"""End-to-end Issue 003 cropping runner.

Reads Issue 002 tracking artefacts from Drive, filters to cam0 tracks,
builds deterministic crop windows, applies the geometric transforms
from :mod:`cropping.clip_builder`, and writes WebDataset-style .tar
shards to ``MyDrive/fall_detection/artifacts/crops/``.

No re-detection happens here. The runner is the seam between "Issue 002
tracking outputs on Drive" and "Issue 005 / 008 training-ready shards
on Drive".

Pipeline (per clip):

    1. Load the Issue 002 detection JSON + clip record (from manifest).
    2. Stage the source frames from Drive to a local-disk sub-folder
       (same I/O reasoning as the Issue 002 perception fix — small-
       file reads from Drive FUSE were the bottleneck).
    3. Filter to ``camera == "cam0"`` (Issue 002 close-out decision).
    4. Group detections by track_id into :class:`TrackedBox` rows.
    5. Call :func:`build_windows_for_track` for each track.
    6. For each emitted :class:`TrackWindow`, read the needed frames
       from the local staged copy, apply :func:`compute_crop_geometry`
       per frame + :func:`apply_crop_to_frame`, write each frame +
       metadata into a shard via :class:`ShardWriter`.
    7. Clean up the local staged copy on exit.
    8. Record every skip reason in the run-level summary.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image
import numpy as np

from cropping.clip_builder import (
    CropConfig,
    apply_crop_to_frame,
    compute_crop_geometry,
)
from cropping.shard_writer import (
    ShardWriter,
    compute_shard_padding_width,
    shard_filename,
)
from cropping.track_windows import (
    TrackedBox,
    WindowBuildResult,
    build_windows_for_track,
)
from data.manifests import ClipRecord, FallLabel, load_manifest
from perception.tracker import DetectionBox


CAMERA_FILTER_DEFAULT: str = "cam0"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardClipRecord:
    """Per-clip summary written to the run summary JSON.

    Captures what was emitted, what was skipped, and which shard the
    clip landed in — enough that a downstream trainer can find the
    shard without re-scanning.
    """

    clip_id: str
    dataset: str
    label: str
    track_id: int
    windows_emitted: int
    coverage: float
    shard_filename: str
    member_keys: tuple[str, ...]


@dataclass
class RunSummary:
    """Output of one full cropping run."""

    started_at_utc: str
    elapsed_seconds: float
    crops_root: str
    crop_config: dict[str, object]
    clips_processed: int = 0
    tracks_processed: int = 0
    windows_emitted: int = 0
    windows_skipped: int = 0
    skip_reason_counts: dict[str, int] = field(default_factory=dict)
    shards_written: int = 0
    shard_clip_records: list[ShardClipRecord] = field(default_factory=list)
    local_staging_root: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shard_clip_records"] = [asdict(r) for r in self.shard_clip_records]
        return payload


# ---------------------------------------------------------------------------
# Loading Issue 002 artefacts
# ---------------------------------------------------------------------------


def load_track_boxes_for_clip(
    detections_path: Path,
) -> dict[int, list[TrackedBox]]:
    """Read an Issue 002 detections JSON and group boxes by ``track_id``.

    Untracked detections (those without a track_id) are skipped — they
    don't form a coherent clip.
    """
    payload = json.loads(Path(detections_path).read_text(encoding="utf-8"))
    boxes_by_track: dict[int, list[TrackedBox]] = {}
    for raw in payload:
        det = DetectionBox(
            frame_index=int(raw["frame_index"]),
            track_id=(int(raw["track_id"]) if raw.get("track_id") is not None else None),
            cls_id=int(raw["cls_id"]),
            confidence=float(raw["confidence"]),
            x_min=float(raw["x_min"]),
            y_min=float(raw["y_min"]),
            x_max=float(raw["x_max"]),
            y_max=float(raw["y_max"]),
        )
        if det.track_id is None:
            continue
        boxes_by_track.setdefault(det.track_id, []).append(TrackedBox(
            frame_index=det.frame_index,
            x_min=det.x_min, y_min=det.y_min,
            x_max=det.x_max, y_max=det.y_max,
            confidence=det.confidence,
        ))
    return boxes_by_track


# ---------------------------------------------------------------------------
# Per-frame geometry application
# ---------------------------------------------------------------------------


def _metadata_for_frame(
    clip_record: ClipRecord,
    track_id: int,
    frame_index: int,
    frame_offset: int,
    crop_config: CropConfig,
    margin: float,
    missing: bool,
    coverage: float,
    shard_filename_str: str,
) -> dict[str, object]:
    """Build the per-frame metadata sidecar written into the shard."""
    return {
        "clip_id": clip_record.clip_id,
        "dataset": clip_record.dataset,
        "label": clip_record.label.value,
        "source_path": clip_record.source_path,
        "track_id": track_id,
        "frame_index": frame_index,
        "frame_offset": frame_offset,
        "missing_frame": missing,
        "coverage": coverage,
        "crop_config": {
            "output_size": crop_config.output_size,
            "margin": crop_config.margin,
            "clip_length": crop_config.clip_length,
        },
        "margin_used": margin,
        "shard_filename": shard_filename_str,
    }


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------


@dataclass
class ClipRunOutcome:
    """Result of running one clip end-to-end."""

    clip_record: ClipRecord
    track_results: list[WindowBuildResult]
    emitted_windows: int = 0
    skipped_reasons: list[str] = field(default_factory=list)


def _load_frames_from_local(
    local_folder: Path,
    source_folder: Path,
    frame_indices: Iterable[int],
) -> dict[int, np.ndarray]:
    """Read frames from a local-disk staged folder keyed by ``frame_NNNNN.png``.

    The local staged layout is a flat directory of
    ``frame_00000.png, frame_00001.png, …`` produced by
    :func:`stage_clip_frames_for_cropping`. We map frame_index →
    ``frame_{index:05d}.png`` directly so we skip the discovery step.

    If a frame_index is missing locally (gap), we fall back to a single
    discovery-based lookup against the original source folder so the
    caller still gets a frame instead of a missing-file crash. This is
    defensive — proper local staging should always cover every index.
    """
    needed = sorted(set(frame_indices))
    out: dict[int, np.ndarray] = {}
    for index in needed:
        candidate = local_folder / f"frame_{index:05d}.png"
        path: Path | None = None
        if candidate.is_file():
            path = candidate
        else:
            # Fallback: discover in source folder. Useful when a frame
            # was missing locally but present on Drive (or vice-versa).
            from perception.frames import discover_frames
            for record in discover_frames(source_folder):
                if record.index == index:
                    path = record.path
                    break
        if path is None:
            continue
        out[index] = np.array(Image.open(path).convert("RGB"))
    return out


def _load_frames(folder: Path, frame_indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Read frames directly from a folder on Drive (legacy path).

    Issue 002's structured tracks carry frame indices, not paths, so
    we read the specific frames instead of the whole sequence. Saves
    I/O when a track has gaps. **Prefer** :func:`_load_frames_from_local`
    on real Colab runs — this function reads from Drive directly and
    triggers the small-file I/O bottleneck.
    """
    from perception.frames import discover_frames
    frames = {frame.index: frame.path for frame in discover_frames(folder)}
    needed = set(frame_indices)
    out: dict[int, np.ndarray] = {}
    for index in sorted(needed & set(frames.keys())):
        out[index] = np.array(Image.open(frames[index]).convert("RGB"))
    return out


def stage_clip_frames_for_cropping(
    clip_id: str,
    source_folder: Path,
    local_root: Path,
) -> tuple[Path, int]:
    """Stage one clip's source frames to local disk for the cropping step.

    Mirrors the perception runner's staging so cropping doesn't pay the
    Drive FUSE small-file I/O cost. Returns ``(local_folder, frame_count)``.
    Returns ``(source_folder, frame_count)`` if ``FALL_DETECTION_SKIP_LOCAL_STAGING``
    is set (caller should treat the returned path as already-local).

    Why a separate stager: the cropping step runs AFTER perception's
    ``StagedClipContext`` has cleaned up its local copy, so we have to
    re-stage. Reusing ``LocalFrameStager`` is possible but its
    filename convention (``frame_NNNNN.png``) is exactly what we want
    here, so we just call the lower-level helpers directly.
    """
    if os.environ.get("FALL_DETECTION_SKIP_LOCAL_STAGING"):
        from perception.frames import discover_frames
        return source_folder, sum(1 for _ in discover_frames(source_folder))

    from perception.frames import discover_frames

    safe = re.sub(r"[^A-Za-z0-9_-]", "_", clip_id) or "unnamed"
    local_folder = local_root / f"crops_{safe}"
    if local_folder.is_dir():
        shutil.rmtree(local_folder, ignore_errors=True)
    local_folder.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    for record in discover_frames(source_folder):
        destination = local_folder / f"frame_{record.index:05d}.png"
        shutil.copy2(record.path, destination)
        frame_count += 1
    return local_folder, frame_count


def _process_clip(
    clip_record: ClipRecord,
    boxes_by_track: dict[int, list[TrackedBox]],
    crop_config: CropConfig,
    crops_root: Path,
    shard_padding: int,
    shard_index: int,
    *,
    layout_root: Path,
    local_root: Path | None = None,
) -> tuple[ClipRunOutcome, int, ShardWriter]:
    """Process one clip end-to-end. Returns the outcome and a new shard index.

    All clips in one run land in the same shard (or rotate when the
    shard reaches a size budget). The caller owns the shard lifecycle.

    If ``local_root`` is provided, the clip's source frames are copied
    to ``<local_root>/crops_<clip_id>/`` BEFORE any frame reads, so the
    runner avoids Drive FUSE small-file I/O. The local copy is removed
    before returning so no stale frames leak into the next clip.

    ``layout_root`` is REQUIRED: it's the root against which
    ``clip_record.source_path`` resolves (the dataset/layout root, NOT
    the crop artefact root). The previous code derived the source root
    from ``crops_root.parent`` — that was wrong in any layout where the
    dataset and the artefact root didn't share a parent (i.e. LOCAL
    mode). Making the parameter explicit kills the ambiguity.
    """
    outcome = ClipRunOutcome(clip_record=clip_record, track_results=[])

    # Determine image dimensions from the first available frame so we
    # can clamp boxes to the image before cropping. Resolve the
    # source folder against ``layout_root`` (the dataset root).
    source_folder = layout_root / clip_record.source_path
    from perception.frames import discover_frames

    # Stage to local disk if requested. The function returns either the
    # original source_folder (when skipping) or the staged local copy.
    effective_source = source_folder
    staged_local: Path | None = None
    if local_root is not None:
        staged_local, _ = stage_clip_frames_for_cropping(
            clip_record.clip_id, source_folder, local_root,
        )
        # When staging is skipped (env var), staged_local == source_folder;
        # otherwise it points at the fresh local copy.
        effective_source = staged_local

    try:
        discovered = discover_frames(effective_source)
        if not discovered:
            outcome.skipped_reasons.append("empty_clip_folder")
            return outcome, shard_index, _open_dummy_shard()
        image_width, image_height = Image.open(discovered[0].path).size

        # One shard per clip for simplicity; a real run can rotate shards
        # when size budget is reached. We track shard_index in the caller.
        shard_path = crops_root / shard_filename(shard_index, shard_padding)
        writer = ShardWriter(shard_path, shard_index=shard_index)
        writer.__enter__()

        for track_id, boxes in sorted(boxes_by_track.items()):
            build = build_windows_for_track(track_id, boxes, crop_config)
            outcome.track_results.append(build)
            outcome.emitted_windows += build.emitted_count
            for reason in build.skipped:
                outcome.skipped_reasons.append(reason.reason)
            for window in build.emitted:
                # Read just the frames this window needs. When we have a
                # local staged folder, use the direct lookup path
                # (cheaper than re-discovering the directory).
                if staged_local is not None:
                    frames = _load_frames_from_local(
                        staged_local, source_folder, window.frame_indices,
                    )
                else:
                    frames = _load_frames(source_folder, window.frame_indices)
                if not frames:
                    outcome.skipped_reasons.append("frames_unreadable")
                    continue
                for offset, (frame_idx, box) in enumerate(zip(window.frame_indices,
                                                              window.boxes)):
                    if frame_idx not in frames:
                        # Frame couldn't be loaded (gap in source folder);
                        # emit a fully-padded placeholder so the clip length
                        # contract holds.
                        placeholder = np.zeros((image_height, image_width, 3),
                                                dtype=np.uint8)
                        frames[frame_idx] = placeholder
                    geometry = compute_crop_geometry(
                        box.x_min, box.y_min, box.x_max, box.y_max,
                        margin=crop_config.margin,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    crop = apply_crop_to_frame(
                        frames[frame_idx], geometry, output_size=crop_config.output_size,
                    )
                    meta = _metadata_for_frame(
                        clip_record=clip_record,
                        track_id=track_id,
                        frame_index=frame_idx,
                        frame_offset=offset,
                        crop_config=crop_config,
                        margin=crop_config.margin,
                        missing=offset in set(window.missing_frames),
                        coverage=window.coverage,
                        shard_filename_str=shard_path.name,
                    )
                    writer.write_clip_member(
                        f"{clip_record.clip_id}_t{track_id}_w{len(writer._clip_keys):03d}",
                        frame_offset=offset,
                        image=crop,
                        metadata=meta,
                    )
        return outcome, shard_index + 1, writer
    finally:
        # Clean up the local staged copy so stale frames can't
        # contaminate the next clip. Defensive: works even if the runner
        # raised before getting here.
        if staged_local is not None and staged_local != source_folder \
                and staged_local.is_dir():
            shutil.rmtree(staged_local, ignore_errors=True)


def _open_dummy_shard() -> ShardWriter:
    """Return a no-op shard writer for error paths."""
    return ShardWriter(path=Path("/dev/null"), shard_index=-1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_cropping(
    *,
    layout_root: Path,
    perception_root: Path,
    crops_root: Path,
    manifest_path: Path,
    crop_config: CropConfig | None = None,
    camera_filter: str = CAMERA_FILTER_DEFAULT,
    max_shards: int = 9999,
    local_root: Path | None = None,
) -> RunSummary:
    """Run the full Issue 003 cropping pipeline.

    Args:
        layout_root: the project's Drive root (e.g.
            ``MyDrive/fall_detection``); used to resolve source paths.
        perception_root: directory containing the Issue 002 per-clip
            ``<clip_id>_detections.json`` files.
        crops_root: directory where WebDataset shards are written.
        manifest_path: path to the URFD debug manifest.
        crop_config: clip geometry config; defaults to PRD defaults.
        camera_filter: only process clips whose manifest notes carry
            this camera (default ``"cam0"`` per Issue 002 close-out).
        max_shards: upper bound for shard-name padding width.
        local_root: if set, each clip's source frames are copied to
            ``<local_root>/crops_<clip_id>/`` BEFORE crop building so
            the runner reads from local disk instead of Drive FUSE.
            Set ``FALL_DETECTION_SKIP_LOCAL_STAGING=1`` to skip even
            when ``local_root`` is set (e.g. when running on a host
            that already has fast disk access to the data).

    Returns:
        A populated :class:`RunSummary` describing what was processed.
    """
    crop_config = crop_config or CropConfig()
    crops_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    summary = RunSummary(
        started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        elapsed_seconds=0.0,
        crops_root=str(crops_root),
        crop_config=asdict(crop_config),
        local_staging_root=str(local_root) if local_root is not None else None,
    )

    manifest = load_manifest(manifest_path)
    eligible_clips = [
        clip for clip in manifest.clips
        if _clip_passes_camera_filter(clip, camera_filter)
    ]

    shard_padding = compute_shard_padding_width(max_shards)
    next_shard_index = 0
    open_writers: dict[int, ShardWriter] = {}

    def _get_writer(shard_index: int) -> ShardWriter:
        writer = open_writers.get(shard_index)
        if writer is not None:
            return writer
        path = crops_root / shard_filename(shard_index, shard_padding)
        writer = ShardWriter(path, shard_index=shard_index)
        writer.__enter__()
        open_writers[shard_index] = writer
        return writer

    for clip in eligible_clips:
        detections_path = perception_root / clip.clip_id / f"{clip.clip_id}_detections.json"
        if not detections_path.is_file():
            summary.clips_processed += 1
            continue
        boxes_by_track = load_track_boxes_for_clip(detections_path)
        if not boxes_by_track:
            summary.clips_processed += 1
            continue
        outcome, next_shard_index, _writer = _process_clip(
            clip, boxes_by_track, crop_config,
            crops_root=crops_root,
            shard_padding=shard_padding,
            shard_index=next_shard_index,
            layout_root=layout_root,
            local_root=local_root,
        )
        # Wire the per-clip windows into the live writer (which the
        # above already opened and wrote into). We rebuild the per-clip
        # shard_clip_records from the track results since _process_clip
        # may have rotated shards.
        for build in outcome.track_results:
            summary.tracks_processed += 1
            summary.windows_emitted += build.emitted_count
            for window in build.emitted:
                shard_filename_str = shard_filename(next_shard_index - 1, shard_padding)
                member_keys = tuple(
                    f"{safe_member_name(clip.clip_id)}_t{window.track_id}_w{{:03d}}_{i:04d}.image.jpg"
                    for i in range(window.length)
                )
                summary.shard_clip_records.append(ShardClipRecord(
                    clip_id=clip.clip_id,
                    dataset=clip.dataset,
                    label=clip.label.value,
                    track_id=window.track_id,
                    windows_emitted=1,
                    coverage=window.coverage,
                    shard_filename=shard_filename_str,
                    member_keys=member_keys,
                ))
        for reason in outcome.skipped_reasons:
            summary.windows_skipped += 1
            summary.skip_reason_counts[reason] = summary.skip_reason_counts.get(reason, 0) + 1
        summary.clips_processed += 1

    # Close any open writers.
    for writer in open_writers.values():
        writer.close()
    summary.shards_written = len(open_writers)

    summary.elapsed_seconds = time.perf_counter() - started
    return summary


def _clip_passes_camera_filter(clip: ClipRecord, camera_filter: str) -> bool:
    """True when the clip's notes include the requested camera.

    The URFD manifest builder writes ``camera=<cam>`` into ``notes``; we
    parse that out rather than adding a dedicated camera field to the
    schema (out of scope for Issue 003 — URFD-only).
    """
    if not camera_filter:
        return True
    if not clip.notes:
        return False
    return f"camera={camera_filter}" in clip.notes


def safe_member_name(stem: str) -> str:
    """Re-export for symmetry with callers that import from this module."""
    from cropping.shard_writer import safe_member_name as _safe
    return _safe(stem)


def write_run_summary(summary: RunSummary, destination: Path) -> Path:
    """Serialise the run summary to JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


__all__: tuple[str, ...] = (
    "CAMERA_FILTER_DEFAULT",
    "ClipRunOutcome",
    "RunSummary",
    "ShardClipRecord",
    "load_track_boxes_for_clip",
    "run_cropping",
    "write_run_summary",
)