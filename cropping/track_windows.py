"""Track → fixed-length window conversion for Issue 003.

A :class:`TrackWindow` is a deterministic, fixed-length slice of one
tracked identity's bounding-box timeline. Building them is the core
"what is a clip?" question this module answers.

Determinism rules (Issue 003 hard requirement: "same input + same config
must produce the same crop pixels, metadata, and shard names"):

    - **Short tracks:** tracks with fewer than ``clip_length`` boxes are
      emitted with empty padding frames (degenerate boxes that the
      clip-builder produces zero-area crops from). The min-coverage gate
      then drops them with reason ``"insufficient_coverage"``.
    - **Missing frames:** a frame index that has no detection is
      carried as a "gap" — the window still covers that index, but the
      geometry falls back to the last-seen box (no spatial jump).
      After ``gap_tolerance`` consecutive gaps the window is dropped
      with reason ``"too_many_gaps"``.
    - **Frame-index gaps:** the window's frame indices are dense
      (``start, start+1, ..., start+clip_length-1``); we never skip
      within a window. If the track has gaps, they show up as missing
      frames in the timeline above.
    - **Boxes partially outside the image:** clamped by
      :func:`clipping.clip_builder.clip_box_to_image` — no shifting.
    - **Very small boxes:** the minimum-coverage gate rejects windows
      where the average box area is below ``min_box_area_px`` or where
      the median box side is below ``min_box_side_px``.

The min-coverage gate is the difference between "silently emit garbage"
and "skip with a clear reason". Every skipped window reports WHY it
was skipped so a human can decide whether the policy is right for the
dataset at hand.

Pure / no I/O / no model dependency. Tested in
``tests/test_track_windows.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from cropping.clip_builder import CropConfig


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedBox:
    """One box at one frame index. Pure data; produced upstream by Issue 002."""

    frame_index: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 0.0

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    @property
    def shorter_side(self) -> float:
        return min(
            max(0.0, self.x_max - self.x_min),
            max(0.0, self.y_max - self.y_min),
        )


@dataclass(frozen=True)
class TrackWindow:
    """One fixed-length clip built from one track.

    ``boxes`` is dense (length == ``config.clip_length``). ``missing_frames``
    marks which indices of ``boxes`` were filled with a fallback geometry
    rather than a real tracked box. ``frame_indices`` are the dense
    absolute indices in the original timeline.
    """

    track_id: int
    boxes: tuple[TrackedBox, ...]
    frame_indices: tuple[int, ...]
    missing_frames: tuple[int, ...] = field(default_factory=tuple)
    coverage: float = 0.0  # fraction of indices that were real (not missing)

    @property
    def length(self) -> int:
        return len(self.boxes)

    @property
    def is_complete(self) -> bool:
        return not self.missing_frames


@dataclass(frozen=True)
class SkipReason:
    """Why one candidate window was not emitted."""

    reason: str  # "short_track", "too_many_gaps", "insufficient_coverage", "empty_track"
    detail: str = ""


@dataclass(frozen=True)
class WindowBuildResult:
    """Outcome of building windows from one track."""

    track_id: int
    emitted: tuple[TrackWindow, ...] = field(default_factory=tuple)
    skipped: tuple[SkipReason, ...] = field(default_factory=tuple)
    source_box_count: int = 0

    @property
    def emitted_count(self) -> int:
        return len(self.emitted)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


# ---------------------------------------------------------------------------
# Window building
# ---------------------------------------------------------------------------


def build_windows_for_track(
    track_id: int,
    boxes: Iterable[TrackedBox],
    config: CropConfig,
    *,
    min_box_area_px: float = 32.0 * 32.0,
    min_box_side_px: float = 16.0,
    gap_tolerance: int = 4,
    max_windows: int = 8,
    stride: int | None = None,
) -> WindowBuildResult:
    """Turn one ordered sequence of boxes into zero-or-more fixed-length windows.

    Args:
        track_id: stable identifier for the track (carried into the window
            metadata).
        boxes: ordered boxes for one track. The function sorts by
            ``frame_index`` defensively (sorted input is the
            caller's responsibility but we don't trust it).
        config: clip geometry configuration.
        min_box_area_px: minimum acceptable average box area; windows
            below this are dropped with reason ``"insufficient_coverage"``.
        min_box_side_px: minimum acceptable median box side; same drop
            reason.
        gap_tolerance: maximum number of consecutive missing frames
            allowed inside one window before the window is dropped
            (``"too_many_gaps"``).
        max_windows: hard cap on how many windows we emit from one
            track. Prevents a 1000-frame track from producing 968
            overlapping windows; a long track should be down-sampled,
            not infinitely sliced.
        stride: separation between window start frames. ``None`` means
            ``clip_length`` (non-overlapping). Smaller values produce
            overlapping windows, larger values produce sparse sampling.

    Returns:
        A :class:`WindowBuildResult` listing every emitted window and
        every skip reason. The caller writes the emitted windows to
        Drive and records the skip reasons in the run log.
    """
    clip_length = config.clip_length
    if stride is None:
        stride = clip_length
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")

    # Sort + deduplicate by frame_index (last write wins on collision —
    # the tracker should never produce collisions, but be defensive).
    by_frame: dict[int, TrackedBox] = {}
    for box in boxes:
        by_frame[box.frame_index] = box

    if not by_frame:
        return WindowBuildResult(
            track_id=track_id,
            skipped=(SkipReason(reason="empty_track", detail="no boxes"),),
        )

    sorted_frames = sorted(by_frame)
    first_frame = sorted_frames[0]
    last_frame = sorted_frames[-1]

    # Sliding window over the dense frame-index timeline. Window starts
    # are at ``first_frame, first_frame + stride, ...`` so the windows
    # tile the track without overlap by default.
    emitted: list[TrackWindow] = []
    skipped: list[SkipReason] = []

    start_frame = first_frame
    while start_frame <= last_frame and len(emitted) < max_windows:
        window_frame_indices = tuple(range(start_frame,
                                           start_frame + clip_length))
        window_boxes: list[TrackedBox] = []
        missing: list[int] = []
        carry: TrackedBox | None = None
        run_of_missing = 0
        aborted = False
        for offset, frame_idx in enumerate(window_frame_indices):
            if frame_idx in by_frame:
                box = by_frame[frame_idx]
                window_boxes.append(box)
                carry = box
                run_of_missing = 0
            else:
                missing.append(offset)
                run_of_missing += 1
                if carry is None:
                    window_boxes.append(TrackedBox(
                        frame_index=frame_idx,
                        x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0,
                        confidence=0.0,
                    ))
                else:
                    window_boxes.append(TrackedBox(
                        frame_index=frame_idx,
                        x_min=carry.x_min, y_min=carry.y_min,
                        x_max=carry.x_max, y_max=carry.y_max,
                        confidence=0.0,
                    ))
                if run_of_missing > gap_tolerance:
                    skipped.append(SkipReason(
                        reason="too_many_gaps",
                        detail=(f"start_frame={start_frame} "
                                f"consecutive_missing={run_of_missing} "
                                f"tolerance={gap_tolerance}"),
                    ))
                    aborted = True
                    break

        if aborted:
            start_frame += stride
            continue

        if not window_boxes:
            start_frame += stride
            continue

        coverage = 1.0 - (len(missing) / len(window_boxes))

        real_boxes = [b for i, b in enumerate(window_boxes)
                       if i not in set(missing)]
        if real_boxes:
            avg_area = sum(b.area for b in real_boxes) / len(real_boxes)
            median_side = _median([b.shorter_side for b in real_boxes])
            if avg_area < min_box_area_px:
                skipped.append(SkipReason(
                    reason="insufficient_coverage",
                    detail=(f"start_frame={start_frame} avg_area={avg_area:.1f} "
                            f"min={min_box_area_px}"),
                ))
                start_frame += stride
                continue
            if median_side < min_box_side_px:
                skipped.append(SkipReason(
                    reason="insufficient_coverage",
                    detail=(f"start_frame={start_frame} median_side={median_side:.1f} "
                            f"min={min_box_side_px}"),
                ))
                start_frame += stride
                continue
        else:
            skipped.append(SkipReason(
                reason="insufficient_coverage",
                detail=(f"start_frame={start_frame} no_real_boxes "
                        f"coverage={coverage:.2f}"),
            ))
            start_frame += stride
            continue

        emitted.append(TrackWindow(
            track_id=track_id,
            boxes=tuple(window_boxes),
            frame_indices=window_frame_indices,
            missing_frames=tuple(missing),
            coverage=coverage,
        ))
        start_frame += stride

    if not emitted and not skipped:
        skipped.append(SkipReason(
            reason="short_track",
            detail=f"track had {len(by_frame)} boxes < clip_length={clip_length}",
        ))

    return WindowBuildResult(
        track_id=track_id,
        emitted=tuple(emitted),
        skipped=tuple(skipped),
        source_box_count=len(by_frame),
    )


def _median(values: Iterable[float]) -> float:
    """Median of a non-empty iterable. No numpy dependency for the math."""
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_values[mid])
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def group_boxes_by_track(
    boxes_per_track: Mapping[int, Iterable[TrackedBox]],
) -> dict[int, list[TrackedBox]]:
    """Helper: convert the perception layer's per-track mapping to lists.

    The Issue 002 perception result hands us a flat list of detections;
    callers group by track_id and pass each group to
    :func:`build_windows_for_track`. This helper makes that conversion
    explicit and testable.
    """
    return {track_id: list(boxes) for track_id, boxes in boxes_per_track.items()}


__all__: tuple[str, ...] = (
    "TrackedBox",
    "TrackWindow",
    "SkipReason",
    "WindowBuildResult",
    "build_windows_for_track",
    "group_boxes_by_track",
)