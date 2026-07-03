"""YOLO26 + ByteTrack tracker wrapper for the perception front-end.

Hard rules (Issue 002):
    - The detector MUST be ``yolo26m``. If the installed ultralytics does
      not expose that exact model, raise — do NOT silently fall back to
      yolo11 / yolov8 / anything else.
    - The tracker MUST be ByteTrack (``tracker="bytetrack.yaml"``) and
      ``persist=True`` so that IDs survive across frames within a stream.
    - Pretrained weights only. No training or fine-tuning in this module.

Public surface:
    - :class:`TrackerConfig` — declarative knobs for the call.
    - :class:`PerceptionRunResult` — what a run produced.
    - :func:`run_tracker_on_folder` — the end-to-end entry point used
      by the notebook and the unit tests.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Constants kept module-level so they can be referenced from tests and
# other modules without instantiating a tracker.
REQUIRED_MODEL: str = "yolo26m"
REQUIRED_TRACKER: str = "bytetrack.yaml"
PERSON_CLASS_ID: int = 0  # COCO "person" — YOLO26 is COCO-pretrained

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerConfig:
    """Declarative configuration for a single perception run.

    All fields are tunable; the model and tracker are deliberately kept
    here (not hidden inside the runner) so tests and notebooks can
    express "what we asked for" without poking private state.

    Fallback levers (apply only after a real URFD baseline run proves
    weak — defaults leave them all ``None``):

    - ``fallback_track_low_thresh``: lowers ByteTrack / BoT-SORT's
      association threshold (more permissive matching). Forwarded via
      Ultralytics ``cfg=`` and accepted by both trackers.
    - ``fallback_tracker``: switches the tracker config file (e.g.
      ``"botsort.yaml"``). Recognised as a top-level ``tracker=`` arg.
    - ``fallback_end2end``: NOT auto-wired. ``end2end`` is a
      **model/runtime argument** to BoT-SORT, not a tracker config
      key, so it cannot be forwarded through ``model.track()``.
      Setting this field records the *intent* in ``run_meta.json``;
      applying it requires a manual code change (instantiating BoT-SORT
      directly with ``end2end=False``).
    """

    model_name: str = REQUIRED_MODEL
    tracker_config: str = REQUIRED_TRACKER
    person_class_id: int = PERSON_CLASS_ID
    confidence_threshold: float = 0.25
    fallback_track_low_thresh: float | None = None
    fallback_tracker: str | None = None
    fallback_end2end: bool | None = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionBox:
    """One bounding box detected in one frame.

    Coords are absolute pixels (xyxy), confidence is the YOLO score,
    and ``track_id`` is the ByteTrack-assigned ID (or ``None`` when the
    detection was not associated to any active track).
    """

    frame_index: int
    track_id: int | None
    cls_id: int
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class TrackSummary:
    """Per-track aggregate produced at the end of a run.

    ``frame_indices`` is the ordered list of frame indices at which
    this track was observed; ``first_frame`` / ``last_frame`` are
    convenience accessors.
    """

    track_id: int
    frame_indices: tuple[int, ...]
    confidences: tuple[float, ...]

    @property
    def first_frame(self) -> int:
        return self.frame_indices[0]

    @property
    def last_frame(self) -> int:
        return self.frame_indices[-1]

    @property
    def length(self) -> int:
        return len(self.frame_indices)


@dataclass
class PerceptionRunResult:
    """All artefacts a single tracker run produced.

    ``detections`` is the flat list of every detected box; ``tracks`` is
    the per-track summary (computed in :func:`run_tracker_on_folder`).
    Timing fields are populated by the runner — FPS is end-to-end
    frames-per-second, latency is wall-clock per frame.

    ``decode_failures`` counts frames whose Ultralytics ``Results`` could
    not be decoded into :class:`DetectionBox` rows (silently returning
    zero detections is exactly the bug this counter exists to surface —
    Issue 002 review).
    """

    clip_id: str
    source_folder: str
    config: TrackerConfig
    detections: list[DetectionBox] = field(default_factory=list)
    tracks: list[TrackSummary] = field(default_factory=list)
    frame_count: int = 0
    detection_count: int = 0
    track_count: int = 0
    elapsed_seconds: float = 0.0
    gpu_name: str | None = None
    fallback_used: str | None = None
    decode_failures: int = 0

    @property
    def fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frame_count / self.elapsed_seconds

    @property
    def latency_ms_per_frame(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        return (self.elapsed_seconds / self.frame_count) * 1000.0


# ---------------------------------------------------------------------------
# Strict model verification
# ---------------------------------------------------------------------------


class UnsupportedModelError(RuntimeError):
    """Raised when the requested detector model is not yolo26m."""


def assert_required_model_available(model_name: str = REQUIRED_MODEL) -> None:
    """Verify that ``model_name`` can be loaded by the installed ultralytics.

    Issue 002 rule: the detector MUST be ``yolo26m``. Any other model
    must abort the run — silently switching to yolo11 / yolov8 / etc.
    would invalidate every downstream evaluation.

    Loading the model is the cheapest, loudest probe; a ``FileNotFoundError``
    on the weights file or an ``AttributeError`` from ``YOLO(...)`` both
    surface as UnsupportedModelError with the original cause attached.
    """
    if model_name != REQUIRED_MODEL:
        raise UnsupportedModelError(
            f"Required model is {REQUIRED_MODEL!r}; got {model_name!r}. "
            f"Per Issue 002: do NOT silently switch to yolo11/yolov8/etc."
        )
    try:
        # Local import so this module loads even before ultralytics is
        # installed (tests for non-tracking code don't need it).
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise UnsupportedModelError(
            "ultralytics is not installed; install it via colab/setup.py "
            "before invoking the perception front-end."
        ) from exc

    try:
        YOLO(model_name)
    except Exception as exc:  # noqa: BLE001 — we want to re-raise as UnsupportedModelError
        raise UnsupportedModelError(
            f"Model {model_name!r} is not available in the installed ultralytics "
            f"({type(exc).__name__}: {exc}). Stop; do not fall back to another model."
        ) from exc


# ---------------------------------------------------------------------------
# GPU info (best-effort, never raises)
# ---------------------------------------------------------------------------


def query_gpu_name() -> str | None:
    """Return the active CUDA device name, or ``None`` when unavailable."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — best-effort
        return None


# ---------------------------------------------------------------------------
# Tracker invocation
# ---------------------------------------------------------------------------


def _build_tracker_kwargs(config: TrackerConfig) -> dict[str, object]:
    """Translate :class:`TrackerConfig` into the kwargs ultralytics wants."""
    kwargs: dict[str, object] = {
        "tracker": config.tracker_config,
        "persist": True,
        "classes": [config.person_class_id],
        "conf": config.confidence_threshold,
        "verbose": False,
    }
    if config.fallback_tracker is not None:
        kwargs["tracker"] = config.fallback_tracker
    return kwargs


def _apply_fallback_kwargs(
    base_kwargs: dict[str, object], config: TrackerConfig
) -> dict[str, object]:
    """Merge fallback-level kwargs into ``base_kwargs``.

    What this DOES forward (auto-wired by ``model.track()``):

    - ``fallback_track_low_thresh`` → ``cfg={"track_low_thresh": ...}``.
      Ultralytics forwards arbitrary tracker keys via the ``cfg=`` dict;
      ByteTrack and BoT-SORT both honour ``track_low_thresh``.

    What this does NOT forward:

    - ``fallback_end2end``: ``end2end`` is a model/runtime argument
      (it's a constructor arg of the BoT-SORT class), not a tracker
      config key. It cannot be expressed through ``model.track()``'s
      ``cfg=``. The field stays in :class:`TrackerConfig` so the
      *intent* is recorded in ``run_meta.json``; applying it requires
      a manual code change (see Issue 002 review).
    """
    out = dict(base_kwargs)
    if config.fallback_track_low_thresh is not None:
        out["cfg"] = {"track_low_thresh": config.fallback_track_low_thresh}
    return out


def _format_fallback_used(config: TrackerConfig) -> str | None:
    """Human-readable record of which fallback lever fired (or ``None``).

    Distinguishes auto-wired levers (already applied) from
    intent-only levers (require manual code change). The split keeps
    the report honest: a reader can see at a glance what was actually
    done vs. what was merely requested.
    """
    applied: list[str] = []
    if config.fallback_track_low_thresh is not None:
        applied.append(f"track_low_thresh={config.fallback_track_low_thresh}")
    if config.fallback_tracker is not None:
        applied.append(f"tracker={config.fallback_tracker}")

    manual_only: list[str] = []
    if config.fallback_end2end is not None:
        manual_only.append(
            f"end2end={config.fallback_end2end} (manual intervention required)"
        )

    parts = applied + manual_only
    return ", ".join(parts) if parts else None


def _flatten_result(
    result: object,
    frame_index: int,
    person_class_id: int,
) -> tuple[list[DetectionBox], bool]:
    """Convert one ultralytics ``Results`` object into :class:`DetectionBox` rows.

    Returns ``(rows, decoded_ok)``. ``decoded_ok=False`` means the
    result object couldn't be interpreted — typically because Ultralytics
    changed its ``Boxes`` API between versions. The caller increments
    ``run.decode_failures`` so the report surfaces the issue rather than
    silently reporting zero detections on every frame.

    A "decoded but no detections" result (a frame with no people in it)
    is NOT a failure — that returns ``(empty, True)``.
    """
    try:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return [], False
        # Newer ultralytics: tensor attributes directly. If any of these
        # attributes are missing, the decode has changed shape.
        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        cls = getattr(boxes, "cls", None)
        if xyxy is None or conf is None or cls is None:
            return [], False
        ids = getattr(boxes, "id", None)  # present when tracking is on
        rows: list[DetectionBox] = []
        for i in range(len(xyxy)):
            cls_id = int(cls[i].item())
            if cls_id != person_class_id:
                continue
            track_id: int | None
            if ids is not None and i < len(ids) and ids[i] is not None:
                track_id = int(ids[i].item())
            else:
                track_id = None
            x1, y1, x2, y2 = (float(v) for v in xyxy[i].tolist())
            rows.append(DetectionBox(
                frame_index=frame_index,
                track_id=track_id,
                cls_id=cls_id,
                confidence=float(conf[i].item()),
                x_min=x1, y_min=y1, x_max=x2, y_max=y2,
            ))
        return rows, True
    except Exception:  # noqa: BLE001 — count any other decode failure
        return [], False


def run_tracker_on_folder(
    clip_id: str,
    frame_paths: Iterable[Path],
    config: TrackerConfig | None = None,
) -> PerceptionRunResult:
    """Run YOLO26 + ByteTrack over an ordered sequence of frame paths.

    Args:
        clip_id: stable identifier for the clip (used in result metadata
            and downstream artefact filenames).
        frame_paths: ordered frame paths in temporal order. The caller is
            responsible for ordering — use :class:`FrameFolderReader`.
        config: tracker configuration. Defaults to the Issue 002 baseline
            (yolo26m + bytetrack.yaml, person class, conf=0.25).

    Returns:
        A populated :class:`PerceptionRunResult` with detections, track
        summaries, timing, GPU info, and the decode-failure counter
        (``run.decode_failures``).

    Raises:
        UnsupportedModelError: when the requested detector is not
            ``yolo26m`` or ultralytics can't load it.
    """
    config = config or TrackerConfig()
    assert_required_model_available(config.model_name)

    # Local import keeps the rest of the module testable without ultralytics.
    from ultralytics import YOLO  # type: ignore

    model = YOLO(config.model_name)
    result = PerceptionRunResult(
        clip_id=clip_id,
        source_folder="",  # filled in by the caller / orchestrator
        config=config,
        gpu_name=query_gpu_name(),
        fallback_used=_format_fallback_used(config),
    )

    tracker_kwargs = _apply_fallback_kwargs(_build_tracker_kwargs(config), config)

    # The frame_paths iterable may be a generator; materialise to a list
    # so we can report frame_count and track the first/last frame_index.
    paths = list(frame_paths)
    result.frame_count = len(paths)

    if result.frame_count == 0:
        return result

    started = time.perf_counter()
    # Ultralytics supports streaming a list of image paths directly; this
    # is the same code path used for video tracks, just with images.
    # ``stream=True`` yields one Results per frame so we can attach the
    # correct frame_index without re-parsing filenames later.
    stream = model.track(
        source=[str(p) for p in paths],
        stream=True,
        **tracker_kwargs,
    )
    for frame_index, ultralytics_result in enumerate(stream):
        rows, decoded_ok = _flatten_result(
            ultralytics_result, frame_index, config.person_class_id
        )
        result.detections.extend(rows)
        if not decoded_ok:
            result.decode_failures += 1
    result.elapsed_seconds = time.perf_counter() - started

    result.detection_count = len(result.detections)
    result.tracks = _summarise_tracks(result.detections)
    result.track_count = len(result.tracks)
    return result


def _summarise_tracks(detections: Iterable[DetectionBox]) -> list[TrackSummary]:
    """Group detections by ``track_id`` and return one :class:`TrackSummary` per track.

    Detections without a ``track_id`` (no association) are skipped — they
    don't belong to any track and would corrupt the fragmentation math.
    """
    by_track: dict[int, list[DetectionBox]] = defaultdict(list)
    for det in detections:
        if det.track_id is None:
            continue
        by_track[det.track_id].append(det)

    summaries: list[TrackSummary] = []
    for track_id in sorted(by_track):
        detections_for_track = sorted(by_track[track_id], key=lambda d: d.frame_index)
        summaries.append(TrackSummary(
            track_id=track_id,
            frame_indices=tuple(d.frame_index for d in detections_for_track),
            confidences=tuple(d.confidence for d in detections_for_track),
        ))
    return summaries


__all__: tuple[str, ...] = (
    "REQUIRED_MODEL",
    "REQUIRED_TRACKER",
    "PERSON_CLASS_ID",
    "TrackerConfig",
    "DetectionBox",
    "TrackSummary",
    "PerceptionRunResult",
    "UnsupportedModelError",
    "assert_required_model_available",
    "query_gpu_name",
    "run_tracker_on_folder",
)