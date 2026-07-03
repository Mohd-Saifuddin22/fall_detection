"""Track-continuity / fragmentation report for the perception front-end.

What this module computes (URFD has no detection/tracking ground truth,
so the formal MOT metrics are intentionally NOT computed — only
structural / count-based diagnostics):

    - per-track length distribution,
    - track fragmentation: how many disjoint frame ranges each track
      covers (1 = continuous, >1 = fragmented),
    - longest continuous track — proxy for "did the falling person
      keep the same ID through the fall window?",
    - ID-switch candidates: frames where the *primary* track changes
      identity (i.e. the dominant track in frame N is not the dominant
      track in frame N+1),
    - detection count per frame (sanity check),
    - summary metrics honest about missing ground truth.

All math is plain Python — no torch / ultralytics dependency. Easy to
unit-test on synthetic :class:`TrackSummary` rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable

from perception.tracker import DetectionBox, PerceptionRunResult, TrackSummary


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FragmentationStat:
    """Per-track fragmentation record."""

    track_id: int
    num_segments: int
    total_length: int
    longest_segment_length: int


@dataclass(frozen=True)
class TrackContinuityReport:
    """All diagnostics for one perception run.

    Every field is plain JSON-serialisable so the notebook can dump this
    straight to ``artifacts/perception/<clip_id>_report.json`` on Drive.
    """

    clip_id: str
    source_folder: str
    frame_count: int
    detection_count: int
    track_count: int
    longest_track_id: int | None
    longest_track_length: int
    primary_track_id: int | None
    id_switch_count: int
    fragmentation: tuple[FragmentationStat, ...]
    fps: float
    latency_ms_per_frame: float
    gpu_name: str | None
    fallback_used: str | None
    # Honest metric-availability flags. URFD has no GT so these are
    # always "n/a (no ground truth)" — included so downstream consumers
    # see the field and don't silently treat the run as comparable to
    # a GT-backed eval.
    metric_availability: dict[str, str] = field(default_factory=lambda: {
        "map_50": "n/a (no detection ground truth)",
        "map_50_95": "n/a (no detection ground truth)",
        "idf1": "n/a (no tracking ground truth)",
        "mota": "n/a (no tracking ground truth)",
        "hota": "n/a (no tracking ground truth)",
    })

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-ready dict (fragmentation rows expanded)."""
        raw = asdict(self)
        raw["fragmentation"] = [asdict(f) for f in self.fragmentation]
        return raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame_segments(frame_indices: tuple[int, ...]) -> list[tuple[int, int]]:
    """Split an ordered list of frame indices into disjoint [start, end] ranges.

    ``[1, 2, 3, 5, 6, 10]`` → ``[(1, 3), (5, 6), (10, 10)]``.
    Consecutive frame indices (delta == 1) join the same segment; gaps
    start a new one.
    """
    if not frame_indices:
        return []
    segments: list[tuple[int, int]] = []
    start = frame_indices[0]
    prev = start
    for index in frame_indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        segments.append((start, prev))
        start = index
        prev = index
    segments.append((start, prev))
    return segments


def _primary_track_per_frame(
    detections: Iterable[DetectionBox],
) -> list[int | None]:
    """Return the most-frequent track_id observed in each frame.

    Returns ``None`` for frames with no detections. Used as the
    "dominant ID" timeline for the ID-switch check; this is a structural
    proxy, not a formal MOT metric.

    When the input has NO tracked detections at all (every track_id is
    None), returns ``[None]`` — the timeline exists, every frame has no
    dominant track. An empty list would mean "the run produced no
    detections at all", which is information the downstream report needs.
    """
    by_frame: dict[int, Counter[int]] = {}
    highest_frame = -1
    for det in detections:
        highest_frame = max(highest_frame, det.frame_index)
        if det.track_id is None:
            continue
        by_frame.setdefault(det.frame_index, Counter())[det.track_id] += 1
    if highest_frame < 0:
        return [None]
    timeline: list[int | None] = []
    for frame_index in range(highest_frame + 1):
        counter = by_frame.get(frame_index)
        if not counter:
            timeline.append(None)
            continue
        timeline.append(counter.most_common(1)[0][0])
    return timeline


def _count_id_switches(timeline: list[int | None]) -> int:
    """Count transitions where the dominant track changes identity.

    A transition where one side is ``None`` (frame had no detections)
    is NOT counted as a switch — only ID-to-ID changes count.
    """
    switches = 0
    last_non_null: int | None = None
    for current in timeline:
        if current is None:
            continue
        if last_non_null is not None and current != last_non_null:
            switches += 1
        last_non_null = current
    return switches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_track_continuity_report(
    run: PerceptionRunResult,
    source_folder: str = "",
) -> TrackContinuityReport:
    """Build a :class:`TrackContinuityReport` from one :class:`PerceptionRunResult`.

    The report intentionally does NOT compute mAP / IDF1 / MOTA / HOTA —
    URFD has no ground truth and reporting them as numbers would be a
    fabrication (Issue 002 rule: "Be honest about metrics").
    """
    fragmentation = _build_fragmentation(run.tracks)
    longest = _longest_track(run.tracks)
    timeline = _primary_track_per_frame(run.detections)
    id_switches = _count_id_switches(timeline)
    primary = _primary_track(timeline)

    return TrackContinuityReport(
        clip_id=run.clip_id,
        source_folder=source_folder,
        frame_count=run.frame_count,
        detection_count=run.detection_count,
        track_count=run.track_count,
        longest_track_id=longest.track_id if longest else None,
        longest_track_length=longest.length if longest else 0,
        primary_track_id=primary,
        id_switch_count=id_switches,
        fragmentation=tuple(fragmentation),
        fps=run.fps,
        latency_ms_per_frame=run.latency_ms_per_frame,
        gpu_name=run.gpu_name,
        fallback_used=run.fallback_used,
    )


def _build_fragmentation(tracks: Iterable[TrackSummary]) -> list[FragmentationStat]:
    """Compute one :class:`FragmentationStat` per track, sorted by length desc."""
    rows: list[FragmentationStat] = []
    for track in tracks:
        segments = _frame_segments(track.frame_indices)
        longest_segment = max((end - start + 1) for start, end in segments)
        rows.append(FragmentationStat(
            track_id=track.track_id,
            num_segments=len(segments),
            total_length=track.length,
            longest_segment_length=longest_segment,
        ))
    rows.sort(key=lambda row: (-row.total_length, row.track_id))
    return rows


def _longest_track(tracks: Iterable[TrackSummary]) -> TrackSummary | None:
    """Return the longest track by total length (ties: lowest track_id)."""
    best: TrackSummary | None = None
    for track in tracks:
        if best is None or track.length > best.length or (
            track.length == best.length and track.track_id < best.track_id
        ):
            best = track
    return best


def _primary_track(timeline: list[int | None]) -> int | None:
    """Return the most-common dominant track across the timeline, or ``None``."""
    counter: Counter[int] = Counter(t for t in timeline if t is not None)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


__all__: tuple[str, ...] = (
    "FragmentationStat",
    "TrackContinuityReport",
    "build_track_continuity_report",
)