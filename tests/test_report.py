"""Unit tests for the perception track-continuity report.

No GPU / ultralytics dependency. Uses synthetic :class:`TrackSummary` and
:class:`DetectionBox` rows to verify the structural math (fragmentation,
ID switches, longest track, primary track).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.report import (  # noqa: E402
    TrackContinuityReport,
    _count_id_switches,
    _frame_segments,
    _primary_track,
    _primary_track_per_frame,
    build_track_continuity_report,
)
from perception.tracker import (  # noqa: E402
    DetectionBox,
    PerceptionRunResult,
    TrackerConfig,
    TrackSummary,
)


def _det(frame: int, track_id: int | None, confidence: float = 0.8) -> DetectionBox:
    """Tiny helper for synthesising detections."""
    return DetectionBox(
        frame_index=frame, track_id=track_id, cls_id=0,
        confidence=confidence, x_min=0, y_min=0, x_max=10, y_max=10,
    )


def _track(track_id: int, frames: list[int]) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        frame_indices=tuple(frames),
        confidences=tuple(0.8 for _ in frames),
    )


class FrameSegmentsTests(unittest.TestCase):
    """Disjoint [start, end] segments per track."""

    def test_continuous_indices_yield_one_segment(self) -> None:
        self.assertEqual(_frame_segments((1, 2, 3, 4)), [(1, 4)])

    def test_gap_starts_a_new_segment(self) -> None:
        self.assertEqual(_frame_segments((1, 2, 3, 5, 6)), [(1, 3), (5, 6)])

    def test_single_index_yields_one_segment(self) -> None:
        self.assertEqual(_frame_segments((7,)), [(7, 7)])

    def test_empty_yields_empty(self) -> None:
        self.assertEqual(_frame_segments(()), [])

    def test_caller_must_pass_sorted_indices(self) -> None:
        # The function does not re-sort — it treats the input as already
        # ordered. An unsorted input simply produces more (smaller)
        # segments, which is the expected behaviour. This test pins that
        # contract so a future "helpful" re-sort doesn't change downstream
        # fragmentation math silently.
        result = _frame_segments((3, 1, 2))
        # 3 starts a segment (no prev). 1 != 3+1 → close (3,3), open new
        # segment starting at 1. 2 == 1+1 → continue the segment.
        self.assertEqual(result, [(3, 3), (1, 2)])


class PrimaryTrackPerFrameTests(unittest.TestCase):
    """Per-frame dominant track id timeline."""

    def test_returns_dominant_track_per_frame(self) -> None:
        detections = [
            _det(0, 1), _det(0, 1),
            _det(1, 1), _det(1, 2),  # frame 1 ties → track 1 inserted first wins
            _det(2, 3),
        ]
        self.assertEqual(_primary_track_per_frame(detections), [1, 1, 3])

    def test_untracked_detections_are_ignored(self) -> None:
        detections = [_det(0, None), _det(0, None)]
        self.assertEqual(_primary_track_per_frame(detections), [None])

    def test_empty_input_returns_single_none(self) -> None:
        # "No detections anywhere" must still produce a non-empty timeline
        # of length 1 — callers can branch on len(timeline) instead of
        # having to special-case the empty-input branch.
        self.assertEqual(_primary_track_per_frame([]), [None])


class IdSwitchCountTests(unittest.TestCase):
    """Consecutive-frame dominant-track transitions."""

    def test_no_switch_when_same_track_keeps_dominating(self) -> None:
        self.assertEqual(_count_id_switches([1, 1, 1, 1]), 0)

    def test_switch_when_track_changes(self) -> None:
        self.assertEqual(_count_id_switches([1, 2]), 1)

    def test_null_gaps_do_not_count_as_switches(self) -> None:
        # Frame 1 has no detections; frame 0 and 2 share track 1 — no switch.
        self.assertEqual(_count_id_switches([1, None, 1]), 0)

    def test_empty_timeline_yields_zero_switches(self) -> None:
        self.assertEqual(_count_id_switches([]), 0)


class PrimaryTrackTests(unittest.TestCase):
    """Most-common dominant track across the timeline."""

    def test_picks_most_frequent(self) -> None:
        self.assertEqual(_primary_track([1, 2, 1, 1]), 1)

    def test_empty_timeline_returns_none(self) -> None:
        self.assertEqual(_primary_track([]), None)


class BuildReportTests(unittest.TestCase):
    """End-to-end: a synthetic PerceptionRunResult → TrackContinuityReport."""

    def _make_run(self) -> PerceptionRunResult:
        # Two tracks: track 1 is continuous (the falling person survived
        # the fall window), track 2 is fragmented (lost and re-acquired).
        detections = []
        for frame in range(10):
            detections.append(_det(frame, 1))
        for frame in [0, 1, 5, 6, 7]:
            detections.append(_det(frame, 2))
        return PerceptionRunResult(
            clip_id="test-clip",
            source_folder="datasets/urfd/fall-01-cam0",
            config=TrackerConfig(),
            detections=detections,
            tracks=[
                _track(1, list(range(10))),
                _track(2, [0, 1, 5, 6, 7]),
            ],
            frame_count=10,
            detection_count=len(detections),
            track_count=2,
            elapsed_seconds=0.5,
            gpu_name="NVIDIA T4",
            fallback_used=None,
        )

    def test_longest_track_is_picked(self) -> None:
        report = build_track_continuity_report(self._make_run())
        self.assertEqual(report.longest_track_id, 1)
        self.assertEqual(report.longest_track_length, 10)

    def test_fragmentation_is_computed(self) -> None:
        report = build_track_continuity_report(self._make_run())
        by_id = {f.track_id: f for f in report.fragmentation}
        self.assertEqual(by_id[1].num_segments, 1)
        self.assertEqual(by_id[1].longest_segment_length, 10)
        self.assertEqual(by_id[2].num_segments, 2)
        self.assertEqual(by_id[2].longest_segment_length, 3)

    def test_id_switch_count_reflects_dominant_track_changes(self) -> None:
        report = build_track_continuity_report(self._make_run())
        # Timeline of dominant track is [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # because track 1 has 1 detection per frame and track 2 has fewer.
        self.assertEqual(report.id_switch_count, 0)

    def test_metric_availability_admits_no_ground_truth(self) -> None:
        report = build_track_continuity_report(self._make_run())
        self.assertIn("map_50", report.metric_availability)
        self.assertEqual(
            report.metric_availability["idf1"], "n/a (no tracking ground truth)"
        )
        self.assertEqual(
            report.metric_availability["mota"], "n/a (no tracking ground truth)"
        )
        self.assertEqual(
            report.metric_availability["hota"], "n/a (no tracking ground truth)"
        )

    def test_fps_and_latency_propagate(self) -> None:
        report = build_track_continuity_report(self._make_run())
        self.assertAlmostEqual(report.fps, 20.0, places=5)
        self.assertAlmostEqual(report.latency_ms_per_frame, 50.0, places=5)
        self.assertEqual(report.gpu_name, "NVIDIA T4")


class ReportSerializationTests(unittest.TestCase):
    """``to_dict`` produces a JSON-ready shape."""

    def test_to_dict_expands_fragmentation(self) -> None:
        run = PerceptionRunResult(
            clip_id="x", source_folder="", config=TrackerConfig(),
            detections=[], tracks=[_track(1, [0, 1, 2])],
            frame_count=3, detection_count=0, track_count=1,
            elapsed_seconds=0.0, gpu_name=None, fallback_used=None,
        )
        report = build_track_continuity_report(run)
        payload = report.to_dict()
        self.assertEqual(payload["clip_id"], "x")
        self.assertIsInstance(payload["fragmentation"], list)
        self.assertEqual(payload["fragmentation"][0]["track_id"], 1)
        # Dataclass default_factory produces a fresh dict per instance.
        self.assertEqual(payload["metric_availability"]["hota"],
                         "n/a (no tracking ground truth)")


if __name__ == "__main__":
    unittest.main()