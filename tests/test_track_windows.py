"""Unit tests for :mod:`cropping.track_windows`.

Covers gap policy, short-track policy, minimum-coverage gate, and
deterministic output (same input → same windows).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cropping.clip_builder import CropConfig  # noqa: E402
from cropping.track_windows import (  # noqa: E402
    SkipReason,
    TrackWindow,
    TrackedBox,
    WindowBuildResult,
    build_windows_for_track,
    group_boxes_by_track,
)


def _box(frame_index: int, x_min: float = 10, y_min: float = 10,
          x_max: float = 50, y_max: float = 60, confidence: float = 0.8) -> TrackedBox:
    return TrackedBox(
        frame_index=frame_index,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        confidence=confidence,
    )


class EmptyTrackTests(unittest.TestCase):
    """A track with no boxes produces one skip reason, no emissions."""

    def test_empty_track_is_skipped(self) -> None:
        result = build_windows_for_track(
            track_id=7, boxes=[], config=CropConfig(),
        )
        self.assertEqual(result.emitted, ())
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].reason, "empty_track")
        self.assertEqual(result.source_box_count, 0)


class ShortTrackTests(unittest.TestCase):
    """A track with fewer boxes than clip_length fails the gap gate."""

    def test_short_track_emits_no_windows(self) -> None:
        # 5 boxes, clip_length 32, default gap_tolerance 4. The single
        # candidate window has 5 real + 27 consecutive missing frames,
        # which exceeds the tolerance → window dropped with ``too_many_gaps``.
        boxes = [_box(i) for i in range(5)]
        result = build_windows_for_track(
            track_id=1, boxes=boxes, config=CropConfig(),
        )
        self.assertEqual(result.emitted, ())
        # The first candidate window aborts due to too_many_gaps.
        gap_drops = [s for s in result.skipped if s.reason == "too_many_gaps"]
        self.assertGreater(len(gap_drops), 0)


class GapPolicyTests(unittest.TestCase):
    """Frame-index gaps inside a window are filled by carry-forward geometry
    until the gap tolerance is exceeded, then the window is dropped."""

    def test_consecutive_gaps_above_tolerance_drop_window(self) -> None:
        # 10 boxes followed by 6 gaps (default tolerance 4) → window dropped.
        boxes = [_box(i) for i in range(10)]
        result = build_windows_for_track(
            track_id=1, boxes=boxes, config=CropConfig(clip_length=32),
            gap_tolerance=4,
        )
        # The first window (start=0) has 32 indices: 10 real + 22 missing.
        # 22 consecutive missing > tolerance 4 → window dropped with
        # ``too_many_gaps``.
        gap_drops = [s for s in result.skipped if s.reason == "too_many_gaps"]
        self.assertGreater(len(gap_drops), 0)

    def test_short_gap_emits_window_with_low_coverage(self) -> None:
        # 30 contiguous boxes; clip_length=32 → one window starts at
        # frame 0 covering frames 0..31. Frames 30..31 are missing
        # (2 consecutive missing, well within tolerance 4). Window
        # emitted with 2 missing slots and coverage 30/32.
        boxes = [_box(i) for i in range(30)]
        result = build_windows_for_track(
            track_id=1, boxes=boxes, config=CropConfig(clip_length=32),
            gap_tolerance=4,
        )
        self.assertEqual(len(result.emitted), 1)
        emitted = result.emitted[0]
        self.assertEqual(len(emitted.missing_frames), 2)
        self.assertAlmostEqual(emitted.coverage, 30 / 32, places=5)


class MinCoverageTests(unittest.TestCase):
    """Tiny boxes fail the min-coverage gate."""

    def test_tiny_boxes_are_dropped(self) -> None:
        # 60 contiguous 8x8 boxes; clip_length=32, stride=32.
        # - Window 1 (frames 0..31): all real, avg area 64 < 1024 → dropped.
        # - Window 2 (frames 32..63): 28 real + 4 missing (within tolerance),
        #   avg area still 64 → dropped.
        boxes = [
            TrackedBox(
                frame_index=i, x_min=0, y_min=0, x_max=8, y_max=8, confidence=0.5,
            )
            for i in range(60)
        ]
        result = build_windows_for_track(
            track_id=1, boxes=boxes, config=CropConfig(),
        )
        self.assertEqual(result.emitted, ())
        # Both windows dropped via the coverage gate (not via gap abort).
        coverage_drops = [s for s in result.skipped if s.reason == "insufficient_coverage"]
        self.assertGreaterEqual(len(coverage_drops), 1)


class DeterminismTests(unittest.TestCase):
    """Same input → same windows (Issue 003 hard rule)."""

    def test_identical_input_produces_identical_windows(self) -> None:
        boxes = [_box(i, x_min=i % 30, y_min=10, x_max=i % 30 + 50, y_max=70)
                  for i in range(60)]
        run_a = build_windows_for_track(track_id=1, boxes=boxes,
                                           config=CropConfig())
        run_b = build_windows_for_track(track_id=1, boxes=boxes,
                                           config=CropConfig())
        self.assertEqual(_window_keys(run_a.emitted), _window_keys(run_b.emitted))

    def test_unsorted_input_is_normalised(self) -> None:
        # Caller might pass boxes out of order; we should sort by frame_index
        # defensively so two runs over the same logical track produce the
        # same windows.
        ordered = [_box(i) for i in range(60)]
        shuffled = list(reversed(ordered))
        run_a = build_windows_for_track(track_id=1, boxes=ordered,
                                           config=CropConfig())
        run_b = build_windows_for_track(track_id=1, boxes=shuffled,
                                           config=CropConfig())
        self.assertEqual(_window_keys(run_a.emitted), _window_keys(run_b.emitted))


class GroupBoxesByTrackTests(unittest.TestCase):
    """The perception → cropping mapping helper."""

    def test_grouping_round_trip(self) -> None:
        boxes_by_track = {
            1: [_box(0), _box(1)],
            2: [_box(5)],
        }
        grouped = group_boxes_by_track(boxes_by_track)
        self.assertEqual(len(grouped[1]), 2)
        self.assertEqual(len(grouped[2]), 1)


def _window_keys(windows: tuple[TrackWindow, ...]) -> list[tuple]:
    """Stable hashable representation of emitted windows for equality checks."""
    return [(w.track_id, w.frame_indices, w.missing_frames,
              round(w.coverage, 6)) for w in windows]


if __name__ == "__main__":
    unittest.main()