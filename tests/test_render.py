"""Unit tests for the perception annotated-frame renderer.

No GPU / ultralytics / Pillow write-out. Tests operate on synthetic
numpy arrays and assert on the drawn pixels directly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.render import (  # noqa: E402
    RenderConfig,
    annotate_frame,
    color_for_track_id,
)
from perception.tracker import DetectionBox  # noqa: E402


def _det(frame_index: int, track_id: int | None, confidence: float = 0.8,
         x_min: float = 10, y_min: float = 20, x_max: float = 80, y_max: float = 120,
         ) -> DetectionBox:
    return DetectionBox(
        frame_index=frame_index, track_id=track_id, cls_id=0,
        confidence=confidence, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
    )


class ColorForTrackIdTests(unittest.TestCase):
    """Track-ID → colour mapping is deterministic and well-separated."""

    def test_consecutive_ids_get_distinct_hues(self) -> None:
        c0 = color_for_track_id(0)
        c1 = color_for_track_id(1)
        self.assertNotEqual(c0, c1)
        # All channels in [0, 255]
        for channel in c0 + c1:
            self.assertGreaterEqual(channel, 0)
            self.assertLessEqual(channel, 255)

    def test_same_id_always_returns_same_colour(self) -> None:
        self.assertEqual(color_for_track_id(7), color_for_track_id(7))


class AnnotateFrameTests(unittest.TestCase):
    """Renderer mutates a copy of the input and respects the config."""

    def setUp(self) -> None:
        # Constant grey image so any drawn pixel is easy to spot.
        self.image = np.full((200, 200, 3), 128, dtype=np.uint8)

    def test_returns_a_copy_not_a_mutation(self) -> None:
        det = _det(0, track_id=1)
        annotated = annotate_frame(self.image, [det])
        # The original still has its uniform grey value.
        self.assertTrue(np.all(self.image == 128))
        # The annotated version differs somewhere.
        self.assertFalse(np.all(annotated == self.image))

    def test_box_outline_changes_pixels(self) -> None:
        det = _det(0, track_id=1, x_min=50, y_min=50, x_max=100, y_max=100)
        annotated = annotate_frame(self.image, [det])
        # A pixel on the top edge of the box must NOT still be grey 128.
        self.assertNotEqual(int(annotated[50, 60, 0]), 128)

    def test_untracked_detection_uses_fallback_color(self) -> None:
        det = _det(0, track_id=None, x_min=50, y_min=50, x_max=100, y_max=100)
        annotated = annotate_frame(self.image, [det])
        # The fallback is (255, 255, 255) — pure white.
        self.assertEqual(int(annotated[50, 60, 0]), 255)

    def test_degenerate_box_is_skipped_silently(self) -> None:
        # x_max <= x_min — degenerate. Renderer must not crash; pixels stay grey.
        det = _det(0, track_id=1, x_min=80, y_min=80, x_max=80, y_max=80)
        annotated = annotate_frame(self.image, [det])
        self.assertTrue(np.all(annotated == 128))

    def test_rejects_non_rgb_image(self) -> None:
        grey = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            annotate_frame(grey, [])

    def test_show_confidence_can_be_disabled(self) -> None:
        config = RenderConfig(show_confidence=False)
        det = _det(0, track_id=1)
        # Should not raise and should still draw the box.
        annotated = annotate_frame(self.image, [det], config=config)
        self.assertFalse(np.all(annotated == self.image))


if __name__ == "__main__":
    unittest.main()