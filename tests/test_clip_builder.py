"""Unit tests for :mod:`cropping.clip_builder`.

Pure tests — no I/O, no fixtures. The crop builder must be byte-deterministic
for any given input + config, so we can hash outputs to verify the
deterministic-output test (Issue 003 requirement).
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cropping.clip_builder import (  # noqa: E402
    MAX_MARGIN,
    MIN_MARGIN,
    CropConfig,
    apply_crop_to_frame,
    clip_box_to_image,
    compute_crop_geometry,
    expand_box,
    square_with_padding,
)


class ExpandBoxTests(unittest.TestCase):
    """Box expansion by a proportional margin."""

    def test_expansion_preserves_centre(self) -> None:
        x_min, y_min, x_max, y_max = expand_box(10, 20, 30, 60, margin=0.4)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        self.assertAlmostEqual(cx, 20.0)
        self.assertAlmostEqual(cy, 40.0)

    def test_expansion_grows_by_margin_times_longer_side_over_two(self) -> None:
        # 100x40 box; longer side = 100; margin 0.2 → pad = 10 on every side.
        x_min, y_min, x_max, y_max = expand_box(10, 10, 110, 50, margin=0.2)
        # The expanded box's side along the long axis should be 100 * 1.2 = 120.
        self.assertAlmostEqual(x_max - x_min, 120.0, places=5)
        self.assertAlmostEqual(y_max - y_min, 120.0, places=5)

    def test_zero_size_box_still_expands(self) -> None:
        # Degenerate zero-area input — must not crash. We clamp the
        # shorter side to >=1.0 internally so the box is a valid
        # 1x1 source.
        x_min, y_min, x_max, y_max = expand_box(5, 5, 5, 5, margin=0.3)
        self.assertGreater(x_max - x_min, 0.0)
        self.assertGreater(y_max - y_min, 0.0)


class ClipBoxToImageTests(unittest.TestCase):
    """Boxes that extend past the image edge are clamped, not shifted."""

    def test_fully_inside_box_is_unchanged(self) -> None:
        result = clip_box_to_image(10, 10, 50, 50, image_width=100, image_height=100)
        self.assertEqual(result, (10, 10, 50, 50))

    def test_partially_outside_box_is_clamped(self) -> None:
        # Box runs off the left and bottom of the image.
        result = clip_box_to_image(-20, -10, 50, 200,
                                    image_width=100, image_height=100)
        self.assertEqual(result, (0.0, 0.0, 50.0, 100.0))

    def test_fully_outside_box_collapses(self) -> None:
        # Box is entirely past the image bounds → empty box.
        result = clip_box_to_image(200, 200, 300, 300,
                                    image_width=100, image_height=100)
        self.assertEqual(result, (100.0, 100.0, 100.0, 100.0))


class SquareWithPaddingTests(unittest.TestCase):
    """Square-pad produces a centred square of side = max(w, h)."""

    def test_square_input_is_padded_by_zero(self) -> None:
        geom = square_with_padding(0, 0, 100, 100)
        self.assertEqual(geom.size, 100.0)
        self.assertEqual(geom.offset_x, 0.0)
        self.assertEqual(geom.offset_y, 0.0)

    def test_landscape_input_is_padded_vertically(self) -> None:
        # 200 wide × 100 tall → square is 200×200 with 50 vertical pad.
        geom = square_with_padding(0, 0, 200, 100)
        self.assertEqual(geom.size, 200.0)
        self.assertEqual(geom.offset_x, 0.0)
        self.assertEqual(geom.offset_y, 50.0)

    def test_portrait_input_is_padded_horizontally(self) -> None:
        # 100 wide × 200 tall → square is 200×200 with 50 horizontal pad.
        geom = square_with_padding(0, 0, 100, 200)
        self.assertEqual(geom.size, 200.0)
        self.assertEqual(geom.offset_x, 50.0)
        self.assertEqual(geom.offset_y, 0.0)


class ComputeCropGeometryTests(unittest.TestCase):
    """End-to-end expand → clip → square for a normal box."""

    def test_centre_box_inside_image(self) -> None:
        geom = compute_crop_geometry(
            40, 40, 60, 60, margin=0.3, image_width=100, image_height=100,
        )
        # 20x20 box, longer side 20, pad = 0.3 * 10 = 3 → square side 26.
        self.assertAlmostEqual(geom.size, 26.0, places=5)


class CropConfigTests(unittest.TestCase):
    """CropConfig validates its own fields at construction time."""

    def test_default_values_match_prd_contract(self) -> None:
        config = CropConfig()
        self.assertEqual(config.output_size, 224)
        self.assertEqual(config.margin, 0.30)
        self.assertEqual(config.clip_length, 32)

    def test_margin_below_minimum_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CropConfig(margin=MIN_MARGIN - 0.01)

    def test_margin_above_maximum_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CropConfig(margin=MAX_MARGIN + 0.01)

    def test_clip_length_must_be_16_or_32(self) -> None:
        with self.assertRaises(ValueError):
            CropConfig(clip_length=24)

    def test_output_size_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            CropConfig(output_size=0)


class ApplyCropToFrameTests(unittest.TestCase):
    """Per-frame cropping produces the right shape and never mutates input."""

    def setUp(self) -> None:
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Paint a white square at (50,50)-(100,100) so cropping can be
        # observed in the resulting array.
        self.frame[50:100, 50:100] = 255

    def test_output_shape_matches_config(self) -> None:
        config = CropConfig(output_size=64, margin=0.3, clip_length=32)
        geom = compute_crop_geometry(
            60, 60, 90, 90, margin=config.margin,
            image_width=self.frame.shape[1], image_height=self.frame.shape[0],
        )
        crop = apply_crop_to_frame(self.frame, geom, output_size=config.output_size)
        self.assertEqual(crop.shape, (64, 64, 3))

    def test_input_is_not_mutated(self) -> None:
        config = CropConfig(output_size=64, margin=0.3)
        geom = compute_crop_geometry(
            60, 60, 90, 90, margin=config.margin,
            image_width=self.frame.shape[1], image_height=self.frame.shape[0],
        )
        snapshot = self.frame.copy()
        _ = apply_crop_to_frame(self.frame, geom, output_size=config.output_size)
        self.assertTrue(np.array_equal(self.frame, snapshot),
                         msg="apply_crop_to_frame must not mutate input")

    def test_degenerate_geometry_returns_uniform_canvas(self) -> None:
        # Box entirely outside the image → empty geometry → uniform canvas.
        geom = compute_crop_geometry(
            9999, 9999, 9999, 9999, margin=0.3,
            image_width=self.frame.shape[1], image_height=self.frame.shape[0],
        )
        crop = apply_crop_to_frame(self.frame, geom, output_size=64)
        # All zeros because the source was zero and the canvas is zero-padded.
        self.assertTrue(np.all(crop == 0))

    def test_non_3d_frame_raises(self) -> None:
        geom = compute_crop_geometry(0, 0, 10, 10, margin=0.3,
                                       image_width=100, image_height=100)
        with self.assertRaises(ValueError):
            apply_crop_to_frame(np.zeros((10, 10), dtype=np.uint8), geom, output_size=64)


class DeterminismTests(unittest.TestCase):
    """Same input + same config → same crop pixels (Issue 003 hard rule)."""

    def test_same_input_yields_identical_pixels(self) -> None:
        frame = np.random.default_rng(42).integers(0, 256, size=(300, 300, 3),
                                                    dtype=np.uint8)
        config = CropConfig(output_size=128, margin=0.25, clip_length=32)
        geom = compute_crop_geometry(
            80, 80, 200, 220, margin=config.margin,
            image_width=frame.shape[1], image_height=frame.shape[0],
        )
        crop_a = apply_crop_to_frame(frame, geom, output_size=config.output_size)
        crop_b = apply_crop_to_frame(frame, geom, output_size=config.output_size)
        # Byte-equal: the hash is a quick stand-in for full equality.
        self.assertEqual(_hash(crop_a), _hash(crop_b))

    def test_resize_is_deterministic_across_pixel_layouts(self) -> None:
        # The nearest-neighbour resize should be order-stable — same
        # input always produces same output, regardless of how we
        # build the input.
        src = np.full((8, 4, 3), 128, dtype=np.uint8)
        out_a = apply_crop_to_frame(
            src,
            compute_crop_geometry(0, 0, 4, 8, margin=0.0,
                                    image_width=4, image_height=8),
            output_size=16,
        )
        out_b = apply_crop_to_frame(
            src,
            compute_crop_geometry(0, 0, 4, 8, margin=0.0,
                                    image_width=4, image_height=8),
            output_size=16,
        )
        self.assertEqual(_hash(out_a), _hash(out_b))


class EdgeClippedCropTests(unittest.TestCase):
    """Issue 003 review: when the expanded box runs past an image edge,
    the crop must (a) slice the exact clipped image region, (b) paste
    it onto a ``size x size`` canvas at the correct pad offsets, and
    (c) leave the rest of the canvas as the configured pad value.

    Each test paints the source frame with a deterministic pattern
    (white box on black background) so the assertion can check both
    that the white box appears at the right slice AND that the pad
    regions are pure black.

    Geometry note: ``expand_box`` centres the box and extends by
    ``half = longer_side / 2`` even when ``margin=0``. So a box at
    (0, 24, 8, 40) (width 8, height 16) is expanded to (-4, 24, 12, 40)
    and clipped to (0, 24, 12, 40) — clipped_width=12, height=16. The
    tests below derive expected values from that math rather than
    hard-coding them.
    """

    IMAGE_SIZE = 64
    PAD_VALUE = 0
    OUTPUT_SIZE = 32

    def _paint_frame(self, white_box: tuple[int, int, int, int]) -> np.ndarray:
        """Black frame with one white rectangle inside ``white_box``."""
        frame = np.full((self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
                        self.PAD_VALUE, dtype=np.uint8)
        x_min, y_min, x_max, y_max = white_box
        frame[y_min:y_max, x_min:x_max] = 255
        return frame

    def _expanded_dims(
        self, x_min: float, y_min: float, x_max: float, y_max: float,
    ) -> tuple[float, float]:
        """Return (clipped_width, clipped_height) for a margin=0 geometry.

        We re-derive from first principles rather than calling
        ``compute_crop_geometry`` so the test pins the math, not the
        implementation: a wider half-side on the long axis extends the
        box ±half_side beyond the original centre even when margin=0.
        """
        longer = max(x_max - x_min, y_max - y_min)
        half = longer / 2.0
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        # Clamp to image rectangle.
        cl_x_min = max(0.0, min(self.IMAGE_SIZE, cx - half))
        cl_y_min = max(0.0, min(self.IMAGE_SIZE, cy - half))
        cl_x_max = max(0.0, min(self.IMAGE_SIZE, cx + half))
        cl_y_max = max(0.0, min(self.IMAGE_SIZE, cy + half))
        return cl_x_max - cl_x_min, cl_y_max - cl_y_min

    def test_fully_in_bounds_box_unchanged(self) -> None:
        # White box well inside the image; expanded box fits comfortably.
        frame = self._paint_frame((20, 20, 44, 44))
        geom = compute_crop_geometry(24, 24, 40, 40, margin=0.0,
                                       image_width=self.IMAGE_SIZE,
                                       image_height=self.IMAGE_SIZE)
        crop = apply_crop_to_frame(frame, geom, output_size=self.OUTPUT_SIZE)
        non_pad = crop[crop != self.PAD_VALUE]
        # The crop contains a non-pad region (the white box) and no
        # garbage pixels.
        self.assertGreater(non_pad.size, 0)
        self.assertTrue(np.all(non_pad == 255),
                         msg=f"non-pad pixels must be pure white, got {np.unique(non_pad)}")

    def test_left_edge_clipped(self) -> None:
        # White box at x=0..8 in the image. Expanded box runs off the
        # left side; the crop must contain the WHITE PORTION (columns
        # 0..clipped_width of the source), padded on the right with the
        # pad value.
        frame = self._paint_frame((0, 24, 8, 40))
        geom = compute_crop_geometry(0, 24, 8, 40, margin=0.0,
                                       image_width=self.IMAGE_SIZE,
                                       image_height=self.IMAGE_SIZE)
        # The clipped region is shorter than the square canvas, so a
        # pad band exists on either side of it. We don't assert exact
        # dimensions here — we assert the BEHAVIOUR (no garbage source
        # pixels, pad bands are pure pad).
        cl_w, _ = self._expanded_dims(0, 24, 8, 40)
        self.assertLess(cl_w, geom.size)  # clipped region is narrower than the canvas

        crop = apply_crop_to_frame(frame, geom, output_size=self.OUTPUT_SIZE)
        # The crop must contain only pad (0) or white (255). Anything
        # else means the slice over-extended into the image.
        unique_values = np.unique(crop)
        self.assertTrue(
            set(unique_values.tolist()).issubset({self.PAD_VALUE, 255}),
            msg=f"left-edge crop must contain only pad and white; got {unique_values}",
        )
        # The clipped region is wider than 0 — there ARE white pixels.
        non_pad = crop[crop != self.PAD_VALUE]
        self.assertGreater(non_pad.size, 0)
        self.assertTrue(np.all(non_pad == 255))

    def test_right_edge_clipped(self) -> None:
        # White box at x=56..64 in the image (right edge). Expanded
        # box runs off the right; the clipped region is shorter than
        # the canvas width.
        frame = self._paint_frame((56, 24, 64, 40))
        geom = compute_crop_geometry(56, 24, 64, 40, margin=0.0,
                                       image_width=self.IMAGE_SIZE,
                                       image_height=self.IMAGE_SIZE)
        cl_w, _ = self._expanded_dims(56, 24, 64, 40)
        self.assertLess(cl_w, geom.size)

        crop = apply_crop_to_frame(frame, geom, output_size=self.OUTPUT_SIZE)
        unique_values = np.unique(crop)
        self.assertTrue(
            set(unique_values.tolist()).issubset({self.PAD_VALUE, 255}),
            msg=f"right-edge crop must contain only pad and white; got {unique_values}",
        )
        non_pad = crop[crop != self.PAD_VALUE]
        self.assertGreater(non_pad.size, 0)
        self.assertTrue(np.all(non_pad == 255))

    def test_top_edge_clipped(self) -> None:
        # White box at y=0..8 (top edge). Vertical clip.
        frame = self._paint_frame((24, 0, 40, 8))
        geom = compute_crop_geometry(24, 0, 40, 8, margin=0.0,
                                       image_width=self.IMAGE_SIZE,
                                       image_height=self.IMAGE_SIZE)
        _, cl_h = self._expanded_dims(24, 0, 40, 8)
        self.assertLess(cl_h, geom.size)

        crop = apply_crop_to_frame(frame, geom, output_size=self.OUTPUT_SIZE)
        unique_values = np.unique(crop)
        self.assertTrue(
            set(unique_values.tolist()).issubset({self.PAD_VALUE, 255}),
            msg=f"top-edge crop must contain only pad and white; got {unique_values}",
        )
        non_pad = crop[crop != self.PAD_VALUE]
        self.assertGreater(non_pad.size, 0)
        self.assertTrue(np.all(non_pad == 255))

    def test_bottom_edge_clipped(self) -> None:
        # White box at y=56..64 (bottom edge). Vertical clip.
        frame = self._paint_frame((24, 56, 40, 64))
        geom = compute_crop_geometry(24, 56, 40, 64, margin=0.0,
                                       image_width=self.IMAGE_SIZE,
                                       image_height=self.IMAGE_SIZE)
        _, cl_h = self._expanded_dims(24, 56, 40, 64)
        self.assertLess(cl_h, geom.size)

        crop = apply_crop_to_frame(frame, geom, output_size=self.OUTPUT_SIZE)
        unique_values = np.unique(crop)
        self.assertTrue(
            set(unique_values.tolist()).issubset({self.PAD_VALUE, 255}),
            msg=f"bottom-edge crop must contain only pad and white; got {unique_values}",
        )
        non_pad = crop[crop != self.PAD_VALUE]
        self.assertGreater(non_pad.size, 0)
        self.assertTrue(np.all(non_pad == 255))

    def test_corner_clipped(self) -> None:
        # White box at the top-left corner — clipped on BOTH top and
        # left. The clipped region is square (8x8); the canvas is 8x8
        # (size = max(8, 8)). No pad is needed because the clipped
        # region IS the square.
        frame = self._paint_frame((0, 0, 8, 8))
        geom = compute_crop_geometry(0, 0, 8, 8, margin=0.0,
                                       image_width=self.IMAGE_SIZE,
                                       image_height=self.IMAGE_SIZE)
        cl_w, cl_h = self._expanded_dims(0, 0, 8, 8)
        self.assertEqual(cl_w, geom.size)
        self.assertEqual(cl_h, geom.size)
        crop = apply_crop_to_frame(frame, geom, output_size=self.OUTPUT_SIZE)
        # The whole crop must be white — no pad needed.
        self.assertTrue(np.all(crop == 255),
                         msg=f"corner-clip crop must be pure white, got {np.unique(crop)}")

    def test_two_edges_clipped_with_pad(self) -> None:
        # White box at top-left with the wide axis horizontal — so the
        # clipped region is wider than it is tall, and the canvas has
        # BOTH a vertical pad band AND a clipped region. This is the
        # shape of a falling person near the frame corner.
        frame = self._paint_frame((0, 0, 16, 8))
        geom = compute_crop_geometry(0, 0, 16, 8, margin=0.0,
                                       image_width=self.IMAGE_SIZE,
                                       image_height=self.IMAGE_SIZE)
        cl_w, cl_h = self._expanded_dims(0, 0, 16, 8)
        # Width equals size (clipped region fits horizontally); height
        # is smaller → vertical pad exists.
        self.assertEqual(cl_w, geom.size)
        self.assertLess(cl_h, geom.size)
        crop = apply_crop_to_frame(frame, geom, output_size=self.OUTPUT_SIZE)
        # Crop contains only pad and white.
        unique_values = np.unique(crop)
        self.assertTrue(
            set(unique_values.tolist()).issubset({self.PAD_VALUE, 255}),
            msg=f"two-edge crop must contain only pad and white; got {unique_values}",
        )
        non_pad = crop[crop != self.PAD_VALUE]
        self.assertGreater(non_pad.size, 0)
        self.assertTrue(np.all(non_pad == 255))

    def test_no_garbage_source_pixels_pulled_in(self) -> None:
        # The previous (buggy) implementation sliced by ``size`` from
        # (x_min, y_min), which would over-extend past the image edge
        # and pull in whatever happens to be there in memory. Verify
        # that the fix doesn't.
        frame = self._paint_frame((0, 24, 8, 40))  # white at left edge
        # Build a much wider sentinel buffer; only the first 64 cols
        # are "the image". Mark columns 64..200 with a distinctive
        # sentinel value (99) that must NOT appear in the crop.
        sentinel = np.full((64, 200, 3), 99, dtype=np.uint8)
        sentinel[:, 0:64] = frame
        # The buggy implementation would have sliced into sentinel[:,64:200]
        # because it used size (which exceeds the visible width). The
        # fixed implementation only slices the visible 64-wide region.
        geom = compute_crop_geometry(0, 24, 8, 40, margin=0.0,
                                       image_width=64, image_height=64)
        crop = apply_crop_to_frame(sentinel[:, 0:64], geom,
                                     output_size=self.OUTPUT_SIZE)
        unique_values = np.unique(crop)
        self.assertNotIn(
            99, unique_values,
            msg=f"sentinel value 99 leaked into the crop: {unique_values}",
        )


def _hash(array: np.ndarray) -> str:
    """Stable content hash for byte-equal comparison."""
    return hashlib.sha256(array.tobytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()