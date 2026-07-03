"""Pure crop math for the Issue 003 crop clip generator.

This module owns the geometric transformations that turn a tracking
bounding box on a raw frame into a fixed-size, fixed-margin crop ready
for classifier consumption. It does NOT read frames, run any model, or
touch the filesystem — those live in :mod:`cropping.track_windows`,
:mod:`cropping.shard_writer`, and :mod:`cropping.runner`.

Pipeline (per box on per frame):

    1. **Expand** the box by a configurable ``margin`` (PRD: 0.20–0.40).
       The expanded box retains floor / limb context — a tight crop of a
       standing person can't tell a fall from a sit.
    2. **Clip** the expanded box to the image bounds. The box is
       allowed to be partially outside the image; we clamp instead of
       shifting, so the same world-space box on a slightly shifted
       frame produces the same crop centre (deterministic).
    3. **Pad** the clipped box up to a square. Padding goes on the side
       where the original box ran off the image, so the person stays
       centred in the canvas.
    4. **Resize** the square crop to the configured target size
       (default 224×224, matching VideoMAE / most classifier heads).

All transforms are pure functions; the same input always produces the
same output. Tests in ``tests/test_clip_builder.py`` pin every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Lower / upper bound on the margin (PRD: configurable in [0.20, 0.40]).
MIN_MARGIN: float = 0.20
MAX_MARGIN: float = 0.40


@dataclass(frozen=True)
class CropGeometry:
    """Square crop box in source-frame coordinates.

    The crop is laid out as:

        Source frame:           Padded square canvas (size x size):

        +-----------+----+        +----+----------------+----+
        |  pad-top  |    |        | p  |  clipped (W,H) | p  |
        +-----------+----+        | a  |                | a  |
        | clipped   | p  |        | d  |  (offset_x,    | d  |
        | (W x H)   | a  |  -->   | -L |   offset_y)    | -R |
        | at (x, y) | d  |        |    |                |    |
        +-----------+----+        +----+----------------+----+
        |  pad-bot  |    |
        +-----------+----+

    Fields (all pixels):
        ``x_min``, ``y_min``      — top-left of the CLIPPED region in the
                                   source frame.
        ``clipped_width``,
        ``clipped_height``        — exact dimensions of the CLIPPED region.
                                   Equals the box's (clamped) width/height
                                   AFTER the image-bound clamp. May be
                                   smaller than the expanded-box dimensions
                                   when the expanded box extends past an
                                   image edge.
        ``size``                  — side length of the square padded
                                   canvas. Equals ``max(clipped_width,
                                   clipped_height)``.
        ``offset_x``, ``offset_y`` — where the clipped region is pasted
                                   onto the canvas (pad on the left/top).
                                   Symmetric pad on the right/bottom is
                                   ``size - clipped_width - offset_x``.

    Why we carry the clipped dimensions explicitly: in a previous version,
    :func:`apply_crop_to_frame` sliced by ``size`` from ``(x_min, y_min)``,
    which over-extended the slice when the expanded box was clipped on
    one or two edges. That pulled unrelated source pixels into the crop
    and made edge cases (a falling person near the frame boundary)
    visually wrong. Carrying the clipped dimensions fixes this.

    Default values: ``clipped_width`` and ``clipped_height`` default to
    ``size`` so a geometry constructed without them behaves as a
    square, no-clipping case (backward compatibility for direct
    constructors in tests).
    """

    x_min: float
    y_min: float
    size: float
    offset_x: float
    offset_y: float
    clipped_width: float = 0.0
    clipped_height: float = 0.0

    def __post_init__(self) -> None:
        # Backward-compat: if the caller didn't set the clipped dims,
        # assume the geometry is square / no-clipping (clipped_width =
        # clipped_height = size). This keeps the dataclass usable for
        # direct construction in tests.
        if self.clipped_width == 0.0 and self.clipped_height == 0.0:
            object.__setattr__(self, "clipped_width", self.size)
            object.__setattr__(self, "clipped_height", self.size)


@dataclass(frozen=True)
class CropConfig:
    """All knobs that affect crop geometry.

    Defaults match the PRD's recommended fixed clip contract:
    16-or-32 frames × 224×224 with a 20–40% margin. We pick 32 frames
    here because VideoMAE's smaller variant accepts 32-frame windows;
    callers can override.
    """

    output_size: int = 224
    margin: float = 0.30  # mid-range of the PRD's [0.20, 0.40]
    clip_length: int = 32

    def __post_init__(self) -> None:
        if not MIN_MARGIN <= self.margin <= MAX_MARGIN:
            raise ValueError(
                f"margin must be in [{MIN_MARGIN}, {MAX_MARGIN}], got {self.margin}."
            )
        if self.clip_length not in (16, 32):
            raise ValueError(
                f"clip_length must be 16 or 32 (PRD contract), got {self.clip_length}."
            )
        if self.output_size <= 0:
            raise ValueError(f"output_size must be positive, got {self.output_size}.")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def expand_box(
    x_min: float, y_min: float, x_max: float, y_max: float,
    margin: float,
) -> tuple[float, float, float, float]:
    """Expand a bounding box by ``margin`` (proportional to its size).

    The expanded box has its centre preserved; each side moves outward by
    ``margin * max(width, height) / 2``. Using the longer side keeps the
    margin relative to the most informative dimension — a tall standing
    person needs more vertical context, a lying one needs more
    horizontal context.
    """
    width = max(x_max - x_min, 0.0)
    height = max(y_max - y_min, 0.0)
    longer = max(width, height, 1.0)
    pad = margin * longer / 2.0
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half = longer / 2.0 + pad
    return (cx - half, cy - half, cx + half, cy + half)


def clip_box_to_image(
    x_min: float, y_min: float, x_max: float, y_max: float,
    image_width: int, image_height: int,
) -> tuple[float, float, float, float]:
    """Clamp a box to the image rectangle without shifting its centre.

    A box that extends past the image edge becomes a smaller box that
    touches the edge. The centre moves only by the over-extension —
    this preserves determinism because we never *translate* the box,
    we only ever shrink it.
    """
    return (
        max(0.0, min(float(image_width), x_min)),
        max(0.0, min(float(image_height), y_min)),
        max(0.0, min(float(image_width), x_max)),
        max(0.0, min(float(image_height), y_max)),
    )


def square_with_padding(
    x_min: float, y_min: float, x_max: float, y_max: float,
) -> CropGeometry:
    """Turn an axis-aligned rectangle into a square with side-padded canvas.

    Padding is added so the original box keeps its position in the
    larger square — the left/top pad equals the right/bottom shortfall.
    Output is a :class:`CropGeometry` carrying the original box origin,
    the clipped dimensions (== the input box's dimensions), the new
    square size, and the pad offsets to apply at resize time.
    """
    clipped_width = max(x_max - x_min, 0.0)
    clipped_height = max(y_max - y_min, 0.0)
    size = max(clipped_width, clipped_height)
    # Centre the original box on the square; offset_x/y is how much pad
    # was added on the left/top before drawing the box at (x_min, y_min).
    offset_x = (size - clipped_width) / 2.0
    offset_y = (size - clipped_height) / 2.0
    return CropGeometry(
        x_min=x_min, y_min=y_min, size=size,
        offset_x=offset_x, offset_y=offset_y,
        clipped_width=clipped_width,
        clipped_height=clipped_height,
    )


def compute_crop_geometry(
    x_min: float, y_min: float, x_max: float, y_max: float,
    margin: float,
    image_width: int, image_height: int,
) -> CropGeometry:
    """End-to-end: expand → clip → square + pad.

    Returns the :class:`CropGeometry` ready for :func:`apply_crop_to_frame`.
    """
    expanded = expand_box(x_min, y_min, x_max, y_max, margin)
    clipped = clip_box_to_image(*expanded, image_width=image_width,
                                image_height=image_height)
    return square_with_padding(*clipped)


# ---------------------------------------------------------------------------
# Pixel operations
# ---------------------------------------------------------------------------


def apply_crop_to_frame(
    frame: np.ndarray,
    geometry: CropGeometry,
    output_size: int,
    pad_value: int = 0,
) -> np.ndarray:
    """Apply a :class:`CropGeometry` to a single frame.

    Steps:
        1. Slice the source region from ``frame`` using the clipped
           dimensions — NOT the square ``size`` — so we never pull
           unrelated source pixels into the crop when the expanded box
           ran past an image edge.
        2. Embed the clipped region in a ``size x size`` canvas, padded
           with ``pad_value`` so the crop is centred at the configured
           offsets.
        3. Resize the canvas to ``(output_size, output_size)`` via
           nearest-neighbour interpolation (no extra deps, deterministic).

    Returns a new ``(output_size, output_size, C)`` array — never mutates
    the input.
    """
    if frame.ndim != 3:
        raise ValueError(f"frame must be HxWxC, got shape {frame.shape}.")
    height, width = frame.shape[:2]
    channels = frame.shape[2]

    # The CLIPPED region — slice the exact image-rectangle we know is
    # valid. Use clipped_width/height, NOT size, so we never grab
    # pixels outside the image (which would either crash or silently
    # pull junk into the crop).
    x_min = int(max(0, min(width, round(geometry.x_min))))
    y_min = int(max(0, min(height, round(geometry.y_min))))
    x_max = int(max(0, min(width, round(geometry.x_min + geometry.clipped_width))))
    y_max = int(max(0, min(height, round(geometry.y_min + geometry.clipped_height))))

    if x_max <= x_min or y_max <= y_min:
        # Degenerate — return an empty canvas so downstream code can
        # detect the bad case via uniform pixels rather than crash.
        return np.full((output_size, output_size, channels), pad_value,
                       dtype=frame.dtype)

    region = frame[y_min:y_max, x_min:x_max]
    # Defensive: if rounding collapsed the region, fall back to a
    # uniform canvas rather than crash on a malformed paste.
    if region.shape[0] == 0 or region.shape[1] == 0:
        return np.full((output_size, output_size, channels), pad_value,
                       dtype=frame.dtype)

    canvas_size = int(round(geometry.size))
    if canvas_size <= 0:
        return np.full((output_size, output_size, channels), pad_value,
                       dtype=frame.dtype)

    canvas = np.full((canvas_size, canvas_size, channels), pad_value,
                     dtype=frame.dtype)
    paste_x = int(round(geometry.offset_x))
    paste_y = int(round(geometry.offset_y))
    # Clip the paste region to the canvas in case rounding overshot.
    # Both width AND height of the region must be clipped — using only
    # one dimension produces a shape mismatch on the assignment below.
    paste_w = min(region.shape[1], canvas_size - paste_x)
    paste_h = min(region.shape[0], canvas_size - paste_y)
    if paste_w <= 0 or paste_h <= 0:
        return np.full((output_size, output_size, channels), pad_value,
                       dtype=frame.dtype)
    canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = \
        region[:paste_h, :paste_w]

    return _resize_nearest_neighbour(canvas, output_size, output_size)


def _resize_nearest_neighbour(
    image: np.ndarray, new_height: int, new_width: int,
) -> np.ndarray:
    """Deterministic nearest-neighbour resize — no scipy / PIL required.

    For each output pixel ``(i, j)``, sample the source pixel
    ``(int(i * H / new_h), int(j * W / new_w))``. Pure, dependency-free,
    deterministic — exactly what tests need.
    """
    src_h, src_w = image.shape[:2]
    if src_h == new_height and src_w == new_width:
        return image.copy()
    # Build coordinate grids by integer multiplication then integer divide
    # so the math is bit-exact across platforms (no float-rounding drift).
    y_indices = (np.arange(new_height, dtype=np.int64) * src_h // new_height)
    x_indices = (np.arange(new_width, dtype=np.int64) * src_w // new_width)
    return image[y_indices[:, None], x_indices[None, :]].copy()


__all__: tuple[str, ...] = (
    "MIN_MARGIN",
    "MAX_MARGIN",
    "CropConfig",
    "CropGeometry",
    "apply_crop_to_frame",
    "clip_box_to_image",
    "compute_crop_geometry",
    "expand_box",
    "square_with_padding",
)