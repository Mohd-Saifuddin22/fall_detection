"""Annotated-frame renderer for the perception front-end.

Draws per-track bounding boxes + track IDs on top of frame pixels so a
human can visually verify that ByteTrack held a stable ID through the
fall window. The renderer is pure with respect to the tracker — it
takes already-decoded :class:`DetectionBox` rows plus the frame image
and returns the annotated image.

No ultralytics dependency here so this module can be unit-tested on
synthetic numpy arrays without a GPU.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from perception.tracker import DetectionBox


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------


def _hsv_to_rgb(hue_degrees: float, saturation: float, value: float) -> tuple[int, int, int]:
    """Convert HSV (hue in [0, 360]) to an RGB int tuple."""
    r, g, b = colorsys.hsv_to_rgb(hue_degrees / 360.0, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def color_for_track_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated colour for a given track ID.

    The hue is derived from the ID modulo 360 so any two consecutive
    IDs get visually distinct colours. Saturation / value are fixed at
    levels that read well against the typical CCTV-grey palette.
    """
    return _hsv_to_rgb(hue_degrees=(track_id * 47) % 360, saturation=0.85, value=0.95)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderConfig:
    """Drawing knobs. Defaults are tuned for 640x480-ish CCTV frames."""

    box_thickness: int = 2
    text_scale: float = 0.6
    text_thickness: int = 2
    text_padding_px: int = 4
    show_confidence: bool = True
    fallback_color: tuple[int, int, int] = (255, 255, 255)  # white for untracked detections


def _draw_box_on_array(
    image: np.ndarray,
    box: DetectionBox,
    color: tuple[int, int, int],
    config: RenderConfig,
) -> None:
    """Mutate ``image`` in-place: draw one bounding box + label."""
    height, width = image.shape[:2]
    x1 = max(0, min(int(round(box.x_min)), width - 1))
    y1 = max(0, min(int(round(box.y_min)), height - 1))
    x2 = max(0, min(int(round(box.x_max)), width - 1))
    y2 = max(0, min(int(round(box.y_max)), height - 1))
    if x2 <= x1 or y2 <= y1:
        return  # degenerate box — skip silently

    # Box outline
    image[y1:y1 + config.box_thickness, x1:x2] = color
    image[y2 - config.box_thickness:y2, x1:x2] = color
    image[y1:y2, x1:x1 + config.box_thickness] = color
    image[y1:y2, x2 - config.box_thickness:x2] = color

    # Label
    label = _format_label(box, config)
    label_width = max(1, int(len(label) * 7 * config.text_scale))
    label_height = max(1, int(14 * config.text_scale))
    label_x1 = x1
    label_y1 = max(0, y1 - label_height - 2 * config.text_padding_px)
    label_x2 = min(width - 1, label_x1 + label_width + 2 * config.text_padding_px)
    label_y2 = min(height - 1, label_y1 + label_height + config.text_padding_px)
    image[label_y1:label_y2, label_x1:label_x2] = color
    _draw_text(image, label, label_x1 + config.text_padding_px,
               label_y1 + label_height, config)


def _format_label(box: DetectionBox, config: RenderConfig) -> str:
    """Return ``track 7 0.83`` or ``det 0.83`` depending on what we have."""
    if box.track_id is not None and config.show_confidence:
        return f"track {box.track_id} {box.confidence:.2f}"
    if box.track_id is not None:
        return f"track {box.track_id}"
    if config.show_confidence:
        return f"det {box.confidence:.2f}"
    return "det"


def _draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    config: RenderConfig,
) -> None:
    """Draw simple bitmap glyphs without depending on PIL.

    Each character is drawn from a tiny 5x7 font table so we don't need
    OpenCV's ``putText`` (which requires a font file and varies by OS).
    The output is intentionally crude — this is a debug artefact, not
    a publication figure.
    """
    for char in text:
        glyph = _GLYPHS.get(char)
        if glyph is None:
            continue
        for row, line in enumerate(glyph):
            for col, bit in enumerate(line):
                if bit == "1":
                    px = x + col
                    py = y + row
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        image[py, px] = (0, 0, 0)
        x += 6  # 5 px glyph + 1 px gap


# A minimal 5x7 ASCII bitmap font covering digits, lowercase letters,
# the space, dot, hyphen, and the percent symbol. Enough for the labels
# we generate; avoids any external font dependency.
_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "a": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "c": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "d": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "e": ("11111", "10000", "11110", "10000", "10000", "10000", "11111"),
    "f": ("11111", "10000", "11110", "10000", "10000", "10000", "10000"),
    "i": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "k": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "l": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "n": ("10001", "11001", "10101", "10101", "10011", "10001", "10001"),
    "o": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "r": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "s": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "t": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "u": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00000", "00100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_frame(
    image: np.ndarray,
    detections: Sequence[DetectionBox],
    config: RenderConfig | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` with boxes + track IDs drawn on it.

    The input array is not mutated; the returned array is a fresh copy
    so callers can keep the original frame around for byte-exact
    comparison.
    """
    config = config or RenderConfig()
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            f"annotate_frame expects an HxWx3 or HxWx4 image; got shape {image.shape}."
        )
    canvas = image.copy()
    if canvas.shape[2] == 4:
        canvas = canvas[:, :, :3]
    for det in detections:
        color = color_for_track_id(det.track_id) if det.track_id is not None \
            else config.fallback_color
        _draw_box_on_array(canvas, det, color, config)
    return canvas


def render_annotated_clip(
    frame_images: Iterable[np.ndarray],
    detections_by_frame: Iterable[Sequence[DetectionBox]],
    output_dir: Path,
    clip_id: str,
    config: RenderConfig | None = None,
) -> list[Path]:
    """Render every frame to ``output_dir / f"{clip_id}_frame_NNNNN.png"``.

    Both iterables are zipped; mismatched lengths raise rather than
    silently truncating, because the order / count of frames is the
    thing the report depends on.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or RenderConfig()
    paths: list[Path] = []
    frame_images_list = list(frame_images)
    detections_list = list(detections_by_frame)
    if len(frame_images_list) != len(detections_list):
        raise ValueError(
            f"Frame / detection count mismatch for clip {clip_id!r}: "
            f"{len(frame_images_list)} frames vs {len(detections_list)} detection rows."
        )
    for index, (image, dets) in enumerate(zip(frame_images_list, detections_list)):
        annotated = annotate_frame(image, dets, config=config)
        try:
            from PIL import Image  # local import — only needed at write time
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to write annotated frames; install via "
                "colab/setup.py (it's in the approved stack)."
            ) from exc
        out_path = output_dir / f"{clip_id}_frame_{index:05d}.png"
        Image.fromarray(annotated).save(out_path)
        paths.append(out_path)
    return paths


__all__: tuple[str, ...] = (
    "RenderConfig",
    "annotate_frame",
    "render_annotated_clip",
    "color_for_track_id",
)