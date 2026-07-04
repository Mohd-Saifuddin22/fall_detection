"""Le2i ground-truth adapter.

Parses one Le2i ``.txt`` annotation file into the project's
:class:`EventGroundTruthWindow` contract (Issue 004 Step 1) and a
list of per-frame detection ground-truth rows.

Verified annotation format
-------------------------

::

    <fall_start_frame>          # line 1, integer, 1-based
    <fall_end_frame>            # line 2, integer, 1-based
    <frame_index>, <flag>, <x1>, <y1>, <x2>, <y2>   # line 3+
    ...

- ``fall_start_frame``, ``fall_end_frame``: integer 1-based
  inclusive frame indices. ``0, 0`` is a **no-fall sentinel** —
  the clip is genuinely no-fall, and the parser emits no event
  window. A mixed ``(0, X)`` or ``(X, 0)`` window is malformed
  and raises :class:`ValueError`.
- ``frame_index``: integer 1-based frame index.
- ``flag``: original Le2i flag column. Preserved verbatim on
  :class:`Le2iFrameDetection.flag` so a downstream consumer can
  distinguish "active annotation" from "potentially unreliable
  frame" — the parser itself does not interpret the flag.
- ``x1, y1, x2, y2``: pixel-corner box. ``0 0 0 0`` means
  "person absent" — :class:`Le2iFrameDetection.present` is set to
  ``False`` and the box columns stay at zero so a downstream
  consumer can skip the row without losing the row's presence.
  Any non-zero box with ``x1 >= x2`` or ``y1 >= y2`` raises
  :class:`ValueError` (a malformed annotation must not be
  silently coerced).
- When ``frame_size`` is supplied, non-zero boxes are also
  validated to lie in-frame (``0 <= x1 < x2 <= width`` and the
  y-equivalent).

Why fail loud on malformed lines
--------------------------------

A misaligned or partially-written annotation file is a
data-integrity bug. Silently dropping the row would let an
evaluation run on partial truth — exactly the failure mode the
Step-3 self-test exists to catch. Raising :class:`ValueError`
with the offending line surfaces the bug at parse time rather
than at evaluation time, and makes the offending file easy to
identify from the traceback.

Missing GT
----------

If a ``.txt`` annotation file does not exist next to a video,
:meth:`parse_le2i_annotation` returns :class:`NotAvailable` and
:meth:`collect_le2i_clips` records ``annotation=None`` on the
:class:`Le2iVideoPair`. Windows / boxes are **never** fabricated
from a missing file. The pair still exposes ``fps`` so the metric
bundle can compute false-alarms-per-hour / delay metrics that do
not require GT.

FPS helper
----------

:meth:`read_le2i_fps` reads the FPS from the ``.avi`` via OpenCV
(lazy-imported so the parser remains usable in environments where
OpenCV is not yet wired). When FPS cannot be read and the caller
did not configure ``fallback_fps``, the helper returns
:class:`NotAvailable` rather than silently substituting a
project-wide constant — every evaluation run gets to see what it
actually used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data.manifests import FallLabel

from evaluation.contracts import EventGroundTruthWindow
from evaluation.not_available import NotAvailable


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Le2iFrameDetection:
    """One parsed per-frame detection ground-truth row.

    All coordinate fields are preserved verbatim from the
    source line. ``present`` is derived from the all-zero box
    check (and never overrides the actual coordinate values).
    """

    frame_index: int
    flag: int
    x1: int
    y1: int
    x2: int
    y2: int
    present: bool

    def __post_init__(self) -> None:
        if self.frame_index < 1:
            raise ValueError(
                f"Le2iFrameDetection.frame_index must be 1-based, "
                f"got {self.frame_index}."
            )
        for name in ("x1", "y1", "x2", "y2"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Le2iFrameDetection.{name} must be a non-negative int, "
                    f"got {value!r}."
                )


@dataclass(frozen=True)
class Le2iAnnotation:
    """One parsed Le2i ``.txt`` file.

    The fall window is preserved as a pair of inclusive 1-based
    frame indices. The :attr:`fall_window` tuple is empty when
    the source file used the no-fall sentinel (``0 0``); the
    :class:`EventGroundTruthWindow` adapter returned by
    :meth:`event_window` is ``None`` in that case.
    """

    source_path: Path
    fall_window: tuple[int, int]  # (start, end) inclusive; () for no-fall
    frame_detections: tuple[Le2iFrameDetection, ...]

    @property
    def has_fall(self) -> bool:
        """``True`` iff a non-empty fall window is present."""
        return len(self.fall_window) > 0

    def event_window(
        self,
        *,
        clip_id: str,
    ) -> EventGroundTruthWindow | None:
        """Map to the Step 1 contract, or ``None`` for no-fall."""
        if not self.has_fall:
            return None
        start, end = self.fall_window
        return EventGroundTruthWindow(
            clip_id=clip_id,
            start_frame=start,
            end_frame=end,
            label=FallLabel.FALL,
        )


@dataclass(frozen=True)
class Le2iVideoPair:
    """A Le2i ``.avi`` file and everything we know about its ground truth.

    ``annotation`` is ``None`` when the clip has no ``.txt``
    alongside it (Office and Lecture-room in real Le2i both have
    videos but no labels). The metric bundle treats that as
    "no event GT available" via :class:`NotAvailable` rather
    than fabricating windows or boxes.

    ``fps`` is :class:`NotAvailable` when OpenCV cannot read it
    *and* the caller did not supply a fallback. The metric
    bundle surfaces that as the no-fps sentinel, not as ``25.0``.
    """

    video_path: Path
    annotation: Le2iAnnotation | None
    fps: float | NotAvailable


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------


def parse_le2i_annotation_text(
    text: str,
    *,
    source_path: Path | None = None,
    frame_size: tuple[int, int] | None = None,
) -> Le2iAnnotation:
    """Parse the body of a Le2i ``.txt`` annotation file.

    Args:
        text: Annotation body. Whitespace lines are ignored.
        source_path: Optional — surfaced on the parsed record for
            diagnostics. Defaults to a sentinel path.
        frame_size: Optional ``(width, height)``. When set,
            non-zero detection boxes are validated to lie
            in-frame (``0 <= x1 < x2 <= width``, and similarly
            for y). When ``None``, that validation is skipped
            (the parser does not require real Le2i videos to
            be available at test time).

    Returns:
        A populated :class:`Le2iAnnotation`. The
        :attr:`Le2iAnnotation.fall_window` is empty when the
        first two lines are the no-fall sentinel ``0 0``.

    Raises:
        ValueError: on malformed content (line count, parse
            failure, in-frame violation, malformed
            coordinates).
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 2:
        raise ValueError(
            "Le2i annotation must have at least 2 lines "
            "(fall-start, fall-end); got "
            f"{len(lines)} non-empty line(s) from {source_path or '<text>'}."
        )

    fall_window = _parse_fall_window(lines[0], lines[1])

    detections: list[Le2iFrameDetection] = []
    for line_number, line in enumerate(lines[2:], start=3):
        detections.append(_parse_detection_line(
            line,
            line_number=line_number,
            source_path=source_path,
            frame_size=frame_size,
        ))

    return Le2iAnnotation(
        source_path=source_path or Path("<text>"),
        fall_window=fall_window,
        frame_detections=tuple(detections),
    )


def _parse_fall_window(start_line: str, end_line: str) -> tuple[int, int]:
    """Decode the first two lines into a fall window or no-fall sentinel."""
    try:
        start_frame = int(start_line)
        end_frame = int(end_line)
    except ValueError as exc:
        raise ValueError(
            f"Le2i annotation first two lines must be integers; got "
            f"{start_line!r} and {end_line!r}."
        ) from exc

    if (start_frame, end_frame) == (0, 0):
        return ()  # no-fall sentinel

    if start_frame == 0 or end_frame == 0:
        raise ValueError(
            f"Le2i fall window must be either (0, 0) or two non-zero "
            f"integers; got ({start_frame}, {end_frame})."
        )

    if start_frame > end_frame:
        raise ValueError(
            f"Le2i fall window start_frame ({start_frame}) must be "
            f"<= end_frame ({end_frame})."
        )

    if start_frame < 1 or end_frame < 1:
        raise ValueError(
            f"Le2i fall window must use 1-based frame indices; got "
            f"({start_frame}, {end_frame})."
        )

    return (start_frame, end_frame)


def _parse_detection_line(
    line: str,
    *,
    line_number: int,
    source_path: Path | None,
    frame_size: tuple[int, int] | None,
) -> Le2iFrameDetection:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"Le2i detection line must have 6 comma-separated integer "
            f"values; got {line!r} (line {line_number}) in {source_path or '<text>'}."
        )
    try:
        frame_index, flag, x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(
            f"Le2i detection line must contain only integers; got "
            f"{line!r} (line {line_number}) in {source_path or '<text>'}."
        ) from exc

    if (x1, y1, x2, y2) == (0, 0, 0, 0):
        present = False
    else:
        present = True
        # Validate non-zero boxes fail loud.
        if x1 >= x2:
            raise ValueError(
                f"Le2i detection line {line_number} of {source_path or '<text>'}: "
                f"x1 ({x1}) must be < x2 ({x2}); line={line!r}."
            )
        if y1 >= y2:
            raise ValueError(
                f"Le2i detection line {line_number} of {source_path or '<text>'}: "
                f"y1 ({y1}) must be < y2 ({y2}); line={line!r}."
            )
        if frame_size is not None:
            width, height = frame_size
            if x2 > width:
                raise ValueError(
                    f"Le2i detection line {line_number} of {source_path or '<text>'}: "
                    f"x2 ({x2}) exceeds frame width ({width}); line={line!r}."
                )
            if y2 > height:
                raise ValueError(
                    f"Le2i detection line {line_number} of {source_path or '<text>'}: "
                    f"y2 ({y2}) exceeds frame height ({height}); line={line!r}."
                )

    return Le2iFrameDetection(
        frame_index=frame_index,
        flag=flag,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        present=present,
    )


def parse_le2i_annotation(
    source_path: Path | str,
    *,
    frame_size: tuple[int, int] | None = None,
) -> Le2iAnnotation | None:
    """Parse a Le2i ``.txt`` annotation file from disk.

    Returns ``None`` when the file does not exist (the
    "missing GT" case — Office and Lecture-room clips, etc.).
    Raises :class:`ValueError` on malformed content.
    """
    path = Path(source_path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return parse_le2i_annotation_text(text, source_path=path, frame_size=frame_size)


# ---------------------------------------------------------------------------
# Video / annotation pairing
# ---------------------------------------------------------------------------


def pair_video_with_annotation(
    video_path: Path,
    annotations_root: Path,
) -> Path | None:
    """Locate the annotation that pairs with ``video_path``.

    Pairing is by filename stem — ``video(i).avi`` pairs with
    ``video(i).txt``. The search tolerates two real-Le2i
    layouts:

      1. The annotation lives directly in ``annotations_root``:
         ``annotations_root / video_path.stem + ".txt"``.
      2. The annotation lives one folder deeper, in a
         same-named subfolder:
         ``annotations_root / video_path.stem / video_path.stem + ".txt"``.

    Both layouts are tried; the first match wins. The function
    returns ``None`` when neither exists. Non-contiguous stems
    work without special-casing — only the file lookup matters.

    Callers that own both ``videos_root`` and ``annotations_root``
    pass each explicitly. Callers that conflate the two
    (``data/le2i-root/videos/(1)/video(1).avi`` and the same
    tree for annotations) pass the same path twice.
    """

    stem = video_path.stem
    candidates = (
        annotations_root / f"{stem}.txt",
        annotations_root / stem / f"{stem}.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# Recognized Le2i video extensions. ``.avi`` is the verified
# default; ``.mp4`` is tolerated for downstream re-encoding but
# not required.
_LE2I_VIDEO_EXTENSIONS: tuple[str, ...] = (".avi", ".mp4")


def collect_le2i_clips(
    videos_root: Path,
    *,
    annotations_root: Path | None = None,
    fallback_fps: float | None = None,
    frame_size: tuple[int, int] | None = None,
) -> list[Le2iVideoPair]:
    """Walk ``videos_root`` for Le2i videos and pair each with its annotation.

    The walk descends into nested folders (any depth). For every
    found ``.avi``/``.mp4``, the matching annotation is looked
    up via :func:`pair_video_with_annotation`. A pair with
    ``annotation=None`` represents the "missing GT" case (Office,
    Lecture-room): the video exists; the GT does not.

    Args:
        videos_root: Directory to walk for video files.
        annotations_root: Directory holding the annotation
            hierarchy. Defaults to ``videos_root`` when ``None``
            — covers the layout where each video sits next to
            its annotation in the same folder.
        fallback_fps: Optional fallback FPS used when the
            per-video OpenCV read fails. Default ``None``
            means "no fallback; missing FPS becomes
            :class:`NotAvailable` per pair".
        frame_size: Optional ``(width, height)`` forwarded to
            :func:`parse_le2i_annotation` for in-frame
            validation. Typically None at this level — the
            caller may pass it after inspecting the video.

    Returns:
        A list of :class:`Le2iVideoPair`, one per video file
        found. Sort order matches :func:`os.walk`'s
        directory order (not deterministic across
        filesystems) — callers wanting a deterministic
        order should sort the result by ``video_path``.
    """

    annotations_root = Path(annotations_root) if annotations_root else videos_root
    videos_root = Path(videos_root)

    if not videos_root.is_dir():
        # Not a directory → no clips. Returning an empty list
        # avoids the caller dealing with a directory-creation
        # error in test environments. Real production code
        # should pre-validate the path.
        return []

    pairs: list[Le2iVideoPair] = []
    for dirpath, _dirnames, filenames in os.walk(videos_root):
        for filename in sorted(filenames):
            video_path = Path(dirpath) / filename
            if video_path.suffix.lower() not in _LE2I_VIDEO_EXTENSIONS:
                continue
            annotation_path = pair_video_with_annotation(video_path, annotations_root)
            if annotation_path is None:
                annotation: Le2iAnnotation | None = None
            else:
                annotation = parse_le2i_annotation(
                    annotation_path, frame_size=frame_size,
                )
            fps_value = read_le2i_fps(video_path, fallback_fps=fallback_fps)
            pairs.append(
                Le2iVideoPair(
                    video_path=video_path,
                    annotation=annotation,
                    fps=fps_value,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# FPS helper
# ---------------------------------------------------------------------------


def read_le2i_fps(
    video_path: Path | str,
    *,
    fallback_fps: float | None = None,
) -> float | NotAvailable:
    """Read the FPS of a Le2i ``.avi`` via OpenCV.

    Behaviour:

    - If OpenCV is installed and the file's ``CAP_PROP_FPS`` is
      ``> 0``: return that float.
    - If OpenCV is installed but the file is unreadable or
      reports FPS ``<= 0``: return ``fallback_fps`` when
      provided, otherwise :class:`NotAvailable` with reason
      ``"cannot read FPS from .avi"``.
    - If OpenCV itself is missing (e.g. on a host that hasn't
      run setup yet): return ``fallback_fps`` when provided,
      otherwise :class:`NotAvailable` with reason
      ``"opencv-python not installed"``.

    The fallback is never silently applied — the caller's
    explicit ``fallback_fps`` keeps the eval end-to-end
    reversible. The default ``None`` surfaces
    :class:`NotAvailable` so the metric bundle reports missing
    fps honestly instead of fabricating a project-wide constant.
    """

    if fallback_fps is not None and fallback_fps <= 0.0:
        raise ValueError(
            f"fallback_fps must be positive, got {fallback_fps!r}."
        )

    try:
        import cv2  # noqa: PLC0415  — lazy import keeps the parser usable.
    except ImportError:
        if fallback_fps is not None:
            return float(fallback_fps)
        return NotAvailable(reason="opencv-python not installed")

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        if fallback_fps is not None:
            return float(fallback_fps)
        return NotAvailable(reason="cannot open video file")

    try:
        raw_fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()

    if raw_fps is None or raw_fps <= 0.0:
        if fallback_fps is not None:
            return float(fallback_fps)
        return NotAvailable(reason="cannot read FPS from .avi")

    return float(raw_fps)


__all__: tuple[str, ...] = (
    "Le2iFrameDetection",
    "Le2iAnnotation",
    "Le2iVideoPair",
    "parse_le2i_annotation",
    "parse_le2i_annotation_text",
    "pair_video_with_annotation",
    "collect_le2i_clips",
    "read_le2i_fps",
)
