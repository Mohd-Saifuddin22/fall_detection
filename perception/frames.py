"""Ordered frame-folder loader for the perception front-end.

URFD and many other fall-detection datasets ship as one folder per clip
containing ordered PNG frames. The tracker contract requires frames in
**temporal order** — out-of-order frames produce invalid tracks, and the
bug is invisible until you visualise the annotated output.

URFD's actual layout (as shipped by ``tanmaydacha/urfd-dataset``) is
**nested**::

    fall-01-cam0-rgb/
        fall-01-cam0-rgb/        <-- the real frame folder
            frame_0001.png
            frame_0002.png
            ...

The clip-level folder is a wrapper around an inner folder that shares
its name. ``discover_frames`` detects this layout and descends into the
single matching inner subfolder automatically — callers don't need to
know it exists.

Other fall datasets (GMDCSA-24, UP-Fall, etc.) typically use a flat
layout where frames sit directly inside the clip folder. Both layouts
are supported.

This module owns:
    - the rule for which file extensions count as frames,
    - the rule for ordering (numeric suffix when present, else lexical),
    - nested-layout detection (single inner matching subfolder),
    - a :class:`FrameFolderReader` that exposes the ordered list without
      loading pixels, and an iterator that yields :class:`FrameRecord`
      rows on demand.

Pure / no side effects. No tracking / no YOLO dependency — easy to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Extensions we treat as image frames. Lower-cased for case-insensitive
# matching; add new ones here deliberately, not at the call site.
DEFAULT_FRAME_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
})

# A trailing numeric suffix — the typical URFD naming convention is
# ``frame_0001.png``; some datasets use ``000123.jpg`` with no prefix.
# Match the last run of digits at the end of the stem (filename minus
# extension), so ``fall-01-cam0_frame_00042`` sorts after ``..._00010``.
_TRAILING_DIGITS = re.compile(r"(\d+)\s*$")


@dataclass(frozen=True)
class FrameRecord:
    """One frame in temporal order.

    ``index`` is the zero-based position in the ordered frame list
    (NOT necessarily the frame number from the dataset — that's what the
    original filename carries; we keep both so reports can show either).
    """

    index: int
    path: Path
    source_index: int  # numeric suffix parsed from the filename, or -1 if none

    @property
    def filename(self) -> str:
        return self.path.name


def extract_trailing_number(stem: str) -> int:
    """Return the trailing integer in ``stem`` or ``-1`` if none.

    Used to numerically sort frames like ``frame_0001.png`` → 1.
    Falls back to ``-1`` (sorts first) rather than raising so the loader
    can still order mixed-naming folders.
    """
    match = _TRAILING_DIGITS.search(stem)
    if match is None:
        return -1
    return int(match.group(1))


def _resolve_scan_folder(
    folder: Path,
    extensions: frozenset[str],
) -> tuple[Path, str]:
    """Resolve the actual folder to scan.

    Rules:
        1. If ``folder`` directly contains frame files, scan it.
        2. Otherwise, if exactly ONE child directory exists AND it
           contains frame files, scan that child (handles URFD's nested
           ``fall-01-cam0-rgb/fall-01-cam0-rgb/*.png`` layout).
        3. Otherwise return ``folder`` unchanged — the caller will get
           an empty list (or raise on a bad path) which is more honest
           than guessing.

    Returns ``(folder_to_scan, layout_description)`` where
    ``layout_description`` is one of ``"flat"`` or ``"nested:<child>"``
    so reports can show which layout was used.
    """
    direct_frames = [
        entry for entry in folder.iterdir()
        if entry.is_file() and entry.suffix.lower() in extensions
    ]
    if direct_frames:
        return folder, "flat"

    child_dirs = [entry for entry in folder.iterdir() if entry.is_dir()]
    if len(child_dirs) == 1:
        only_child = child_dirs[0]
        child_frames = [
            entry for entry in only_child.iterdir()
            if entry.is_file() and entry.suffix.lower() in extensions
        ]
        if child_frames:
            return only_child, f"nested:{only_child.name}"

    # No frames here, no single matching child — leave it alone; downstream
    # code will report zero frames so the operator can see the problem.
    return folder, "flat"


def discover_frames(
    folder: Path,
    extensions: frozenset[str] = DEFAULT_FRAME_EXTENSIONS,
) -> list[FrameRecord]:
    """Return all frames under ``folder``, ordered numerically.

    Sorting rule:
        1. If a frame has a trailing integer in its stem, sort by that
           integer ascending.
        2. If multiple frames share the same trailing integer (rare,
           some datasets pad then prefix), tie-break by full filename
           (lexical) for determinism.
        3. Frames with no trailing integer sort to the END (after all
           numeric frames) by filename, so they don't poison the start
           of the timeline.

    The folder is scanned at the top level; the URFD nested layout
    (``<clip>/<clip>/*.png``) is auto-resolved by :func:`_resolve_scan_folder`
    so callers can pass the clip-level folder they were given.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Frame folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Frame path is not a directory: {folder}")

    scan_folder, _layout = _resolve_scan_folder(folder, extensions)

    candidates: list[tuple[int, str, Path]] = []
    for entry in scan_folder.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in extensions:
            continue
        number = extract_trailing_number(entry.stem)
        # Sort key: (has_number_flag, number, filename). has_number_flag=0
        # puts numeric frames first; -1 entries sort after real numbers
        # because -1 < any positive n.
        has_number = 0 if number >= 0 else 1
        candidates.append((has_number, number, entry))

    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))

    return [
        FrameRecord(index=i, path=path, source_index=number)
        for i, (_has_num, number, path) in enumerate(candidates)
    ]


def describe_layout(folder: Path) -> str:
    """Return ``"flat"`` or ``"nested:<child>"`` for diagnostics.

    Useful in the run report so a human reviewer can see at a glance
    which layout the loader found — helpful when the staging script
    is updated and the folder shape changes.
    """
    if not folder.is_dir():
        return "missing"
    _, layout = _resolve_scan_folder(folder, DEFAULT_FRAME_EXTENSIONS)
    return layout


class FrameFolderReader:
    """Reads an ordered list of frame paths from a clip folder.

    Use :meth:`frames` to get the ordered list without loading pixels,
    or :meth:`iter_frames` to stream :class:`FrameRecord` rows.
    """

    def __init__(
        self,
        folder: Path,
        extensions: frozenset[str] = DEFAULT_FRAME_EXTENSIONS,
    ) -> None:
        self._folder = Path(folder)
        self._extensions = extensions
        self._cache: list[FrameRecord] | None = None

    @property
    def folder(self) -> Path:
        return self._folder

    def frames(self) -> list[FrameRecord]:
        """Return the ordered frame list (cached after first call)."""
        if self._cache is None:
            self._cache = discover_frames(self._folder, self._extensions)
        return list(self._cache)

    def __len__(self) -> int:
        return len(self.frames())

    def iter_frames(self) -> Iterator[FrameRecord]:
        """Yield frames in temporal order."""
        return iter(self.frames())


__all__: tuple[str, ...] = (
    "DEFAULT_FRAME_EXTENSIONS",
    "FrameRecord",
    "FrameFolderReader",
    "describe_layout",
    "discover_frames",
    "extract_trailing_number",
)