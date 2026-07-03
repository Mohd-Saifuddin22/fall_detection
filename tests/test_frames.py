"""Unit tests for the perception frame loader (numeric sort, ext filter).

No ultralytics / torch / GPU dependency. All tests run on a synthetic
folder of fake frame files created via tempfile.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.frames import (  # noqa: E402
    DEFAULT_FRAME_EXTENSIONS,
    FrameFolderReader,
    discover_frames,
    extract_trailing_number,
)


def _touch(folder: Path, name: str) -> Path:
    """Create an empty file in ``folder`` and return its path."""
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _make_folder_with_files(parent: Path, names: list[str]) -> Path:
    """Create a temp folder under ``parent`` containing exactly the named files."""
    folder = parent / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()
    return folder


class ExtractTrailingNumberTests(unittest.TestCase):
    """Pure-function coverage for the numeric-suffix parser."""

    def test_simple_digit_suffix(self) -> None:
        self.assertEqual(extract_trailing_number("frame_0001"), 1)

    def test_no_suffix_returns_minus_one(self) -> None:
        self.assertEqual(extract_trailing_number("fall-01-cam0_frame"), -1)

    def test_suffix_after_underscore(self) -> None:
        self.assertEqual(extract_trailing_number("frame_00042"), 42)

    def test_zero_suffix(self) -> None:
        self.assertEqual(extract_trailing_number("frame_0"), 0)


class DiscoverFramesTests(unittest.TestCase):
    """Folder-level discovery: extensions, ordering, error paths."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.parent = Path(self._tmp.name)

    def test_orders_files_by_trailing_integer(self) -> None:
        # Deliberately write out of order; loader must fix it.
        folder = _make_folder_with_files(self.parent, [
            "frame_0010.png", "frame_0002.png", "frame_0001.png",
            "frame_0003.png", "frame_0009.png",
        ])
        frames = discover_frames(folder)
        self.assertEqual([f.source_index for f in frames], [1, 2, 3, 9, 10])

    def test_filters_to_supported_extensions_only(self) -> None:
        folder = _make_folder_with_files(self.parent, [
            "frame_0001.png", "frame_0002.jpg", "frame_0003.txt",
            "frame_0004.png", "README.md",
        ])
        frames = discover_frames(folder)
        names = [f.filename for f in frames]
        self.assertEqual(names, ["frame_0001.png", "frame_0002.jpg", "frame_0004.png"])

    def test_case_insensitive_extension_matching(self) -> None:
        folder = _make_folder_with_files(self.parent, ["frame_0001.PNG", "frame_0002.Jpeg"])
        frames = discover_frames(folder)
        self.assertEqual(len(frames), 2)

    def test_non_numeric_files_sort_after_numeric_ones(self) -> None:
        folder = _make_folder_with_files(self.parent, [
            "frame_0002.png", "fall-01-cam0_frame.png", "frame_0001.png",
        ])
        frames = discover_frames(folder)
        # First two by integer (1, 2), then the non-numeric one.
        self.assertEqual([f.filename for f in frames], [
            "frame_0001.png", "frame_0002.png", "fall-01-cam0_frame.png",
        ])

    def test_empty_folder_returns_empty_list(self) -> None:
        folder = self.parent / "empty"
        folder.mkdir()
        self.assertEqual(discover_frames(folder), [])

    def test_missing_folder_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            discover_frames(self.parent / "does-not-exist")

    def test_non_directory_raises(self) -> None:
        file_path = _touch(self.parent, "not-a-folder.txt")
        with self.assertRaises(NotADirectoryError):
            discover_frames(file_path)

    def test_default_extensions_include_common_image_formats(self) -> None:
        self.assertIn(".png", DEFAULT_FRAME_EXTENSIONS)
        self.assertIn(".jpg", DEFAULT_FRAME_EXTENSIONS)
        self.assertIn(".jpeg", DEFAULT_FRAME_EXTENSIONS)

    def test_urfd_nested_layout_descends_into_inner_matching_folder(self) -> None:
        # URFD ships as fall-NN-camM-rgb/fall-NN-camM-rgb/*.png — the
        # outer folder is a wrapper. The loader must auto-descend into
        # the single matching child so callers don't have to.
        outer = self.parent / "fall-01-cam0-rgb"
        inner = outer / "fall-01-cam0-rgb"
        inner.mkdir(parents=True)
        # Deliberately write frames out of order; numeric sort must fix it.
        for name in ("frame_0003.png", "frame_0001.png", "frame_0010.png",
                     "frame_0002.png"):
            (inner / name).touch()
        # Add a sibling file in the OUTER folder to prove the loader
        # correctly chose the inner folder.
        (outer / "README.txt").touch()

        frames = discover_frames(outer)
        self.assertEqual(len(frames), 4)
        self.assertEqual([f.source_index for f in frames], [1, 2, 3, 10])

    def test_urfd_nested_layout_does_not_descend_when_inner_has_no_frames(self) -> None:
        # If the only child is empty, the loader must NOT pretend there
        # are frames — it returns zero rows so the operator sees the bug.
        outer = self.parent / "fall-02-cam0-rgb"
        inner = outer / "fall-02-cam0-rgb"
        inner.mkdir(parents=True)
        (inner / "no_frames_here.txt").touch()
        self.assertEqual(discover_frames(outer), [])

    def test_urfd_nested_layout_ignores_unrelated_subfolders(self) -> None:
        # If there are MULTIPLE subfolders and no direct frames, we
        # don't guess — return zero so the operator sees the mismatch.
        outer = self.parent / "fall-03-cam0-rgb"
        (outer / "subdir_a").mkdir(parents=True)
        (outer / "subdir_b").mkdir(parents=True)
        self.assertEqual(discover_frames(outer), [])

    def test_describe_layout_reports_urfd_nested(self) -> None:
        from perception.frames import describe_layout
        outer = self.parent / "fall-04-cam0-rgb"
        inner = outer / "fall-04-cam0-rgb"
        inner.mkdir(parents=True)
        (inner / "frame_0001.png").touch()
        layout = describe_layout(outer)
        self.assertTrue(layout.startswith("nested:"), msg=f"got {layout!r}")


class FrameFolderReaderTests(unittest.TestCase):
    """The reader caches and exposes the same frames as :func:`discover_frames`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.parent = Path(self._tmp.name)

    def test_caches_result_across_calls(self) -> None:
        folder = _make_folder_with_files(self.parent, ["f_0001.png", "f_0002.png"])
        reader = FrameFolderReader(folder)
        first = reader.frames()
        second = reader.frames()
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        # Same list contents, fresh list object (caller can mutate freely).
        self.assertEqual([f.filename for f in first], [f.filename for f in second])

    def test_len_returns_frame_count(self) -> None:
        folder = _make_folder_with_files(self.parent, [
            "a_0001.png", "a_0002.png", "a_0003.png",
        ])
        reader = FrameFolderReader(folder)
        self.assertEqual(len(reader), 3)

    def test_iter_frames_yields_ordered_frames(self) -> None:
        folder = _make_folder_with_files(self.parent, [
            "a_0003.png", "a_0001.png", "a_0002.png",
        ])
        reader = FrameFolderReader(folder)
        indices = [f.source_index for f in reader.iter_frames()]
        self.assertEqual(indices, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()