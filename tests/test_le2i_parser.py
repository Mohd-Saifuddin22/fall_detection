"""Tests for :mod:`data.le2i`.

The parser is exercised against small in-memory text fixtures that
match the verified Le2i annotation format — no Kaggle download,
no real video file, no OpenCV dependency at test time. The
OpenCV-backed :func:`data.le2i.read_le2i_fps` is exercised only
against its non-OpenCV paths (fallback-only + missing-file).

Coverage target (per the Step 5 task spec):

- No-fall fixture (``0, 0`` sentinel) → no event window,
  no-zero frame-1 detection becomes a present=False row.
- Fall fixture (``144, 164`` window) → ``[144, 164]`` event,
  frame-1 row is ``present=True`` with the parsed bbox.
- Malformed bbox with ``x1 >= x2`` raises.
- All-zero bbox is treated as ``present=False``.
- Missing annotation returns :class:`NotAvailable` /
  ``annotation=None`` without crashing.
- Video/annotation pairing tolerates non-contiguous stems.
- Nested same-named subfolder resolution works.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.le2i import (
    Le2iAnnotation,
    Le2iFrameDetection,
    Le2iVideoPair,
    collect_le2i_clips,
    pair_video_with_annotation,
    parse_le2i_annotation,
    parse_le2i_annotation_text,
    read_le2i_fps,
)
from evaluation.not_available import NotAvailable


# ---------------------------------------------------------------------------
# Annotation parsing — no-fall fixture
# ---------------------------------------------------------------------------


class NoFallFixtureTests(unittest.TestCase):
    """``0, 0`` sentinel produces no event window."""

    NO_FALL_TEXT = (
        "0\n"
        "0\n"
        "1, 1, 204, 61, 273, 198\n"
        "10, 1, 0, 0, 0, 0\n"
    )

    def test_no_fall_sentinel_yields_empty_fall_window(self) -> None:
        ann = parse_le2i_annotation_text(self.NO_FALL_TEXT)
        self.assertFalse(ann.has_fall)
        self.assertEqual(ann.fall_window, ())

    def test_no_fall_event_window_adapter_returns_none(self) -> None:
        ann = parse_le2i_annotation_text(self.NO_FALL_TEXT)
        self.assertIsNone(ann.event_window(clip_id="clip-x"))

    def test_no_fall_frame_1_box_is_parsed(self) -> None:
        # From the brief: row ``1, 1, 204, 61, 273, 198`` →
        # frame 1, present, bbox (204, 61, 273, 198).
        ann = parse_le2i_annotation_text(self.NO_FALL_TEXT)
        self.assertEqual(len(ann.frame_detections), 2)
        first = ann.frame_detections[0]
        self.assertEqual(first.frame_index, 1)
        self.assertEqual(first.flag, 1)
        self.assertEqual(first.x1, 204)
        self.assertEqual(first.y1, 61)
        self.assertEqual(first.x2, 273)
        self.assertEqual(first.y2, 198)
        self.assertTrue(first.present)

    def test_no_fall_absent_person_row_is_marked_absent(self) -> None:
        # ``10, 1, 0, 0, 0, 0`` → person absent on frame 10.
        # The box stays at zero and ``present=False`` so a
        # downstream consumer can skip the row without losing it.
        ann = parse_le2i_annotation_text(self.NO_FALL_TEXT)
        absent_row = ann.frame_detections[1]
        self.assertEqual(absent_row.frame_index, 10)
        self.assertFalse(absent_row.present)
        self.assertEqual(
            (absent_row.x1, absent_row.y1, absent_row.x2, absent_row.y2),
            (0, 0, 0, 0),
        )


# ---------------------------------------------------------------------------
# Annotation parsing — fall fixture
# ---------------------------------------------------------------------------


class FallFixtureTests(unittest.TestCase):
    """Fall fixture maps cleanly to the Step 1 contract."""

    FALL_TEXT = (
        "144\n"
        "164\n"
        "1, 1, 205, 70, 259, 170\n"
        "200, 1, 300, 100, 360, 200\n"
    )

    def test_fall_window_is_parsed_inclusive(self) -> None:
        ann = parse_le2i_annotation_text(self.FALL_TEXT)
        self.assertTrue(ann.has_fall)
        self.assertEqual(ann.fall_window, (144, 164))

    def test_fall_event_window_adapter_returns_event(self) -> None:
        ann = parse_le2i_annotation_text(self.FALL_TEXT)
        window = ann.event_window(clip_id="le2i-fall-clip")
        self.assertIsNotNone(window)
        self.assertEqual(window.start_frame, 144)
        self.assertEqual(window.end_frame, 164)
        # ``label`` is a :class:`FallLabel` enum; compare by its
        # ``.value`` to avoid an extra import in the test.
        self.assertEqual(window.label.value, "fall")

    def test_fall_frame_1_box_is_parsed(self) -> None:
        ann = parse_le2i_annotation_text(self.FALL_TEXT)
        first = ann.frame_detections[0]
        self.assertEqual(first.frame_index, 1)
        self.assertEqual(first.flag, 1)
        self.assertEqual(first.x1, 205)
        self.assertEqual(first.y1, 70)
        self.assertEqual(first.x2, 259)
        self.assertEqual(first.y2, 170)
        self.assertTrue(first.present)


# ---------------------------------------------------------------------------
# Annotation parsing — malformed input
# ---------------------------------------------------------------------------


class MalformedAnnotationTests(unittest.TestCase):
    """Every malformed annotation raises ``ValueError`` (fail loud)."""

    def test_x1_greater_or_equal_to_x2_raises(self) -> None:
        text = "0\n0\n5, 1, 200, 100, 100, 200\n"   # x1=200, x2=100 → invalid
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text(text)

    def test_y1_greater_or_equal_to_y2_raises(self) -> None:
        text = "0\n0\n5, 1, 100, 200, 200, 100\n"   # y1=200, y2=100 → invalid
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text(text)

    def test_mixed_zero_fall_window_raises(self) -> None:
        # (0, 164) or (144, 0) are not the recognised no-fall
        # sentinel — must raise.
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text("0\n164\n1, 1, 0, 0, 0, 0\n")
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text("144\n0\n1, 1, 0, 0, 0, 0\n")

    def test_inverted_fall_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text("164\n144\n1, 1, 0, 0, 0, 0\n")

    def test_window_uses_zero_index_raises(self) -> None:
        # 1-based frame indices — ``(1, 1)`` is the minimum valid
        # pair. Any 0 in the window outside the all-zero sentinel
        # is rejected.
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text("1\n0\n1, 1, 0, 0, 0, 0\n")

    def test_too_few_lines_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text("144\n")     # missing end-frame line
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text("")

    def test_detection_line_with_too_few_columns_raises(self) -> None:
        text = "0\n0\n1, 1, 5, 10, 20\n"   # 5 columns instead of 6
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text(text)

    def test_detection_line_with_non_integer_raises(self) -> None:
        text = "0\n0\n1, 1, foo, 10, 20, 30\n"
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text(text)

    def test_frame_size_validation_catches_out_of_bounds_box(self) -> None:
        # Box reaches x2=999, frame is only 320 wide → invalid.
        text = "0\n0\n1, 1, 100, 100, 999, 200\n"
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text(text, frame_size=(320, 240))

    def test_frame_size_validation_allows_all_zero_box(self) -> None:
        # All-zero box never violates in-frame bounds (it is the
        # absent-person marker). With frame_size set, the
        # parser still accepts it.
        text = "0\n0\n5, 1, 0, 0, 0, 0\n"
        ann = parse_le2i_annotation_text(text, frame_size=(320, 240))
        self.assertFalse(ann.frame_detections[0].present)

    def test_zero_indexed_frame_raises(self) -> None:
        text = "0\n0\n0, 1, 100, 100, 200, 200\n"   # frame_index=0
        with self.assertRaises(ValueError):
            parse_le2i_annotation_text(text)


class AllZeroBboxTests(unittest.TestCase):
    """All-zero bbox is the absent-person marker."""

    def test_zero_box_is_parsed_with_present_false(self) -> None:
        text = "0\n0\n7, 0, 0, 0, 0, 0\n"   # flag=0 also
        ann = parse_le2i_annotation_text(text)
        row = ann.frame_detections[0]
        self.assertFalse(row.present)
        # Flag preserved verbatim — the parser does not interpret it.
        self.assertEqual(row.flag, 0)

    def test_zero_box_validates_in_frame_when_frame_size_provided(self) -> None:
        # An all-zero box does NOT trigger in-frame validation;
        # providing frame_size never breaks the absent case.
        text = "0\n0\n5, 1, 0, 0, 0, 0\n"
        ann = parse_le2i_annotation_text(text, frame_size=(640, 480))
        self.assertFalse(ann.frame_detections[0].present)


# ---------------------------------------------------------------------------
# Container shape
# ---------------------------------------------------------------------------


class ContainerShapeTests(unittest.TestCase):
    """Container constructors reject invalid input at the type boundary."""

    def test_frame_detection_rejects_zero_index(self) -> None:
        with self.assertRaises(ValueError):
            Le2iFrameDetection(
                frame_index=0, flag=1,
                x1=10, y1=10, x2=20, y2=20,
                present=True,
            )

    def test_frame_detection_rejects_negative_coordinate(self) -> None:
        with self.assertRaises(ValueError):
            Le2iFrameDetection(
                frame_index=1, flag=1,
                x1=-5, y1=10, x2=20, y2=20,
                present=True,
            )


# ---------------------------------------------------------------------------
# File-path + missing-GT
# ---------------------------------------------------------------------------


class MissingGTTests(unittest.TestCase):
    """Missing ``.txt`` files must not crash; represent as ``None`` / ``NotAvailable``."""

    def test_parse_le2i_annotation_returns_none_for_missing_file(self) -> None:
        result = parse_le2i_annotation(Path("/does/not/exist.txt"))
        self.assertIsNone(result)

    def test_parse_le2i_annotation_loads_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write("0\n0\n1, 1, 100, 100, 200, 200\n")
            tmp_path = Path(tmp.name)
        try:
            ann = parse_le2i_annotation(tmp_path)
            self.assertIsNotNone(ann)
            self.assertIsNone(ann.event_window(clip_id="x"))
            self.assertEqual(len(ann.frame_detections), 1)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_collect_le2i_clips_records_none_annotation_for_missing_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos"
            videos.mkdir()
            # Place a video with NO annotation next to it.
            (videos / "fall-01-cam0.avi").write_bytes(b"")
            pairs = collect_le2i_clips(videos, fallback_fps=25.0)
            self.assertEqual(len(pairs), 1)
            self.assertIsNone(pairs[0].annotation)
            # Fallback FPS surfaces the project-supplied constant.
            self.assertEqual(pairs[0].fps, 25.0)

    def test_read_le2i_fps_returns_not_available_when_no_fallback(self) -> None:
        # Non-existent file → cv2 cannot open → fallback absent →
        # ``NotAvailable``. The helper must NEVER silently
        # invent an FPS.
        result = read_le2i_fps(Path("/does/not/exist.avi"))
        self.assertIsInstance(result, NotAvailable)

    def test_read_le2i_fps_uses_provided_fallback(self) -> None:
        result = read_le2i_fps(Path("/does/not/exist.avi"), fallback_fps=25.0)
        self.assertEqual(result, 25.0)

    def test_read_le2i_fps_rejects_non_positive_fallback(self) -> None:
        with self.assertRaises(ValueError):
            read_le2i_fps(Path("/anywhere.avi"), fallback_fps=0.0)


# ---------------------------------------------------------------------------
# Pairing helpers
# ---------------------------------------------------------------------------


def _touch(*parts: str) -> Path:
    """Helper: build a path under ``root`` and ensure its parent exists."""
    path = Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("annotation\n", encoding="utf-8")
    return path


def _touch_video(*parts: str) -> Path:
    """Helper: build a video file path with empty bytes; cv2 isn't called."""
    path = Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class PairingHelperTests(unittest.TestCase):
    """Pairing is by filename stem and tolerates the two real-Le2i layouts."""

    def test_pair_next_to_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = _touch_video(root, "video(1).avi")
            _touch(root, "video(1).txt")
            found = pair_video_with_annotation(video, root)
            self.assertEqual(found, root / "video(1).txt")

    def test_pair_nested_same_name_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = _touch_video(root, "fall-01-cam0.avi")
            _touch(root, "fall-01-cam0", "fall-01-cam0.txt")
            found = pair_video_with_annotation(video, root)
            # The annotation lives under ``fall-01-cam0/`` (same name
            # as the video stem). The pairer descends one level.
            self.assertEqual(
                found, root / "fall-01-cam0" / "fall-01-cam0.txt",
            )

    def test_pair_returns_none_when_neither_layout_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = _touch_video(root, "missing.avi")
            self.assertIsNone(pair_video_with_annotation(video, root))

    def test_pair_with_non_contiguous_indices(self) -> None:
        # Indices 1 and 3 present; 2 absent — neither the
        # pairer nor the collector should care that 2 is missing.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_1 = _touch_video(root, "video(1).avi")
            _touch(root, "video(1).txt")
            video_3 = _touch_video(root, "video(3).avi")
            _touch(root, "video(3).txt")
            self.assertIsNotNone(pair_video_with_annotation(video_1, root))
            self.assertIsNotNone(pair_video_with_annotation(video_3, root))

    def test_pair_direct_match_takes_precedence_over_subfolder(self) -> None:
        # When both layouts exist, the direct match wins. This
        # matters in mixed-layout datasets where a tree already
        # has placeholder subfolders.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = _touch_video(root, "video(1).avi")
            direct = _touch(root, "video(1).txt")
            direct.write_text("direct\n", encoding="utf-8")
            _touch(root, "video(1)", "video(1).txt").write_text(
                "subfolder\n", encoding="utf-8",
            )
            found = pair_video_with_annotation(video, root)
            self.assertEqual(found, direct)


class CollectLe2iClipsTests(unittest.TestCase):
    """``collect_le2i_clips`` walks + pairs across nested layouts."""

    def test_collect_pairs_each_video_with_correct_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos"
            videos.mkdir()
            annotations = root / "annotations"
            annotations.mkdir()
            video_1 = _touch_video(videos, "video(1).avi")
            _touch(annotations, "video(1).txt").write_text(
                "0\n0\n1, 1, 100, 100, 200, 200\n", encoding="utf-8",
            )
            video_3 = _touch_video(videos, "video(3).avi")
            _touch(annotations, "video(3).txt").write_text(
                "100\n110\n", encoding="utf-8",
            )
            # Office / Lecture-room style: a third video with
            # no annotation whatsoever.
            video_5 = _touch_video(videos, "video(5).avi")
            pairs = collect_le2i_clips(
                videos, annotations_root=annotations,
                fallback_fps=25.0,
            )
            by_video = {p.video_path.name: p for p in pairs}
            self.assertEqual(len(pairs), 3)
            self.assertIsNotNone(by_video[video_1.name].annotation)
            self.assertIsNotNone(by_video[video_3.name].annotation)
            # Missing GT → ``annotation=None``, FPS fallback surfaces.
            self.assertIsNone(by_video[video_5.name].annotation)
            self.assertEqual(by_video[video_5.name].fps, 25.0)

    def test_collect_descends_into_nested_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos"
            annotations = root / "annotations"
            videos.mkdir()
            annotations.mkdir()
            video = _touch_video(videos, "fall-01-cam0.avi")
            _touch(annotations, "fall-01-cam0", "fall-01-cam0.txt").write_text(
                "10\n20\n", encoding="utf-8",
            )
            pairs = collect_le2i_clips(
                videos, annotations_root=annotations,
                fallback_fps=30.0,
            )
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].video_path, video)
            self.assertIsNotNone(pairs[0].annotation)
            self.assertEqual(pairs[0].annotation.fall_window, (10, 20))

    def test_collect_returns_empty_when_videos_root_missing(self) -> None:
        # Missing root → empty list, NOT a crash. Real production
        # code pre-validates; the helper is permissive.
        pairs = collect_le2i_clips(Path("/does/not/exist"))
        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
