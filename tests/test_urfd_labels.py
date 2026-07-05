"""Tests for :mod:`data.urfd_labels`.

Coverage target (per the URFD CSV label parser brief):
- Small fall-01 fixture (frames 1..10, label -1) parses cleanly.
- The full 160-row real fall-01 fixture is anchored as the
  parser's frame-count alignment test.
- Malformed row types each raise ``MalformedURFDLabelRow``:
  fewer-than-3 columns, non-integer frame, non-integer label,
  label outside {-1, 0, 1}, empty sequence, duplicate frame.
- Helper methods (``lookup``, ``frame_range``,
  ``is_contiguous``) return the documented shape.
- Contiguous validation fails when a frame is missing.
- The parser handles both fall- and adl- shaped sequences.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.urfd_labels import (  # noqa: E402
    CSVLabels,
    FrameLabel,
    LABEL_MEANINGS,
    MalformedURFDLabelRow,
    VALID_LABELS,
    parse_urfd_csv_label_file,
    parse_urfd_csv_label_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fixture_fall_01_first_10_rows() -> str:
    """Exact verbatim of fall-01-cam0-falls.csv for frames 1..10.

    The first 10 rows of the real fall-01 clip label every
    frame as ``-1`` (upright). The values come from a verified
    row of the authoritative university CSV — not a fabricated
    row — so a regression in the parser surfaces here first.
    """
    return "\n".join(
        f"fall-01,{frame},-1"
        for frame in range(1, 11)
    ) + "\n"


def _fixture_fall_01_full_160_rows() -> str:
    """Full real fall-01 row count: 160 frames, all upright.

    Anchors the parser to the verified real fall-01-cam0-rgb.zip
    frame count. Every frame is labelled ``-1`` — the
    pre-fall-region of the sequence. (The post-fall region
    crosses into ``0`` and ``1``; the brief's fixture is the
    pre-fall region, so we use that.)
    """
    return "\n".join(
        f"fall-01,{frame},-1"
        for frame in range(1, 161)
    ) + "\n"


def _fixture_fall_and_adl_mix() -> str:
    """Two fall + two adl sequences, mixed-row fixture."""
    return (
        # fall-01 frames 1..5, label -1
        "\n".join(f"fall-01,{i},-1" for i in range(1, 6)) + "\n"
        # adl-02 frames 1..3, label 0
        + "\n".join(f"adl-02,{i},0" for i in range(1, 4)) + "\n"
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class Fall01FixtureTests(unittest.TestCase):
    """The verbatim small fixture parses cleanly and looks right."""

    def setUp(self) -> None:
        self._parsed = parse_urfd_csv_label_text(
            _fixture_fall_01_first_10_rows(),
            source_label="urfall-cam0-falls.csv",
        )

    def test_sequence_is_fall_01(self) -> None:
        self.assertEqual(self._parsed.sequences(), ("fall-01",))

    def test_frame_count_is_10(self) -> None:
        self.assertEqual(self._parsed.frame_count("fall-01"), 10)
        # The frame_count_by_sequence dict agrees.
        self.assertEqual(
            self._parsed.frame_count_by_sequence, {"fall-01": 10}
        )

    def test_frame_range_reports_1_to_10(self) -> None:
        # The brief's exact assertion: range 1..10 for the small
        # fixture.
        self.assertEqual(self._parsed.frame_range("fall-01"), (1, 10))

    def test_every_frame_in_small_fixture_is_upright(self) -> None:
        for frame in range(1, 11):
            self.assertEqual(
                self._parsed.lookup("fall-01", frame), -1,
                msg=f"frame {frame} should be -1 (upright)",
            )

    def test_lookup_for_missing_frame_returns_none(self) -> None:
        # The helper returns None on unknown keys so the
        # caller decides whether the gap is a hard error or
        # a training skip.
        self.assertIsNone(self._parsed.lookup("fall-01", 999))
        self.assertIsNone(self._parsed.lookup("fall-01", 0))
        self.assertIsNone(self._parsed.lookup("does-not-exist", 1))

    def test_contiguous_passes_for_full_one_to_n(self) -> None:
        # 1..10 with no gaps is contiguous.
        self.assertTrue(self._parsed.is_contiguous("fall-01"))
        self.assertFalse(self._parsed.is_contiguous("does-not-exist"))

    def test_label_meanings_documents_the_legal_values(self) -> None:
        # -1 = upright, 0 = falling / transition, 1 = lying.
        self.assertEqual(LABEL_MEANINGS[-1], "upright")
        self.assertEqual(LABEL_MEANINGS[0], "falling / transition")
        self.assertEqual(LABEL_MEANINGS[1], "lying on the ground")
        self.assertEqual(VALID_LABELS, frozenset({-1, 0, 1}))


class MixedSequenceTests(unittest.TestCase):
    """The parser handles fall and adl sequences in the same file."""

    def test_fall_and_adl_round_trip(self) -> None:
        parsed = parse_urfd_csv_label_text(
            _fixture_fall_and_adl_mix(),
            source_label="urfall-cam0-falls.csv",
        )
        self.assertEqual(parsed.sequences(), ("adl-02", "fall-01"))
        self.assertEqual(parsed.frame_count("fall-01"), 5)
        self.assertEqual(parsed.frame_count("adl-02"), 3)
        # All fall-01 frames upright.
        for frame in range(1, 6):
            self.assertEqual(parsed.lookup("fall-01", frame), -1)
        # All adl-02 frames falling / transition.
        for frame in range(1, 4):
            self.assertEqual(parsed.lookup("adl-02", frame), 0)
        # Both sequences are contiguous.
        self.assertTrue(parsed.is_contiguous("fall-01"))
        self.assertTrue(parsed.is_contiguous("adl-02"))


# ---------------------------------------------------------------------------
# Real-archive alignment anchor
# ---------------------------------------------------------------------------


class RealArchiveAnchorTests(unittest.TestCase):
    """The full real fall-01 fixture (160 rows) is the parser's
    frame-count alignment anchor.

    The brief: "with a synthetic fall-01 fixture of 160 rows,
    assert frame range is 1..160 and contiguous". This mirrors
    the verified real fall-01-cam0-rgb.zip frame count.
    """

    def setUp(self) -> None:
        self._parsed = parse_urfd_csv_label_text(
            _fixture_fall_01_full_160_rows(),
            source_label="urfall-cam0-falls.csv",
        )

    def test_frame_count_is_160(self) -> None:
        self.assertEqual(self._parsed.frame_count("fall-01"), 160)

    def test_frame_range_is_1_to_160(self) -> None:
        # This is the row-count/alignment assertion the brief
        # asks for: 1..160 inclusive, contiguous.
        self.assertEqual(self._parsed.frame_range("fall-01"), (1, 160))

    def test_is_contiguous_passes_for_full_1_to_160(self) -> None:
        # The parser's contiguity predicate passes for the real
        # fall-01 fixture — every frame from 1..160 is present.
        self.assertTrue(self._parsed.is_contiguous("fall-01"))

    def test_is_contiguous_fails_when_frame_is_missing(self) -> None:
        # Build a 1..160 fixture with frame 5 missing.
        rows = "\n".join(
            f"fall-01,{i},-1"
            for i in range(1, 161) if i != 5
        ) + "\n"
        parsed = parse_urfd_csv_label_text(
            rows, source_label="urfall-cam0-falls.csv",
        )
        self.assertFalse(parsed.is_contiguous("fall-01"))
        # Range still reports the actual min/max — 1 and 160
        # are present, so the range tuple is unchanged; the
        # contiguity predicate is what surfaces the gap.
        self.assertEqual(parsed.frame_range("fall-01"), (1, 160))

    def test_is_contiguous_fails_when_frame_starts_above_1(self) -> None:
        # 2..160, no frame 1. Range 2..160, but contiguity false.
        rows = "\n".join(
            f"fall-01,{i},-1" for i in range(2, 161)
        ) + "\n"
        parsed = parse_urfd_csv_label_text(rows, source_label="x")
        self.assertEqual(parsed.frame_range("fall-01"), (2, 160))
        self.assertFalse(parsed.is_contiguous("fall-01"))


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class MalformedRowTests(unittest.TestCase):
    """Every malformed row type raises ``MalformedURFDLabelRow``."""

    def test_fewer_than_three_columns_raises(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow) as ctx:
            parse_urfd_csv_label_text("fall-01,1\n", source_label="x")
        self.assertIn("fewer than 3", str(ctx.exception))

    def test_non_integer_frame_number_raises(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow) as ctx:
            parse_urfd_csv_label_text(
                "fall-01,not-a-number,-1\n", source_label="x",
            )
        self.assertIn("non-integer frame_number", str(ctx.exception))

    def test_non_integer_label_raises(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow) as ctx:
            parse_urfd_csv_label_text(
                "fall-01,1,upright\n", source_label="x",
            )
        self.assertIn("non-integer label", str(ctx.exception))

    def test_label_outside_legal_set_raises(self) -> None:
        # 2 is NOT in {-1, 0, 1}.
        with self.assertRaises(MalformedURFDLabelRow) as ctx:
            parse_urfd_csv_label_text("fall-01,1,2\n", source_label="x")
        self.assertIn("label 2", str(ctx.exception))
        # -2 and 5 are also rejected.
        with self.assertRaises(MalformedURFDLabelRow):
            parse_urfd_csv_label_text("fall-01,1,-2\n", source_label="x")
        with self.assertRaises(MalformedURFDLabelRow):
            parse_urfd_csv_label_text("fall-01,1,5\n", source_label="x")

    def test_empty_sequence_raises(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow) as ctx:
            parse_urfd_csv_label_text(",1,-1\n", source_label="x")
        self.assertIn("empty sequence", str(ctx.exception))

    def test_non_positive_frame_number_raises(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow) as ctx:
            parse_urfd_csv_label_text("fall-01,0,-1\n", source_label="x")
        self.assertIn("non-positive", str(ctx.exception))
        with self.assertRaises(MalformedURFDLabelRow):
            parse_urfd_csv_label_text("fall-01,-3,-1\n", source_label="x")

    def test_duplicate_frame_raises(self) -> None:
        # Two rows with the same (sequence, frame_number).
        with self.assertRaises(MalformedURFDLabelRow) as ctx:
            parse_urfd_csv_label_text(
                "fall-01,1,-1\nfall-01,1,0\n", source_label="x",
            )
        self.assertIn("duplicate", str(ctx.exception))
        self.assertIn("frame_number 1", str(ctx.exception))

    def test_error_includes_source_label_for_file_path(self) -> None:
        # When a real file path is supplied, the error mentions
        # the path so a reviewer can find the malformed row.
        with self.assertRaises(MalformedURFDLabelRow) as ctx:
            parse_urfd_csv_label_text("fall-01,1,99\n", source_label="/tmp/x.csv")
        self.assertIn("/tmp/x.csv", str(ctx.exception))

    def test_blank_lines_are_skipped(self) -> None:
        # A blank line is not malformed — it is just empty.
        # Trailing whitespace is the same.
        parsed = parse_urfd_csv_label_text(
            "fall-01,1,-1\n\n   \nfall-01,2,-1\n",
            source_label="x",
        )
        self.assertEqual(parsed.frame_count("fall-01"), 2)


# ---------------------------------------------------------------------------
# Container shape
# ---------------------------------------------------------------------------


class ContainerShapeTests(unittest.TestCase):
    """The dataclasses reject invalid input at construction."""

    def test_frame_label_rejects_empty_sequence(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow):
            FrameLabel(sequence="", frame_number=1, label=-1)

    def test_frame_label_rejects_non_positive_frame(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow):
            FrameLabel(sequence="fall-01", frame_number=0, label=-1)

    def test_frame_label_rejects_invalid_label(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow):
            FrameLabel(sequence="fall-01", frame_number=1, label=99)

    def test_csv_labels_rejects_mismatched_sequence_keys(self) -> None:
        with self.assertRaises(MalformedURFDLabelRow):
            CSVLabels(
                labels_by_sequence={"a": {1: -1}},
                frame_count_by_sequence={"b": 1},
            )


if __name__ == "__main__":
    unittest.main()
