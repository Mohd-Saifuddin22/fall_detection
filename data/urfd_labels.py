"""URFD cam0 CSV label parser.

Parses the two authoritative university URFD cam0 CSV files:

- ``urfall-cam0-falls.csv`` — one row per annotated frame of
  every fall clip (fall-01..fall-30).
- ``urfall-cam0-adls.csv`` — one row per annotated frame of
  every adl clip (adl-01..adl-40).

CSV format (verified):
    - No header.
    - 11 comma-separated columns.
    - Only the first 3 columns are used:
        1. ``sequence``     — e.g. ``"fall-01"`` / ``"adl-01"``.
        2. ``frame_number``  — 1-based integer (the clip's
           intra-clip frame index, NOT a global video
           timestamp).
        3. ``label``         — one of ``{-1, 0, 1}``:
            - ``-1`` = upright (no fall in progress)
            - `` 0`` = falling / transition
            - `` 1`` = lying on the ground

      Columns 4..11 are ignored. (Real university CSVs carry
      extra per-frame metadata — bounding boxes, sub-classes,
      etc. — that the project does not consume.)

Failure modes (all raise :class:`MalformedURFDLabelRow`):
    - fewer than 3 columns
    - empty sequence
    - non-integer frame number
    - frame number < 1
    - non-integer label
    - label outside ``{-1, 0, 1}``
    - duplicate frame number within the same sequence

Public API:
    - :func:`parse_urfd_csv_label_file` — file-based entry.
    - :func:`parse_urfd_csv_label_text` — text-based entry
      (testable, no real CSV file required).
    - :class:`CSVLabels` — structured output with helper
      methods for lookups, frame ranges, and contiguous
      validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The only legal label values. ``-1`` = upright,
#: ``0`` = falling / transition, ``1`` = lying on the ground.
VALID_LABELS: frozenset[int] = frozenset({-1, 0, 1})

#: A human-readable mapping of label value → description. Useful
#: for run summaries and reviewer logs.
LABEL_MEANINGS: dict[int, str] = {
    -1: "upright",
    0: "falling / transition",
    1: "lying on the ground",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MalformedURFDLabelRow(ValueError):
    """Raised on any malformed or out-of-range URFD label CSV row.

    Inherits :class:`ValueError` so callers that catch the
    broader family still work.
    """


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameLabel:
    """One annotated frame: ``(sequence, frame_number, label)``."""

    sequence: str
    frame_number: int
    label: int

    def __post_init__(self) -> None:
        # Mirror the parser's invariants so a hand-built
        # :class:`FrameLabel` is rejected the same way a malformed
        # CSV row would be.
        if not isinstance(self.sequence, str) or not self.sequence.strip():
            raise MalformedURFDLabelRow(
                f"empty sequence on FrameLabel: {self!r}"
            )
        if not isinstance(self.frame_number, int) or self.frame_number < 1:
            raise MalformedURFDLabelRow(
                f"non-positive frame_number on FrameLabel: {self!r}"
            )
        if self.label not in VALID_LABELS:
            raise MalformedURFDLabelRow(
                f"label {self.label!r} outside {sorted(VALID_LABELS)} on FrameLabel"
            )


@dataclass(frozen=True)
class CSVLabels:
    """One parsed CSV file's label lookup.

    Index shape:
        - ``labels_by_sequence[sequence][frame_number] == label``
          for every annotated row.
        - ``frame_count_by_sequence[sequence]`` — number of
          annotated rows for the sequence (== ``len(...)``).

    Helper methods cover the lookups the rest of the pipeline
    needs (e.g. Pipeline A's contiguity check before building a
    training example).
    """

    labels_by_sequence: dict[str, dict[int, int]]
    frame_count_by_sequence: dict[str, int]

    def __post_init__(self) -> None:
        # Sanity: every sequence in the count map must be in
        # the labels map, and vice versa. Defends against a
        # hand-built partial :class:`CSVLabels`; the parser
        # already enforces both halves are consistent.
        if set(self.labels_by_sequence) != set(self.frame_count_by_sequence):
            raise MalformedURFDLabelRow(
                "labels_by_sequence and frame_count_by_sequence "
                "must carry the same sequence keys"
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, sequence: str, frame_number: int) -> int | None:
        """Return the label for ``(sequence, frame_number)``, or ``None``.

        Returns ``None`` (not raises) when either the sequence
        or the frame number is unknown — the caller decides
        whether the gap is a training skip or a hard error.
        """
        per_sequence = self.labels_by_sequence.get(sequence)
        if per_sequence is None:
            return None
        return per_sequence.get(frame_number)

    def sequences(self) -> tuple[str, ...]:
        """Tuple of sequence names, sorted."""
        return tuple(sorted(self.labels_by_sequence))

    def frame_count(self, sequence: str) -> int:
        """Annotated-row count for ``sequence``, or 0 if unknown."""
        return self.frame_count_by_sequence.get(sequence, 0)

    def frame_range(self, sequence: str) -> tuple[int, int]:
        """``(min_frame, max_frame)`` for ``sequence``.

        Returns ``(0, 0)`` when the sequence is unknown. The
        range is INCLUSIVE on both ends — the caller computes
        ``max - min + 1`` for the expected row count.
        """
        per_sequence = self.labels_by_sequence.get(sequence)
        if not per_sequence:
            return (0, 0)
        frames = per_sequence.keys()
        return (min(frames), max(frames))

    def is_contiguous(self, sequence: str) -> bool:
        """``True`` iff ``sequence``'s frames form a contiguous 1..N.

        The "contiguous" check is what Pipeline A's loader uses
        to confirm every clip has a complete label set before
        building a training example. A clip with frame 5
        missing is genuinely broken — the parser surfaces it.
        """
        per_sequence = self.labels_by_sequence.get(sequence)
        if not per_sequence:
            return False
        frames = sorted(per_sequence)
        if frames[0] != 1:
            return False
        expected = list(range(1, len(frames) + 1))
        return frames == expected


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_urfd_csv_label_text(
    text: str,
    *,
    source_label: str = "<text>",
) -> CSVLabels:
    """Parse the body of one URFD label CSV.

    Args:
        text: CSV body. Blank lines are skipped; lines with
            only whitespace are also skipped. Every other
            non-empty line must have at least 3 comma-separated
            fields; column 4..N are ignored.
        source_label: surfaced in error messages so a reviewer
            knows which file is malformed. Default ``"<text>"``;
            the file-based entry passes the file path so a real
            file's error points at the on-disk source.

    Returns:
        A populated :class:`CSVLabels`.

    Raises:
        MalformedURFDLabelRow: on any malformed row.
    """
    labels_by_sequence: dict[str, dict[int, int]] = {}
    line_number = 0
    for raw_line in text.splitlines():
        line_number += 1
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            raise MalformedURFDLabelRow(
                f"{source_label} line {line_number}: fewer than 3 "
                f"comma-separated columns (got {len(parts)}): {raw_line!r}"
            )
        sequence, frame_str, label_str = parts[0], parts[1], parts[2]

        if not sequence:
            raise MalformedURFDLabelRow(
                f"{source_label} line {line_number}: empty sequence"
            )

        try:
            frame_number = int(frame_str)
        except (TypeError, ValueError):
            raise MalformedURFDLabelRow(
                f"{source_label} line {line_number}: non-integer "
                f"frame_number {frame_str!r}"
            ) from None
        if frame_number < 1:
            raise MalformedURFDLabelRow(
                f"{source_label} line {line_number}: non-positive "
                f"frame_number {frame_number}"
            )

        try:
            label = int(label_str)
        except (TypeError, ValueError):
            raise MalformedURFDLabelRow(
                f"{source_label} line {line_number}: non-integer "
                f"label {label_str!r}"
            ) from None
        if label not in VALID_LABELS:
            raise MalformedURFDLabelRow(
                f"{source_label} line {line_number}: label {label} "
                f"outside {sorted(VALID_LABELS)}"
            )

        per_sequence = labels_by_sequence.setdefault(sequence, {})
        if frame_number in per_sequence:
            raise MalformedURFDLabelRow(
                f"{source_label} line {line_number}: duplicate "
                f"frame_number {frame_number} in sequence {sequence!r}"
            )
        per_sequence[frame_number] = label

    return CSVLabels(
        labels_by_sequence=labels_by_sequence,
        frame_count_by_sequence={
            seq: len(per_seq) for seq, per_seq in labels_by_sequence.items()
        },
    )


def parse_urfd_csv_label_file(path: Path | str) -> CSVLabels:
    """Parse one URFD label CSV file from disk.

    Thin wrapper around :func:`parse_urfd_csv_label_text` that
    reads the file and surfaces the file path in error messages.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return parse_urfd_csv_label_text(text, source_label=str(path))


__all__: tuple[str, ...] = (
    "CSVLabels",
    "FrameLabel",
    "LABEL_MEANINGS",
    "MalformedURFDLabelRow",
    "VALID_LABELS",
    "parse_urfd_csv_label_file",
    "parse_urfd_csv_label_text",
)
