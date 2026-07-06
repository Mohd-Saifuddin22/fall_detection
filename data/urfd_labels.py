"""URFD cam0 CSV label parser + window labeling.

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

Sparse-label note (real university CSVs):
    Real university URFD CSVs are sparse: some RGB frames have
    no corresponding label row (only frames an annotator actually
    labelled appear). The parser still requires every
    PRESENT row to be valid (the failure-mode list above still
    applies per-row); it does NOT pad missing frames with
    placeholders. The downstream :func:`label_window` SKIPS
    frames not in the CSV rather than raising.

Public API:
    - :func:`parse_urfd_csv_label_file` — file-based entry.
    - :func:`parse_urfd_csv_label_text` — text-based entry
      (testable, no real CSV file required).
    - :class:`CSVLabels` — structured output with helper
      methods for lookups, frame ranges, and contiguous
      validation.
    - :func:`clip_id_to_sequence` — manifest clip id → CSV
      sequence mapping.
    - :class:`WindowLabelingRule` (and
      :class:`DefaultWindowLabelingRule`) — pluggable per-frame
      → window-label rule so Issue 006 can swap the rule
      without re-cropping.
    - :func:`label_window` — one-shot call that maps a clip id
      + the window's frame indices to ``(window_label,
      is_confuser)`` using the supplied rule.

Frame-index alignment decision
-----------------------------

The crop metadata sidecar (``window.frame_indices[i]`` carried
into ``"frame_index"``) is **0-based** — it is the zero-based
absolute index produced by :func:`enumerate` in
:func:`perception.tracker.run_tracker_on_folder` and propagated
through :class:`cropping.track_windows.TrackedBox` →
:class:`cropping.track_windows.TrackWindow` →
:mod:`cropping.runner`. The URFD CSV ``frame_number`` is
**1-based** (and university RGB filenames are 1-based too:
``fall-01-cam0-rgb-001.png``). The two are aligned only after
applying a ``+1`` shift.

So a real crop caller MUST apply ``+1`` to every crop frame
index before CSV lookup. The supported entry points for that
alignment are:

- :data:`CROP_TO_CSV_FRAME_OFFSET` — the constant ``1``. Pass
  it explicitly to :func:`label_window` as
  ``frame_index_offset``.
- :func:`label_window_from_crop_meta` — wrapper that applies
  :data:`CROP_TO_CSV_FRAME_OFFSET` for you. **Use this from
  every real crop path** (Issue 006, the full pipeline
  notebook's smoke cell, the cropping runner's downstream
  training step). The raw :func:`label_window` is the
  primitive — kept for synthetic fixtures, future CSV
  formats, and explicit-shift adapters — and is reserved for
  callers that genuinely do not want the crop convention
  applied.

The default for :func:`label_window` is ``frame_index_offset=0``
(no shift) because the function is the project-wide primitive
that any caller — synthetic fixtures, future CSV formats,
adapters — can plug into without inheriting the crop
convention. Real crop callers must opt in via the helper or
the constant; the default is NOT a no-op for real crop data.

The :func:`label_window` signature still exposes
``frame_index_offset`` as an explicit parameter — a future
change to the crop metadata's frame-indexing convention can
be applied at the call site rather than at the parser. The
helper is the supported default for crop callers; it is NOT
a replacement for the primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

#: Offset a real crop caller must apply to a 0-based crop metadata
#: frame index to align it with the 1-based URFD CSV ``frame_number``.
#:
#: Crop metadata ``frame_index`` is 0-based (perception ``enumerate``);
#: URFD CSV ``frame_number`` is 1-based. The two are aligned by adding
#: this constant. Exposed as a single named value so the alignment is
#: documented at the call site (the literal ``1`` is the alignment
#: contract, not a magic number) and so :func:`label_window_from_crop_meta`
#: is the single supported entry point for real crop callers.
#:
#: See the module-level "Frame-index alignment decision" section for
#: the full rationale.
CROP_TO_CSV_FRAME_OFFSET: int = 1


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


# ---------------------------------------------------------------------------
# Clip-id ↔ CSV-sequence mapping
# ---------------------------------------------------------------------------


#: Prefix the manifest builder uses for every URFD-derived clip.
#: Real layout: ``urfd-debug-{sequence}-cam0-rgb`` where
#: ``{sequence}`` is e.g. ``"fall-01"`` or ``"adl-02"``.
_URFD_CLIP_ID_PREFIX: str = "urfd-debug-"
#: Suffix every Issue 003 staged folder carries. The manifest
#: id is built from the on-disk folder name, so the suffix
#: must be present for the mapping to round-trip.
_URFD_CLIP_ID_SUFFIX: str = "-cam0-rgb"


def clip_id_to_sequence(clip_id: str) -> str:
    """Map a manifest clip id to its URFD CSV sequence.

    Examples:
        ``urfd-debug-fall-01-cam0-rgb``  →  ``"fall-01"``
        ``urfd-debug-adl-02-cam0-rgb``   →  ``"adl-02"``

    Raises:
        ValueError: when ``clip_id`` does not match the expected
            ``urfd-debug-<sequence>-cam0-rgb`` shape, or when
            ``<sequence>`` does not start with ``"fall-"`` or
            ``"adl-"``. Unknown sequences are not URFD fall / ADL
            clips — the loader's caller decides whether to
            surface the error.
    """
    if not isinstance(clip_id, str):
        raise ValueError(
            f"clip_id must be a string, got {type(clip_id).__name__}"
        )
    if not clip_id.startswith(_URFD_CLIP_ID_PREFIX):
        raise ValueError(
            f"clip_id {clip_id!r} is not a URFD clip id — expected "
            f"prefix {_URFD_CLIP_ID_PREFIX!r}"
        )
    if not clip_id.endswith(_URFD_CLIP_ID_SUFFIX):
        raise ValueError(
            f"clip_id {clip_id!r} is not a URFD clip id — expected "
            f"suffix {_URFD_CLIP_ID_SUFFIX!r}"
        )
    sequence = clip_id[len(_URFD_CLIP_ID_PREFIX):-len(_URFD_CLIP_ID_SUFFIX)]
    if not sequence:
        raise ValueError(
            f"clip_id {clip_id!r} has an empty sequence between the "
            f"prefix and suffix"
        )
    if not (sequence.startswith("fall-") or sequence.startswith("adl-")):
        raise ValueError(
            f"clip_id {clip_id!r} yields sequence {sequence!r} which "
            "is not a recognised URFD fall/adl sequence (must start "
            "with 'fall-' or 'adl-')"
        )
    return sequence


def sequence_to_clip_type(sequence: str) -> str:
    """Map a URFD CSV sequence to its clip type.

    Returns ``"fall"`` for ``fall-*`` and ``"adl"`` for ``adl-*``.
    Other prefixes raise :class:`ValueError` — the loader is
    expected to pass sequences that have already been validated
    by :func:`clip_id_to_sequence`.
    """
    if sequence.startswith("fall-"):
        return "fall"
    if sequence.startswith("adl-"):
        return "adl"
    raise ValueError(
        f"sequence {sequence!r} is not a recognised URFD fall/adl sequence"
    )


# ---------------------------------------------------------------------------
# Window labeling
# ---------------------------------------------------------------------------


#: Window-label return values — the same strings the rest of the
#: pipeline uses (``pipeline_a.label_to_int`` etc.).
WINDOW_LABEL_FALL: str = "fall"
WINDOW_LABEL_NO_FALL: str = "no_fall"

#: Sentinel returned by :func:`label_window` when a fall clip's
#: window has zero available CSV labels (e.g. the window's
#: frame indices fall entirely outside the labelled range).
#: Real university ADL labels are sparse — some RGB frames have
#: no corresponding CSV row — so the loader must treat a
#: no-label-available window as a separate outcome, NOT a
#: silent fall or a hard error.
#:
#: Issue 006 can use this sentinel to drop the example
#: deterministically (the project's "no labels available" is a
#: legitimate reason to skip an example, distinct from
#: "labels say no_fall" which IS training data).
WINDOW_LABEL_UNLABELED: str = "unlabeled"


class WindowLabelingError(ValueError):
    """Raised on any malformed / inconsistent label-window call.

    Inherits :class:`ValueError` so callers that catch the
    broader family still work.
    """


class WindowLabelingRule:
    """Strategy that turns a list of per-frame labels into a window label.

    Issue 006 may want to swap the default rule for a stricter or
    looser one — the Issue 003 crop shards are NOT relabelled
    here, the rule is applied at training time so a future
    rule change does not require re-cropping.

    Implementations must be deterministic and pure (no I/O).
    """

    def apply(
        self,
        clip_type: str,
        per_frame_labels: tuple[int, ...],
    ) -> tuple[str, bool]:
        """Return ``(window_label, is_confuser)``.

        ``clip_type`` is one of ``"fall"`` / ``"adl"`` (see
        :func:`sequence_to_clip_type`). ``per_frame_labels`` is
        the per-frame integer labels in the order the caller
        passed them. ``is_confuser`` is the caller-flag for
        "this window is a confuser example" — see the default
        rule below for the canonical meaning.
        """
        raise NotImplementedError


class DefaultWindowLabelingRule(WindowLabelingRule):
    """The default rule Issue 005 ships with.

    Fall clips:
        - if any frame in the window has label ``0`` (falling)
          or ``1`` (lying), the window is ``fall``.
        - if every frame in the window is ``-1`` (upright), the
          window is ``no_fall`` — this is the pre-fall region
          of a fall clip, which the default rule treats as a
          clean negative rather than a noisy positive.
        - if the window has zero available labels (the real
          university ADL CSV is sparse; some RGB frames have
          no corresponding label row, and a window's frame
          indices can all land outside the labelled range),
          the rule returns the explicit sentinel
          :data:`WINDOW_LABEL_UNLABELED` so the loader can
          deterministically skip the example. It does NOT
          raise — "no labels available" is a legitimate
          outcome, not a bug.

    ADL clips:
        - always ``no_fall`` (the source clip type is non-fall).
        - if any available frame in the window has label
          ``1`` (lying), the window is flagged
          ``is_confuser=True`` so a downstream training run
          can choose whether to mix it into the no-fall pool,
          weight it down, or drop it. The default keeps the
          flag on so the Project's evaluation can study
          confuser-aware behaviour without further label
          surgery.
        - if the window has zero available labels, the rule
          still returns ``no_fall, is_confuser=False`` — an
          ADL window with no labels is still a non-fall by
          source-clip type, and the confuser flag requires a
          positive label-1 signal. A zero-label ADL window is
          a normal no-fall training example; sparse labels
          do not change that.
    """

    def apply(
        self,
        clip_type: str,
        per_frame_labels: tuple[int, ...],
    ) -> tuple[str, bool]:
        if clip_type == "fall":
            if not per_frame_labels:
                # No labels available → explicit sentinel. The
                # loader / Issue 006 can drop the example
                # deterministically. Distinct from
                # "no_fall" (which is a real training
                # signal) and from "fall" (which would be
                # wrong here).
                return WINDOW_LABEL_UNLABELED, False
            window_label = (
                WINDOW_LABEL_FALL
                if any(lbl in (0, 1) for lbl in per_frame_labels)
                else WINDOW_LABEL_NO_FALL
            )
            return window_label, False
        if clip_type == "adl":
            # ADL clip type is non-fall by definition; a
            # zero-label window is still ``no_fall``. The
            # confuser flag requires an actual label-1
            # signal — a sparse label set must not flip the
            # flag to True on absence of evidence.
            is_confuser = any(lbl == 1 for lbl in per_frame_labels)
            return WINDOW_LABEL_NO_FALL, is_confuser
        raise WindowLabelingError(
            f"unknown clip_type {clip_type!r}; expected 'fall' or 'adl'"
        )


#: The default rule applied by :func:`label_window` when no
#: explicit rule is supplied. Issue 006 can construct a
#: different :class:`WindowLabelingRule` subclass and pass it
#: to :func:`label_window` without re-cropping.
DEFAULT_WINDOW_LABELING_RULE: WindowLabelingRule = (
    DefaultWindowLabelingRule()
)


def label_window(
    clip_id: str,
    frame_indices: Sequence[int],
    csv_labels: CSVLabels,
    *,
    frame_index_offset: int = 0,
    labeling_rule: WindowLabelingRule = DEFAULT_WINDOW_LABELING_RULE,
) -> tuple[str, bool]:
    """Assign a clean window label + confuser flag for one window.

    Args:
        clip_id: manifest clip id (``urfd-debug-fall-01-cam0-rgb`` or
            ``urfd-debug-adl-01-cam0-rgb``).
        frame_indices: the window's frame indices (1-based absolute
            frame numbers — the same indexing the URFD CSV uses).
        csv_labels: the parsed :class:`CSVLabels` from
            :func:`parse_urfd_csv_label_text` /
            :func:`parse_urfd_csv_label_file`.
        frame_index_offset: integer added to each ``frame_index``
            before CSV lookup. Default ``0`` (no shift).

            The default ``0`` is the project-wide primitive
            default — synthetic fixtures, future CSV formats,
            and explicit-shift adapters all use it without
            inheriting any crop-specific convention.

            **Real crop callers** carry 0-based frame indices
            (perception ``enumerate`` → :class:`TrackedBox`
            → :class:`TrackWindow` → crop metadata
            ``frame_index``) and must apply ``+1`` before CSV
            lookup. The supported entry points are
            :data:`CROP_TO_CSV_FRAME_OFFSET` (pass as this
            argument) or, easier, :func:`label_window_from_crop_meta`
            (the wrapper that applies the offset for you).

            The parameter is still exposed so a future change
            to the crop metadata's frame-indexing convention
            can be applied at the call site rather than at the
            parser.
        labeling_rule: pluggable rule. Defaults to
            :data:`DEFAULT_WINDOW_LABELING_RULE` (the Issue 005
            default). Issue 006 may construct a different
            subclass to swap the rule without re-cropping.

    Returns:
        ``(window_label, is_confuser)`` where
        ``window_label`` is one of :data:`WINDOW_LABEL_FALL` /
        :data:`WINDOW_LABEL_NO_FALL` /
        :data:`WINDOW_LABEL_UNLABELED` and ``is_confuser`` is
        ``True`` iff the rule flagged this window as a
        confuser example (default rule: ADL window with at
        least one available label ``1`` (lying)).

    Sparse-label handling (real university ADL is sparse):
        Real university ADL CSVs skip frames where no
        annotator labelled the RGB. A window's frame indices
        may all fall outside the labelled range, or only
        partially overlap it. :func:`label_window` SKIPS
        frames not present in the CSV — the missing ones
        contribute no signal. The default rule's
        zero-availability paths return the explicit
        :data:`WINDOW_LABEL_UNLABELED` sentinel for fall
        clips and ``("no_fall", False)`` for ADL clips so the
        downstream loader can skip / weight down the example
        without crashing.

    Raises:
        WindowLabelingError: ONLY on a genuine bug — not on
            sparse CSV labels. Specifically:

            - ``frame_indices`` is empty.
            - ``clip_id`` does not match the expected
              ``urfd-debug-<sequence>-cam0-rgb`` shape.
            - A non-integer frame index in the caller's list.
            - A non-positive adjusted frame index.

            A non-contiguous CSV or a missing frame is **not**
            a failure mode — those are valid sparse-label
            outcomes and the function handles them silently.
    """
    if not frame_indices:
        raise WindowLabelingError(
            f"label_window: frame_indices is empty for clip_id "
            f"{clip_id!r}"
        )

    sequence = clip_id_to_sequence(clip_id)
    clip_type = sequence_to_clip_type(sequence)

    per_frame_labels: list[int] = []
    for raw_frame_index in frame_indices:
        if not isinstance(raw_frame_index, int):
            raise WindowLabelingError(
                f"label_window: frame index {raw_frame_index!r} is "
                f"not an integer (clip_id {clip_id!r})"
            )
        adjusted = raw_frame_index + frame_index_offset
        if adjusted < 1:
            raise WindowLabelingError(
                f"label_window: adjusted frame index {adjusted} is "
                f"non-positive (raw={raw_frame_index}, "
                f"frame_index_offset={frame_index_offset}, "
                f"clip_id {clip_id!r})"
            )
        # Sparse-label handling: missing frames in the CSV are
        # silently skipped, NOT raised. The default rule's
        # zero-availability paths handle the no-signal case
        # explicitly via the unlabeled sentinel (fall) or a
        # clean no_fall (adl). ``is_contiguous`` is now an
        # informational helper for callers who want to audit
        # the CSV independently — it is NOT a hard gate here.
        lbl = csv_labels.lookup(sequence, adjusted)
        if lbl is not None:
            per_frame_labels.append(lbl)

    return labeling_rule.apply(clip_type, tuple(per_frame_labels))


def label_window_from_crop_meta(
    clip_id: str,
    crop_frame_indices: Sequence[int],
    csv_labels: CSVLabels,
    *,
    labeling_rule: WindowLabelingRule = DEFAULT_WINDOW_LABELING_RULE,
) -> tuple[str, bool]:
    """Label a window whose ``frame_index`` came from real crop metadata.

    This is the supported entry point for every real crop caller
    (Issue 006's training step, the full pipeline notebook's
    smoke cell, any future training / eval code that reads crop
    shards). It wraps :func:`label_window` and applies the
    project-wide crop-metadata → CSV alignment in one place:

        - Crop metadata ``frame_index`` is 0-based (perception
          :func:`enumerate` → :class:`TrackedBox` →
          :class:`TrackWindow` → crop metadata sidecar).
        - URFD CSV ``frame_number`` is 1-based.
        - The two are aligned by adding
          :data:`CROP_TO_CSV_FRAME_OFFSET` (``+1``).

    Why a wrapper instead of changing
    :func:`label_window`'s default to ``+1``: the primitive's
    default of ``0`` is the project-wide convention that
    synthetic fixtures, future CSV formats, and explicit-shift
    adapters all rely on without inheriting any crop-specific
    convention. Real crop callers opt in to the crop convention
    by going through this wrapper — Issue 006 cannot
    accidentally skip the ``+1`` shift at a call site because
    the only supported crop-caller entry point already has it.

    Args:
        clip_id: Manifest clip id (``urfd-debug-fall-01-cam0-rgb``
            or ``urfd-debug-adl-01-cam0-rgb``).
        crop_frame_indices: The window's frame indices as they
            appear in real crop metadata — i.e. **0-based**
            absolute frame numbers from the perception
            ``enumerate``. A 32-frame window starting at the
            first frame of a clip is ``range(0, 32)`` here
            (not ``range(1, 33)``).
        csv_labels: The parsed :class:`CSVLabels` from
            :func:`parse_urfd_csv_label_text` /
            :func:`parse_urfd_csv_label_file`.
        labeling_rule: Pluggable rule; same shape as
            :func:`label_window`'s argument.

    Returns:
        ``(window_label, is_confuser)`` with the same semantics
        as :func:`label_window`.

    Raises:
        WindowLabelingError: passed through from
            :func:`label_window`. The wrapper does not swallow
            any error — empty ``crop_frame_indices``, non-int
            entries, non-positive adjusted indices, and unknown
            ``clip_id`` shapes all surface the same way they
            would via the primitive.

    See also:
        - :data:`CROP_TO_CSV_FRAME_OFFSET` — the ``+1`` constant
          applied internally.
        - :func:`label_window` — the primitive; use it only when
          you have a genuine reason not to go through this
          wrapper.
    """
    return label_window(
        clip_id,
        crop_frame_indices,
        csv_labels,
        frame_index_offset=CROP_TO_CSV_FRAME_OFFSET,
        labeling_rule=labeling_rule,
    )


__all__: tuple[str, ...] = (
    "CSVLabels",
    "CROP_TO_CSV_FRAME_OFFSET",
    "DefaultWindowLabelingRule",
    "DEFAULT_WINDOW_LABELING_RULE",
    "FrameLabel",
    "LABEL_MEANINGS",
    "MalformedURFDLabelRow",
    "VALID_LABELS",
    "WINDOW_LABEL_FALL",
    "WINDOW_LABEL_NO_FALL",
    "WINDOW_LABEL_UNLABELED",
    "WindowLabelingError",
    "WindowLabelingRule",
    "clip_id_to_sequence",
    "label_window",
    "label_window_from_crop_meta",
    "parse_urfd_csv_label_file",
    "parse_urfd_csv_label_text",
    "sequence_to_clip_type",
)
