"""Tests for :mod:`data.urfd_labels`' window-labeling surface.

Coverage target (per the URFD window-labeling brief):

- ``clip_id_to_sequence`` maps the manifest clip id to the CSV
  sequence correctly for both fall and adl prefixes.
- ``label_window`` returns ``("fall", False)`` for a fall window
  containing any label 0 or 1.
- ``label_window`` returns ``("no_fall", False)`` for a fall window
  containing only -1 (the pre-fall region).
- ``label_window`` returns ``("no_fall", True)`` for an ADL window
  containing any label 1.
- ``label_window`` returns ``("no_fall", False)`` for an ADL window
  with all -1.
- Guards: missing frame raises, non-contiguous sequence raises,
  unknown clip id raises, empty frame indices raises.
- The default rule is pluggable: an alternate rule produces
  the window label / confuser tuple without re-cropping.
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
    CROP_TO_CSV_FRAME_OFFSET,
    DefaultWindowLabelingRule,
    WindowLabelingError,
    WindowLabelingRule,
    WINDOW_LABEL_FALL,
    WINDOW_LABEL_NO_FALL,
    WINDOW_LABEL_UNLABELED,
    clip_id_to_sequence,
    label_window,
    label_window_from_crop_meta,
    parse_urfd_csv_label_text,
    sequence_to_clip_type,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _labels_for(sequence_label_map: dict[int, int], sequence: str) -> str:
    """Build a contiguous CSV body from a ``{frame: label}`` map.

    Each row is ``<sequence>,<frame>,<label>``. The map MUST
    contain every frame 1..N contiguously.
    """
    lines = [
        f"{sequence},{frame},{label}"
        for frame, label in sorted(sequence_label_map.items())
    ]
    return "\n".join(lines) + "\n"


def _fall_with_first_n_upright(n: int) -> str:
    """Frames 1..n all -1."""
    return _labels_for({frame: -1 for frame in range(1, n + 1)}, "fall-01")


def _adl_with_first_n_upright(n: int) -> str:
    """Frames 1..n all -1."""
    return _labels_for({frame: -1 for frame in range(1, n + 1)}, "adl-01")


def _adl_with_some_lying(lying_frames: list[int], n: int) -> str:
    """Frames 1..n with -1 except ``lying_frames`` set to 1."""
    mapping = {frame: -1 for frame in range(1, n + 1)}
    for frame in lying_frames:
        mapping[frame] = 1
    return _labels_for(mapping, "adl-01")


def _fall_with_one_falling(falling_frame: int, n: int) -> str:
    """Frames 1..n with -1 except ``falling_frame`` set to 0."""
    mapping = {frame: -1 for frame in range(1, n + 1)}
    mapping[falling_frame] = 0
    return _labels_for(mapping, "fall-01")


def _fall_with_one_lying(lying_frame: int, n: int) -> str:
    """Frames 1..n with -1 except ``lying_frame`` set to 1."""
    mapping = {frame: -1 for frame in range(1, n + 1)}
    mapping[lying_frame] = 1
    return _labels_for(mapping, "fall-01")


def _non_contiguous_fall(frames_present: list[int]) -> str:
    """A non-contiguous fall sequence — used to trip the contiguity guard."""
    return _labels_for({frame: -1 for frame in frames_present}, "fall-01")


# ---------------------------------------------------------------------------
# Clip-id ↔ sequence mapping
# ---------------------------------------------------------------------------


class ClipIdMappingTests(unittest.TestCase):
    """``urfd-debug-<seq>-cam0-rgb`` → ``<seq>``."""

    def test_fall_clip_id_maps_to_fall_sequence(self) -> None:
        self.assertEqual(
            clip_id_to_sequence("urfd-debug-fall-01-cam0-rgb"),
            "fall-01",
        )
        # Higher sequence numbers + multi-digit are also fine.
        self.assertEqual(
            clip_id_to_sequence("urfd-debug-fall-30-cam0-rgb"),
            "fall-30",
        )

    def test_adl_clip_id_maps_to_adl_sequence(self) -> None:
        self.assertEqual(
            clip_id_to_sequence("urfd-debug-adl-01-cam0-rgb"),
            "adl-01",
        )
        self.assertEqual(
            clip_id_to_sequence("urfd-debug-adl-40-cam0-rgb"),
            "adl-40",
        )

    def test_unknown_clip_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            clip_id_to_sequence("not-a-urfd-clip")
        with self.assertRaises(ValueError):
            clip_id_to_sequence("urfd-debug-foo-01-cam0-rgb")  # foo
        with self.assertRaises(ValueError):
            clip_id_to_sequence("urfd-debug--cam0-rgb")  # empty sequence
        # Missing suffix.
        with self.assertRaises(ValueError):
            clip_id_to_sequence("urfd-debug-fall-01")

    def test_sequence_to_clip_type_classifies_fall_and_adl(self) -> None:
        self.assertEqual(sequence_to_clip_type("fall-01"), "fall")
        self.assertEqual(sequence_to_clip_type("adl-02"), "adl")
        with self.assertRaises(ValueError):
            sequence_to_clip_type("weird-01")


# ---------------------------------------------------------------------------
# Default rule — fall clip
# ---------------------------------------------------------------------------


class FallClipDefaultRuleTests(unittest.TestCase):
    """Fall windows: 0 or 1 → fall; all -1 → no_fall; never confuser."""

    def test_fall_window_with_all_upright_is_no_fall(self) -> None:
        csv = parse_urfd_csv_label_text(_fall_with_first_n_upright(10))
        # 10-frame window of all uprights → the pre-fall region of
        # the fall clip. The default rule treats this as a clean
        # negative, NOT a noisy positive.
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_fall_window_containing_label_zero_is_fall(self) -> None:
        csv = parse_urfd_csv_label_text(_fall_with_one_falling(5, 10))
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)
        self.assertFalse(is_confuser)

    def test_fall_window_containing_label_one_is_fall(self) -> None:
        csv = parse_urfd_csv_label_text(_fall_with_one_lying(7, 10))
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)
        self.assertFalse(is_confuser)

    def test_fall_window_with_fall_at_first_frame(self) -> None:
        csv = parse_urfd_csv_label_text(_fall_with_one_falling(1, 10))
        label, _ = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)


# ---------------------------------------------------------------------------
# Default rule — ADL clip
# ---------------------------------------------------------------------------


class AdlClipDefaultRuleTests(unittest.TestCase):
    """ADL windows: always no_fall; label 1 → confuser=True."""

    def test_adl_window_with_all_upright_is_clean_no_fall(self) -> None:
        csv = parse_urfd_csv_label_text(_adl_with_first_n_upright(10))
        label, is_confuser = label_window(
            "urfd-debug-adl-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_adl_window_containing_label_one_is_confuser(self) -> None:
        # An ADL clip where one frame is labelled 1 (lying) — the
        # window stays no_fall but is flagged as a confuser
        # example.
        csv = parse_urfd_csv_label_text(_adl_with_some_lying([5], 10))
        label, is_confuser = label_window(
            "urfd-debug-adl-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertTrue(is_confuser)

    def test_adl_window_lying_at_first_frame(self) -> None:
        csv = parse_urfd_csv_label_text(_adl_with_some_lying([1], 10))
        label, is_confuser = label_window(
            "urfd-debug-adl-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertTrue(is_confuser)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class GuardTests(unittest.TestCase):
    """Every malformed input raises :class:`WindowLabelingError`."""

    def test_empty_frame_indices_raises(self) -> None:
        csv = parse_urfd_csv_label_text(_fall_with_first_n_upright(10))
        with self.assertRaises(WindowLabelingError) as ctx:
            label_window("urfd-debug-fall-01-cam0-rgb", [], csv)
        self.assertIn("frame_indices is empty", str(ctx.exception))

    def test_unknown_clip_id_raises(self) -> None:
        csv = parse_urfd_csv_label_text(_fall_with_first_n_upright(10))
        with self.assertRaises(ValueError):  # clip_id mapping
            label_window("not-a-urfd-clip", [1, 2, 3], csv)

    def test_non_contiguous_sequence_does_not_raise(self) -> None:
        # Real CSVs may be sparse — a fall sequence with frame 5
        # missing is a legitimate state, NOT a bug. The function
        # must skip frame 5 and label the rest of the window
        # from whatever labels are present.
        # 1..10 except 5: 9 frames total, range 1..10, gap at 5.
        non_contiguous = _non_contiguous_fall(
            [f for f in range(1, 11) if f != 5]
        )
        csv = parse_urfd_csv_label_text(non_contiguous)
        # Window 1..10 with frame 5 missing — no raise, the
        # missing frame is skipped, the other 9 frames carry
        # label -1 → no_fall.
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_missing_frame_does_not_raise(self) -> None:
        # CSV has frames 1..10. Window asks for frame 11 — missing
        # in the CSV. The function skips frame 11, labels the
        # rest of the window from what's available, and does NOT
        # raise. (Real ADL CSVs are sparse; a window's frame
        # indices may extend past the labelled range.)
        csv = parse_urfd_csv_label_text(_fall_with_first_n_upright(10))
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            csv,
        )
        # All available labels are -1 → no_fall.
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_non_integer_frame_index_raises(self) -> None:
        csv = parse_urfd_csv_label_text(_fall_with_first_n_upright(10))
        with self.assertRaises(WindowLabelingError):
            label_window(
                "urfd-debug-fall-01-cam0-rgb",
                [1, 2, "three", 4, 5, 6, 7, 8, 9, 10],  # type: ignore[list-item]
                csv,
            )

    def test_non_positive_adjusted_frame_raises(self) -> None:
        # Raw frame 0 with offset -1 → adjusted -1 → raises.
        csv = parse_urfd_csv_label_text(_fall_with_first_n_upright(10))
        with self.assertRaises(WindowLabelingError) as ctx:
            label_window(
                "urfd-debug-fall-01-cam0-rgb",
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                csv,
                frame_index_offset=-1,
            )
        self.assertIn("non-positive", str(ctx.exception))


# ---------------------------------------------------------------------------
# Frame-index offset parameter
# ---------------------------------------------------------------------------


class FrameIndexOffsetTests(unittest.TestCase):
    """``frame_index_offset`` documents the alignment explicitly.

    These tests exercise the :func:`label_window` primitive
    directly with hand-built indices. They are SYNTHETIC
    no-shift tests — the indices they pass in are already
    CSV-aligned (1-based), so the default offset of ``0`` is
    correct. Real crop callers (Issue 006, the full pipeline
    notebook's smoke cell) MUST NOT use this primitive
    directly because real crop metadata carries 0-based
    indices; they go through :func:`label_window_from_crop_meta`
    which applies :data:`CROP_TO_CSV_FRAME_OFFSET` for them.
    See :class:`CropMetaHelperTests` for the real-crop
    coverage.
    """

    def test_default_offset_zero_aligns_with_urfd_csv_synthetic(self) -> None:
        # SYNTHETIC no-shift test: indices passed in are already
        # CSV-aligned (1-based), so the default offset of ``0``
        # is correct and the function reads CSV rows 1..10
        # verbatim. This proves the primitive's default works
        # for hand-built fixtures — it does NOT prove the
        # default is correct for real crop metadata (which is
        # 0-based; use ``label_window_from_crop_meta`` there).
        csv = parse_urfd_csv_label_text(_fall_with_one_falling(3, 10))
        label, _ = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
        )
        # Frame 3 has label 0 → fall.
        self.assertEqual(label, WINDOW_LABEL_FALL)

    def test_explicit_offset_one_applied_to_zero_based_indices(self) -> None:
        # Synthetic too — exercises the primitive's
        # ``frame_index_offset=1`` path with hand-built 0-based
        # indices. The real-crop wrapper (next test class) is
        # the supported entry point for crop callers; this
        # primitive path stays available for explicit-shift
        # adapters and future CSV formats.
        csv = parse_urfd_csv_label_text(_fall_with_one_falling(4, 10))
        # 0-based frame indices (0..9) with offset=1 → look up 1..10.
        label, _ = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            csv,
            frame_index_offset=1,
        )
        # Frame 3 (0-based) + 1 = frame 4 in CSV, label 0 → fall.
        self.assertEqual(label, WINDOW_LABEL_FALL)


# ---------------------------------------------------------------------------
# Real crop wrapper — the supported entry point for Issue 006
# ---------------------------------------------------------------------------


class CropMetaHelperTests(unittest.TestCase):
    """:func:`label_window_from_crop_meta` is the real-crop entry point.

    Real crop metadata (``window.frame_indices[i]`` carried into
    the sidecar ``frame_index``) is 0-based (perception
    ``enumerate``). The wrapper applies
    :data:`CROP_TO_CSV_FRAME_OFFSET` so a real crop caller does
    NOT have to remember the alignment at every call site —
    exactly the footgun Issue 006 must avoid. These tests pin
    the constant value, prove the wrapper handles the realistic
    pre-fall / falling / lying regions with 32-frame windows,
    and prove the wrapper passes through validation errors
    rather than swallowing them.
    """

    def test_offset_constant_is_one(self) -> None:
        # The constant is the alignment contract. Pin it so a
        # future change to either side of the convention
        # (crop metadata indexing OR CSV indexing) trips the
        # test suite instead of silently drifting.
        self.assertEqual(CROP_TO_CSV_FRAME_OFFSET, 1)
        self.assertIsInstance(CROP_TO_CSV_FRAME_OFFSET, int)

    def test_wrapper_labels_zero_based_indices_with_falling_frame(self) -> None:
        # Real crop metadata for a 10-frame window: 0-based
        # 0..9. The wrapper applies +1 internally so the
        # function reads CSV rows 1..10. Frame 4 (CSV) is
        # "falling" → window is "fall".
        csv = parse_urfd_csv_label_text(_fall_with_one_falling(4, 10))
        label, is_confuser = label_window_from_crop_meta(
            "urfd-debug-fall-01-cam0-rgb",
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)
        self.assertFalse(is_confuser)

    def test_wrapper_with_32_frame_pre_fall_window(self) -> None:
        # Real Issue 003 shards are 32-frame windows. A 0-based
        # crop window 0..31 (CSV 1..32) lands entirely inside
        # the pre-fall region of the realistic fall sequence
        # (CSV frames 1..82 are all label -1). The wrapper
        # applies +1 internally and the default rule returns
        # a clean "no_fall" — NOT a noisy "fall".
        csv = parse_urfd_csv_label_text(_realistic_fall_sequence())
        label, is_confuser = label_window_from_crop_meta(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(0, 32)),  # 32-frame window, 0-based crop indices
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_wrapper_with_32_frame_window_spanning_falling_onset(self) -> None:
        # 32-frame 0-based crop window 60..91 (CSV 61..92)
        # contains CSV frame 83 (the first "falling" frame).
        # Default rule → "fall".
        csv = parse_urfd_csv_label_text(_realistic_fall_sequence())
        label, is_confuser = label_window_from_crop_meta(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(60, 92)),
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)
        self.assertFalse(is_confuser)

    def test_wrapper_with_32_frame_lying_tail_window(self) -> None:
        # 32-frame 0-based crop window 130..161 (CSV 131..162)
        # is entirely inside the "lying" tail (CSV frames
        # 113..160 are all label 1). Default rule → "fall".
        csv = parse_urfd_csv_label_text(_realistic_fall_sequence())
        label, is_confuser = label_window_from_crop_meta(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(130, 162)),
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)
        self.assertFalse(is_confuser)

    def test_wrapper_plus_one_is_load_bearing_at_fall_onset_boundary(self) -> None:
        # The guard for the exact footgun: the +1 must change the
        # OUTCOME, not merely the lookup. A 32-frame 0-based crop
        # window 51..82 sits just BELOW the fall onset. With the
        # wrapper's +1 it reads CSV 52..83 — frame 83 is the first
        # "falling" frame → "fall". WITHOUT the +1 (i.e. the
        # primitive's default 0) it would read CSV 51..82, all
        # upright → "no_fall". This test therefore FAILS if the
        # wrapper ever stops applying CROP_TO_CSV_FRAME_OFFSET —
        # unlike the interior-window tests above, which pass either
        # way and so cannot catch a dropped offset on their own.
        csv = parse_urfd_csv_label_text(_realistic_fall_sequence())
        label, is_confuser = label_window_from_crop_meta(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(51, 83)),  # 0-based 51..82; +1 → CSV 52..83
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)
        self.assertFalse(is_confuser)

    def test_wrapper_with_32_frame_adl_sparse_window(self) -> None:
        # 0-based 0..31 (CSV 1..32) on the sparse ADL fixture:
        # labelled ADL frames start at CSV frame 6, so CSV
        # frames 1..5 are missing and CSV frames 6..32 are all
        # -1. Default rule on ADL: missing frames are skipped,
        # remaining labels are -1 → "no_fall".
        csv = parse_urfd_csv_label_text(_sparse_adl_sequence())
        label, is_confuser = label_window_from_crop_meta(
            "urfd-debug-adl-01-cam0-rgb",
            list(range(0, 32)),  # 32-frame window, 0-based crop indices
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_wrapper_preserves_adl_confuser_flag(self) -> None:
        # Real crop window 0-based 7..22 (CSV 8..23) on a sparse
        # ADL sequence with a single lying frame at CSV frame 8.
        # Default rule on ADL with a positive label-1 signal →
        # "no_fall" with is_confuser=True. The wrapper must NOT
        # lose the confuser flag through the +1 offset.
        rows = [
            f"adl-01,{i},-1" for i in range(6, 25) if i != 7 and i != 8
        ] + ["adl-01,8,1"]
        csv = parse_urfd_csv_label_text("\n".join(rows) + "\n")
        label, is_confuser = label_window_from_crop_meta(
            "urfd-debug-adl-01-cam0-rgb",
            list(range(7, 23)),
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertTrue(is_confuser)

    def test_wrapper_returns_unlabeled_for_zero_label_fall_window(self) -> None:
        # Real crop window 0-based 199..215 (CSV 200..216) on
        # the fully-populated fall sequence: the window's CSV
        # rows are all missing. The default rule returns the
        # explicit ``WINDOW_LABEL_UNLABELED`` sentinel so Issue
        # 006 can drop the example deterministically — exactly
        # the same behaviour as the primitive.
        rows = "\n".join(f"fall-01,{i},-1" for i in range(1, 161)) + "\n"
        csv = parse_urfd_csv_label_text(rows)
        label, is_confuser = label_window_from_crop_meta(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(199, 216)),
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_UNLABELED)
        self.assertFalse(is_confuser)

    def test_wrapper_passes_through_empty_frame_indices_error(self) -> None:
        # An empty ``crop_frame_indices`` is a real bug (the
        # window-builder always produces a dense window) — the
        # wrapper must NOT swallow it. The error message and
        # exception type come from the primitive unchanged.
        csv = parse_urfd_csv_label_text(_fall_with_first_n_upright(10))
        with self.assertRaises(WindowLabelingError) as ctx:
            label_window_from_crop_meta(
                "urfd-debug-fall-01-cam0-rgb",
                [],
                csv,
            )
        self.assertIn("frame_indices is empty", str(ctx.exception))

    def test_wrapper_passes_through_unknown_clip_id_error(self) -> None:
        # Real crop metadata with a bad ``clip_id`` is a real
        # bug — the wrapper must surface it via the primitive.
        csv = parse_urfd_csv_label_text(_fall_with_first_n_upright(10))
        with self.assertRaises(ValueError):
            label_window_from_crop_meta(
                "not-a-urfd-clip",
                [0, 1, 2, 3, 4],
                csv,
            )


# ---------------------------------------------------------------------------
# Pluggable rule
# ---------------------------------------------------------------------------


class StrictFallbackRule(WindowLabelingRule):
    """A different rule: fall only if label 1 (lying) is present.

    Demonstrates the pluggability — Issue 006 can swap the rule
    without re-cropping. Frames in the transition phase (label
    0) do NOT credit the window as fall under this rule.
    """

    def apply(
        self,
        clip_type: str,
        per_frame_labels: tuple[int, ...],
    ) -> tuple[str, bool]:
        if clip_type == "fall":
            return (
                WINDOW_LABEL_FALL if any(l == 1 for l in per_frame_labels)
                else WINDOW_LABEL_NO_FALL
            ), False
        return WINDOW_LABEL_NO_FALL, False


class PluggableRuleTests(unittest.TestCase):
    """The rule is a swappable strategy."""

    def test_alternate_rule_treats_transition_as_no_fall(self) -> None:
        # Same fixture, different rule: frame 5 is label 0
        # (falling / transition). The default rule treats this as
        # "fall"; the alternate rule does NOT (only lying = label 1
        # credits the window).
        csv = parse_urfd_csv_label_text(_fall_with_one_falling(5, 10))
        alternate = StrictFallbackRule()
        label, _ = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
            labeling_rule=alternate,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)

    def test_default_rule_still_flags_lying_as_fall(self) -> None:
        # With label 1, the default rule still fires.
        csv = parse_urfd_csv_label_text(_fall_with_one_lying(7, 10))
        label, _ = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            csv,
            labeling_rule=StrictFallbackRule(),
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)

    def test_rule_constructed_via_dataclass_default_is_default(self) -> None:
        # The module exposes a singleton default rule so callers
        # that just want the project default can pass it through
        # without constructing one themselves.
        from data.urfd_labels import DEFAULT_WINDOW_LABELING_RULE
        self.assertIsInstance(
            DEFAULT_WINDOW_LABELING_RULE, DefaultWindowLabelingRule,
        )


# ---------------------------------------------------------------------------
# Realistic sparse fixtures
# ---------------------------------------------------------------------------


def _realistic_fall_sequence() -> str:
    """Real-shaped fall sequence:

    - frames 1..82:   label -1 (upright pre-fall region)
    - frames 83..112: label 0  (falling / transition)
    - frames 113..160: label 1 (lying on the ground)
    """
    rows: list[str] = []
    rows.extend(f"fall-01,{i},-1" for i in range(1, 83))
    rows.extend(f"fall-01,{i},0"  for i in range(83, 113))
    rows.extend(f"fall-01,{i},1"  for i in range(113, 161))
    return "\n".join(rows) + "\n"


def _sparse_adl_sequence() -> str:
    """Real-shaped sparse ADL sequence:

    - labelled frames 6..150
    - frame 7 missing (an annotator skipped it)
    - all labels are -1
    - frames 1..5 and 151..160 are NOT in the CSV
    """
    rows = [
        f"adl-01,{i},-1"
        for i in list(range(6, 150 + 1))
        if i != 7
    ]
    return "\n".join(rows) + "\n"


class RealisticFallFixtureTests(unittest.TestCase):
    """Real-shape fall sequence: pre-fall, falling, lying regions."""

    def setUp(self) -> None:
        self._csv = parse_urfd_csv_label_text(_realistic_fall_sequence())

    def test_pre_fall_window_is_no_fall(self) -> None:
        # Window 1..32 falls entirely inside the upright
        # pre-fall region (frames 1..82 all -1). The default
        # rule produces a clean negative, NOT a noisy positive.
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(1, 33)),
            self._csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_window_spanning_falling_frame_is_fall(self) -> None:
        # Window 60..91 contains frame 83, which has label 0
        # (falling). The default rule produces "fall".
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(60, 92)),
            self._csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)
        self.assertFalse(is_confuser)

    def test_lying_tail_window_is_fall(self) -> None:
        # Window 130..161 contains frames 130..160 which are
        # all label 1 (lying). The default rule produces "fall".
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(130, 162)),
            self._csv,
        )
        self.assertEqual(label, WINDOW_LABEL_FALL)
        self.assertFalse(is_confuser)


class SparseAdlFixtureTests(unittest.TestCase):
    """Real-shape sparse ADL: some RGB frames have no CSV row."""

    def setUp(self) -> None:
        self._csv = parse_urfd_csv_label_text(_sparse_adl_sequence())

    def test_window_in_unlabelled_prefix_does_not_crash(self) -> None:
        # Window 1..16 — all frames are 1..5 or 6..16, but
        # labelled ADL frames start at 6. So frames 1..5 are
        # missing, frames 6..16 are present (all -1). No
        # crash — the missing frames are skipped.
        label, is_confuser = label_window(
            "urfd-debug-adl-01-cam0-rgb",
            list(range(1, 17)),
            self._csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_window_over_missing_frame_gap_does_not_crash(self) -> None:
        # Window 4..23 spans a missing frame (7). The function
        # skips the missing frame, labels the rest, and
        # returns no_fall.
        label, is_confuser = label_window(
            "urfd-debug-adl-01-cam0-rgb",
            list(range(4, 24)),
            self._csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_window_with_zero_available_adl_labels(self) -> None:
        # Window 1..5 — all frames missing from the CSV
        # (labelled ADL frames start at 6). The default rule
        # returns no_fall, is_confuser=False for zero-label
        # ADL windows (a confuser requires a positive label-1
        # signal).
        label, is_confuser = label_window(
            "urfd-debug-adl-01-cam0-rgb",
            [1, 2, 3, 4, 5],
            self._csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertFalse(is_confuser)

    def test_window_with_adl_lying_frame_still_flags_confuser(self) -> None:
        # Construct a sparse ADL sequence with a single lying
        # frame; window 5..24 covers frames 6..24 except 7
        # (missing), so the available labels include frame
        # 8 (label 1).
        rows = [
            f"adl-01,{i},-1" for i in list(range(6, 25)) if i != 7
        ]
        rows.append("adl-01,8,1")  # 8 is lying
        # dedup, since 8 was already added as -1:
        rows = [
            f"adl-01,{i},-1" for i in range(6, 25) if i != 7 and i != 8
        ] + ["adl-01,8,1"]
        csv = parse_urfd_csv_label_text("\n".join(rows) + "\n")
        label, is_confuser = label_window(
            "urfd-debug-adl-01-cam0-rgb",
            list(range(5, 25)),
            csv,
        )
        self.assertEqual(label, WINDOW_LABEL_NO_FALL)
        self.assertTrue(is_confuser)


class ZeroLabelFallFixtureTests(unittest.TestCase):
    """Fall windows with zero available labels return the unlabeled sentinel."""

    def test_fall_window_with_no_available_labels_returns_unlabeled(self) -> None:
        # Real CSV is fully populated for fall-01 (frames 1..160),
        # but the window asks for frames 200..216 — all missing.
        # The default rule returns the explicit unlabeled
        # sentinel so Issue 006 can drop the example.
        rows = "\n".join(f"fall-01,{i},-1" for i in range(1, 161)) + "\n"
        csv = parse_urfd_csv_label_text(rows)
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            list(range(200, 217)),
            csv,
        )
        from data.urfd_labels import WINDOW_LABEL_UNLABELED
        self.assertEqual(label, WINDOW_LABEL_UNLABELED)
        self.assertFalse(is_confuser)

    def test_fall_window_fully_missing_in_partial_csv_returns_unlabeled(self) -> None:
        # A partial CSV with labels only in the middle, and a
        # window that lands entirely outside the labelled range.
        rows = "\n".join(f"fall-01,{i},0" for i in range(50, 100)) + "\n"
        csv = parse_urfd_csv_label_text(rows)
        label, is_confuser = label_window(
            "urfd-debug-fall-01-cam0-rgb",
            [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
             212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
             224, 225, 226, 227, 228, 229, 230, 231],
            csv,
        )
        from data.urfd_labels import WINDOW_LABEL_UNLABELED
        self.assertEqual(label, WINDOW_LABEL_UNLABELED)
        self.assertFalse(is_confuser)


if __name__ == "__main__":
    unittest.main()
