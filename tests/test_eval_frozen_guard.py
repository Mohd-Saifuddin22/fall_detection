"""Tests for the frozen-tier read guard (:mod:`evaluation.execution_context`).

The guard is the project's defense against training/tuning code
silently reading the frozen_unseen_test role. These tests prove the
guard fails closed on every path that could leak frozen data:

- TRAINING and TUNING cannot receive frozen records.
- EVALUATION and FINAL_JUDGEMENT can.
- UNKNOWN / None / typo'd strings cannot receive frozen records.
- :func:`get_frozen_clips` raises FrozenAccessError on denial.
- :func:`enforce_no_frozen_in_iterable` catches leakage even when a
  caller hand-built a clip list and skipped the context-aware path.
- The guard coexists with the existing manifest validator — adding a
  well-formed vault manifest to ``validate_manifest`` does not
  invalidate it (vault isolation is enforced at the validator; this
  guard is the second layer against context misuse).
"""

from __future__ import annotations

import unittest

from data.manifests import ClipRole, Manifest, validate_manifest

from evaluation.execution_context import (
    ExecutionContext,
    FROZEN_ALLOWED_CONTEXTS,
    FrozenAccessError,
    coerce_execution_context,
    enforce_no_frozen_in_iterable,
    frozen_clips_present,
    get_frozen_clips,
    is_frozen_allowed,
    select_clips_for_context,
)

# Reuse the test fixtures: in-memory clip builders + a mixed-role manifest.
from tests.eval_fixtures import make_clip, make_mixed_manifest


def _allowed_contexts_substring(message: str) -> str:
    """Extract the ``['ctx1', 'ctx2', ...]`` portion of a FrozenAccessError.

    The denial error embeds the allow-list between ``Only`` and
    ``may.``; this helper isolates that slice so a test can assert
    on its contents without coupling to the rest of the message.
    """
    marker_start = message.find("Only ")
    if marker_start < 0:
        return ""
    slice_from = message.find("[", marker_start)
    slice_to = message.find("]", slice_from if slice_from >= 0 else marker_start)
    if slice_from < 0 or slice_to < 0:
        return ""
    return message[slice_from:slice_to + 1]





class CoercionTests(unittest.TestCase):
    """``coerce_execution_context`` never silently promotes to a privileged value."""

    def test_enum_passes_through(self) -> None:
        self.assertEqual(
            coerce_execution_context(ExecutionContext.TRAINING),
            ExecutionContext.TRAINING,
        )

    def test_string_value_is_accepted(self) -> None:
        self.assertEqual(coerce_execution_context("training"), ExecutionContext.TRAINING)
        self.assertEqual(
            coerce_execution_context("FINAL_JUDGEMENT"),
            ExecutionContext.FINAL_JUDGEMENT,
        )

    def test_none_becomes_unknown(self) -> None:
        # Critical — a default-None kwarg must NOT silently grant
        # access. Coercion returns UNKNOWN so the guard denies.
        self.assertEqual(coerce_execution_context(None), ExecutionContext.UNKNOWN)

    def test_typo_string_becomes_unknown(self) -> None:
        # "Traning" is a typo; should fail closed, not be coerced.
        self.assertEqual(coerce_execution_context("Traning"), ExecutionContext.UNKNOWN)
        self.assertEqual(coerce_execution_context("final"), ExecutionContext.UNKNOWN)

    def test_arbitrary_object_becomes_unknown(self) -> None:
        # Int, dict, list etc. must all map to UNKNOWN, not crash.
        self.assertEqual(coerce_execution_context(42), ExecutionContext.UNKNOWN)
        self.assertEqual(coerce_execution_context({"x": 1}), ExecutionContext.UNKNOWN)
        self.assertEqual(coerce_execution_context([1, 2]), ExecutionContext.UNKNOWN)


class AllowListTests(unittest.TestCase):
    """The frozen allow-list is exactly FINAL_JUDGEMENT — single entry by governance."""

    def test_only_final_judgement_can_see_frozen(self) -> None:
        # Governance rule: the vault opens ONLY for FINAL_JUDGEMENT.
        # EVALUATION, despite its name, is NOT allowed to read frozen
        # — operators wanting to spot-check frozen behaviour must
        # declare FINAL_JUDGEMENT (or accept the denial error).
        self.assertEqual(
            FROZEN_ALLOWED_CONTEXTS,
            frozenset({ExecutionContext.FINAL_JUDGEMENT}),
        )

    def test_is_frozen_allowed_matches_the_allow_list(self) -> None:
        for ctx, expected in (
            (ExecutionContext.TRAINING, False),
            (ExecutionContext.TUNING, False),
            (ExecutionContext.EVALUATION, False),  # governance: denied
            (ExecutionContext.FINAL_JUDGEMENT, True),
            (ExecutionContext.UNKNOWN, False),
        ):
            with self.subTest(context=ctx):
                self.assertEqual(is_frozen_allowed(ctx), expected)

    def test_evaluation_is_not_in_the_allow_list(self) -> None:
        # Belt-and-braces: explicit assertion independent of any
        # iteration order. If a future governance change re-adds
        # EVALUATION, this test will fail loudly so the review
        # surfaces in test output rather than in production.
        self.assertNotIn(ExecutionContext.EVALUATION, FROZEN_ALLOWED_CONTEXTS)


class SelectClipsForContextTests(unittest.TestCase):
    """End-to-end proof: training/tuning/unknown never see frozen clips."""

    def _frozen_ids(self, clips) -> set[str]:
        return {c.clip_id for c in clips if c.role is ClipRole.FROZEN_UNSEEN_TEST}

    def test_training_context_excludes_frozen_clips(self) -> None:
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, ExecutionContext.TRAINING)
        self.assertEqual(self._frozen_ids(chosen), set())

    def test_training_context_string_form_also_excludes_frozen(self) -> None:
        # The guard must work even when the caller passes a string
        # (which is what notebooks tend to build from env vars).
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, "training")
        self.assertEqual(self._frozen_ids(chosen), set())

    def test_tuning_context_excludes_frozen_clips(self) -> None:
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, ExecutionContext.TUNING)
        self.assertEqual(self._frozen_ids(chosen), set())

    def test_evaluation_context_excludes_frozen_clips(self) -> None:
        # Governance change: EVALUATION no longer reads frozen. It
        # operates only on debug / train / validate. Previously
        # this case asserted the inverse; the policy is now fail
        # closed for EVALUATION too.
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, ExecutionContext.EVALUATION)
        self.assertEqual(self._frozen_ids(chosen), set(),
                         msg="EVALUATION must not see frozen_unseen_test records.")

    def test_evaluation_string_form_also_excludes_frozen(self) -> None:
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, "evaluation")
        self.assertEqual(self._frozen_ids(chosen), set())

    def test_final_judgement_context_includes_frozen_clips(self) -> None:
        # FINAL_JUDGEMENT remains the one context that can read
        # frozen — this case must stay positive through the
        # governance change.
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, ExecutionContext.FINAL_JUDGEMENT)
        self.assertEqual(
            self._frozen_ids(chosen),
            {c.clip_id for c in manifest.clips
             if c.role.value == "frozen_unseen_test"},
        )

    def test_unknown_context_denies_frozen_access(self) -> None:
        # Explicit UNKNOWN — same denied-path as training.
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, ExecutionContext.UNKNOWN)
        self.assertEqual(self._frozen_ids(chosen), set())

    def test_none_context_denies_frozen_access(self) -> None:
        # Default-None argument is the realistic "I forgot to specify"
        # case. Must NOT silently get a privileged context.
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, None)
        self.assertEqual(self._frozen_ids(chosen), set())

    def test_unknown_string_context_denies_frozen_access(self) -> None:
        # A typo ("traning") or an unrecognised flag must NOT bubble
        # up to a privileged context.
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, "traning")
        self.assertEqual(self._frozen_ids(chosen), set())

    def test_training_still_returns_non_frozen_clips(self) -> None:
        # The guard removes frozen but does not otherwise zero the
        # manifest — debug / train / validate must still come through.
        manifest = make_mixed_manifest()
        chosen = select_clips_for_context(manifest, ExecutionContext.TRAINING)
        role_counts: dict[str, int] = {}
        for clip in chosen:
            role_counts[clip.role.value] = role_counts.get(clip.role.value, 0) + 1
        self.assertGreater(role_counts.get("train", 0), 0)
        self.assertGreater(role_counts.get("validate", 0), 0)
        self.assertGreater(role_counts.get("debug", 0), 0)
        self.assertNotIn("frozen_unseen_test", role_counts)


class GetFrozenClipsTests(unittest.TestCase):
    """Explicit frozen accessor raises on denial and returns on allow."""

    def test_evaluation_now_raises_frozen_access_error(self) -> None:
        # Governance change: EVALUATION used to be allowed. It is no
        # longer — the vault opens ONLY for FINAL_JUDGEMENT. This
        # case flipped from "can read frozen" to "raises".
        manifest = make_mixed_manifest()
        with self.assertRaises(FrozenAccessError):
            get_frozen_clips(manifest, ExecutionContext.EVALUATION)

    def test_evaluation_string_form_raises_frozen_access_error(self) -> None:
        manifest = make_mixed_manifest()
        with self.assertRaises(FrozenAccessError):
            get_frozen_clips(manifest, "evaluation")

    def test_final_judgement_can_read_frozen(self) -> None:
        manifest = make_mixed_manifest()
        clips = get_frozen_clips(manifest, ExecutionContext.FINAL_JUDGEMENT)
        self.assertTrue(all(c.role.value == "frozen_unseen_test" for c in clips))
        self.assertGreater(len(clips), 0)

    def test_final_judgement_string_form_can_read_frozen(self) -> None:
        manifest = make_mixed_manifest()
        clips = get_frozen_clips(manifest, "final_judgement")
        self.assertTrue(all(c.role.value == "frozen_unseen_test" for c in clips))

    def test_training_raises_frozen_access_error(self) -> None:
        manifest = make_mixed_manifest()
        with self.assertRaises(FrozenAccessError):
            get_frozen_clips(manifest, ExecutionContext.TRAINING)

    def test_tuning_raises_frozen_access_error(self) -> None:
        manifest = make_mixed_manifest()
        with self.assertRaises(FrozenAccessError):
            get_frozen_clips(manifest, ExecutionContext.TUNING)

    def test_unknown_raises_frozen_access_error(self) -> None:
        manifest = make_mixed_manifest()
        with self.assertRaises(FrozenAccessError):
            get_frozen_clips(manifest, ExecutionContext.UNKNOWN)

    def test_none_raises_frozen_access_error(self) -> None:
        manifest = make_mixed_manifest()
        with self.assertRaises(FrozenAccessError):
            get_frozen_clips(manifest, None)

    def test_error_message_lists_only_final_judgement_as_allowed(self) -> None:
        # After the governance change, the only allowed context is
        # FINAL_JUDGEMENT. EVALUATION / training / tuning strings
        # must NOT appear in the allow-list portion of the message.
        manifest = make_mixed_manifest()
        with self.assertRaises(FrozenAccessError) as ctx:
            get_frozen_clips(manifest, ExecutionContext.TRAINING)
        message = str(ctx.exception)
        # Allow-list slice: must contain ``final_judgement`` exactly.
        allowed = _allowed_contexts_substring(message)
        self.assertEqual(allowed, "['final_judgement']")

    def test_empty_manifest_returns_empty_list_for_allowed_context(self) -> None:
        # Even an empty manifest must accept the allowed context
        # without raising — it just has no frozen clips to return.
        empty = Manifest(schema_version="1.1")
        self.assertEqual(
            get_frozen_clips(empty, ExecutionContext.FINAL_JUDGEMENT),
            [],
        )


class EnforceNoFrozenIterableTests(unittest.TestCase):
    """Boundary check on hand-built clip lists."""

    def test_passes_through_when_no_frozen_present(self) -> None:
        non_frozen = [
            make_clip("a", role=ClipRole.TRAIN),
            make_clip("b", role=ClipRole.DEBUG),
        ]
        # Training is a non-allowed context but no frozen present, so
        # the check passes vacuously.
        enforce_no_frozen_in_iterable(non_frozen, ExecutionContext.TRAINING)

    def test_raises_when_frozen_slip_through(self) -> None:
        leaked = [
            make_clip("train-1", role=ClipRole.TRAIN),
            make_clip("vault-1", role=ClipRole.FROZEN_UNSEEN_TEST),
        ]
        with self.assertRaises(FrozenAccessError):
            enforce_no_frozen_in_iterable(leaked, ExecutionContext.TRAINING)

    def test_allows_frozen_for_privileged_context(self) -> None:
        clips = [make_clip("vault-1", role=ClipRole.FROZEN_UNSEEN_TEST)]
        # No raise — final_judgement is allowed.
        enforce_no_frozen_in_iterable(clips, ExecutionContext.FINAL_JUDGEMENT)

    def test_error_message_identifies_leaked_clip(self) -> None:
        leaked = [make_clip("very-specific-id", role=ClipRole.FROZEN_UNSEEN_TEST)]
        with self.assertRaises(FrozenAccessError) as ctx:
            enforce_no_frozen_in_iterable(leaked, ExecutionContext.TUNING)
        self.assertIn("very-specific-id", str(ctx.exception))


class FrozenClipsPresentTests(unittest.TestCase):
    """Diagnostics helper for callers that want to assert vault presence first."""

    def test_returns_true_when_manifest_carries_frozen_clips(self) -> None:
        self.assertTrue(frozen_clips_present(make_mixed_manifest()))

    def test_returns_false_when_no_frozen_clips(self) -> None:
        manifest = Manifest(
            schema_version="1.1",
            clips=[make_clip("t-1", role=ClipRole.TRAIN)],
        )
        self.assertFalse(frozen_clips_present(manifest))


class CoexistenceWithValidatorTests(unittest.TestCase):
    """The guard is a second layer on top of ``validate_manifest``."""

    def test_validator_still_accepts_well_formed_vault_manifest(self) -> None:
        # Vault isolation is enforced at the validator layer; this
        # test guards against regressing that behaviour by adding
        # Step 1 modules.
        manifest = make_mixed_manifest()
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"validator errors:\n{report.errors}")

    def test_guard_does_not_silently_unfreeze_a_validator_violation(self) -> None:
        # Build a manifest where a vault dataset leaks into train —
        # the validator catches it AND the guard still removes frozen
        # rows from the training view (so a buggy manifest cannot
        # accidentally train on vault data via the guard).
        bad_manifest = Manifest(
            schema_version="1.1",
            clips=[
                make_clip("train-1"),
                make_clip("vault-1", dataset="omnifall", role=ClipRole.TRAIN),
            ],
        )
        report = validate_manifest(bad_manifest)
        self.assertFalse(report.is_valid)
        chosen = select_clips_for_context(bad_manifest, ExecutionContext.TRAINING)
        self.assertEqual(
            [c.clip_id for c in chosen if c.role is ClipRole.FROZEN_UNSEEN_TEST],
            [],
            msg="guard must not return frozen rows even when the manifest is malformed",
        )


if __name__ == "__main__":
    unittest.main()
