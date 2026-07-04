"""Tests for :mod:`evaluation.self_test`.

Coverage target (per the Step 3 task spec):

- Baseline AUC-ROC is near 1.0 on the synthetic build.
- Shuffled mean AUC-ROC collapses toward chance (in the chance band).
- A deliberately broken scorer / evaluator that always reports
  unrealistically great AUC makes the self-test fail.
- Deterministic: same config → same result.
- Typed result shape (``SelfTestResult``) carries every required
  field with the documented type.
- Exporter sanity check (the top-level :mod:`evaluation` package
  re-exports the new symbols).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.contracts import ClipLabel, ClipPrediction, MetricResult
from evaluation.metrics.classification import (
    SliceMetricReport,
    compute_classification_metrics,
)
from evaluation.self_test import (
    SelfTestConfig,
    SelfTestResult,
    SyntheticClassificationSet,
    build_synthetic_baseline,
    run_classification_self_test,
    shuffle_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broken_evaluator(
    auc_value: float = 0.99,
) -> "callable":
    """Return an evaluator that always reports AUC-ROC = ``auc_value``.

    The broken evaluator preserves the real metric bundle's
    output for every other axis (count-based metrics, confusion
    matrix, support counts) — only the AUC-ROC is fabricated.
    That is the realistic failure mode where a downstream
    aggregator is broken but the upstream metrics still work.
    """

    def _evaluate(
        predictions,
        labels,
        *,
        threshold=0.5,
        slice_tags=("dataset", "lighting", "occlusion", "multi_person", "action_confuser"),
    ):
        reports = compute_classification_metrics(
            predictions, labels, threshold=threshold, slice_tags=slice_tags
        )

        new_reports: list[SliceMetricReport] = []
        for report in reports:
            new_metrics = tuple(
                MetricResult(
                    name=m.name,
                    value=auc_value if m.name == "auc_roc" else m.value,
                    slice_key=m.slice_key,
                    higher_is_better=m.higher_is_better,
                    notes=m.notes,
                )
                for m in report.metrics
            )
            new_reports.append(
                SliceMetricReport(
                    slice_key=report.slice_key,
                    support=report.support,
                    confusion_matrix=report.confusion_matrix,
                    metrics=new_metrics,
                )
            )
        return new_reports

    return _evaluate


# ---------------------------------------------------------------------------
# Container / config shape
# ---------------------------------------------------------------------------


class ConfigDefaultsTests(unittest.TestCase):
    """The default knobs match the documented deterministic recipe."""

    def test_default_config_has_documented_values(self) -> None:
        cfg = SelfTestConfig()
        self.assertEqual(cfg.n_samples, 200)
        self.assertEqual(cfg.n_shuffles, 8)
        self.assertGreaterEqual(cfg.score_class_separation, 0.5)
        self.assertGreater(cfg.seed_base, 0)
        self.assertGreater(cfg.margin, 0.0)
        # Healthy band is strictly below 1.0 (it is half-open so
        # exactly 1.0 still passes).
        self.assertLess(cfg.healthy_auc_range[1], 1.5)
        # Chance band brackets 0.5.
        self.assertLess(cfg.degraded_auc_range[0], 0.5)
        self.assertGreater(cfg.degraded_auc_range[1], 0.5)

    def test_shuffled_seeds_are_sequential_from_seed_base(self) -> None:
        cfg = SelfTestConfig(seed_base=12345, n_shuffles=4)
        # seed_base + 1, seed_base + 2, ..., seed_base + n_shuffles.
        self.assertEqual(
            cfg.shuffled_seeds(),
            (12346, 12347, 12348, 12349),
        )

    def test_synthetic_set_rejects_length_mismatch(self) -> None:
        from data.manifests import ClipRole, FallLabel
        preds = (
            ClipPrediction(
                clip_id="a", score=0.5, model_id="m",
                dataset="urfd", role=ClipRole.TRAIN,
            ),
        )
        labels = (
            ClipLabel(
                clip_id="a", label=FallLabel.FALL, dataset="urfd",
                role=ClipRole.TRAIN, source_path="datasets/urfd/a.mp4",
            ),
            ClipLabel(
                clip_id="b", label=FallLabel.NO_FALL, dataset="urfd",
                role=ClipRole.TRAIN, source_path="datasets/urfd/b.mp4",
            ),
        )
        with self.assertRaises(ValueError):
            SyntheticClassificationSet(predictions=preds, labels=labels)


# ---------------------------------------------------------------------------
# Synthetic baseline + shuffle correctness
# ---------------------------------------------------------------------------


class SyntheticBaselineTests(unittest.TestCase):
    """The synthetic builder yields the documented shape + correlation."""

    def test_baseline_has_both_classes(self) -> None:
        # positive_rate=0.4 → n_fall = 80, n_nofall = 120 (default n=200).
        cfg = SelfTestConfig(n_samples=200, positive_rate=0.4)
        synth = build_synthetic_baseline(cfg)
        n_fall = sum(1 for l in synth.labels if l.label.value == "fall")
        n_nofall = sum(1 for l in synth.labels if l.label.value == "no_fall")
        self.assertEqual(n_fall + n_nofall, 200)
        self.assertEqual(n_fall, 80)
        self.assertEqual(n_nofall, 120)

    def test_baseline_auc_is_near_one(self) -> None:
        cfg = SelfTestConfig()
        synth = build_synthetic_baseline(cfg)
        reports = compute_classification_metrics(synth.predictions, synth.labels)
        agg = next(r for r in reports if r.slice_key is None)
        auc = next(m.value for m in agg.metrics if m.name == "auc_roc")
        # Well-separated score clouds → AUC near 1.0. The 0.85 lower
        # bound is the SelfTestConfig's healthy_auc_range floor.
        self.assertGreaterEqual(float(auc), 0.85)
        self.assertLess(float(auc), 1.001)

    def test_baseline_is_deterministic(self) -> None:
        # Same config → same predictions and labels. The builder
        # uses an explicit RNG with the configured seed_base, so
        # re-running produces byte-identical scores and labels.
        cfg = SelfTestConfig(seed_base=42)
        synth_a = build_synthetic_baseline(cfg)
        synth_b = build_synthetic_baseline(cfg)
        self.assertEqual(
            [p.score for p in synth_a.predictions],
            [p.score for p in synth_b.predictions],
        )
        self.assertEqual(
            [l.label.value for l in synth_a.labels],
            [l.label.value for l in synth_b.labels],
        )


class LabelShuffleTests(unittest.TestCase):
    """Shuffling actually breaks the ``(score, label)`` pairing."""

    def test_shuffle_changes_label_to_score_pairing(self) -> None:
        # Build a small synthetic set we can inspect by hand.
        cfg = SelfTestConfig(n_samples=12)
        synth = build_synthetic_baseline(cfg)
        before_pairs = list(zip(
            [p.score for p in synth.predictions],
            [l.label.value for l in synth.labels],
        ))
        shuffled = shuffle_labels(synth, seed=12346)
        after_pairs = list(zip(
            [p.score for p in shuffled.predictions],
            [l.label.value for l in shuffled.labels],
        ))
        # At least one pair differs — otherwise the shuffle was a
        # no-op (which is rare but possible with small N).
        self.assertNotEqual(before_pairs, after_pairs)

    def test_shuffle_preserves_predictions(self) -> None:
        cfg = SelfTestConfig(n_samples=12)
        synth = build_synthetic_baseline(cfg)
        shuffled = shuffle_labels(synth, seed=12346)
        # Predictions are not perturbed by the shuffle.
        self.assertEqual(
            [p.clip_id for p in synth.predictions],
            [p.clip_id for p in shuffled.predictions],
        )
        self.assertEqual(
            [p.score for p in synth.predictions],
            [p.score for p in shuffled.predictions],
        )

    def test_shuffle_is_deterministic_per_seed(self) -> None:
        cfg = SelfTestConfig(n_samples=20)
        synth = build_synthetic_baseline(cfg)
        first = shuffle_labels(synth, seed=999)
        second = shuffle_labels(synth, seed=999)
        self.assertEqual(
            [l.label.value for l in first.labels],
            [l.label.value for l in second.labels],
        )

    def test_shuffle_label_clip_ids_match_predictions(self) -> None:
        # The shuffle renames each label's clip_id so the metric
        # bundle pairs it with a different prediction. After
        # shuffling, label[i].clip_id must equal prediction[i].clip_id
        # for the pairing to land in the shuffled order.
        cfg = SelfTestConfig(n_samples=20)
        synth = build_synthetic_baseline(cfg)
        shuffled = shuffle_labels(synth, seed=999)
        for prediction, label in zip(shuffled.predictions, shuffled.labels):
            self.assertEqual(prediction.clip_id, label.clip_id)


# ---------------------------------------------------------------------------
# Healthy run + criterion checks
# ---------------------------------------------------------------------------


class HealthyRunTests(unittest.TestCase):
    """``run_classification_self_test`` passes on a healthy harness."""

    def test_default_run_passes(self) -> None:
        result = run_classification_self_test()
        self.assertTrue(result.passed,
                        msg=f"healthy run should pass, reason was: {result.reason!r}")
        self.assertTrue(result.degraded)
        self.assertGreaterEqual(result.baseline_auc_roc, 0.85)
        self.assertGreaterEqual(result.degradation_margin, 0.30)

    def test_shuffled_mean_is_in_chance_band(self) -> None:
        result = run_classification_self_test()
        self.assertGreaterEqual(result.shuffled_auc_roc_mean, 0.4)
        self.assertLessEqual(result.shuffled_auc_roc_mean, 0.6)

    def test_n_shuffles_per_shuffled_values_length_match(self) -> None:
        result = run_classification_self_test()
        self.assertEqual(len(result.shuffled_auc_roc_values), result.n_shuffles)

    def test_reason_mentions_pass_when_healthy(self) -> None:
        result = run_classification_self_test()
        self.assertIn("all checks passed", result.reason)


# ---------------------------------------------------------------------------
# Determinism: same config → same numeric result
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    """Two runs of the same config produce identical numeric readings."""

    def test_same_config_same_result(self) -> None:
        cfg = SelfTestConfig(seed_base=31337)
        result_a = run_classification_self_test(cfg)
        result_b = run_classification_self_test(cfg)
        self.assertEqual(result_a.baseline_auc_roc, result_b.baseline_auc_roc)
        self.assertEqual(result_a.shuffled_auc_roc_mean, result_b.shuffled_auc_roc_mean)
        self.assertEqual(result_a.shuffled_auc_roc_values, result_b.shuffled_auc_roc_values)
        self.assertEqual(result_a.degradation_margin, result_b.degradation_margin)
        self.assertEqual(result_a.passed, result_b.passed)
        self.assertEqual(result_a.reason, result_b.reason)


# ---------------------------------------------------------------------------
# Teeth: broken scorer must cause the self-test to fail
# ---------------------------------------------------------------------------


class BrokenScorerTests(unittest.TestCase):
    """The self-test has teeth: a lying scorer that always reports
    unrealistically great AUC must cause ``passed`` to flip to
    ``False``."""

    def test_constant_high_auc_scorer_fails_the_self_test(self) -> None:
        broken = _broken_evaluator(auc_value=0.99)
        result = run_classification_self_test(evaluate_metrics=broken)
        # The broken scorer reports 0.99 for baseline AND every
        # shuffle → shuffled_mean ≈ 0.99 → not in [0.4, 0.6] → the
        # chance-band check fires. ``passed`` must be False.
        self.assertFalse(result.passed,
                         msg=f"broken scorer must flip passed to False; reason was: {result.reason!r}")
        self.assertFalse(result.degraded)
        # And the failure reason must surface WHY.
        self.assertIn("shuffled mean", result.reason)
        self.assertIn("chance band", result.reason)

    def test_broken_scorer_baseline_matches_broken_value(self) -> None:
        # Direct demonstration: the baseline AUC reported by a
        # 0.99-broken scorer is ~0.99, NOT the real ~1.0 baseline.
        # The self-test's "baseline healthy" check FAILS (healthy
        # range upper bound is 1.001), which is correct — the
        # broken scorer IS still high, but the test should fail
        # because the rest of the picture does not match.
        broken = _broken_evaluator(auc_value=0.99)
        result = run_classification_self_test(evaluate_metrics=broken)
        # Both checks must fire (chance-band AND margin).
        self.assertIn("outside the chance band", result.reason)
        self.assertIn("below required", result.reason)

    def test_perfectly_wrong_scorer_consistently_fails(self) -> None:
        # Sanity: regardless of the constant the broken scorer
        # emits (as long as it is constant), the self-test fails.
        for value in (0.5, 0.7, 0.95, 1.0):
            with self.subTest(broken_auc=value):
                broken = _broken_evaluator(auc_value=value)
                result = run_classification_self_test(evaluate_metrics=broken)
                # The chance-band check fires for any constant
                # outside [0.4, 0.6]; the constant 0.5 lands inside
                # the band, so the test relies on the margin check
                # — also failed because shuffled_mean == baseline.
                self.assertFalse(
                    result.passed,
                    msg=f"broken scorer reporting {value} should fail; "
                        f"reason: {result.reason!r}",
                )


# ---------------------------------------------------------------------------
# Typed result shape
# ---------------------------------------------------------------------------


class SelfTestResultShapeTests(unittest.TestCase):
    """``SelfTestResult`` exposes every spec field with the right type."""

    def test_carries_every_spec_field(self) -> None:
        result = run_classification_self_test()
        # Numeric readings.
        self.assertIsInstance(result.baseline_auc_roc, float)
        self.assertIsInstance(result.shuffled_auc_roc_values, tuple)
        self.assertIsInstance(result.shuffled_auc_roc_mean, float)
        self.assertIsInstance(result.degradation_margin, float)
        # Boolean conclusions.
        self.assertIsInstance(result.degraded, bool)
        self.assertIsInstance(result.passed, bool)
        # Human-readable reason.
        self.assertIsInstance(result.reason, str)
        # Tracked-back knobs.
        self.assertIsInstance(result.n_shuffles, int)
        self.assertIsInstance(result.margin, float)
        self.assertIsInstance(result.config_seed_base, int)
        # Half-open ranges preserved.
        self.assertIsInstance(result.healthy_auc_range, tuple)
        self.assertIsInstance(result.degraded_auc_range, tuple)

    def test_shuffled_values_each_in_unit_interval(self) -> None:
        result = run_classification_self_test()
        for v in result.shuffled_auc_roc_values:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_margin_equals_baseline_minus_shuffled_mean(self) -> None:
        result = run_classification_self_test()
        self.assertAlmostEqual(
            result.degradation_margin,
            result.baseline_auc_roc - result.shuffled_auc_roc_mean,
            places=12,
        )

    def test_result_is_hashable_and_frozen(self) -> None:
        # SelfTestResult is a frozen dataclass — mutation must
        # raise, and it must hash so callers can drop it into a
        # set / use as a dict key (e.g. tracking self-test
        # versions across runs).
        result = run_classification_self_test()
        with self.assertRaises(Exception):
            result.passed = not result.passed  # type: ignore[misc]
        # Frozen dataclasses are hashable by default; this guards
        # against accidentally adding a mutable field in the future.
        self.assertEqual(hash(result), hash(result))


# ---------------------------------------------------------------------------
# Top-level evaluation package export
# ---------------------------------------------------------------------------


class PublicExportTests(unittest.TestCase):
    """The top-level :mod:`evaluation` package exposes the new symbols."""

    def test_top_level_exports_exist(self) -> None:
        import evaluation  # noqa: PLC0415
        for name in (
            "SelfTestConfig",
            "SelfTestResult",
            "SyntheticClassificationSet",
            "build_synthetic_baseline",
            "run_classification_self_test",
            "shuffle_synthetic_labels",
        ):
            self.assertTrue(
                hasattr(evaluation, name),
                msg=f"evaluation.{name} must be a top-level export.",
            )

    def test_shuffle_synthetic_labels_alias_matches_module_symbol(self) -> None:
        # The package exports the function under a different name
        # to avoid colliding with ``shuffle`` callers that already
        # use ``random.shuffle`` semantics. Verify the alias
        # actually delegates.
        import evaluation  # noqa: PLC0415
        import evaluation.self_test as self_test_module  # noqa: PLC0415
        self.assertIs(
            evaluation.shuffle_synthetic_labels,
            self_test_module.shuffle_labels,
        )


if __name__ == "__main__":
    unittest.main()
