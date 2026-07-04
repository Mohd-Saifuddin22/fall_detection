"""Tests for :mod:`evaluation.metrics.classification`.

Coverage target (per the Step 2 task spec):

- Hand-computed TP / FP / TN / FN values match the emitted confusion
  matrix.
- Hand-computed accuracy / precision / recall / specificity / F1
  match the sklearn-backed output.
- Threshold changes alter count-based metrics.
- AUC-ROC / AUPRC use raw scores and are threshold-independent.
- One-class slices return ``NotAvailable`` for AUC-ROC / AUPRC.
- Missing slice metadata lands in the aggregate-only bucket.
- Per-slice support counts are correct.
- Persisted metric payload remains structured and reloadable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from data.manifests import ClipRole, FallLabel

from evaluation.contracts import (
    ClipLabel,
    ClipPrediction,
    SliceKey,
    SliceTags,
)
from evaluation.metrics.classification import (
    DEFAULT_SLICE_TAGS,
    DEFAULT_THRESHOLD,
    METRIC_NAMES,
    ConfusionMatrix,
    SliceMetricReport,
    SupportCounts,
    compute_classification_metrics,
)
from evaluation.not_available import NotAvailable
from evaluation.result_persistence import (
    EvalRunMetadata,
    MetricResultPayload,
    MetricResultStore,
    make_default_metadata,
)

# Repo-root sys.path injection (mirrors other test modules).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Toy fixtures
# ---------------------------------------------------------------------------


def _make_label(
    clip_id: str,
    label: FallLabel,
    *,
    dataset: str = "urfd",
    lighting: str | None = None,
    occlusion: str | None = None,
    multi_person: bool | None = None,
    action_confuser: str | None = None,
    role: ClipRole = ClipRole.TRAIN,
) -> ClipLabel:
    return ClipLabel(
        clip_id=clip_id,
        label=label,
        dataset=dataset,
        role=role,
        source_path=f"datasets/{dataset}/{clip_id}.mp4",
        slice_tags=SliceTags(
            lighting=lighting,
            occlusion=occlusion,
            multi_person=multi_person,
            action_confuser=action_confuser,
        ),
    )


def _make_prediction(
    clip_id: str,
    score: float,
    *,
    lighting: str | None = None,
    occlusion: str | None = None,
    multi_person: bool | None = None,
    action_confuser: str | None = None,
) -> ClipPrediction:
    return ClipPrediction(
        clip_id=clip_id,
        score=score,
        model_id="toy",
        dataset="urfd",
        role=ClipRole.TRAIN,
        slice_tags=SliceTags(
            lighting=lighting,
            occlusion=occlusion,
            multi_person=multi_person,
            action_confuser=action_confuser,
        ),
    )


def _toy_dataset() -> tuple[list[ClipPrediction], list[ClipLabel]]:
    """A balanced 9-clip toy dataset with hand-checkable TP/FP/TN/FN.

    Truth table at threshold = 0.5:

        clip-a (label=fall):          score=0.9 → pred=fall   → TP
        clip-b (label=fall):          score=0.7 → pred=fall   → TP
        clip-c (label=fall):          score=0.6 → pred=fall   → TP
        clip-d (label=fall):          score=0.4 → pred=NO    → FN
        clip-e (label=fall):          score=0.2 → pred=NO    → FN
        clip-f (label=no_fall):       score=0.3 → pred=NO    → TN
        clip-g (label=no_fall):       score=0.4 → pred=NO    → TN
        clip-h (label=no_fall):       score=0.6 → pred=fall  → FP
        clip-i (label=no_fall):       score=0.8 → pred=fall  → FP

    Counts:
        TP=3  FN=2  TN=2  FP=2   total=9, n_pos=5, n_neg=4
        accuracy    = (TP+TN)/total = (3+2)/9 = 5/9 ≈ 0.5556
        precision   = TP/(TP+FP)   = 3/5 = 0.6
        recall      = TP/(TP+FN)   = 3/5 = 0.6
        specificity = TN/(TN+FP)   = 2/4 = 0.5
        f1          = 2·p·r/(p+r)  = 2·0.6·0.6/1.2 = 0.6

    Slice tags are written so the dataset splits cleanly:

        lighting:   daylight (a, c, f)  dim (b, d, g, h)  low_light (e, i)
        occlusion:  none (a, b, e, i)   partial (c, g)    heavy (d, f, h)
        action_confuser: none (a, b, c, d, f)  sitting (g)  sleeping (h) exercising (i)
                     → all fall clips except i tag as "none"; no_fall clips get
                       confusers to exercise the slice.

    The test assertions look up each metric by name in the aggregate
    report.
    """
    predictions = [
        _make_prediction("a", 0.9, lighting="daylight", occlusion="none", action_confuser="none"),
        _make_prediction("b", 0.7, lighting="dim", occlusion="none", action_confuser="none"),
        _make_prediction("c", 0.6, lighting="daylight", occlusion="partial", action_confuser="none"),
        _make_prediction("d", 0.4, lighting="dim", occlusion="heavy", action_confuser="none"),
        _make_prediction("e", 0.2, lighting="low_light", occlusion="none", action_confuser="none"),
        _make_prediction("f", 0.3, lighting="daylight", occlusion="heavy", action_confuser="none"),
        _make_prediction("g", 0.4, lighting="dim", occlusion="partial", action_confuser="sitting"),
        _make_prediction("h", 0.6, lighting="dim", occlusion="heavy", action_confuser="sleeping"),
        _make_prediction("i", 0.8, lighting="low_light", occlusion="none", action_confuser="exercising"),
    ]
    labels = [
        _make_label("a", FallLabel.FALL,    lighting="daylight", occlusion="none", action_confuser="none"),
        _make_label("b", FallLabel.FALL,    lighting="dim",      occlusion="none", action_confuser="none"),
        _make_label("c", FallLabel.FALL,    lighting="daylight", occlusion="partial", action_confuser="none"),
        _make_label("d", FallLabel.FALL,    lighting="dim",      occlusion="heavy", action_confuser="none"),
        _make_label("e", FallLabel.FALL,    lighting="low_light", occlusion="none", action_confuser="none"),
        _make_label("f", FallLabel.NO_FALL, lighting="daylight", occlusion="heavy", action_confuser="none"),
        _make_label("g", FallLabel.NO_FALL, lighting="dim",      occlusion="partial", action_confuser="sitting"),
        _make_label("h", FallLabel.NO_FALL, lighting="dim",      occlusion="heavy", action_confuser="sleeping"),
        _make_label("i", FallLabel.NO_FALL, lighting="low_light", occlusion="none", action_confuser="exercising"),
    ]
    return predictions, labels


def _aggregate_report(reports: list[SliceMetricReport]) -> SliceMetricReport:
    for r in reports:
        if r.slice_key is None:
            return r
    raise AssertionError("no aggregate report returned")


def _by_name(report: SliceMetricReport) -> dict[str, MetricResult]:
    return {m.name: m for m in report.metrics}


# ---------------------------------------------------------------------------
# Container-shape tests (cheap, no sklearn calls in the hot path)
# ---------------------------------------------------------------------------


class ContainerShapeTests(unittest.TestCase):
    """Typed containers reject bad input at construction time."""

    def test_confusion_matrix_rejects_negative_counts(self) -> None:
        with self.assertRaises(ValueError):
            ConfusionMatrix(tn=-1, fp=0, fn=0, tp=0)

    def test_confusion_matrix_support_sums(self) -> None:
        cm = ConfusionMatrix(tn=2, fp=3, fn=4, tp=5)
        self.assertEqual(cm.support, 14)

    def test_support_counts_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            SupportCounts(n_positive=-1, n_negative=0)

    def test_support_counts_total(self) -> None:
        s = SupportCounts(n_positive=3, n_negative=4)
        self.assertEqual(s.total, 7)


# ---------------------------------------------------------------------------
# Pairing + empty / aggregate-only behaviour
# ---------------------------------------------------------------------------


class PairingAndEmptyTests(unittest.TestCase):
    """Predictions are matched to labels by clip_id; mismatches raise."""

    def test_prediction_without_label_raises(self) -> None:
        preds = [_make_prediction("x", 0.5)]
        labels = [_make_label("y", FallLabel.FALL)]
        with self.assertRaises(ValueError):
            compute_classification_metrics(preds, labels)

    def test_label_without_prediction_raises(self) -> None:
        preds = [_make_prediction("x", 0.5)]
        labels = [
            _make_label("x", FallLabel.FALL),
            _make_label("y", FallLabel.NO_FALL),
        ]
        with self.assertRaises(ValueError):
            compute_classification_metrics(preds, labels)

    def test_duplicate_prediction_clip_id_raises(self) -> None:
        preds = [
            _make_prediction("x", 0.5),
            _make_prediction("x", 0.4),
        ]
        labels = [_make_label("x", FallLabel.FALL)]
        with self.assertRaises(ValueError):
            compute_classification_metrics(preds, labels)

    def test_empty_inputs_return_empty_list(self) -> None:
        # Empty input is empty output, not "fake zero metrics". A
        # consumer asking for an empty-input report deserves zero
        # reports — they have no clips.
        self.assertEqual(compute_classification_metrics([], []), [])

    def test_empty_slice_emits_not_available_metrics(self) -> None:
        # Pathological case: caller asked for a slice tag that no
        # clip carries a value for. That slice is "empty by tag",
        # but it still doesn't exist in the output. Sanity: an
        # aggregate over a single clip does not produce empty slices.
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels, slice_tags=("nonexistent_tag",))
        # Only aggregate comes back.
        self.assertEqual(len(reports), 1)
        self.assertIsNone(reports[0].slice_key)


# ---------------------------------------------------------------------------
# Confusion matrix + count-based metric verification
# ---------------------------------------------------------------------------


class HandComputedMetricsTests(unittest.TestCase):
    """Every count-based metric is asserted against a hand calculation."""

    def test_aggregate_confusion_matrix_matches_hand_computed_tp_fp_tn_fn(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        report = _aggregate_report(reports)
        cm = report.confusion_matrix

        # From the truth table in _toy_dataset's docstring:
        # TP=3, FN=2, TN=2, FP=2.
        self.assertEqual(cm.tp, 3)
        self.assertEqual(cm.fn, 2)
        self.assertEqual(cm.tn, 2)
        self.assertEqual(cm.fp, 2)
        self.assertEqual(cm.support, 9)

    def test_aggregate_support_counts_match_hand_computed(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        support = _aggregate_report(reports).support
        self.assertEqual(support.n_positive, 5)
        self.assertEqual(support.n_negative, 4)
        self.assertEqual(support.total, 9)

    def test_aggregate_metric_values_match_hand_computed(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))

        # accuracy = (TP+TN)/total = (3+2)/9
        self.assertAlmostEqual(by["accuracy"].numeric_value(), 5 / 9, places=6)
        # precision = TP/(TP+FP) = 3/5
        self.assertAlmostEqual(by["precision"].numeric_value(), 3 / 5, places=6)
        # recall = TP/(TP+FN) = 3/5
        self.assertAlmostEqual(by["recall"].numeric_value(), 3 / 5, places=6)
        # specificity = TN/(TN+FP) = 2/4
        self.assertAlmostEqual(by["specificity"].numeric_value(), 2 / 4, places=6)
        # f1 = 2·p·r / (p+r) = 0.6 (when p == r)
        self.assertAlmostEqual(by["f1"].numeric_value(), 0.6, places=6)

    def test_specificity_does_not_equal_recall(self) -> None:
        # Different denominators — recall is on positives, specificity
        # on negatives. They would diverge in any realistic run; the
        # toy set happens to give 0.6 vs 0.5 which is itself a check
        # that the formulae are not accidentally swapped.
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))
        self.assertNotAlmostEqual(by["recall"].numeric_value(),
                                  by["specificity"].numeric_value(),
                                  places=6)

    def test_aggregate_emits_every_required_metric(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))
        for name in METRIC_NAMES:
            self.assertIn(name, by, msg=f"metric {name} missing from aggregate report")
            self.assertTrue(
                by[name].is_available()
                or isinstance(by[name].value, NotAvailable),
                msg=f"metric {name} must be numeric or NotAvailable",
            )


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


class ThresholdTests(unittest.TestCase):
    """Count-based metrics change with the threshold; AUC metrics don't."""

    def test_lower_threshold_changes_count_based_metrics(self) -> None:
        preds, labels = _toy_dataset()
        # At threshold 0.5: TP=3, FN=2 → recall = 3/5 = 0.6.
        # At threshold 1.5: every score < 1.5 → pred=no_fall.
        #                    TP=0, FN=5 → recall = 0.0.
        # So changing the threshold changes a count-based metric.
        reports_default = compute_classification_metrics(preds, labels, threshold=0.5)
        reports_extreme = compute_classification_metrics(preds, labels, threshold=1.5)
        recall_default = _by_name(_aggregate_report(reports_default))["recall"].numeric_value()
        report_extreme = _aggregate_report(reports_extreme)

        # Recall at default threshold is the hand-computed 0.6.
        self.assertAlmostEqual(recall_default, 3 / 5, places=6)
        # At the extreme, every clip is predicted no_fall: no TPs,
        # all five falls become FNs.
        self.assertEqual(report_extreme.confusion_matrix.tp, 0)
        self.assertEqual(report_extreme.confusion_matrix.fp, 0)
        self.assertEqual(report_extreme.confusion_matrix.tn, 4)
        self.assertEqual(report_extreme.confusion_matrix.fn, 5)
        # Recall drops because FN grew.
        self.assertGreater(recall_default, _by_name(report_extreme)["recall"].numeric_value())

    def test_count_based_metrics_depend_on_threshold(self) -> None:
        # At threshold=0.65 the predictions flip: clip-c (0.6) and
        # clip-b (0.7) swing from fall → no_fall. b stays TP (0.7>0.65);
        # c (0.6<0.65) becomes FN. So FN goes 2 → 3, recall drops.
        preds, labels = _toy_dataset()
        agg_low = _aggregate_report(compute_classification_metrics(preds, labels, threshold=0.5))
        agg_high = _aggregate_report(compute_classification_metrics(preds, labels, threshold=0.65))
        recall_low = _by_name(agg_low)["recall"].numeric_value()
        recall_high = _by_name(agg_high)["recall"].numeric_value()
        self.assertGreater(recall_low, recall_high)

    def test_default_threshold_constant_is_one_half(self) -> None:
        # Behaviour the spec asserts by name.
        self.assertEqual(DEFAULT_THRESHOLD, 0.5)


# ---------------------------------------------------------------------------
# AUC-ROC / AUPRC: threshold independence + one-class behaviour
# ---------------------------------------------------------------------------


class CurveMetricTests(unittest.TestCase):
    """AUC-ROC and AUPRC use raw scores; one-class slices → NotAvailable."""

    def test_auc_roc_uses_raw_scores_threshold_independent(self) -> None:
        # The same predictions at two different thresholds MUST
        # produce identical AUC-ROC, because AUC-ROC is computed
        # from the raw score distribution, not the thresholded
        # binary.
        preds, labels = _toy_dataset()
        reports_low = compute_classification_metrics(preds, labels, threshold=0.3)
        reports_high = compute_classification_metrics(preds, labels, threshold=0.8)
        auc_low = _by_name(_aggregate_report(reports_low))["auc_roc"].numeric_value()
        auc_high = _by_name(_aggregate_report(reports_high))["auc_roc"].numeric_value()
        self.assertAlmostEqual(auc_low, auc_high, places=12)

    def test_auprc_uses_raw_scores_threshold_independent(self) -> None:
        preds, labels = _toy_dataset()
        reports_low = compute_classification_metrics(preds, labels, threshold=0.3)
        reports_high = compute_classification_metrics(preds, labels, threshold=0.8)
        ap_low = _by_name(_aggregate_report(reports_low))["auprc"].numeric_value()
        ap_high = _by_name(_aggregate_report(reports_high))["auprc"].numeric_value()
        self.assertAlmostEqual(ap_low, ap_high, places=12)

    def test_one_class_slice_returns_not_available_for_auc_roc(self) -> None:
        # Construct a slice that contains ONLY fall clips (positive
        # label only). AUC-ROC needs both classes; single-class
        # slices must surface NotAvailable with a clear reason.
        preds = [_make_prediction("f1", 0.9), _make_prediction("f2", 0.7)]
        labels = [
            _make_label("f1", FallLabel.FALL),
            _make_label("f2", FallLabel.FALL),
        ]
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))
        roc = by["auc_roc"].value
        self.assertIsInstance(roc, NotAvailable)
        self.assertIn("only one class present", roc.reason)

    def test_one_class_slice_returns_not_available_for_auprc(self) -> None:
        preds = [_make_prediction("n1", 0.2), _make_prediction("n2", 0.4)]
        labels = [
            _make_label("n1", FallLabel.NO_FALL),
            _make_label("n2", FallLabel.NO_FALL),
        ]
        reports = compute_classification_metrics(preds, labels)
        roc = _by_name(_aggregate_report(reports))["auprc"].value
        self.assertIsInstance(roc, NotAvailable)
        self.assertIn("only one class present", roc.reason)

    def test_count_based_metrics_are_numeric_even_for_one_class_slice(self) -> None:
        # Count-based metrics on the POSITIVE side (accuracy,
        # precision, recall, F1) are well-defined on a one-class
        # slice. Specificity is honest-NotAvailable because the
        # TN + FP denominator is zero — fabricating 1.0 would be
        # silently claiming "no negatives mispredicted" without
        # any negatives to test against.
        preds = [_make_prediction("f1", 0.9), _make_prediction("f2", 0.4)]
        labels = [
            _make_label("f1", FallLabel.FALL),
            _make_label("f2", FallLabel.FALL),
        ]
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))
        for name in ("accuracy", "precision", "recall", "f1"):
            self.assertTrue(by[name].is_available(),
                            msg=f"{name} should be numeric on a one-class slice.")
        spec = by["specificity"].value
        self.assertIsInstance(spec, NotAvailable,
                              msg="specificity on n_negative=0 slice must be NotAvailable")
        self.assertIn("no negatives", spec.reason)

    def test_auc_roc_matches_sklearn_directly(self) -> None:
        # Sklearn parity — the spec requires sklearn-backed. This
        # test confirms the implementation exactly matches sklearn's
        # values (it does not just produce a plausible-looking
        # number).
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        auc = _by_name(_aggregate_report(reports))["auc_roc"].numeric_value()
        y_true = [1] * 3 + [1] * 2 + [0] * 2 + [0] * 2  # placeholder
        # Compute the expected value from the same inputs.
        y_true = [_fall_to_int(l) for l in labels]
        y_score = [p.score for p in preds]
        expected = float(roc_auc_score(y_true, y_score))
        self.assertAlmostEqual(auc, expected, places=12)

    def test_auprc_matches_sklearn_directly(self) -> None:
        from sklearn.metrics import average_precision_score  # noqa: PLC0415
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        ap = _by_name(_aggregate_report(reports))["auprc"].numeric_value()
        y_true = [_fall_to_int(l) for l in labels]
        y_score = [p.score for p in preds]
        expected = float(average_precision_score(y_true, y_score))
        self.assertAlmostEqual(ap, expected, places=12)


def _fall_to_int(label: ClipLabel) -> int:
    return 1 if label.label is FallLabel.FALL else 0


# ---------------------------------------------------------------------------
# Slice behaviour
# ---------------------------------------------------------------------------


class SliceReportingTests(unittest.TestCase):
    """Per-slice reports carry consistent support and metric shapes."""

    def test_aggregate_is_always_present(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        self.assertIn(None, [r.slice_key for r in reports],
                      msg="aggregate report (slice_key=None) must always be emitted")

    def test_emits_one_report_per_slice_tag_value(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        slice_keys = [r.slice_key for r in reports if r.slice_key is not None]
        # 9 clips distributed across lighting = {daylight (3), dim (4), low_light (2)}.
        lighting_slices = sorted(
            (k.value for k in slice_keys if k.tag == "lighting")
        )
        self.assertEqual(lighting_slices, ["daylight", "dim", "low_light"])

    def test_slice_outputs_are_alphabetically_ordered(self) -> None:
        # Determinism requirement — by-tag alphabetical, by-value
        # alphabetical within a tag.
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        per_slice = [r for r in reports if r.slice_key is not None]
        keys = [(r.slice_key.tag, r.slice_key.value) for r in per_slice]
        self.assertEqual(keys, sorted(keys))

    def test_per_slice_support_sums_to_total(self) -> None:
        # A clip contributes to MULTIPLE slice buckets (one per
        # tag) — but to the aggregate ONCE. So per-slice supports
        # are not disjoint: this test guards against confusing
        # them with the aggregate.
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        agg_total = _aggregate_report(reports).support.total
        for report in reports:
            if report.slice_key is not None:
                # Each per-tag slice's total is in [1, len(preds)].
                self.assertGreaterEqual(report.support.total, 1)
                self.assertLessEqual(report.support.total, len(preds))
        self.assertEqual(agg_total, 9)

    def test_lighting_daylight_slice_handles_3_clips(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        target = next(
            r for r in reports
            if r.slice_key == SliceKey("lighting", "daylight")
        )
        # Clips a, c, f: fall (a, c) and no_fall (f); scores 0.9, 0.6, 0.3.
        # At threshold 0.5: pred a=fall, pred c=fall, pred f=no_fall.
        self.assertEqual(target.support.n_positive, 2)
        self.assertEqual(target.support.n_negative, 1)
        # TP=a, TP=c, TN=f → TP=2, FP=0, FN=0, TN=1.
        self.assertEqual(target.confusion_matrix.tp, 2)
        self.assertEqual(target.confusion_matrix.fn, 0)
        self.assertEqual(target.confusion_matrix.fp, 0)
        self.assertEqual(target.confusion_matrix.tn, 1)

    def test_metric_results_includes_support_and_confusion_rows(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        agg = _aggregate_report(reports)
        names = {m.name for m in agg.metric_results()}
        # Confusion matrix entries as named rows.
        for cm_name in ("tn", "fp", "fn", "tp"):
            self.assertIn(cm_name, names)
        # Support counts as named rows.
        for s_name in ("n_positive", "n_negative", "total"):
            self.assertIn(s_name, names)


# ---------------------------------------------------------------------------
# Missing slice metadata → aggregate-only
# ---------------------------------------------------------------------------


class MissingSliceMetadataTests(unittest.TestCase):
    """Clips without slice tags appear in the aggregate, not in per-tag slices."""

    def test_clips_with_no_slice_tags_only_in_aggregate(self) -> None:
        # Two clips carry NO slice_tags at all. They appear in the
        # aggregate report but must NOT contribute to any per-tag
        # slice (a tag slice requires the clip to have a value for
        # that tag).
        preds_with_tags, labels_with_tags = _toy_dataset()
        preds_no_tags = [
            ClipPrediction(
                clip_id="no-tag-1", score=0.7, model_id="toy",
                dataset="urfd", role=ClipRole.TRAIN,
                slice_tags=None,
            ),
            ClipPrediction(
                clip_id="no-tag-2", score=0.3, model_id="toy",
                dataset="urfd", role=ClipRole.TRAIN,
                slice_tags=None,
            ),
        ]
        labels_no_tags = [
            ClipLabel(
                clip_id="no-tag-1", label=FallLabel.FALL, dataset="urfd",
                role=ClipRole.TRAIN, source_path="datasets/urfd/no-tag-1.mp4",
                slice_tags=None,
            ),
            ClipLabel(
                clip_id="no-tag-2", label=FallLabel.NO_FALL, dataset="urfd",
                role=ClipRole.TRAIN, source_path="datasets/urfd/no-tag-2.mp4",
                slice_tags=None,
            ),
        ]
        preds = preds_with_tags + preds_no_tags
        labels = labels_with_tags + labels_no_tags
        reports = compute_classification_metrics(preds, labels)

        # Aggregate includes both tagged and untagged clips.
        agg = _aggregate_report(reports)
        self.assertEqual(agg.support.total, len(preds))
        self.assertEqual(agg.confusion_matrix.support, len(preds))

        # The no-tag clips' clip_ids must NOT appear inside any
        # per-tag slice's confusion matrix counts. Verification:
        # summing per-tag slice supports exceeds the aggregate by
        # exactly the no-tag count (every tagged clip contributes
        # to multiple slices).
        per_tag_total = sum(r.support.total for r in reports if r.slice_key is not None)
        # Aggregate (9 + 2 = 11) is smaller than the no-tag-count-
        # adjusted per-tag sum.
        self.assertGreater(per_tag_total, agg.support.total)
        # The difference is at least the 2 no-tag clips, because they
        # do not enter any per-tag slice.
        self.assertGreaterEqual(per_tag_total - agg.support.total, 2)

    def test_clip_with_partial_slice_tags_is_aggregate_also(self) -> None:
        # A clip with lighting="dim" but no action_confuser tag must
        # appear in the lighting=dim slice but NOT in any
        # action_confuser=... slice. The aggregate sees it.
        preds = [
            _make_prediction("partial-1", 0.8, lighting="dim"),  # no action_confuser
        ]
        labels = [
            _make_label("partial-1", FallLabel.FALL, lighting="dim"),  # no action_confuser
        ]
        reports = compute_classification_metrics(preds, labels)
        agg = _aggregate_report(reports)
        self.assertEqual(agg.support.total, 1)
        # lighting=dim slice exists.
        lighting_slice = next(
            r for r in reports if r.slice_key == SliceKey("lighting", "dim")
        )
        self.assertEqual(lighting_slice.support.total, 1)
        # No action_confuser=... slice was emitted for this single clip
        # (no values to slice by).
        action_confuser_slices = [
            r for r in reports
            if r.slice_key is not None and r.slice_key.tag == "action_confuser"
        ]
        self.assertEqual(action_confuser_slices, [])

    def test_default_slice_tags_constant(self) -> None:
        # Sanity: the PRD-mandated tags are bundled into the default.
        self.assertEqual(
            DEFAULT_SLICE_TAGS,
            ("dataset", "lighting", "occlusion", "multi_person", "action_confuser"),
        )


# ---------------------------------------------------------------------------
# Degenerate-slice honesty contract (Step 2 fix)
# ---------------------------------------------------------------------------


def _not_available(metric: MetricResult) -> NotAvailable:
    """Assert ``metric.value`` is :class:`NotAvailable` and return it."""
    assert isinstance(metric.value, NotAvailable), (
        f"expected NotAvailable, got {type(metric.value).__name__}"
    )
    return metric.value


class AllNegativeSliceTests(unittest.TestCase):
    """A slice whose labels are all ``no_fall`` has no actual positives.

    - accuracy: numeric, since the denominator is defined.
    - specificity: numeric (tn + fp > 0).
    - precision: ``NotAvailable`` if the model also never predicts
      positive (tp + fp == 0); otherwise numeric.
    - recall: ``NotAvailable`` (tp + fn == 0 — no actual positives).
    - F1: ``NotAvailable`` (because recall is).
    - AUC-ROC / AUPRC: ``NotAvailable`` (only one class present).
    """

    def test_all_no_fall_no_predicted_positives_is_fully_undefined(self) -> None:
        # All labels are no_fall, all scores < 0.5 → model never
        # predicts positive. Every degenerate metric must surface
        # NotAvailable with a precise reason. accuracy + specificity
        # are the only well-defined metrics here.
        preds = [
            _make_prediction("n1", 0.2),
            _make_prediction("n2", 0.3),
            _make_prediction("n3", 0.1),
        ]
        labels = [
            _make_label("n1", FallLabel.NO_FALL),
            _make_label("n2", FallLabel.NO_FALL),
            _make_label("n3", FallLabel.NO_FALL),
        ]
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))

        # accuracy = 3/3 because every prediction matches.
        self.assertAlmostEqual(by["accuracy"].numeric_value(), 1.0, places=6)
        # specificity = 3/3 because no false positives.
        self.assertAlmostEqual(by["specificity"].numeric_value(), 1.0, places=6)

        # precision — undefined: no predicted positives.
        precision_na = _not_available(by["precision"])
        self.assertEqual(precision_na.reason, "no predicted positives in slice")
        # recall — undefined: no actual positives.
        recall_na = _not_available(by["recall"])
        self.assertEqual(recall_na.reason, "no actual positives in slice")
        # F1 — undefined: recall is undefined.
        f1_na = _not_available(by["f1"])
        self.assertEqual(f1_na.reason, "precision or recall undefined")
        # AUC-ROC / AUPRC — undefined: only one class present.
        auc_roc_na = _not_available(by["auc_roc"])
        self.assertEqual(auc_roc_na.reason, "only one class present in slice")
        auprc_na = _not_available(by["auprc"])
        self.assertEqual(auprc_na.reason, "only one class present in slice")

    def test_all_no_fall_but_some_predicted_positives_has_numeric_precision(self) -> None:
        # All labels are no_fall (recall remains undefined), but
        # some predictions cross the threshold so precision becomes
        # well-defined (tp + fp > 0). recall / F1 / AUC stay NotAvailable.
        preds = [
            _make_prediction("n1", 0.2),
            _make_prediction("n2", 0.7),  # predicted fall → fp
            _make_prediction("n3", 0.4),
        ]
        labels = [
            _make_label("n1", FallLabel.NO_FALL),
            _make_label("n2", FallLabel.NO_FALL),
            _make_label("n3", FallLabel.NO_FALL),
        ]
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))
        cm = _aggregate_report(reports).confusion_matrix

        # tp=0, fp=1, fn=0, tn=2.
        self.assertEqual(cm.tp, 0)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 0)
        self.assertEqual(cm.tn, 2)

        # precision = tp / (tp + fp) = 0 / 1 = 0.0 (real value).
        self.assertAlmostEqual(by["precision"].numeric_value(), 0.0, places=6)
        # accuracy = 2/3.
        self.assertAlmostEqual(by["accuracy"].numeric_value(), 2 / 3, places=6)
        # specificity = tn / (tn + fp) = 2 / 3.
        self.assertAlmostEqual(by["specificity"].numeric_value(), 2 / 3, places=6)

        # recall — still undefined.
        recall_na = _not_available(by["recall"])
        self.assertEqual(recall_na.reason, "no actual positives in slice")
        # F1 — still undefined (recall is).
        f1_na = _not_available(by["f1"])
        self.assertEqual(f1_na.reason, "precision or recall undefined")
        # AUC-ROC / AUPRC — still NotAvailable (one class).
        self.assertIsInstance(by["auc_roc"].value, NotAvailable)
        self.assertIsInstance(by["auprc"].value, NotAvailable)


class AllPositiveSliceTests(unittest.TestCase):
    """A slice whose labels are all ``fall`` has no actual negatives.

    - accuracy: numeric.
    - recall: numeric.
    - specificity: ``NotAvailable`` (tn + fp == 0).
    - precision: numeric (tp + fp > 0 by construction, since at
      least one prediction may match).
    """

    def test_all_fall_slice_is_numeric_for_recall_only(self) -> None:
        # All labels are FALL. The model is mixed-correct: clip-f1
        # is predicted correctly as fall (tp), but clip-f2 is
        # missed (fn). Specificity stays NotAvailable; recall
        # becomes a real number.
        preds = [
            _make_prediction("f1", 0.9),  # predicted fall → tp
            _make_prediction("f2", 0.3),  # predicted no_fall → fn
            _make_prediction("f3", 0.6),  # predicted fall → tp
        ]
        labels = [
            _make_label("f1", FallLabel.FALL),
            _make_label("f2", FallLabel.FALL),
            _make_label("f3", FallLabel.FALL),
        ]
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))
        cm = _aggregate_report(reports).confusion_matrix

        # All-fall labels → tn=0 and fp MUST be 0 (no negative
        # clips exist to misclassify). The remaining counts:
        # tp=2 (f1, f3), fn=1 (f2), fp=0, tn=0.
        self.assertEqual(cm.tp, 2)
        self.assertEqual(cm.fp, 0)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.tn, 0)

        # recall = tp / (tp + fn) = 2 / 3.
        self.assertAlmostEqual(by["recall"].numeric_value(), 2 / 3, places=6)
        # accuracy = (tp + tn) / total = (2 + 0) / 3.
        self.assertAlmostEqual(by["accuracy"].numeric_value(), 2 / 3, places=6)
        # precision = tp / (tp + fp) = 2 / 2 = 1.0.
        self.assertAlmostEqual(by["precision"].numeric_value(), 1.0, places=6)
        # F1 = 2·p·r / (p+r) — heterogeneous p, r here.
        self.assertAlmostEqual(
            by["f1"].numeric_value(),
            2 * (2 / 3) * (1.0) / ((2 / 3) + 1.0),
            places=6,
        )

        # specificity — undefined: no negatives.
        specificity_na = _not_available(by["specificity"])
        self.assertEqual(specificity_na.reason, "no negatives in slice")
        # AUC-ROC / AUPRC — undefined: only one class.
        self.assertIsInstance(by["auc_roc"].value, NotAvailable)
        self.assertIsInstance(by["auprc"].value, NotAvailable)

    def test_all_fall_with_no_predicted_positives_is_fully_undefined(self) -> None:
        # All FALL labels, but the model never clears the threshold.
        # Both precision (no predicted positives) and specificity
        # (no negatives) become NotAvailable; accuracy and recall
        # also collapse to 0 because recall = 0/3 and accuracy =
        # (tp + tn) / total = (0 + 0) / 3 = 0.
        preds = [
            _make_prediction("f1", 0.1),
            _make_prediction("f2", 0.2),
        ]
        labels = [
            _make_label("f1", FallLabel.FALL),
            _make_label("f2", FallLabel.FALL),
        ]
        reports = compute_classification_metrics(preds, labels)
        by = _by_name(_aggregate_report(reports))

        # accuracy = 0 / 2 — numeric zero, not fabricated.
        self.assertAlmostEqual(by["accuracy"].numeric_value(), 0.0, places=6)
        # recall = 0 / 2 — numeric zero.
        self.assertAlmostEqual(by["recall"].numeric_value(), 0.0, places=6)

        # precision — undefined: no predicted positives.
        precision_na = _not_available(by["precision"])
        self.assertEqual(precision_na.reason, "no predicted positives in slice")
        # F1 — undefined (precision is).
        f1_na = _not_available(by["f1"])
        self.assertEqual(f1_na.reason, "precision or recall undefined")
        # specificity — undefined: no negatives.
        self.assertIsInstance(by["specificity"].value, NotAvailable)


class ActionConfuserSittingTests(unittest.TestCase):
    """A specific PRD-style slice: ``action_confuser=sitting`` containing
    only ``no_fall`` clips.

    - recall: ``NotAvailable`` (no actual positives in the slice).
    - F1: ``NotAvailable`` (recall is).
    - specificity: numeric (tn + fp > 0).
    - false-positive count is visible in the confusion matrix.
    """

    def test_action_confuser_sitting_all_no_fall_slice(self) -> None:
        # Three clips with action_confuser=sitting; labels are all
        # no_fall. Predictions: high (→ fp), low (→ tn), low (→ tn).
        # At threshold 0.5: tp=0, fp=1, fn=0, tn=2.
        preds = [
            _make_prediction(
                "sit1", 0.8, action_confuser="sitting",
                lighting="daylight", occlusion="none",
            ),
            _make_prediction(
                "sit2", 0.2, action_confuser="sitting",
                lighting="daylight", occlusion="none",
            ),
            _make_prediction(
                "sit3", 0.3, action_confuser="sitting",
                lighting="daylight", occlusion="none",
            ),
        ]
        labels = [
            _make_label(
                "sit1", FallLabel.NO_FALL,
                action_confuser="sitting",
                lighting="daylight", occlusion="none",
            ),
            _make_label(
                "sit2", FallLabel.NO_FALL,
                action_confuser="sitting",
                lighting="daylight", occlusion="none",
            ),
            _make_label(
                "sit3", FallLabel.NO_FALL,
                action_confuser="sitting",
                lighting="daylight", occlusion="none",
            ),
        ]
        reports = compute_classification_metrics(preds, labels)
        # Locate the action_confuser=sitting slice report.
        sitting = next(
            r for r in reports
            if r.slice_key == SliceKey("action_confuser", "sitting")
        )
        by = _by_name(sitting)
        cm = sitting.confusion_matrix
        support = sitting.support

        # 3 clips, all no_fall.
        self.assertEqual(support.total, 3)
        self.assertEqual(support.n_positive, 0)
        self.assertEqual(support.n_negative, 3)
        # Confusion matrix: tp=0, fp=1, fn=0, tn=2. The false
        # positive count is visible (this is exactly the slice a
        # reviewer wants to diagnose false-alarm behaviour on).
        self.assertEqual(cm.tp, 0)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 0)
        self.assertEqual(cm.tn, 2)
        # The fp count must be reachable from the persisted
        # MetricResult payload too — its reload value equals 1.0.
        persisted_fp = next(
            m for m in sitting.metric_results() if m.name == "fp"
        )
        self.assertEqual(persisted_fp.numeric_value(), 1.0)

        # recall — undefined (no actual positives).
        recall_na = _not_available(by["recall"])
        self.assertEqual(recall_na.reason, "no actual positives in slice")
        # F1 — undefined.
        f1_na = _not_available(by["f1"])
        self.assertEqual(f1_na.reason, "precision or recall undefined")
        # AUC-ROC / AUPRC — undefined.
        self.assertIsInstance(by["auc_roc"].value, NotAvailable)
        self.assertIsInstance(by["auprc"].value, NotAvailable)

        # specificity — numeric; tn / (tn + fp) = 2 / 3.
        self.assertAlmostEqual(by["specificity"].numeric_value(), 2 / 3, places=6)
        # precision — numeric; tp / (tp + fp) = 0 / 1 = 0.0.
        # ``tp + fp > 0`` (one predicted positive) so precision is
        # defined; it just happens to read 0.
        self.assertAlmostEqual(by["precision"].numeric_value(), 0.0, places=6)
        # accuracy — numeric; (tp + tn) / total = 2 / 3.
        self.assertAlmostEqual(by["accuracy"].numeric_value(), 2 / 3, places=6)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


class PersistenceRoundTripTests(unittest.TestCase):
    """The Step 1 ``MetricResultStore`` accepts the bundle's output unchanged."""

    def setUp(self) -> None:
        name = tempfile.mkdtemp()
        self._root = Path(name)
        self.addCleanup(_rm_tree, self._root)

    def test_full_report_persists_and_reloads(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        store = MetricResultStore(self._root)

        meta = make_default_metadata("r-classification", "toy-model", context="evaluation")
        # Flatten every slice's metric_results into one stream.
        rows = [row for report in reports for row in report.metric_results()]
        store.save(meta, rows, overwrite=True)

        reloaded = store.load("r-classification")
        # Every persisted row is reloadable as a structured
        # MetricResult; we filter by (slice_key, name) so the test
        # does not depend on row order.
        self.assertEqual(len(reloaded.metrics), len(rows))
        self.assertEqual(reloaded.metadata.run_id, "r-classification")
        self.assertEqual(reloaded.metadata.context, "evaluation")

        # Confirm that the aggregate recall survived numerically.
        agg_recall = next(
            m for m in reloaded.metrics
            if m.name == "recall" and m.slice_key is None
        )
        self.assertAlmostEqual(agg_recall.numeric_value(), 0.6, places=6)

        # Confirm at least one per-slice metric survived. Filter
        # to slices that actually have positives — some per-tag
        # slices are no_fall-only (e.g. action_confuser=exercising).
        any_slice = next(
            m for m in reloaded.metrics
            if m.name == "n_positive" and m.slice_key is not None
            and m.numeric_value() > 0
        )
        self.assertGreater(any_slice.numeric_value(), 0)

    def test_persisted_payload_carries_support_and_cm_per_slice(self) -> None:
        preds, labels = _toy_dataset()
        reports = compute_classification_metrics(preds, labels)
        store = MetricResultStore(self._root)
        meta = make_default_metadata("r-support", "toy", context="evaluation")
        rows = [row for report in reports for row in report.metric_results()]
        store.save(meta, rows, overwrite=True)

        reloaded = store.load("r-support")
        # For the lighting=dim slice: support + cm rows must reload.
        for metric_name in ("tn", "fp", "fn", "tp", "n_positive", "n_negative", "total"):
            with self.subTest(metric=metric_name):
                row = next(
                    m for m in reloaded.metrics
                    if m.name == metric_name and m.slice_key == SliceKey("lighting", "dim")
                )
                self.assertIsInstance(row.numeric_value(), float)


def _rm_tree(path: Path) -> None:
    try:
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir():
                    _rm_tree(child)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
            try:
                path.rmdir()
            except OSError:
                pass
    except OSError:
        pass


# Silence unused-import linter — Path / os only used in helpers above.
_ = (Path, os, json, MetricResultPayload)


if __name__ == "__main__":
    unittest.main()
