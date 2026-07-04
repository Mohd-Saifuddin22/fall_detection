"""Detector-of-the-detector self-test.

Issues 004 Step 1 + Step 2 stand up the contracts and the
classification metric bundle. Step 3 builds the self-test that
**proves** the bundle catches a known-bad change. The PRD's
*Testing Decisions* are explicit:

    > Test the detector. Periodically inject a known-bad change
    > (shuffled labels, degraded tracking) into a shadow path and
    > confirm the eval catches it — an eval nobody validates is
    > decorative.

This module is the first half of that requirement (classification
sentinel). Tracking / skeleton sentinels follow the same pattern
and are listed in the extension seam at the bottom of this file —
they are NOT implemented here.

How it works
------------

1. Build a deterministic synthetic classification set where
   ``score`` is strongly correlated with ``label`` (AUC-ROC ≈ 1).
2. Run the **real** :func:`compute_classification_metrics` on the
   baseline. Read the aggregate AUC-ROC. This is the
   ``baseline_auc_roc`` reading.
3. For ``n_shuffles`` deterministic seeds (seeds step from
   ``config.seed_base + 1`` upward), build a corrupted copy of
   the synthetic set whose labels have been shuffled relative to
   the predictions. Run the real metrics bundle on each.
4. Average the per-shuffle AUC-ROC readings → ``shuffled_mean``.
5. Apply three pass criteria:

       a. ``baseline_auc_roc`` is in the healthy range
          (:attr:`SelfTestConfig.healthy_auc_range`).
       b. ``shuffled_mean`` is in the chance range
          (:attr:`SelfTestConfig.degraded_auc_range`).
       c. ``baseline - shuffled_mean >= margin``.

   All three must hold for ``passed=True``.

6. Return a :class:`SelfTestResult` carrying every relevant number
   so a reviewer / CI hook can decide on its own.

A broken scorer (the test with teeth)
--------------------------------------

The scorer is injectable via ``evaluate_metrics``. The default
delegates to :func:`compute_classification_metrics`. Callers —
typically the self-test's test suite — can substitute a
deterministic broken scorer that always reports AUC-ROC = 0.99
regardless of input. With that broken scorer the baseline reading
is 0.99 AND the shuffled readings are 0.99, the margin collapses
to 0, and ``passed`` flips to ``False``. That is exactly the
"decorative eval that doesn't catch" failure mode the test exists
to detect.

Determinism
-----------

All randomness lives behind ``random.Random(seed)``. The same
``SelfTestConfig`` (same seed) → same ``SelfTestResult`` (modulo
the per-shuffle value list ordering). Without an explicit seed
the module refuses to run — there is no implicit global RNG.

Extension seam (future known-bad component tests)
------------------------------------------------

This module owns the classification sentinel only. Future
sentinels reuse the same pattern (synthetic build + deterministic
shuffle + real harness + sentinel metric + pass criteria):

- Tracking sentinel: synthetic ByteTrack output with injected ID
  switches; sentinel metric is IDF1 / MOTA / HOTA on the real
  event-metric bundle.
- Skeleton sentinel: synthetic keypoint sequence with degraded
  confidence; sentinel metric is missing-keypoint rate on the
  skeleton extractor.
- Post-verification sentinel: degraded fall score sequence;
  sentinel metric is detection delay.

The cheap place to start each: a sibling
``run_<component>_self_test`` function in this module (or, for
size, a new ``evaluation/self_test/<component>.py``) that mirrors
the same shape: ``Synthetic<Component>`` builder + a deterministic
corruption strategy + a sentinel reading + a
``run_<component>_self_test`` orchestrator returning a
``SelfTestResult``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from data.manifests import ClipRole, FallLabel

from evaluation.contracts import ClipLabel, ClipPrediction, MetricResult, SliceKey
from evaluation.metrics.classification import compute_classification_metrics
from evaluation.not_available import NotAvailable


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfTestConfig:
    """Knobs for the classification self-test.

    Defaults are tuned for a stable, fast, deterministic run on a
    laptop CPU in well under a second — change a knob only with a
    reason; every default below is also covered by a regression
    test. See module docstring for the deterministic-shuffle
    policy.
    """

    #: Number of synthetic clips in the baseline set. Larger
    #: samples stabilise the AUC-ROC readings across shuffles but
    #: slow the run.
    n_samples: int = 200

    #: Class separation in score-space: fall-clip mean score minus
    #: no-fall-clip mean score. Larger → healthier baseline AUC.
    score_class_separation: float = 0.6

    #: Fraction of synthetic clips labelled ``fall``. The remainder
    #: are ``no_fall``. Balanced-ish keeps both AUC computations
    #: well-defined.
    positive_rate: float = 0.4

    #: Number of seeded shuffles to average over. The mean of
    #: ``n_shuffles`` independent shuffles is non-flaky for
    #: ``n_shuffles >= 4`` at these sample sizes.
    n_shuffles: int = 8

    #: Base seed for the synthetic builder. Shuffles use
    #: ``seed_base + 1, seed_base + 2, ...``.
    seed_base: int = 12345

    #: Required gap ``baseline_auc_roc - shuffled_mean``. Below
    #: this margin the self-test concludes "the harness did not
    #: see the corruption".
    margin: float = 0.30

    #: Healthy AUC-ROC band for the baseline reading. The
    #: interval is half-open on the upper end (``[low, high)``)
    #: so an unrealistically perfect reading pushes the test out
    #: of compliance (a constant 1.0 is by definition not what a
    #: finite noisy sample produces — flagging it catches a
    #: scorer that rounds up).
    healthy_auc_range: tuple[float, float] = (0.85, 1.001)

    #: Degraded AUC-ROC band for the shuffled mean. The PRD's
    #: "approaches chance" wording maps to roughly ``[0.4, 0.6]``.
    degraded_auc_range: tuple[float, float] = (0.4, 0.6)

    #: Standard deviation of per-class score noise around the
    #: per-class mean. Smaller → more deterministic AUC-ROC.
    score_noise: float = 0.10

    def shuffled_seeds(self) -> tuple[int, ...]:
        """The deterministic per-shuffle seeds used by the runner.

        Exposed for testability — a reviewer can rebuild a single
        shuffle with this seed and compare.
        """
        return tuple(self.seed_base + i + 1 for i in range(self.n_shuffles))


@dataclass(frozen=True)
class SyntheticClassificationSet:
    """A self-contained synthetic classification set.

    Predictions and labels are aligned by index: ``predictions[i]``
    pairs with ``labels[i]``. Use :func:`shuffle_labels` to
    perturb this alignment without changing the predictions.
    """

    predictions: tuple[ClipPrediction, ...]
    labels: tuple[ClipLabel, ...]

    def __post_init__(self) -> None:
        if len(self.predictions) != len(self.labels):
            raise ValueError(
                f"SyntheticClassificationSet length mismatch: "
                f"{len(self.predictions)} predictions vs {len(self.labels)} labels."
            )


@dataclass(frozen=True)
class SelfTestResult:
    """Outcome of :func:`run_classification_self_test`.

    Carries every numeric reading so a reviewer / CI hook can
    decide on its own whether the harness is healthy — and if not,
    exactly which reading failed.
    """

    baseline_auc_roc: float
    shuffled_auc_roc_values: tuple[float, ...]
    shuffled_auc_roc_mean: float
    degradation_margin: float
    degraded: bool
    passed: bool
    reason: str

    #: Echo knobs needed to reproduce the run. Bundled as a tuple
    #: (no dataclass dependency) so JSON-serialisation stays flat.
    n_shuffles: int = 0
    margin: float = 0.0
    config_seed_base: int = 0
    healthy_auc_range: tuple[float, float] = (0.0, 0.0)
    degraded_auc_range: tuple[float, float] = (0.0, 0.0)


# Type alias for the injectable evaluator. The default is the
# real :func:`compute_classification_metrics`; the broken-harness
# tests substitute a function with the same signature that lies
# about AUC-ROC.
EvaluateMetricsFn = Callable[
    [Sequence[ClipPrediction], Sequence[ClipLabel]],
    list,
]


# ---------------------------------------------------------------------------
# Synthetic-baseline builders
# ---------------------------------------------------------------------------


def build_synthetic_baseline(
    config: SelfTestConfig | None = None,
    *,
    rng: random.Random | None = None,
) -> SyntheticClassificationSet:
    """Build a deterministic, strongly-correlated synthetic baseline.

    ``score`` for a fall clip is sampled near
    ``0.5 + config.score_class_separation / 2``; ``score`` for a
    no-fall clip is sampled near
    ``0.5 - config.score_class_separation / 2``. The two score
    clouds are well-separated, so AUC-ROC against the original
    labels is near 1.

    The synthetic set is a flat list (no per-clip slice tags).
    This is intentional — the self-test exercises the aggregate
    path, and a per-tag slice sweep would only add variance.
    """

    config = config or SelfTestConfig()
    rng = rng or random.Random(config.seed_base)

    n_fall = int(round(config.n_samples * config.positive_rate))
    n_nofall = config.n_samples - n_fall

    # Tag clips deterministically; shuffle the order so the
    # synthetic set is not monotonic in label.
    labels_first = ([FallLabel.FALL] * n_fall) + ([FallLabel.NO_FALL] * n_nofall)
    rng.shuffle(labels_first)

    fall_mean = 0.5 + config.score_class_separation / 2
    nofall_mean = 0.5 - config.score_class_separation / 2
    span = config.score_noise

    preds: list[ClipPrediction] = []
    labs: list[ClipLabel] = []
    for i, label in enumerate(labels_first):
        clip_id = f"synth-{i:04d}"
        mean = fall_mean if label is FallLabel.FALL else nofall_mean
        # Sample + clip to a valid probability.
        raw = rng.gauss(mean, span)
        score = max(0.0, min(1.0, raw))
        preds.append(
            ClipPrediction(
                clip_id=clip_id,
                score=score,
                model_id="self-test-synthetic",
                dataset="urfd",
                role=ClipRole.TRAIN,
            )
        )
        labs.append(
            ClipLabel(
                clip_id=clip_id,
                label=label,
                dataset="urfd",
                role=ClipRole.TRAIN,
                source_path=f"datasets/urfd/{clip_id}.mp4",
            )
        )
    return SyntheticClassificationSet(
        predictions=tuple(preds),
        labels=tuple(labs),
    )


def shuffle_labels(
    synth: SyntheticClassificationSet,
    seed: int,
    *,
    rng: random.Random | None = None,
) -> SyntheticClassificationSet:
    """Deterministically break the ``(score, label)`` bond.

    The predictions list is preserved in its original order.
    The labels are permuted AND each label's ``clip_id`` is
    rewritten to match the prediction at its new index — that
    is what actually corrupts the pairing the metric bundle
    uses.

    Why relabel the ``clip_id``: the metric bundle pairs
    predictions to labels by ``clip_id``. If we merely permute
    the label tuple while preserving each label's ``clip_id``,
    the pairing is unchanged by construction — labels still
    match the prediction whose ``clip_id`` they carry.
    Renaming each label's ``clip_id`` to the prediction at
    its new index assigns the LABEL VALUE of one synthetic
    clip to the SCORE of a different one. That randomisation
    is what the self-test depends on.
    """

    rng = rng or random.Random(seed)
    n = len(synth.labels)
    indices = list(range(n))
    rng.shuffle(indices)

    new_labels: list[ClipLabel] = []
    for new_index, original_index in enumerate(indices):
        original = synth.labels[original_index]
        target_prediction = synth.predictions[new_index]
        new_labels.append(
            ClipLabel(
                clip_id=target_prediction.clip_id,
                label=original.label,
                dataset=original.dataset,
                role=original.role,
                source_path=f"datasets/urfd/{target_prediction.clip_id}.mp4",
                slice_tags=original.slice_tags,
            )
        )
    return SyntheticClassificationSet(
        predictions=synth.predictions,
        labels=tuple(new_labels),
    )


# ---------------------------------------------------------------------------
# Sentinel extraction (aggregate AUC-ROC of the real metric bundle)
# ---------------------------------------------------------------------------


def _aggregate_auc_roc(reports) -> float:
    """Return the aggregate AUC-ROC from a list of :class:`SliceMetricReport`.

    Raises ``ValueError`` if the aggregate is missing or its
    AUC-ROC is :class:`NotAvailable` — both are signals that the
    metric bundle has a real problem (e.g. only one class is
    present).
    """

    for report in reports:
        if report.slice_key is None:
            for metric in report.metrics:
                if metric.name == "auc_roc":
                    if isinstance(metric.value, NotAvailable):
                        raise ValueError(
                            "Aggregate AUC-ROC is NotAvailable; "
                            "self-test requires a numeric reading on the synthetic set."
                        )
                    return float(metric.value)
    raise ValueError("No aggregate report found; metric bundle returned no slice_key=None.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_classification_self_test(
    config: SelfTestConfig | None = None,
    *,
    evaluate_metrics: EvaluateMetricsFn | None = None,
) -> SelfTestResult:
    """Run the classification sentinel end-to-end.

    Args:
        config: Knobs (see :class:`SelfTestConfig`). ``None``
            uses the deterministic defaults.
        evaluate_metrics: Injectable metric function. The default
            is :func:`compute_classification_metrics`. Tests that
            prove the sentinel has teeth substitute a function
            that ignores its inputs and reports AUC-ROC = 0.99 —
            the sentinel must then return ``passed=False``.

    Returns:
        A :class:`SelfTestResult` carrying every relevant number.

    Failure modes:

    - Synthetic baseline fails to construct → ``ValueError`` (the
      builder is internal; this should never happen with valid
      config).
    - Aggregate AUC-ROC is :class:`NotAvailable` → ``ValueError``
      (the synthetic set is constructed to have both classes, so
      this is a harness bug rather than a data bug).
    - The sentinel concludes ``passed=False`` when the harness
      reports ``AUC ≈ 1.0`` for both baseline and shuffled (the
      broken-scorer failure mode).
    """

    config = config or SelfTestConfig()
    evaluate_metrics = evaluate_metrics or compute_classification_metrics

    baseline = build_synthetic_baseline(config)
    baseline_reports = evaluate_metrics(baseline.predictions, baseline.labels)
    baseline_auc = _aggregate_auc_roc(baseline_reports)

    shuffled_values: list[float] = []
    for seed in config.shuffled_seeds():
        shuffled = shuffle_labels(baseline, seed)
        reports = evaluate_metrics(shuffled.predictions, shuffled.labels)
        shuffled_values.append(_aggregate_auc_roc(reports))

    shuffled_mean = sum(shuffled_values) / len(shuffled_values)
    margin_observed = baseline_auc - shuffled_mean

    checks: list[str] = []
    healthy_low, healthy_high = config.healthy_auc_range
    degraded_low, degraded_high = config.degraded_auc_range

    if not (healthy_low <= baseline_auc < healthy_high):
        checks.append(
            f"baseline AUC-ROC {baseline_auc:.4f} is outside the healthy range "
            f"[{healthy_low}, {healthy_high})"
        )
    if not (degraded_low <= shuffled_mean <= degraded_high):
        checks.append(
            f"shuffled mean AUC-ROC {shuffled_mean:.4f} is outside the chance band "
            f"[{degraded_low}, {degraded_high}]"
        )
    if margin_observed < config.margin:
        checks.append(
            f"degradation margin {margin_observed:.4f} is below required {config.margin:.3f}"
        )

    degraded = not checks
    passed = degraded

    reason = "all checks passed" if passed else "; ".join(checks)

    return SelfTestResult(
        baseline_auc_roc=baseline_auc,
        shuffled_auc_roc_values=tuple(shuffled_values),
        shuffled_auc_roc_mean=shuffled_mean,
        degradation_margin=margin_observed,
        degraded=degraded,
        passed=passed,
        reason=reason,
        n_shuffles=config.n_shuffles,
        margin=config.margin,
        config_seed_base=config.seed_base,
        healthy_auc_range=config.healthy_auc_range,
        degraded_auc_range=config.degraded_auc_range,
    )


__all__: tuple[str, ...] = (
    "SelfTestConfig",
    "SyntheticClassificationSet",
    "SelfTestResult",
    "EvaluateMetricsFn",
    "build_synthetic_baseline",
    "run_classification_self_test",
    "shuffle_labels",
)
