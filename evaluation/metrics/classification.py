"""Clip-level classification metric bundle.

Consumes :class:`evaluation.contracts.ClipPrediction` and
:class:`evaluation.contracts.ClipLabel`; emits
:class:`evaluation.contracts.MetricResult` objects grouped into
:class:`SliceMetricReport` records (one per slice + an aggregate).

Implementation rules (Step 2):

- **All counts come from sklearn.** No hand-rolled TP/FP/TN/FN math.
  Accuracy / precision / recall / F1 from
  :func:`sklearn.metrics.accuracy_score`,
  :func:`sklearn.metrics.precision_score`,
  :func:`sklearn.metrics.recall_score`,
  :func:`sklearn.metrics.f1_score`. Confusion matrix from
  :func:`sklearn.metrics.confusion_matrix`. AUC-ROC from
  :func:`sklearn.metrics.roc_auc_score`. AUPRC from
  :func:`sklearn.metrics.average_precision_score`. Specificity is
  computed from the confusion matrix (sklearn does not expose it as
  a one-liner).
- **Threshold is configurable.** Default 0.5. Count-based metrics
  (accuracy, precision, recall, specificity, F1) use the thresholded
  hard label. AUC-ROC and AUPRC use **raw scores** — they are
  threshold-independent and must not be silently affected by the
  operating-point choice.
- **Slice tags are explicit.** Default slices: ``dataset``,
  ``lighting``, ``occlusion``, ``multi_person``, ``action_confuser``.
  Per-tag slices are emitted for each distinct value present. Clips
  that lack a value for a particular tag are excluded from that
  tag's slices but still contribute to the aggregate report.
  Clips that lack ANY slice metadata at all appear in the aggregate
  only ("aggregate-only bucket").
- **`NotAvailable` is honest.** Empty input / empty slice → every
  metric is :class:`NotAvailable`-marked. AUC-ROC / AUPRC with only
  one class present → :class:`NotAvailable` with reason
  ``"only one class present in slice"``. **Degenerate-slice
  precision / recall / F1** also become :class:`NotAvailable`
  rather than fabricating ``0.0`` via sklearn's ``zero_division``
  knob:

    - precision is :class:`NotAvailable` (``"no predicted positives
      in slice"``) when ``tp + fp == 0`` — the denominator is
      literally undefined.
    - recall is :class:`NotAvailable` (``"no actual positives in
      slice"``) when ``tp + fn == 0``.
    - specificity is :class:`NotAvailable` (``"no negatives in
      slice"``) when ``tn + fp == 0`` — already implemented in
      Step 2 (this fix tightens the phrasing).
    - F1 is :class:`NotAvailable` (``"precision or recall
      undefined"``) when either precision or recall is itself
      :class:`NotAvailable` for the above reasons.

  Numerically-defined degenerate cases (e.g. ``tp=0`` with
  ``fp>0`` → precision ``= 0.0``; ``tp=0`` with ``fn>0`` →
  recall ``= 0.0``) ARE computed normally — they are real
  ``0.0`` readings, not undefined values. The ``NotAvailable``
  marker only fires when the denominator itself is zero.
- **Recall and AUPRC are surfaced clearly** because they are the
  priority metrics for rare fall detection (PRD user story 38:
  false negatives treated as the most costly error).

Public surface
--------------

- :class:`ConfusionMatrix` — typed ``tn`` / ``fp`` / ``fn`` / ``tp``
  + ``support`` shortcut.
- :class:`SupportCounts` — typed ``n_positive`` / ``n_negative`` +
  ``total`` shortcut.
- :class:`SliceMetricReport` — one slice's full report (support,
  confusion matrix, list of :class:`MetricResult` rows).
- :func:`compute_classification_metrics` — main entry point.

The :class:`SliceMetricReport.metric_results` method produces the
flat row list that the Step 1 :class:`MetricResultStore` persists
unchanged, including the confusion matrix entries and support counts
as named rows so a reload produces an identical, structured payload.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from evaluation.contracts import ClipLabel, ClipPrediction, MetricResult, SliceKey
from evaluation.not_available import NotAvailable


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Default operating threshold — a prediction with score >= 0.5
#: is treated as ``fall``; below 0.5 is ``no_fall``. This is the
#: default the PRD cites ("fall probability > 0.80 for 10
#: consecutive frames" describes the post-verification rule; the
#: classifier-level operating point is left at a simple 0.5 here).
DEFAULT_THRESHOLD: float = 0.5

#: Slice tags the bundle aggregates over by default. The PRD
#: requires slicing by lighting, occlusion, multi-person, and
#: action-confusers; ``dataset`` is added so a cross-dataset
#: comparison is one slice lookup away.
DEFAULT_SLICE_TAGS: tuple[str, ...] = (
    "dataset",
    "lighting",
    "occlusion",
    "multi_person",
    "action_confuser",
)

#: Names the metric bundle emits per slice. Excludes support / CM
#: counts (those are added by :meth:`SliceMetricReport.metric_results`).
METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auc_roc",
    "auprc",
)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionMatrix:
    """A 2x2 confusion matrix with named fields.

    Convention follows sklearn's default (``labels=[0, 1]`` mapped to
    our ``FallLabel.NO_FALL=0`` and ``FallLabel.FALL=1``):

        |            | pred=0 (no_fall) | pred=1 (fall) |
        | true=0     | tn              | fp           |
        | true=1     | fn              | tp           |
    """

    tn: int
    fp: int
    fn: int
    tp: int

    def __post_init__(self) -> None:
        for name in ("tn", "fp", "fn", "tp"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"ConfusionMatrix.{name} must be a non-negative int, got {value!r}."
                )

    @property
    def support(self) -> int:
        """Total number of examples summarised by this confusion matrix."""
        return self.tn + self.fp + self.fn + self.tp


@dataclass(frozen=True)
class SupportCounts:
    """Class-count summary for one slice.

    ``n_positive`` is the number of ``FallLabel.FALL`` clips in the
    slice; ``n_negative`` is the number of ``FallLabel.NO_FALL``
    clips. ``total`` is the sum.
    """

    n_positive: int
    n_negative: int

    def __post_init__(self) -> None:
        if not isinstance(self.n_positive, int) or self.n_positive < 0:
            raise ValueError(
                f"SupportCounts.n_positive must be a non-negative int, got {self.n_positive!r}."
            )
        if not isinstance(self.n_negative, int) or self.n_negative < 0:
            raise ValueError(
                f"SupportCounts.n_negative must be a non-negative int, got {self.n_negative!r}."
            )

    @property
    def total(self) -> int:
        return self.n_positive + self.n_negative


@dataclass(frozen=True)
class SliceMetricReport:
    """One slice's classification-metric report.

    A "slice" is either a single ``(tag, value)`` pair (per-tag slices
    like ``lighting=daylight``) or the aggregate over all clips
    (``slice_key=None``). The aggregate is always present.

    Attributes:
        slice_key: The ``SliceKey`` identifying the slice, or
            ``None`` for the aggregate.
        support: Class-count summary (positive / negative).
        confusion_matrix: 2x2 confusion matrix.
        metrics: Per-slice :class:`MetricResult` rows for the
            count-based metrics + AUC-ROC + AUPRC. Does NOT include
            the support / confusion-matrix entries — those are added
            by :meth:`metric_results` for persistence.
    """

    slice_key: SliceKey | None
    support: SupportCounts
    confusion_matrix: ConfusionMatrix
    metrics: tuple[MetricResult, ...]

    def metric_results(self) -> tuple[MetricResult, ...]:
        """All :class:`MetricResult` rows for persistence.

        Appends the confusion matrix entries (``tn``, ``fp``, ``fn``,
        ``tp``) and support entries (``n_positive``, ``n_negative``,
        ``total``) to ``self.metrics`` so the persisted payload is
        complete and reloadable without recomputation.
        """
        sk = self.slice_key
        rows: list[MetricResult] = list(self.metrics)
        rows.extend(_confusion_matrix_metric_rows(self.confusion_matrix, sk))
        rows.extend(_support_metric_rows(self.support, sk))
        return tuple(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_to_hard_label(score: float, threshold: float) -> int:
    """Threshold a raw fall probability into ``1`` (fall) or ``0`` (no_fall)."""
    return 1 if score >= threshold else 0


def _fall_label_to_int(label: object) -> int:
    """Map a :class:`FallLabel` enum member to its 0/1 numeric value.

    Defined here — not on :class:`FallLabel` — so the metric bundle
    does not need to be re-imported just to convert a label, and so
    any label-shaped object that happens to look like ``FallLabel.FALL``
    is rejected if it isn't.
    """
    from data.manifests import FallLabel  # noqa: PLC0415
    if not isinstance(label, FallLabel):
        raise ValueError(
            f"Expected FallLabel, got {type(label).__name__} ({label!r}). "
            f"Only the two project-defined label values are metric-compatible."
        )
    return 1 if label is FallLabel.FALL else 0


def _pair_predictions_and_labels(
    predictions: Sequence[ClipPrediction],
    labels: Sequence[ClipLabel],
) -> list[tuple[ClipPrediction, ClipLabel]]:
    """Match each prediction to its label by ``clip_id``.

    Raises:
        ValueError: on duplicate ``clip_id`` in either input or on
            any unpaired entry. Eval code is expected to be self-
            consistent — silently dropping a prediction would
            fabricate a metric.
    """
    # Check both sides for duplicates + collect IDs by side. Doing
    # both checks up-front lets us report the precise source of a
    # duplicate (predictions vs labels) instead of two errors
    # stacked on top of each other.
    pred_ids: dict[str, int] = {}
    for prediction in predictions:
        pred_ids[prediction.clip_id] = pred_ids.get(prediction.clip_id, 0) + 1
    for clip_id, count in pred_ids.items():
        if count > 1:
            raise ValueError(
                f"Duplicate clip_id {clip_id!r} in predictions; eval expects exactly one per clip."
            )

    label_by_id: dict[str, ClipLabel] = {}
    for label in labels:
        if label.clip_id in label_by_id:
            raise ValueError(
                f"Duplicate clip_id {label.clip_id!r} in labels; eval expects exactly one per clip."
            )
        label_by_id[label.clip_id] = label

    pairs: list[tuple[ClipPrediction, ClipLabel]] = []
    for prediction in predictions:
        if prediction.clip_id not in label_by_id:
            raise ValueError(
                f"prediction for clip_id {prediction.clip_id!r} has no matching label."
            )
        pairs.append((prediction, label_by_id[prediction.clip_id]))

    # Labels without a matching prediction are also a bug — a
    # labelled clip the model never scored will silently inflate
    # support counts on per-slice reports without contributing to
    # any metric.
    prediction_ids = {prediction.clip_id for prediction in predictions}
    for label in labels:
        if label.clip_id not in prediction_ids:
            raise ValueError(
                f"label for clip_id {label.clip_id!r} has no matching prediction."
            )

    return pairs


def _sliceable_tags(slice_tags: object, configured: tuple[str, ...]) -> dict[str, str]:
    """Pick the configured tags from a ``SliceTags`` (or similar) object.

    Clips that lack values for a particular tag do not appear in
    that tag's slices — that's how the "missing metadata lands in
    aggregate-only" behaviour falls out naturally.
    """
    out: dict[str, str] = {}
    if slice_tags is None:
        return out
    for tag in configured:
        value = getattr(slice_tags, tag, None)
        if value is None:
            continue
        # ``multi_person`` is rendered as "true"/"false" elsewhere;
        # we don't re-stringify here because the caller may have
        # pre-rendered.
        out[tag] = str(value)
    return out


# ---------------------------------------------------------------------------
# Confusion matrix + support counts (sklearn-backed)
# ---------------------------------------------------------------------------


def _compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> ConfusionMatrix:
    """Compute a 2x2 confusion matrix using sklearn.

    ``labels=[0, 1]`` is passed explicitly so the layout is
    deterministic regardless of which classes happen to be present.
    The output is reordered to :class:`ConfusionMatrix` (tn, fp, fn,
    tp) — sklearn returns ``[[tn, fp], [fn, tp]]`` when labels are
    in numeric order, so the indexing is direct.
    """
    from sklearn.metrics import confusion_matrix  # noqa: PLC0415
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel().tolist())
    return ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp)


def _compute_support_counts(y_true: Sequence[int]) -> SupportCounts:
    n_positive = sum(1 for v in y_true if v == 1)
    n_negative = sum(1 for v in y_true if v == 0)
    return SupportCounts(n_positive=n_positive, n_negative=n_negative)


# ---------------------------------------------------------------------------
# Per-metric emission (degenerate-slice honesty lives here)
# ---------------------------------------------------------------------------


def _emit_metric(
    name: str,
    value: float | int | NotAvailable,
    slice_key: SliceKey | None,
    *,
    higher_is_better: bool = True,
    notes: str | None = None,
) -> MetricResult:
    return MetricResult(
        name=name,
        value=value,  # type: ignore[arg-type]
        slice_key=slice_key,
        higher_is_better=higher_is_better,
        notes=notes,
    )


def _emit_metrics_for_slice(
    cm: ConfusionMatrix,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    slice_key: SliceKey | None,
) -> tuple[MetricResult, ...]:
    """Emit the seven primary :class:`MetricResult` rows for one slice.

    Implements the **degenerate-slice honesty contract**:

    - accuracy: numeric when the slice is non-empty.
    - precision: :class:`NotAvailable` when ``tp + fp == 0``;
      otherwise sklearn-backed numeric.
    - recall: :class:`NotAvailable` when ``tp + fn == 0``;
      otherwise sklearn-backed numeric.
    - specificity: :class:`NotAvailable` when ``tn + fp == 0``;
      otherwise ``tn / (tn + fp)``.
    - F1: :class:`NotAvailable` when precision OR recall is
      :class:`NotAvailable`; otherwise sklearn-backed numeric.
    - AUC-ROC / AUPRC: :class:`NotAvailable` when only one class is
      present in ``y_true``; otherwise sklearn-backed numeric.

    sklearn's ``zero_division`` knob is NOT used to fabricate ``0.0``
    for undefined denominators — that's the whole point of this
    function. We rely on sklearn only when the metric is
    mathematically defined (positive denominator).
    """
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    tp, fp, fn, tn = cm.tp, cm.fp, cm.fn, cm.tn
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    # accuracy — well-defined for any non-empty slice.
    accuracy_value = float(accuracy_score(y_true_list, y_pred_list))

    # precision — undefined when the model never predicts positive.
    # We never pass ``zero_division``; either the call returns the
    # real value (positive denominator) or we surface NotAvailable.
    if tp + fp == 0:
        precision_value: float | NotAvailable = NotAvailable(
            reason="no predicted positives in slice",
            metric_name="precision",
        )
    else:
        precision_value = float(precision_score(y_true_list, y_pred_list))

    # recall — undefined when no actual positives exist.
    if tp + fn == 0:
        recall_value: float | NotAvailable = NotAvailable(
            reason="no actual positives in slice",
            metric_name="recall",
        )
    else:
        recall_value = float(recall_score(y_true_list, y_pred_list))

    # F1 — undefined when precision OR recall is undefined.
    if isinstance(precision_value, NotAvailable) or isinstance(recall_value, NotAvailable):
        f1_value: float | NotAvailable = NotAvailable(
            reason="precision or recall undefined",
            metric_name="f1",
        )
    else:
        f1_value = float(f1_score(y_true_list, y_pred_list))

    # specificity — derived from the confusion matrix; sklearn has
    # no direct primitive. Same honesty rule.
    if tn + fp == 0:
        specificity_value: float | NotAvailable = NotAvailable(
            reason="no negatives in slice",
            metric_name="specificity",
        )
    else:
        specificity_value = tn / (tn + fp)

    # AUC-ROC / AUPRC — NotAvailable when only one class is present.
    auc_roc_value = _compute_auc_roc(y_true, y_score, slice_key)
    auprc_value = _compute_auprc(y_true, y_score, slice_key)

    return (
        _emit_metric("accuracy", accuracy_value, slice_key, higher_is_better=True),
        _emit_metric("precision", precision_value, slice_key, higher_is_better=True),
        _emit_metric("recall", recall_value, slice_key, higher_is_better=True),
        _emit_metric("specificity", specificity_value, slice_key, higher_is_better=True),
        _emit_metric("f1", f1_value, slice_key, higher_is_better=True),
        _emit_metric("auc_roc", auc_roc_value, slice_key, higher_is_better=True),
        _emit_metric("auprc", auprc_value, slice_key, higher_is_better=True),
    )


def _compute_auc_roc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    slice_key: SliceKey | None,
) -> float | NotAvailable:
    """AUC-ROC from raw scores; :class:`NotAvailable` with one class only.

    We compute the class balance up-front rather than relying on
    sklearn's exception — sklearn's behaviour for one-class inputs
    has drifted across versions (some emit warnings + return 0.0
    instead of raising), and depending on the exception makes the
    outcome version-sensitive. The dedicated check here makes the
    semantics explicit and stable.
    """
    unique_classes = set(y_true)
    if len(unique_classes) < 2:
        return NotAvailable(
            reason="only one class present in slice",
            metric_name="auc_roc",
        )
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    return float(roc_auc_score(y_true, y_score))


def _compute_auprc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    slice_key: SliceKey | None,
) -> float | NotAvailable:
    """AUPRC from raw scores; :class:`NotAvailable` with one class only.

    Same rationale as :func:`_compute_auc_roc` — explicit class-balance
    check rather than relying on sklearn's potentially-warn-only
    behaviour.
    """
    unique_classes = set(y_true)
    if len(unique_classes) < 2:
        return NotAvailable(
            reason="only one class present in slice",
            metric_name="auprc",
        )
    from sklearn.metrics import average_precision_score  # noqa: PLC0415
    return float(average_precision_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Confusion-matrix / support rows for persistence
# ---------------------------------------------------------------------------


def _confusion_matrix_metric_rows(
    cm: ConfusionMatrix,
    slice_key: SliceKey | None,
) -> tuple[MetricResult, ...]:
    return (
        MetricResult(name="tn", value=float(cm.tn), slice_key=slice_key, higher_is_better=True),
        MetricResult(name="fp", value=float(cm.fp), slice_key=slice_key, higher_is_better=False),
        MetricResult(name="fn", value=float(cm.fn), slice_key=slice_key, higher_is_better=False),
        MetricResult(name="tp", value=float(cm.tp), slice_key=slice_key, higher_is_better=True),
    )


def _support_metric_rows(
    support: SupportCounts,
    slice_key: SliceKey | None,
) -> tuple[MetricResult, ...]:
    return (
        MetricResult(name="n_positive", value=float(support.n_positive), slice_key=slice_key),
        MetricResult(name="n_negative", value=float(support.n_negative), slice_key=slice_key),
        MetricResult(name="total", value=float(support.total), slice_key=slice_key),
    )


# ---------------------------------------------------------------------------
# One-slice computation (used by :func:`compute_classification_metrics`)
# ---------------------------------------------------------------------------


def _thresholded_pairs(
    local_pairs: Sequence[tuple[ClipPrediction, ClipLabel]],
    threshold: float,
) -> tuple[list[int], list[int], list[float]]:
    """Apply threshold + label-encoding once for an entire slice.

    Returns ``(y_true, y_pred, y_score)``. ``y_pred`` is the
    thresholded hard label; ``y_score`` is the raw fall probability
    (used by AUC-ROC / AUPRC, which never consult the threshold).
    """
    y_t: list[int] = []
    y_p: list[int] = []
    y_s: list[float] = []
    for prediction, label in local_pairs:
        y_t.append(_fall_label_to_int(label.label))
        y_p.append(_score_to_hard_label(float(prediction.score), threshold))
        y_s.append(float(prediction.score))
    return y_t, y_p, y_s


def _empty_report(slice_key: SliceKey | None) -> SliceMetricReport:
    """Build a slice report for an empty input.

    All seven primary metrics are :class:`NotAvailable` with reason
    ``"empty slice"``. ``support`` / ``confusion_matrix`` are still
    emitted as zero-valued typed objects so a caller iterating a list
    of reports always sees a consistent shape.
    """
    empty_na = NotAvailable(reason="empty slice", metric_name=None)
    return SliceMetricReport(
        slice_key=slice_key,
        support=SupportCounts(n_positive=0, n_negative=0),
        confusion_matrix=ConfusionMatrix(tn=0, fp=0, fn=0, tp=0),
        metrics=tuple(
            _emit_metric(name, empty_na, slice_key, higher_is_better=higher)
            for name, higher in (
                ("accuracy", True),
                ("precision", True),
                ("recall", True),
                ("specificity", True),
                ("f1", True),
                ("auc_roc", True),
                ("auprc", True),
            )
        ),
    )


def _report(
    local_pairs: Sequence[tuple[ClipPrediction, ClipLabel]],
    slice_key: SliceKey | None,
    threshold: float,
) -> SliceMetricReport:
    """Build a :class:`SliceMetricReport` for one slice.

    Splits the empty path (every metric :class:`NotAvailable`) from
    the non-empty path. The non-empty path delegates to
    :func:`_emit_metrics_for_slice`, which carries the degenerate-
    slice honesty contract.
    """
    if not local_pairs:
        return _empty_report(slice_key)
    y_t, y_p, y_s = _thresholded_pairs(local_pairs, threshold)
    support = _compute_support_counts(y_t)
    cm = _compute_confusion_matrix(y_t, y_p)
    metrics = _emit_metrics_for_slice(cm, y_t, y_p, y_s, slice_key)
    return SliceMetricReport(
        slice_key=slice_key, support=support, confusion_matrix=cm, metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    predictions: Iterable[ClipPrediction],
    labels: Iterable[ClipLabel],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    slice_tags: Sequence[str] = DEFAULT_SLICE_TAGS,
) -> list[SliceMetricReport]:
    """Compute clip-level classification metrics overall and per slice.

    Args:
        predictions: Model outputs, one per clip. ``score`` is the
            raw fall probability.
        labels: Ground-truth labels, one per clip.
        threshold: Operating point for converting scores into hard
            labels (default :data:`DEFAULT_THRESHOLD`, 0.5).
            Count-based metrics use ``score >= threshold`` as the
            predicted "fall" label. AUC-ROC and AUPRC are computed
            from raw scores and ignore this argument.
        slice_tags: Slice-axis names to aggregate over (default:
            the PRD-mandated five — ``dataset``, ``lighting``,
            ``occlusion``, ``multi_person``, ``action_confuser``).

    Returns:
        A list of :class:`SliceMetricReport` records: the aggregate
        (``slice_key=None``) first, then one report per emitted
        slice. The list is ordered deterministically:
        ``slice_key.tag`` alphabetical, then ``slice_key.value``
        alphabetical. Returns an empty list if both inputs are empty.

    Behavior:

    - Predictions and labels are paired by ``clip_id``. A prediction
      with no matching label (or vice versa) raises ``ValueError`` —
      silently dropping a prediction would fabricate a metric.
    - The aggregate and every emitted slice carry a
      :class:`ConfusionMatrix` and :class:`SupportCounts`. Empty
      slices carry zero-valued objects and ``NotAvailable``-marked
      metrics so downstream iteration sees a consistent shape.
    - **Degenerate-slice semantics** (the Step 2 fix): precision /
      recall / F1 / specificity become :class:`NotAvailable` when
      their respective denominators are zero, rather than
      fabricating ``0.0`` via sklearn's ``zero_division`` knob.
      AUC-ROC / AUPRC remain :class:`NotAvailable` when only one
      class is present. See :mod:`evaluation.metrics.classification`
      module docstring for the full table.
    """
    predictions_list = list(predictions)
    labels_list = list(labels)
    pairs = _pair_predictions_and_labels(predictions_list, labels_list)

    if not pairs:
        return []

    # Group pairs by (tag, value) for each requested slice tag.
    # A clip can contribute to multiple slices (one per tag), and
    # is excluded from a tag's slices when it lacks a value.
    by_tag_value: dict[str, dict[str, list[tuple[ClipPrediction, ClipLabel]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for prediction, label in pairs:
        tags_from_prediction = _sliceable_tags(prediction.slice_tags, tuple(slice_tags))
        tags_from_label = _sliceable_tags(label.slice_tags, tuple(slice_tags))
        # Prefer the prediction's slice_tags when both are present —
        # they're usually identical, but in case a caller passed a
        # richer label, the prediction wins (it's the row that drives
        # the metric).
        merged = {**tags_from_label, **tags_from_prediction}
        for tag, value in merged.items():
            by_tag_value[tag][value].append((prediction, label))

    reports: list[SliceMetricReport] = [_report(pairs, None, threshold)]

    # Emit per-tag slices in alphabetical order so the output is
    # deterministic and trivially diffable across runs — independent
    # of the order in which ``slice_tags`` was passed.
    for tag in sorted(by_tag_value.keys()):
        for value in sorted(by_tag_value[tag].keys()):
            slice_pairs = by_tag_value[tag][value]
            if not slice_pairs:
                continue
            reports.append(
                _report(slice_pairs, SliceKey(tag=tag, value=value), threshold)
            )

    return reports


__all__: tuple[str, ...] = (
    "DEFAULT_THRESHOLD",
    "DEFAULT_SLICE_TAGS",
    "METRIC_NAMES",
    "ConfusionMatrix",
    "SupportCounts",
    "SliceMetricReport",
    "compute_classification_metrics",
)
