"""Evaluation metrics bundle.

Step 2: clip-level classification metrics.

Submodules:

- :mod:`evaluation.metrics.classification` — accuracy, precision,
  recall, specificity, F1, AUC-ROC, AUPRC, confusion matrix,
  reported overall and by slice. sklearn-backed; threshold-
  configurable; honours :class:`NotAvailable` for unsupportable
  metrics.

Subsequent steps (out of scope here) will add event-level metrics
and the slice aggregator.
"""

from evaluation.metrics.classification import (
    DEFAULT_SLICE_TAGS,
    DEFAULT_THRESHOLD,
    METRIC_NAMES,
    ConfusionMatrix,
    SliceMetricReport,
    SupportCounts,
    compute_classification_metrics,
)

__all__: tuple[str, ...] = (
    "DEFAULT_THRESHOLD",
    "DEFAULT_SLICE_TAGS",
    "METRIC_NAMES",
    "ConfusionMatrix",
    "SupportCounts",
    "SliceMetricReport",
    "compute_classification_metrics",
)
