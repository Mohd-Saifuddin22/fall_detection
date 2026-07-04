"""Evaluation metrics bundle.

Submodules:

- :mod:`evaluation.metrics.classification` — clip-level
  classification metrics (accuracy, precision, recall,
  specificity, F1, AUC-ROC, AUPRC, confusion matrix). sklearn-backed,
  slice-aggregated, honest ``NotAvailable`` semantics on degenerate
  slices. (Step 2)

- :mod:`evaluation.metrics.event` — system/event-level metrics
  (event-level recall, false alarms / hour, detection delay
  mean + p95 in frames and seconds, cross-dataset F1) plus the
  pure alert-derivation function (threshold + persistence) and a
  component-metric scaffold (mAP / IDF1 / MOTA / HOTA / PCK)
  returning ``NotAvailable`` until real ground truth + library
  integrations land. (Step 4)
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
from evaluation.metrics.event import (
    DEFAULT_ALERT_PERSISTENCE,
    DEFAULT_ALERT_THRESHOLD,
    DEFAULT_EVENT_TOLERANCE_FRAMES,
    AlertRule,
    EventMatching,
    EventMetricBundle,
    aggregate_event_metrics_by_dataset,
    compute_component_metrics,
    compute_event_metrics_for_clip,
    compute_event_metrics_for_stream,
    derive_alert_frame_indices,
    match_alerts_to_events,
)

__all__: tuple[str, ...] = (
    # classification (Step 2)
    "DEFAULT_THRESHOLD",
    "DEFAULT_SLICE_TAGS",
    "METRIC_NAMES",
    "ConfusionMatrix",
    "SupportCounts",
    "SliceMetricReport",
    "compute_classification_metrics",
    # event metrics (Step 4)
    "DEFAULT_ALERT_THRESHOLD",
    "DEFAULT_ALERT_PERSISTENCE",
    "DEFAULT_EVENT_TOLERANCE_FRAMES",
    "AlertRule",
    "EventMatching",
    "EventMetricBundle",
    "derive_alert_frame_indices",
    "match_alerts_to_events",
    "compute_event_metrics_for_clip",
    "compute_event_metrics_for_stream",
    "aggregate_event_metrics_by_dataset",
    "compute_component_metrics",
)
