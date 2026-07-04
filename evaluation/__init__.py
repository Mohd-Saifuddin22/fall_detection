"""Evaluation harness public surface for Step 1 + Step 2 + Step 3 + Step 4 (Issue 004).

Re-exports contracts, the NotAvailable marker, the execution-context
frozen-tier guard, the result-persistence stub, the clip-level
classification metrics bundle, the detector-of-the-detector
self-test, and the event-level metric bundle so notebooks and
pipeline code can ``from evaluation import ...`` without reaching
into submodules.

Step 1 establishes:

- Data contracts for clip-level predictions / labels, slice metadata,
  event prediction streams, event ground-truth windows, and metric
  results.
- A machine-readable ``NotAvailable`` type for metrics that cannot be
  computed (carries a reason; distinct from numeric 0.0).
- A frozen-tier read guard that fails closed against the
  ``frozen_unseen_test`` manifest role for training / tuning / unknown
  contexts.
- A versioned, reloadable result-persistence stub rooted at the active
  layout's ``metrics/`` artifact directory.

Step 2 adds:

- A clip-level classification metric bundle (accuracy, precision,
  recall, specificity, F1, AUC-ROC, AUPRC, confusion matrix)
  reported overall and per slice; sklearn-backed; configurable
  operating threshold; honest ``NotAvailable`` semantics including
  degenerate-slice denominators (precision / recall / F1 / specificity
  become NotAvailable when their denominator is zero).

Step 3 adds:

- A detector-of-the-detector self-test (``run_classification_self_test``)
  that proves the Step 2 harness catches a shuffled-label corruption.
  Synthetic baseline + deterministic seeded shuffles + injected
  scorer hook so the test has teeth against a broken / lying harness.

Step 4 adds:

- An event-level metric bundle (event-level recall, false alarms /
  hour, detection delay mean + p95 in frames and seconds,
  cross-dataset event F1) on top of the Step 1 event contracts.
- A pure alert-derivation function (threshold + persistence) and an
  externally-suppliable alert-frame path so a future Pipeline C
  verification engine can feed final alerts without rewriting the
  metrics.
- A component-metric scaffold (mAP / IDF1 / MOTA / HOTA / PCK) that
  returns ``NotAvailable`` rows with precise reasons until real
  detection / tracking / pose ground truth + library integrations
  (sklearn / motmetrics / TrackEval) land.
"""

from evaluation.contracts import (
    ClipLabel,
    ClipPrediction,
    EventGroundTruthWindow,
    EventPredictionStream,
    MetricResult,
    SliceKey,
    SliceTags,
)
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
from evaluation.not_available import (
    NOT_AVAILABLE_JSON_KEY,
    NotAvailable,
    from_dict as not_available_from_dict,
    is_not_available_marker as is_not_available_payload,
)
from evaluation.result_persistence import (
    RESULT_PAYLOAD_FORMAT_VERSION,
    EvalRunMetadata,
    MetricResultPayload,
    MetricResultStore,
    encode_value,
    make_default_metadata,
)
from evaluation.metrics.classification import (
    DEFAULT_SLICE_TAGS,
    DEFAULT_THRESHOLD,
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
from evaluation.self_test import (
    SelfTestConfig,
    SelfTestResult,
    SyntheticClassificationSet,
    build_synthetic_baseline,
    run_classification_self_test,
    shuffle_labels as shuffle_synthetic_labels,
)

__all__: tuple[str, ...] = (
    # contracts
    "SliceKey",
    "SliceTags",
    "ClipPrediction",
    "ClipLabel",
    "EventPredictionStream",
    "EventGroundTruthWindow",
    "MetricResult",
    # NotAvailable
    "NOT_AVAILABLE_JSON_KEY",
    "NotAvailable",
    "not_available_from_dict",
    "is_not_available_payload",
    # execution-context guard
    "ExecutionContext",
    "FROZEN_ALLOWED_CONTEXTS",
    "FrozenAccessError",
    "coerce_execution_context",
    "enforce_no_frozen_in_iterable",
    "frozen_clips_present",
    "get_frozen_clips",
    "is_frozen_allowed",
    "select_clips_for_context",
    # result persistence
    "RESULT_PAYLOAD_FORMAT_VERSION",
    "EvalRunMetadata",
    "MetricResultPayload",
    "MetricResultStore",
    "encode_value",
    "make_default_metadata",
    # classification metrics (Step 2)
    "DEFAULT_THRESHOLD",
    "DEFAULT_SLICE_TAGS",
    "ConfusionMatrix",
    "SupportCounts",
    "SliceMetricReport",
    "compute_classification_metrics",
    # event metrics + component scaffold (Step 4)
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
    # detector-of-the-detector (Step 3)
    "SelfTestConfig",
    "SelfTestResult",
    "SyntheticClassificationSet",
    "build_synthetic_baseline",
    "run_classification_self_test",
    "shuffle_synthetic_labels",
)

