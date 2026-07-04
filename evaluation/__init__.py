"""Evaluation harness public surface for Step 1 + Step 2 (Issue 004).

Re-exports contracts, the NotAvailable marker, the execution-context
frozen-tier guard, the result-persistence stub, and the clip-level
classification metrics bundle so notebooks and pipeline code can
``from evaluation import ...`` without reaching into submodules.

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
  operating threshold; honest ``NotAvailable`` semantics.
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
)

