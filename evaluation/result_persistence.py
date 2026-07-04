"""Result-persistence stub for evaluation runs.

Writes **versioned** metric-result payloads under the active layout's
``metrics/`` artifact root. The on-disk shape is a directory per run:

    <metrics_root>/<run_id>/
        results.json          # the structured, reloadable payload
        summary.txt           # one-line human-readable summary (for grep)

A "result payload" is intentionally narrow on Step 1: a format-
version string + run metadata + a list of :class:`MetricResult`
rows. Computation of classification and event metrics lands in
Step 2+; this module is the contract that those metrics will land
in.

Why versioned
-------------

A version string at the root of every payload makes future schema
changes survivable. If the payload format gains a new required field
(e.g. prediction-source provenance), the loader can branch on
``format_version`` instead of refusing older payloads or silently
dropping fields.

Why a stub
----------

Step 1 builds the container only — no metric computation, no
aggregation, no slicing logic. The unit tests therefore exercise
the write/read round-trip with hand-built :class:`MetricResult`
lists, including the :class:`NotAvailable` shape. Step 2+ replaces
the hand-built lists with metric outputs from ``metrics/classification.py``
and ``metrics/event.py``.

Why under ``layout.metrics``, not hardcoded Drive
-----------------------------------------------

The result-persistence layer takes a path. The notebooks and
pipeline code pass ``layout.metrics`` — never a literal Google
Drive path. This keeps the persistence surface testable on any
host, and lets a developer point the artifact root at a temp
directory for debugging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from evaluation.contracts import MetricResult, SliceKey
from evaluation.execution_context import (
    ExecutionContext,
    coerce_execution_context,
)
from evaluation.not_available import (
    NOT_AVAILABLE_JSON_KEY,
    NotAvailable,
    from_dict as na_from_dict,
    is_not_available_marker,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Format version of the persisted payload. Bump when the payload
# schema changes in a way that's not transparently loadable by
# older code; the loader's ``format_version`` check then forces
# the consumer to upgrade.
RESULT_PAYLOAD_FORMAT_VERSION: str = "1.0"

# Filename of the per-run payload, and the sidecar summary.
RESULTS_FILENAME: str = "results.json"
SUMMARY_FILENAME: str = "summary.txt"


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRunMetadata:
    """Provenance for one evaluation run.

    Carried verbatim into the persisted payload so any future reader
    (a reviewer, a follow-up run, a comparison table) knows exactly
    which model produced the numbers and on which context.
    """

    run_id: str
    model_id: str
    created_at: str  # ISO-8601 UTC, e.g. "2026-07-04T12:00:00+00:00"
    context: str     # :class:`ExecutionContext` value
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("EvalRunMetadata.run_id must be a non-empty string.")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("EvalRunMetadata.model_id must be a non-empty string.")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("EvalRunMetadata.created_at must be a non-empty ISO-8601 string.")
        if not isinstance(self.context, str) or not self.context:
            raise ValueError("EvalRunMetadata.context must be a non-empty context label.")


def make_default_metadata(
    run_id: str,
    model_id: str,
    *,
    context: ExecutionContext | str | None = None,
    notes: str | None = None,
) -> EvalRunMetadata:
    """Build :class:`EvalRunMetadata` with the current UTC timestamp.

    Provided as a convenience so callers don't have to import
    ``datetime`` + ``timezone`` just to stamp the run.
    """
    resolved = coerce_execution_context(context)
    return EvalRunMetadata(
        run_id=run_id,
        model_id=model_id,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        context=resolved.value,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResultPayload:
    """The versioned result payload persisted to disk.

    ``metrics`` is a tuple (not a list) so the payload is hashable
    and cannot be mutated after construction.
    """

    format_version: str
    metadata: EvalRunMetadata
    metrics: tuple[MetricResult, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-friendly dict.

        ``SliceKey`` and :class:`NotAvailable` are rendered in their
        stable on-disk shapes (see :func:`encode_value`).
        """
        return {
            "format_version": self.format_version,
            "metadata": {
                "run_id": self.metadata.run_id,
                "model_id": self.metadata.model_id,
                "created_at": self.metadata.created_at,
                "context": self.metadata.context,
                "notes": self.metadata.notes,
            },
            "metrics": [_serialise_metric(m) for m in self.metrics],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "MetricResultPayload":
        """Inverse of :meth:`to_dict`.

        Strict on shape: a malformed payload raises ``ValueError``
        rather than guessing, so a future schema bump catches old
        consumers immediately rather than silently losing fields.

        Raises:
            ValueError: if ``payload`` is not a dict, ``format_version``
                is missing, ``metadata`` is missing a required field,
                or a metric record is malformed.
        """
        if not isinstance(payload, dict):
            raise ValueError(
                f"MetricResultPayload.from_dict expects a dict, got {type(payload).__name__}."
            )

        format_version = payload.get("format_version")
        if not isinstance(format_version, str) or not format_version:
            raise ValueError("MetricResultPayload missing 'format_version' string.")
        if format_version != RESULT_PAYLOAD_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported metric-result format_version {format_version!r}; "
                f"this loader understands {RESULT_PAYLOAD_FORMAT_VERSION!r}."
            )

        raw_metadata = payload.get("metadata")
        if not isinstance(raw_metadata, dict):
            raise ValueError("MetricResultPayload 'metadata' must be a dict.")
        metadata = EvalRunMetadata(
            run_id=str(raw_metadata.get("run_id", "")),
            model_id=str(raw_metadata.get("model_id", "")),
            created_at=str(raw_metadata.get("created_at", "")),
            context=str(raw_metadata.get("context", "")),
            notes=_optional_str(raw_metadata.get("notes")),
        )

        raw_metrics = payload.get("metrics", [])
        if not isinstance(raw_metrics, list):
            raise ValueError("MetricResultPayload 'metrics' must be a list.")

        metrics: list[MetricResult] = []
        for index, raw in enumerate(raw_metrics):
            metrics.append(_deserialise_metric(raw, index))

        return cls(
            format_version=format_version,
            metadata=metadata,
            metrics=tuple(metrics),
        )


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _serialise_metric(metric: MetricResult) -> dict[str, object]:
    """Render one :class:`MetricResult` as a JSON-friendly dict."""
    value = encode_value(metric.value, metric_name=metric.name)
    out: dict[str, object] = {
        "name": metric.name,
        "value": value,
        "higher_is_better": metric.higher_is_better,
    }
    if metric.slice_key is not None:
        out["slice_key"] = metric.slice_key.to_dict()
    if metric.notes is not None:
        out["notes"] = metric.notes
    return out


def _deserialise_metric(raw: object, index: int) -> MetricResult:
    """Inverse of :func:`_serialise_metric`. Strict on shape."""
    if not isinstance(raw, dict):
        raise ValueError(f"metrics[{index}] must be a dict, got {type(raw).__name__}.")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"metrics[{index}].name must be a non-empty string.")
    raw_value = raw.get("value")
    if raw_value is None:
        raise ValueError(f"metrics[{index}].value is missing.")
    higher_is_better = raw.get("higher_is_better", True)
    if not isinstance(higher_is_better, bool):
        raise ValueError(f"metrics[{index}].higher_is_better must be a bool.")

    if is_not_available_marker(raw_value):
        marker = na_from_dict(raw_value)
        # Preserve the original metric_name on the marker so the
        # post-load ``MetricResult`` is self-describing.
        value: object = NotAvailable(reason=marker.reason, metric_name=marker.metric_name or name)
    elif isinstance(raw_value, bool):
        # Belt-and-braces: a JSON boolean is technically also a JSON
        # "number"-shaped thing in some encoders, but the spec says
        # ``false`` here would be very unexpected — refuse explicitly.
        raise ValueError(f"metrics[{index}].value must be numeric or a NotAvailable marker.")
    elif isinstance(raw_value, (int, float)):
        value = float(raw_value)
    else:
        raise ValueError(
            f"metrics[{index}].value has unexpected type {type(raw_value).__name__}; "
            f"expected numeric or a NotAvailable marker."
        )

    slice_key_raw = raw.get("slice_key")
    slice_key: SliceKey | None = None
    if slice_key_raw is not None:
        slice_key = SliceKey.from_dict(slice_key_raw)

    return MetricResult(
        name=name,
        value=value,  # type: ignore[arg-type]
        slice_key=slice_key,
        higher_is_better=higher_is_better,
        notes=_optional_str(raw.get("notes")),
    )


def encode_value(value: object, metric_name: str | None = None) -> object:
    """Render a single metric value (numeric or NotAvailable) for JSON.

    Numeric values pass through unchanged. :class:`NotAvailable` is
    rendered as its marker dict. Everything else raises — the
    payload schema does not permit other shapes.

    The ``metric_name`` kwarg is advisory: when supplied AND the
    NotAvailable has no metric_name of its own, the marker is
    stamped so a payload consumer that has lost the surrounding
    MetricResult ordering can still pair marker → metric name from
    the marker alone. The stamp is reversible — see
    :meth:`MetricResultPayload.from_dict`.
    """
    if isinstance(value, NotAvailable):
        if metric_name is not None and value.metric_name is None:
            # Stamp the metric_name onto the marker so a reader of
            # the payload sees the pairing even if it didn't preserve
            # the surrounding MetricResult record ordering.
            stamped = NotAvailable(reason=value.reason, metric_name=metric_name)
            return stamped.to_dict()
        return value.to_dict()
    if isinstance(value, bool):
        # Booleans are not legitimate metric values; refuse rather
        # than round-trip as numeric and silently coerce.
        raise ValueError(
            f"encode_value: metric value must be numeric or NotAvailable, got bool {value!r}."
        )
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(
        f"encode_value: metric value must be numeric or NotAvailable, got {type(value).__name__}."
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MetricResultStore:
    """Persistence handle for one evaluation run.

    Constructed with the directory the metrics will live under —
    typically ``layout.metrics`` from :mod:`colab.data_mode`, never a
    hardcoded Drive path. The store does not assume any particular
    layout; the caller owns the file arrangement.

    Overwrite behaviour (failure mode on re-save)
    ---------------------------------------------

    :meth:`save` refuses to overwrite an existing run by default.
    The motivation:

    - A ``run_id`` is the result's identity on disk; overwriting
      silently means a re-run of the same id loses the prior result.
    - ``save`` is the natural seam to capture *new* evaluation work.
      A caller that genuinely wants to replace an existing run must
      pass ``overwrite=True`` so the replacement is explicit in the
      call site.
    - Test code that intentionally re-saves the same id (e.g. to
      round-trip its own writes, or to test idempotence under the
      same metadata) is the only legitimate reason to override; the
      explicit kwarg flags that intent.

    The store never silently clears or rotates other runs. To remove
    a run, delete its directory directly via :meth:`run_dir` —
    ``shutil.rmtree(store.run_dir("..."))`` — there's no special
    delete method because a misclick that drops results is worse
    than one that requires a few lines of shutil to undo.
    """

    def __init__(self, metrics_root: Path | str) -> None:
        self._root = Path(metrics_root)

    @property
    def root(self) -> Path:
        """The root directory payload files are written under."""
        return self._root

    def run_dir(self, run_id: str) -> Path:
        """The directory one run's payload lands in."""
        return self._root / run_id

    def ensure(self) -> None:
        """Create the metrics root if it does not yet exist. Idempotent."""
        self._root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        metadata: EvalRunMetadata,
        metrics: Iterable[MetricResult],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist one run's payload.

        Args:
            metadata: Run provenance. The ``run_id`` field determines
                the on-disk directory name.
            metrics: The metric results to persist. May include
                :class:`NotAvailable` markers.
            overwrite: When ``True``, an existing run directory for
                ``metadata.run_id`` is replaced. When ``False``
                (default), an existing run triggers
                :class:`FileExistsError` so a re-run of the same id
                does not silently drop the prior result.

        Returns:
            The directory containing the persisted payload.

        Raises:
            ValueError: if ``metadata.run_id`` is empty / contains a
                path separator (would let a run_id escape its
                directory); or if the metrics contain invalid values.
            FileExistsError: if a run directory already exists for
                ``metadata.run_id`` and ``overwrite`` is ``False``.
        """
        _validate_run_id(metadata.run_id)
        self.ensure()
        run_dir = self.run_dir(metadata.run_id)
        if run_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Run directory {run_dir!s} already exists; refusing to overwrite. "
                    f"Pass overwrite=True to replace, or use a unique run_id."
                )
            # Caller explicitly opted in — clean the existing files
            # but DO NOT touch siblings. Reproducible payload bytes
            # depend on the run_dir being freshly populated, so we
            # remove the named files only.
            for stale in (run_dir / RESULTS_FILENAME, run_dir / SUMMARY_FILENAME):
                if stale.exists():
                    stale.unlink()

        run_dir.mkdir(parents=True, exist_ok=True)

        payload = MetricResultPayload(
            format_version=RESULT_PAYLOAD_FORMAT_VERSION,
            metadata=metadata,
            metrics=tuple(metrics),
        )
        payload_dict = payload.to_dict()

        results_path = run_dir / RESULTS_FILENAME
        results_path.write_text(
            json.dumps(payload_dict, indent=2, sort_keys=False, ensure_ascii=False),
            encoding="utf-8",
        )
        summary_path = run_dir / SUMMARY_FILENAME
        summary_path.write_text(
            _summary_line(payload),
            encoding="utf-8",
        )
        return run_dir

    def load(self, run_id: str) -> MetricResultPayload:
        """Read back a previously-saved payload.

        Strict on shape so a corrupted / schema-bumped file surfaces
        immediately rather than silently returning a degraded view.
        """
        run_dir = self.run_dir(run_id)
        results_path = run_dir / RESULTS_FILENAME
        if not results_path.exists():
            raise FileNotFoundError(
                f"No metric-results file at {results_path!s}; expected one written by save()."
            )
        try:
            raw = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Metric-results file at {results_path!s} is not valid JSON: {exc}."
            ) from exc
        return MetricResultPayload.from_dict(raw)


def _validate_run_id(run_id: str) -> None:
    """A run_id becomes a directory name; refuse anything that would escape."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string.")
    if run_id in {".", ".."}:
        raise ValueError(f"run_id {run_id!r} is reserved.")
    if "/" in run_id or "\\" in run_id:
        raise ValueError(
            f"run_id {run_id!r} must not contain a path separator; "
            f"passed-in run_ids are filename segments, not paths."
        )


def _summary_line(payload: MetricResultPayload) -> str:
    """One-line grep-friendly summary written next to ``results.json``."""
    num_metrics = len(payload.metrics)
    num_available = sum(1 for m in payload.metrics if m.is_available())
    head = (
        f"run_id={payload.metadata.run_id} "
        f"model_id={payload.metadata.model_id} "
        f"context={payload.metadata.context} "
        f"created_at={payload.metadata.created_at} "
        f"format_version={payload.format_version} "
        f"metrics={num_available}/{num_metrics}"
    )
    if payload.metadata.notes:
        head = f"{head} notes={payload.metadata.notes}"
    return head + "\n"


__all__: tuple[str, ...] = (
    "RESULT_PAYLOAD_FORMAT_VERSION",
    "RESULTS_FILENAME",
    "SUMMARY_FILENAME",
    "EvalRunMetadata",
    "MetricResultPayload",
    "MetricResultStore",
    "make_default_metadata",
    "encode_value",
)
