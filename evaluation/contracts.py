"""Evaluation-harness data contracts.

Public surface:

    - :class:`SliceKey` — (tag, value) label for one evaluation slice.
    - :class:`SliceTags` — all slice tags from a clip, with ``from_clip`` helper.
    - :class:`ClipPrediction` — model output for one clip.
    - :class:`ClipLabel` — ground-truth label for one clip.
    - :class:`EventPredictionStream` — frame-indexed fall score stream for one clip.
    - :class:`EventGroundTruthWindow` — temporal ground-truth window.
    - :class:`MetricResult` — one computed metric (number or :class:`NotAvailable`).

These contracts are deliberately **schema-only**. They describe what
flows through the evaluation pipeline — not how the metrics are
computed. Step 1 (this issue) is to lock the shapes so downstream
metrics code, frozen-tier guards, and persistence work against a
shared vocabulary. Implementation of full classification and event
metrics lands in Step 2+ and is explicitly out of scope here.

All contracts are keyword-only ``frozen=True`` dataclasses so callers
cannot accidentally swap fields positionally, and cannot mutate a
value out from under the metric code.

Naming convention: every contract carries a ``clip_id`` (``str``) and
a ``model_id`` / ``model_label`` where applicable so two model runs
over the same clip are line-item distinguishable in the persisted
payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from data.manifests import ClipRecord, ClipRole, FallLabel


# ---------------------------------------------------------------------------
# Slice metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceKey:
    """One (tag, value) slice label.

    Used both as an in-memory label and as the persisted ``slice_key``
    on :class:`MetricResult`. Examples:

        SliceKey("lighting", "daylight")
        SliceKey("dataset", "urfd")
        SliceKey("role", "validate")
        SliceKey("action_confuser", "sitting")
    """

    tag: str
    value: str

    def __post_init__(self) -> None:
        # Validation is structural, not semantic — slice tags are
        # free-form strings in the manifest, but here we harden the
        # contract so an empty tag/value never makes it into a result.
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError(f"SliceKey.tag must be a non-empty string, got {self.tag!r}.")
        if not isinstance(self.value, str) or not self.value:
            raise ValueError(f"SliceKey.value must be a non-empty string, got {self.value!r}.")

    def label(self) -> str:
        """Human-readable ``"tag=value"`` form, used in CSV / log lines."""
        return f"{self.tag}={self.value}"

    def to_dict(self) -> dict[str, str]:
        return {"tag": self.tag, "value": self.value}

    @classmethod
    def from_dict(cls, payload: object) -> "SliceKey":
        if not isinstance(payload, dict):
            raise ValueError(f"SliceKey.from_dict expects a dict, got {type(payload).__name__}.")
        try:
            return cls(tag=str(payload["tag"]), value=str(payload["value"]))
        except KeyError as exc:
            raise ValueError(f"SliceKey.from_dict missing key {exc!s}.") from exc


@dataclass(frozen=True)
class SliceTags:
    """All slice tags from a clip, carrying the manifest's optional fields.

    Every field is nullable so :class:`SliceTags` mirrors the manifest
    shape directly. ``from_clip`` builds one from a :class:`ClipRecord`
    so eval code never has to reach into the manifest by hand.
    """

    lighting: str | None = None
    occlusion: str | None = None
    multi_person: bool | None = None
    action_confuser: str | None = None

    @classmethod
    def from_clip(cls, clip: ClipRecord) -> "SliceTags":
        return cls(
            lighting=clip.lighting,
            occlusion=clip.occlusion,
            multi_person=clip.multi_person,
            action_confuser=clip.action_confuser,
        )

    def tags_set(self) -> tuple[str, ...]:
        """Names of the slice axes that have a concrete value.

        Used by eval code to decide which slice dimensions are
        meaningful to aggregate over (``lighting``, ``occlusion``,
        ``multi_person``, ``action_confuser``).
        """
        return tuple(
            name for name, value in (
                ("lighting", self.lighting),
                ("occlusion", self.occlusion),
                ("multi_person", self.multi_person),
                ("action_confuser", self.action_confuser),
            )
            if value is not None
        )

    def keys(self) -> tuple[SliceKey, ...]:
        """Materialise a :class:`SliceTags` into a tuple of :class:`SliceKey`.

        Only tags with a non-None value are included. ``multi_person``
        is rendered as the literal string ``"true"`` / ``"false"`` so
        it round-trips through the persisted JSON without losing the
        boolean distinction.
        """
        items: list[SliceKey] = []
        if self.lighting is not None:
            items.append(SliceKey("lighting", self.lighting))
        if self.occlusion is not None:
            items.append(SliceKey("occlusion", self.occlusion))
        if self.multi_person is not None:
            items.append(SliceKey("multi_person", "true" if self.multi_person else "false"))
        if self.action_confuser is not None:
            items.append(SliceKey("action_confuser", self.action_confuser))
        return tuple(items)


# ---------------------------------------------------------------------------
# Clip-level predictions and labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipPrediction:
    """One model's output for one clip.

    ``score`` is the model-issued fall probability in ``[0, 1]``.
    Out-of-range values are not validated here — that's a problem for
    the model wrapper, not the eval contract.

    ``slice_tags`` is optional because the model can be evaluated on a
    subset of clips that lack slice metadata; the metric code then
    reports an aggregate-only result for those rows.
    """

    clip_id: str
    score: float
    model_id: str
    dataset: str
    role: ClipRole
    slice_tags: SliceTags | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.clip_id, str) or not self.clip_id:
            raise ValueError(f"ClipPrediction.clip_id must be a non-empty string.")
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"ClipPrediction.score must be numeric, got {type(self.score).__name__}.")


@dataclass(frozen=True)
class ClipLabel:
    """Ground-truth label for one clip.

    Bundled with the manifest metadata so downstream metric code can
    group by dataset / role / slice without re-reading the manifest —
    eval is self-contained once labels and predictions are paired.
    """

    clip_id: str
    label: FallLabel
    dataset: str
    role: ClipRole
    source_path: str
    slice_tags: SliceTags | None = None


# ---------------------------------------------------------------------------
# Event-level predictions and ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventPredictionStream:
    """Frame-indexed fall-score stream for one clip.

    ``frame_scores`` is an ordered tuple of ``(frame_index, score)``
    pairs. Sparse (only trigger frames) AND dense (every frame)
    streams are legal — the eval code that consumes this decides the
    aggregation strategy.

    ``clip_start_frame`` / ``clip_end_frame`` carry the absolute frame
    range of the underlying clip so event metrics can compute
    detection delay against a global timeline.
    """

    clip_id: str
    frame_scores: tuple[tuple[int, float], ...]
    model_id: str
    clip_start_frame: int = 0
    clip_end_frame: int = 0

    def __post_init__(self) -> None:
        for pair in self.frame_scores:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    f"EventPredictionStream.frame_scores entries must be (frame_index, score) tuples; "
                    f"got {pair!r}."
                )
            frame_index, score = pair
            if not isinstance(frame_index, int):
                raise ValueError(f"frame_index must be int, got {type(frame_index).__name__}.")
            if not isinstance(score, (int, float)):
                raise ValueError(f"score must be numeric, got {type(score).__name__}.")


@dataclass(frozen=True)
class EventGroundTruthWindow:
    """Temporal ground-truth window for one event.

    The convention is **inclusive** on both ends. ``start_frame == end_frame``
    is a legal one-frame window for instantaneous events.

    One clip can carry multiple windows (e.g. two fall events) — these
    are emitted as separate :class:`EventGroundTruthWindow` rows.
    """

    clip_id: str
    start_frame: int
    end_frame: int
    label: FallLabel

    def __post_init__(self) -> None:
        if self.start_frame > self.end_frame:
            raise ValueError(
                f"EventGroundTruthWindow {self.clip_id!r}: start_frame ({self.start_frame}) must be "
                f"<= end_frame ({self.end_frame})."
            )


# ---------------------------------------------------------------------------
# Metric results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """One computed metric.

    ``value`` may be either a Python ``float`` or a :class:`NotAvailable`
    instance — both are legal, and downstream code (including the
    result-persistence stub) treats them as distinct shapes.

    ``slice_key`` is ``None`` for aggregate (non-sliced) results. When
    set, it identifies which slice this metric is the result for.

    ``higher_is_better`` records the metric's natural direction so
    downstream consumers (model selection, gating) can pick the
    comparator without hard-coding per-metric knowledge.
    """

    name: str
    value: float | object  # narrow in practice to float | NotAvailable
    slice_key: SliceKey | None = None
    higher_is_better: bool = True
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"MetricResult.name must be a non-empty string.")
        # Lazy-import NotAvailable to avoid a circular import — the
        # type lives in its own module so the JSON encoder can pick it
        # up without pulling contracts.
        from evaluation.not_available import NotAvailable  # noqa: PLC0415
        if not isinstance(self.value, (float, int, NotAvailable)):
            raise ValueError(
                f"MetricResult.value must be numeric or NotAvailable; got {type(self.value).__name__}."
            )

    def is_available(self) -> bool:
        """``True`` iff ``value`` is a real number, not a NotAvailable marker."""
        from evaluation.not_available import NotAvailable  # noqa: PLC0415
        return not isinstance(self.value, NotAvailable)

    def numeric_value(self) -> float:
        """Return ``value`` as a ``float``. Raises if the metric is :class:`NotAvailable`."""
        from evaluation.not_available import NotAvailable  # noqa: PLC0415
        if isinstance(self.value, NotAvailable):
            raise ValueError(
                f"MetricResult {self.name!r} is not available: {self.value.reason}"
            )
        return float(self.value)


__all__: tuple[str, ...] = (
    "SliceKey",
    "SliceTags",
    "ClipPrediction",
    "ClipLabel",
    "EventPredictionStream",
    "EventGroundTruthWindow",
    "MetricResult",
)
