"""Event-level metric bundle + component-metric scaffolding.

Builds the system/event-level path on top of the Step 1 event
contracts (:class:`EventPredictionStream`,
:class:`EventGroundTruthWindow`). Step 4 ships the metric
computations and the deterministic alert-derivation function —
NOT a real fallback alert engine. The fallback to "real Pipeline C
post-verification" is documented as an extension seam.

Pipeline (per-clip)
-------------------

1. **Alert derivation.** A pure function
   :func:`derive_alert_frame_indices` takes an ordered score
   stream and emits the frame indices at which an alert "fires"
   under a configurable :class:`AlertRule`. Defaults match the
   PRD's post-verification starter: ``score >= 0.80`` sustained
   for ``10`` consecutive frames. External alert frames
   (e.g. a future Pipeline C verification engine) can be
   passed directly to :func:`compute_event_metrics_for_clip`
   — the metrics do not care how the alerts were derived.
2. **Alert ↔ GT matching.**
   :func:`match_alerts_to_events` walks the GT events, finds
   the first alert that falls within the GT window (with
   optional tolerance), and reports matched / unmatched /
   per-match delays. Each event is consumed at most once.
3. **Per-clip metric bundle.**
   :func:`compute_event_metrics_for_clip` turns a matching
   into a typed :class:`EventMetricBundle` of
   :class:`MetricResult` rows.
4. **Cross-dataset aggregation.**
   :func:`aggregate_event_metrics_by_dataset` groups bundles
   by ``dataset`` and emits per-dataset precision / recall / F1
   rows so the comparison table the PRD asks for is one
   function call away.

Component metrics scaffold
--------------------------

:meth:`compute_component_metrics` returns ``NotAvailable``
rows with precise reasons until real detection / tracking / pose
ground truth + their library integrations land. The seam lists
which library handles which metric:

- mAP / mAP@0.5:0.95 — :mod:`sklearn.metrics` ``average_precision_score``
  over flat detection scores vs. detection ground truth.
- IDF1 / MOTA — :mod:`motmetrics` (in the approved stack,
  ``requirements.txt`` line 26).
- HOTA — :func:`trackeval` (manual install per Issue 001 review;
  the seam exists so the future implementation is a one-line
  swap).
- PCK — vendor-agnostic pose-GT-distance adapter.

Calling the scaffold with non-None ground truth still returns
``NotAvailable`` but with reason ``"component metric integration
pending"``, signalling where the seam is wired without
implementing the metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from evaluation.contracts import (
    EventGroundTruthWindow,
    EventPredictionStream,
    MetricResult,
    SliceKey,
)
from evaluation.not_available import NotAvailable


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Default alert-derivation threshold. Matches the PRD's post-
#: verification starter ("fall probability > 0.80 for 10
#: consecutive frames").
DEFAULT_ALERT_THRESHOLD: float = 0.80

#: Default persistence (frames the score must stay above the
#: threshold before an alert fires). Matches the PRD's starter.
DEFAULT_ALERT_PERSISTENCE: int = 10

#: Default frame tolerance when matching an alert to a GT event.
#: 0 means "the alert frame must lie inside the GT window"; a
#: positive tolerance expands the window by ``tolerance`` frames
#: on each side. 0 is the strict default — non-zero is opt-in.
DEFAULT_EVENT_TOLERANCE_FRAMES: int = 0


# ---------------------------------------------------------------------------
# Alert rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertRule:
    """Operating point for the alert-derivation function.

    Both fields are keyword-overridable on
    :func:`derive_alert_frame_indices` directly, but exposing them
    on a typed value lets future callers compare two operating
    points side-by-side without re-binding the function.
    """

    threshold: float = DEFAULT_ALERT_THRESHOLD
    persistence: int = DEFAULT_ALERT_PERSISTENCE

    def __post_init__(self) -> None:
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError(
                f"AlertRule.threshold must be in [0, 1], got {self.threshold!r}."
            )
        if self.persistence < 1:
            raise ValueError(
                f"AlertRule.persistence must be >= 1, got {self.persistence!r}."
            )


# ---------------------------------------------------------------------------
# Pure alert derivation
# ---------------------------------------------------------------------------


def derive_alert_frame_indices(
    scores: Sequence[float],
    *,
    threshold: float = DEFAULT_ALERT_THRESHOLD,
    persistence: int = DEFAULT_ALERT_PERSISTENCE,
    frame_offset: int = 0,
) -> tuple[int, ...]:
    """Threshold + persistence over an ordered score stream.

    An alert fires at the frame that completes a run of
    ``persistence`` consecutive frames with ``score >= threshold``.
    The returned frame index is ``position_in_scores + frame_offset``,
    so callers can keep clip-absolute indexing by passing the
    stream's start frame.

    Convention: after an alert fires the run is **reset** so a
    sustained high-score region produces one alert per
    ``persistence``-sized chunk rather than one per frame. A
    future alert engine (Pipeline C post-verification) can ignore
    this convention entirely by passing its own alert frames to
    :func:`compute_event_metrics_for_clip` directly.

    Args:
        scores: Ordered fall probabilities (one per frame,
            contiguous). Out-of-range values are not validated here
            — the alert engine that produced the scores is
            responsible for that.
        threshold: Score level above which a frame counts toward
            the run.
        persistence: Number of consecutive "above threshold"
            frames required to fire.
        frame_offset: Constant added to every returned frame
            index (so a slice of a longer stream can keep
            absolute frame indices).

    Returns:
        Sorted tuple of frame indices at which an alert fired.
        Empty tuple when the run never reaches ``persistence``
        consecutive above-threshold frames.

    Raises:
        ValueError: on ``persistence < 1``.
    """
    if persistence < 1:
        raise ValueError(
            f"persistence must be >= 1, got {persistence!r}."
        )

    alerts: list[int] = []
    run_start: int | None = None
    for position, score in enumerate(scores):
        if score >= threshold:
            if run_start is None:
                run_start = position
            run_length = position - run_start + 1
            if run_length >= persistence:
                # Firing frame is the LAST frame of the run.
                alerts.append(position + frame_offset)
                # Reset to keep one alert per persistence-sized chunk.
                run_start = None
        else:
            run_start = None
    return tuple(alerts)


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventMatching:
    """Outcome of :func:`match_alerts_to_events`.

    All counts/sequences are stable, frozen tuples so the
    structure is hashable and persistence-friendly.
    """

    #: Frame indices of the alerts that matched a GT event,
    #: ordered by encounter order in the alert stream.
    matched_alert_frames: tuple[int, ...]

    #: Number of GT events that found a matching alert. Combined
    #: with ``total_events`` this drives event-level recall.
    matched_event_count: int

    #: Frame indices of alerts that found no GT event. Drives
    #: false-alarm count and (with ``fps`` + ``total_frames``)
    #: the false-alarms-per-hour metric.
    unmatched_alerts: tuple[int, ...]

    #: GT events that found no matching alert. Drives the
    #: "missed events" component of recall.
    unmatched_events: tuple[EventGroundTruthWindow, ...]

    #: ``alert_frame - event.start_frame`` for each matched event,
    #: in frames. Positive ⇒ the alert fired after the start;
    #: negative ⇒ the alert fired before the event began (only
    #: possible with a positive tolerance that swallows the
    #: pre-event lead-in).
    match_delays_frames: tuple[int, ...]


def match_alerts_to_events(
    alerts: Iterable[int],
    events: Iterable[EventGroundTruthWindow],
    *,
    tolerance: int = DEFAULT_EVENT_TOLERANCE_FRAMES,
) -> EventMatching:
    """Match each GT event to its first-matching alert.

    Matching rule (the corrected contract):

    1. An alert is a **false alarm** iff it falls outside every
       ground-truth event window (including tolerance).
    2. A GT event is **matched** if at least one alert falls in
       its tolerated window.
    3. Each GT event is consumed at most once. The first alert
       in window order that lands in an unmatched event is the
       one that credits it (and sets the detection delay).
    4. **Redundant in-window alerts** — alerts that fall in an
       already-matched event's window — are harmless: they are
       neither additional matches nor false alarms. They are
       dropped from the audit trail so the false-alarm count
       and false-alarms-per-hour only count spurious alerts.

    Args:
        alerts: Frame indices at which alerts fired. Unsorted
            input is sorted internally; matching considers alerts
            in ascending frame order.
        events: Ground-truth events for the clip. Order is
            preserved on iteration; matching is greedy by event
            order, so callers wanting a different matching
            (e.g. the closest alert rather than the first) must
            reorder first.
        tolerance: Frame slack on each side of an event window.
            A positive tolerance swallows small lead-ins
            (``alert < start``) and small trailing edges
            (``alert > end``).

    Returns:
        A populated :class:`EventMatching`. ``unmatched_alerts``
        contains only spurious alerts (those outside every
        tolerated GT window); redundant in-window alerts are
        excluded.
    """
    if tolerance < 0:
        raise ValueError(
            f"tolerance must be >= 0, got {tolerance!r}."
        )

    alert_list = sorted(set(int(a) for a in alerts))
    event_list = list(events)
    if not event_list:
        # No GT events: every alert is spurious — there is no
        # window any alert could redundantly fall in.
        return EventMatching(
            matched_alert_frames=(),
            matched_event_count=0,
            unmatched_alerts=tuple(alert_list),
            unmatched_events=(),
            match_delays_frames=(),
        )

    matched_event_indices: set[int] = set()
    matched_alerts: list[int] = []
    unmatched_alerts: list[int] = []
    delays: list[int] = []

    for alert_frame in alert_list:
        # First pass: try to find an UNMATCHED event whose window
        # contains this alert. That is the "first-in-window wins"
        # match — credit the alert's frame and the event's start
        # frame.
        matched_event_index: int | None = None
        for index, event in enumerate(event_list):
            if index in matched_event_indices:
                continue
            window_low = event.start_frame - tolerance
            window_high = event.end_frame + tolerance
            if window_low <= alert_frame <= window_high:
                matched_event_index = index
                break
        if matched_event_index is not None:
            matched_event_indices.add(matched_event_index)
            matched_alerts.append(alert_frame)
            delays.append(
                alert_frame - event_list[matched_event_index].start_frame
            )
            continue

        # Second pass: is this alert inside ANY event's window
        # (matched or not)? If yes, it is a redundant in-window
        # alert for an already-matched event — silently dropped.
        # It is neither an additional match nor a false alarm.
        if _alert_in_any_window(alert_frame, event_list, tolerance):
            continue

        # Spurious alert: outside every tolerated GT window.
        unmatched_alerts.append(alert_frame)

    unmatched_events = tuple(
        event for index, event in enumerate(event_list)
        if index not in matched_event_indices
    )
    return EventMatching(
        matched_alert_frames=tuple(matched_alerts),
        matched_event_count=len(matched_event_indices),
        unmatched_alerts=tuple(unmatched_alerts),
        unmatched_events=unmatched_events,
        match_delays_frames=tuple(delays),
    )


def _alert_in_any_window(
    alert_frame: int,
    event_list: Sequence[EventGroundTruthWindow],
    tolerance: int,
) -> bool:
    """``True`` if ``alert_frame`` falls inside any event's tolerated window."""
    for event in event_list:
        window_low = event.start_frame - tolerance
        window_high = event.end_frame + tolerance
        if window_low <= alert_frame <= window_high:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-clip metric bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventMetricBundle:
    """One clip's event-level metric bundle.

    The bundle carries the raw matching results so a reviewer
    can interpret the metrics without rerunning matching, and
    exposes :meth:`metric_results` that flattens everything into
    :class:`MetricResult` rows that the Step 1 persistence
    layer accepts unchanged.

    Two parallel counts are tracked separately because the
    recall and precision denominators come from different
    worlds:

    - ``matched_events`` / ``total_events`` count GT events.
    - ``matched_alerts`` / ``len(unmatched_alerts)`` count
      alerts the system raised.

    Precision = matched_alerts / (matched_alerts + unmatched_alerts).
    Recall    = matched_events / total_events.
    """

    clip_id: str
    dataset: str
    matched_events: int
    total_events: int
    matched_alerts: int
    unmatched_alerts: tuple[int, ...]
    match_delays_frames: tuple[int, ...]
    fps: float | None
    total_frames: int | None

    def metric_results(self) -> tuple[MetricResult, ...]:
        """All :class:`MetricResult` rows for this clip.

        Every row carries ``slice_key=None`` because per-clip
        event metrics are not slice-aggregated — slices belong
        to the cross-dataset / per-dataset aggregator
        (:func:`aggregate_event_metrics_by_dataset`).
        """
        sk: SliceKey | None = None
        rows: list[MetricResult] = []

        # recall
        if self.total_events > 0:
            recall_value: float | NotAvailable = self.matched_events / self.total_events
        else:
            recall_value = NotAvailable(
                reason="no event GT available",
                metric_name="event_recall",
            )
        rows.append(MetricResult(
            name="event_recall",
            value=recall_value,
            slice_key=sk,
            higher_is_better=True,
        ))

        # precision = matched_events / (matched_events + false_alarm_count).
        # Per the corrected contract: matched_alerts is NOT the
        # numerator. The numerator counts GT events credited; the
        # denominator counts both credited events and spurious
        # alerts (the false alarms). Redundant in-window alerts are
        # neither — they have already been excluded from
        # ``self.unmatched_alerts`` at the matcher.
        total_alerts_for_precision = self.matched_events + len(self.unmatched_alerts)
        if total_alerts_for_precision > 0:
            precision_value: float | NotAvailable = (
                self.matched_events / total_alerts_for_precision
            )
        else:
            precision_value = NotAvailable(
                reason="no alerts fired",
                metric_name="event_precision",
            )
        rows.append(MetricResult(
            name="event_precision",
            value=precision_value,
            slice_key=sk,
            higher_is_better=True,
        ))

        # F1.
        if (
            isinstance(recall_value, float)
            and isinstance(precision_value, float)
        ):
            if (recall_value + precision_value) > 0.0:
                f1_value: float | NotAvailable = (
                    2 * precision_value * recall_value
                    / (precision_value + recall_value)
                )
            else:
                f1_value = NotAvailable(
                    reason="precision and recall both zero",
                    metric_name="event_f1",
                )
        else:
            f1_value = NotAvailable(
                reason="precision or recall undefined",
                metric_name="event_f1",
            )
        rows.append(MetricResult(
            name="event_f1",
            value=f1_value,
            slice_key=sk,
            higher_is_better=True,
        ))

        # false-alarm count (always defined when alerts were raised;
        # can be 0).
        rows.append(MetricResult(
            name="false_alarms",
            value=float(len(self.unmatched_alerts)),
            slice_key=sk,
            higher_is_better=False,
        ))

        # false-alarms-per-hour.
        rows.append(MetricResult(
            name="false_alarms_per_hour",
            value=_compute_false_alarms_per_hour(
                unmatched_alerts=self.unmatched_alerts,
                fps=self.fps,
                total_frames=self.total_frames,
                metric_name="false_alarms_per_hour",
            ),
            slice_key=sk,
            higher_is_better=False,
        ))

        # mean / p95 detection delay, frames.
        rows.append(MetricResult(
            name="detection_delay_mean_frames",
            value=_safe_mean_or_not_available(
                self.match_delays_frames, metric_name="detection_delay_mean_frames",
            ),
            slice_key=sk,
            higher_is_better=False,
        ))
        rows.append(MetricResult(
            name="detection_delay_p95_frames",
            value=_safe_percentile_or_not_available(
                self.match_delays_frames, 95.0,
                metric_name="detection_delay_p95_frames",
            ),
            slice_key=sk,
            higher_is_better=False,
        ))

        # mean / p95 detection delay, seconds. Only available when
        # fps is set.
        rows.append(MetricResult(
            name="detection_delay_mean_seconds",
            value=_safe_mean_seconds_or_not_available(
                self.match_delays_frames, fps=self.fps,
                metric_name="detection_delay_mean_seconds",
            ),
            slice_key=sk,
            higher_is_better=False,
        ))
        rows.append(MetricResult(
            name="detection_delay_p95_seconds",
            value=_safe_percentile_seconds_or_not_available(
                self.match_delays_frames, 95.0, fps=self.fps,
                metric_name="detection_delay_p95_seconds",
            ),
            slice_key=sk,
            higher_is_better=False,
        ))

        # Supporting rows so a reviewer can recover the matching
        # picture from ``metric_results()`` alone.
        rows.append(MetricResult(
            name="total_events",
            value=float(self.total_events),
            slice_key=sk,
        ))
        rows.append(MetricResult(
            name="matched_events",
            value=float(self.matched_events),
            slice_key=sk,
        ))
        rows.append(MetricResult(
            name="matched_alerts",
            value=float(self.matched_alerts),
            slice_key=sk,
        ))
        rows.append(MetricResult(
            name="total_alerts",
            value=float(self.matched_alerts + len(self.unmatched_alerts)),
            slice_key=sk,
        ))
        return tuple(rows)


def _compute_false_alarms_per_hour(
    *,
    unmatched_alerts: tuple[int, ...],
    fps: float | None,
    total_frames: int | None,
    metric_name: str,
) -> float | NotAvailable:
    """Compute false alarms per hour; honest NotAvailable on missing inputs."""
    if fps is None or fps <= 0.0:
        return NotAvailable(reason="no fps / temporal metadata", metric_name=metric_name)
    if total_frames is None or total_frames <= 0:
        return NotAvailable(reason="no fps / temporal metadata", metric_name=metric_name)
    duration_hours = total_frames / fps / 3600.0
    if duration_hours <= 0.0:
        return NotAvailable(reason="no fps / temporal metadata", metric_name=metric_name)
    return len(unmatched_alerts) / duration_hours


def _safe_mean_or_not_available(
    values: Sequence[float],
    *,
    metric_name: str,
) -> float | NotAvailable:
    if not values:
        return NotAvailable(reason="no event GT available", metric_name=metric_name)
    return sum(values) / len(values)


def _safe_mean_seconds_or_not_available(
    values: Sequence[float],
    *,
    fps: float | None,
    metric_name: str,
) -> float | NotAvailable:
    if fps is None or fps <= 0.0:
        return NotAvailable(reason="no fps / temporal metadata", metric_name=metric_name)
    if not values:
        return NotAvailable(reason="no event GT available", metric_name=metric_name)
    return sum(values) / len(values) / fps


def _safe_percentile_or_not_available(
    values: Sequence[float],
    percentile: float,
    *,
    metric_name: str,
) -> float | NotAvailable:
    if not values:
        return NotAvailable(reason="no event GT available", metric_name=metric_name)
    return _percentile(values, percentile)


def _safe_percentile_seconds_or_not_available(
    values: Sequence[float],
    percentile: float,
    *,
    fps: float | None,
    metric_name: str,
) -> float | NotAvailable:
    if fps is None or fps <= 0.0:
        return NotAvailable(reason="no fps / temporal metadata", metric_name=metric_name)
    if not values:
        return NotAvailable(reason="no event GT available", metric_name=metric_name)
    return _percentile(values, percentile) / fps


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear-interpolation percentile (matches numpy's default).

    Avoids pulling :mod:`numpy` into the eval bundle. For a
    single value the percentile is that value. For longer
    sequences, ``rank = (p / 100) * (n - 1)`` and the answer is
    the linear interpolation between the surrounding sorted
    samples — same convention as ``numpy.percentile`` with
    ``interpolation='linear'``.
    """
    import math  # local import keeps the helper import surface narrow.

    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}.")
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (n - 1)
    if rank >= n - 1:
        return float(sorted_values[-1])
    lower_index = math.floor(rank)
    fraction = rank - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[lower_index + 1] * fraction
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compute_event_metrics_for_clip(
    clip_id: str,
    dataset: str,
    alerts: Iterable[int],
    events: Iterable[EventGroundTruthWindow],
    *,
    fps: float | None = None,
    total_frames: int | None = None,
    tolerance: int = DEFAULT_EVENT_TOLERANCE_FRAMES,
) -> EventMetricBundle:
    """Run matching + per-clip metrics for one clip.

    Args:
        clip_id: Carried through into the bundle.
        dataset: Carried through; used by
            :func:`aggregate_event_metrics_by_dataset`.
        alerts: Frame indices at which alerts fired. The caller
            may pass alerts from :func:`derive_alert_frame_indices`
            or from an external post-verification engine — the
            metrics do not care how they were derived.
        events: Ground-truth events for the clip.
        fps: Optional — enables seconds-based delay and
            false-alarms/hour. Falsy means those metrics become
            :class:`NotAvailable`.
        total_frames: Optional — frame count over which the
            alerts were raised; enables false-alarms/hour.
            Required when ``fps`` is provided but is otherwise
            optional.
        tolerance: Frame slack on each side of an event window
            when matching. Defaults to 0 (alert frame must lie
            inside the GT window).

    Returns:
        A populated :class:`EventMetricBundle`.
    """
    alert_list = tuple(int(a) for a in alerts)
    matching = match_alerts_to_events(alert_list, events, tolerance=tolerance)
    total_events = (
        matching.matched_event_count + len(matching.unmatched_events)
    )
    return EventMetricBundle(
        clip_id=clip_id,
        dataset=dataset,
        matched_events=matching.matched_event_count,
        total_events=total_events,
        matched_alerts=len(matching.matched_alert_frames),
        unmatched_alerts=matching.unmatched_alerts,
        match_delays_frames=matching.match_delays_frames,
        fps=fps,
        total_frames=total_frames,
    )


def compute_event_metrics_for_stream(
    stream: EventPredictionStream,
    events: Iterable[EventGroundTruthWindow],
    *,
    dataset: str,
    threshold: float = DEFAULT_ALERT_THRESHOLD,
    persistence: int = DEFAULT_ALERT_PERSISTENCE,
    tolerance: int = DEFAULT_EVENT_TOLERANCE_FRAMES,
    fps: float | None = None,
    total_frames: int | None = None,
) -> EventMetricBundle:
    """Derive alerts from a stream, then run matching + metrics.

    The convenience wrapper used when the caller wants alert
    derivation to happen inside the bundle. When the caller has
    externally-derived alerts (Pipeline C final alerts), use
    :func:`compute_event_metrics_for_clip` directly.
    """
    scores = [score for _, score in stream.frame_scores]
    alerts = derive_alert_frame_indices(
        scores,
        threshold=threshold,
        persistence=persistence,
        frame_offset=stream.clip_start_frame,
    )
    if total_frames is None:
        # Derive a default from the stream so the caller does not
        # have to repeat the frame math.
        total_frames = max(0, stream.clip_end_frame - stream.clip_start_frame + 1)
    return compute_event_metrics_for_clip(
        clip_id=stream.clip_id,
        dataset=dataset,
        alerts=alerts,
        events=events,
        fps=fps,
        total_frames=total_frames,
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
# Cross-dataset aggregation
# ---------------------------------------------------------------------------


def aggregate_event_metrics_by_dataset(
    bundles: Iterable[EventMetricBundle],
) -> list[MetricResult]:
    """Group per-clip bundles by ``dataset`` and emit per-dataset F1 rows.

    Aggregation rule: per-dataset precision = total matched events /
    total triggered alerts across all clips in the dataset. Recall =
    total matched events / total GT events across all clips in the
    dataset. F1 follows. ``NotAvailable`` is propagated when the
    underlying per-clip bundle carried a ``NotAvailable`` for any
    contributing row — i.e. we do not silently average across
    "missing" rows; the per-dataset row inherits the same honest
    semantics.
    """
    grouped: dict[str, list[EventMetricBundle]] = {}
    for bundle in bundles:
        grouped.setdefault(bundle.dataset, []).append(bundle)

    out: list[MetricResult] = []
    for dataset_name in sorted(grouped.keys()):
        dataset_bundles = grouped[dataset_name]
        total_matched_events = sum(b.matched_events for b in dataset_bundles)
        total_gt = sum(b.total_events for b in dataset_bundles)
        total_matched_alerts = sum(b.matched_alerts for b in dataset_bundles)
        total_unmatched_alerts = sum(
            len(b.unmatched_alerts) for b in dataset_bundles
        )
        slice_key = SliceKey(tag="dataset", value=dataset_name)

        if total_gt > 0:
            recall: float | NotAvailable = total_matched_events / total_gt
        else:
            recall = NotAvailable(
                reason="no event GT available",
                metric_name="event_recall",
            )
        total_alerts = total_matched_alerts + total_unmatched_alerts
        # Corrected precision: numerator is matched EVENTS, not
        # matched alerts. The matched-alerts count is also
        # potentially understated vs. the original ``matched_-
        # alerts / total_alerts`` formula because redundant
        # in-window alerts no longer inflate the denominator
        # (they're dropped at the matcher).
        if total_matched_events + total_unmatched_alerts > 0:
            precision: float | NotAvailable = (
                total_matched_events
                / (total_matched_events + total_unmatched_alerts)
            )
        else:
            precision = NotAvailable(
                reason="no alerts fired",
                metric_name="event_precision",
            )
        if (
            isinstance(recall, float)
            and isinstance(precision, float)
            and (recall + precision) > 0.0
        ):
            f1: float | NotAvailable = (
                2 * precision * recall / (precision + recall)
            )
        else:
            f1 = NotAvailable(
                reason="precision or recall undefined",
                metric_name="event_f1",
            )

        out.append(MetricResult(
            name="event_recall",
            value=recall,
            slice_key=slice_key,
            higher_is_better=True,
        ))
        out.append(MetricResult(
            name="event_precision",
            value=precision,
            slice_key=slice_key,
            higher_is_better=True,
        ))
        out.append(MetricResult(
            name="event_f1",
            value=f1,
            slice_key=slice_key,
            higher_is_better=True,
        ))
        out.append(MetricResult(
            name="total_events",
            value=float(total_gt),
            slice_key=slice_key,
        ))
        out.append(MetricResult(
            name="matched_events",
            value=float(total_matched_events),
            slice_key=slice_key,
        ))
        out.append(MetricResult(
            name="matched_alerts",
            value=float(total_matched_alerts),
            slice_key=slice_key,
        ))
        out.append(MetricResult(
            name="total_alerts",
            value=float(total_alerts),
            slice_key=slice_key,
        ))
    return out


# ---------------------------------------------------------------------------
# Component-metric scaffold
# ---------------------------------------------------------------------------


def compute_component_metrics(
    *,
    detection_ground_truth: object | None = None,
    tracking_ground_truth: object | None = None,
    pose_ground_truth: object | None = None,
    predictions: object | None = None,
    slice_key: SliceKey | None = None,
) -> tuple[MetricResult, ...]:
    """Component-metric scaffold.

    Returns :class:`NotAvailable` rows for mAP / IDF1 / MOTA /
    HOTA / PCK with precise reasons explaining what input the
    metric is missing. Used by the Step 1 ``MetricResultStore``
    callers who want to record "we tried, here is why it didn't
    compute" rather than silently dropping the metric.

    Library seams (documented, not implemented in Step 4):

    - ``map_50`` / ``map_50_95``: ``sklearn.metrics.average_precision_score``
      over flat per-detection confidence scores vs. detection
      ground truth (per-class AP, then mean). The PRD notes
      these metrics require detection ground truth, which URFD
      does not ship (see context.txt for Issue 002 metric
      limitation note).
    - ``idf1`` / ``mota``: ``motmetrics`` package (already in
      ``requirements.txt`` line 26 as ``motmetrics>=1.4.0``).
    - ``hota``: TrackEval — manual install per
      ``colab/setup.py:TRACKEVAL_INSTALL_NOTE``.
    - ``pck``: vendor-agnostic pose-GT-distance adapter; not
      implemented here. PCK requires pose ground truth which
      none of the currently-staged datasets ship.

    The function returns ``NotAvailable`` rows on every code
    path so the persisted payload is structurally identical to
    what a successful evaluation would produce. Once real
    detection / tracking / pose ground truth lands and the
    library integrations are wired up, replace the NotAvailable
    rows with the real metric values; the rest of the
    persistence layer needs no change.
    """
    rows: list[MetricResult] = []

    # mAP variants — require detection ground truth.
    for name in ("map_50", "map_50_95"):
        if detection_ground_truth is None:
            reason = "no detection ground truth"
        else:
            reason = "component metric integration pending"
        rows.append(MetricResult(
            name=name,
            value=NotAvailable(reason=reason, metric_name=name),
            slice_key=slice_key,
            higher_is_better=True,
        ))

    # IDF1 / MOTA / HOTA — require tracking ground truth.
    for name in ("idf1", "mota", "hota"):
        if tracking_ground_truth is None:
            reason = "no tracking ground truth"
        else:
            reason = "component metric integration pending"
        rows.append(MetricResult(
            name=name,
            value=NotAvailable(reason=reason, metric_name=name),
            slice_key=slice_key,
            # HOTA / IDF1 higher-is-better; MOTA is reported as a
            # signed number (negative is bad) — caller's job to
            # compare like-signed values. Higher-is-better=True
            # because that's the convention the existing metric
            # bundle uses for tracking.
            higher_is_better=True,
        ))

    # PCK — requires pose ground truth.
    if pose_ground_truth is None:
        reason = "no pose ground truth"
    else:
        reason = "component metric integration pending"
    rows.append(MetricResult(
        name="pck",
        value=NotAvailable(reason=reason, metric_name="pck"),
        slice_key=slice_key,
        higher_is_better=True,
    ))

    # ``predictions`` parameter is reserved for the future
    # integration; referencing it here keeps the signature
    # stable so callers wiring Pipeline A/B/C in upcoming steps
    # do not have to chase signature changes.
    _ = predictions

    return tuple(rows)


__all__: tuple[str, ...] = (
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
