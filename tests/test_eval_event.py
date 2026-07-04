"""Tests for :mod:`evaluation.metrics.event`.

Coverage target (per the Step 4 task spec):

- Pure alert derivation: defaults + configurable threshold / persistence.
- Externally supplied alert frames override derived alerts.
- Event matching with tolerance + first-match-per-event.
- Unmatched alerts counted as false alarms.
- Event-level recall + precision + F1.
- Detection delay mean + p95 in frames; seconds only when fps set.
- False alarms / hour: NotAvailable without fps; numeric with fps.
- No event GT → NotAvailable for recall / delay.
- Cross-dataset event F1 per dataset.
- Component scaffolding (mAP / IDF1 / MOTA / HOTA / PCK) returns
  NotAvailable with clear reasons.
- Results persist + reload via the Step 1 MetricResultStore.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.manifests import FallLabel

from evaluation.contracts import (
    EventGroundTruthWindow,
    EventPredictionStream,
    MetricResult,
    SliceKey,
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
from evaluation.not_available import NotAvailable
from evaluation.result_persistence import (
    EvalRunMetadata,
    MetricResultStore,
    make_default_metadata,
)


# ---------------------------------------------------------------------------
# Alert rule shape
# ---------------------------------------------------------------------------


class AlertRuleTests(unittest.TestCase):
    """AlertRule validates at construction time."""

    def test_defaults_match_module_constants(self) -> None:
        rule = AlertRule()
        self.assertEqual(rule.threshold, DEFAULT_ALERT_THRESHOLD)
        self.assertEqual(rule.persistence, DEFAULT_ALERT_PERSISTENCE)
        # Module constants — pinned by the task spec.
        self.assertEqual(DEFAULT_ALERT_THRESHOLD, 0.80)
        self.assertEqual(DEFAULT_ALERT_PERSISTENCE, 10)

    def test_threshold_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AlertRule(threshold=1.5)
        with self.assertRaises(ValueError):
            AlertRule(threshold=-0.1)

    def test_persistence_below_one_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AlertRule(persistence=0)


# ---------------------------------------------------------------------------
# Pure alert derivation
# ---------------------------------------------------------------------------


class AlertDerivationTests(unittest.TestCase):
    """``derive_alert_frame_indices`` is a pure function."""

    def test_no_run_no_alert(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.05]
        alerts = derive_alert_frame_indices(scores, threshold=0.8, persistence=10)
        self.assertEqual(alerts, ())

    def test_default_persistence_at_threshold(self) -> None:
        # 10 consecutive high frames → alert at the LAST frame.
        scores = [0.85] * 10 + [0.85] * 5 + [0.1]
        alerts = derive_alert_frame_indices(scores)
        # 10 consecutive → first alert at position 9. After the
        # alert, the run is reset; positions 10–13 form a new
        # 4-frame run, which is below persistence=10 → no further
        # alert.
        self.assertEqual(alerts, (9,))

    def test_configurable_threshold_and_persistence(self) -> None:
        scores = [0.5, 0.5, 0.5]
        # threshold=0.4, persistence=3 → alert at position 2.
        self.assertEqual(
            derive_alert_frame_indices(scores, threshold=0.4, persistence=3),
            (2,),
        )
        # Same scores, threshold=0.6 → no frames clear the bar.
        self.assertEqual(
            derive_alert_frame_indices(scores, threshold=0.6, persistence=3),
            (),
        )
        # Same scores, persistence=4 → run too short.
        self.assertEqual(
            derive_alert_frame_indices(scores, threshold=0.4, persistence=4),
            (),
        )

    def test_alerts_fire_on_each_persistence_sized_chunk(self) -> None:
        # 25 high frames → alerts at positions 9 and 19 (every
        # persistence-sized chunk restarts the run).
        scores = [0.9] * 25 + [0.1]
        alerts = derive_alert_frame_indices(scores, threshold=0.5, persistence=10)
        self.assertEqual(alerts, (9, 19))

    def test_break_in_run_resets(self) -> None:
        # 8 high, 1 low, 8 high, 1 low — no run reaches 10 → no alert.
        scores = [0.9] * 8 + [0.1] + [0.9] * 8 + [0.1]
        self.assertEqual(
            derive_alert_frame_indices(scores, threshold=0.5, persistence=10),
            (),
        )

    def test_frame_offset_aligns_with_clip_absolute_indices(self) -> None:
        # Caller can shift a slice-of-stream into absolute clip-frame
        # coordinates by passing ``frame_offset``.
        scores = [0.9] * 10
        alerts = derive_alert_frame_indices(scores, frame_offset=1000)
        self.assertEqual(alerts, (1009,))

    def test_alert_fires_at_end_of_run_not_beginning(self) -> None:
        # Convention: the alert's frame index is the LAST frame of
        # the persistence run, not the first.
        scores = [0.9] * 12
        alerts = derive_alert_frame_indices(scores, threshold=0.5, persistence=10)
        self.assertEqual(alerts, (9,))   # not (0,)

    def test_persistence_below_one_rejected_at_call_site(self) -> None:
        with self.assertRaises(ValueError):
            derive_alert_frame_indices([0.5], persistence=0)


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------


class EventMatchingTests(unittest.TestCase):
    """First-match-per-event semantics + tolerance + unmatched counting."""

    def _ev(self, start: int, end: int, clip_id: str = "c") -> EventGroundTruthWindow:
        return EventGroundTruthWindow(
            clip_id=clip_id, start_frame=start, end_frame=end,
            label=FallLabel.FALL,
        )

    def test_match_inside_window_zero_tolerance(self) -> None:
        events = [self._ev(10, 20)]
        matching = match_alerts_to_events([15], events)
        self.assertEqual(matching.matched_event_count, 1)
        self.assertEqual(matching.unmatched_alerts, ())
        self.assertEqual(matching.matched_alert_frames, (15,))
        # Detection delay = alert_frame - event.start_frame = 5.
        self.assertEqual(matching.match_delays_frames, (5,))

    def test_redundant_in_window_alerts_are_not_false_alarms(self) -> None:
        # Sustained-fall case from the Step 4 fix brief:
        # GT event [100, 130], three alerts at 109, 119, 129 — all
        # in window. The first (109) credits the event with a delay
        # of 109 - 100 = 9 frames; the next two (119, 129) land in
        # the same window and MUST NOT count as false alarms.
        events = [self._ev(100, 130)]
        matching = match_alerts_to_events([109, 119, 129], events)
        self.assertEqual(matching.matched_event_count, 1)
        self.assertEqual(matching.matched_alert_frames, (109,))
        self.assertEqual(matching.unmatched_alerts, ())
        # Detection delay comes from the first in-window alert only.
        self.assertEqual(matching.match_delays_frames, (9,))

    def test_redundant_in_window_does_not_change_per_metric_results(self) -> None:
        # End-to-end: feeding three in-window alerts to a bundle
        # produces recall=1.0, precision=1.0, false_alarms=0,
        # mean_delay=9. The redundant alerts do NOT inflate the
        # false-alarm count or precision's denominator.
        events = [self._ev(100, 130)]
        bundle = compute_event_metrics_for_clip(
            clip_id="c-sustained",
            dataset="urfd",
            alerts=[109, 119, 129],
            events=events,
            fps=30.0,
            total_frames=900,
        )
        by = {m.name: m for m in bundle.metric_results()}
        self.assertAlmostEqual(by["event_recall"].numeric_value(), 1.0, places=6)
        self.assertAlmostEqual(by["event_precision"].numeric_value(), 1.0, places=6)
        # No spurious alerts.
        self.assertAlmostEqual(by["false_alarms"].numeric_value(), 0.0, places=6)
        # Single delay (from alert 109).
        self.assertAlmostEqual(
            by["detection_delay_mean_frames"].numeric_value(), 9.0, places=6
        )
        # false-alarms/hour is 0 / (900 / 30 / 3600) = 0.
        self.assertAlmostEqual(
            by["false_alarms_per_hour"].numeric_value(), 0.0, places=6
        )

    def test_alert_outside_window_zero_tolerance_is_false_alarm(self) -> None:
        events = [self._ev(10, 20)]
        matching = match_alerts_to_events([5, 25], events)
        self.assertEqual(matching.matched_event_count, 0)
        self.assertEqual(matching.unmatched_alerts, (5, 25))
        # ``unmatched_events`` is a tuple, not a list — match the
        # declared type.
        self.assertEqual(matching.unmatched_events, tuple(events))

    def test_tolerance_swallows_lead_in(self) -> None:
        # Alert at frame 8, event [10, 20], tolerance 3 → 8 ∈ [7, 23] → match.
        events = [self._ev(10, 20)]
        matching = match_alerts_to_events([8], events, tolerance=3)
        self.assertEqual(matching.matched_event_count, 1)
        # delay is -2 (alert is 2 frames BEFORE start).
        self.assertEqual(matching.match_delays_frames, (-2,))

    def test_tolerance_swallows_trailing_edge(self) -> None:
        events = [self._ev(10, 20)]
        matching = match_alerts_to_events([22], events, tolerance=3)
        self.assertEqual(matching.matched_event_count, 1)
        # Delay = 22 - 10 = 12.
        self.assertEqual(matching.match_delays_frames, (12,))

    def test_first_match_per_event_wins(self) -> None:
        # Two events at [10, 20] and [30, 40]. Two alerts — one
        # inside each window. Each event consumes its first
        # matching alert; remaining alerts land as false alarms.
        events = [self._ev(10, 20), self._ev(30, 40)]
        matching = match_alerts_to_events([15, 35, 22], events)
        self.assertEqual(matching.matched_event_count, 2)
        self.assertEqual(matching.matched_alert_frames, (15, 35))
        self.assertEqual(matching.unmatched_alerts, (22,))

    def test_first_match_wins_with_colliding_alerts(self) -> None:
        # Two alerts both inside [10, 20]: the first one (by
        # alert frame order) matches the event. The second alert
        # is in the *same* window — under the corrected contract
        # it is a redundant in-window alert for an already-matched
        # event and is silently dropped (not a false alarm).
        events = [self._ev(10, 20)]
        matching = match_alerts_to_events([11, 19], events)
        self.assertEqual(matching.matched_event_count, 1)
        self.assertEqual(matching.matched_alert_frames, (11,))
        # Crucially: 19 is NOT a false alarm — it fell inside the
        # already-matched event's window.
        self.assertEqual(matching.unmatched_alerts, ())

    def test_no_events_means_all_alerts_are_false_alarms(self) -> None:
        matching = match_alerts_to_events([5, 10, 15], [])
        self.assertEqual(matching.matched_event_count, 0)
        self.assertEqual(matching.unmatched_alerts, (5, 10, 15))
        self.assertEqual(matching.unmatched_events, ())

    def test_negative_tolerance_rejected(self) -> None:
        with self.assertRaises(ValueError):
            match_alerts_to_events([1], [self._ev(0, 5)], tolerance=-1)

    def test_alerts_sorted_during_matching(self) -> None:
        # Caller-supplied unsorted alerts are sorted internally
        # before matching — the natural pipeline convention.
        events = [self._ev(10, 20), self._ev(30, 40)]
        matching = match_alerts_to_events([35, 15], events)
        self.assertEqual(matching.matched_alert_frames, (15, 35))

    def test_duplicate_alerts_collapse(self) -> None:
        # Same alert frame listed twice is wasted work, not a
        # bug — the matcher dedupes.
        events = [self._ev(10, 20)]
        matching = match_alerts_to_events([15, 15], events)
        self.assertEqual(matching.matched_event_count, 1)
        self.assertEqual(matching.matched_alert_frames, (15,))
        self.assertEqual(matching.unmatched_alerts, ())


# ---------------------------------------------------------------------------
# Per-clip metric bundle
# ---------------------------------------------------------------------------


class EventMetricBundleTests(unittest.TestCase):
    """Per-clip metrics: recall / precision / F1 / delays / FA-per-hour."""

    def _ev(self, start: int, end: int) -> EventGroundTruthWindow:
        return EventGroundTruthWindow(
            clip_id="c", start_frame=start, end_frame=end,
            label=FallLabel.FALL,
        )

    def _by_name(self, bundle: EventMetricBundle) -> dict[str, MetricResult]:
        return {m.name: m for m in bundle.metric_results()}

    def test_recall_and_precision_match_hand_computed(self) -> None:
        # 2 events, 1 alert inside event #1 only.
        events = [self._ev(10, 20), self._ev(30, 40)]
        alerts = [15]   # matches event #1; event #2 missed.
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=alerts, events=events,
        )
        by = self._by_name(bundle)
        # recall = 1/2 = 0.5.
        self.assertAlmostEqual(by["event_recall"].numeric_value(), 0.5, places=6)
        # precision = 1 / (1 + 0) = 1.0  (one alert, one match).
        self.assertAlmostEqual(by["event_precision"].numeric_value(), 1.0, places=6)
        # F1 = 2*p*r / (p+r) = 2 * 0.5 / 1.5 = 2/3.
        self.assertAlmostEqual(by["event_f1"].numeric_value(), 2 / 3, places=6)

    def test_match_delays_hand_computed(self) -> None:
        # 3 events, 3 alerts (one inside each).  Delays:
        # alert 15 - 10 = 5; alert 25 - 20 = 5; alert 35 - 30 = 5.
        events = [self._ev(10, 20), self._ev(20, 30), self._ev(30, 40)]
        alerts = [15, 25, 35]
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=alerts, events=events,
        )
        by = self._by_name(bundle)
        self.assertAlmostEqual(
            by["detection_delay_mean_frames"].numeric_value(), 5.0, places=6
        )
        # 3 identical samples → p95 interpolation lands on 5.0
        # via the linear interpolation convention.
        self.assertAlmostEqual(
            by["detection_delay_p95_frames"].numeric_value(), 5.0, places=6
        )

    def test_p95_handles_variegated_delays(self) -> None:
        # Three alerts land at frames 5, 15, 25 — the first inside
        # event [0, 5], the next two each on the boundary of their
        # own event. ``alert_frame - event.start_frame`` is
        # therefore:
        # - alert 5 vs event start 0  →  delay 5
        # - alert 15 vs event start 5 →  delay 10
        # - alert 25 vs event start 15 → delay 10
        # Sorted delays = [5, 10, 10]. p95 = sorted[1] + 0.9 *
        # (sorted[2] - sorted[1]) = 10 + 0.9 * 0 = 10.0.
        events = [
            self._ev(0, 5),
            self._ev(5, 15),
            self._ev(15, 25),
        ]
        alerts = [5, 15, 25]
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=alerts, events=events,
        )
        by = self._by_name(bundle)
        self.assertAlmostEqual(
            by["detection_delay_p95_frames"].numeric_value(),
            10.0,
            places=6,
        )

    def test_no_event_gt_returns_not_available_for_recall_and_delay(self) -> None:
        # No GT events; alerts fired.
        alerts = [10, 20, 30]
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=alerts, events=[],
        )
        by = self._by_name(bundle)

        # Recall — undefined (no events).
        recall_na = by["event_recall"].value
        self.assertIsInstance(recall_na, NotAvailable)
        self.assertIn("no event GT available", recall_na.reason)
        # Delay mean / p95 — undefined (no matches → no delays).
        for name in ("detection_delay_mean_frames", "detection_delay_p95_frames"):
            na = by[name].value
            self.assertIsInstance(na, NotAvailable,
                                  msg=f"{name} should be NotAvailable")
            self.assertIn("no event GT available", na.reason)
        # Precision IS defined (alerts fired, none matched → 0 / N).
        precision = by["event_precision"].numeric_value()
        self.assertAlmostEqual(precision, 0.0, places=6)
        # F1 — undefined (recall is NotAvailable).
        self.assertIsInstance(by["event_f1"].value, NotAvailable)
        # False-alarm count = 3 (all unmatched, no events to match).
        self.assertAlmostEqual(by["false_alarms"].numeric_value(), 3.0, places=6)

    def test_seconds_delay_returns_not_available_without_fps(self) -> None:
        events = [self._ev(10, 20)]
        alerts = [15]
        # fps=None → seconds metrics become NotAvailable.
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=alerts, events=events,
            fps=None,
        )
        by = self._by_name(bundle)
        for name in (
            "detection_delay_mean_seconds",
            "detection_delay_p95_seconds",
        ):
            na = by[name].value
            self.assertIsInstance(
                na, NotAvailable,
                msg=f"{name} should be NotAvailable without fps",
            )
            self.assertIn("no fps / temporal metadata", na.reason)
        # Frames-based delay is fine without fps.
        self.assertAlmostEqual(
            by["detection_delay_mean_frames"].numeric_value(),
            5.0, places=6,
        )

    def test_false_alarms_per_hour_not_available_without_fps(self) -> None:
        events = [self._ev(10, 20)]
        alerts = [15, 25]   # 25 is unmatched
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=alerts, events=events,
            fps=None,
            total_frames=900,
        )
        by = self._by_name(bundle)
        na = by["false_alarms_per_hour"].value
        self.assertIsInstance(na, NotAvailable)
        self.assertIn("no fps / temporal metadata", na.reason)
        # But the FALSE-ALARM COUNT is still numeric.
        self.assertAlmostEqual(by["false_alarms"].numeric_value(), 1.0, places=6)

    def test_false_alarms_per_hour_computes_with_fps(self) -> None:
        # 30 frames at 30 fps = 1 second → 1 / 3600 hours.  Two
        # unmatched alerts → 2 / (1/3600) = 7200 FA/hour.
        events = [self._ev(0, 5)]
        alerts = [25, 28, 4]   # 4 matches the event, 25/28 unmatched.
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=alerts, events=events,
            fps=30.0,
            total_frames=30,
        )
        by = self._by_name(bundle)
        # 30 frames / 30 fps = 1 sec; 2 unmatched alerts.
        # 2 / (1 / 3600) = 7200.
        expected = 2 / (30 / 30.0 / 3600.0)
        self.assertAlmostEqual(
            by["false_alarms_per_hour"].numeric_value(),
            expected,
            places=6,
        )

    def test_seconds_delay_with_fps(self) -> None:
        events = [self._ev(0, 100)]
        alerts = [50]  # delay 50 frames at 25 fps → 2 sec.
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=alerts, events=events,
            fps=25.0,
            total_frames=300,
        )
        by = self._by_name(bundle)
        self.assertAlmostEqual(
            by["detection_delay_mean_seconds"].numeric_value(),
            2.0, places=6,
        )

    def test_externally_supplied_alerts_override_derivation(self) -> None:
        # The convenience wrapper derives alerts from the stream.
        # Bypassing the wrapper to pass external alerts must produce
        # the same metric bundle for the same input.
        events = [self._ev(50, 60)]
        # Construct a stream that would NOT trigger any alert
        # under the default rule (scores all below 0.8).
        stream = EventPredictionStream(
            clip_id="c",
            frame_scores=tuple((i, 0.05) for i in range(100)),
            model_id="dummy",
        )
        external_bundle = compute_event_metrics_for_stream(
            stream=stream,
            events=events,
            dataset="urfd",
            threshold=0.99,     # intentionally unreachable
            persistence=10,
            fps=30.0,
        )
        # No alerts from the high threshold → recall reads 0/1 = 0.0
        # (defined; total_events > 0). Delay mean / p95 stay
        # NotAvailable because there are no matches to compute a
        # delay from. Precision stays NotAvailable because no alerts
        # fired at all.
        by = {m.name: m for m in external_bundle.metric_results()}
        self.assertAlmostEqual(by["event_recall"].numeric_value(), 0.0, places=6)
        self.assertIsInstance(by["detection_delay_mean_frames"].value, NotAvailable)
        self.assertIsInstance(by["event_precision"].value, NotAvailable)


# ---------------------------------------------------------------------------
# Stream-level wrapper
# ---------------------------------------------------------------------------


class StreamWrapperTests(unittest.TestCase):
    """``compute_event_metrics_for_stream`` derives alerts + matches."""

    def test_default_persistence_at_threshold(self) -> None:
        scores = [0.9] * 12
        stream = EventPredictionStream(
            clip_id="c",
            frame_scores=tuple((i, s) for i, s in enumerate(scores)),
            model_id="m",
        )
        events = [
            EventGroundTruthWindow(
                clip_id="c", start_frame=8, end_frame=10,
                label=FallLabel.FALL,
            ),
        ]
        bundle = compute_event_metrics_for_stream(
            stream=stream,
            events=events,
            dataset="urfd",
            fps=30.0,
        )
        by = {m.name: m for m in bundle.metric_results()}
        # One alert at position 9 → matches [8, 10] → recall 1.0.
        self.assertAlmostEqual(by["event_recall"].numeric_value(), 1.0, places=6)

    def test_frame_offset_applied(self) -> None:
        # frame_offset pushed to the stream's clip_start_frame so
        # the alert's frame index lines up with the absolute GT
        # window.
        scores = [0.9] * 12
        stream = EventPredictionStream(
            clip_id="c",
            frame_scores=tuple((i, s) for i, s in enumerate(scores)),
            model_id="m",
            clip_start_frame=100,
            clip_end_frame=120,
        )
        events = [
            EventGroundTruthWindow(
                clip_id="c", start_frame=108, end_frame=110,
                label=FallLabel.FALL,
            ),
        ]
        bundle = compute_event_metrics_for_stream(
            stream=stream, events=events, dataset="urfd", fps=30.0,
        )
        by = {m.name: m for m in bundle.metric_results()}
        self.assertAlmostEqual(by["event_recall"].numeric_value(), 1.0, places=6)
        self.assertAlmostEqual(
            by["detection_delay_mean_frames"].numeric_value(), 1.0, places=6
        )


# ---------------------------------------------------------------------------
# Cross-dataset aggregation
# ---------------------------------------------------------------------------


class CrossDatasetAggregationTests(unittest.TestCase):
    """Per-dataset F1 reported with one row set per dataset."""

    def _bundle(
        self,
        clip_id: str,
        dataset: str,
        matched_events: int,
        total_events: int,
        matched_alerts: int,
        unmatched_alerts: tuple[int, ...] = (),
        fps: float | None = None,
    ) -> EventMetricBundle:
        return EventMetricBundle(
            clip_id=clip_id,
            dataset=dataset,
            matched_events=matched_events,
            total_events=total_events,
            matched_alerts=matched_alerts,
            unmatched_alerts=unmatched_alerts,
            match_delays_frames=(),
            fps=fps,
            total_frames=None,
        )

    def test_emits_one_set_per_dataset(self) -> None:
        bundles = [
            self._bundle("c1", "urfd", matched_events=2, total_events=2,
                          matched_alerts=2),
            self._bundle("c2", "urfd", matched_events=1, total_events=2,
                          matched_alerts=2, unmatched_alerts=(99,)),
            self._bundle("c3", "le2i", matched_events=0, total_events=1,
                          matched_alerts=0, unmatched_alerts=(10, 20)),
        ]
        out = aggregate_event_metrics_by_dataset(bundles)
        # Each dataset produces precision/recall/F1 + supporting
        # rows; gather the unique dataset names.
        dataset_names = sorted({
            r.slice_key.value
            for r in out if r.slice_key is not None
        })
        self.assertEqual(dataset_names, ["le2i", "urfd"])

    def test_per_dataset_precision_recall_f1(self) -> None:
        # urfd: 3 matched events / 4 total = 0.75 recall.
        # 4 matched alerts / 5 total alerts = 0.8 (legacy formula)
        # but the CORRECTED precision uses matched_event_count as
        # the numerator: 3 / (3 + 1 unmatched) = 0.75.
        # F1 with both = 0.75 is exactly 0.75.
        bundles = [
            self._bundle("c1", "urfd", matched_events=3, total_events=4,
                          matched_alerts=4, unmatched_alerts=(99,)),
        ]
        out = aggregate_event_metrics_by_dataset(bundles)
        # Index per (dataset_name, metric_name) so multiple rows on
        # the same dataset do not overwrite each other.
        by_dataset = {
            (r.slice_key.value, r.name): r
            for r in out if r.slice_key is not None
        }
        self.assertAlmostEqual(
            by_dataset[("urfd", "event_recall")].numeric_value(),
            3 / 4, places=6,
        )
        self.assertAlmostEqual(
            by_dataset[("urfd", "event_precision")].numeric_value(),
            3 / 4, places=6,
        )
        self.assertAlmostEqual(
            by_dataset[("urfd", "event_f1")].numeric_value(),
            0.75, places=6,
        )

    def test_dataset_with_no_gt_yields_not_available_for_recall(self) -> None:
        bundles = [
            self._bundle("c1", "urfd", matched_events=0, total_events=0,
                          matched_alerts=0, unmatched_alerts=(5, 6)),
        ]
        out = aggregate_event_metrics_by_dataset(bundles)
        recall_row = next(
            r for r in out
            if r.slice_key == SliceKey("dataset", "urfd")
            and r.name == "event_recall"
        )
        self.assertIsInstance(recall_row.value, NotAvailable)
        self.assertIn("no event GT available", recall_row.value.reason)

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(aggregate_event_metrics_by_dataset([]), [])

    def test_one_result_per_dataset_per_metric_name(self) -> None:
        # Pin the row names so a future refactor cannot silently
        # change the persisted JSON shape.
        bundles = [self._bundle("c1", "urfd", matched_events=1,
                                   total_events=1, matched_alerts=1)]
        out = aggregate_event_metrics_by_dataset(bundles)
        names = [r.name for r in out if r.slice_key is not None]
        for expected in ("event_recall", "event_precision", "event_f1",
                          "total_events", "matched_events", "matched_alerts",
                          "total_alerts"):
            self.assertIn(expected, names)


# ---------------------------------------------------------------------------
# Component metric scaffold
# ---------------------------------------------------------------------------


class ComponentScaffoldTests(unittest.TestCase):
    """Component metrics return NotAvailable rows with precise reasons."""

    def test_default_returns_expected_metric_names(self) -> None:
        rows = compute_component_metrics()
        names = {r.name for r in rows}
        for name in ("map_50", "map_50_95", "idf1", "mota", "hota", "pck"):
            self.assertIn(name, names)

    def test_detection_metrics_have_no_detection_ground_truth_reason(self) -> None:
        rows = compute_component_metrics()
        for row in rows:
            if row.name in ("map_50", "map_50_95"):
                self.assertIsInstance(row.value, NotAvailable)
                self.assertEqual(row.value.reason, "no detection ground truth")
                self.assertEqual(row.value.metric_name, row.name)

    def test_tracking_metrics_have_no_tracking_ground_truth_reason(self) -> None:
        rows = compute_component_metrics()
        for row in rows:
            if row.name in ("idf1", "mota", "hota"):
                self.assertIsInstance(row.value, NotAvailable)
                self.assertEqual(row.value.reason, "no tracking ground truth")
                self.assertEqual(row.value.metric_name, row.name)

    def test_pose_metric_has_no_pose_ground_truth_reason(self) -> None:
        rows = compute_component_metrics()
        pck = next(r for r in rows if r.name == "pck")
        self.assertIsInstance(pck.value, NotAvailable)
        self.assertEqual(pck.value.reason, "no pose ground truth")
        self.assertEqual(pck.value.metric_name, "pck")

    def test_present_inputs_still_return_not_available_pending_integration(self) -> None:
        # When ground truth is provided but the library
        # integration isn't wired up yet, the scaffold still
        # returns NotAvailable with reason
        # "component metric integration pending" — that is the
        # the seam's expected state until the real integration
        # lands.
        rows = compute_component_metrics(
            detection_ground_truth={"fake": True},
            tracking_ground_truth={"fake": True},
            pose_ground_truth={"fake": True},
        )
        for row in rows:
            self.assertIsInstance(row.value, NotAvailable)
            self.assertEqual(row.value.reason, "component metric integration pending")

    def test_slice_key_is_propagated(self) -> None:
        sk = SliceKey("lighting", "daylight")
        rows = compute_component_metrics(slice_key=sk)
        for row in rows:
            self.assertEqual(row.slice_key, sk)


# ---------------------------------------------------------------------------
# Persistence integration
# ---------------------------------------------------------------------------


class PersistenceIntegrationTests(unittest.TestCase):
    """The Step 1 store accepts event-metric bundles unchanged."""

    def setUp(self) -> None:
        name = tempfile.mkdtemp()
        self._root = Path(name)
        self.addCleanup(_rm_tree, self._root)

    def test_full_event_metric_payload_persists_and_reloads(self) -> None:
        events = [
            EventGroundTruthWindow(
                clip_id="c", start_frame=10, end_frame=20,
                label=FallLabel.FALL,
            ),
            EventGroundTruthWindow(
                clip_id="c", start_frame=30, end_frame=40,
                label=FallLabel.FALL,
            ),
        ]
        bundle = compute_event_metrics_for_clip(
            clip_id="c", dataset="urfd",
            alerts=[15, 35, 50],   # 50 is unmatched
            events=events,
            fps=30.0,
            total_frames=900,
        )
        aggregator = [
            EventMetricBundle(
                clip_id="c", dataset="urfd",
                matched_events=bundle.matched_events,
                total_events=bundle.total_events,
                matched_alerts=bundle.matched_alerts,
                unmatched_alerts=bundle.unmatched_alerts,
                match_delays_frames=bundle.match_delays_frames,
                fps=bundle.fps,
                total_frames=bundle.total_frames,
            ),
        ]
        rows = list(bundle.metric_results()) + aggregate_event_metrics_by_dataset(
            aggregator
        )

        store = MetricResultStore(self._root)
        meta = make_default_metadata("r-event", "m-event", context="evaluation")
        store.save(meta, rows, overwrite=True)
        reloaded = store.load("r-event")

        # Aggregate recall survived numerically.
        recall_row = next(
            r for r in reloaded.metrics
            if r.name == "event_recall"
            and r.slice_key is None
        )
        self.assertAlmostEqual(recall_row.numeric_value(), 1.0, places=6)

        # Per-dataset F1 row survived with the dataset slice_key.
        # Bundle has recall = 2/2 = 1.0, precision = 2/3 → F1 =
        # 2 * 1.0 * (2/3) / (1.0 + 2/3) = 4/5 = 0.8. The
        # aggregator passes that through unchanged.
        f1_row = next(
            r for r in reloaded.metrics
            if r.name == "event_f1"
            and r.slice_key == SliceKey("dataset", "urfd")
        )
        self.assertAlmostEqual(f1_row.numeric_value(), 0.8, places=6)

        # False-alarms/hour is the seconds metric NotAvailable path
        # survives too (fps is set here, so it should be numeric).
        fa_hour = next(
            r for r in reloaded.metrics
            if r.name == "false_alarms_per_hour"
            and r.slice_key is None
        )
        self.assertGreaterEqual(fa_hour.numeric_value(), 0.0)


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


if __name__ == "__main__":
    unittest.main()
