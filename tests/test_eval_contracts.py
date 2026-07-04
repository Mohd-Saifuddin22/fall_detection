"""Tests for evaluation data contracts (:mod:`evaluation.contracts`).

Step 1 only — schema tests, no metric computation. The goal is to
prove every contract accepts the shapes eval code will feed it and
rejects malformed shapes loudly.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from data.manifests import ClipRecord, ClipRole, FallLabel

from evaluation.contracts import (
    ClipLabel,
    ClipPrediction,
    EventGroundTruthWindow,
    EventPredictionStream,
    MetricResult,
    SliceKey,
    SliceTags,
)
from evaluation.not_available import NotAvailable

from tests.eval_fixtures import make_clip


class SliceKeyTests(unittest.TestCase):
    """SliceKey is hashable, equality-based, and refuses empty fields."""

    def test_basic_construction_and_label(self) -> None:
        k = SliceKey("lighting", "daylight")
        self.assertEqual(k.label(), "lighting=daylight")
        self.assertEqual(k.tag, "lighting")
        self.assertEqual(k.value, "daylight")

    def test_hashable_and_equal(self) -> None:
        # Critical: SliceKey keys must hash so they work as dict keys
        # for the slice aggregator in Step 2+.
        a = SliceKey("dataset", "urfd")
        b = SliceKey("dataset", "urfd")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_inequality_on_differing_field(self) -> None:
        self.assertNotEqual(
            SliceKey("lighting", "daylight"),
            SliceKey("lighting", "dim"),
        )
        self.assertNotEqual(
            SliceKey("lighting", "daylight"),
            SliceKey("occlusion", "daylight"),
        )

    def test_empty_tag_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SliceKey("", "x")

    def test_empty_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SliceKey("x", "")

    def test_to_from_dict_round_trips(self) -> None:
        k = SliceKey("action_confuser", "sleeping")
        self.assertEqual(SliceKey.from_dict(k.to_dict()), k)

    def test_from_dict_rejects_non_dict(self) -> None:
        with self.assertRaises(ValueError):
            SliceKey.from_dict("lighting=daylight")
        with self.assertRaises(ValueError):
            SliceKey.from_dict({"value": "daylight"})  # missing tag
        with self.assertRaises(ValueError):
            SliceKey.from_dict({"tag": "lighting"})  # missing value


class SliceTagsTests(unittest.TestCase):
    """SliceTags mirrors ClipRecord's slice fields and is materialisable."""

    def test_from_clip_propagates_all_slice_fields(self) -> None:
        clip = make_clip(
            "c-01",
            lighting="low_light",
            occlusion="heavy",
            multi_person=True,
            action_confuser="crawling",
        )
        tags = SliceTags.from_clip(clip)
        self.assertEqual(tags.lighting, "low_light")
        self.assertEqual(tags.occlusion, "heavy")
        self.assertEqual(tags.multi_person, True)
        self.assertEqual(tags.action_confuser, "crawling")

    def test_from_clip_with_all_missing_fields(self) -> None:
        # Placeholder rows: every slice tag is None.
        clip = make_clip("c-02")
        tags = SliceTags.from_clip(clip)
        self.assertIsNone(tags.lighting)
        self.assertIsNone(tags.occlusion)
        self.assertIsNone(tags.multi_person)
        self.assertIsNone(tags.action_confuser)

    def test_tags_set_lists_only_set_fields(self) -> None:
        tags = SliceTags(lighting="daylight", occlusion="none", multi_person=False)
        self.assertEqual(set(tags.tags_set()), {"lighting", "occlusion", "multi_person"})

    def test_tags_set_is_empty_when_nothing_set(self) -> None:
        self.assertEqual(SliceTags().tags_set(), ())

    def test_keys_materialises_non_null_tags(self) -> None:
        tags = SliceTags(lighting="daylight", multi_person=True)
        keys = tags.keys()
        self.assertEqual(
            {k.label() for k in keys},
            {"lighting=daylight", "multi_person=true"},
        )

    def test_multi_person_renders_as_true_false_string(self) -> None:
        # Boolean values must round-trip as the string "true"/"false"
        # so the persisted JSON keeps the distinction.
        self.assertIn(
            SliceKey("multi_person", "true"),
            SliceTags(multi_person=True).keys(),
        )
        self.assertIn(
            SliceKey("multi_person", "false"),
            SliceTags(multi_person=False).keys(),
        )

    def test_keys_is_empty_when_all_none(self) -> None:
        self.assertEqual(SliceTags().keys(), ())


class ClipPredictionTests(unittest.TestCase):
    """ClipPrediction holds model output for one clip."""

    def test_minimal_construction(self) -> None:
        pred = ClipPrediction(
            clip_id="c-01",
            score=0.42,
            model_id="m-01",
            dataset="urfd",
            role=ClipRole.TRAIN,
        )
        self.assertEqual(pred.clip_id, "c-01")
        self.assertEqual(pred.score, 0.42)

    def test_optional_slice_tags(self) -> None:
        pred = ClipPrediction(
            clip_id="c-01",
            score=0.5,
            model_id="m",
            dataset="urfd",
            role=ClipRole.VALIDATE,
            slice_tags=SliceTags(lighting="dim"),
        )
        self.assertIsNotNone(pred.slice_tags)

    def test_empty_clip_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ClipPrediction(
                clip_id="",
                score=0.5,
                model_id="m",
                dataset="urfd",
                role=ClipRole.TRAIN,
            )

    def test_non_numeric_score_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ClipPrediction(
                clip_id="c-01",
                score="high",  # type: ignore[arg-type]
                model_id="m",
                dataset="urfd",
                role=ClipRole.TRAIN,
            )

    def test_frozen_dataclass_blocks_mutation(self) -> None:
        pred = ClipPrediction(
            clip_id="c-01",
            score=0.5,
            model_id="m",
            dataset="urfd",
            role=ClipRole.TRAIN,
        )
        with self.assertRaises(FrozenInstanceError):
            pred.score = 0.99  # type: ignore[misc]


class ClipLabelTests(unittest.TestCase):
    """ClipLabel bundles ground truth with manifest metadata."""

    def test_carries_label_and_provenance(self) -> None:
        label = ClipLabel(
            clip_id="c-01",
            label=FallLabel.FALL,
            dataset="urfd",
            role=ClipRole.TRAIN,
            source_path="datasets/urfd/c-01.mp4",
        )
        self.assertEqual(label.label, FallLabel.FALL)
        self.assertEqual(label.source_path, "datasets/urfd/c-01.mp4")


class EventPredictionStreamTests(unittest.TestCase):
    """EventPredictionStream stores ordered (frame, score) pairs."""

    def test_dense_stream_round_trips(self) -> None:
        stream = EventPredictionStream(
            clip_id="c-01",
            frame_scores=((0, 0.1), (1, 0.2), (2, 0.9)),
            model_id="m-01",
            clip_start_frame=0,
            clip_end_frame=10,
        )
        self.assertEqual(len(stream.frame_scores), 3)

    def test_sparse_stream_is_allowed(self) -> None:
        # A sparse stream with only the trigger frames is a legal
        # representation — eval code decides how to densify.
        stream = EventPredictionStream(
            clip_id="c-02",
            frame_scores=((42, 0.95),),
            model_id="m-01",
        )
        self.assertEqual(stream.frame_scores[0][1], 0.95)

    def test_invalid_pair_shape_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EventPredictionStream(
                clip_id="c-03",
                frame_scores=((0, 0.1, "extra"),),  # type: ignore[arg-type]
                model_id="m",
            )

    def test_non_int_frame_index_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EventPredictionStream(
                clip_id="c-04",
                frame_scores=(("zero", 0.1),),  # type: ignore[arg-type]
                model_id="m",
            )

    def test_non_numeric_score_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EventPredictionStream(
                clip_id="c-05",
                frame_scores=((0, "high"),),  # type: ignore[arg-type]
                model_id="m",
            )


class EventGroundTruthWindowTests(unittest.TestCase):
    """EventGroundTruthWindow is a temporal [start_frame, end_frame] interval."""

    def test_minimal_window(self) -> None:
        win = EventGroundTruthWindow(
            clip_id="c-01",
            start_frame=10,
            end_frame=20,
            label=FallLabel.FALL,
        )
        self.assertEqual(win.start_frame, 10)
        self.assertEqual(win.end_frame, 20)

    def test_single_frame_window_allowed(self) -> None:
        # An instantaneous event is a legal one-frame window.
        win = EventGroundTruthWindow(
            clip_id="c-02",
            start_frame=42,
            end_frame=42,
            label=FallLabel.FALL,
        )
        self.assertEqual(win.start_frame, win.end_frame)

    def test_inverted_window_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EventGroundTruthWindow(
                clip_id="c-03",
                start_frame=20,
                end_frame=10,
                label=FallLabel.FALL,
            )


class MetricResultTests(unittest.TestCase):
    """MetricResult accepts numeric and NotAvailable values; refuses other shapes."""

    def test_numeric_value_passes_through(self) -> None:
        result = MetricResult(name="accuracy", value=0.85)
        self.assertTrue(result.is_available())
        self.assertEqual(result.numeric_value(), 0.85)

    def test_not_available_value_is_handled(self) -> None:
        na = NotAvailable(reason="no detection ground truth", metric_name="map_50")
        result = MetricResult(name="map_50", value=na)
        self.assertFalse(result.is_available())
        # The MetricResult container itself is truthy (the dataclass
        # default); the *value* it carries is falsy. Guard clauses
        # therefore check ``result.is_available()`` or
        # ``bool(result.value)`` — not ``bool(result)``.
        self.assertTrue(bool(result))
        self.assertFalse(bool(result.value))
        with self.assertRaises(ValueError):
            result.numeric_value()

    def test_metric_value_zero_is_kept_distinct_from_not_available(self) -> None:
        # 0.0 is a real metric value; NotAvailable is "could not
        # compute". They must not be confusable.
        zero_result = MetricResult(name="false_alarms_per_hour", value=0.0)
        na_result = MetricResult(
            name="false_alarms_per_hour",
            value=NotAvailable(reason="missing temporal metadata"),
        )
        self.assertTrue(zero_result.is_available())
        self.assertFalse(na_result.is_available())
        self.assertNotEqual(zero_result, na_result)

    def test_metric_value_none_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricResult(name="x", value=None)  # type: ignore[arg-type]

    def test_metric_value_string_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricResult(name="x", value="0.5")  # type: ignore[arg-type]

    def test_empty_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricResult(name="", value=0.5)

    def test_slice_key_carries_through(self) -> None:
        result = MetricResult(
            name="recall",
            value=0.7,
            slice_key=SliceKey("lighting", "low_light"),
        )
        self.assertEqual(result.slice_key, SliceKey("lighting", "low_light"))

    def test_frozen_dataclass_blocks_mutation(self) -> None:
        result = MetricResult(name="x", value=0.5)
        with self.assertRaises(FrozenInstanceError):
            result.value = 0.9  # type: ignore[misc]

    def test_higher_is_better_default_true(self) -> None:
        result = MetricResult(name="x", value=0.5)
        self.assertTrue(result.higher_is_better)

    def test_higher_is_better_can_be_false(self) -> None:
        # delay, false_alarms_per_hour — lower is better.
        result = MetricResult(
            name="delay",
            value=1.2,
            higher_is_better=False,
        )
        self.assertFalse(result.higher_is_better)


if __name__ == "__main__":
    unittest.main()
