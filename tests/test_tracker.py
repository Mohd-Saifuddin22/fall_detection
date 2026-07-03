"""Unit tests for the perception tracker wrapper.

Covers the **contract** (strict yolo26m + ByteTrack, fallback shapes,
synthetic result flattening) without doing live YOLO inference — those
tests would need a GPU. The full pipeline is exercised by the Colab
notebook.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.tracker import (  # noqa: E402
    PERSON_CLASS_ID,
    REQUIRED_MODEL,
    REQUIRED_TRACKER,
    DetectionBox,
    PerceptionRunResult,
    TrackerConfig,
    UnsupportedModelError,
    _apply_fallback_kwargs,
    _build_tracker_kwargs,
    _format_fallback_used,
    _flatten_result,
    _summarise_tracks,
    assert_required_model_available,
)


class RequiredConstantsTests(unittest.TestCase):
    """The constants Issue 002 pins must be the ones the rules demand."""

    def test_required_model_is_yolo26m(self) -> None:
        self.assertEqual(REQUIRED_MODEL, "yolo26m")

    def test_required_tracker_is_bytetrack(self) -> None:
        self.assertEqual(REQUIRED_TRACKER, "bytetrack.yaml")

    def test_person_class_id_is_zero(self) -> None:
        self.assertEqual(PERSON_CLASS_ID, 0)


class StrictModelCheckTests(unittest.TestCase):
    """The model check must fail loud on anything that isn't yolo26m."""

    def test_non_yolo26_model_is_rejected_before_load(self) -> None:
        with self.assertRaises(UnsupportedModelError):
            assert_required_model_available("yolo11n")

    def test_yolo26m_passes_when_ultralytics_can_load_it(self) -> None:
        # Skip the live download in CI: importing YOLO + probing a model
        # pulls weights over the network (~42 MB), which is too slow for
        # every test run. The contract we care about — "the model name
        # check accepts yolo26m" — is exercised here via the public
        # function, with the YOLO constructor patched so no download
        # happens.
        from unittest.mock import patch
        try:
            with patch("ultralytics.YOLO") as mock_yolo:
                mock_yolo.return_value = object()
                assert_required_model_available("yolo26m")
                mock_yolo.assert_called_once_with("yolo26m")
        except UnsupportedModelError:
            self.skipTest("ultralytics not available in this environment")


class TrackerConfigTests(unittest.TestCase):
    """Configuration object defaults + dataclass freeze."""

    def test_defaults_match_issue_002_baseline(self) -> None:
        config = TrackerConfig()
        self.assertEqual(config.model_name, REQUIRED_MODEL)
        self.assertEqual(config.tracker_config, REQUIRED_TRACKER)
        self.assertTrue(config.person_class_id == 0)
        self.assertEqual(config.confidence_threshold, 0.25)
        self.assertIsNone(config.fallback_track_low_thresh)
        self.assertIsNone(config.fallback_tracker)
        self.assertIsNone(config.fallback_end2end)

    def test_config_is_frozen(self) -> None:
        config = TrackerConfig()
        with self.assertRaises(Exception):  # FrozenInstanceError subclasses Exception
            config.confidence_threshold = 0.99  # type: ignore[misc]


class BuildTrackerKwargsTests(unittest.TestCase):
    """The kwargs dict must always carry tracker + persist + classes + conf."""

    def test_baseline_kwargs_contain_persist_and_bytetrack(self) -> None:
        kwargs = _build_tracker_kwargs(TrackerConfig())
        self.assertEqual(kwargs["tracker"], "bytetrack.yaml")
        self.assertTrue(kwargs["persist"])
        self.assertEqual(kwargs["classes"], [PERSON_CLASS_ID])
        self.assertEqual(kwargs["conf"], 0.25)

    def test_fallback_tracker_overrides_default(self) -> None:
        config = TrackerConfig(fallback_tracker="botsort.yaml")
        kwargs = _build_tracker_kwargs(config)
        self.assertEqual(kwargs["tracker"], "botsort.yaml")


class ApplyFallbackKwargsTests(unittest.TestCase):
    """Fallback knobs translate to the right ``cfg`` entries — and the wrong
    ones don't get silently wired at all."""

    def test_track_low_thresh_adds_cfg_entry(self) -> None:
        config = TrackerConfig(fallback_track_low_thresh=0.1)
        merged = _apply_fallback_kwargs(_build_tracker_kwargs(config), config)
        self.assertEqual(merged["cfg"], {"track_low_thresh": 0.1})

    def test_end2end_is_NOT_auto_wired_via_cfg(self) -> None:
        # Issue 002 review: ``end2end`` is a model/runtime argument, not
        # a tracker config key. Auto-wiring it via ``cfg=`` would be
        # silently ignored by ByteTrack AND would mislead BoT-SORT into
        # treating it as a config override. We must NOT forward it.
        config = TrackerConfig(fallback_tracker="botsort.yaml", fallback_end2end=False)
        merged = _apply_fallback_kwargs(_build_tracker_kwargs(config), config)
        # Track_low_thresh not set, so cfg should be absent entirely.
        self.assertNotIn("cfg", merged)

    def test_end2end_alone_does_not_create_cfg(self) -> None:
        config = TrackerConfig(fallback_end2end=False)
        merged = _apply_fallback_kwargs(_build_tracker_kwargs(config), config)
        self.assertNotIn("cfg", merged)


class FormatFallbackUsedTests(unittest.TestCase):
    """The human-readable fallback record reflects auto-wired vs manual-only levers."""

    def test_no_fallbacks_returns_none(self) -> None:
        self.assertIsNone(_format_fallback_used(TrackerConfig()))

    def test_auto_wired_levers_appear_plain(self) -> None:
        config = TrackerConfig(
            fallback_track_low_thresh=0.1,
            fallback_tracker="botsort.yaml",
        )
        text = _format_fallback_used(config)
        assert text is not None
        self.assertIn("track_low_thresh=0.1", text)
        self.assertIn("tracker=botsort.yaml", text)
        # No "manual intervention required" annotation for auto-wired levers.
        self.assertNotIn("manual intervention", text)

    def test_end2end_is_marked_as_manual_intervention(self) -> None:
        config = TrackerConfig(fallback_end2end=False)
        text = _format_fallback_used(config)
        assert text is not None
        self.assertIn("end2end=False", text)
        self.assertIn("manual intervention required", text)


class FlattenResultTests(unittest.TestCase):
    """The result-row extractor filters by class and surfaces decode failures.

    Each call returns ``(rows, decoded_ok)``; ``decoded_ok=False`` means
    the Ultralytics result couldn't be interpreted and the caller should
    increment its ``decode_failures`` counter (Issue 002 review).
    """

    class _FakeBoxes:
        def __init__(self, xyxy, conf, cls, ids):
            self.xyxy = xyxy
            self.conf = conf
            self.cls = cls
            self.id = ids

    class _FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes

    def _make_result(self):
        import numpy as np
        return self._FakeResult(self._FakeBoxes(
            xyxy=np.array([[0.0, 0.0, 10.0, 20.0], [1.0, 1.0, 5.0, 5.0]]),
            conf=np.array([0.9, 0.5]),
            cls=np.array([0, 1]),  # second one is not a person
            ids=np.array([7, 8]),
        ))

    def test_filters_non_person_classes(self) -> None:
        rows, decoded_ok = _flatten_result(
            self._make_result(), frame_index=3, person_class_id=0,
        )
        self.assertTrue(decoded_ok)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].track_id, 7)
        self.assertEqual(rows[0].frame_index, 3)
        self.assertEqual(rows[0].cls_id, 0)

    def test_missing_boxes_is_a_decode_failure_not_a_silent_zero(self) -> None:
        rows, decoded_ok = _flatten_result(object(), frame_index=0, person_class_id=0)
        self.assertFalse(decoded_ok, msg="missing boxes must be flagged as decode failure")
        self.assertEqual(rows, [])

    def test_boxes_with_missing_attributes_is_a_decode_failure(self) -> None:
        # Boxes object without ``xyxy`` / ``conf`` / ``cls`` — simulates
        # an Ultralytics version bump that renamed an attribute.
        class _IncompleteBoxes:
            pass

        result = self._FakeResult(_IncompleteBoxes())
        rows, decoded_ok = _flatten_result(result, frame_index=0, person_class_id=0)
        self.assertFalse(decoded_ok)
        self.assertEqual(rows, [])

    def test_boxes_with_no_persons_is_decoded_ok_with_empty_rows(self) -> None:
        # A frame with no person detections is a valid (decoded) result
        # that happens to be empty — different from a decode failure.
        import numpy as np

        class _Boxes:
            xyxy = np.empty((0, 4))
            conf = np.empty((0,))
            cls = np.empty((0,))
            id = np.empty((0,))

        rows, decoded_ok = _flatten_result(self._FakeResult(_Boxes()), 0, 0)
        self.assertTrue(decoded_ok)
        self.assertEqual(rows, [])


class DecodeFailuresCounterTests(unittest.TestCase):
    """The :class:`PerceptionRunResult` exposes ``decode_failures`` so the
    artefact JSON carries the signal — no more silent zero-detection runs.
    """

    def test_default_decode_failures_is_zero(self) -> None:
        run = PerceptionRunResult(clip_id="x", source_folder="",
                                  config=TrackerConfig())
        self.assertEqual(run.decode_failures, 0)

    def test_decode_failures_can_be_incremented(self) -> None:
        run = PerceptionRunResult(clip_id="x", source_folder="",
                                  config=TrackerConfig())
        run.decode_failures += 1
        run.decode_failures += 1
        self.assertEqual(run.decode_failures, 2)

    def test_decode_failures_appears_in_run_meta_artifact(self) -> None:
        # End-to-end: increment the counter on a run, write the artefact
        # JSON, and confirm the value reaches disk. This is the signal a
        # human reviewer needs to know the tracker decoded everything
        # cleanly (or, conversely, that something is wrong).
        import json
        import tempfile
        from pathlib import Path
        from perception.artifacts import write_perception_artifacts
        from perception.report import build_track_continuity_report

        run = PerceptionRunResult(clip_id="x", source_folder="",
                                  config=TrackerConfig())
        run.decode_failures = 7
        report = build_track_continuity_report(run)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x"
            paths = write_perception_artifacts(out, run, report)
            meta = json.loads(paths["run_meta"].read_text(encoding="utf-8"))
        self.assertEqual(meta["decode_failures"], 7)


class SummariseTracksTests(unittest.TestCase):
    """Per-track summary math is correct and skips untracked detections."""

    def test_groups_by_track_id_and_sorts_by_frame(self) -> None:
        detections = [
            DetectionBox(frame_index=5, track_id=2, cls_id=0, confidence=0.9,
                         x_min=0, y_min=0, x_max=10, y_max=10),
            DetectionBox(frame_index=2, track_id=2, cls_id=0, confidence=0.8,
                         x_min=0, y_min=0, x_max=10, y_max=10),
            DetectionBox(frame_index=3, track_id=1, cls_id=0, confidence=0.7,
                         x_min=0, y_min=0, x_max=10, y_max=10),
            DetectionBox(frame_index=4, track_id=None, cls_id=0, confidence=0.6,
                         x_min=0, y_min=0, x_max=10, y_max=10),
        ]
        summaries = _summarise_tracks(detections)
        self.assertEqual([s.track_id for s in summaries], [1, 2])
        track_2 = summaries[1]
        self.assertEqual(track_2.frame_indices, (2, 5))
        self.assertEqual(track_2.first_frame, 2)
        self.assertEqual(track_2.last_frame, 5)
        self.assertEqual(track_2.length, 2)

    def test_empty_input_yields_empty_summaries(self) -> None:
        self.assertEqual(_summarise_tracks([]), [])


class PerceptionRunResultTests(unittest.TestCase):
    """FPS / latency accessors behave on degenerate inputs."""

    def test_fps_is_zero_when_no_frames(self) -> None:
        run = PerceptionRunResult(clip_id="x", source_folder="",
                                  config=TrackerConfig(), frame_count=0)
        self.assertEqual(run.fps, 0.0)
        self.assertEqual(run.latency_ms_per_frame, 0.0)

    def test_fps_and_latency_compute_from_elapsed_seconds(self) -> None:
        run = PerceptionRunResult(clip_id="x", source_folder="",
                                  config=TrackerConfig(), frame_count=100,
                                  elapsed_seconds=2.0)
        self.assertAlmostEqual(run.fps, 50.0, places=5)
        self.assertAlmostEqual(run.latency_ms_per_frame, 20.0, places=5)


if __name__ == "__main__":
    unittest.main()