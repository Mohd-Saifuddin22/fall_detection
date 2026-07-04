"""Tests for the result-persistence stub (:mod:`evaluation.result_persistence`).

Covers:

- MetricResultPayload round-trips through JSON with both numeric and
  NotAvailable values preserved.
- MetricResultStore writes and reloads a payload from a configurable
  root (no hardcoded Drive paths).
- The structured payload is grep-friendly (run summary sidecar).
- Strict load failures: malformed JSON, wrong format_version, bad
  metric shape, missing run_id metadata.
- The store integrates with the active layout concept
  (``colab.data_mode.DataLayout.metrics``) without hardcoding Drive.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Repo-root sys.path injection (mirrors test_data_mode.py etc).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from evaluation.contracts import MetricResult, SliceKey  # noqa: E402
from evaluation.not_available import (  # noqa: E402
    NOT_AVAILABLE_JSON_KEY,
    NotAvailable,
)
from evaluation.result_persistence import (  # noqa: E402
    RESULT_PAYLOAD_FORMAT_VERSION,
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    EvalRunMetadata,
    MetricResultPayload,
    MetricResultStore,
    encode_value,
    make_default_metadata,
)


class EncodeValueTests(unittest.TestCase):
    """``encode_value`` renders numeric + NotAvailable in their stable shapes."""

    def test_numeric_value_passes_through_as_float(self) -> None:
        # JSON has only one number type — float. We standardise on
        # float so the loader never has to guess between int / float.
        self.assertEqual(encode_value(0.5), 0.5)
        self.assertEqual(encode_value(1), 1.0)

    def test_not_available_renders_as_marker_dict(self) -> None:
        na = NotAvailable(reason="missing temporal metadata", metric_name="delay")
        encoded = encode_value(na)
        self.assertIsInstance(encoded, dict)
        self.assertIs(encoded[NOT_AVAILABLE_JSON_KEY], True)
        self.assertEqual(encoded["reason"], "missing temporal metadata")
        self.assertEqual(encoded["metric_name"], "delay")

    def test_metric_name_arg_stamps_marker_when_not_set(self) -> None:
        # A caller that hands encode_value a NotAvailable without
        # metric_name + supplies the metric_name kwarg → the marker
        # carries the metric_name on disk.
        na = NotAvailable(reason="no detection ground truth")
        encoded = encode_value(na, metric_name="map_50")
        self.assertEqual(encoded["metric_name"], "map_50")

    def test_metric_name_arg_does_not_overwrite_explicit_metric_name(self) -> None:
        # If the NotAvailable already has a metric_name, leave it
        # alone — callers may have set it deliberately.
        na = NotAvailable(reason="x", metric_name="explicit")
        encoded = encode_value(na, metric_name="ignored")
        self.assertEqual(encoded["metric_name"], "explicit")

    def test_bool_rejected(self) -> None:
        # Booleans are not legitimate metric values.
        with self.assertRaises(ValueError):
            encode_value(True)  # type: ignore[arg-type]

    def test_none_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_value(None)

    def test_string_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_value("0.5")  # type: ignore[arg-type]


class MakeDefaultMetadataTests(unittest.TestCase):
    """``make_default_metadata`` stamps the current UTC time and coerces context."""

    def test_default_metadata_uses_current_utc_timestamp(self) -> None:
        meta = make_default_metadata("r-1", "m-1")
        self.assertTrue(meta.created_at.endswith("+00:00") or meta.created_at.endswith("Z"),
                        msg=f"timestamp should be UTC ISO-8601, got {meta.created_at!r}")

    def test_context_string_is_coerced(self) -> None:
        meta = make_default_metadata("r-2", "m-2", context="final_judgement")
        self.assertEqual(meta.context, "final_judgement")

    def test_context_none_becomes_unknown(self) -> None:
        meta = make_default_metadata("r-3", "m-3", context=None)
        self.assertEqual(meta.context, "unknown")

    def test_notes_pass_through_when_set(self) -> None:
        meta = make_default_metadata("r-4", "m-4", notes="smoke run")
        self.assertEqual(meta.notes, "smoke run")


class RoundTripTests(unittest.TestCase):
    """End-to-end JSON round-trip preserves the payload's semantic shape."""

    def _payload(self) -> MetricResultPayload:
        return MetricResultPayload(
            format_version=RESULT_PAYLOAD_FORMAT_VERSION,
            metadata=EvalRunMetadata(
                run_id="r-001",
                model_id="videomae-tiny",
                created_at="2026-07-04T12:00:00+00:00",
                context="evaluation",
                notes="smoke run",
            ),
            metrics=(
                MetricResult(name="accuracy", value=0.85),
                MetricResult(
                    name="map_50",
                    value=NotAvailable(reason="no detection ground truth"),
                ),
                MetricResult(
                    name="recall",
                    value=0.7,
                    slice_key=SliceKey("lighting", "low_light"),
                ),
                MetricResult(
                    name="delay",
                    value=NotAvailable(reason="missing temporal metadata"),
                    higher_is_better=False,
                ),
            ),
        )

    def test_payload_to_json_is_valid_json(self) -> None:
        text = json.dumps(self._payload().to_dict())
        # Round-trip through stdlib JSON to confirm the shape is
        # fully JSON-native (no leftover set / tuple representations).
        json.loads(text)

    def test_payload_round_trips_through_json(self) -> None:
        original = self._payload()
        text = json.dumps(original.to_dict())
        reloaded = MetricResultPayload.from_dict(json.loads(text))
        self.assertEqual(reloaded.format_version, original.format_version)
        self.assertEqual(reloaded.metadata, original.metadata)
        self.assertEqual(len(reloaded.metrics), len(original.metrics))
        for left, right in zip(reloaded.metrics, original.metrics):
            self.assertEqual(left.name, right.name)
            self.assertEqual(left.slice_key, right.slice_key)
            self.assertEqual(left.higher_is_better, right.higher_is_better)
            # ``value`` may be NotAvailable (frozen dataclass) or
            # numeric. Compare semantically rather than with ``==``
            # so the encode-time ``metric_name`` stamp on a marker
            # without one doesn't show up as inequality.
            if isinstance(right.value, NotAvailable):
                self.assertIsInstance(left.value, NotAvailable)
                self.assertEqual(left.value.reason, right.value.reason)
                # The stamp may have set metric_name on the reloaded
                # marker; either it matches the metric's name or it
                # matches the original (which was None / set).
                self.assertIn(left.value.metric_name, (None, left.name, right.value.metric_name))
            else:
                self.assertEqual(float(left.value), float(right.value))

    def test_not_available_round_trips_losslessly(self) -> None:
        payload = self._payload()
        text = json.dumps(payload.to_dict())
        reloaded = MetricResultPayload.from_dict(json.loads(text))
        map_50 = next(m for m in reloaded.metrics if m.name == "map_50")
        self.assertIsInstance(map_50.value, NotAvailable)
        self.assertEqual(map_50.value.reason, "no detection ground truth")

    def test_metric_name_is_preserved_on_marker_round_trip(self) -> None:
        payload = MetricResultPayload(
            format_version=RESULT_PAYLOAD_FORMAT_VERSION,
            metadata=EvalRunMetadata(
                run_id="r-002",
                model_id="m",
                created_at="2026-07-04T12:00:00+00:00",
                context="final_judgement",
            ),
            metrics=(MetricResult(
                name="hota",
                value=NotAvailable(reason="no tracking GT", metric_name="hota"),
            ),),
        )
        text = json.dumps(payload.to_dict())
        reloaded = MetricResultPayload.from_dict(json.loads(text))
        marker = reloaded.metrics[0].value
        self.assertIsInstance(marker, NotAvailable)
        self.assertEqual(marker.metric_name, "hota")

    def test_slice_key_round_trips(self) -> None:
        payload = MetricResultPayload(
            format_version=RESULT_PAYLOAD_FORMAT_VERSION,
            metadata=EvalRunMetadata(
                run_id="r-003",
                model_id="m",
                created_at="2026-07-04T12:00:00+00:00",
                context="evaluation",
            ),
            metrics=(MetricResult(
                name="recall",
                value=0.6,
                slice_key=SliceKey("occlusion", "heavy"),
            ),),
        )
        text = json.dumps(payload.to_dict())
        reloaded = MetricResultPayload.from_dict(json.loads(text))
        self.assertEqual(reloaded.metrics[0].slice_key, SliceKey("occlusion", "heavy"))


class LoadStrictnessTests(unittest.TestCase):
    """Loaders reject malformed / out-of-version payloads loudly."""

    def _good_metadata(self) -> EvalRunMetadata:
        return EvalRunMetadata(
            run_id="r-good",
            model_id="m",
            created_at="2026-07-04T12:00:00+00:00",
            context="evaluation",
        )

    def test_wrong_format_version_rejected(self) -> None:
        # Format-version mismatch is the loader's invitation to crash
        # loudly — better than silently losing a field on schema bump.
        with self.assertRaises(ValueError):
            MetricResultPayload.from_dict({
                "format_version": "999.0",
                "metadata": {
                    "run_id": "x",
                    "model_id": "m",
                    "created_at": "2026-07-04T12:00:00+00:00",
                    "context": "evaluation",
                },
                "metrics": [],
            })

    def test_missing_format_version_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricResultPayload.from_dict({
                "metadata": {"run_id": "x"},
                "metrics": [],
            })

    def test_non_numeric_value_rejected(self) -> None:
        # Boolean is structurally also a numeric in some parsers, but
        # we explicitly reject the shape here.
        with self.assertRaises(ValueError):
            MetricResultPayload.from_dict({
                "format_version": RESULT_PAYLOAD_FORMAT_VERSION,
                "metadata": {
                    "run_id": "x",
                    "model_id": "m",
                    "created_at": "t",
                    "context": "evaluation",
                },
                "metrics": [{"name": "x", "value": True}],
            })

    def test_malformed_metric_record_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricResultPayload.from_dict({
                "format_version": RESULT_PAYLOAD_FORMAT_VERSION,
                "metadata": {
                    "run_id": "x",
                    "model_id": "m",
                    "created_at": "t",
                    "context": "evaluation",
                },
                "metrics": [{"name": "x"}],  # missing value
            })

    def test_non_dict_payload_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricResultPayload.from_dict([1, 2, 3])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MetricResultPayload.from_dict("string")  # type: ignore[arg-type]


class EvalRunMetadataValidationTests(unittest.TestCase):
    """Metadata rejects empty / non-string fields at construction time."""

    def test_empty_run_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvalRunMetadata(
                run_id="",
                model_id="m",
                created_at="t",
                context="evaluation",
            )

    def test_empty_model_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvalRunMetadata(
                run_id="r",
                model_id="",
                created_at="t",
                context="evaluation",
            )


class MetricResultStoreTests(unittest.TestCase):
    """End-to-end save/load against a configurable root directory."""

    def setUp(self) -> None:
        name = tempfile.mkdtemp()
        self._root = Path(name)
        self.addCleanup(_rm_tree, self._root)
        self.store = MetricResultStore(self._root)

    def test_ensure_is_idempotent(self) -> None:
        self.store.ensure()
        self.store.ensure()  # second call must not raise
        self.assertTrue(self._root.is_dir())

    def test_save_creates_run_subdir_and_results_json(self) -> None:
        payload = MetricResultPayload(
            format_version=RESULT_PAYLOAD_FORMAT_VERSION,
            metadata=EvalRunMetadata(
                run_id="r-1",
                model_id="videomae-tiny",
                created_at="2026-07-04T12:00:00+00:00",
                context="evaluation",
            ),
            metrics=(MetricResult(name="accuracy", value=0.85),),
        )
        run_dir = self.store.save(payload.metadata, payload.metrics)
        self.assertTrue((run_dir / RESULTS_FILENAME).exists())
        self.assertTrue((run_dir / SUMMARY_FILENAME).exists())

    def test_save_writes_reloadable_json(self) -> None:
        meta = EvalRunMetadata(
            run_id="r-2",
            model_id="m",
            created_at="2026-07-04T12:00:00+00:00",
            context="final_judgement",
        )
        metrics = (
            MetricResult(name="accuracy", value=0.85),
            MetricResult(
                name="map_50",
                value=NotAvailable(reason="no detection ground truth"),
            ),
            MetricResult(
                name="recall",
                value=0.7,
                slice_key=SliceKey("lighting", "low_light"),
            ),
        )
        run_dir = self.store.save(meta, metrics)
        reloaded = self.store.load("r-2")
        self.assertEqual(reloaded.metadata, meta)
        self.assertEqual(len(reloaded.metrics), 3)
        # Numeric preserved.
        acc = next(m for m in reloaded.metrics if m.name == "accuracy")
        self.assertEqual(acc.numeric_value(), 0.85)
        # NotAvailable preserved as a marker.
        map_50 = next(m for m in reloaded.metrics if m.name == "map_50")
        self.assertIsInstance(map_50.value, NotAvailable)
        self.assertEqual(map_50.value.reason, "no detection ground truth")
        # Slice key preserved.
        recall = next(m for m in reloaded.metrics if m.name == "recall")
        self.assertEqual(recall.slice_key, SliceKey("lighting", "low_light"))

    def test_summary_is_grep_friendly(self) -> None:
        meta = EvalRunMetadata(
            run_id="r-3",
            model_id="videomae-tiny",
            created_at="2026-07-04T12:00:00+00:00",
            context="evaluation",
            notes="smoke run",
        )
        self.store.save(meta, (MetricResult(name="accuracy", value=0.5),))
        summary_text = (self.store.run_dir("r-3") / SUMMARY_FILENAME).read_text(
            encoding="utf-8"
        )
        # Grep-friendly means a single line with identifiable fields.
        first_line = summary_text.splitlines()[0]
        for needle in ("run_id=r-3", "model_id=videomae-tiny", "context=evaluation", "metrics=1/1"):
            self.assertIn(needle, first_line, msg=f"summary missing {needle!r}: {first_line!r}")
        self.assertIn("smoke run", first_line)

    def test_load_reports_missing_file_clearly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store.load("does-not-exist")

    def test_load_rejects_corrupt_json(self) -> None:
        run_dir = self.store.run_dir("r-bad")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / RESULTS_FILENAME).write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.load("r-bad")

    def test_load_rejects_unknown_format_version(self) -> None:
        run_dir = self.store.run_dir("r-version")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / RESULTS_FILENAME).write_text(
            json.dumps({"format_version": "999.0", "metadata": {}, "metrics": []}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            self.store.load("r-version")

    def test_run_id_with_path_separator_rejected(self) -> None:
        meta = EvalRunMetadata(
            run_id="../../etc/passwd",
            model_id="m",
            created_at="t",
            context="evaluation",
        )
        with self.assertRaises(ValueError):
            self.store.save(meta, ())

    def test_run_id_dot_dot_rejected(self) -> None:
        meta = EvalRunMetadata(
            run_id="..",
            model_id="m",
            created_at="t",
            context="evaluation",
        )
        with self.assertRaises(ValueError):
            self.store.save(meta, ())

    def test_overwrite_default_refuses_to_replace_existing_run(self) -> None:
        # Governing behaviour: re-saving the same run_id without an
        # explicit opt-in must NOT silently drop the prior result.
        # Re-running the same id should surface as FileExistsError,
        # not a quiet loss of prior work.
        meta = EvalRunMetadata(
            run_id="r-no-overwrite",
            model_id="m-A",
            created_at="2026-07-04T12:00:00+00:00",
            context="final_judgement",
        )
        self.store.save(meta, (MetricResult(name="accuracy", value=0.5),))
        with self.assertRaises(FileExistsError):
            # Identical run_id, no overwrite=True → refusal.
            self.store.save(meta, (MetricResult(name="accuracy", value=0.9),))

    def test_overwrite_true_replaces_existing_run(self) -> None:
        # Explicit opt-in: callers that deliberately want to replace
        # a run (e.g. test fixtures, re-runs with identical metadata)
        # pass overwrite=True. The store cleans the old payload files
        # but never touches siblings.
        meta = EvalRunMetadata(
            run_id="r-overwrite",
            model_id="m",
            created_at="2026-07-04T12:00:00+00:00",
            context="final_judgement",
        )
        self.store.save(
            meta,
            (MetricResult(name="accuracy", value=0.5),),
            overwrite=True,
        )
        self.store.save(
            meta,
            (MetricResult(name="accuracy", value=0.9),),
            overwrite=True,
        )
        # The latest payload is what was written; reload confirms it.
        reloaded = self.store.load("r-overwrite")
        self.assertEqual(reloaded.metrics[0].numeric_value(), 0.9)

    def test_overwrite_only_clears_named_payload_files(self) -> None:
        # A re-save with overwrite=True must remove the named payload
        # files (results.json, summary.txt) but must not touch any
        # sibling artefacts a caller may have placed in run_dir
        # alongside the payload.
        run_dir = self.store.run_dir("r-keep-sibling")
        run_dir.mkdir(parents=True, exist_ok=True)
        sibling = run_dir / "user_notes.md"
        sibling.write_text("manual analysis", encoding="utf-8")
        meta = EvalRunMetadata(
            run_id="r-keep-sibling",
            model_id="m",
            created_at="2026-07-04T12:00:00+00:00",
            context="final_judgement",
        )
        self.store.save(meta, (), overwrite=True)
        self.assertTrue(sibling.exists(),
                        msg="overwrite=True must not delete caller-managed sibling files.")
        self.assertEqual(sibling.read_text(encoding="utf-8"), "manual analysis")


class MetricsRootIntegrationTests(unittest.TestCase):
    """``MetricResultStore`` accepts the active layout's ``metrics/`` root."""

    def test_accepts_a_path_object(self) -> None:
        store = MetricResultStore(Path("/tmp/fake_metrics"))
        self.assertEqual(store.root, Path("/tmp/fake_metrics"))

    def test_accepts_a_string_path(self) -> None:
        store = MetricResultStore("/tmp/fake_metrics")
        self.assertEqual(store.root, Path("/tmp/fake_metrics"))

    def test_run_dir_uses_metrics_root(self) -> None:
        # The on-disk layout must keep metrics under the supplied root
        # — no hardcoded Drive paths.
        store = MetricResultStore(Path("/tmp/fake_metrics_root"))
        self.assertEqual(store.run_dir("my-run"), Path("/tmp/fake_metrics_root/my-run"))

    def test_round_trip_works_against_layout_style_root(self) -> None:
        name = tempfile.mkdtemp()
        root = Path(name)
        self.addCleanup(_rm_tree, root)
        # Mirrors `layout.metrics` from colab.data_mode.DataLayout —
        # the store sees a plain Path and never knows it's a layout.
        store = MetricResultStore(root / "metrics")
        store.ensure()
        meta = make_default_metadata("r-layout", "m-layout", context="evaluation")
        store.save(meta, (MetricResult(name="accuracy", value=0.9),))
        # Files actually land under <root>/metrics/r-layout/...
        run_dir = root / "metrics" / "r-layout"
        self.assertTrue((run_dir / RESULTS_FILENAME).exists())
        reloaded = store.load("r-layout")
        self.assertEqual(reloaded.metrics[0].numeric_value(), 0.9)


def _rm_tree(path: Path) -> None:
    """Best-effort cleanup helper for temp roots."""
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
