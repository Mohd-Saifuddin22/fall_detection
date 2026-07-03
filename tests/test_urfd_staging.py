"""Unit tests for URFD folder-name parsing and manifest construction.

No Kaggle / GPU dependency. Tests operate on synthetic staged folders.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.build_urfd_manifest import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    build_clip_id,
    build_clip_record,
    build_urfd_manifest,
    write_urfd_manifest,
)
from data.manifests import (  # noqa: E402
    ClipRole,
    FallLabel,
    load_manifest,
    validate_manifest,
)
from data.stage_urfd import (  # noqa: E402
    ALLOWED_KAGGLE_SLUG,
    STAGING_MARKER_FILENAME,
    UrfdStagingResult,
    is_urfd_already_staged,
    parse_urfd_folder_name,
    stage_urfd_from_kaggle,
)


class ParseUrfdFolderNameTests(unittest.TestCase):
    """Folder-name → :class:`StagedClipFolder` mapping."""

    def test_fall_folder_parses_to_fall_label(self) -> None:
        parsed = parse_urfd_folder_name("fall-01-cam0")
        assert parsed is not None
        self.assertEqual(parsed.label, "fall")
        self.assertEqual(parsed.camera, "cam0")
        self.assertEqual(parsed.clip_sequence, "01")

    def test_adl_folder_parses_to_no_fall_label(self) -> None:
        parsed = parse_urfd_folder_name("adl-02-cam1")
        assert parsed is not None
        self.assertEqual(parsed.label, "no_fall")
        self.assertEqual(parsed.camera, "cam1")
        self.assertEqual(parsed.clip_sequence, "02")

    def test_unknown_folder_returns_none(self) -> None:
        self.assertIsNone(parse_urfd_folder_name("random-stuff"))
        self.assertIsNone(parse_urfd_folder_name(""))
        self.assertIsNone(parse_urfd_folder_name("fall"))  # no sequence number

    def test_case_insensitive(self) -> None:
        parsed = parse_urfd_folder_name("Fall-01-CAM0")
        assert parsed is not None
        self.assertEqual(parsed.label, "fall")
        self.assertEqual(parsed.camera, "cam0")


class IsUrfdAlreadyStagedTests(unittest.TestCase):
    """Idempotency check for the staging script."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_empty_root_is_not_staged(self) -> None:
        staged = self.root / "datasets" / "urfd"
        staged.mkdir(parents=True)
        self.assertFalse(is_urfd_already_staged(staged))

    def test_marker_file_alone_is_not_enough(self) -> None:
        staged = self.root / "datasets" / "urfd"
        staged.mkdir(parents=True)
        (staged / STAGING_MARKER_FILENAME).write_text("x")
        # Marker present but no clip folders — still considered not staged.
        self.assertFalse(is_urfd_already_staged(staged))

    def test_marker_plus_real_folder_is_staged(self) -> None:
        staged = self.root / "datasets" / "urfd"
        staged.mkdir(parents=True)
        (staged / STAGING_MARKER_FILENAME).write_text("x")
        (staged / "fall-01-cam0").mkdir()
        self.assertTrue(is_urfd_already_staged(staged))


class StageUrfdSlugTests(unittest.TestCase):
    """Slug whitelist is enforced."""

    def test_default_slug_is_the_only_allowed_one(self) -> None:
        self.assertEqual(ALLOWED_KAGGLE_SLUG, "tanmaydacha/urfd-dataset")

    def test_arbitrary_slug_is_rejected(self) -> None:
        # We never let stage_urfd_from_kaggle touch a non-whitelisted slug.
        # We can't easily reach the kagglehub path without credentials,
        # so we just confirm the slug check fires before any download.
        with self.assertRaises(RuntimeError):
            stage_urfd_from_kaggle(Path("/tmp"), kaggle_slug="something-else")


class BuildClipIdTests(unittest.TestCase):
    """Clip-ID construction is stable."""

    def test_fall_clip_id(self) -> None:
        self.assertEqual(
            build_clip_id(parse_urfd_folder_name("fall-01-cam0")),  # type: ignore[arg-type]
            "urfd-debug-fall-01-cam0",
        )

    def test_adl_clip_id(self) -> None:
        self.assertEqual(
            build_clip_id(parse_urfd_folder_name("adl-02-cam1")),  # type: ignore[arg-type]
            "urfd-debug-adl-02-cam1",
        )


class BuildClipRecordTests(unittest.TestCase):
    """The manifest row carries the right label / role / dataset."""

    def _make_folder(self, name: str) -> "StagedClipFolder":  # type: ignore[name-defined]
        from data.stage_urfd import StagedClipFolder
        return StagedClipFolder(
            absolute_path=Path("/tmp/fake"),
            folder_name=name,
            label="fall" if name.startswith("fall") else "no_fall",
            camera="cam0",
            clip_sequence="01",
        )

    def test_fall_folder_produces_fall_record(self) -> None:
        record = build_clip_record(self._make_folder("fall-01-cam0"))
        self.assertEqual(record.dataset, "urfd")
        self.assertEqual(record.role, ClipRole.DEBUG)
        self.assertEqual(record.label, FallLabel.FALL)
        self.assertEqual(record.clip_id, "urfd-debug-fall-01-cam0")
        self.assertEqual(record.source_path, "datasets/urfd/fall-01-cam0")
        self.assertIn("camera=cam0", record.notes or "")

    def test_adl_folder_produces_no_fall_record(self) -> None:
        record = build_clip_record(self._make_folder("adl-01-cam0"))
        self.assertEqual(record.label, FallLabel.NO_FALL)


class BuildUrfdManifestTests(unittest.TestCase):
    """End-to-end: a staged tree → a valid :class:`Manifest`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.staged = Path(self._tmp.name) / "datasets" / "urfd"
        self.staged.mkdir(parents=True)
        for name in ("fall-01-cam0", "fall-01-cam1", "adl-01-cam0", "adl-01-cam1"):
            (self.staged / name).mkdir()
        (self.staged / STAGING_MARKER_FILENAME).write_text("staged\n")

    def test_builds_valid_manifest_with_one_row_per_folder(self) -> None:
        manifest = build_urfd_manifest(self.staged)
        self.assertEqual(len(manifest.clips), 4)
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Errors: {report.errors}")

    def test_falls_and_adls_have_correct_labels(self) -> None:
        manifest = build_urfd_manifest(self.staged)
        labels = {c.clip_id: c.label for c in manifest.clips}
        self.assertEqual(labels["urfd-debug-fall-01-cam0"], FallLabel.FALL)
        self.assertEqual(labels["urfd-debug-adl-01-cam0"], FallLabel.NO_FALL)

    def test_empty_staged_tree_raises(self) -> None:
        empty = self.staged.parent / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            build_urfd_manifest(empty)

    def test_uses_schema_1_1(self) -> None:
        manifest = build_urfd_manifest(self.staged)
        self.assertEqual(manifest.schema_version, MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest.schema_version, "1.1")

    def test_write_and_reload_round_trips_through_yaml(self) -> None:
        manifest = build_urfd_manifest(self.staged)
        out_path = self.staged.parent / "manifest.yaml"
        write_urfd_manifest(manifest, out_path)
        reloaded = load_manifest(out_path)
        self.assertEqual(len(reloaded.clips), 4)
        report = validate_manifest(reloaded)
        self.assertTrue(report.is_valid, msg=f"Errors: {report.errors}")


class PlaceholderManifestTests(unittest.TestCase):
    """The shipped placeholder URFD manifest must validate cleanly."""

    def test_placeholder_validates(self) -> None:
        path = _REPO_ROOT / "data" / "manifests" / "urfd_debug_placeholder.yaml"
        manifest = load_manifest(path)
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Errors: {report.errors}")
        # Every clip is debug / urfd / has a label.
        for clip in manifest.clips:
            self.assertEqual(clip.dataset, "urfd")
            self.assertEqual(clip.role, ClipRole.DEBUG)
            self.assertIn(clip.label, (FallLabel.FALL, FallLabel.NO_FALL))


if __name__ == "__main__":
    unittest.main()