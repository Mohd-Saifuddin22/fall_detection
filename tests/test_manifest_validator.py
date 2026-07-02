"""Tests for the dataset manifest schema + validator.

Run with::

    python -m unittest tests.test_manifest_validator

The tests exercise every individual check inside :func:`validate_manifest`,
plus the happy-path sample manifest. Each test asserts on the public
contract (errors / warnings returned), not on internal helpers.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Allow running this file directly (``python tests/test_manifest_validator.py``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _write_temp(body: str, suffix: str) -> Path:
    """Write ``body`` to a fresh temp file with ``suffix`` and close the handle.

    Windows holds temp files open if you return the path straight from
    ``tempfile.mkstemp``; closing the fd here lets the cleanup hook delete
    it without a ``PermissionError``.
    """
    fd, name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
    except Exception:
        # If write fails, ensure the file is still removed.
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return Path(name)

from data.manifests import (  # noqa: E402
    ClipRecord,
    ClipRole,
    FallLabel,
    FROZEN_VAULT_DATASETS,
    Manifest,
    load_manifest,
    validate_manifest,
)


def _make_clip(
    clip_id: str = "test-01",
    dataset: str = "urfd",
    role: ClipRole = ClipRole.TRAIN,
    label: FallLabel = FallLabel.FALL,
    *,
    source_path: str | None = None,
    subject_id: str | None = None,
) -> ClipRecord:
    """Construct a minimal valid :class:`ClipRecord` for tests.

    ``source_path`` and ``subject_id`` are accepted as keyword overrides
    so tests can deliberately craft path / subject collisions.
    """
    return ClipRecord(
        clip_id=clip_id,
        dataset=dataset,
        role=role,
        label=label,
        source_path=source_path or f"datasets/{dataset}/{clip_id}.mp4",
        subject_id=subject_id,
    )


class SampleManifestTests(unittest.TestCase):
    """The shipped sample manifest must validate with zero errors."""

    def test_sample_manifest_validates_cleanly(self) -> None:
        path = _REPO_ROOT / "data" / "manifests" / "sample_manifest.yaml"
        manifest = load_manifest(path)
        report = validate_manifest(manifest)
        self.assertTrue(
            report.is_valid,
            msg=f"Sample manifest must validate, got errors:\n"
                + "\n".join(report.errors),
        )


class RequiredFieldsTests(unittest.TestCase):
    """The required-fields check must surface every missing field."""

    def test_empty_manifest_is_valid_but_uninteresting(self) -> None:
        report = validate_manifest(Manifest(schema_version="1.0"))
        self.assertTrue(report.is_valid)


class DisjointTrainValidateTests(unittest.TestCase):
    """A clip appearing in both train and validate must be flagged."""

    def test_clip_in_both_train_and_validate_is_an_error(self) -> None:
        manifest = Manifest(
            schema_version="1.0",
            clips=[
                _make_clip("dup-01", role=ClipRole.TRAIN),
                _make_clip("dup-01", role=ClipRole.VALIDATE),
            ],
        )
        report = validate_manifest(manifest)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("dup-01" in err and "train" in err and "validate" in err for err in report.errors),
            msg=f"Expected a train/validate conflict for dup-01; got {report.errors}",
        )

    def test_clip_in_train_and_debug_is_allowed(self) -> None:
        # Clip IDs are unique per design — each row is one clip with one role.
        # The cross-listed datasets (le2i, gmdcsa24) get separate clip IDs in
        # each role; same DATASET across roles is fine, same CLIP_ID is not.
        manifest = Manifest(
            schema_version="1.0",
            clips=[
                _make_clip("le2i-train-01", dataset="le2i", role=ClipRole.TRAIN),
                _make_clip("le2i-debug-01", dataset="le2i", role=ClipRole.DEBUG),
            ],
        )
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Unexpected errors: {report.errors}")


class FrozenVaultIsolationTests(unittest.TestCase):
    """Vault datasets may ONLY appear in ``frozen_unseen_test``."""

    def test_vault_dataset_in_train_is_an_error(self) -> None:
        manifest = Manifest(
            schema_version="1.0",
            clips=[_make_clip("v-01", dataset="omnifall", role=ClipRole.TRAIN)],
        )
        report = validate_manifest(manifest)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("omnifall" in err and "frozen-vault" in err for err in report.errors),
            msg=f"Expected vault-isolation error for omnifall; got {report.errors}",
        )

    def test_non_vault_dataset_in_vault_role_is_an_error(self) -> None:
        manifest = Manifest(
            schema_version="1.0",
            clips=[_make_clip("v-02", dataset="urfd", role=ClipRole.FROZEN_UNSEEN_TEST)],
        )
        report = validate_manifest(manifest)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("reserved for vault datasets" in err for err in report.errors),
            msg=f"Expected role-reservation error; got {report.errors}",
        )

    def test_all_four_vault_datasets_are_enumerated(self) -> None:
        expected = {"omnifall", "caucafall", "mcfd", "fallvision"}
        self.assertEqual(expected, FROZEN_VAULT_DATASETS)


class DuplicateClipIdTests(unittest.TestCase):
    """Duplicate ``clip_id`` values must be reported clearly."""

    def test_duplicate_clip_id_with_same_role_is_an_error(self) -> None:
        manifest = Manifest(
            schema_version="1.0",
            clips=[
                _make_clip("dup-01", role=ClipRole.TRAIN),
                _make_clip("dup-01", role=ClipRole.TRAIN),
            ],
        )
        report = validate_manifest(manifest)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("duplicate clip_id 'dup-01'" in err for err in report.errors),
            msg=f"Expected duplicate-id error for dup-01; got {report.errors}",
        )


class RoleAndLabelEnforcementTests(unittest.TestCase):
    """The loader rejects unknown roles / labels at parse time."""

    def _write_yaml(self, body: str) -> Path:
        tmp = _write_temp(body, suffix=".yaml")
        self.addCleanup(tmp.unlink)
        return tmp

    def test_unknown_role_is_rejected(self) -> None:
        path = self._write_yaml(
            "schema_version: '1.0'\n"
            "clips:\n"
            "  - clip_id: x-01\n"
            "    dataset: urfd\n"
            "    role: holdout\n"
            "    label: fall\n"
            "    source_path: x.mp4\n"
        )
        with self.assertRaises(ValueError):
            load_manifest(path)

    def test_unknown_label_is_rejected(self) -> None:
        path = self._write_yaml(
            "schema_version: '1.0'\n"
            "clips:\n"
            "  - clip_id: x-02\n"
            "    dataset: urfd\n"
            "    role: train\n"
            "    label: maybe\n"
            "    source_path: x.mp4\n"
        )
        with self.assertRaises(ValueError):
            load_manifest(path)

    def test_missing_required_field_is_rejected(self) -> None:
        path = self._write_yaml(
            "schema_version: '1.0'\n"
            "clips:\n"
            "  - clip_id: x-03\n"
            "    dataset: urfd\n"
            "    role: train\n"
            "    label: fall\n"
            "    # source_path deliberately missing\n"
        )
        with self.assertRaises(ValueError):
            load_manifest(path)


class LoaderFormatTests(unittest.TestCase):
    """The loader accepts both YAML and JSON; rejects unknown extensions."""

    def test_json_round_trip(self) -> None:
        clip = _make_clip("json-01", role=ClipRole.DEBUG)
        manifest = Manifest(schema_version="1.1", clips=[clip])
        path = _write_temp(json.dumps(manifest.to_serialisable()), suffix=".json")
        self.addCleanup(path.unlink)

        reloaded = load_manifest(path)
        self.assertEqual(len(reloaded.clips), 1)
        self.assertEqual(reloaded.clips[0].clip_id, "json-01")
        self.assertEqual(reloaded.clips[0].role, ClipRole.DEBUG)

    def test_unsupported_extension_raises(self) -> None:
        path = _write_temp("clip_id\nx\n", suffix=".csv")
        self.addCleanup(path.unlink)
        with self.assertRaises(ValueError):
            load_manifest(path)


class SourcePathDisjointTests(unittest.TestCase):
    """The same raw video file must not appear in both train and validate.

    Distinct ``clip_id``s with the same ``source_path`` are the realistic
    failure mode this guards against — different annotators writing
    different clip IDs for overlapping windows of the same recording.
    """

    def test_same_source_path_in_train_and_validate_is_an_error(self) -> None:
        manifest = Manifest(
            schema_version="1.1",
            clips=[
                _make_clip("train-clip-A", role=ClipRole.TRAIN,
                           source_path="datasets/up_fall/room1.mp4"),
                _make_clip("validate-clip-B", role=ClipRole.VALIDATE,
                           source_path="datasets/up_fall/room1.mp4"),
            ],
        )
        report = validate_manifest(manifest)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("room1.mp4" in err and "raw video file" in err for err in report.errors),
            msg=f"Expected source_path conflict for room1.mp4; got {report.errors}",
        )

    def test_same_source_path_in_train_and_debug_is_allowed(self) -> None:
        # Debug clips may legally be cut from the same video used in train
        # — the debug tier is exactly for plumbing tests on real material.
        manifest = Manifest(
            schema_version="1.1",
            clips=[
                _make_clip("train-A", role=ClipRole.TRAIN,
                           source_path="datasets/le2i/scene1.mp4"),
                _make_clip("debug-A", role=ClipRole.DEBUG,
                           source_path="datasets/le2i/scene1.mp4"),
            ],
        )
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Unexpected errors: {report.errors}")


class SubjectIdDisjointTests(unittest.TestCase):
    """The same person must not appear in both train and validate.

    A subject showing up in both splits lets the model memorise
    subject-specific cues rather than learning the fall pattern.
    """

    def test_same_subject_in_train_and_validate_is_an_error(self) -> None:
        manifest = Manifest(
            schema_version="1.1",
            clips=[
                _make_clip("train-A", role=ClipRole.TRAIN,
                           subject_id="subject-07",
                           source_path="datasets/up_fall/clip-A.mp4"),
                _make_clip("validate-B", role=ClipRole.VALIDATE,
                           subject_id="subject-07",
                           source_path="datasets/up_fall/clip-B.mp4"),
            ],
        )
        report = validate_manifest(manifest)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("subject-07" in err and "same person" in err for err in report.errors),
            msg=f"Expected subject_id conflict for subject-07; got {report.errors}",
        )

    def test_different_subjects_in_train_and_validate_are_allowed(self) -> None:
        manifest = Manifest(
            schema_version="1.1",
            clips=[
                _make_clip("train-A", role=ClipRole.TRAIN,
                           subject_id="subject-01",
                           source_path="datasets/up_fall/clip-A.mp4"),
                _make_clip("validate-B", role=ClipRole.VALIDATE,
                           subject_id="subject-02",
                           source_path="datasets/up_fall/clip-B.mp4"),
            ],
        )
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Unexpected errors: {report.errors}")

    def test_missing_subject_id_on_either_side_does_not_block_validation(self) -> None:
        # When subject_id is None on at least one side of the comparison
        # there is nothing to check, so the rule must NOT block. We surface
        # this as a caveat in context.txt instead — population quality
        # determines how loud the check fires.
        manifest = Manifest(
            schema_version="1.1",
            clips=[
                _make_clip("train-A", role=ClipRole.TRAIN,
                           subject_id="subject-01",
                           source_path="datasets/up_fall/clip-A.mp4"),
                _make_clip("validate-B", role=ClipRole.VALIDATE,
                           subject_id=None,
                           source_path="datasets/up_fall/clip-B.mp4"),
                _make_clip("validate-C", role=ClipRole.VALIDATE,
                           subject_id=None,
                           source_path="datasets/up_fall/clip-C.mp4"),
            ],
        )
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Unexpected errors: {report.errors}")


class SliceTagTests(unittest.TestCase):
    """Slice tags are optional; known values pass, unknown values warn."""

    def _make_tagged_clip(
        self,
        clip_id: str,
        role: ClipRole,
        *,
        lighting: str | None = None,
        occlusion: str | None = None,
        multi_person: bool | None = None,
        action_confuser: str | None = None,
    ) -> ClipRecord:
        return ClipRecord(
            clip_id=clip_id,
            dataset="urfd",
            role=role,
            label=FallLabel.FALL,
            source_path=f"datasets/urfd/{clip_id}.mp4",
            lighting=lighting,
            occlusion=occlusion,
            multi_person=multi_person,
            action_confuser=action_confuser,
        )

    def _no_slice_tag_warnings(self, warnings: tuple[str, ...]) -> None:
        """Helper: assert no warning string mentions a slice-tag keyword.

        Other warnings (e.g. cross-listed-dataset warnings) are allowed to
        coexist — the test only cares about slice-tag noise here.
        """
        offenders = [w for w in warnings if any(
            kw in w for kw in ("lighting", "occlusion", "action_confuser")
        )]
        self.assertEqual(offenders, [],
                         msg=f"Unexpected slice-tag warnings: {offenders}")

    def test_known_slice_tags_do_not_warn(self) -> None:
        manifest = Manifest(
            schema_version="1.1",
            clips=[
                self._make_tagged_clip(
                    "tagged-01", ClipRole.TRAIN,
                    lighting="daylight", occlusion="partial",
                    multi_person=True, action_confuser="sleeping",
                ),
            ],
        )
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Unexpected errors: {report.errors}")
        self._no_slice_tag_warnings(report.warnings)

    def test_unknown_lighting_value_warns_but_does_not_fail(self) -> None:
        manifest = Manifest(
            schema_version="1.1",
            clips=[
                self._make_tagged_clip(
                    "tagged-02", ClipRole.TRAIN, lighting="neon",
                ),
            ],
        )
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Unexpected errors: {report.errors}")
        self.assertTrue(
            any("lighting 'neon'" in w for w in report.warnings),
            msg=f"Expected lighting warning; got {report.warnings}",
        )

    def test_unknown_action_confuser_value_warns_but_does_not_fail(self) -> None:
        manifest = Manifest(
            schema_version="1.1",
            clips=[
                self._make_tagged_clip(
                    "tagged-03", ClipRole.TRAIN, action_confuser="yoga",
                ),
            ],
        )
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Unexpected errors: {report.errors}")
        self.assertTrue(
            any("action_confuser 'yoga'" in w for w in report.warnings),
            msg=f"Expected action_confuser warning; got {report.warnings}",
        )

    def test_missing_slice_tags_do_not_warn(self) -> None:
        # Placeholder rows are legal until Issue 004 starts aggregating.
        manifest = Manifest(
            schema_version="1.1",
            clips=[_make_clip("tagged-04", role=ClipRole.TRAIN)],
        )
        report = validate_manifest(manifest)
        self.assertTrue(report.is_valid, msg=f"Unexpected errors: {report.errors}")
        self._no_slice_tag_warnings(report.warnings)


class SchemaVersionTests(unittest.TestCase):
    """Unsupported schema versions are rejected loudly."""

    def test_unsupported_schema_version_is_an_error(self) -> None:
        manifest = Manifest(
            schema_version="99.0",
            clips=[_make_clip("v-01")],
        )
        report = validate_manifest(manifest)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("schema_version '99.0'" in err for err in report.errors),
            msg=f"Expected schema-version error; got {report.errors}",
        )

    def test_supported_schema_versions_pass_version_check(self) -> None:
        for version in ("1.0", "1.1"):
            with self.subTest(version=version):
                manifest = Manifest(schema_version=version, clips=[_make_clip(f"v-{version}")])
                report = validate_manifest(manifest)
                self.assertNotIn(
                    "schema_version", " ".join(report.errors),
                    msg=f"version {version} should not trigger schema-version error; "
                        f"got {report.errors}",
                )


if __name__ == "__main__":
    unittest.main()