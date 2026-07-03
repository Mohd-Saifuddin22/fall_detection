"""Unit tests for :mod:`colab.data_mode`.

Covers the LOCAL / DRIVE mode split that lets the active pipeline
avoid Drive FUSE small-file reads while still persisting outputs to
Drive.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from colab.data_mode import (  # noqa: E402
    DEFAULT_DATA_MODE,
    DEFAULT_DRIVE_ROOT,
    DEFAULT_LOCAL_DATA_ROOT,
    DATA_MODE_ENV_VAR,
    DRIVE_ROOT_ENV_VAR,
    LOCAL_DATA_ROOT_ENV_VAR,
    DataLayout,
    DataMode,
    describe_layout,
    resolve_data_layout,
    select_active_paths,
)


class DataModeEnumTests(unittest.TestCase):
    """The enum values are stable + lowercase."""

    def test_default_mode_is_local(self) -> None:
        self.assertEqual(DEFAULT_DATA_MODE, DataMode.LOCAL)

    def test_mode_values_are_lowercase(self) -> None:
        self.assertEqual(DataMode.LOCAL.value, "local")
        self.assertEqual(DataMode.DRIVE.value, "drive")


class DefaultConstantsTests(unittest.TestCase):
    """The documented default roots are the ones the notebooks assume."""

    def test_default_local_root_is_content_fall_local(self) -> None:
        self.assertEqual(DEFAULT_LOCAL_DATA_ROOT.parts[-2:],
                          ("content", "fall_local"))

    def test_default_drive_root_is_content_drive_fall_detection(self) -> None:
        # Last 4 components: ('content', 'drive', 'MyDrive', 'fall_detection')
        # — on POSIX; on Windows the leading / is normalized to a
        # drive-letter root, so we check the tail.
        self.assertEqual(DEFAULT_DRIVE_ROOT.parts[-4:],
                          ("content", "drive", "MyDrive", "fall_detection"))

    def test_env_var_names_are_stable(self) -> None:
        # Renaming any of these would break user-provided overrides.
        self.assertEqual(LOCAL_DATA_ROOT_ENV_VAR, "FALL_DETECTION_LOCAL_DATA_ROOT")
        self.assertEqual(DRIVE_ROOT_ENV_VAR, "FALL_DETECTION_DRIVE_ROOT")
        self.assertEqual(DATA_MODE_ENV_VAR, "FALL_DETECTION_DATA_MODE")


class ResolveDataLayoutTests(unittest.TestCase):
    """The factory produces the right layout for each mode + override."""

    def setUp(self) -> None:
        # Make sure no env var from a sibling test leaks in.
        for var in (DATA_MODE_ENV_VAR, LOCAL_DATA_ROOT_ENV_VAR, DRIVE_ROOT_ENV_VAR):
            if var in __import__("os").environ:
                del __import__("os").environ[var]

    def test_default_mode_is_local(self) -> None:
        layout = resolve_data_layout()
        self.assertEqual(layout.mode, DataMode.LOCAL)

    def test_local_mode_resolves_dataset_root_under_content_fall_local(self) -> None:
        layout = resolve_data_layout(mode="local")
        self.assertEqual(layout.dataset_root, DEFAULT_LOCAL_DATA_ROOT)
        # And the Drive artefact root stays default.
        self.assertEqual(layout.artifact_root, DEFAULT_DRIVE_ROOT)

    def test_drive_mode_collapses_both_roots(self) -> None:
        layout = resolve_data_layout(mode="drive")
        self.assertEqual(layout.mode, DataMode.DRIVE)
        self.assertEqual(layout.dataset_root, layout.artifact_root)
        self.assertEqual(layout.dataset_root, DEFAULT_DRIVE_ROOT)

    def test_local_root_override_takes_effect(self) -> None:
        custom = Path("/tmp/my_local_data")
        layout = resolve_data_layout(mode="local", local_root=custom)
        self.assertEqual(layout.dataset_root, custom)
        # Artefact root unchanged.
        self.assertEqual(layout.artifact_root, DEFAULT_DRIVE_ROOT)

    def test_drive_root_override_takes_effect(self) -> None:
        custom = Path("/tmp/my_drive_data")
        layout = resolve_data_layout(mode="drive", drive_root=custom)
        self.assertEqual(layout.dataset_root, custom)
        self.assertEqual(layout.artifact_root, custom)

    def test_env_var_overrides_when_no_explicit_arg(self) -> None:
        custom_local = Path("/tmp/env_local")
        __import__("os").environ[LOCAL_DATA_ROOT_ENV_VAR] = str(custom_local)
        layout = resolve_data_layout(mode="local")
        self.assertEqual(layout.dataset_root, custom_local)

    def test_env_var_drive_mode_overrides(self) -> None:
        __import__("os").environ[DATA_MODE_ENV_VAR] = "drive"
        layout = resolve_data_layout()
        self.assertEqual(layout.mode, DataMode.DRIVE)

    def test_invalid_mode_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_data_layout(mode="nonsense")


class DataLayoutPropertiesTests(unittest.TestCase):
    """The convenience paths on the layout resolve correctly."""

    def _make(self, *, dataset: Path, artefact: Path) -> DataLayout:
        return DataLayout(mode=DataMode.LOCAL, dataset_root=dataset,
                          artifact_root=artefact)

    def test_root_alias_points_at_dataset_root(self) -> None:
        layout = self._make(dataset=Path("/x"), artefact=Path("/y"))
        self.assertEqual(layout.root, Path("/x"))

    def test_datasets_path_is_dataset_root_plus_datasets(self) -> None:
        layout = self._make(dataset=Path("/data"), artefact=Path("/art"))
        self.assertEqual(layout.datasets, Path("/data/datasets"))

    def test_artefacts_path_is_artefact_root_plus_artifacts(self) -> None:
        layout = self._make(dataset=Path("/data"), artefact=Path("/art"))
        self.assertEqual(layout.artifacts, Path("/art/artifacts"))

    def test_logs_path_is_artefact_root_plus_logs(self) -> None:
        layout = self._make(dataset=Path("/data"), artefact=Path("/art"))
        self.assertEqual(layout.logs, Path("/art/logs"))

    def test_is_local_mode_property(self) -> None:
        local = self._make(dataset=Path("/x"), artefact=Path("/y"))
        drive = DataLayout(mode=DataMode.DRIVE,
                            dataset_root=Path("/y"), artifact_root=Path("/y"))
        self.assertTrue(local.is_local_mode())
        self.assertFalse(drive.is_local_mode())


class DescribeLayoutTests(unittest.TestCase):
    """The log-friendly description mentions both roots when they differ."""

    def test_local_layout_includes_both_roots(self) -> None:
        layout = DataLayout(mode=DataMode.LOCAL,
                              dataset_root=Path("/content/fall_local"),
                              artifact_root=Path("/content/drive/MyDrive/fall_detection"))
        text = describe_layout(layout)
        self.assertIn("local", text)
        # Path round-trip through Windows-style backslashes too.
        self.assertTrue(
            "fall_local" in text and "fall_detection" in text,
            msg=f"expected both roots in description, got: {text}",
        )

    def test_drive_layout_collapses_to_one_root_in_description(self) -> None:
        layout = DataLayout(mode=DataMode.DRIVE,
                              dataset_root=Path("/x"),
                              artifact_root=Path("/x"))
        text = describe_layout(layout)
        self.assertIn("drive", text)
        # When both roots are the same, the description folds them.
        self.assertIn("dataset_root=artifact_root", text)


class SelectActivePathsTests(unittest.TestCase):
    """The function returns the dirs that the active pipeline reads from."""

    def test_local_mode_yields_only_dataset_dirs(self) -> None:
        layout = DataLayout(mode=DataMode.LOCAL,
                              dataset_root=Path("/content/fall_local"),
                              artifact_root=Path("/content/drive/MyDrive/fall_detection"))
        paths = set(select_active_paths(layout))
        # In LOCAL mode, active processing never touches the Drive
        # artefact root for reads.
        self.assertEqual(paths, {layout.datasets, layout.dataset_root})
        self.assertNotIn(layout.artifact_root, paths)

    def test_drive_mode_yields_drive_dataset_dirs(self) -> None:
        layout = DataLayout(mode=DataMode.DRIVE,
                              dataset_root=Path("/x"),
                              artifact_root=Path("/x"))
        paths = set(select_active_paths(layout))
        # When roots collapse, both roots and datasets are returned.
        self.assertIn(layout.datasets, paths)


class StageUrfdContractTests(unittest.TestCase):
    """stage_urfd_from_kaggle now accepts any data_root (not just Drive).

    We can't actually download in tests; we just confirm the signature
    accepts the new parameter name and rejects the old whitelist
    violations.
    """

    def test_stage_urfd_signature_uses_data_root(self) -> None:
        import inspect
        from data.stage_urfd import stage_urfd_from_kaggle
        sig = inspect.signature(stage_urfd_from_kaggle)
        params = list(sig.parameters)
        self.assertIn("data_root", params)
        self.assertNotIn("drive_root", params)

    def test_non_whitelisted_slug_raises_before_io(self) -> None:
        from data.stage_urfd import stage_urfd_from_kaggle, ALLOWED_KAGGLE_SLUG
        self.assertEqual(ALLOWED_KAGGLE_SLUG, "tanmaydacha/urfd-dataset")
        with self.assertRaises(RuntimeError):
            stage_urfd_from_kaggle(Path("/tmp"), kaggle_slug="other-slug")


if __name__ == "__main__":
    unittest.main()