"""Unit tests for :mod:`cropping.runner` local-frame staging.

Covers the Issue 002 runtime fix extended to cropping: copy each clip's
source frames to local disk before reading them, clean up afterwards,
preserve the source folder on Drive.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cropping.runner import (  # noqa: E402
    stage_clip_frames_for_cropping,
)
from perception.frames import discover_frames  # noqa: E402


def _touch(folder: Path, name: str, payload: bytes = b"x") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(payload)
    return p


def _make_clip(folder: Path, names: list[str]) -> Path:
    """Create a clip folder with named frames and deterministic payloads."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for i, name in enumerate(names):
        _touch(folder, name, payload=f"clip-{i}-{name}".encode())
    return folder


class StageClipFramesForCroppingTests(unittest.TestCase):
    """The cropping-side stager mirrors the perception-side one."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.source = Path(self._tmp.name) / "datasets" / "urfd" / "fall-01-cam0"
        self.local = Path(self._tmp.name) / "local"
        self.local.mkdir(parents=True)

    def test_local_folder_contains_numerically_named_frames(self) -> None:
        _make_clip(self.source, ["frame_0010.png", "frame_0001.png", "frame_0005.png"])
        local_folder, count = stage_clip_frames_for_cropping(
            "fall-01-cam0", self.source, self.local,
        )
        self.assertEqual(count, 3)
        names = sorted(p.name for p in local_folder.iterdir())
        # Indexed by the discover_frames index (zero-based), NOT by
        # the source filename. Discover_frames assigns indices in the
        # sorted numeric order: frame_0001=0, frame_0005=1, frame_0010=2.
        self.assertEqual(names, ["frame_00000.png", "frame_00001.png",
                                  "frame_00002.png"])

    def test_source_drive_folder_is_not_modified(self) -> None:
        _make_clip(self.source, ["frame_0001.png", "frame_0002.png"])
        snapshot = {
            entry.name: (entry.stat().st_size, entry.read_bytes())
            for entry in self.source.iterdir()
        }
        stage_clip_frames_for_cropping("fall-01-cam0", self.source, self.local)
        for entry in self.source.iterdir():
            self.assertIn(entry.name, snapshot)
            size, content = snapshot[entry.name]
            self.assertEqual(entry.stat().st_size, size)
            self.assertEqual(entry.read_bytes(), content)

    def test_clip_id_used_for_local_subfolder_name(self) -> None:
        _make_clip(self.source, ["frame_0001.png"])
        local_folder, _ = stage_clip_frames_for_cropping(
            "urfd/fall 01/cam0", self.source, self.local,
        )
        # Un-safe characters get collapsed to underscores; the prefix
        # ``crops_`` makes the staging dir self-documenting.
        self.assertEqual(local_folder.name, "crops_urfd_fall_01_cam0")
        self.assertTrue(local_folder.is_dir())

    def test_repeated_call_replaces_previous_staging(self) -> None:
        # First call stages 3 frames; second call replaces the local
        # folder entirely so stale frames from the first call can't
        # leak into the second.
        _make_clip(self.source, ["frame_0001.png", "frame_0002.png",
                                  "frame_0003.png"])
        first_folder, _ = stage_clip_frames_for_cropping(
            "fall-01-cam0", self.source, self.local,
        )
        self.assertEqual(len(list(first_folder.iterdir())), 3)

        # Shrink the source to 1 frame; second call must wipe the
        # previous 3-frame local folder and stage only 1 frame.
        _make_clip(self.source, ["frame_0009.png"])
        second_folder, count = stage_clip_frames_for_cropping(
            "fall-01-cam0", self.source, self.local,
        )
        self.assertEqual(count, 1)
        self.assertEqual(second_folder, first_folder,
                          msg="same clip_id should land in the same local folder")
        self.assertEqual(len(list(second_folder.iterdir())), 1)

    def test_skip_staging_env_var_returns_source_folder(self) -> None:
        _make_clip(self.source, ["frame_0001.png", "frame_0002.png"])
        try:
            os.environ["FALL_DETECTION_SKIP_LOCAL_STAGING"] = "1"
            effective, count = stage_clip_frames_for_cropping(
                "fall-01-cam0", self.source, self.local,
            )
            # We didn't actually copy — the returned path is the source.
            self.assertEqual(effective, self.source)
            self.assertEqual(count, 2)
            # No new folder was created under local_root.
            self.assertEqual(list(self.local.iterdir()), [])
        finally:
            os.environ.pop("FALL_DETECTION_SKIP_LOCAL_STAGING", None)

    def test_local_frames_can_be_read_by_discover_frames_via_symlink(self) -> None:
        # Sanity: the local staged frames are discoverable by the same
        # discovery routine that drives the tracker. We verify by
        # re-discovering the staged folder.
        _make_clip(self.source, ["frame_0001.png", "frame_0002.png"])
        local_folder, _ = stage_clip_frames_for_cropping(
            "fall-01-cam0", self.source, self.local,
        )
        discovered = discover_frames(local_folder)
        self.assertEqual(len(discovered), 2)
        # Sorted indices match the numeric source ordering.
        self.assertEqual([d.index for d in discovered], [0, 1])


if __name__ == "__main__":
    unittest.main()