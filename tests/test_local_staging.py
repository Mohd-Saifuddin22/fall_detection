"""Unit tests for :mod:`perception.local_staging`.

Covers the Issue 002 performance fix: copy frames to local disk before
tracking, preserve numeric order, isolate clips, never modify the
source Drive folder.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.frames import discover_frames  # noqa: E402
from perception.local_staging import (  # noqa: E402
    DEFAULT_LOCAL_ROOT,
    LocalFrameStager,
    StagedClipContext,
    stage_clip_frames,
)


def _make_clip_frames(parent: Path, clip_name: str, names: list[str],
                        contents: list[bytes] | None = None) -> Path:
    """Create a clip folder with named files. Returns the folder path.

    Each file gets a unique byte payload by default so we can detect
    accidental copy-mixups. Caller can pass an explicit ``contents``
    list (one per file) to control payloads.
    """
    folder = parent / clip_name
    folder.mkdir(parents=True, exist_ok=True)
    if contents is None:
        contents = [f"{clip_name}-{name}".encode() for name in names]
    assert len(contents) == len(names), "contents must match names"
    for name, payload in zip(names, contents):
        (folder / name).write_bytes(payload)
    return folder


class StageClipTests(unittest.TestCase):
    """The per-clip staging produces a flat, numerically-ordered local copy."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.source_root = Path(self._tmp.name) / "datasets" / "urfd"
        self.source_root.mkdir(parents=True)
        self.local_root = Path(self._tmp.name) / "local"
        self.local_root.mkdir(parents=True)

    def _make_frames(self) -> Path:
        return _make_clip_frames(
            self.source_root, "fall-01-cam0",
            ["frame_0003.png", "frame_0001.png", "frame_0010.png",
             "frame_0002.png", "frame_0009.png"],
        )

    def test_staging_preserves_numeric_order(self) -> None:
        # Drive folder is written out of order; staging must fix it.
        source_folder = self._make_frames()
        ordered = discover_frames(source_folder)
        # Sanity: discover_frames already sorts; verify the staging
        # step takes those sorted paths and produces a flat sequence.
        stager = LocalFrameStager(local_root=self.local_root)
        with stager.stage_clip("fall-01-cam0", source_folder, ordered) as ctx:
            staged = sorted(ctx.local_folder.iterdir())
            names = [p.name for p in staged]
            self.assertEqual(
                names,
                ["frame_00000.png", "frame_00001.png", "frame_00002.png",
                 "frame_00003.png", "frame_00004.png"],
            )
            # The bytes are also intact — confirms shutil.copy2 worked
            # on each ordered path, not on a shuffled subset.
            self.assertEqual((ctx.local_folder / "frame_00000.png").read_bytes(),
                              b"fall-01-cam0-frame_0001.png")
            self.assertEqual((ctx.local_folder / "frame_00004.png").read_bytes(),
                              b"fall-01-cam0-frame_0010.png")

    def test_context_manager_cleans_up_on_exit(self) -> None:
        source_folder = self._make_frames()
        ordered = discover_frames(source_folder)
        stager = LocalFrameStager(local_root=self.local_root)
        with stager.stage_clip("fall-01-cam0", source_folder, ordered) as ctx:
            self.assertTrue(ctx.local_folder.is_dir())
            self.assertGreater(len(list(ctx.local_folder.iterdir())), 0)
        # Context exit removed the local folder.
        self.assertFalse(ctx.local_folder.exists())

    def test_stale_local_frames_are_removed_between_clips(self) -> None:
        # First clip writes a frame; second clip's staging must wipe
        # the destination before copying — so a smaller second clip
        # can't leave stale first-clip frames behind.
        clip_a = _make_clip_frames(self.source_root, "clip-A",
                                    ["frame_0001.png", "frame_0002.png",
                                     "frame_0003.png", "frame_0004.png",
                                     "frame_0005.png", "frame_0006.png"])
        clip_b = _make_clip_frames(self.source_root, "clip-B",
                                    ["frame_0001.png"])

        stager = LocalFrameStager(local_root=self.local_root)
        # Stage clip-A.
        with stager.stage_clip("clip-A", clip_a, discover_frames(clip_a)) as ctx_a:
            count_a = len(list(ctx_a.local_folder.iterdir()))
        # Stage clip-B at the SAME local path (different clip_id would
        # normally land in a different sub-folder; we force the
        # collision by passing clip_b's id as "clip-A" too).
        with stager.stage_clip("clip-A", clip_b, discover_frames(clip_b)) as ctx_b:
            count_b = len(list(ctx_b.local_folder.iterdir()))
        self.assertEqual(count_a, 6)
        self.assertEqual(count_b, 1)
        # And after exit, the second staging is gone too.
        self.assertFalse(ctx_b.local_folder.exists())

    def test_source_drive_folder_is_not_modified(self) -> None:
        # Snapshot every file's size + mtime + content BEFORE staging;
        # assert nothing changed AFTER staging.
        source_folder = self._make_frames()
        snapshot = {
            entry.name: (entry.stat().st_size, entry.stat().st_mtime_ns,
                          entry.read_bytes())
            for entry in source_folder.iterdir()
        }

        stager = LocalFrameStager(local_root=self.local_root)
        with stager.stage_clip("fall-01-cam0", source_folder,
                                  discover_frames(source_folder)):
            pass

        for entry in source_folder.iterdir():
            self.assertIn(entry.name, snapshot)
            size, mtime, content = snapshot[entry.name]
            self.assertEqual(entry.stat().st_size, size,
                              msg=f"{entry.name}: size changed")
            self.assertEqual(entry.stat().st_mtime_ns, mtime,
                              msg=f"{entry.name}: mtime changed")
            self.assertEqual(entry.read_bytes(), content,
                              msg=f"{entry.name}: content changed")

    def test_skip_env_var_skips_copy(self) -> None:
        # When FALL_DETECTION_SKIP_LOCAL_STAGING is set, the stager
        # reports a zero-byte / zero-second result and the local
        # folder is never touched.
        source_folder = self._make_frames()
        try:
            os.environ["FALL_DETECTION_SKIP_LOCAL_STAGING"] = "1"
            stager = LocalFrameStager(local_root=self.local_root)
            with stager.stage_clip("fall-01-cam0", source_folder,
                                      discover_frames(source_folder)) as ctx:
                self.assertTrue(ctx.result.frame_count > 0)
                self.assertEqual(ctx.result.bytes_copied, 0)
                self.assertEqual(ctx.result.elapsed_seconds, 0.0)
                # local_folder should equal the source — we didn't copy.
                self.assertEqual(ctx.local_folder, source_folder)
        finally:
            os.environ.pop("FALL_DETECTION_SKIP_LOCAL_STAGING", None)

    def test_staging_returns_zero_files_for_empty_clip(self) -> None:
        # Empty source folder: zero frames copied, local folder exists
        # but contains nothing.
        empty = self.source_root / "empty-clip"
        empty.mkdir()
        stager = LocalFrameStager(local_root=self.local_root)
        with stager.stage_clip("empty-clip", empty, discover_frames(empty)) as ctx:
            self.assertEqual(ctx.result.frame_count, 0)
            self.assertEqual(ctx.result.bytes_copied, 0)
            self.assertEqual(list(ctx.local_folder.iterdir()), [])

    def test_handles_missing_source_frame_gracefully(self) -> None:
        # If a frame path disappears between discovery and copy, the
        # error propagates rather than silently skipping — we never
        # want to claim a successful staging with missing files.
        source_folder = self._make_frames()
        ordered = discover_frames(source_folder)
        # Delete one frame after discovery.
        ordered[0].path.unlink()
        stager = LocalFrameStager(local_root=self.local_root)
        with self.assertRaises(FileNotFoundError):
            with stager.stage_clip("fall-01-cam0", source_folder, ordered):
                pass

    def test_local_root_default(self) -> None:
        # Sanity: the documented default points at the Colab local disk.
        # Compare the trailing components rather than the full string
        # so the test passes on POSIX (``/content/fall_detection_local``)
        # AND Windows (where the leading slash normalises to a drive
        # letter or to a backslash-rooted path).
        self.assertEqual(DEFAULT_LOCAL_ROOT.parts[-2:], ("content", "fall_detection_local"))
        self.assertTrue(str(DEFAULT_LOCAL_ROOT).replace("\\", "/").endswith(
            "/content/fall_detection_local"))

    def test_stage_clip_frames_one_shot(self) -> None:
        source_folder = self._make_frames()
        ctx, ordered = stage_clip_frames(
            "fall-01-cam0", source_folder, local_root=self.local_root,
        )
        try:
            self.assertIsInstance(ctx, StagedClipContext)
            self.assertEqual(len(ordered), 5)
            # Local folder has the staged files.
            self.assertEqual(len(list(ctx.local_folder.iterdir())), 5)
        finally:
            ctx._stager.cleanup_one(ctx.local_folder)
        self.assertFalse(ctx.local_folder.exists())


class CleanupTests(unittest.TestCase):
    """The stager's cleanup removes every folder it created."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.source_root = Path(self._tmp.name) / "src"
        self.source_root.mkdir(parents=True)
        self.local_root = Path(self._tmp.name) / "local"
        self.local_root.mkdir(parents=True)

    def test_cleanup_is_idempotent(self) -> None:
        clip = _make_clip_frames(self.source_root, "clip-x",
                                  ["a_0001.png", "a_0002.png"])
        stager = LocalFrameStager(local_root=self.local_root)
        with stager.stage_clip("clip-x", clip, discover_frames(clip)) as ctx:
            folder = ctx.local_folder
        stager.cleanup()
        # Second call must be a no-op even though folders are already gone.
        stager.cleanup()
        self.assertFalse(folder.exists())

    def test_cleanup_does_not_remove_local_root(self) -> None:
        clip = _make_clip_frames(self.source_root, "clip-y",
                                  ["a_0001.png"])
        stager = LocalFrameStager(local_root=self.local_root)
        with stager.stage_clip("clip-y", clip, discover_frames(clip)):
            pass
        stager.cleanup()
        # local_root itself stays — we only clean the per-clip sub-folders.
        self.assertTrue(self.local_root.exists())


if __name__ == "__main__":
    unittest.main()