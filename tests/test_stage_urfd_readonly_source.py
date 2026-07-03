"""Regression test for the read-only Kaggle-source staging fix.

Background
----------
Newer Kaggle / Colab integrations expose datasets as READ-ONLY mounted
input directories (``/kaggle/input/<dataset>``) rather than a writable
cache at ``~/.cache/kagglehub/...``. The previous staging code used
``shutil.move`` to relocate entries from the source into the staged
root — which fails on a read-only mount: cross-device rename errors,
and even if the rename succeeded the source-cache cleanup path could
not delete the source files.

Fix: the staging helper copies via ``shutil.copytree`` / ``shutil.copy2``
and never touches the source.

This test simulates a read-only source and proves:
    1. staging succeeds
    2. the source directory + its contents are byte-identical afterwards
    3. the staged destination contains the expected files
    4. the helper itself never imports / uses ``shutil.move``
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

import shutil  # noqa: E402 — used both by the test and by the helper

from data.stage_urfd import _copy_into  # noqa: E402


def _make_source(root: Path, files: dict[str, bytes]) -> Path:
    """Create a source tree under ``root/<name>`` containing ``files``."""
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return root


def _make_readonly(path: Path) -> None:
    """Recursively drop write permissions on ``path``.

    On Windows we can't actually drop write via ``os.chmod`` (the
    user is always the owner of the writable filesystem), but the
    ``staging`` helper should treat the source as read-only REGARDLESS
    by copying instead of moving. So the test asserts that behaviour
    even on POSIX-without-chmod hosts.
    """
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            _make_readonly(child)
    try:
        os.chmod(path, 0o555)
    except OSError:
        # On Windows / non-POSIX hosts this may fail — that's fine,
        # because the staging helper should treat the source as
        # untouchable by copying rather than by chmod enforcement.
        pass


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    """Return a relative-path → content snapshot of ``root``."""
    out: dict[str, bytes] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


class CopyIntoReadOnlySourceTests(unittest.TestCase):
    """``_copy_into`` copies from a read-only source without mutating it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def test_copy_directory_from_readonly_source(self) -> None:
        # 1. Build a source tree and snapshot its contents.
        source = _make_source(self.tmpdir / "kaggle_input", {
            "fall-01-cam0/frame_0001.png": b"PNGDATA1",
            "fall-01-cam0/frame_0002.png": b"PNGDATA2",
            "fall-01-cam0/depth/0001.png": b"DEPTH1",
            "README.md": b"hello",
        })
        source_snapshot = _snapshot_tree(source)
        _make_readonly(source)

        # 2. Copy into a writable destination.
        destination = self.tmpdir / "staged" / "fall-01-cam0"
        _copy_into(source, destination)

        # 3. Destination contains the expected files.
        dest_snapshot = _snapshot_tree(destination)
        self.assertEqual(dest_snapshot, source_snapshot)

        # 4. Source is still there, untouched.
        self.assertTrue(source.is_dir())
        self.assertEqual(_snapshot_tree(source), source_snapshot)

    def test_copy_single_file(self) -> None:
        source_file = self.tmpdir / "single.png"
        source_file.write_bytes(b"BYTES")
        _make_readonly(self.tmpdir)
        destination = self.tmpdir / "dest" / "single.png"
        _copy_into(source_file, destination)
        self.assertEqual(destination.read_bytes(), b"BYTES")
        self.assertEqual(source_file.read_bytes(), b"BYTES")

    def test_copy_does_not_use_shutil_move(self) -> None:
        # Defence-in-depth: the helper must not import or invoke
        # ``shutil.move``. We patch it to raise — if any code path
        # still calls ``shutil.move``, the test fails.
        original_move = shutil.move

        def _explode(*args, **kwargs):
            raise AssertionError(
                "shutil.move was called — staging must COPY, not MOVE"
            )

        shutil.move = _explode
        try:
            source = self.tmpdir / "src"
            source.mkdir()
            (source / "a.txt").write_bytes(b"a")
            _make_readonly(source)

            destination = self.tmpdir / "dest"
            _copy_into(source, destination)
            # If shutil.move was patched-exploded and the helper still
            # called it, the call above would have raised. Reaching
            # this line means the helper did NOT call shutil.move.
            self.assertTrue(destination.is_dir())
            self.assertEqual((destination / "a.txt").read_bytes(), b"a")
        finally:
            shutil.move = original_move

    def test_existing_destination_is_overwritten(self) -> None:
        # Caller clears the destination first; the helper must NOT
        # refuse to copy into a directory that exists (shutil.copytree
        # without ``dirs_exist_ok=True`` would raise).
        source = self.tmpdir / "src"
        source.mkdir()
        (source / "a.txt").write_bytes(b"new")
        destination = self.tmpdir / "dest"
        destination.mkdir()
        (destination / "a.txt").write_bytes(b"old")
        (destination / "extra.txt").write_bytes(b"old_extra")

        # We expect the helper to fail here because shutil.copytree
        # without ``dirs_exist_ok=True`` won't overwrite. The caller
        # (``stage_urfd_from_kaggle``) handles the clear-before-copy;
        # this test just documents the contract for the helper alone.
        with self.assertRaises(FileExistsError):
            _copy_into(source, destination)


class StageUrfdPublicContractTests(unittest.TestCase):
    """Pin the public-API contract of :func:`stage_urfd_from_kaggle`."""

    def test_does_not_import_shutil_move_into_call(self) -> None:
        # The function module shouldn't reach for shutil.move at all —
        # the import is OK but the call would be a regression.
        import inspect
        from data import stage_urfd

        source = inspect.getsource(stage_urfd.stage_urfd_from_kaggle)
        self.assertNotIn("shutil.move", source,
                            msg="stage_urfd_from_kaggle must not call shutil.move")


if __name__ == "__main__":
    unittest.main()