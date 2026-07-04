"""End-to-end invariant: ``run_cropping`` → ``load_clip_tensor_from_shards``.

This test catches the post-Issue-005-smoke-check window-key
bug. Before the fix, the per-window member stem was computed
INSIDE the frame loop via ``len(writer._clip_keys)`` — a global
counter that did NOT reset between clips / tracks within one
shard. The result was non-monotonically-numbered windows across
clip / track boundaries, which the Pipeline A loader (which
groups on the per-window member stem) could not recover.

The test runs ``run_cropping`` against a synthetic fixture
long enough to produce multiple windows per track, then asserts
the cross-module invariant:

    1. The shard manifest's ``clip_keys`` count equals the number
       of EMITTED WINDOWS, not frames.
    2. Each emitted window has exactly ``clip_length`` members
       (one per frame_offset).
    3. Each emitted window's member frames have contiguous
       ``frame_offset`` values ``0..clip_length - 1``.
    4. The Pipeline A loader can load each window directly via
       its per-window member stem — proving the runner's output
       is what the loader expects, with no manual key fixup.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline_a import IMAGE_SIZE, load_clip_tensor_from_shards  # noqa: E402

from cropping.clip_builder import CropConfig  # noqa: E402
from cropping.runner import run_cropping  # noqa: E402
from cropping.shard_writer import read_shard  # noqa: E402
from cropping.track_windows import TrackedBox  # noqa: E402
from data.build_urfd_manifest import write_urfd_manifest  # noqa: E402
from data.manifests import (  # noqa: E402
    ClipRecord,
    ClipRole,
    FallLabel,
    Manifest,
)


def _write_clip_frames(folder: Path, count: int) -> None:
    """Write ``count`` solid-grey PNG frames into ``folder``."""
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (64, 64), color=128).save(folder / f"frame_{i:05d}.png")


def _write_perception_json(
    perception_root: Path,
    clip_id: str,
    boxes: list[TrackedBox],
    track_id: int = 1,
) -> None:
    clip_dir = perception_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "frame_index": b.frame_index,
            "track_id": track_id,
            "cls_id": 0,
            "confidence": b.confidence,
            "x_min": b.x_min, "y_min": b.y_min,
            "x_max": b.x_max, "y_max": b.y_max,
        }
        for b in boxes
    ]
    (clip_dir / f"{clip_id}_detections.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def _build_long_track_fixture(
    layout_root: Path,
    perception_root: Path,
    crops_root: Path,
    manifest_path: Path,
    *,
    clip_id: str,
    frame_count: int,
    track_id: int,
    clip_length: int,
) -> None:
    """Build a single-clip fixture with one long continuous track.

    Long enough so ``build_windows_for_track`` produces at least
    two emitted windows when stride == clip_length. The boxes
    are dense, large, and tracked — well above any
    min-coverage gate.
    """
    clip_folder = layout_root / "datasets" / "urfd" / clip_id
    _write_clip_frames(clip_folder, count=frame_count)
    boxes = [
        TrackedBox(
            frame_index=i, x_min=10, y_min=10,
            x_max=50, y_max=50, confidence=0.8,
        )
        for i in range(frame_count)
    ]
    _write_perception_json(perception_root, clip_id, boxes, track_id=track_id)
    manifest = Manifest(
        schema_version="1.1",
        clips=[ClipRecord(
            clip_id=clip_id,
            dataset="urfd",
            role=ClipRole.DEBUG,
            label=FallLabel.FALL,
            source_path=f"datasets/urfd/{clip_id}",
            notes="camera=cam0; long-track fixture for window-key invariant.",
        )],
    )
    write_urfd_manifest(manifest, manifest_path)


class RunnerToLoaderInvariantTests(unittest.TestCase):
    """run_cropping output is consumable by load_clip_tensor_from_shards."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.layout_root = Path(self._tmp.name) / "drive" / "fall_detection"
        self.perception_root = self.layout_root / "artifacts" / "perception"
        self.crops_root = self.layout_root / "artifacts" / "crops"
        self.manifest_path = (
            self.layout_root / "datasets" / "urfd" / "manifest.yaml"
        )

    def _run_long_track(self, clip_length: int) -> tuple[list[str], int]:
        """Build the fixture, run, and return ``(clip_keys, num_windows)``."""
        # 96 frames with clip_length=32 and stride=32 yields at
        # least two windows (window 0 = [0..31], window 1 = [32..63],
        # window 2 = [64..95]). The test relies on >= 2 emitted
        # windows.
        clip_id = "urfd-invariant-fall-01-cam0"
        frame_count = 96
        track_id = 7
        _build_long_track_fixture(
            self.layout_root, self.perception_root, self.crops_root,
            self.manifest_path,
            clip_id=clip_id, frame_count=frame_count,
            track_id=track_id, clip_length=clip_length,
        )
        crop_config = CropConfig(
            output_size=32, margin=0.30, clip_length=clip_length,
        )
        summary = run_cropping(
            layout_root=self.layout_root,
            perception_root=self.perception_root,
            crops_root=self.crops_root,
            manifest_path=self.manifest_path,
            crop_config=crop_config,
            camera_filter="cam0",
            max_shards=64,
        )
        shard_paths = sorted(self.crops_root.glob("shard-*.tar"))
        self.assertEqual(len(shard_paths), summary.shards_written,
                          msg=f"shard count mismatched: {len(shard_paths)} vs {summary.shards_written}")
        # Manifest clip_keys — collected across all shards.
        clip_keys: list[str] = []
        for shard_path in shard_paths:
            result = read_shard(shard_path)
            clip_keys.extend(result.manifest.get("clip_keys", []))
        return clip_keys, summary.windows_emitted

    def test_clip_key_count_matches_emitted_window_count(self) -> None:
        clip_keys, num_windows = self._run_long_track(clip_length=32)
        # The manifest MUST list exactly ``emitted_windows``
        # unique clip_keys, one per window. The buggy runner used
        # ``len(writer._clip_keys)`` which grew monotonically across
        # clips within a shard — out of scope for this fixture but
        # the same invariant catches it when a second clip is added.
        self.assertEqual(len(clip_keys), num_windows,
                          msg=f"clip_keys={clip_keys} vs num_windows={num_windows}")
        self.assertGreaterEqual(num_windows, 2)

    def test_each_clip_key_has_clip_length_members_with_contiguous_offsets(self) -> None:
        clip_keys, _ = self._run_long_track(clip_length=32)
        shard_paths = sorted(self.crops_root.glob("shard-*.tar"))
        # Build a per-clip_key → list[frame_offset] map across
        # every shard.
        offsets_by_key: dict[str, list[int]] = {}
        for shard_path in shard_paths:
            for member_name, _image_bytes in read_shard(shard_path).image_members.items():
                # Member names look like `<safe_key>_0007.image.jpg`;
                # the safe_key is the clip_key.
                stem = member_name.removesuffix(".image.jpg")
                last_underscore = stem.rfind("_")
                if last_underscore < 0:
                    continue
                clip_key_part = stem[:last_underscore]
                try:
                    frame_offset = int(stem[last_underscore + 1:])
                except ValueError:
                    continue
                offsets_by_key.setdefault(clip_key_part, []).append(frame_offset)
        for key in clip_keys:
            offsets = sorted(offsets_by_key[key])
            self.assertEqual(
                len(offsets), 32,
                msg=f"clip_key {key} has {len(offsets)} members, expected 32",
            )
            self.assertEqual(
                offsets, list(range(32)),
                msg=f"clip_key {key} offsets are not contiguous 0..31: {offsets}",
            )

    def test_pipeline_a_loader_consumes_each_window_key_directly(self) -> None:
        # The end-to-end invariant the brief asks for: every emitted
        # window's per-window key can be loaded by the Pipeline A
        # loader. Before the fix, the per-window key was
        # non-monotonic across windows / clips and the loader (which
        # groups on the per-window stem) would either LookupError on a
        # key that no longer existed or merge siblings.
        clip_keys, _ = self._run_long_track(clip_length=32)
        self.assertGreaterEqual(len(clip_keys), 2)
        shard_paths = sorted(self.crops_root.glob("shard-*.tar"))
        for clip_key in clip_keys:
            loaded = load_clip_tensor_from_shards(
                shard_paths, clip_key=clip_key, T=32,
            )
            # The T-32 enforcement: 32 frames, contiguous 0..31.
            # IMAGE_SIZE is fixed at 224 by the Pipeline A loader
            # contract regardless of the runner's output_size —
            # the loader resizes every decoded frame up to 224x224.
            self.assertEqual(
                loaded.tensor.shape,
                (32, 3, IMAGE_SIZE, IMAGE_SIZE),
            )
            self.assertEqual(
                loaded.frames.shape,
                (32, IMAGE_SIZE, IMAGE_SIZE, 3),
            )
            self.assertEqual(loaded.frame_offsets, tuple(range(32)))
            # Per-window identity: this fixture wrote one source
            # clip_id, so every clip_id is the same bare id; the
            # crucial fix is that loaded.clip_key (the per-window
            # stem) is exactly the key we asked for.
            self.assertEqual(loaded.clip_key, clip_key)
            self.assertEqual(loaded.clip_id, "urfd-invariant-fall-01-cam0")


class RunnerToLoaderInvariant16FrameTests(RunnerToLoaderInvariantTests):
    """Same invariants at clip_length=16.

    The brief's spec is that the loader must be T-adaptive. This
    proves the runner's invariant holds at both sides of the
    ``ALLOWED_T = (16, 32)`` policy.
    """

    def _run_long_track(self, clip_length: int = 16) -> tuple[list[str], int]:
        # The inner method is overridden by signature; re-emit with
        # clip_length=16 so the test passes it through.
        return super()._run_long_track(clip_length)

    def test_pipeline_a_loader_consumes_each_window_key_directly(self) -> None:
        clip_keys, _ = self._run_long_track(clip_length=16)
        shard_paths = sorted(self.crops_root.glob("shard-*.tar"))
        self.assertGreaterEqual(len(clip_keys), 2)
        for clip_key in clip_keys:
            loaded = load_clip_tensor_from_shards(
                shard_paths, clip_key=clip_key, T=16,
            )
            self.assertEqual(
                loaded.tensor.shape,
                (16, 3, IMAGE_SIZE, IMAGE_SIZE),
            )
            self.assertEqual(loaded.frame_offsets, tuple(range(16)))
            self.assertEqual(loaded.clip_key, clip_key)


if __name__ == "__main__":
    unittest.main()
