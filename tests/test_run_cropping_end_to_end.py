"""End-to-end regression for crop shard finalization.

The bug this guards against:
    ``run_cropping`` discarded the writer returned by ``_process_clip``
    and never closed it, leaving the tar file unterminated and the
    ``_manifest.json`` empty (or missing). ``summary.shards_written``
    always reported 0.

This test runs ``run_cropping`` against a small valid fixture and
asserts:
    1. at least one window is emitted (the data fixture is rich enough),
    2. the shard tar contains a non-empty ``_manifest.json``,
    3. ``summary.shards_written`` reflects the actual shard count,
    4. the produced tar is finalized + readable via ``tarfile.open``.
"""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cropping.clip_builder import CropConfig  # noqa: E402
from cropping.runner import run_cropping  # noqa: E402
from cropping.shard_writer import read_shard  # noqa: E402
from cropping.track_windows import TrackedBox  # noqa: E402
from data.manifests import (  # noqa: E402
    ClipRecord,
    ClipRole,
    FallLabel,
    Manifest,
    _parse_clip,  # used to build the manifest dict
)
from data.build_urfd_manifest import write_urfd_manifest  # noqa: E402


def _write_clip_frames(folder: Path, count: int = 64) -> None:
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
    """Write a minimal Issue-002 detections.json for one clip."""
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


class RunCroppingFinalizationTests(unittest.TestCase):
    """The shard writer is registered with run_cropping and finalised."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # Single-root layout: dataset and artefacts under one Drive root.
        # This keeps the fixture simple — the split-root bug is
        # already covered by tests/test_cropping_split_root.py.
        self.layout_root = Path(self._tmp.name) / "drive" / "fall_detection"
        self.perception_root = self.layout_root / "artifacts" / "perception"
        self.crops_root = self.layout_root / "artifacts" / "crops"
        self.dataset_root = self.layout_root
        self.manifest_path = self.layout_root / "datasets" / "urfd" / "manifest.yaml"

    def _build_fixture(self, clip_id: str = "urfd-debug-fall-01-cam0") -> None:
        # Dataset frames.
        clip_folder = self.dataset_root / "datasets" / "urfd" / clip_id
        _write_clip_frames(clip_folder, count=64)
        # Perception JSON for one clip + one track.
        boxes = [
            TrackedBox(
                frame_index=i, x_min=10, y_min=10,
                x_max=50, y_max=50, confidence=0.8,
            )
            for i in range(64)
        ]
        _write_perception_json(self.perception_root, clip_id, boxes)
        # Manifest with the matching source_path.
        manifest = Manifest(
            schema_version="1.1",
            clips=[ClipRecord(
                clip_id=clip_id,
                dataset="urfd",
                role=ClipRole.DEBUG,
                label=FallLabel.FALL,
                source_path=f"datasets/urfd/{clip_id}",
                notes="camera=cam0; frame-folder (PNGs); TEST",
            )],
        )
        write_urfd_manifest(manifest, self.manifest_path)

    def test_run_cropping_finalizes_shards_with_manifest(self) -> None:
        self._build_fixture()

        crop_config = CropConfig(output_size=32, margin=0.30, clip_length=32)
        summary = run_cropping(
            layout_root=self.layout_root,
            perception_root=self.perception_root,
            crops_root=self.crops_root,
            manifest_path=self.manifest_path,
            crop_config=crop_config,
            camera_filter="cam0",
            max_shards=64,
        )

        # 1. At least one window emitted.
        self.assertGreater(
            summary.windows_emitted, 0,
            msg=f"emitted={summary.windows_emitted}, skipped={summary.skip_reason_counts}",
        )

        # 2. shards_written reflects the real shard count. Before the fix
        # this was always 0 because the close loop iterated over an empty
        # open_writers dict.
        self.assertGreater(summary.shards_written, 0,
                            msg=f"shards_written={summary.shards_written}")

        # 3. Shard files exist on disk and are readable.
        shard_paths = sorted(self.crops_root.glob("shard-*.tar"))
        self.assertEqual(len(shard_paths), summary.shards_written)

        # 4. The tar contains a non-empty _manifest.json (the bug
        # surfaced as an empty / missing manifest because close was
        # never called).
        for shard_path in shard_paths:
            with tarfile.open(shard_path, mode="r") as tar:
                names = tar.getnames()
                self.assertIn("_manifest.json", names,
                               msg=f"missing manifest in {shard_path}")
                manifest_member = tar.getmember("_manifest.json")
                self.assertGreater(manifest_member.size, 0,
                                     msg=f"empty manifest in {shard_path}")
                # And the manifest actually lists at least one member.
                manifest_text = tar.extractfile(manifest_member).read()
                payload = json.loads(manifest_text)
                self.assertGreater(
                    len(payload.get("member_keys", [])),
                    0,
                    msg=f"manifest has no member_keys in {shard_path}",
                )

        # 5. read_shard() round-trips the same metadata — confirms the
        # finalised shard is usable by downstream trainers.
        result = read_shard(shard_paths[0])
        self.assertGreater(len(result.image_members), 0)
        self.assertGreater(len(result.metadata_members), 0)
        self.assertGreater(len(result.manifest.get("clip_keys", [])), 0)


if __name__ == "__main__":
    unittest.main()