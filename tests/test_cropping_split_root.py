"""Regression tests for the Issue 003 real-run crop-stage path bug.

The bug:
    cropping/runner.py resolved source frames as
    ``crops_root.parent / clip_record.source_path`` — i.e. it assumed
    the dataset root and the crop artefact root shared a parent.
    In LOCAL mode they live on different filesystems (local dataset vs.
    Drive artefact), so source_path resolved to
    ``<artifact_root>/datasets/...`` which doesn't exist.

The fix:
    ``layout_root`` is now threaded through ``_process_clip`` and used
    as the root for source-path resolution. When ``layout_root`` is
    omitted, the runner falls back to ``crops_root.parent`` so legacy
    single-root layouts still work.

These tests intentionally keep the dataset root and the crop artefact
root on different paths so the bug repros if it ever comes back.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cropping.runner import _process_clip  # noqa: E402
from data.manifests import ClipRecord, ClipRole, FallLabel  # noqa: E402
from cropping.clip_builder import CropConfig  # noqa: E402
from cropping.track_windows import TrackedBox  # noqa: E402


def _make_clip_record(clip_id: str, source_path: str) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        dataset="urfd",
        role=ClipRole.DEBUG,
        label=FallLabel.FALL,
        source_path=source_path,
        notes="camera=cam0; frame-folder (PNGs); TEST",
    )


def _write_fake_perception_json(
    perception_root: Path,
    clip_id: str,
    track_id: int,
    boxes: list[TrackedBox],
) -> Path:
    """Write a tiny Issue-002 detections.json for one clip + track."""
    clip_dir = perception_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    det_path = clip_dir / f"{clip_id}_detections.json"
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
    det_path.write_text(json.dumps(payload), encoding="utf-8")
    return det_path


def _make_clip_frames(folder: Path, count: int = 64) -> None:
    """Write ``count`` solid-grey PNG frames into ``folder``.

    64 frames lets two non-overlapping clip_length=32 windows emit
    cleanly (frames 0..31 and 32..63). Frames are deterministic in
    colour (grey 128) so we can assert on them later if we want to.
    """
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (64, 64), color=128).save(folder / f"frame_{i:05d}.png")


class SplitRootRegressionTests(unittest.TestCase):
    """The dataset root and crop artefact root are intentionally different."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # Dataset root lives on a "local" path.
        self.dataset_root = Path(self._tmp.name) / "local" / "fall_local"
        # Crop artefact root lives on a "Drive" path.
        self.drive_root = Path(self._tmp.name) / "drive" / "MyDrive" / "fall_detection"
        # Perception root on Drive (Issue 002 artefacts).
        self.perception_root = self.drive_root / "artifacts" / "perception"
        # Crop artefact root on Drive.
        self.crops_root = self.drive_root / "artifacts" / "crops"

    def _make_urfd_layout(self, clip_id: str = "urfd-debug-fall-01-cam0") -> Path:
        # Build the dataset tree under the LOCAL dataset root, NOT under
        # the drive artefact root. This is the configuration that broke
        # the old code: the dataset lives on one path, the artefact
        # writes go to a different path.
        clip_folder = self.dataset_root / "datasets" / "urfd" / clip_id
        _make_clip_frames(clip_folder, count=64)
        return clip_folder

    def test_process_clip_resolves_source_folder_under_layout_root(self) -> None:
        """Without the fix, this would crash with FileNotFoundError."""
        clip_id = "urfd-debug-fall-01-cam0"
        clip_folder = self._make_urfd_layout(clip_id)
        # Sanity: the dataset folder lives under the dataset root.
        self.assertTrue(clip_folder.is_dir())

        record = _make_clip_record(
            clip_id=clip_id,
            source_path=f"datasets/urfd/{clip_id}",
        )
        boxes_by_track = {1: [
            TrackedBox(frame_index=i, x_min=10, y_min=10,
                        x_max=50, y_max=50, confidence=0.8)
            for i in range(64)
        ]}
        _write_fake_perception_json(self.perception_root, clip_id,
                                      track_id=1,
                                      boxes=boxes_by_track[1])

        crop_config = CropConfig(output_size=32, margin=0.30, clip_length=32)
        outcome, _next_shard, _writer = _process_clip(
            clip_record=record,
            boxes_by_track=boxes_by_track,
            crop_config=crop_config,
            crops_root=self.crops_root,
            shard_padding=5,
            shard_index=0,
            layout_root=self.dataset_root,
        )
        # The fix landed: at least one window emitted, no frames_unreadable
        # skip reason. If the source_folder had been resolved against
        # crops_root.parent, the runner would have failed to discover
        # frames and recorded "empty_clip_folder" or "frames_unreadable".
        self.assertEqual(outcome.skipped_reasons, [])
        self.assertGreater(outcome.emitted_windows, 0,
                            msg=f"emitted={outcome.emitted_windows}, "
                                f"skipped={outcome.skipped_reasons}")

    def test_old_path_would_have_crashed(self) -> None:
        """Sanity: the OLD path (crops_root.parent / source_path) would
        resolve to a folder that does NOT contain the dataset frames.

        This test pins the regression by demonstrating that under the
        split-root layout, the OLD resolution produces a non-existent
        path. The new code uses ``layout_root`` instead, which the test
        above proves works.
        """
        clip_id = "urfd-debug-fall-01-cam0"
        clip_folder = self._make_urfd_layout(clip_id)

        # OLD (buggy) resolution:
        old_source_folder = self.crops_root.parent / f"datasets/urfd/{clip_id}"
        # NEW (correct) resolution:
        new_source_folder = self.dataset_root / f"datasets/urfd/{clip_id}"

        # The two paths differ — the bug was using the first when the
        # caller expected the second.
        self.assertNotEqual(old_source_folder, new_source_folder)
        # The buggy path does not exist; the correct one does.
        self.assertFalse(old_source_folder.is_dir())
        self.assertTrue(new_source_folder.is_dir())
        self.assertEqual(new_source_folder, clip_folder)


if __name__ == "__main__":
    unittest.main()