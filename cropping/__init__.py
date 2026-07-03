"""Issue 003 — deterministic crop clip generator.

Public surface:
    - :mod:`cropping.clip_builder` — pure crop math (no I/O, no model).
    - :mod:`cropping.track_windows` — track → fixed-length windows
      with deterministic gap / coverage policy.
    - :mod:`cropping.shard_writer` — WebDataset-style .tar shards.
    - :mod:`cropping.runner` — end-to-end orchestration that consumes
      Issue 002 artefacts and writes shards to Drive.

Pipeline:
    Issue 002 detections.json
        → load_track_boxes_for_clip (group by track_id)
        → build_windows_for_track (gap/coverage policy)
        → compute_crop_geometry + apply_crop_to_frame (per frame)
        → ShardWriter.write_clip_member (image + JSON sidecar per frame)
"""

from __future__ import annotations

from cropping import clip_builder, runner, shard_writer, track_windows

__all__: tuple[str, ...] = (
    "clip_builder",
    "runner",
    "shard_writer",
    "track_windows",
)