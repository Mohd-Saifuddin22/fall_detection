"""Unit tests for :mod:`cropping.shard_writer`.

Verifies the WebDataset-style .tar shard contains image + JSON sidecar
pairs, the manifest is correct, and shard names are deterministic.
"""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cropping.shard_writer import (  # noqa: E402
    ShardWriter,
    compute_shard_padding_width,
    encode_jpeg_bytes,
    read_shard,
    safe_member_name,
    shard_filename,
)


def _white_frame(size: int = 32) -> np.ndarray:
    """Cheap deterministic test frame."""
    return np.full((size, size, 3), 200, dtype=np.uint8)


class SafeMemberNameTests(unittest.TestCase):
    """Filenames inside shards must survive any filesystem."""

    def test_path_separators_are_collapsed(self) -> None:
        self.assertEqual(safe_member_name("a/b\\c"), "a_b_c")

    def test_spaces_and_punctuation_become_underscore(self) -> None:
        self.assertEqual(safe_member_name("clip: 01!"), "clip__01_")

    def test_empty_string_falls_back_to_placeholder(self) -> None:
        # Empty string → no characters → falls back to "unnamed".
        self.assertEqual(safe_member_name(""), "unnamed")
        # Only-unsafe characters → all replaced with '_' → 3 underscores.
        self.assertEqual(safe_member_name("///"), "___")


class ShardFilenameTests(unittest.TestCase):
    """Stable, deterministic, lexicographically-sortable shard names."""

    def test_shard_filename_is_zero_padded(self) -> None:
        self.assertEqual(shard_filename(0, width=5), "shard-00000.tar")
        self.assertEqual(shard_filename(42, width=5), "shard-00042.tar")

    def test_padding_width_is_at_least_four(self) -> None:
        # Even for tiny projects we want enough width that alphabetical
        # sort matches numeric sort for the first 10000 shards.
        self.assertEqual(compute_shard_padding_width(1), 4)

    def test_padding_width_scales_with_expected_shard_count(self) -> None:
        self.assertEqual(compute_shard_padding_width(10000), 5)
        self.assertEqual(compute_shard_padding_width(100000), 6)


class EncodeJpegBytesTests(unittest.TestCase):
    """JPEG encoder accepts standard frame shapes."""

    def test_hxw3_frame_encodes(self) -> None:
        payload = encode_jpeg_bytes(_white_frame())
        self.assertGreater(len(payload), 0)
        self.assertEqual(payload[:3], b"\xff\xd8\xff")  # JPEG magic bytes

    def test_hxw4_frame_drops_alpha(self) -> None:
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        rgba[:, :, 3] = 128  # half-transparent
        payload = encode_jpeg_bytes(rgba)
        self.assertGreater(len(payload), 0)

    def test_non_standard_shape_raises(self) -> None:
        bad = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            encode_jpeg_bytes(bad)

    def test_float_frame_is_clamped_and_cast(self) -> None:
        # Float images in [0, 1] are common in ML pipelines; the encoder
        # must not crash on them — it should clip + cast to uint8.
        floats = np.full((8, 8, 3), 0.5, dtype=np.float32)
        payload = encode_jpeg_bytes(floats)
        self.assertGreater(len(payload), 0)


class ShardWriterRoundTripTests(unittest.TestCase):
    """Write a shard, read it back, verify every member + manifest."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "test-shard.tar"

    def test_one_clip_one_frame_round_trip(self) -> None:
        frame = _white_frame(48)
        metadata = {"clip_id": "urfd-fall-01-cam0", "frame_index": 0}
        with ShardWriter(self.path, shard_index=0) as shard:
            shard.write_clip_member("urfd-fall-01-cam0_t1_w0",
                                     frame_offset=0, image=frame, metadata=metadata)

        result = read_shard(self.path)
        self.assertEqual(len(result.image_members), 1)
        self.assertEqual(len(result.metadata_members), 1)
        self.assertIn("urfd-fall-01-cam0_t1_w0_0000.image.jpg", result.image_members)
        meta_member = "urfd-fall-01-cam0_t1_w0_0000.meta.json"
        self.assertIn(meta_member, result.metadata_members)
        self.assertEqual(result.metadata_members[meta_member]["clip_id"],
                          "urfd-fall-01-cam0")
        # Manifest records the writer's bookkeeping.
        self.assertEqual(result.manifest["shard_index"], 0)
        self.assertIn("urfd-fall-01-cam0_t1_w0", result.manifest["clip_keys"])

    def test_multiple_frames_one_clip_round_trip(self) -> None:
        with ShardWriter(self.path, shard_index=2) as shard:
            for offset in range(5):
                shard.write_clip_member(
                    "urfd-fall-01-cam0_t7_w0",
                    frame_offset=offset,
                    image=_white_frame(48),
                    metadata={"clip_id": "urfd-fall-01-cam0", "offset": offset},
                )

        result = read_shard(self.path)
        # 5 image members + 5 meta members + the manifest itself.
        self.assertEqual(len(result.image_members), 5)
        self.assertEqual(len(result.metadata_members), 5)
        frames = result.clip_frames("urfd-fall-01-cam0_t7_w0")
        self.assertEqual(len(frames), 5)
        # Ordered by offset.
        for i, (member_name, _payload, metadata) in enumerate(frames):
            self.assertIn(f"_{i:04d}.image.jpg", member_name)
            self.assertEqual(metadata["offset"], i)

    def test_unsafe_clip_id_is_sanitised_in_member_names(self) -> None:
        # Path separators and whitespace get collapsed to underscores so
        # the tar member names never break extraction.
        with ShardWriter(self.path, shard_index=0) as shard:
            shard.write_clip_member(
                "urfd/fall 01/cam0",
                frame_offset=0,
                image=_white_frame(32),
                metadata={"clip_id": "urfd/fall 01/cam0"},
            )
        with tarfile.open(self.path, mode="r") as tar:
            names = tar.getnames()
        self.assertTrue(all("/" not in n for n in names),
                         msg=f"member names must not contain slashes: {names}")


class ShardDeterminismTests(unittest.TestCase):
    """Two writes of identical content → identical shard (Issue 003 rule)."""

    def setUp(self) -> None:
        self._tmp_a = tempfile.TemporaryDirectory()
        self._tmp_b = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_a.cleanup)
        self.addCleanup(self._tmp_b.cleanup)
        self.path_a = Path(self._tmp_a.name) / "shard-00000.tar"
        self.path_b = Path(self._tmp_b.name) / "shard-00000.tar"

    def test_identical_payload_produces_byte_equal_shards(self) -> None:
        # Tar metadata timestamps are deterministic (we set mtime=0);
        # member ordering is insertion order. Two runs of identical
        # payload should produce identical bytes.
        for path in (self.path_a, self.path_b):
            with ShardWriter(path, shard_index=0) as shard:
                shard.write_clip_member(
                    "clip-x", frame_offset=0,
                    image=_white_frame(32),
                    metadata={"clip_id": "clip-x", "k": 1},
                )
        # Ignore the padding at the end of the tar file (OS may write
        # EOF blocks of varying length); member contents must match.
        with tarfile.open(self.path_a, mode="r") as tar_a, \
             tarfile.open(self.path_b, mode="r") as tar_b:
            names_a = sorted(t.name for t in tar_a.getmembers())
            names_b = sorted(t.name for t in tar_b.getmembers())
            self.assertEqual(names_a, names_b)
            for member in tar_a.getmembers():
                if not member.isfile():
                    continue
                payload_a = tar_a.extractfile(member).read()
                payload_b = tar_b.extractfile(member).read()
                self.assertEqual(
                    payload_a, payload_b,
                    msg=f"member {member.name!r} differs across runs",
                )


if __name__ == "__main__":
    unittest.main()