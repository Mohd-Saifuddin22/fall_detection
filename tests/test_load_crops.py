"""Tests for :mod:`pipeline_a.load_crops`.

Coverage target (per the Issue 005 Step 1 task spec):

- Synthetic shards generated via :class:`cropping.shard_writer.ShardWriter`
  so the loader is exercised against the exact on-disk format it
  reads in production.
- Tensor shape / dtype / RGB channel layout.
- Normalization constants match the HF ``VideoMAEImageProcessor``
  defaults (``MCG-NJU/videomae-base``).
- Label encoding is correct (no_fall→0, fall→1).
- Provenance + missing-frame mask round-trip.
- Frame offsets sort correctly even when tar member order is not
  numeric.
- Failure paths: duplicate frame offset, missing frame offset,
  inconsistent provenance, invalid ``T``, missing required
  metadata, missing shard path.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

# Repo-root sys.path injection (mirrors the other test modules).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.manifests import FallLabel
from cropping.shard_writer import ShardWriter, encode_jpeg_bytes, read_shard, safe_member_name
from pipeline_a import (
    ALLOWED_T,
    DEFAULT_T,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CHANNELS,
    REQUIRED_METADATA_FIELDS,
    VIDEOMAE_PROCESSOR_MODEL_ID,
    CropShardsMissingError,
    DuplicateFrameOffsetError,
    InconsistentClipProvenanceError,
    InvalidClipLengthError,
    LoadedClip,
    MissingFrameOffsetError,
    MissingMetadataFieldError,
    int_to_label,
    label_to_int,
    list_clip_keys,
    load_clip_batch_from_shards,
    load_clip_tensor_from_shards,
    read_videomae_constants,
)


# ---------------------------------------------------------------------------
# Synthetic-shard builders
# ---------------------------------------------------------------------------


def _rgb_frame(
    r: int,
    g: int,
    b: int,
    *,
    h: int = IMAGE_SIZE,
    w: int = IMAGE_SIZE,
) -> np.ndarray:
    """Build one deterministic RGB uint8 frame.

    The deterministic contents let the parity tests assert
    exact pixel values rather than just shape + dtype.
    """
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = r % 256
    arr[..., 1] = g % 256
    arr[..., 2] = b % 256
    return arr


def _frame_metadata(
    *,
    clip_id: str,
    dataset: str,
    label: str,
    track_id: int,
    frame_index: int,
    frame_offset: int,
    source_path: str,
    missing_frame: bool = False,
    coverage: float = 1.0,
) -> dict[str, object]:
    """Build the per-frame metadata dict the writer consumes."""
    return {
        "clip_id": clip_id,
        "dataset": dataset,
        "label": label,
        "source_path": source_path,
        "track_id": track_id,
        "frame_index": frame_index,
        "frame_offset": frame_offset,
        "missing_frame": missing_frame,
        "coverage": coverage,
        "crop_config": {
            "output_size": IMAGE_SIZE,
            "margin": 0.30,
            "clip_length": DEFAULT_T,
        },
        "margin_used": 0.30,
        "shard_filename": "shard-000000.tar",
    }


def _write_clip_shard(
    shard_path: Path,
    *,
    clips: Sequence[tuple[str, str, int, str, list[bool], list[float]]],
    shard_index: int = 0,
    jpeg_quality: int = 90,
) -> Path:
    """Write one Issue 003-style shard with the given clips.

    Each ``clips`` entry is ``(clip_id, label, track_id, source_path,
    missing_per_frame, coverage_per_frame)``. Frames are
    default-True RGB except where the ``missing_per_frame`` slot
    says False (in which case the frame is the marker
    ``(224, 224, 3)`` array — irrelevant, the loader treats the
    sidecar's ``missing_frame`` flag as authoritative).
    """
    shard_path = Path(shard_path)
    with ShardWriter(shard_path, shard_index=shard_index) as writer:
        for (
            clip_id,
            label,
            track_id,
            source_path,
            missing_per_frame,
            coverage_per_frame,
        ) in clips:
            for frame_offset in range(DEFAULT_T):
                missing = missing_per_frame[frame_offset]
                coverage = coverage_per_frame[frame_offset]
                image = (
                    _rgb_frame(0, 0, 0)
                    if missing
                    else _rgb_frame(*clip_id_to_rgb(clip_id, frame_offset))
                )
                metadata = _frame_metadata(
                    clip_id=clip_id,
                    dataset="urfd",
                    label=label,
                    track_id=track_id,
                    frame_index=100 + frame_offset,
                    frame_offset=frame_offset,
                    source_path=source_path,
                    missing_frame=missing,
                    coverage=coverage,
                )
                writer.write_clip_member(
                    clip_key=clip_id,
                    frame_offset=frame_offset,
                    image=image,
                    metadata=metadata,
                    jpeg_quality=jpeg_quality,
                )
    return shard_path


def clip_id_to_rgb(clip_id: str, frame_offset: int) -> tuple[int, int, int]:
    """Deterministic pixel color per clip id + frame offset.

    Used so each clip's frames carry a distinct RGB fingerprint,
    letting the tests assert the loader reassembled frames in the
    right order without ambiguity.
    """
    base = sum(ord(c) for c in clip_id) * 17
    return (
        (base + frame_offset) % 256,
        (base + frame_offset * 3) % 256,
        (base + frame_offset * 5) % 256,
    )


# ---------------------------------------------------------------------------
# Tensor-shape + dtype tests
# ---------------------------------------------------------------------------


class TensorShapeTests(unittest.TestCase):
    """The output contract: ``(T, 3, H, W)`` float32, RGB."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(_rm_tree, Path(self._tmp))
        self.shard_path = _write_clip_shard(
            Path(self._tmp) / "shard-000000.tar",
            clips=[
                ("urfd-fall-01-cam0", "fall", 7, "datasets/urfd/fall-01.mp4",
                 [False] * DEFAULT_T, [1.0] * DEFAULT_T),
            ],
        )

    def test_tensor_shape_is_t_c_h_w(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="urfd-fall-01-cam0",
        )
        self.assertEqual(loaded.tensor.shape, (DEFAULT_T, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

    def test_tensor_dtype_is_float32(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="urfd-fall-01-cam0",
        )
        self.assertEqual(loaded.tensor.dtype, np.float32)

    def test_uint8_frames_shape_is_t_h_w_3(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="urfd-fall-01-cam0",
        )
        self.assertEqual(loaded.frames.shape, (DEFAULT_T, IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertEqual(loaded.frames.dtype, np.uint8)

    def test_rgb_channel_order_preserved(self) -> None:
        # Build a deliberately red frame; the loader must produce
        # a (3, H, W) slice where channel 0 (R) is much higher
        # than channel 2 (B). This proves we're not silently
        # converting to BGR anywhere along the way.
        clip_id = "urfd-rgb-canon"
        shard_path = Path(self._tmp) / "shard-rgb.tar"
        with ShardWriter(shard_path, shard_index=1) as writer:
            for frame_offset in range(DEFAULT_T):
                writer.write_clip_member(
                    clip_key=clip_id,
                    frame_offset=frame_offset,
                    image=_rgb_frame(220, 30, 60),  # strong red
                    metadata=_frame_metadata(
                        clip_id=clip_id, dataset="urfd",
                        label=FallLabel.FALL.value, track_id=1,
                        frame_index=100 + frame_offset,
                        frame_offset=frame_offset,
                        source_path="datasets/urfd/rgb.mp4",
                    ),
                )
        loaded = load_clip_tensor_from_shards(
            [shard_path], clip_key=clip_id,
        )
        first = loaded.tensor[0]
        # Channel 0 (red) must mean a larger normalised value
        # than channel 2 (blue) on a red-dominant frame. The
        # loader must NOT be doing any silent channel swap.
        self.assertGreater(
            float(first[0].mean()), float(first[2].mean()),
            msg=(
                f"channel means red={float(first[0].mean()):.3f} "
                f"blue={float(first[2].mean()):.3f} — red must dominate "
                "a red-dominant frame."
            ),
        )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class NormalizationTests(unittest.TestCase):
    """Local constants match HF ``VideoMAE-base`` processor defaults."""

    def test_local_constants_pinned_to_videomae_base(self) -> None:
        # Most reliable check: load the processor from cache and
        # compare. transformers + HF cache must be installed.
        try:
            from transformers import VideoMAEImageProcessor  # noqa: PLC0415
            proc = VideoMAEImageProcessor.from_pretrained(
                VIDEOMAE_PROCESSOR_MODEL_ID,
            )
        except Exception as exc:
            self.skipTest(
                f"VideoMAEImageProcessor not importable / not cached locally: {exc}"
            )
        self.assertEqual(tuple(proc.image_mean), IMAGENET_MEAN)
        self.assertEqual(tuple(proc.image_std), IMAGENET_STD)
        # The processor's size.shortest_edge is 224 for videomae-base.
        self.assertEqual(int(proc.size["shortest_edge"]), IMAGE_SIZE)

    def test_local_constants_are_documented_imagenet_values(self) -> None:
        # Sanity — these constants are the canonical ImageNet
        # values reported across the literature. A future refactor
        # that drifts them away from this canonical set would be a
        # regression for any model that trained on ImageNet stats.
        self.assertEqual(
            IMAGENET_MEAN, (0.485, 0.456, 0.406),
        )
        self.assertEqual(
            IMAGENET_STD, (0.229, 0.224, 0.225),
        )

    def test_tensor_normalisation_is_byte_exact_at_clip_level(self) -> None:
        # Pick deterministic pixel values and run them through the
        # loader's normalisation pipeline by hand. Tensor values
        # must match the same arithmetic.
        clip_id = "urfd-normalisation-canon"
        shard_path = _write_clip_shard(
            Path(tempfile.mkdtemp()) / "shard-norm.tar",
            clips=[(clip_id, "fall", 1, "x.mp4",
                    [False] * DEFAULT_T, [1.0] * DEFAULT_T)],
            shard_index=2,
        )
        try:
            loaded = load_clip_tensor_from_shards(
                [shard_path], clip_key=clip_id,
            )
        finally:
            _rm_tree(shard_path.parent)
        # Spot-check the first frame's tensor against a hand
        # computation of ImageNet normalisation.
        first_frames_uint8 = loaded.frames[0]
        first_tensor = loaded.tensor[0]
        expected = first_frames_uint8.astype(np.float32) / 255.0
        expected = (expected - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
            IMAGENET_STD, dtype=np.float32
        )
        # HWC expected → CHW for comparison.
        expected_chw = expected.transpose(2, 0, 1)
        np.testing.assert_allclose(first_tensor, expected_chw, rtol=1e-6)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------


class LabelEncodingTests(unittest.TestCase):
    """``no_fall`` → 0, ``fall`` → 1, anything else → :class:`ValueError`."""

    def test_no_fall_to_zero(self) -> None:
        self.assertEqual(label_to_int(FallLabel.NO_FALL.value), 0)

    def test_fall_to_one(self) -> None:
        self.assertEqual(label_to_int(FallLabel.FALL.value), 1)

    def test_int_to_label_round_trip(self) -> None:
        self.assertEqual(int_to_label(0), FallLabel.NO_FALL.value)
        self.assertEqual(int_to_label(1), FallLabel.FALL.value)

    def test_unknown_label_raises(self) -> None:
        with self.assertRaises(ValueError):
            label_to_int("unknown_label")
        with self.assertRaises(ValueError):
            int_to_label(2)


# ---------------------------------------------------------------------------
# Provenance round-trip
# ---------------------------------------------------------------------------


class ProvenanceRoundTripTests(unittest.TestCase):
    """Provenance + missing-frame mask survive the loader pass."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(_rm_tree, Path(self._tmp))
        self.shard_path = _write_clip_shard(
            Path(self._tmp) / "shard-000000.tar",
            clips=[
                ("urfd-fall-01-cam0_t7_w0", "fall", 7,
                 "datasets/urfd/fall-01.mp4",
                 # Frames 0 and 7 missing.
                 [True, False, False, False, False, False, False, True,
                  False, False, False, False, False, False, False, False],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            ],
        )

    def test_required_provenance_fields_round_trip(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path],
            clip_key="urfd-fall-01-cam0_t7_w0",
        )
        self.assertEqual(loaded.clip_id, "urfd-fall-01-cam0_t7_w0")
        self.assertEqual(loaded.dataset, "urfd")
        self.assertEqual(loaded.label, FallLabel.FALL.value)
        self.assertEqual(loaded.track_id, 7)
        self.assertEqual(loaded.source_path, "datasets/urfd/fall-01.mp4")

    def test_missing_mask_round_trips(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path],
            clip_key="urfd-fall-01-cam0_t7_w0",
        )
        self.assertEqual(loaded.missing_mask.dtype, bool)
        self.assertEqual(loaded.missing_mask.shape, (DEFAULT_T,))
        self.assertEqual(
            [bool(v) for v in loaded.missing_mask],
            [True, False, False, False, False, False, False, True,
             False, False, False, False, False, False, False, False],
        )
        self.assertEqual(loaded.missing_frame_count, 2)

    def test_coverage_is_average_over_frames(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path],
            clip_key="urfd-fall-01-cam0_t7_w0",
        )
        # 14 / 16 = 0.875.
        self.assertAlmostEqual(loaded.coverage, 14 / DEFAULT_T, places=6)

    def test_frame_indices_and_offsets_round_trip(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path],
            clip_key="urfd-fall-01-cam0_t7_w0",
        )
        self.assertEqual(
            loaded.frame_offsets, tuple(range(DEFAULT_T)),
        )
        # Absolute frame_index started at 100 + offset.
        self.assertEqual(
            loaded.frame_indices,
            tuple(100 + offset for offset in range(DEFAULT_T)),
        )

    def test_shard_filename_provenance(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path],
            clip_key="urfd-fall-01-cam0_t7_w0",
        )
        self.assertEqual(loaded.shard_filename, "shard-000000.tar")


# ---------------------------------------------------------------------------
# Sort-by-frame-offset behaviour
# ---------------------------------------------------------------------------


class FrameOffsetSortTests(unittest.TestCase):
    """``frame_offsets`` reflect numeric sort regardless of tar member order."""

    def test_loader_sorts_by_numeric_offset_not_member_name(self) -> None:
        # Hand-roll a shard where the metadata sidecars claim
        # offsets in a non-numeric order. PIL/JPEG writer always
        # writes them in numeric order too, but the **loader** is
        # the contract here — it must sort by the metadata's
        # ``frame_offset``, not by member name.
        clip_id = "urfd-shuffle-order"
        shard_path = Path(tempfile.mkdtemp()) / "shard-000007.tar"
        try:
            offsets_in_storage_order = [9, 0, 5, 1, 4, 2, 7, 3, 6, 8, 10, 11,
                                         12, 13, 14, 15]
            for frame_offset in offsets_in_storage_order:
                # Build a deterministic frame whose dominant
                # channel signals its offset (so per-frame pixels
                # are unique).
                intensity = (frame_offset + 1) * 16
                frame = _rgb_frame(intensity, intensity, intensity)
                metadata = _frame_metadata(
                    clip_id=clip_id, dataset="urfd",
                    label=FallLabel.NO_FALL.value, track_id=1,
                    frame_index=100 + frame_offset,
                    frame_offset=frame_offset,
                    source_path="x.mp4",
                )
            # The above loop overwrote the same in-memory image;
            # rebuild with the right values per offset.
            with ShardWriter(shard_path, shard_index=7) as writer:
                for frame_offset in offsets_in_storage_order:
                    intensity = (frame_offset + 1) * 16
                    frame = _rgb_frame(intensity, intensity, intensity)
                    metadata = _frame_metadata(
                        clip_id=clip_id, dataset="urfd",
                        label=FallLabel.NO_FALL.value, track_id=1,
                        frame_index=100 + frame_offset,
                        frame_offset=frame_offset,
                        source_path="x.mp4",
                    )
                    writer.write_clip_member(
                        clip_key=clip_id,
                        frame_offset=frame_offset,
                        image=frame,
                        metadata=metadata,
                    )

            loaded = load_clip_tensor_from_shards(
                [shard_path], clip_key=clip_id,
            )
            # Offsets are 0..15 in numeric sort order.
            self.assertEqual(
                loaded.frame_offsets, tuple(range(DEFAULT_T)),
            )
            # Frame 0 corresponds to the lowest offset, so its
            # tensor must carry the lowest-channel offset
            # signature.
            first_frame_uint8 = loaded.frames[0]
            # Intensity for offset 0 was 16 → mean ≈ 16 (cast to
            # uint8, so exact 16/255 ≈ 0.0627).
            self.assertAlmostEqual(float(first_frame_uint8.mean()),
                                   16.0, places=2)
        finally:
            _rm_tree(shard_path.parent)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class FailurePathTests(unittest.TestCase):
    """Negative tests: malformed shards raise the right errors."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(_rm_tree, Path(self._tmp))
        self.shards_root = Path(self._tmp)
        self.good_shard = _write_clip_shard(
            self.shards_root / "shard-good.tar",
            clips=[
                ("urfd-good-clip", "fall", 1, "x.mp4",
                 [False] * DEFAULT_T, [1.0] * DEFAULT_T),
            ],
        )

    # --- shape / T -----------------------------------------------------

    def test_invalid_T_raises(self) -> None:
        with self.assertRaises(InvalidClipLengthError):
            load_clip_tensor_from_shards(
                [self.good_shard], clip_key="urfd-good-clip", T=8,
            )

    def test_default_T_is_16(self) -> None:
        self.assertEqual(DEFAULT_T, 16)

    def test_allowed_T_pins_to_16_and_32(self) -> None:
        self.assertEqual(set(ALLOWED_T), {16, 32})

    # --- duplicate + missing offsets -----------------------------------

    def _write_custom_shard(
        self,
        name: str,
        clip_id: str,
        frame_offsets: Sequence[int],
    ) -> Path:
        """Write a shard with one clip whose frame offsets follow ``frame_offsets``."""
        shard_path = self.shards_root / name
        with ShardWriter(shard_path, shard_index=99) as writer:
            for frame_offset in frame_offsets:
                intensity = (frame_offset + 1) * 16 if frame_offset >= 0 else 0
                writer.write_clip_member(
                    clip_key=clip_id,
                    frame_offset=frame_offset,
                    image=_rgb_frame(intensity, intensity, intensity),
                    metadata=_frame_metadata(
                        clip_id=clip_id, dataset="urfd",
                        label=FallLabel.FALL.value, track_id=1,
                        frame_index=100 + frame_offset,
                        frame_offset=frame_offset,
                        source_path="x.mp4",
                    ),
                )
        return shard_path

    def test_duplicate_frame_offset_raises(self) -> None:
        # Two image members for the same clip carrying the same
        # ``frame_offset`` — duplicate. Constructed by appending a
        # second image+meta pair to the tar with the SAME
        # ``frame_offset`` value but a unique member name. The
        # loader's image_member walker picks both up; the
        # duplicate-detection step trips before the contiguity
        # check fires.
        import tarfile

        clip_id = "urfd-dup-clip"
        shard_path = self._write_custom_shard(
            "shard-dup.tar", clip_id, list(range(DEFAULT_T)),
        )

        safe = safe_member_name(clip_id)
        with tarfile.open(shard_path, mode="a") as tar:
            # A second image member with a different name — same
            # ``frame_offset`` value via its meta sidecar.
            dup_frame = _rgb_frame(99, 99, 99)
            dup_jpg = encode_jpeg_bytes(dup_frame)
            dup_image_info = tarfile.TarInfo(name=f"{safe}_0000_dup.image.jpg")
            dup_image_info.size = len(dup_jpg)
            dup_image_info.mtime = 0
            tar.addfile(dup_image_info, io.BytesIO(dup_jpg))

            dup_meta_info = tarfile.TarInfo(name=f"{safe}_0000_dup.meta.json")
            dup_meta_payload = _to_meta_payload(
                clip_id=clip_id,
                frame_offset=0,
                frame_index=200,
            )
            dup_meta_info.size = len(dup_meta_payload)
            dup_meta_info.mtime = 0
            tar.addfile(dup_meta_info, io.BytesIO(dup_meta_payload))

        with self.assertRaises(DuplicateFrameOffsetError):
            load_clip_tensor_from_shards(
                [shard_path], clip_key=clip_id,
            )

    def test_missing_frame_offset_raises(self) -> None:
        # Skip frame_offset=5 → not contiguous.
        clip_id = "urfd-missing-offset"
        offsets = [o for o in range(DEFAULT_T) if o != 5]
        shard_path = self._write_custom_shard(
            "shard-missing.tar", clip_id, offsets,
        )
        with self.assertRaises(MissingFrameOffsetError):
            load_clip_tensor_from_shards(
                [shard_path], clip_key=clip_id,
            )

    def test_clip_with_wrong_frame_count_raises(self) -> None:
        # 17 frames in a T=16 clip — too many.
        clip_id = "urfd-long-clip"
        shard_path = self._write_custom_shard(
            "shard-long.tar", clip_id, list(range(17)),
        )
        with self.assertRaises(InvalidClipLengthError):
            load_clip_tensor_from_shards(
                [shard_path], clip_key=clip_id,
            )

    def test_clip_with_too_few_frame_count_raises_missing_offset(self) -> None:
        # 8 frames in a T=16 clip — too few. Surfaced as
        # ``MissingFrameOffsetError`` because the obvious
        # diagnostic is "offsets 8..15 are missing".
        clip_id = "urfd-short-clip"
        shard_path = self._write_custom_shard(
            "shard-short.tar", clip_id, [0, 1, 2, 3, 4, 5, 6, 7],
        )
        with self.assertRaises(MissingFrameOffsetError):
            load_clip_tensor_from_shards(
                [shard_path], clip_key=clip_id,
            )

    # --- provenance consistency ---------------------------------------

    def test_inconsistent_provenance_raises(self) -> None:
        clip_id_a = "urfd-inconsistent-clip-a"
        clip_id_b = "urfd-inconsistent-clip-b"
        # Same offset key — but different clip_ids — would be
        # filtered by clip_key. To test inconsistent provenance,
        # we put BOTH clips under the same clip_id in the shard's
        # metadata by writing it manually. Simpler: write two
        # clips with different labels but force the loader to
        # see them as one clip.
        # Easier path: use two clips' frames with different
        # ``source_path`` under one clip_id.
        shard_path = self.shards_root / "shard-inconsistent.tar"
        with ShardWriter(shard_path, shard_index=8) as writer:
            for frame_offset in range(DEFAULT_T):
                # Half the frames have source_path A, half B.
                src = "x.mp4" if frame_offset < DEFAULT_T // 2 else "y.mp4"
                writer.write_clip_member(
                    clip_key="urfd-inconsistent-clip",
                    frame_offset=frame_offset,
                    image=_rgb_frame(64, 64, 64),
                    metadata=_frame_metadata(
                        clip_id="urfd-inconsistent-clip", dataset="urfd",
                        label=FallLabel.FALL.value, track_id=1,
                        frame_index=100 + frame_offset,
                        frame_offset=frame_offset,
                        source_path=src,
                    ),
                )
        with self.assertRaises(InconsistentClipProvenanceError):
            load_clip_tensor_from_shards(
                [shard_path], clip_key="urfd-inconsistent-clip",
            )

    # --- required metadata --------------------------------------------

    def test_missing_required_metadata_field_raises(self) -> None:
        # Hand-roll a metadata dict missing ``source_path``.
        clip_id = "urfd-bad-meta"
        shard_path = self.shards_root / "shard-bad-meta.tar"
        with ShardWriter(shard_path, shard_index=10) as writer:
            for frame_offset in range(DEFAULT_T):
                image = _rgb_frame(64, 64, 64)
                bad_meta = _frame_metadata(
                    clip_id=clip_id, dataset="urfd",
                    label=FallLabel.FALL.value, track_id=1,
                    frame_index=100 + frame_offset,
                    frame_offset=frame_offset,
                    source_path="x.mp4",
                )
                del bad_meta["source_path"]
                writer.write_clip_member(
                    clip_key=clip_id,
                    frame_offset=frame_offset,
                    image=image,
                    metadata=bad_meta,
                )
        with self.assertRaises(MissingMetadataFieldError):
            load_clip_tensor_from_shards(
                [shard_path], clip_key=clip_id,
            )

    # --- missing shard path -------------------------------------------

    def test_missing_shard_path_raises_with_clear_message(self) -> None:
        # The error must mention Issue 003 so the user can recover.
        with self.assertRaises(CropShardsMissingError) as ctx:
            load_clip_tensor_from_shards(
                [self.shards_root / "does-not-exist.tar"],
                clip_key="urfd-anything",
            )
        message = str(ctx.exception)
        self.assertIn("not found", message.lower())
        self.assertIn("Issue 003", message)

    def test_missing_clip_id_raises_lookup_error(self) -> None:
        # Sanity: a present shard with no matching clip_id
        # surfaces a LookupError, NOT a generic crash.
        with self.assertRaises(LookupError):
            load_clip_tensor_from_shards(
                [self.good_shard], clip_key="urfd-not-in-shard",
            )


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------


class BatchLoaderTests(unittest.TestCase):
    """``load_clip_batch_from_shards`` builds ``(B, T, 3, H, W)``."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(_rm_tree, Path(self._tmp))
        self.shard_path = _write_clip_shard(
            Path(self._tmp) / "shard-000000.tar",
            clips=[
                ("urfd-clip-a", FallLabel.FALL.value, 1, "a.mp4",
                 [False] * DEFAULT_T, [1.0] * DEFAULT_T),
                ("urfd-clip-b", FallLabel.NO_FALL.value, 2, "b.mp4",
                 [False] * DEFAULT_T, [1.0] * DEFAULT_T),
            ],
        )

    def test_batch_shape_and_dtype(self) -> None:
        batch, clips = load_clip_batch_from_shards(
            [self.shard_path],
            clip_keys=["urfd-clip-a", "urfd-clip-b"],
        )
        self.assertEqual(batch.shape, (2, DEFAULT_T, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(batch.dtype, np.float32)
        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0].clip_id, "urfd-clip-a")
        self.assertEqual(clips[1].clip_id, "urfd-clip-b")

    def test_batch_returns_individual_clips_in_order(self) -> None:
        _, clips = load_clip_batch_from_shards(
            [self.shard_path],
            clip_keys=["urfd-clip-b", "urfd-clip-a"],
        )
        self.assertEqual([c.clip_id for c in clips],
                         ["urfd-clip-b", "urfd-clip-a"])

    def test_empty_clip_keys_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_clip_batch_from_shards([self.shard_path], clip_keys=[])


# ---------------------------------------------------------------------------
# Convenience surface
# ---------------------------------------------------------------------------


class ConvenienceTests(unittest.TestCase):
    """``list_clip_keys`` + ``read_videomae_constants``."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(_rm_tree, Path(self._tmp))
        self.shard_path = _write_clip_shard(
            Path(self._tmp) / "shard-000000.tar",
            clips=[
                ("urfd-clip-a", "fall", 1, "a.mp4",
                 [False] * DEFAULT_T, [1.0] * DEFAULT_T),
                ("urfd-clip-b", "no_fall", 2, "b.mp4",
                 [False] * DEFAULT_T, [1.0] * DEFAULT_T),
            ],
        )

    def test_list_clip_keys_returns_shard_clips(self) -> None:
        keys = list_clip_keys(self.shard_path)
        self.assertIn("urfd-clip-a", keys)
        self.assertIn("urfd-clip-b", keys)

    def test_list_clip_keys_missing_file_raises_crop_shards_error(self) -> None:
        with self.assertRaises(CropShardsMissingError):
            list_clip_keys(Path(self._tmp) / "missing.tar")

    def test_read_videomae_constants_round_trip(self) -> None:
        cfg = read_videomae_constants()
        self.assertEqual(cfg["model_id"], VIDEOMAE_PROCESSOR_MODEL_ID)
        self.assertEqual(cfg["image_size"], IMAGE_SIZE)
        self.assertEqual(tuple(cfg["image_mean"]), IMAGENET_MEAN)
        self.assertEqual(tuple(cfg["image_std"]), IMAGENET_STD)
        self.assertEqual(cfg["default_T"], DEFAULT_T)
        self.assertEqual(list(cfg["allowed_T"]), list(ALLOWED_T))


# ---------------------------------------------------------------------------
# Constants + module surface
# ---------------------------------------------------------------------------


class ModuleSurfaceTests(unittest.TestCase):
    """Top-level exports exist + REQUIRED_METADATA_FIELDS covers the brief."""

    def test_required_metadata_fields_matches_brief(self) -> None:
        self.assertEqual(
            set(REQUIRED_METADATA_FIELDS),
            {
                "clip_id",
                "dataset",
                "label",
                "track_id",
                "frame_index",
                "frame_offset",
                "source_path",
            },
        )

    def test_module_exports_match_brief(self) -> None:
        import pipeline_a  # noqa: PLC0415
        for name in (
            "LoadedClip",
            "load_clip_tensor_from_shards",
            "load_clip_batch_from_shards",
            "IMAGENET_MEAN",
            "IMAGENET_STD",
            "IMAGE_SIZE",
            "ALLOWED_T",
            "DEFAULT_T",
            "label_to_int",
            "int_to_label",
        ):
            self.assertTrue(hasattr(pipeline_a, name),
                            msg=f"pipeline_a.{name} must be exported.")


# ---------------------------------------------------------------------------
# Regression: real-Issue-003 multi-window shards
# ---------------------------------------------------------------------------


def _write_multi_window_shard(
    shard_path: Path,
    *,
    source_clip_id: str,
    windows: Sequence[tuple[str, int, str, str, list[bool], list[float]]],
) -> Path:
    """Write one Issue 003-style shard with multiple windows of the same source.

    Each entry of ``windows`` is::

        (per_window_member_stem,
         track_id,
         label,
         source_path,
         missing_per_frame,
         coverage_per_frame)

    Every window's metadata sidecar carries the same
    ``clip_id`` (the bare source) — this is the real Issue 003
    layout: two windows from source ``"X"`` share the bare
    metadata id ``"X"`` but write to two different tar member
    stems (``"X_t7_w000_..."`` and ``"X_t7_w001_..."``).
    """
    shard_path = Path(shard_path)
    with ShardWriter(shard_path, shard_index=0) as writer:
        for window_stem, track_id, label, source_path, missing, coverage in windows:
            for frame_offset in range(DEFAULT_T):
                intensity = (sum(ord(c) for c in window_stem) + frame_offset) % 256
                writer.write_clip_member(
                    clip_key=window_stem,
                    frame_offset=frame_offset,
                    image=_rgb_frame(intensity, intensity // 2, intensity // 4),
                    metadata=_frame_metadata(
                        clip_id=source_clip_id, dataset="urfd",
                        label=label, track_id=track_id,
                        frame_index=200 + frame_offset,
                        frame_offset=frame_offset,
                        source_path=source_path,
                        missing_frame=missing[frame_offset],
                        coverage=coverage[frame_offset],
                    ),
                )
    return shard_path


class MultiWindowShardTests(unittest.TestCase):
    """Real-Issue-003 multi-window shards: bare clip_id is shared.

    Source clip ``"X"`` produces two windows:
    ``"X_t7_w000"`` and ``"X_t7_w001"``. Both write
    metadata sidecars with bare ``clip_id = "X"``. The OLD
    loader grouped on bare ``clip_id``, merged both windows
    into 32 frames, and raised ``InvalidClipLengthError``.
    The corrected loader groups on the per-window member
    stem and loads each window as its own 16-frame example.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(_rm_tree, Path(self._tmp))
        self.shard_path = _write_multi_window_shard(
            Path(self._tmp) / "shard-000000.tar",
            source_clip_id="X",
            windows=[
                (
                    "X_t7_w000", 7, FallLabel.FALL.value, "datasets/urfd/x.mp4",
                    [False] * DEFAULT_T, [1.0] * DEFAULT_T,
                ),
                (
                    "X_t7_w001", 7, FallLabel.FALL.value, "datasets/urfd/x.mp4",
                    [False] * DEFAULT_T, [1.0] * DEFAULT_T,
                ),
            ],
        )

    def test_list_clip_keys_returns_per_window_stems_only(self) -> None:
        # list_clip_keys reads the manifest — which the Issue 003
        # writer populates with the per-window keys it actually
        # accepted. No bare "X" should leak through.
        keys = list_clip_keys(self.shard_path)
        self.assertIn("X_t7_w000", keys)
        self.assertIn("X_t7_w001", keys)
        self.assertNotIn("X", keys,
                         msg="list_clip_keys must not surface bare source clip_id.")

    def test_each_window_loads_as_its_own_t_frames(self) -> None:
        for window_key in ("X_t7_w000", "X_t7_w001"):
            with self.subTest(window=window_key):
                loaded = load_clip_tensor_from_shards(
                    [self.shard_path], clip_key=window_key,
                )
                # Tensor + frames shape.
                self.assertEqual(
                    loaded.tensor.shape,
                    (DEFAULT_T, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
                )
                self.assertEqual(
                    loaded.frames.shape,
                    (DEFAULT_T, IMAGE_SIZE, IMAGE_SIZE, 3),
                )
                # Per-window identity vs bare source id.
                self.assertEqual(loaded.clip_key, window_key)
                self.assertEqual(loaded.clip_id, "X")

    def test_windows_are_not_merged(self) -> None:
        # Each window has different per-frame intensity
        # (sum(window_stem ord) + offset). If the windows were
        # merged, the union's frame_offsets would range 0..31
        # and the count would exceed DEFAULT_T — raising
        # InvalidClipLengthError. The point of the regression
        # test: prove we get two clean 16-frame tensors.
        a = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="X_t7_w000",
        )
        b = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="X_t7_w001",
        )
        # Frame 0 of each window is a different intensity.
        self.assertFalse(
            np.array_equal(a.frames[0], b.frames[0]),
            msg="Loaded windows must not be the same byte sequence.",
        )
        # Each window's frame_offsets are 0..T-1.
        self.assertEqual(a.frame_offsets, tuple(range(DEFAULT_T)))
        self.assertEqual(b.frame_offsets, tuple(range(DEFAULT_T)))

    def test_no_duplicate_offset_or_invalid_length_error_per_window(self) -> None:
        # Sanity: loading either window independently does NOT
        # raise a duplicate-offset or invalid-length error.
        # Each window has exactly one frame per offset, so the
        # loader's T-validations succeed.
        for window_key in ("X_t7_w000", "X_t7_w001"):
            with self.subTest(window=window_key):
                loaded = load_clip_tensor_from_shards(
                    [self.shard_path], clip_key=window_key,
                )
                self.assertEqual(
                    loaded.frame_offsets, tuple(range(DEFAULT_T)),
                )

    def test_batch_loads_each_window_as_distinct_example(self) -> None:
        # The regression test for the broader bug: the batch
        # loader must produce B = 2 distinct examples, not 1
        # merged one.
        from pipeline_a import load_clip_batch_from_shards  # noqa: PLC0415
        batch, clips = load_clip_batch_from_shards(
            [self.shard_path],
            clip_keys=["X_t7_w000", "X_t7_w001"],
        )
        self.assertEqual(batch.shape,
                         (2, DEFAULT_T, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual([c.clip_key for c in clips],
                         ["X_t7_w000", "X_t7_w001"])
        # And both examples are bare "X".
        self.assertEqual([c.clip_id for c in clips], ["X", "X"])

    def test_per_window_provenance_consistent(self) -> None:
        # All frames in a window carry the same provenance.
        # Since the writer built the metadata correctly, this
        # passes — but it's the regression that catches a buggy
        # loader that mixed window frames.
        loaded = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="X_t7_w000",
        )
        self.assertEqual(loaded.dataset, "urfd")
        self.assertEqual(loaded.label, FallLabel.FALL.value)
        self.assertEqual(loaded.track_id, 7)
        self.assertEqual(loaded.source_path, "datasets/urfd/x.mp4")


class PrefixCollisionGuardTests(unittest.TestCase):
    """Prefix-match guard: ``X_t7_w000`` must not capture ``X_t70_w000``.

    Both windows are written into the same shard; their bare
    metadata ``clip_id`` is identical (the source-clip id
    collision is yet another reason grouping on bare ``clip_id``
    is wrong). The trailing-underscore prefix
    ``X_t7_w000_`` is the disambiguation; the brief spells out
    the assert.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(_rm_tree, Path(self._tmp))
        # Two windows whose member names share a stem prefix
        # but differ in characters 4 / 7. The loader's prefix
        # is ``safe(clip_key) + "_"``.
        self.shard_path = _write_multi_window_shard(
            Path(self._tmp) / "shard-prefix.tar",
            source_clip_id="X",
            windows=[
                (
                    "X_t7_w000", 7, FallLabel.FALL.value, "datasets/urfd/x.mp4",
                    [False] * DEFAULT_T, [1.0] * DEFAULT_T,
                ),
                (
                    "X_t70_w000", 70, FallLabel.NO_FALL.value, "datasets/urfd/x.mp4",
                    [False] * DEFAULT_T, [1.0] * DEFAULT_T,
                ),
            ],
        )

    def test_loading_X_t7_w000_does_not_capture_X_t70_w000(self) -> None:
        loaded = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="X_t7_w000",
        )
        # ``track_id`` matches the X_t7_w000 window (7), not the
        # X_t70_w000 window (70). If the loader wrongly grabbed
        # both windows, track_id consistency would still pass
        # (because both share the bare clip_id), but coverage /
        # missing-frame count would either round up to 32 frames
        # or raise an InvalidClipLengthError — neither is what
        # we want.
        self.assertEqual(loaded.track_id, 7)
        self.assertEqual(loaded.frame_offsets, tuple(range(DEFAULT_T)))
        # And the second window still loads independently.
        loaded_70 = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="X_t70_w000",
        )
        self.assertEqual(loaded_70.track_id, 70)
        # Crucially: a different track_id proves the loader
        # did NOT collapse the two windows into a single example.
        self.assertNotEqual(loaded.track_id, loaded_70.track_id)

    def test_loading_X_t7_w000_returns_only_its_own_frames(self) -> None:
        # Sanity: a hand-computable frame-offset count check.
        # If the prefix guard failed, the loader would assemble
        # 32 frames, fail the count check, and surface
        # InvalidClipLengthError. With the guard in place,
        # only the 16 frames prefixed with ``X_t7_w000_`` are
        # gathered and the window loads cleanly.
        loaded = load_clip_tensor_from_shards(
            [self.shard_path], clip_key="X_t7_w000",
        )
        self.assertEqual(loaded.tensor.shape[0], DEFAULT_T)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_meta_payload(
    *,
    clip_id: str,
    frame_offset: int,
    frame_index: int,
) -> bytes:
    """Build a JSON-encoded metadata sidecar carrying one frame's fields.

    Used by tests that hand-roll a second image+meta pair to
    manufacture ``frame_offset`` duplicates the writer would
    never have produced in the first place.
    """
    import json  # local — kept narrow for test surfaces only.
    payload = {
        "clip_id": clip_id,
        "dataset": "urfd",
        "label": FallLabel.FALL.value,
        "source_path": "x.mp4",
        "track_id": 1,
        "frame_index": frame_index,
        "frame_offset": frame_offset,
        "missing_frame": False,
        "coverage": 1.0,
        "crop_config": {
            "output_size": IMAGE_SIZE,
            "margin": 0.30,
            "clip_length": DEFAULT_T,
        },
        "margin_used": 0.30,
        "shard_filename": "shard-dup.tar",
    }
    return json.dumps(payload).encode("utf-8")


def _rm_tree(path: Path) -> None:
    try:
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir():
                    _rm_tree(child)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
            try:
                path.rmdir()
            except OSError:
                pass
    except OSError:
        pass


if __name__ == "__main__":
    unittest.main()
