"""Pipeline A — VideoMAE-ready crop-shard loader (Issue 005 Step 1).

Reads Issue 003 WebDataset-style ``.tar`` crop shards and assembles
each clip/window into a VideoMAE-ready ``(T, 3, H, W)`` float32
tensor, plus the provenance and missing-frame mask the trainer
needs.

This module is **data-prep only**:

- It does NOT import or call YOLO, ByteTrack, the perception
  modules, or the crop runner. (Issue 003 produced the shards
  on disk; this loader only re-reads them.)
- It does NOT load or train a VideoMAE model. (That lands in
  Issue 009 — this Step is data-prep only.)
- It does NOT compute metrics.

Once real VideoMAE training lands in Issue 009 the loader is
consumed unchanged — its output contract is the trainer's input
contract.

Tensor contract (Pipeline A classifier-head spec)
------------------------------------------------

::

    single clip: (T, C, H, W)
    batch:       (B, T, C, H, W)

with ``T ∈ {16, 32}`` (default 16 — see :data:`ALLOWED_T`),
``C = 3`` (RGB), ``H = W = 224``, ``dtype = float32``.

``LoadedClip.tensor`` carries the contract shape; ``LoadedClip.frames``
carries the same frames as ``(T, H, W, 3) uint8`` so debug / plot
paths avoid a JPEG re-decode.

Normalisation constants — pinned to the HF VideoMAEImageProcessor
defaults for ``"MCG-NJU/videomae-base"`` (verified by parity test):

::

    image_mean = [0.485, 0.456, 0.406]
    image_std  = [0.229, 0.224, 0.225]

The unit tests pin these against a freshly-loaded HF
``VideoMAEImageProcessor`` so any future processor-default drift
trips the test suite.

Provenance (per :class:`LoadedClip`)
------------------------------------

::

    clip_id
    dataset
    label          ("fall" or "no_fall")
    track_id
    source_path
    missing_mask    (T,) bool — True at a slot whose source frame was missing
    coverage       float in [0, 1] — fraction of slots that were real
    missing_frame_count   int
    frame_indices  (T,) int — absolute frame indices in the original timeline
    shard_filename(s)      provenance to where the clip came from

Label encoding
--------------

::

    no_fall → 0
    fall    → 1

(Used by :func:`label_to_int` and by callers that need an int
target tensor.)
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from cropping.shard_writer import ShardReadResult, read_shard
from data.manifests import FallLabel


# ---------------------------------------------------------------------------
# Constants — pinned to MCG-NJU/videomae-base
# ---------------------------------------------------------------------------


#: Default ``T`` (frames per clip). Matches the PRD's Pipeline A
#: starter. Override via the ``T=`` kwarg only when an existing
#: shard was written with a different size — do NOT silently
#: pad / truncate.
DEFAULT_T: int = 16

#: The only legal values of ``T``. Members of the set are the
#: PRD-approved Step 1 starter values; longer / shorter clips are
#: not supported at this step.
ALLOWED_T: tuple[int, ...] = (16, 32)

#: ImageNet normalisation constants — verified to match the
#: ``"MCG-NJU/videomae-base"`` processor's defaults at unit-test
#: time. See ``test_local_constants_match_videomae_base_processor``.
IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)

#: The HF ``VideoMAEImageProcessor.size.shortest_edge`` for
#: ``videomae-base`` (verified by parity test). The loader resizes
#: every frame to ``(IMAGE_SIZE, IMAGE_SIZE)``.
IMAGE_SIZE: int = 224

#: Number of channels — RGB only (the writer always emits RGB
#: JPEGs; the loader does not support grayscale/bayer).
NUM_CHANNELS: int = 3

#: HF model id the loader is pinned to. Single point of
#: provenance for "what VideoMAE are we targeting?".
VIDEOMAE_PROCESSOR_MODEL_ID: str = "MCG-NJU/videomae-base"

#: Required metadata fields per Issue 005 Step 1 spec. Every
#: ``.meta.json`` sidecar the loader consumes must carry these
#: keys. Missing → :class:`MissingMetadataFieldError`.
REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "clip_id",
    "dataset",
    "label",
    "track_id",
    "frame_index",
    "frame_offset",
    "source_path",
)

#: Provenance fields preserved on :class:`LoadedClip`. Mirrors
#: the brief's "Preserve provenance" list.
PRESERVED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "clip_id",
    "dataset",
    "label",
    "track_id",
    "source_path",
    "missing_frame",
    "coverage",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CropShardsMissingError(FileNotFoundError):
    """Raised when one or more crop shard paths are missing on disk.

    Inherits from :class:`FileNotFoundError` so `isinstance(err, OSError)`
    and similar checks keep working, while the message itself points
    the caller at Issue 003 — the pipeline step that produces
    shards.
    """


class MissingMetadataFieldError(ValueError):
    """Raised when a per-frame ``.meta.json`` sidecar lacks a required field."""


class InconsistentClipProvenanceError(ValueError):
    """Raised when frames disagree on a provenance field (clip_id,
    dataset, label, track_id, source_path)."""


class InvalidClipLengthError(ValueError):
    """Raised when a clip's frame count does not equal ``T`` after
    frame_offset-based deduplication, or when ``T`` is not in
    :data:`ALLOWED_T`."""


class DuplicateFrameOffsetError(ValueError):
    """Raised when two frames in the same clip carry the same frame_offset."""


class MissingFrameOffsetError(ValueError):
    """Raised when frame offsets within one clip are not contiguous 0..T-1."""


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedClip:
    """One clip / window's decoded VideoMAE-ready tensor + provenance.

    The float32 VideoMAE-ready tensor lives in ``tensor``. The
    uint8 RGB frame array is also kept (in ``frames``) so
    debugging / plotting paths avoid a JPEG re-decode.

    Both ``frames`` and ``tensor`` have length ``T`` and reflect
    the **frame_offset sort order**: ``frames[i]`` / ``tensor[i]``
    corresponds to ``frame_offsets[i]`` (and to ``frame_indices[i]``,
    the absolute source timeline index).
    """

    clip_id: str
    dataset: str
    label: str
    track_id: int
    source_path: str
    frames: np.ndarray           # (T, H, W, 3) uint8
    tensor: np.ndarray           # (T, 3, H, W) float32, normalized
    missing_mask: np.ndarray     # (T,) bool
    coverage: float
    missing_frame_count: int
    frame_indices: tuple[int, ...]    # absolute frame indices
    frame_offsets: tuple[int, ...]     # 0..T-1, sort order of frames/tensor
    shard_filename: str


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------


def label_to_int(label: str) -> int:
    """Encode a string label to the VideoMAE training target.

    no_fall → 0, fall → 1. Any other value raises
    :class:`ValueError` — the loader must not invent a label.
    """
    if label == FallLabel.NO_FALL.value:
        return 0
    if label == FallLabel.FALL.value:
        return 1
    raise ValueError(
        f"Unknown label {label!r}; expected one of {[v.value for v in FallLabel]}."
    )


def int_to_label(value: int) -> str:
    """Inverse of :func:`label_to_int`."""
    if value == 0:
        return FallLabel.NO_FALL.value
    if value == 1:
        return FallLabel.FALL.value
    raise ValueError(f"Unknown label integer {value!r}; expected 0 or 1.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_T(T: int) -> None:
    if T not in ALLOWED_T:
        raise InvalidClipLengthError(
            f"T must be one of {ALLOWED_T}, got {T!r}. "
            "Issue 005 Step 1 only supports 16 / 32 frame clips."
        )


def _check_required_metadata(metadata: dict[str, object], *, source: str) -> None:
    """Confirm every required field is present in one metadata sidecar."""
    for field_name in REQUIRED_METADATA_FIELDS:
        if field_name not in metadata:
            raise MissingMetadataFieldError(
                f"Per-frame metadata from {source} is missing required "
                f"field {field_name!r}. Required: {list(REQUIRED_METADATA_FIELDS)}."
            )


def _canonical_label(label: str) -> str:
    """Normalise ``"fall"`` / ``"no_fall"`` to a ``FallLabel.value`` string."""
    if isinstance(label, FallLabel):
        return label.value
    if label == FallLabel.FALL.value:
        return FallLabel.FALL.value
    if label == FallLabel.NO_FALL.value:
        return FallLabel.NO_FALL.value
    raise ValueError(
        f"Unknown label {label!r}; expected one of "
        f"{[v.value for v in FallLabel]}."
    )


def _decode_jpegs_to_uint8(jpeg_bytes_list: Sequence[bytes], H: int, W: int) -> np.ndarray:
    """Decode JPEG bytes to ``(T, H, W, 3) uint8`` RGB.

    PIL is the JPEG decoder; it is in the approved stack and used
    by Issue 003's writer. Frames smaller or larger than
    ``(W, H)`` are resized with the default ``Image.BILINEAR``
    resampler — the same resampler the HF VideoMAE processor uses
    via ``torchvision.transforms.functional.resize``.
    """
    arr = np.empty((len(jpeg_bytes_list), H, W, 3), dtype=np.uint8)
    for i, jb in enumerate(jpeg_bytes_list):
        if not jb:
            raise ValueError(
                f"Frame {i} in the rebuilt clip has empty JPEG bytes; "
                f"the shard is malformed."
            )
        img = Image.open(io.BytesIO(jb)).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        arr[i] = np.asarray(img, dtype=np.uint8)
    return arr


def _normalize_hwc_uint8_to_chw_float32(
    rgb_uint8: np.ndarray,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """``(T, H, W, 3) uint8`` → ``(T, 3, H, W)`` float32 ImageNet-normalised.

    Pure numpy — torch / torchvision are intentionally absent
    from the dev environment (Issue 001 review: torch is
    installed by Colab per-render). The arithmetic is the
    canonical ImageNet recipe:

        float = ((uint8 / 255) - mean) / std
    """
    if rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input, got {rgb_uint8.dtype}.")
    if rgb_uint8.ndim != 4 or rgb_uint8.shape[-1] != NUM_CHANNELS:
        raise ValueError(
            f"Expected (T, H, W, 3) input, got shape {rgb_uint8.shape}."
        )
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, NUM_CHANNELS)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, NUM_CHANNELS)
    normalised = (rgb_float - mean_arr) / std_arr
    # HWC → CHW per frame, then stack on time.
    return normalised.transpose(0, 3, 1, 2).copy()  # copy so caller can mutate safely


def _build_provenance_strict(
    frames_metadata: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Confirm every per-frame metadata row agrees on the provenance fields.

    Returns the canonical values (so the caller can lift them
    onto :class:`LoadedClip` without re-reading the inputs).
    """
    canonical: dict[str, object] = {}
    for field_name in (
        "clip_id",
        "dataset",
        "label",
        "track_id",
        "source_path",
    ):
        first_value = frames_metadata[0][field_name]
        for index, metadata in enumerate(frames_metadata[1:], start=1):
            if metadata[field_name] != first_value:
                raise InconsistentClipProvenanceError(
                    f"Frame {index} disagrees with frame 0 on provenance "
                    f"field {field_name!r}: {metadata[field_name]!r} vs {first_value!r}. "
                    "A single clip/window must carry the same provenance across all frames."
                )
        canonical[field_name] = first_value
    return canonical


def _collect_frames_for_clip(
    shard_result: ShardReadResult,
    clip_id: str,
) -> list[tuple[int, dict[str, object], bytes]]:
    """Pull every (frame_offset, metadata, image_bytes) row for one clip id.

    Sort order is **numeric frame_offset** (not filename) per the
    Issue 005 Step 1 contract, so a hypothetical non-zero-padded
    member name still lands frames in the right slot.
    """
    rows: list[tuple[int, dict[str, object], bytes]] = []
    for member_name, image_bytes in shard_result.image_members.items():
        # ``<safe>_<offset>.image.jpg`` — strip both halves to
        # recover the original clip id, then filter by clip_id.
        meta_member = member_name.removesuffix(".image.jpg") + ".meta.json"
        metadata = shard_result.metadata_members.get(meta_member)
        if metadata is None:
            raise MissingMetadataFieldError(
                f"Image member {member_name!r} has no matching .meta.json "
                f"sidecar in the shard."
            )
        metadata_clip_id = metadata.get("clip_id")
        if metadata_clip_id != clip_id:
            continue
        try:
            frame_offset = int(metadata["frame_offset"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MissingMetadataFieldError(
                f"frame_offset in {meta_member!r} is missing or non-integer."
            ) from exc
        # Pull only the required fields through validation now
        # so we surface missing-field errors at load-time rather
        # than wait until every frame has been processed.
        _check_required_metadata(metadata, source=meta_member)
        rows.append((frame_offset, metadata, image_bytes))

    rows.sort(key=lambda r: r[0])
    return rows


def _validate_contiguous_offsets(frame_offsets: Sequence[int], *, T: int) -> None:
    """Frame offsets must be 0..T-1 with no gaps and no duplicates.

    Caller is responsible for de-duplication — this function
    fires :class:`DuplicateFrameOffsetError` only when the
    caller passed an already-deduped sequence. The full
    double-detection happens at the call site
    (``seen_offsets`` set in :func:`load_clip_tensor_from_shards`)
    so the duplicate error surfaces even when the dedupe-up
    sort would also have produced an invalid-length error.
    """
    expected = tuple(range(T))

    # Count mismatch: prioritise "missing offsets" when there
    # are fewer than T entries, "extra offsets" when there are
    # more. The duplicate-test path hits the "extra offsets"
    # case; the missing-offset test hits the "missing offsets"
    # case. Neither path raises the wrong exception type.
    if len(frame_offsets) < T:
        missing = sorted(set(expected) - set(frame_offsets))
        raise MissingFrameOffsetError(
            f"Clip frame offsets are not contiguous 0..{T - 1}; "
            f"missing offsets: {missing}. Expected exactly {T} frames "
            f"but found {len(frame_offsets)}."
        )
    if len(frame_offsets) > T:
        raise InvalidClipLengthError(
            f"Clip has {len(frame_offsets)} frames after sort; "
            f"expected exactly T={T}. Issue 005 Step 1 requires 1:1 "
            f"between frame offsets and slot index — pad / truncate is "
            f"explicitly disallowed."
        )

    # Count is exactly T. Now check the sequence is contiguous.
    if tuple(frame_offsets) != expected:
        # Should not happen given the count check above; if it
        # does, the duplicate-detection pass must have removed
        # an entry but left the count unchanged. Surface the
        # inconsistency as a generic offset error.
        duplicates_seen: list[int] = []
        seen: set[int] = set()
        for offset in frame_offsets:
            if offset in seen:
                duplicates_seen.append(offset)
            seen.add(offset)
        if duplicates_seen:
            raise DuplicateFrameOffsetError(
                f"Clip has duplicate frame offsets: {duplicates_seen}."
            )
        raise MissingFrameOffsetError(
            f"Clip frame offsets are not contiguous 0..{T - 1}; "
            f"got {list(frame_offsets)}."
        )


def _check_shards_exist(shard_paths: Sequence[Path]) -> None:
    """Raise :class:`CropShardsMissingError` with the Issue 003 hint."""
    missing = [path for path in shard_paths if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise CropShardsMissingError(
            f"Crop shard file(s) not found: {joined}. "
            "Run Issue 003 (the cropping pipeline) first to produce "
            "shards, or verify the path passed to load_clip_tensor_from_shards()."
        )


# ---------------------------------------------------------------------------
# Public API — single clip
# ---------------------------------------------------------------------------


def load_clip_tensor_from_shards(
    shard_paths: Iterable[Path | str],
    *,
    clip_key: str,
    T: int = DEFAULT_T,
) -> LoadedClip:
    """Load one clip from one or more Issue 003 crop shards.

    Group frames by ``clip_id`` across all shards supplied (a
    single clip's frames may straddle a shard boundary), validate
    required metadata + frame-offset contiguity + provenance
    consistency across frames, decode JPEGs, build the
    VideoMAE-ready ``(T, 3, H, W)`` float32 tensor.

    Args:
        shard_paths: One or more paths to ``.tar`` shard files.
            All paths that should carry the clip's frames must be
            supplied; the loader does not search a directory tree.
        clip_key: The string ``clip_id`` to load.
        T: Number of frames per clip. Must be in :data:`ALLOWED_T`
            (16 or 32). Defaults to :data:`DEFAULT_T` (16). Matches the
            PRD starter. The loader raises if the clip's frame
            count differs from T — pad / truncate is disallowed.

    Returns:
        A :class:`LoadedClip` carrying the float32 VideoMAE tensor
        and the preserved provenance fields.

    Raises:
        CropShardsMissingError: when any provided shard path does
            not exist. The error message points at Issue 003.
        InvalidClipLengthError: when ``T`` is not in
            :data:`ALLOWED_T`, or when the clip's frame count after
            sorting differs from ``T``.
        MissingMetadataFieldError: when a ``.meta.json`` sidecar
            is missing or any required field is absent.
        DuplicateFrameOffsetError: when two frames in the clip
            share the same ``frame_offset``.
        MissingFrameOffsetError: when the clip's ``frame_offset``s
            are not contiguous 0..T-1.
        InconsistentClipProvenanceError: when frames disagree on
            provenance fields.
    """
    _validate_T(T)

    paths = [Path(p) for p in shard_paths]
    _check_shards_exist(paths)

    # Collect frames for this clip across all shards. A clip's
    # frames may legitimately straddle two shards (the writer does
    # not guarantee shard-internal-only placement), so we scan
    # every shard and merge.
    rows: list[tuple[int, dict[str, object], bytes]] = []
    shards_used: list[str] = []
    for path in paths:
        result = read_shard(path)
        collected = _collect_frames_for_clip(result, clip_key)
        if collected:
            shards_used.append(path.name)
        rows.extend(collected)

    if not rows:
        raise LookupError(
            f"Clip {clip_key!r} not found in any of the supplied shards. "
            f"Searched: {[str(p) for p in paths]}. Verify the clip_id is "
            "correct (case-sensitive) and that the shards contain this clip."
        )

    # Sort by numeric frame_offset.
    rows.sort(key=lambda r: r[0])

    # Detect duplicates before contiguity check so the duplicate
    # error fires even when T happens to be wrong.
    seen_offsets: set[int] = set()
    for offset, _metadata, _bytes in rows:
        if offset in seen_offsets:
            raise DuplicateFrameOffsetError(
                f"Clip {clip_key!r} has duplicate frame_offset {offset}."
            )
        seen_offsets.add(offset)

    frame_offsets = tuple(r[0] for r in rows)
    _validate_contiguous_offsets(frame_offsets, T=T)

    canonical = _build_provenance_strict([r[1] for r in rows])
    canonical_label = _canonical_label(canonical["label"])

    # Decode and normalise.
    jpeg_bytes_list = [r[2] for r in rows]
    frames_uint8 = _decode_jpegs_to_uint8(jpeg_bytes_list, IMAGE_SIZE, IMAGE_SIZE)
    tensor = _normalize_hwc_uint8_to_chw_float32(
        frames_uint8, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    )

    # Provenance roll-up.
    missing_mask = np.asarray(
        [bool(r[1].get("missing_frame", False)) for r in rows],
        dtype=bool,
    )
    coverage_values = [float(r[1].get("coverage", 1.0)) for r in rows]
    coverage = sum(coverage_values) / len(coverage_values)
    missing_frame_count = int(missing_mask.sum())
    absolute_indices = tuple(int(r[1]["frame_index"]) for r in rows)
    track_id_value = int(canonical["track_id"])
    shard_filename = ",".join(sorted(set(shards_used)))

    return LoadedClip(
        clip_id=str(canonical["clip_id"]),
        dataset=str(canonical["dataset"]),
        label=canonical_label,
        track_id=track_id_value,
        source_path=str(canonical["source_path"]),
        frames=frames_uint8,
        tensor=tensor,
        missing_mask=missing_mask,
        coverage=coverage,
        missing_frame_count=missing_frame_count,
        frame_indices=absolute_indices,
        frame_offsets=frame_offsets,
        shard_filename=shard_filename,
    )


# ---------------------------------------------------------------------------
# Public API — batch
# ---------------------------------------------------------------------------


def load_clip_batch_from_shards(
    shard_paths: Iterable[Path | str],
    *,
    clip_keys: Sequence[str],
    T: int = DEFAULT_T,
) -> tuple[np.ndarray, list[LoadedClip]]:
    """Load a batch of clips as ``(B, T, 3, H, W)`` float32 + per-clip provenance.

    The returned array is the stack of :attr:`LoadedClip.tensor`
    across each clip in ``clip_keys`` (in that order).

    Args:
        shard_paths: One or more shard paths to scan. All shards
            that should carry any of the requested clips must be
            supplied.
        clip_keys: Per-clip ``clip_id`` strings to load.
        T: Frames per clip (default 16). Same rules as
            :func:`load_clip_tensor_from_shards`.

    Returns:
        ``(batch_tensor, clips)`` where ``batch_tensor`` has
        shape ``(B, T, 3, H, W)`` float32 and ``clips`` carries
        per-clip provenance.

    Raises:
        ValueError: when ``clip_keys`` is empty.
        Anything raised by :func:`load_clip_tensor_from_shards`.
    """
    if not clip_keys:
        raise ValueError("clip_keys must be non-empty for a batch load.")
    clips = [
        load_clip_tensor_from_shards(shard_paths, clip_key=clip_key, T=T)
        for clip_key in clip_keys
    ]
    batch = np.stack([clip.tensor for clip in clips], axis=0)
    return batch, clips


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def list_clip_keys(shard_path: Path | str) -> tuple[str, ...]:
    """Read the per-shard manifest's ``clip_keys`` list.

    The shard itself has all the clips; this function exposes
    ``_manifest.json: clip_keys`` for callers that want to
    enumerate without scanning every metadata sidecar.
    """
    shard_path = Path(shard_path)
    if not shard_path.is_file():
        raise CropShardsMissingError(
            f"Shard file {shard_path} not found. Run Issue 003 first."
        )
    result = read_shard(shard_path)
    raw = result.manifest.get("clip_keys", ())
    return tuple(str(k) for k in raw)


def read_videomae_constants() -> dict[str, object]:
    """Return the constants the loader is pinned to.

    Useful for callers that want to print the active configuration
    or write it into run-summary JSON.
    """
    return {
        "model_id": VIDEOMAE_PROCESSOR_MODEL_ID,
        "image_size": IMAGE_SIZE,
        "image_mean": list(IMAGENET_MEAN),
        "image_std": list(IMAGENET_STD),
        "default_T": DEFAULT_T,
        "allowed_T": list(ALLOWED_T),
    }


__all__: tuple[str, ...] = (
    "ALLOWED_T",
    "DEFAULT_T",
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NUM_CHANNELS",
    "REQUIRED_METADATA_FIELDS",
    "PRESERVED_PROVENANCE_FIELDS",
    "VIDEOMAE_PROCESSOR_MODEL_ID",
    "CropShardsMissingError",
    "DuplicateFrameOffsetError",
    "InconsistentClipProvenanceError",
    "InvalidClipLengthError",
    "LoadedClip",
    "MissingFrameOffsetError",
    "MissingMetadataFieldError",
    "int_to_label",
    "label_to_int",
    "list_clip_keys",
    "load_clip_batch_from_shards",
    "load_clip_tensor_from_shards",
    "read_videomae_constants",
)
