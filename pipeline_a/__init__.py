"""Pipeline A — VideoMAE data-prep (Issue 005 Step 1).

Public surface:

    :mod:`pipeline_a.load_crops` — Issue 003 crop-shard reader that
        builds VideoMAE-ready ``(T, 3, H, W)`` float32 tensors.

Issue 005 Step 1 ships the data-prep layer only:

- It does NOT load a VideoMAE model. (Issue 009.)
- It does NOT compute metrics. (Issue 007 / 010 / 013 / 014 / 015.)
- It does NOT import or call YOLO / ByteTrack / perception / crop
  runner. Issue 003 already wrote the shards on disk; this
  module only re-reads them.
"""

from pipeline_a.load_crops import (
    ALLOWED_T,
    DEFAULT_T,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CHANNELS,
    PRESERVED_PROVENANCE_FIELDS,
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

__all__: tuple[str, ...] = (
    "ALLOWED_T",
    "DEFAULT_T",
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NUM_CHANNELS",
    "PRESERVED_PROVENANCE_FIELDS",
    "REQUIRED_METADATA_FIELDS",
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
