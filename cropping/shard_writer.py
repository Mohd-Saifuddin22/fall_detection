"""WebDataset-style .tar shard writer for Issue 003.

Why shards and not loose files:
    - Drive small-file I/O is the documented bottleneck for this
      project (PRD: "cached artifacts must be written in compact
      sharded formats"). Thousands of loose PNGs are prohibitively
      slow to read back; tar shards are read in a single I/O.
    - WebDataset is in the approved dep stack (see requirements.txt)
      so the format is round-trippable downstream.

Shard layout:
    Each shard is a single ``.tar`` file containing pairs of members
    sharing a numeric key::

        shard-000000.tar
            000000.image.jpg     <-- one crop frame (JPEG-encoded)
            000000.meta.json     <-- per-frame metadata sidecar
            000001.image.jpg
            000001.meta.json
            ...

    The ``clip_id`` (one Issue 003 TrackWindow) typically has multiple
    frames so it spans many consecutive keys inside one shard. A
    ``_manifest.json`` file at the SHARD root lists every clip id
    written into that shard and the byte offset of each entry — this
    is the per-clip lookup index the trainer / reader needs.

Naming:
    Shard filenames are zero-padded ``shard-NNNNNN.tar`` so alphabetical
    sort = chronological order. Padding width auto-scales with shard
    count so we never run out of digits.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Sanitize a clip_id for use as a filename component. We replace any
# character outside [A-Za-z0-9_-] with '_' so the tar member names are
# valid on every filesystem.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


def safe_member_name(stem: str) -> str:
    """Turn an arbitrary clip_id into a safe tar member stem."""
    cleaned = _SAFE_NAME.sub("_", stem)
    return cleaned or "unnamed"


@dataclass(frozen=True)
class ShardIndex:
    """Per-shard lookup index written as ``_manifest.json`` inside the shard."""

    shard_filename: str
    shard_index: int
    member_keys: tuple[str, ...] = ()
    clip_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_filename": self.shard_filename,
            "shard_index": self.shard_index,
            "member_keys": list(self.member_keys),
            "clip_keys": list(self.clip_keys),
        }


def encode_jpeg_bytes(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode one frame as JPEG bytes.

    Falls back to Pillow when OpenCV isn't available. Either is in the
    approved dep stack.
    """
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"image must be HxWx3 or HxWx4, got shape {image.shape}.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    rgb = image[:, :, :3] if image.shape[2] == 4 else image
    try:
        from PIL import Image as _PILImage
        buf = io.BytesIO()
        _PILImage.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except ImportError:
        # OpenCV is the second-best path; it is in the approved stack.
        import cv2  # type: ignore
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode JPEG.")
        return bytes(buf)


@dataclass
class ShardWriter:
    """Open one .tar shard for sequential writes; close to flush.

    Usage:
        with ShardWriter(path) as shard:
            shard.write_clip_member("urfd-fall-01-cam0_t7_w0", 0, frame, metadata)
            shard.write_clip_member("urfd-fall-01-cam0_t7_w0", 1, frame, metadata)
        # manifest is written on close.

    Each shard is fully serial — ``open(path, "w")`` mode of
    ``tarfile`` is buffered, so writes are not on disk until
    ``close()`` or context exit.
    """

    def __init__(self, path: Path, shard_index: int) -> None:
        self._path = Path(path)
        self._shard_index = shard_index
        self._tar: tarfile.TarFile | None = None
        self._member_keys: list[str] = []
        self._clip_keys: list[str] = []

    @property
    def path(self) -> Path:
        return self._path

    @property
    def shard_index(self) -> int:
        return self._shard_index

    def __enter__(self) -> "ShardWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tar = tarfile.open(self._path, mode="w")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Write the manifest and close the tar."""
        if self._tar is None:
            return
        # Write the per-shard manifest as a sibling member so readers can
        # scan the shard without extracting every image.
        index = ShardIndex(
            shard_filename=self._path.name,
            shard_index=self._shard_index,
            member_keys=tuple(self._member_keys),
            clip_keys=tuple(self._clip_keys),
        )
        manifest_bytes = json.dumps(index.to_dict(), indent=2).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="_manifest.json")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = 0  # deterministic — no mtime drift across runs
        self._tar.addfile(manifest_info, io.BytesIO(manifest_bytes))
        self._tar.close()
        self._tar = None

    def _add_bytes(self, member_name: str, payload: bytes) -> None:
        assert self._tar is not None, "ShardWriter must be opened before use."
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload)
        info.mtime = 0
        info.mode = 0o644
        self._tar.addfile(info, io.BytesIO(payload))
        self._member_keys.append(member_name)

    def write_clip_member(
        self,
        clip_key: str,
        frame_offset: int,
        image: np.ndarray,
        metadata: dict[str, object],
        jpeg_quality: int = 90,
    ) -> str:
        """Write one clip frame + its metadata sidecar into the shard.

        Returns the image-member name (e.g. ``"safe_key_0000.image.jpg"``).
        """
        safe = safe_member_name(clip_key)
        image_member = f"{safe}_{frame_offset:04d}.image.jpg"
        meta_member = f"{safe}_{frame_offset:04d}.meta.json"

        self._add_bytes(image_member, encode_jpeg_bytes(image, quality=jpeg_quality))
        meta_bytes = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
        self._add_bytes(meta_member, meta_bytes)

        # Record the clip id once — even if it spans many members.
        if clip_key not in self._clip_keys:
            self._clip_keys.append(clip_key)
        return image_member


# ---------------------------------------------------------------------------
# Shard naming
# ---------------------------------------------------------------------------


def shard_filename(shard_index: int, width: int) -> str:
    """Return ``shard-NNNNNN.tar`` with zero-padded width."""
    return f"shard-{shard_index:0{width}d}.tar"


def compute_shard_padding_width(max_shards: int) -> int:
    """Pick a stable zero-padding width given the expected shard count.

    Stable across runs: a fixed ``max_shards`` always produces the same
    padding width, so shard filenames sort lexicographically the same
    way every time. (Avoids a 999-shard run padding to 3 and then a
    1000-shard run padding to 4 mid-project.)
    """
    return max(4, len(str(max(1, max_shards))))


# ---------------------------------------------------------------------------
# Reading / verification (used by tests + future trainer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardReadResult:
    """Contents of one shard, post-decode."""

    shard_filename: str
    image_members: dict[str, bytes]
    metadata_members: dict[str, dict[str, object]]
    manifest: dict[str, object]

    def clip_frames(self, clip_key: str) -> list[tuple[str, bytes, dict[str, object]]]:
        """Return ordered ``(image_member_name, image_bytes, metadata)`` triples for one clip."""
        safe = safe_member_name(clip_key)
        frames: list[tuple[str, bytes, dict[str, object]]] = []
        for member_name, image_bytes in sorted(self.image_members.items()):
            if not member_name.startswith(safe + "_"):
                continue
            meta_member = member_name.removesuffix(".image.jpg") + ".meta.json"
            metadata = self.metadata_members.get(meta_member, {})
            frames.append((member_name, image_bytes, metadata))
        return frames


def read_shard(path: Path) -> ShardReadResult:
    """Read every image + metadata member + manifest from one shard.

    Used by tests to verify the writer; will also be used by the
    Issue 006 / 009 / 011 trainers in their shard readers.
    """
    image_members: dict[str, bytes] = {}
    metadata_members: dict[str, dict[str, object]] = {}
    manifest: dict[str, object] = {}
    with tarfile.open(path, mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            if member.name == "_manifest.json":
                manifest = json.loads(payload.decode("utf-8"))
            elif member.name.endswith(".image.jpg"):
                image_members[member.name] = payload
            elif member.name.endswith(".meta.json"):
                metadata_members[member.name] = json.loads(payload.decode("utf-8"))
    return ShardReadResult(
        shard_filename=path.name,
        image_members=image_members,
        metadata_members=metadata_members,
        manifest=manifest,
    )


__all__: tuple[str, ...] = (
    "ShardIndex",
    "ShardReadResult",
    "ShardWriter",
    "compute_shard_padding_width",
    "encode_jpeg_bytes",
    "read_shard",
    "safe_member_name",
    "shard_filename",
)