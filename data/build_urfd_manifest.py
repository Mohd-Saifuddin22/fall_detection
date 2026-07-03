"""Build the URFD debug manifest from a staged Drive folder tree.

Walks ``MyDrive/fall_detection/datasets/urfd/`` and emits a
schema-1.1 manifest whose every clip is a folder of ordered PNG frames.

The folder-name convention (from ``data/stage_urfd.py:parse_urfd_folder_name``):

    ``fall-NN-camM``  →  role=debug, label=fall
    ``adl-NN-camM``   →  role=debug, label=no_fall

Clip IDs are stable: ``urfd-debug-<sequence>-<camera>`` so re-running
this script over the same tree produces the same manifest.

Validates the result before writing so a typo in the folder tree can
never silently produce a broken manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from data.manifests import (
    ClipRecord,
    ClipRole,
    FallLabel,
    Manifest,
    load_manifest,
    validate_manifest,
)
from data.stage_urfd import (
    DATASET_SUBDIR_NAME,
    StagedClipFolder,
    _enumerate_staged_clips,
)


MANIFEST_RELATIVE_PATH: str = "datasets/urfd/manifest.yaml"
MANIFEST_SCHEMA_VERSION: str = "1.1"


def build_clip_id(folder: StagedClipFolder) -> str:
    """Stable clip ID for one URFD folder.

    ``fall-01-cam0`` → ``urfd-debug-fall-01-cam0``
    ``adl-02-cam1``  → ``urfd-debug-adl-02-cam1``
    """
    return f"urfd-debug-{folder.folder_name.lower()}"


def build_clip_record(folder: StagedClipFolder) -> ClipRecord:
    """Construct one :class:`ClipRecord` from one staged folder."""
    notes_parts: list[str] = []
    if folder.clip_sequence:
        notes_parts.append(f"sequence={folder.clip_sequence}")
    if folder.camera:
        notes_parts.append(f"camera={folder.camera}")
    notes_parts.append("frame-folder (PNGs)")
    notes = "; ".join(notes_parts)

    label = FallLabel.FALL if folder.label == "fall" else FallLabel.NO_FALL

    return ClipRecord(
        clip_id=build_clip_id(folder),
        dataset=DATASET_SUBDIR_NAME,
        role=ClipRole.DEBUG,
        label=label,
        source_path=folder.drive_relative_path,
        notes=notes,
    )


def build_urfd_manifest(staged_root: Path) -> Manifest:
    """Walk the staged tree and return a fully-populated :class:`Manifest`.

    Validates the manifest in-memory before returning so callers can
    trust the result; raising :class:`ValueError` from this function
    means "the staged tree is malformed" — fix the tree, not the manifest.
    """
    staged_root = Path(staged_root)
    clip_folders = _enumerate_staged_clips(staged_root)
    if not clip_folders:
        raise ValueError(
            f"No URFD-shaped clip folders found under {staged_root}. "
            f"Expected folders matching 'fall-NN-camM' or 'adl-NN-camM'."
        )

    clips: list[ClipRecord] = [build_clip_record(folder) for folder in clip_folders]
    manifest = Manifest(schema_version=MANIFEST_SCHEMA_VERSION, clips=clips)
    report = validate_manifest(manifest)
    if not report.is_valid:
        raise ValueError(
            "URFD manifest failed validation. Fix the staged tree, not the manifest.\n"
            + "\n".join(report.errors)
        )
    return manifest


def write_urfd_manifest(manifest: Manifest, destination: Path) -> Path:
    """Serialise the manifest to YAML at ``destination``."""
    import yaml  # local import — PyYAML is in the approved stack

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(manifest.to_serialisable(), sort_keys=False),
        encoding="utf-8",
    )
    return destination


def reload_manifest(path: Path) -> Manifest:
    """Convenience: load a previously written URFD manifest from disk."""
    return load_manifest(path)


def expected_clip_paths(staged_root: Path) -> Iterable[Path]:
    """Yield every Drive-relative path the manifest will reference.

    Useful for sanity checks: "do these folders actually exist on Drive?"
    """
    for folder in _enumerate_staged_clips(staged_root):
        yield Path("datasets") / DATASET_SUBDIR_NAME / folder.folder_name


__all__: tuple[str, ...] = (
    "MANIFEST_RELATIVE_PATH",
    "MANIFEST_SCHEMA_VERSION",
    "build_clip_id",
    "build_clip_record",
    "build_urfd_manifest",
    "write_urfd_manifest",
    "reload_manifest",
    "expected_clip_paths",
)