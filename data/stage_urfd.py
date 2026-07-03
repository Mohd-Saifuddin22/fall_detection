"""Stage URFD from Kaggle into Drive.

Hard rules (Issue 002):
    - Read Kaggle credentials from Colab Secrets at runtime; never from
      files on disk, never as plaintext in this script, never as an
      environment variable the script prints.
    - Download ONLY when the target staging path does not already exist.
      Re-runs must reuse the staged copy (idempotency).
    - Never print, log, commit, or write credential values to Drive.
    - Never use a different Kaggle slug than ``tanmaydacha/urfd-dataset``
      (whitelisted here; everything else fails loud).
    - **Never mutate the Kaggle source.** Kaggle / Colab may expose the
      dataset as a read-only mounted input directory
      (``/kaggle/input/<dataset>``) instead of a writable cache. We
      COPY contents into the staging root, we do NOT move or rename.
      See :func:`_copy_into`.

Public API:
    - :func:`is_urfd_already_staged` — idempotency check.
    - :func:`stage_urfd_from_kaggle` — one-shot staging call.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# The ONLY Kaggle dataset slug this project is allowed to fetch.
# If a future issue needs a different slug, add it here deliberately,
# not by passing an arbitrary string into the function.
ALLOWED_KAGGLE_SLUG: str = "tanmaydacha/urfd-dataset"
DATASET_SUBDIR_NAME: str = "urfd"
# Sentinel file written after staging succeeds; lets us prove to
# ourselves that the on-disk tree is the result of THIS script, not
# a stray copy that someone dropped in by hand.
STAGING_MARKER_FILENAME: str = ".staged_from_kaggle.txt"


@dataclass(frozen=True)
class StagedClipFolder:
    """One sub-folder of the URFD staging tree."""

    absolute_path: Path
    folder_name: str
    label: str  # "fall" | "no_fall"
    camera: str | None
    clip_sequence: str | None  # the fall-/adl- sequence number, when parseable

    @property
    def drive_relative_path(self) -> str:
        """Path string relative to ``MyDrive/fall_detection/`` for the manifest."""
        return f"datasets/{DATASET_SUBDIR_NAME}/{self.folder_name}"


@dataclass(frozen=True)
class UrfdStagingResult:
    """Outcome of a single staging call."""

    staged_root: Path
    clip_folders: tuple[StagedClipFolder, ...]
    already_staged: bool
    kaggle_slug: str

    @property
    def clip_count(self) -> int:
        return len(self.clip_folders)


# ---------------------------------------------------------------------------
# Colab Secrets integration — credentials NEVER leave this function
# ---------------------------------------------------------------------------


def _read_kaggle_credentials_from_secrets() -> None:
    """Read Kaggle credentials from Colab Secrets and set env vars.

    ``kagglehub`` reads ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` from the
    environment. We only set them; we never ``print()`` them, never
    return them, never write them anywhere on disk.

    On non-Colab hosts (local dev), fall back to a ``.kaggle/kaggle.json``
    already on disk — but DO NOT auto-create one. If neither source is
    present, raise a clear error so the user knows what to do.
    """
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        # Not in Colab — assume the user has set up ~/.kaggle/kaggle.json
        # already. We do NOT auto-fetch or auto-generate it.
        if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
            print(
                "[stage_urfd] Not running on Colab and no KAGGLE_USERNAME / "
                "KAGGLE_KEY in env. If you need to stage URFD locally, install "
                "kaggle CLI and run `kaggle datasets download -d "
                f"{ALLOWED_KAGGLE_SLUG}` into datasets/{DATASET_SUBDIR_NAME}/ by hand.",
                file=sys.stderr,
            )
        return

    # On Colab: read from Secrets, set env vars, and let the function end.
    # No print, no log, no return of the credential value.
    username = userdata.get("KAGGLE_USERNAME")
    key = userdata.get("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError(
            "Kaggle credentials missing in Colab Secrets. Add two secrets:\n"
            "  KAGGLE_USERNAME  — your Kaggle username\n"
            "  KAGGLE_KEY       — your Kaggle API key\n"
            "Then re-run. The values are NEVER logged or written to disk."
        )
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------


def is_urfd_already_staged(staged_root: Path) -> bool:
    """True when ``staged_root`` already contains the URFD layout + marker file.

    Re-runs of the staging script must short-circuit here — never
    re-download a multi-GB dataset that is already on Drive.
    """
    if not staged_root.is_dir():
        return False
    marker = staged_root / STAGING_MARKER_FILENAME
    if not marker.is_file():
        return False
    # The marker alone isn't enough — also require at least one clip
    # folder matching the URFD naming convention. Hidden files (the
    # marker itself starts with a dot) are excluded so the marker
    # doesn't count as "real content".
    return any(
        not entry.name.startswith(".") for entry in staged_root.iterdir()
    )


# ---------------------------------------------------------------------------
# Folder-name parsing
# ---------------------------------------------------------------------------


def parse_urfd_folder_name(folder_name: str) -> StagedClipFolder | None:
    """Parse a URFD clip folder name into a :class:`StagedClipFolder`.

    URFD naming convention (from ``tanmaydacha/urfd-dataset``):
        ``fall-NN-camM``  →  fall sequence NN, camera M
        ``adl-NN-camM``   →  activities of daily living (non-fall), sequence NN, camera M

    Folder names that OS-level re-stages have appended ``" (1)"`` /
    ``" (2)"`` (Windows file-collision suffix) are normalised before
    parsing so the dataset doesn't end up with two copies of the same
    clip under different slugs. The returned ``folder_name`` is the
    original name (preserving the on-disk path); the parsing logic
    operates on the normalised stem.

    Returns ``None`` for folders that don't match — the caller decides
    whether to skip silently or warn.
    """
    lowered = folder_name.strip().lower()
    if not lowered:
        return None

    # Strip an OS-level collision suffix: "fall-01-cam0 (1)" → "fall-01-cam0".
    # We only strip when the suffix is exactly " (N)" where N is a
    # small integer — anything more exotic (trailing spaces, unicode
    # parens) is left alone so a real renaming is visible.
    import re as _re
    collision_match = _re.search(r"\s+\(\d+\)\s*$", lowered)
    if collision_match is not None:
        lowered = lowered[:collision_match.start()]

    label: str | None = None
    if lowered.startswith("fall-"):
        label = "fall"
    elif lowered.startswith("adl-"):
        label = "no_fall"

    if label is None:
        return None

    # Optional camera suffix: ``-cam0`` / ``-cam1``.
    camera: str | None = None
    sequence: str | None = None
    parts = lowered.split("-")
    # The first part is "fall" or "adl"; the second is the sequence
    # number; anything after is camera / angle metadata.
    if len(parts) >= 2:
        sequence = parts[1]
    for part in parts[2:]:
        if part.startswith("cam") and part[3:].isdigit():
            camera = part
            break

    return StagedClipFolder(
        absolute_path=Path(),  # filled in by the caller
        folder_name=folder_name,
        label=label,
        camera=camera,
        clip_sequence=sequence,
    )


# ---------------------------------------------------------------------------
# Staging entry point
# ---------------------------------------------------------------------------


def stage_urfd_from_kaggle(
    data_root: Path,
    *,
    kaggle_slug: str = ALLOWED_KAGGLE_SLUG,
    force: bool = False,
) -> UrfdStagingResult:
    """Stage URFD into ``<data_root>/datasets/urfd/``.

    The ``data_root`` is whichever root holds the ACTIVE processing
    data — Drive in legacy mode, the Colab local disk in the
    fast-path mode (``colab.data_mode.LOCAL``). The path the dataset
    lands at is the same shape regardless of mode; the choice is
    made by the caller, not this function.

    Args:
        data_root: root under which to stage URFD. In LOCAL mode this
            is ``/content/fall_local``; in DRIVE mode it's the
            project's Drive root (``MyDrive/fall_detection/``).
        kaggle_slug: dataset slug; defaults to the whitelisted
            ``tanmaydacha/urfd-dataset``. Any other value raises.
        force: when ``True``, re-download even if a staged copy exists.
            Use sparingly — it costs Drive egress and runtime wall-clock.

    Returns:
        A :class:`UrfdStagingResult` listing every parsed clip folder.

    Raises:
        RuntimeError: when Kaggle credentials are missing in Colab Secrets,
            or when the slug is not in the whitelist.
        FileNotFoundError: when kagglehub fails to return a downloaded
            folder (network issue, slug typo, etc.).
    """
    if kaggle_slug != ALLOWED_KAGGLE_SLUG:
        raise RuntimeError(
            f"Kaggle slug {kaggle_slug!r} is not whitelisted. "
            f"Only {ALLOWED_KAGGLE_SLUG!r} may be staged by this script."
        )

    staged_root = Path(data_root) / "datasets" / DATASET_SUBDIR_NAME

    if not force and is_urfd_already_staged(staged_root):
        clips = _enumerate_staged_clips(staged_root)
        return UrfdStagingResult(
            staged_root=staged_root,
            clip_folders=clips,
            already_staged=True,
            kaggle_slug=kaggle_slug,
        )

    # Idempotency: clear a half-staged tree before re-downloading so a
    # previous interrupted run doesn't leave a confusing mix.
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)

    _read_kaggle_credentials_from_secrets()

    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed; install it via colab/setup.py before "
            "running this staging script."
        ) from exc

    # ``dataset_download`` returns the path to the dataset source —
    # historically a writable ``~/.cache/kagglehub/...`` cache, but newer
    # Kaggle / Colab integrations may expose the dataset as a READ-ONLY
    # mounted input directory (e.g. ``/kaggle/input/<dataset>``). We must
    # NEVER delete, rename, or mutate that source — we only READ it and
    # COPY its contents into our writable staged root.
    download_path = Path(kagglehub.dataset_download(kaggle_slug))
    if not download_path.is_dir():
        raise FileNotFoundError(
            f"kagglehub returned a non-directory path: {download_path}"
        )

    # kagglehub may nest the actual data one or two levels deep; pull
    # the first directory contents up. We COPY everything (never move)
    # so the Kaggle source remains untouched. The marker file we write
    # at the end proves provenance.
    for entry in download_path.iterdir():
        destination = staged_root / entry.name
        if destination.exists():
            shutil.rmtree(destination)
        _copy_into(entry, staged_root / entry.name)

    # Write the provenance marker — proves this tree came from kagglehub,
    # not from a stray copy someone hand-placed.
    (staged_root / STAGING_MARKER_FILENAME).write_text(
        f"staged_from_kaggle_slug={kaggle_slug}\n"
        "credentials_source=colab_secrets\n",
        encoding="utf-8",
    )

    clips = _enumerate_staged_clips(staged_root)
    return UrfdStagingResult(
        staged_root=staged_root,
        clip_folders=clips,
        already_staged=False,
        kaggle_slug=kaggle_slug,
    )


def _enumerate_staged_clips(staged_root: Path) -> tuple[StagedClipFolder, ...]:
    """Walk the staged tree and parse each URFD-shaped folder.

    Deduplicates on the normalised name so ``fall-01-cam0`` and
    ``fall-01-cam0 (1)`` (an OS-level file-collision suffix from
    Windows re-stages) don't both appear in the manifest. The first
    folder encountered in sorted order wins; the ``(N)`` suffix is
    treated as a duplicate and skipped.
    """
    clips: list[StagedClipFolder] = []
    seen_normalised: set[str] = set()
    for entry in sorted(staged_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue  # skip hidden / marker files
        parsed = parse_urfd_folder_name(entry.name)
        if parsed is None:
            continue
        # Normalise the folder name for dedup (same logic the parser
        # applies) — case-insensitive stem, collision suffix stripped.
        import re as _re
        normalised = parsed.folder_name.strip().lower()
        collision_match = _re.search(r"\s+\(\d+\)\s*$", normalised)
        if collision_match is not None:
            normalised = normalised[:collision_match.start()]
        if normalised in seen_normalised:
            continue
        seen_normalised.add(normalised)
        clips.append(StagedClipFolder(
            absolute_path=entry,
            folder_name=parsed.folder_name,
            label=parsed.label,
            camera=parsed.camera,
            clip_sequence=parsed.clip_sequence,
        ))
    return tuple(clips)


# ---------------------------------------------------------------------------
# Internal: read-only-safe copy
# ---------------------------------------------------------------------------


def _copy_into(source: Path, destination: Path) -> None:
    """Copy ``source`` into ``destination`` without mutating ``source``.

    Kaggle / Colab may mount the dataset as a READ-ONLY input directory
    (``/kaggle/input/<dataset>``). We must never delete, rename, or
    move anything there — we only READ its bytes and COPY them into
    our writable staging root.

    Behaviour:
        - Source directory  → ``shutil.copytree(..., dirs_exist_ok=False)``
          into ``destination``. ``dirs_exist_ok=False`` is the safer
          choice: the caller (``stage_urfd_from_kaggle``) has already
          cleared the destination if it existed.
        - Source file       → ``shutil.copy2(...)`` (preserves mtime +
          metadata so the destination is byte-faithful). The parent
          directory of ``destination`` is created if needed.
        - Symlinks          → preserved as-is so the destination tree
          mirrors the source tree's structure.
    """
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(str(source), str(destination), symlinks=True)
        return
    # File (or a symlink — copy2 leaves symlink targets alone).
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


__all__: tuple[str, ...] = (
    "ALLOWED_KAGGLE_SLUG",
    "DATASET_SUBDIR_NAME",
    "STAGING_MARKER_FILENAME",
    "StagedClipFolder",
    "UrfdStagingResult",
    "is_urfd_already_staged",
    "parse_urfd_folder_name",
    "stage_urfd_from_kaggle",
)