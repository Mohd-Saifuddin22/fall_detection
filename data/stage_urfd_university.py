"""Stage URFD from the university repository (Issue 005+).

The default URFD source moved away from the Kaggle mirror. The
authoritative source is the University of Rzeszów's URFD page:

    https://fenix.ur.edu.pl/~mkepski/ds/data/

This module downloads every cam0 RGB zip (fall-01..fall-30 +
adl-01..adl-40) and both label CSVs (urfall-cam0-falls.csv,
urfall-cam0-adls.csv) into a local staging tree that matches
the layout the rest of the pipeline (Issue 003 manifest,
perception runner) already consumes.

Real university zip shape (verified):
    - The downloaded archive extracts to
      ``<staged_root>/<fall|adl>-NN-cam0-rgb/<fall|adl>-NN-cam0-rgb/*.png``
      — a double-nested layout. The existing
      :class:`perception.frames.FrameFolderReader` already detects
      the inner matching subfolder and descends into it; we
      preserve the ``-rgb`` suffix on the folder name so the
      manifest clip id (``urfd-debug-fall-NN-cam0-rgb``) matches
      the on-disk folder verbatim.
    - Each fall/adl zip carries 160 frames.
    - Frame files are 1-based and 3-digit zero-padded
      (``fall-01-cam0-rgb-001.png`` … ``fall-01-cam0-rgb-160.png``).
    - The full 70-clip download is roughly 4 GB.

Hard rules (Issue 005+ university source):
    - Whitelist the ONLY base URL allowed. Any other host /
      scheme / path-prefix fails loud BEFORE the network is
      touched.
    - Never re-download a complete staged tree unless ``force=True``
      is set. The marker file ``.staged_from_university.txt``
      + the presence of every expected file is the idempotency key.
    - Every downloaded zip is verified as a valid zip archive
      before extraction. A truncated or corrupt zip fails loud —
      a partial clip is worse than a missing clip because the
      manifest would still list it.
    - CSVs are copied to a persistent path (``staged_root /
    ``csvs``) and survive ``Run All`` re-runs on Colab. They
      carry the same provenance as the frame zips.
    - Existing manifest / frame-reader pipeline must keep working
      unchanged. The folder naming convention
      (``fall-NN-cam0`` / ``adl-NN-cam0``) is preserved.

Public API:
    - :func:`build_frame_zip_urls` — enumerate the 70 zip URLs.
    - :func:`build_csv_urls` — enumerate the 2 CSV URLs.
    - :func:`is_urfd_university_already_staged` — idempotency check.
    - :func:`stage_urfd_from_university` — one-shot staging call.
"""

from __future__ import annotations

import io
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------


#: The ONLY base URL this project is allowed to fetch URFD from.
#: Any other base URL, host, or scheme must fail loud BEFORE the
#: network is touched. Adding a new authoritative source means
#: adding it here deliberately, not by passing an arbitrary
#: string into the function.
ALLOWED_UNIVERSITY_BASE_URL: str = "https://fenix.ur.edu.pl/~mkepski/ds/data/"

#: Sentinel file written after a successful staging run; lets us
#: prove the on-disk tree is the result of THIS script, not a
#: stray hand-placed copy.
STAGING_MARKER_FILENAME: str = ".staged_from_university.txt"

#: URFD sequences the university page publishes.
FALL_SEQUENCES: tuple[int, ...] = tuple(range(1, 31))   # 1..30 inclusive
ADL_SEQUENCES: tuple[int, ...] = tuple(range(1, 41))    # 1..40 inclusive

#: Camera-0 suffix appended to every frame-zip URL.
CAMERA_SUFFIX: str = "cam0-rgb"

#: Label CSV filenames (cam-0 only — Issue 002 cam1 = hard slice).
FALL_CSV_FILENAME: str = "urfall-cam0-falls.csv"
ADL_CSV_FILENAME: str = "urfall-cam0-adls.csv"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StagedClipFolder:
    """One sub-folder of the university-staged URFD tree.

    Mirrors :class:`data.stage_urfd.StagedClipFolder` so the existing
    manifest / frame-reader pipeline can enumerate the staged
    tree without knowing which staging path was used.
    """

    absolute_path: Path
    folder_name: str
    label: str  # "fall" | "no_fall"
    camera: str | None
    clip_sequence: str | None  # the fall-/adl- sequence number

    @property
    def drive_relative_path(self) -> str:
        return f"datasets/urfd/{self.folder_name}"


@dataclass(frozen=True)
class UrfdUniversityStagingResult:
    """Outcome of a single university-staging call.

    The structured shape mirrors :class:`UrfdStagingResult` so a
    caller that already consumes the Kaggle mirror's result
    can swap implementations without changing its bookkeeping
    code.
    """

    staged_root: Path
    clip_folders: tuple[StagedClipFolder, ...]
    csv_paths: dict[str, Path]  # filename -> absolute path
    succeeded_clips: tuple[str, ...]
    failed_clips: dict[str, str]  # clip name -> error message
    already_staged: bool
    source_base_url: str

    @property
    def clip_count(self) -> int:
        return len(self.clip_folders)


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def build_frame_zip_urls(base_url: str = ALLOWED_UNIVERSITY_BASE_URL) -> tuple[str, ...]:
    """Return the 70 frame-zip URLs in deterministic order.

    Each URL is ``<base_url>/<seq>-cam0-rgb.zip``. Camera 0 only;
    the Issue 002 cam1 = hard slice decision is preserved.
    """
    _verify_base_url(base_url)
    urls: list[str] = []
    for seq in FALL_SEQUENCES:
        urls.append(f"{base_url}fall-{seq:02d}-{CAMERA_SUFFIX}.zip")
    for seq in ADL_SEQUENCES:
        urls.append(f"{base_url}adl-{seq:02d}-{CAMERA_SUFFIX}.zip")
    return tuple(urls)


def build_csv_urls(base_url: str = ALLOWED_UNIVERSITY_BASE_URL) -> dict[str, str]:
    """Return the two label-CSV URLs keyed by filename.

    Keys match the on-disk filename the script persists; values
    are the full remote URL.
    """
    _verify_base_url(base_url)
    return {
        FALL_CSV_FILENAME: f"{base_url}{FALL_CSV_FILENAME}",
        ADL_CSV_FILENAME: f"{base_url}{ADL_CSV_FILENAME}",
    }


def _verify_base_url(base_url: str) -> None:
    """Fail loud when ``base_url`` is not in the whitelist.

    The check is a strict equality (no scheme-rewriting, no path
    suffix, no trailing-slash tolerance) — any drift from the
    pinned base URL is a code-review failure, not a runtime
    convenience.
    """
    if base_url != ALLOWED_UNIVERSITY_BASE_URL:
        raise RuntimeError(
            f"Base URL {base_url!r} is not whitelisted. "
            f"Only {ALLOWED_UNIVERSITY_BASE_URL!r} may be staged "
            "by this script. Add the new source to "
            "ALLOWED_UNIVERSITY_BASE_URL deliberately, not by "
            "passing an arbitrary string into the function."
        )
    parsed = urlparse(base_url)
    if parsed.scheme != "https":
        raise RuntimeError(
            f"Base URL {base_url!r} is not https. The whitelist "
            "is scheme-strict to keep a future typo from silently "
            "downloading from a different host."
        )


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def expected_files(staged_root: Path) -> tuple[Path, ...]:
    """Every file the marker requires to be present for "fully staged".

    The marker alone isn't enough — the marker is one marker, but
    a half-staged tree could carry it. We require every file the
    script intends to write:
        - 30 fall clip folders (``fall-NN-cam0-rgb``)
        - 40 adl clip folders (``adl-NN-cam0-rgb``)
        - 2 CSV files
        - the marker

    The ``-rgb`` suffix is the real university layout: the
    downloaded zip extracts to
    ``<root>/fall-NN-cam0-rgb/fall-NN-cam0-rgb/*.png`` and the
    existing :class:`perception.frames.FrameFolderReader` already
    descends into that nested shape. Stripping ``-rgb`` would
    diverge from the on-disk folder name and break the manifest
    builder's clip-id contract (``urfd-debug-fall-NN-cam0-rgb``).
    """
    files: list[Path] = [staged_root / STAGING_MARKER_FILENAME]
    files.append(staged_root / "csvs" / FALL_CSV_FILENAME)
    files.append(staged_root / "csvs" / ADL_CSV_FILENAME)
    for seq in FALL_SEQUENCES:
        files.append(staged_root / f"fall-{seq:02d}-cam0-rgb")
    for seq in ADL_SEQUENCES:
        files.append(staged_root / f"adl-{seq:02d}-cam0-rgb")
    return tuple(files)


def is_urfd_university_already_staged(staged_root: Path) -> bool:
    """True when the marker + every expected file is present.

    Re-runs of the staging script must short-circuit here — never
    re-download a multi-GB dataset that is already on disk.
    """
    if not staged_root.is_dir():
        return False
    for path in expected_files(staged_root):
        if not path.exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Folder-name parsing (kept identical to the Kaggle stager's shape)
# ---------------------------------------------------------------------------


def parse_university_folder_name(folder_name: str) -> StagedClipFolder | None:
    """Parse a URFD clip folder name into a :class:`StagedClipFolder`.

    Same convention as the Kaggle mirror's parser, plus the
    university source's ``-rgb`` suffix:
        ``fall-NN-camM``  →  fall sequence NN, camera M
        ``fall-NN-camM-rgb``  →  same — the ``-rgb`` suffix is
            kept by the university stager and flows through to
            the manifest clip id.
        ``adl-NN-camM``   →  activities of daily living (non-fall), sequence NN, camera M
        ``adl-NN-camM-rgb``  →  same, university source shape.

    Folder names with OS-level re-stages like ``"fall-01-cam0-rgb (1)"``
    are normalised so the manifest doesn't end up with two
    copies of the same clip under different slugs.
    """
    import re as _re
    lowered = folder_name.strip().lower()
    if not lowered:
        return None
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

    camera: str | None = None
    sequence: str | None = None
    parts = lowered.split("-")
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
# Network helpers
# ---------------------------------------------------------------------------


def _download_with_retry(
    url: str,
    *,
    max_attempts: int = 3,
    timeout_seconds: float = 60.0,
) -> bytes:
    """Download ``url`` with up to ``max_attempts`` linear retries.

    Fail loud on every permanent failure — a truncated download
    is not silently treated as a 0-byte payload.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = Request(url, headers={"User-Agent": "fall-detection-urfd-stager/1.0"})
            with urlopen(req, timeout=timeout_seconds) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts:
                continue
    raise RuntimeError(
        f"Download failed after {max_attempts} attempts: {url} "
        f"(last error: {type(last_exc).__name__}: {last_exc})"
    )


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def _verify_zip_bytes(payload: bytes, source: str) -> None:
    """Fail loud on a truncated or corrupt zip.

    The :class:`zipfile.ZipFile` constructor itself checks the
    central directory + End-of-Central-Directory record; a
    truncated download fails here, not later at extract time.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(
                    f"Corrupt zip from {source}: bad entry {bad!r}"
                )
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Truncated or corrupt zip from {source}: {exc}"
        ) from exc


def _safe_extract_zip(
    payload: bytes, destination: Path, *, source: str
) -> None:
    """Extract ``payload`` into ``destination`` after a validity check.

    :class:`zipfile.ZipFile.extractall` is the standard path; we
    call :func:`_verify_zip_bytes` first so a bad zip raises
    BEFORE the destination is touched. A partial extract followed
    by a failure would leave the staged tree in a state that
    passes the existence check but breaks the manifest.
    """
    _verify_zip_bytes(payload, source=source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        # ``extractall`` raises if any member writes outside
        # ``destination`` — defence in depth against a zip bomb
        # pointing at ``../../etc``.
        zf.extractall(path=destination)


def _enumerated_clips(staged_root: Path) -> tuple[StagedClipFolder, ...]:
    """Walk the staged tree and parse each URFD-shaped folder.

    Same dedup logic as the Kaggle mirror: ``fall-01-cam0`` and
    ``fall-01-cam0 (1)`` (an OS-level file-collision suffix) are
    treated as one logical clip.
    """
    import re as _re
    clips: list[StagedClipFolder] = []
    seen: set[str] = set()
    for entry in sorted(staged_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        parsed = parse_university_folder_name(entry.name)
        if parsed is None:
            continue
        normalised = parsed.folder_name.strip().lower()
        collision_match = _re.search(r"\s+\(\d+\)\s*$", normalised)
        if collision_match is not None:
            normalised = normalised[:collision_match.start()]
        if normalised in seen:
            continue
        seen.add(normalised)
        clips.append(StagedClipFolder(
            absolute_path=entry,
            folder_name=parsed.folder_name,
            label=parsed.label,
            camera=parsed.camera,
            clip_sequence=parsed.clip_sequence,
        ))
    return tuple(clips)


def stage_urfd_from_university(
    data_root: Path,
    *,
    base_url: str = ALLOWED_UNIVERSITY_BASE_URL,
    force: bool = False,
    max_attempts: int = 3,
) -> UrfdUniversityStagingResult:
    """Stage URFD from the university repository.

    Args:
        data_root: root under which to stage URFD. The path the
            dataset lands at is the same shape as the Kaggle
            mirror's: ``<data_root>/datasets/urfd/``.
        base_url: must equal the whitelisted
            :data:`ALLOWED_UNIVERSITY_BASE_URL`. Any other value
            raises.
        force: when ``True``, re-download even if a complete
            staged tree exists.
        max_attempts: linear-retry count per download.

    Returns:
        A :class:`UrfdUniversityStagingResult` listing every
        successfully extracted clip folder, the persistent CSV
        paths, and the failure map for clips that errored mid-run.

    Raises:
        RuntimeError: when the base URL is not whitelisted, or
            when any downloaded zip is truncated or corrupt.
    """
    _verify_base_url(base_url)

    staged_root = Path(data_root) / "datasets" / "urfd"

    if not force and is_urfd_university_already_staged(staged_root):
        enumerated = _enumerated_clips(staged_root)
        return UrfdUniversityStagingResult(
            staged_root=staged_root,
            clip_folders=enumerated,
            csv_paths=_discover_csvs(staged_root),
            succeeded_clips=tuple(f.folder_name for f in enumerated),
            failed_clips={},
            already_staged=True,
            source_base_url=base_url,
        )

    # Idempotency: clear a half-staged tree before re-downloading so
    # a previous interrupted run doesn't leave a confusing mix.
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)

    succeeded: list[str] = []
    failed: dict[str, str] = {}
    csv_paths: dict[str, Path] = {}

    # Frame zips: 30 fall + 40 adl.
    for url in build_frame_zip_urls(base_url):
        clip_folder_name = _url_to_clip_folder_name(url)
        try:
            payload = _download_with_retry(url, max_attempts=max_attempts)
            destination = staged_root / clip_folder_name
            _safe_extract_zip(payload, destination, source=url)
            succeeded.append(clip_folder_name)
        except Exception as exc:  # noqa: BLE001
            failed[clip_folder_name] = (
                f"{type(exc).__name__}: {exc}"
            )

    # CSVs to a persistent path under the staged root.
    csv_destination = staged_root / "csvs"
    csv_destination.mkdir(parents=True, exist_ok=True)
    for filename, url in build_csv_urls(base_url).items():
        destination = csv_destination / filename
        try:
            payload = _download_with_retry(url, max_attempts=max_attempts)
            destination.write_bytes(payload)
            csv_paths[filename] = destination
        except Exception as exc:  # noqa: BLE001
            failed[filename] = f"{type(exc).__name__}: {exc}"

    # Write the marker ONLY when every expected file landed. A
    # half-staged tree is a real failure: not only is the marker
    # not written, the run raises so the caller / CI / a manual
    # human sees a clear "staging failed" rather than a silent
    # partial result. The brief: "fail loud on truncated / corrupt
    # zip — do not silently stage partial clips".
    if failed:
        details = "\n".join(
            f"  - {name}: {msg}" for name, msg in failed.items()
        )
        raise RuntimeError(
            f"URFD university staging failed — {len(failed)} file(s) "
            "did not land. Inspect the partial tree at "
            f"{staged_root} and re-run with force=True after fixing "
            "the network or source.\n"
            f"{details}"
        )

    (staged_root / STAGING_MARKER_FILENAME).write_text(
        f"staged_from={base_url}\n"
        "frame_zip_count=70\n"
        f"fall_zip_count={len(FALL_SEQUENCES)}\n"
        f"adl_zip_count={len(ADL_SEQUENCES)}\n",
        encoding="utf-8",
    )

    enumerated = _enumerated_clips(staged_root)
    return UrfdUniversityStagingResult(
        staged_root=staged_root,
        clip_folders=enumerated,
        csv_paths=csv_paths,
        succeeded_clips=tuple(succeeded),
        failed_clips={},
        already_staged=False,
        source_base_url=base_url,
    )

    (staged_root / STAGING_MARKER_FILENAME).write_text(
        f"staged_from={base_url}\n"
        "frame_zip_count=70\n"
        f"fall_zip_count={len(FALL_SEQUENCES)}\n"
        f"adl_zip_count={len(ADL_SEQUENCES)}\n",
        encoding="utf-8",
    )

    return UrfdUniversityStagingResult(
        staged_root=staged_root,
        clip_folders=_enumerated_clips(staged_root),
        csv_paths=csv_paths,
        succeeded_clips=tuple(succeeded),
        failed_clips={},
        already_staged=False,
        source_base_url=base_url,
    )


def _url_to_clip_folder_name(url: str) -> str:
    """Extract ``fall-01-cam0-rgb.zip`` → ``fall-01-cam0-rgb``.

    The university page publishes the cam0 RGB zips with a
    ``-rgb.zip`` suffix. The double-nested structure
    ``staged_root/fall-01-cam0-rgb/fall-01-cam0-rgb/*.png`` is
    what the real archive extracts to; the existing
    :func:`data.build_urfd_manifest.build_clip_id` and
    :class:`perception.frames.FrameFolderReader` already handle
    the ``-rgb`` folder name and the nested-frame layout, so we
    preserve the suffix here. Only the ``.zip`` archive extension
    is stripped.
    """
    name = url.rsplit("/", 1)[-1]
    if name.endswith(".zip"):
        name = name[: -len(".zip")]
    return name


def _discover_csvs(staged_root: Path) -> dict[str, Path]:
    """Return the on-disk CSV paths the script persists, or empty."""
    csv_dir = staged_root / "csvs"
    out: dict[str, Path] = {}
    if not csv_dir.is_dir():
        return out
    for entry in sorted(csv_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".csv":
            out[entry.name] = entry
    return out


__all__: tuple[str, ...] = (
    "ALLOWED_UNIVERSITY_BASE_URL",
    "ADL_CSV_FILENAME",
    "ADL_SEQUENCES",
    "CAMERA_SUFFIX",
    "FALL_CSV_FILENAME",
    "FALL_SEQUENCES",
    "STAGING_MARKER_FILENAME",
    "StagedClipFolder",
    "UrfdUniversityStagingResult",
    "build_csv_urls",
    "build_frame_zip_urls",
    "expected_files",
    "is_urfd_university_already_staged",
    "parse_university_folder_name",
    "stage_urfd_from_university",
)
