"""Local-disk frame staging for the Issue 002 perception front-end.

Why this module exists
---------------------
Live Issue 002 runs showed identical ~1.8 fps on L4 and A100 GPUs — the
GPU is NOT the bottleneck. The runner was reading PNG frames directly
from the mounted Google Drive (a FUSE filesystem) one frame at a time,
and Drive's small-file I/O is the documented bottleneck for this
project (PRD: "small-file I/O from Drive is prohibitively slow").
FUSE reads can also be arbitrarily cached/served by Colab's mount,
which makes the runtime appear stable but leaves the GPU idle.

The fix is to copy each clip's frames from Drive to a local
``/content/...`` directory BEFORE running the tracker, then read from
local disk. After tracking, only the project's artefact paths on
Drive are written to — the raw dataset folder on Drive is never
modified.

Public surface
--------------
    - :class:`LocalFrameStager` — per-clip staging with safe cleanup.
    - :class:`StagingResult` — per-clip counts + timing.
    - :func:`stage_clip_frames` — convenience one-shot call.
    - :data:`DEFAULT_LOCAL_ROOT` — ``/content/fall_detection_local``
      (override via ``FALL_DETECTION_LOCAL_ROOT`` env var).
    - :data:`COLAB_LOCAL_ROOT_DEFAULT` — same as above; documented
      separately so the notebook can echo it back to the user.

Why the staged files are renamed
-------------------------------
The Drive folder may nest one level deep (``fall-01-cam0-rgb/
fall-01-cam0-rgb/*.png`` — see ``perception.frames``). We copy the
ordered frame list (which already accounts for the nested layout)
rather than ``shutil.copytree``, so the destination directory is
flat ``frame_NNNNN.png`` files in numeric order. This makes the
staging directory self-documenting and removes any chance of a
nested copy escaping the staging root.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from perception.frames import FrameRecord, discover_frames


# Default staging root inside the Colab runtime's local disk.
# ``/content/`` is the Colab ephemeral disk (fast, large); everything
# under it is wiped on runtime reset, which is exactly the lifetime
# we want for staged frames.
DEFAULT_LOCAL_ROOT: Path = Path("/content/fall_detection_local")
COLAB_LOCAL_ROOT_DEFAULT: Path = DEFAULT_LOCAL_ROOT

# Staging is opt-out via the environment. Useful for tests and for
# hosts where the runtime already has a faster path to data.
SKIP_STAGING_ENV_VAR: str = "FALL_DETECTION_SKIP_LOCAL_STAGING"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StagingResult:
    """Outcome of staging one clip's frames to local disk."""

    clip_id: str
    source_folder: str
    local_folder: Path
    frame_count: int
    bytes_copied: int
    elapsed_seconds: float

    @property
    def fps_copy(self) -> float:
        """Throughput of the copy step (frames / second)."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frame_count / self.elapsed_seconds


class LocalStagingError(RuntimeError):
    """Raised when local staging cannot proceed safely."""


# ---------------------------------------------------------------------------
# Stager
# ---------------------------------------------------------------------------


class LocalFrameStager:
    """Per-clip staging helper.

    Usage::

        stager = LocalFrameStager(local_root=Path("/content/fall_detection_local"))
        for clip in clips:
            with stager.stage_clip(clip_id, source_folder) as staged_folder:
                # run tracker against `staged_folder`
                ...

    The context manager guarantees the staged folder is removed on
    exit even if the tracker raises, so stale frames from a failed
    clip can't contaminate the next clip.
    """

    def __init__(
        self,
        local_root: Path | None = None,
        *,
        skip_if_env_set: bool = True,
    ) -> None:
        self._local_root = Path(local_root or os.environ.get(
            "FALL_DETECTION_LOCAL_ROOT", DEFAULT_LOCAL_ROOT,
        ))
        self._skip_if_env_set = skip_if_env_set
        self._active_folders: set[Path] = set()

    @property
    def local_root(self) -> Path:
        return self._local_root

    def stage_clip(
        self,
        clip_id: str,
        source_folder: Path,
        ordered_frames: Iterable[FrameRecord],
    ) -> "StagedClipContext":
        """Copy ``ordered_frames`` to a fresh local sub-folder for this clip.

        Args:
            clip_id: stable identifier for the clip; used as the
                destination sub-folder name so different clips never
                collide on the local disk.
            source_folder: original Drive folder (recorded for logging
                but not modified).
            ordered_frames: the temporal-order frame list. Caller is
                responsible for ordering (typically from
                :class:`FrameFolderReader`).

        Returns:
            A :class:`StagedClipContext` whose ``local_folder`` is the
            freshly-populated local directory.
        """
        if self._skip_if_env_set and os.environ.get(SKIP_STAGING_ENV_VAR):
            # Caller opted out — pretend the source folder is already
            # "local". Useful when running on a host where the data
            # already lives on fast disk.
            return StagedClipContext(
                clip_id=clip_id,
                local_folder=Path(source_folder),
                stager=self,
                bytes_copied=0,
                elapsed_seconds=0.0,
                frame_count=sum(1 for _ in ordered_frames),
                skipped=True,
            )

        local_folder = self._local_root / _safe_local_name(clip_id)
        # Defensive cleanup: a previous run that crashed before the
        # context exit may have left frames here. Remove them so they
        # can't be silently picked up by the next clip.
        _safe_rmtree(local_folder)
        local_folder.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        bytes_copied = 0
        frame_count = 0
        for index, frame in enumerate(ordered_frames):
            # Sequential numeric suffix keeps the staged layout
            # self-documenting and matches the colab notebook's
            # expectations (frame_00001.png, frame_00002.png, …).
            destination = local_folder / f"frame_{index:05d}.png"
            shutil.copy2(frame.path, destination)
            try:
                bytes_copied += destination.stat().st_size
            except OSError:
                # Source unreadable mid-copy — let the exception
                # propagate to the caller; we don't want to claim a
                # successful staging when files are missing.
                raise
            frame_count += 1
        elapsed = time.perf_counter() - started

        self._active_folders.add(local_folder)
        return StagedClipContext(
            clip_id=clip_id,
            local_folder=local_folder,
            stager=self,
            bytes_copied=bytes_copied,
            elapsed_seconds=elapsed,
            frame_count=frame_count,
            skipped=False,
        )

    def cleanup(self) -> None:
        """Remove every staged folder this stager created.

        Idempotent — safe to call multiple times or after a partial
        run. Does NOT remove the ``local_root`` itself (we may want
        to inspect staged folders post-mortem).
        """
        for folder in list(self._active_folders):
            _safe_rmtree(folder)
        self._active_folders.clear()

    def cleanup_one(self, local_folder: Path) -> None:
        """Remove one staged folder (used by the context manager)."""
        _safe_rmtree(local_folder)
        self._active_folders.discard(local_folder)


class StagedClipContext:
    """Context manager returned by :meth:`LocalFrameStager.stage_clip`.

    Exposes the local folder via :attr:`local_folder` so the tracker
    can be pointed at it. On exit, the folder is removed.
    """

    def __init__(
        self,
        *,
        clip_id: str,
        local_folder: Path,
        stager: LocalFrameStager,
        bytes_copied: int,
        elapsed_seconds: float,
        frame_count: int,
        skipped: bool,
    ) -> None:
        self.clip_id = clip_id
        self.local_folder = local_folder
        self._stager = stager
        self.result = StagingResult(
            clip_id=clip_id,
            source_folder=str(local_folder),
            local_folder=local_folder,
            frame_count=frame_count,
            bytes_copied=bytes_copied,
            elapsed_seconds=elapsed_seconds,
        )
        self._skipped = skipped

    def __enter__(self) -> "StagedClipContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._skipped:
            self._stager.cleanup_one(self.local_folder)


# ---------------------------------------------------------------------------
# Convenience one-shot
# ---------------------------------------------------------------------------


def stage_clip_frames(
    clip_id: str,
    source_folder: Path,
    *,
    local_root: Path | None = None,
) -> tuple[StagedClipContext, list[FrameRecord]]:
    """One-shot staging that pairs the context manager with discovery.

    Args:
        clip_id: stable identifier for the clip.
        source_folder: Drive folder containing the clip frames
            (possibly nested; the loader handles that).
        local_root: optional override for the local staging root.

    Returns:
        A ``(context, ordered_frames)`` tuple. Use the context as a
        context manager so cleanup is guaranteed; pass the ordered
        frames to the tracker.
    """
    stager = LocalFrameStager(local_root=local_root)
    ordered = discover_frames(source_folder)
    context = stager.stage_clip(clip_id, source_folder, ordered)
    return context, ordered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_local_name(clip_id: str) -> str:
    """Sanitise a clip_id for use as a local directory name.

    Conservative: replace any character outside ``[A-Za-z0-9_-]`` with
    ``_``. Empty result falls back to ``unnamed``.
    """
    import re as _re
    cleaned = _re.sub(r"[^A-Za-z0-9_-]", "_", clip_id)
    return cleaned or "unnamed"


def _safe_rmtree(path: Path) -> None:
    """Remove ``path`` if it exists; never raise on missing."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


__all__: tuple[str, ...] = (
    "COLAB_LOCAL_ROOT_DEFAULT",
    "DEFAULT_LOCAL_ROOT",
    "SKIP_STAGING_ENV_VAR",
    "LocalFrameStager",
    "LocalStagingError",
    "StagedClipContext",
    "StagingResult",
    "stage_clip_frames",
)