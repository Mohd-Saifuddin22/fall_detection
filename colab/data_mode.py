"""Data-mode resolution for the Colab pipeline.

This module answers ONE question: "where do I read frames from, and
where do I write artefacts to?"

The default mode is ``LOCAL``: datasets live on the Colab local disk
(``/content/fall_local/``) so the active pipeline never pays the Drive
FUSE small-file I/O cost. Final artefacts (tracking outputs, crop
shards, logs, metrics) still land on Drive so they survive Colab
runtime resets.

Why this is a first-class mode, not an env-var hack
-------------------------------------------------
Before Issue 003-fix-2, the only way to get the fast local path was
to set two env vars by hand:

    FALL_DETECTION_DRIVE_ROOT=/content/fall_local
    SKIP_LOCAL_STAGING=1

That worked but it was a manual hack — easy to forget, easy to typo,
and impossible to discover from the notebook alone. This module
turns the same behaviour into a single declarative flag the
notebooks set at the top, with logging that confirms the mode at
runtime so a human reviewer can verify the fast path actually fired.

Public surface
--------------
- :class:`DataMode` — enum ``LOCAL`` / ``DRIVE``.
- :data:`DEFAULT_DATA_MODE` — ``LOCAL`` (the fast path).
- :data:`DEFAULT_LOCAL_DATA_ROOT` — ``/content/fall_local``.
- :data:`DEFAULT_DRIVE_ROOT` — ``/content/drive/MyDrive/fall_detection``.
- :class:`DataLayout` — holds dataset_root + artifact_root + helpers.
- :func:`resolve_data_layout` — factory used by notebooks.

Backward compatibility
----------------------
``resolve_data_layout(mode="drive")`` returns a layout where
``dataset_root`` and ``artifact_root`` are the SAME Drive directory
— i.e. legacy behaviour. Existing code that uses
``layout.artifacts`` / ``layout.logs`` keeps working unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class DataMode(str, Enum):
    """Where active processing reads frames from.

    - ``LOCAL``: datasets live on the Colab local disk. Artefacts still
      land on Drive so they persist across runtime resets.
    - ``DRIVE``: legacy mode — datasets AND artefacts live on Drive.
      Kept for compatibility and for hosts that don't have a fast
      local disk available.
    """

    LOCAL = "local"
    DRIVE = "drive"


# The default mode for the pipeline. Local-first because Drive FUSE
# small-file reads are the documented bottleneck for this project
# (PRD: "small-file I/O from Drive is prohibitively slow").
DEFAULT_DATA_MODE: DataMode = DataMode.LOCAL

# Where datasets land in LOCAL mode. Override via the
# ``FALL_DETECTION_LOCAL_DATA_ROOT`` env var if a host needs a
# different path (e.g. for testing).
DEFAULT_LOCAL_DATA_ROOT: Path = Path("/content/fall_local")

# Where artefacts land in LOCAL mode AND where everything lives in
# DRIVE mode. Matches the legacy ``DriveLayout`` default.
DEFAULT_DRIVE_ROOT: Path = Path("/content/drive/MyDrive/fall_detection")

# Env-var overrides (preserved for compatibility with the prior
# manual-hack workflow).
LOCAL_DATA_ROOT_ENV_VAR: str = "FALL_DETECTION_LOCAL_DATA_ROOT"
DRIVE_ROOT_ENV_VAR: str = "FALL_DETECTION_DRIVE_ROOT"
DATA_MODE_ENV_VAR: str = "FALL_DETECTION_DATA_MODE"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataLayout:
    """Resolved roots for one pipeline run.

    The two roots answer different questions:

        ``dataset_root`` — where Stage URFD puts the raw data and
                            where the perception/cropping runners READ
                            frames from during active processing.

        ``artifact_root`` — where the perception/cropping runners
                              WRITE tracking outputs, shards, logs,
                              and metrics. Should be on Drive so they
                              survive a Colab reset.

    In ``LOCAL`` mode the two roots differ (local vs Drive). In
    ``DRIVE`` mode they collapse to the same directory — legacy
    behaviour preserved for compatibility.
    """

    mode: DataMode
    dataset_root: Path
    artifact_root: Path

    @property
    def root(self) -> Path:
        """Alias for :attr:`dataset_root`.

        Kept for backward compatibility with cells that pre-date the
        data_mode split — they reference ``layout.root`` expecting a
        single root. In LOCAL mode that's the local data root; in
        DRIVE mode it collapses to ``artifact_root``.
        """
        return self.dataset_root

    @property
    def datasets(self) -> Path:
        """Where URFD (and future staged datasets) live."""
        return self.dataset_root / "datasets"

    @property
    def artifacts(self) -> Path:
        """Where tracking outputs and crop shards are written."""
        return self.artifact_root / "artifacts"

    @property
    def checkpoints(self) -> Path:
        return self.artifact_root / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.artifact_root / "metrics"

    @property
    def logs(self) -> Path:
        return self.artifact_root / "logs"

    def ensure(self) -> None:
        """Create every directory the pipeline writes to.

        Idempotent. Includes ``dataset_root`` in LOCAL mode so a
        ``stage_urfd_from_kaggle`` call has a place to land; in DRIVE
        mode the dataset dir already exists on Drive from a prior run.
        """
        for path in (
            self.datasets,
            self.artifacts,
            self.checkpoints,
            self.metrics,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def is_local_mode(self) -> bool:
        return self.mode is DataMode.LOCAL


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def resolve_data_layout(
    mode: str | DataMode | None = None,
    *,
    local_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> DataLayout:
    """Resolve the data + artefact roots for one pipeline run.

    Resolution order (first non-None wins):
        1. The explicit ``mode`` argument.
        2. The ``FALL_DETECTION_DATA_MODE`` env var.
        3. The module-level ``DEFAULT_DATA_MODE`` (``LOCAL``).

    Within a mode:
        - LOCAL: ``local_root`` arg → env var → ``/content/fall_local``.
                 Drive root is the standard ``/content/drive/MyDrive/fall_detection``.
        - DRIVE: ``drive_root`` arg → env var → default. Both roots
                  collapse to the same Drive directory.

    Args:
        mode: ``"local"`` / ``"drive"`` / :class:`DataMode` / ``None``.
        local_root: override the local dataset root (for testing or
            hosts without ``/content``).
        drive_root: override the Drive artefact root.

    Returns:
        A populated :class:`DataLayout`.
    """
    if mode is None:
        env_mode = os.environ.get(DATA_MODE_ENV_VAR)
        if env_mode:
            mode = env_mode
        else:
            mode = DEFAULT_DATA_MODE
    if isinstance(mode, str):
        mode = DataMode(mode.strip().lower())

    drive = Path(drive_root or os.environ.get(DRIVE_ROOT_ENV_VAR)
                  or DEFAULT_DRIVE_ROOT)

    if mode is DataMode.LOCAL:
        local = Path(local_root or os.environ.get(LOCAL_DATA_ROOT_ENV_VAR)
                     or DEFAULT_LOCAL_DATA_ROOT)
        return DataLayout(mode=mode, dataset_root=local, artifact_root=drive)

    # DRIVE mode: dataset and artefact roots collapse to the same Drive
    # directory — legacy behaviour for compatibility with code that
    # expected everything under one root.
    return DataLayout(mode=mode, dataset_root=drive, artifact_root=drive)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def describe_layout(layout: DataLayout) -> str:
    """Return a one-line, log-friendly summary of the active layout.

    Used by notebooks to confirm the fast path is actually firing.
    """
    same = layout.dataset_root == layout.artifact_root
    if same:
        return (f"data_mode={layout.mode.value}  dataset_root=artifact_root="
                f"{layout.dataset_root}")
    return (f"data_mode={layout.mode.value}  dataset_root={layout.dataset_root}  "
            f"artifact_root={layout.artifact_root}")


def select_active_paths(layout: DataLayout) -> Iterable[Path]:
    """Yield the directories the pipeline reads from during active work.

    In LOCAL mode this is the local dataset root; in DRIVE mode it
    collapses to the artefact root. Used by tests + diagnostics to
    assert that no Drive frame reads happen during active processing.
    """
    yield layout.datasets
    if layout.dataset_root != layout.artifact_root:
        yield layout.dataset_root


__all__: tuple[str, ...] = (
    "DEFAULT_DATA_MODE",
    "DEFAULT_DRIVE_ROOT",
    "DEFAULT_LOCAL_DATA_ROOT",
    "DATA_MODE_ENV_VAR",
    "DRIVE_ROOT_ENV_VAR",
    "DataLayout",
    "DataMode",
    "LOCAL_DATA_ROOT_ENV_VAR",
    "describe_layout",
    "resolve_data_layout",
    "select_active_paths",
)