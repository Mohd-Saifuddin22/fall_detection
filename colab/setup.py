"""Reusable Google Colab environment setup helpers for the fall-detection project.

This module is the single source of truth for:
    - the persistent Google Drive directory layout,
    - the approved pip dependency stack (Colab-compatible),
    - the post-install environment lock + run-log capture.

Design rules (see ``context.txt``):
    - Never reinstall torch / torchvision — use what Colab ships with.
    - TrackEval is intentionally not in the default install (no clean pip wheel);
      it is documented in ``TRACKEVAL_INSTALL_NOTE`` for manual install.
    - Every helper is callable from both a notebook and a plain script, and
      performs no work that cannot be undone by re-running it (idempotent).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Drive layout
# ---------------------------------------------------------------------------

DRIVE_ROOT_ENV_VAR = "FALL_DETECTION_DRIVE_ROOT"
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/fall_detection")

DRIVE_SUBDIRS: tuple[str, ...] = (
    "datasets",
    "artifacts",
    "checkpoints",
    "metrics",
    "logs",
)


@dataclass(frozen=True)
class DriveLayout:
    """Resolved on-disk paths for the project's persistent Drive directory tree."""

    root: Path
    datasets: Path
    artifacts: Path
    checkpoints: Path
    metrics: Path
    logs: Path

    @classmethod
    def resolve(cls, drive_root: Path | None = None) -> "DriveLayout":
        """Resolve the Drive layout from an explicit path or the env var override."""
        if drive_root is None:
            env_root = os.environ.get(DRIVE_ROOT_ENV_VAR)
            drive_root = Path(env_root) if env_root else DEFAULT_DRIVE_ROOT
        return cls(
            root=drive_root,
            datasets=drive_root / "datasets",
            artifacts=drive_root / "artifacts",
            checkpoints=drive_root / "checkpoints",
            metrics=drive_root / "metrics",
            logs=drive_root / "logs",
        )

    def ensure(self) -> None:
        """Create every directory if missing. Safe to call repeatedly."""
        for path in (self.root, self.datasets, self.artifacts,
                     self.checkpoints, self.metrics, self.logs):
            path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dependency stack
# ---------------------------------------------------------------------------

# These are the ONLY packages the project installs via pip. torch and
# torchvision are intentionally omitted — Colab ships a CUDA-matched build
# and reinstalling it risks pulling a CPU-only wheel.
APPROVED_PIP_PACKAGES: tuple[str, ...] = (
    "ultralytics",
    # Issue 006 Prompt #2 — pinned exactly. The Kinetics-finetuned
    # VideoMAE backbone integrity check depends on a specific
    # transformers implementation; a silent upgrade risks an
    # incompatible shape or attribute change.
    "transformers==4.46.3",
    "accelerate",
    "webdataset",
    "opencv-python",
    "decord",
    "Pillow",
    "scikit-learn",
    "torchmetrics",
    "motmetrics",
    "omegaconf",
    "pyyaml",
    "tqdm",
    # Issue 002 — dataset staging from Kaggle.
    # Used only by data/stage_urfd.py and only against Colab Secrets at
    # runtime; never invoked from inference or tracking code.
    "kagglehub",
    # Issue 002 close-out — linear-assignment solver used by MOT metrics
    # (TrackEval-style IDF1 / MOTA computation). Pinned here so future
    # eval code can rely on it being importable without surprise.
    "lap",
)

# TrackEval has no clean pip release; install from source. Listed separately
# so the default install stays a single `pip install` line and the source
# install is opt-in + documented.
TRACKEVAL_INSTALL_NOTE = (
    "TrackEval is NOT installed by default. It has no maintained pip wheel "
    "(its README installs from git). To install:\n"
    "    pip install git+https://github.com/JonathonLuiten/TrackEval.git\n"
    "Run that once per runtime if MOT metrics from TrackEval are needed."
)


def install_approved_packages(quiet: bool = True) -> None:
    """Install the approved dependency stack, skipping torch / torchvision.

    A single pip invocation keeps the install atomic and fast. ``quiet=True``
    suppresses per-package chatter but still surfaces hard errors.
    """
    cmd = [sys.executable, "-m", "pip", "install", *APPROVED_PIP_PACKAGES]
    if quiet:
        # `--quiet` would silence real errors too; instead we capture stdout
        # and only print on failure.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise RuntimeError(
                "Approved dependency install failed; see pip output above."
            )
    else:
        subprocess.run(cmd, check=True)


def capture_pip_freeze(destination: Path) -> Path:
    """Write a real ``pip freeze`` snapshot to ``destination`` and return it.

    This is the project's authoritative environment lock — a literal pip
    freeze, not a curated subset — so any transitive change in Colab's base
    image is reflected in version control.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    destination.write_text(result.stdout, encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _try_query_gpu() -> tuple[str | None, float | None]:
    """Best-effort GPU name + total VRAM (GiB) via ``nvidia-smi``.

    Returns ``(None, None)`` if ``nvidia-smi`` is unavailable or fails — the
    run log is allowed to record "unknown" rather than crash the setup.
    """
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, None
    if not out:
        return None, None
    # Use the first GPU line; multi-GPU hosts are out of scope for Colab.
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return parts[0], None
    try:
        vram_mib = float(parts[1])
        return parts[0], round(vram_mib / 1024.0, 2)
    except ValueError:
        return parts[0], None


def _cuda_version_from_torch() -> str | None:
    """Return torch's reported CUDA version, or None if torch is unavailable."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    return torch.version.cuda


def capture_run_log(
    destination: Path,
    layout: DriveLayout,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write a machine-readable environment snapshot to ``destination``.

    Fields captured: timestamp (UTC, ISO 8601), GPU name, GPU VRAM (GiB),
    CUDA version (from torch), Python version, and the resolved Drive layout
    paths. Caller-supplied ``extra`` fields are merged in last so they take
    precedence — useful for tagging the run with an issue number.
    """
    gpu_name, vram_gib = _try_query_gpu()
    payload: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "gpu_vram_gib": vram_gib,
        "cuda_version": _cuda_version_from_torch(),
        "python_version": platform.python_version(),
        "drive_layout": {
            "root": str(layout.root),
            "datasets": str(layout.datasets),
            "artifacts": str(layout.artifacts),
            "checkpoints": str(layout.checkpoints),
            "metrics": str(layout.metrics),
            "logs": str(layout.logs),
        },
    }
    if extra:
        payload.update(extra)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------


def run_setup(
    drive_root: Path | None = None,
    *,
    install_deps: bool = True,
    write_freeze: bool = True,
    write_run_log: bool = True,
    extra_log_fields: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Run the full Colab setup in one call.

    Steps (all idempotent):
        1. Resolve and create the Drive layout.
        2. Optionally install the approved pip stack.
        3. Optionally write a real ``pip freeze`` lock to Drive.
        4. Optionally write a run-log JSON to Drive.

    Returns a mapping of artifact name → on-disk path for the caller to log
    or assert against.
    """
    layout = DriveLayout.resolve(drive_root)
    layout.ensure()

    artifacts: dict[str, Path] = {}

    if install_deps:
        install_approved_packages(quiet=True)

    if write_freeze:
        artifacts["pip_freeze"] = capture_pip_freeze(layout.artifacts / "pip_freeze.txt")

    if write_run_log:
        artifacts["run_log"] = capture_run_log(
            layout.logs / "setup_run_log.json",
            layout,
            extra=extra_log_fields,
        )

    return artifacts


__all__: Iterable[str] = (
    "DRIVE_ROOT_ENV_VAR",
    "DEFAULT_DRIVE_ROOT",
    "DRIVE_SUBDIRS",
    "DriveLayout",
    "APPROVED_PIP_PACKAGES",
    "TRACKEVAL_INSTALL_NOTE",
    "install_approved_packages",
    "capture_pip_freeze",
    "capture_run_log",
    "run_setup",
)