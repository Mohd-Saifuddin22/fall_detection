"""Artifact writer for the perception front-end.

A single function that takes the outputs of a perception run + the
report and writes three things to a per-clip sub-directory on Drive:

    - ``<clip_id>_detections.json`` — flat list of every detected box.
    - ``<clip_id>_tracks.json`` — per-track summary.
    - ``<clip_id>_report.json`` — the track-continuity report (also the
      run-level summary a human reads first).
    - ``<clip_id>_run_meta.json`` — what config + GPU + timing was used.

Kept deliberately simple — the goal is "open one folder on Drive and
see everything about this clip's perception run."
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from perception.report import TrackContinuityReport
from perception.tracker import DetectionBox, PerceptionRunResult, TrackSummary


def _write_json(path: Path, payload: object) -> Path:
    """Write ``payload`` to ``path`` as JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_perception_artifacts(
    output_dir: Path,
    run: PerceptionRunResult,
    report: TrackContinuityReport,
) -> dict[str, Path]:
    """Write all run-level JSON artefacts to ``output_dir``.

    Returns a mapping of artefact name → on-disk path so the caller can
    log or assert against the exact paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_payload = [asdict(d) for d in run.detections]
    tracks_payload = [asdict(t) for t in run.tracks]
    run_meta = {
        "clip_id": run.clip_id,
        "source_folder": run.source_folder,
        "config": asdict(run.config),
        "frame_count": run.frame_count,
        "detection_count": run.detection_count,
        "track_count": run.track_count,
        "decode_failures": run.decode_failures,
        "elapsed_seconds": run.elapsed_seconds,
        "fps": run.fps,
        "latency_ms_per_frame": run.latency_ms_per_frame,
        "gpu_name": run.gpu_name,
        "fallback_used": run.fallback_used,
    }

    return {
        "detections": _write_json(
            output_dir / f"{run.clip_id}_detections.json", detections_payload),
        "tracks": _write_json(
            output_dir / f"{run.clip_id}_tracks.json", tracks_payload),
        "report": _write_json(
            output_dir / f"{run.clip_id}_report.json", report.to_dict()),
        "run_meta": _write_json(
            output_dir / f"{run.clip_id}_run_meta.json", run_meta),
    }


def detections_grouped_by_frame(
    detections: Iterable[DetectionBox],
    frame_count: int,
) -> list[list[DetectionBox]]:
    """Bucket detections into per-frame lists for the renderer.

    Returns a list of length ``frame_count``; empty lists are explicit
    (a frame with no detections is "no boxes to draw", not "missing").
    """
    buckets: list[list[DetectionBox]] = [[] for _ in range(frame_count)]
    for det in detections:
        if 0 <= det.frame_index < frame_count:
            buckets[det.frame_index].append(det)
    return buckets


__all__: tuple[str, ...] = (
    "write_perception_artifacts",
    "detections_grouped_by_frame",
)