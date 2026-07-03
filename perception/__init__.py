"""Perception front-end — Issue 002 vertical slice.

Public surface:
    - :mod:`perception.frames` — ordered frame-folder loader.
    - :mod:`perception.tracker` — YOLO26 + ByteTrack wrapper.
    - :mod:`perception.render` — annotated-frame renderer.
    - :mod:`perception.report` — track-continuity / fragmentation report.
    - :mod:`perception.artifacts` — write per-clip JSON outputs to Drive.

The pipeline the Issue 002 notebook / scripts run is:

    FrameFolderReader
        → run_tracker_on_folder       (perception.tracker)
        → build_track_continuity_report (perception.report)
        → write_perception_artifacts  (perception.artifacts)
        → render_annotated_clip       (perception.render)
"""

from __future__ import annotations

from perception import artifacts, frames, render, report, tracker

__all__: tuple[str, ...] = (
    "artifacts",
    "frames",
    "render",
    "report",
    "tracker",
)