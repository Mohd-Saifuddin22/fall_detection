"""Restore cell 12 of the Issue 002 notebook with the full code body."""
import json
from pathlib import Path

NB = Path("colab/002_perception_urfd.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

RESTORED_CELL = [
    "from perception.frames import FrameFolderReader\n",
    "from perception.tracker import (\n",
    "    TrackerConfig,\n",
    "    run_tracker_on_folder,\n",
    "    query_gpu_name,\n",
    ")\n",
    "from perception.report import build_track_continuity_report\n",
    "from perception.artifacts import (\n",
    "    write_perception_artifacts,\n",
    "    detections_grouped_by_frame,\n",
    ")\n",
    "from perception.render import render_annotated_clip\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "\n",
    "MAX_FRAMES_PER_CLIP = None  # None = full-length (required by Issue 003); set a small int only for smoke-testing.\n",
    "ARTIFACTS_ROOT = layout.artifacts / \"perception\"\n",
    "ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "print(f\"GPU detected: {query_gpu_name() or 'none (CPU-only runtime)'}\")\n",
    "print()\n",
    "\n",
    "config = TrackerConfig()\n",
    "summary_rows = []\n",
    "\n",
    "for clip in manifest.clips:\n",
    "    clip_folder = layout.root / clip.source_path\n",
    "    if not clip_folder.is_dir():\n",
    "        print(f\"[skip] {clip.clip_id}: folder missing on Drive: {clip_folder}\")\n",
    "        continue\n",
    "\n",
    "    reader = FrameFolderReader(clip_folder)\n",
    "    ordered = reader.frames()\n",
    "    if not ordered:\n",
    "        print(f\"[skip] {clip.clip_id}: no frames in {clip_folder}\")\n",
    "        continue\n",
    "\n",
    "    # Debug-tier cap: take the first N frames when MAX_FRAMES_PER_CLIP\n",
    "    # is set; None means full-length (required by Issue 003).\n",
    "    capped = ordered if MAX_FRAMES_PER_CLIP is None else ordered[:MAX_FRAMES_PER_CLIP]\n",
    "    paths = [frame.path for frame in capped]\n",
    "\n",
    "    print(f\"[run]  {clip.clip_id}: {len(paths)} frames from {clip_folder.name}\")\n",
    "    run = run_tracker_on_folder(clip.clip_id, paths, config=config)\n",
    "    run.source_folder = str(clip_folder.relative_to(layout.root))\n",
    "\n",
    "    report = build_track_continuity_report(run, source_folder=run.source_folder)\n",
    "    out_dir = ARTIFACTS_ROOT / clip.clip_id\n",
    "    paths_written = write_perception_artifacts(out_dir, run, report)\n",
    "\n",
    "    # Render annotated frames for visual review.\n",
    "    images = [np.array(Image.open(p).convert(\"RGB\")) for p in paths]\n",
    "    detections_by_frame = detections_grouped_by_frame(run.detections, run.frame_count)\n",
    "    annotated_paths = render_annotated_clip(\n",
    "        images, detections_by_frame,\n",
    "        output_dir=out_dir / \"annotated\",\n",
    "        clip_id=clip.clip_id,\n",
    "    )\n",
    "\n",
    "    summary_rows.append({\n",
    "        \"clip_id\": clip.clip_id,\n",
    "        \"frames\": run.frame_count,\n",
    "        \"detections\": run.detection_count,\n",
    "        \"tracks\": run.track_count,\n",
    "        \"longest_track_id\": report.longest_track_id,\n",
    "        \"longest_track_length\": report.longest_track_length,\n",
    "        \"id_switch_count\": report.id_switch_count,\n",
    "        \"fps\": round(report.fps, 2),\n",
    "        \"latency_ms_per_frame\": round(report.latency_ms_per_frame, 2),\n",
    "    })\n",
    "    print(f\"       -> {report.detection_count} detections, \"\n",
    "          f\"{report.track_count} tracks, longest=track {report.longest_track_id} \"\n",
    "          f\"({report.longest_track_length} frames), \"\n",
    "          f\"{report.id_switch_count} id switches, {report.fps:.1f} fps\")\n",
    "    print(f\"       -> artefacts: {out_dir}\")\n",
]

# Find and replace cell 12. Be defensive in case the indexing has shifted.
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "MAX_FRAMES_PER_CLIP = None" in src and "ARTIFACTS_ROOT = layout.artifacts" in src:
        cell["source"] = RESTORED_CELL
        print(f"Restored cell {idx}")
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        break
else:
    raise SystemExit("Could not locate the run loop cell to restore.")