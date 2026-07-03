"""One-shot script to patch the Issue 002 notebook: remove the debug frame cap."""
import json
from pathlib import Path

NB_PATH = Path("colab/002_perception_urfd.ipynb")

NEW_RUN_INTRO_MARKDOWN = [
    "## 6. Run YOLO26 + ByteTrack on every debug clip\n",
    "\n",
    "For each clip:\n",
    "\n",
    "  1. Resolve the frame folder on Drive.\n",
    "  2. Order frames numerically via `FrameFolderReader` (CRITICAL —\n",
    "     out-of-order frames produce invalid tracks).\n",
    "  3. Run `run_tracker_on_folder(clip_id, ordered_paths, config)`.\n",
    "  4. Build a `TrackContinuityReport` and write the per-clip artefacts.\n",
    "  5. Render annotated frames so a human can review track-through-fall.\n",
    "\n",
    "Issue 003 consumes the full-length cam0 tracks from this step, so the\n",
    "default here is NO frame cap. Set `MAX_FRAMES_PER_CLIP` to a small\n",
    "integer only when smoke-testing the pipeline (e.g. 60); leave it\n",
    "`None` for real runs.\n",
]

NEW_RUN_INTRO_CODE = [
    "MAX_FRAMES_PER_CLIP = None  # None = full-length (required by Issue 003); set a small int only for smoke-testing.\n",
    "ARTIFACTS_ROOT = layout.artifacts / \"perception\"\n",
    "ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)\n",
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    patched_code = False
    patched_md = False
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "code" and (
            "MAX_FRAMES_PER_CLIP = 120" in src
            or "MAX_FRAMES_PER_CLIP = None" in src
        ):
            cell["source"] = NEW_RUN_INTRO_CODE
            patched_code = True
        elif cell["cell_type"] == "markdown" and (
            "Run starts at `max_frames_per_clip`" in src
            or "Raise the limit for the second pass" in src
            or "Issue 003 consumes the full-length cam0 tracks" in src
        ):
            cell["source"] = NEW_RUN_INTRO_MARKDOWN
            patched_md = True

    if not (patched_code and patched_md):
        print(
            f"No-op (already patched): code={patched_code} md={patched_md}"
        )
        return

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Patched {NB_PATH}: removed MAX_FRAMES_PER_CLIP=120 debug cap.")


if __name__ == "__main__":
    main()