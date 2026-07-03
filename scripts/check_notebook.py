"""Diagnostic: list every cell that references MAX_FRAMES_PER_CLIP."""
import json
from pathlib import Path

NB = Path("colab/002_perception_urfd.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "MAX_FRAMES_PER_CLIP" in src or "max_frames_per_clip" in src:
        print(f"cell {i} ({cell['cell_type']}): {src[:200]}")
        print("---")