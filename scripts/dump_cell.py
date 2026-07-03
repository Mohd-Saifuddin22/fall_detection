"""Dump a single notebook cell by index (Windows-safe encoding)."""
import json
import sys
from pathlib import Path

NB = Path("colab/002_perception_urfd.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
i = int(sys.argv[1])
cell = nb["cells"][i]
src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
print(f"cell {i} ({cell['cell_type']}):")
print(src)