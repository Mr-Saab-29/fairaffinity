# tests/conftest.py
from pathlib import Path
import sys

# Repo root (the folder that contains `src/` and `tests/`)
ROOT = Path(__file__).resolve().parent.parent  # == parents[1]

# Ensure ROOT is on sys.path so `import src...` works
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)