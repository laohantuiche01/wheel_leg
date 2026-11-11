"""Source-tree entry point; no editable install or old repository PYTHONPATH needed."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
