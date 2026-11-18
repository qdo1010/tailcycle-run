import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# every module reads the config at import: point it at the shipped example
os.environ["TAILCYCLE_RUN_CONFIG"] = str(ROOT / "config.example.toml")
