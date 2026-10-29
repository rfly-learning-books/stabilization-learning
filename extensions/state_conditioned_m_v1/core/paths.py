"""Make the extension importable without modifying the frozen packages."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT / "stabilization-gym", ROOT / "stabilization-learning-control", ROOT / "migration/next_envs_v1"):
    text = str(folder)
    if text not in sys.path:
        sys.path.insert(0, text)
