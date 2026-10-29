"""Unified, local, random-origin application-training entry point."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from extensions.scratch_apps_v1.suite import main
if __name__=='__main__':main()
