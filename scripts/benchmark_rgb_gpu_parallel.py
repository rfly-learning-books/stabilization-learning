"""Entry point for isolated GPU RGB environment measurement."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT/"stabilization-gym", ROOT/"stabilization-learning-control"):
    sys.path.insert(0, str(folder))

if __name__ == "__main__":
    from extensions.rgb_gpu_parallel.benchmark import main
    main()
