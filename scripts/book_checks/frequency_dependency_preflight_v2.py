"""Validate the isolated compatibility dependency and actual thread limits."""
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.linalg import eigvalsh
import threadpoolctl


def main():
    target = Path(sys.argv[1]).resolve()
    expected = {"numpy": "1.26.4", "scipy": "1.11.1", "threadpoolctl": "3.6.0"}
    versions = {key: importlib.metadata.version(key) for key in expected}
    if versions != expected:
        raise RuntimeError(f"Unexpected dependency versions: {versions}")
    module_path = Path(threadpoolctl.__file__).resolve()
    distribution_path = Path(importlib.metadata.distribution("threadpoolctl").locate_file("")).resolve()
    if not module_path.is_relative_to(target) or distribution_path != target:
        raise RuntimeError("threadpoolctl module and metadata must both use isolated target")
    # Force both NumPy and SciPy BLAS/LAPACK libraries to be loaded, then verify
    # the original public threadpool_limits context actually constrains them.
    np.eye(2) @ np.eye(2)
    eigvalsh(np.eye(2))
    before = threadpoolctl.threadpool_info()
    with threadpoolctl.threadpool_limits(limits=1):
        limited = threadpoolctl.threadpool_info()
        if not limited or not all(item.get("num_threads") == 1 for item in limited):
            raise RuntimeError(f"Actual thread pools were not constrained to one thread: {limited}")
    after = threadpoolctl.threadpool_info()
    print(json.dumps({"passed": True, "python": sys.executable, "versions": versions,
        "module_paths": {"numpy": np.__file__, "scipy": scipy.__file__,
                         "threadpoolctl": str(module_path)},
        "threadpoolctl_distribution_path": str(distribution_path),
        "before": before, "inside_limits_one": limited, "after": after}, indent=2))


if __name__ == "__main__":
    main()
