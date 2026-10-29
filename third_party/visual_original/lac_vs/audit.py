from __future__ import annotations

import importlib.metadata
import io
import platform
import subprocess
import sys
import time
import unittest
from pathlib import Path

import torch

from .config import ROOT, json_write


def run_audit(output):
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"passed": result.wasSuccessful(), "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "details": stream.getvalue(),
              "python": sys.executable, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "seconds": time.perf_counter() - started,
              "scope": "Math/gradient/data-contract audit; no task-performance or full-state stability claim"}
    json_write(output, report)
    print(stream.getvalue())
    return report


def doctor(cfg, output):
    from . import legacy
    versions = {}
    for name in ("torch", "torchvision", "mujoco", "numpy", "opencv-python", "scipy", "PyYAML", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"python": sys.executable, "platform": platform.platform(), "packages": versions,
              "cuda_available": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "dependencies": legacy.dependency_hashes(cfg)}
    json_write(output, report)
    return report


def git_cache_cleanup(roots, output, apply=False):
    """Untrack only generated ignored files, never delete working files/history.

    A root must itself be the Git top level; nested third-party repositories are
    never traversed. No git clean/reset/gc or recursive filesystem deletion.
    """
    import shutil
    rows = []
    for root in roots:
        root = Path(root).resolve()
        probe = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if probe.returncode:
            rows.append({"root": str(root), "status": "no_git_repository", "untracked_from_index": []})
            continue
        top = Path(probe.stdout.strip()).resolve()
        if top != root:
            rows.append({"root": str(root), "status": "parent_repository_not_touched", "untracked_from_index": []})
            continue
        query = subprocess.run(["git", "-C", str(root), "ls-files", "-ci", "--exclude-standard", "-z"],
                               capture_output=True, check=True)
        names = [name.decode("utf-8") for name in query.stdout.split(b"\0") if name]
        def generated(name):
            path = Path(name)
            return ("__pycache__" in path.parts or ".pytest_cache" in path.parts or path.parts[0] in ("runs", "runtime")
                    or path.suffix.lower() in (".pyc", ".pyo", ".npz", ".npy", ".pt", ".pth", ".mp4", ".avi", ".log", ".tmp"))
        names = [name for name in names if generated(name)]
        row = {"root": str(root), "status": "nothing_to_untrack" if not names else "dry_run", "candidates": names,
               "untracked_from_index": []}
        if names and apply:
            backup = Path(output).resolve().parent / f"git_index_backup_{root.name}_{time.time_ns()}"
            index = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-path", "index"], capture_output=True, text=True, check=True).stdout.strip()
            index = Path(index) if Path(index).is_absolute() else root / index
            if index.exists():
                shutil.copy2(index, backup)
                row["index_backup"] = str(backup)
            for start in range(0, len(names), 100):
                subprocess.run(["git", "-C", str(root), "rm", "--cached", "--ignore-unmatch", "--", *names[start:start+100]],
                               capture_output=True, check=True)
            row.update(status="untracked_only_working_files_preserved", untracked_from_index=names)
        rows.append(row)
    result = {"applied": apply, "roots": rows, "physical_files_deleted": 0, "git_history_changed": False,
              "third_party_repositories_touched": False}
    json_write(output, result)
    return result
