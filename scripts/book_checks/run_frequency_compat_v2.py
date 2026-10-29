"""Run only the original frequency check with a hash-pinned isolated dependency.

No mathematical source, tolerance, original runner, or shared package is edited.
The old failed output remains immutable. A fresh --output directory is required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def run_json(command, environment):
    completed = subprocess.run(command, env=environment, cwd=PROJECT,
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if completed.returncode:
        raise RuntimeError(f"Preflight failed: {completed.stderr}")
    return json.loads(completed.stdout)


def shared_snapshot(environment):
    program = """import hashlib,importlib.metadata,importlib.util,json,pathlib
dist=importlib.metadata.distribution('threadpoolctl')
module=pathlib.Path(importlib.util.find_spec('threadpoolctl').origin)
paths=[module]+[pathlib.Path(dist.locate_file(p)) for p in dist.files if str(p).endswith(('METADATA','RECORD','WHEEL'))]
print(json.dumps({'version':dist.version,'module':str(module),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}}))
"""
    return run_json([sys.executable, "-B", "-c", program], environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((PROJECT / "runs").resolve()):
        raise ValueError("Output must be a fresh directory below project runs")
    manifest_path = HERE / "manifest_frequency_compat_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repair = manifest["compatibility_repair"]
    original_path = PROJECT / repair["original_manifest"]
    if sha(original_path) != repair["original_manifest_sha256"]:
        raise RuntimeError("Original manifest changed")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if manifest["groups"] != original["groups"]:
        raise RuntimeError("Mathematical groups/conditions must remain unchanged")
    expected_deps = dict(original["dependency_versions"], threadpoolctl="3.6.0")
    if manifest["dependency_versions"] != expected_deps:
        raise RuntimeError("Only threadpoolctl may differ from original dependencies")
    dependency_root = PROJECT / repair["dependency_root"]
    lock_path = dependency_root / "dependency_lock.json"
    if sha(lock_path) != repair["dependency_lock_sha256"]:
        raise RuntimeError("Dependency lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha(dependency_root / lock["wheel_filename"]) != lock["wheel_sha256"]:
        raise RuntimeError("Wheel hash mismatch")
    for row in lock["installed_files"]:
        if sha(dependency_root / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Installed dependency hash mismatch: {row['path']}")
    for row in manifest["runner_files"]:
        if row["root"] != "project" or sha(PROJECT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Runner hash mismatch: {row['path']}")
    failed_evidence = {row["path"]: sha(PROJECT / row["path"])
                       for row in repair["v1_failure_evidence"]}
    if failed_evidence != {row["path"]: row["sha256"] for row in repair["v1_failure_evidence"]}:
        raise RuntimeError("Historical failure evidence changed")
    clean = os.environ.copy()
    clean.pop("PYTHONPATH", None)
    clean.pop("PYTHONVERBOSE", None)
    clean.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    shared_before = shared_snapshot(clean)
    if shared_before["version"] != "2.2.0":
        raise RuntimeError("Shared environment no longer matches original package snapshot")
    environment = clean.copy()
    target = Path(lock["target"]).resolve()
    environment.update(PYTHONPATH=str(target), CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    preflight_command = [sys.executable, "-B", str(HERE / "frequency_dependency_preflight_v2.py"), str(target)]
    preflight = run_json(preflight_command, environment)
    # Import tracing records the actual origin in BOTH unchanged source processes.
    # It only adds diagnostic stderr, and does not patch imports or numerical code.
    environment["PYTHONVERBOSE"] = "1"
    command = [sys.executable, "-B", str(HERE / "run_book_checks.py"), "--mode", "run",
        "--manifest", str(manifest_path), "--groups", "frequency_energy", "--output", str(output),
        "--timeout-seconds", str(args.timeout_seconds)]
    completed = subprocess.run(command, env=environment, cwd=PROJECT, capture_output=True,
        text=True, encoding="utf-8", timeout=2 * args.timeout_seconds + 120, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if not output.is_dir():
        raise RuntimeError(f"Original runner did not create output: {completed.stderr}")
    (output / "COMPAT_RUNNER.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "COMPAT_RUNNER.stderr.log").write_text(completed.stderr, encoding="utf-8")
    save(output / "DEPENDENCY_PREFLIGHT.json", preflight)
    save(output / "ISOLATED_DEPENDENCY_LOCK.json", lock)
    shared_after = shared_snapshot(clean)
    import_origins = []
    group = output / "frequency_energy"
    for name in ("01_compute_example_Q.stderr.log", "02_compute_example_controller.stderr.log"):
        path = group / name
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        origins = [line for line in lines if "threadpoolctl.py" in line and "code object from" in line]
        expected_module = str(target / "threadpoolctl.py").replace("\\", "/").lower()
        matches = bool(origins) and all(expected_module in line.replace("\\", "/").lower() for line in origins)
        import_origins.append({"stderr": str(path), "origin_lines": origins, "passed": matches})
    history_unchanged = all(sha(PROJECT / path) == value for path, value in failed_evidence.items())
    installed_unchanged = all(sha(dependency_root / row["path"]) == row["sha256"] for row in lock["installed_files"])
    passed = (completed.returncode == 0 and preflight["passed"] and shared_before == shared_after
              and history_unchanged and installed_unchanged and all(row["passed"] for row in import_origins))
    record = {"schema": "book-frequency-compatibility-execution-v2",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
        "command": command, "exit_code": completed.returncode, "preflight_command": preflight_command,
        "manifest_sha256": sha(manifest_path), "dependency_lock_sha256": sha(lock_path),
        "process_environment_overrides": {key: environment[key] for key in
            ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONVERBOSE",
             "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "source_process_imports": import_origins, "shared_before": shared_before, "shared_after": shared_after,
        "shared_package_unchanged": shared_before == shared_after, "historical_failure_unchanged": history_unchanged,
        "isolated_package_unchanged": installed_unchanged, "all_book_tasks_complete": False,
        "scope": "Only frequency_energy retry; original equations, sources, audit and tolerances unchanged."}
    save(output / "COMPATIBILITY_REPAIR.json", record)
    save(output / "compatibility_artifact_manifest.json", [
        {"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "compatibility_artifact_manifest.json"])
    print(completed.stdout, end="")
    print(json.dumps({"compatibility_passed": passed, "output": str(output),
                      "source_process_imports_verified": all(row["passed"] for row in import_origins)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
