"""Stage and audit six book teaching-example groups without touching the book.

Default mode is read-only inspection. Numerical execution requires --mode run.
Every prepare/run requires a fresh directory under this project's runs/.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n", encoding="utf-8")


def source_path(source, roots):
    root = roots[source["root"]].resolve()
    candidate = (root/source["path"]).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Source escapes its declared root: {source['path']}")
    return candidate


def inspect_source(source, roots):
    path = source_path(source, roots)
    row = {"path": str(path), "expected_sha256": source["sha256"],
           "exists": path.is_file(), "passed": False}
    if not path.is_file():
        return row
    row["sha256"] = sha(path)
    row["bytes"] = path.stat().st_size
    row["hash_matches"] = row["sha256"] == source["sha256"]
    if path.suffix == ".py":
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            row["syntax_valid"] = True
        except SyntaxError as error:
            row["syntax_valid"] = False
            row["syntax_error"] = str(error)
    else:
        row["syntax_valid"] = None
    row["passed"] = row["hash_matches"] and row["syntax_valid"] is not False
    return row


def inspect_manifest(manifest, selected, roots):
    packages = []
    for name, expected in manifest["dependency_versions"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages.append({"package": name, "expected": expected, "actual": actual,
                         "passed": actual == expected})
    runner = [inspect_source(row, roots) for row in manifest["runner_files"]]
    groups = []
    for group in selected:
        executable = [inspect_source(row, roots) for row in group["sources"]]
        evidence = [inspect_source(row, roots) for row in group.get("evidence_sources", [])]
        groups.append({"id": group["id"], "title": group["title"],
                       "implemented": group["implemented"],
                       "scope": group["scope"], "sources": executable,
                       "evidence_sources": evidence,
                       "missing_items": group.get("missing_items", []),
                       "ready_to_run": group["implemented"] and all(x["passed"] for x in executable),
                       "source_checks_passed": all(x["passed"] for x in executable+evidence)})
    passed = (all(row["passed"] for row in packages+runner)
              and all(group["source_checks_passed"] for group in groups))
    return {"schema": "book-numerical-inspection-v1", "timestamp_utc": now(),
            "mode": "static_only_no_numerical_execution", "python": sys.executable,
            "python_version": sys.version, "dependency_versions": packages,
            "runner_files": runner, "groups": groups,
            "static_checks_passed": passed,
            "implemented_group_count": sum(g["implemented"] for g in selected),
            "requested_group_count": len(selected)}


def now():
    return datetime.now(timezone.utc).isoformat()


def new_output(path):
    runs_root = (PROJECT/"runs").resolve()
    output = Path(path)
    if not output.is_absolute():
        output = PROJECT/output
    output = output.resolve()
    if output == runs_root or not output.is_relative_to(runs_root):
        raise ValueError("Output must be a new subdirectory of D:/stabilization-learning/runs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite or resume this lightweight suite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def stage(group, output, roots):
    workspace = output/group["id"]/"workspace"
    workspace.mkdir(parents=True)
    copied = []
    for source in group["sources"]:
        original = source_path(source, roots)
        destination = (workspace/source["stage_as"]).resolve()
        if not destination.is_relative_to(workspace.resolve()):
            raise ValueError("Staged path escapes workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        actual = sha(destination)
        if actual != source["sha256"]:
            raise RuntimeError(f"Staged source failed hash validation: {destination}")
        copied.append({"source": str(original), "copy": str(destination), "sha256": actual})
    return workspace, copied


def output_inventory(directory):
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {"SUITE_RESULTS.json", "artifact_manifest.json"}:
            result.append({"path": path.relative_to(directory).as_posix(),
                           "bytes": path.stat().st_size, "sha256": sha(path)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "prepare", "run"), default="inspect")
    parser.add_argument("--manifest", type=Path, default=HERE/"manifest.json")
    parser.add_argument("--book-root", type=Path)
    parser.add_argument("--groups", nargs="+", help="Default: all six groups; unknown IDs are rejected")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600,
                        help="Per source process; timeout evidence is retained")
    args = parser.parse_args()
    if args.mode == "inspect" and args.output is not None:
        parser.error("inspect is read-only; use prepare to save and stage a static inspection")
    if args.mode != "inspect" and args.output is None:
        parser.error("prepare/run requires --output pointing to a new runs subdirectory")
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    roots = {"book": (args.book_root or Path(manifest["book_root"])), "project": PROJECT}
    by_id = {g["id"]: g for g in manifest["groups"]}
    requested = args.groups or list(by_id)
    if len(requested) != len(set(requested)) or any(x not in by_id for x in requested):
        parser.error(f"Use unique group IDs from {list(by_id)}")
    selected = [by_id[name] for name in requested]
    inspection = inspect_manifest(manifest, selected, roots)
    inspection["manifest_path"] = str(args.manifest.resolve())
    inspection["manifest_sha256"] = sha(args.manifest)
    if args.mode == "inspect":
        print(json.dumps(inspection, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if inspection["static_checks_passed"] else 1
    output = new_output(args.output)
    save_json(output/"requested_manifest.json", manifest)
    save_json(output/"STATIC_INSPECTION.json", inspection)
    if not inspection["static_checks_passed"]:
        save_json(output/"SUITE_RESULTS.json", {
            "status": "blocked_static_checks", "numerical_execution_started": False,
            "inspection": "STATIC_INSPECTION.json"})
        print(f"Static checks failed; no numerical process started. Evidence: {output}")
        return 1
    staged = {}
    for group in selected:
        if group["implemented"]:
            workspace, copies = stage(group, output, roots)
            staged[group["id"]] = {"workspace": str(workspace), "files": copies}
    save_json(output/"staging_manifest.json", staged)
    if args.mode == "prepare":
        save_json(output/"SUITE_RESULTS.json", {
            "status": "prepared_not_executed", "numerical_execution_started": False,
            "selected_groups": requested, "static_checks_passed": True,
            "instruction": "Numerical execution requires an explicit --mode run with a fresh output directory."})
        save_json(output/"artifact_manifest.json", output_inventory(output))
        print(f"Prepared only; no numerical process started: {output}")
        return 0
    # Import numerical validators only after explicit run mode and all static guards.
    from audit_results import audit
    results = []
    for group in selected:
        started = time.monotonic()
        entry = {"id": group["id"], "title": group["title"], "scope": group["scope"],
                 "missing_items": group.get("missing_items", []),
                 "full_book_task_complete": False}
        if not group["implemented"]:
            entry.update(status="blocked_missing_implementation", numerical_execution_started=False)
            results.append(entry)
            continue
        workspace = Path(staged[group["id"]]["workspace"])
        group_output = workspace.parent
        stdout_paths = []
        processes = []
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "NUMEXPR_NUM_THREADS": "1", "PYTHONUTF8": "1",
                            "PYTHONDONTWRITEBYTECODE": "1",
                            "MPLCONFIGDIR": str(group_output/"mpl_cache")})
        try:
            for index, script in enumerate(group["commands"]):
                stdout_path = group_output/f"{index+1:02d}_{Path(script).stem}.stdout.log"
                stderr_path = group_output/f"{index+1:02d}_{Path(script).stem}.stderr.log"
                command = [sys.executable, "-B", str(workspace/script)]
                process = {"command": command, "cwd": str(workspace),
                           "stdout": str(stdout_path), "stderr": str(stderr_path),
                           "started_utc": now()}
                processes.append(process)
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    completed = subprocess.run(command, cwd=workspace, env=environment,
                        stdout=stdout, stderr=stderr, timeout=args.timeout_seconds,
                        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0), check=False)
                process["exit_code"] = completed.returncode
                if completed.returncode != 0:
                    raise RuntimeError(f"Source process exited with code {completed.returncode}: {script}")
                stdout_paths.append(stdout_path)
            checked = audit(group["id"], workspace, stdout_paths)
            save_json(group_output/"AUDIT.json", checked)
            entry.update(status="passed_implemented_scope" if checked["passed"] else "failed_audit",
                         numerical_execution_started=True, audit="AUDIT.json",
                         check_count=checked["check_count"])
        except Exception as error:
            entry.update(status="failed_execution", numerical_execution_started=True,
                         error={"type": type(error).__name__, "message": str(error)})
        entry["source_processes"] = processes
        entry["elapsed_seconds"] = time.monotonic()-started
        results.append(entry)
        save_json(group_output/"GROUP_RESULT.json", entry)
        print(f"{group['id']}: {entry['status']}", flush=True)
    # Rehash original sources after the subprocesses. No test may modify the book.
    after = inspect_manifest(manifest, selected, roots)
    save_json(output/"POST_SOURCE_INSPECTION.json", after)
    failed = any(r["status"].startswith("failed") for r in results) or not after["static_checks_passed"]
    blocked = any(r["status"].startswith("blocked") for r in results)
    status = "failed" if failed else "incomplete_missing_implementation" if blocked else "passed_implemented_scopes"
    summary = {"schema": "book-numerical-results-v1", "completed_utc": now(), "status": status,
               "numerical_execution_started": True, "groups": results,
               "original_sources_unchanged": after["static_checks_passed"],
               "all_book_tasks_complete": False,
               "limitations": manifest["limitations"],
               "manifest_sha256": inspection["manifest_sha256"]}
    save_json(output/"SUITE_RESULTS.json", summary)
    save_json(output/"artifact_manifest.json", output_inventory(output))
    print(f"Suite: {status}; evidence: {output}")
    return 1 if failed else 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
