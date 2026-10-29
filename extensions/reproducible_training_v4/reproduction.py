"""Compare complete, independently launched runs with explicit provenance gates."""
from pathlib import Path
import json
import math
import os
import re
import subprocess
import sys
import time

import torch
import numpy as np

from stabilization_learning.unified_v1 import runner as files
from extensions.unified_training_v3.numerical import require_finite
from .runtime import content_hash, VOLATILE


def process_snapshot(roots):
    """Real external OS enumeration; unknown identities never imply exit."""
    if os.name != "nt":
        raise RuntimeError("This audited process snapshot implementation requires Windows PowerShell")
    script = r"""$ErrorActionPreference='Stop'
$items=@(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {$_.Name -match '^(python(?:w|[0-9.]+)?|py)\.exe$'} | Select-Object ProcessId,ParentProcessId,Name,CommandLine)
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
[ordered]@{schema='unified_training_process_snapshot';query_success=$true;created_unix=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();processes=$items} | ConvertTo-Json -Depth 5
"""
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("External process enumeration failed: " + result.stderr.decode(errors="replace"))
    snapshot = json.loads(result.stdout.decode("utf-8-sig"))
    if (snapshot.get("schema") != "unified_training_process_snapshot" or
            snapshot.get("query_success") is not True or not isinstance(snapshot.get("processes"), list)):
        raise RuntimeError("Complete external process inventory required")
    created = snapshot.get("created_unix")
    if type(created) not in (int, float) or not -5 <= time.time()-created <= 60:
        raise RuntimeError("External process inventory expired")
    processes = snapshot["processes"]
    for proc in processes:
        if (not isinstance(proc, dict) or type(proc.get("ProcessId")) is not int or proc["ProcessId"] <= 0 or
                type(proc.get("ParentProcessId")) is not int or proc["ParentProcessId"] < 0 or
                not isinstance(proc.get("Name"), str) or not re.fullmatch(r"(?:python(?:w|[0-9.]+)?|py)\.exe", proc["Name"], re.I) or
                not isinstance(proc.get("CommandLine"), str) or not proc["CommandLine"].strip()):
            raise RuntimeError("Unknown or empty Python process identity does not prove writer exit")
    ancestors, current = set(), os.getpid()
    by_pid = {p["ProcessId"]: p for p in processes}
    if len(by_pid) != len(processes) or current not in by_pid:
        raise RuntimeError("Incomplete or duplicate external Python process inventory")
    while current in by_pid and current not in ancestors:
        ancestors.add(current); current = by_pid[current]["ParentProcessId"]
    normalized = [str(Path(root).resolve()).replace("\\", "/").lower() for root in roots]
    relative = []
    for root in roots:
        try:
            relative.append(str(Path(root).resolve().relative_to(Path.cwd().resolve())).replace("\\", "/").lower())
        except ValueError:
            pass
    for proc in processes:
        command = str(proc["CommandLine"]).strip().replace("\\", "/").lower()
        is_writer = bool(re.search(r"run_unified_training_v\d+\.py[\"']?\s+(?:train|_preflight_job)\b", command))
        output_arg = re.search(r"--output(?:=|\s+)(?:\"([^\"]+)\"|'([^']+)'|(\S+))", command)
        output = next((value for value in output_arg.groups() if value is not None), "") if output_arg else ""
        # CIM does not expose another process's working directory. A relative
        # writer target with the same basename is ambiguous, so reject it.
        relative_target = (is_writer and output and not Path(output).is_absolute() and
            (output.removeprefix("./") in relative or Path(output).name in {Path(root).name.lower() for root in roots}))
        target = any(root in command for root in normalized) or relative_target
        if target and (proc["ProcessId"] not in ancestors or is_writer):
            raise RuntimeError("A target run still has a Python process: " + command)
    snapshot["scope"] = "All Python processes enumerated; reject any target-run writer; complete identities of unrelated runs retained"
    return snapshot


_EXCLUDED = VOLATILE | {"wall_s", "policy_wall_s"}
_STATE_FIELDS = {"actor", "policy", "network", "value", "metric", "features", "learner", "q1", "q2", "target1", "target2",
                 "optimizer", "actor_optimizer", "q_optimizer", "alpha_optimizer", "value_optimizer",
                 "initial", "initial_state", "initial_actor", "best_actor", "best_optimizer", "replay", "pending", "environments",
                 "rng", "torch_rng", "numpy_rng", "python_rng", "cuda_rng", "replay_rng", "collection_rng"}


class _Evidence:
    """Remember every file actually read and detect concurrent replacements."""
    def __init__(self, root):
        self.root, self.digests, self.listings = Path(root), {}, {}

    def fingerprint(self, path):
        path = Path(path).resolve()
        digest = files.sha256(path)
        previous = self.digests.setdefault(str(path), digest)
        if previous != digest:
            raise RuntimeError("Evidence changed while being compared: " + str(path))
        return digest

    def read(self, path):
        self.fingerprint(path)
        result = files.read(path)
        self.fingerprint(path)
        return result

    def listing(self, folder, pattern):
        folder = Path(folder).resolve()
        result = sorted(folder.glob(pattern))
        self.listings[(str(folder), pattern)] = [str(p) for p in result]
        return result

    def verify(self):
        if (self.root / "WRITER.lock").exists():
            raise RuntimeError("A writer appeared during comparison")
        for path in tuple(self.digests):
            self.fingerprint(path)
        for (folder, pattern), expected in self.listings.items():
            if [str(p) for p in sorted(Path(folder).glob(pattern))] != expected:
                raise RuntimeError("Evidence file set changed while being compared")


def _inside(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise RuntimeError("Own-run evidence path escapes its job: " + str(path))
    return path


def _archive(path, owner):
    owner.fingerprint(path)
    with np.load(path, allow_pickle=False) as archive:
        digest = content_hash({key: archive[key] for key in sorted(archive.files)})
    owner.fingerprint(path)
    return digest


def _stable(value, root, folder, owner, *, protected=False):
    """Normalize publication paths/timings, verifying external arrays first."""
    if isinstance(value, dict):
        result = {key: _stable(item, root, folder, owner, protected=protected or key in _STATE_FIELDS)
                  for key, item in value.items() if protected or key not in _EXCLUDED}
        if not protected and isinstance(value.get("trajectory"), str):
            ref = Path(value["trajectory"])
            path = _inside(folder, ref if ref.is_absolute() else folder / ref)
            digest = owner.fingerprint(path)
            if "trajectory_sha256" in value and value["trajectory_sha256"] != digest:
                raise RuntimeError("Referenced trajectory hash mismatch")
            result["trajectory"] = str(path.relative_to(root)).replace("\\", "/")
            result["trajectory_sha256"] = _archive(path, owner)
        return result
    if isinstance(value, (tuple, list)):
        parts = [_stable(item, root, folder, owner, protected=protected) for item in value]
        return tuple(parts) if isinstance(value, tuple) else parts
    if isinstance(value, str) and not protected:
        # Output directory names are publication details; relative paths still
        # identify which artifact was used and are not erased from comparison.
        prefix = str(root).replace("\\", "/") + "/"
        normalized = value.replace("\\", "/")
        if normalized.lower().startswith(prefix.lower()):
            return "<RUN>/" + normalized[len(prefix):]
    return value


def _expected_stages(epochs):
    result = {"initialize", "validate_initial", "collect_initial", "warmup_representation",
              "pretrain_evaluator", "validate_ready"}
    for epoch in range(1, epochs + 1):
        result.update(f"{kind}_epoch_{epoch:04d}" for kind in ("collect", "train", "validate"))
    return result


def _inventory(root, *, metadata_only=False):
    root = Path(root).resolve()
    if (root/"WRITER.lock").exists():
        raise RuntimeError("Writer lock remains: " + str(root))
    owner = _Evidence(root)
    final = owner.read(root/"FINAL_STATE.json")
    if final.get("status") != "complete" or final.get("cleanup_errors"):
        raise RuntimeError("Only complete training runs can establish reproducibility")
    protocol = owner.read(root/"protocol.json")
    if protocol.get("runner_revision") != "v4_reproducible_contracts":
        raise ValueError("Compare requires the versioned v4 lifecycle")
    jobs = [j["id"] for j in protocol["config"]["jobs"]]
    if len(jobs) != len(set(jobs)) or not jobs or set(final.get("jobs", {})) != set(jobs) or any(
            final["jobs"][job].get("status") != "complete" for job in jobs):
        raise RuntimeError("Incomplete declared job group")
    files.check_frozen(root, jobs)
    global_freeze = owner.read(root/"GLOBAL_FROZEN.json")
    if global_freeze.get("schema") != "unified_frozen_group_v1":
        raise RuntimeError("Invalid global freeze schema")
    suite_results = owner.read(root/"RESULTS.json")
    if suite_results.get("schema") != "unified_suite_results_v1" or set(suite_results.get("jobs", {})) != set(jobs):
        raise RuntimeError("Missing complete suite results")
    sources = protocol["sources"]
    for path, expected in sources.items():
        if not Path(path).is_file() or owner.fingerprint(path) != expected:
            raise RuntimeError("Frozen source/config/asset changed: " + path)
    contract = protocol["config"]["reproducibility_contract"]
    if sources.get(contract["declared_config_path"]) != contract["declared_config_sha256"]:
        raise RuntimeError("Declared configuration is not bound by the source manifest")
    declared = owner.read(contract["declared_config_path"])
    excluded = {"maximum_minutes", "absolute_deadline_unix", "initializer_preflight", "reproducibility_contract"}
    if {k: v for k, v in protocol["config"].items() if k not in excluded} != {k: v for k, v in declared.items() if k not in excluded}:
        raise RuntimeError("Actual training configuration differs from the declared configuration")
    preflight = protocol["config"]["initializer_preflight"]
    if (preflight["original_maximum_minutes"] != declared["maximum_minutes"] or
            not 0 < protocol["config"]["maximum_minutes"] <= declared["maximum_minutes"] or
            protocol["deadline_unix"] > preflight["cutoff_unix"] or
            protocol["deadline_unix"] > declared.get("absolute_deadline_unix", float("inf"))):
        raise RuntimeError("Training budget differs from its declared bound")
    preflight_path = str(Path(preflight["path"]).resolve())
    if sources.get(preflight_path) != preflight["sha256"] or owner.read(preflight_path).get("status") != "passed":
        raise RuntimeError("Missing successful immutable initializer preflight")
    # Reject an incomplete group before unpickling any model, including a
    # missing stage/checkpoint in a later job than the first one.
    for job in protocol["config"]["jobs"]:
        folder = root/"jobs"/job["id"]
        epochs = job["schedule"]["epochs"]
        expected_names = {"initial", "ready"} | {f"epoch_{e:04d}" for e in range(1, epochs+1)}
        if ({p.name for p in owner.listing(folder/"checkpoints", "*.COMMITTED.json")} !=
                {n+".COMMITTED.json" for n in expected_names} or
                {p.name for p in owner.listing(folder/"checkpoints", "*.pt")} != {n+".pt" for n in expected_names}):
            raise RuntimeError("Missing committed training checkpoints")
        if {p.stem for p in owner.listing(folder/"stages", "*.json")} != _expected_stages(epochs):
            raise RuntimeError("Missing or unexpected training/validation stages")
        test = owner.read(folder/"TEST_RESULTS.json")
        if test.get("started_after_group_freeze") is not True or test.get("group_freeze_sha256") != owner.fingerprint(root/"GLOBAL_FROZEN.json"):
            raise RuntimeError("Test was not declared after group freeze")
    if metadata_only:
        owner.verify()
        return owner
    checkpoints, evaluations, stages, selections, arrays = {}, {}, {}, {}, {}
    for job in protocol["config"]["jobs"]:
        jid = job["id"]
        folder = root/"jobs"/jid
        job_protocol = owner.read(folder/"protocol.json")
        if job_protocol.get("job") != job:
            raise RuntimeError("Actual job differs from the declared configuration")
        epochs = job["schedule"]["epochs"]
        expected_names = {"initial", "ready"} | {f"epoch_{e:04d}" for e in range(1, epochs+1)}
        names = owner.listing(folder/"checkpoints", "*.COMMITTED.json")
        if {p.name for p in names} != {n+".COMMITTED.json" for n in expected_names} or {
                p.name for p in owner.listing(folder/"checkpoints", "*.pt")} != {n+".pt" for n in expected_names}:
            raise RuntimeError("Missing committed training checkpoints")
        markers = {}
        for marker_path in names:
            marker = owner.read(marker_path)
            path = _inside(folder, folder/marker["checkpoint"])
            if path != marker_path.with_name(marker_path.name.replace(".COMMITTED.json", ".pt")) or owner.fingerprint(path) != marker["sha256"]:
                raise RuntimeError("Checkpoint integrity failed")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            owner.fingerprint(path)
            if not isinstance(payload, dict):
                raise RuntimeError("Unsupported checkpoint payload")
            # PPO uses +inf only as its explicitly uninitialized score sentinel.
            # Never allow NaN/Inf in model, optimizer, replay, or RNG contents.
            numerical = dict(payload)
            if not payload.get("history") and type(payload.get("best_score")) is float and payload["best_score"] == math.inf:
                numerical.pop("best_score")
            require_finite(numerical, "reproducibility checkpoint")
            stable = _stable(payload, root, folder, owner)
            checkpoints[jid+"/"+path.name] = dict(whole=content_hash(stable),
                fields={key: content_hash(value) for key, value in stable.items()})
            markers[marker["checkpoint"]] = marker
            # RGB archives live outside torch.save. The checkpoint's row index
            # alone does not establish equality of the training data it names.
            for row in payload.get("replay_rows", []):
                archive = _inside(folder/"replay", folder/"replay"/row["file"])
                arrays[str(archive.relative_to(root)).replace("\\", "/")] = _archive(archive, owner)
        test = owner.read(folder/"TEST_RESULTS.json")
        if test.get("started_after_group_freeze") is not True or test.get("group_freeze_sha256") != owner.fingerprint(root/"GLOBAL_FROZEN.json"):
            raise RuntimeError("Test was not declared after group freeze")
        evaluations[jid] = content_hash(_stable(test["result"], root, folder, owner))
        paths = owner.listing(folder/"stages", "*.json")
        if {p.stem for p in paths} != _expected_stages(epochs):
            raise RuntimeError("Missing or unexpected training/validation stages")
        stage_records = {p.stem: owner.read(p) for p in paths}
        if any(record.get("stage") != name for name, record in stage_records.items()):
            raise RuntimeError("Training stage identity mismatch")
        stages[jid] = {name: content_hash(_stable(record["result"], root, folder, owner)) for name, record in stage_records.items()}
        freeze = owner.read(folder/"FROZEN.json")
        if not all(freeze.get(k) == v for k, v in markers.get(freeze["checkpoint"], {}).items()) or freeze["checkpoint"] not in markers:
            raise RuntimeError("Selected freeze is not a committed checkpoint")
        selections[jid] = {key: freeze[key] for key in ("selected_epoch", "candidate_stage", "selection_score")}
        retained = suite_results["jobs"][jid]
        if (retained.get("frozen") != freeze or retained.get("test") != test["result"] or
                {item["stage"]: item for item in retained.get("stages", [])} != stage_records or
                retained.get("backend") != job["backend"]):
            raise RuntimeError("Suite summary differs from its detailed evidence")
        if job["backend"] == "state_model_gradient":
            for variant, group in test["result"]["groups"].items():
                for scale in {row["scale"] for row in group["rows"]}:
                    archive = folder/"trajectories"/"test"/variant/f"scale_{scale:g}.npz"
                    arrays[str(archive.relative_to(root)).replace("\\", "/")] = _archive(archive, owner)
    owner.verify()
    return dict(config_sha256=contract["declared_config_sha256"], runtime=protocol["runtime"],
                sources={p: h for p, h in sources.items() if p != preflight_path},
                checkpoints=checkpoints, evaluations=evaluations, stages=stages, selections=selections,
                arrays=arrays, _owner=owner)


def compare(first, second, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    first, second = Path(first).resolve(), Path(second).resolve()
    if first == second:
        raise ValueError("Reproducibility requires two distinct runs")
    snapshot = process_snapshot([first, second])
    initial_evidence = [_inventory(root, metadata_only=True) for root in (first, second)]
    a, b = _inventory(first), _inventory(second)
    differences = []
    for part in ("config_sha256", "runtime", "sources", "evaluations", "stages", "selections", "arrays"):
        if a[part] != b[part]:
            differences.append(dict(component=part, first_hash=content_hash(a[part]), second_hash=content_hash(b[part])))
    for name in sorted(set(a["checkpoints"]) | set(b["checkpoints"])):
        left, right = a["checkpoints"].get(name), b["checkpoints"].get(name)
        if left != right:
            keys = set((left or {}).get("fields", {})) | set((right or {}).get("fields", {}))
            differences.append(dict(component="checkpoint", name=name,
                differing_fields=sorted(k for k in keys if (left or {}).get("fields", {}).get(k) != (right or {}).get("fields", {}).get(k))))
    # Both OS identity and every source, marker, stage, result and archive are
    # checked again; checking just the selected checkpoint misses later writes.
    after_snapshot = process_snapshot([first, second])
    for owner in initial_evidence:
        owner.verify()
    a["_owner"].verify(); b["_owner"].verify()
    report = dict(schema="training_reproducibility_comparison_v4", passed=not differences,
                  first=str(first), second=str(second), config_sha256=a["config_sha256"],
                  jobs=len(a["evaluations"]), checkpoints=len(a["checkpoints"]), differences=differences,
                  compared=["all committed checkpoint tensor/optimizer/RNG/replay contents", "training stage metrics",
                            "validation selection", "frozen test trajectories and metrics", "runtime and source/config/asset hashes"],
                  excluded_publication_fields=sorted(_EXCLUDED), process_snapshot=snapshot,
                  process_snapshot_after_read=after_snapshot,
                  scope="Exact same-runtime engineering reproduction; not a long-run convergence or cross-device guarantee")
    files.write_new(output, report)
    if differences:
        raise RuntimeError("Reproducibility mismatch; full evidence retained in " + str(output))
    return report


def reproduce(config_path, output, *, project):
    from .runner import _run_owned_child, validate_training_config
    output, config_path, project = Path(output).resolve(), Path(config_path).resolve(), Path(project).resolve()
    config = validate_training_config(files.read(config_path))
    if output.exists():
        raise FileExistsError("Choose a new reproducibility directory")
    output.mkdir(parents=True, exist_ok=False)
    frozen = output/"DECLARED_CONFIG.json"
    files.write_new(frozen, config)
    started = time.time()
    cutoff = min(started + 2*60*config["maximum_minutes"]+60,
                 config.get("absolute_deadline_unix", float("inf")))
    files.write_new(output/"REPRODUCTION_PROTOCOL.json", dict(config_sha256=files.sha256(frozen),
        original_config=str(config_path), started_unix=started, deadline_unix=cutoff,
        runs=["run_a", "run_b"], restart_on_failure=False, scope="two independent new processes; same fixed engineering configuration"))
    for name in ("run_a", "run_b"):
        remaining = cutoff-time.time()
        if remaining <= 0 or (output/"STOP").exists():
            raise TimeoutError("Reproduction deadline or STOP")
        code, stdout, stderr = _run_owned_child([sys.executable, "-X", "utf8",
            str(project/"scripts/run_unified_training_v4.py"), "train", "--config", str(frozen),
            "--output", str(output/name)], cwd=project, timeout=min(remaining, config["maximum_minutes"]*60+15))
        (output/(name+".stdout.log")).write_text(stdout, encoding="utf-8")
        (output/(name+".stderr.log")).write_text(stderr, encoding="utf-8")
        files.write_new(output/(name+".PROCESS_EXIT.json"), dict(returncode=code, reaped_unix=time.time()))
        if code:
            raise RuntimeError("Fresh process failed; no automatic restart: " + name)
    return compare(output/"run_a", output/"run_b", output/"REPRODUCIBILITY.json")
