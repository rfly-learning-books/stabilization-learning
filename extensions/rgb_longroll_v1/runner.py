"""Single-job, source-frozen long-trajectory RGB run; no automatic resume."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from stabilization_learning.unified_v1 import runner as v1
from extensions.unified_training_v2.runner import visual_source_inventory


def validate(config):
    v1.validate_config(config)
    if len(config["jobs"]) != 1 or config["jobs"][0]["backend"] != "rgb_legacy":
        raise ValueError("Declare precisely one RGB job")
    job = config["jobs"][0]
    cfg = job["config"]
    if cfg.get("case_manifest"):
        raise ValueError("Formal runs must use the fixed reference evaluation starts")
    expected = dict(warmup_representation=1000, pretrain_evaluator=1000, epochs=24,
                    representation_updates=300, evaluator_updates=0, certificate_updates=200)
    if job["schedule"] != expected:
        raise ValueError("Preserve the full reference update budget")
    if cfg["seed"] != 41 or cfg["batch_size"] != 4 or cfg["collection_steps"] != 120:
        raise ValueError("This experiment fixes seed41/batch4/120-step regular collection")
    if cfg["initial_collection_steps"] != 24 or cfg["parallel_envs"] != 16:
        raise ValueError("Preserve initial collection and maximum GPU parallelism")
    if len(cfg["regular_collection_quotas"]) != 24 or cfg["post_success_steps"] != 50:
        raise ValueError("Declare all 24 collection budgets and 5-second retention")
    if cfg["evaluation_steps"] != 200 or cfg["validation_cases"] != 48 or cfg["test_cases"] != 72:
        raise ValueError("Preserve reference evaluation conditions")
    required = {"selected", "initial", "identity", "shadow", "historical_trained_reference",
                "historical_trained_identity", "previous_short_selected", "previous_short_identity"}
    if set(job["test_variants"]) != required:
        raise ValueError("Declare all frozen reference tests")
    if not isinstance(config.get("absolute_deadline_unix"), (float, int)):
        raise ValueError("An absolute deadline is required")
    for value in config["comparative_sources"].values():
        if not Path(value).is_absolute() or not Path(value).is_file():
            raise FileNotFoundError(value)


def run_suite(config, root, *, project, config_path):
    validate(config)
    project = Path(project).resolve()
    original_inventory = v1.source_inventory
    effective = json.loads(json.dumps(config))
    def inventory(root_project):
        sources = original_inventory(root_project)
        sources.update(visual_source_inventory(root_project))
        paths = list((project/"extensions/rgb_longroll_v1").glob("*.py"))
        paths += [project/"scripts/run_rgb_longroll_v1.py", Path(config_path),
                  project/"extensions/unified_training_v2/historical_rgb.py", project/"extensions/unified_training_v2/runner.py"]
        paths += [Path(path) for path in config["comparative_sources"].values()]
        for path in paths:
            sources[str(path.resolve())] = v1.sha256(path)
        seconds = config["absolute_deadline_unix"]-time.time()-10
        if seconds < 60:
            raise TimeoutError("Insufficient time before absolute deadline")
        effective["maximum_minutes"] = min(config["maximum_minutes"], seconds/60)
        return sources
    def factory(name, cfg, output, *, event, deadline):
        from .backend import LongRollRGBBackend
        return LongRollRGBBackend(cfg, output, event=event, deadline=min(deadline, config["absolute_deadline_unix"]))
    v1.source_inventory = inventory
    try:
        return v1.run_suite(effective, root, project=project, factory=factory)
    finally:
        v1.source_inventory = original_inventory


def supervise(config_path, root, *, project):
    """Only the spawned child tree may be stopped at the declared deadline."""
    config = v1.read(config_path)
    validate(config)
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("A new run directory is required")
    if time.time() >= config["absolute_deadline_unix"]:
        raise TimeoutError("Declared deadline has expired")
    command = [sys.executable, "-X", "utf8", "-u", str(Path(project)/"scripts/run_rgb_longroll_v1.py"),
               "worker", "--config", str(Path(config_path).resolve()), "--output", str(root)]
    started = time.time()
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(json.dumps(dict(event="supervisor_started", pid=os.getpid(), child_pid=child.pid,
                          root=str(root), deadline_unix=config["absolute_deadline_unix"])), flush=True)
    stop_seen = None
    forced = False
    while child.poll() is None:
        now = time.time()
        stopping = (root/"STOP").exists() or now >= config["absolute_deadline_unix"]
        if stopping and stop_seen is None:
            stop_seen = now
        if stop_seen is not None and now-stop_seen >= 15:
            # taskkill's /T is scoped to this still-live, directly spawned child.
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL, capture_output=True, timeout=15,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
            else:
                child.terminate()
            forced = True
            break
        time.sleep(1)
    code = child.wait(timeout=20)
    result = dict(schema="rgb_longroll_supervisor_v1", parent_pid=os.getpid(), child_pid=child.pid,
                  started_unix=started, finished_unix=time.time(), exit_code=code,
                  forced_child_tree_stop=forced, automatic_resume=False)
    if root.exists():
        v1.write_new(root/"SUPERVISOR_FINAL.json", result)
    print(json.dumps(result), flush=True)
    return result


def report_suite(root, output, snapshot):
    from stabilization_learning.unified_v1.reporting import report_suite as base_report
    result = base_report(root, output, snapshot)
    root, output = Path(root), Path(output)
    protocol = v1.read(root/"protocol.json")
    cfg = protocol["config"]["jobs"][0]["config"]
    folder = root/"jobs"/protocol["config"]["jobs"][0]["id"]
    same = v1.read(folder/"RGB_CASES.json") == v1.read(cfg["previous_cases"])
    extras = dict(same_previous_short_cases=same,
                  intervention="120-step trajectories + up to 50 post-success steps + alive-only triplets",
                  exact_per_room_transition_budget=True, single_seed=41,
                  previous_short_reference_selection="previous run frozen validation-selected epoch12; no new selection")
    retention = folder/"RGB_RETENTION_TEST/RESULTS.json"
    if retention.exists():
        import numpy as np
        values = []
        for row in v1.read(retention)["rows"]:
            path = Path(row["trajectory"])
            if v1.sha256(path) != row["trajectory_sha256"]:
                raise RuntimeError("Retention trajectory hash changed")
            with np.load(path, allow_pickle=False) as a:
                e, m = a["error"].astype(float), a["M"].astype(float)
                actual = .5*np.einsum("bi,ij,bj->b", e, m, e)
                if not np.allclose(actual, a["V"], atol=2e-5, rtol=1e-5):
                    raise AssertionError("Independent retention V recomputation failed")
                values.append(float(np.max(abs(actual-a["V"]))))
        extras["retention_independent_V"] = dict(trajectories=len(values), maximum_absolute_difference=max(values) if values else None, passed=bool(values))
    v1.write_new(output/"LONGROLL_AUDIT.json", extras)
    return dict(base=result, longroll=extras)
