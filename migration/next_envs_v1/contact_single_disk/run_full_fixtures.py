"""Five fixed CPU engineering fixtures. No learner, optimizer, or tuning.

Each committed step is an atomic, hash-chained transaction containing all four
physical substeps, candidate diagnostics, and complete boundary simulator state.
Restoration assembles the committed prefix and passes the environment's existing
full loader. Completed fixtures are never executed again.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OUTPUT = HERE / "checks/full_fixtures_v1"
EXPECTED = {
    "contact_single_disk_env.py": "39dc1282df245740329c4b3598c1c2e8070d2e12bcc15de8eef4118bb5d3c1e3",
    "DESIGN_PROTOCOL.json": "43a31a8acb402223ba5f914fca2d4de08dbefa12320b2d31c716ceb88df609ce",
    "check_contact_preflight.py": "de3c3d0d4088b8e4389b0d6428d68e8b0cc4cdfd554005f1040b14eef58f2b95",
}
REVIEW = HERE / "review/SAVED_PREFLIGHT_INDEPENDENT_REVIEW_20261023_082100.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, default=ready, allow_nan=False) + "\n").encode())


def commit(path, value):
    if Path(path).exists():
        raise RuntimeError("Refusing to overwrite committed physical evidence: " + str(path))
    payload = pickle.dumps(value, protocol=5)
    envelope = {"sha256": hashlib.sha256(payload).hexdigest(), "payload": payload}
    atomic_bytes(path, gzip.compress(pickle.dumps(envelope, protocol=5), compresslevel=1, mtime=0))
    return sha(path)


def read_commit(path):
    envelope = pickle.loads(gzip.decompress(Path(path).read_bytes()))
    if hashlib.sha256(envelope["payload"]).hexdigest() != envelope["sha256"]:
        raise RuntimeError("Committed payload checksum mismatch")
    return pickle.loads(envelope["payload"])


def resource_status():
    usage = shutil.disk_usage("D:/")
    total = sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file()) if OUTPUT.exists() else 0
    command = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    if completed.returncode:
        raise RuntimeError("GPU resource gate unavailable: " + completed.stderr[-300:])
    gpu = [dict(zip(("used_mib", "total_mib"), map(float, line.split(","))))
           for line in completed.stdout.strip().splitlines()]
    if not gpu:
        raise RuntimeError("GPU resource gate returned no device")
    result = {"utc": utc(), "free_gib": usage.free / 2**30, "output_gib": total / 2**30,
              "gpu": gpu, "gpu_allocations_by_this_runner": 0}
    result["safe"] = result["free_gib"] >= 50 and result["output_gib"] <= 35 and all(
        g["used_mib"] / g["total_mib"] <= .65 for g in gpu)
    return result


def verify_sources():
    if {name: sha(HERE / name) for name in EXPECTED} != EXPECTED:
        raise RuntimeError("Local executed source/protocol mismatch; no implicit repair")
    preflight = json.loads((HERE / "checks/preflight_v1/SUMMARY.json").read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    if not preflight["passed"] or not review["passed"] or review["sources"] != EXPECTED:
        raise RuntimeError("Independent preflight gate not passed for these sources")
    frozen = json.loads((PROJECT / "runs/book_m_night_suite_20261023_v1/protocol.json").read_text(encoding="utf-8"))["sources"]
    if {p: sha(PROJECT / p) for p in frozen} != frozen:
        raise RuntimeError("Protected shared source changed")
    return {**EXPECTED, "run_full_fixtures.py": sha(__file__)}, frozen


def light_snapshot(env):
    """Same frozen state_dict schema; growing trace/log committed once per row."""
    if env._pending_plan is not None:
        raise RuntimeError("Snapshot requires a complete physical-step boundary")
    env._assert_fixed()
    return {"schema": "contact-environment-integration-boundary-v1", "engine_version": "3.2.3",
        "xml_sha256": env.xml_sha256, "source_sha256": env.source_sha256,
        "protocol_sha256": env.protocol_sha256, "integration_spec": int(env.state_spec),
        "integration": env._integration().copy(), "goal_xy": env.goal_xy.copy(),
        "context": env._context.copy(), "initial_state": env._initial_state.copy(),
        "initial_seed": env._seed_record, "rng": deepcopy(env.np_random.bit_generator.state),
        "elapsed_steps": env._elapsed_steps, "hold_steps": env._hold_steps, "done": env._done,
        "execution_mode": env._execution_mode, "origin": env._origin,
        "episode_metric": deepcopy(env._episode_metric), "in_goal_since": env._in_goal_since,
        "first_hold_confirmation": env._first_hold_confirmation, "planning_failure": env._planning_failure,
        "budget": deepcopy(env._budget), "trace": [], "plan_log": [], "last_info": deepcopy(env._last_info)}


def restore_prefix(env, folder, fixture, source_hashes):
    files = sorted(folder.glob("step_*.pkl.gz"))
    previous_hash = None
    trace, logs, snapshot = [], [], None
    for index, path in enumerate(files, 1):
        if path.name != f"step_{index:06d}.pkl.gz":
            raise RuntimeError("Gap or duplicate in committed physical prefix")
        record = read_commit(path)
        if (record["fixture"] != fixture or record["sources"] != source_hashes or
                record["previous_commit_sha256"] != previous_hash or record["step"] != index):
            raise RuntimeError("Committed fixture/source/hash-chain mismatch")
        trace.append(record["row"])
        logs.append(record["plan_log"])
        snapshot = record["boundary"]
        previous_hash = sha(path)
    if snapshot is not None:
        snapshot["trace"], snapshot["plan_log"] = trace, logs
        env.load_state_dict(snapshot)
    return previous_hash


def summarize(env, fixture):
    import numpy as np
    rows = env._trace
    states = np.array([env._initial_state] + [r["state_after"] for r in rows])
    M = np.eye(8)
    errors = np.stack([env.tensor_error(x).numpy() for x in states])
    values = np.einsum("ni,ij,nj->n", errors, M, errors)
    costs = np.array([env.physical_cost(x) for x in states])
    raw_cost = float(.02 * costs[:-1].sum())
    failure = bool(rows[-1]["failure"]) if rows else False
    tail = max(float(costs[-1]), 25.) * max(0., 30. - .02 * len(rows)) if failure else 0.
    valid = np.array([env.is_success(x) for x in states])
    final_hold = len(rows) == 1500 and bool(valid[-51:].all()) and not failure
    since = None
    if final_hold:
        since_index = len(valid) - 1
        while since_index > 0 and valid[since_index - 1]:
            since_index -= 1
        since = .02 * since_index
    commands = np.array([r["applied_command"] for r in rows])
    dv = np.diff(values)
    hdelta = values[125:] - values[:-125] if len(values) > 125 else np.array([])
    contact_count = sum(bool(s["contacts"]) for r in rows for s in r["substeps"])
    stored_cost = sum(r["cost"] for r in rows)
    if abs(stored_cost - raw_cost) > 1e-10 or final_hold != bool(env._last_info["success"]):
        raise RuntimeError("Independent full-prefix cost/hold mismatch")
    return {"fixture": fixture, "complete": env._done, "success": final_hold, "failure": failure,
        "failure_reason": env._last_info.get("failure_reason"), "real_control_steps": len(rows),
        "physical_duration_seconds": .02 * len(rows), "intended_horizon_seconds": 30.,
        "raw_physical_cost": raw_cost, "failure_tail": tail, "fixed_horizon_cost": raw_cost + tail,
        "terminal_physical_error": float(costs[-1]),
        "terminal_position_error_m": float(np.linalg.norm(states[-1, :2] - env.goal_xy)),
        "final_51_sample_hold": final_hold, "settling_time_or_30s": since if final_hold else 30.,
        "arrival_with_1s_hold_or_30s": since + 1. if final_hold else 30.,
        "V_initial": float(values[0]), "V_final": float(values[-1]),
        "V_rise_steps_tol_1e_10": int((dv > 1e-10).sum()), "V_delta_max": float(dv.max()) if len(dv) else 0.,
        "V_H125_rises_tol_1e_10": int((hdelta > 1e-10).sum()), "V_H125_comparisons": len(hdelta),
        "command_total_variation": float(np.linalg.norm(np.diff(commands, axis=0), axis=1).sum()) if len(commands) > 1 else 0.,
        "contact_physics_substeps": contact_count, "budget": deepcopy(env._budget),
        "M": M.tolist(), "learned_policy": False, "training_updates": 0, "optimizers": 0,
        "state_feedback_not_vision": True, "certificate_claim": "candidate_only"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=75.)
    args = parser.parse_args()
    if not 0 < args.max_minutes <= 75:
        raise ValueError("Safety slice must lie in (0,75] minutes")
    source_hashes, frozen = verify_sources()
    existed = (OUTPUT / "START_MANIFEST.json").exists()
    if existed and not args.resume:
        raise RuntimeError("Existing output requires explicit --resume")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "EXECUTION.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    # Handle held until process exit. A stale metadata PID cannot confer ownership.
    write_json(OUTPUT / "lock_owner.json", {"pid": os.getpid(), "utc": utc(), "single_writer": True})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    for p in (PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control",
              PROJECT / "third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages"):
        sys.path.insert(0, str(p))
    import numpy as np
    import psutil
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from contact_single_disk_env import ContactSingleDiskEnv, ContactMFacade, PROTOCOL, _digest
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    allowed = process.cpu_affinity()
    process.cpu_affinity([allowed[-1]])
    started = time.perf_counter()
    resource = resource_status()
    manifest = {"sources": source_hashes, "protected_sources": frozen, "fixtures": PROTOCOL["five_full_fixtures"],
        "M": np.eye(8).tolist(), "independent_review_sha256": sha(REVIEW),
        "review": str(REVIEW), "max_fixture_count": 5, "max_control_steps_per_fixture": 1500,
        "max_prediction_substeps_per_query": 9000, "optimizer_count": 0, "learned_policy": False,
        "cpu_affinity": process.cpu_affinity(), "torch_threads": 1, "gpu_allocations": 0,
        "fixture_horizon_seconds": 30., "safe_slice_minutes": 75.,
        "source_signature_limitation": "Frozen facade cached signature predates description replacement; actual description digest recorded separately. No training acceptance claimed."}
    if existed:
        old = json.loads((OUTPUT / "START_MANIFEST.json").read_text(encoding="utf-8"))
        for key in ("sources", "protected_sources", "fixtures", "M", "independent_review_sha256"):
            if old[key] != manifest[key]:
                raise RuntimeError("Resume protocol changed: " + key)
    else:
        write_json(OUTPUT / "START_MANIFEST.json", {**manifest, "utc": utc(), "initial_resource": resource})
    summaries = []
    env = None
    status = {"phase": "initializing", "pid": os.getpid(), "utc": utc(), "resource": resource,
              "completed_fixtures": [], "training_updates": 0, "optimizers": 0}
    write_json(OUTPUT / "status.json", status)
    try:
      with threadpool_limits(limits=1):
        write_json(OUTPUT / "THREADS.json", {"threadpools": threadpool_info(), "affinity": process.cpu_affinity()})
        if not resource["safe"]:
            status.update(phase="safe_stopped", reason="resource_gate")
            return
        for fi, fixture in enumerate(PROTOCOL["five_full_fixtures"]):
            folder = OUTPUT / f"{fi+1:02d}_{fixture['name']}"
            folder.mkdir(exist_ok=True)
            env = ContactSingleDiskEnv()
            env.reset(seed=41, options={k: v for k, v in fixture.items() if k != "name"})
            previous_hash = restore_prefix(env, folder, fixture, source_hashes)
            if (folder / "COMPLETE.json").exists():
                completed = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
                if (not env._done or completed.get("last_commit_sha256") != previous_hash or
                        completed.get("fixture") != fixture or completed.get("sources") != source_hashes):
                    raise RuntimeError("Completed fixture has missing/changed transactions; refusing any re-execution")
            facade = ContactMFacade(env)
            if not (folder / "INTERFACE.json").exists():
                desc = facade.describe().to_dict()
                write_json(folder / "INTERFACE.json", {"description": desc, "actual_description_sha256": _digest(desc)})
            status.update(phase="running", fixture=fixture["name"], fixture_index=fi, steps=env._elapsed_steps,
                          completed_fixtures=[s["fixture"]["name"] for s in summaries])
            write_json(OUTPUT / "status.json", {**status, "utc": utc()})
            last_gate = time.perf_counter()
            while not env._done:
                now = time.perf_counter()
                if now - started >= args.max_minutes * 60 or (OUTPUT / "STOP").exists():
                    status.update(phase="safe_stopped", reason="time_slice" if not (OUTPUT / "STOP").exists() else "user_STOP")
                    return
                if now - last_gate >= 30:
                    resource = resource_status(); last_gate = now
                    status["resource"] = resource
                    if verify_sources() != (source_hashes, frozen):
                        raise RuntimeError("Runner or protected source changed after launch")
                    if not resource["safe"]:
                        status.update(phase="safe_stopped", reason="resource_gate")
                        return
                if env._elapsed_steps >= 1500:
                    raise RuntimeError("Physical step cap exceeded")
                error, context = env.build_error(env.state)
                command = facade.calculate_action_from_M(torch.eye(8, dtype=torch.float64), error, context=context)
                if env._pending_plan["budget"]["prediction_substeps"] > 9000:
                    raise RuntimeError("Per-query prediction cap exceeded")
                env.step(command.numpy())
                index = env._elapsed_steps
                record = {"fixture": fixture, "sources": source_hashes, "step": index,
                    "previous_commit_sha256": previous_hash, "row": env._trace[-1],
                    "plan_log": env._plan_log[-1], "boundary": light_snapshot(env)}
                previous_hash = commit(folder / f"step_{index:06d}.pkl.gz", record)
                status.update(steps=index, last_commit_sha256=previous_hash, budget=deepcopy(env._budget))
                if index % 25 == 0 or env._done:
                    write_json(OUTPUT / "status.json", {**status, "utc": utc()})
                if index % 100 == 0 or env._done:
                    print(json.dumps({"utc": utc(), "fixture": fixture["name"], "step": index,
                                      "physical_error": env.physical_cost(env.state), "wall_seconds": now-started}), flush=True)
            summary = summarize(env, fixture)
            summary.update(last_commit_sha256=previous_hash, sources=source_hashes)
            if (folder / "COMPLETE.json").exists():
                if json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8")) != summary:
                    raise RuntimeError("Previously complete fixture summary differs")
            else:
                write_json(folder / "COMPLETE.json", summary)
            summaries.append(summary)
            write_json(OUTPUT / "RESULTS.json", {"fixtures": summaries, "complete": len(summaries) == 5,
                       "training_updates": 0, "optimizers": 0, "M": "identity", "utc": utc()})
        if verify_sources() != (source_hashes, frozen):
            raise RuntimeError("Runner or protected source changed after launch")
        status.update(phase="complete", completed_fixtures=[s["fixture"]["name"] for s in summaries],
                      successful_fixtures=sum(s["success"] for s in summaries),
                      real_control_steps_total=sum(s["real_control_steps"] for s in summaries),
                      budget_total={k: sum(s["budget"][k] for s in summaries) for k in summaries[0]["budget"]})
        write_json(OUTPUT / "COMPLETE.json", {**status, "utc": utc(), "sources": source_hashes})
    except BaseException as exc:
        status.update(phase="error", error=str(exc), traceback=traceback.format_exc(),
                      recovery_note="Only committed control boundaries are replayable; do not bypass source checks.")
        write_json(OUTPUT / ("ERROR_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"), status)
        raise
    finally:
        write_json(OUTPUT / "status.json", {**status, "utc": utc(), "process_wall_seconds": time.perf_counter()-started})
        lock.close()


if __name__ == "__main__":
    main()
