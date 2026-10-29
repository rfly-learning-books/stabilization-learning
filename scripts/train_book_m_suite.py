"""Serial, resumable CPU benchmarks through the shared M-learning task interface.

This orchestrates the existing episode-M PPO without changing its mathematics.
The complete training batch is the recovery boundary: a crash before committing
last.pt replays that batch with saved RNG; uncommitted logs never count as data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "stabilization-gym", ROOT / "stabilization-learning-control", ROOT / "scripts"):
    sys.path.insert(0, str(folder))

import numpy as np
import torch

from stabilization_gym.interfaces.m_learning import PortableMFacade
from stabilization_learning.trainer import StabilizationLearner, TrainConfig, module_hash, atomic_save
from stabilization_learning.wrappers import rollout_metric
from m_learning_suite.task_runtime import make_task, audit_task, independent_physical_step


class SafeStop(RuntimeError):
    """Interrupt an uncommitted batch; last.pt remains the replay boundary."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class WriterLock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.file.close()
            raise RuntimeError("Another suite writer holds the lock")
        return self

    def __exit__(self, *args):
        import msvcrt
        self.file.seek(0)
        msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        self.file.close()


def source_manifest(config):
    paths = [Path(__file__).resolve(), Path(config).resolve(), ROOT / "scripts/start_book_m_suite.ps1",
             ROOT / "scripts/audit_book_m_results.py"]
    for folder in (ROOT / "stabilization-gym/stabilization_gym", ROOT / "stabilization-learning-control/stabilization_learning", ROOT / "scripts/m_learning_suite"):
        paths.extend(folder.rglob("*.py"))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(set(paths))}


def validate_config(cfg):
    if len(set(cfg["seeds"])) != len(cfg["seeds"]) or not cfg["seeds"]:
        raise ValueError("Training seeds must be nonempty and unique")
    ids = [task["id"] for task in cfg["tasks"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Task IDs must be nonempty and unique")
    if any(not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name) for name in ids):
        raise ValueError("Task IDs must be simple lowercase directory names")
    count = cfg["test_seed_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Positive integer test_seed_count required")
    validation = set(cfg["training"]["validation_seeds"])
    testing = set(range(cfg["test_seed_start"], cfg["test_seed_start"]+count))
    if validation & testing:
        raise ValueError("Validation and test seeds overlap")
    for seed in cfg["seeds"]:
        training = set(range(seed, seed+cfg["training"]["episodes"]))
        if training & (validation | testing):
            raise ValueError("Training seeds overlap validation/test seeds")
        config_for(cfg, seed)
    factors = cfg["test_initial_scale_factors"]
    if not factors or len(set(factors)) != len(factors) or any(not math.isfinite(x) or x <= 0 for x in factors):
        raise ValueError("Initial-scale factors must be positive, finite and unique")
    safety = cfg["safety"]
    if safety["minimum_free_gib"] < 50 or not 1 < safety["maximum_run_gib"] <= 35 or not 0 < safety["maximum_gpu_fraction"] <= .65:
        raise ValueError("Safety limits may not exceed the authorized disk/GPU envelope")


def preparation_gate(cfg, config_path):
    path = ROOT / cfg["preparation_audit"]
    if not path.is_file():
        raise RuntimeError("Awaiting independent suite engineering audit: " + str(path))
    audit = read(path)
    if audit.get("passed") is not True or audit.get("scope") != "engineering_only_no_formal_performance_claim":
        raise RuntimeError("Suite engineering audit did not pass")
    if audit.get("source_sha256") != source_manifest(config_path):
        raise RuntimeError("Suite engineering audit does not match current source/config hashes")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def prerequisites(cfg):
    gate = ROOT / cfg["depends_on"]
    if not gate.exists() or read(gate).get("passed") is not True:
        raise RuntimeError("Awaiting independent final visual audit: " + str(gate))
    # The audit alone is not a substitute for the full original final run.
    if not (gate.parent / "complete.json").exists():
        raise RuntimeError("Visual final evaluation is not complete")


def resources(output, cfg):
    free = shutil.disk_usage(ROOT).free / 1024**3
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) / 1024**3
    safe = cfg["safety"]
    snapshot = {"free_gib": free, "run_gib": size, "device": "cpu", "gpu_memory_fractions": []}
    try:
        queried = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=5,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0), check=False)
        if queried.returncode == 0:
            for line in queried.stdout.strip().splitlines():
                used, total = map(float, line.split(","))
                snapshot["gpu_memory_fractions"].append(used/total)
        else:
            snapshot["gpu_query"] = "unavailable; this suite uses CPU only"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        snapshot["gpu_query"] = "unavailable; this suite uses CPU only"
    if free < safe["minimum_free_gib"] + 1 or size > safe["maximum_run_gib"] - 1:
        return "resource_margin", snapshot
    if any(x > safe["maximum_gpu_fraction"] for x in snapshot["gpu_memory_fractions"]):
        return "shared_GPU_memory_budget", snapshot
    return None, snapshot


def guard_physical_boundaries(learner, should_stop):
    """Check between actual physics steps, without changing PPO batch semantics."""
    original_step, original_reset = learner.env.step, learner.env.reset
    def checked_step(action):
        if should_stop():
            raise SafeStop("Uncommitted batch/episode interrupted; replay from last.pt on resume")
        return original_step(action)
    def checked_reset(**kwargs):
        if should_stop():
            raise SafeStop("Stopped at an episode boundary")
        return original_reset(**kwargs)
    learner.env.step, learner.env.reset = checked_step, checked_reset


def same_tree(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(same_tree(a,b) for a,b in zip(left,right))
    return left == right


def committed_best_payload(checkpoint):
    selected = min(checkpoint["history"], key=lambda row: row["validation"]["objective"])
    if selected["validation"]["objective"] != checkpoint["best_score"]:
        raise RuntimeError("Checkpoint best score/history disagree")
    return {"protocol": checkpoint["protocol"], "actor": checkpoint["best_actor"],
            "score": checkpoint["best_score"], "selected_update": selected["updates"],
            "selected_physical_steps": selected["physical_steps"]}


def freeze_training_artifacts(folder):
    """Recover a stale best sidecar from committed last.pt, then freeze final files."""
    last = folder/"last.pt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    if checkpoint["episodes"] != checkpoint["protocol"]["config"]["episodes"]:
        raise RuntimeError("Cannot finalize an incomplete training checkpoint")
    expected_best = committed_best_payload(checkpoint)
    completion = folder/"training_complete.json"
    if completion.exists():
        frozen = read(completion)
        for name, digest in frozen["immutable_sha256"].items():
            if not (folder/name).is_file() or sha(folder/name) != digest:
                raise RuntimeError("Immutable training artifact changed: " + name)
        if frozen["last_sha256"] != sha(last):
            raise RuntimeError("Training last.pt changed after final freeze")
        if not same_tree(torch.load(folder/"best.pt", map_location="cpu", weights_only=False), expected_best):
            raise RuntimeError("Frozen selected policy differs from committed validation history")
        return checkpoint, frozen
    best_path = folder/"best.pt"
    existing = None
    if best_path.exists():
        try:
            existing = torch.load(best_path, map_location="cpu", weights_only=False)
        except Exception:
            pass
    if not same_tree(existing, expected_best):
        if best_path.exists():
            evidence = folder/("best.pre_recovery."+sha(best_path)[:16]+".pt")
            if not evidence.exists():
                shutil.copyfile(best_path, evidence)
        atomic_save(expected_best, best_path)
    final_training = folder/"final_training.pt"
    if final_training.exists():
        if sha(final_training) != sha(last):
            raise RuntimeError("Existing final_training.pt differs; never overwrite it")
    else:
        temporary = final_training.with_suffix(".pt.tmp")
        shutil.copyfile(last, temporary)
        os.replace(temporary, final_training)
    final_policy = {"role": "final_training_actor_not_validation_selected",
                    "actor": checkpoint["actor"], "updates": checkpoint["updates"],
                    "episodes": checkpoint["episodes"], "source_last_sha256": sha(last)}
    if (folder/"final_policy.pt").exists():
        if not same_tree(torch.load(folder/"final_policy.pt", map_location="cpu", weights_only=False), final_policy):
            raise RuntimeError("Existing final_policy.pt differs; never overwrite it")
    else:
        atomic_save(final_policy, folder/"final_policy.pt")
    frozen = {"episodes": checkpoint["episodes"], "updates": checkpoint["updates"],
              "physical_steps": checkpoint["physical_steps"], "last_sha256": sha(last),
              "immutable_sha256": {name: sha(folder/name) for name in
                  ("initial_policy.pt", "best.pt", "final_training.pt", "final_policy.pt")},
              "best_selected_update": expected_best["selected_update"],
              "best_is_initial": expected_best["selected_update"] == 0}
    write(completion, frozen)
    return checkpoint, frozen


def config_for(cfg, seed):
    values = dict(cfg["training"], seed=seed)
    values["validation_seeds"] = tuple(values["validation_seeds"])
    return TrainConfig(**values)


def make_learner(task, cfg, seed, scale=1.0):
    kwargs = dict(task["env_kwargs"])
    physical = make_task(task["env"], kwargs)
    if scale != 1.0:
        physical.unwrapped.initial_scale *= scale
    env = PortableMFacade(physical)
    env.require_algorithm("episode_metric_ppo")
    learner = StabilizationLearner(env, config=config_for(cfg, seed), env_id=task["env"], env_kwargs=kwargs)
    return learner, physical


def paired_summary(records):
    methods = sorted({r["method"] for r in records})
    summaries = {}
    keys = ("fixed_horizon_cost", "success", "failure", "arrival_time", "terminal_physical_error", "V_rise_fraction", "actual_decay_violation_fraction", "action_total_variation")
    for method in methods:
        items = [r for r in records if r["method"] == method]
        summaries[method] = {k: float(np.mean([r[k] for r in items])) for k in keys}
        summaries[method]["episodes"] = len(items)
    comparisons = {}
    lookup = {(r["method"], r["seed"]): r for r in records}
    for method in methods:
        if method == "identity":
            continue
        items = [(r, lookup[("identity", r["seed"])]) for r in records if r["method"] == method]
        jointly = [(a,b) for a,b in items if a["success"] and b["success"]]
        comparisons[method + "_vs_identity"] = {
            "n": len(items),
            "success_gain_count": sum(int(a["success"])-int(b["success"]) for a,b in items),
            "learned_only_success": sum(a["success"] and not b["success"] for a,b in items),
            "identity_only_success": sum(b["success"] and not a["success"] for a,b in items),
            "mean_paired_fixed_cost_delta": float(np.mean([a["fixed_horizon_cost"]-b["fixed_horizon_cost"] for a,b in items])),
            "joint_success_n": len(jointly),
            "joint_success_time_delta": float(np.mean([a["arrival_time"]-b["arrival_time"] for a,b in jointly])) if jointly else None,
            "all_start_capped_time_delta": float(np.mean([a["arrival_time"]-b["arrival_time"] for a,b in items])),
        }
    return {"summaries": summaries, "paired": comparisons, "candidate_certificate_only": True}


def recompute_episode(physical, arrays, decay, certificate_weight):
    """Reconstruct the applied controller, physics, V, costs and final hold."""
    M = np.asarray(arrays["M"])
    states, following = np.asarray(arrays["states"]), np.asarray(arrays["following"])
    actions = np.asarray(arrays["actions"])
    n = len(states)
    if n == 0 or n > physical.max_steps or len(following) != n or len(actions) != n:
        raise RuntimeError("Invalid evaluation trajectory length")
    if not all(np.isfinite(v).all() for v in (M, states, following, actions)):
        raise RuntimeError("Nonfinite saved trajectory")
    if n > 1 and not np.array_equal(states[1:], following[:-1]):
        raise RuntimeError("Saved states do not form a contiguous trajectory")
    errors, next_errors, commanded, stepped = [], [], [], []
    for state, action, nxt in zip(states, actions, following):
        with torch.no_grad():
            e, context = physical.inner_env.build_error(state)
            e_next, _ = physical.inner_env.build_error(nxt)
            command = physical.inner_env.calculate_action_from_M(torch.as_tensor(M).to(e), e, context=context)
        errors.append(e.numpy()); next_errors.append(e_next.numpy())
        commanded.append(np.clip(command.numpy(), physical.action_space.low, physical.action_space.high))
        stepped.append(independent_physical_step(physical, state, action))
    errors, next_errors = np.asarray(errors), np.asarray(next_errors)
    commanded, stepped = np.asarray(commanded), np.asarray(stepped)
    checks = {
        "actual_controller_action": bool(np.allclose(actions, commanded, atol=1e-9, rtol=1e-8)),
        "independent_physical_steps": bool(np.allclose(following, stepped, atol=1e-10, rtol=1e-9)),
        "error_before_recomputed": bool(np.allclose(arrays["errors"], errors, atol=1e-10, rtol=1e-9)),
        "error_after_recomputed": bool(np.allclose(arrays["errors_after"], next_errors, atol=1e-10, rtol=1e-9)),
        "actions_in_limits": bool(np.all(actions >= physical.action_space.low-1e-10)
            and np.all(actions <= physical.action_space.high+1e-10)),
        "metric_shape": M.shape == (physical.error_dim, physical.error_dim),
        "metric_symmetric": bool(np.allclose(M, M.T, atol=1e-7, rtol=1e-7)),
    }
    eigen = np.linalg.eigvalsh(M)
    checks["metric_spectrum_and_trace"] = bool(eigen.min() >= physical.metric_spec.eigen_floor-2e-6
        and eigen.max() <= physical.metric_spec.eigen_ceiling+2e-6
        and abs(np.trace(M)-physical.metric_spec.trace) < 2e-5)
    before = np.einsum("bi,ij,bj->b", errors, M, errors)
    after = np.einsum("bi,ij,bj->b", next_errors, M, next_errors)
    checks["V_before_recomputed"] = bool(np.allclose(arrays["V_before"], before, atol=1e-9, rtol=1e-8))
    checks["V_after_recomputed"] = bool(np.allclose(arrays["V_after"], after, atol=1e-9, rtol=1e-8))
    costs_before = np.asarray([physical.inner_env.physical_cost(s) for s in states])
    terminal = float(physical.inner_env.physical_cost(following[-1]))
    costs = physical.dt * costs_before
    checks["integrated_costs_recomputed"] = bool(np.allclose(arrays["costs"], costs, atol=1e-10, rtol=1e-9))
    checks["physical_error_labels_recomputed"] = bool(
        np.allclose(arrays["physical_error_before"],costs_before,atol=1e-10,rtol=1e-9)
        and np.allclose(arrays["physical_error_after"],
            [physical.inner_env.physical_cost(s) for s in following],atol=1e-10,rtol=1e-9))
    all_states = np.concatenate((states[:1], following), axis=0)
    in_goal = np.asarray([physical.inner_env.is_success(s) for s in all_states], dtype=bool)
    failures = np.asarray([physical.inner_env.is_failure(s) for s in following], dtype=bool)
    failure = bool(failures[-1])
    checks["no_steps_after_failure"] = not bool(failures[:-1].any())
    checks["full_horizon_or_failure"] = n == physical.max_steps or failure
    checks["goal_flags_recomputed"] = (np.array_equal(arrays["in_goal_before"], in_goal[:-1])
                                      and np.array_equal(arrays["in_goal_after"], in_goal[1:]))
    start_index = (int(np.flatnonzero(~in_goal)[-1])+1 if not in_goal.all() else 0) if in_goal[-1] else n+1
    hold_seconds = max(0., (n-start_index)*physical.dt)
    success = bool(n == physical.max_steps and not failure and in_goal[-1]
                   and hold_seconds+1e-10 >= physical.required_hold_seconds)
    horizon = physical.dt*physical.max_steps
    arrival = start_index*physical.dt+physical.required_hold_seconds if success else horizon
    settling = start_index*physical.dt if success else horizon
    accumulated = float(costs.sum())
    fixed = accumulated + max(0, physical.max_steps-n)*physical.dt*(max(terminal,25.) if failure else terminal)
    norm2 = np.square(errors).sum(axis=1)
    residual = after-before+decay*physical.dt*norm2
    penalty_scale = np.maximum(norm2,1e-6)
    penalty = float(np.sum(np.maximum(0.,after-before+decay*physical.dt*penalty_scale)/penalty_scale))
    checks["decay_residual_recomputed"] = bool(np.allclose(arrays["decay_residual"], residual, atol=1e-9, rtol=1e-8))
    result = {"success": success, "failure": failure, "arrival_time": float(arrival),
              "settling_time": float(settling), "hold_seconds": float(hold_seconds),
              "required_hold_seconds": float(physical.required_hold_seconds),
              "fixed_horizon_cost": fixed, "accumulated_cost": accumulated,
              "terminal_physical_error": terminal, "physical_steps": n,
              "certificate_penalty": penalty, "objective": fixed+certificate_weight*penalty,
              "V_rise_fraction": float(np.mean(after-before > 1e-8)),
              "actual_decay_violation_fraction": float(np.mean(residual > 1e-8)),
              "actual_decay_violation_max": float(max(0.,residual.max())),
              "action_total_variation": float(np.linalg.norm(np.diff(actions,axis=0),axis=1).sum()),
              "horizon_seconds": float(horizon), "M_eigen_min": float(eigen.min()),
              "M_eigen_max": float(eigen.max()), "M_trace": float(np.trace(M)),
              "physical_terminated": bool(failure or success),
              "physical_truncated": bool(n == physical.max_steps and not (failure or success))}
    if not all(checks.values()):
        raise RuntimeError("Independent episode checks failed: "+str([k for k,v in checks.items() if not v]))
    return result, {"passed": True, "checks": checks,
        "max_controller_action_abs_error": float(np.max(np.abs(actions-commanded))),
        "max_physical_step_abs_error": float(np.max(np.abs(following-stepped))),
        "scope": "engineering_recomputation_of_recorded_trajectory_not_global_stability"}


def check_cached_episode(learner, physical, method, seed, directory):
    row = read(directory/f"{method}_{seed}.json")
    expected = {"method":method, "seed":seed, "policy_hash":module_hash(learner.actor)}
    if any(row.get(k) != v for k,v in expected.items()):
        raise RuntimeError("Cached evaluation policy/method/seed changed")
    trajectory = directory/row["trajectory"]
    if trajectory.resolve().parent != directory.resolve() or sha(trajectory) != row["trajectory_sha256"]:
        raise RuntimeError("Cached trajectory changed or escapes evaluation directory")
    obs,_ = learner.env.reset(seed=seed)
    with torch.no_grad():
        metric = (torch.eye(learner.env.error_dim)*learner.env.metric_spec.trace/learner.env.error_dim
                  if method == "identity" else learner.actor(learner._context(obs)))
    with np.load(trajectory, allow_pickle=False) as data:
        if not np.array_equal(data["states"][0], obs) or not np.array_equal(data["M"], metric.numpy()):
            raise RuntimeError("Cached initial state or policy-produced M differs")
        recomputed, _ = recompute_episode(physical, data, learner.config.decay, learner.config.certificate_weight)
    for name,value in recomputed.items():
        if name not in row or not math.isclose(float(row[name]),float(value),abs_tol=1e-8,rel_tol=1e-8):
            raise RuntimeError("Cached result disagrees with independent recomputation: "+name)
    return row


def evaluate_one(learner, physical, method, seed, directory):
    obs, _ = learner.env.reset(seed=seed)
    with torch.no_grad():
        M = (torch.eye(learner.env.error_dim)*learner.env.metric_spec.trace/learner.env.error_dim
             if method == "identity" else learner.actor(learner._context(obs)))
    result = rollout_metric(learner.env, M, obs, certificate_weight=learner.config.certificate_weight,
                            decay=learner.config.decay, save_trajectory=True)
    trace = result.pop("trajectory")
    errors = np.asarray([x["error_before"] for x in physical.trace], dtype=float)
    states = np.asarray([x["state_before"] for x in physical.trace], dtype=float)
    following = np.asarray([x["state_after"] for x in physical.trace], dtype=float)
    errors_after = np.asarray([x["error_after"] for x in physical.trace], dtype=float)
    actions = np.asarray([x["action"] for x in physical.trace], dtype=float)
    submitted = np.asarray([x["action"] for x in trace], dtype=float)
    if not np.allclose(actions,submitted,atol=1e-10,rtol=1e-10):
        raise RuntimeError("Submitted and actually applied actions differ")
    before = np.asarray([x["V_before"] for x in trace])
    after = np.asarray([x["V_after"] for x in trace])
    residual = after-before + learner.config.decay*physical.dt*np.sum(errors**2, axis=1)
    arrays = dict(M=M.numpy(), states=states, following=following, errors=errors, errors_after=errors_after,
                  actions=actions, V_before=before, V_after=after, decay_residual=residual,
                  costs=np.asarray([x["cost"] for x in physical.trace]),
                  physical_error_before=np.asarray([x["physical_error_before"] for x in physical.trace]),
                  physical_error_after=np.asarray([x["physical_error_after"] for x in physical.trace]),
                  in_goal_before=np.asarray([x["in_goal_before"] for x in physical.trace]),
                  in_goal_after=np.asarray([x["in_goal_after"] for x in physical.trace]),
                  goal=np.asarray(physical.goal), dt=np.asarray(physical.dt),
                  times=(np.arange(len(states))+1)*physical.dt,
                  required_hold_seconds=np.asarray(physical.required_hold_seconds))
    recomputed, audit = recompute_episode(physical,arrays,learner.config.decay,learner.config.certificate_weight)
    for name in ("fixed_horizon_cost","accumulated_cost","objective","certificate_penalty","V_rise_fraction"):
        if not math.isclose(float(result[name]),recomputed[name],abs_tol=1e-8,rel_tol=1e-8):
            raise RuntimeError("Rollout and independent result disagree: "+name)
    result.update(recomputed)
    result.update(method=method, seed=seed, policy_hash=module_hash(learner.actor),
                  independent_episode_audit=audit, env_id=physical.env_id,
                  env_kwargs=dict(learner.env_kwargs,initial_scale=float(physical.inner_env.initial_scale)),
                  protocol_hold_seconds=1.0, tracking_cycles=float(physical.tracking_cycles))
    key = f"{method}_{seed}"
    target = directory / (key+".npz")
    temporary = directory / (key+".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary,target)
    result["trajectory"] = target.name
    result["trajectory_sha256"] = sha(target)
    write(directory/(key+".json"), result)
    return result


def report(output, cfg):
    tasks=[]
    for seed in cfg["seeds"]:
        for task in cfg["tasks"]:
            path=output/(task["id"]+f"_seed{seed}")/"complete.json"
            if path.exists():
                value=read(path)
                if value.get("task") != task["id"] or value.get("seed") != seed:
                    raise RuntimeError("Unexpected task/seed in completed record")
                tasks.append(value)
    result={"completed_settings_seeds":len(tasks), "expected_settings_seeds":len(cfg["tasks"])*len(cfg["seeds"]),
            "complete":len(tasks)==len(cfg["tasks"])*len(cfg["seeds"]), "runs":tasks,
            "interpretation":"A completed experiment can fail empirical stabilization or fail to improve performance. SPD alone is not stability. No test selection.",
            "original_protocols":"Original Push and indoor visual runs remain separate controls; this suite does not replace their metrics or claim identical trainers."}
    write(output/"RESULTS.json",result)
    (output/"REPORT.html").write_text('<!doctype html><meta charset="utf-8"><title>M learning book suite</title><h1>书稿 M 学习长测</h1><p>统一接口与完整时域；工程正确、实测下降和性能分开。以下仅列已完成实验。</p><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>',encoding="utf-8")
    return result


def probe_policy_effects(learner, physical, initial_state, best_state, final_state):
    """Paired fixed-state probes; no physics step, no optimizer, no validation selection."""
    actor=deepcopy(learner.actor)
    maxima={name:{"max_M_delta":0.,"max_action_delta":0.} for name in ("learned","final")}
    snapshots=[]
    for seed in (7241,7242,7243,7244):
        observation,_=physical.reset(seed=seed)
        error,context=physical.inner_env.build_error(observation)
        condition=learner._context(observation)
        matrices,actions={},{}
        for name,state in (("initial",initial_state),("learned",best_state),("final",final_state)):
            actor.load_state_dict(state)
            with torch.no_grad():
                matrices[name]=actor(condition).to(error)
                actions[name]=physical.inner_env.calculate_action_from_M(matrices[name],error,context=context)
        for name in maxima:
            maxima[name]["max_M_delta"]=max(maxima[name]["max_M_delta"],float((matrices[name]-matrices["initial"]).abs().max()))
            maxima[name]["max_action_delta"]=max(maxima[name]["max_action_delta"],float((actions[name]-actions["initial"]).abs().max()))
        snapshots.append({"seed":seed,"state":np.asarray(observation).tolist()})
    return {"against":"initial_policy_on_identical_state_context", "fixed_probes":snapshots,
            "effects":maxima,"interpretation":"Measured policy effect only; not evidence of performance improvement."}


def completion_hashes(folder):
    names=["initial_policy.pt","last.pt","best.pt","final_training.pt","final_policy.pt",
           "training_complete.json","ENGINEERING_AUDIT.json","POLICY_EFFECT_PROBES.json"]
    paths=[folder/name for name in names]
    paths.extend(p for p in folder.glob("test_scale_*/*") if p.suffix in (".json",".npz"))
    return {p.relative_to(folder).as_posix():sha(p) for p in sorted(paths)}


def verify_completed(folder, task, seed):
    row=read(folder/"complete.json")
    if row.get("task")!=task["id"] or row.get("seed")!=seed or not row.get("artifact_sha256"):
        raise RuntimeError("Invalid completed task record")
    for relative,digest in row["artifact_sha256"].items():
        path=(folder/relative).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or sha(path)!=digest:
            raise RuntimeError("Completed task artifact changed: "+relative)


def run(args):
    cfg = read(args.config)
    validate_config(cfg)
    out = args.output.resolve()
    if not out.is_relative_to((ROOT/"runs").resolve()) or out==(ROOT/"runs").resolve():
        raise ValueError("Outputs must remain within this project's runs directory")
    if not math.isfinite(args.max_minutes) or args.max_minutes<=0:
        raise ValueError("max-minutes must be positive and finite")
    torch.set_num_threads(1)
    if args.preflight:
        for task in cfg["tasks"]:
            learner, physical = make_learner(task,cfg,cfg["seeds"][0])
            print(json.dumps({"task":task["id"],"facade":type(learner.env).__name__,"dt":physical.dt,"horizon":physical.max_steps,"error_dim":physical.error_dim}))
            learner.env.close()
        print(json.dumps({"preflight":True,"training_started":False,"dependency_ready":(ROOT/cfg["depends_on"]).exists()}))
        return
    prerequisites(cfg)
    engineering_gate=preparation_gate(cfg,args.config)
    if (out/"STOP").exists():
        print("User STOP exists; no run resumed",flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    expected={"config":cfg,"sources":source_manifest(args.config),"algorithm":"shared_episode_M_PPO",
              "runtime":{"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,
                         "scipy":importlib.metadata.version("scipy"),
                         "gymnasium":importlib.metadata.version("gymnasium"),"device":"cpu",
                         "torch_threads":torch.get_num_threads()},
              "preparation_gate":engineering_gate,
              "visual_final_audit_sha256":sha(ROOT/cfg["depends_on"]),
              "recovery":"committed full-batch checkpoints; incomplete batch replayed, not counted twice"}
    deadline=time.monotonic()+args.max_minutes*60
    last_resource_time=0.0
    last_status_time=0.0
    last_resources=(None,{})

    def stop(stage):
        nonlocal last_resource_time,last_status_time,last_resources
        if time.monotonic()-last_resource_time>15:
            last_resources=resources(out,cfg)
            last_resource_time=time.monotonic()
        reason,snapshot=last_resources
        if (out/"STOP").exists(): reason="user_STOP"
        if time.monotonic()>=deadline: reason=reason or "time_slice"
        if reason is not None or time.monotonic()-last_status_time>5:
            write(out/"status.json",{"stage":stage,"stop_reason":reason,"resources":snapshot,
                  "recovery":"Replay any uncommitted batch/episode from the last committed checkpoint",
                  "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            last_status_time=time.monotonic()
        return reason is not None

    with WriterLock(ROOT/"runs/book_m_suite_slot.lock"), WriterLock(out/"writer.lock"):
        if (out/"protocol.json").exists():
            if read(out/"protocol.json")!=expected:
                raise RuntimeError("Sources/config changed; preserve run and create a new protocol")
        else:
            write(out/"protocol.json",expected)
            for relative,digest in expected["sources"].items():
                destination=out/"provenance"/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/relative,destination)
                if sha(destination)!=digest: raise RuntimeError("Source changed during protocol capture")
        for seed in cfg["seeds"]:
            for task in cfg["tasks"]:
                tag=task["id"]+f"_seed{seed}"
                folder=out/tag
                if (folder/"complete.json").exists():
                    verify_completed(folder,task,seed)
                    continue
                if stop(tag): return
                folder.mkdir(exist_ok=True)
                if not (folder/"ENGINEERING_AUDIT.json").exists():
                    audit=audit_task(task["env"],task["env_kwargs"])
                    write(folder/"ENGINEERING_AUDIT.json",audit)
                if not read(folder/"ENGINEERING_AUDIT.json").get("passed"):
                    raise RuntimeError("Task engineering audit failed: "+tag)
                learner, physical=make_learner(task,cfg,seed)
                initial_state=deepcopy(learner.actor.state_dict())
                if (folder/"initial_policy.pt").exists():
                    if not same_tree(torch.load(folder/"initial_policy.pt",map_location="cpu",weights_only=False),initial_state):
                        raise RuntimeError("Initial policy differs from the declared seed/initialization")
                else:
                    if (folder/"last.pt").exists(): raise RuntimeError("Existing training checkpoint lost its initial policy")
                    atomic_save(initial_state,folder/"initial_policy.pt")
                try:
                    if (folder/"last.pt").exists(): learner.restore(folder/"last.pt")
                    guard_physical_boundaries(learner,lambda:stop(tag+":training"))
                    while learner.episodes<learner.config.episodes:
                        if stop(tag+":training"): return
                        learner.train(folder,resume=(folder/"last.pt").exists(),max_updates=1)
                        print(json.dumps({"task":tag,"episodes":learner.episodes,"updates":learner.updates,"physical_steps":learner.physical_steps}),flush=True)
                    checkpoint,frozen=freeze_training_artifacts(folder)
                    final_state=deepcopy(checkpoint["actor"])
                    best_state=deepcopy(checkpoint["best_actor"])
                    effects=probe_policy_effects(learner,physical,initial_state,best_state,final_state)
                    if (folder/"POLICY_EFFECT_PROBES.json").exists():
                        if read(folder/"POLICY_EFFECT_PROBES.json")!=effects:
                            raise RuntimeError("Fixed policy-effect probes changed on resume")
                    else: write(folder/"POLICY_EFFECT_PROBES.json",effects)
                except SafeStop:
                    return
                finally:
                    learner.env.close()
                evaluated={}
                for factor in cfg["test_initial_scale_factors"]:
                    eval_folder=folder/("test_scale_"+str(factor))
                    eval_folder.mkdir(exist_ok=True)
                    evaluator,physical=make_learner(task,cfg,seed,scale=factor)
                    evaluation_protocol={"factor":factor,"task":task,"training_seed":seed,
                        "test_seeds":list(range(cfg["test_seed_start"],cfg["test_seed_start"]+cfg["test_seed_count"])),
                        "frozen_training":frozen,"protocol_sha256":sha(out/"protocol.json")}
                    if (eval_folder/"evaluation_protocol.json").exists():
                        if read(eval_folder/"evaluation_protocol.json")!=evaluation_protocol:
                            raise RuntimeError("Evaluation protocol or frozen training changed")
                    else: write(eval_folder/"evaluation_protocol.json",evaluation_protocol)
                    guard_physical_boundaries(evaluator,lambda:stop(tag+":evaluation"))
                    records=[]
                    try:
                        for method,state in (("identity",initial_state),("initial",initial_state),("learned",best_state),("final",final_state)):
                            evaluator.actor.load_state_dict(state)
                            before_hash=module_hash(evaluator.actor)
                            for test_seed in range(cfg["test_seed_start"],cfg["test_seed_start"]+cfg["test_seed_count"]):
                                path=eval_folder/f"{method}_{test_seed}.json"
                                if path.exists():
                                    row=check_cached_episode(evaluator,physical,method,test_seed,eval_folder)
                                else:
                                    if stop(tag+":evaluation"): return
                                    row=evaluate_one(evaluator,physical,method,test_seed,eval_folder)
                                records.append(row)
                            if module_hash(evaluator.actor)!=before_hash: raise RuntimeError("Evaluation modified actor")
                    except SafeStop:
                        return
                    finally:
                        evaluator.env.close()
                    for test_seed in range(cfg["test_seed_start"],cfg["test_seed_start"]+cfg["test_seed_count"]):
                        with np.load(eval_folder/f"identity_{test_seed}.npz") as base:
                            start=base["states"][0].copy()
                        for method in ("initial","learned","final"):
                            with np.load(eval_folder/f"{method}_{test_seed}.npz") as paired:
                                if not np.array_equal(start,paired["states"][0]):
                                    raise RuntimeError("Paired test starts differ")
                    evaluated[str(factor)]=paired_summary(records)
                    write(eval_folder/"SUMMARY.json",evaluated[str(factor)])
                best_info=torch.load(folder/"best.pt",map_location="cpu",weights_only=False)
                completed={"task":task["id"],"seed":seed,"best_selected_update":best_info["selected_update"],"best_is_initial":best_info["selected_update"]==0,
                           "evaluations":evaluated,"final_training_sha256":sha(folder/"final_training.pt"),"best_sha256":sha(folder/"best.pt"),
                           "actor_parameters_changed":any(not torch.equal(final_state[k],initial_state[k]) for k in final_state),
                           "policy_effect_probes":effects,"candidate_certificate_only":True,
                           "artifact_sha256":completion_hashes(folder),
                           "independent_final_audit":"pending; completion here means training/evaluation coverage only"}
                write(folder/"complete.json",completed)
                report(out,cfg)
        final=report(out,cfg)
        write(out/"complete.json",{"evaluation_complete":final["complete"],"independent_final_audit":"required"})
        write(out/"status.json",{"stage":"evaluation_complete_awaiting_independent_scientific_audit",
              "stop_reason":None,"candidate_certificate_only":True,
              "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z")})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/book_m_night_suite.json")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/book_m_night_suite_20261023_v1")
    parser.add_argument("--max-minutes",type=float,default=120)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        safe_output=args.output.resolve()
        if (not args.preflight and safe_output.is_relative_to((ROOT/"runs").resolve())
                and safe_output!=(ROOT/"runs").resolve() and (safe_output/"protocol.json").exists()):
            write(args.output/"error.json",{"type":type(exc).__name__,"message":str(exc),"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        raise
