"""Independent scientific audit of recorded full-horizon M-learning episodes.

This file does not import the suite runner or modify its sources/checkpoints.
``passed`` means reproducible measurements and complete coverage, never good
control performance or a global Lyapunov proof. Cached audits are content keyed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m_learning_suite.task_runtime import make_task, independent_physical_step
import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def independent_errors(task, states):
    """NumPy physical error coordinates, independent of tensor_error."""
    s = np.asarray(states, dtype=np.float64)
    env, kind = task.inner_env, task.env_id
    if kind == "unicycle_tracking":
        phase, heading = s[..., 3], s[..., 2]
        dx = s[..., 0] - env.radius*np.sin(phase)
        dy = s[..., 1] - env.radius*(1-np.cos(phase))
        return np.stack((np.cos(heading)*dx+np.sin(heading)*dy,
                         -np.sin(heading)*dx+np.cos(heading)*dy,
                         wrap(heading-phase)), axis=-1)
    if kind == "multirotor_tracking":
        reference = np.broadcast_to(env.goal[:6], s[..., :6].shape).copy()
        if env.trajectory == "circle":
            phase, radius, omega = s[..., 6], env.radius, env.reference_omega
            ax, az = -radius*omega**2*np.sin(phase), radius*omega**2*np.cos(phase)
            fx, fz = -ax, env.gravity+az
            fx_dot, fz_dot = radius*omega**3*np.cos(phase), -radius*omega**3*np.sin(phase)
            reference = np.stack((env.goal[0]+radius*np.sin(phase), env.goal[1]+radius*(1-np.cos(phase)),
                radius*omega*np.cos(phase), radius*omega*np.sin(phase), np.arctan2(fx,fz),
                (fz*fx_dot-fx*fz_dot)/(fz**2+fx**2)), axis=-1)
        difference = s[..., :6] - reference
        difference[..., 4] = wrap(difference[..., 4])
        return difference
    difference = s - env.goal
    if kind == "pendulum":
        difference[..., 0] = wrap(s[..., 0])
    elif kind == "cartpole":
        difference[..., 2] = wrap(s[..., 2])
    elif kind == "two_link_arm":
        difference[..., :2] = wrap(difference[..., :2])
    return difference


def local_closed_loop(task, metric, epsilon=1e-5):
    """Finite differences of the executed discrete map in physical error coordinates.

    M stays fixed. No learning gradient, time-varying-M derivative or moving
    target is silently omitted. Static checks are explicitly inapplicable to
    moving references. A failed inequality is a result, not corrupt data.
    """
    if task.moving_reference:
        return {"applicable": False, "reason": "moving_reference_requires_time_varying_or_periodic_analysis",
                "global_certificate_claim": False}
    env, dimension = task.inner_env, task.error_dim
    metric = np.asarray(metric, dtype=np.float64)
    origin = np.asarray(env.goal, dtype=np.float64).copy()
    matrix = torch.as_tensor(metric)

    def transition(error):
        state = origin.copy()
        state[:dimension] += error
        with torch.no_grad():
            e, context = env.build_error(state)
            action = env.calculate_action_from_M(matrix.to(e), e, context=context).numpy()
        action = np.clip(action, env.action_space.low, env.action_space.high)
        following = independent_physical_step(task, state, action)
        return independent_errors(task, following), action

    residual, zero_action = transition(np.zeros(dimension))
    jacobians = []
    for size in (epsilon, epsilon/2):
        columns = []
        for axis in np.eye(dimension):
            plus, _ = transition(size*axis)
            minus, _ = transition(-size*axis)
            columns.append((plus-minus)/(2*size))
        jacobians.append(np.column_stack(columns))
    coarse, fine = jacobians
    eig = np.linalg.eigvals(fine)
    lyapunov = fine.T @ metric @ fine - metric
    lyapunov = (lyapunov+lyapunov.T)/2
    lyapunov_eig = np.linalg.eigvalsh(lyapunov)
    residual_norm = float(np.linalg.norm(residual))
    fd_disagreement = float(np.max(np.abs(fine-coarse)))
    action_margin = float(min(np.min(zero_action-env.action_space.low), np.min(env.action_space.high-zero_action)))
    meaningful = residual_norm < 1e-8 and fd_disagreement < 1e-5 and action_margin > 1e-7
    radius = float(np.max(np.abs(eig)))
    return {"applicable": True, "scope": "local_numerical_evidence_only_at_static_equilibrium",
        "global_certificate_claim": False, "equilibrium_error_residual_norm": residual_norm,
        "equilibrium_action": zero_action.tolist(), "equilibrium_actuator_margin": action_margin,
        "finite_difference_steps": [epsilon,epsilon/2], "finite_difference_disagreement_max": fd_disagreement,
        "numerically_meaningful": meaningful, "A_cl_discrete": fine.tolist(),
        "closed_loop_eigenvalues": [{"real":float(x.real),"imag":float(x.imag)} for x in eig],
        "spectral_radius": radius, "spectral_radius_below_one": bool(radius < 1-1e-7) if meaningful else None,
        "A_transpose_M_A_minus_M": lyapunov.tolist(), "lyapunov_matrix_eigenvalues": lyapunov_eig.tolist(),
        "quadratic_matrix_negative_definite": bool(lyapunov_eig.max() < -1e-7) if meaningful else None,
        "interpretation": "Stable local eigenvalues and descent of this particular M are separate numerical findings; neither establishes a global basin or robustness."}


def audit_episode(task, arrays, row, *, decay, certificate_weight):
    """Recompute a complete episode, including the final uninterrupted hold."""
    required = ("M", "states", "following", "actions", "errors", "errors_after", "V_before", "V_after",
                "costs", "physical_error_before", "physical_error_after", "in_goal_before", "in_goal_after",
                "decay_residual", "dt", "goal", "required_hold_seconds")
    missing = sorted(set(required)-set(arrays))
    if missing:
        return {"passed": False, "error": "Missing trajectory fields", "missing": missing}
    metric, states = np.asarray(arrays["M"]), np.asarray(arrays["states"])
    following, actions = np.asarray(arrays["following"]), np.asarray(arrays["actions"])
    n, dimension = len(states), task.error_dim
    shapes_valid = (metric.shape == (dimension,dimension) and states.ndim == 2
        and states.shape == following.shape and states.shape[1:] == task.observation_space.shape
        and actions.shape == (n,)+task.action_space.shape and 0 < n <= task.max_steps)
    finite = all(np.isfinite(np.asarray(arrays[key])).all() for key in required)
    if not shapes_valid or not finite:
        return {"passed": False, "error": "Malformed or nonfinite physical arrays"}
    errors = independent_errors(task, states)
    errors_after = independent_errors(task, following)
    with torch.no_grad():
        builtin_errors, context = task.inner_env.build_error(states)
        command = task.inner_env.calculate_action_from_M(torch.as_tensor(metric).to(builtin_errors),
            builtin_errors, context=context).numpy()
    command = np.clip(command, task.action_space.low, task.action_space.high)
    advanced = np.asarray([independent_physical_step(task, state, action) for state,action in zip(states,actions)])
    norm2 = np.einsum("bi,bi->b", errors, errors)
    after_norm2 = np.einsum("bi,bi->b", errors_after, errors_after)
    before = np.einsum("bi,ij,bj->b",errors,metric,errors)
    after = np.einsum("bi,ij,bj->b",errors_after,metric,errors_after)
    delta = after-before
    residual = delta+decay*task.dt*norm2
    all_states = np.vstack((states[:1],following))
    good = np.asarray([task.inner_env.is_success(s) for s in all_states],dtype=bool)
    failed = np.asarray([task.inner_env.is_failure(s) for s in all_states],dtype=bool)
    # A trajectory starting outside the admissible region must report failure;
    # after reset the first physical step may be recorded before termination.
    final_failure = bool(failed[-1])
    final_segment = n+1
    if good[-1]:
        final_segment = 0 if good.all() else int(np.flatnonzero(~good)[-1])+1
    hold = max(0., (n-final_segment)*task.dt)
    success = bool(n == task.max_steps and not final_failure and good[-1]
                   and hold+1e-10 >= task.required_hold_seconds)
    horizon = task.horizon_seconds
    arrival = final_segment*task.dt+task.required_hold_seconds if success else horizon
    settling = final_segment*task.dt if success else horizon
    accumulated = float(task.dt*norm2.sum())
    terminal = float(after_norm2[-1])
    fixed = accumulated+(task.max_steps-n)*task.dt*(max(terminal,25.) if final_failure else terminal)
    normalization = np.maximum(norm2,1e-6)
    penalty = float(np.sum(np.maximum(0.,delta+decay*task.dt*normalization)/normalization))
    eigen = np.linalg.eigvalsh((metric+metric.T)/2)
    expected_start = task.reset(seed=int(row["seed"]))[0]
    checks = {
        "finite_valid_shapes": shapes_valid and finite,
        "contiguous_state_chain": bool(np.array_equal(states[1:],following[:-1])),
        "seed_reset_start_exact": bool(np.array_equal(states[0],expected_start)),
        "physical_error_coordinates": bool(np.allclose(builtin_errors.numpy(), errors,atol=1e-10,rtol=1e-9)),
        "recorded_error_before": bool(np.allclose(arrays["errors"],errors,atol=1e-10,rtol=1e-9)),
        "recorded_error_after": bool(np.allclose(arrays["errors_after"],errors_after,atol=1e-10,rtol=1e-9)),
        "controller_applied_action": bool(np.allclose(actions,command,atol=1e-9,rtol=1e-8)),
        "independent_physical_successor": bool(np.allclose(following,advanced,atol=1e-10,rtol=1e-9)),
        "actuator_limits": bool(np.all(actions >= task.action_space.low-1e-10) and np.all(actions <= task.action_space.high+1e-10)),
        "M_symmetric": bool(np.allclose(metric,metric.T,atol=1e-7,rtol=1e-7)),
        "M_spectrum_trace": bool(eigen.min() >= task.metric_spec.eigen_floor-2e-6 and
            eigen.max() <= task.metric_spec.eigen_ceiling+2e-6 and abs(np.trace(metric)-task.metric_spec.trace)<2e-5),
        "V_before": bool(np.allclose(arrays["V_before"],before,atol=1e-9,rtol=1e-8)),
        "V_after": bool(np.allclose(arrays["V_after"],after,atol=1e-9,rtol=1e-8)),
        "physical_error_before": bool(np.allclose(arrays["physical_error_before"],norm2,atol=1e-10,rtol=1e-9)),
        "physical_error_after": bool(np.allclose(arrays["physical_error_after"],after_norm2,atol=1e-10,rtol=1e-9)),
        "integrated_physical_costs": bool(np.allclose(arrays["costs"],task.dt*norm2,atol=1e-10,rtol=1e-9)),
        "decay_residual": bool(np.allclose(arrays["decay_residual"],residual,atol=1e-9,rtol=1e-8)),
        "goal_flags": bool(np.array_equal(arrays["in_goal_before"],good[:-1]) and np.array_equal(arrays["in_goal_after"],good[1:])),
        "no_steps_after_physical_failure": not bool(failed[1:-1].any()),
        "full_horizon_or_failure": n == task.max_steps or final_failure,
        "goal_matches": bool(np.array_equal(arrays["goal"],task.goal)),
        "dt_matches": math.isclose(float(arrays["dt"]),task.dt,abs_tol=1e-12),
        "hold_requirement_matches": math.isclose(float(arrays["required_hold_seconds"]),task.required_hold_seconds,abs_tol=1e-12),
    }
    if "times" in arrays:
        checks["step_times"] = bool(np.allclose(arrays["times"],np.arange(1,n+1)*task.dt,atol=1e-12,rtol=1e-12))
    measurements = {"success":success,"failure":final_failure,"arrival_time":float(arrival),
        "settling_time":float(settling),"hold_seconds":float(hold),"required_hold_seconds":task.required_hold_seconds,
        "fixed_horizon_cost":float(fixed),"accumulated_cost":accumulated,"terminal_physical_error":terminal,
        "physical_steps":n,"certificate_penalty":penalty,"objective":float(fixed+certificate_weight*penalty),
        "V_rise_fraction":float(np.mean(delta>1e-8)),"actual_decay_violation_fraction":float(np.mean(residual>1e-8)),
        "actual_decay_violation_max":float(max(0.,residual.max())),
        "action_total_variation":float(np.linalg.norm(np.diff(actions,axis=0),axis=1).sum()),
        "horizon_seconds":horizon,"M_eigen_min":float(eigen.min()),"M_eigen_max":float(eigen.max()),
        "M_trace":float(np.trace(metric)),"physical_terminated":bool(final_failure or success),
        "physical_truncated":bool(n == task.max_steps and not (final_failure or success))}
    mismatches = [key for key,value in measurements.items() if key not in row or
                  not math.isclose(float(value),float(row[key]),rel_tol=1e-8,abs_tol=1e-8)]
    checks["episode_summary_recomputed"] = not mismatches
    checks["identity_method_uses_unit_metric"] = row.get("method") != "identity" or bool(np.array_equal(metric,np.eye(dimension)))
    return {"passed":all(checks.values()),"checks":checks,"summary_mismatches":mismatches,
        "measurements":measurements,"initial_state":states[0].tolist(),
        "max_controller_action_abs_error":float(np.max(np.abs(actions-command))),
        "max_physical_step_abs_error":float(np.max(np.abs(following-advanced))),
        "observed_V_nonincrease_on_all_recorded_steps":bool(np.all(delta <= 1e-8)),
        "observed_declared_decay_on_all_recorded_steps":bool(np.all(residual <= 1e-8)),
        "initial_state_admissible":not bool(failed[0]),
        "local_closed_loop":local_closed_loop(task,metric)}


def aggregate(records):
    """Paired physical measurements; no score or model selection occurs here."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row["method"]].append(row)
    keys = ("fixed_horizon_cost","success","failure","arrival_time","terminal_physical_error",
            "V_rise_fraction","actual_decay_violation_fraction","action_total_variation")
    methods = {method:{**{key:float(np.mean([x[key] for x in values])) for key in keys},
                       "episodes":len(values)} for method,values in sorted(grouped.items())}
    lookup = {(r["method"],r["seed"]):r for r in records}
    comparisons = {}
    for method, rows in sorted(grouped.items()):
        if method == "identity":
            continue
        pairs = [(x,lookup[("identity",x["seed"])]) for x in rows if ("identity",x["seed"]) in lookup]
        if not pairs:
            continue
        joint = [(a,b) for a,b in pairs if a["success"] and b["success"]]
        comparisons[method+"_vs_identity"] = {
            "n":len(pairs),"success_gain_count":sum(int(a["success"])-int(b["success"]) for a,b in pairs),
            "learned_only_success":sum(bool(a["success"] and not b["success"]) for a,b in pairs),
            "identity_only_success":sum(bool(b["success"] and not a["success"]) for a,b in pairs),
            "mean_paired_fixed_cost_delta":float(np.mean([a["fixed_horizon_cost"]-b["fixed_horizon_cost"] for a,b in pairs])),
            "joint_success_n":len(joint),
            "joint_success_time_delta":float(np.mean([a["arrival_time"]-b["arrival_time"] for a,b in joint])) if joint else None,
            "all_start_capped_time_delta":float(np.mean([a["arrival_time"]-b["arrival_time"] for a,b in pairs]))}
    return {"summaries":methods,"paired":comparisons,"candidate_certificate_only":True}


def same_numbers(a, b):
    if isinstance(a,dict):
        return isinstance(b,dict) and all(k in b and same_numbers(v,b[k]) for k,v in a.items())
    if a is None:
        return b is None
    if isinstance(a,(int,float,bool)):
        return isinstance(b,(int,float,bool)) and math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-8)
    return a == b


def run_suite(output, *, allow_partial=False, max_minutes=None):
    output = Path(output).resolve()
    protocol = read(output/"protocol.json")
    cfg = protocol["config"]
    current_sources = {}
    for relative,expected in protocol["sources"].items():
        source = ROOT/relative
        if not source.is_file() or sha(source) != expected:
            raise RuntimeError("Current environment/source differs from frozen protocol: "+relative)
        frozen = output/"provenance"/relative
        if not frozen.is_file() or sha(frozen) != expected:
            raise RuntimeError("Frozen source provenance differs: "+relative)
        current_sources[relative] = expected
    audit_identity = hashlib.sha256(json.dumps({"audit":sha(__file__),"sources":current_sources},sort_keys=True).encode()).hexdigest()
    started = time.monotonic()
    cases, problems, local_reports = [], [], {}
    audited_count = 0
    expected_count = len(cfg["tasks"])*len(cfg["seeds"])*len(cfg["test_initial_scale_factors"])*4*cfg["test_seed_count"]
    timed_out = False
    for setting in cfg["tasks"]:
        for train_seed in cfg["seeds"]:
            folder = output/(setting["id"]+f"_seed{train_seed}")
            training_complete = folder/"training_complete.json"
            if training_complete.exists():
                frozen = read(training_complete)
                for name,digest in frozen.get("immutable_sha256",{}).items():
                    if not (folder/name).is_file() or sha(folder/name) != digest:
                        problems.append(str(folder/name)+": immutable training hash differs")
                if set(frozen.get("immutable_sha256",{})) != {"initial_policy.pt","best.pt","final_training.pt","final_policy.pt"}:
                    problems.append(str(training_complete)+": missing final artifact coverage")
                if not (folder/"last.pt").is_file() or sha(folder/"last.pt") != frozen.get("last_sha256"):
                    problems.append(str(folder/"last.pt")+": final checkpoint changed")
            else:
                problems.append(str(training_complete)+": not yet present")
            for factor in cfg["test_initial_scale_factors"]:
                directory = folder/("test_scale_"+str(factor))
                records, starts, complete = [], {}, True
                for method in ("identity","initial","learned","final"):
                    for seed in range(cfg["test_seed_start"],cfg["test_seed_start"]+cfg["test_seed_count"]):
                        if max_minutes is not None and time.monotonic()-started >= max_minutes*60:
                            timed_out = True; complete = False; break
                        path = directory/f"{method}_{seed}.json"
                        if not path.exists():
                            complete = False; continue
                        row = read(path)
                        try:
                            if row.get("method") != method or row.get("seed") != seed:
                                raise ValueError("Episode method/seed metadata differs from filename")
                            trajectory = (directory/row["trajectory"]).resolve()
                            if trajectory.parent != directory.resolve() or sha(trajectory) != row["trajectory_sha256"]:
                                raise ValueError("Trajectory SHA or directory differs")
                            parameters = dict(row.get("env_kwargs",setting["env_kwargs"]))
                            task = make_task(setting["env"],parameters,
                                hold_seconds=row.get("protocol_hold_seconds",1.),tracking_cycles=row.get("tracking_cycles",2.))
                            try:
                                if "env_kwargs" not in row:
                                    task.inner_env.initial_scale *= factor
                                if row.get("env_id",setting["env"]) != setting["env"]:
                                    raise ValueError("Episode environment differs from frozen config")
                                key = hashlib.sha256((audit_identity+sha(path)+row["trajectory_sha256"]).encode()).hexdigest()
                                cache = output/"independent_audit_cache"/(key+".json")
                                if cache.exists():
                                    audited = read(cache)
                                else:
                                    with np.load(trajectory,allow_pickle=False) as arrays:
                                        audited = audit_episode(task,arrays,row,decay=cfg["training"]["decay"],
                                            certificate_weight=cfg["training"]["certificate_weight"])
                                    write(cache,audited)
                            finally:
                                task.close()
                            if not audited["passed"]:
                                problems.append(str(path)+": "+str(audited.get("summary_mismatches") or audited.get("checks") or audited.get("error")))
                                complete = False; continue
                            audited_count += 1
                            records.append(dict(audited["measurements"],method=method,seed=seed))
                            starts[(method,seed)] = audited["initial_state"]
                            local_key = "/".join((folder.name,str(factor),method))
                            reports = local_reports.setdefault(local_key,{})
                            local = audited["local_closed_loop"]
                            local_digest = hashlib.sha256(json.dumps(local,sort_keys=True).encode()).hexdigest()
                            reports[local_digest] = local
                            if audited_count % 128 == 0:
                                print(json.dumps({"independently_audited_episodes":audited_count,"expected":expected_count}),flush=True)
                        except Exception as exc:
                            complete = False
                            problems.append(str(path)+": "+type(exc).__name__+": "+str(exc))
                    if timed_out: break
                pairing = all(starts.get(("identity",seed)) == value for (method,seed),value in starts.items())
                if not pairing:
                    problems.append(str(directory)+": paired initial states differ")
                summary = aggregate(records)
                if complete:
                    if not (directory/"SUMMARY.json").exists() or not same_numbers(summary,read(directory/"SUMMARY.json")):
                        problems.append(str(directory)+": aggregate summary differs or missing")
                cases.append({"task":setting["id"],"train_seed":train_seed,"initial_scale_factor":factor,
                    "complete":complete,"paired_initial_states_exact":pairing,**summary})
                if timed_out: break
            if timed_out: break
        if timed_out: break
    coverage = audited_count == expected_count and all(case["complete"] for case in cases)
    complete_marker = (output/"complete.json").exists() and read(output/"complete.json").get("evaluation_complete") is True
    if not coverage: problems.append("Final episode coverage is incomplete")
    if not complete_marker: problems.append("Suite final completion marker is missing")
    result = {"passed":bool(coverage and complete_marker and not problems),
        "meaning_of_passed":"complete_coverage_and_independently_reproduced_measurements_not_performance_or_stability",
        "audit_script_sha256":sha(__file__),"source_identity":audit_identity,"audited_episodes":audited_count,
        "expected_episodes":expected_count,"coverage_complete":coverage,"time_slice_ended":timed_out,
        "problems":problems,"comparisons":cases,"local_numerical_checks":{k:list(v.values()) for k,v in local_reports.items()},
        "global_certificate_claim":False,"limitations":[
            "Finite trajectories cannot establish global Lyapunov descent or the true region of attraction.",
            "Local static matrix checks freeze M and reference; tracking cases are explicitly skipped.",
            "Initial-condition scale tests are empirical success fractions, not a proven basin boundary or parameter-robustness theorem.",
            "Training seed is the independent experimental unit; episode count does not replace multiple training seeds.",
            "Audit success is independent of whether learning improves performance or satisfies measured certificate descent."]}
    write(output/"INDEPENDENT_SCIENTIFIC_AUDIT.json",result)
    lines = ["# M 学习长测：独立科学审核","",f"测量与覆盖核验通过：**{result['passed']}**。已核验 {audited_count}/{expected_count} 回合。",
        "","这里的通过仅表示完整记录可以独立重算，不表示性能提高或获得全局稳定性证明。","",
        "| 任务 | 训练种子 | 初始范围倍数 | 策略 | 成功率 | 固定时域代价 | 失败封顶到达时间 | 实际下降违例比例 |",
        "|---|---:|---:|---|---:|---:|---:|---:|"]
    for case in cases:
        for method,summary in case["summaries"].items():
            lines.append(f"| {case['task']} | {case['train_seed']} | {case['initial_scale_factor']} | {method} | {summary['success']:.3f} | {summary['fixed_horizon_cost']:.5g} | {summary['arrival_time']:.4g} | {summary['actual_decay_violation_fraction']:.3f} |")
    lines.extend(["","静态任务的局部离散闭环谱与二次型矩阵检验详见同名 JSON；跟踪任务不套用静态平衡判据。",
        "","## 尚未通过的记录或覆盖项",""]+(["- "+item for item in problems] or ["无测量一致性或覆盖问题。性能和证书是否改善仍应分别阅读数值。"]))
    (output/"INDEPENDENT_SCIENTIFIC_AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if not result["passed"] and not allow_partial:
        raise RuntimeError("Independent final audit did not pass; see saved JSON for evidence")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--max-minutes",type=float)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    if arguments.max_minutes is not None and arguments.max_minutes <= 0:
        parser.error("--max-minutes must be positive")
    result = run_suite(arguments.run,allow_partial=arguments.allow_partial,max_minutes=arguments.max_minutes)
    print(json.dumps({key:result[key] for key in ("passed","audited_episodes","expected_episodes","coverage_complete")},ensure_ascii=False))
