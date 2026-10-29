"""Hash-chained full-horizon engineering, never an optimizer or a trainer.

--preflight uses standard-library file/AST/resource checks only. No task,
Torch, NumPy or MuJoCo module is imported until the explicit full gate passes.
An immutable pre-attempt marker makes interrupted, uncommitted physical work
an explicit diagnosis requirement rather than silently replaying it.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


sys.dont_write_bytecode = True
EXTENSION = Path(__file__).resolve().parents[1]
PROJECT = EXTENSION.parents[1]
FORMAL_PROTOCOL = PROJECT / "runs/book_m_night_suite_20261023_v1/protocol.json"
DEFAULT_CONFIG = EXTENSION / "configs/disk_engineering_v1.json"
DEFAULT_FREEZE = EXTENSION / "review/EXECUTION_FREEZE.json"
DEFAULT_OUTPUT = EXTENSION / "checks/disk_full_v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_project(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (PROJECT / path).resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError("A bound source/evidence path escapes the project: " + str(path))
    return path


def immutable_json(path, value):
    """One writer plus rename-to-unused-name, retaining every stale pending file."""
    path = Path(path)
    if path.exists():
        raise RuntimeError("Refusing to overwrite immutable evidence: " + str(path))
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise RuntimeError("Evidence destination unexpectedly appeared")
    os.rename(temporary, path)
    return sha(path)


def atomic_json(path, value):
    """Mutable writer-owned status; external readers must wait for writer exit."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stop_paths(output):
    return (output / "STOP", EXTENSION / "STOP", PROJECT / "STOP")


def resource_status(output):
    usage = shutil.disk_usage("D:/")
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) if output.exists() else 0
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("Cannot verify GPU memory limit: " + result.stderr[-300:])
    gpu = [dict(zip(("used_mib", "total_mib"), map(float, line.split(","))))
        for line in result.stdout.strip().splitlines() if line.strip()]
    if not gpu:
        raise RuntimeError("No GPU resource record returned")
    record = {"utc": utc(), "free_gib": usage.free / 2**30, "output_gib": size / 2**30,
        "gpu": gpu, "gpu_allocations_by_runner": 0}
    record["safe"] = record["free_gib"] >= 50. and record["output_gib"] <= 35. and all(
        row["total_mib"] > 0. and row["used_mib"] / row["total_mib"] <= .65 for row in gpu)
    return record


def xml_literal_hash(path):
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MODEL_XML" for target in node.targets):
            return hashlib.sha256(ast.literal_eval(node.value).encode()).hexdigest()
    raise RuntimeError("Executor MODEL_XML literal absent")


def verify_sources(args):
    config_path, freeze_path = args.config.resolve(), args.freeze.resolve()
    config, freeze = read_json(config_path), read_json(freeze_path)
    if freeze.get("schema") != "push_any_shape_integral_execution_freeze_v1":
        raise RuntimeError("Unrecognized extension source freeze")
    declared = freeze["source_sha256"]
    actual_paths = {path.resolve() for path in EXTENSION.rglob("*.py")}
    actual_paths.add(config_path)
    if {resolve_project(key) for key in declared} != actual_paths:
        raise RuntimeError("Extension Python/config inventory changed")
    for relative, expected in declared.items():
        path = resolve_project(relative)
        if sha(path) != expected:
            raise RuntimeError("Frozen extension source changed: " + relative)
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if freeze["config_sha256"] != sha(config_path) or resolve_project(freeze["config_path"]) != config_path:
        raise RuntimeError("Configuration differs from the frozen protocol")
    protected = freeze["protected_sources51"]
    formal = read_json(FORMAL_PROTOCOL)["sources"]
    if len(protected) != 51 or protected != formal:
        raise RuntimeError("The complete protected 51-source protocol differs")
    for relative, expected in protected.items():
        if sha(resolve_project(relative)) != expected:
            raise RuntimeError("Protected formal source changed: " + relative)
    current_protected = {str(path.relative_to(PROJECT)).replace("\\", "/")
        for folder in (PROJECT / "stabilization-gym/stabilization_gym",
            PROJECT / "stabilization-learning-control/stabilization_learning", PROJECT / "scripts/m_learning_suite")
        for path in folder.rglob("*.py")}
    protected_package_keys = {key for key in protected if any(key.startswith(prefix) for prefix in (
        "stabilization-gym/stabilization_gym/", "stabilization-learning-control/stabilization_learning/", "scripts/m_learning_suite/"))}
    if current_protected != protected_package_keys:
        raise RuntimeError("Protected package Python inventory changed")
    xml_sha = xml_literal_hash(EXTENSION / "push_any_shape/contact_executor.py")
    if xml_sha != freeze["xml_sha256"]:
        raise RuntimeError("Frozen XML differs")
    if config["evaluation"]["seconds"] != 30. or config["evaluation"]["control_steps_per_fixture"] != 1500 or len(config["fixtures"]) != 5:
        raise RuntimeError("This runner only implements the declared five 30-second fixtures")
    if len({fixture["name"] for fixture in config["fixtures"]}) != 5:
        raise RuntimeError("Fixture names must be unique")
    if config["actor"]["optimizer_updates"] != 0 or config["actor"]["optimizer_instances"] != 0:
        raise RuntimeError("Engineering runner cannot construct an optimizer")
    gate_record = None
    if args.gate is not None:
        gate_record = read_json(args.gate)
        if gate_record.get("passed") is not True or gate_record.get("execution") != "full_fixtures" or gate_record.get("freeze_sha256") != sha(freeze_path):
            raise RuntimeError("Root full-fixture gate is absent or binds another freeze")
        short = gate_record["short_independent_audit"]
        short_path = resolve_project(short["path"])
        if sha(short_path) != short["sha256"] or read_json(short_path).get("passed") is not True:
            raise RuntimeError("Bound independent short engineering audit did not pass")
    if args.execute_reviewed_full and gate_record is None:
        raise RuntimeError("Full physical execution requires an explicit root gate")
    return config, {"freeze_sha256": sha(freeze_path), "config_sha256": sha(config_path),
        "gate_sha256": sha(args.gate) if args.gate else None,
        "source_sha256": declared, "protected_sources51": protected, "xml_sha256": xml_sha}


def certify_no_other_writer():
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'run_disk_engineering.py' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if done.returncode:
        raise RuntimeError("Cannot inspect competing engineering writers")
    rows = json.loads(done.stdout) if done.stdout.strip() else []
    rows = [rows] if isinstance(rows, dict) else rows
    competitors = [row for row in rows if int(row["ProcessId"]) != os.getpid() and "--preflight" not in row["CommandLine"]]
    if competitors:
        raise RuntimeError("Another Disk engineering writer exists: " + str(competitors))


def physical_error(state, goal):
    dx, dy = state[0] - goal[0], state[1] - goal[1]
    gap = math.hypot(state[3] - state[0], state[4] - state[1]) - .019
    return dx * dx + dy * dy + sum(value * value for value in state[5:10]) + max(.002 - gap, 0.)**2


def goal_conditions(state, goal):
    gap = math.hypot(state[3] - state[0], state[4] - state[1]) - .019
    return (math.hypot(state[0] - goal[0], state[1] - goal[1]) <= .005 and
        math.hypot(state[5], state[6]) <= .003 and abs(state[7]) <= .05 and
        math.hypot(state[8], state[9]) <= .003 and gap >= .002)


def certificate_record(task, actor, sample=None):
    sample = task.error_context() if sample is None else sample
    result = task.certificate(actor, sample)
    return plain({"E": sample.error, "M": result.M, "L": result.lower,
        "V": result.value, "full_error_gradient": result.full_error_gradient,
        "claim": result.claim})


def nominal_prediction(task, actor, sample, result):
    """One selected nominal model step, never a physical rollout or a new plan."""
    import torch
    from push_any_shape.controller import torch_step
    from push_any_shape.integral_certificate import integrate_eta, integral_derivative, masked_error, pose_coordinates
    data, cfg = sample.control_context["data"], task.cfg
    e, eta, mask = data["e"], data["eta"], data["task_mask"]
    C, valid = sample.metric_context["C"], sample.metric_context["valid"]
    c = C[:, result.action["mode"]]
    speed = e.new_tensor([result.action["u"]])
    ep = torch_step(e, c, speed, cfg.dt, cfg.length_scale)
    D, Dp = pose_coordinates(e) * mask, pose_coordinates(ep) * mask
    eta_next = integrate_eta(eta, D, Dp, cfg.dt, task_mask=mask, goal_hold=data["goal_hold"], cfg=cfg)
    E_next = masked_error(ep, eta_next, mask)
    _, M_next = actor(E_next, C, valid)
    V_next = torch.einsum("bi,bij,bj->b", E_next, M_next, E_next)
    a = e[:, 2]
    flow_D = torch.stack(((a.cos() * c[:, 0] - a.sin() * c[:, 1]) / cfg.length_scale,
        (a.sin() * c[:, 0] + a.cos() * c[:, 1]) / cfg.length_scale,
        (a / 2).cos() * c[:, 2] / math.sqrt(2.)), -1) * speed[:, None] * mask
    flow_eta = integral_derivative(D, eta, task_mask=mask, goal_hold=data["goal_hold"], cfg=cfg)
    flow = cfg.dt * (result.certificate.full_error_gradient * torch.cat((flow_D, flow_eta), -1)).sum(-1)
    pose_next = [float(ep[0, 0]) * cfg.length_scale + float(task.target[0]),
        float(ep[0, 1]) * cfg.length_scale + float(task.target[1]), float(ep[0, 2])]
    return plain({"source": "nominal_ideal_no_slip_not_calibrated", "physical_rollout": False,
        "requested_dt": cfg.dt, "nominal_pose_next": pose_next, "nominal_E_next": E_next,
        "nominal_M_next": M_next, "nominal_V_next": V_next,
        "nominal_delta_V": V_next - result.certificate.value, "nominal_linearized_delta_V": flow})


def load_prefix(folder, fixture, binding, actor_sha):
    """Immutable data only; called after exclusive writer acquisition."""
    initial = read_json(folder / "INITIAL.json")
    if initial["fixture"] != fixture or initial["binding"] != binding or initial["actor_sha256"] != actor_sha:
        raise RuntimeError("Fixture initial state or execution binding changed")
    rows, previous = [], sha(folder / "INITIAL.json")
    commits = sorted(folder.glob("step_*.json"))
    attempts = sorted(folder.glob("attempt_*.json"))
    for index, path in enumerate(commits, 1):
        if path.name != f"step_{index:06d}.json":
            raise RuntimeError("Committed physical chain contains a gap")
        row = read_json(path)
        attempt_path = folder / f"attempt_{index:06d}.json"
        if row["step"] != index or row["fixture_name"] != fixture["name"] or row["previous_sha256"] != previous or row["actor_sha256"] != actor_sha or row["binding_sha256"] != digest(binding):
            raise RuntimeError("Committed physical chain/protocol/actor mismatch")
        if not attempt_path.is_file() or row["attempt_sha256"] != sha(attempt_path):
            raise RuntimeError("Missing or changed pre-execution marker")
        attempt = read_json(attempt_path)
        previous_snapshot = initial["snapshot"] if not rows else rows[-1]["snapshot_after"]
        if attempt["before_snapshot_sha256"] != digest(previous_snapshot):
            raise RuntimeError("Attempt marker does not bind the prior committed state")
        if abs(float(row["actual_dt"]) - (row["snapshot_after"]["integral"]["accepted_time_s"][0] - previous_snapshot["integral"]["accepted_time_s"][0])) > 1e-10:
            raise RuntimeError("Accepted physical/integral duration mismatch")
        rows.append(row)
        previous = sha(path)
    if len(attempts) != len(rows) or any(path.name != f"attempt_{index:06d}.json" for index, path in enumerate(attempts, 1)):
        raise RuntimeError("An uncommitted physical attempt exists; diagnose it without replaying")
    if list(folder.glob("*.pending")):
        raise RuntimeError("A pending evidence transaction exists; diagnose without deleting/replaying")
    return initial, rows, previous


def summarize(initial, rows, fixture):
    states = [initial["state"]] + [row["after_state"] for row in rows]
    times = [0.] + [row["accepted_time_s"] for row in rows]
    values = [float(initial["certificate"]["V"][0])] + [float(row["certificate_after"]["V"][0]) for row in rows]
    dt = [row["actual_dt"] for row in rows]
    goal = fixture["goal_xy"]
    costs = [physical_error(state, goal) for state in states]
    raw_cost = sum(cost * h for cost, h in zip(costs, dt))
    failure = bool(rows[-1]["terminated"]) if rows else False
    duration = times[-1]
    tail = max(costs[-1], 25.) * max(0., 30. - duration) if failure else 0.
    valid = [goal_conditions(state, goal) for state in states]
    final_hold = bool(len(rows) == 1500 and abs(duration - 30.) < 1e-8 and all(valid[-51:]) and not failure)
    since = None
    if final_hold:
        index = len(valid) - 1
        while index > 0 and valid[index - 1]:
            index -= 1
        since = times[index]
    dv = [right - left for left, right in zip(values, values[1:])]
    commands = [row["info"]["execution"]["command"] for row in rows]
    variation = sum(math.dist(left, right) for left, right in zip(commands, commands[1:]))
    phases = dict(Counter(row["info"]["execution"]["phase"] for row in rows))
    return {"fixture": fixture, "complete": failure or len(rows) == 1500,
        "success": final_hold, "failure": failure,
        "failure_reason": rows[-1]["info"]["execution"].get("failure_reason") if rows else None,
        "control_attempts": len(rows), "accepted_control_steps": sum(h > 0 for h in dt),
        "accepted_physical_substeps": sum(row["info"]["execution"]["accepted_substeps"] for row in rows),
        "physical_duration_s": duration, "intended_horizon_s": 30.,
        "raw_physical_cost": raw_cost, "failure_tail_cost": tail, "fixed_horizon_cost": raw_cost + tail,
        "terminal_physical_error_E": costs[-1], "terminal_position_error_m": math.dist(states[-1][:2], goal),
        "final_51_endpoint_hold": final_hold, "sustained_arrival_s_or_30": since if final_hold else 30.,
        "hold_confirmation_s_or_30": min(30., since + 1.) if final_hold else 30.,
        "V_initial": values[0], "V_final": values[-1],
        "actual_V_rises_tolerance_1e_10": sum(value > 1e-10 for value in dv),
        "actual_delta_V_max": max(dv, default=0.), "phases": phases,
        "no_descent_candidate_steps": sum(row["info"]["no_descent_candidate"] for row in rows),
        "command_total_variation_mps": variation,
        "response_calibrated": False, "metric": "identity I6 from new Cholesky actor; fixed weights",
        "optimizer_instances": 0, "optimizer_updates": 0,
        "scope": "bounded planar-contact engineering; no M-learning, stability proof or cross-shape contact claim"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute-reviewed-full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-wall-seconds", type=float, default=600.)
    args = parser.parse_args()
    args.output = resolve_project(args.output)
    if not 0. < args.max_wall_seconds <= 600.:
        raise ValueError("Maximum invocation duration must lie in (0,600] seconds")
    config, binding = verify_sources(args)
    resources = resource_status(args.output)
    if not resources["safe"] or any(path.exists() for path in stop_paths(args.output)):
        raise RuntimeError("Resource/STOP gate rejected execution")
    if args.preflight:
        print(json.dumps({"passed": True, "scope": "read_only_static_no_environment_import",
            "binding": binding, "resources": resources, "physical_execution": False}))
        return
    if not args.execute_reviewed_full:
        raise RuntimeError("Physical execution requires --execute-reviewed-full")
    certify_no_other_writer()
    existed = args.output.exists() and any(args.output.iterdir())
    if existed and not args.resume:
        raise RuntimeError("Existing output requires explicit --resume; no implicit replay")
    args.output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (args.output / "writer.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    # The held handle is the actual exclusivity guarantee; metadata is advisory.
    atomic_json(args.output / "writer_metadata.json", {"pid": os.getpid(), "utc": utc(), "binding": binding})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    for path in (PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control", EXTENSION):
        sys.path.insert(0, str(path))
    import torch
    from push_any_shape import IntegralConfig, IntegralMActor, make_task
    from push_any_shape.task import actor_digest
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(config["actor"]["seed"])
    cfg = IntegralConfig(**config["integral_config"])
    actor = IntegralMActor(cfg).to(device="cpu", dtype=torch.float64).eval()
    actor.requires_grad_(False)
    actor_path = args.output / "identity_actor.pt"
    if actor_path.exists():
        saved_actor = torch.load(actor_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(saved_actor["state_dict"])
        if saved_actor["binding_sha256"] != digest(binding):
            raise RuntimeError("Saved actor execution binding differs")
    else:
        temporary = actor_path.with_name(actor_path.name + ".pending")
        if temporary.exists():
            raise RuntimeError("Incomplete actor save requires diagnosis")
        torch.save({"schema": "disk_engineering_identity6_v1", "state_dict": actor.state_dict(),
            "binding_sha256": digest(binding), "torch_rng": torch.get_rng_state(), "optimizer_instances": 0}, temporary)
        os.rename(temporary, actor_path)
    actor_sha, actor_file_sha = actor_digest(actor), sha(actor_path)
    start_path = args.output / "START_MANIFEST.json"
    if start_path.exists():
        started = read_json(start_path)
        if started["binding"] != binding or started["actor_sha256"] != actor_sha or started["actor_file_sha256"] != actor_file_sha:
            raise RuntimeError("Existing run manifest changed")
    else:
        immutable_json(start_path, {"schema": "push_integral_disk_full_engineering_v1", "utc": utc(),
            "binding": binding, "config": config, "actor_sha256": actor_sha, "actor_file_sha256": actor_file_sha,
            "optimizer_instances": 0, "optimizer_updates": 0, "resources": resources})
    deadline = time.monotonic() + args.max_wall_seconds
    invocation = utc().replace(":", "").replace("+", "_")
    complete_summaries = []
    active_task = None
    try:
        for fixture_index, fixture in enumerate(config["fixtures"], 1):
            folder = args.output / f"{fixture_index:02d}_{fixture['name']}"
            folder.mkdir(exist_ok=True)
            initial_path = folder / "INITIAL.json"
            if initial_path.exists():
                initial, rows, previous = load_prefix(folder, fixture, binding, actor_sha)
            else:
                if any(folder.iterdir()):
                    raise RuntimeError("Fixture directory is nonempty without initial state")
                initial, rows, previous = None, [], None
            completed_path = folder / "COMPLETE.json"
            if completed_path.exists():
                completed = read_json(completed_path)
                if completed["last_commit_sha256"] != previous or completed["summary"] != summarize(initial, rows, fixture):
                    raise RuntimeError("Completed fixture chain/summary changed")
                complete_summaries.append(completed["summary"])
                continue  # No environment construction or reset for completed fixtures.
            if rows and (rows[-1]["terminated"] or len(rows) == 1500):
                summary = summarize(initial, rows, fixture)
                immutable_json(completed_path, {"utc": utc(), "summary": summary, "last_commit_sha256": previous,
                    "final_snapshot": rows[-1]["snapshot_after"], "actor_sha256": actor_sha,
                    "actor_file_sha256": actor_file_sha, "binding": binding})
                complete_summaries.append(summary)
                continue
            if any(path.exists() for path in stop_paths(args.output)) or time.monotonic() >= deadline:
                atomic_json(args.output / "status.json", {"state": "safe_boundary_stop", "utc": utc(), "completed_fixtures": len(complete_summaries)})
                return
            _, current_binding = verify_sources(args)
            if current_binding != binding or not resource_status(args.output)["safe"]:
                raise RuntimeError("Source/resource gate changed before fixture")
            active_task = make_task("Disk", "contact", cfg=cfg, max_steps=1500, dtype=torch.float64)
            if initial is None:
                options = {key: value for key, value in fixture.items() if key != "name"}
                state, reset_info = active_task.reset(seed=config["actor"]["seed"], options=options)
                initial = {"fixture": fixture, "binding": binding, "actor_sha256": actor_sha,
                    "state": plain(state), "reset_info": plain(reset_info),
                    "certificate": certificate_record(active_task, actor), "snapshot": active_task.snapshot(actor)}
                matrix = torch.as_tensor(initial["certificate"]["M"][0], dtype=torch.float64)
                if not torch.allclose(matrix, torch.eye(6, dtype=torch.float64), rtol=0., atol=1e-12):
                    raise RuntimeError("New actor is not the predeclared identity M")
                previous = immutable_json(initial_path, initial)
            else:
                active_task.restore(rows[-1]["snapshot_after"] if rows else initial["snapshot"], actor)
            for step_index in range(len(rows) + 1, 1501):
                if any(path.exists() for path in stop_paths(args.output)) or time.monotonic() >= deadline:
                    atomic_json(args.output / "status.json", {"state": "safe_boundary_stop", "utc": utc(),
                        "fixture": fixture["name"], "committed_attempts": len(rows)})
                    return
                if step_index == 1 or step_index % 50 == 0:
                    if not resource_status(args.output)["safe"]:
                        atomic_json(args.output / "status.json", {"state": "resource_safe_stop", "utc": utc(), "fixture": fixture["name"], "committed_attempts": len(rows)})
                        return
                    if actor_digest(actor) != actor_sha:
                        raise RuntimeError("Actor weights changed during zero-optimization engineering")
                if step_index % 250 == 0 and verify_sources(args)[1] != binding:
                    raise RuntimeError("Frozen execution sources changed")
                before_snapshot = active_task.snapshot(actor)
                sample = active_task.error_context()
                before_cert = certificate_record(active_task, actor, sample)
                control = active_task.control(actor, sample)
                nominal = nominal_prediction(active_task, actor, sample, control)
                attempt_path = folder / f"attempt_{step_index:06d}.json"
                attempt_sha = immutable_json(attempt_path, {"utc": utc(), "step": step_index,
                    "fixture_name": fixture["name"], "previous_sha256": previous,
                    "before_snapshot_sha256": digest(before_snapshot), "actor_sha256": actor_sha,
                    "binding_sha256": digest(binding), "action": plain(control.action)})
                state, reward, terminated, truncated, info = active_task.step(control, actor=actor)
                after_cert = certificate_record(active_task, actor)
                snapshot = active_task.snapshot(actor)
                if abs(before_cert["V"][0] - info["V_before"]) > 1e-12 or abs(after_cert["V"][0] - info["V_after"]) > 1e-12:
                    raise RuntimeError("Actual endpoint V differs from independently queried certificate")
                applicable = bool(info["execution"]["diagnostics"].get("nominal_C_prediction_applicable", False))
                nominal["applicable_execution_phase"] = applicable
                nominal["actual_pose_difference_from_nominal_m"] = math.dist(plain(state)[:2], nominal["nominal_pose_next"][:2])
                row = plain({"schema": "disk_physical_step_v1", "fixture_name": fixture["name"], "step": step_index,
                    "previous_sha256": previous, "attempt_sha256": attempt_sha, "binding_sha256": digest(binding),
                    "actor_sha256": actor_sha, "before_state": info["execution"]["before"], "after_state": state,
                    "action": control.action, "control_diagnostics": control.diagnostics,
                    "certificate_before": before_cert, "certificate_after": after_cert,
                    "eta_before": before_snapshot["integral"]["eta"], "eta_after": snapshot["integral"]["eta"],
                    "nominal_prediction": nominal, "info": info, "actual_dt": info["actual_dt"],
                    "accepted_time_s": info["accepted_time_s"], "reward_returned_by_task": reward,
                    "physical_error_before": physical_error(plain(info["execution"]["before"]), fixture["goal_xy"]),
                    "physical_error_after": physical_error(plain(state), fixture["goal_xy"]),
                    "terminated": bool(terminated), "truncated": bool(truncated), "snapshot_after": snapshot})
                previous = immutable_json(folder / f"step_{step_index:06d}.json", row)
                rows.append(row)
                if step_index % 50 == 0 or terminated or truncated:
                    atomic_json(args.output / "status.json", {"state": "running", "utc": utc(), "fixture": fixture["name"],
                        "committed_attempts": len(rows), "accepted_time_s": info["accepted_time_s"], "phase": info["execution"]["phase"]})
                if terminated or truncated:
                    break
            summary = summarize(initial, rows, fixture)
            if not summary["complete"]:
                raise RuntimeError("Fixture ended without physical failure or full horizon")
            immutable_json(completed_path, {"utc": utc(), "summary": summary, "last_commit_sha256": previous,
                "final_snapshot": rows[-1]["snapshot_after"], "actor_sha256": actor_sha,
                "actor_file_sha256": actor_file_sha, "binding": binding})
            complete_summaries.append(summary)
            active_task.close()
            active_task = None
        if actor_digest(actor) != actor_sha or verify_sources(args)[1] != binding:
            raise RuntimeError("Final actor/source audit failed")
        summary = {"schema": "disk_integral_engineering_full_v1", "complete": True, "utc": utc(),
            "fixtures": complete_summaries, "fixture_count": len(complete_summaries),
            "task_successes": sum(item["success"] for item in complete_summaries),
            "all_tasks_successful": all(item["success"] for item in complete_summaries),
            "binding": binding, "actor_sha256": actor_sha, "actor_file_sha256": actor_file_sha,
            "optimizer_instances": 0, "optimizer_updates": 0, "independent_final_audit": "pending"}
        final_path = args.output / "COMPLETE.json"
        if final_path.exists():
            previous_final = read_json(final_path)
            if any(previous_final[key] != summary[key] for key in (
                    "fixtures", "binding", "actor_sha256", "actor_file_sha256", "fixture_count", "task_successes")):
                raise RuntimeError("Existing final evidence differs from the verified committed chains")
            print(json.dumps({"already_complete": True, "physical_execution": False,
                "task_successes": summary["task_successes"], "fixture_count": 5}))
            return
        immutable_json(final_path, summary)
        atomic_json(args.output / "status.json", {"state": "complete", "utc": utc(), "fixture_count": 5,
            "task_successes": summary["task_successes"], "independent_final_audit": "pending"})
        print(json.dumps({"complete": True, "task_successes": summary["task_successes"], "fixture_count": 5}))
    except BaseException as exc:
        error = {"utc": utc(), "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
            "physical_replay_authorized": False, "pending_attempt_policy": "diagnose all uncommitted markers; do not delete or replay"}
        immutable_json(args.output / f"ERROR_{invocation}.json", error)
        raise
    finally:
        if active_task is not None:
            active_task.close()
        lock.close()


if __name__ == "__main__":
    main()
