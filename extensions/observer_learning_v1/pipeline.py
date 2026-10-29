"""Single-writer multiseed observer training, selection, testing and audit."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stabilization_gym.observation import make_observation_task
from stabilization_gym.observation.metrics import summarize_episode
from stabilization_learning.observation import MetricObserver, make_observer, reference_metric, value
from .learning import MetricNetwork, FrozenNetworkMetric, batch_data, episode_data, rollout
from .tensor_model import TensorModel

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def event(root, kind, **payload):
    path = Path(root)/"events"/f"{time.time_ns()}_{kind}.json"
    write(path, dict(event=kind, time=datetime.now(timezone(timedelta(hours=8))).isoformat(), **payload))


def source_files():
    files = [*sorted((ROOT/"extensions/observer_learning_v1").glob("*.py")),
        *sorted((ROOT/"stabilization-gym/stabilization_gym/observation").glob("*.py")),
        *sorted((ROOT/"stabilization-learning-control/stabilization_learning/observation").glob("*.py")),
        ROOT/"scripts/analysis/run_observer_learning_v1.py", ROOT/"configs/observer_learning_night_v1.json"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def check_sources(protocol):
    for name, digest in protocol["sources"].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError("Frozen observer source changed: "+name)


def stopped(root, deadline, parent_stop):
    if (root/"STOP").exists() or (parent_stop is not None and parent_stop.exists()):
        return "user_STOP"
    if time.time() >= deadline:
        return "protocol_time_limit"
    return None


def save_checkpoint(path, network, optimizer, **state):
    payload = dict(network=network.state_dict(), optimizer=optimizer.state_dict(),
                   torch_rng=torch.get_rng_state(), **state)
    temp = Path(str(path)+".tmp"); torch.save(payload, temp); temp.replace(path)


def validation(network, model, batch):
    with torch.no_grad():
        result = rollout(network, model, batch)
    if not torch.isfinite(result["loss"]):
        return None
    return {"successes": int(result["success"].sum()), "episodes": len(result["success"]),
            "mean_cost": float(result["cost"].mean()), "terminal_mse": float(result["terminal_mse"].mean()),
            "certificate_loss": float(result["certificate_loss"])}


def score(row):
    return (row["successes"], -row["mean_cost"])


def train_job(root, task, seed, config, protocol, deadline, parent_stop):
    folder = root/"jobs"/f"{task}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"SELECTION.json").exists():
        selection = json.loads((folder/"SELECTION.json").read_text(encoding="utf-8"))
        if sha(folder/"selected.pt") != selection["sha256"]:
            raise RuntimeError("Completed selected checkpoint changed")
        return selection
    torch.manual_seed(seed)
    model = make_observation_task(task).model
    tensor_model = TensorModel(model, config["device"])
    network = MetricNetwork(model, config["hidden"], config["metric_radius"]).to(config["device"])
    optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
    development = config.get("development_only", False)
    val_episodes = [episode_data(task, "engineering" if development else "validation", i+(1200 if development else 0), s)
                    for s in config["scenarios"] for i in config["validation_indices"]]
    val_batch = batch_data(model, val_episodes, config["device"])
    validation_rows = []; training_rows = []; next_update = 0; rejected = 0
    if (folder/"last.pt").exists():
        saved = torch.load(folder/"last.pt", map_location=config["device"], weights_only=False)
        if saved["protocol_sha256"] != sha(root/"protocol.json"):
            raise RuntimeError("Whole-update resume protocol changed")
        network.load_state_dict(saved["network"]); optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"]); next_update = saved["next_update"]
        best, best_update, best_state = saved["best"], saved["best_update"], saved["best_state"]
        validation_rows, training_rows = saved["validation_rows"], saved["training_rows"]
    else:
        best = validation(network, tensor_model, val_batch)
        if best is None: raise RuntimeError("Initial validation is nonfinite")
        best_update = 0; best_state = deepcopy(network.state_dict())
        validation_rows.append(dict(update=0, **best))
        torch.save({"network": best_state, "update": 0}, folder/"initial.pt")
    event(root, "job_started", task=task, seed=seed, next_update=next_update)
    for update in range(next_update, config["updates"]):
        reason = stopped(root, deadline, parent_stop)
        if reason:
            save_checkpoint(folder/"last.pt", network, optimizer, protocol_sha256=sha(root/"protocol.json"),
                next_update=update, best=best, best_update=best_update, best_state=best_state,
                validation_rows=validation_rows, training_rows=training_rows)
            event(root, "stopped", task=task, seed=seed, reason=reason, next_update=update)
            raise InterruptedError(reason)
        data = [episode_data(task, "engineering" if development else "train", (4000 if development else 0)+(seed-41)*config["training_seed_stride"]+update*config["batch_size"]+b,
                             config["scenarios"][(update+b+seed)%len(config["scenarios"])]) for b in range(config["batch_size"])]
        batch = batch_data(model, data, config["device"])
        optimizer.zero_grad(set_to_none=True)
        result = rollout(network, tensor_model, batch)
        finite = bool(torch.isfinite(result["loss"]))
        if finite:
            result["loss"].backward()
            finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in network.parameters())
        if not finite:
            rejected += 1
            event(root, "nonfinite_candidate", task=task, seed=seed, update=update+1, consecutive=rejected)
            if rejected >= config["maximum_consecutive_nonfinite"]:
                raise RuntimeError("Consecutive nonfinite gradients; preserving last completed boundary")
            training_rows.append(dict(update=update+1, rejected_nonfinite=True))
            continue
        rejected = 0
        torch.nn.utils.clip_grad_norm_(network.parameters(), config["gradient_clip"])
        optimizer.step()
        training_rows.append(dict(update=update+1, loss=float(result["loss"].detach()),
            mean_cost=float(result["cost"].mean().detach()), certificate_loss=float(result["certificate_loss"].detach())))
        del result, batch
        if (update+1)%config["validation_every"] == 0 or update+1 == config["updates"]:
            check_sources(protocol)
            row = validation(network, tensor_model, val_batch)
            if row is not None:
                validation_rows.append(dict(update=update+1, **row))
                if score(row) > score(best):
                    best, best_update, best_state = row, update+1, deepcopy(network.state_dict())
            save_checkpoint(folder/"last.pt", network, optimizer, protocol_sha256=sha(root/"protocol.json"),
                next_update=update+1, best=best, best_update=best_update, best_state=best_state,
                validation_rows=validation_rows, training_rows=training_rows)
            write(folder/"TRAINING.json", {"training": training_rows, "validation": validation_rows})
            event(root, "validation_committed", task=task, seed=seed, update=update+1,
                  validation=row, best_update=best_update, best=best)
    # Last validation exists even if the final attempted update was rejected.
    save_checkpoint(folder/"last.pt", network, optimizer, protocol_sha256=sha(root/"protocol.json"),
        next_update=config["updates"], best=best, best_update=best_update, best_state=best_state,
        validation_rows=validation_rows, training_rows=training_rows)
    torch.save({"network": best_state, "update": best_update, "protocol_sha256": sha(root/"protocol.json")}, folder/"selected.pt")
    np.savez(folder/"selected_numpy_weights.npz", **{k: v.detach().cpu().numpy() for k, v in best_state.items()})
    selection = dict(task=task, seed=seed, best_update=best_update, best_validation=best,
        attempted_updates=config["updates"], optimizer_updates=sum(not r.get("rejected_nonfinite", False) for r in training_rows),
        sha256=sha(folder/"selected.pt"), numpy_weights_sha256=sha(folder/"selected_numpy_weights.npz"))
    write(folder/"TRAINING.json", {"training": training_rows, "validation": validation_rows})
    write(folder/"SELECTION.json", selection)
    event(root, "selection_committed", **selection)
    return selection


def numpy_matrix(weights, estimate, radius):
    features = np.tanh(estimate/weights["feature_scale"])
    hidden = np.tanh(features@weights["net.0.weight"].T+weights["net.0.bias"])
    raw = hidden@weights["net.2.weight"].T+weights["net.2.bias"]
    diagonal = weights["diagonal"].astype(bool)
    off = np.where(diagonal, 0., np.tanh(raw)); off = .25*off/(1+np.linalg.norm(off))
    factor = np.eye(weights["base_cholesky"].shape[0])
    factor[weights["row"], weights["column"]] += off
    factor = np.exp(radius*np.tanh(raw[diagonal]/radius))[:, None]*factor
    composed = weights["base_cholesky"]@factor
    return composed@composed.T


def evaluate_policy(root, task, name, seed, config, selection=None, *, deadline=None, parent_stop=None):
    gate_path = root/"GLOBAL_SELECTION_COMMITTED.json"
    if not gate_path.is_file():
        raise RuntimeError("All observer selections must be frozen before test access")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected_jobs = {f"{t}_seed{s}" for t in config["tasks"] for s in config["seeds"]}
    if set(gate) != expected_jobs:
        raise RuntimeError("Incomplete global observer selection commitment")
    for job, row in gate.items():
        if sha(root/"jobs"/job/"selected.pt") != row["sha256"]:
            raise RuntimeError("An observer checkpoint changed after global selection")
    model = make_observation_task(task).model
    weights = None; selected = None
    if selection:
        folder = root/"jobs"/f"{task}_seed{seed}"
        if sha(folder/"selected.pt") != selection["sha256"]: raise RuntimeError("Selected weights changed")
        if sha(folder/"selected_numpy_weights.npz") != selection["numpy_weights_sha256"]:
            raise RuntimeError("Independent NumPy weights changed after selection")
        network = MetricNetwork(model, config["hidden"], config["metric_radius"])
        selected = torch.load(folder/"selected.pt", map_location="cpu", weights_only=False)
        network.load_state_dict(selected["network"])
        metric = FrozenNetworkMetric(network)
        with np.load(folder/"selected_numpy_weights.npz", allow_pickle=False) as payload:
            weights = {k: payload[k] for k in payload.files}
    else:
        metric = reference_metric(model)
    version = metric.version
    rows = []; audit_max = dict(matrix=0., correction=0., certificate=0., prediction=0.)
    for scenario in config["test_scenarios"] + (config["pose_test_stress"] if task == "cooperative_pose" else []):
        for index in config["test_indices"]:
            reason = stopped(root, deadline if deadline is not None else float("inf"), parent_stop)
            if reason:
                raise InterruptedError(reason)
            env = make_observation_task(task); obs, _ = env.reset(options={"split": "engineering" if config.get("development_only") else "test",
                "index": index+(2200 if config.get("development_only") else 0), "scenario": scenario})
            observer = MetricObserver(env.model, metric) if selection else make_observer(name, env.model)
            records = []; corrections_ms = []
            while True:
                before = time.perf_counter(); action = observer.correction(obs); corrections_ms.append((time.perf_counter()-before)*1000)
                if selection:
                    independent = numpy_matrix(weights, obs["estimate"], config["metric_radius"])
                    audit_max["matrix"] = max(audit_max["matrix"], float(np.max(np.abs(independent-metric.matrix(obs["estimate"])))) )
                    h = env.model.measurement_jacobian(obs["estimate"])*np.asarray(env.model.spec.error_scale)[None, :]
                    residual = (obs["measurement"]-env.model.measure(obs["estimate"]))*obs["measurement_mask"]
                    expected = env.model.spec.dt*np.asarray(env.model.spec.error_scale)*np.linalg.solve(independent, h.T@(obs["confidence"]*residual/np.asarray(env.model.spec.measurement_scale)**2))
                    audit_max["correction"] = max(audit_max["correction"], float(np.max(np.abs(expected-action))))
                obs, _, terminated, truncated, _ = env.step(action)
                record = env.evaluation_record(); records.append(record)
                if len(records) > 1:
                    expected = env.model.predict(records[-2]["estimate"], records[-2]["observation"]["known_input"])
                    audit_max["prediction"] = max(audit_max["prediction"], float(np.max(np.abs(expected-record["observation"]["estimate"]))))
                if terminated or truncated: break
            summary = summarize_episode(env.model, records, lambda x,z,v: value(env.model, x,z,metric,v),
                certificate_version=version, noise_allowance_per_second=0. if scenario=="nominal" else .01)
            errors = np.array([r["error"] for r in records]); estimates = np.array([r["estimate"] for r in records])
            matrices = np.array([metric.matrix(z) for z in estimates])
            vv = np.einsum("ni,nij,nj->n", errors/np.asarray(model.spec.error_scale), matrices, errors/np.asarray(model.spec.error_scale))
            for r, vv_i in zip(records, vv):
                audit_max["certificate"] = max(audit_max["certificate"], abs(vv_i-value(env.model,r["truth"],r["estimate"],metric,version)))
            tail_count = max(2, round(model.spec.hold_seconds/model.spec.dt))
            tail = np.linalg.norm(errors[-tail_count:]/np.asarray(model.spec.error_scale), axis=1)
            filename = f"test/{task}/{name}_seed{seed}_{scenario}_{index}.npz"
            path = root/filename; path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists(): raise FileExistsError("Never overwrite a partially tested group")
            np.savez_compressed(path, time=np.array([r["time"] for r in records]), truth=np.array([r["truth"] for r in records]),
                estimate=estimates, error=errors, matrix=matrices, value=vv,
                correction=np.array([r["correction"] for r in records]),
                measurement=np.array([r["observation"]["measurement"] for r in records]),
                mask=np.array([r["observation"]["measurement_mask"] for r in records]),
                known_input=np.array([r["observation"]["known_input"] for r in records]))
            summary.update(task=task, method=name, seed=seed, scenario=scenario, test_index=index,
                empirical_terminal_error_p95=float(np.quantile(tail,.95)), empirical_terminal_error_max=float(tail.max()),
                empirical_bound_is_not_a_proof=True, matrix_span=float(np.max(np.abs(matrices-matrices[0]))),
                mean_inference_ms=float(np.mean(corrections_ms)), trajectory=filename, trajectory_sha256=sha(path),
                replay_sha256=env.supervision()["episode"]["replay_sha256"])
            rows.append(summary)
    audit = {"passed": all(v<1e-6 for v in audit_max.values()), "maxima": audit_max,
             "independent_numpy_network_and_action": selection is not None, "episodes": len(rows)}
    write(root/"test"/task/f"{name}_seed{seed}_AUDIT.json", audit)
    if not audit["passed"]: raise RuntimeError("Independent NumPy consistency failed")
    event(root, "test_group_committed", task=task, method=name, seed=seed, episodes=len(rows), audit=audit)
    return rows


def run(root, config, *, parent_stop=None, resume=False):
    root = Path(root); parent_stop = Path(parent_stop) if parent_stop else None
    torch.set_num_threads(config["cpu_threads"]); torch.use_deterministic_algorithms(True)
    if (root/"STOP").exists() or (root/"FINAL_STATE.json").exists():
        raise RuntimeError("Existing STOP/final result prohibits automatic restart")
    if root.exists() and not resume:
        raise FileExistsError("Use a fresh output")
    if resume:
        protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
        if protocol["config"] != config: raise RuntimeError("Resume configuration differs")
        if (root/"test").exists(): raise RuntimeError("Partial tests require separate analysis; no overwrite/reselection")
        check_sources(protocol)
    else:
        root.mkdir(parents=True)
        protocol = dict(schema="observer-learning-night-v1", config=config, sources=source_files(),
            deadline_unix=time.time()+config["maximum_hours"]*3600,
            task_specs={name:asdict(make_observation_task(name).model.spec) for name in config["tasks"]},
            data_rule="Whole-trajectory split; shared paired tests across seeds, not new independent starts",
            metric_input="Estimated state only; measurement residual/mask used by the gain, never true error",
            action="Internal observer tangent correction; object motion independent of the observer",
            certificate="Full successor e^T M(xhat) e; sampled evidence, no nonlinear domain proof")
        write(root/"protocol.json", protocol)
        for name in protocol["sources"]:
            path = root/"sources"/name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name,path)
    event(root, "started", pid=__import__("os").getpid(), sources=len(protocol["sources"]))
    selections = []
    try:
        for task in config["tasks"]:
            for seed in config["seeds"]:
                selections.append(train_job(root,task,seed,config,protocol,protocol["deadline_unix"],parent_stop))
        commitment = {f"{r['task']}_seed{r['seed']}":r for r in selections}
        write(root/"GLOBAL_SELECTION_COMMITTED.json", commitment)
        event(root,"all_selections_frozen", jobs=len(selections))
        rows=[]
        for task in config["tasks"]:
            for method in config["baselines"]+(config["pose_baselines"] if task=="cooperative_pose" else []):
                reason=stopped(root,protocol["deadline_unix"],parent_stop)
                if reason: raise InterruptedError(reason)
                rows.extend(evaluate_policy(root,task,method,0,config,deadline=protocol["deadline_unix"],parent_stop=parent_stop))
            for selection in [s for s in selections if s["task"]==task]:
                reason=stopped(root,protocol["deadline_unix"],parent_stop)
                if reason: raise InterruptedError(reason)
                rows.extend(evaluate_policy(root,task,"learned_M",selection["seed"],config,selection,deadline=protocol["deadline_unix"],parent_stop=parent_stop))
        check_sources(protocol)
        write(root/"RESULTS.json", {"selections":selections,"episodes":rows,"training_updates":sum(s["optimizer_updates"] for s in selections),
            "scope":"5 seeds on 3 observation tasks; shared paired starts; conditional on fixed task protocol",
            "no_stability_proof":True})
        from .report import create_report
        create_report(root, config, rows, selections)
        write(root/"FINAL_STATE.json", {"status":"complete","jobs":len(selections),"test_episodes":len(rows),
            "finished":datetime.now(timezone(timedelta(hours=8))).isoformat()})
        event(root,"complete",jobs=len(selections),episodes=len(rows))
    except InterruptedError as exc:
        write(root/"STOP_STATE.json", {"reason":str(exc),"resume_boundary":"whole_update_only_before_any_test",
            "resumable":not (root/"STOP").exists() and not (root/"test").exists() and str(exc)!="protocol_time_limit"})
        raise
    except Exception as exc:
        import traceback
        write(root/"ERROR.json", {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        event(root,"failed",error=str(exc)); raise
