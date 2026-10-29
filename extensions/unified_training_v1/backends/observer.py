"""Staged full-sequence observer-M learning with sensor/estimator separation.

The preserved learner has no AE and no critic. Training truth is used only in
loss/evaluation. Deployment evaluates M on the public estimator state.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from stabilization_gym.observation import make_observation_task
from stabilization_gym.observation.metrics import summarize_episode
from stabilization_gym.tasks_v2.state_observer import ObservationReplayEnv, ObserverTaskAdapter
from stabilization_learning.observation import reference_metric, value
from extensions.observer_learning_v1.learning import MetricNetwork, FrozenNetworkMetric, episode_data, batch_data, rollout
from extensions.observer_learning_v1.tensor_model import TensorModel
from extensions.observer_learning_v1.pipeline import numpy_matrix


TASKS = ("motor_speed", "cartpole_output", "cooperative_pose")
ROOT = Path(__file__).resolve().parents[3]


def _digest(network):
    result = hashlib.sha256()
    for key, tensor in network.state_dict().items():
        result.update(key.encode()); result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


class _FixedMetric:
    def __init__(self, matrix, label):
        self.base = np.array(matrix, dtype=np.float64); self.label = label
        self.version = hashlib.sha256(self.base.tobytes()+label.encode()).hexdigest()
        eig = np.linalg.eigvalsh(self.base); self.bounds = (float(eig.min()), float(eig.max()))
    def matrix(self, estimate): return self.base.copy()


class Backend:
    family = "observer"
    def __init__(self, config, output, event=None, deadline=None):
        self.config = dict(config); self.output = Path(output)
        self.event = event or (lambda *a, **kw: None); self.deadline = deadline
        self.name = self.config.get("task", "motor_speed")
        if self.name not in TASKS: raise ValueError("Unknown observer task")
        self.updates = 0; self.train_index = int(self.config.get("training_index_start", 60000))
        self.collected_transitions = 0; self.history = []; self.pending = []
        self.initialized = False

    def guard(self):
        if any((p/"STOP").exists() for p in (self.output, self.output.parent, self.output.parent.parent)):
            raise InterruptedError("STOP")
        if self.deadline is not None and time.time() >= self.deadline:
            raise InterruptedError("protocol_deadline")

    def source_files(self):
        folders = (ROOT/"extensions/observer_learning_v1",
            ROOT/"stabilization-gym/stabilization_gym/observation",
            ROOT/"stabilization-learning-control/stabilization_learning/observation")
        return [str(Path(__file__).resolve()), str(ROOT/"stabilization-gym/stabilization_gym/tasks_v2/state_observer.py")]+[
            str(p) for folder in folders for p in sorted(folder.glob("*.py"))]

    def initialize(self):
        self.guard()
        if self.initialized: raise RuntimeError("Backend already initialized")
        torch.set_num_threads(int(self.config.get("cpu_threads", 1)))
        torch.manual_seed(int(self.config.get("seed", 81)))
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = make_observation_task(self.name).model
        self.tensor_model = TensorModel(self.model, "cpu")
        self.network = MetricNetwork(self.model, int(self.config.get("hidden", 32)),
            float(self.config.get("metric_radius", 3.)))
        self.initial_state = deepcopy(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.config.get("learning_rate", .003)))
        self.scenarios = tuple(self.config.get("scenarios", ("nominal", "noise", "occlusion", "mismatch")))
        self.validation_descriptors = self._descriptors("validation")
        self.validation_batch = batch_data(self.model, [episode_data(self.name, "validation", i, s)
            for i,s in self.validation_descriptors])
        self.initialized = True
        return dict(task=self.name, algorithm="preserved_full_sequence_observer_model_gradient",
            initialization="reference fixed-M residual initialization; random hidden layer, zero output layer; empty Adam",
            representation="public estimate and sensor residual; no feature learner", evaluator="none",
            initial_parameter_hash=_digest(self.network), horizon_seconds=self.model.spec.horizon,
            horizon_steps=round(self.model.spec.horizon/self.model.spec.dt), dt=self.model.spec.dt,
            validation_descriptors=[dict(index=i,scenario=s) for i,s in self.validation_descriptors],
            test_descriptors=[dict(index=i,scenario=s) for i,s in self._descriptors("test")],
            certificate_factor=1., truth_usage="supervised loss/evaluation only", action_kind="estimator_update")

    def _descriptors(self, split):
        count = int(self.config.get(split+"_count", 4 if split == "validation" else 8))
        start = int(self.config.get(split+"_index_start", 60000))
        scenarios = tuple(self.config.get(split+"_scenarios", self.config.get("scenarios", ("nominal","noise","occlusion","mismatch"))))
        if not scenarios or count <= 0 or start < 0 or start+count > 100000: raise ValueError("Invalid episode split budget")
        return [(start+k, scenarios[k%len(scenarios)]) for k in range(count)]

    def _fresh(self):
        self.guard(); count = int(self.config.get("batch_size", 2)); episodes = []
        for b in range(count):
            index = self.train_index; self.train_index += 1
            scenario = self.scenarios[index%len(self.scenarios)]
            episodes.append(episode_data(self.name, "train", index, scenario))
        self.collected_transitions += sum(len(e["truth"])-1 for e in episodes)
        return episodes

    def collect(self, initial=False):
        self.guard()
        fresh = self._fresh(); self.pending.extend(fresh)
        return dict(initial=bool(initial), episodes=len(fresh),
            new_transitions=sum(len(e["truth"])-1 for e in fresh),
            total_transitions=self.collected_transitions,
            sequences=[dict(index=e["index"], scenario=e["scenario"], replay_sha256=e["replay_sha256"]) for e in fresh],
            physical_action="declared known input", estimator_rollout="recomputed with current M during updates")

    def warmup_representation(self, updates):
        return dict(skipped=True, requested_updates=int(updates), actual_updates=0, reason="no independent feature extractor")

    def pretrain_evaluator(self, updates):
        return dict(skipped=True, requested_updates=int(updates), actual_updates=0, reason="known differentiable estimator model; no critic")

    def train_epoch(self, epoch, representation_updates=0, evaluator_updates=0, certificate_updates=4):
        rows = []
        for _ in range(int(certificate_updates)):
            self.guard(); start = time.perf_counter()
            count = int(self.config.get("batch_size", 2))
            if self.pending:
                episodes = self.pending[:count]; self.pending = self.pending[count:]
            else:
                episodes = self._fresh()
            batch = batch_data(self.model, episodes); before = _digest(self.network)
            self.optimizer.zero_grad(set_to_none=True)
            result = rollout(self.network, self.tensor_model, batch)
            if not torch.isfinite(result["loss"]): raise FloatingPointError("Nonfinite observer loss before update")
            result["loss"].backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                raise FloatingPointError("Nonfinite observer gradient before update")
            norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(self.config.get("gradient_clip", 2.)))
            self.guard(); self.optimizer.step(); self.updates += 1
            with torch.no_grad():
                probe = self.network(torch.as_tensor(self.model.initial_estimate(), dtype=torch.float64))
            row = dict(epoch=int(epoch), update=self.updates, loss=float(result["loss"].detach()),
                mean_cost=float(result["cost"].mean().detach()), certificate_loss=float(result["certificate_loss"].detach()),
                gradient_norm=float(norm), parameters_changed=before != _digest(self.network),
                collected_transitions=self.collected_transitions, wall_seconds=time.perf_counter()-start,
                fixed_public_estimate_M=probe.tolist(),
                training_episodes=[dict(index=e["index"], scenario=e["scenario"], replay_sha256=e["replay_sha256"]) for e in episodes])
            rows.append(row); self.history.append(row); self.event("certificate_update", task=self.name, **row)
        return dict(epoch=int(epoch), actual_updates=self.updates, certificate_updates=len(rows),
            representation_updates=0, evaluator_updates=0, rows=rows)

    def validate(self):
        self.guard()
        with torch.no_grad(): result=rollout(self.network, self.tensor_model, self.validation_batch)
        if not torch.isfinite(result["loss"]): raise FloatingPointError("Nonfinite validation")
        rows=[dict(index=i,scenario=s,success=bool(result["success"][k]),
            cost=float(result["cost"][k]),terminal_mse=float(result["terminal_mse"][k]))
            for k,(i,s) in enumerate(self.validation_descriptors)]
        summary=dict(episodes=len(rows),successes=sum(r["success"] for r in rows),
            success_rate=float(np.mean([r["success"] for r in rows])),mean_cost=float(result["cost"].mean()),
            mean_terminal_error=float(result["terminal_mse"].mean()),certificate_loss=float(result["certificate_loss"]))
        return dict(summary=summary,rows=rows,selection_score=[summary["successes"],-summary["mean_cost"]],
            parameter_hash=_digest(self.network))

    def save(self, path):
        self.guard(); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError("Never overwrite a checkpoint: "+str(path))
        payload=dict(config=self.config, network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),
            initial_state=self.initial_state, updates=self.updates,train_index=self.train_index,
            collected_transitions=self.collected_transitions,pending=self.pending,history=self.history,
            torch_rng=torch.get_rng_state(),parameter_hash=_digest(self.network))
        temporary=path.with_suffix(path.suffix+".tmp");torch.save(payload,temporary);temporary.replace(path)
        return dict(path=str(path),parameter_hash=payload["parameter_hash"],actual_updates=self.updates)

    def load(self, path):
        self.guard(); payload=torch.load(path,map_location="cpu",weights_only=False)
        if payload["config"] != self.config: raise ValueError("Backend config mismatch")
        self.network.load_state_dict(payload["network"]);self.optimizer.load_state_dict(payload["optimizer"])
        for key in ("initial_state","updates","train_index","collected_transitions","pending","history"):
            setattr(self,key,payload[key])
        torch.set_rng_state(payload["torch_rng"])
        if _digest(self.network)!=payload["parameter_hash"]: raise RuntimeError("Checkpoint parameter digest mismatch")
        return dict(parameter_hash=_digest(self.network),actual_updates=self.updates)

    def evaluate(self, split, variants=("selected","initial","identity")):
        if split not in ("validation","test"): raise ValueError("Unknown split")
        self.guard()
        if split=="test":
            frozen=self.output/"FROZEN.json";global_file=self.output.parent.parent/"GLOBAL_FROZEN.json"
            if not frozen.is_file() or not global_file.is_file():
                raise RuntimeError("All jobs must freeze before test")
            gate=json.loads(frozen.read_text(encoding="utf-8"));global_gate=json.loads(global_file.read_text(encoding="utf-8"))
            job_protocol=json.loads((self.output/"protocol.json").read_text(encoding="utf-8"))
            if (global_gate.get("jobs",{}).get(self.output.name)!=hashlib.sha256(frozen.read_bytes()).hexdigest()
                    or gate.get("protocol_sha256")!=hashlib.sha256((self.output/"protocol.json").read_bytes()).hexdigest()
                    or global_gate.get("protocol_sha256")!=job_protocol.get("suite_protocol_sha256")):
                raise RuntimeError("Job freeze is not in the committed experiment group")
            if hashlib.sha256((self.output/gate["checkpoint"]).read_bytes()).hexdigest()!=gate["sha256"]:
                raise RuntimeError("Frozen checkpoint changed")
            saved=torch.load(self.output/gate["checkpoint"],map_location="cpu",weights_only=False)
            if saved["parameter_hash"]!=_digest(self.network):
                raise RuntimeError("Load the selected frozen model before testing")
        selected=deepcopy(self.network.state_dict());before=_digest(self.network);groups={};audits={}
        prior_training_mode=self.network.training
        try:
            for variant in variants:
                if variant in ("selected","initial"):
                    self.network.load_state_dict(selected if variant=="selected" else self.initial_state)
                    metric=FrozenNetworkMetric(self.network)
                    weights={k:v.detach().cpu().numpy() for k,v in self.network.state_dict().items()}
                    independent=lambda estimate: numpy_matrix(weights,estimate,self.network.radius)
                    label="learned_M" if variant=="selected" else "initial_nominal_fixed_M"
                elif variant in ("identity","fixed_M"):
                    matrix=np.eye(self.model.spec.error_dim) if variant=="identity" else reference_metric(self.model).base
                    metric=_FixedMetric(matrix,variant);independent=lambda estimate,m=matrix: m.copy()
                    label="literal_identity_M" if variant=="identity" else "reference_fixed_M"
                else: raise ValueError("Unknown variant: "+variant)
                rows=[];maxima=dict(matrix=0.,correction=0.,certificate=0.)
                for index,scenario in self._descriptors(split):
                    self.guard();env=ObservationReplayEnv(self.name)
                    observation,_=env.reset(options=dict(split=split,index=index,scenario=scenario))
                    adapter=ObserverTaskAdapter(env.model);memory=adapter.initial_memory(scenario=scenario)
                    records=[];version=metric.version;dt=env.model.spec.dt
                    while True:
                        self.guard();labels=env.supervision();context=adapter.make_context(observation,memory=memory)
                        plan=adapter.construct_action(context,metric)
                        score=adapter.evaluate_supervision(labels,context,plan)
                        prior=np.asarray(memory["estimate"]);matrix=independent(prior)
                        maxima["matrix"]=max(maxima["matrix"],float(np.max(np.abs(matrix-metric.matrix(prior)))))
                        mask=observation["measurement_mask"].astype(bool)
                        h=env.model.measurement_jacobian(prior)[mask]*np.asarray(env.model.spec.error_scale)[None,:]
                        residual=(observation["measurement"]-env.model.measure(prior))[mask]
                        expected=dt*np.asarray(env.model.spec.error_scale)*np.linalg.solve(matrix,h.T@(residual/np.asarray(env.model.spec.measurement_scale)[mask]**2))
                        correction=plan.action.detach().numpy()
                        maxima["correction"]=max(maxima["correction"],float(np.max(np.abs(expected-correction))))
                        record=dict(time=float(observation["time"][0]),truth=labels["truth"],estimate=score["estimate"],
                            error=score["error"],correction=correction,observation=dict(observation,estimate=prior),
                            failure=score["failure"],complete=score["complete"],failure_reason="model_domain_exit" if score["failure"] else None)
                        records.append(record);memory=plan.memory
                        if score["failure"] or score["complete"]: break
                        observation,_,terminated,truncated,_=env.step(observation["known_input"])
                        if terminated or truncated: break
                    summary=summarize_episode(env.model,records,lambda x,z,v:value(env.model,x,z,metric,v),
                        certificate_version=version,noise_allowance_per_second=0. if scenario=="nominal" else .01)
                    errors=np.array([r["error"] for r in records])/np.asarray(env.model.spec.error_scale)
                    estimates=np.array([r["estimate"] for r in records])
                    matrices=np.array([independent(z) for z in estimates])
                    vv=np.einsum("ni,nij,nj->n",errors,matrices,errors)
                    reference=np.array([value(env.model,r["truth"],r["estimate"],metric,version) for r in records])
                    maxima["certificate"]=max(maxima["certificate"],float(np.max(np.abs(vv-reference))))
                    folder=self.output/"trajectories"/split/variant;folder.mkdir(parents=True,exist_ok=True)
                    target=folder/f"{scenario}_{index}.npz"
                    if target.exists(): raise FileExistsError("Never overwrite frozen evaluation: "+str(target))
                    np.savez_compressed(target,time=np.array([r["time"] for r in records]),truth=np.array([r["truth"] for r in records]),
                        estimate=estimates,normalized_error=errors,matrix=matrices,value=vv,
                        correction=np.array([r["correction"] for r in records]),
                        measurement=np.array([r["observation"]["measurement"] for r in records]),
                        known_input=np.array([r["observation"]["known_input"] for r in records]))
                    summary.update(index=index,scenario=scenario,variant=variant,variant_meaning=label,
                        trajectory=str(target.relative_to(self.output)),replay_sha256=env.supervision()["episode"]["replay_sha256"],
                        correction_square_integral=float(dt*np.sum(np.array([r["correction"] for r in records[:-1]])**2)),
                        correction_square_sum=float(np.sum(np.array([r["correction"] for r in records])**2)),
                        input_scope="left-rectangle time proxy of estimator correction; separate all-update squared sum; not actuator energy")
                    rows.append(summary)
                summary=dict(episodes=len(rows),successes=sum(r["success"] for r in rows),failures=sum(r["failure"] for r in rows),
                    success_rate=float(np.mean([r["success"] for r in rows])),mean_cost=float(np.mean([r["fixed_horizon_error_cost"] for r in rows])),
                    mean_input_square_integral=float(np.mean([r["correction_square_integral"] for r in rows])),
                    mean_V_rise_fraction=float(np.mean([r["certificate"]["raw_increase_count"]/max(1,r["certificate"]["transitions"]) for r in rows])))
                groups[variant]=dict(summary=summary,rows=rows,variant_meaning=label)
                audits[variant]=dict(maxima=maxima,passed=all(v<1e-6 for v in maxima.values()),
                    full_successor_metric=True,independent_numpy_network=True,physical_correction_separated=True)
        finally:
            self.network.load_state_dict(selected)
            self.network.train(prior_training_mode)
        if _digest(self.network)!=before: raise RuntimeError("Evaluation changed selected parameters")
        paired={}
        if "selected" in groups:
            for variant,group in groups.items():
                if variant=="selected":continue
                pairs=[(a,b) for a,b in zip(groups["selected"]["rows"],group["rows"]) if a["success"] and b["success"]]
                paired[variant]=dict(common_successes=len(pairs),selected_minus_baseline_arrival_seconds=
                    float(np.mean([a["arrival_time"]-b["arrival_time"] for a,b in pairs])) if pairs else None)
        return dict(task=self.name,split=split,groups=groups,audit=audits,paired_time=paired,
            scope="same public sensor sequences; candidate certificate; true state used only for evaluation")


ObserverBackend = Backend
