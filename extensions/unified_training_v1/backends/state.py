"""Staged adapter for the existing full-horizon state-M learner.

No AE or value critic is invented. Original dynamics, objective, feasibility
gates, and whole-update rollback are retained. The initial M is nominal LQR.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from stabilization_gym import make
from stabilization_gym.envs.classic_control.sdc_full_horizon import FullHorizonSDCTask
from stabilization_learning.policies.state_metric_residual_v3 import AnchoredStateMetricPolicy
from stabilization_learning.algos.pytorch.state_metric_full_horizon import FullHorizonConfig, FullHorizonMetricLearner
from stabilization_learning.algos.pytorch.state_metric_full_horizon.evaluation import evaluate as evaluate_state
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash


TASKS = ("coupled_linear", "point_mass", "pendulum", "cartpole")
ROOT = Path(__file__).resolve().parents[3]


def _digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _summary(rows):
    return dict(episodes=len(rows), successes=sum(r["success"] for r in rows),
        failures=sum(r["failure"] for r in rows),
        success_rate=float(np.mean([r["success"] for r in rows])),
        mean_cost=float(np.mean([r["fixed_horizon_cost"] for r in rows])),
        mean_terminal_error=float(np.mean([r["terminal_physical_error"] for r in rows])),
        mean_input_square_integral=float(np.mean([r["control_energy"] for r in rows])),
        mean_V_rise_fraction=float(np.mean([r["V_rise_fraction"] for r in rows])))


class Backend:
    family = "state"
    def __init__(self, config, output, event=None, deadline=None):
        self.config = dict(config); self.output = Path(output)
        self.event = event or (lambda *a, **kw: None); self.deadline = deadline
        self.name = self.config.get("task", "point_mass")
        if self.name not in TASKS:
            raise ValueError("This adapter covers only the four declared state tasks")
        self.collect_calls = 0; self.initialized = False; self.history = []

    def guard(self):
        if any((folder/"STOP").exists() for folder in (self.output, self.output.parent, self.output.parent.parent)):
            raise InterruptedError("STOP")
        if self.deadline is not None and time.time() >= self.deadline:
            raise InterruptedError("protocol_deadline")

    def _should_stop(self):
        self.guard(); return False

    def source_files(self):
        return [str(Path(__file__).resolve()),
            str(ROOT/"stabilization-learning-control/stabilization_learning/algos/pytorch/state_metric_full_horizon/learner.py"),
            str(ROOT/"stabilization-learning-control/stabilization_learning/algos/pytorch/state_metric_full_horizon/objectives.py"),
            str(ROOT/"stabilization-learning-control/stabilization_learning/algos/pytorch/state_metric_full_horizon/evaluation.py"),
            str(ROOT/"stabilization-learning-control/stabilization_learning/policies/state_metric_residual_v3.py"),
            str(ROOT/"stabilization-learning-control/stabilization_learning/policies/state_metric_v2.py"),
            str(ROOT/"stabilization-gym/stabilization_gym/envs/classic_control/classic.py"),
            str(ROOT/"stabilization-gym/stabilization_gym/envs/classic_control/sdc_metric.py")]

    def initialize(self):
        self.guard()
        if self.initialized: raise RuntimeError("Backend already initialized")
        torch.set_num_threads(int(self.config.get("cpu_threads", 1)))
        torch.manual_seed(int(self.config.get("seed", 81)))
        self.output.mkdir(parents=True, exist_ok=True)
        self.task = FullHorizonSDCTask(make(self.name, **self.config.get("physical", {})), self.name)
        self.policy = AnchoredStateMetricPolicy(self.task.metric_spec, self.task.L0,
            goal=self.task.goal, context_scale=max(self.task.physical.initial_scale, .1),
            hidden=int(self.config.get("hidden", 16)))
        self.initial_state = deepcopy(self.policy.state_dict())
        base = 71000000 + TASKS.index(self.name)*100000
        known = {f.name for f in fields(FullHorizonConfig)}
        overrides = {k:v for k,v in self.config.items() if k in known}
        overrides.update(seed=int(self.config.get("seed", 81)),
            updates=int(self.config.get("total_updates", 8)),
            batch_states=int(self.config.get("batch_size", 4)),
            train_seed_start=base,
            validation_seeds=tuple(self.config.get("validation_seeds", range(base+20000, base+20000+int(self.config.get("validation_count", 4))))),
            test_seeds=tuple(self.config.get("test_seeds", range(base+40000, base+40000+int(self.config.get("test_count", 8))))),
            train_probe_seed=base+60000, validation_probe_seed=base+70000)
        if "scales" in overrides: overrides["scales"] = tuple(overrides["scales"])
        cfg = FullHorizonConfig(**overrides)
        source = hashlib.sha256("".join(hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                      for p in self.source_files()).encode()).hexdigest()
        self.learner = FullHorizonMetricLearner(self.task, self.policy, cfg, source_digest=source)
        self.learner.should_stop = self._should_stop
        self.learner.initialize()
        self.fixed_probe_error = self.task.error(torch.as_tensor(
            self.task.initial_states([cfg.train_probe_seed], scale=1.), dtype=torch.float64))
        self.initialized = True
        return dict(task=self.name, algorithm="preserved_full_horizon_model_gradient",
            initialization="nominal LQR metric with random hidden weights and zero residual output; empty Adam",
            representation="physical_state_identity", evaluator="none", certificate_factor=1.,
            horizon_steps=self.task.max_steps, dt=self.task.dt,
            initial_parameter_hash=_digest(self.policy), model_transitions=self.learner.model_transitions,
            fixed_training_probe_error=self.fixed_probe_error.detach().cpu().tolist(),
            fixed_training_probe_M=self._probe_matrix(),
            training_seed_start=base, validation_seeds=list(cfg.validation_seeds), test_seeds=list(cfg.test_seeds),
            acceptance="original validation feasibility gates, including failures, cost, arrival and input constraints")

    def _probe_matrix(self):
        with torch.no_grad():
            return self.policy(self.fixed_probe_error).detach().cpu().tolist()

    def collect(self, initial=False):
        self.guard(); self.collect_calls += 1
        return dict(skipped=True, reason="exact differentiable model generates fresh full trajectories inside each M update",
                    initial=bool(initial), model_transitions=self.learner.model_transitions)

    def warmup_representation(self, updates):
        return dict(skipped=True, requested_updates=int(updates), actual_updates=0, reason="physical state coordinates; no feature learner")

    def pretrain_evaluator(self, updates):
        return dict(skipped=True, requested_updates=int(updates), actual_updates=0, reason="known differentiable dynamics; no critic")

    def train_epoch(self, epoch, representation_updates=0, evaluator_updates=0, certificate_updates=4):
        rows = []
        for _ in range(int(certificate_updates)):
            self.guard(); before = _digest(self.policy); start = time.perf_counter()
            row = self.learner.update()
            row.update(epoch=int(epoch), wall_seconds=time.perf_counter()-start,
                       parameters_changed=_digest(self.policy) != before,
                       fixed_training_probe_error=self.fixed_probe_error.detach().cpu().tolist(),
                       fixed_training_probe_M=self._probe_matrix())
            rows.append(row); self.history.append(row)
            self.event("certificate_update", task=self.name, **row)
        return dict(epoch=int(epoch), representation_updates=0, evaluator_updates=0,
            certificate_updates=len(rows), actual_updates=self.learner.updates,
            model_transitions=self.learner.model_transitions, rows=rows)

    def validate(self):
        self.guard()
        candidate = self.learner.validate()
        rows = []
        # Validation may perform the original rollback. These rows explicitly
        # describe the active model after that decision, not the rejected model.
        for scale in self.learner.config.scales:
            group, _ = evaluate_state(self.task, self.policy,
                self.task.initial_states(self.learner.config.validation_seeds, scale=scale),
                decay=self.learner.config.decay, should_stop=self._should_stop)
            for seed, row in zip(self.learner.config.validation_seeds, group):
                row.update(seed=int(seed), scale=float(scale))
            rows.extend(group)
        summary = _summary(rows)
        return dict(summary=summary, rows=rows, candidate_validation=candidate,
            active_parameter_hash=_digest(self.policy),
            selection_score=[int(candidate["qualified"]), summary["successes"], -summary["mean_cost"]],
            original_rollback_retained=True)

    def save(self, path):
        self.guard(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise FileExistsError("Never overwrite a committed checkpoint: "+str(path))
        temporary = path.with_suffix(path.suffix+".tmp")
        self.learner.save(temporary)
        payload = torch.load(temporary, map_location="cpu", weights_only=False)
        payload["unified_backend"] = dict(config=self.config, initial_state=self.initial_state,
            collect_calls=self.collect_calls, history=self.history, torch_rng=torch.get_rng_state())
        torch.save(payload, temporary); temporary.replace(path)
        return dict(path=str(path), parameter_hash=_digest(self.policy), actual_updates=self.learner.updates)

    def load(self, path):
        self.guard()
        self.learner.restore(Path(path))
        payload = torch.load(path, map_location="cpu", weights_only=False)["unified_backend"]
        if payload["config"] != self.config: raise ValueError("Backend config mismatch")
        self.initial_state = payload["initial_state"]; self.collect_calls = payload["collect_calls"]
        self.history = payload["history"]; torch.set_rng_state(payload["torch_rng"])
        return dict(parameter_hash=_digest(self.policy), actual_updates=self.learner.updates)

    def evaluate(self, split, variants=("selected", "initial", "identity")):
        if split not in ("validation", "test"): raise ValueError("Unknown evaluation split")
        self.guard()
        if split == "test":
            frozen = self.output/"FROZEN.json"
            global_file = self.output.parent.parent/"GLOBAL_FROZEN.json"
            if not frozen.is_file() or not global_file.is_file():
                raise RuntimeError("All jobs must be frozen before test")
            gate = json.loads(frozen.read_text(encoding="utf-8"))
            global_gate = json.loads(global_file.read_text(encoding="utf-8"))
            job_protocol = json.loads((self.output/"protocol.json").read_text(encoding="utf-8"))
            if (global_gate.get("jobs", {}).get(self.output.name) != hashlib.sha256(frozen.read_bytes()).hexdigest()
                    or gate.get("protocol_sha256") != hashlib.sha256((self.output/"protocol.json").read_bytes()).hexdigest()
                    or global_gate.get("protocol_sha256") != job_protocol.get("suite_protocol_sha256")):
                raise RuntimeError("Job freeze is not in the committed experiment group")
            if hashlib.sha256((self.output/gate["checkpoint"]).read_bytes()).hexdigest() != gate["sha256"]:
                raise RuntimeError("Frozen checkpoint changed")
            saved = torch.load(self.output/gate["checkpoint"], map_location="cpu", weights_only=False)
            if saved["policy_hash"] != module_hash(self.policy):
                raise RuntimeError("Load the selected frozen model before testing")
        seeds = self.learner.config.test_seeds if split == "test" else self.learner.config.validation_seeds
        selected = deepcopy(self.policy.state_dict()); groups = {}; audit = {}
        before = _digest(self.policy)
        try:
            for variant in variants:
                self.guard()
                if variant == "selected":
                    self.policy.load_state_dict(selected); model = self.policy
                elif variant == "initial":
                    self.policy.load_state_dict(self.initial_state); model = self.policy
                elif variant in ("identity", "fixed_M"):
                    matrix = torch.eye(self.task.error_dim, dtype=torch.float64) if variant == "identity" else torch.tensor(self.task.P0)
                    model = lambda e, m=matrix: m.expand(*e.shape[:-1], *m.shape)
                else: raise ValueError("Unknown variant: "+variant)
                rows = []; maximum = 0.
                for scale in self.learner.config.scales:
                    starts = self.task.initial_states(seeds, scale=scale)
                    results, trace = evaluate_state(self.task, model, starts,
                        decay=self.learner.config.decay, record=True, should_stop=self._should_stop)
                    with torch.no_grad():
                        for endpoint, key in (("states", "V_before"), ("following", "V_after")):
                            e = self.task.error(torch.as_tensor(trace[endpoint]))
                            expected = torch.einsum("...i,...ij,...j->...", e, model(e), e).numpy()
                            maximum = max(maximum, float(np.max(np.abs(expected-trace[key]))))
                    for seed, row in zip(seeds, results): row.update(seed=int(seed), scale=float(scale))
                    rows.extend(results)
                    folder=self.output/"trajectories"/split/variant; folder.mkdir(parents=True, exist_ok=True)
                    target=folder/f"scale_{scale:g}.npz"
                    if target.exists(): raise FileExistsError("Never overwrite frozen evaluation: "+str(target))
                    np.savez_compressed(target, seeds=np.array(seeds), **trace)
                groups[variant] = dict(summary=_summary(rows), rows=rows)
                audit[variant] = dict(full_successor_V_max_abs=maximum, passed=maximum < 1e-8,
                    certificate_factor=1., matrix="literal_identity" if variant == "identity" else "state_conditioned_or_nominal")
        finally:
            self.policy.load_state_dict(selected)
        if _digest(self.policy) != before: raise RuntimeError("Evaluation changed selected parameters")
        paired = {}
        if "selected" in groups:
            left=groups["selected"]["rows"]
            for variant, data in groups.items():
                if variant == "selected": continue
                pairs=[(a,b) for a,b in zip(left,data["rows"]) if a["success"] and b["success"]]
                paired[variant]=dict(common_successes=len(pairs),
                    selected_minus_baseline_arrival_seconds=float(np.mean([a["arrival_time"]-b["arrival_time"] for a,b in pairs])) if pairs else None)
        return dict(task=self.name, split=split, groups=groups, audit=audit, paired_time=paired,
            scope="same starts; original full horizon, candidate V only; model execution, not hardware")


StateBackend = Backend
