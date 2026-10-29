"""Full-successor V and matched analytic control for the independent DC motor.

This new protocol fixes both-endpoint metric evaluation. The historical optional
candidate-screen controller remains historical; this route trains and evaluates
the same analytic applied-voltage controller, without claiming a hard CLF proof.
"""
from copy import deepcopy
import math
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from extensions.state_conditioned_m_v1.envs.dc_motor import StateDependentDCMotorEnv
from extensions.state_conditioned_m_v1.core.metric import StateDependentMetricActor
from extensions.state_conditioned_m_v1.core.identity import IdentityMetricActor
from extensions.state_conditioned_m_v1.core.certificate import quadratic
from extensions.unified_training_v3.guards import NumericalGuard
from extensions.unified_training_v3.numerical import require_finite
from .independent_backend import EpisodeBackend, assert_test_gate
from .runtime import content_hash
from .validation import positive_int, seeds


def validate_settings(config, schedule=None):
    allowed = {"seed", "total_updates", "batch_size", "hidden", "learning_rate", "decay",
               "descent_weight", "target_speed", "validation_seeds", "test_seeds", "cpu_threads"}
    if set(config)-allowed:
        raise ValueError("Unknown DC motor configuration: " + repr(sorted(set(config)-allowed)))
    for name, default in (("seed", 81), ("total_updates", 2), ("batch_size", 4), ("hidden", 32)):
        positive_int(config.get(name, default), name, 0 if name == "seed" else 1)
    updates, batch = config.get("total_updates", 2), config.get("batch_size", 4)
    if schedule and schedule["epochs"] * schedule["certificate_updates"] != updates:
        raise ValueError("Motor schedule must exactly match total_updates")
    for key, default, minimum in (("learning_rate", .001, 0), ("decay", .05, -1e-100),
                                  ("descent_weight", 1., -1e-100)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= minimum:
            raise ValueError("Invalid motor setting: " + key)
    StateDependentDCMotorEnv._validate_target(config.get("target_speed", .5))
    val = seeds(config.get("validation_seeds", [910001, 910002]), "validation_seeds")
    test = seeds(config.get("test_seeds", [920001, 920002]), "test_seeds")
    train = set(range(config.get("seed", 81)+1000, config.get("seed", 81)+1000+updates*batch))
    if train & set(val) or train & set(test) or set(val) & set(test):
        raise ValueError("DC motor training/validation/test seeds overlap")
    return val, test


def full_successor_loss(actor, env, episode_seeds, *, decay=.05, descent_weight=1.):
    costs, residuals, frozen_residuals = [], [], []
    for seed in episode_seeds:
        state, _ = env.reset(seed=int(seed))
        error, context = env.build_error(state)
        metric = actor(env.metric_context(state))
        action = env.controller_action(metric, error, context)
        following = env.torch_step(torch.as_tensor(state, dtype=torch.float64), action.to(torch.float64))
        next_error = env.tensor_error(following)
        next_metric = actor(env.context_for_states(following))
        before = quadratic(error, metric.to(error))
        after = quadratic(next_error, next_metric.to(next_error))
        norm = error.square().sum()
        residuals.append(after-before+decay*env.dt*norm)
        frozen_residuals.append(quadratic(next_error, metric.to(next_error))-before+decay*env.dt*norm)
        costs.append(next_error.square().sum())
    raw = torch.stack(residuals)
    performance = torch.stack(costs).mean()
    loss = performance + descent_weight*raw.relu().mean()
    require_finite((loss, raw, performance), "full successor motor objective")
    return loss, dict(loss=float(loss.detach()), performance=float(performance.detach()),
                     full_successor_residual=raw.detach().tolist(),
                     frozen_metric_residual_diagnostic=torch.stack(frozen_residuals).detach().tolist())


@torch.no_grad()
def rollout(env, actor, seed, *, decay=.05):
    observation, info = env.reset(seed=int(seed))
    rows, total = [], 0.
    for index in range(env.max_steps):
        error, context = env.build_error(observation)
        metric = actor(env.metric_context(observation))
        action = env.controller_action(metric, error, context)
        before = quadratic(error, metric.to(error))
        observation, cost, terminated, truncated, info = env.step(action)
        next_error = env.tensor_error(torch.as_tensor(observation, dtype=torch.float64))
        next_metric = actor(env.context_for_states(torch.as_tensor(observation)))
        after = quadratic(next_error, next_metric.to(next_error))
        row = dict(step=index+1, error=error.tolist(), error_next=next_error.tolist(), M=metric.tolist(),
                   M_next=next_metric.tolist(), V=float(before), V_next=float(after),
                   residual=float(after-before+decay*env.dt*error.square().sum()),
                   commanded_action=action.tolist(), applied_action=np.asarray(info.get("applied_action", action)).tolist(),
                   cost=float(cost))
        require_finite(row, "motor actual successor")
        rows.append(row); total += float(cost)
        if terminated or truncated:
            break
    failure = bool(info.get("failure", False))
    result = dict(seed=seed, success=bool(info.get("success", False)) and not failure, failure=failure,
                  steps=len(rows), total_cost=total, terminal_physical_error=float(info["physical_error"]),
                  V_rise_steps=sum(r["V_next"] > r["V"]+1e-8 for r in rows),
                  descent_residual_sum=sum(max(0., r["residual"]) for r in rows), trace=rows)
    require_finite(result, "motor rollout")
    return result


class MotorBackend:
    family = "state_dependent_M_motor"
    guard = EpisodeBackend.guard
    collect = EpisodeBackend.collect
    warmup_representation = EpisodeBackend.warmup_representation
    pretrain_evaluator = EpisodeBackend.pretrain_evaluator

    def __init__(self, config, output, event=None, deadline=None):
        self.config, self.output = deepcopy(config), Path(output)
        self.validation_seeds, self.test_seeds = validate_settings(config)
        self.event, self.deadline = event or (lambda *_a, **_k: None), deadline
        self.initialized, self.updates, self.history = False, 0, []

    def guard(self):
        import time
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError("Original deadline reached")
        if any((p/"STOP").exists() for p in (self.output, self.output.parent, self.output.parent.parent)):
            raise InterruptedError("STOP")
        if self.initialized:
            self.numerical.guard()

    def initialize(self):
        self.guard()
        if self.initialized:
            raise RuntimeError("Backend already initialized")
        torch.manual_seed(self.config.get("seed", 81))
        torch.set_num_threads(1)
        self.env = StateDependentDCMotorEnv(target_speed=self.config.get("target_speed", .5))
        self.actor = StateDependentMetricActor(4, self.env.metric_spec, self.config.get("hidden", 32))
        self.initial_state = deepcopy(self.actor.state_dict())
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.get("learning_rate", .001))
        self.numerical = NumericalGuard("full-successor DC motor v4")
        self.numerical.watch_module(self.actor, "actor")
        self.numerical.watch_optimizer(self.optimizer, "Adam")
        self.output.mkdir(parents=True, exist_ok=True)
        self.initialized = True
        return dict(task="dc_motor", algorithm="full_successor_model_gradient_v4", controller="matched analytic voltage",
                    objective="one-step physical error plus complete V(e_next,M(e_next))-V(e,M(e)) descent penalty",
                    horizon_steps=self.env.max_steps, dt=self.env.dt, initial_parameter_hash=content_hash(self.actor.state_dict()))

    def train_epoch(self, epoch, representation_updates=0, evaluator_updates=0, certificate_updates=1):
        if representation_updates or evaluator_updates:
            raise ValueError("Unsupported motor stage")
        positive_int(certificate_updates, "certificate_updates", 0)
        if self.updates+certificate_updates > self.config.get("total_updates", 2):
            raise ValueError("Motor epoch exceeds the declared whole-group seed budget")
        rows = []
        for _ in range(certificate_updates):
            self.guard()
            if self.updates >= self.config.get("total_updates", 2):
                raise ValueError("Motor update budget exhausted")
            snapshot = deepcopy((self.actor.state_dict(), self.optimizer.state_dict(), torch.get_rng_state()))
            try:
                batch = self.config.get("batch_size", 4)
                start = self.config.get("seed", 81)+1000+self.updates*batch
                self.optimizer.zero_grad(set_to_none=True)
                loss, row = full_successor_loss(self.actor, self.env, range(start, start+batch),
                    decay=self.config.get("decay", .05), descent_weight=self.config.get("descent_weight", 1.))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1., error_if_nonfinite=True)
                self.numerical.check(norm, "gradient norm")
                self.optimizer.step(); self.numerical.check_all()
            except BaseException as error:
                self.actor.load_state_dict(snapshot[0]); self.optimizer.load_state_dict(snapshot[1])
                torch.set_rng_state(snapshot[2]); self.numerical.fail(error)
                raise
            self.updates += 1
            row.update(epoch=epoch, update=self.updates, parameter_hash=content_hash(self.actor.state_dict()),
                       seeds=list(range(start, start+batch)))
            rows.append(row); self.history.append(row)
            self.event("certificate_update", task="dc_motor", **row)
        return dict(epoch=epoch, certificate_updates=len(rows), actual_updates=self.updates, rows=rows)

    def validate(self):
        self.guard()
        try:
            rows = []
            for seed in self.validation_seeds:
                self.guard()
                rows.append(rollout(self.env, self.actor, seed, decay=self.config.get("decay", .05)))
            summary = dict(successes=sum(r["success"] for r in rows), failures=sum(r["failure"] for r in rows),
                           mean_cost=float(np.mean([r["total_cost"] for r in rows])))
            result = dict(summary=summary, selection_score=[summary["successes"], -summary["mean_cost"]], rows=rows)
            self.numerical.check(result, "validation")
            return result
        except BaseException as error:
            self.numerical.fail(error)
            raise

    def save(self, path):
        self.guard(); self.numerical.check_all()
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema="full_successor_motor_v4", config=self.config, actor=self.actor.state_dict(), optimizer=self.optimizer.state_dict(),
                        initial=self.initial_state, updates=self.updates, history=self.history,
                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
        require_finite(payload, "motor checkpoint")
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        torch.save(payload, temporary)
        if path.exists():
            raise FileExistsError(path)
        temporary.rename(path)
        return dict(path=str(path), updates=self.updates)

    def load(self, path):
        self.guard()
        snapshot = deepcopy((self.actor.state_dict(), self.optimizer.state_dict(), self.initial_state,
            self.updates, self.history, torch.get_rng_state(), np.random.get_state(), random.getstate()))
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require_finite(payload, "motor checkpoint")
            if payload.get("schema") != "full_successor_motor_v4" or payload["config"] != self.config:
                raise ValueError("Motor checkpoint configuration/schema differs")
            count = positive_int(payload["updates"], "checkpoint updates", 0)
            if count > self.config.get("total_updates", 2) or not isinstance(payload["history"], list):
                raise ValueError("Invalid motor checkpoint counters/history")
            probe = deepcopy(self.actor); probe.load_state_dict(payload["initial"])
            self.actor.load_state_dict(payload["actor"]); self.optimizer.load_state_dict(payload["optimizer"])
            for parameter, state in self.optimizer.state.items():
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and state[key].shape != parameter.shape:
                        raise ValueError("Motor checkpoint Adam moment shape differs")
            self.initial_state, self.updates, self.history = payload["initial"], count, payload["history"]
            torch.set_rng_state(payload["torch_rng"]); np.random.set_state(payload["numpy_rng"])
            random.setstate(payload["python_rng"])
            self.numerical.check_all()
        except BaseException as error:
            self.actor.load_state_dict(snapshot[0]); self.optimizer.load_state_dict(snapshot[1])
            self.initial_state, self.updates, self.history = snapshot[2:5]
            torch.set_rng_state(snapshot[5]); np.random.set_state(snapshot[6]); random.setstate(snapshot[7])
            self.optimizer.zero_grad(set_to_none=True)
            self.numerical.fail(error)
            raise

    def evaluate(self, split="test", variants=("selected", "initial", "identity")):
        self.guard()
        if split == "test":
            assert_test_gate(self.output, content_hash(self.actor.state_dict()))
        elif split != "validation":
            raise ValueError("Unknown split")
        chosen = deepcopy(self.actor.state_dict())
        groups = {}
        try:
            for variant in variants:
                if variant not in {"selected", "initial", "identity"}:
                    raise ValueError("Unknown variant")
                self.actor.load_state_dict(self.initial_state if variant == "initial" else chosen)
                model = IdentityMetricActor(4, self.env.metric_spec) if variant == "identity" else self.actor
                group = [rollout(self.env, model, seed, decay=self.config.get("decay", .05))
                         for seed in (self.test_seeds if split == "test" else self.validation_seeds)]
                groups[variant] = dict(episodes=group, successes=sum(r["success"] for r in group),
                                       mean_cost=float(np.mean([r["total_cost"] for r in group])))
                self.numerical.check(groups[variant], "evaluation")
                self.guard()
        except BaseException as error:
            self.numerical.fail(error)
            raise
        finally:
            self.actor.load_state_dict(chosen)
        return dict(task="dc_motor", variants=groups, controller="analytic", candidate_certificate_only=True)

    def close(self):
        if self.initialized:
            self.numerical.close(); self.env.close(); self.initialized = False
