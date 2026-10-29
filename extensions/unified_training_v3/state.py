"""State-task stability guards for fresh runs; preserved objectives and gates."""
from copy import deepcopy
from dataclasses import fields
from numbers import Integral, Real
from pathlib import Path
import math

import torch

from extensions.unified_training_v1.backends.state import Backend, TASKS
from stabilization_learning.algos.pytorch.state_metric_full_horizon import FullHorizonConfig, FullHorizonMetricLearner
from stabilization_learning.algos.pytorch.state_metric_full_horizon.objectives import (
    full_horizon_components, certificate_transition, falsify)
from .guards import NumericalGuard
from .numerical import require_finite


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def validate_config(config):
    """Validate the adapter's effective config before any environment starts."""
    config = dict(config)
    task = config.get("task", "point_mass")
    if task not in TASKS:
        raise ValueError("Unknown state task")
    for key, default in (("total_updates", 8), ("batch_size", 4), ("hidden", 16),
                         ("cpu_threads", 1), ("validation_count", 4), ("test_count", 8)):
        _integer(config.get(key, default), key)
    _integer(config.get("seed", 81), "seed", 0)
    base = 71000000 + TASKS.index(task)*100000
    defaults = FullHorizonConfig()
    known = {f.name for f in fields(FullHorizonConfig)}
    values = {key: value for key, value in config.items() if key in known}
    fixed = dict(seed=config.get("seed", 81), updates=config.get("total_updates", 8),
        batch_states=config.get("batch_size", 4), train_seed_start=base,
        train_probe_seed=base+60000, validation_probe_seed=base+70000)
    for key, value in fixed.items():
        if key in config and config[key] != value:
            raise ValueError(f"{key} conflicts with the state adapter's declared configuration")
    values.update(fixed)
    for key, count, offset in (("validation_seeds", "validation_count", 20000),
                                ("test_seeds", "test_count", 40000)):
        seeds = config.get(key, range(base+offset, base+offset+config.get(count, 4 if offset==20000 else 8)))
        if not isinstance(seeds, (tuple, list, range)) or not seeds:
            raise ValueError(f"{key} must be a nonempty integer sequence")
        values[key] = tuple(_integer(seed, key, 0) for seed in seeds)
    scales = config.get("scales", defaults.scales)
    if not isinstance(scales, (tuple, list)) or not scales or len(set(scales)) != len(scales):
        raise ValueError("scales must be nonempty and unique")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0 for v in scales):
        raise ValueError("scales must be finite and positive")
    values["scales"] = tuple(scales)
    for key in known - {"scales", "validation_seeds", "test_seeds"}:
        value = values.get(key, getattr(defaults, key))
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Invalid state setting: {key}")
    cfg = FullHorizonConfig(**values)
    train_end = cfg.train_seed_start + (cfg.updates+2)*cfg.batch_states + 128
    probes = {cfg.train_probe_seed, cfg.validation_probe_seed}
    if any(cfg.train_seed_start <= seed < train_end for seed in probes):
        raise ValueError("Training range overlaps an independent probe seed")
    if probes & (set(cfg.validation_seeds) | set(cfg.test_seeds)):
        raise ValueError("Holdout seeds overlap an independent probe seed")
    physical = config.get("physical", {})
    if not isinstance(physical, dict):
        raise ValueError("physical must be a mapping")
    try:
        require_finite(physical, "state physical settings")
    except FloatingPointError as error:
        raise ValueError(str(error)) from error
    if "max_steps" in physical:
        _integer(physical["max_steps"], "physical.max_steps")
    for key in ("dt", "initial_scale"):
        if key in physical and (isinstance(physical[key], bool) or not isinstance(physical[key], Real) or physical[key] <= 0):
            raise ValueError(f"physical.{key} must be finite and positive")
    return cfg


_COUNTERS = ("updates", "model_transitions", "validation_transitions", "dual", "replay",
    "best_actor", "best_optimizer", "last_candidate", "best_score", "best_update", "stale",
    "history", "baseline", "cost_scale", "in_update")


class _StableLearner(FullHorizonMetricLearner):
    def update(self):
        """Preserved full-horizon update with checks at its commit boundaries."""
        if self.cost_scale is None:
            raise RuntimeError("Initialize baseline and normalization first")
        if self.in_update:
            raise RuntimeError("Update already in progress")
        self.in_update = True
        cfg = self.config
        ids = range(cfg.train_seed_start + 128 + self.updates*cfg.batch_states,
                    cfg.train_seed_start + 128 + (self.updates+1)*cfg.batch_states)
        starts = self.task.initial_states(ids)
        factors = torch.tensor(cfg.scales, dtype=torch.float64)[torch.randint(len(cfg.scales),
            (len(starts),), generator=self.rng)]
        goal = starts.new_tensor(self.task.goal)
        starts = goal + (starts-goal)*factors[:,None]
        fresh = self.probes(cfg.probe_batch, self.rng)
        if len(self.replay):
            indices = torch.randint(len(self.replay), (cfg.probe_batch//2,), generator=self.rng)
            fresh = torch.cat((fresh, self.replay[indices]))
        hard, rates = falsify(self.task, self.policy, fresh, decay=cfg.decay,
            steps=cfg.attack_steps, lower=self.lower, upper=self.upper)
        require_finite((hard, rates), "training counterexamples")
        P0 = starts.new_tensor(self.task.P0)
        def nominal(e):
            return P0.expand(*e.shape[:-1],*P0.shape)
        with torch.no_grad():
            reference = full_horizon_components(self.task, nominal, starts, decay=cfg.decay,
                checkpoint_stride=0, should_stop=self.should_stop).mean(0)
        self.optimizer.zero_grad(set_to_none=True)
        components = full_horizon_components(self.task, self.policy, starts, decay=cfg.decay,
            checkpoint_stride=cfg.checkpoint_stride, should_stop=self.should_stop).mean(0)
        risk = certificate_transition(self.task, self.policy, hard, decay=cfg.decay)[-1].relu().square().mean()
        horizon_seconds = self.task.dt*self.task.max_steps
        energy_ratio = components[4]/reference[4].clamp_min(1e-15)
        budget_excess = (energy_ratio-(cfg.max_energy_ratio-cfg.validation_energy_margin)).relu().square()
        constraint = components[1]/horizon_seconds + risk + cfg.energy_budget_weight*budget_excess
        loss = (components[0]/self.cost_scale + self.dual*constraint + cfg.hold_weight*components[2]
            + cfg.arrival_weight*components[3] + cfg.effort_weight*components[4]/self.cost_scale)
        require_finite((reference, components, risk, energy_ratio, constraint, loss), "state loss")
        loss.backward()
        require_finite([p.grad for p in self.policy.parameters()], "full-horizon gradient")
        # Finite leaves can still have an overflowing aggregate norm. Do not
        # turn them into zero/NaN gradients and let Adam appear to succeed.
        gradient = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1., error_if_nonfinite=True)
        if self.should_stop and self.should_stop():
            raise InterruptedError("STOP before committing optimizer update")
        self.optimizer.step()
        require_finite((self.policy.state_dict(), self.optimizer.state_dict()), "post-Adam state")
        self.updates += 1
        self.model_transitions += len(starts)*self.task.max_steps
        next_dual = self.dual+cfg.dual_step*float(constraint.detach())
        require_finite(next_dual, "dual update")
        self.dual = min(cfg.dual_max, next_dual)
        pool = torch.cat((self.replay, hard))
        with torch.no_grad():
            ranks = certificate_transition(self.task, self.policy, pool, decay=cfg.decay)[-1]
            require_finite(ranks, "post-update counterexample ranking")
            keep = torch.topk(ranks, min(len(pool), cfg.replay_capacity), sorted=True).indices
        self.replay = pool[keep].detach().clone()
        self.in_update = False
        result = dict(update=self.updates, loss=float(loss.detach()),
            physical_training_cost=float(components[0].detach()), constraint=float(constraint.detach()),
            paired_training_energy_ratio=float(energy_ratio.detach()),
            gradient_norm=float(gradient), dual=self.dual, replay_size=len(self.replay),
            model_transitions=self.model_transitions)
        self.history.append(dict(kind="training", **result))
        return result


class StableStateBackend(Backend):
    """Reject corrupt updates atomically while retaining normal validation rollback."""
    def __init__(self, config, output, event=None, deadline=None):
        self._validated_config = validate_config(config)
        self._numerical = NumericalGuard("state")
        super().__init__(config, output, event=event, deadline=deadline)

    def guard(self):
        self._numerical.guard()
        return super().guard()

    def source_files(self):
        return super().source_files() + [str(Path(__file__).resolve()),
            str(Path(__file__).with_name("guards.py").resolve()),
            str(Path(__file__).with_name("numerical.py").resolve()),
            str(Path(__file__).resolve().parents[2] / "stabilization-gym/stabilization_gym/envs/classic_control/sdc_full_horizon.py")]

    def _check(self):
        self._numerical.check_all()
        for key in _COUNTERS:
            self._numerical.check(getattr(self.learner, key), "learner."+key)
        self._numerical.check(self.initial_state, "initial state")

    def _snapshot(self):
        return dict(policy=deepcopy(self.policy.state_dict()),
            optimizer=deepcopy(self.learner.optimizer.state_dict()),
            rng=self.learner.rng.get_state().clone(), torch_rng=torch.get_rng_state().clone(),
            counters={key: deepcopy(getattr(self.learner, key)) for key in _COUNTERS},
            history=deepcopy(self.history), initial_state=deepcopy(self.initial_state),
            collect_calls=self.collect_calls)

    def _restore_snapshot(self, snapshot):
        self.policy.load_state_dict(snapshot["policy"])
        self.learner.optimizer.load_state_dict(snapshot["optimizer"])
        self.learner.rng.set_state(snapshot["rng"])
        torch.set_rng_state(snapshot["torch_rng"])
        for key, value in snapshot["counters"].items():
            setattr(self.learner, key, value)
        self.history = snapshot["history"]
        self.initial_state = snapshot["initial_state"]
        self.collect_calls = snapshot["collect_calls"]
        self.learner.optimizer.zero_grad(set_to_none=True)

    def _transaction(self, operation, label):
        self.guard()
        self._check()
        snapshot = self._snapshot()
        try:
            result = operation()
            self._numerical.check(result, label)
            self._check()
            return result
        except BaseException as error:
            self._restore_snapshot(snapshot)
            self._numerical.fail(error)
            raise

    def initialize(self):
        self.guard()
        try:
            result = super().initialize()
            self._numerical.watch_module(self.policy, "policy")
            self._numerical.watch_optimizer(self.learner.optimizer, "Adam")
            self._check()
            self._numerical.check(result, "initialization")
            def check_output(_module, _args, value):
                self._numerical.check(value, "state metric output")
            self._forward_handle = self.policy.register_forward_hook(check_output)
            for method in ("requested_control", "step_model"):
                original = getattr(self.task, method)
                def checked(*args, _original=original, _label=method, **kwargs):
                    result = _original(*args, **kwargs)
                    self._numerical.check(result, "state "+_label)
                    return result
                setattr(self.task, method, checked)
            self.learner.__class__ = _StableLearner
            old_update = self.learner.update
            old_validate = self.learner.validate
            def checked_update():
                self.guard()
                if self.learner.updates >= self.learner.config.updates:
                    raise ValueError("Declared state update budget exhausted")
                return self._transaction(old_update, "full-horizon update")
            self.learner.update = checked_update
            self.learner.validate = lambda: self._transaction(old_validate, "state validation")
            return result
        except BaseException as error:
            self._numerical.fail(error)
            raise

    def train_epoch(self, epoch, representation_updates=0, evaluator_updates=0, certificate_updates=4):
        self.guard()
        _integer(certificate_updates, "certificate_updates", 0)
        if self.learner.updates + certificate_updates > self.learner.config.updates:
            raise ValueError("Schedule exceeds the declared state update/seed budget")
        return super().train_epoch(epoch, representation_updates, evaluator_updates, certificate_updates)

    def validate(self):
        return self._transaction(super().validate, "state validation report")

    def save(self, path):
        self.guard()
        self._check()
        return super().save(path)

    def load(self, path):
        return self._transaction(lambda: super(StableStateBackend, self).load(path), "state load")

    def evaluate(self, split, variants=("selected", "initial", "identity")):
        return self._transaction(lambda: super(StableStateBackend, self).evaluate(split, variants), "state evaluation")

    def close(self):
        if hasattr(self, "_forward_handle"):
            self._forward_handle.remove()
        self._numerical.close()
        if hasattr(self, "task"):
            self.task.physical.close()
