"""Complete CPU training mechanism with full-horizon gradients and safe selection."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import numpy as np
import torch
from stabilization_learning.common.contracts_v2 import identity
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import atomic_save, module_hash
from .objectives import full_horizon_components, certificate_transition, falsify
from .evaluation import evaluate, local_check, selection_gate


@dataclass(frozen=True)
class FullHorizonConfig:
    seed: int = 41
    updates: int = 64
    batch_states: int = 16
    learning_rate: float = .003
    validation_every: int = 8
    checkpoint_stride: int = 32
    decay: float = .05
    initial_dual: float = 2.
    dual_step: float = .5
    dual_max: float = 100.
    hold_weight: float = 1.
    arrival_weight: float = .02
    effort_weight: float = .001
    energy_budget_weight: float = 10.
    replay_capacity: int = 256
    probe_batch: int = 64
    attack_steps: int = 4
    patience: int = 6
    scales: tuple = (1., 1.5, 2.)
    validation_seeds: tuple = tuple(range(5100001, 5100017))
    test_seeds: tuple = tuple(range(6100001, 6100065))
    train_seed_start: int = 4100001
    validation_probe_seed: int = 5200001
    train_probe_seed: int = 4200001
    max_arrival_delay: float = .15
    max_energy_ratio: float = 1.1
    validation_energy_margin: float = .02
    rate_tolerance: float = 1e-6

    def __post_init__(self):
        integers = ("updates", "batch_states", "validation_every", "checkpoint_stride", "probe_batch",
                    "replay_capacity", "attack_steps", "patience")
        if any(type(getattr(self, k)) is not int or getattr(self, k) <= 0 for k in integers):
            raise ValueError("Positive integer training budgets required")
        if any(not math.isfinite(x) or x <= 0 for x in self.scales):
            raise ValueError("Invalid reset scales")
        if (not self.validation_seeds or not self.test_seeds
            or len(set(self.validation_seeds)) != len(self.validation_seeds)
            or len(set(self.test_seeds)) != len(self.test_seeds)
            or set(self.validation_seeds) & set(self.test_seeds)):
            raise ValueError("Nonempty disjoint validation and test seeds required")
        train = set(range(self.train_seed_start, self.train_seed_start + (self.updates + 2) * self.batch_states + 128))
        if train & (set(self.validation_seeds) | set(self.test_seeds)):
            raise ValueError("Training overlaps holdout seeds")
        if self.train_probe_seed == self.validation_probe_seed:
            raise ValueError("Training and validation probe RNGs must differ")
        numeric = (self.learning_rate, self.decay, self.initial_dual, self.dual_step, self.dual_max,
                   self.hold_weight, self.arrival_weight, self.effort_weight, self.energy_budget_weight,
                   self.rate_tolerance,self.max_arrival_delay,self.max_energy_ratio,self.validation_energy_margin)
        if any(not math.isfinite(v) or v < 0 for v in numeric) or self.learning_rate == 0:
            raise ValueError("Finite nonnegative objective settings required")
        if self.max_energy_ratio-self.validation_energy_margin < 1:
            raise ValueError("Validation energy margin excludes the nominal reference")


class FullHorizonMetricLearner:
    """Task/policy injection. No imports from an experiment directory."""
    def __init__(self, task, policy, config, *, source_digest):
        task.require_algorithm("state_metric_full_horizon_v3")
        if next(policy.parameters()).device.type != "cpu":
            raise ValueError("Only CPU whole-update recovery has been qualified")
        self.task, self.policy, self.config = task, policy, config
        self.source_digest = source_digest
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.rng = torch.Generator().manual_seed(config.train_probe_seed)
        self.updates = 0; self.model_transitions = 0; self.validation_transitions = 0
        self.dual = config.initial_dual
        self.replay = torch.empty((0, task.error_dim), dtype=torch.float64)
        self.best_actor = deepcopy(policy.state_dict())
        self.best_optimizer = deepcopy(self.optimizer.state_dict())
        self.last_candidate = deepcopy(policy.state_dict())
        self.best_score = 1.; self.best_update = 0; self.stale = 0
        self.history = []; self.baseline = None; self.cost_scale = None
        self.should_stop = None; self.in_update = False
        extent = float(task.physical.initial_scale) * max(config.scales)
        self.lower = torch.tensor(task.goal, dtype=torch.float64) - extent
        self.upper = torch.tensor(task.goal, dtype=torch.float64) + extent

    def protocol(self):
        return dict(schema="state_metric_full_horizon_v3", config=asdict(self.config),
            source_digest=self.source_digest, task=self.task.describe().to_dict(),
            physical=dict(dt=self.task.dt, horizon=self.task.max_steps, goal=self.task.goal.tolist(),
                initial_scale=float(self.task.physical.initial_scale), tolerance=self.task.success_tolerance,
                hold=self.task.required_hold_seconds, low=self.task.action_low.tolist(), high=self.task.action_high.tolist(),
                Q=self.task.Q.tolist(),R=self.task.R.tolist(),P0=self.task.P0.tolist()),
            policy={k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in self.policy.state_dict().items()},
            fixed_policy_buffers={k:v.tolist() for k,v in self.policy.named_buffers()},
            runtime=dict(torch=torch.__version__, numpy=np.__version__, device="cpu", dtype="float64"),
            gradient="full physical horizon, no truncated state gradients, segment recomputation",
            counterexamples="training-only empirical PGD and bounded replay; not a verifier",
            selection="all feasibility gates then mean validation physical-cost ratio; test is sealed",
            recovery="whole optimizer update; optimizer, dual, RNG, replay, selection and schedule included")

    def probes(self, count, generator):
        z = torch.rand((count, self.task.error_dim), generator=generator, dtype=torch.float64) * 2 - 1
        scale = torch.ones(count, dtype=torch.float64)
        scale[count//2:] = torch.exp(-torch.rand(count-count//2, generator=generator,
            dtype=torch.float64) * math.log(1e4))
        return torch.tensor(self.task.goal) + z * (self.upper-self.lower) / 2 * scale[:, None]

    def initialize(self):
        if self.baseline is not None:
            return
        cfg = self.config
        self.baseline = {}
        for scale in cfg.scales:
            rows,_ = evaluate(self.task, self.policy, self.task.initial_states(cfg.validation_seeds, scale=scale),
                decay=cfg.decay, should_stop=self.should_stop)
            self.baseline[scale] = rows
            self.validation_transitions += sum(r["physical_steps"] for r in rows)
        # Objective normalization uses TRAINING starts only, not validation costs.
        starts = self.task.initial_states(range(cfg.train_seed_start, cfg.train_seed_start+32), scale=1.5)
        with torch.no_grad():
            component = full_horizon_components(self.task, self.policy, starts, decay=cfg.decay,
                checkpoint_stride=0, should_stop=self.should_stop)
        self.cost_scale = max(float(component[:,0].mean()), 1e-12)
        result = self.validate()
        if not result["qualified"]:
            raise ValueError("Nominal policy fails the declared validation gates: " + repr(result["reasons"]))

    def update(self):
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
        hard, _ = falsify(self.task, self.policy, fresh, decay=cfg.decay,
            steps=cfg.attack_steps, lower=self.lower, upper=self.upper)
        # Paired TRAINING reference: a validation energy limit alone supplies
        # no gradient explaining why an otherwise cheaper policy was rejected.
        P0=starts.new_tensor(self.task.P0)
        def nominal(e):return P0.expand(*e.shape[:-1],*P0.shape)
        with torch.no_grad():
            reference=full_horizon_components(self.task,nominal,starts,decay=cfg.decay,
                checkpoint_stride=0,should_stop=self.should_stop).mean(0)
        self.optimizer.zero_grad(set_to_none=True)
        components = full_horizon_components(self.task, self.policy, starts, decay=cfg.decay,
            checkpoint_stride=cfg.checkpoint_stride, should_stop=self.should_stop).mean(0)
        risk = certificate_transition(self.task, self.policy, hard, decay=cfg.decay)[-1].relu().square().mean()
        horizon_seconds = self.task.dt*self.task.max_steps
        energy_ratio=components[4]/reference[4].clamp_min(1e-15)
        budget_excess=(energy_ratio-(cfg.max_energy_ratio-cfg.validation_energy_margin)).relu().square()
        constraint = components[1]/horizon_seconds + risk + cfg.energy_budget_weight*budget_excess
        loss = (components[0]/self.cost_scale + self.dual*constraint + cfg.hold_weight*components[2]
            + cfg.arrival_weight*components[3] + cfg.effort_weight*components[4]/self.cost_scale)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.policy.parameters()):
            raise FloatingPointError("Nonfinite full-horizon gradient; update not committed")
        gradient = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
        if self.should_stop and self.should_stop():
            raise InterruptedError("STOP before committing optimizer update")
        self.optimizer.step()
        self.updates += 1
        self.model_transitions += len(starts)*self.task.max_steps
        self.dual = min(cfg.dual_max, self.dual+cfg.dual_step*float(constraint.detach()))
        pool = torch.cat((self.replay, hard))
        with torch.no_grad():
            ranks = certificate_transition(self.task, self.policy, pool, decay=cfg.decay)[-1]
            keep = torch.topk(ranks, min(len(pool),cfg.replay_capacity), sorted=True).indices
        self.replay = pool[keep].detach().clone()
        self.in_update = False
        result = dict(update=self.updates, loss=float(loss.detach()),
            physical_training_cost=float(components[0].detach()), constraint=float(constraint.detach()),
            paired_training_energy_ratio=float(energy_ratio.detach()),
            gradient_norm=float(gradient), dual=self.dual, replay_size=len(self.replay),
            model_transitions=self.model_transitions)
        self.history.append(dict(kind="training", **result))
        return result

    def validate(self):
        cfg = self.config
        groups = {}
        for scale in cfg.scales:
            rows,_ = evaluate(self.task, self.policy, self.task.initial_states(cfg.validation_seeds, scale=scale),
                decay=cfg.decay, should_stop=self.should_stop)
            groups[scale] = rows
            self.validation_transitions += sum(r["physical_steps"] for r in rows)
        generator = torch.Generator().manual_seed(cfg.validation_probe_seed)
        probes = self.probes(256, generator)
        _, rates = falsify(self.task, self.policy, probes, decay=cfg.decay, steps=cfg.attack_steps,
            lower=self.lower, upper=self.upper)
        local = local_check(self.task, self.policy, decay=cfg.decay)
        result = selection_gate(groups, self.baseline, local=local, probe_rate=float(rates.max()),
            rate_tolerance=cfg.rate_tolerance, max_arrival_delay=cfg.max_arrival_delay,
            max_energy_ratio=cfg.max_energy_ratio-cfg.validation_energy_margin)
        result.update(update=self.updates, actor_hash=module_hash(self.policy))
        self.last_candidate = deepcopy(self.policy.state_dict())
        improved = result["qualified"] and result["score"] < self.best_score - 1e-5
        result["selected"] = improved
        if improved:
            self.best_actor = deepcopy(self.policy.state_dict())
            self.best_optimizer = deepcopy(self.optimizer.state_dict())
            self.best_score = result["score"]; self.best_update = self.updates; self.stale = 0
        elif self.updates:
            self.stale += 1
        if self.updates and not result["qualified"]:
            rate = self.optimizer.param_groups[0]["lr"] * .5
            self.policy.load_state_dict(self.best_actor)
            self.optimizer.load_state_dict(deepcopy(self.best_optimizer))
            for group in self.optimizer.param_groups:
                group["lr"] = rate
            result["rollback"] = dict(to_update=self.best_update, learning_rate=rate)
        self.history.append(dict(kind="validation", **result))
        return result

    def save(self, path):
        if self.in_update:
            raise RuntimeError("Only whole-update snapshots may be committed")
        counters = {key:deepcopy(getattr(self,key)) for key in ("updates","model_transitions",
            "validation_transitions","dual","replay","best_actor","best_optimizer","last_candidate",
            "best_score","best_update","stale","history","baseline","cost_scale")}
        atomic_save(dict(schema="full_horizon_whole_update_v3", protocol=self.protocol(),
            policy=self.policy.state_dict(), policy_hash=module_hash(self.policy),
            optimizer=self.optimizer.state_dict(), rng=self.rng.get_state(), counters=counters), Path(path))

    def restore(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "full_horizon_whole_update_v3" or identity(payload["protocol"]) != identity(self.protocol()):
            raise ValueError("Checkpoint protocol, source, task or runtime mismatch")
        self.policy.load_state_dict(payload["policy"])
        if module_hash(self.policy) != payload["policy_hash"]:
            raise ValueError("Checkpoint policy digest mismatch")
        self.optimizer.load_state_dict(payload["optimizer"])
        self.rng.set_state(payload["rng"])
        for k,v in payload["counters"].items():
            setattr(self,k,v)
        self.in_update = False
