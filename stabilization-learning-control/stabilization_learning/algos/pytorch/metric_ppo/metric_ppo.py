"""Cross-task episode-level PPO on certificate parameters.

Only raw Cholesky actions are stochastic. Physics and hard mode selection do
not need derivatives. This is a new M-policy algorithm, not a reproduction of
the original Push mode-index PPO or SLC's LAC objective.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from numbers import Integral
import math
import numpy as np
import torch
from torch import nn
from stabilization_learning.policies.metric import MetricActor
from stabilization_learning.wrappers import rollout_metric
from stabilization_learning.common.compat import canonical_protocol


@dataclass
class TrainConfig:
    episodes: int = 96
    batch_episodes: int = 12
    epochs: int = 4
    learning_rate: float = 2e-4
    value_learning_rate: float = 1e-3
    std: float = .15
    clip: float = .2
    target_kl: float = .03
    certificate_weight: float = .1
    decay: float = .05
    seed: int = 41
    validation_seeds: tuple = (10001, 10002, 10003, 10004)

    def __post_init__(self):
        if any(not isinstance(v, Integral) or isinstance(v, bool) or v <= 0 for v in (self.episodes, self.batch_episodes, self.epochs)) or self.std <= 0:
            raise ValueError("Positive episode/update counts and exploration std required")
        if not all(math.isfinite(v) for v in (self.std, self.learning_rate, self.value_learning_rate,
                                              self.clip, self.target_kl, self.certificate_weight, self.decay)):
            raise ValueError("Training parameters must be finite")
        if not 0 < self.clip < 1 or self.target_kl <= 0:
            raise ValueError("Require 0 < PPO clip < 1 and positive target_kl")
        if min(self.learning_rate, self.value_learning_rate) <= 0 or self.certificate_weight < 0 or self.decay < 0:
            raise ValueError("Invalid optimizer or certificate-penalty parameters")
        if not self.validation_seeds:
            raise ValueError("Validation seeds must be declared before training")


def module_hash(module):
    if module is None:
        return None
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def summarize(records):
    keys = ("fixed_horizon_cost", "accumulated_cost", "objective", "certificate_penalty",
            "V_rise_fraction", "success", "failure", "arrival_time", "terminal_physical_error")
    return {key: float(np.mean([r[key] for r in records])) for key in keys}


class StabilizationLearner:
    """Learner owns M/optimizers; env owns physical action construction.

    Representation is pretrained separately and frozen for this run. Changing
    it requires a fresh policy run, keeping old-policy ratios and contexts valid.
    """
    def __init__(self, env, representation=None, config=None, *, env_id=None, env_kwargs=None):
        self.env, self.representation = env, representation
        self.config = config or TrainConfig()
        self.env_id, self.env_kwargs = env_id or type(env).__name__, env_kwargs or {}
        self.environment_signature = {"class": type(env).__module__ + "." + type(env).__name__,
            "goal": np.asarray(getattr(env, "goal", [])).tolist(),
            "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist(),
            "settings": {key: value for key, value in vars(env).items()
                         if not key.startswith("_") and isinstance(value, (str, bool, int, float))}}
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        if representation is not None:
            representation.eval()
            representation.requires_grad_(False)
            if hasattr(env, "bind_representation"):
                env.bind_representation(representation)
        self.actor = MetricActor(env.metric_context_dim, env.metric_spec, std=self.config.std)
        self.value = nn.Sequential(nn.Linear(env.metric_context_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.config.value_learning_rate)
        self.episodes, self.updates, self.physical_steps = 0, 0, 0
        self.history = []
        self.best_score, self.best_actor = float("inf"), deepcopy(self.actor.state_dict())
        self.representation_hash = module_hash(representation or getattr(env, "representation", None))

    def _protocol(self):
        # Configuration includes train/validation seeds; test seeds are not used here.
        return {"env_id": self.env_id, "env_kwargs": self.env_kwargs,
                "metric_spec": asdict(self.env.metric_spec), "context_dim": self.env.metric_context_dim,
                "physical_horizon": [float(self.env.dt), int(self.env.max_steps)],
                "environment_signature": self.environment_signature,
                "asset_manifest": getattr(self.env, "asset_manifest", None),
                "config": asdict(self.config), "representation_hash": self.representation_hash,
                "algorithm": "episode_M_PPO_v1", "reward": "negative_fixed_horizon_cost_plus_empirical_descent_penalty"}

    def _context(self, observation):
        context = self.env.metric_context(observation).detach().to(dtype=torch.float32, device="cpu")
        if context.shape != (self.env.metric_context_dim,) or not torch.isfinite(context).all():
            raise ValueError("Invalid metric context")
        return context

    def evaluate(self, seeds=None, *, identity=False, best=False, trajectories=False):
        seeds = tuple(self.config.validation_seeds if seeds is None else seeds)
        if not seeds:
            raise ValueError("At least one evaluation seed is required")
        actor = self.actor
        if best:
            actor = deepcopy(self.actor)
            actor.load_state_dict(self.best_actor)
        records = []
        for seed in seeds:
            obs, _ = self.env.reset(seed=int(seed))
            with torch.no_grad():
                M = (torch.eye(self.env.error_dim) * self.env.metric_spec.trace / self.env.error_dim
                     if identity else actor(self._context(obs)))
            result = rollout_metric(self.env, M, obs, representation=self.representation,
                                    certificate_weight=self.config.certificate_weight,
                                    decay=self.config.decay, save_trajectory=trajectories)
            result["seed"] = int(seed)
            records.append(result)
        return {"summary": summarize(records), "episodes": records}

    def _collect(self, count):
        contexts, raws, logps, returns, records = [], [], [], [], []
        for _ in range(count):
            seed = self.config.seed + self.episodes
            if seed in self.config.validation_seeds:
                raise ValueError("Training seed collides with declared validation seeds")
            obs, _ = self.env.reset(seed=seed)
            context = self._context(obs)
            with torch.no_grad():
                dist = self.actor.distribution(context)
                raw = dist.sample()
                logp = dist.log_prob(raw)
                M = self.actor.metric(raw)
            result = rollout_metric(self.env, M, obs, representation=self.representation,
                                    certificate_weight=self.config.certificate_weight, decay=self.config.decay)
            contexts.append(context); raws.append(raw); logps.append(logp)
            # Fixed horizon scaling preserves within-task optimum and avoids
            # varying time units across plants; complete Monte Carlo, no TD.
            returns.append(-result["objective"] / (self.env.dt * self.env.max_steps))
            records.append(result)
            self.episodes += 1
            self.physical_steps += result["physical_steps"]
        return torch.stack(contexts), torch.stack(raws), torch.stack(logps), torch.tensor(returns), records

    def _update(self, context, raw, old_logp, returns):
        with torch.no_grad():
            advantage = returns - self.value(context).squeeze(-1)
            # Centering a single sample would erase its only policy signal.
            if len(advantage) > 1:
                advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        epochs_run = 0
        for _ in range(self.config.epochs):
            logp = self.actor.distribution(context).log_prob(raw)
            ratio = (logp - old_logp).exp()
            kl = ((ratio - 1) - (logp - old_logp)).mean()
            if kl.item() > self.config.target_kl:
                break
            policy_loss = -torch.minimum(ratio * advantage,
                ratio.clamp(1-self.config.clip, 1+self.config.clip) * advantage).mean()
            self.optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            self.optimizer.step()
            value_loss = nn.functional.mse_loss(self.value(context).squeeze(-1), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 10.)
            self.value_optimizer.step()
            epochs_run += 1
        self.updates += 1
        return {"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                "approx_kl": float(kl.detach()), "epochs_run": epochs_run}

    def save(self, path):
        atomic_save({"protocol": self._protocol(), "actor": self.actor.state_dict(),
                     "value": self.value.state_dict(), "optimizer": self.optimizer.state_dict(),
                     "value_optimizer": self.value_optimizer.state_dict(), "episodes": self.episodes,
                     "updates": self.updates, "physical_steps": self.physical_steps,
                     "best_score": self.best_score, "best_actor": self.best_actor, "history": self.history,
                     "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                     "python_rng": random.getstate()}, path)

    def restore(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if canonical_protocol(checkpoint["protocol"]) != canonical_protocol(self._protocol()):
            raise ValueError("Checkpoint protocol differs: environment, phi, metric or training configuration changed")
        for name in ("actor", "value", "optimizer", "value_optimizer"):
            getattr(self, name).load_state_dict(checkpoint[name])
        for name in ("episodes", "updates", "physical_steps", "best_score", "best_actor", "history"):
            setattr(self, name, checkpoint[name])
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])

    def train(self, output_dir, *, resume=False, max_updates=None):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        last = output / "last.pt"
        if resume:
            self.restore(last)
        elif last.exists():
            raise FileExistsError("Existing run: use resume=True or choose a new output directory")
        if not self.history:
            initial = self.evaluate()
            self.best_score = initial["summary"]["objective"]
            self.best_actor = deepcopy(self.actor.state_dict())
            self.history.append({"updates": 0, "episodes": 0, "physical_steps": 0,
                                 "validation": initial["summary"]})
            self.save(last)
        count = 0
        while self.episodes < self.config.episodes:
            if max_updates is not None and count >= max_updates:
                break
            batch = self._collect(min(self.config.batch_episodes, self.config.episodes - self.episodes))
            losses = self._update(*batch[:4])
            validation = self.evaluate()["summary"]
            if validation["objective"] < self.best_score:
                self.best_score = validation["objective"]
                self.best_actor = deepcopy(self.actor.state_dict())
            self.history.append({"updates": self.updates, "episodes": self.episodes,
                "physical_steps": self.physical_steps, "train": summarize(batch[4]),
                "validation": validation, **losses})
            if module_hash(self.representation or getattr(self.env, "representation", None)) != self.representation_hash:
                raise RuntimeError("Representation changed during the policy run")
            self.save(last)
            (output / "progress.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            count += 1
        selected = min(self.history, key=lambda row: row["validation"]["objective"])
        atomic_save({"protocol": self._protocol(), "actor": self.best_actor, "score": self.best_score,
                     "selected_update": selected["updates"], "selected_physical_steps": selected["physical_steps"]}, output / "best.pt")
        self._write_report(output)
        return self.history

    def _write_report(self, output):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [h["physical_steps"] for h in self.history]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, key in zip(axes, ("fixed_horizon_cost", "success", "V_rise_fraction")):
            ax.plot(x, [h["validation"][key] for h in self.history], "o-")
            ax.set(xlabel="New physical transitions", ylabel=key, title="Fixed validation seeds")
            ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(output / "learning.png", dpi=140); plt.close(fig)
        status = {"complete": self.episodes >= self.config.episodes, "episodes": self.episodes,
                  "physical_steps": self.physical_steps, "updates": self.updates, "best_validation_objective": self.best_score,
                  "representation_hash": self.representation_hash, "candidate_certificate_only": True}
        selected = min(self.history, key=lambda row: row["validation"]["objective"])
        status.update(best_selected_update=selected["updates"], best_is_initial_identity=selected["updates"] == 0)
        (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        (output / "REPORT.html").write_text('<!doctype html><meta charset="utf-8"><title>Stabilization learning</title>'
            '<h1>Episode-held M training</h1><p>Validation only; no test-based model selection. '
            'Positive M does not establish a Lyapunov proof.</p><img width="100%" src="learning.png"><pre>'
            + json.dumps(status, indent=2) + '</pre>', encoding="utf-8")
