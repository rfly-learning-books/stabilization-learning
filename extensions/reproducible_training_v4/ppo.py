"""Versioned episode-M PPO with explicit rollouts and whole-update guards."""
from copy import deepcopy
from pathlib import Path
from numbers import Integral
import hashlib
import random
import numpy as np
import torch
from torch import nn

from stabilization_learning.trainer import StabilizationLearner, TrainConfig, summarize
from extensions.unified_training_v3.guards import NumericalGuard
from extensions.unified_training_v3.numerical import require_finite
from .rollout import rollout_metric


class StableEpisodeLearner(StabilizationLearner):
    def __init__(self, env, representation=None, config=None, *, env_id=None, env_kwargs=None):
        config = deepcopy(config or TrainConfig())
        config.__post_init__()
        seeds = (config.seed, *config.validation_seeds)
        if any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in seeds):
            raise ValueError("Integer nonnegative seeds required")
        if len(set(config.validation_seeds)) != len(config.validation_seeds):
            raise ValueError("Validation seeds must be unique")
        if set(range(config.seed, config.seed+config.episodes)) & set(config.validation_seeds):
            raise ValueError("Training and validation seeds overlap")
        self.numerical = NumericalGuard("episode PPO v4")
        super().__init__(env, representation, config, env_id=env_id, env_kwargs=env_kwargs)
        self.numerical.watch_module(self.actor, "actor")
        self.numerical.watch_module(self.value, "value")
        self.numerical.watch_optimizer(self.optimizer, "actor Adam")
        self.numerical.watch_optimizer(self.value_optimizer, "value Adam")
        self._source_hashes = self._sources()

    def _sources(self):
        folder = Path(__file__).parent
        root = folder.parent.parent
        sources = [folder / name for name in ("ppo.py", "rollout.py", "independent.py")]
        sources.extend((folder.parent / "unified_training_v3" / name) for name in ("guards.py", "numerical.py"))
        sources.extend(root / name for name in (
            "scripts/m_learning_suite/task_runtime.py",
            "extensions/fast_m_learning_v2/discrete_tracking.py",
            "stabilization-gym/stabilization_gym/envs/base.py",
            "stabilization-gym/stabilization_gym/envs/robotics/robotics.py",
            "stabilization-gym/stabilization_gym/envs/classic_control/classic.py",
            "stabilization-gym/stabilization_gym/envs/tasks/visual_servo.py",
            "stabilization-learning-control/stabilization_learning/algos/pytorch/metric_ppo/metric_ppo.py",
            "stabilization-learning-control/stabilization_learning/policies/metric.py"))
        return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    def _protocol(self):
        protocol = super()._protocol()
        protocol.update(algorithm="episode_M_PPO_reproducible_v4",
            rollout_semantics="actual successor; denominator-only norm floor; final-hold arrival; applied action trace",
            physical_reference_model_version=getattr(self.env, "reference_model_version", "explicit_external_env"),
            sources=self._source_hashes)
        return protocol

    def _check(self):
        if self._sources() != self._source_hashes:
            error = RuntimeError("An independent-task scientific source changed after initialization")
            self.numerical.fail(error)
            raise error
        self.numerical.check_all()
        self.numerical.check(self.history, "history")
        self.numerical.check(self.best_actor, "selected actor")
        if self.history:
            self.numerical.check(self.best_score, "selected score")

    def _snapshot(self):
        return dict(modules={name:deepcopy(getattr(self,name).state_dict()) for name in
                            ("actor", "value", "optimizer", "value_optimizer")},
            fields={name:deepcopy(getattr(self,name)) for name in
                    ("episodes", "updates", "physical_steps", "best_score", "best_actor", "history")},
            torch_rng=torch.get_rng_state().clone(), numpy_rng=np.random.get_state(), python_rng=random.getstate())

    def _rollback(self, saved):
        for name, value in saved["modules"].items():
            getattr(self,name).load_state_dict(value)
        for name, value in saved["fields"].items():
            setattr(self,name,value)
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        self.optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)

    def evaluate(self, seeds=None, *, identity=False, best=False, trajectories=False):
        self.numerical.guard(); self._check()
        seeds = tuple(self.config.validation_seeds if seeds is None else seeds)
        if not seeds or any(isinstance(s,bool) or not isinstance(s,Integral) or s < 0 for s in seeds):
            raise ValueError("Declare nonempty integer evaluation seeds")
        actor = deepcopy(self.actor) if best else self.actor
        if best:
            actor.load_state_dict(self.best_actor)
        records = []
        try:
            for seed in seeds:
                obs, _ = self.env.reset(seed=int(seed))
                with torch.no_grad():
                    M = (torch.eye(self.env.error_dim)*self.env.metric_spec.trace/self.env.error_dim
                         if identity else actor(self._context(obs)))
                row = rollout_metric(self.env, M, obs, representation=self.representation,
                    certificate_weight=self.config.certificate_weight, decay=self.config.decay,
                    save_trajectory=trajectories)
                row["seed"] = int(seed); records.append(row)
            result = dict(summary=summarize(records), episodes=records)
            self.numerical.check(result, "evaluation")
            return result
        except BaseException as error:
            self.numerical.fail(error)
            raise

    def _collect(self, count):
        self.numerical.guard(); self._check()
        if isinstance(count,bool) or not isinstance(count,Integral) or count < 1 or self.episodes+count > self.config.episodes:
            raise ValueError("Collection exceeds the declared episode/seed budget")
        contexts, raws, logps, returns, records = [], [], [], [], []
        try:
            for _ in range(count):
                seed = self.config.seed+self.episodes
                obs, _ = self.env.reset(seed=seed)
                context = self._context(obs)
                with torch.no_grad():
                    dist = self.actor.distribution(context)
                    raw = dist.sample(); logp = dist.log_prob(raw)
                    M = self.actor.metric(raw)
                row = rollout_metric(self.env, M, obs, representation=self.representation,
                    certificate_weight=self.config.certificate_weight, decay=self.config.decay)
                contexts.append(context); raws.append(raw); logps.append(logp)
                returns.append(-row["objective"]/(self.env.dt*self.env.max_steps)); records.append(row)
                self.episodes += 1; self.physical_steps += row["physical_steps"]
            result = torch.stack(contexts), torch.stack(raws), torch.stack(logps), torch.tensor(returns), records
            self.numerical.check(result, "collected batch")
            return result
        except BaseException as error:
            self.numerical.fail(error)
            raise

    def _update(self, context, raw, old_logp, returns):
        self.numerical.guard(); self._check()
        saved = self._snapshot()
        try:
            self.numerical.check((context,raw,old_logp,returns), "PPO input")
            with torch.no_grad():
                advantage = returns-self.value(context).squeeze(-1)
                if len(advantage)>1:
                    advantage = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
            self.numerical.check(advantage, "advantage")
            epochs_run = 0
            policy_loss = value_loss = returns.new_zeros(())
            for _ in range(self.config.epochs):
                logp = self.actor.distribution(context).log_prob(raw)
                ratio = (logp-old_logp).exp()
                kl = ((ratio-1)-(logp-old_logp)).mean()
                self.numerical.check((logp,ratio,kl), "PPO ratio")
                if kl.item()>self.config.target_kl:
                    break
                policy_loss = -torch.minimum(ratio*advantage,
                    ratio.clamp(1-self.config.clip,1+self.config.clip)*advantage).mean()
                value_loss = nn.functional.mse_loss(self.value(context).squeeze(-1),returns)
                self.numerical.check((policy_loss,value_loss), "PPO losses")
                self.optimizer.zero_grad(); policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True)
                self.optimizer.step()
                self.value_optimizer.zero_grad(); value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(),10.,error_if_nonfinite=True)
                self.value_optimizer.step()
                self.numerical.check_all()
                epochs_run += 1
            self.updates += 1
            result = dict(policy_loss=float(policy_loss.detach()),value_loss=float(value_loss.detach()),
                          approx_kl=float(kl.detach()),epochs_run=epochs_run)
            self.numerical.check(result,"PPO update")
            return result
        except BaseException as error:
            self._rollback(saved)
            self.numerical.fail(error)
            raise

    def save(self, path):
        self.numerical.guard(); self._check()
        return super().save(path)

    def restore(self, path):
        self.numerical.guard(); self._check()
        saved = self._snapshot()
        try:
            payload = torch.load(path,map_location="cpu",weights_only=False)
            require_finite({key:value for key,value in payload.items() if key!="best_score"},"checkpoint")
            if payload.get("history"):
                require_finite(payload["best_score"],"checkpoint score")
            super().restore(path)
            self._check()
        except BaseException as error:
            self._rollback(saved)
            self.numerical.fail(error)
            raise

    def close(self):
        self.numerical.close()
        self.env.close()
