"""Capability-neutral rollout driver with explicit execution sessions.

The driver never switches on a task name.  It asks an adapter for an ActionPlan
and delegates execution to an injected session.  In particular an observer's
correction is an estimator update, not a physical actuator command.  Labels are
kept in a separate field and never fed back into make_context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import torch

from stabilization_gym.tasks_v2.contracts import ActionPlan, TaskContext


def snapshot(value):
    """Replay owns detached copies; module objects never belong in observations."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    return value


@dataclass
class EpisodeStart:
    observation: Any
    reference: Any = None
    memory: Mapping[str, Any] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    applied_action: Any
    memory: Mapping[str, Any] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)
    supervision: Mapping[str, Any] | None = None


@dataclass
class CollectedStep:
    observation: Any
    next_observation: Any
    plan: ActionPlan
    applied_action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    supervision: Mapping[str, Any] | None = None
    candidate_distribution: Mapping[str, Any] | None = None


@dataclass
class Rollout:
    steps: list[CollectedStep]
    final_memory: Mapping[str, Any]
    terminated: bool
    truncated: bool
    collector_cutoff: bool

    @property
    def total_reward(self):
        return sum(step.reward for step in self.steps)


class PhysicalActionSession:
    """Execute the adapter's physical action using the standard Gym boundary."""
    def __init__(self, physical):
        self.physical = physical

    def reset(self, *, seed=None, options=None, reference=None, memory=None):
        observation, info = self.physical.reset(seed=seed, options=options)
        clean = dict(info)
        clean.pop("supervision", None)
        return EpisodeStart(snapshot(observation), reference,
                            {} if memory is None else snapshot(memory), snapshot(clean))

    def apply_plan(self, context: TaskContext, plan: ActionPlan):
        if plan.action is None:
            raise ValueError("No physical action: inject a session that explicitly handles adapter failure")
        observation, reward, terminated, truncated, info = self.physical.step(plan.action)
        clean = dict(info)
        labels = clean.pop("supervision", None)
        return ExecutionResult(snapshot(observation), float(reward), bool(terminated), bool(truncated),
                               snapshot(clean.get("applied_action", plan.action)), snapshot(plan.memory),
                               snapshot(clean), snapshot(labels))


class ObserverSession:
    """Apply internal estimate memory; send only known forcing to the plant.

    Optional labels provide estimation reward and true-error audit.  Inference
    mode neither calls physical.supervision nor computes a hidden-error reward.
    This session retains the legacy correction-then-prediction timing contract.
    """
    def __init__(self, physical, adapter, *, supervised=False):
        self.physical, self.adapter = physical, adapter
        self.supervised = bool(supervised)
        self.observation = None

    def reset(self, *, seed=None, options=None, reference=None, memory=None):
        observation, info = self.physical.reset(seed=seed, options=options)
        self.observation = snapshot(observation)
        if memory is None:
            memory = self.adapter.initial_memory(scenario=(options or {}).get("scenario", "nominal"))
        clean = dict(info)
        clean.pop("supervision", None)
        return EpisodeStart(snapshot(observation), reference, snapshot(memory), snapshot(clean))

    def apply_plan(self, context: TaskContext, plan: ActionPlan):
        if self.observation is None:
            raise RuntimeError("Reset observer session before execution")
        # Labels are passed only to the explicitly offline scoring method.
        labels = (self.adapter.evaluate_supervision(self.physical.supervision(), context, plan)
                  if self.supervised else None)
        failed = bool(plan.diagnostics.get("prior_failure", False) or
                      (labels is not None and labels["failure"]))
        physical_action = plan.diagnostics["physical_action"]
        if failed:
            observation = snapshot(self.observation)
            terminated, truncated = True, False
            info = {"dt": 0., "failure": True, "reason": "observer_domain_exit"}
        else:
            observation, _, terminated, truncated, info = self.physical.step(physical_action)
            info = dict(info)
        info.pop("supervision", None)
        reward = 0. if labels is None else float(labels["reward"])
        info["estimation_cost_available"] = labels is not None
        if labels is not None:
            info["physical_cost"] = float(labels["physical_cost"])
            info["complete"] = bool(labels["complete"])
        self.observation = snapshot(observation)
        return ExecutionResult(snapshot(observation), reward, bool(terminated), bool(truncated),
                               snapshot(physical_action), snapshot(plan.memory), snapshot(info), snapshot(labels))


class RolloutDriver:
    """Sample a frozen strategy; learning recomputes graphs from raw records.

    max_steps is a collector bound.  If reached it is recorded separately from
    the physical task's terminated/truncated flags, preserving bootstrap meaning.
    apply_plan is an injected callable, not a task-name dispatch.
    This is a deterministic execution collector, not a PPO sampler.  Candidate
    Categorical probabilities are retained as diagnostics only; the adapter may
    choose their argmax, so they are not behavior-policy likelihoods.
    """
    def __init__(self, adapter, certificate, apply_plan: Callable, *, features=None, model=None, binding=None):
        self.adapter, self.certificate, self.apply_plan = adapter, certificate, apply_plan
        self.features, self.model, self.binding = features, model, binding

    def run(self, start: EpisodeStart, *, max_steps: int, guard: Callable | None = None):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("Collector max_steps must be a positive integer")
        observation, memory = snapshot(start.observation), snapshot(start.memory)
        steps = []
        terminated = truncated = False
        for _ in range(max_steps):
            if guard is not None:
                guard()
            with torch.no_grad():
                context = self.adapter.make_context(observation, start.reference,
                    features=self.features, model=self.model, memory=memory, binding=self.binding)
                plan = self.adapter.construct_action(context, self.certificate)
            candidate_distribution = None
            if plan.distribution is not None:
                if not isinstance(plan.distribution, torch.distributions.Categorical):
                    raise TypeError("Unsupported candidate distribution: provide explicit replay diagnostics")
                candidate_distribution = {
                    "kind": "categorical_candidates",
                    "logits": snapshot(plan.distribution.logits),
                    "probabilities": snapshot(plan.distribution.probs),
                    "behavior_log_probability_available": False,
                }
            execution = self.apply_plan(context, plan)
            if not np.isfinite(execution.reward):
                raise FloatingPointError("Nonfinite reward in physical transition")
            if "supervision" in execution.info:
                raise ValueError("Execution labels must use the separate supervision field")
            copied_plan = ActionPlan(snapshot(plan.action), snapshot(plan.memory), snapshot(plan.diagnostics), None)
            steps.append(CollectedStep(snapshot(observation), snapshot(execution.observation), copied_plan,
                snapshot(execution.applied_action), float(execution.reward), bool(execution.terminated),
                bool(execution.truncated), snapshot(execution.info), snapshot(execution.supervision),
                candidate_distribution))
            observation, memory = snapshot(execution.observation), snapshot(execution.memory)
            terminated, truncated = bool(execution.terminated), bool(execution.truncated)
            if terminated or truncated:
                break
        return Rollout(steps, memory, terminated, truncated, not (terminated or truncated))
