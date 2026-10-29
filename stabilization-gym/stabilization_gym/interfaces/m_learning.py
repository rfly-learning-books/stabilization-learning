"""Explicit M-learning contract, without changing frozen research trainers.

PortableMFacade is accepted by the existing episode-M learner. Original task
facades expose verified control operations only: neither pretends to support
that learner's incompatible rollout or optimization semantics. All certificate
values here are candidates; positive definiteness is not a stability proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import torch


class CapabilityError(ValueError):
    """The selected backend does not implement the requested learning path."""


class MetricScope(str, Enum):
    EPISODE_FIXED = "episode_fixed"
    GOAL_CONDITIONED = "goal_conditioned_episode_fixed"
    STATE_DEPENDENT = "state_dependent"


@dataclass(frozen=True)
class GradientCapabilities:
    action_wrt_M: bool
    action_wrt_error: bool
    candidate_scores_wrt_certificate: bool = False
    full_state_certificate_derivative: bool = False
    error_wrt_representation: bool = False
    reference_wrt_representation: bool = False
    model_wrt_action: bool = False
    physical_step_autograd: bool = False
    note: str = "Gradients may be piecewise or zero at saturation."


@dataclass(frozen=True)
class TaskDescription:
    name: str
    backend: str
    support_level: str
    metric_scope: MetricScope
    conditioning: tuple[str, ...]
    error_dim: int
    raw_metric_dim: int
    action_description: str
    certificate_factor: float
    gradients: GradientCapabilities
    controller_form: str
    controller_has_state: bool
    control_call_is_pure: bool
    cost_convention: str
    control_call_side_effects: tuple[str, ...] = ()
    supported_algorithms: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    certificate_claim: str = "candidate_only"

    def to_dict(self):
        result = asdict(self)
        result["metric_scope"] = self.metric_scope.value
        return result


@dataclass
class ErrorContext:
    error: torch.Tensor
    metric_context: torch.Tensor | Mapping[str, Any]
    control_context: Mapping[str, Any]
    representation_version: Any = None


@dataclass
class CertificateResult:
    M: torch.Tensor
    value: torch.Tensor
    lower: torch.Tensor | None = None
    full_error_gradient: torch.Tensor | None = None
    claim: str = "candidate_only"


@dataclass
class ControlResult:
    action: Any
    certificate: CertificateResult
    next_controller_state: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    policy_logits: torch.Tensor | None = None


@dataclass
class LearningAssessment:
    """Separate evidence layers; an absent result is unknown, never passed.

    Suggested trajectory keys: evaluated_steps, rise_fraction, decay_violation,
    goal_region_excluded, comparable_fraction, horizon, seed and domain.
    Suggested performance keys: success, held_success, physical_cost,
    paired_arrival_time, failures_capped_at_horizon and heldout_split.
    """
    structural: Mapping[str, Any] = field(default_factory=dict)
    control_coupling: Mapping[str, Any] = field(default_factory=dict)
    trajectory_decrease: Mapping[str, Any] = field(default_factory=dict)
    task_performance: Mapping[str, Any] = field(default_factory=dict)
    analytical_guarantee: Mapping[str, Any] = field(default_factory=lambda: {
        "established": False, "scope": "not_proved"})


def _quadratic(M, error, factor):
    if not torch.is_tensor(M) or not torch.is_tensor(error):
        raise TypeError("M and error must be torch tensors; graph conversion is caller-owned")
    if M.shape[-2:] != (error.shape[-1], error.shape[-1]):
        raise ValueError("M dimensions do not match error coordinates")
    return factor * torch.einsum("...i,...ij,...j->...", error, M, error)


class _ContractFacade:
    def describe(self):
        return self.description

    def require_algorithm(self, name):
        if name not in self.description.supported_algorithms:
            raise CapabilityError(
                f"{self.description.name} does not implement {name!r} through this facade. "
                + "; ".join(self.description.missing_capabilities))
        return self

    def require_capability(self, name):
        if name == "pure_control":
            value = self.description.control_call_is_pure
        elif hasattr(self.description.gradients, name):
            value = getattr(self.description.gradients, name)
        else:
            raise CapabilityError("Unknown capability: " + name)
        if value is not True:
            raise CapabilityError(f"{self.description.name} does not support {name}")
        return self


class PortableMFacade(_ContractFacade):
    """Real compatibility facade for existing fixed-M physical environments.

    The legacy duck-typed methods remain available so StabilizationLearner can
    consume this object directly. Typed error_context/certificate/control are
    extra inspection paths. Controller-state mutation in portable Push is
    explicitly reported; this wrapper does not silently repair old semantics.
    """
    def __init__(self, env, *, name=None):
        for attribute in ("metric_spec", "error_dim", "metric_context_dim", "dt", "max_steps",
                          "action_space", "observation_space", "controller_capabilities"):
            if not hasattr(env, attribute):
                raise CapabilityError("Physical environment lacks " + attribute)
        for method in ("reset", "step", "build_error", "metric_context", "calculate_action_from_M"):
            if not callable(getattr(env, method, None)):
                raise CapabilityError("Physical environment lacks " + method)
        caps = env.controller_capabilities
        if not caps.get("episode_fixed_metric") or caps.get("state_dependent_certificate"):
            raise CapabilityError("PortableMFacade requires episode-fixed M semantics")
        self.env = env
        backend = type(env).__module__ + "." + type(env).__name__
        underlying = getattr(env, "inner_env", getattr(env, "physical_env", getattr(env, "env", env)))
        native_backend = type(underlying).__module__ + "." + type(underlying).__name__
        push = "push_any_shape" in native_backend
        visual = "visual_servo" in native_backend
        audit_pending = hasattr(env, "_pending_metric")
        side_effects = (("recovery_and_pending_action",) if push else ())
        if audit_pending:
            side_effects += ("audit_pending_metric",)
        side_effects = tuple(getattr(env, "control_call_side_effects", side_effects))
        pure = bool(getattr(env, "control_call_is_pure", not side_effects))
        form = getattr(env, "controller_form", None)
        if form is None:
            form = ("fixed_metric_candidate_selection" if push else
                    "weighted_fixed_reference_feedback" if visual else
                    "nominal_feedback_plus_metric_correction" if "classic_control" in native_backend else
                    "analytic_feedback_weighted_by_M")
        gradient = bool(caps.get("action_gradient", False))
        self.description = TaskDescription(
            name=name or type(env).__name__, backend=backend, support_level="implemented_fixed_M",
            metric_scope=MetricScope.GOAL_CONDITIONED if visual else MetricScope.EPISODE_FIXED,
            conditioning=("goal_image_feature",) if visual else
                         ("calibrated_modes", "target") if push else ("task_metric_context",),
            error_dim=env.error_dim, raw_metric_dim=env.metric_spec.raw_dim,
            action_description="physical action in env.action_space; not raw M parameters",
            certificate_factor=1.,
            gradients=GradientCapabilities(
                action_wrt_M=gradient, action_wrt_error=gradient,
                candidate_scores_wrt_certificate=bool(caps.get("differentiable_candidate_scores", False)),
                error_wrt_representation=bool(caps.get("differentiable_representation", False)),
                reference_wrt_representation=bool(caps.get("differentiable_representation", False)),
                model_wrt_action=bool(caps.get("model_gradient", False))),
            controller_form=form, controller_has_state=push,
            control_call_is_pure=pure, cost_convention=caps.get("cost_convention", "unspecified"),
            control_call_side_effects=side_effects,
            supported_algorithms=("episode_metric_ppo",),
            missing_capabilities=("state-dependent certificate training", "joint online Phi updates",
                                  "global Lyapunov or region-of-attraction proof"))
        # The existing learner records scalar facade fields. This digest keeps
        # the underlying scalar settings and interface semantics in that record.
        signature = dict(description=self.description.to_dict(),
            settings={k: v for k, v in vars(env).items() if not k.startswith("_")
                      and isinstance(v, (str, bool, int, float))})
        self.source_signature_sha256 = hashlib.sha256(
            json.dumps(signature, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, physical_action):
        return self.env.step(physical_action)

    def build_error(self, observation, goal=None, representation=None, history=None):
        return self.env.build_error(observation, goal, representation=representation, history=history)

    def metric_context(self, observation=None, goal=None):
        return self.env.metric_context(observation, goal)

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        if certificate is not None:
            raise CapabilityError("Fixed-M facade cannot discard a state-dependent certificate derivative")
        return self.env.calculate_action_from_M(M, e, context=context,
                                               controller_state=controller_state, certificate=None)

    def error_context(self, observation, *, representation=None):
        error, context = self.build_error(observation, representation=representation)
        return ErrorContext(error, self.metric_context(observation), context)

    def certificate(self, M, sample):
        if not torch.is_tensor(M):
            raise TypeError("M must be a torch tensor")
        M = M.to(sample.error)
        return CertificateResult(M, _quadratic(M, sample.error, 1.))

    def control(self, M, sample, *, controller_state=None, training=False):
        if training and not self.description.gradients.action_wrt_M:
            raise CapabilityError("This hard controller has no action gradient; use candidate scores or episode PPO")
        certificate = self.certificate(M, sample)
        action = self.calculate_action_from_M(certificate.M, sample.error, context=sample.control_context,
                                             controller_state=controller_state)
        diagnostics = dict(getattr(self.env, "_diagnostics", {}))
        diagnostics["control_call_is_pure"] = self.description.control_call_is_pure
        state = {"recovery": getattr(self.env, "_recovery", None)} if self.description.controller_has_state else None
        return ControlResult(action, certificate, state, diagnostics)


class OriginalPushMFacade(_ContractFacade):
    """Original full M(D,C) control only; native PPO remains a separate backend."""
    def __init__(self, task, bank, environment):
        if task.name != "push_any_shape" or task.contract.scope != "state_dependent":
            raise CapabilityError("Expected OriginalPushTask with full state-dependent certificate")
        self.task, self.bank, self.environment = task, bank, environment
        self.description = TaskDescription(
            name="original_push_any_shape", backend="native_full_m_ppo", support_level="partial_control_facade",
            metric_scope=MetricScope.STATE_DEPENDENT, conditioning=("D", "C", "valid_modes"),
            error_dim=3, raw_metric_dim=6, action_description="{mode: [B] int, u: [B] speed}",
            certificate_factor=1., gradients=GradientCapabilities(False, False, True, True,
                model_wrt_action=True, note="Hard choice has no action gradient; training logits and full dV/dD do."),
            controller_form="full_certificate_candidate_policy_with_attitude_recovery",
            controller_has_state=True, control_call_is_pure=True, cost_convention="native_reward_maximize",
            missing_capabilities=("common reset/step rollout protocol", "episode-fixed M PPO is incompatible",
                                  "native categorical PPO must be invoked through its original trainer"))

    def error_context(self, data):
        context = self.task.metric_context(data, self.bank)
        return ErrorContext(context["D"], context, {"data": data})

    def certificate(self, actor, sample):
        if torch.is_tensor(actor) or not all(hasattr(actor, x) for x in ("conditioned", "encoder", "value")):
            raise CapabilityError("Pass the full M(D,C) actor, never a detached matrix")
        with torch.enable_grad():
            D = sample.error if sample.error.requires_grad else sample.error.detach().requires_grad_(True)
            context = sample.metric_context
            condition = actor.encoder(context["C"], context["valid_modes"], self.task.cfg.length_scale)
            lower, M = actor.conditioned(D, condition)
            value = _quadratic(M, D, 1.)
            gradient, = torch.autograd.grad(value.sum(), D, create_graph=True, retain_graph=True)
        return CertificateResult(M, value, lower, gradient)

    def control(self, actor, sample, *, controller_state=None, training=False):
        if torch.is_tensor(actor):
            raise CapabilityError("State-dependent control needs a callable full certificate")
        if controller_state is not None:
            raise CapabilityError("Native controller state is owned by the supplied environment; do not override it")
        data = sample.control_context["data"]
        if training:
            candidates = self.task.candidate_scores(actor, data, self.bank, training=True)
            action, next_state = None, None
        else:
            decision, candidates = self.task.calculate_action_from_certificate(actor, data, self.bank, self.environment)
            mode, speed, recovery, failure = decision
            action = {"mode": mode, "u": speed}
            next_state = {"recovery": recovery, "forced_failure": failure}
        cert = CertificateResult(candidates["M"], candidates["V"], candidates["L"], candidates["gradient"])
        return ControlResult(action, cert, next_state, candidates, candidates["logits"])


class OriginalVisualMFacade(_ContractFacade):
    """Original goal-only M controller; renderer/trainer stay external and explicit.

    Goal binding caches detached J exactly as the original deployment path.
    Optional encoder gradients below go through the error, not through cached J.
    This facade does not expose simulator pose to the encoder or M network.
    """
    def __init__(self, adapter, ae, metric):
        self.adapter, self.ae, self.metric = adapter, ae, metric
        self.reference = None
        self.description = TaskDescription(
            name="original_visual_servo", backend="original_derivative_critic", support_level="partial_control_facade",
            metric_scope=MetricScope.GOAL_CONDITIONED, conditioning=("z_goal",),
            error_dim=16, raw_metric_dim=136,
            action_description="[B,4] body velocity [m/s,m/s,m/s,rad/s]",
            certificate_factor=.5, gradients=GradientCapabilities(True, True, False, False,
                error_wrt_representation=True, reference_wrt_representation=False,
                model_wrt_action=False,
                note="Cached J is detached; original trainer has a separate differentiable P0 model."),
            controller_form="original_normalized_command_weighted_fixed_reference_feedback",
            controller_has_state=False, control_call_is_pure=True, cost_convention="labels_and_metrics_external",
            missing_capabilities=("common reset/step and replay interface", "joint Phi/Critic/M optimization scheduler",
                                  "explicit P0/P1 label capability adapter", "closed-loop backend chosen outside this facade"))

    def _versions(self):
        return tuple(p._version for module in (self.ae, self.metric)
                     for p in (*module.parameters(), *module.buffers()))

    def bind_reference(self, goal_rgb, rotation, *, depth=None, device="cpu"):
        self.goal_rgb = goal_rgb.detach().clone() if torch.is_tensor(goal_rgb) else np.array(goal_rgb, copy=True)
        self.device = device
        self.reference = self.adapter.reference(self.ae, self.metric, self.goal_rgb, rotation,
                                                depth=depth, device=device)
        self.bound_versions = self._versions()
        self.binding_generation = getattr(self, "binding_generation", 0) + 1
        return self.reference

    def _check_reference(self):
        if self.reference is None:
            raise CapabilityError("Bind the goal image/reference before computing control")
        if self._versions() != self.bound_versions:
            raise CapabilityError("Phi or M changed; rebind reference between episodes")

    def error_context(self, current_rgb, *, representation_grad=False):
        self._check_reference()
        with torch.set_grad_enabled(representation_grad):
            error, goal = self.adapter.encode_error(self.ae, current_rgb, self.goal_rgb, self.device)
        token = (self.binding_generation, self.bound_versions)
        return ErrorContext(error, {"z_goal": goal}, {"reference": self.reference}, token)

    def certificate(self, M, sample):
        self._check_reference()
        if sample.representation_version != (self.binding_generation, self.bound_versions):
            raise CapabilityError("Error context belongs to a different goal/representation binding")
        if not torch.is_tensor(M):
            raise TypeError("M must be a torch tensor")
        M = M.to(sample.error)
        return CertificateResult(M, _quadratic(M, sample.error, .5))

    def control(self, M, sample, *, controller_state=None, training=False):
        if controller_state is not None:
            raise CapabilityError("This original visual controller is stateless for a fixed reference")
        cert = self.certificate(M, sample)
        action = self.adapter.calculate_action_from_M(cert.M, sample.error, sample.control_context["reference"])
        return ControlResult(action, cert, diagnostics={"reference_J_detached": True})


__all__ = ["CapabilityError", "MetricScope", "GradientCapabilities", "TaskDescription", "ErrorContext",
           "CertificateResult", "ControlResult", "LearningAssessment", "PortableMFacade",
           "OriginalPushMFacade", "OriginalVisualMFacade"]
