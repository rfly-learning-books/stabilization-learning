"""Visual task geometry and differentiable control, without trainable ownership.

The policy accepts only public RGB/goal observations and image tangents. Physical
pose labels are arguments to the *training* model/cost methods, never the policy.
No rendering derivative is fabricated here: tangents are a local image model and
must be checked against real image transitions by an environment integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
from torch import Tensor

from .contracts import ActionPlan, Capabilities, TaskContext


@dataclass(frozen=True)
class VisualObservation:
    """Public policy inputs, all batched; tangents have shape B,A,C,H,W."""

    image: Tensor
    goal_image: Tensor
    image_tangents: Tensor

    def validate(self) -> None:
        if self.image.ndim != 4 or self.goal_image.shape != self.image.shape:
            raise ValueError("Current and goal RGB must have the same B,C,H,W shape")
        if self.image.shape[1] != 3:
            raise ValueError("This adapter expects RGB images")
        expected = (len(self.image), 4, *self.image.shape[1:])
        if tuple(self.image_tangents.shape) != expected:
            raise ValueError(f"Expected command image tangents {expected}")
        for value in (self.image, self.goal_image, self.image_tangents):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Policy inputs must be finite floating point tensors")
            if value.device != self.image.device or value.dtype != self.image.dtype:
                raise ValueError("Policy input dtype/device mismatch")


@dataclass(frozen=True)
class VisualControlConfig:
    gain: float = 0.7
    damping: float = 0.02
    diagonal_damping: float = 0.01
    translation_speed: float = 0.4
    yaw_speed: float = 0.35
    yaw_error_scale: float = math.radians(25.0)

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.gain, self.damping, self.diagonal_damping, self.translation_speed,
            self.yaw_speed, self.yaw_error_scale,
        )):
            raise ValueError("Control configuration must be finite")
        if min(self.gain, self.damping, self.translation_speed,
               self.yaw_speed, self.yaw_error_scale) <= 0:
            raise ValueError("Gains, damping and physical units must be positive")
        if self.diagonal_damping < 0:
            raise ValueError("Diagonal damping cannot be negative")


@dataclass
class VisualPolicyOutput:
    action: Tensor
    raw_action: Tensor
    error: Tensor
    goal: Tensor
    interaction: Tensor
    metric: Tensor
    value: Tensor
    hessian: Tensor


def radial_limit(action: Tensor) -> Tensor:
    """One positive scale for all four components preserves action direction."""
    scale = torch.maximum(action[..., :3].norm(dim=-1), action[..., 3].abs())
    return action / scale.clamp_min(1.0).unsqueeze(-1)


def quadratic_value(error: Tensor, metric: Tensor) -> Tensor:
    return 0.5 * torch.einsum("...i,...ij,...j->...", error, metric, error)


def pinhole_image_tangents(
    image: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    camera_from_command: Tensor | None = None,
) -> Tensor:
    """Local brightness-constancy image model for a stationary scene.

    ``camera_from_command`` maps four normalized commands to camera twist ordered
    vx,vy,vz,wx,wy,wz. Its units are metres/radians per second. The default assumes
    aligned camera/body axes and scales 0.4,0.4,0.4,0.35. RGB derivatives therefore
    have units RGB units / second / normalized command. Depth is an online
    estimate, not privileged scene truth. Occlusion/lighting effects are absent.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected B,3,H,W RGB")
    b, _, h, w = image.shape
    if min(h, w) < 3:
        raise ValueError("At least 3 by 3 pixels are required for spatial gradients")
    if depth.ndim == 3:
        depth = depth[:, None]
    if depth.shape != (b, 1, h, w):
        raise ValueError("Depth must have B,1,H,W shape")
    if not torch.isfinite(image).all() or not torch.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError("Finite RGB and strictly positive finite depth are required")
    k = torch.as_tensor(intrinsics, device=image.device, dtype=image.dtype)
    if k.shape == (3, 3):
        k = k.expand(b, -1, -1)
    if k.shape != (b, 3, 3) or not torch.isfinite(k).all():
        raise ValueError("Intrinsics must be finite 3,3 or B,3,3")
    fx, fy = k[:, 0, 0, None, None], k[:, 1, 1, None, None]
    if (fx <= 0).any() or (fy <= 0).any():
        raise ValueError("Focal lengths must be positive")
    yy, xx = torch.meshgrid(torch.arange(h, device=image.device, dtype=image.dtype),
                           torch.arange(w, device=image.device, dtype=image.dtype), indexing="ij")
    x = (xx[None] - k[:, 0, 2, None, None]) / fx
    y = (yy[None] - k[:, 1, 2, None, None]) / fy
    iz = depth[:, 0].reciprocal()
    zero = torch.zeros_like(x)
    lx = torch.stack((-iz, zero, x * iz, x * y, -(1 + x.square()), y), dim=1) * fx[:, None]
    ly = torch.stack((zero, -iz, y * iz, 1 + y.square(), -x * y, -x), dim=1) * fy[:, None]
    gy, gx = torch.gradient(image, dim=(-2, -1))
    # Brightness transport: dI/dt = -grad(I) dot image velocity.
    camera_tangents = -(gx[:, None] * lx[:, :, None] + gy[:, None] * ly[:, :, None])
    if camera_from_command is None:
        a = image.new_zeros(6, 4)
        a[0, 0] = a[1, 1] = a[2, 2] = 0.4
        a[5, 3] = 0.35
    else:
        a = torch.as_tensor(camera_from_command, device=image.device, dtype=image.dtype)
    if a.shape == (6, 4):
        a = a.expand(b, -1, -1)
    if a.shape != (b, 6, 4) or not torch.isfinite(a).all():
        raise ValueError("camera_from_command must be finite 6,4 or B,6,4")
    return torch.einsum("btchw,bta->bachw", camera_tangents, a)


class VisualTaskAdapter:
    """Task-side construction of a goal-metric visual velocity controller."""

    capabilities = Capabilities(
        action_gradient=True, feature_gradient=True,
        interaction_parameter_gradient=True, candidate_probabilities=False,
        prediction_gradient=True, physical_step_gradient=False,
        action_kind="normalized_body_velocity_and_yaw_rate",
        note="Differentiable current-image J and nominal P0 label model; no renderer gradient",
    )

    def __init__(self, config: VisualControlConfig | None = None):
        self.config = config or VisualControlConfig()

    @staticmethod
    def interaction(features: Any, image: Tensor, tangents: Tensor,
                    *, create_graph: bool = True) -> Tensor:
        """JVPs retain mixed input/parameter derivatives in training mode."""
        columns = []
        for axis in range(tangents.shape[1]):
            _, tangent = torch.autograd.functional.jvp(
                features.encode, image, tangents[:, axis],
                create_graph=create_graph, strict=False,
            )
            columns.append(tangent)
        return torch.stack(columns, dim=-1)

    def solve(self, error: Tensor, interaction: Tensor, metric: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if metric.shape[-2:] != (error.shape[-1], error.shape[-1]):
            raise ValueError("Metric dimension does not match visual features")
        if interaction.shape != (*error.shape, 4):
            raise ValueError("Expected B,feature_dim,4 interaction matrix")
        jt = interaction.transpose(-1, -2)
        gram = jt @ metric @ interaction
        eye = torch.eye(4, dtype=error.dtype, device=error.device)
        hessian = (gram + self.config.diagonal_damping * torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1))
                   + self.config.damping * eye)
        gradient = (jt @ metric @ error.unsqueeze(-1)).squeeze(-1)
        raw = -self.config.gain * torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
        return radial_limit(raw), raw, hessian

    def policy(self, features: Any, metric_provider: Any, observation: VisualObservation,
               *, create_graph: bool = True) -> VisualPolicyOutput:
        observation.validate()
        z = features.encode(observation.image)
        goal = features.encode(observation.goal_image)
        error = z - goal
        interaction = self.interaction(features, observation.image, observation.image_tangents,
                                       create_graph=create_graph)
        metric = metric_provider(goal)
        action, raw, hessian = self.solve(error, interaction, metric)
        return VisualPolicyOutput(action, raw, error, goal, interaction, metric,
                                  quadratic_value(error, metric), hessian)

    def make_context(self, observation: Tensor | VisualObservation, reference: Any = None, *,
                     features: Any = None, model: Any = None, memory: Any = None,
                     binding: Any = None) -> TaskContext:
        if features is None:
            raise ValueError("Visual features must be provided explicitly")
        if isinstance(observation, VisualObservation):
            public = observation
        else:
            if model is None or not hasattr(model, "image_tangents"):
                raise ValueError("A public image-tangent model is required")
            public = VisualObservation(observation, reference, model.image_tangents(observation))
        public.validate()
        z, goal = features.encode(public.image), features.encode(public.goal_image)
        j = self.interaction(features, public.image, public.image_tangents,
                             create_graph=torch.is_grad_enabled())
        return TaskContext(error=z - goal, metric_context=goal,
                           model_context={"interaction": j}, memory=memory or {}, binding=binding)

    def construct_action(self, context: TaskContext, certificate: Any) -> ActionPlan:
        # Bound certificates must perform their binding check. A raw callable is
        # an explicitly unbound goal-metric provider, not a checked certificate.
        if hasattr(certificate, "evaluate"):
            if certificate.binding.metric_scope not in {"goal_conditioned", "episode_fixed"}:
                raise ValueError("This visual controller requires a goal-only, episode-fixed metric")
            result = certificate.evaluate(context.error, context.metric_context, binding=context.binding)
            metric, value = result.M, result.value
        else:
            if context.binding is not None:
                raise ValueError("A bound context requires a version-checking certificate")
            metric = certificate(context.metric_context)
            value = quadratic_value(context.error, metric)
        action, raw, hessian = self.solve(context.error, context.model_context["interaction"], metric)
        return ActionPlan(action=action, memory=context.memory,
                          diagnostics={"V": value,
                                       "M": metric, "J": context.model_context["interaction"],
                                       "raw_action": raw, "hessian": hessian})

    def nominal_step(self, offset: Tensor, action: Tensor, dt: Tensor | float) -> Tensor:
        """Differentiable P0 body-velocity model, used ONLY with training labels."""
        dt = torch.as_tensor(dt, dtype=offset.dtype, device=offset.device)
        if dt.ndim == offset.ndim and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)
        if (dt <= 0).any() or not torch.isfinite(dt).all():
            raise ValueError("Positive finite transition durations are required")
        yaw_mid = offset[..., 3] + 0.5 * self.config.yaw_speed * action[..., 3] * dt
        c, s = yaw_mid.cos(), yaw_mid.sin()
        v = self.config.translation_speed
        delta = torch.stack((v * (c * action[..., 0] - s * action[..., 1]) * dt,
                             v * (s * action[..., 0] + c * action[..., 1]) * dt,
                             v * action[..., 2] * dt,
                             self.config.yaw_speed * action[..., 3] * dt), dim=-1)
        return offset + delta

    def physical_error(self, offset: Tensor) -> Tensor:
        scales = offset.new_tensor((1.0, 1.0, 1.0, self.config.yaw_error_scale))
        return (offset / scales).square().sum(dim=-1)


__all__ = ["VisualObservation", "VisualControlConfig", "VisualPolicyOutput",
           "VisualTaskAdapter", "pinhole_image_tangents", "quadratic_value", "radial_limit"]
