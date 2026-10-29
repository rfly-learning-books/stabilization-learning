"""Observer-specific context and update contract.

Truth is intentionally a separate object.  The policy actor and gain generator
only receive ObserverContext, so a context cannot accidentally carry the true
state or a test label.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..core.gain import information_gain


@dataclass(frozen=True)
class ObserverContext:
    estimate: torch.Tensor
    innovation: torch.Tensor
    jacobian: torch.Tensor
    quality: torch.Tensor
    control: torch.Tensor
    prior_metric: torch.Tensor | None = None

    def features(self) -> torch.Tensor:
        parts = [self.estimate.reshape(*self.estimate.shape[:-1], -1),
                 self.innovation.reshape(*self.innovation.shape[:-1], -1),
                 self.jacobian.reshape(*self.jacobian.shape[:-3], -1),
                 self.quality.reshape(*self.quality.shape[:-1], -1),
                 self.control.reshape(*self.control.shape[:-1], -1)]
        if self.prior_metric is not None:
            parts.append(self.prior_metric.reshape(*self.prior_metric.shape[:-2], -1))
        return torch.cat(parts, dim=-1)


@dataclass(frozen=True)
class ObserverTruth:
    """Training/evaluation-only hidden state. Never pass to an actor."""
    state: torch.Tensor
    timestamp: float | None = None


@dataclass(frozen=True)
class ObserverUpdate:
    prior_estimate: torch.Tensor
    posterior_estimate: torch.Tensor
    correction: torch.Tensor
    gain: torch.Tensor
    metric: torch.Tensor


class MetricObserver:
    """M-generated observer update with a closed-form gain."""

    def __init__(self, actor, measurement_covariance: torch.Tensor):
        self.actor = actor
        self.measurement_covariance = measurement_covariance

    def update(self, context: ObserverContext) -> ObserverUpdate:
        features = context.features()
        param = next(self.actor.parameters(), None)
        if param is not None:
            features = features.to(dtype=param.dtype, device=param.device)
        M = self.actor(features).to(context.estimate)
        R = self.measurement_covariance.to(M)
        L = information_gain(M, context.jacobian.to(M), R)
        correction = (L @ context.innovation.unsqueeze(-1)).squeeze(-1)
        posterior = context.estimate + correction
        return ObserverUpdate(context.estimate, posterior, correction, L, M)


