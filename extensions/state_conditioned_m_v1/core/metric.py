"""Trace-normalized state/target conditioned metric actor.

The actor consumes only deployment-legal context supplied by the task.  The
resulting SPD matrix has fixed trace and a spectral floor through the same
Cholesky construction used by the public learning package.
"""
from __future__ import annotations
import torch
from torch import nn

from .paths import ROOT  # noqa: F401 - path side effect
from stabilization_gym.common.types import MetricSpec
from stabilization_learning.policies.metric import CholeskyMetric


class StateDependentMetricActor(nn.Module):
    """Context -> SPD metric with identity initialization.

    The network predicts a residual in Cholesky raw coordinates.  The identity
    metric is the initial policy, and a zero final layer makes the initial raw
    action exactly the public identity parameterization.
    """

    def __init__(self, context_dim: int, spec: MetricSpec, hidden: int = 64):
        super().__init__()
        if context_dim <= 0 or hidden <= 0:
            raise ValueError("Positive context/hidden dimensions required")
        self.spec = spec
        self.metric = CholeskyMetric(spec)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, spec.raw_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("center", self.metric.identity_raw())
        self.context_dim = int(context_dim)

    def raw(self, context: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(context) or not context.is_floating_point():
            raise TypeError("Metric context must be a floating tensor")
        if context.shape[-1] != self.context_dim or not torch.isfinite(context).all():
            raise ValueError("Invalid metric context")
        return self.center.to(context) + self.net(context)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.metric(self.raw(context))
