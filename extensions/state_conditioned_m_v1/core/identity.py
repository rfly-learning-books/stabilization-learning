"""Fixed identity metric provider for paired baselines."""
from __future__ import annotations
import torch
from torch import nn

from .paths import ROOT  # noqa: F401
from stabilization_gym.common.types import MetricSpec
from stabilization_learning.policies.metric import CholeskyMetric


class IdentityMetricActor(nn.Module):
    def __init__(self, context_dim: int, spec: MetricSpec):
        super().__init__()
        self.context_dim = int(context_dim)
        self.spec = spec
        self.metric = CholeskyMetric(spec)
        raw = self.metric.identity_raw()
        self.register_buffer("raw_identity", raw)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] != self.context_dim:
            raise ValueError("Invalid context dimension")
        raw = self.raw_identity.to(context).reshape((1,) * (context.ndim - 1) + (-1,))
        raw = raw.expand(*context.shape[:-1], -1)
        return self.metric(raw)
