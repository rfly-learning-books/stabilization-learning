"""Trace-normalized Cholesky metrics. Positive definiteness is structural."""
import math
import torch
from torch import nn
from stabilization_gym.common.types import MetricSpec


class CholeskyMetric(nn.Module):
    def __init__(self, spec: MetricSpec):
        super().__init__()
        self.spec = spec
        rows, cols = torch.tril_indices(spec.error_dim, spec.error_dim)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.register_buffer("diagonal", rows == cols)

    def identity_raw(self, *, dtype=torch.float32, device=None):
        device = self.rows.device if device is None else device
        raw = torch.zeros(self.spec.raw_dim, dtype=dtype, device=device)
        raw[self.diagonal.to(device)] = math.log(math.expm1(1.0))
        return raw

    def forward(self, raw):
        if raw.shape[-1] != self.spec.raw_dim:
            raise ValueError("Wrong number of lower-triangular parameters")
        if not torch.isfinite(raw).all():
            raise ValueError("Cholesky parameters must be finite")
        values = torch.where(self.diagonal, torch.nn.functional.softplus(raw) + 1e-8, raw)
        L = raw.new_zeros((*raw.shape[:-1], self.spec.error_dim, self.spec.error_dim))
        L[..., self.rows, self.cols] = values
        L = L / L.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        gram = L @ L.transpose(-1, -2)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        eye = torch.eye(self.spec.error_dim, dtype=raw.dtype, device=raw.device)
        return self.spec.eigen_floor * eye + (self.spec.trace - self.spec.error_dim * self.spec.eigen_floor) * gram / trace.clamp_min(1e-20)


def quadratic_certificate(error, M):
    return torch.einsum("...i,...ij,...j->...", error, M, error)


class MetricActor(nn.Module):
    """Goal/context -> distribution on one episode-held Cholesky factor.

    The Gaussian is in raw parameter space. No logdet correction is needed for
    PPO ratios because the sampled action is raw, not its many-to-one SPD map.
    """
    def __init__(self, context_dim, spec, hidden=64, std=.15):
        super().__init__()
        self.metric = CholeskyMetric(spec)
        self.net = nn.Sequential(nn.Linear(context_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, spec.raw_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("center", self.metric.identity_raw())
        self.register_buffer("std", torch.full((spec.raw_dim,), float(std)))

    def mean(self, context):
        return self.center + self.net(context)

    def distribution(self, context):
        return torch.distributions.Independent(torch.distributions.Normal(self.mean(context), self.std), 1)

    def forward(self, context):
        return self.metric(self.mean(context))
