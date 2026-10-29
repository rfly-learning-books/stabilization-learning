from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def frozen_parameters(*modules):
    """Freeze weights, not input/action derivatives needed by control."""
    old = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    for p, _ in old:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in old:
            p.requires_grad_(flag)


class MetricActor(nn.Module):
    def __init__(self, latent=32, hidden=128, floor=0.1, identity=False):
        super().__init__()
        self.latent, self.floor, self.identity = latent, floor, identity
        if not 0 < floor < 1:
            raise ValueError("Metric eigenvalue floor must be in (0,1)")
        self.net = nn.Sequential(nn.Linear(3 * latent, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, latent * (latent + 1) // 2))
        rows, cols = torch.tril_indices(latent, latent)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.register_buffer("diagonal", rows == cols)
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[self.diagonal] = math.log(math.expm1(1.0 - 1e-3))

    def metric(self, error, reference):
        # Reconstruct z from e: differentiation w.r.t. e includes the z input
        # path of M, rather than incorrectly holding current z constant.
        current = reference + math.sqrt(self.latent) * error
        raw = self.net(torch.cat([current, reference, error], dim=-1))
        raw = torch.where(self.diagonal, F.softplus(raw) + 1e-3, raw)
        lower = raw.new_zeros(*raw.shape[:-1], self.latent, self.latent)
        lower[..., self.rows, self.cols] = raw
        eye = torch.eye(self.latent, device=raw.device, dtype=raw.dtype)
        gram = lower @ lower.transpose(-1, -2)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        matrix = self.floor * eye + self.latent * (1 - self.floor) * gram / trace
        if self.identity:
            matrix = eye.expand_as(matrix) + raw.sum(-1)[..., None, None] * 0.0
        return lower, matrix

    def certificate(self, error, reference):
        lower, matrix = self.metric(error, reference)
        value = 0.5 * torch.einsum("...i,...ij,...j->...", error, matrix, error)
        return value, lower, matrix


class CostCritic(nn.Module):
    """Nonnegative action cost; not automatically a strict state Lyapunov function."""
    def __init__(self, latent=32, hidden=256, output=32):
        super().__init__()
        self.latent = latent
        self.net = nn.Sequential(nn.Linear(3 * latent + 4, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output))

    def forward(self, current, reference, action):
        error = (current - reference) / math.sqrt(self.latent)
        vector = self.net(torch.cat([current, reference, error, action], dim=-1))
        return vector.square().mean(-1)


@torch.no_grad()
def ema(target, source, rate):
    for to, source_value in zip(target.parameters(), source.parameters()):
        to.lerp_(source_value, rate)
    for to, source_value in zip(target.buffers(), source.buffers()):
        to.copy_(source_value)
