"""Bounded geometry-conditioned features and causal normalized adaptation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .config import ObserverConfig
from .state import CONTEXT_DIM, RESIDUAL_SCALE


@dataclass
class Memory:
    weights: torch.Tensor
    hidden: torch.Tensor

    def detach(self):
        return Memory(self.weights.detach(), self.hidden.detach())

    def expand(self, count):
        return Memory(self.weights.expand(count, -1, -1), self.hidden.expand(count, -1))


def projected_update(weights, phi, error, eta, epsilon, radius):
    """Euclidean projection onto a Frobenius ball, differentiable a.e."""
    denominator = epsilon + phi.square().sum(-1, keepdim=True)
    updated = weights - (eta / denominator).unsqueeze(-1) * error.unsqueeze(-1) * phi.unsqueeze(-2)
    norm = torch.linalg.vector_norm(updated, dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return updated * (radius / norm).clamp(max=1.0)


class ResidualObserver(nn.Module):
    def __init__(self, config: ObserverConfig):
        super().__init__()
        self.settings = config
        self.image_encoder = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten(),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(16 * 8 * 8 + 6, config.geometry_dim), nn.Tanh()
        )
        self.feature_network = nn.Sequential(
            nn.Linear(CONTEXT_DIM + config.geometry_dim, 128), nn.SiLU(),
            nn.Linear(128, config.feature_dim), nn.Tanh(),
        )
        if config.static_residual_head:
            self.static_residual_network = nn.Sequential(
                nn.Linear(CONTEXT_DIM + config.geometry_dim, 128), nn.SiLU(),
                nn.Linear(128, 12),
            )
            # Upgrading an old observer starts with exactly the old prediction.
            nn.init.zeros_(self.static_residual_network[-1].weight)
            nn.init.zeros_(self.static_residual_network[-1].bias)
        if config.gain == "learned":
            self.feedback_gru = nn.GRUCell(26, config.hidden_dim)
            self.gain_head = nn.Linear(config.hidden_dim, 1)
        self.register_buffer("residual_scale", torch.tensor(RESIDUAL_SCALE.copy()))

    def initial(self, batch=1):
        parameter = next(self.parameters())
        return Memory(parameter.new_zeros(batch, 12, self.settings.feature_dim),
                      parameter.new_zeros(batch, self.settings.hidden_dim))

    def encode_geometry(self, grid, dimensions):
        return self.geometry_encoder(torch.cat([self.image_encoder(grid), dimensions], dim=-1))

    def predict(self, context, geometry, memory):
        # No GRU history inside phi: the realizability premise uses a static feature map.
        phi = self.features(context, geometry)
        residual = self.base_residual(context, geometry)
        residual = residual + torch.einsum("bnr,br->bn", memory.weights, phi)
        return residual, phi

    def base_residual(self, context, geometry):
        if not self.settings.static_residual_head:
            return context.new_zeros((*context.shape[:-1], 12))
        value = self.static_residual_network(torch.cat([context, geometry], dim=-1))
        return self.settings.static_residual_limit * torch.tanh(value)

    def features(self, context, geometry):
        """Supports BxTxD inputs so shared features can be evaluated in one GPU batch."""
        if not self.settings.adaptation:
            return context.new_zeros((*context.shape[:-1], self.settings.feature_dim))
        phi = self.feature_network(torch.cat([context, geometry], dim=-1))
        phi = phi / math.sqrt(self.settings.feature_dim)  # ||phi|| <= 1
        return phi

    def feedback(self, memory, phi, error, target, action_mps):
        """Called only AFTER comparing the saved prediction with the true next state."""
        c = self.settings
        if not c.adaptation:
            eta = error.new_zeros((len(error), 1))
            return memory, eta
        if c.gain == "learned":
            innovation = torch.cat([error, target, action_mps / .30], dim=-1)
            hidden = self.feedback_gru(innovation, memory.hidden)
            eta = c.eta_min + (c.eta_max - c.eta_min) * torch.sigmoid(self.gain_head(hidden))
        else:
            hidden = memory.hidden
            eta = error.new_full((len(error), 1), c.eta)
        weights = projected_update(memory.weights, phi, error, eta, c.epsilon, c.radius)
        return Memory(weights, hidden), eta

    @property
    def dissipation_coefficient(self):
        c = self.settings
        if not c.adaptation:
            return 0.
        lower, upper = (c.eta, c.eta) if c.gain == "constant" else (c.eta_min, c.eta_max)
        return lower * (2 - upper) / (c.epsilon + 1.0)
