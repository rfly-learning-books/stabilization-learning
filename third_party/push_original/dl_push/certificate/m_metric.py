"""State--shape conditioned SPD metrics for the instantaneous CLF study.

The production contract in ``pusht_mjx_requirements`` freezes a C-only
certificate once per episode.  This module adds the teacher-inspired research
variant in which the metric also sees the current normalized pose error.  The
state dependence is deliberate and explicit: the derivative used by the
training objective is the full autograd derivative of
``D.T @ P @ M(D, shape) @ P @ D`` and therefore includes the ``M_dot`` term.

The geometry is represented without a shape name.  A padded set of measured
mode descriptors is processed with a masked permutation-invariant encoder;
the descriptor contains the calibrated response, contact point, normal, face
count, and validity bit.  A small fixed geometry summary is supplied as a
second shape input.  This is a research artifact, not a claim that the
state-conditioned metric already satisfies the frozen-Q theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _trace_zero_matrix(parameters: torch.Tensor) -> torch.Tensor:
    """Map five parameters to a symmetric trace-zero 3 by 3 matrix."""

    if parameters.shape[-1] != 5:
        raise ValueError("metric parameters must end in dimension 5")
    a, b, c, d, e = parameters.unbind(dim=-1)
    return torch.stack([a, b, c, b, d, e, c, e, -a - d], dim=-1).reshape(
        *parameters.shape[:-1], 3, 3
    )


def spd_from_parameters(
    parameters: torch.Tensor,
    eigenvalue_floor: float = 0.05,
    trace: float = 3.0,
    logit_scale: float = 0.8,
) -> torch.Tensor:
    """Return a smooth fixed-scale SPD matrix from five unconstrained values.

    ``torch.matrix_exp`` is used instead of an eigendecomposition so the
    training derivative is well behaved at the isotropic initialization.  The
    trace and floor are the same contract as the requirements' Q parameter.
    """

    if not 0.0 < float(eigenvalue_floor) < 1.0:
        raise ValueError("eigenvalue_floor must be in (0,1)")
    if float(trace) <= 0.0 or float(logit_scale) <= 0.0:
        raise ValueError("trace and logit_scale must be positive")
    # Bounding the generator keeps the overnight experiment numerically
    # stable while retaining a nontrivial family of ellipsoidal metrics.
    S = float(logit_scale) * torch.tanh(_trace_zero_matrix(parameters))
    exponential = torch.matrix_exp(S)
    normalized = exponential / torch.diagonal(
        exponential, dim1=-2, dim2=-1
    ).sum(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(3, dtype=parameters.dtype, device=parameters.device)
    return (
        float(eigenvalue_floor) * eye
        + float(trace) * (1.0 - float(eigenvalue_floor)) * normalized
    )


@dataclass(frozen=True)
class MetricConfig:
    """Architecture and certificate-scale settings."""

    mode_feature_dim: int = 9
    shape_feature_dim: int = 6
    hidden: int = 96
    eigenvalue_floor: float = 0.05
    trace: float = 3.0
    logit_scale: float = 0.8
    state_conditioned: bool = True

    def __post_init__(self):
        if min(self.mode_feature_dim, self.shape_feature_dim, self.hidden) < 1:
            raise ValueError("metric dimensions must be positive")


class StateShapeMetric(nn.Module):
    """Permutation-invariant metric conditioned on error and calibrated shape.

    Inputs:
      ``D``: ``(B,3)`` normalized position/yaw error;
      ``mode_features``: ``(B,N,9)`` padded measured mode descriptors;
      ``valid``: ``(B,N)`` mode mask;
      ``shape_features``: ``(B,6)`` scale-free polygon summary.

    The output is an SPD matrix ``M`` with fixed trace.  Setting
    ``state_conditioned=False`` gives the proof-safe episode-frozen shape
    metric while retaining the same interface.
    """

    def __init__(self, config: MetricConfig | None = None):
        super().__init__()
        self.config = config or MetricConfig()
        h = int(self.config.hidden)
        self.mode_net = nn.Sequential(
            nn.Linear(self.config.mode_feature_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.shape_net = nn.Sequential(
            nn.Linear(self.config.shape_feature_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.state_net = nn.Sequential(
            nn.Linear(5, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * h + 1, h),
            nn.SiLU(),
            nn.Linear(h, 5),
        )
        # Start from the isotropic metric without running a Q=I feasibility
        # preflight.  The first reported metric is already a learned output.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        D: torch.Tensor,
        mode_features: torch.Tensor,
        valid: torch.Tensor,
        shape_features: torch.Tensor,
    ) -> torch.Tensor:
        if D.ndim != 2 or D.shape[-1] != 3:
            raise ValueError("D must have shape (B,3)")
        if mode_features.ndim != 3 or mode_features.shape[-1] != self.config.mode_feature_dim:
            raise ValueError("mode_features has an invalid shape")
        if valid.shape != mode_features.shape[:2]:
            raise ValueError("valid must have shape (B,N)")
        if shape_features.ndim != 2 or shape_features.shape[-1] != self.config.shape_feature_dim:
            raise ValueError("shape_features has an invalid shape")
        if shape_features.shape[0] != D.shape[0] or mode_features.shape[0] != D.shape[0]:
            raise ValueError("all metric inputs must share the batch dimension")

        mask = valid.to(dtype=mode_features.dtype).unsqueeze(-1)
        embedded = self.mode_net(mode_features)
        masked = embedded.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        maximum = masked.amax(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (embedded * mask).sum(dim=1) / count
        set_summary = torch.cat([mean, maximum, count / mode_features.shape[1]], dim=-1)

        # Position and attitude are both explicit inputs.  Sine/cosine gives
        # the network a smooth local representation of the yaw coordinate.
        state_input = torch.cat(
            [D, torch.sin(D[:, 2:3]), torch.cos(D[:, 2:3])], dim=-1
        )
        if not self.config.state_conditioned:
            state_input = torch.zeros_like(state_input)
        state_embedding = self.state_net(state_input)
        shape_embedding = self.shape_net(shape_features)
        parameters = self.head(
            torch.cat([set_summary, state_embedding, shape_embedding], dim=-1)
        )
        return spd_from_parameters(
            parameters,
            eigenvalue_floor=self.config.eigenvalue_floor,
            trace=self.config.trace,
            logit_scale=self.config.logit_scale,
        )


__all__ = ["MetricConfig", "StateShapeMetric", "spd_from_parameters"]
