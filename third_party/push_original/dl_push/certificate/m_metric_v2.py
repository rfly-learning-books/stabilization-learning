"""Phase-isolated SPD metric (v2) for the instantaneous CLF study.

Fixes the P0-1 confounding of ``m_metric.py``: the metric state input is the
*phase-relevant* error ``P_sigma D`` rather than the full error.  With
``input_mode='state_shape'`` the certificate blocks become

  * ``M_J = M(D)``
  * ``M_A = M([0, 0, e_theta])``
  * ``M_P = M([e_x, e_y, 0])``

so ``V_A`` cannot fall because the object translated, and ``V_P`` cannot fall
because the object rotated.  Two frozen baselines share the same interface:

  * ``episode_shape``: the state branch is removed; the metric depends only on
    the measured mode set and the fixed geometry summary (frozen per episode);
  * ``c_only``: additionally removes the geometry summary and the non-response
    mode-descriptor channels, leaving a strict C-only set function.

The parameterization is recorded exactly as implemented: five unconstrained
values are mapped to a symmetric matrix with zero trace, scaled elementwise by
``tanh``, exponentiated, and trace-normalized.  The elementwise ``tanh`` does
not preserve the zero-trace property of the generator; the fixed trace and the
eigenvalue floor are nevertheless guaranteed by the affine map
``Q = floor * I + trace * (1 - floor) * normalized``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

INPUT_MODES = ("c_only", "episode_shape", "state_shape")


def _trace_zero_matrix(parameters: torch.Tensor) -> torch.Tensor:
    """Map five parameters to a symmetric zero-trace 3 by 3 generator."""

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
    """Return a smooth fixed-scale SPD matrix from five unconstrained values."""

    if not 0.0 < float(eigenvalue_floor) < 1.0:
        raise ValueError("eigenvalue_floor must be in (0,1)")
    if float(trace) <= 0.0 or float(logit_scale) <= 0.0:
        raise ValueError("trace and logit_scale must be positive")
    generator = float(logit_scale) * torch.tanh(_trace_zero_matrix(parameters))
    exponential = torch.matrix_exp(generator)
    trace_value = torch.diagonal(exponential, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    normalized = exponential / trace_value.unsqueeze(-1)
    eye = torch.eye(3, dtype=parameters.dtype, device=parameters.device)
    return (
        float(eigenvalue_floor) * eye
        + float(trace) * (1.0 - float(eigenvalue_floor)) * normalized
    )


@dataclass(frozen=True)
class MetricConfigV2:
    """Architecture and certificate-scale settings for the v2 metric."""

    input_mode: str = "state_shape"
    mode_feature_dim: int = 9
    shape_feature_dim: int = 6
    hidden: int = 96
    eigenvalue_floor: float = 0.05
    trace: float = 3.0
    logit_scale: float = 0.8

    def __post_init__(self):
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}")
        if min(self.mode_feature_dim, self.shape_feature_dim, self.hidden) < 1:
            raise ValueError("metric dimensions must be positive")

    @property
    def uses_state(self) -> bool:
        return self.input_mode == "state_shape"

    @property
    def uses_shape(self) -> bool:
        return self.input_mode != "c_only"

    @property
    def mode_input_dim(self) -> int:
        # The measured response is always the first three channels.
        return 3 if self.input_mode == "c_only" else self.mode_feature_dim


class PhaseMetricV2(nn.Module):
    """Permutation-invariant metric with phase-isolated state conditioning."""

    def __init__(self, config: MetricConfigV2 | None = None):
        super().__init__()
        self.config = config or MetricConfigV2()
        h = int(self.config.hidden)
        self.mode_net = nn.Sequential(
            nn.Linear(self.config.mode_input_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.shape_net = (
            nn.Sequential(
                nn.Linear(self.config.shape_feature_dim, h),
                nn.SiLU(),
                nn.Linear(h, h),
                nn.SiLU(),
            )
            if self.config.uses_shape
            else None
        )
        self.state_net = (
            nn.Sequential(
                nn.Linear(5, h),
                nn.SiLU(),
                nn.Linear(h, h),
                nn.SiLU(),
            )
            if self.config.uses_state
            else None
        )
        head_width = 2 * h + 1
        head_width += h if self.config.uses_shape else 0
        head_width += h if self.config.uses_state else 0
        self.head = nn.Sequential(
            nn.Linear(head_width, h),
            nn.SiLU(),
            nn.Linear(h, 5),
        )
        # Isotropic initialization without any Q=I feasibility preflight.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        state_error: torch.Tensor,
        mode_features: torch.Tensor,
        valid: torch.Tensor,
        shape_features: torch.Tensor,
    ) -> torch.Tensor:
        if state_error.ndim != 2 or state_error.shape[-1] != 3:
            raise ValueError("state_error must have shape (B,3)")
        if mode_features.ndim != 3 or mode_features.shape[-1] != self.config.mode_feature_dim:
            raise ValueError("mode_features has an invalid shape")
        if valid.shape != mode_features.shape[:2]:
            raise ValueError("valid must have shape (B,N)")
        if shape_features.ndim != 2 or shape_features.shape[-1] != self.config.shape_feature_dim:
            raise ValueError("shape_features has an invalid shape")
        if shape_features.shape[0] != state_error.shape[0] or mode_features.shape[0] != state_error.shape[0]:
            raise ValueError("all metric inputs must share the batch dimension")

        mask = valid.to(dtype=mode_features.dtype).unsqueeze(-1)
        features = mode_features[..., : self.config.mode_input_dim]
        embedded = self.mode_net(features)
        masked = embedded.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        maximum = masked.amax(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (embedded * mask).sum(dim=1) / count
        summary = [mean, maximum, count / mode_features.shape[1]]
        if self.state_net is not None:
            state_input = torch.cat(
                [
                    state_error,
                    torch.sin(state_error[:, 2:3]),
                    torch.cos(state_error[:, 2:3]),
                ],
                dim=-1,
            )
            summary.append(self.state_net(state_input))
        if self.shape_net is not None:
            summary.append(self.shape_net(shape_features))
        parameters = self.head(torch.cat(summary, dim=-1))
        return spd_from_parameters(
            parameters,
            eigenvalue_floor=self.config.eigenvalue_floor,
            trace=self.config.trace,
            logit_scale=self.config.logit_scale,
        )


__all__ = [
    "INPUT_MODES",
    "MetricConfigV2",
    "PhaseMetricV2",
    "spd_from_parameters",
]
