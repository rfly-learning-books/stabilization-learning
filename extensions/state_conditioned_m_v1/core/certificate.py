"""Quadratic certificate and same-version one-step residuals."""
from __future__ import annotations
import torch


def quadratic(error: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...i,...ij,...j->...", error, M, error)


def same_metric_residual(error_before: torch.Tensor,
                         error_after: torch.Tensor,
                         M_used_for_gain: torch.Tensor,
                         *,
                         decay: float,
                         dt: float) -> torch.Tensor:
    """Residual using one M version for both endpoints.

    The gain at step k uses M_used_for_gain.  For a one-step certificate we
    evaluate both endpoints with that same metric; changing M at the next
    measurement update requires a separate dwell/version condition.
    """
    if decay < 0 or dt <= 0:
        raise ValueError("Require decay >= 0 and dt > 0")
    vb = quadratic(error_before, M_used_for_gain)
    va = quadratic(error_after, M_used_for_gain)
    return torch.relu(va - vb + decay * dt * error_before.square().sum(-1))
