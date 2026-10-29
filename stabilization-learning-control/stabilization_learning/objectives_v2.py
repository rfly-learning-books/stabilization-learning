"""Versioned empirical descent loss; the safety floor never changes the target."""
from __future__ import annotations
import math
import torch

OBJECTIVE_VERSION = "true_norm_descent_denominator_floor_v2"


def descent_terms(delta_v, norm2, *, dt, decay=.05, denominator_floor=1e-6):
    if not all(math.isfinite(x) for x in (dt, decay, denominator_floor)):
        raise ValueError("Finite descent configuration required")
    if dt <= 0 or decay < 0 or denominator_floor <= 0:
        raise ValueError("Require dt>0, decay>=0 and denominator_floor>0")
    if delta_v.shape != norm2.shape or not delta_v.is_floating_point() or not norm2.is_floating_point():
        raise ValueError("Floating delta_V and norm2 must have matching shapes")
    if not torch.isfinite(delta_v).all() or not torch.isfinite(norm2).all() or (norm2 < 0).any():
        raise ValueError("Finite delta_V and nonnegative physical squared error required")
    residual = delta_v + decay * dt * norm2
    denominator = norm2.clamp_min(denominator_floor)
    return dict(residual=residual, normalized_residual=residual / denominator,
                penalty=residual.clamp_min(0) / denominator)
