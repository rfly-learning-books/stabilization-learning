"""Adapt the existing one-sided disk GMM experiment to model-based contracts.

The source implementation is a zero-input-preserving mixture of local gains,
not a general joint-distribution GMR implementation.
"""
from __future__ import annotations

import math
import numpy as np

from .algo import fit as fit_disk_route


def _column(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty (samples, 1) array")
    return result[:, 0]


class DiskGMMTransitionModel:
    def __init__(self, model):
        if model.route != "GMM":
            raise ValueError("The disk adapter only accepts the existing GMM route")
        self.model = model

    def predict_next(self, state, action):
        x, a = _column(state, "state"), _column(action, "action")
        if len(x) != len(a):
            raise ValueError("state and action sample counts differ")
        return (x + self.model.increment(x, a))[:, None]

    def memberships(self, state, action):
        x, a = _column(state, "state"), _column(action, "action")
        if len(x) != len(a):
            raise ValueError("state and action sample counts differ")
        return self.model.memberships(x, a)


class DiskGMMFitter:
    def __init__(self, config, *, components=4):
        if type(components) is not int or components <= 0:
            raise ValueError("components must be a positive integer")
        self.config, self.components = config, components

    def fit(self, batch, *, seed):
        if not math.isclose(batch.dt_seconds, self.config.dt, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("batch sampling period differs from the disk model")
        x = _column(batch.state, "state")
        a = _column(batch.applied_action, "applied_action")
        xn = _column(batch.next_state, "next_state")
        if not (len(x) == len(a) == len(xn)):
            raise ValueError("training sample counts differ")
        model, diagnostics = fit_disk_route(
            "GMM", {"observed_x": x, "action": a, "observed_next_x": xn},
            self.config, seed=seed, gmm_components=self.components,
        )
        return DiskGMMTransitionModel(model), diagnostics
