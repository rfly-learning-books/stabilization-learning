"""NumPy diagnostics for discrete certificate changes, not a stability proof.

These helpers deliberately operate on NumPy arrays. They are not a
differentiable training path for GPU/tensor models.
"""
from __future__ import annotations

import numpy as np


def _states(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty (samples, dimensions) array")
    return array


def _values(certificate, state):
    value = np.asarray(certificate.value(state.copy()), dtype=np.float64)
    if value.shape != (state.shape[0],) or not np.isfinite(value).all():
        raise ValueError("certificate must return one finite value per state")
    return value


def observed_discrete_change(certificate, state, next_state):
    """Compute V(next_state)-V(state) for actual saved successors."""
    current = _states(state, "state")
    successor = _states(next_state, "next_state")
    if current.shape != successor.shape:
        raise ValueError("state and next_state must have the same shape")
    return _values(certificate, successor) - _values(certificate, current)


def predicted_discrete_change(model, certificate, state, action):
    """Compute V(model.predict_next(state, action))-V(state)."""
    current = _states(state, "state")
    actions = _states(action, "action")
    if len(current) != len(actions):
        raise ValueError("state and action must have the same sample count")
    predicted = _states(model.predict_next(current.copy(), actions.copy()), "predicted successor")
    if predicted.shape != current.shape:
        raise ValueError("predicted successor must match state shape")
    return _values(certificate, predicted) - _values(certificate, current)
