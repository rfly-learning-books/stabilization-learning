"""Pure numerical contracts for the sampling-first CLF controller.

The functions in this module are intentionally simulator-independent.  They
implement the mathematical appendix of ``pusht_mjx_requirements`` and are
used by both the mode-library audit and the real physics runner.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np


STAGES = ("JOINT", "ATTITUDE", "POSITION")
Stage = Literal["JOINT", "ATTITUDE", "POSITION"]


def _finite_array(value, shape=None, name="array"):
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def rot2(angle: float) -> np.ndarray:
    """Return the world/body planar rotation matrix for ``angle`` radians."""

    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    c, s = np.cos(float(angle)), np.sin(float(angle))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def wrap_angle(angle):
    """Wrap an angle to ``[-pi, pi]`` without changing its array shape."""

    value = np.asarray(angle, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("angle must be finite")
    return np.arctan2(np.sin(value), np.cos(value))


def shape_error(object_q, target_q, feature_length, orientation_period=2 * np.pi):
    """Return target-frame normalized error ``D = [e_px, e_py, e_theta]``.

    ``object_q`` and ``target_q`` are ``[x, y, yaw]``.  The orientation period
    is fixed for the episode and represents a declared geometric symmetry;
    it is never inferred from the current trajectory.
    """

    q = _finite_array(object_q, (3,), "object_q")
    target = _finite_array(target_q, (3,), "target_q")
    if not np.isfinite(feature_length) or feature_length <= 0:
        raise ValueError("feature_length must be positive")
    if not np.isfinite(orientation_period) or orientation_period <= 0:
        raise ValueError("orientation_period must be positive")
    order = 2 * np.pi / float(orientation_period)
    position = rot2(-target[2]) @ (q[:2] - target[:2]) / float(feature_length)
    angle = wrap_angle(order * (q[2] - target[2])) / order
    return np.array([position[0], position[1], angle], dtype=np.float64)


def response_in_error_frame(D, c, feature_length):
    """Return ``g_i(D)`` for ``dot(D) = g_i(D) u``.

    ``c`` is a raw three-component body-frame response.  Its third component
    is angular response per unit scalar input; it is not rotated or divided by
    the feature length.
    """

    error = _finite_array(D, (3,), "D")
    response = _finite_array(c, (3,), "c")
    if not np.isfinite(feature_length) or feature_length <= 0:
        raise ValueError("feature_length must be positive")
    return np.r_[rot2(error[2]) @ response[:2] / float(feature_length), response[2]]


def _symmetric_parameter_matrix(parameters):
    theta = _finite_array(parameters, (5,), "parameters")
    a, b, c, d, e = theta
    return np.array([[a, b, c], [b, d, e], [c, e, -a - d]], dtype=np.float64)


def q_from_parameters(parameters, eigenvalue_floor=0.05, trace=3.0):
    """Map five unconstrained parameters to a fixed-scale SPD matrix.

    The trace and eigenvalue floor are part of the certificate contract.  The
    implementation uses a symmetric eigendecomposition rather than a generic
    matrix exponential, which is stable and keeps the result symmetric to
    floating-point precision.
    """

    if not np.isfinite(eigenvalue_floor) or not 0 < eigenvalue_floor < 1:
        raise ValueError("eigenvalue_floor must be in (0,1)")
    if not np.isfinite(trace) or trace <= 0:
        raise ValueError("trace must be positive")
    values, vectors = np.linalg.eigh(_symmetric_parameter_matrix(parameters))
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    normalized = (vectors * exp_values) @ vectors.T
    normalized /= float(np.trace(normalized))
    Q = float(eigenvalue_floor) * np.eye(3)
    Q += float(trace) * (1.0 - float(eigenvalue_floor)) * normalized
    Q = 0.5 * (Q + Q.T)
    return Q


def validate_q(Q, eigenvalue_floor=0.0, trace=None, tolerance=1e-9):
    """Validate symmetry, positive definiteness, and optional scale."""

    matrix = _finite_array(Q, (3, 3), "Q")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0):
        raise ValueError("Q must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] < float(eigenvalue_floor) - tolerance:
        raise ValueError("Q violates the eigenvalue floor")
    if trace is not None and not np.isclose(np.trace(matrix), trace, atol=tolerance, rtol=0):
        raise ValueError("Q violates the fixed trace")
    return matrix


def stage_projection(stage: Stage) -> np.ndarray:
    """Return the fixed J/A/P diagonal projection mask."""

    if stage not in STAGES:
        raise ValueError(f"unknown CLF stage {stage!r}")
    return {
        "JOINT": np.ones(3, dtype=np.float64),
        "ATTITUDE": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "POSITION": np.array([1.0, 1.0, 0.0], dtype=np.float64),
    }[stage]


def certificate(D, Q, stage: Stage, response=None, feature_length=1.0):
    """Return ``(V, gradient, beta)`` for one stage and optional mode.

    ``beta`` is ``gradient.T @ g_i(D)``.  It is ``None`` when no response is
    supplied, which is useful for diagnostics of the certificate itself.
    """

    error = _finite_array(D, (3,), "D")
    matrix = validate_q(Q)
    mask = stage_projection(stage)
    P = mask[:, None] * matrix * mask[None, :]
    value = float(error @ P @ error)
    gradient = 2.0 * P @ error
    beta = None
    if response is not None:
        beta = float(gradient @ response_in_error_frame(error, response, feature_length))
    return value, gradient, beta


def _sinc0(value):
    value = float(value)
    if abs(value) < 1e-6:
        value2 = value * value
        return 1.0 - value2 / 6.0 + value2 * value2 / 120.0
    return float(np.sin(value) / value)


def _cosc0(value):
    value = float(value)
    if abs(value) < 1e-4:
        value2 = value * value
        return value / 2.0 - value * value2 / 24.0 + value * value2 * value2 / 720.0
    return float((1.0 - np.cos(value)) / value)


def exact_se2_step(q, c, u, duration):
    """Exactly integrate a constant body twist over one control period.

    The angle is deliberately left unwrapped.  Angle-chart management belongs
    to the task/supervisor layer, not to the integrator.
    """

    state = _finite_array(q, (3,), "q")
    response = _finite_array(c, (3,), "c")
    if not np.isfinite(u) or u < 0 or not np.isfinite(duration) or duration < 0:
        raise ValueError("u and duration must be finite and nonnegative")
    z = float(response[2] * float(u) * float(duration))
    sinc = _sinc0(z)
    cosc = _cosc0(z)
    B = np.array([[sinc, -cosc], [cosc, sinc]], dtype=np.float64)
    displacement = rot2(state[2]) @ B @ response[:2] * float(u) * float(duration)
    return state + np.array([displacement[0], displacement[1], z], dtype=np.float64)


def minimum_linear_input(A, b, lower, upper, tolerance=1e-12):
    """Solve ``min u`` subject to scalar linear inequalities and bounds.

    Infeasible intervals return ``None``.  In particular, the function never
    clips a lower bound to the actuator limit and then labels the result as
    feasible.
    """

    lhs = np.asarray(A, dtype=np.float64).reshape(-1)
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    if lhs.shape != rhs.shape or not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("A and b must be equal-sized finite arrays")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or upper < lower:
        raise ValueError("invalid nonnegative input interval")
    lo, hi = float(lower), float(upper)
    for coefficient, bound in zip(lhs, rhs):
        if abs(coefficient) <= tolerance:
            if bound < -tolerance:
                return None
        elif coefficient > 0:
            hi = min(hi, bound / coefficient)
        else:
            lo = max(lo, bound / coefficient)
    if lo > hi + tolerance or lo > upper + tolerance:
        return None
    if np.any(lhs * lo > rhs + 10.0 * tolerance):
        return None
    return float(max(lo, lower))


def matrix_hash(Q) -> str:
    """Canonical hash for the fixed certificate matrix."""

    matrix = validate_q(Q)
    payload = json.dumps(matrix.tolist(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

