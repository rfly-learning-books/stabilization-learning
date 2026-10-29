"""Candidate e.T M(xhat) e certificates, bound to a single parameter version."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.linalg import solve_continuous_are


class StateMetric:
    """Bounded analytic probe; no fitted parameters or training claims.

    Future learned providers can implement the same matrix(estimate)/version
    contract. Context deliberately contains the estimate only, never true error.
    """
    def __init__(self, base, amplitude=0.):
        base = np.array(base, dtype=np.float64, copy=True)
        if base.ndim != 2 or base.shape[0] != base.shape[1] or not np.isfinite(base).all():
            raise ValueError("Finite square metric required")
        if not np.allclose(base, base.T, atol=1e-12, rtol=1e-12) or np.linalg.eigvalsh(base).min() <= 0:
            raise ValueError("Metric must be symmetric positive definite")
        if not np.isfinite(amplitude) or not 0 <= amplitude <= .4:
            raise ValueError("Probe amplitude must be in [0,.4]")
        self.base = base; self.base.flags.writeable = False
        self.amplitude = float(amplitude)
        eig = np.linalg.eigvalsh(base)
        self.bounds = ((1-amplitude)*float(eig.min()), (1+amplitude)*float(eig.max()))

    @property
    def version(self):
        return hashlib.sha256(self.base.tobytes()+repr(self.amplitude).encode()+b"estimate-tanh-probe-v1").hexdigest()

    def matrix(self, estimate):
        estimate = np.asarray(estimate, dtype=np.float64)
        if estimate.ndim != 1 or not np.isfinite(estimate).all():
            raise ValueError("Finite deployment estimate required")
        weights = np.linspace(.7, 1.3, len(estimate))
        factor = 1 + self.amplitude*np.tanh(weights @ estimate)
        return factor*self.base


def reference_metric(model, amplitude=0.):
    """Continuous dual CARE at the declared nominal prior, not online SDRE.

    A degenerate camera case uses the SAME nominal design as other pose cases;
    its actual Jacobian still uses the degenerate geometry. No retuning on tests.
    """
    from stabilization_gym.observation.models import MODEL_TYPES
    design_model = MODEL_TYPES[model.spec.name]()
    spec = design_model.spec; estimate = design_model.initial_estimate()
    scale = np.asarray(spec.error_scale)
    a = design_model.flow_jacobian(estimate, np.zeros(spec.input_dim))*scale[None, :]/scale[:, None]
    h = design_model.measurement_jacobian(estimate)*scale[None, :]
    r = np.diag(np.asarray(spec.measurement_scale)**2)
    covariance = solve_continuous_are(a.T, h.T, np.eye(spec.error_dim), r)
    matrix = np.linalg.solve(covariance, np.eye(spec.error_dim))
    return StateMetric((matrix+matrix.T)/2, amplitude)


def checked_matrix(metric, estimate, expected_version=None):
    version = metric.version
    if expected_version is not None and version != expected_version:
        raise ValueError("Certificate parameter version changed")
    matrix = np.asarray(metric.matrix(estimate), dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10)):
        raise ValueError("Invalid symmetric certificate matrix")
    eigenvalues = np.linalg.eigvalsh(matrix)
    lower, upper = metric.bounds
    if (eigenvalues.min() <= 0 or lower <= 0 or eigenvalues.min() < lower-1e-8*lower-1e-12
            or eigenvalues.max() > upper+1e-8*max(1., upper)):
        raise ValueError("Certificate spectral bounds violated")
    if metric.version != version:
        raise ValueError("Certificate changed during query")
    return matrix


def value(model, truth, estimate, metric, expected_version=None):
    error = model.error(truth, estimate)/np.asarray(model.spec.error_scale)
    matrix = checked_matrix(metric, estimate, expected_version)
    if matrix.shape != (len(error), len(error)):
        raise ValueError("Certificate coordinates do not match task")
    return float(error@matrix@error)


def difference(model, truth, estimate, following_truth, following_estimate, metric):
    version = metric.version
    before = value(model, truth, estimate, metric, version)
    after = value(model, following_truth, following_estimate, metric, version)
    return {"before": before, "after": after, "delta": after-before, "version": version}


def local_decay_matrix(model, estimate, control, metric, epsilon=1e-5):
    """Sampled continuous local diagnostic INCLUDING D_f M; not a domain proof.

    Pose uses right-tangent coordinates. This local diagnostic does not certify
    the split prediction/correction implementation or visibility switching.
    """
    version = metric.version
    scale = np.asarray(model.spec.error_scale)
    a = model.flow_jacobian(estimate, control)*scale[None, :]/scale[:, None]
    h = model.measurement_jacobian(estimate)*scale[None, :]
    w = 1/np.asarray(model.spec.measurement_scale)**2
    matrix = checked_matrix(metric, estimate, version)
    plus = checked_matrix(metric, model.predict(estimate, control, epsilon), version)
    minus = checked_matrix(metric, model.predict(estimate, control, -epsilon), version)
    derivative = (plus-minus)/(2*epsilon)
    result = derivative+a.T@matrix+matrix@a-2*h.T@(w[:, None]*h)
    return {"matrix": (result+result.T)/2, "metric_flow_derivative": derivative,
            "max_eigenvalue": float(np.linalg.eigvalsh((result+result.T)/2).max()), "version": version}
