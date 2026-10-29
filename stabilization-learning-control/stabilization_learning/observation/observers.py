"""Reference observers receive the PUBLIC model and observation only."""
from __future__ import annotations
import numpy as np
from .certificate import checked_matrix, reference_metric


class PredictionObserver:
    name = "prediction_only"

    def __init__(self, model):
        self.model = model
        self.diagnostics = {}

    def correction(self, observation):
        return np.zeros(self.model.spec.error_dim)


class MetricObserver(PredictionObserver):
    """Split-step implementation of L=M^-1 H.T W (normalized tangent).

    At each measurement: delta = dt*S*M^-1*(H*S).T*W*r, then the
    environment predicts the next prior. No hidden gain, saturation, optimizer,
    or true-state input. Actual discrete certificate differences are evaluated.
    """
    def __init__(self, model, metric=None):
        super().__init__(model)
        self.metric = reference_metric(model) if metric is None else metric
        self.version = self.metric.version
        self.name = "variable_M_probe" if self.metric.amplitude else "fixed_M"

    def correction(self, observation):
        spec = self.model.spec; estimate = observation["estimate"]
        matrix = checked_matrix(self.metric, estimate, self.version)
        mask = observation["measurement_mask"].astype(bool)
        if not mask.any():
            self.diagnostics = {"visible_channels": 0, "metric_version": self.version}
            return np.zeros(spec.error_dim)
        h = self.model.measurement_jacobian(estimate)[mask] * np.asarray(spec.error_scale)[None, :]
        residual = observation["measurement"][mask]-self.model.measure(estimate)[mask]
        weights = observation["confidence"][mask]/np.asarray(spec.measurement_scale)[mask]**2
        gain = np.linalg.solve(matrix, h.T*weights)
        delta = float(observation["dt"][0])*np.asarray(spec.error_scale)*(gain@residual)
        self.diagnostics = {"visible_channels": int(mask.sum()), "metric_version": self.version,
                            "gain_norm": float(np.linalg.norm(gain)), "matrix": matrix.copy()}
        return delta


class ExtendedKalmanObserver(PredictionObserver):
    """Known-model KF/EKF baseline. Covariance is distinct from certificate M."""
    name = "kalman_ekf"

    def __init__(self, model, process_variance_per_second=.0025):
        super().__init__(model)
        self.covariance = np.eye(model.spec.error_dim)*.25
        self.process_variance = float(process_variance_per_second)

    def correction(self, observation):
        spec = self.model.spec; scale = np.asarray(spec.error_scale)
        estimate = observation["estimate"]; mask = observation["measurement_mask"].astype(bool)
        covariance = self.covariance.copy()
        if mask.any():
            h = self.model.measurement_jacobian(estimate)[mask]*scale[None, :]
            residual = observation["measurement"][mask]-self.model.measure(estimate)[mask]
            r = np.diag(np.maximum(np.asarray(spec.sensor_std)[mask]**2, 1e-8))
            innovation = h@covariance@h.T+r
            gain = np.linalg.solve(innovation, h@covariance).T
            delta = scale*(gain@residual)
            identity = np.eye(spec.error_dim)-gain@h
            posterior = identity@covariance@identity.T+gain@r@gain.T
            nis = float(residual@np.linalg.solve(innovation, residual))
        else:
            delta = np.zeros(spec.error_dim); posterior = covariance; nis = None
        corrected = self.model.retract(estimate, delta)
        f = self.model.transition_jacobian(corrected, observation["known_input"])*scale[None, :]/scale[:, None]
        # Transport attitude covariance into the corrected right-tangent frame.
        if spec.name == "cooperative_pose":
            from scipy.spatial.transform import Rotation
            # Exact reset Jacobian d Log(Exp(-d) Exp(d+epsilon))/d epsilon.
            d = delta[3:6]; eps = 1e-6; reset = np.eye(spec.error_dim)
            base = Rotation.from_rotvec(d).inv()
            for j in range(3):
                direction = np.zeros(3); direction[j] = eps
                reset[3:6, 3+j] = ((base*Rotation.from_rotvec(d+direction)).as_rotvec()
                                  -(base*Rotation.from_rotvec(d-direction)).as_rotvec())/(2*eps)
            posterior = reset@posterior@reset.T
        dt = float(observation["dt"][0])
        self.covariance = f@posterior@f.T+self.process_variance*dt*np.eye(spec.error_dim)
        self.covariance = (self.covariance+self.covariance.T)/2
        self.diagnostics = {"visible_channels": int(mask.sum()), "nis": nis,
                            "posterior_covariance_normalized": posterior.copy()}
        return delta


class GeometryFilterObserver(PredictionObserver):
    """Known-point reprojection least squares followed by a fixed alpha-beta filter."""
    name = "geometry_filter"

    def __init__(self, model):
        super().__init__(model)
        if model.spec.name != "cooperative_pose":
            raise ValueError("Geometric pose baseline only applies to the camera task")

    def correction(self, observation):
        from scipy.optimize import least_squares
        estimate = observation["estimate"]; mask = observation["measurement_mask"].astype(bool)
        h = self.model.measurement_jacobian(estimate)[mask, :6]
        if mask.sum() < 8 or np.linalg.matrix_rank(h) < 6:
            self.diagnostics = {"geometry_available": False, "rank": int(np.linalg.matrix_rank(h)) if h.size else 0}
            return np.zeros(12)
        def residual(delta):
            state = self.model.retract(estimate, np.r_[delta, np.zeros(6)])
            return (self.model.measure(state)[mask]-observation["measurement"][mask])
        fit = least_squares(residual, np.zeros(6), max_nfev=20,
            bounds=([-2., -2., -2., -.8, -.8, -.8], [2., 2., 2., .8, .8, .8]),
            ftol=1e-7, xtol=1e-7, gtol=1e-7)
        delta = np.zeros(12)
        if fit.success and np.isfinite(fit.x).all():
            delta[:6] = .45*fit.x
            delta[6:9] = .05/observation["dt"][0]*fit.x[:3]
            delta[9:12] = .05/observation["dt"][0]*fit.x[3:6]
        self.diagnostics = {"geometry_available": bool(fit.success), "reprojection_cost": float(fit.cost)}
        return delta


def make_observer(name, model):
    if name == "prediction_only":
        return PredictionObserver(model)
    if name == "kalman_ekf":
        return ExtendedKalmanObserver(model)
    if name == "geometry_filter":
        return GeometryFilterObserver(model)
    if name in {"fixed_M", "variable_M_probe"}:
        return MetricObserver(model, reference_metric(model, .15 if name == "variable_M_probe" else 0.))
    raise ValueError(f"Unknown observer: {name}")
