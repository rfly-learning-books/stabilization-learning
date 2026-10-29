"""Public prediction/measurement models; no trajectory truth or learning imports."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


def vector(value, size):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"Expected a finite vector of length {size}")
    return value.copy()


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@dataclass(frozen=True)
class TaskSpec:
    name: str
    state_dim: int
    error_names: tuple
    error_units: tuple
    error_scale: tuple
    measurement_dim: int
    measurement_scale: tuple
    input_dim: int
    dt: float
    horizon: float
    hold_seconds: float
    group_names: tuple
    group_tolerances: tuple
    sensor_std: tuple
    description: str

    @property
    def error_dim(self):
        return len(self.error_names)


class ObservationModel:
    """Pure, known model. Error/correction coordinates have declared physical units."""
    spec: TaskSpec

    def retract(self, estimate, correction):
        return vector(estimate, self.spec.state_dim) + vector(correction, self.spec.error_dim)

    def error(self, truth, estimate):
        return vector(truth, self.spec.state_dim) - vector(estimate, self.spec.state_dim)

    def groups(self, error):
        return np.abs(error)

    def measurement_jacobian(self, estimate):
        raise NotImplementedError

    def transition_jacobian(self, estimate, control, dt=None):
        """Derivative in the same local tangent coordinates as error/retract."""
        dt = self.spec.dt if dt is None else dt
        base = self.predict(estimate, control, dt)
        result = np.empty((self.spec.error_dim, self.spec.error_dim))
        for j in range(self.spec.error_dim):
            delta = np.zeros(self.spec.error_dim); delta[j] = 1e-6
            plus = self.predict(self.retract(estimate, delta), control, dt)
            minus = self.predict(self.retract(estimate, -delta), control, dt)
            result[:, j] = (self.error(plus, base) - self.error(minus, base)) / 2e-6
        return result

    def initial_estimate(self):
        return np.zeros(self.spec.state_dim)

    def visible(self, state):
        return np.ones(self.spec.measurement_dim, dtype=bool)

    def known_input(self, time, phase):
        return np.zeros(self.spec.input_dim)

    def truth_step(self, state, control, time, scenario):
        return self.predict(state, control)

    def domain_ok(self, state):
        return bool(np.isfinite(state).all())


class MotorModel(ObservationModel):
    spec = TaskSpec("motor_speed", 2, ("current", "speed"), ("A", "rad/s"),
        (1., 1.), 1, (1.,), 1, .01, 2., .4, ("current_A", "speed_rad_s"),
        (.03, .03), (.02,), "Book ch2 motor: speed-only measurement, exact ZOH")
    A = np.array([[-100., -10.], [10., -10.]])
    B = np.array([[100.], [0.]])
    C = np.array([[0., 1.]])

    def __init__(self):
        self.Ad, self.Bd = self._zoh(self.A, self.spec.dt)
        mismatch = self.A.copy(); mismatch[0, 0] *= 1.1
        self.mismatch_Ad, self.mismatch_Bd = self._zoh(mismatch, self.spec.dt)

    def _zoh(self, a, dt):
        block = np.zeros((3, 3)); block[:2, :2] = a; block[:2, 2:] = self.B
        result = expm(dt * block)
        return result[:2, :2], result[:2, 2:]

    def predict(self, state, control, dt=None):
        state, control = vector(state, 2), vector(control, 1)
        a, b = (self.Ad, self.Bd) if dt is None or dt == self.spec.dt else self._zoh(self.A, dt)
        return a @ state + b @ control

    def truth_step(self, state, control, time, scenario):
        if scenario == "mismatch":
            return self.mismatch_Ad @ state + self.mismatch_Bd @ control
        return self.predict(state, control)

    def measure(self, state):
        return self.C @ vector(state, 2)

    def measurement_jacobian(self, estimate):
        return self.C.copy()

    def flow_jacobian(self, estimate, control):
        return self.A.copy()

    def transition_jacobian(self, estimate, control, dt=None):
        return self.Ad.copy() if dt is None else self._zoh(self.A, dt)[0]

    def initial_truth(self, rng):
        return rng.uniform(-.6, .6, 2)

    def known_input(self, time, phase):
        return np.array([1. + .5 * np.sin(4 * time + phase)])

    def domain_ok(self, state):
        return bool(np.isfinite(state).all() and np.max(np.abs(state)) <= 6.)


class CartPoleModel(ObservationModel):
    spec = TaskSpec("cartpole_output", 4,
        ("position", "velocity", "angle", "angular_velocity"),
        ("m", "m/s", "rad", "rad/s"), (1., 1., 1., 1.),
        2, (1., 1.), 1, .01, 1.2, .2,
        ("position_m", "velocity_m_s", "angle_rad", "angular_velocity_rad_s"),
        (.015, .05, .015, .08), (.002, .002),
        "Book ch2 outputs x/theta; existing cart-pole mechanics, short open-loop observation")
    C = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.]])

    @staticmethod
    def flow(state, control, masspole=.1):
        _, velocity, theta, omega = state
        sine, cosine = np.sin(theta), np.cos(theta)
        total, length = 1. + masspole, .5
        temp = (control[0] + masspole * length * omega**2 * sine) / total
        angular_acc = (9.81 * sine - cosine * temp) / (length * (4/3 - masspole * cosine**2 / total))
        acceleration = temp - masspole * length * angular_acc * cosine / total
        return np.array([velocity, acceleration, omega, angular_acc])

    def _integrate(self, state, control, dt, masspole=.1):
        flow = lambda x: self.flow(x, control, masspole)
        k1 = flow(state); k2 = flow(state + dt*k1/2)
        k3 = flow(state + dt*k2/2); k4 = flow(state + dt*k3)
        return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    def predict(self, state, control, dt=None):
        return self._integrate(vector(state, 4), vector(control, 1), self.spec.dt if dt is None else dt)

    def truth_step(self, state, control, time, scenario):
        return self._integrate(state, control, self.spec.dt, .13 if scenario == "mismatch" else .1)

    def error(self, truth, estimate):
        result = super().error(truth, estimate)
        result[2] = np.arctan2(np.sin(result[2]), np.cos(result[2]))
        return result

    def measure(self, state):
        return self.C @ vector(state, 4)

    def measurement_jacobian(self, estimate):
        return self.C.copy()

    def flow_jacobian(self, estimate, control):
        result = np.empty((4, 4))
        for j in range(4):
            delta = np.zeros(4); delta[j] = 1e-6
            result[:, j] = (self.flow(estimate+delta, control) - self.flow(estimate-delta, control))/2e-6
        return result

    def initial_truth(self, rng):
        return rng.uniform(-1., 1., 4) * [.04, .02, .003, .004]

    def known_input(self, time, phase):
        return np.array([.02 * np.sin(3*time + phase)])

    def domain_ok(self, state):
        return bool(np.isfinite(state).all() and abs(state[0]) <= 2.4 and abs(state[2]) <= np.pi/4
                    and np.max(np.abs(state[[1, 3]])) <= 20.)


class PoseModel(ObservationModel):
    """13 stored coordinates [p, quaternion_xyzw, v, body_omega]; 12-D tangent."""
    spec = TaskSpec("cooperative_pose", 13,
        tuple([f"position_{a}" for a in "xyz"] + [f"rotation_{a}" for a in "xyz"]
              + [f"velocity_{a}" for a in "xyz"] + [f"angular_velocity_{a}" for a in "xyz"]),
        ("m",)*3 + ("rad",)*3 + ("m/s",)*3 + ("rad/s",)*3, (1.,)*12,
        16, (100.,)*16, 0, .05, 4., .5,
        ("position_norm_m", "rotation_geodesic_rad", "velocity_norm_m_s", "angular_velocity_norm_rad_s"),
        (.05, np.deg2rad(2.), .08, .08), (1.,)*16,
        "Book ch15 calibrated known-geometry target; synthetic keypoints, not an RGB detector")

    def __init__(self, degenerate=False):
        self.points = np.array([[x, y, z] for x in (-.25, .25)
                               for y in (-.18, .18) for z in (-.12, .12)])
        if degenerate:
            self.points = np.column_stack([np.linspace(-.25, .25, 8), np.zeros((8, 2))])
        self.K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])

    def initial_estimate(self):
        return np.array([0., 0., 3.5, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    def initial_truth(self, rng):
        state = self.initial_estimate()
        state[:3] += rng.uniform(-1., 1., 3) * [.2, .15, .25]
        state[3:7] = Rotation.from_rotvec(rng.uniform(-.15, .15, 3)).as_quat()
        state[7:10] = rng.uniform(-.06, .06, 3)
        state[10:13] = rng.uniform(-.08, .08, 3)
        return state

    def predict(self, state, control, dt=None):
        state = vector(state, 13); vector(control, 0)
        dt = self.spec.dt if dt is None else dt
        state[:3] += dt * state[7:10]
        state[3:7] = (Rotation.from_quat(state[3:7]) * Rotation.from_rotvec(dt*state[10:13])).as_quat()
        return state

    def truth_step(self, state, control, time, scenario):
        result = self.predict(state, control)
        if scenario == "mismatch":
            acceleration = np.array([.03*np.sin(time), .02*np.cos(time), .01])
            angular_acc = np.array([.015, .02*np.sin(time), -.01])
            result[:3] += .5*self.spec.dt**2 * acceleration
            result[7:10] += self.spec.dt*acceleration
            result[3:7] = (Rotation.from_quat(result[3:7]) * Rotation.from_rotvec(.5*self.spec.dt**2*angular_acc)).as_quat()
            result[10:13] += self.spec.dt*angular_acc
        return result

    def retract(self, estimate, correction):
        result, delta = vector(estimate, 13), vector(correction, 12)
        result[:3] += delta[:3]
        result[3:7] = (Rotation.from_quat(result[3:7]) * Rotation.from_rotvec(delta[3:6])).as_quat()
        result[7:10] += delta[6:9]; result[10:13] += delta[9:12]
        return result

    def error(self, truth, estimate):
        truth, estimate = vector(truth, 13), vector(estimate, 13)
        rotation = (Rotation.from_quat(estimate[3:7]).inv()*Rotation.from_quat(truth[3:7])).as_rotvec()
        return np.r_[truth[:3]-estimate[:3], rotation, truth[7:10]-estimate[7:10], truth[10:13]-estimate[10:13]]

    def groups(self, error):
        return np.linalg.norm(np.asarray(error).reshape(4, 3), axis=1)

    def camera_points(self, state):
        state = vector(state, 13)
        return state[:3] + Rotation.from_quat(state[3:7]).apply(self.points)

    def measure(self, state):
        points = self.camera_points(state)
        # Invalid depths are masked, never turned into usable measurements.
        depth = np.maximum(points[:, 2], .01)
        return (points[:, :2]/depth[:, None]*[self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]).ravel()

    def visible(self, state):
        points = self.camera_points(state); pixels = self.measure(state).reshape(-1, 2)
        mask = (points[:, 2] >= .2) & (pixels[:, 0] >= 0) & (pixels[:, 0] < 640) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 480)
        return np.repeat(mask, 2)

    def measurement_jacobian(self, estimate):
        rotation = Rotation.from_quat(estimate[3:7]).as_matrix()
        rows = []
        for point, body_point in zip(self.camera_points(estimate), self.points):
            x, y, z = point
            if z <= .01:
                raise ValueError("Prediction is outside positive-depth camera domain")
            projection = np.array([[600/z, 0., -600*x/z**2], [0., 600/z, -600*y/z**2]])
            rows.append(np.column_stack([projection, -projection@rotation@skew(body_point), np.zeros((2, 6))]))
        return np.vstack(rows)

    def flow_jacobian(self, estimate, control):
        a = np.zeros((12, 12)); a[:3, 6:9] = np.eye(3)
        a[3:6, 3:6] = -skew(estimate[10:13]); a[3:6, 9:12] = np.eye(3)
        return a

    def domain_ok(self, state):
        return bool(np.isfinite(state).all() and abs(np.linalg.norm(state[3:7])-1.) < 1e-6
                    and np.min(self.camera_points(state)[:, 2]) >= .2
                    and np.linalg.norm(state[:3]) <= 20. and np.max(np.abs(state[7:])) <= 10.)


MODEL_TYPES = {"motor_speed": MotorModel, "cartpole_output": CartPoleModel, "cooperative_pose": PoseModel}
