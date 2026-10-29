"""One physical state layout and a single residual-normalization boundary."""
from __future__ import annotations

import numpy as np

DT = 0.05
ACTION_SCALE = 0.30
STATE_NAMES = ("x", "y", "z", "yaw", "vx", "vy", "vz", "omega",
               "pusher_x", "pusher_y", "pusher_vx", "pusher_vy")
RESIDUAL_SCALE = np.array(
    [.01, .01, .005, .10, .10, .10, .05, 1., .01, .01, .10, .10],
    dtype=np.float32,
)
CONTEXT_DIM = 36


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class YawLift:
    """Continuous yaw along measured time; prior obs keeps its original sin/cos."""
    def __init__(self):
        self.previous = None

    def __call__(self, state):
        state = np.asarray(state, dtype=np.float32).copy()
        if self.previous is not None:
            state[3] = self.previous + wrap(float(state[3]) - self.previous)
        self.previous = float(state[3])
        return state


def residual_target(real_next, nominal_next):
    # Both states must already be in the same lifted angle chart.
    return (np.asarray(real_next) - np.asarray(nominal_next)) / RESIDUAL_SCALE


def corrected_state(nominal, normalized_residual):
    return np.asarray(nominal) + np.asarray(normalized_residual) * RESIDUAL_SCALE


def state12_to_obs17(
    states, goal, orientation_period,
    body_center_xy=(0., 0.), model_origin_xy=(0., 0.),
):
    """Rebuild the frozen priors' 17-D observation from physical state.

    This is used only for model-space planning.  The real control loop keeps
    using the observation returned by PhysX.
    """
    x = np.asarray(states, dtype=np.float32)
    if x.shape[-1] != 12:
        raise ValueError("Expected physical state with final dimension 12")
    goal = np.asarray(goal, dtype=np.float32)
    if goal.shape != (3,) or not np.isfinite(orientation_period) or orientation_period <= 0:
        raise ValueError("Expected a finite goal and positive orientation period")
    order = 2 * np.pi / orientation_period
    yaw = x[..., 3]
    delta_yaw = wrap(order * (goal[2] - yaw)) / order
    center = np.asarray(body_center_xy, dtype=np.float32)
    origin = np.asarray(model_origin_xy, dtype=np.float32)
    if center.shape != (2,) or origin.shape != (2,):
        raise ValueError("Body center and model origin must be two-dimensional")
    offset = origin - center
    c, s = np.cos(yaw), np.sin(yaw)
    rotated_offset = np.stack(
        (c * offset[0] - s * offset[1],
         s * offset[0] + c * offset[1]),
        axis=-1,
    )
    omega = x[..., 7]
    origin_velocity_from_com = np.stack(
        (-omega * rotated_offset[..., 1], omega * rotated_offset[..., 0]),
        axis=-1,
    )
    center_velocity = x[..., 4:6] - origin_velocity_from_com
    relative = x[..., 8:10] - x[..., :2]
    pusher_local = np.stack(
        (c * relative[..., 0] + s * relative[..., 1],
         -s * relative[..., 0] + c * relative[..., 1]),
        axis=-1,
    )
    observation = np.concatenate(
        (
            x[..., 8:10],
            x[..., 10:12],
            x[..., :2],
            center_velocity,
            np.stack((np.sin(yaw), np.cos(yaw)), axis=-1),
            x[..., 7:8],
            goal[:2] - x[..., :2],
            np.stack((np.sin(delta_yaw), np.cos(delta_yaw)), axis=-1),
            pusher_local,
        ),
        axis=-1,
    )
    if observation.shape[-1] != 17 or not np.isfinite(observation).all():
        raise ValueError("Could not reconstruct a finite prior observation")
    return observation.astype(np.float32)


def model_context(state, actions, nominal, local_sdf):
    state = np.asarray(state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    nominal = np.asarray(nominal, dtype=np.float32)
    if state.shape != (12,) or nominal.shape != (len(actions), 12) or actions.shape[1:] != (2,):
        raise ValueError("Expected real state (12,), actions (N,2), nominal (N,12)")
    state_features = np.r_[state[:3] / [.25, .25, .04], np.sin(state[3]), np.cos(state[3]),
                           state[4:7] / .30, state[7] / 5.,
                           state[8:10] / .232, state[10:12] / .30]
    features = np.c_[np.tile(state_features, (len(actions), 1)),
                     actions / ACTION_SCALE, (nominal - state) / RESIDUAL_SCALE,
                     np.tile(np.asarray(local_sdf) / .10, (len(actions), 1))]
    if features.shape != (len(actions), CONTEXT_DIM) or not np.isfinite(features).all():
        raise ValueError("Invalid causal model features")
    return features.astype(np.float32)
