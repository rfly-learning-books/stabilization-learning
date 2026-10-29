"""Small continuous-control tasks with differentiable metric-driven controllers.

Each controller combines a nominal local LQR with a correction derived from
``grad_e V = 2 M e``. Identity M recovers the nominal controller exactly; an
arbitrary positive-definite M is *not* by itself a stability guarantee. The
nonlinear pendulum and cart-pole tasks intentionally target local stabilization,
not swing-up. State, force and velocity units are SI (angles are radians).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from scipy.linalg import solve_continuous_are
from torch import Tensor

from ..base import StateMetricEnv


def _lqr(A: np.ndarray, B: np.ndarray, q: Sequence[float], r: float) -> Tensor:
    Q = np.diag(np.asarray(q, dtype=np.float64))
    R = np.eye(B.shape[1], dtype=np.float64) * r
    P = solve_continuous_are(A, B, Q, R)
    return torch.from_numpy(np.linalg.solve(R, B.T @ P))


def _rk4(state: Tensor, action: Tensor, dt: float, flow: Any) -> Tensor:
    """One fixed-step RK4 update; supports arbitrary leading batch dimensions."""
    k1 = flow(state, action)
    k2 = flow(state + 0.5 * dt * k1, action)
    k3 = flow(state + 0.5 * dt * k2, action)
    k4 = flow(state + dt * k3, action)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _bounded(action: Tensor, low: Tensor, high: Tensor) -> Tensor:
    return torch.minimum(torch.maximum(action, low.to(action)), high.to(action))


def _metric_feedback(M: Tensor, e: Tensor, K: Tensor, B: Tensor, gain: float) -> Tensor:
    # The nominal term supplies restoring forces even when B^T e only contains
    # velocity errors. The correction equals -gain/2 B^T grad(V_M - V_I).
    weighted = torch.matmul(M, e.unsqueeze(-1)).squeeze(-1) - e
    return -(e @ K.to(e).T) - gain * (weighted @ B.to(e))


class CoupledLinearEnv(StateMetricEnv):
    """Two coupled masses with spring and damping forces.

    State/error order is ``[q1, q2, dq1, dq2]``; action is two applied forces.
    The fixed goal has zero velocity. The model and its controller support
    batched PyTorch evaluation independently of the Gymnasium simulator state.
    """

    def __init__(
        self,
        *,
        dt: float = 0.04,
        max_steps: int = 300,
        initial_scale: float = 0.7,
        goal: Sequence[float] | None = None,
        metric_gain: float = 0.6,
        render_mode: str | None = None,
    ) -> None:
        target = np.zeros(4) if goal is None else np.asarray(goal, dtype=np.float64)
        if target.shape != (4,) or not np.allclose(target[2:], 0.0):
            raise ValueError("CoupledLinearEnv requires a four-state goal with zero velocity")
        self._stiffness = torch.tensor([[2.0, -0.6], [-0.6, 1.5]], dtype=torch.float64)
        self._damping = torch.tensor([[0.25, 0.03], [0.03, 0.3]], dtype=torch.float64)
        A = np.block([[np.zeros((2, 2)), np.eye(2)],
                      [-self._stiffness.numpy(), -self._damping.numpy()]])
        B = np.vstack([np.zeros((2, 2)), np.eye(2)])
        self._A, self._B = torch.from_numpy(A), torch.from_numpy(B)
        self._K = _lqr(A, B, [4.0, 4.0, 1.0, 1.0], 0.5)
        self.metric_gain = float(metric_gain)
        self._low, self._high = torch.full((2,), -8.0), torch.full((2,), 8.0)
        super().__init__(state_dim=4, action_low=[-8.0, -8.0], action_high=[8.0, 8.0],
                         error_dim=4, dt=dt, max_steps=max_steps, goal=target,
                         initial_scale=initial_scale, success_tolerance=0.05,
                         render_mode=render_mode)

    def _flow(self, state: Tensor, action: Tensor) -> Tensor:
        return state @ self._A.to(state).T + action @ self._B.to(state).T

    def torch_step(self, state: Tensor, action: Tensor) -> Tensor:
        return _rk4(state, action, self.dt, self._flow)

    def calculate_action_from_M(self, M: Tensor, e: Tensor, *, context: Mapping[str, Tensor],
                                controller_state: Any = None, certificate: Any = None) -> Tensor:
        goal = context["goal"].to(e)
        feedforward = goal[..., :2] @ self._stiffness.to(e).T
        action = _metric_feedback(M, e, self._K, self._B, self.metric_gain) + feedforward
        return _bounded(action, self._low, self._high)


class PointMassEnv(StateMetricEnv):
    """Planar double integrator with acceleration control.

    State/error order is ``[x, y, vx, vy]``. ``set_reference`` can be called before
    each control step for trajectory tracking; its velocity and acceleration
    are reference feedforward data, not true-state observations. Passing a
    tensor ``reference_acceleration`` in controller context supports a fully
    differentiable external trajectory generator.
    """

    def __init__(
        self,
        *,
        dt: float = 0.04,
        max_steps: int = 300,
        initial_scale: float = 0.7,
        goal: Sequence[float] | None = None,
        metric_gain: float = 0.6,
        render_mode: str | None = None,
    ) -> None:
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [np.zeros((2, 4))]])
        B = np.vstack([np.zeros((2, 2)), np.eye(2)])
        self._B = torch.from_numpy(B)
        self._K = _lqr(A, B, [4.0, 4.0, 1.0, 1.0], 0.5)
        self.metric_gain = float(metric_gain)
        self._reference_acceleration = torch.zeros(2, dtype=torch.float64)
        self._low, self._high = torch.full((2,), -6.0), torch.full((2,), 6.0)
        super().__init__(state_dim=4, action_low=[-6.0, -6.0], action_high=[6.0, 6.0],
                         error_dim=4, dt=dt, max_steps=max_steps, goal=goal,
                         initial_scale=initial_scale, success_tolerance=0.05,
                         render_mode=render_mode)

    def set_reference(self, position: Sequence[float], velocity: Sequence[float] = (0.0, 0.0),
                      acceleration: Sequence[float] = (0.0, 0.0)) -> None:
        p, v, a = (np.asarray(x, dtype=np.float64) for x in (position, velocity, acceleration))
        if any(x.shape != (2,) or not np.isfinite(x).all() for x in (p, v, a)):
            raise ValueError("Reference position, velocity and acceleration must be finite 2-vectors")
        self.goal = np.concatenate((p, v))
        self._reference_acceleration = torch.from_numpy(a.copy())

    def torch_step(self, state: Tensor, action: Tensor) -> Tensor:
        # Exact constant-acceleration discretization, preserving all gradients.
        p = state[..., :2] + self.dt * state[..., 2:] + 0.5 * self.dt**2 * action
        v = state[..., 2:] + self.dt * action
        return torch.cat((p, v), dim=-1)

    def calculate_action_from_M(self, M: Tensor, e: Tensor, *, context: Mapping[str, Tensor],
                                controller_state: Any = None, certificate: Any = None) -> Tensor:
        feedforward = context.get("reference_acceleration", self._reference_acceleration).to(e)
        action = _metric_feedback(M, e, self._K, self._B, self.metric_gain) + feedforward
        return _bounded(action, self._low, self._high)


class PendulumEnv(StateMetricEnv):
    """Torque-controlled upright pendulum; local stabilization, not swing-up."""

    def __init__(self, *, dt: float = 0.02, max_steps: int = 400,
                 initial_scale: float = 0.15, metric_gain: float = 0.5,
                 render_mode: str | None = None) -> None:
        self.gravity, self.length, self.mass, self.damping = 9.81, 1.0, 1.0, 0.15
        A = np.array([[0.0, 1.0], [self.gravity / self.length, -self.damping]])
        B = np.array([[0.0], [1.0 / (self.mass * self.length**2)]])
        self._B = torch.from_numpy(B)
        self._K = _lqr(A, B, [12.0, 1.0], 0.5)
        self.metric_gain = float(metric_gain)
        self._low, self._high = torch.tensor([-8.0]), torch.tensor([8.0])
        super().__init__(state_dim=2, action_low=[-8.0], action_high=[8.0], error_dim=2,
                         dt=dt, max_steps=max_steps, goal=np.zeros(2),
                         initial_scale=initial_scale, success_tolerance=0.04,
                         render_mode=render_mode)

    def tensor_error(self, state: Tensor) -> Tensor:
        return torch.stack((torch.atan2(torch.sin(state[..., 0]), torch.cos(state[..., 0])),
                            state[..., 1]), dim=-1)

    def _flow(self, state: Tensor, action: Tensor) -> Tensor:
        acceleration = (self.gravity / self.length * torch.sin(state[..., 0])
                        - self.damping * state[..., 1]
                        + action[..., 0] / (self.mass * self.length**2))
        return torch.stack((state[..., 1], acceleration), dim=-1)

    def torch_step(self, state: Tensor, action: Tensor) -> Tensor:
        return _rk4(state, action, self.dt, self._flow)

    def calculate_action_from_M(self, M: Tensor, e: Tensor, *, context: Mapping[str, Tensor],
                                controller_state: Any = None, certificate: Any = None) -> Tensor:
        action = _metric_feedback(M, e, self._K, self._B, self.metric_gain)
        return _bounded(action, self._low, self._high)

    def is_failure(self, state: np.ndarray) -> bool:
        if super().is_failure(state):
            return True
        angle = np.arctan2(np.sin(state[0]), np.cos(state[0]))
        return bool(abs(angle) > np.pi / 2.0 or abs(state[1]) > 12.0)


class CartPoleEnv(StateMetricEnv):
    """Continuous-force cart-pole near the upright configuration.

    State/error order is ``[x, dx, theta, dtheta]``. The controller is designed
    for the upright region; falling beyond 45 degrees ends a failed episode.
    """

    def __init__(self, *, dt: float = 0.02, max_steps: int = 500,
                 initial_scale: float = 0.08, metric_gain: float = 0.2,
                 render_mode: str | None = None) -> None:
        self.gravity, self.masscart, self.masspole, self.length = 9.81, 1.0, 0.1, 0.5
        total = self.masscart + self.masspole
        den = self.length * (4.0 / 3.0 - self.masspole / total)
        A = np.zeros((4, 4))
        A[0, 1], A[2, 3] = 1.0, 1.0
        A[1, 2] = -self.masspole * self.length * self.gravity / (total * den)
        A[3, 2] = self.gravity / den
        B = np.array([[0.0], [1.0 / total + self.masspole * self.length / (total**2 * den)],
                      [0.0], [-1.0 / (total * den)]])
        self._B = torch.from_numpy(B)
        self._K = _lqr(A, B, [2.0, 0.5, 16.0, 1.0], 0.1)
        self.metric_gain = float(metric_gain)
        self._low, self._high = torch.tensor([-10.0]), torch.tensor([10.0])
        super().__init__(state_dim=4, action_low=[-10.0], action_high=[10.0], error_dim=4,
                         dt=dt, max_steps=max_steps, goal=np.zeros(4),
                         initial_scale=initial_scale, success_tolerance=0.04,
                         render_mode=render_mode)

    def tensor_error(self, state: Tensor) -> Tensor:
        angle = torch.atan2(torch.sin(state[..., 2]), torch.cos(state[..., 2]))
        return torch.stack((state[..., 0], state[..., 1], angle, state[..., 3]), dim=-1)

    def _flow(self, state: Tensor, action: Tensor) -> Tensor:
        theta, omega = state[..., 2], state[..., 3]
        sine, cosine = torch.sin(theta), torch.cos(theta)
        total = self.masscart + self.masspole
        moment = self.masspole * self.length
        temp = (action[..., 0] + moment * omega.square() * sine) / total
        theta_acc = (self.gravity * sine - cosine * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cosine.square() / total))
        x_acc = temp - moment * theta_acc * cosine / total
        return torch.stack((state[..., 1], x_acc, omega, theta_acc), dim=-1)

    def torch_step(self, state: Tensor, action: Tensor) -> Tensor:
        return _rk4(state, action, self.dt, self._flow)

    def calculate_action_from_M(self, M: Tensor, e: Tensor, *, context: Mapping[str, Tensor],
                                controller_state: Any = None, certificate: Any = None) -> Tensor:
        action = _metric_feedback(M, e, self._K, self._B, self.metric_gain)
        return _bounded(action, self._low, self._high)

    def is_failure(self, state: np.ndarray) -> bool:
        if super().is_failure(state):
            return True
        angle = np.arctan2(np.sin(state[2]), np.cos(state[2]))
        return bool(abs(state[0]) > 2.4 or abs(angle) > np.pi / 4.0)
