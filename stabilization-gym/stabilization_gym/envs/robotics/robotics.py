"""Small physical robot tasks with differentiable metric-driven controllers.

These are transparent research environments, not adapters to a hardware robot.
Each controller is a candidate stabilizer: positive definiteness of ``M`` alone
does not prove stability of the nonlinear, saturated closed loop.  Physical
success criteria are independent of the learned metric.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch

from ..base import StateMetricEnv


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)


def _bound(action: torch.Tensor, low, high) -> torch.Tensor:
    lower = torch.as_tensor(low, dtype=action.dtype, device=action.device)
    upper = torch.as_tensor(high, dtype=action.dtype, device=action.device)
    return torch.maximum(lower, torch.minimum(upper, action))


class TwoLinkArmEnv(StateMetricEnv):
    """Planar two-link arm with coupled inertia, Coriolis and gravity.

    State is ``[q1, q2, dq1, dq2]``; joint angles are measured from the
    downward vertical.  Actions are joint torques in N m.  The virtual
    acceleration uses the gradient of the quadratic candidate,
    ``a = -kp (M e)[:2] - kd (M e)[2:]``; computed torque then cancels the
    known nominal mechanics.  ``M=I`` gives the ordinary computed-torque PD
    controller.  Arbitrary SPD cross terms and actuator saturation require
    closed-loop validation and are not asserted to preserve stability.
    """

    def __init__(self, *, dt=.02, max_steps=400, initial_scale=.2,
                 goal=None, success_tolerance=.05, render_mode=None):
        self.mass1 = self.mass2 = 1.0
        self.length1 = self.length2 = 1.0
        self.gravity = 9.81
        self.kp = 9.0
        self.kd = 6.0
        target = np.array([.3, -.45, 0., 0.]) if goal is None else np.asarray(goal)
        if target.shape != (4,) or not np.allclose(target[2:], 0.):
            raise ValueError("TwoLinkArmEnv requires a four-vector static joint target")
        super().__init__(state_dim=4, action_low=[-35., -35.],
                         action_high=[35., 35.], error_dim=4, dt=dt,
                         max_steps=max_steps, goal=target,
                         initial_scale=initial_scale,
                         success_tolerance=success_tolerance,
                         render_mode=render_mode)

    def tensor_error(self, state):
        target = torch.as_tensor(self.goal, dtype=state.dtype, device=state.device)
        return torch.cat((_wrap(state[..., :2] - target[:2]),
                          state[..., 2:] - target[2:]), dim=-1)

    def mechanics(self, state):
        """Return inertia H(q), Coriolis vector C(q,dq), and gravity G(q)."""
        q1, q2, v1, v2 = state.unbind(-1)
        m1, m2 = self.mass1, self.mass2
        l1, l2 = self.length1, self.length2
        c1, c2 = l1 / 2., l2 / 2.
        i1, i2 = m1 * l1**2 / 12., m2 * l2**2 / 12.
        h11 = i1 + i2 + m1*c1*c1 + m2*(l1*l1 + c2*c2 + 2*l1*c2*torch.cos(q2))
        h12 = i2 + m2*(c2*c2 + l1*c2*torch.cos(q2))
        h22 = torch.ones_like(q2) * (i2 + m2*c2*c2)
        inertia = torch.stack((torch.stack((h11, h12), -1),
                               torch.stack((h12, h22), -1)), -2)
        h = -m2*l1*c2*torch.sin(q2)
        coriolis = torch.stack((h*(2*v1*v2 + v2*v2), -h*v1*v1), -1)
        gravity = torch.stack((self.gravity*((m1*c1 + m2*l1)*torch.sin(q1)
                                            + m2*c2*torch.sin(q1 + q2)),
                               self.gravity*m2*c2*torch.sin(q1 + q2)), -1)
        return inertia, coriolis, gravity

    def torch_step(self, state, action):
        torque = _bound(action, self.action_space.low, self.action_space.high)
        inertia, coriolis, gravity = self.mechanics(state)
        acceleration = torch.linalg.solve(inertia, (torque-coriolis-gravity).unsqueeze(-1)).squeeze(-1)
        velocity = state[..., 2:] + self.dt*acceleration
        position = _wrap(state[..., :2] + self.dt*velocity)
        return torch.cat((position, velocity), -1)

    def calculate_action_from_M(self, M, e, *, context: Mapping,
                                controller_state=None, certificate=None):
        state = context["state"]
        weighted_error = _mv(M, e)
        acceleration = -self.kp*weighted_error[..., :2] - self.kd*weighted_error[..., 2:]
        inertia, coriolis, gravity = self.mechanics(state)
        torque = _mv(inertia, acceleration) + coriolis + gravity
        return _bound(torque, self.action_space.low, self.action_space.high)

    def is_success(self, state):
        error = self.tensor_error(torch.as_tensor(state, dtype=torch.float64)).detach().cpu().numpy()
        return bool(np.linalg.norm(error[:2]) < self.success_tolerance
                    and np.linalg.norm(error[2:]) < 2*self.success_tolerance)

    def is_failure(self, state):
        return bool(not np.isfinite(state).all() or np.max(np.abs(state[2:])) > 40.)


class UnicycleTrackingEnv(StateMetricEnv):
    """Nonholonomic unicycle tracking a moving circular reference.

    State is ``[x, y, heading, reference_phase]``.  The reference has strictly
    positive speed; this is trajectory tracking, not a claim of smooth global
    stabilization at an arbitrary stationary pose.  Actions are ``[v, omega]``.
    The reference phase belongs to state, so ``torch_step`` has no hidden clock.
    At identity M the controller is the usual body-frame feedforward/feedback
    law; M shapes its translational and heading errors.
    """

    def __init__(self, *, dt=.05, max_steps=400, initial_scale=.15,
                 radius=1., reference_speed=.4, success_tolerance=.06,
                 render_mode=None):
        if radius <= 0 or reference_speed <= 0:
            raise ValueError("radius and reference_speed must be strictly positive")
        self.radius = float(radius)
        self.reference_speed = float(reference_speed)
        self.reference_omega = self.reference_speed/self.radius
        super().__init__(state_dim=4, action_low=[-1., -2.5],
                         action_high=[1., 2.5], error_dim=3, dt=dt,
                         max_steps=max_steps, goal=np.zeros(4),
                         initial_scale=initial_scale,
                         success_tolerance=success_tolerance,
                         render_mode=render_mode)

    def reference(self, phase):
        return torch.stack((self.radius*torch.sin(phase),
                            self.radius*(1-torch.cos(phase)), phase), -1)

    def tensor_error(self, state):
        reference = self.reference(state[..., 3])
        dx = state[..., 0] - reference[..., 0]
        dy = state[..., 1] - reference[..., 1]
        heading = state[..., 2]
        return torch.stack((torch.cos(heading)*dx + torch.sin(heading)*dy,
                            -torch.sin(heading)*dx + torch.cos(heading)*dy,
                            _wrap(heading-reference[..., 2])), -1)

    def torch_step(self, state, action):
        action = _bound(action, self.action_space.low, self.action_space.high)
        speed, omega = action.unbind(-1)
        middle_heading = state[..., 2] + .5*self.dt*omega
        return torch.stack((state[..., 0] + self.dt*speed*torch.cos(middle_heading),
                            state[..., 1] + self.dt*speed*torch.sin(middle_heading),
                            _wrap(state[..., 2] + self.dt*omega),
                            _wrap(state[..., 3] + self.dt*self.reference_omega)), -1)

    def calculate_action_from_M(self, M, e, *, context: Mapping,
                                controller_state=None, certificate=None):
        sine_error = torch.cat((e[..., :2], torch.sin(e[..., 2:3])), -1)
        weighted_error = _mv(M, sine_error)
        speed = self.reference_speed*torch.cos(e[..., 2]) - 1.2*weighted_error[..., 0]
        omega = self.reference_omega - self.reference_speed*(3.*weighted_error[..., 1]
                                                             + 3.*weighted_error[..., 2])
        return _bound(torch.stack((speed, omega), -1),
                      self.action_space.low, self.action_space.high)

    def is_success(self, state):
        error = self.tensor_error(torch.as_tensor(state, dtype=torch.float64)).detach().cpu().numpy()
        return bool(np.linalg.norm(error[:2]) < self.success_tolerance
                    and abs(error[2]) < self.success_tolerance)

    def is_failure(self, state):
        return bool(not np.isfinite(state).all() or np.linalg.norm(state[:2]) > 20.)


class MultirotorTrackingEnv(StateMetricEnv):
    """Planar quadrotor hover or circular-trajectory tracking.

    This deliberately models a planar vehicle, not a full six-DOF multirotor.
    State is ``[x, z, vx, vz, pitch, pitch_rate, reference_phase]`` and action
    is ``[total_thrust, pitch_torque]``.  Horizontal acceleration depends on
    thrust and pitch.  A metric-shaped outer position loop supplies the desired
    force; the inner attitude loop realizes that force.  M also shapes attitude
    residuals.  The moving reference phase is explicit even in hover mode.
    """

    def __init__(self, *, dt=.02, max_steps=500, initial_scale=.08,
                 trajectory="hover", radius=.3, reference_omega=.35,
                 goal=None, success_tolerance=.06, render_mode=None):
        if trajectory not in ("hover", "circle"):
            raise ValueError("trajectory must be 'hover' or 'circle'")
        if radius <= 0 or reference_omega <= 0:
            raise ValueError("radius and reference_omega must be positive")
        self.trajectory = trajectory
        self.radius = float(radius)
        self.reference_omega = float(reference_omega)
        self.mass = 1.0
        self.inertia = .025
        self.gravity = 9.81
        target = np.array([0., 1., 0., 0., 0., 0., 0.]) if goal is None else np.asarray(goal)
        if target.shape != (7,) or not np.allclose(target[2:6], 0.):
            raise ValueError("goal must be [x, z, 0, 0, 0, 0, phase]")
        super().__init__(state_dim=7, action_low=[0., -.6],
                         action_high=[2.5*self.mass*self.gravity, .6],
                         error_dim=6, dt=dt, max_steps=max_steps, goal=target,
                         initial_scale=initial_scale,
                         success_tolerance=success_tolerance,
                         render_mode=render_mode)

    def reference(self, phase):
        zero = torch.zeros_like(phase)
        origin = torch.as_tensor(self.goal, dtype=phase.dtype, device=phase.device)
        if self.trajectory == "hover":
            state = torch.stack((zero+origin[0], zero+origin[1], zero, zero, zero, zero), -1)
            return state, torch.stack((zero, zero), -1)
        r, w = self.radius, self.reference_omega
        ax, az = -r*w*w*torch.sin(phase), r*w*w*torch.cos(phase)
        fx, fz = -ax, self.gravity+az
        pitch = torch.atan2(fx, fz)
        fx_dot, fz_dot = r*w**3*torch.cos(phase), -r*w**3*torch.sin(phase)
        pitch_rate = (fz*fx_dot-fx*fz_dot)/(fz*fz+fx*fx)
        state = torch.stack((origin[0]+r*torch.sin(phase),
                             origin[1]+r*(1-torch.cos(phase)),
                             r*w*torch.cos(phase), r*w*torch.sin(phase),
                             pitch, pitch_rate), -1)
        return state, torch.stack((ax, az), -1)

    def tensor_error(self, state):
        reference, _ = self.reference(state[..., 6])
        difference = state[..., :6]-reference
        return torch.cat((difference[..., :4], _wrap(difference[..., 4:5]),
                          difference[..., 5:6]), -1)

    def torch_step(self, state, action):
        action = _bound(action, self.action_space.low, self.action_space.high)
        thrust, torque = action.unbind(-1)
        pitch = state[..., 4]
        vx = state[..., 2] - self.dt*thrust*torch.sin(pitch)/self.mass
        vz = state[..., 3] + self.dt*(thrust*torch.cos(pitch)/self.mass-self.gravity)
        omega = state[..., 5]+self.dt*torque/self.inertia
        phase_speed = self.reference_omega if self.trajectory == "circle" else 0.
        return torch.stack((state[..., 0]+self.dt*vx, state[..., 1]+self.dt*vz,
                            vx, vz, _wrap(pitch+self.dt*omega), omega,
                            _wrap(state[..., 6]+self.dt*phase_speed)), -1)

    def calculate_action_from_M(self, M, e, *, context: Mapping,
                                controller_state=None, certificate=None):
        state = context["state"]
        reference, feedforward = self.reference(state[..., 6])
        weighted_error = _mv(M, e)
        acceleration = feedforward-2.5*weighted_error[..., :2]-3.*weighted_error[..., 2:4]
        fx, fz = acceleration[..., 0], acceleration[..., 1]+self.gravity
        desired_pitch = torch.atan2(-fx, fz)
        pitch_error = _wrap(state[..., 4]-desired_pitch)
        thrust = self.mass*(-fx*torch.sin(state[..., 4])+fz*torch.cos(state[..., 4]))
        torque = self.inertia*(-25.*pitch_error-8.*(state[..., 5]-reference[..., 5])
                              -2.*(weighted_error[..., 4]-e[..., 4])
                              -(weighted_error[..., 5]-e[..., 5]))
        return _bound(torch.stack((thrust, torque), -1),
                      self.action_space.low, self.action_space.high)

    def is_success(self, state):
        error = self.tensor_error(torch.as_tensor(state, dtype=torch.float64)).detach().cpu().numpy()
        tolerance = self.success_tolerance
        return bool(np.linalg.norm(error[:2]) < tolerance
                    and np.linalg.norm(error[2:4]) < 2*tolerance
                    and abs(error[4]) < tolerance
                    and abs(error[5]) < 2*tolerance)

    def is_failure(self, state):
        return bool(not np.isfinite(state).all() or state[1] < 0.
                    or np.linalg.norm(state[:2]) > 20.
                    or abs(state[4]) > math.pi/2.)


__all__ = ["TwoLinkArmEnv", "UnicycleTrackingEnv", "MultirotorTrackingEnv"]
