"""Gymnasium physical-action boundary and differentiable controller facade.

The second step result is positive integrated cost (SLC convention). Conversion
to a maximized reward belongs to an algorithm wrapper, never to the physics.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from numbers import Integral

from ..common.types import MetricSpec


class StateMetricEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, state_dim, action_low, action_high, error_dim, dt,
                 max_steps, goal=None, initial_scale=.1, success_tolerance=.05,
                 render_mode=None, metric_floor=.05):
        if not np.isfinite(dt) or dt <= 0 or not isinstance(max_steps, Integral) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("Require positive finite dt and a positive integer max_steps")
        self.state_dim, self.error_dim = int(state_dim), int(error_dim)
        self.dt, self.max_steps = float(dt), int(max_steps)
        self.initial_scale, self.success_tolerance = initial_scale, success_tolerance
        self.goal = np.zeros(state_dim) if goal is None else np.asarray(goal, dtype=np.float64)
        if self.goal.shape != (state_dim,) or not np.isfinite(self.goal).all():
            raise ValueError("goal must be a finite full-state vector")
        self.state = self.goal.copy()
        self.action_space = gym.spaces.Box(np.asarray(action_low, dtype=np.float32),
                                           np.asarray(action_high, dtype=np.float32))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float64)
        self.metric_spec = MetricSpec(error_dim, metric_floor, float(error_dim))
        self.metric_context_dim = state_dim
        self.controller_capabilities = {
            "action_gradient": True, "model_gradient": True,
            "episode_fixed_metric": True, "state_dependent_certificate": False,
            "certificate_guarantee": "candidate_only", "cost_convention": "positive_integral",
        }
        self.render_mode = render_mode
        self._elapsed_steps, self._hold_steps = 0, 0
        self._done = True

    def _reset_state(self, options):
        if "state" in options:
            return np.asarray(options["state"], dtype=np.float64).copy()
        return self.goal + self.np_random.uniform(-1., 1., self.state_dim) * self.initial_scale

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options
        self.state = np.asarray(self._reset_state(options), dtype=np.float64)
        if self.state.shape != (self.state_dim,) or not np.isfinite(self.state).all():
            raise ValueError("reset produced an invalid state")
        self._elapsed_steps, self._hold_steps, self._done = 0, 0, False
        return self.state.copy(), self._info()

    def tensor_error(self, state):
        return state - torch.as_tensor(self.goal, dtype=state.dtype, device=state.device)

    def build_error(self, observation, goal=None, representation=None, history=None):
        state = observation if torch.is_tensor(observation) else torch.as_tensor(observation, dtype=torch.float64)
        target = torch.as_tensor(self.goal if goal is None else goal, dtype=state.dtype, device=state.device)
        if representation is not None:
            # Generic state controllers need a representation preserving their
            # physical coordinates. Image controllers override this contract.
            if not getattr(representation, "preserves_physical_coordinates", False):
                raise ValueError("State controller coordinates are physical; use IdentityRepresentation")
        if goal is not None and not torch.equal(target, torch.as_tensor(self.goal).to(target)):
            raise ValueError("Set the environment reference before constructing its error")
        return self.tensor_error(state), {"state": state, "goal": target}

    def metric_context(self, observation=None, goal=None):
        return torch.as_tensor(self.goal if goal is None else goal, dtype=torch.float32)

    def physical_cost(self, state):
        e = self.tensor_error(torch.as_tensor(state, dtype=torch.float64))
        return float(e.square().sum().detach().cpu())

    def is_success(self, state):
        return self.physical_cost(state) <= self.success_tolerance ** 2

    def is_failure(self, state):
        return not np.isfinite(state).all() or np.linalg.norm(state) > 100

    def _info(self):
        error = self.physical_cost(self.state)
        return {"physical_error": error, "distance": float(np.sqrt(max(error, 0.))),
                "success": self._hold_steps >= 5, "in_goal": bool(self.is_success(self.state)),
                "failure": bool(self.is_failure(self.state)), "steps": self._elapsed_steps,
                "cost_convention": "positive_integral"}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode")
        u = np.asarray(action, dtype=np.float64)
        if u.shape != self.action_space.shape or not np.isfinite(u).all():
            raise ValueError("Action must be a finite physical-action vector")
        u = np.clip(u, self.action_space.low, self.action_space.high)
        cost_before = self.physical_cost(self.state)
        with torch.no_grad():
            self.state = self.torch_step(torch.as_tensor(self.state), torch.as_tensor(u)).cpu().numpy()
        self._elapsed_steps += 1
        self._hold_steps = self._hold_steps + 1 if self.is_success(self.state) else 0
        info = self._info()
        cost = self.dt * cost_before
        terminated = info["success"] or info["failure"]
        truncated = self._elapsed_steps >= self.max_steps and not terminated
        self._done = terminated or truncated
        info.update(cost=cost, applied_action=u.copy())
        return self.state.copy(), cost, terminated, truncated, info

    def torch_step(self, state, action):
        raise NotImplementedError

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        raise NotImplementedError

    def render(self):
        # A compact state/error plot, not a physical simulator animation.
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(5, 3), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.bar(np.arange(self.state_dim), self.state - self.goal)
        ax.set(title=type(self).__name__ + " state error", xlabel="state coordinate", ylabel="error")
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()
