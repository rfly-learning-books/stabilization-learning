"""Unregistered full-state DC motor preparation, outside frozen packages.

The book continuous parameters are re-discretized by exact ZOH. Observation,
input limits and full-horizon success are a NEW nominal benchmark, not a
reproduction of the book's erroneous printed discrete matrices or speed-only
measurement. M defines the candidate V and the one-step control objective.
"""
from __future__ import annotations

from copy import deepcopy
import math

import gymnasium as gym
import numpy as np
from scipy.linalg import expm
import torch

from stabilization_gym.envs.base import StateMetricEnv


class DCMotorExactZOHEnv(StateMetricEnv):
    """Physical action is voltage; M is supplied through the control interface.

    calculate_action_from_M preserves autograd. A detached pending record is
    committed only by step to enforce episode-fixed executed M. Thus control
    queries do not advance physics but are explicitly NOT side-effect free.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}
    controller_form = "same_M_one_step_normalized_state_energy_physical_voltage_regularization"
    control_call_is_pure = False
    control_call_side_effects = ("audit_pending_metric_and_action",)

    def __init__(self, *, target_speed=None, render_mode=None):
        super().__init__(state_dim=2, action_low=[-5.], action_high=[5.], error_dim=2,
            dt=.01, max_steps=200, goal=[0., 0.], initial_scale=3.,
            success_tolerance=.02, render_mode=render_mode, metric_floor=.05)
        self._fixed_target = None if target_speed is None else self._validate_target(target_speed)
        self._target_speed = 0. if target_speed is None else self._fixed_target
        self.goal = np.full(2, self._target_speed, dtype=np.float64)
        self._goal_snapshot = self.goal.copy()
        self.metric_context_dim = 1
        self.current_scale_A = self.speed_scale_rad_s = 6.
        self.current_limit_A = self.speed_limit_rad_s = 6.
        self.current_tolerance_A = self.speed_tolerance_rad_s = .02
        self.required_hold_seconds = 1.
        self.horizon_seconds = 2.
        self.voltage_deviation_weight = .01 / 25.
        self.protocol_kind = "dc_motor_exact_zoh_full_state_preparation_v1"
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float64)
        self.A = np.array([[-100., -10.], [10., -10.]])
        self.Bu = np.array([[100.], [0.]])
        lifted = np.zeros((3, 3)); lifted[:2, :2], lifted[:2, 2:] = self.A, self.Bu
        transition = expm(self.dt * lifted)
        self.Ad, self.Bd = transition[:2, :2].copy(), transition[:2, 2:].copy()
        self.error_scale = np.array([6., 6.])
        self.Ae = self.Ad * self.error_scale[None, :] / self.error_scale[:, None]
        self.Be = self.Bd[:, 0] / self.error_scale
        self.controller_capabilities.update(full_horizon_evaluation=True,
            full_state_observation=True, state_coordinate_units=("A", "rad/s"),
            physical_step_autograd=False, differentiable_torch_model=True,
            certificate_guarantee="candidate_only")
        self._pending_control = self._episode_metric = None
        self._execution_mode = None
        self._in_goal_since = None
        self._first_hold_confirmation = None
        self._metric_steps = 0
        self._initial_state = self.state.copy()
        self._seed_record = None
        self._trace = []
        self._last_info = {}

    @staticmethod
    def _validate_target(value):
        if isinstance(value, (bool, np.bool_)) or np.asarray(value).shape != ():
            raise ValueError("Target speed must be a scalar in rad/s")
        value = float(value)
        if not math.isfinite(value) or not -2. <= value <= 2.:
            raise ValueError("Target speed must lie in [-2,2] rad/s")
        return value

    @property
    def target_speed(self):
        return self._target_speed

    @property
    def equilibrium_voltage(self):
        return 1.1 * self.target_speed

    def _assert_reference_fixed(self):
        if not np.array_equal(self.goal, self._goal_snapshot) or not np.all(self.goal == self.target_speed):
            raise ValueError("Reference changed; use reset to set a new target")

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = {} if options is None else dict(options)
        if set(options) - {"target_speed", "state"}:
            raise ValueError("Only target_speed and state are supported reset options")
        target = options.get("target_speed", self._fixed_target)
        self._target_speed = self._validate_target(self.np_random.uniform(-2., 2.) if target is None else target)
        self.goal = np.full(2, self.target_speed, dtype=np.float64)
        self._goal_snapshot = self.goal.copy()
        state = np.asarray(options.get("state", self.np_random.uniform(-3., 3., 2)), dtype=np.float64)
        if state.shape != (2,) or not np.isfinite(state).all() or np.any(abs(state) > 6.):
            raise ValueError("Reset state must be finite [current A,speed rad/s] in [-6,6]^2")
        self.state = state.copy()
        self._initial_state, self._seed_record = self.state.copy(), seed
        self._elapsed_steps, self._hold_steps, self._done = 0, 0, False
        self._pending_control = self._episode_metric = None
        self._execution_mode = None
        self._metric_steps = 0
        self._in_goal_since = 0. if self.is_success(self.state) else None
        self._first_hold_confirmation = None
        self._trace = []
        self._last_info = self._info()
        return self.state.copy(), dict(self._last_info)

    def tensor_error(self, state):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float64)
        if not state.is_floating_point() or state.shape[-1:] != (2,):
            raise ValueError("State must have floating last dimension 2")
        return (state - torch.as_tensor(self.goal, dtype=state.dtype, device=state.device)) / state.new_tensor(self.error_scale)

    def build_error(self, observation, goal=None, representation=None, history=None):
        self._assert_reference_fixed()
        if representation is not None and not getattr(representation, "preserves_physical_coordinates", False):
            raise ValueError("This explicit full-state task requires physical coordinates")
        state = observation if torch.is_tensor(observation) else torch.as_tensor(observation, dtype=torch.float64)
        target = torch.as_tensor(self.goal if goal is None else goal, dtype=state.dtype, device=state.device)
        if target.shape != (2,) or not torch.equal(target, state.new_tensor(self.goal)):
            raise ValueError("Set the physical reference through reset")
        return self.tensor_error(state), {"state": state, "goal": target,
            "target_speed": self.target_speed, "equilibrium_voltage": self.equilibrium_voltage}

    def metric_context(self, observation=None, goal=None):
        self._assert_reference_fixed()
        if goal is not None and not np.array_equal(np.asarray(goal), self.goal):
            raise ValueError("Metric context cannot use another reference")
        return torch.tensor([self.target_speed / 2.], dtype=torch.float32)

    def physical_error_components(self, state):
        return np.asarray(state, dtype=np.float64) - self.goal

    def physical_cost(self, state):
        # Canonical physical units (1 A and 1 rad/s); not the /6 certificate scale.
        difference = self.physical_error_components(state)
        return float(difference @ difference)

    def is_success(self, state):
        difference = self.physical_error_components(state)
        return bool(np.isfinite(difference).all() and np.all(abs(difference) <= .02))

    def is_failure(self, state):
        state = np.asarray(state)
        return bool(not np.isfinite(state).all() or np.any(abs(state) > 6. + 1e-10))

    def torch_step(self, state, action):
        if not torch.is_tensor(state) or not torch.is_tensor(action):
            raise TypeError("Differentiable model requires torch state and action")
        if not state.is_floating_point() or state.shape[-1:] != (2,) or action.shape[-1:] != (1,):
            raise ValueError("Expected state[...,2] and voltage[...,1]")
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError("State and voltage must be finite")
        action = action.to(state)
        applied = action.clamp(-5., 5.)
        return state @ state.new_tensor(self.Ad).T + applied @ state.new_tensor(self.Bd).T

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        self._assert_reference_fixed()
        if controller_state is not None or certificate is not None:
            raise ValueError("Only the fixed quadratic M controller is supported")
        if not torch.is_tensor(M) or not torch.is_tensor(e) or not e.is_floating_point():
            raise TypeError("M and error must be floating tensors; retain the caller graph")
        if M.shape[-2:] != (2, 2) or e.shape[-1:] != (2,) or not torch.isfinite(M).all() or not torch.isfinite(e).all():
            raise ValueError("Invalid finite M/error dimensions")
        if context.get("target_speed") != self.target_speed or context.get("equilibrium_voltage") != self.equilibrium_voltage:
            raise ValueError("Stale or mismatched control reference")
        M = M.to(e)
        eigen = torch.linalg.eigvalsh(M)
        if (not torch.allclose(M, M.transpose(-1, -2), atol=1e-6, rtol=1e-6)
            or torch.any(eigen < self.metric_spec.eigen_floor - 1e-6)
            or torch.any(eigen > self.metric_spec.eigen_ceiling + 1e-6)
            or not torch.allclose(M.diagonal(dim1=-2, dim2=-1).sum(-1),
                                  M.new_full(M.shape[:-2], 2.), atol=2e-6, rtol=1e-6)):
            raise ValueError("M must obey the shared SPD/trace specification")
        prediction = e @ e.new_tensor(self.Ae).T
        Mb = torch.einsum("...ij,j->...i", M, e.new_tensor(self.Be))
        numerator = (Mb * prediction).sum(-1)
        denominator = (Mb * e.new_tensor(self.Be)).sum(-1) + self.voltage_deviation_weight
        correction = -numerator / denominator
        action = (correction + self.equilibrium_voltage).clamp(-5., 5.).unsqueeze(-1)
        # Detached audit data does not participate in the returned control graph.
        self._pending_control = None
        if M.shape == (2, 2) and e.shape == (2,):
            self._pending_control = {"mode": "metric", "M": M.detach().cpu().numpy().copy(),
                "error": e.detach().cpu().numpy().copy(), "action": action.detach().cpu().numpy().copy()}
        return action

    def fixed_feedforward_action(self):
        self._assert_reference_fixed()
        action = torch.tensor([self.equilibrium_voltage], dtype=torch.float64)
        self._pending_control = {"mode": "fixed_feedforward", "M": None,
            "error": (self.state-self.goal)/self.error_scale, "action": action.numpy().copy()}
        return action

    def unit_metric_action(self, observation=None):
        error, context = self.build_error(self.state if observation is None else observation)
        return self.calculate_action_from_M(torch.eye(2, dtype=error.dtype, device=error.device), error, context=context)

    def _info(self):
        elapsed = self._elapsed_steps * self.dt
        good, failure = self.is_success(self.state), self.is_failure(self.state)
        hold = 0. if self._in_goal_since is None else elapsed-self._in_goal_since
        hold_met = bool(good and hold + 1e-12 >= self.required_hold_seconds)
        success = bool(self._elapsed_steps >= self.max_steps and hold_met and not failure)
        error = self.physical_error_components(self.state)
        return {"success": success, "failure": failure, "in_goal": good,
            "hold_condition_met": hold_met, "hold_seconds": max(0., hold),
            "required_hold_seconds": self.required_hold_seconds,
            "settling_time": float(self._in_goal_since) if success else self.horizon_seconds,
            "arrival_time": float(self._in_goal_since+self.required_hold_seconds) if success else self.horizon_seconds,
            "first_hold_confirmation_time": self._first_hold_confirmation,
            "steps": self._elapsed_steps, "physical_steps": self._elapsed_steps,
            "elapsed_seconds": elapsed, "horizon_seconds": self.horizon_seconds,
            "physical_error": self.physical_cost(self.state),
            "current_error_A": float(error[0]), "speed_error_rad_s": float(error[1]),
            "normalized_error": (error/self.error_scale).tolist(),
            "terminal_physical_error": self.physical_cost(self.state),
            "target_speed_rad_s": self.target_speed, "equilibrium_voltage_V": self.equilibrium_voltage,
            "metric_control_steps": self._metric_steps, "execution_mode": self._execution_mode,
            "metric_fixed_in_episode": self._execution_mode == "metric",
            "cost_convention": "positive_integral", "protocol_kind": self.protocol_kind,
            "success_scope": "sampled_closed_final_interval_not_intersample_certificate"}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode")
        self._assert_reference_fixed()
        submitted = np.asarray(action, dtype=np.float64)
        if submitted.shape != (1,) or not np.isfinite(submitted).all():
            raise ValueError("Action must be one finite physical voltage")
        applied = np.clip(submitted, -5., 5.)
        pending = self._pending_control
        mode = "external_physical_action" if pending is None else pending["mode"]
        if self._execution_mode is not None and mode != self._execution_mode:
            raise ValueError("Cannot mix execution policies inside a declared episode")
        if pending is not None:
            if not np.allclose(pending["error"], (self.state-self.goal)/self.error_scale, atol=1e-7, rtol=1e-6):
                raise ValueError("Pending action was calculated for a different physical state")
            if not np.allclose(pending["action"], applied, atol=1e-10, rtol=1e-7):
                raise ValueError("Submitted action differs from the recorded controller output")
            if mode == "metric" and self._episode_metric is not None and not np.array_equal(pending["M"], self._episode_metric):
                raise ValueError("M cannot change inside an executed metric episode")
        before = self.state.copy()
        with torch.no_grad():
            following = self.torch_step(torch.as_tensor(before), torch.as_tensor(applied)).cpu().numpy()
        if not np.isfinite(following).all():
            raise FloatingPointError("Nonfinite physical transition; refusing an invalid record")
        good_before = self.is_success(before)
        self._execution_mode = mode
        if mode == "metric":
            if self._episode_metric is None:
                self._episode_metric = pending["M"].copy()
            self._metric_steps += 1
        self._pending_control = None
        self.state = following.copy()  # Never clip state to manufacture safety.
        self._elapsed_steps += 1
        now = self._elapsed_steps*self.dt
        good = self.is_success(self.state)
        if not good:
            self._in_goal_since = None
        elif not good_before or self._in_goal_since is None:
            self._in_goal_since = now
        hold = 0. if self._in_goal_since is None else now-self._in_goal_since
        self._hold_steps = int(round(max(0., hold)/self.dt))
        if good and not self.is_failure(self.state) and hold+1e-12 >= self.required_hold_seconds and self._first_hold_confirmation is None:
            self._first_hold_confirmation = now
        info = self._info()
        cost = self.dt*self.physical_cost(before)
        terminated = bool(info["failure"] or info["success"])
        truncated = bool(self._elapsed_steps >= self.max_steps and not terminated)
        self._done = terminated or truncated
        info.update(cost=cost, applied_action=applied.copy(), submitted_action=submitted.copy(),
                    physical_terminated=terminated, physical_truncated=truncated)
        e0, e1 = (before-self.goal)/self.error_scale, (self.state-self.goal)/self.error_scale
        metric = self._episode_metric
        self._trace.append({"step": self._elapsed_steps, "time": now,
            "state_before": before.tolist(), "state_after": self.state.tolist(),
            "action": applied.tolist(), "cost": cost, "in_goal": good,
            "success": info["success"], "failure": info["failure"], "hold_seconds": info["hold_seconds"],
            "V_before": None if metric is None else float(e0@metric@e0),
            "V_after": None if metric is None else float(e1@metric@e1)})
        self._last_info = info
        return self.state.copy(), cost, terminated, truncated, dict(info)

    def get_episode_audit(self):
        return {**self._last_info, "complete": self._done, "initial_seed": self._seed_record,
            "initial_state": self._initial_state.tolist(), "goal": self.goal.tolist(),
            "metric": None if self._episode_metric is None else self._episode_metric.tolist(),
            "trace": deepcopy(self._trace), "training_updates": 0,
            "registered": False, "formal_training_authorized_here": False}
