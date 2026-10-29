"""Full-horizon physical task protocol around immutable seven-task sources.

No learning code lives here. The wrapped task still owns its error, M context,
differentiable controller and physical step. Only episode/evaluation semantics
are strengthened: no early success, a final sustained hold and full traces.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "stabilization-gym", ROOT / "stabilization-learning-control"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stabilization_gym.envs import make as make_original

TASKS = ("coupled_linear", "point_mass", "pendulum", "cartpole",
         "two_link_arm", "unicycle_tracking", "multirotor_tracking")
FORMS = {
    "coupled_linear": "nominal_LQR_plus_metric_gradient_correction",
    "point_mass": "nominal_LQR_plus_metric_gradient_correction",
    "pendulum": "local_LQR_plus_metric_gradient_correction",
    "cartpole": "local_LQR_plus_metric_gradient_correction",
    "two_link_arm": "metric_virtual_acceleration_then_computed_torque",
    "unicycle_tracking": "metric_body_error_feedforward_feedback",
    "multirotor_tracking": "metric_outer_force_and_inner_pitch_feedback",
}


class FullHorizonTask(gym.Wrapper):
    """Run the full declared horizon unless a physical failure occurs.

    ``arrival_time`` includes the required hold, using the FINAL uninterrupted
    in-goal segment. This is stricter than the first transient tolerance visit.
    It is horizon-capped on failure/timeout. ``settling_time`` is the start of
    that final segment. Both require final success to be reported as arrivals.
    """

    def __init__(self, env, env_id, *, hold_seconds=1.0, tracking_cycles=2.0):
        super().__init__(env)
        if env_id not in TASKS:
            raise ValueError("Full-horizon protocol supports only the seven declared state tasks")
        if not math.isfinite(hold_seconds) or hold_seconds < 1.:
            raise ValueError("Static goal hold must be at least one second")
        if not math.isfinite(tracking_cycles) or tracking_cycles < 2.:
            raise ValueError("Moving-reference horizon must cover at least two periods")
        self.inner_env = env
        self.env_id = env_id
        self.controller_form = FORMS[env_id]
        self.dt = float(env.dt)
        self.error_dim, self.metric_context_dim = env.error_dim, env.metric_context_dim
        self.metric_spec = env.metric_spec
        self.controller_capabilities = dict(env.controller_capabilities,
            episode_fixed_metric=True, full_horizon_evaluation=True,
            certificate_guarantee="candidate_only")
        self.moving_reference = (env_id == "unicycle_tracking" or
            (env_id == "multirotor_tracking" and env.trajectory == "circle"))
        self.reference_period = 2 * math.pi / float(env.reference_omega) if self.moving_reference else 0.
        requested_hold = self.reference_period if self.moving_reference else float(hold_seconds)
        self.required_hold_steps = int(math.ceil(requested_hold / self.dt - 1e-12))
        self.required_hold_seconds = self.required_hold_steps * self.dt
        minimum_horizon = int(math.ceil(tracking_cycles * self.reference_period / self.dt - 1e-12)) if self.moving_reference else self.required_hold_steps + 1
        self.max_steps = max(int(env.max_steps), minimum_horizon)
        self.horizon_seconds = self.max_steps * self.dt
        self.tracking_cycles = float(tracking_cycles)
        self.protocol_kind = "seven_task_full_horizon_metric_v1"
        self.trace = []
        self._done = True
        self._steps = 0
        self._in_goal_since = None
        self._first_hold_confirmation = None
        self._episode_metric = None
        self._pending_metric = None
        self._metric_control_steps = 0
        self._last_info = {}
        self._reset_state_snapshot = None
        self._reset_seed = None

    @property
    def goal(self):
        return self.inner_env.goal

    @property
    def state(self):
        return self.inner_env.state

    def tensor_error(self, state):
        return self.inner_env.tensor_error(state)

    def physical_cost(self, state):
        return self.inner_env.physical_cost(state)

    def build_error(self, observation, goal=None, representation=None, history=None):
        return self.inner_env.build_error(observation, goal=goal,
            representation=representation, history=history)

    def metric_context(self, observation=None, goal=None):
        return self.inner_env.metric_context(observation, goal=goal)

    def torch_step(self, state, action):
        return self.inner_env.torch_step(state, action)

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        action = self.inner_env.calculate_action_from_M(M, e, context=context,
            controller_state=controller_state, certificate=certificate)
        # Preserve the autograd graph of the returned action. The detached copy
        # is only an execution audit and is committed on an actual physical step.
        candidate = torch.as_tensor(M).detach().cpu().numpy()
        self._pending_metric = candidate.copy() if candidate.shape == (self.error_dim, self.error_dim) else None
        return action

    def reset(self, *, seed=None, options=None):
        observation, _ = self.inner_env.reset(seed=seed, options=options)
        self._steps, self._done = 0, False
        self.trace = []
        self._reset_seed = seed
        self._reset_state_snapshot = np.asarray(observation).copy()
        good = bool(self.inner_env.is_success(observation))
        self._in_goal_since = 0. if good else None
        self._first_hold_confirmation = None
        self._episode_metric = self._pending_metric = None
        self._metric_control_steps = 0
        self._last_info = self._info(good, bool(self.inner_env.is_failure(observation)))
        return observation, dict(self._last_info)

    def _info(self, good, failure):
        elapsed = self._steps * self.dt
        hold = 0. if self._in_goal_since is None else elapsed - self._in_goal_since
        met = bool(good and hold + 1e-10 >= self.required_hold_seconds)
        success = bool(self._steps >= self.max_steps and met and not failure)
        arrival = self._in_goal_since + self.required_hold_seconds if success else self.horizon_seconds
        error = float(self.physical_cost(self.state))
        return dict(success=success, failure=bool(failure), in_goal=bool(good),
            hold_condition_met=met, hold_seconds=float(max(0., hold)), required_hold_seconds=self.required_hold_seconds,
            arrival_time=float(arrival), settling_time=float(self._in_goal_since) if success else self.horizon_seconds,
            first_hold_confirmation_time=self._first_hold_confirmation,
            horizon_seconds=self.horizon_seconds, physical_steps=self._steps, steps=self._steps,
            elapsed_seconds=elapsed, physical_error=error, terminal_physical_error=error,
            distance=math.sqrt(max(error, 0.)), cost_convention="positive_integral",
            moving_reference=self.moving_reference, reference_period=self.reference_period,
            metric_control_steps=self._metric_control_steps, metric_fixed_in_episode=True,
            protocol_kind=self.protocol_kind)

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset before stepping a finished full-horizon episode")
        submitted = np.asarray(action, dtype=np.float64)
        if submitted.shape != self.action_space.shape or not np.isfinite(submitted).all():
            raise ValueError("Action must be a finite physical-action vector")
        applied = np.clip(submitted, self.action_space.low, self.action_space.high)
        pending = self._pending_metric
        if pending is not None:
            if not np.isfinite(pending).all():
                raise ValueError("Nonfinite episode metric")
            if self._episode_metric is None:
                if not np.allclose(pending, pending.T, atol=1e-10, rtol=1e-8) or np.linalg.eigvalsh(pending).min() <= 0:
                    raise ValueError("Executed metric must be symmetric positive definite")
                self._episode_metric = pending.copy()
            elif not np.array_equal(pending, self._episode_metric):
                raise ValueError("M changed inside a fixed-M physical episode")
            self._metric_control_steps += 1
        self._pending_metric = None
        before = self.state.copy()
        before_error = self.tensor_error(torch.as_tensor(before)).detach().cpu().numpy()
        good_before = bool(self.inner_env.is_success(before))
        cost_before = float(self.physical_cost(before))
        with torch.no_grad():
            after = self.inner_env.torch_step(torch.as_tensor(before), torch.as_tensor(applied)).cpu().numpy()
        if not np.isfinite(after).all():
            raise FloatingPointError("Nonfinite physical state; cannot produce a valid training record")
        self.inner_env.state = after.copy()
        self._steps += 1
        self.inner_env._elapsed_steps = self._steps
        good = bool(self.inner_env.is_success(after))
        failure = bool(self.inner_env.is_failure(after))
        time_now = self._steps * self.dt
        if not good:
            self._in_goal_since = None
        elif not good_before or self._in_goal_since is None:
            self._in_goal_since = time_now
        hold = 0. if self._in_goal_since is None else time_now - self._in_goal_since
        if good and not failure and hold + 1e-10 >= self.required_hold_seconds and self._first_hold_confirmation is None:
            self._first_hold_confirmation = time_now
        info = self._info(good, failure)
        cost = self.dt * cost_before
        terminated = failure or info["success"]
        truncated = self._steps >= self.max_steps and not terminated
        self._done = bool(terminated or truncated)
        self.inner_env._done = self._done
        self.inner_env._hold_steps = int(round(max(hold, 0.) / self.dt))
        after_error = self.tensor_error(torch.as_tensor(after)).detach().cpu().numpy()
        record = dict(step=self._steps, time=time_now, state_before=before.tolist(), state_after=after.tolist(),
            error_before=before_error.tolist(), error_after=after_error.tolist(), action=applied.tolist(),
            physical_error_before=cost_before, physical_error_after=info["physical_error"], cost=cost,
            in_goal_before=good_before, in_goal_after=good, hold_seconds=info["hold_seconds"],
            success=info["success"], failure=failure, terminated=bool(terminated), truncated=bool(truncated))
        self.trace.append(record)
        info.update(cost=cost, applied_action=applied.copy(), state_before=before,
            state_after=after.copy(), error_before=before_error, error_after=after_error,
            physical_terminated=bool(terminated), physical_truncated=bool(truncated))
        self._last_info = info
        return after.copy(), cost, bool(terminated), bool(truncated), dict(info)

    def get_episode_audit(self):
        info = self._last_info
        summary_keys = ("success", "failure", "arrival_time", "settling_time", "horizon_seconds",
            "hold_seconds", "required_hold_seconds", "physical_steps", "terminal_physical_error",
            "first_hold_confirmation_time", "moving_reference", "reference_period", "metric_control_steps")
        result = {key: info.get(key) for key in summary_keys}
        result.update(complete=self._done, protocol_kind=self.protocol_kind, env_id=self.env_id,
            controller_form=self.controller_form, initial_seed=self._reset_seed,
            initial_state=None if self._reset_state_snapshot is None else self._reset_state_snapshot.tolist(),
            metric=None if self._episode_metric is None else self._episode_metric.tolist(),
            metric_fixed_in_episode=True, trace=deepcopy(self.trace))
        return result


def make_task(env_id, env_kwargs=None, *, hold_seconds=1.0, tracking_cycles=2.0):
    """Create an existing physical task under the new full-horizon protocol."""
    if env_id not in TASKS:
        raise ValueError("Unknown seven-task ID: " + str(env_id))
    return FullHorizonTask(make_original(env_id, **dict(env_kwargs or {})), env_id,
                           hold_seconds=hold_seconds, tracking_cycles=tracking_cycles)


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def independent_physical_step(task, state, action):
    """NumPy equation check independent of the environment's torch_step code."""
    env, kind = task.inner_env, task.env_id
    s, u = np.asarray(state, float), np.asarray(action, float)
    u = np.clip(u, env.action_space.low, env.action_space.high)
    dt = task.dt
    if kind == "point_mass":
        return np.r_[s[:2] + dt * s[2:] + .5 * dt * dt * u, s[2:] + dt * u]
    if kind in ("coupled_linear", "pendulum", "cartpole"):
        def flow(x):
            if kind == "coupled_linear":
                return np.r_[x[2:], -np.array([[2., -.6], [-.6, 1.5]]) @ x[:2]
                    - np.array([[.25, .03], [.03, .3]]) @ x[2:] + u]
            if kind == "pendulum":
                return np.array([x[1], env.gravity / env.length * np.sin(x[0]) - env.damping * x[1] + u[0] / (env.mass * env.length ** 2)])
            total, moment = env.masscart + env.masspole, env.masspole * env.length
            sine, cosine = np.sin(x[2]), np.cos(x[2])
            temp = (u[0] + moment * x[3] ** 2 * sine) / total
            theta_acc = (env.gravity * sine - cosine * temp) / (env.length * (4. / 3. - env.masspole * cosine ** 2 / total))
            return np.array([x[1], temp - moment * theta_acc * cosine / total, x[3], theta_acc])
        a = flow(s); b = flow(s + .5 * dt * a); c = flow(s + .5 * dt * b); d = flow(s + dt * c)
        return s + dt / 6 * (a + 2 * b + 2 * c + d)
    if kind == "two_link_arm":
        q1, q2, v1, v2 = s
        m1, m2, l1, l2 = env.mass1, env.mass2, env.length1, env.length2
        c1, c2 = l1 / 2, l2 / 2
        i1, i2 = m1 * l1 ** 2 / 12, m2 * l2 ** 2 / 12
        h11 = i1 + i2 + m1 * c1 ** 2 + m2 * (l1 ** 2 + c2 ** 2 + 2 * l1 * c2 * np.cos(q2))
        h12 = i2 + m2 * (c2 ** 2 + l1 * c2 * np.cos(q2)); h22 = i2 + m2 * c2 ** 2
        h = -m2 * l1 * c2 * np.sin(q2)
        C = np.array([h * (2 * v1 * v2 + v2 ** 2), -h * v1 ** 2])
        G = env.gravity * np.array([(m1 * c1 + m2 * l1) * np.sin(q1) + m2 * c2 * np.sin(q1 + q2), m2 * c2 * np.sin(q1 + q2)])
        acceleration = np.linalg.solve(np.array([[h11, h12], [h12, h22]]), u - C - G)
        velocity = s[2:] + dt * acceleration
        return np.r_[_wrap(s[:2] + dt * velocity), velocity]
    if kind == "unicycle_tracking":
        heading = s[2] + .5 * dt * u[1]
        return np.array([s[0] + dt * u[0] * np.cos(heading), s[1] + dt * u[0] * np.sin(heading),
                         _wrap(s[2] + dt * u[1]), _wrap(s[3] + dt * env.reference_omega)])
    if kind == "multirotor_tracking":
        vx = s[2] - dt * u[0] * np.sin(s[4]) / env.mass
        vz = s[3] + dt * (u[0] * np.cos(s[4]) / env.mass - env.gravity)
        omega = s[5] + dt * u[1] / env.inertia
        phase_speed = env.reference_omega if env.trajectory == "circle" else 0.
        return np.array([s[0] + dt * vx, s[1] + dt * vz, vx, vz,
                         _wrap(s[4] + dt * omega), omega, _wrap(s[6] + dt * phase_speed)])
    raise ValueError(kind)


def audit_task(env_id, env_kwargs=None, *, seed=7241, verify_full_episode=True):
    """CPU mathematical/protocol audit; does not train or claim learned gains."""
    from stabilization_learning.metrics import CholeskyMetric
    task = make_task(env_id, env_kwargs)
    paired = make_task(env_id, env_kwargs)
    try:
        observation, _ = task.reset(seed=seed)
        other, _ = paired.reset(seed=seed)
        seed_pair = np.array_equal(observation, other)
        e, context = task.build_error(observation)
        construct = CholeskyMetric(task.metric_spec).double()
        raw = construct.identity_raw(dtype=torch.float64)
        raw = (raw + .01 * torch.arange(1, len(raw) + 1, dtype=torch.float64) / len(raw)).requires_grad_()
        def action_of(value):
            return task.calculate_action_from_M(construct(value), e, context=context)
        jac = torch.autograd.functional.jacobian(action_of, raw)
        row, column = np.unravel_index(int(jac.detach().abs().argmax()), tuple(jac.shape))
        eps = 1e-5
        direction = torch.zeros_like(raw); direction[column] = 1.
        numeric = (action_of(raw.detach() + eps * direction)[row] - action_of(raw.detach() - eps * direction)[row]) / (2 * eps)
        analytic = jac[row, column]
        fd_error = float(abs(numeric - analytic))
        M = construct(raw).detach()
        u = action_of(raw).detach()
        changed_u = action_of(raw.detach() + .1 * direction).detach()
        action_change = float(torch.linalg.vector_norm(changed_u - u))
        # Restore pending metric to the one associated with the actual action.
        u = task.calculate_action_from_M(M, e, context=context).detach().numpy()
        expected = independent_physical_step(task, observation, u)
        following, _, _, _, _ = task.step(u)
        step_error = float(np.max(np.abs(following - expected)))
        paired_e, paired_context = paired.build_error(other)
        paired_u = paired.calculate_action_from_M(M, paired_e, context=paired_context).detach().numpy()
        paired_next = paired.step(paired_u)[0]
        seed_pair = bool(seed_pair and np.array_equal(u, paired_u) and np.array_equal(following, paired_next))
        eigen = torch.linalg.eigvalsh(M)
        checks = dict(seed_initial_and_step_pair_exact=seed_pair, cholesky_action_gradient_nonzero=bool(jac.abs().max() > 1e-9),
            cholesky_action_finite_difference=bool(fd_error < 1e-6 + 2e-4 * abs(float(analytic))),
            M_changes_executed_controller=action_change > 1e-8, independent_physical_step=step_error < 1e-10,
            SPD_trace_bounds=bool(eigen.min() >= task.metric_spec.eigen_floor - 1e-10 and
                eigen.max() <= task.metric_spec.eigen_ceiling + 1e-10 and abs(float(M.trace()) - task.metric_spec.trace) < 1e-10))
        nominal = None
        if verify_full_episode:
            obs, _ = task.reset(seed=seed)
            identity = torch.eye(task.error_dim, dtype=torch.float64)
            for step in range(task.max_steps):
                error, ctx = task.build_error(obs)
                command = task.calculate_action_from_M(identity, error, context=ctx)
                obs, _, terminated, truncated, info = task.step(command.detach().numpy())
                if terminated or truncated:
                    break
            nominal = task.get_episode_audit()
            checks["no_premature_success"] = not nominal["success"] or nominal["physical_steps"] == task.max_steps
            checks["success_implies_required_final_hold"] = not nominal["success"] or nominal["hold_seconds"] + 1e-10 >= nominal["required_hold_seconds"]
            checks["full_horizon_or_physical_failure"] = nominal["physical_steps"] == task.max_steps or nominal["failure"]
            checks["metric_used_and_frozen_every_step"] = nominal["metric_control_steps"] == nominal["physical_steps"]
            checks["tracking_at_least_two_periods"] = not task.moving_reference or task.horizon_seconds + 1e-10 >= 2 * task.reference_period
            nominal = {key: value for key, value in nominal.items() if key not in ("trace", "metric")}
        return dict(passed=all(checks.values()), env_id=env_id, checks=checks,
            action_gradient_max_abs=float(jac.abs().max()), finite_difference_abs_error=fd_error,
            action_change_norm=action_change, independent_step_max_abs=step_error,
            nominal_protocol_check=nominal, scope="CPU engineering audit, no learning-performance claim")
    finally:
        task.close(); paired.close()
