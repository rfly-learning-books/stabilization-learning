"""State/target-conditioned M version of the isolated DC motor.

The physical model, limits, normalization, hold rule and success criterion are
reused from the reviewed isolated environment.  This version deliberately does
NOT enforce an episode-fixed executed M; each measurement update may select a
new metric.  The one-step controller and the candidate CLF screen both consume
the metric selected for that state.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch

from . import _path_setup  # noqa: F401
from dc_motor_env import DCMotorExactZOHEnv
from ..core.certificate import quadratic
from ..core.gain import trace_eigen_bounds


class StateDependentDCMotorEnv(DCMotorExactZOHEnv):
    """DC motor with state/target-conditioned M.

    The class keeps truth and goal internally, but the metric context exposed to
    the actor contains only normalized state and target.  The controller uses
    the current observed state and the fixed target, as in normal state feedback.
    """

    controller_form = "state_conditioned_M_one_step_voltage_plus_optional_CLF_candidate"
    controller_call_is_pure = True
    control_call_side_effects = ()

    def __init__(self, *, target_speed=None, render_mode=None):
        super().__init__(target_speed=target_speed, render_mode=render_mode)
        self.metric_context_dim = 4
        self.controller_capabilities.update(
            state_dependent_certificate=True,
            episode_fixed_metric=False,
            certificate_guarantee="candidate_only",
        )
        self._active_metric = None
        self._active_action = None
        self._metric_trace = []

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._active_metric = None
        self._active_action = None
        self._metric_trace = []
        return observation, info

    def metric_context(self, observation=None, goal=None):
        """Return deployment-legal normalized [state, goal] context."""
        self._assert_reference_fixed()
        if goal is not None and not np.array_equal(np.asarray(goal), self.goal):
            raise ValueError("Metric context cannot use another reference")
        state = self.state if observation is None else observation
        state = state if torch.is_tensor(state) else torch.as_tensor(state, dtype=torch.float32)
        state = state.to(dtype=torch.float32)
        if state.shape[-1:] != (2,) or not torch.isfinite(state).all():
            raise ValueError("Expected finite two-dimensional motor state")
        goal_t = torch.as_tensor(self.goal, dtype=state.dtype, device=state.device)
        scale = state.new_tensor(self.error_scale)
        return torch.cat([state / scale, goal_t / scale], dim=-1)

    def context_for_states(self, states):
        """Batched metric context for candidate next-state evaluation."""
        states = states if torch.is_tensor(states) else torch.as_tensor(states, dtype=torch.float64)
        if states.shape[-1:] != (2,):
            raise ValueError("Expected states[...,2]")
        states = states.to(dtype=torch.float32)
        goal_t = torch.as_tensor(self.goal, dtype=states.dtype, device=states.device)
        scale = states.new_tensor(self.error_scale)
        return torch.cat([states / scale, goal_t.expand_as(states) / scale], dim=-1)

    def analytic_action_from_M(self, M, e, context=None):
        """One-step quadratic voltage derived from M and normalized error."""
        if M.shape[-2:] != (2, 2) or e.shape[-1:] != (2,):
            raise ValueError("Expected M[...,2,2] and e[...,2]")
        if not torch.isfinite(M).all() or not torch.isfinite(e).all():
            raise ValueError("Non-finite M or error")
        M = M.to(e)
        threshold = 1e-6
        eig = torch.linalg.eigvalsh(M)
        if (not torch.allclose(M, M.transpose(-1, -2), atol=threshold, rtol=threshold)
                or torch.any(eig < self.metric_spec.eigen_floor - threshold)
                or torch.any(eig > self.metric_spec.eigen_ceiling + threshold)
                or not torch.allclose(M.diagonal(dim1=-2, dim2=-1).sum(-1),
                                      M.new_full(M.shape[:-2], 2.), atol=2e-6, rtol=1e-6)):
            raise ValueError("M violates the configured SPD/trace bounds")
        prediction = e @ e.new_tensor(self.Ae).T
        Mb = torch.einsum("...ij,j->...i", M, e.new_tensor(self.Be))
        numerator = (Mb * prediction).sum(-1)
        denominator = (Mb * e.new_tensor(self.Be)).sum(-1) + self.voltage_deviation_weight
        correction = -numerator / denominator
        action = (correction + self.equilibrium_voltage).clamp(-5., 5.)
        return action.unsqueeze(-1)

    def controller_action(self, M, e, context=None):
        action = self.analytic_action_from_M(M, e, context)
        self._active_metric = M.detach().cpu().numpy().copy()
        self._active_action = action.detach().cpu().numpy().copy()
        return action

    @torch.no_grad()
    def clf_candidate_action(self, actor, state, M, *, alpha=0.05, points=21):
        """Hard one-step candidate screen using M(s,s*) and same-version V.

        Candidates are a fixed voltage grid plus the analytic action.  The
        selected action is the lowest physical-error candidate satisfying the
        one-step residual; if none is feasible, the minimum-residual action is
        returned and the caller must record infeasibility.
        """
        state = torch.as_tensor(state, dtype=torch.float64)
        e0 = self.tensor_error(state)
        M = M.to(e0)
        v0 = quadratic(e0, M)
        analytic = self.analytic_action_from_M(M, e0).squeeze(-1)
        grid = torch.linspace(-5., 5., int(points), dtype=torch.float64)
        candidates = torch.unique(torch.cat([grid, analytic.reshape(1)]))
        following = self.torch_step(state.expand(candidates.numel(), 2), candidates.unsqueeze(-1))
        e1 = self.tensor_error(following)
        context1 = self.context_for_states(following)
        M1 = actor(context1).to(e1)
        v1 = quadratic(e1, M1)
        residual = v1 - v0 + alpha * self.dt * e0.square().sum().expand_as(v1)
        physical = e1.square().sum(-1) + self.voltage_deviation_weight * candidates.square()
        feasible = residual <= 0.
        if bool(feasible.any()):
            masked = physical.masked_fill(~feasible, float("inf"))
            index = int(torch.argmin(masked))
        else:
            index = int(torch.argmin(residual))
        action = candidates[index].reshape(1)
        audit = {
            "candidate_count": int(candidates.numel()),
            "feasible_count": int(feasible.sum()),
            "selected_residual": float(residual[index]),
            "selected_physical_cost": float(physical[index]),
            "M_current": M.detach().cpu().tolist(),
            "M_next": M1[index].detach().cpu().tolist(),
        }
        self._active_metric = M.detach().cpu().numpy().copy()
        self._active_action = action.detach().cpu().numpy().copy()
        return action, audit

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode")
        self._assert_reference_fixed()
        submitted = np.asarray(action.detach().cpu() if torch.is_tensor(action) else action, dtype=np.float64)
        if submitted.shape != (1,) or not np.isfinite(submitted).all():
            raise ValueError("Action must be one finite physical voltage")
        applied = np.clip(submitted, -5., 5.)
        if self._active_action is not None and not np.allclose(self._active_action, applied, atol=1e-10, rtol=1e-7):
            raise ValueError("Submitted action differs from state-dependent controller output")
        before = self.state.copy()
        with torch.no_grad():
            following = self.torch_step(torch.as_tensor(before), torch.as_tensor(applied)).cpu().numpy()
        if not np.isfinite(following).all():
            raise FloatingPointError("Nonfinite physical transition; refusing an invalid record")
        good_before = self.is_success(before)
        self.state = following.copy()
        self._elapsed_steps += 1
        now = self._elapsed_steps * self.dt
        good = self.is_success(self.state)
        if not good:
            self._in_goal_since = None
        elif not good_before or self._in_goal_since is None:
            self._in_goal_since = now
        hold = 0. if self._in_goal_since is None else now - self._in_goal_since
        self._hold_steps = int(round(max(0., hold) / self.dt))
        if good and not self.is_failure(self.state) and hold + 1e-12 >= self.required_hold_seconds and self._first_hold_confirmation is None:
            self._first_hold_confirmation = now
        info = self._info()
        cost = self.dt * self.physical_cost(before)
        terminated = bool(info["failure"] or info["success"])
        truncated = bool(self._elapsed_steps >= self.max_steps and not terminated)
        self._done = terminated or truncated
        info.update(cost=cost, applied_action=applied.copy(), submitted_action=submitted.copy(),
                    physical_terminated=terminated, physical_truncated=truncated,
                    state_dependent_metric=True)
        e0 = (before - self.goal) / self.error_scale
        e1 = (self.state - self.goal) / self.error_scale
        M = self._active_metric
        v0 = None if M is None else float(e0 @ M @ e0)
        v1 = None if M is None else float(e1 @ M @ e1)
        self._metric_trace.append({
            "step": self._elapsed_steps, "time": now,
            "state_before": before.tolist(), "state_after": self.state.tolist(),
            "action": applied.tolist(), "cost": cost, "in_goal": good,
            "success": info["success"], "failure": info["failure"],
            "hold_seconds": info["hold_seconds"], "M": None if M is None else M.tolist(),
            "V_before": v0, "V_after": v1,
            "same_M_one_step_residual": None if M is None else v1 - v0 + 0.05 * self.dt * float(e0 @ e0),
        })
        self._active_metric = None
        self._active_action = None
        self._last_info = info
        return self.state.copy(), cost, terminated, truncated, dict(info)

    def get_episode_audit(self):
        return {**self._last_info, "complete": self._done, "initial_seed": self._seed_record,
                "initial_state": self._initial_state.tolist(), "goal": self.goal.tolist(),
                "state_dependent_metric": True,
                "metric_trace": deepcopy(self._metric_trace),
                "training_updates": 0, "registered": False,
                "formal_training_authorized_here": False}


