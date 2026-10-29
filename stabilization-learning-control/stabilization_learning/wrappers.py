"""One metric decision -> one complete physical episode (finite-horizon SMDP)."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch
from .metrics import CholeskyMetric, quadratic_certificate


def rollout_metric(env, M, observation, *, representation=None, certificate_weight=0., decay=.05,
                   failure_error=25., save_trajectory=False):
    """Use fixed M, fixed phi and physical success labels for the whole rollout.

    Physical cost uses a fixed horizon: after early success retain terminal
    error, after failure retain max(terminal error, failure_error). A fast crash
    therefore cannot obtain an artificially short low-cost trajectory.
    """
    M = M.detach()
    # A bound, unchanged encoder uses the frozen reference cached by reset.
    # Explicit representation paths are for offline differentiable training.
    if representation is getattr(env, "representation", None):
        representation = None
    dt, horizon = float(env.dt), int(env.max_steps)
    total_cost = penalty = 0.
    rises = steps = 0
    trajectory = []
    info = {}
    for step in range(horizon):
        e, context = env.build_error(observation, representation=representation)
        m = M.to(e)
        before = float(quadratic_certificate(e, m).detach())
        action = env.calculate_action_from_M(m, e, context=context)
        applied = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        following, cost, terminated, truncated, info = env.step(applied)
        next_e, _ = env.build_error(following, representation=representation)
        after = float(quadratic_certificate(next_e, m.to(next_e)).detach())
        delta = after - before
        scale = max(float(e.detach().square().sum()), 1e-6)
        residual = max(0., delta + decay * dt * scale) / scale
        total_cost += float(cost)
        penalty += residual
        rises += int(delta > 1e-8)
        steps = step + 1
        if save_trajectory:
            trajectory.append({"step": steps, "time": steps * dt, "V_before": before,
                               "V_after": after, "action": applied.tolist(), "cost": float(cost),
                               "physical_error": float(info.get("physical_error", cost / dt))})
        observation = following
        if terminated or truncated:
            break
    terminal_error = float(info.get("physical_error", float(cost) / dt))
    failure, success = bool(info.get("failure", False)), bool(info.get("success", False))
    tail = max(terminal_error, failure_error) if failure else terminal_error
    fixed_cost = total_cost + max(0, horizon - steps) * dt * tail
    result = {"fixed_horizon_cost": fixed_cost, "accumulated_cost": total_cost,
              "objective": fixed_cost + certificate_weight * penalty,
              "certificate_penalty": penalty, "V_rise_fraction": rises / max(steps, 1),
              "success": success, "failure": failure, "physical_steps": steps,
              "arrival_time": steps * dt if success else horizon * dt,
              "terminal_physical_error": terminal_error,
              "physical_terminated": bool(terminated), "physical_truncated": bool(truncated)}
    if not all(np.isfinite(result[k]) for k in ("fixed_horizon_cost", "objective", "terminal_physical_error")):
        raise FloatingPointError("Non-finite physical rollout; refusing to train on invalid data")
    if save_trajectory:
        result["trajectory"] = trajectory
    return result


class EpisodeMetricEnv(gym.Env):
    """SLC-compatible positive-cost M-action environment.

    step(raw) executes all physical steps and always terminates this metric
    decision. A physical time limit is a finite-horizon endpoint here, so no
    artificial value bootstrap across the reset to a different task occurs.
    """
    metadata = {"render_modes": []}

    def __init__(self, physical_env, representation=None, certificate_weight=0., decay=.05, raw_bound=5.):
        self.physical_env, self.representation = physical_env, representation
        if representation is not None:
            representation.eval().requires_grad_(False)
            if hasattr(physical_env, "bind_representation"):
                physical_env.bind_representation(representation)
        self.metric = CholeskyMetric(physical_env.metric_spec)
        self.certificate_weight, self.decay = certificate_weight, decay
        self.action_space = gym.spaces.Box(-raw_bound, raw_bound, (physical_env.metric_spec.raw_dim,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (physical_env.metric_context_dim,), np.float32)
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._observation, info = self.physical_env.reset(seed=seed, options=options)
        self._context = self.physical_env.metric_context(self._observation).detach().cpu().numpy().astype(np.float32)
        self._done = False
        return self._context.copy(), info

    def step(self, action):
        if self._done:
            raise RuntimeError("reset required before a new M decision")
        raw = torch.as_tensor(action, dtype=torch.float32)
        if raw.shape != self.action_space.shape or not torch.isfinite(raw).all():
            raise ValueError("Invalid Cholesky action")
        raw = raw.clamp(float(self.action_space.low[0]), float(self.action_space.high[0]))
        info = rollout_metric(self.physical_env, self.metric(raw), self._observation,
                              representation=self.representation,
                              certificate_weight=self.certificate_weight, decay=self.decay)
        self._done = True
        return self._context.copy(), info["objective"], True, False, info

    def close(self):
        self.physical_env.close()


class CostToReward(gym.RewardWrapper):
    """Explicit sign adapter for reward-maximizing off-the-shelf algorithms."""
    def reward(self, cost):
        return -cost
