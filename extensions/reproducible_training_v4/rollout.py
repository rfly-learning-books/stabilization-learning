"""Actual-successor episode-M rollouts with explicit physical-action evidence."""
import math
import numpy as np
import torch
from stabilization_learning.metrics import quadratic_certificate
from extensions.unified_training_v3.numerical import require_finite


def rollout_metric(env, M, observation, *, representation=None, certificate_weight=0., decay=.05,
                   failure_error=25., save_trajectory=False):
    if any(not math.isfinite(x) or x < 0 for x in (certificate_weight, decay, failure_error)):
        raise ValueError("Rollout costs must be finite and nonnegative")
    require_finite(M, "episode metric")
    M = M.detach()
    if representation is getattr(env, "representation", None):
        representation = None
    dt, horizon = float(env.dt), int(env.max_steps)
    if not math.isfinite(dt) or dt <= 0 or horizon < 1:
        raise ValueError("Invalid physical time horizon")
    total_cost = penalty = 0.
    rises = steps = 0
    trajectory = []
    for step in range(horizon):
        e, context = env.build_error(observation, representation=representation)
        require_finite(e, "current feature error")
        m = M.to(e)
        before = float(quadratic_certificate(e, m).detach())
        action = env.calculate_action_from_M(m, e, context=context)
        command = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        require_finite(command, "commanded action")
        following, cost, terminated, truncated, info = env.step(command)
        applied = np.asarray(info.get("applied_action", command))
        require_finite((cost, applied), "physical transition")
        next_e, _ = env.build_error(following, representation=representation)
        require_finite(next_e, "actual successor feature error")
        after = float(quadratic_certificate(next_e, m.to(next_e)).detach())
        norm2 = float(e.detach().square().sum())
        delta = after-before
        # Only the denominator is floored. A zero error must have zero decay
        # demand; stabilizing that denominator must not invent a positive cost.
        residual = max(0., delta + decay*dt*norm2) / max(norm2, 1e-6)
        require_finite((before, after, norm2, residual), "certificate transition")
        total_cost += float(cost)
        penalty += residual
        rises += int(delta > 1e-8)
        steps = step+1
        if save_trajectory:
            trajectory.append(dict(step=steps, time=steps*dt, V_before=before, V_after=after,
                action=applied.tolist(), applied_action=applied.tolist(), commanded_action=command.tolist(),
                cost=float(cost), physical_error=float(info.get("physical_error", cost/dt))))
        observation = following
        if terminated or truncated:
            break
    terminal_error = float(info.get("physical_error", float(cost)/dt))
    failure, success = bool(info.get("failure", False)), bool(info.get("success", False))
    tail = max(terminal_error, failure_error) if failure else terminal_error
    fixed_cost = total_cost + max(0, horizon-steps)*dt*tail
    arrival = float(info.get("arrival_time", steps*dt)) if success else horizon*dt
    result = dict(fixed_horizon_cost=fixed_cost, accumulated_cost=total_cost,
        objective=fixed_cost+certificate_weight*penalty, certificate_penalty=penalty,
        V_rise_fraction=rises/max(steps,1), success=success, failure=failure,
        physical_steps=steps, arrival_time=arrival, terminal_physical_error=terminal_error,
        physical_terminated=bool(terminated), physical_truncated=bool(truncated))
    require_finite(result, "episode summary")
    if save_trajectory:
        result["trajectory"] = trajectory
    return result
