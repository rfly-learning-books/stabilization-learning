"""Physical finite-horizon evaluation, separate from the training surrogate."""
import math
import numpy as np
import torch
from .objectives import quadratic


@torch.no_grad()
def evaluate(task, policy, starts, *, decay=.05, record=False, should_stop=None):
    state = starts.detach().clone().to(dtype=torch.float64, device="cpu")
    count = len(state)
    active = torch.ones(count, dtype=torch.bool)
    failed = torch.zeros_like(active)
    steps = torch.zeros(count, dtype=torch.long)
    cost = torch.zeros(count, dtype=torch.float64)
    energy = cost.clone(); penalty = cost.clone(); maximum = torch.full_like(cost, -1e100)
    peak = cost.clone(); rises = steps.clone(); violations = steps.clone()
    terminal = task.error(state).square().sum(-1)
    last_bad = torch.where(terminal <= task.success_tolerance**2, -1, 0)
    buffers = {} if record else None
    for k in range(task.max_steps):
        if should_stop and should_stop():
            raise InterruptedError("STOP during evaluation")
        if not active.any():
            break
        e = task.error(state); M = policy(e)
        requested = task.requested_control(M, e, state=state)
        u = torch.maximum(task.action_low, torch.minimum(task.action_high, requested))
        safe = torch.where(active[:, None], state, state.new_tensor(task.goal))
        usafe = torch.where(active[:, None], u, task.feedforward(safe))
        following = task.step_model(safe, usafe)
        if not torch.isfinite(following).all():
            raise FloatingPointError("Nonfinite evaluation state")
        en = task.error(following); Mn = policy(en)
        norm = e.square().sum(-1); nnorm = en.square().sum(-1)
        v0 = quadratic(e, M); v1 = quadratic(en, Mn)
        residual = v1 - v0 + decay * task.dt * norm
        rate = residual / (task.dt * norm.clamp_min(1e-18))
        p = residual.relu() / norm.clamp_min(1e-6)
        bad = task.failure(following); good = nnorm <= task.success_tolerance**2
        cost += torch.where(active, task.dt * norm, 0.)
        energy += torch.where(active, task.dt * u.square().sum(-1), 0.)
        penalty += torch.where(active, p, 0.)
        maximum = torch.maximum(maximum, torch.where(active, rate, -1e100))
        peak = torch.maximum(peak, torch.where(active, u.abs().amax(-1), 0.))
        rises += active & (v1 - v0 > 1e-8)
        violations += active & (rate > 1e-6)
        steps += active
        terminal = torch.where(active, nnorm, terminal)
        last_bad = torch.where(active & ~good, torch.full_like(last_bad, k + 1), last_bad)
        if record:
            values = dict(states=state, following=following, M_current=M, M_next=Mn,
                requested_actions=requested, actions=u, V_before=v0, V_after=v1,
                costs=task.dt * norm, penalties=p, active=active, failure=bad,
                in_goal=good, decay_residual=residual)
            for key, value in values.items():
                buffers.setdefault(key, []).append(value.numpy().copy())
        failed |= active & bad
        state = torch.where(active[:, None], following, state)
        active = active & ~bad
    start = (last_bad + 1).clamp_min(0)
    hold = (steps - start).clamp_min(0).to(torch.float64) * task.dt
    success = (~failed) & (steps == task.max_steps) & (hold + 1e-10 >= task.required_hold_seconds)
    remaining = (task.max_steps - steps).to(torch.float64) * task.dt
    fixed = cost + remaining * torch.where(failed, terminal.clamp_min(25.), terminal)
    rows = []
    for i in range(count):
        rows.append(dict(success=bool(success[i]), failure=bool(failed[i]), physical_steps=int(steps[i]),
            accumulated_cost=float(cost[i]), fixed_horizon_cost=float(fixed[i]),
            certificate_penalty=float(penalty[i]), objective=float(fixed[i] + .1 * penalty[i]),
            terminal_physical_error=float(terminal[i]), hold_seconds=float(hold[i]),
            settling_time=float(start[i]) * task.dt if success[i] else task.max_steps * task.dt,
            arrival_time=float(start[i]) * task.dt + task.required_hold_seconds if success[i] else task.max_steps * task.dt,
            control_energy=float(energy[i]), peak_action=float(peak[i]),
            max_normalized_decay_rate=float(maximum[i]),
            V_rise_fraction=float(rises[i]) / max(int(steps[i]), 1),
            decay_violation_fraction=float(violations[i]) / max(int(steps[i]), 1)))
    return rows, None if not record else {k: np.stack(v) for k, v in buffers.items()}


def local_check(task, policy, *, decay=.05):
    goal = torch.tensor(task.goal, dtype=torch.float64)
    def closed(x):
        e = task.error(x)
        return task.step_model(x, task.control(policy(e), e, state=x))
    F = torch.autograd.functional.jacobian(closed, goal).detach().numpy()
    with torch.no_grad():
        M0 = policy(task.error(goal)).numpy()
        drift = float(torch.max(torch.abs(closed(goal) - goal)))
    D = M0 - F.T @ M0 @ F - decay * task.dt * np.eye(task.error_dim)
    minimum = float(np.linalg.eigvalsh((D + D.T) / 2).min())
    radius = float(np.max(np.abs(np.linalg.eigvals(F))))
    anchor_error = float(np.max(np.abs(M0 - task.P0)))
    return dict(passed=drift < 1e-12 and minimum > 0 and radius < 1 and anchor_error < 1e-10,
        fixed_target_drift=drift, spectral_radius=radius, decay_matrix_min_eigenvalue=minimum,
        anchor_error=anchor_error, F=F.tolist(), M0=M0.tolist(),
        claim="floating-point local linearization check, not a nonlinear-domain proof")


def selection_gate(groups, baseline, *, local, probe_rate, rate_tolerance=1e-6,
                   max_arrival_delay=.15, max_energy_ratio=1.1):
    """Feasibility gates first; only then rank by physical cost on validation."""
    reasons = []; ratios = []; details = []
    if not local["passed"]:
        reasons.append("local_linearization")
    if not math.isfinite(probe_rate) or probe_rate > rate_tolerance:
        reasons.append("independent_validation_probes")
    for scale, rows in groups.items():
        base = baseline[scale]
        cost = np.mean([r["fixed_horizon_cost"] for r in rows])
        bcost = np.mean([r["fixed_horizon_cost"] for r in base])
        ratio = float(cost / max(bcost, 1e-15))
        delay = float(np.mean([r["arrival_time"] - b["arrival_time"] for r, b in zip(rows, base)]))
        energy = float(np.mean([r["control_energy"] for r in rows]) /
                       max(np.mean([r["control_energy"] for r in base]), 1e-15))
        # Baseline-successful lanes must remain successful, not merely equal counts.
        if any(b["success"] and not r["success"] for r, b in zip(rows, base)) or any(r["failure"] for r in rows):
            reasons.append(f"success_or_failure:{scale}")
        if not math.isfinite(ratio) or ratio > 1 + 1e-10:
            reasons.append(f"physical_cost:{scale}")
        if max(r["max_normalized_decay_rate"] for r in rows) > rate_tolerance:
            reasons.append(f"trajectory_descent:{scale}")
        if delay > max_arrival_delay + 1e-12:
            reasons.append(f"arrival_delay:{scale}")
        if energy > max_energy_ratio + 1e-12:
            reasons.append(f"control_energy:{scale}")
        ratios.append(ratio)
        details.append(dict(scale=scale, cost_ratio=ratio, arrival_delay=delay, energy_ratio=energy,
            successes=sum(r["success"] for r in rows), episodes=len(rows)))
    return dict(qualified=not reasons, reasons=reasons, score=float(np.mean(ratios)), groups=details,
                probe_max_normalized_decay_rate=float(probe_rate), local=local)
