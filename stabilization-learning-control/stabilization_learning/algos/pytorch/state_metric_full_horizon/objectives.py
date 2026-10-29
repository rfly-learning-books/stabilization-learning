"""Full episode objective, with exact segment recomputation during backward.

No gradient is detached at segment boundaries. Boolean termination remains a
piecewise branch; no gradient through the first failure time is claimed.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint


def quadratic(e, M):
    return torch.einsum("...i,...ij,...j->...", e, M, e)


def certificate_transition(task, policy, state, *, decay=.05):
    e = task.error(state)
    M = policy(e)
    action = task.control(M, e, state=state)
    following = task.step_model(state, action)
    en = task.error(following)
    # Successor M remains in the graph, including its dependence on en.
    delta = quadratic(en, policy(en)) - quadratic(e, M)
    norm2 = e.square().sum(-1)
    residual = delta + decay * task.dt * norm2
    rate = residual / (task.dt * norm2.clamp_min(1e-18))
    return following, action, norm2, en.square().sum(-1), residual, rate


def full_horizon_components(task, policy, starts, *, decay=.05, checkpoint_stride=32,
                            horizon=None, should_stop=None):
    """Per-lane [physical cost, descent risk, hold excess, soft arrival, effort].

The optional horizon is for gradient fixtures only. Production callers always
use the complete task.max_steps. Failure tails use float64 physical time.
"""
    T = task.max_steps if horizon is None else horizon
    if T < 1 or T > task.max_steps or checkpoint_stride < 0:
        raise ValueError("Invalid horizon or recomputation segment size")
    state = starts
    if state.dtype != torch.float64 or state.device.type != "cpu":
        raise ValueError("This recovery protocol is CPU float64 only")
    active = torch.ones(len(starts), dtype=torch.bool, device=starts.device)
    total = starts.new_zeros((len(starts), 5))
    goal = starts.new_tensor(task.goal)
    hold_steps = int(math.ceil(task.required_hold_seconds / task.dt))
    stride = checkpoint_stride or T
    for first in range(0, T, stride):
        if should_stop is not None and should_stop():
            raise InterruptedError("STOP before an uncommitted optimizer update")
        # Bind loop indices: checkpoint replay must use the same segment.
        def segment(x, alive, first=first, end=min(first + stride, T)):
            sums = x.new_zeros((len(x), 5))
            for k in range(first, end):
                y, u, norm, next_norm, _, rate = certificate_transition(task, policy, x, decay=decay)
                bad = task.failure(y)
                just_failed = alive & bad
                remaining_seconds = float(T - k - 1) * float(task.dt)
                physical = task.dt * norm + torch.where(
                    just_failed, remaining_seconds * next_norm.clamp_min(25.), 0.)
                # A differentiable boundary penalty accompanies the discrete failure cost.
                hold = task.boundary_excess(y) * 100.
                if k >= T - hold_steps - 1:
                    hold = hold + (next_norm / task.success_tolerance**2 - 1).relu().square()
                columns = torch.stack((physical,
                    task.dt * rate.relu().square(), task.dt * hold,
                    task.dt * norm / (norm + task.success_tolerance**2),
                    task.dt * u.square().sum(-1)), -1)
                sums = sums + torch.where(alive[:, None], columns, 0.)
                alive = alive & ~bad
                x = torch.where(alive[:, None], y, goal)
            return x, alive, sums
        if checkpoint_stride and torch.is_grad_enabled():
            state, active, part = checkpoint(segment, state, active,
                use_reentrant=False, preserve_rng_state=False)
        else:
            state, active, part = segment(state, active)
        total = total + part
    if not torch.isfinite(total).all():
        raise FloatingPointError("Nonfinite full-horizon objective")
    return total


def falsify(task, policy, starts, *, decay=.05, steps=4, step_fraction=.08,
            lower, upper):
    """Empirical projected-gradient search. It is NOT a formal verifier."""
    state = starts.detach().clone()
    width = upper - lower
    best = state.clone()
    with torch.no_grad():
        best_rate = certificate_transition(task, policy, state, decay=decay)[-1]
    for _ in range(steps):
        with torch.enable_grad():
            state.requires_grad_(True)
            rate = certificate_transition(task, policy, state, decay=decay)[-1]
            grad, = torch.autograd.grad(rate.sum(), state)
        state = torch.minimum(upper, torch.maximum(lower,
            state.detach() + step_fraction * width * grad.sign()))
        with torch.no_grad():
            rate = certificate_transition(task, policy, state, decay=decay)[-1]
            better = rate > best_rate
            best = torch.where(better[:, None], state, best)
            best_rate = torch.maximum(best_rate, rate)
    return best.detach(), best_rate.detach()
