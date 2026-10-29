"""Small one-step differentiable objective for the DC motor M actor."""
from __future__ import annotations
import torch

from .certificate import quadratic


def dc_motor_one_step_loss(actor, env, *, seeds, decay: float = .05,
                           descent_weight: float = 1.0,
                           performance_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    e_list, next_e_list, v0_list, v1_list = [], [], [], []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        context = env.metric_context(observation)
        M = actor(context)
        error, control_context = env.build_error(observation)
        action = env.controller_action(M, error, control_context)
        following = env.torch_step(torch.as_tensor(observation, dtype=torch.float64),
                                   action.to(torch.float64))
        next_error = env.tensor_error(following)
        v0 = quadratic(error, M.to(error))
        v1 = quadratic(next_error, M.to(next_error))
        e_list.append(error.square().sum())
        next_e_list.append(next_error.square().sum())
        v0_list.append(v0)
        v1_list.append(v1)
    performance = torch.stack(next_e_list).mean()
    v0 = torch.stack(v0_list)
    v1 = torch.stack(v1_list)
    residual = torch.relu(v1 - v0 + decay * env.dt * torch.stack(e_list)).mean()
    loss = performance_weight * performance + descent_weight * residual
    metrics = {
        "loss": float(loss.detach()),
        "performance": float(performance.detach()),
        "descent_residual": float(residual.detach()),
        "V_rise_count": int((v1 > v0 + 1e-8).sum().item()),
    }
    return loss, metrics
