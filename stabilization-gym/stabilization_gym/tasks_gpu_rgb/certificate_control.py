"""GPU candidate controller for an explicit, goal-conditioned quadratic certificate.

The response J maps normalized commands to feature velocity. Selection enforces
the complete finite-step *model* residual; feasibility of actual transitions is
measured separately. This module owns no network or optimizer.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CertificateControlConfig:
    dt: float = .1
    decay: float = .05
    ridge: float = .02
    temperature: float = .15
    input_weight: float = .01


def value(error, matrix):
    return .5 * torch.einsum('bi,bij,bj->b', error, matrix, error)


def bounded(action):
    divisor = torch.maximum(action[..., :3].norm(dim=-1), action[..., 3].abs()).clamp_min(1.)
    return action / divisor[..., None]


def candidates(error, matrix, response, config=CertificateControlConfig(), identity=False):
    """Return differentiable scores and an explicit hard model-feasibility mask."""
    if error.ndim != 2 or response.shape != (*error.shape, 4):
        raise ValueError('Expected error B,d and response B,d,4')
    if identity:
        matrix = torch.eye(error.shape[1], device=error.device, dtype=error.dtype).expand(len(error), -1, -1)
    jt = response.transpose(-1, -2)
    gram = jt @ matrix @ response
    system = gram + config.ridge * torch.eye(4, device=error.device, dtype=error.dtype)
    nominal = bounded(-.7 * torch.linalg.solve(system, (jt @ matrix @ error[..., None])).squeeze(-1))
    basis = torch.cat((torch.eye(4, device=error.device, dtype=error.dtype),
                       -torch.eye(4, device=error.device, dtype=error.dtype)))
    scales = error.new_tensor([.05, .15, .4, .8])
    grid = (scales[:, None, None] * basis[None]).reshape(-1, 4)
    policies = torch.stack((nominal*.25, nominal*.5, nominal, bounded(nominal*1.5)), 1)
    actions = torch.cat((torch.zeros_like(nominal[:, None]), policies,
                         grid[None].expand(len(error), -1, -1)), 1)
    successor = error[:, None] + config.dt * torch.einsum('bdi,bki->bkd', response, actions)
    before = value(error, matrix)
    after = .5 * torch.einsum('bki,bij,bkj->bk', successor, matrix, successor)
    delta = after - before[:, None]
    residual = delta + config.decay * config.dt * error.square().sum(-1)[:, None]
    feasible = residual <= 0.
    has_feasible = feasible.any(-1)
    # Infeasibility is returned explicitly; recovery minimizes residual and is
    # never included in the count of certified-model actions.
    recovery = residual.argmin(-1)
    allowed = feasible.clone()
    allowed[~has_feasible, recovery[~has_feasible]] = True
    normalized_delta = delta / before.detach().clamp_min(1e-5)[:, None]
    logits = -normalized_delta / config.temperature - config.input_weight * actions.square().sum(-1)
    logits = logits.masked_fill(~allowed, -torch.inf)
    logp = torch.log_softmax(logits, -1).masked_fill(~allowed, 0.)
    probability = logp.exp().masked_fill(~allowed, 0.)
    return dict(actions=actions, probability=probability, logp=logp, matrix=matrix,
                before=before, after=after, delta=delta, residual=residual,
                feasible=feasible, has_feasible=has_feasible, allowed=allowed)


def choose(result, explore=False, generator=None):
    if explore:
        index = torch.multinomial(result['probability'], 1, generator=generator).squeeze(-1)
    else:
        index = result['probability'].argmax(-1)
    rows = torch.arange(len(index), device=index.device)
    return result['actions'][rows, index], dict(
        model_feasible=result['feasible'][rows, index],
        model_delta=result['delta'][rows, index],
        model_residual=result['residual'][rows, index],
        recovery=~result['has_feasible'], index=index)
