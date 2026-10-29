"""RGB task formulas: metric parameters -> certificate -> legacy velocity.

The critic's eta is a 136-vector of raw Cholesky entries. It is distinct from
the scalar numerical ridge in the four-dimensional velocity solve.
"""
import torch
from torch.nn import functional as F


def metric_from_eta(eta, dim=16, floor=.1):
    if eta.shape[-1] != dim*(dim+1)//2:
        raise ValueError('eta must contain raw lower-triangular metric parameters')
    rows, cols = torch.tril_indices(dim, dim, device=eta.device)
    entries = torch.where(rows == cols, F.softplus(eta)+1e-3, eta)
    lower = eta.new_zeros(*eta.shape[:-1], dim, dim)
    lower[..., rows, cols] = entries
    gram = lower @ lower.transpose(-1, -2)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    matrix = floor*torch.eye(dim, device=eta.device, dtype=eta.dtype) + dim*(1-floor)*gram/trace
    return lower, matrix


def value(error, matrix):
    return .5*torch.einsum('...i,...ij,...j->...', error, matrix, error)


def command(error, interaction, matrix):
    """Unchanged goal_fixed_core.policy, including joint radial saturation."""
    jt = interaction.transpose(-1, -2)
    gram = jt @ matrix @ interaction
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    ridge = torch.maximum(trace*1.01/(1e4-1), trace.new_full(trace.shape, 1e-8))
    h = gram + .01*torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1))
    h = h + ridge[..., None, None]*torch.eye(4, device=h.device, dtype=h.dtype)
    g = (jt @ matrix @ error.unsqueeze(-1)).squeeze(-1)
    raw = -.7*torch.linalg.solve(h, g.unsqueeze(-1)).squeeze(-1)
    divisor = torch.maximum(torch.ones_like(raw[..., 0]), raw[..., :3].norm(dim=-1))
    divisor = torch.maximum(divisor, raw[..., 3].abs())
    return raw/divisor[..., None], h


def p0_step(offset, action, dt=.1):
    yaw = offset[..., 3]+.5*.35*action[..., 3]*dt
    c, s = yaw.cos(), yaw.sin()
    return offset + torch.stack((.4*dt*(c*action[..., 0]-s*action[..., 1]),
        .4*dt*(s*action[..., 0]+c*action[..., 1]), .4*dt*action[..., 2], .35*dt*action[..., 3]), -1)
