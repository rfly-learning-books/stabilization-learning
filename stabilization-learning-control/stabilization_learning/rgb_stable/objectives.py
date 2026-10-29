"""Task-scale losses and optimizer-independent update checks.

Physical offsets are training labels. These routines are never policy inputs.
All successor comparisons use one parameter version and the current command.
"""
import math
import torch
from torch.nn import functional as F


def component_energy(offset):
    return torch.stack((offset[..., :3].square().sum(-1) / .02**2,
                        offset[..., 3].square() / math.radians(1.)**2), -1)


def progress_risk(offset, successor, fraction=.04, interior=.5):
    """Separate arrival components; normalized progress and interior retention.

fraction is a per-transition contraction target, not a proven decay rate.
The small next-state term retains a descent gradient after the hinge is met.
"""
    rho, following = component_energy(offset), component_energy(successor)
    residual = (following-rho+fraction*F.relu(rho-interior))/(rho+1.)
    progress = F.relu(residual).square()
    hold = (rho <= 1.).to(rho.dtype)*F.relu(following-.8).square()/(rho+1.).square()
    constraints = progress+hold
    per_component = constraints+.05*following/(rho+1.)
    per_sample = per_component.amax(-1)
    tail = per_sample.topk(max(1, math.ceil(len(per_sample)/4))).values.mean()
    risk = per_sample.mean()+.25*tail
    # Guard the progress/retention constraints separately from the optimization
    # incentive. A harmless motion inside the retention region is permissible.
    return risk, constraints, residual


def priority_gradient(primary, auxiliary, cap=.2):
    """Remove a conflicting auxiliary component, then limit its norm.

This is a first-order construction only. The actual optimizer proposal must
still pass the nonlinear progress and control-change checks.
"""
    pp = torch.dot(primary, primary)
    dot = torch.dot(primary, auxiliary)
    adjusted = auxiliary - torch.minimum(dot, dot.new_zeros(()))*primary/pp.clamp_min(1e-20)
    scale = torch.minimum(pp.sqrt()*cap/adjusted.norm().clamp_min(1e-20), pp.new_ones(()))
    adjusted = adjusted*scale
    return primary+adjusted, dict(primary_norm=float(pp.sqrt()), auxiliary_norm=float(auxiliary.norm()),
        gradient_dot=float(dot), auxiliary_scale=float(scale), conflict=bool(dot < 0))


def trust_statistics(previous_u, candidate_u, previous_m, candidate_m):
    delta = (candidate_u-previous_u).norm(dim=-1)
    matrix = (candidate_m-previous_m).flatten(-2).norm(dim=-1)/previous_m.flatten(-2).norm(dim=-1).clamp_min(1e-8)
    return dict(action_change_mean=float(delta.mean()), action_change_max=float(delta.max()),
                metric_relative_change_mean=float(matrix.mean()), metric_relative_change_max=float(matrix.max()))


def group_risks(components, offsets):
    rho = component_energy(offsets).amax(-1)
    groups = (rho <= 1, (rho > 1) & (rho <= 6.25), rho > 6.25)
    # Use both position and yaw within each available phase; neither can hide
    # degradation in the other component during proposal acceptance.
    return torch.cat([components[mask].mean(0) for mask in groups if bool(mask.any())])
