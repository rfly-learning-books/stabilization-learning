from __future__ import annotations

import math

import torch


def saturate(action):
    """One positive scale preserves direction; xyz norm <=1 and |yaw|<=1."""
    scale = torch.maximum(torch.linalg.vector_norm(action[..., :3], dim=-1), action[..., 3].abs())
    return action / scale.clamp_min(1.0).unsqueeze(-1)


def certificate_control(actor, current, reference, jacobian, gain=0.7, damping=0.01,
                        ridge=1e-8, max_condition=10000.0, create_graph=False):
    """J = d[(z-z*)/sqrt(m)]/d(normalized physical command), shape B,m,4.

    Includes all derivatives of L and trace normalization. Encoder weights / J
    may be frozen while gradients to the certificate network remain enabled.
    """
    if damping < 0 or ridge <= 0 or max_condition <= 1:
        raise ValueError("Require nonnegative damping, positive ridge and condition limit > 1")
    with torch.enable_grad():
        error = ((current - reference) / math.sqrt(current.shape[-1])).detach().requires_grad_(True)
        value, lower, matrix = actor.certificate(error, reference.detach())
        grad, = torch.autograd.grad(value.sum(), error, create_graph=create_graph)
        jt = jacobian.transpose(-1, -2)
        hessian = jt @ matrix @ jacobian
        diagonal = hessian.diagonal(dim1=-2, dim2=-1)
        # For B=J'MJ>=0: lambda_max(B+d*diag(B)) <= (1+d)*tr(B).
        # This sufficient ridge bound gives cond_2(H) <= max_condition in
        # exact arithmetic, even for rank-deficient J. No eigensolver is used.
        relative = (diagonal.sum(-1) * (1 + damping) / (max_condition - 1)).clamp_min(ridge)
        eye = torch.eye(4, device=current.device, dtype=current.dtype)
        regularized = hessian + damping * torch.diag_embed(diagonal) + relative[:, None, None] * eye
        control_gradient = (jt @ grad.unsqueeze(-1)).squeeze(-1)
        raw = -gain * torch.linalg.solve(regularized, control_gradient.unsqueeze(-1)).squeeze(-1)
        action = saturate(raw)
        derivative = (control_gradient * action).sum(-1)
    return {"action": action, "V": value, "L": lower, "M": matrix, "error": error,
            "gradient": grad, "Vdot": derivative, "g": control_gradient, "H": regularized}


def candidate_residuals(actor, result, reference, jacobian, dt):
    """Push Any Shape-style existence test on a LOCAL latent model, not future RGB.

    Candidates: +/- four axes at three amplitudes, plus four scales of the
    certificate-derived continuous command. All satisfy physical command bounds.
    """
    identity = torch.eye(4, device=reference.device, dtype=reference.dtype)
    axes = torch.cat([identity, -identity], 0)
    fixed = torch.cat([axes * scale for scale in (0.25, 0.5, 1.0)], 0)
    fixed = fixed[None].expand(len(reference), -1, -1)
    scales = reference.new_tensor([1.0, 0.5, 0.25, 0.125])
    derived = result["action"][:, None] * scales[None, :, None]
    actions = torch.cat([fixed, derived], 1)
    error_next = result["error"][:, None] + dt * torch.einsum("bmi,bki->bkm", jacobian, actions)
    ref = reference[:, None].expand(-1, actions.shape[1], -1)
    future, _, _ = actor.certificate(error_next, ref)
    denominator = result["V"].detach().clamp_min(1e-6)[:, None]
    flow = dt * torch.einsum("bi,bki->bk", result["g"], actions) / denominator
    endpoint = (future - result["V"][:, None]) / denominator
    return torch.maximum(flow, endpoint)


def hard_backtracking(actor, result, reference, jacobian, dt, scales, tolerance):
    """Deployment-only estimated-model gate; no unobserved future image oracle."""
    with torch.no_grad():
        candidates = result["action"].detach()[:, None] * reference.new_tensor(scales)[None, :, None]
        future_error = result["error"].detach()[:, None] + dt * torch.einsum("bmi,bki->bkm", jacobian, candidates)
        ref = reference[:, None].expand(-1, len(scales), -1)
        future, _, _ = actor.certificate(future_error, ref)
        denominator = result["V"].detach().clamp_min(1e-6)[:, None]
        flow = dt * torch.einsum("bi,bki->bk", result["g"].detach(), candidates) / denominator
        endpoint = (future - result["V"].detach()[:, None]) / denominator
        feasible = (flow < -tolerance) & (endpoint < -tolerance)
        index = feasible.to(torch.int64).argmax(-1)
        chosen = candidates[torch.arange(len(reference), device=reference.device), index]
        chosen = torch.where(feasible.any(-1)[:, None], chosen, torch.zeros_like(chosen))
    return chosen, feasible.any(-1)
