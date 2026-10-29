"""Original Push mode/speed selection with an augmented integral certificate.

Only one calibrated/declared response-model step is predicted.  No contact
planner or long-horizon cost optimizer is introduced here. All functions are
pure tensor queries: integral state is committed only by the execution owner.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .integral_certificate import (IntegralConfig, certificate_terms,
                                   integrate_eta, integral_derivative,
                                   masked_error, pose_coordinates)


def torch_step(e, c, u, h, ell=.1):
    """Copied original body-response integration; e retains physical yaw."""
    s = u * h
    a = c[..., 2] * s
    heading = e[..., 2] + a / 2
    amp = s * torch.sinc(a / (2 * math.pi)) / ell
    return e + torch.stack((amp * (torch.cos(heading) * c[..., 0] -
                                  torch.sin(heading) * c[..., 1]),
                             amp * (torch.sin(heading) * c[..., 0] +
                                    torch.cos(heading) * c[..., 1]), a), -1)


def tensor_candidates(actor, e, eta, C, valid, cfg=None, *, task_mask,
                      current_mode, physical_mask=None, training=False,
                      speed_points=None, goal_hold=None, blocked_axes=None,
                      speed_upper=None):
    """Evaluate complete E-dependent flow and one-step endpoint candidates.

    C is [B,N,3] in object-body coordinates. task_mask never removes physical
    yaw from body-response rotation. goal_hold and blocked_axes are supplied
    facts, not mutable caches or deductions from a zero previous command.
    physical_mask may be a pure callback(e_next)->bool[B,N*K].
    """
    cfg = cfg or IntegralConfig()
    if e.ndim != 2 or e.shape[-1] != 3 or eta.shape != e.shape:
        raise ValueError('e and eta must be matching [B,3] tensors')
    B, N = C.shape[:2]
    if B != len(e) or C.ndim != 3 or C.shape[-1] != 3 or valid.shape != (B, N):
        raise ValueError('Response/valid arrays must be [B,N,3]/[B,N]')
    if valid.dtype != torch.bool or not bool(valid.any(-1).all()):
        raise ValueError('Nonempty boolean valid modes required')
    K = cfg.train_speed_points if training else cfg.eval_speed_points
    K = K if speed_points is None else speed_points
    if not isinstance(K, int) or K < 2:
        raise ValueError('At least two speed candidates required')
    upper = torch.as_tensor(cfg.u_max if speed_upper is None else speed_upper,
                            device=e.device, dtype=e.dtype)
    upper = torch.broadcast_to(upper, (B, N)).clamp_max(cfg.u_max)
    if not bool(torch.isfinite(upper).all()) or bool((upper <= cfg.speed_min_search).any()):
        raise ValueError('Finite per-mode speed upper bounds must exceed search minimum')
    fraction = torch.linspace(0., 1., K, device=e.device, dtype=e.dtype)
    speeds = torch.exp(math.log(cfg.speed_min_search) + fraction[None, None, :] *
                       (upper.log()[..., None] - math.log(cfg.speed_min_search)))
    E = masked_error(e, eta, task_mask)
    cert = certificate_terms(actor, E, C, valid, task_mask=task_mask, training=training)
    mask = cert['task_mask']
    D = pose_coordinates(e) * mask
    a = e[:, 2, None]
    D_flow = torch.stack(((torch.cos(a) * C[..., 0] - torch.sin(a) * C[..., 1]) /
                          cfg.length_scale,
                          (torch.sin(a) * C[..., 0] + torch.cos(a) * C[..., 1]) /
                          cfg.length_scale,
                          torch.cos(a / 2) * C[..., 2] / math.sqrt(2.)), -1)
    D_flow = D_flow * mask[:, None, :]
    eta_flow = integral_derivative(D, eta, task_mask=mask, blocked=blocked_axes,
                                   goal_hold=goal_hold, cfg=cfg)
    pose_dot = D_flow[:, :, None, :] * speeds[..., None]
    eta_dot = eta_flow[:, None, None, :].expand(-1, N, K, -1)
    E_dot = torch.cat((pose_dot, eta_dot), -1)
    flow = cfg.dt * torch.einsum('bi,bnki->bnk', cert['gradient'], E_dot)
    ep = torch_step(e[:, None, None, :], C[:, :, None, :], speeds,
                    cfg.dt, cfg.length_scale).reshape(B, N * K, 3)
    Dp = pose_coordinates(ep) * mask[:, None, :]
    blocked = None if blocked_axes is None else torch.broadcast_to(torch.as_tensor(
        blocked_axes, dtype=torch.bool, device=e.device), (B, 3))[:, None, :]
    hold = None if goal_hold is None else torch.broadcast_to(torch.as_tensor(
        goal_hold, dtype=torch.bool, device=e.device), (B,))[:, None]
    etap = integrate_eta(eta[:, None, :].expand(-1, N * K, -1),
                          D[:, None, :].expand(-1, N * K, -1), Dp, cfg.dt,
                          task_mask=mask[:, None, :], blocked=blocked,
                          goal_hold=hold, cfg=cfg)
    Ep = torch.cat((Dp, etap), -1)
    # Re-encode once inside this query, never cache conditions across updates.
    condition = actor.encoder(C, valid, cfg.length_scale)
    _, Mp = actor.conditioned(Ep, condition)
    Vp = torch.einsum('bni,bnij,bnj->bn', Ep, Mp, Ep)
    norm = (cert['E'].detach().square().sum(-1) + 1e-6)[:, None]
    flow = flow.reshape(B, -1) / norm
    endpoint = (Vp - cert['V'][:, None]) / norm
    both = torch.maximum(flow, endpoint)
    physical = valid[:, :, None].expand(-1, -1, K).reshape(B, -1).clone()
    if physical_mask is not None:
        extra = physical_mask(ep) if callable(physical_mask) else physical_mask
        extra = torch.as_tensor(extra, device=e.device, dtype=torch.bool)
        if extra.shape != physical.shape:
            raise ValueError('Physical candidate mask has the wrong shape')
        physical = physical & extra
    feasible = physical & (flow < -cfg.residual_tol) & (endpoint < -cfg.residual_tol)
    modes = torch.arange(N, device=e.device).repeat_interleave(K)
    command = speeds.reshape(B, -1)
    mode = torch.as_tensor(current_mode, device=e.device, dtype=torch.int64)
    mode = torch.broadcast_to(mode, (B,))
    cost = cfg.input_cost * (command / cfg.u_max).square()
    logits = (F.logsigmoid(-both / cfg.residual_temperature) -
              endpoint / cfg.progress_temperature - cost -
              cfg.stay_bias * (modes[None, :] != mode[:, None]))
    logits = logits.masked_fill(~physical, -1e9)
    return dict(**cert, V_next=Vp, M_next=Mp, E_next=Ep, eta_next=etap,
                e_next=ep, flow=flow, endpoint=endpoint, both=both,
                physical=physical, feasible=feasible, logits=logits,
                modes=modes, speeds=command, condition=condition,
                beta=torch.einsum('bi,bni->bn', cert['gradient'][..., :3], D_flow),
                eta_flow=eta_flow, dt=cfg.dt, goal_hold=goal_hold,
                blocked_axes=blocked_axes)


def candidate_geometry(e_next, target, ids, bank, cfg, task_mask):
    """Original endpoint polygon/workspace test, free-yaw chart handled by mask."""
    a = target[:, 2:3]
    x, y = e_next[..., 0] * cfg.length_scale, e_next[..., 1] * cfg.length_scale
    px = target[:, 0:1] + torch.cos(a) * x - torch.sin(a) * y
    py = target[:, 1:2] + torch.sin(a) * x + torch.cos(a) * y
    yaw = a + e_next[..., 2]
    v = bank.vertices[ids]
    mask = bank.vertex_mask[ids][:, None, :]
    wx = px[..., None] + torch.cos(yaw)[..., None] * v[:, None, :, 0] - torch.sin(yaw)[..., None] * v[:, None, :, 1]
    wy = py[..., None] + torch.sin(yaw)[..., None] * v[:, None, :, 0] + torch.cos(yaw)[..., None] * v[:, None, :, 1]
    inside = ((wx.masked_fill(~mask, torch.inf).amin(-1) >= 0) &
              (wy.masked_fill(~mask, torch.inf).amin(-1) >= 0) &
              (wx.masked_fill(~mask, -torch.inf).amax(-1) <= cfg.workspace_m) &
              (wy.masked_fill(~mask, -torch.inf).amax(-1) <= cfg.workspace_m))
    yaw_controlled = torch.as_tensor(task_mask, device=e_next.device)[..., 2].bool()
    return inside & (~yaw_controlled[:, None] | (e_next[..., 2].abs() < cfg.angle_chart_limit))


def candidates(actor, data, bank, cfg=None, training=False, speed_points=None):
    """Original-shaped dictionary/bank adapter; tensor_candidates is standalone."""
    cfg = cfg or IntegralConfig()
    ids = data['shape']
    physical = None
    if hasattr(bank, 'vertices') and hasattr(bank, 'vertex_mask'):
        physical = lambda ep: candidate_geometry(ep, data['target'], ids, bank, cfg,
                                                  data['task_mask'])
    return tensor_candidates(actor, data['e'], data['eta'], bank.C[ids], bank.valid[ids], cfg,
                              task_mask=data['task_mask'], current_mode=data['mode'],
                              physical_mask=physical, training=training,
                              speed_points=speed_points, goal_hold=data.get('goal_hold'),
                              blocked_axes=data.get('blocked_axes'),
                              speed_upper=data.get('speed_upper'))


def hard_choice(cand, data, cfg=None, recovery=None, at_goal=None):
    """Original mode persistence plus explicitly separate pose-yaw recovery.

    Disk mask disables yaw recovery. Failure/goal command is zero. No zero
    candidate is accepted as evidence of task progress or convergence.
    """
    cfg = cfg or IntegralConfig()
    e = data['e']
    B = len(e)
    at_goal = torch.zeros(B, dtype=torch.bool, device=e.device) if at_goal is None else at_goal
    recovery = torch.zeros_like(at_goal) if recovery is None else recovery
    at_goal = torch.as_tensor(at_goal, device=e.device, dtype=torch.bool)
    recovery = torch.as_tensor(recovery, device=e.device, dtype=torch.bool)
    yaw_controlled = cand['task_mask'][:, 2].bool()
    feasible = cand['feasible']
    has = feasible.any(-1)
    angle = e[:, 2]
    recovery = recovery & yaw_controlled & (angle.abs() > cfg.angle_tol)
    continuing = recovery.clone()
    recovery = recovery | (~has & yaw_controlled & (angle.abs() > cfg.angle_tol) & ~at_goal)
    modes, us = cand['modes'], cand['speeds']
    current = modes[None, :] == data['mode'][:, None]
    pool = feasible & torch.where((feasible & current).any(-1)[:, None],
                                  current, torch.ones_like(current))
    normal_cost = (cand['V_next'] / (cand['V'].detach()[:, None] + 1e-6) +
                   cfg.input_cost * (us / cfg.u_max).square())
    normal_index = normal_cost.masked_fill(~pool, torch.inf).argmin(-1)
    next_angle = cand['e_next'][..., 2]
    omega = (next_angle - angle[:, None]) / cfg.dt
    recover_pool = (cand['physical'] & (angle[:, None] * omega < 0) &
                    (next_angle.abs() < angle.abs()[:, None] - 1e-8))
    recover_pool = recover_pool & ((next_angle * angle[:, None] >= 0) |
                                    (next_angle.abs() <= cfg.angle_tol))
    keep = continuing & (recover_pool & current).any(-1)
    recover_pool = recover_pool & torch.where(keep[:, None], current,
                                              torch.ones_like(current))
    score = -angle.sign()[:, None] * omega
    recover_index = score.masked_fill(~recover_pool, -torch.inf).argmax(-1)
    index = torch.where(recovery, recover_index, normal_index)
    fail = torch.where(recovery, ~recover_pool.any(-1), ~has) & ~at_goal
    command = us.gather(1, index[:, None]).squeeze(1)
    command = torch.where(at_goal | fail, torch.zeros_like(command), command)
    return modes[index], command, recovery & ~at_goal, fail
