"""Pure tensor certificates and committed-time signed integral state.

This extension does not load old weights or construct a physical environment.
The shared Cholesky implementation is imported read-only.  SPD is structural;
no Lyapunov guarantee is implied by a returned candidate certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import torch
from torch import nn

from stabilization_gym.common.types import MetricSpec
from stabilization_learning.policies.metric import CholeskyMetric


@dataclass(frozen=True)
class IntegralConfig:
    dt: float = .02
    length_scale: float = .1
    integral_gain: float = 1.
    leak_rate: float = .2
    integral_limit: float = 2.
    eigen_floor: float = .05
    trace: float = 6.
    hidden: int = 64
    u_max: float = .1
    speed_min_search: float = 1e-4
    train_speed_points: int = 17
    eval_speed_points: int = 65
    residual_tol: float = 1e-6
    residual_temperature: float = .05
    progress_temperature: float = .1
    stay_bias: float = 2.
    input_cost: float = .02
    angle_tol: float = math.radians(2.)
    angle_chart_limit: float = math.pi - .05
    workspace_m: float = .5

    def __post_init__(self):
        positive = (self.dt, self.length_scale, self.integral_gain,
                    self.integral_limit, self.u_max, self.speed_min_search,
                    self.residual_temperature, self.progress_temperature,
                    self.angle_tol, self.angle_chart_limit, self.workspace_m)
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError('Finite positive time, scale, bounds and temperatures required')
        if not math.isfinite(self.leak_rate) or self.leak_rate < 0:
            raise ValueError('Nonnegative finite leakage required')
        if self.speed_min_search >= self.u_max:
            raise ValueError('Speed search minimum must be below maximum')
        if any(not isinstance(x, int) or x < 2 for x in
               (self.hidden, self.train_speed_points, self.eval_speed_points)):
            raise ValueError('Network width and speed point counts must be integers >= 2')
        if not math.isfinite(self.residual_tol) or self.residual_tol < 0:
            raise ValueError('Nonnegative finite descent tolerance required')
        if not all(math.isfinite(x) and x >= 0 for x in
                   (self.stay_bias, self.input_cost)):
            raise ValueError('Finite nonnegative ranking penalties required')
        MetricSpec(6, self.eigen_floor, self.trace)
        if self.trace != 6.:
            raise ValueError('This version declares trace six')

    @property
    def signature(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def pose_coordinates(e: torch.Tensor) -> torch.Tensor:
    """Target-frame position/length_scale and continuous relative yaw -> D."""
    if e.shape[-1] != 3 or not bool(torch.isfinite(e).all()):
        raise ValueError('Finite error tensor [...,3] required')
    return torch.cat((e[..., :2], math.sqrt(2.) * torch.sin(e[..., 2:3] / 2)), -1)


def _mask(task_mask, reference):
    value = torch.as_tensor(task_mask, dtype=reference.dtype, device=reference.device)
    value = torch.broadcast_to(value, reference.shape)
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError('Task mask entries must be zero or one')
    if not bool(value[..., :2].bool().all()):
        raise ValueError('This version always controls both position coordinates')
    return value


def _blocked(blocked, reference):
    if blocked is None:
        return torch.zeros_like(reference, dtype=torch.bool)
    value = torch.as_tensor(blocked, dtype=torch.bool, device=reference.device)
    return torch.broadcast_to(value, reference.shape)


def _goal_hold(goal_hold, reference):
    if goal_hold is None:
        return torch.zeros(reference.shape[:-1], dtype=torch.bool, device=reference.device)
    value = torch.as_tensor(goal_hold, dtype=torch.bool, device=reference.device)
    return torch.broadcast_to(value, reference.shape[:-1])


def _duration(actual_dt, reference):
    dt = torch.as_tensor(actual_dt, dtype=reference.dtype, device=reference.device)
    dt = torch.broadcast_to(dt, reference.shape[:-1])
    if not bool(torch.isfinite(dt).all()) or bool((dt < 0).any()):
        raise ValueError('Actual accepted duration must be finite and nonnegative')
    return dt


def masked_error(e, eta, task_mask):
    """Build E=[masked D,masked eta]; physical yaw may remain in e."""
    if e.shape != eta.shape or e.shape[-1] != 3:
        raise ValueError('Matching error and eta tensors [...,3] required')
    mask = _mask(task_mask, e)
    return torch.cat((pose_coordinates(e) * mask, eta * mask), -1)


def integral_derivative(D, eta, *, task_mask, blocked=None, goal_hold=None,
                        cfg=None):
    """Projected vector field; blocked/hold removes error drive, retains leak.

    A zero command is never implicitly interpreted as actuator blocking.
    Projection at a hard integral bound is piecewise differentiable.
    """
    cfg = cfg or IntegralConfig()
    if D.shape != eta.shape or D.shape[-1] != 3:
        raise ValueError('Matching D and eta tensors [...,3] required')
    if not bool(torch.isfinite(D).all()) or not bool(torch.isfinite(eta).all()):
        raise ValueError('Finite integral state and error required')
    mask = _mask(task_mask, D)
    drive_enabled = ~_blocked(blocked, D) & ~_goal_hold(goal_hold, D)[..., None]
    drive = cfg.integral_gain * D * drive_enabled * mask
    derivative = drive - cfg.leak_rate * eta * mask
    outward = ((eta >= cfg.integral_limit) & (derivative > 0)) | (
        (eta <= -cfg.integral_limit) & (derivative < 0))
    return torch.where(outward, torch.zeros_like(derivative), derivative) * mask


def integrate_eta(eta, D_before, D_after, actual_dt, *, task_mask,
                  blocked=None, goal_hold=None, cfg=None):
    """Pure accepted-time update with a leaky trapezoidal error-drive rule.

    The exponential leakage is exact. The average endpoint input approximates
    a changing D; it is not an exact integral for arbitrary physical motion.
    Hard clipping provides bounded antiwindup. Explicit blocked coordinates
    and goal_hold suppress the error input while preserving exponential leak.
    """
    cfg = cfg or IntegralConfig()
    if eta.shape != D_before.shape or eta.shape != D_after.shape or eta.shape[-1] != 3:
        raise ValueError('Matching eta and endpoint D tensors [...,3] required')
    if not all(bool(torch.isfinite(x).all()) for x in (eta, D_before, D_after)):
        raise ValueError('Integral inputs must be finite')
    mask = _mask(task_mask, eta)
    dt = _duration(actual_dt, eta)[..., None]
    a = torch.exp(-cfg.leak_rate * dt)
    b = (cfg.integral_gain * dt if cfg.leak_rate == 0 else
         cfg.integral_gain * (-torch.expm1(-cfg.leak_rate * dt)) / cfg.leak_rate)
    enabled = ~_blocked(blocked, eta) & ~_goal_hold(goal_hold, eta)[..., None]
    drive = (D_before + D_after) * .5 * enabled * mask
    updated = (a * eta + b * drive).clamp(-cfg.integral_limit, cfg.integral_limit)
    # A rejected / zero-duration step cannot commit a hidden state change.
    return torch.where(dt > 0, updated, eta) * mask


@dataclass(frozen=True)
class IntegralState:
    eta: torch.Tensor
    accepted_time_s: torch.Tensor
    accepted_steps: torch.Tensor
    blocked_axes: torch.Tensor
    config_signature: str

    @classmethod
    def zeros(cls, batch_size, *, cfg=None, device='cpu', dtype=torch.float64):
        cfg = cfg or IntegralConfig()
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Positive integer batch size required')
        eta = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        return cls(eta, torch.zeros(batch_size, device=device, dtype=dtype),
                   torch.zeros(batch_size, device=device, dtype=torch.int64),
                   torch.zeros_like(eta, dtype=torch.bool), cfg.signature)

    def advance(self, D_before, D_after, actual_dt, *, task_mask,
                blocked=None, goal_hold=None, active=None, cfg=None):
        """Return a new state; only the environment's committed step calls it."""
        cfg = cfg or IntegralConfig()
        if self.config_signature != cfg.signature:
            raise ValueError('Integral configuration changed; reset or version explicitly')
        dt = _duration(actual_dt, self.eta)
        if active is not None:
            enabled = torch.broadcast_to(torch.as_tensor(active, dtype=torch.bool,
                                         device=dt.device), dt.shape)
            dt = dt * enabled
        eta = integrate_eta(self.eta, D_before, D_after, dt, task_mask=task_mask,
                            blocked=blocked, goal_hold=goal_hold, cfg=cfg)
        return IntegralState(eta, self.accepted_time_s + dt,
                             self.accepted_steps + (dt > 0).to(torch.int64),
                             _blocked(blocked, self.eta).clone(), cfg.signature)

    def snapshot(self) -> dict[str, Any]:
        return dict(schema='push_signed_integral_state_v1',
                    config_signature=self.config_signature,
                    eta=self.eta.detach().cpu().tolist(),
                    accepted_time_s=self.accepted_time_s.detach().cpu().tolist(),
                    accepted_steps=self.accepted_steps.detach().cpu().tolist(),
                    blocked_axes=self.blocked_axes.detach().cpu().tolist())

    @classmethod
    def from_snapshot(cls, payload, *, cfg=None, device='cpu', dtype=torch.float64):
        cfg = cfg or IntegralConfig()
        if payload.get('schema') != 'push_signed_integral_state_v1' or payload.get(
                'config_signature') != cfg.signature:
            raise ValueError('Integral snapshot schema/configuration mismatch')
        eta = torch.as_tensor(payload['eta'], device=device, dtype=dtype).clone()
        times = torch.as_tensor(payload['accepted_time_s'], device=device, dtype=dtype).clone()
        raw_steps = payload['accepted_steps']
        if (not isinstance(raw_steps, list) or
                any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                    for x in raw_steps)):
            raise ValueError('Snapshot accepted steps must be nonnegative integer counts')
        steps = torch.as_tensor(raw_steps, device=device, dtype=torch.int64).clone()
        blocked = torch.as_tensor(payload['blocked_axes'], device=device, dtype=torch.bool).clone()
        if eta.ndim != 2 or eta.shape[-1] != 3 or blocked.shape != eta.shape:
            raise ValueError('Malformed integral snapshot dimensions')
        if times.shape != eta.shape[:-1] or steps.shape != times.shape:
            raise ValueError('Malformed integral snapshot counters')
        if not bool(torch.isfinite(eta).all()) or bool((eta.abs() > cfg.integral_limit).any()):
            raise ValueError('Snapshot integral outside its declared finite bounds')
        if not bool(torch.isfinite(times).all()) or bool((times < 0).any()) or bool((steps < 0).any()):
            raise ValueError('Snapshot contains invalid accepted-time counters')
        return cls(eta, times, steps, blocked, cfg.signature)


class ModeEncoder(nn.Module):
    """Original shared-per-mode mean/max conditioning, copied into extension."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())

    def forward(self, C, valid, ell=.1):
        if C.ndim != 3 or C.shape[-1] != 3 or valid.shape != C.shape[:2]:
            raise ValueError('C[B,N,3] and valid[B,N] required')
        if valid.dtype != torch.bool or not bool(valid.any(-1).all()):
            raise ValueError('Nonempty boolean valid-mode rows required')
        if not bool(torch.isfinite(C).all()):
            raise ValueError('Finite shape responses required')
        values = torch.where(valid[..., None], C * C.new_tensor([1., 1., ell]), 0.)
        H = self.net(values)
        return torch.cat(((H * valid[..., None]).sum(1) / valid.sum(1, keepdim=True),
                          H.masked_fill(~valid[..., None], -torch.inf).amax(1)), -1)


class IntegralMActor(nn.Module):
    """E,C -> 21 public-tril raw parameters -> SPD six-dimensional M.

    Exact identity initialization is deliberate. Full state derivatives are
    computed by certificate_terms; never replace them with 2 M E after learning.
    """
    schema = 'push_integral_actor_6d_public_tril_v1'

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or IntegralConfig()
        h = self.cfg.hidden
        self.encoder = ModeEncoder(h)
        self.state = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(3 * h, 2 * h), nn.SiLU(),
                                  nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 21))
        self.metric = CholeskyMetric(MetricSpec(6, self.cfg.eigen_floor, self.cfg.trace))
        self.register_buffer('identity_center', self.metric.identity_raw())
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def conditioned(self, E, condition):
        if E.shape[-1] != 6 or not bool(torch.isfinite(E).all()):
            raise ValueError('Finite six-dimensional augmented error required')
        while condition.ndim < E.ndim:
            condition = condition.unsqueeze(-2)
        condition = condition.expand(*E.shape[:-1], condition.shape[-1])
        raw = self.identity_center + self.head(torch.cat((self.state(E), condition), -1))
        M = self.metric(raw)
        return torch.linalg.cholesky(M), M

    def forward(self, E, C, valid):
        return self.conditioned(E, self.encoder(C, valid, self.cfg.length_scale))

    def value(self, E, C, valid):
        _, M = self(E, C, valid)
        return torch.einsum('...i,...ij,...j->...', E, M, E)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if ('state.0.weight' in state_dict and state_dict['state.0.weight'].shape[-1] != 6
                or 'head.4.weight' in state_dict and state_dict['head.4.weight'].shape[0] != 21):
            raise ValueError('Old three-dimensional Push weights cannot load into a six-dimensional actor')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class LegacyPoseCertificate(nn.Module):
    """Read-only explicit bridge: diag(M_old(masked D,C), I3), trace six.

    This is a newly augmented controller with legacy pose weights. It is not an
    original-protocol replay. The old actor object must already be independently
    hash-verified by the caller. Its input-dependent M derivative is retained.
    """
    schema = 'push_legacy_pose_blockdiag_integral_v1'

    def __init__(self, old_actor, cfg=None):
        super().__init__()
        self.cfg = cfg or IntegralConfig()
        if not hasattr(old_actor, 'conditioned') or not hasattr(old_actor, 'encoder'):
            raise TypeError('Verified original FullMActor interface required')
        old_cfg = getattr(old_actor, 'cfg', None)
        if old_cfg is None or old_cfg.length_scale != self.cfg.length_scale:
            raise ValueError('Legacy actor length scale must match the extension')
        # Independent immutable weight copy: do not freeze or change the mode of
        # the caller's original actor. Input-D derivatives remain available.
        self.old_actor = deepcopy(old_actor)
        self.old_actor.requires_grad_(False)
        self.old_actor.eval()

    @property
    def encoder(self):
        return self.old_actor.encoder

    def conditioned(self, E, condition):
        if E.shape[-1] != 6:
            raise ValueError('Six-dimensional masked error required')
        _, old_M = self.old_actor.conditioned(E[..., :3], condition)
        zero = torch.zeros_like(old_M)
        eye = torch.eye(3, device=E.device, dtype=E.dtype).expand_as(old_M)
        M = torch.cat((torch.cat((old_M, zero), -1), torch.cat((zero, eye), -1)), -2)
        return torch.linalg.cholesky(M), M

    def forward(self, E, C, valid):
        return self.conditioned(E, self.encoder(C, valid, self.cfg.length_scale))

    def value(self, E, C, valid):
        _, M = self(E, C, valid)
        return torch.einsum('...i,...ij,...j->...', E, M, E)


def certificate_terms(actor, E, C, valid, *, task_mask, training=False):
    """Return complete dV/dE with masked axes identically absent from the graph."""
    if E.shape[-1] != 6:
        raise ValueError('Augmented error dimension is six')
    mask = _mask(task_mask, E[..., :3])
    mask6 = torch.cat((mask, mask), -1)
    with torch.enable_grad():
        base = E if E.requires_grad else E.clone().requires_grad_(True)
        selected = base * mask6
        L, M = actor(selected, C, valid)
        V = torch.einsum('...i,...ij,...j->...', selected, M, selected)
        gradient, = torch.autograd.grad(V.sum(), base, create_graph=training,
                                        retain_graph=training)
    return dict(E=selected, L=L, M=M, V=V, gradient=gradient * mask6,
                claim='candidate_only', task_mask=mask)
