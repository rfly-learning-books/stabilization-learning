"""Adapter for the preserved, calibrated Push Any Shape numeric plant.

This is a reduced-order shape/mode model, not a contact simulator. Hard mode
selection is deliberately non-differentiable; the candidate values are not.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ...common.types import MetricSpec
from ...common.action_spaces import ModeSpeedSpace
from ...common.paths import asset_path


WORKSPACE = asset_path("legacy_workspace")
DEFAULT_SOURCE = asset_path("push_source")
DEFAULT_LIBRARY = asset_path("push_library")


def _load_source(source_root):
    root = Path(source_root).resolve()
    if not (root / "dl_push/certificate/numeric_env.py").is_file():
        raise FileNotFoundError(f"Push Any Shape source missing: {root}. Supply source_root.")
    previous = sys.modules.get("dl_push")
    if previous is not None and Path(previous.__file__).resolve().parent != root / "dl_push":
        raise RuntimeError("A different dl_push source is already loaded in this process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return (importlib.import_module("dl_push.certificate.numeric_env"),
            importlib.import_module("dl_push.certificate.full_m_control"))


class PushAnyShapeEnv(gym.Env):
    """Unified metric interface over 32 calibrated shapes and 12 push modes.

    ``step`` accepts ``[mode, speed]`` or the native ``{'mode', 'u'}``.
    The learner's M stays fixed for an episode. This is a new fixed-M baseline,
    not the original state-dependent M(D,C) PPO controller. Its hard controller
    preserves the original candidate feasibility, mode persistence and attitude
    recovery rules, and exposes the recovery flag instead of hiding it.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    error_dim = 3
    metric_context_dim = 39
    controller_capabilities = {
        "action_gradient": False, "model_gradient": True,
        "episode_fixed_metric": True, "state_dependent_certificate": False,
        "cost_convention": "positive_integral",
        "differentiable_action": False,
        "differentiable_candidate_scores": True,
        "differentiable_model": True,
        "representation": "analytic_target_frame",
        "metric_scope": "episode",
        "plant": "calibrated_reduced_order_exact_mode_flow",
        "certificate_guarantee": False,
    }

    def __init__(self, shape_name="T", *, source_root=None, library_root=None,
                 max_steps=100, speed_points=65, render_mode="rgb_array", **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected Push options: {sorted(kwargs)}")
        numeric, self._control = _load_source(source_root or DEFAULT_SOURCE)
        root = Path(library_root or DEFAULT_LIBRARY).resolve()
        if not (root / shape_name / "report.json").exists():
            raise FileNotFoundError(f"Calibrated Push library missing for {shape_name}: {root}")
        self.bank = self._control.ModeBank(root)
        self.cfg = self._control.FullMConfig(max_steps=int(max_steps))
        self.metric_spec = MetricSpec(3, eigen_floor=.05, trace=3.)
        self.speed_points = int(speed_points)
        if self.speed_points < 2:
            raise ValueError("speed_points must be at least 2")
        self.dt = self.cfg.dt
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.plant = numeric.NumericPushEnv(shape_name, root, numeric.NumericConfig(
            max_steps=max_steps, success_mode="none", terminate_on_outside=True))
        self.observation_space = self.plant.observation_space
        self.action_space = ModeSpeedSpace(12, .6)
        self.asset_manifest = {"source_root": str(source_root or DEFAULT_SOURCE),
                               "library_root": str(root),
                               "library_fingerprints": self.bank.fingerprints}
        self._obs = None
        self._hold = 0
        self._recovery = False
        self._done = True
        self._diagnostics = {}
        self._pending_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._obs, info = self.plant.reset(seed=seed, options=options)
        self._hold, self._recovery, self._done = 0, False, False
        self._diagnostics = {}
        self._pending_action = None
        return self._obs, self._info(info)

    def metric_context(self, observation=None, goal=None):
        obs = self._obs if observation is None else observation
        if obs is None:
            raise RuntimeError("reset must precede metric_context")
        target = obs["target"] if goal is None else goal
        # C and target remain fixed during one episode; pose is excluded.
        return torch.cat([torch.as_tensor(obs["C"], dtype=torch.float32).flatten(),
                          torch.as_tensor(target, dtype=torch.float32)])

    def build_error(self, observation, goal=None, *, representation=None, history=None):
        if representation is not None:
            raise ValueError("Push error uses calibrated geometry; no learned encoder is required")
        raw = torch.as_tensor(observation["error"], dtype=torch.float32)
        if goal is not None:
            q = torch.as_tensor(observation["pose"], dtype=torch.float32)
            target = torch.as_tensor(goal, dtype=torch.float32)
            delta = q[:2] - target[:2]
            c, s = target[2].cos(), target[2].sin()
            raw = torch.stack([(c * delta[0] + s * delta[1]) / .1,
                               (-s * delta[0] + c * delta[1]) / .1, q[2] - target[2]])
        return self._control.coordinates(raw), {
            "raw_error": raw, "target": torch.as_tensor(observation["target"] if goal is None else goal),
            "shape": int(observation["shape_id"]), "mode": int(observation["current_mode"]) - 1,
            "recovery": self._recovery, "coverage": float(self.plant.vector.coverage[0]),
        }

    def candidate_scores(self, M, e, *, context):
        """Real candidate endpoint/flow values with gradients to M and e."""
        M = torch.as_tensor(M)
        e = torch.as_tensor(e, device=M.device, dtype=M.dtype)
        raw = context["raw_error"].to(e)
        speeds = torch.logspace(math.log10(self.cfg.speed_min_search), math.log10(.6),
                               self.speed_points, device=e.device, dtype=e.dtype)
        sid = int(context["shape"])
        C, valid = self.bank.C[sid].to(e), self.bank.valid[sid].to(e.device)
        ep = self._control.torch_step(raw[None, None], C[:, None], speeds[None], self.dt)
        ep = ep.reshape(-1, 3)
        en = self._control.coordinates(ep)
        V = torch.einsum("i,ij,j->", e, M, e)
        Vn = torch.einsum("ni,ij,nj->n", en, M, en)
        a = raw[2]
        b = torch.stack([(a.cos()*C[:, 0] - a.sin()*C[:, 1])/.1,
                         (a.sin()*C[:, 0] + a.cos()*C[:, 1])/.1,
                         (a/2).cos()*C[:, 2]/math.sqrt(2)], -1)
        beta = b @ ((M + M.T) @ e)
        norm = e.detach().square().sum() + 1e-6
        flow = (self.dt * beta[:, None] * speeds).flatten() / norm
        endpoint = (Vn - V) / norm
        # The immutable geometric bank is CPU based; transfer only for geometry.
        if e.device.type != "cpu":
            physical = self._control.candidate_geometry(ep[None].cpu(),
                context["target"].float()[None].cpu(), torch.tensor([sid]), self.bank, self.cfg)[0].to(e.device)
        else:
            physical = self._control.candidate_geometry(ep[None],
                context["target"].to(e)[None], torch.tensor([sid]), self.bank, self.cfg)[0]
        physical = physical & valid.repeat_interleave(self.speed_points)
        feasible = physical & (flow < -self.cfg.residual_tol) & (endpoint < -self.cfg.residual_tol)
        return {"V": V[None], "V_next": Vn[None], "e_next": ep[None],
                "flow": flow[None], "endpoint": endpoint[None], "physical": physical[None],
                "feasible": feasible[None], "modes": torch.arange(12, device=e.device).repeat_interleave(self.speed_points),
                "speeds": speeds.repeat(12)}

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        if certificate is not None:
            raise ValueError("This adapter evaluates episode-fixed M; state-dependent certificates need their full derivative")
        cand = self.candidate_scores(M, e, context=context)
        raw = context["raw_error"].to(e)
        goal = bool(raw[:2].norm()*.1 <= .005 and raw[2].abs() <= math.radians(2)
                    and context["coverage"] >= .95)
        data = {"e": raw[None], "mode": torch.tensor([context["mode"]], device=e.device)}
        mode, speed, recovery, fail = self._control.hard_choice(cand, data, self.cfg,
            torch.tensor([context["recovery"]], device=e.device), torch.tensor([goal], device=e.device))
        self._recovery = bool(recovery[0])
        self._diagnostics = {"recovery": self._recovery, "no_feasible_action": bool(fail[0]),
                             "candidate_feasible_count": int(cand["feasible"].sum())}
        action = torch.stack([mode[0].to(speed.dtype), speed[0]]).detach()
        self._pending_action = action.cpu().numpy().copy()
        return action

    def physical_cost(self, observation=None, action=None, next_observation=None):
        obs = next_observation if next_observation is not None else observation
        obs = self._obs if obs is None else obs
        raw = np.asarray(obs["error"], dtype=float)
        return float(raw[:2] @ raw[:2] + 1 - np.cos(raw[2]))

    def _info(self, info):
        return {**info, **self._diagnostics, "physical_error": self.physical_cost(),
                "success": self._hold >= 5, "is_success": self._hold >= 5,
                "failure": bool(info.get("outside_workspace", False)), "cost_convention": "positive_integral",
                "plant_fidelity": "calibrated numeric mode flow; no contact or pusher dynamics"}

    def step(self, physical_action):
        if self._done:
            raise RuntimeError("reset is required before stepping a completed episode")
        if isinstance(physical_action, dict):
            speed_array = np.asarray(physical_action["u"]).reshape(-1)
            if speed_array.shape != (1,):
                raise ValueError("Push speed must have exactly one component")
            a = np.asarray([physical_action["mode"], speed_array[0]], dtype=float)
        else:
            a = torch.as_tensor(physical_action).detach().cpu().numpy()
        if a.shape != (2,) or not np.isfinite(a).all() or abs(a[0] - round(float(a[0]))) > 1e-6:
            raise ValueError("Push action must be finite [integer mode, speed]")
        mode, speed = int(a[0]), float(a[1])
        if not 0 <= int(mode) < 12 or not 0 <= speed <= .600001:
            raise ValueError("Push action outside mode/speed limits")
        chosen = self._pending_action is not None and np.allclose(a, self._pending_action, rtol=0., atol=1e-7)
        grid_failure = bool(chosen and self._diagnostics.get("no_feasible_action", False))
        # A decision's fail flag belongs only to its returned action, never a
        # caller's later override or a stale action from an earlier step.
        self._pending_action = None
        if not chosen:
            self._diagnostics = {}
        self._obs, reward, term, trunc, info = self.plant.step({"mode": int(mode), "u": np.array([speed])})
        raw = self._obs["error"]
        good = np.linalg.norm(raw[:2])*.1 <= .005 and abs(raw[2]) <= math.radians(2) and info["coverage"] >= .95
        self._hold = self._hold + 1 if good and speed == 0 else 0
        chart_failure = abs(raw[2]) >= self.cfg.angle_chart_limit
        term = bool(term or self._hold >= 5 or chart_failure or grid_failure)
        trunc = bool(trunc and not term)
        cost = self.dt * self.physical_cost()
        self._done = term or trunc
        info = self._info(info)
        failure = bool(info["failure"] or chart_failure or grid_failure)
        reason = ("ANGLE_CHART_EXIT" if chart_failure else "GRID_NO_ACTION" if grid_failure else
                  "SUCCESS" if self._hold >= 5 else info.get("reason", ""))
        info.update(cost=cost, original_reward=float(reward), failure=failure, reason=reason,
                    applied_action=np.array([mode, speed]), quadrature="right_endpoint")
        return self._obs, cost, term, trunc, info

    def render(self):
        return self.plant.render()

    def close(self):
        self.plant.close()
