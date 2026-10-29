"""RGB multirotor visual servo and an optional preserved-experiment adapter.

The portable backend is a NEW downward-camera perspective texture simulator.
It is not a reconstruction of the deleted indoor room renderer or old results.
Images alone form errors; simulator pose appears only in cost/evaluation info.
"""
from __future__ import annotations

import ast
import importlib
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...common.types import MetricSpec
from ...common.action_spaces import BodyVelocitySpace
from ...common.paths import asset_path

WORKSPACE = asset_path("legacy_workspace")
VS_ROOT = asset_path("visual_legacy_project")


class ImageDCTEncoder(nn.Module):
    """Deterministic 16-D RGB descriptor for a dependency-free smoke baseline.

    A learned shared encoder can replace this through ``representation=``.
    This descriptor is never described as the old trained AE.
    """
    def __init__(self, size=64, dim=16):
        super().__init__()
        n = torch.arange(size, dtype=torch.float32) + .5
        bases = []
        for i in range(4):
            for j in range(4):
                bases.append(torch.cos(math.pi*i*n/size)[:, None] * torch.cos(math.pi*j*n/size)[None, :])
        self.register_buffer("basis", torch.stack(bases[:dim]) / size**2)
        self.output_dim = dim

    def encode(self, x):
        gray = (x * x.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
        return 8 * torch.einsum("bhw,khw->bk", gray, self.basis.to(x))

    def forward(self, x):
        return self.encode(x)


class ReadOnlyFastDepth:
    """Load existing pure tensor weights without running upstream trainer code."""
    def __init__(self, root=None, device="cpu"):
        root = Path(root or asset_path("fastdepth"))
        for name in ("models.py", "extracted_state_dict.pt"):
            if not (root / name).is_file():
                raise FileNotFoundError(f"FastDepth asset missing: {root / name}")
        tree = ast.parse((root / "models.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        scope = {"torch": torch, "nn": nn, "F": F, "math": math, "time": time}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / "models.py"), "exec"), scope)
        self.model = scope["FastDepth"]().to(device).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(root / "extracted_state_dict.pt", map_location=device, weights_only=True))
        self.device = torch.device(device)

    @torch.no_grad()
    def predict(self, image):
        x = F.interpolate(image.to(self.device), size=(224, 224), mode="bilinear", align_corners=False)
        d = self.model(x)
        return F.interpolate(d, size=image.shape[-2:], mode="bilinear", align_corners=False).clamp(.25, 15.)


class VisualServoEnv(gym.Env):
    """Four-axis body-velocity servo using shared RGB features and fixed J*.

    Observation: current RGB and reference RGB only. Action: physical body
    ``[vx,vy,vz,yaw_rate]`` in m/s and rad/s. The controller uses an estimated
    reference depth (default constant 2 m), never simulator true depth.

    Call ``bind_representation(phi)`` then reset before each rollout. The shared
    encoder is frozen by contract during an episode. ``build_error(...,
    representation=phi)`` is also a differentiable offline training interface;
    its JVP is rebuilt from phi, not a stale detached encoder cache.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    controller_capabilities = {
        "action_gradient": True, "model_gradient": True,
        "episode_fixed_metric": True, "state_dependent_certificate": False,
        "cost_convention": "positive_integral",
        "differentiable_action": True,
        "differentiable_model": True,
        "differentiable_representation": True,
        "reference": "goal_fixed_estimated_depth_image_JVP",
        "metric_scope": "episode",
        "plant": "new_downward_perspective_RGB_velocity_plant",
        "certificate_guarantee": False,
    }

    def __init__(self, *, representation=None, error_dim=16, image_size=64,
                 max_steps=300, split="train", scene_id=0, depth_mode="constant",
                 estimated_depth=2., depth_scale=1., fastdepth_root=None,
                 plant="P0", render_mode="rgb_array", device="cpu", **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected visual servo options: {sorted(kwargs)}")
        if split not in ("train", "val", "test") or plant not in ("P0", "P1"):
            raise ValueError("split must be train/val/test and plant must be P0/P1")
        if image_size < 16 or error_dim < 1 or max_steps < 1 or estimated_depth <= 0 or depth_scale <= 0:
            raise ValueError("Invalid image/dimension/horizon/depth configuration")
        self.error_dim = int(error_dim)
        self.metric_context_dim = self.error_dim
        self.metric_spec = MetricSpec(self.error_dim, eigen_floor=.1, trace=float(self.error_dim))
        self.size, self.max_steps, self.dt = int(image_size), int(max_steps), .1
        self.device = torch.device(device)
        if representation is None and error_dim != 16:
            raise ValueError("Default image descriptor is 16-D; supply representation for another dimension")
        self.representation = representation or ImageDCTEncoder(self.size, self.error_dim)
        self.representation = self.representation.to(self.device).eval()
        self.split, self.scene_id, self.plant_name = split, int(scene_id), plant
        self.depth_mode, self.estimated_depth, self.depth_scale = depth_mode, estimated_depth, depth_scale
        if depth_mode not in ("constant", "fastdepth"):
            raise ValueError("depth_mode must be constant or fastdepth")
        self.depth = ReadOnlyFastDepth(fastdepth_root, device) if depth_mode == "fastdepth" else None
        self.render_mode = render_mode
        image_space = gym.spaces.Box(0, 255, (self.size, self.size, 3), np.uint8)
        self.observation_space = gym.spaces.Dict({"rgb": image_space, "goal_rgb": image_space})
        self.action_space = BodyVelocitySpace(.4, .35)
        grid = (torch.arange(self.size, dtype=torch.float32) + .5 - self.size/2) / (self.size*.8)
        self._Y, self._X = torch.meshgrid(grid, grid, indexing="ij")
        self._done, self._ready = True, False
        self.goal = np.array([0., 0., 2., 0.], dtype=np.float64)
        self.asset_manifest = {"backend": "portable_perspective_texture_v1", "legacy_results_reproduced": False,
                               "depth": depth_mode, "default_encoder": type(self.representation).__name__}

    def bind_representation(self, representation):
        """Change phi only between episodes; reset rebuilds its goal and J*."""
        if self._ready and not self._done:
            raise RuntimeError("Finish the current episode before binding a new representation")
        self.representation = representation.to(self.device).eval()

    def _versions(self):
        return tuple(p._version for p in (*self.representation.parameters(), *self.representation.buffers()))

    def _image(self, pose):
        # Analytic pinhole projection of a textured horizontal plane. No pose
        # information is encoded into the observation outside rendered pixels.
        p = torch.as_tensor(pose, dtype=torch.float32)
        x = p[0] + p[2]*(p[3].cos()*self._X - p[3].sin()*self._Y)
        y = p[1] + p[2]*(p[3].sin()*self._X + p[3].cos()*self._Y)
        c = self._texture
        value = (x[..., None]*c[None, None, :, 0] + y[..., None]*c[None, None, :, 1] + c[None, None, :, 2]).sin()
        rgb = (.5 + .17*torch.einsum("hwn,nc->hwc", value, self._colors)).clamp(0., 1.)
        return (rgb.detach().numpy()*255).round().astype(np.uint8)

    def _tensor(self, image):
        if isinstance(image, torch.Tensor):
            x = image.to(self.device)
            if x.ndim == 3:
                x = x.permute(2, 0, 1)[None] if x.shape[-1] == 3 else x[None]
            return x.float()/255 if x.dtype == torch.uint8 else x.float()
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].to(self.device).float()/255

    def _encode(self, representation, x):
        value = representation.encode(x) if hasattr(representation, "encode") else representation(x)
        if isinstance(value, tuple):
            value = value[0]
        if value.shape != (x.shape[0], self.error_dim):
            raise ValueError(f"Encoder must return [B,{self.error_dim}], got {value.shape}")
        return value

    def _tangents(self, image, depth):
        # Calibrated image derivatives from perspective geometry. This is not
        # an optical-flow estimator or a current-frame feature tracker.
        fy = fx = self.size*.8
        iy, ix = torch.gradient(image, dim=(-2, -1))
        X, Y = self._X.to(image), self._Y.to(image)
        return torch.stack([ix*fx/depth, iy*fy/depth,
            (ix*fx*X + iy*fy*Y)/depth, -ix*fx*Y + iy*fy*X], 1)

    def _reference_features(self, representation, goal, *, graph):
        xg = self._tensor(goal)
        depth = self._estimated_depth.to(xg)
        tangent = self._tangents(xg, depth)
        zg = self._encode(representation, xg)
        columns = []
        for k in range(4):
            _, col = torch.autograd.functional.jvp(lambda x: self._encode(representation, x),
                xg, tangent[:, k], create_graph=graph, strict=False)
            columns.append(col[0]/math.sqrt(self.error_dim))
        return zg[0], torch.stack(columns, -1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        # Texture split is independent of start seed; held-out scene IDs do not
        # leak through metric context, which contains only the goal image code.
        scene_seed = {"train": 11000, "val": 21000, "test": 31000}[self.split] + self.scene_id
        rng = np.random.default_rng(scene_seed)
        self._texture = torch.from_numpy(np.c_[rng.uniform(1., 8., (10, 2)), rng.uniform(-math.pi, math.pi, 10)].astype(np.float32))
        self._colors = torch.from_numpy(rng.normal(0, .55, (10, 3)).astype(np.float32))
        self.goal = np.asarray(options.get("goal", [0., 0., 2., 0.]), np.float64)
        offset = np.asarray(options.get("offset", np.r_[self.np_random.uniform(-.18, .18, 3), self.np_random.uniform(-.15, .15)]))
        self.state = np.asarray(options.get("state", self.goal + offset), dtype=np.float64)
        if self.state.shape != (4,) or self.goal.shape != (4,) or not np.isfinite(self.state).all() or self.state[2] <= .3:
            raise ValueError("State/goal must be finite [x,y,height,yaw] with height > .3 m")
        self._velocity = np.zeros(4)
        self._goal_rgb = self._image(self.goal)
        self._rgb = self._image(self.state)
        xg = self._tensor(self._goal_rgb)
        self._estimated_depth = (self.depth.predict(xg) if self.depth else torch.full_like(xg[:, :1], self.estimated_depth))*self.depth_scale
        self._goal_feature, self._J = self._reference_features(self.representation, self._goal_rgb, graph=False)
        self._goal_feature, self._J = self._goal_feature.detach(), self._J.detach()
        self._parameter_versions = self._versions()
        self.steps, self._hold = 0, 0
        self._done, self._ready = False, True
        return self._observation(), self._info()

    def _observation(self):
        return {"rgb": self._rgb.copy(), "goal_rgb": self._goal_rgb.copy()}

    def metric_context(self, observation=None, goal=None):
        if not self._ready:
            raise RuntimeError("reset must precede metric_context")
        if observation is None and goal is None:
            return self._goal_feature.detach().clone()
        rgb = goal if goal is not None else observation["goal_rgb"]
        with torch.no_grad():
            return self._encode(self.representation, self._tensor(rgb))[0]

    def build_error(self, observation, goal=None, *, representation=None, history=None):
        if not self._ready:
            raise RuntimeError("reset must precede build_error")
        phi = self.representation if representation is None else representation
        if representation is None and self._versions() != self._parameter_versions:
            raise RuntimeError("Phi changed inside an episode; reset to rebuild the fixed reference")
        x = self._tensor(observation["rgb"])
        if representation is None and goal is None:
            zg, J = self._goal_feature, self._J
        else:
            zg, J = self._reference_features(phi, goal if goal is not None else observation["goal_rgb"], graph=torch.is_grad_enabled())
        z = self._encode(phi, x)[0]
        return (z-zg)/math.sqrt(self.error_dim), {"J": J, "goal_feature": zg, "rgb": x,
            "goal_rgb": self._tensor(observation["goal_rgb"]), "reference_frozen": representation is None}

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        if certificate is not None:
            raise ValueError("Use fixed M for this weighted-reference controller")
        M = torch.as_tensor(M)
        e = e.to(M)
        J = context["J"].to(M)
        b = J.transpose(-1, -2) @ M @ J
        ridge = torch.maximum(b.diagonal(dim1=-2, dim2=-1).sum(-1)*1.01/9999,
                              b.new_tensor(1e-8))
        h = b + .01*torch.diag_embed(b.diagonal(dim1=-2, dim2=-1)) + ridge[..., None, None]*torch.eye(4, device=M.device, dtype=M.dtype)
        raw = -.7*torch.linalg.solve(h, (J.transpose(-1, -2) @ M @ e[..., None])).squeeze(-1)
        scale = torch.maximum(torch.ones_like(raw[..., 0]), raw[..., :3].norm(dim=-1)/.4)
        scale = torch.maximum(scale, raw[..., 3].abs()/.35)
        return raw/scale[..., None]

    def predict_next(self, state, action):
        """Differentiable ideal P0 physical step, never used to infer images."""
        state, action = torch.as_tensor(state), torch.as_tensor(action)
        yaw = state[..., 3] + self.dt*.5*action[..., 3]
        dx = yaw.cos()*action[..., 0] - yaw.sin()*action[..., 1]
        dy = yaw.sin()*action[..., 0] + yaw.cos()*action[..., 1]
        return state + self.dt*torch.stack([dx, dy, action[..., 2], action[..., 3]], -1)

    def physical_cost(self, observation=None, action=None, next_observation=None):
        # Privileged pose is available only in labels/info, never feature context.
        d = self.state - self.goal
        yaw = math.atan2(math.sin(d[3]), math.cos(d[3]))
        return float(d[:3] @ d[:3] + (yaw/math.radians(25))**2)

    def _info(self, failure=False):
        return {"physical_error": self.physical_cost(), "position_error_m": float(np.linalg.norm(self.state[:3]-self.goal[:3])),
                "yaw_error_rad": float(abs(math.atan2(math.sin(self.state[3]-self.goal[3]), math.cos(self.state[3]-self.goal[3])))),
                "success": self._hold >= 11, "is_success": self._hold >= 11, "failure": bool(failure),
                "scene_id": self.scene_id, "split": self.split, "plant": self.plant_name,
                "backend": "new portable downward-camera scene; not old indoor benchmark",
                "depth_source": self.depth_mode, "simulated_time": self.steps*self.dt,
                "evaluation_state": self.state.copy()}

    def step(self, physical_action):
        if self._done:
            raise RuntimeError("reset is required before stepping a completed episode")
        if self._versions() != self._parameter_versions:
            raise RuntimeError("Phi changed inside an episode; fixed reference invalid")
        action = torch.as_tensor(physical_action).detach().cpu().numpy().astype(float)
        if action.shape != (4,) or not np.isfinite(action).all() or np.linalg.norm(action[:3]) > .40001 or abs(action[3]) > .35001:
            raise ValueError("Action exceeds physical limits or is nonfinite")
        if self.plant_name == "P1":
            self._velocity += .45*(action-self._velocity)
            applied = self._velocity
        else:
            applied = action
        self.state = self.predict_next(torch.from_numpy(self.state), torch.from_numpy(applied)).numpy()
        self.steps += 1
        failure = bool(np.linalg.norm(self.state[:2]) > 3 or self.state[2] < .35 or self.state[2] > 4)
        yaw = math.atan2(math.sin(self.state[3]-self.goal[3]), math.cos(self.state[3]-self.goal[3]))
        good = np.linalg.norm(self.state[:3]-self.goal[:3]) < .02 and abs(yaw) < math.radians(1.)
        self._hold = self._hold+1 if good else 0
        terminated = failure or self._hold >= 11
        truncated = self.steps >= self.max_steps and not terminated
        self._done = terminated or truncated
        self._rgb = self._image(self.state)
        cost = self.dt*self.physical_cost() + (20. if failure else 0.)
        info = self._info(failure)
        info.update(cost=cost, commanded_action=action.copy(), applied_action=applied.copy(),
                    cost_convention="positive_integral", quadrature="right_endpoint")
        return self._observation(), cost, terminated, truncated, info

    def render(self):
        if not self._ready:
            raise RuntimeError("reset before render")
        return np.concatenate([self._rgb, self._goal_rgb], axis=1)

    def close(self):
        self._done = True


class LegacyVisualServoEnv(VisualServoEnv):
    """Optional original room plant; requires restoration of its deleted source.

    This code path is unavailable on the current machine and therefore is not
    included in the portable benchmark's validation claim. It never substitutes
    the new renderer or an inferred AE architecture for missing original files.
    """
    def __init__(self, *, legacy_root=None, project_root=None, checkpoint=None,
                 representation=None, max_steps=300, split="train", plant="P0", device="cpu"):
        root = Path(legacy_root or WORKSPACE / "output/stabilization_rgb_indoor_v1")
        needed = [root / "src" / name for name in ("room_env.py", "common.py", "models.py", "vision.py", "scene_builders.py")]
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            raise FileNotFoundError("Old RGB benchmark source was removed with its cache. Required assets: " + ", ".join(missing))
        import json
        project = Path(project_root or VS_ROOT).resolve()
        for path in (project, project / "runtime"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        self._legacy_core = importlib.import_module("goal_fixed_core")
        self.cfg = self._legacy_core.load_config(project / "configs/derivative_critic.yaml")
        self.cfg["paths"]["legacy_root"] = str(root)
        self.device = torch.device(device)
        self.error_dim = self.metric_context_dim = 16
        self.metric_spec = MetricSpec(16, eigen_floor=.1, trace=16.)
        self.max_steps, self.dt, self.split, self.plant_name = int(max_steps), .1, split, plant
        if split not in ("train", "val", "test") or plant not in ("P0", "P1"):
            raise ValueError("Invalid split/plant")
        run = project / "runs/ae_derivative_critic_seed41_20261019_v1"
        self.scenes = [s for s in json.loads((run / "scenes.json").read_text(encoding="utf-8"))["scenes"] if s["split"] == split]
        self.representation = representation or self._legacy_core.make_model(self.cfg, initialize=False)
        if representation is None:
            state = torch.load(checkpoint or run / "best.pt", map_location="cpu", weights_only=False)
            self.representation.load_state_dict(state["learner"]["ae"])
        self.representation = self.representation.to(device).eval()
        # Requiring extracted tensor weights prevents FastDepth's old loader
        # from creating a new extracted asset in the historical directory.
        extracted = project / self.cfg["goal_fixed"]["depth_path"] / "extracted_state_dict.pt"
        if not extracted.exists():
            raise FileNotFoundError(f"Read-only FastDepth weights missing: {extracted}")
        self.depth = self._legacy_core.FastDepth(self.cfg, device=device)
        self._room = self._legacy_core.legacy.module(self.cfg, "room_env")
        self._plant = self._room.PLANTS[plant]
        self.instance = None
        self._ready, self._done = False, True
        self.render_mode = "rgb_array"
        self.size = 256
        box = gym.spaces.Box(0, 255, (256, 256, 3), np.uint8)
        self.observation_space = gym.spaces.Dict({"rgb": box, "goal_rgb": box})
        self.action_space = BodyVelocitySpace(.4, .35)
        self.controller_capabilities = {**VisualServoEnv.controller_capabilities, "plant": "original_room_renderer_when_restored"}
        self.asset_manifest = {"legacy_root": str(root), "project_root": str(project), "validated_on_current_machine": False}

    def _tensor(self, image):
        return self._legacy_core.image_tensor(self.cfg, [image], self.device)

    def _reference_features(self, representation, goal, *, graph):
        x, tangent = self._legacy_core.images_and_tangents(self.cfg, [goal],
            [self._goal_depth], [self.frame.camera_from_body], self.device)
        zg = representation.encode(x)[0]
        # Original tangent columns are with respect to normalized commands.
        J = self._legacy_core.encoder_jvp(representation, x, tangent)[0].transpose(0, 1)
        J = J / J.new_tensor([.4, .4, .4, .35])
        return (zg, J) if graph else (zg.detach(), J.detach())

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = dict(options or {})
        if self.instance is not None:
            self.instance.close()
        self.scene = self.scenes[int(options.get("scene_index", 0))]
        self.scene_id = self.scene["id"]
        self.instance = self._room.RoomInstance({k: self.scene[k] for k in ("id", "split", "family", "seed", "params")})
        def forbidden(*args, **kwargs):
            raise RuntimeError("Simulator true-depth rendering is forbidden in this adapter")
        self.instance.renderer.enable_depth_rendering = forbidden
        gi = int(options.get("goal_index", 0))
        self.frame = self._room.ReferenceFrame(np.asarray(self.scene["refs"][gi]["goal_matrix"]))
        start = options.get("offset")
        if start is None:
            start = self._legacy_core.sample_start(self.cfg, self.scene, self.frame, self.np_random, "small")
        start = np.asarray(start, dtype=float)
        self._room_state = {"position": self.frame.position_from_offset(start), "yaw": float(start[3]),
                            "body0": self.frame.body0, **self._plant.initial_extra()}
        self._goal_rgb = self.instance.goal_rgb(self.frame)
        self._goal_depth = self.depth.predict(self._goal_rgb)
        self._rgb = self.instance.render_rgb(self.frame, self._room_state["position"], self._room_state["yaw"])
        self._goal_feature, self._J = self._reference_features(self.representation, self._goal_rgb, graph=False)
        self._parameter_versions = self._versions()
        self.steps, self._hold, self._ready, self._done = 0, 0, True, False
        return self._observation(), self._info()

    def physical_cost(self, observation=None, action=None, next_observation=None):
        _, E = self._legacy_core.task_error(self.cfg, self.frame, self._room_state)
        return float(E)

    def _info(self, failure=False):
        offset, E = self._legacy_core.task_error(self.cfg, self.frame, self._room_state)
        return {"physical_error": float(E), "position_error_m": float(np.linalg.norm(offset[:3])),
                "yaw_error_rad": float(abs(offset[3])), "success": self._hold >= 11,
                "is_success": self._hold >= 11, "failure": failure, "scene_id": self.scene_id,
                "split": self.split, "plant": self.plant_name, "depth_source": "fastdepth",
                "simulated_time": self.steps*self.dt, "evaluation_offset": offset.copy()}

    def step(self, physical_action):
        if self._done:
            raise RuntimeError("reset is required before step")
        if self._versions() != self._parameter_versions:
            raise RuntimeError("Phi changed inside an episode; fixed reference invalid")
        a = torch.as_tensor(physical_action).detach().cpu().numpy().astype(float)
        if a.shape != (4,) or not np.isfinite(a).all() or np.linalg.norm(a[:3]) > .40001 or abs(a[3]) > .35001:
            raise ValueError("Action exceeds physical limits")
        self._room_state = self._plant.step_state(self._room_state, a/np.array([.4, .4, .4, .35]))
        self.steps += 1
        offset, E = self._legacy_core.task_error(self.cfg, self.frame, self._room_state)
        failure = bool(self.instance.collides(self._room_state["position"]) or self.instance.outside_workspace(self._room_state["position"]))
        good = np.linalg.norm(offset[:3]) < .02 and abs(offset[3]) < math.radians(1.)
        self._hold = self._hold+1 if good else 0
        terminated = failure or self._hold >= 11
        truncated = self.steps >= self.max_steps and not terminated
        self._done = terminated or truncated
        self._rgb = self.instance.render_rgb(self.frame, self._room_state["position"], self._room_state["yaw"])
        cost = .1*float(E) + (20. if failure else 0.)
        info = self._info(failure)
        info.update(cost=cost, commanded_action=a.copy(), cost_convention="positive_integral",
                    quadrature="right_endpoint")
        return self._observation(), cost, terminated, truncated, info

    def close(self):
        if self.instance is not None:
            self.instance.close()
            self.instance = None
        self._done = True
