"""Batched CUDA implementation of the existing reconstructed-room P0/P1 task.

The physical plant remains the four-DOF velocity outer loop, not a rigid-body
quadrotor. PyTorch advances its state; MJWarp renders the unchanged MJCF assets.
No feature network, certificate, depth estimator or optimizer belongs here.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import torch


def rotation_z(angle):
    c, s = angle.cos(), angle.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(-1, 3, 3)


def initial_offsets(cfg, scenes, seeds, goal_ids=None, difficulty="small"):
    """CPU setup only, using the original rejection sampler and independent RNGs."""
    from lac_vs import legacy
    from lac_vs.environment import sample_start
    room = legacy.module(cfg, "room_env")
    goal_ids = [0] * len(scenes) if goal_ids is None else list(goal_ids)
    return np.stack([sample_start(cfg, scene, room.ReferenceFrame(np.asarray(scene["refs"][g]["goal_matrix"])),
                                  np.random.default_rng(seed), difficulty)
                     for scene, seed, g in zip(scenes, seeds, goal_ids)])


class _RoomRenderer:
    def __init__(self, cfg, scene, batch_size, device, cache_dir, samples_per_axis):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        from lac_vs import legacy
        self.mjw, self.wp, self.device = mjw, wp, torch.device(device)
        # Warp's default Windows cache can be unavailable; do not alter global ACLs.
        wp.config.kernel_cache_dir = str(Path(cache_dir).resolve())
        wp.init()
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        room = legacy.module(cfg, "room_env")
        spec = {k: scene[k] for k in ("id", "split", "family", "seed", "params")}
        built = room.build_instance(spec)
        self.cpu_model = room._uri_loaded_model(built["xml"])
        self.scene_id = scene["id"]
        self.camera_id = mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
        cpu_data = mujoco.MjData(self.cpu_model)
        mujoco.mj_forward(self.cpu_model, cpu_data)
        size = legacy.module(cfg, "common").SIZE
        with wp.ScopedDevice(str(self.device)), wp.ScopedStream(self.stream):
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.put_data(self.cpu_model, cpu_data, nworld=batch_size)
            self.context = mjw.create_render_context(
                self.cpu_model, nworld=batch_size, cam_res=(size, size),
                cam_active=[self.camera_id], render_rgb=True, render_depth=False,
                render_seg=False, use_textures=True, use_shadows=True,
                render_skybox=True, samples_per_pixel=samples_per_axis,
                use_fast_math=False, deterministic=True,
            )
            # Rooms are static. Camera motion does not change geometry BVHs.
            mjw.refit_bvh(self.model, self.data, self.context)
            self.rgb_buffer = wp.zeros((batch_size, size, size), dtype=wp.vec3, device=str(self.device))
        self.camera_position = wp.to_torch(self.data.cam_xpos)
        self.camera_matrix = wp.to_torch(self.data.cam_xmat)
        self.rgb_float = wp.to_torch(self.rgb_buffer)
        self.stats = dict(scene_id=scene["id"], worlds=batch_size, ngeom=self.cpu_model.ngeom,
                          nmesh=self.cpu_model.nmesh, ntex=self.cpu_model.ntex,
                          samples_per_axis=samples_per_axis, simulator_depth=False)

    def render(self, position, matrix):
        # Both frameworks execute in the caller's CUDA stream; shared arrays have
        # no host staging. Return owned RGB: the renderer reuses its output buffer.
        stream = self.wp.stream_from_torch(torch.cuda.current_stream(self.device))
        with self.wp.ScopedStream(stream):
            self.camera_position[:, self.camera_id].copy_(position.float())
            self.camera_matrix[:, self.camera_id].copy_(matrix.float())
            self.mjw.render(self.model, self.data, self.context)
            self.mjw.get_rgb(self.context, 0, self.rgb_buffer)
            return (self.rgb_float*255.).round().clamp(0, 255).to(torch.uint8)


class GPUParallelRGBEnv:
    """Tensor vector environment with explicit next-step autoreset semantics.

    Each lane follows its original goal, costs, hold criterion and horizon.
    A completed lane resets on its next step; that call has transition_mask=False
    and is never counted as a training transition. Automatic resets reuse the
    explicitly sampled initial conditions. Call reset with new seeds/offsets to
    change them. Returned observations and supervision are separate GPU tensors.
    """
    metadata = {"render_modes": ["rgb_tensor"], "autoreset_mode": "NextStep"}

    def __init__(self, cfg, scenes, *, plant="P0", max_steps=200, device="cuda:0",
                 cache_dir=None, samples_per_axis=2):
        from lac_vs import legacy
        if plant not in {"P0", "P1"} or not scenes or max_steps < 1:
            raise ValueError("Declare nonempty scenes, P0/P1, and a positive horizon")
        self.cfg, self.scenes = deepcopy(cfg), deepcopy(list(scenes))
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("This explicit GPU backend requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.room, self.common = legacy.module(cfg, "room_env"), legacy.module(cfg, "common")
        if not math.isclose(cfg["plant"]["dt"], self.common.DT, abs_tol=1e-12):
            raise ValueError("Physical time step differs from the reference task")
        if cfg["cost"]["position_scale_m"] != 1 or cfg["cost"]["yaw_scale_deg"] != 25:
            raise ValueError("Physical error coordinates differ from the reference task")
        self.num_envs, self.plant, self.max_steps = len(scenes), plant, int(max_steps)
        self.dt, self.closed = float(self.common.DT), False
        self.scale = self._tensor(self.common.ACTION_SCALE)
        self.mix = self.room.PlantP1.mix
        self._groups = []
        groups = OrderedDict()
        for i, scene in enumerate(scenes):
            groups.setdefault(scene["id"], []).append(i)
        cache_dir = cache_dir or Path(__file__).resolve().parents[3] / ".warp_cache"
        for indices in groups.values():
            renderer = _RoomRenderer(cfg, scenes[indices[0]], len(indices), self.device,
                                     cache_dir, samples_per_axis)
            self._groups.append((torch.tensor(indices, device=self.device), renderer))
        max_obstacles = max(len(s["obstacles"]) for s in scenes)
        centers = np.zeros((len(scenes), max_obstacles, 3))
        half = np.zeros_like(centers)
        mask = np.zeros((len(scenes), max_obstacles), dtype=bool)
        for i, scene in enumerate(scenes):
            for j, ob in enumerate(scene["obstacles"]):
                centers[i, j], half[i, j], mask[i, j] = ob["center"], ob["halfsize"], True
        self.centers, self.half = self._tensor(centers), self._tensor(half)
        self.obstacle_mask = torch.as_tensor(mask, device=self.device)
        self.dims = self._tensor([[s["dims"][k] for k in ("hw", "hl", "height")] for s in scenes])
        self._initialized = False

    def _tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float64)

    def _offset(self):
        delta = torch.einsum("bij,bj->bi", self.body0.transpose(1, 2), self.position - self.goal_position)
        yaw = torch.atan2(self.yaw.sin(), self.yaw.cos())
        return torch.cat((delta, yaw[:, None]), -1)

    @staticmethod
    def physical_error(offset):
        return offset[:, :3].square().sum(-1) + (offset[:, 3] / math.radians(25)).square()

    def _render(self, position, yaw):
        matrix = rotation_z(yaw) @ self.goal_matrix @ self.optical_to_gl
        if len(self._groups) == 1:
            return self._groups[0][1].render(position, matrix)
        rgb = torch.empty((self.num_envs, self.common.SIZE, self.common.SIZE, 3),
                          dtype=torch.uint8, device=self.device)
        for indices, renderer in self._groups:
            rgb[indices] = renderer.render(position[indices], matrix[indices])
        return rgb

    def _observation(self):
        return dict(rgb=self.current, goal_rgb=self.goal_rgb, rotation=self.rotation,
                    previous_action=self.previous.float())

    def reset(self, *, seed=811, options=None):
        from lac_vs.environment import valid_start
        if self.closed:
            raise RuntimeError("Environment is closed")
        options = {} if options is None else dict(options)
        if set(options) - {"offsets", "goal_ids", "difficulty"}:
            raise ValueError("Unknown vector reset option")
        self.goal_ids = list(options.get("goal_ids", [0] * self.num_envs))
        if len(self.goal_ids) != self.num_envs:
            raise ValueError("One goal id per environment is required")
        frames = [self.room.ReferenceFrame(np.asarray(s["refs"][g]["goal_matrix"]))
                  for s, g in zip(self.scenes, self.goal_ids)]
        if "offsets" in options:
            offsets = np.asarray(options["offsets"], dtype=np.float64)
        else:
            seeds = [int(seed) + i for i in range(self.num_envs)]
            offsets = initial_offsets(self.cfg, self.scenes, seeds, self.goal_ids,
                                      options.get("difficulty", "small"))
        if offsets.shape != (self.num_envs, 4) or not np.isfinite(offsets).all():
            raise ValueError("Initial offsets must be finite B,4")
        if any(not valid_start(self.cfg, s, f, e) for s, f, e in zip(self.scenes, frames, offsets)):
            raise ValueError("Invalid initial workspace or clearance")
        self.goal_position = self._tensor(np.stack([f.goal[:3, 3] for f in frames]))
        self.goal_matrix = self._tensor(np.stack([f.goal[:3, :3] for f in frames]))
        self.body0 = self._tensor(np.stack([f.body0 for f in frames]))
        self.rotation = self._tensor(np.stack([f.camera_from_body for f in frames])).float()
        self.optical_to_gl = self._tensor(self.common.OPTICAL_TO_GL)
        off = self._tensor(offsets)
        self.initial_position = self.goal_position + torch.einsum("bij,bj->bi", self.body0, off[:, :3])
        self.initial_yaw = off[:, 3].clone()
        self.position, self.yaw = self.initial_position.clone(), self.initial_yaw.clone()
        self.previous = torch.zeros((self.num_envs, 4), dtype=torch.float64, device=self.device)
        self.actual = torch.zeros_like(self.previous)
        self.steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.hold = torch.zeros_like(self.steps)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_rgb = self._render(self.goal_position, torch.zeros_like(self.yaw))
        self.current = self._render(self.position, self.yaw)
        self._initialized = True
        initial = self._offset()
        return self._observation(), dict(supervision=dict(offset=initial, next_offset=initial.clone(),
                                                         physical_error=self.physical_error(initial)))

    @torch.no_grad()
    def step(self, action):
        if not self._initialized or self.closed:
            raise RuntimeError("Reset an open environment before stepping")
        if not torch.is_tensor(action) or action.shape != (self.num_envs, 4) or action.device != self.device:
            raise ValueError("GPU vector actions must have B,4 shape on the environment device")
        if not torch.isfinite(action).all() or (action[:, :3].norm(dim=-1) > 1+1e-6).any() or (action[:, 3].abs() > 1+1e-6).any():
            raise ValueError("Nonfinite or out-of-envelope action")
        active = ~self.done
        # Next-step autoreset is mask-based, including resets in measured runtime.
        self.position = torch.where(active[:, None], self.position, self.initial_position)
        self.yaw = torch.where(active, self.yaw, self.initial_yaw)
        self.previous = torch.where(active[:, None], self.previous, 0.)
        self.actual = torch.where(active[:, None], self.actual, 0.)
        self.steps = torch.where(active, self.steps, 0)
        self.hold = torch.where(active, self.hold, 0)
        before = self._offset()
        previous = self.previous.clone()
        command = torch.where(active[:, None], action.double(), 0.)
        if self.plant == "P1":
            actual = (1-self.mix)*self.actual + self.mix*self.scale*self.previous
        else:
            actual = self.scale*command
        body = rotation_z(self.yaw + .5*actual[:, 3]*self.dt) @ self.body0
        self.position = self.position + torch.einsum("bij,bj->bi", body, actual[:, :3])*self.dt
        self.yaw = self.yaw + actual[:, 3]*self.dt
        self.actual, self.previous = actual, command
        after = self._offset()
        error = self.physical_error(after)
        distance = ((self.position[:, None] - self.centers).abs() - self.half).clamp_min(0)
        collision = ((distance.square().sum(-1) <= self.common.COLLISION_RADIUS**2) & self.obstacle_mask).any(-1)
        hw, hl, height = self.dims.unbind(-1)
        p = self.position
        inside = ((p[:, 0] > -hw+.3) & (p[:, 0] < hw-.3) & (p[:, 1] > -hl+.3) & (p[:, 1] < hl-.3)
                  & (p[:, 2] > self.common.WORKSPACE_Z[0]-.05) & (p[:, 2] < height-.4))
        outside = ~inside
        good = ((after[:, :3].norm(dim=-1) < self.cfg["collection"]["position_tolerance_m"])
                & (after[:, 3].abs() < math.radians(self.cfg["collection"]["yaw_tolerance_deg"])))
        self.hold = torch.where(active & good, self.hold+1, 0)
        self.steps = self.steps + active.long()
        success = active & (self.hold >= self.cfg["collection"]["required_hold_observations"]) & ~(collision | outside)
        terminal = active & (success | collision | outside)
        truncated = active & (self.steps >= self.max_steps) & ~terminal
        self.done = terminal | truncated
        input_cost = command.square().sum(-1)*self.dt
        cost = self.dt*error + self.cfg["cost"]["effort"]*input_cost
        cost = cost + (collision | outside).to(cost.dtype)*self.cfg["cost"]["collision_terminal"]
        cost = torch.where(active, cost, 0.)
        self.current = self._render(self.position, self.yaw)
        info = dict(transition_mask=active, autoreset_mask=~active, success=success,
                    failure=active & (collision | outside), collision=active & collision, outside=active & outside,
                    steps=self.steps.clone(), hold_observations=self.hold.clone(),
                    dt=torch.where(active, self.dt, 0.),
                    supervision=dict(offset=before, next_offset=after,
                        physical_error=error, physical_error_before=self.physical_error(before), physical_cost=cost,
                        actual_velocity=actual, previous_action=previous,
                        integrated_error=torch.where(active, self.dt*error, 0.),
                        normalized_input_squared_integral=input_cost))
        return self._observation(), -cost, terminal, truncated, info

    def render(self):
        return self.current.clone()

    def close(self):
        self.closed = True
        self._groups.clear()

