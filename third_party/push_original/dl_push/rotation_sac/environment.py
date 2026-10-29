"""SAC-facing rotation task with explicit angular and drift objectives."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import torch
import torch.nn.functional as F

from dl_push.six_primitives.environment import SixConfig, SixPrimitiveEnv, SPLITS, TASKS


class RotationSACEnv:
    """Vectorized PhysX environment for clockwise/counter-clockwise rotation.

    The object starts at the same origin and yaw in every reset.  ``contact_reset``
    is a reset-only curriculum that places the pusher at a high-leverage corner;
    it never changes the object pose or inserts actions into replay.
    """

    def __init__(self, shape='T', num_envs=32, horizon=240, contact_reset=True,
                 angle_min=.15, angle_max=1.0, seed=20260911, task=None):
        split = next(name for name, names in SPLITS.items() if shape in names)
        self.shape = shape
        self.contact_reset = bool(contact_reset)
        if task is not None and task not in (4, 5):
            raise ValueError('Rotation task must be 4 (CW) or 5 (CCW)')
        task_ids = (task,) if task is not None else (4, 5)
        self.task = task
        self.base = SixPrimitiveEnv(SixConfig(
            num_envs=num_envs, split=split, horizon=horizon, fixed_shape=shape,
            fixed_task=task, task_ids=task_ids, reset_mode='fixed_origin',
            angle_min=angle_min, angle_max=angle_max, contact_state=True,
        ))
        self.device = self.base.device
        self.num_envs = num_envs
        self.dt = self.base.dt
        self.prev_action = torch.zeros(num_envs, 2, device=self.device)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        self.obs_dim = 16 * 16 + 18 + 1 + 2 + 2 + 1 + 1 + 1 + len(SPLITS[split])
        self._last_obs = None

    def _place_contact(self, env_idx=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_idx is None else torch.as_tensor(
            env_idx, device=self.device, dtype=torch.long)
        active_shape = self.base.raw._active_shape_id[ids]
        tasks = self.base.task[ids]
        positions = self.base.raw.pusher.pose.p[:, :2].clone()
        centers = self.base.initial_xy[ids]
        for shape_id, spec in enumerate(self.base.raw._shape_specs):
            selected = active_shape == shape_id
            if not selected.any():
                continue
            corners = torch.tensor([
                [cx + sx * hx, cy + sy * hy]
                for (cx, cy), (hx, hy) in spec.boxes
                for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
            ], device=self.device)
            # CW uses the upper-right lever; CCW uses the lower-right lever.
            score_cw = corners[:, 0] + corners[:, 1]
            score_ccw = corners[:, 0] - corners[:, 1]
            corner_cw = corners[score_cw.argmax()]
            corner_ccw = corners[score_ccw.argmax()]
            chosen = torch.where((tasks[selected] == 4)[:, None], corner_cw, corner_ccw)
            radial = chosen / chosen.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # Place the circular pusher on the actual expanded boundary. The
            # previous +1 mm gap is 0.01 in contact_gap's normalized units and
            # left SAC collecting mostly non-contact transitions.
            positions[ids[selected]] = centers[selected] + chosen + radial * (
                self.base.raw.agent.pusher_radius + .0001)
        positions.clamp_(-self.base.raw._PUSHER_JOINT_LIMIT, self.base.raw._PUSHER_JOINT_LIMIT)
        self.base.raw.agent.reset(positions)
        self.base.raw.scene._gpu_apply_all()
        self.base.raw.scene.px.gpu_update_articulation_kinematics()
        self.base.raw.scene._gpu_fetch_all()

    def _shape_code(self):
        names = SPLITS[self.base.config.split]
        code = F.one_hot(self.base.raw._active_shape_id, num_classes=len(names)).float()
        return code

    def _encode(self, raw_obs=None):
        raw_obs = self.base.observe() if raw_obs is None else raw_obs
        # Reflect CCW observations into the CW canonical task frame. The actor
        # therefore learns one signed rotation problem; step() maps its action
        # back to world coordinates for task 5.
        ccw = self.base.task == 5
        shape = raw_obs['shape'].clone()
        if ccw.any():
            shape[ccw] = shape[ccw].flip(-2)
        pooled = F.avg_pool2d(shape, kernel_size=4).flatten(1)
        state = raw_obs['s'].clone()
        if ccw.any():
            for index in (1, 3, 5, 7, 11, 14):
                state[ccw, index] *= -1
            for index in (8, 10, 15):
                state[ccw, index] *= -1
        metrics = self.base.metrics()
        contact = (self.base.raw._last_control_pusher_tee_force / 30.).clamp(0, 1)[:, None]
        gap = self.base.contact_gap()[:, None].clamp(0, 1)
        drift = (self.base.translation_error_sum / .2).clamp(0, 5)[:, None]
        # Task one-hot is only the two rotation entries; retain direction exactly.
        task = torch.zeros_like(raw_obs['task'][:, 4:6]); task[:, 0] = 1.
        vector = torch.cat((pooled, state, raw_obs['g'] / 1.57,
                            task, self.prev_action, contact, gap, drift,
                            self._shape_code()), dim=-1)
        if vector.shape[-1] != self.obs_dim:
            raise RuntimeError(f'Rotation observation width changed: {vector.shape[-1]}')
        return vector.float().clone()

    def reset(self, seed=None, env_idx=None):
        obs, info = self.base.reset(seed=seed, env_idx=env_idx)
        ids = torch.arange(self.num_envs, device=self.device) if env_idx is None else torch.as_tensor(
            env_idx, device=self.device, dtype=torch.long)
        self.prev_action[ids] = 0
        if self.contact_reset:
            self._place_contact(ids)
            obs, info = self.base.observe(), self.base.metrics()
        encoded = self._encode(obs)
        self._last_obs = encoded
        return encoded, info

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).clamp(-1, 1)
        canonical_action = action.clone()
        world_action = action.clone()
        ccw = self.base.task == 5
        world_action[ccw, 1] *= -1
        before = self.base.metrics()
        previous = self.prev_action.clone()
        raw_next, _, terminated, truncated, info = self.base.step(world_action)
        scale = torch.ones_like(info['g'])
        progress = before['g'] - info['g']
        d_constraint = torch.relu(info['d_violation'])
        drift_penalty = info['position_drift'] / .02
        angle_penalty = info['g'].abs() / self.base.amount.clamp_min(.15)
        smooth_penalty = (canonical_action - previous).square().sum(-1)
        velocity_penalty = (info['omega'] / .08).square()
        # D is a soft姿态收敛 constraint; the discrete success bonus is separate.
        near_goal = torch.exp(-(info['g'] / .04).square())
        contact_bonus = (info['contact_force'] / 10.).clamp(0, 1) * (info['g'] > self.base.config.angle_tolerance).float()
        reward = (8.0 * progress - 1.0 * drift_penalty - .5 * angle_penalty
                  - 4.0 * d_constraint - .03 * smooth_penalty
                  - .05 * velocity_penalty + .15 * contact_bonus + 1.0 * near_goal)
        reward = reward + 100.0 * info['success'].float() - 5.0 * info['fail'].float()
        reward = reward + .25 * (info['hold'] > 0).float()
        info = dict(info)
        info.update(rotation_reward=reward.clone(), angle_progress=progress.clone(),
                    drift_penalty=drift_penalty.clone(), d_constraint=d_constraint.clone(),
                    action_delta=torch.linalg.vector_norm(action - previous, dim=-1),
                    contact=(info['contact_force'] >= .05) | (self.base.contact_gap() <= .001))
        done = terminated | truncated
        self.prev_action = canonical_action.clone()
        # Reset finished lanes immediately for the next vector transition. The
        # terminal mask remains in the returned batch, so SAC does not bootstrap it.
        if done.any():
            reset_ids = done.nonzero().flatten()
            self.base.reset(env_idx=reset_ids)
            self.prev_action[reset_ids] = 0
            if self.contact_reset:
                self._place_contact(reset_ids)
        encoded = self._encode(self.base.observe())
        self._last_obs = encoded
        return encoded, reward, terminated, truncated, info

    def provenance(self):
        files = sorted(Path(__file__).parent.glob('*.py'))
        return dict(route='rotation_sac_physx_v1', backend='SAPIEN GPU PhysX',
                    shape=self.shape, task_ids=[4, 5], contact_reset=self.contact_reset,
                    object_reset='fixed_origin: xy=(0,0), yaw=0',
                    action='normalized pusher XY velocity, u=.30*a m/s',
                    control_dt=self.dt, files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in files}, config=asdict(self.base.config))

    def close(self):
        self.base.close()

    def set_angle_max(self, value):
        value = float(value)
        if not .15 <= value <= 1.57:
            raise ValueError('angle_max must lie in [.15, 1.57] rad')
        self.base.config.angle_max = value
