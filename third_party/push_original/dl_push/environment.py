"""Thin real PhysX wrapper; preserve prior observations and terminal feedback."""
from __future__ import annotations

import numpy as np

from .runtime import activate_reference, REFERENCE
from .state import YawLift


class PushWorld:
    def __init__(self, spec, config, mass=.8, friction=3., render_mode=None):
        activate_reference()
        import gymnasium as gym
        import ca_glpr_unseen_shape.env.mixed_shapes as module
        from ca_glpr_unseen_shape.env import shapes
        from starter.state import physical_state12
        from pathlib import Path
        import mani_skill
        if not Path(mani_skill.__file__).resolve().is_relative_to(REFERENCE):
            raise RuntimeError("ManiSkill must be loaded from reference/")
        # Extend a process-local truth catalogue; never edit reference source files.
        shapes.MIXED_SHAPES[spec.name] = spec
        if spec.name not in module.ALL_SHAPE_NAMES:
            module.ALL_SHAPE_NAMES = (*module.ALL_SHAPE_NAMES, spec.name)
        environment_type = type("DLPushShapeEnv", (module.ManiSkillPushMixedShapesEnv,), {
            "_SHAPE_NAMES": (spec.name,), "_TEE_MASS": float(mass),
            "_TEE_STATIC_FRICTION": float(friction), "_TEE_DYNAMIC_FRICTION": float(friction),
        })
        raw = environment_type(
            num_envs=config.num_envs, obs_mode="state", sim_backend="gpu",
            control_mode="velocity_xy", fixed_shape=spec.name, reset_mode=config.reset_mode,
            contact_diagnostics=True, reward_success_mode="instant_pose_velocity",
            observation_width=64, observation_height=64,
            render_mode=render_mode,
        )
        self.env = gym.wrappers.TimeLimit(raw, max_episode_steps=config.episode_steps)
        self.raw = raw
        self.config, self.spec, self._reader = config, spec, physical_state12
        self.goal = np.array([*raw._GOAL_XY, raw._GOAL_YAW], np.float32)
        if raw._sim_steps_per_control != 10 or raw.pusher_max_speed != .30:
            self.close()
            raise RuntimeError("Reference timing or action contract changed")
        self.lifts = []

    def _states(self):
        states = []
        for lane, lift in enumerate(self.lifts):
            # image-frame model origin matches the native truth actor origin.
            state = lift(self._reader(self.env, lane, (0., 0.), 0.))
            if not np.isfinite(state).all():
                raise FloatingPointError("Non-finite true PhysX state")
            states.append(state)
        return np.stack(states)

    def reset(self, seed):
        self.lifts = [YawLift() for _ in range(self.config.num_envs)]
        obs, info = self.env.reset(seed=seed)
        if obs.shape != (self.config.num_envs, 17):
            raise RuntimeError("Prior observation contract changed")
        # Reference evaluate() also runs at reset. Remove that non-step from its hold counter.
        self.raw._success_hold_count.zero_()
        return obs, self._states(), info

    def step(self, physical_actions):
        import torch
        actions = np.asarray(physical_actions, dtype=np.float32)
        if actions.shape != (self.config.num_envs, 2) or not np.isfinite(actions).all():
            raise ValueError("Expected finite per-lane physical actions")
        if np.any(np.abs(actions) > .300001):
            raise ValueError("Physical action exceeds velocity interface")
        result = self.env.step(torch.as_tensor(actions / .30, device=self.raw.device))
        if not self.config.terminate_on_success:
            obs, reward, terminated, truncated, info = result
            # Keep the original success signal, but evaluate beyond its loose 20 mm gate.
            fail = info.get("fail", torch.zeros_like(terminated))
            terminated = (terminated & ~info["success"]) | fail
            result = obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = result
        # No autoreset wrapper: read the actual terminal physical state exactly once.
        return obs, self._states(), reward, terminated, truncated, info

    def close(self):
        self.env.close()
