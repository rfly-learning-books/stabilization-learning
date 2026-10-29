"""Audited reconstructed RGB rooms behind a physical-only Gymnasium boundary.

No historical AE, M or replay is loaded.  The geometry provider uses only the
fixed public FastDepth model, current RGB and camera calibration.  Physical
pose and actual actuator velocity are returned under a separate supervision
key and never become feature/control inputs.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Mapping

import gymnasium as gym
import numpy as np
import torch
from torch import nn


PUBLIC_KEYS = frozenset({"rgb", "goal_rgb", "rotation", "previous_action"})


def load_runtime(root):
    """Read audited code/config/scene assets; initialize no learned weights.

    The source adapter validates the existing renderer and source hashes.  Only
    metadata is retained from the historical protocol; its policy/replay paths
    are removed before returning the new configuration.
    """
    import json
    from ..envs.tasks.original_visual_online import OriginalVisualOnlineAdapter

    root = Path(root).resolve()
    reconstruction = root / "migration/visual_scene_reconstruction"
    adapter = OriginalVisualOnlineAdapter(root, reconstruction / "reconstructed_legacy_root",
                                         reconstruction / "SCENE_AUDIT.json")
    cfg = deepcopy(adapter.cfg)
    cfg["paths"]["initial_autoencoder"] = None
    cfg["encoder"]["initialization"] = "random_in_new_run"
    for key in ("source_run", "initialization_checkpoint"):
        cfg["derivative"].pop(key, None)
    cfg["goal_fixed"].pop("repair_source_run", None)
    cfg["joint_v2"] = {"root": str(root), "original_rgb_equivalent": False,
                         "old_policy_weights_loaded": False, "old_replay_loaded": False,
                         "scene_manifest": str(reconstruction / "scenes.json"),
                         "scene_manifest_sha256": adapter.scene_audit["scene_manifest_sha256"]}
    scenes = json.loads((reconstruction / "scenes.json").read_text(encoding="utf-8"))["scenes"]
    return adapter, cfg, scenes


def _array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb(rgb):
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected a uint8 RGB image")
    return array


def _public(observation):
    if set(observation) != PUBLIC_KEYS:
        raise ValueError("Only RGB, target RGB, camera rotation and previous action are public")
    rgb, goal = _rgb(observation["rgb"]), _rgb(observation["goal_rgb"])
    rotation = np.asarray(observation["rotation"], dtype=np.float32)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Invalid public camera/body calibration")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5) or np.linalg.det(rotation) < 0:
        raise ValueError("Camera/body calibration must be a proper rotation")
    return rgb, goal, rotation


class ReconstructedRGBEnv(gym.Env):
    """Four-DOF P0/P1 plant with RGB observations and normalized commands.

    This is a visual velocity outer-loop testbed.  P1 models one-sample delay
    and first-order velocity tracking, not motor/attitude quadrotor dynamics.
    The second step result is reward=-physical_cost.  Pose labels and the
    actuator state are isolated in info['supervision'].
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg, scene, *, plant="P0", max_steps=200):
        from lac_vs import legacy
        from train_goal_fixed import new_instance
        if plant not in {"P0", "P1"}:
            raise ValueError("Only the declared P0/P1 plants are supported")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.cfg, self.scene = deepcopy(cfg), deepcopy(scene)
        self.room = legacy.module(cfg, "room_env")
        self.common = legacy.module(cfg, "common")
        if not math.isclose(float(cfg["plant"]["dt"]), float(self.common.DT), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Config dt differs from the frozen physical plant")
        if (not math.isclose(float(cfg["cost"]["position_scale_m"]), 1.) or
            not math.isclose(float(cfg["cost"]["yaw_scale_deg"]), 25.)):
            raise ValueError("This bridge preserves the goal-fixed physical cost coordinates")
        self.plant_name, self.plant = plant, self.room.PLANTS[plant]
        self.instance = new_instance(cfg, scene)
        self.max_steps = max_steps
        self.dt = float(self.common.DT)
        self.action_space = gym.spaces.Box(-1., 1., (4,), dtype=np.float32)
        size = self.common.SIZE
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(0, 255, (size, size, 3), dtype=np.uint8),
            "goal_rgb": gym.spaces.Box(0, 255, (size, size, 3), dtype=np.uint8),
            "rotation": gym.spaces.Box(-1., 1., (3, 3), dtype=np.float32),
            "previous_action": gym.spaces.Box(-1., 1., (4,), dtype=np.float32)})
        self._done, self._closed = True, False

    def _offset(self):
        error = np.r_[self.frame.body0.T @ (self._state["position"]-self.frame.goal[:3, 3]),
                      self._state["yaw"]]
        error[3] = np.arctan2(np.sin(error[3]), np.cos(error[3]))
        return error.astype(np.float64)

    @staticmethod
    def physical_error(offset):
        offset = np.asarray(offset, dtype=np.float64)
        return float(offset[:3] @ offset[:3] + (offset[3]/math.radians(25.))**2)

    def _observation(self):
        return {"rgb": self._current.copy(), "goal_rgb": self._goal.copy(),
                "rotation": self.frame.camera_from_body.astype(np.float32).copy(),
                "previous_action": self._previous.copy()}

    def reset(self, *, seed=None, options=None):
        from lac_vs.environment import sample_start, valid_start
        if self._closed:
            raise RuntimeError("Cannot reset a closed room")
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        if set(options) - {"goal_id", "offset", "difficulty"}:
            raise ValueError("Only goal_id, offset and difficulty reset options are accepted")
        goal = options.get("goal_id", 0)
        if isinstance(goal, bool) or not isinstance(goal, int) or not 0 <= goal < len(self.scene["refs"]):
            raise ValueError("Invalid declared goal index")
        self.goal_id = goal
        self.frame = self.room.ReferenceFrame(np.asarray(self.scene["refs"][goal]["goal_matrix"]))
        difficulty = options.get("difficulty", "small")
        if difficulty not in {"small", "medium", "large"}:
            raise ValueError("Unknown start difficulty")
        if "offset" in options:
            offset = np.asarray(options["offset"], dtype=np.float64)
            if offset.shape != (4,) or not np.isfinite(offset).all():
                raise ValueError("Initial offset must be a finite four-vector")
            if not valid_start(self.cfg, self.scene, self.frame, offset):
                raise ValueError("Initial offset violates the declared room workspace/clearance")
        else:
            offset = sample_start(self.cfg, self.scene, self.frame, self.np_random, difficulty)
        self._state = {"position": self.frame.position_from_offset(offset), "yaw": float(offset[3]),
                       "body0": self.frame.body0, **self.plant.initial_extra()}
        self._goal = self.instance.goal_rgb(self.frame)
        self._current = self.instance.render_rgb(self.frame, self._state["position"], self._state["yaw"])
        self._previous = np.zeros(4, dtype=np.float32)
        self._steps, self._hold, self._done = 0, 0, False
        initial = self._offset()
        info = {"scene_id": self.scene["id"], "split": self.scene["split"], "goal_id": goal,
                "plant": self.plant_name, "dt": 0., "success": False, "failure": False,
                "supervision": {"offset": initial.copy(), "next_offset": initial.copy(),
                                "physical_error": self.physical_error(initial), "physical_cost": 0.}}
        return self._observation(), info

    def step(self, action):
        if self._done or self._closed:
            raise RuntimeError("Reset an active, open room before stepping")
        command = np.asarray(_array(action), dtype=np.float64)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError("Action must be a finite normalized four-vector")
        if np.linalg.norm(command[:3]) > 1.+1e-6 or abs(command[3]) > 1.+1e-6:
            raise ValueError("Command exceeds the normalized translation/yaw envelope")
        before = self._offset()
        self._state = self.plant.step_state(self._state, command)
        after = self._offset()
        E = self.physical_error(after)
        collision = bool(self.instance.collides(self._state["position"]))
        outside = bool(self.instance.outside_workspace(self._state["position"]))
        good = (np.linalg.norm(after[:3]) < self.cfg["collection"]["position_tolerance_m"] and
                abs(after[3]) < math.radians(self.cfg["collection"]["yaw_tolerance_deg"]))
        self._hold = self._hold+1 if good else 0
        self._steps += 1
        success = self._hold >= self.cfg["collection"]["required_hold_observations"] and not (collision or outside)
        terminal = bool(collision or outside or success)
        truncated = self._steps >= self.max_steps and not terminal
        cost = self.dt*(E+self.cfg["cost"]["effort"]*float(command @ command))
        if collision or outside:
            cost += float(self.cfg["cost"]["collision_terminal"])
        actual = (self._state["actual_velocity"].copy() if self.plant_name == "P1"
                  else self.common.ACTION_SCALE*command)
        self._previous = command.astype(np.float32)
        self._current = self.instance.render_rgb(self.frame, self._state["position"], self._state["yaw"])
        self._done = bool(terminal or truncated)
        reason = "collision" if collision else "outside" if outside else "success" if success else "time_limit" if truncated else "running"
        info = {"scene_id": self.scene["id"], "split": self.scene["split"], "goal_id": self.goal_id,
                "plant": self.plant_name, "dt": self.dt, "steps": self._steps,
                "success": bool(success), "failure": bool(collision or outside), "reason": reason,
                "hold_observations": self._hold, "reward_convention": "negative_physical_cost",
                "supervision": {"offset": before, "next_offset": after,
                    "physical_error": E, "physical_error_before": self.physical_error(before),
                    "physical_cost": float(cost), "actual_velocity": actual,
                    "integrated_error": self.dt*E,
                    "normalized_input_squared_integral": self.dt*float(command @ command)}}
        return self._observation(), -float(cost), terminal, bool(truncated), info

    def render(self):
        if not hasattr(self, "_current"):
            raise RuntimeError("Reset before rendering")
        return self._current.copy()

    def close(self):
        if not self._closed:
            self.instance.close()
            self._closed = True


class RGBGeometry:
    """Current-image geometry from public FastDepth, never simulator depth.

    Tangents are derivatives per second for NORMALIZED commands and already
    include action scales.  The feature provider is responsible for applying
    exactly the same latent normalization to features and their JVP.
    """
    def __init__(self, cfg, device="cpu"):
        from goal_fixed_core import ROOT, FastDepth
        from lac_vs import vision
        path = ROOT / cfg["goal_fixed"]["depth_path"] / "extracted_state_dict.pt"
        if not path.is_file():
            raise FileNotFoundError("Pre-extracted public depth weights are required; this bridge never writes old assets")
        self.cfg, self.device, self.vision = deepcopy(cfg), torch.device(device), vision
        self.depth = FastDepth(cfg, device)
        self.provenance = {"kind": "fixed_public_FastDepth", "path": str(path),
                           "sha256": self.depth.hash, "trainable": False,
                           "simulator_depth_used": False}

    def observe(self, public):
        rgb, goal, rotation = _public(public)
        estimated_depth = self.depth.predict(rgb)
        image, tangents = self.vision.images_and_tangents(
            self.cfg, [rgb], [estimated_depth], [rotation], self.device)
        goal_image = self.vision.image_tensor(self.cfg, [goal], self.device)
        if not torch.isfinite(tangents).all():
            raise FloatingPointError("Estimated-depth image tangents are nonfinite")
        return image, goal_image, tangents


class RandomLegacyAutoencoder(nn.Module):
    """Raw legacy RGB backbone with independent normalization owned by learner."""
    def __init__(self, cfg):
        super().__init__()
        from goal_fixed_core import make_model
        clean = deepcopy(cfg)
        clean["paths"]["initial_autoencoder"] = None
        # initialize=False prohibits both old AE weights and bottleneck SVD.
        self.base = make_model(clean, initialize=False).base

    def encode(self, image):
        return self.base.encode(image)

    def decode(self, code):
        return self.base.decoder(self.base.expand_spatial(self.base.expand(code)[..., None, None]))

    def forward(self, image):
        code = self.encode(image)
        return code, self.decode(code)


def random_raw_autoencoder(cfg, device="cpu"):
    return RandomLegacyAutoencoder(cfg).to(device)
