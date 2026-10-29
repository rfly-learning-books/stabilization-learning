"""MuJoCo room environment: reference frames, RGB rendering (no ground-truth depth in
any live path), kinematic plants P0/P1 and evaluation-only collision checks."""
from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .common import (ACTION_SCALE, COLLISION_RADIUS, DT, OPTICAL_TO_GL, SIZE, TAU, X, Y, WORKSPACE_Z, rot_z)
from .scene_builders import build_instance, build_shell  # noqa: F401


class GroundTruthDepthForbidden(RuntimeError):
    pass


def _uri_loaded_model(xml):
    compiled = xml
    virtual_files = {}
    for i, filename in enumerate(dict.fromkeys(re.findall(r'file="([^"]+)"', xml))):
        path = Path(filename)
        alias = f"resource_{i}{path.suffix}"
        virtual_files[alias] = path.read_bytes()
        compiled = compiled.replace(f'file="{filename}"', f'file="{alias}"')
    return mujoco.MjModel.from_xml_string(compiled, assets=virtual_files)


class ReferenceFrame:
    def __init__(self, goal):
        self.goal = np.asarray(goal, float)
        forward = self.goal[:3, 2].copy()
        forward[2] = 0
        forward /= np.linalg.norm(forward)
        self.forward = forward
        self.body0 = np.column_stack([forward, np.cross([0.0, 0.0, 1.0], forward), [0.0, 0.0, 1.0]])
        self.camera_from_body = self.goal[:3, :3].T @ self.body0
        self.mount_yaw = self.camera_from_body[:, 2]
        a = np.zeros((SIZE, SIZE, 2, 3), np.float32)
        a[..., 0, 0], a[..., 0, 2] = -1.0, X
        a[..., 1, 1], a[..., 1, 2] = -1.0, Y
        self.trans = a @ self.camera_from_body
        kx, ky, kz = self.mount_yaw
        self.rot = np.stack([X * Y * kx - (1 + X ** 2) * ky + Y * kz,
                             (1 + Y ** 2) * kx - X * Y * ky - X * kz], axis=-1).astype(np.float32)

    def position_from_offset(self, offset_body):
        return self.goal[:3, 3] + self.body0 @ np.asarray(offset_body, float)[:3]

    def pose(self, position, yaw):
        t = self.goal.copy()
        t[:3, :3] = rot_z(yaw) @ self.goal[:3, :3]
        t[:3, 3] = position
        return t


class RoomInstance:
    """One compiled room. A scene owns one model and several reference frames."""

    def __init__(self, spec, build=True, built=None):
        self.spec = spec
        self.id = spec["id"]
        built = build_instance(spec) if (built is None and build) else built
        if built:
            self.xml = built["xml"]
            self.obstacles = built["obstacles"]
            self.dims = built["dims"]
            self.model = _uri_loaded_model(self.xml)
            self.data = mujoco.MjData(self.model)
            self.cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
            self.renderer = mujoco.Renderer(self.model, SIZE, SIZE)
            self.references = []
            for ref in built["refs"]:
                from .common import look_at

                goal = look_at(ref["eye"], ref["target"])
                frame = ReferenceFrame(goal)
                frame.eye = ref["eye"]
                frame.clearance_m = ref["clearance_m"]
                self.references.append(frame)
        else:
            raise ValueError("RoomInstance requires build=True")

    def close(self):
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None

    def render_rgb(self, frame, position, yaw):
        pose = frame.pose(position, yaw)
        q = Rotation.from_matrix(pose[:3, :3] @ OPTICAL_TO_GL).as_quat()
        self.model.cam_pos[self.cid] = position
        self.model.cam_quat[self.cid] = q[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cid)
        return self.renderer.render().copy()

    def goal_rgb(self, frame):
        return self.render_rgb(frame, frame.goal[:3, 3], 0.0)

    def render(self, *args, **kwargs):
        raise GroundTruthDepthForbidden("Ground-truth depth rendering is forbidden in the RGB-only path")

    def collides(self, position, radius=COLLISION_RADIUS):
        p = np.asarray(position, float)
        for ob in self.obstacles:
            c = np.asarray(ob["center"])
            h = np.asarray(ob["halfsize"])
            d = np.maximum(np.abs(p - c) - h, 0.0)
            if float(d @ d) <= radius * radius:
                return True
        return False

    def outside_workspace(self, position):
        hw, hl, height = self.dims["hw"], self.dims["hl"], self.dims["height"]
        p = np.asarray(position, float)
        return not (-hw + 0.3 < p[0] < hw - 0.3 and -hl + 0.3 < p[1] < hl - 0.3
                    and WORKSPACE_Z[0] - 0.05 < p[2] < height - 0.4)


class PlantP0:
    name = "P0"

    @staticmethod
    def step_state(state, command):
        command = ACTION_SCALE * np.asarray(command, float)
        position, yaw = state["position"], state["yaw"]
        body = rot_z(yaw + 0.5 * command[3] * DT) @ state["body0"]
        position = position + body @ np.asarray(command, float)[:3] * DT
        yaw = yaw + float(command[3]) * DT
        return {"position": position, "yaw": yaw, "body0": state["body0"]}

    @staticmethod
    def initial_extra():
        return {}


class PlantP1:
    name = "P1"
    mix = 1 - math.exp(-DT / TAU)

    @staticmethod
    def step_state(state, command):
        actual = ((1 - PlantP1.mix) * state["actual_velocity"]
                  + PlantP1.mix * ACTION_SCALE * state["previous_command"])
        body = rot_z(state["yaw"] + 0.5 * actual[3] * DT) @ state["body0"]
        position = state["position"] + body @ actual[:3] * DT
        yaw = state["yaw"] + actual[3] * DT
        return {"position": position, "yaw": yaw, "body0": state["body0"],
                "actual_velocity": actual, "previous_command": np.asarray(command, float).copy()}

    @staticmethod
    def initial_extra():
        return {"actual_velocity": np.zeros(4), "previous_command": np.zeros(4)}


PLANTS = {"P0": PlantP0, "P1": PlantP1}
