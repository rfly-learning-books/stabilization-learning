"""Paired dense-flow IBVS with per-pixel flow and 16x16 block depth.

All mutations and outputs stay in this experiment directory. Existing packaged
scene functions, native source, and cached model weights are read-only.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import ast
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import platform
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CODE = ROOT / "视觉伺服_代码与经验_20261012" / "code" / "seese3"
SIZE, GRID, BLOCK = 256, 16, 16
FOCAL = SIZE / (2 * math.tan(math.pi / 6))
YY, XX = np.mgrid[:SIZE, :SIZE].astype(np.float32)
X, Y = (XX - (SIZE - 1) / 2) / FOCAL, (YY - (SIZE - 1) / 2) / FOCAL
RAYS = np.stack([X, Y, np.ones_like(X)], axis=-1)
OPTICAL_TO_GL = np.diag([1., -1., -1.])
FAMILIES = ["room", "office", "corridor", "warehouse", "depth_boundary"]
METHOD_LABELS = {
    "full_true": "逐像素真值深度",
    "block_inverse": "16×16 块深度：逆深度平均",
    "block_mean": "16×16 块深度：深度平均",
    "block_scale05": "块深度 ×0.5",
    "block_scale20": "块深度 ×2.0",
    "block_noise30": "块深度：逐帧 30% 随机误差",
    "block_noise60": "块深度：逐帧 60% 随机误差",
    "dpt_full": "DINOv2＋DPT 逐像素估计",
    "dpt_block": "DINOv2＋DPT 16×16 块估计",
}
CORE_METHODS = ["full_true", "block_inverse", "block_mean", "block_scale05", "block_scale20", "block_noise30", "block_noise60", "dpt_full", "dpt_block"]
IMAGE_METHODS = ["full_true", "block_inverse", "dpt_full", "dpt_block"]
STARTS = [[.18, -.12, .07, 6.], [-.22, .15, -.08, -7.], [.28, .18, .10, 9.]]


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def source_functions(path, names, ns):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError(f"Missing pure functions: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def pool(array):
    tail = array.shape[2:]
    return array.reshape(GRID, BLOCK, GRID, BLOCK, *tail).mean(axis=(1, 3))


def expand(array):
    return np.repeat(np.repeat(array, BLOCK, axis=0), BLOCK, axis=1)


class NVOFA:
    """NVIDIA driver optical-flow API; dense output keeps original pixel units."""
    def __init__(self, grid=1, preset=5):
        self.grid, self.preset = grid, preset
        self.lib = ct.CDLL(str(OUT / "native" / "nvofa_bridge.dll"))
        self.lib.nvofa_create.argtypes = [ct.c_int] * 4 + [ct.c_char_p, ct.c_size_t]
        self.lib.nvofa_create.restype = ct.c_void_p
        self.lib.nvofa_compute.argtypes = [ct.c_void_p] * 6 + [ct.c_char_p, ct.c_size_t]
        self.lib.nvofa_compute.restype = ct.c_int
        self.lib.nvofa_caps.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_uint32, ct.c_char_p, ct.c_size_t]
        self.lib.nvofa_caps.restype = ct.c_int
        self.lib.nvofa_destroy.argtypes = [ct.c_void_p]
        self.error = ct.create_string_buffer(2048)
        self.handle = self.lib.nvofa_create(SIZE, SIZE, grid, preset, self.error, len(self.error))
        if not self.handle:
            raise RuntimeError(self.error.value.decode(errors="replace"))
        self.times = []

    def caps(self):
        result = {}
        for name, cap in [("output_grid_sizes", 0), ("min_width", 4), ("min_height", 5)]:
            values = np.zeros(32, np.uint32)
            n = self.lib.nvofa_caps(self.handle, cap, values.ctypes.data, len(values), self.error, len(self.error))
            if n < 0:
                raise RuntimeError(self.error.value.decode(errors="replace"))
            result[name] = values[:n].tolist()
        return result

    def calc(self, rgb_a, rgb_b):
        a, b = [np.ascontiguousarray(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)) for x in (rgb_a, rgb_b)]
        n = (SIZE + self.grid - 1) // self.grid
        raw, cost, timing = np.empty((n, n, 2), np.int16), np.empty((n, n), np.uint8), np.zeros(4)
        status = self.lib.nvofa_compute(self.handle, a.ctypes.data, b.ctypes.data, raw.ctypes.data, cost.ctypes.data,
                                       timing.ctypes.data, self.error, len(self.error))
        if status:
            raise RuntimeError(self.error.value.decode(errors="replace"))
        self.times.append(timing[-1])
        flow = raw.astype(np.float32) / 32.0
        if self.grid != 1:
            flow = cv2.resize(flow, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            cost = cv2.resize(cost, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        return flow, cost

    def close(self):
        if self.handle:
            self.lib.nvofa_destroy(self.handle)
            self.handle = None


class DPT:
    def __init__(self):
        import torch
        import torch.nn.functional as F
        spec = importlib.util.spec_from_file_location("previous_depth_demo", ROOT / "output" / "depth_demo_windows_20261012" / "run_demo.py")
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        self.torch, self.F, self.device = torch, F, torch.device("cuda")
        self.model = previous.load_model(self.device)
        self.mean = torch.tensor([123.675, 116.28, 103.53], device=self.device)[None, :, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375], device=self.device)[None, :, None, None]
        self.checkpoints = [{"name": name, "sha256": sha(previous.WEIGHTS / name)} for name in previous.FILENAMES]
        self.times = []

    def calc(self, rgb):
        t, F = self.torch, self.F
        x = t.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float()[None]
        x = (F.interpolate(x, (224, 224), mode="bilinear", align_corners=False) - self.mean) / self.std
        with t.inference_mode():
            t.cuda.synchronize()
            begin = time.perf_counter()
            z = self.model.encode_decode(x, img_metas=None, rescale=True, size=(224, 224))
            z = F.interpolate(z.float(), (SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
            t.cuda.synchronize()
            self.times.append((time.perf_counter() - begin) * 1000)
            return np.clip(z.cpu().numpy(), .10, 30.)


class Scene:
    def __init__(self, family, pilot=False):
        self.family = family
        family_id = FAMILIES.index(family)
        seed = 2026101300 + family_id * 100 + (90000 if pilot else 0)
        self.spec = {"name": f"blockdepth_{family}_{seed}", "family": family, "layout_seed": seed, "pose_seed": seed + 37}
        ns = {"np": np, "OPTICAL_TO_GL": OPTICAL_TO_GL}
        source_functions(CODE / "servo_benchmark.py", {"look_at"}, ns)
        source_functions(CODE / "build_depth_cross_scene_dataset.py", {"build_scene", "optical_hub"}, ns)
        if family != "depth_boundary":
            xml, dims = ns["build_scene"](self.spec)
            self.goal = ns["optical_hub"](1, dims["width_m"], dims["depth_m"], np.random.default_rng(self.spec["pose_seed"]))
        else:
            dims = {"width_m": 8., "depth_m": 12., "height_m": 3.}
            geoms = ['<geom name="floor" type="box" pos="0 1 -.05" size="4 6 .05" rgba=".7 .7 .7 1"/>',
                     '<geom name="back" type="box" pos="0 5 1.5" size="4 .05 1.5" rgba=".85 .82 .78 1"/>',
                     '<geom name="ceiling" type="box" pos="0 1 3.05" size="4 6 .05" rgba=".8 .8 .8 1"/>']
            for i, x in enumerate(np.linspace(-2., 2., 17)):
                y = -.20 + .12 * (i % 3)
                geoms.append(f'<geom name="bar_{i}" type="box" pos="{x} {y} 1.25" size=".065 .06 1.1" rgba=".35 .48 .60 1"/>')
            xml = '<mujoco><compiler angle="radian"/><visual><global offwidth="512" offheight="512"/><quality offsamples="0" shadowsize="512"/></visual><worldbody><light pos="-2 -2 6" diffuse=".85 .85 .85"/>' + ''.join(geoms) + '<camera name="depth_camera" fovy="60"/></worldbody></mujoco>'
            self.goal = ns["look_at"]([0, -3., 1.3], [0, 3., 1.3])
            self.goal[:3, :3] = self.goal[:3, :3] @ OPTICAL_TO_GL
        self.dims = dims
        rng = np.random.default_rng(seed + 781)
        coarse = rng.uniform(.20, .90, (64, 64, 3))
        texture = cv2.resize(coarse, (512, 512), interpolation=cv2.INTER_LINEAR)
        texture = np.clip(texture + rng.normal(0, .035, texture.shape), 0, 1)
        texture = (255 * texture).astype(np.uint8)
        asset_name = "benchmark_texture.png"
        buffer = cv2.imencode(".png", cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))[1].tobytes()
        assets = '<asset><texture name="flow_texture" type="2d" file="benchmark_texture.png"/><material name="flow_material" texture="flow_texture" texrepeat="1 1" texuniform="true" reflectance="0"/></asset>'
        xml = xml.replace("<worldbody>", assets + "<worldbody>").replace("<geom ", '<geom material="flow_material" ')
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml, assets={asset_name: buffer})
        self.model.vis.headlight.ambient[:] = .30
        self.model.vis.headlight.diffuse[:] = .65
        self.data = mujoco.MjData(self.model)
        self.cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
        self.renderer = mujoco.Renderer(self.model, SIZE, SIZE)
        forward = self.goal[:3, 2].copy()
        forward[2] = 0
        forward /= np.linalg.norm(forward)
        left = np.cross([0., 0., 1.], forward)
        self.body0 = np.column_stack([forward, left, [0., 0., 1.]])
        self.camera_from_body = self.goal[:3, :3].T @ self.body0
        self.mount_yaw = self.camera_from_body[:, 2]
        self.trans = np.empty((SIZE, SIZE, 2, 3), np.float32)
        a = np.zeros_like(self.trans)
        a[..., 0, 0], a[..., 0, 2] = -1., X
        a[..., 1, 1], a[..., 1, 2] = -1., Y
        self.trans[:] = a @ self.camera_from_body
        kx, ky, kz = self.mount_yaw
        self.rot = np.stack([X * Y * kx - (1 + X ** 2) * ky + Y * kz,
                             (1 + Y ** 2) * kx - X * Y * ky - X * kz], axis=-1).astype(np.float32)
        self.goal_rgb, self.goal_depth = self.render(self.goal[:3, 3], 0.)
        self.goal_gray = cv2.cvtColor(self.goal_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def pose(self, position, yaw):
        T = self.goal.copy()
        T[:3, :3] = rot_z(yaw) @ self.goal[:3, :3]
        T[:3, 3] = position
        return T

    def render(self, position, yaw):
        pose = self.pose(position, yaw)
        gl = pose[:3, :3] @ OPTICAL_TO_GL
        q = Rotation.from_matrix(gl).as_quat()
        self.model.cam_pos[self.cid] = position
        self.model.cam_quat[self.cid] = q[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cid)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cid)
        depth = self.renderer.render().copy().astype(np.float32)
        self.renderer.disable_depth_rendering()
        return rgb, np.clip(depth, .10, 80.)

    def oracle_flow(self, position, yaw, depth):
        """Geometric ideal sensor ONLY for the explicitly named oracle arm."""
        current = self.pose(position, yaw)
        xyz = RAYS * depth[..., None]
        relative_R = self.goal[:3, :3].T @ current[:3, :3]
        relative_t = self.goal[:3, :3].T @ (position - self.goal[:3, 3])
        target = xyz @ relative_R.T + relative_t
        uv = target[..., :2] / np.maximum(target[..., 2:3], .001) * FOCAL + (SIZE - 1) / 2
        flow = uv - np.stack([XX, YY], axis=-1)
        target_z = cv2.remap(self.goal_depth, uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32),
                            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = (target[..., 2] > .1) & (uv[..., 0] > 2) & (uv[..., 0] < SIZE - 3) & (uv[..., 1] > 2) & (uv[..., 1] < SIZE - 3)
        valid &= np.abs(target_z - target[..., 2]) < np.maximum(.035, .02 * target[..., 2])
        return flow.astype(np.float32), valid.astype(np.float32)

    def close(self):
        self.renderer.close()


def image_sensor(flow_front, rgb, goal_rgb):
    """Only RGB enters this arm; masks never use simulator pose/depth."""
    flow, cost = flow_front.calc(rgb, goal_rgb)
    backward, _ = flow_front.calc(goal_rgb, rgb)
    ux, uy = XX + flow[..., 0], YY + flow[..., 1]
    back_warped = cv2.remap(backward, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    fb = np.linalg.norm(flow + back_warped, axis=-1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    goal_gray = cv2.cvtColor(goal_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    photo = np.abs(gray - cv2.remap(goal_gray, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    valid = (ux > 3) & (ux < SIZE - 4) & (uy > 3) & (uy < SIZE - 4)
    valid &= (fb < (1.25 + .035 * np.linalg.norm(flow, axis=-1))) & (photo < 35) & (gradient > 3)
    weights = valid.astype(np.float32) / (1 + cost.astype(np.float32) / 32.)
    weights *= np.exp(-.5 * (fb / 1.5) ** 2)
    return flow, weights, {"fb_median_px": float(np.median(fb[valid])) if valid.any() else 1000.,
                           "photo_median": float(np.median(photo[valid])) if valid.any() else 1000.}


def depth_observation(method, true_depth, rgb, predictor, rng):
    if method == "full_true":
        return true_depth.copy()
    if method.startswith("dpt_"):
        predicted = predictor.calc(rgb)
        if method == "dpt_full":
            return predicted
        return expand(pool(predicted))
    if method == "block_mean":
        coarse = pool(true_depth)
    else:
        coarse = 1 / pool(1 / true_depth)
    if method == "block_scale05":
        coarse *= .5
    elif method == "block_scale20":
        coarse *= 2
    elif method in ("block_noise30", "block_noise60"):
        sigma = .30 if method == "block_noise30" else .60
        multiplier = rng.lognormal(-.5 * sigma ** 2, sigma, (GRID, GRID))
        coarse *= multiplier
    return expand(np.clip(coarse, .10, 80.)).astype(np.float32)


def solve_control(flow, observed_depth, weights, scene, gain=.6, damping=1e-5):
    # Dense 65536 pixel samples remain dense for every depth condition.
    J = np.empty((SIZE, SIZE, 2, 4), np.float32)
    J[..., :3] = scene.trans / observed_depth[..., None, None]
    J[..., 3] = scene.rot
    error = -flow / FOCAL
    w = weights.reshape(-1).copy()
    if np.count_nonzero(w > .01) < 500:
        return np.zeros(4), {"valid_pixels": int(np.count_nonzero(w > .01)), "condition": 1e9, "flow_rms_px": 1000.}
    a, e = J.reshape(-1, 2, 4), error.reshape(-1, 2)
    # Same Huber reweighting for all conditions; no oracle residual enters it.
    command = np.zeros(4)
    for _ in range(2):
        denom = max(float(w.sum()), 1.)
        af = a.reshape(-1, 4)
        wf = np.repeat(w, 2)
        H = af.T @ (af * wf[:, None]) / denom
        b = af.T @ (e.reshape(-1) * wf) / denom
        command = -gain * np.linalg.solve(H + damping * np.eye(4), b)
        residual_px = np.linalg.norm(np.einsum("nij,j->ni", a, command) + gain * e, axis=-1) * FOCAL
        w = weights.reshape(-1) * np.minimum(1., 2. / np.maximum(residual_px, .001))
    command[:3] *= min(1., .40 / max(np.linalg.norm(command[:3]), 1e-8))
    command[3] = np.clip(command[3], -.35, .35)
    eig = np.linalg.eigvalsh(H)
    stats = {"valid_pixels": int(np.count_nonzero(weights > .01)),
             "condition": float(math.sqrt(max(eig[-1], 1e-10) / max(eig[0], 1e-10))),
             "flow_rms_px": float(np.sqrt(np.sum(weights * np.sum(flow ** 2, axis=-1)) / max(weights.sum(), 1.)))}
    return command, stats


def verify(scene, nvofa):
    rng = np.random.default_rng(42)
    z = rng.uniform(1, 8, (SIZE, SIZE)).astype(np.float32)
    block = expand(1 / pool(1 / z))
    assert block.shape == (SIZE, SIZE)
    assert np.max(np.std(block.reshape(GRID, BLOCK, GRID, BLOCK), axis=(1, 3))) < 1e-5
    flow, mask = scene.oracle_flow(scene.goal[:3, 3], 0, scene.goal_depth)
    identity_error = float(np.max(np.abs(flow)))
    assert identity_error < 1e-4
    # Independent finite-difference projection of fixed 3D points checks all four columns.
    points_world = (RAYS.astype(np.float64) * scene.goal_depth[..., None]) @ scene.goal[:3, :3].T + scene.goal[:3, 3]
    good = (scene.goal_depth > .2) & (scene.goal_depth < 40)
    errors = []
    eps = 1e-5
    J = np.concatenate([scene.trans / scene.goal_depth[..., None, None], scene.rot[..., None]], axis=-1)
    for axis in range(4):
        command = np.zeros(4)
        command[axis] = 1
        p = scene.goal[:3, 3] + scene.body0 @ command[:3] * eps
        T = scene.pose(p, command[3] * eps)
        xyz = (points_world - p) @ T[:3, :3]
        uv = xyz[..., :2] / xyz[..., 2:3]
        fd = (uv - RAYS[..., :2]) / eps
        errors.append(float(np.sqrt(np.mean((fd[good] - J[good, :, axis]) ** 2))))
    assert max(errors) < 2e-4, errors
    identity_nvofa, _ = nvofa.calc(scene.goal_rgb, scene.goal_rgb)
    nvofa_median = float(np.median(np.linalg.norm(identity_nvofa, axis=-1)))
    assert nvofa_median < .2, nvofa_median
    c = solve_control(flow, scene.goal_depth, mask, scene)[0]
    assert np.linalg.norm(c) < 1e-4
    report = {"passed": True, "identity_geometric_flow_max_px": identity_error,
              "interaction_matrix_fd_rmse_by_body_axis": errors, "nvofa_identity_flow_median_px": nvofa_median,
              "dense_flow_size": [SIZE, SIZE, 2], "depth_size": [GRID, GRID], "depth_block_coverage": [BLOCK, BLOCK],
              "flow_is_downsampled_in_controller": False, "nvofa_caps": nvofa.caps(), "nvofa_native_grid": nvofa.grid}
    json_write(OUT / "verification.json", report)
    return report


def initial_pose(scene, start):
    return scene.goal[:3, 3] + scene.body0 @ np.asarray(start[:3]), math.radians(start[3])


def run_episode(scene, trial, method, front_kind, nvofa, predictor, steps, save_frames=False):
    position, yaw = initial_pose(scene, STARTS[trial])
    actual_velocity, previous_command = np.zeros(4), np.zeros(4)
    dt, tau = .10, .15
    mix = 1 - math.exp(-dt / tau)
    rng = np.random.default_rng(801 + FAMILIES.index(scene.family) * 1000 + trial * 100)
    rows, frames, positions = [], [], []
    first = None
    failure = None
    begin = time.perf_counter()
    for t in range(steps):
        rgb, depth = scene.render(position, yaw)
        observed_depth = depth_observation(method, depth, rgb, predictor, rng)
        if front_kind == "oracle":
            flow, weights = scene.oracle_flow(position, yaw, depth)
        else:
            flow, weights, _ = image_sensor(nvofa, rgb, scene.goal_rgb)
        command, info = solve_control(flow, observed_depth, weights, scene)
        error_m = float(np.linalg.norm(position - scene.goal[:3, 3]))
        yaw_deg = abs(math.degrees(yaw))
        absrel = float(np.mean(np.abs(observed_depth - depth) / depth))
        rows.append([t * dt, error_m, yaw_deg, info["flow_rms_px"], absrel, info["valid_pixels"], info["condition"], *command])
        positions.append([*position, yaw])
        if first is None:
            first = {"rgb": rgb.copy(), "gt": depth.copy(), "observed": observed_depth.copy(), "flow": flow.copy(), "weights": weights.copy()}
        if save_frames and (t % 3 == 0 or t == steps - 1):
            frames.append(rgb.copy())
        if error_m > 2.5 or yaw_deg > 60 or info["valid_pixels"] < 500:
            failure = "escaped" if error_m > 2.5 or yaw_deg > 60 else "insufficient_correspondences"
            break
        # Camera actuation only: first-order flight-controller lag and one-step command delay.
        actual_velocity = (1 - mix) * actual_velocity + mix * previous_command
        body_rotation = rot_z(yaw + .5 * actual_velocity[3] * dt) @ scene.body0
        position = position + body_rotation @ actual_velocity[:3] * dt
        yaw = yaw + actual_velocity[3] * dt
        previous_command = command.copy()
    history = np.asarray(rows, np.float64)
    tail = history[-min(20, len(history)):]
    good = (history[:, 1] < .02) & (history[:, 2] < 1.)
    strict = (history[:, 1] < .01) & (history[:, 2] < 1.)
    settle = None
    for i in range(len(good)):
        if np.all(good[i:]) and len(good) - i >= 20:
            settle = float(history[i, 0])
            break
    energy = (history[:, 1] / .2) ** 2 + (np.deg2rad(history[:, 2]) / .15) ** 2
    result = {"scene": scene.family, "trial": trial, "frontend": front_kind, "method": method,
              "steps": len(history), "failure": failure, "success_2cm_1deg_hold": bool(failure is None and np.all(good[-20:])),
              "success_1cm_1deg_hold": bool(failure is None and np.all(strict[-20:])),
              "initial_translation_m": float(history[0, 1]), "initial_yaw_deg": float(history[0, 2]),
              "final_translation_m": float(history[-1, 1]), "final_yaw_deg": float(history[-1, 2]),
              "hold_translation_max_m": float(tail[:, 1].max()), "hold_yaw_max_deg": float(tail[:, 2].max()),
              "settle_s": settle, "mean_depth_absrel": float(history[:, 4].mean()),
              "min_valid_pixels": int(history[:, 5].min()), "mean_valid_pixels": float(history[:, 5].mean()),
              "pose_energy_increase_fraction": float(np.mean(np.diff(energy) > 1e-5)) if len(energy) > 1 else 1.,
              "command_total_variation": float(np.linalg.norm(np.diff(history[:, 7:11], axis=0), axis=1).sum()),
              "runtime_s": time.perf_counter() - begin}
    name = f"{front_kind}__{scene.family}__trial{trial}__{method}"
    np.savez_compressed(OUT / "traces" / f"{name}.npz", history=history, positions=np.asarray(positions), goal_pose=scene.goal,
                        columns=np.array(["time_s", "translation_m", "yaw_deg", "flow_rms_px", "depth_absrel", "valid_pixels", "condition", "vx_cmd", "vy_cmd", "vz_cmd", "yawrate_cmd"]))
    if save_frames:
        np.savez_compressed(OUT / "demos" / f"{name}.npz", frames=np.stack(frames), history=history,
                            goal_rgb=scene.goal_rgb, **first)
    return result


def summarize(results):
    summary = {}
    for front in ("oracle", "nvofa"):
        summary[front] = {}
        for method in METHOD_LABELS:
            subset = [r for r in results if r["frontend"] == front and r["method"] == method]
            if not subset:
                continue
            values = np.array([r["final_translation_m"] for r in subset])
            times = [r["settle_s"] for r in subset if r["settle_s"] is not None]
            summary[front][method] = {"n": len(subset), "success_2cm_1deg_hold": sum(r["success_2cm_1deg_hold"] for r in subset),
                                    "success_1cm_1deg_hold": sum(r["success_1cm_1deg_hold"] for r in subset),
                                    "median_final_translation_cm": float(np.median(values) * 100),
                                    "max_final_translation_cm": float(values.max() * 100),
                                    "median_final_yaw_deg": float(np.median([r["final_yaw_deg"] for r in subset])),
                                    "median_settle_s_successes_only": float(np.median(times)) if times else None,
                                    "mean_depth_absrel": float(np.mean([r["mean_depth_absrel"] for r in subset])),
                                    "failures": sum(r["failure"] is not None for r in subset)}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["smoke", "core", "image", "all"], default="all")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--method", choices=list(METHOD_LABELS))
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    for d in ("traces", "demos", "scenes"):
        (OUT / d).mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NVOFA/DPT sensor arm")
    front = NVOFA(grid=1)
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if args.resume and results_path.exists() else []
    completed = {(r["frontend"], r["scene"], r["trial"], r["method"]) for r in results}
    predictor = None
    scene = None
    try:
        pilot = Scene("room", pilot=True)
        checks = verify(pilot, front)
        print("VERIFIED", json.dumps(checks, ensure_ascii=False), flush=True)
        if args.stage == "smoke":
            predictor = DPT()
            for method in ("full_true", "block_inverse", "dpt_block"):
                record = run_episode(pilot, 0, method, "nvofa", front, predictor, args.steps, True)
                print("SMOKE", json.dumps(record, ensure_ascii=False), flush=True)
            pilot.close()
            return
        pilot.close()
        stages = ["oracle", "nvofa"] if args.stage == "all" else ["oracle" if args.stage == "core" else "nvofa"]
        protocol = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "image_size": [SIZE, SIZE],
                    "dense_flow_shape": [SIZE, SIZE, 2], "depth_grid": [GRID, GRID], "depth_block_pixels": [BLOCK, BLOCK],
                    "flow_downsampled_in_controller": False, "nvofa_native_grid": front.grid, "nvofa_preset": "SLOW",
                    "flow_unit": "original RGB pixels per current-to-reference correspondence; not temporal velocity",
                    "controller": "weighted dense point IBVS; analytic interaction matrix; body FLU velocity + world-vertical yaw rate",
                    "gain": .6, "damping": 1e-5, "huber_residual_threshold_px": 2., "translation_limit_m_s": .4, "yaw_limit_rad_s": .35,
                    "plant": "kinematic camera with first-order velocity response (tau=.15 s), one-step command delay; no rigid-body/collision flight simulation",
                    "dt_s": .1, "steps": args.steps, "starts_body_m_yaw_deg": STARTS, "families": FAMILIES,
                    "layout_seed_rule": "2026101300 + family_index*100; independent pilot adds 90000",
                    "oracle_depth_profiles": CORE_METHODS, "rgb_sensor_depth_profiles": IMAGE_METHODS,
                    "noise": "unbiased multiplicative lognormal block noise; sigma=.30 or .60; new errors each frame; paired seeds",
                    "dpt": "official frozen DINOv2-B/14 + NYU DPT; RGB256 resized224; output256; arithmetic mean16x16 + nearest broadcast; no GT scale correction",
                    "success": "position <2cm and yaw <1deg on ALL last20 samples; strict position <1cm",
                    "controller_pose_access": False, "rgb_mask_gt_access": False,
                    "oracle_frontend": "exact geometric correspondence and visibility synthesized by simulator; uses GT only inside explicitly labeled oracle sensor",
                    "evaluation": "simulator pose errors independently of selected pixels; paired episodes, same gain/masks/frontend per comparison",
                    "python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "mujoco": mujoco.__version__, "script_sha256": sha(Path(__file__)), "bridge_sha256": sha(OUT / "native" / "nvofa_bridge.dll")}
        if not (OUT / "protocol.json").exists():
            json_write(OUT / "protocol.json", protocol)
        families = [args.family] if args.family else FAMILIES
        for front_kind in stages:
            methods = [args.method] if args.method else CORE_METHODS if front_kind == "oracle" else IMAGE_METHODS
            if any(m.startswith("dpt") for m in methods) and predictor is None:
                print("Loading cached official DPT model", flush=True)
                predictor = DPT()
                json_write(OUT / "model_provenance.json", {"checkpoints": predictor.checkpoints, "gt_scale_alignment": False})
            for family in families:
                scene = Scene(family)
                Image.fromarray(scene.goal_rgb).save(OUT / "scenes" / f"{family}_reference.png")
                np.save(OUT / "scenes" / f"{family}_reference_depth.npy", scene.goal_depth)
                json_write(OUT / "scenes" / f"{family}_manifest.json", {"spec": scene.spec, "dimensions": scene.dims, "goal_camera_to_world": scene.goal.tolist()})
                for trial in range(args.trials):
                    for method in methods:
                        key = (front_kind, family, trial, method)
                        if key in completed:
                            continue
                        record = run_episode(scene, trial, method, front_kind, front, predictor, args.steps,
                                             save_frames=(trial == 0 and method in ("full_true", "block_inverse", "dpt_block")))
                        results.append(record)
                        completed.add(key)
                        json_write(results_path, results)
                        json_write(OUT / "summary.json", summarize(results))
                        json_write(OUT / "status.json", {"last": record, "completed_episodes": len(results), "phase": front_kind})
                        print("RESULT", json.dumps(record, ensure_ascii=False), flush=True)
                scene.close()
                scene = None
        timing = {"nvofa_calls": len(front.times), "nvofa_median_ms": float(np.median(front.times)),
                  "dpt_calls": len(predictor.times) if predictor else 0,
                  "dpt_median_ms": float(np.median(predictor.times)) if predictor else None}
        json_write(OUT / f"timing_{args.stage}.json", timing)
        json_write(OUT / "status.json", {"phase": "completed", "completed_episodes": len(results)})
        print("SUMMARY", json.dumps(summarize(results), ensure_ascii=False), flush=True)
    finally:
        if scene:
            scene.close()
        front.close()


if __name__ == "__main__":
    main()
