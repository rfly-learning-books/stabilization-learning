"""Shared constants, paths, IO and small math helpers for the RGB stabilizing representation experiment.

Protocol values follow experiment_plan.md (version 1.0, 2026-10-15). Any deviation is
recorded in the interpretation journal (logs/interpretation_journal.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path(r"D:/书稿论文/output/stabilization_rgb_indoor_v1")
WORKSPACE = Path(r"D:/书稿论文")
OLD_AEVS_DIR = WORKSPACE / "output" / "aevs_fixed_scene_20261014"
OLD_DEPTH_DIR = WORKSPACE / "output" / "aevs_estimated_depth_20261014"
OLD_SCENES_DIR = WORKSPACE / "output" / "bright_natural_dense_servo_20261013"
OLD_DEPTH_MODEL_DIR = WORKSPACE / "output" / "li_depth_vs_fd_20261014" / "model"
OLD_DEPTH_RUNTIME = WORKSPACE / "output" / "li_depth_vs_fd_20261014" / "runtime_packages"
OLD_AEVS_CHECKPOINT = OLD_AEVS_DIR / "best.pt"
DEPTH_MODEL_FILE = OLD_DEPTH_MODEL_DIR / "model.safetensors"

import sys as _sys

if str(OLD_DEPTH_RUNTIME) not in _sys.path:
    _sys.path.insert(0, str(OLD_DEPTH_RUNTIME))

AEVS_CHECKPOINT_SHA256 = "2ab72ee1e673abfa3f2a7917705f9b4c8cc7a5bc5c1b0a00f77da9e834025758"
DEPTH_MODEL_SHA256 = "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3"

CONFIG_DIR = ROOT / "config"
MANIFEST_DIR = ROOT / "manifests"
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "cache"
RESULT_DIR = ROOT / "results"
VIDEO_DIR = ROOT / "videos"
AUDIT_DIR = ROOT / "audit"

SIZE = 256
NET_SIZE = 128
LATENT = 32
FOCAL = SIZE / (2 * math.tan(math.pi / 6))
YY, XX = np.mgrid[:SIZE, :SIZE].astype(np.float32)
X = (XX - (SIZE - 1) / 2) / FOCAL
Y = (YY - (SIZE - 1) / 2) / FOCAL
OPTICAL_TO_GL = np.diag([1.0, -1.0, -1.0])

DT = 0.1
TAU = 0.15
HORIZON = 20.0
STEPS = int(round(HORIZON / DT)) + 1
GAIN = 0.7
MU = 0.01
VMAX = 0.4
WMAX = 0.35
ACTION_SCALE = np.asarray([VMAX, VMAX, VMAX, WMAX], np.float32)
PIPELINE_VERSION = "rgb_sl_v2"
S_P = 1.0
S_YAW = math.radians(25.0)
ETA = 0.03
VC = 1.0
DELTA0 = 1e-6
VF = 1e-5
C1 = 0.25
C2 = 2.0
GAIN5 = 1 - (1 - ETA) ** 5
DELTA5 = 5 * DELTA0
COLLISION_RADIUS = 0.15
WORKSPACE_Z = (0.35, 3.05)

FAMILIES = ("classroom", "office", "meeting", "library")
TRAIN_SEEDS = (41, 42, 43)

LOSS_WEIGHTS = {
    "bound": 2.0,
    "anchor": 0.2,
    "dec": 1.0,
    "policy": 1.0,
    "tangent": 0.1,
    "model": 0.1,
    "zsmooth": 0.05,
    "usmooth": 0.05,
    "nuisance": 0.05,
    "rec": 0.02,
}

METHODS = ("PVS", "DOM120", "Multi-AE32", "Metric-Feat32", "SL-Core32", "SL-Full32")
LEARNED_METHODS = ("Multi-AE32", "Metric-Feat32", "SL-Core32", "SL-Full32")
METHOD_STAGE = {"Multi-AE32": "mixed_ae", "Metric-Feat32": "metric", "SL-Core32": "stabilizing_core", "SL-Full32": "stabilizing_full"}

STAGE_ORDER = ("mixed_ae", "metric", "stabilizing_core", "stabilizing_full")


def ensure_dirs():
    for d in (CONFIG_DIR, MANIFEST_DIR, LOG_DIR, CKPT_DIR, CACHE_DIR, RESULT_DIR, VIDEO_DIR, AUDIT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path, blocks=4 * 1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(blocks), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def progress(event, **kw):
    row = {"tag": event, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    append_jsonl(LOG_DIR / "journal.jsonl", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    goal = np.eye(4)
    goal[:3, :3] = np.column_stack([right, down, forward])
    goal[:3, 3] = eye
    return goal


def pose_error_squared(offset_xyz, yaw_rad):
    q = np.concatenate([np.asarray(offset_xyz, float) / S_P, [wrap_angle(float(yaw_rad)) / S_YAW]])
    return float(q @ q)


def seed_all(seed):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch():
    import torch

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def disable_tf32():
    import torch

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self):
        now = time.perf_counter()
        out = now - self.t0
        self.t0 = now
        return out
