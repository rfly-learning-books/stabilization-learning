"""Parameterized room instances for classroom / office / meeting room / library.

Reuses the frozen Builder from bright_natural_dense_servo_20261013 (read only) for
materials, meshes and MJCF assembly. Every solid furniture piece is also recorded as
an axis-aligned obstacle box for evaluation-only collision checking.
"""
from __future__ import annotations

import importlib.util
import math

import numpy as np

from .common import FAMILIES, OLD_SCENES_DIR

_BUILDER = None


def _builder_class():
    global _BUILDER
    if _BUILDER is None:
        spec = importlib.util.spec_from_file_location("bright_scenes_readonly", OLD_SCENES_DIR / "scenes.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BUILDER = module.Builder
    return _BUILDER


def floats(values):
    return " ".join(f"{float(x):.7g}" for x in values)


class RoomBuilder:
    def __init__(self, seed):
        self.b = _builder_class()(outdoor=False, seed=seed)
        self.obstacles = []

    def solid(self, pos, half, material, yaw=0.0):
        self.b.box(pos, half, material, yaw=yaw)
        self.obstacles.append({"center": [float(v) for v in pos],
                               "halfsize": [float(v) for v in half]})

    def decor(self, pos, half, material, yaw=0.0):
        self.b.box(pos, half, material, yaw=yaw)

    def cylinder(self, a, b, radius, material="metal"):
        self.b.cylinder(a, b, radius, material)

    def prop(self, identifier, pos, yaw=0.0, scale=1.0, obstacle=None):
        self.b.prop(identifier, pos, yaw=yaw, scale=scale)
        if obstacle is not None:
            self.obstacles.append({"center": [float(v) for v in pos],
                                   "halfsize": [float(v) for v in obstacle]})

    def plant(self, x, y, scale=1.0):
        self.b.plant(x, y, scale)
        self.obstacles.append({"center": [float(x), float(y), 0.25],
                               "halfsize": [0.2 * scale, 0.2 * scale, 0.25]})

    def desk(self, cx, cy, yaw=0.0, variant=0, chair=True):
        self.solid((cx, cy, 0.73), (0.62, 0.36, 0.035), "wood", yaw=yaw)
        self.solid((cx, cy, 0.67), (0.56, 0.30, 0.025), "trim", yaw=yaw)
        for dx in (-0.52, 0.52):
            for dy in (-0.27, 0.27):
                self.cylinder((cx + dx, cy + dy, 0.03), (cx + dx, cy + dy, 0.70), 0.024)
        if chair:
            self.prop("plastic_monobloc_chair_01", (cx, cy - 0.66, 0), yaw=math.pi,
                      obstacle=(0.34, 0.34, 0.44))
        if variant % 3 == 0:
            self.decor((cx + 0.14, cy + 0.05, 0.78), (0.17, 0.12, 0.015), "blue")
            self.decor((cx + 0.14, cy + 0.045, 0.80), (0.163, 0.115, 0.004), "white")
        elif variant % 3 == 1:
            self.decor((cx - 0.08, cy, 0.77), (0.19, 0.14, 0.005), "white")

    def office_desk(self, cx, cy, yaw=0.0, monitor=True):
        self.solid((cx, cy, 0.72), (0.72, 0.42, 0.035), "wood", yaw=yaw)
        for dx in (-0.6, 0.6):
            for dy in (-0.32, 0.32):
                self.cylinder((cx + dx, cy + dy, 0.03), (cx + dx, cy + dy, 0.69), 0.028)
        if monitor:
            self.decor((cx, cy + 0.18, 1.06), (0.28, 0.03, 0.19), "metal", yaw=yaw)
            self.decor((cx, cy + 0.16, 0.76), (0.13, 0.09, 0.012), "white", yaw=yaw)
        self.prop("plastic_monobloc_chair_01", (cx, cy - 0.72, 0), yaw=math.pi,
                  obstacle=(0.36, 0.36, 0.44))

    def cabinet(self, cx, cy, yaw=0.0, width=0.5, height=0.65):
        self.solid((cx, cy, height), (width, 0.24, height), "metal", yaw=yaw)
        for k in range(2):
            self.decor((cx, cy - 0.245, 0.35 + 0.6 * k), (width * 0.92, 0.01, 0.03), "trim", yaw=yaw)

    def bookshelf(self, cx, cy, yaw=0.0, half_len=0.9, height=0.95, levels=3):
        self.solid((cx, cy, height), (0.18, half_len, height), "wood", yaw=yaw)
        for li in range(levels):
            z = 0.32 + li * (1.55 / max(levels - 1, 1))
            self.decor((cx, cy, z), (0.155, half_len - 0.06, 0.02), "wood", yaw=yaw)
            for j in range(int(half_len * 9)):
                offset = -half_len + 0.14 + j * (2 * (half_len - 0.14) / max(int(half_len * 9) - 1, 1))
                h = float(np.clip(0.24 + 0.16 * ((j * 7) % 5) / 4 + 0.05 * ((j * 3) % 2), 0.2, 0.38))
                material = "blue" if j % 3 == 0 else "ochre" if j % 3 == 1 else "white"
                self.decor((cx - 0.02, cy + offset, z + h / 2 + 0.02), (0.1, 0.05, h / 2), material, yaw=yaw)
        self.decor((cx, cy, 0.02), (0.2, half_len, 0.02), "wood", yaw=yaw)


def _window_wall(rb, x, hl, height, n_win, door=False):
    half_h = height / 2
    rb.decor((x, 0, 0.45), (0.1, hl, 0.45), "wall_light")
    rb.decor((x, 0, height - 0.3), (0.1, hl, 0.3), "wall_light")
    span = 2 * hl / (n_win + 1)
    for w in range(n_win + 1):
        y = -hl + w * span
        center_z = (0.45 + (height - 0.3)) / 2
        rb.decor((x, y, center_z), (0.1, 0.09, (height - 0.3 - 0.45) / 2), "white")
    for w in range(n_win):
        y = -hl + (w + 0.5) * span
        rb.decor((x + 0.03, y, 1.6), (0.035, 0.035, 1.05), "trim")
        rb.decor((x + 0.05, y, 1.6), (0.035, span * 0.46, 0.028), "trim")
    if door:
        rb.decor((x, hl - 0.7, 1.05), (0.11, 0.45, 1.0), "trim")
        rb.decor((x + 0.03, hl - 0.7, 1.05), (0.02, 0.38, 0.92), "wood")


def _plain_wall(rb, pos, half):
    rb.decor(pos, half, "wall_light")


def _trim(rb, x, hl, height):
    for side in (-1, 1):
        rb.decor((x + side * 0.001, 0, 0.09), (0.0 + 0.035, hl, 0.09), "trim")



# Everything below this marker is an explicit reconstruction. It is NOT the
# unavailable original scene_builders.py and is governed by a new protocol.
import copy
import json
from pathlib import Path

PROTOCOL_KIND = "reconstructed_environment"
ORIGINAL_RGB_EQUIVALENT = False
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "scenes.json"


def _records():
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {row["id"]: row for row in data["scenes"]}


def build_shell(rb, hw, hl, height, window_side, n_win=4):
    """Inferred wall/ceiling layout, using recovered window-wall helpers."""
    _plain_wall(rb, (0, hl + .1, height / 2), (hw + .1, .1, height / 2))
    _plain_wall(rb, (0, -hl - .1, height / 2), (hw + .1, .1, height / 2))
    wx = -(hw + .1) if window_side == "left" else hw + .1
    _window_wall(rb, wx, hl, height, n_win)
    _plain_wall(rb, (-wx, 0, height / 2), (.1, hl, height / 2))
    rb.decor((0, 0, height + .05), (hw + .1, hl + .1, .05), "white")
    for x in (-hw + .05, hw - .05):
        _trim(rb, x, hl, height)


def _near(values, expected):
    return np.allclose(values, expected, rtol=0, atol=1e-12)


def _chair_yaw(scene, pos):
    if scene["family"] in ("classroom", "office"):
        return math.pi
    if scene["family"] == "meeting":
        return math.pi if pos[1] < 0 else 0.0
    table = min(scene["params"]["tables"], key=lambda xy: np.linalg.norm(np.asarray(xy) - np.asarray(pos[:2])))
    return math.pi if pos[1] < table[1] else 0.0


def _furniture_from_manifest(rb, scene):
    """Keep every historical collision box; infer only visual decoration.

    The original collision model is axis-aligned and does not rotate AABBs for
    yawed meshes. That behavior is deliberately preserved, not 'corrected'.
    """
    obs = scene["obstacles"]
    i, desk_index = 0, 0
    while i < len(obs):
        row = obs[i]
        pos, half = row["center"], row["halfsize"]
        x, y, z = pos
        if _near(half, [.62, .36, .035]) and abs(z - .73) < 1e-12:
            rb.desk(x, y, variant=desk_index)
            desk_index += 1
            i += 3
            continue
        if _near(half, [.72, .42, .035]) and abs(z - .72) < 1e-12:
            spec = min(scene["params"]["desks"], key=lambda d: (d[0] - x) ** 2 + (d[1] - y) ** 2)
            rb.office_desk(x, y, yaw=spec[2], monitor=True)
            i += 2
            continue
        if abs(half[0] - .18) < 1e-12 and abs(z - .95) < 1e-12 and abs(half[2] - .95) < 1e-12:
            rb.bookshelf(x, y, half_len=half[1])
        elif abs(z - .25) < 1e-12 and abs(half[2] - .25) < 1e-12:
            rb.plant(x, y, scale=half[0] / .2)
        elif abs(z) < 1e-12 and abs(half[2] - .44) < 1e-12:
            rb.prop("plastic_monobloc_chair_01", pos, yaw=_chair_yaw(scene, pos), obstacle=half)
        elif _near(half, [.28, .24, .65]):
            rb.cabinet(x, y, width=half[0], height=half[2])
        else:
            material = "metal" if abs(z - .35) < 1e-12 else "wood"
            rb.solid(pos, half, material)
            if scene["family"] == "library" and _near(half, [.8, .5, .035]):
                for dx in (-.68, .68):
                    for dy in (-.38, .38):
                        rb.cylinder((x + dx, y + dy, .03), (x + dx, y + dy, .70), .025)
        i += 1
    if len(rb.obstacles) != len(obs):
        raise RuntimeError("Reconstruction changed obstacle count: " + scene["id"])
    for got, expected in zip(rb.obstacles, obs):
        if not _near(got["center"], expected["center"]) or not _near(got["halfsize"], expected["halfsize"]):
            raise RuntimeError("Reconstruction changed an AABB: " + scene["id"])
    # Return original floating-point payloads exactly, including last-bit values.
    rb.obstacles = copy.deepcopy(obs)


def _visual_decoration(rb, scene):
    """Inferred non-collision appearance. Never optimized against held-out RGB."""
    p, family = scene["params"], scene["family"]
    hw, hl, height = p["hw"], p["hl"], scene["dims"]["height"]
    floor = p.get("floor", "concrete" if family == "office" else "wood" if family == "library" else "floor_light")
    rb.decor((0, 0, -.08), (hw, hl, .08), floor)
    build_shell(rb, hw, hl, height, p["window_side"], {"classroom": 4, "office": 3, "meeting": 3, "library": 5}[family])
    if family == "classroom":
        rb.decor((0, hl - .06, 1.86), (2.12, .04, .89), "metal")
        rb.decor((0, hl - .115, 1.86), (2.05, .012, .83), "board")
        rb.decor((hw - 1.2, hl - .08, 2.1), (.43, .025, .56), "poster")
        rb.decor((-hw + 1.1, hl - .09, 2.6), (.30, .02, .30), "clock")
    elif family == "office":
        rb.decor((.6, hl - .07, 1.85), (.55, .025, .65), "poster")
    elif family == "meeting":
        if p.get("screen", False):
            rb.decor((0, hl - .08, 1.9), (1.35, .04, .75), "metal")
        else:
            rb.decor((0, hl - .08, 1.9), (1.35, .04, .75), "board")
    else:
        rb.decor((-hw + .9, hl - .08, 2.55), (.30, .02, .30), "clock")
    for y in np.linspace(-hl * .55, hl * .55, 3):
        rb.decor((0, y, height - .04), (.85, .12, .018), "white")


def make_scene_specs():
    """Frozen historical layouts only: no guessed sampler or newly drawn seeds."""
    return [copy.deepcopy({k: row[k] for k in ("id", "split", "family", "seed", "params")}) for row in _records().values()]


def build_instance(spec):
    scene = _records().get(spec["id"])
    if scene is None:
        raise ValueError("Only the 44 frozen scene IDs are supported")
    for key in ("id", "split", "family", "seed", "params", "refs", "obstacles", "dims"):
        if key in spec and spec[key] != scene[key]:
            raise ValueError("Frozen scene field changed: " + key)
    rb = RoomBuilder(seed=scene["seed"])
    _visual_decoration(rb, scene)
    _furniture_from_manifest(rb, scene)
    height = scene["dims"]["height"]
    hw, hl = scene["params"]["hw"], scene["params"]["hl"]
    lights = []
    for x, y in [(-3, -3), (3, 1), (0, 4)]:
        x = float(np.clip(x, -hw + .6, hw - .6))
        y = float(np.clip(y, -hl + .6, hl - .6))
        lights.append(f'<light pos="{x} {y} {height - .2}" dir="0 0 -1" diffuse=".18 .18 .17" ambient=".03 .03 .03" specular=".03 .03 .03" castshadow="false"/>')
    direction = 1 if scene["params"]["window_side"] == "left" else -1
    lights.append(f'<light name="window_sun" directional="true" pos="{-hw - 3} {-hl + 1} {height + 5}" dir="{direction} .3 -1" diffuse=".20 .19 .18" ambient=".1 .1 .1" castshadow="false"/>')
    xml = ('<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>'
           '<statistic extent="20"/><visual><global offwidth="640" offheight="640"/>'
           '<quality offsamples="4" shadowsize="2048"/><map znear=".003" zfar="30" shadowscale="2"/>'
           '<headlight ambient=".65 .65 .65" diffuse="0 0 0" specular="0 0 0"/></visual>'
           '<default><geom contype="0" conaffinity="0" group="0"/></default><asset>'
           + "".join(rb.b.assets) + '</asset><worldbody>' + "".join(lights + rb.b.geoms)
           + '<camera name="depth_camera" fovy="60"/></worldbody></mujoco>')
    return {"xml": xml, "obstacles": copy.deepcopy(scene["obstacles"]),
            "refs": copy.deepcopy(scene["refs"]), "dims": copy.deepcopy(scene["dims"])}


def point_clearance(point, obstacles):
    best = 1e9
    for ob in obstacles:
        c, h = np.asarray(ob["center"]), np.asarray(ob["halfsize"])
        d = np.maximum(np.abs(np.asarray(point) - c) - h, 0.0)
        best = min(best, float(np.linalg.norm(d)))
    return best


def in_bounds(point, hw, hl, height):
    return (-hw + .3 < point[0] < hw - .3 and -hl + .3 < point[1] < hl - .3
            and .45 < point[2] < height - .4)
