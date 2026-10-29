"""Bright 3D classroom/campus/park scenes with physical face UVs and CC0 materials."""
from pathlib import Path
import importlib.util
import json
import math
import re

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets"
SOURCE = ROOT.parent / "dense_flow_block_depth_servo_20261013"
spec = importlib.util.spec_from_file_location("sealed_base", SOURCE / "experiment.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
NAMES = {"classroom": "日光教室", "classroom_side": "侧向教室", "courtyard": "室外校园庭院", "park": "室外公园步道"}
FONT = "C:/Windows/Fonts/arial.ttf"


def floats(values):
    return " ".join(f"{float(x):.7g}" for x in values)


def look_at(eye, target):
    eye, target = np.array(eye, dtype=float), np.array(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    result = np.eye(4)
    result[:3, :3] = np.column_stack([right, down, forward])
    result[:3, 3] = eye
    return result


def graphic_assets():
    out = ASSET / "classroom_graphics"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("board", "poster", "clock"):
        path = out / f"{name}.png"
        if path.exists():
            continue
        im = Image.new("RGB", (1024, 512) if name == "board" else (512, 512), "#f6f5ed")
        d = ImageDraw.Draw(im)
        font = ImageFont.truetype(FONT, 40)
        small = ImageFont.truetype(FONT, 27)
        if name == "board":
            d.text((55, 28), "GEOMETRY & MOTION", font=font, fill="#244e67")
            d.line((55, 88, 945, 88), fill="#69868b", width=3)
            for i, text in enumerate(["P = (x, y, z)", "u = f x / z", "v = f y / z", "Observe - Predict - Check"]):
                d.text((70, 125 + 70 * i), text, font=small, fill="#355873")
            d.line([(570, 355), (830, 355), (750, 135), (570, 355)], fill="#cf764c", width=5)
            d.line([(750, 135), (750, 355)], fill="#647d93", width=3)
            d.arc((690, 280, 830, 410), 180, 260, fill="#50885f", width=4)
            d.text((770, 240), "z", font=font, fill="#355873")
        elif name == "poster":
            d.rectangle((0, 0, 512, 95), fill="#487878")
            d.text((32, 25), "OUR PLANET", font=font, fill="white")
            d.ellipse((80, 135, 430, 465), fill="#99c9de", outline="#4678a1", width=4)
            d.polygon([(170, 165), (275, 184), (294, 229), (246, 277), (225, 365), (166, 351), (135, 265)], fill="#7caa73")
            d.polygon([(330, 217), (414, 278), (351, 374), (316, 327)], fill="#98ac73")
        else:
            d.ellipse((12, 12, 500, 500), fill="#fcfcf9", outline="#3d454b", width=16)
            for i in range(12):
                angle = i * math.pi / 6
                p = [(256 + radius * math.sin(angle), 256 - radius * math.cos(angle)) for radius in (195, 220)]
                d.line(p, fill="#333f48", width=8)
            d.line((256, 256, 152, 178), fill="#333f48", width=14)
            d.line((256, 256, 368, 126), fill="#333f48", width=9)
        im.save(path)


class Builder:
    def __init__(self, outdoor=False, seed=37):
        self.outdoor = outdoor
        self.rng = np.random.default_rng(seed)
        self.assets, self.geoms, self.lights = [], [], []
        self.counter, self.cache = 0, {}
        self.photo = {"wood": ("wood_floor", 2.), "wall": ("painted_plaster_wall", 3.),
                      "concrete": ("concrete_floor_02", 3.), "brick": ("brick_wall_001", 2.),
                      "grass": ("grass_ground", 2.5), "gravel": ("gravel", 2.)}
        for mat, (identifier, _) in self.photo.items():
            self.texture(mat, ASSET / identifier / "diffuse.png", specular=.04)
        # Repainted pale plaster / light vinyl retain natural surface variation.
        for name, source, tint, strength, tile in [
            ("wall_light", "painted_plaster_wall", [225, 224, 218], .13, 3.),
            ("floor_light", "concrete_floor_02", [199, 200, 194], .20, 3.),
        ]:
            target = ASSET / "classroom_graphics" / f"{name}.png"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                source_rgb = np.array(Image.open(ASSET / source / "diffuse.png"), dtype=float)
                gray = source_rgb.mean(2)
                variation = 1 + strength * (gray / gray.mean() - 1)
                Image.fromarray(np.clip(variation[..., None] * tint, 0, 255).astype(np.uint8)).save(target)
            self.texture(name, target, .04)
            self.photo[name] = (None, tile)
        self.assets.append('<texture type="skybox" builtin="gradient" rgb1="0.45 0.68 0.9" rgb2="0.93 0.97 1" width="512" height="3072"/>')
        for name, rgba in {"white": (.92, .92, .89, 1), "metal": (.30, .34, .37, 1),
                           "blue": (.22, .43, .60, 1), "glass": (.57, .77, .87, 1),
                           "trim": (.79, .82, .83, 1), "ochre": (.76, .60, .38, 1),
                           "trunk": (.35, .25, .15, 1), "leaf": (.25, .43, .11, 1),
                           "terracotta": (.61, .31, .20, 1)}.items():
            self.assets.append(f'<material name="{name}" rgba="{floats(rgba)}" specular=".12" shininess=".18"/>')
        graphic_assets()
        for name in ("board", "poster", "clock"):
            self.texture(name, ASSET / "classroom_graphics" / f"{name}.png", specular=.02)
            self.photo[name] = (None, None)

    def texture(self, name, path, specular=.08):
        self.assets.append(f'<texture name="tex_{name}" type="2d" file="{path.as_posix()}"/>')
        self.assets.append(f'<material name="{name}" texture="tex_{name}" texuniform="false" specular="{specular}" shininess=".15"/>')

    def mesh_box(self, halfsize, material):
        key = (tuple(halfsize), material)
        if key in self.cache:
            return self.cache[key]
        name = f"box_{len(self.cache)}"
        tile = self.photo[material][1]
        vertices, texcoords, faces = [], [], []
        # Independent face vertices/UVs prevent the old stretched box mapping.
        for axis in range(3):
            for sign in (-1, 1):
                uaxis, vaxis = (axis + 1) % 3, (axis + 2) % 3
                points = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
                if sign < 0:
                    points = points[::-1]
                start = len(vertices)
                for u, v in points:
                    p = np.zeros(3)
                    p[axis] = sign * halfsize[axis]
                    p[uaxis], p[vaxis] = u * halfsize[uaxis], v * halfsize[vaxis]
                    vertices.append(p)
                    if tile:
                        texcoords.append(((u + 1) * halfsize[uaxis] / tile, (v + 1) * halfsize[vaxis] / tile))
                    elif axis == 1:
                        texcoords.append(((p[0] / halfsize[0] + 1) / 2, (1 - p[2] / halfsize[2]) / 2))
                    else:
                        texcoords.append(((u + 1) / 2, (v + 1) / 2))
                faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
        self.assets.append(f'<mesh name="{name}" vertex="{floats(np.array(vertices).ravel())}" texcoord="{floats(np.array(texcoords).ravel())}" face="{floats(np.array(faces).ravel())}"/>')
        self.cache[key] = name
        return name

    def box(self, pos, size, material="white", yaw=0, rgba=None):
        extra = f' rgba="{floats(rgba)}"' if rgba else ""
        if material in self.photo:
            geom = f'type="mesh" mesh="{self.mesh_box(size, material)}"'
        else:
            geom = f'type="box" size="{floats(size)}"'
        self.geoms.append(f'<geom {geom} pos="{floats(pos)}" euler="0 0 {yaw}" material="{material}"{extra}/>')

    def cylinder(self, a, b, radius, material="metal"):
        self.geoms.append(f'<geom type="cylinder" fromto="{floats([*a, *b])}" size="{radius}" material="{material}"/>')

    def prop(self, identifier, pos, yaw=0, scale=1.):
        parts = json.loads((ASSET / identifier / "converted.json").read_text())
        for i, part in enumerate(parts):
            if part["alpha_mode"] != "OPAQUE":
                continue
            key = (identifier, i, scale)
            if key not in self.cache:
                name = f"prop_{len(self.cache)}"
                path = ASSET / identifier / part["mesh"]
                self.assets.append(f'<mesh name="{name}" file="{path.as_posix()}" scale="{scale} {scale} {scale}" maxhullvert="64"/>')
                if part["texture"]:
                    self.texture(name, ASSET / identifier / part["texture"], .08)
                else:
                    self.assets.append(f'<material name="{name}" rgba="{floats(part["rgba"])}"/>')
                self.cache[key] = name
            name = self.cache[key]
            self.geoms.append(f'<geom type="mesh" mesh="{name}" material="{name}" pos="{floats(pos)}" euler="0 0 {yaw}"/>')

    def desk(self, x, y, variant=0):
        self.box((x, y, .75), (.58, .35, .035), "wood")
        self.box((x, y, .69), (.52, .29, .025), "trim")
        for dx in (-.49, .49):
            for dy in (-.26, .26):
                self.cylinder((x + dx, y + dy, .03), (x + dx, y + dy, .72), .024)
        self.prop("plastic_monobloc_chair_01", (x, y - .63, 0), yaw=math.pi)
        if variant % 3 == 0:
            self.box((x + .12, y + .04, .80), (.17, .12, .015), "blue")
            self.box((x + .12, y + .035, .818), (.163, .115, .004), "white")
        elif variant % 3 == 1:
            self.box((x - .08, y, .79), (.19, .14, .005), "white")

    def plant(self, x, y, scale=1.):
        self.cylinder((x, y, .05), (x, y, .35 * scale), .18 * scale, "terracotta")
        for angle in np.linspace(0, 2 * math.pi, 13, endpoint=False):
            z = self.rng.uniform(.65, 1.05) * scale
            dx, dy = .28 * scale * np.cos(angle), .28 * scale * np.sin(angle)
            self.cylinder((x, y, .25 * scale), (x + dx, y + dy, z), .009 * scale, "leaf")
            self.geoms.append(f'<geom type="ellipsoid" pos="{floats((x+dx, y+dy, z))}" size="{floats((.09*scale,.04*scale,.22*scale))}" euler=".45 .6 {angle}" material="leaf"/>')

    def tree(self, x, y, scale=1.):
        self.cylinder((x, y, 0), (x, y, 2.9 * scale), .10 * scale, "trunk")
        for angle in np.linspace(0, 2 * math.pi, 9, endpoint=False):
            self.cylinder((x, y, 1.8 * scale),
                          (x + .75 * scale * np.cos(angle), y + .75 * scale * np.sin(angle), 3.0 * scale), .032 * scale, "trunk")
        key = ("leaf_mesh", scale)
        if key not in self.cache:
            name = f"leaves_{len(self.cache)}"
            rng = np.random.default_rng(281)
            vertices, faces = [], []
            for _ in range(2100):
                center = rng.normal(size=3)
                center /= max(np.linalg.norm(center), 1e-10)
                center *= rng.uniform(.25, 1) ** (1 / 3)
                center = center * [1.35, 1.25, 1.10] + [0, 0, 3.1]
                r = Rotation.random(random_state=rng).as_matrix()
                local = np.array([[0, -.105, 0], [.043, -.025, .012], [.033, .055, .012],
                                  [0, .12, 0], [-.033, .055, .012], [-.043, -.025, .012], [0, 0, .027]])
                start = len(vertices)
                vertices.extend((local @ r.T + center) * scale)
                for i in range(6):
                    faces.extend([[start + i, start + (i + 1) % 6, start + 6],
                                  [start + 6, start + (i + 1) % 6, start + i]])
            self.assets.append(f'<mesh name="{name}" vertex="{floats(np.array(vertices).ravel())}" face="{floats(np.array(faces).ravel())}" maxhullvert="48"/>')
            self.cache[key] = name
        name = self.cache[key]
        self.geoms.append(f'<geom type="mesh" mesh="{name}" material="leaf" pos="{x} {y} 0" euler="0 0 {self.rng.uniform(-3,3)}"/>')

    def xml(self):
        if self.outdoor:
            self.lights.append('<light name="sun" directional="true" pos="-6 -8 12" dir=".4 .5 -1" ambient=".15 .15 .15" diffuse=".85 .82 .75" specular=".1 .1 .1" castshadow="true"/>')
        else:
            for x, y in [(-3, -3), (3, 1), (0, 4)]:
                self.lights.append(f'<light pos="{x} {y} 3.2" dir="0 0 -1" diffuse=".18 .18 .17" ambient=".03 .03 .03" specular=".03 .03 .03" castshadow="false"/>')
            self.lights.append('<light name="window_sun" directional="true" pos="-8 -5 8" dir="1 .3 -1" diffuse=".20 .19 .18" ambient=".1 .1 .1" castshadow="false"/>')
        return ('<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>'
                '<statistic extent="15"/><visual><global offwidth="640" offheight="640"/>'
                '<quality offsamples="4" shadowsize="2048"/><map znear=".003" zfar="12" shadowscale="2"/>'
                '<headlight ambient=".65 .65 .65" diffuse="0 0 0" specular="0 0 0"/></visual>'
                '<default><geom contype="0" conaffinity="0" group="0"/></default><asset>'
                + "".join(self.assets) + '</asset><worldbody>' + "".join(self.lights + self.geoms)
                + '<camera name="depth_camera" fovy="60"/></worldbody></mujoco>')


def build(family):
    outdoor = family in ("courtyard", "park")
    b = Builder(outdoor, seed=710 + list(NAMES).index(family))
    if not outdoor:
        b.box((0, 0, -.08), (5, 6, .08), "floor_light")
        b.box((0, 6.1, 1.7), (5.1, .1, 1.7), "wall_light")
        b.box((0, -6.1, 1.7), (5.1, .1, 1.7), "wall_light")
        b.box((5.1, 0, 1.7), (.1, 6., 1.7), "wall_light")
        b.box((0, 0, 3.45), (5.1, 6.1, .05), "white")
        # Actual window openings, with bright sky outside and separate mullions.
        b.box((-5.1, 0, .48), (.1, 6, .48), "wall_light")
        b.box((-5.1, 0, 3.1), (.1, 6, .3), "wall_light")
        for y in (-5.7, -2.6, .5, 3.6, 5.8):
            b.box((-5.1, y, 1.86), (.1, .11, .91), "white")
        for y in (-4.2, -1.1, 2, 4.7):
            b.box((-5.07, y, 1.86), (.035, .035, .88), "trim")
            b.box((-5.06, y, 1.88), (.035, 1.4, .027), "trim")
        for side in (-1, 1):
            b.box((side * 4.95, 0, .09), (.035, 6, .09), "trim")
        b.box((0, 5.94, 1.86), (2.12, .04, .89), "metal")
        b.box((0, 5.885, 1.86), (2.05, .012, .83), "board")
        b.box((3.2, 5.92, 2.1), (.43, .025, .56), "poster")
        b.box((-3.5, 5.91, 2.6), (.30, .02, .30), "clock")
        b.box((0, 4.7, .76), (1., .48, .045), "wood")
        for x in (-.83, .83):
            b.box((x, 4.7, .36), (.035, .38, .36), "metal")
        for row, y in enumerate((-2.2, -.1, 2.0)):
            for col, x in enumerate((-3.2, -1.5, 1.5, 3.2)):
                b.desk(x + (.13 * (row % 2) if family == "classroom_side" else 0), y, row * 4 + col)
        b.plant(-4.15, 4.9, 1.4)
        b.plant(4.2, 5.1, 1.2)
        # Bookshelf on the right wall, with books of varying sizes/colors.
        b.box((4.75, 2.8, .98), (.22, 1.35, .98), "wood")
        for z in (.3, .85, 1.4):
            b.box((4.43, 2.8, z), (.18, 1.27, .025), "wood")
            for j in range(13):
                h = b.rng.uniform(.24, .40)
                b.box((4.37, 1.65 + j * .18, z + h / 2 + .035), (.10, .055, h / 2),
                      "blue" if j % 3 == 0 else "ochre" if j % 3 == 1 else "white")
        for y in (-3, 0, 3):
            b.box((0, y, 3.36), (.85, .12, .018), "white")
        eye = [0., -4.8, 1.55] if family == "classroom" else [1.0, -4.65, 1.60]
        target = [0, 5, 1.32] if family == "classroom" else [-.35, 4.5, 1.4]
    elif family == "courtyard":
        b.box((0, 4, -.10), (18, 24, .10), "grass")
        b.box((0, 3, .015), (2.0, 10, .02), "concrete")
        b.box((0, 11, 2.8), (9., .45, 2.8), "brick")
        b.box((-9, 5, 2.8), (.45, 6., 2.8), "wall")
        for x in np.arange(-7.2, 8, 2.4):
            for z in (1.6, 4.1):
                b.box((x, 10.48, z), (.76, .08, .83), "white")
                b.box((x, 10.37, z), (.67, .025, .72), "glass")
                b.box((x, 10.32, z), (.023, .02, .73), "trim")
                b.box((x, 10.32, z), (.67, .02, .025), "trim")
        b.box((0, 10.23, 1.1), (.74, .14, 1.10), "metal")
        b.box((0, 10.04, 1.21), (.61, .025, .85), "glass")
        b.box((0, 9.6, 2.45), (1.65, .65, .055), "trim")
        for x, y, angle in [(-3.1, -.5, -.25), (3.1, 3.2, math.pi + .15), (-3.2, 6.8, 0)]:
            b.prop("painted_wooden_bench", (x, y, .04), yaw=angle, scale=1.35)
        for x, y, scale in [(-4.3, 2., 1.1), (4.5, 4.5, 1.25), (-5., 8., 1.0), (6.4, 9.2, 1.3)]:
            b.box((x, y, .22), (.65, .65, .22), "concrete")
            b.tree(x, y, scale)
        for x in (-2.5, 2.5):
            b.plant(x, 8.7, 1.5)
        eye, target = [0, -5.2, 1.65], [.25, 8, 1.55]
    else:
        b.box((0, 5, -.10), (24, 30, .10), "grass")
        b.box((0, 4, .015), (1.6, 15, .018), "gravel")
        for x, y, scale in [(-3.8, -.1, 1.05), (4.6, 2.4, 1.3), (-4., 5.1, 1.15),
                            (3.6, 8.1, 1.0), (-3.4, 12., 1.25), (6.8, 14., 1.4), (-7, 17, 1.5)]:
            b.tree(x, y, scale)
        for x, y, angle in [(-2.8, 3.5, .15), (2.9, 8., math.pi)]:
            b.prop("painted_wooden_bench", (x, y, .02), yaw=angle, scale=1.5)
        for x, y, scale in [(-2.5, -.2, 7.), (2.7, 2.1, 9.), (-3.1, 6.5, 11.), (2.5, 11.5, 8.)]:
            b.prop("rock_09", (x, y, .0), yaw=b.rng.uniform(-3, 3), scale=scale)
        b.cylinder((2.35, 5.0, 0), (2.35, 5., 1.9), .05, "trunk")
        b.box((2.35, 5., 1.6), (.46, .04, .22), "blue")
        b.box((2.35, 4.95, 1.61), (.30, .015, .03), "white")
        # A distant park building gives finite, realistic background structure.
        b.box((1, 22, 2.3), (7, .6, 2.3), "brick")
        for x in (-4, -1, 2, 5):
            b.box((x, 21.34, 2.4), (.75, .04, .9), "glass")
        eye, target = [0, -5.0, 1.7], [.2, 8, 1.2]
    return b.xml(), look_at(eye, target)


class Scene(base.Scene):
    def __init__(self, family):
        self.family = family
        self.xml, self.goal = build(family)
        # MuJoCo's Windows file loader cannot open all Unicode paths; feed bytes.
        compiled_xml = self.xml
        virtual_files = {}
        for i, filename in enumerate(dict.fromkeys(re.findall(r'file="([^"]+)"', self.xml))):
            path = Path(filename)
            alias = f"resource_{i}{path.suffix}"
            virtual_files[alias] = path.read_bytes()
            compiled_xml = compiled_xml.replace(f'file="{filename}"', f'file="{alias}"')
        self.model = mujoco.MjModel.from_xml_string(compiled_xml, assets=virtual_files)
        self.data = mujoco.MjData(self.model)
        self.cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
        self.renderer = mujoco.Renderer(self.model, base.SIZE, base.SIZE)
        forward = self.goal[:3, 2].copy()
        forward[2] = 0
        forward /= np.linalg.norm(forward)
        self.body0 = np.column_stack([forward, np.cross([0., 0., 1.], forward), [0., 0., 1.]])
        self.camera_from_body = self.goal[:3, :3].T @ self.body0
        self.mount_yaw = self.camera_from_body[:, 2]
        a = np.zeros((base.SIZE, base.SIZE, 2, 3), np.float32)
        a[..., 0, 0], a[..., 0, 2] = -1., base.X
        a[..., 1, 1], a[..., 1, 2] = -1., base.Y
        self.trans = a @ self.camera_from_body
        kx, ky, kz = self.mount_yaw
        self.rot = np.stack([base.X * base.Y * kx - (1 + base.X ** 2) * ky + base.Y * kz,
                             (1 + base.Y ** 2) * kx - base.X * base.Y * ky - base.X * kz], axis=-1).astype(np.float32)
        self.goal_rgb, self.goal_depth = self.render(self.goal[:3, 3], 0.)


def preview():
    destination = ROOT / "scenes"
    destination.mkdir(exist_ok=True)
    records = []
    canvas = Image.new("RGB", (1024, 1120), "#f2f5f7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 25)
    for i, family in enumerate(NAMES):
        scene = Scene(family)
        try:
            rgb = scene.goal_rgb
            Image.fromarray(rgb).save(destination / f"{family}_reference.png")
            (destination / f"{family}.xml").write_text(scene.xml, encoding="utf-8")
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            record = {"family": family, "mean_gray": float(gray.mean()),
                      "dark_pixel_fraction_gray_below_30": float(np.mean(gray < 30)),
                      "saturated_pixel_fraction_gray_above_250": float(np.mean(gray > 250)),
                      "geoms": scene.model.ngeom, "mesh_triangles": scene.model.nmeshface,
                      "goal_pose": scene.goal.tolist(), "xml_sha256": base.sha(destination / f"{family}.xml")}
            records.append(record)
            print("SCENE", json.dumps(record, ensure_ascii=False), flush=True)
            x, y = (i % 2) * 512, (i // 2) * 560
            draw.text((x + 15, y + 10), NAMES[family], font=font, fill="#22394b")
            canvas.paste(Image.fromarray(rgb).resize((512, 512), Image.Resampling.LANCZOS), (x, y + 45))
        finally:
            scene.close()
    canvas.save(ROOT / "scene_gallery.png")
    (ROOT / "scene_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    preview()
