"""Independent geometry/dynamics/schema audit; only train RGB is compared.

This script freezes source identity BEFORE accessing held-out scene geometry.
It does not fit, optimize, or change the reconstructed scene builder.
"""
from __future__ import annotations
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time
import types

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
LEGACY = BASE / "reconstructed_legacy_root"
DEPENDENCIES = ROOT / "third_party/visual_original/assets/scene_dependencies"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    package = types.ModuleType("reconstructed_scene_audit")
    package.__path__ = [str(LEGACY / "src")]
    sys.modules[package.__name__] = package
    common = importlib.import_module(package.__name__ + ".common")
    common.ROOT = LEGACY
    common.WORKSPACE = ROOT
    common.OLD_SCENES_DIR = DEPENDENCIES / "bright_natural_dense_servo_20261013"
    return common, importlib.import_module(package.__name__ + ".scene_builders"), importlib.import_module(package.__name__ + ".room_env")


def main():
    start = time.time()
    frozen_sha = sha(LEGACY / "src/scene_builders.py")
    (BASE / "SOURCE_FREEZE.json").write_text(json.dumps({"schema": 1, "frozen_at_unix": start,
        "builder_sha256": frozen_sha, "frozen_before_heldout_geometry_audit": True,
        "pixel_tuning_scope": "train only; no validation or test RGB accessed"}, indent=2), encoding="utf-8")
    common, builders, env = load()
    manifest = json.loads((BASE / "scenes.json").read_text(encoding="utf-8"))
    rows, scene_index, asset_files = [], {}, set()
    xml_dir = BASE / "compiled_scene_xml"
    xml_dir.mkdir(exist_ok=True)
    for scene in manifest["scenes"]:
        built = builders.build_instance(scene)
        repeated = builders.build_instance(scene)
        assert all(built[k] == scene[k] for k in ("refs", "obstacles", "dims"))
        assert built == repeated
        scene_index[scene["id"]] = scene
        (xml_dir / (scene["id"] + ".xml")).write_text(built["xml"], encoding="utf-8")
        asset_files.update(Path(p) for p in re.findall(r'file="([^"]+)"', built["xml"]))
        # Compile all 44 models on CPU. Rendering is reserved for four train rooms.
        model = env._uri_loaded_model(built["xml"])
        ref_checks = []
        probe = object.__new__(env.RoomInstance)
        probe.obstacles, probe.dims = built["obstacles"], built["dims"]
        for ref in scene["refs"]:
            goal = common.look_at(ref["eye"], ref["target"])
            assert np.array_equal(goal, np.asarray(ref["goal_matrix"]))
            clearance = builders.point_clearance(ref["eye"], built["obstacles"])
            assert clearance == ref["clearance_m"]
            assert not probe.collides(ref["eye"]) and not probe.outside_workspace(ref["eye"])
            assert builders.in_bounds(ref["eye"], **built["dims"])
            ref_checks.append(clearance)
        rows.append({"id": scene["id"], "split": scene["split"], "ngeom": int(model.ngeom),
                     "nmesh": int(model.nmesh), "obstacles": len(built["obstacles"]),
                     "minimum_reference_clearance_m": min(ref_checks), "deterministic_xml": True,
                     "structured_geometry_exact": True})
        del model
    # All fixed evaluation starts: metadata/geometry only, no held-out image read.
    starts_path = ROOT / "assets/original_visual_initialization/evaluation_starts.json"
    starts = json.loads(starts_path.read_text(encoding="utf-8"))
    start_count, min_clearance = 0, float("inf")
    for split, cases in starts.items():
        for row in cases:
            scene = scene_index[row["scene_id"]]
            frame = env.ReferenceFrame(np.asarray(scene["refs"][row["goal_id"]]["goal_matrix"]))
            pos = frame.position_from_offset(row["offset"])
            assert builders.in_bounds(pos, **scene["dims"])
            clearance = builders.point_clearance(pos, scene["obstacles"])
            assert clearance > .16
            probe = object.__new__(env.RoomInstance)
            probe.obstacles, probe.dims = scene["obstacles"], scene["dims"]
            assert not probe.collides(pos) and not probe.outside_workspace(pos)
            start_count += 1
            min_clearance = min(min_clearance, clearance)
    # Collision tests include exact tangency and a strictly separated point.
    probe = object.__new__(env.RoomInstance)
    probe.obstacles = [{"center": [0., 0., 0.], "halfsize": [1., 1., 1.]}]
    collision_tests = {"interior": probe.collides([0, 0, 0]),
                       "tangent": probe.collides([1.15, 0, 0]),
                       "outside": not probe.collides([1.151, 0, 0])}
    assert all(collision_tests.values())
    rng = np.random.default_rng(741)
    max_p0, max_p1 = 0., 0.
    for _ in range(100):
        yaw = float(rng.uniform(-2, 2))
        cmd = rng.uniform(-1, 1, 4)
        state = {"position": rng.normal(size=3), "yaw": yaw, "body0": common.rot_z(.3),
                 "actual_velocity": rng.normal(size=4), "previous_command": rng.uniform(-1, 1, 4)}
        command = common.ACTION_SCALE * cmd
        body = common.rot_z(yaw + .5 * command[3] * common.DT) @ state["body0"]
        expected_pos = state["position"] + body @ command[:3] * common.DT
        expected_yaw = yaw + command[3] * common.DT
        got = env.PlantP0.step_state(state, cmd)
        max_p0 = max(max_p0, float(np.max(np.abs(got["position"] - expected_pos))), abs(got["yaw"] - expected_yaw))
        actual = (1 - env.PlantP1.mix) * state["actual_velocity"] + env.PlantP1.mix * common.ACTION_SCALE * state["previous_command"]
        body = common.rot_z(yaw + .5 * actual[3] * common.DT) @ state["body0"]
        expected_pos = state["position"] + body @ actual[:3] * common.DT
        got = env.PlantP1.step_state(state, cmd)
        max_p1 = max(max_p1, float(np.max(np.abs(got["position"] - expected_pos))), abs(got["yaw"] - (yaw + actual[3] * common.DT)))
        assert np.array_equal(got["previous_command"], cmd)
    assert max_p0 == max_p1 == 0
    render_rows = []
    for family in common.FAMILIES:
        scene = scene_index[family + "_train_00"]
        instance = env.RoomInstance(scene)
        try:
            frame = instance.references[0]
            image = instance.goal_rgb(frame)
            repeated = instance.goal_rgb(frame)
            assert image.shape == (256, 256, 3) and image.dtype == np.uint8
            assert np.array_equal(image, repeated) and image.std() > 5
            Image.fromarray(image).save(BASE / (family + "_reconstructed_reference.png"))
            historical_path = BASE / (family + "_historical_reference.png")
            historical = np.asarray(Image.open(historical_path).convert("RGB"))
            delta = image.astype(float) - historical.astype(float)
            mse = float(np.mean(delta ** 2))
            panel = np.concatenate([historical, image, np.clip(np.abs(delta) * 3, 0, 255).astype(np.uint8)], axis=1)
            Image.fromarray(panel).save(BASE / (family + "_train_comparison.png"))
            # Four short dynamics/render feedback sequences, intentionally no
            # learner or learned-M performance claim in this rendering audit.
            shapes = []
            state = {"position": frame.position_from_offset([.1, .03, .01, .02]), "yaw": .02,
                     "body0": frame.body0, **env.PlantP0.initial_extra()}
            for _ in range(5):
                observed = instance.render_rgb(frame, state["position"], state["yaw"])
                shapes.append(list(observed.shape))
                assert not instance.collides(state["position"]) and not instance.outside_workspace(state["position"])
                offset = frame.body0.T @ (state["position"] - frame.goal[:3, 3])
                # State feedback is ONLY an engineering dynamics/render loop.
                command = np.clip(-.2 * np.r_[offset, state["yaw"]] / common.ACTION_SCALE, -.1, .1)
                state = env.PlantP0.step_state(state, command)
            render_rows.append({"scene_id": scene["id"], "goal_id": 0, "split": "train", "rgb_shape": list(image.shape),
                "deterministic_rgb": True, "mae_to_historical": float(np.mean(np.abs(delta))),
                "rmse_to_historical": mse ** .5, "psnr_to_historical": float(10 * np.log10(255 ** 2 / max(mse, 1e-12))),
                "matching_pixel_fraction": float(np.mean(np.all(image == historical, axis=-1))),
                "engineering_feedback_render_steps": len(shapes), "engineering_feedback_is_servo_result": False})
        finally:
            instance.close()
    original_root = ROOT / "assets/recovered_visual_legacy/src"
    exact_files = ["common.py", "models.py", "vision.py", "room_env.py", "depth_net.py"]
    assert all(sha(LEGACY / "src" / name) == sha(original_root / name) for name in exact_files)
    assert frozen_sha == sha(LEGACY / "src/scene_builders.py")
    sources = sorted((LEGACY / "src").glob("*.py")) + [
        DEPENDENCIES / "bright_natural_dense_servo_20261013/scenes.py",
        DEPENDENCIES / "dense_flow_block_depth_servo_20261013/experiment.py"]
    files = [{"path": str(path.resolve()), "sha256": sha(path)} for path in sources]
    assets = [{"path": str(path.resolve()), "sha256": sha(path)} for path in sorted(asset_files)]
    checks = {"structured_geometry_preserved": True, "collision_semantics_preserved": True,
        "p0_p1_dynamics_preserved": True, "all_44_xml_compiled": True,
        "all_fixed_starts_valid": True, "original_module_byte_copies": True,
        "train_only_rgb_comparison": True, "source_frozen_before_heldout_geometry": True,
        "deterministic_xml_and_train_render": True}
    audit = {"schema": 1, "passed": True, "reconstruction": True,
        "protocol_kind": "reconstructed_environment", "original_rgb_equivalent": False,
        "legacy_root": str(LEGACY.resolve()), "scene_manifest": str((BASE / "scenes.json").resolve()),
        "scene_manifest_sha256": sha(BASE / "scenes.json"), "frozen_builder_sha256": frozen_sha,
        "source_files": files, "files": files, "asset_files": assets,
        "geometry_manifest_preserved": True, "p0_p1_dynamics_parity_passed": True,
        "checks": checks, "scenes": rows, "train_render_comparisons": render_rows,
        "fixed_start_count": start_count, "minimum_fixed_start_clearance_m": min_clearance,
        "collision_probe_checks": collision_tests, "p0_max_abs": max_p0, "p1_max_abs": max_p1,
        "visual_changes": ["Missing original shell/family assembly reconstructed explicitly", "Original samplers replaced by frozen manifest lookup", "Decor placement, office monitors and shelf details are inferred"],
        "usage": "Regenerate current AND goal RGB, replay and anchors under a NEW protocol. Never resume old RGB protocols.",
        "limitations": "Structured collision geometry and kinematics match. Appearance is not strictly equivalent; historical success rates are not migration results.",
        "elapsed_seconds": time.time() - start}
    (BASE / "SCENE_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": True, "scenes": len(rows), "fixed_starts": start_count,
        "train_comparisons": render_rows, "elapsed": time.time() - start}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
