from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import legacy
from .config import digest, json_write
from .controller import certificate_control, hard_backtracking, saturate
from .vision import calibrated_frame, frozen_features


def make_manifest(cfg):
    builders = legacy.module(cfg, "scene_builders")
    common = legacy.module(cfg, "common")
    scenes = []
    for split_index, split in enumerate(("train", "val", "test")):
        for family_index, family in enumerate(cfg["scenes"]["families"]):
            for k in range(cfg["scenes"]["per_family"][split]):
                for retry in range(100):
                    seed = cfg["scenes"]["seed_base"] + split_index * 100000 + family_index * 10000 + k * 100 + retry
                    rng = np.random.default_rng(seed)
                    spec = {"id": f"{family}_{split}_{k:02d}", "split": split, "family": family,
                            "seed": seed, "params": builders.SAMPLERS[family](rng)}
                    try:
                        built = builders.build_instance(spec)
                        break
                    except RuntimeError:
                        continue
                else:
                    raise RuntimeError(f"Could not construct valid scene {family}/{split}/{k}")
                refs = []
                for ref in built["refs"][:cfg["scenes"]["references"][split]]:
                    refs.append({**ref, "goal_matrix": common.look_at(ref["eye"], ref["target"]).tolist()})
                if len(refs) != cfg["scenes"]["references"][split]:
                    raise ValueError(f"Scene {spec['id']} did not provide enough fixed goals")
                scenes.append({**spec, "refs": refs, "dims": built["dims"], "obstacles": built["obstacles"]})
    return {"schema": 1, "scenes": scenes, "new_scene_seeds": True,
            "claim": "Held-out parameterized room instances; not arbitrary real-world environment generalization"}


def valid_start(cfg, scene, frame, offset):
    p = frame.position_from_offset(offset)
    dims = scene["dims"]
    bounds = (-dims["hw"] + 0.3 < p[0] < dims["hw"] - 0.3 and
              -dims["hl"] + 0.3 < p[1] < dims["hl"] - 0.3 and 0.45 < p[2] < dims["height"] - 0.4)
    clearance = legacy.module(cfg, "scene_builders").point_clearance(p, scene["obstacles"])
    return bounds and clearance > 0.16


def sample_start(cfg, scene, frame, rng, difficulty):
    limits = {"small": (0.05, 0.18, 1, 5), "medium": (0.25, 0.5, 6, 12), "large": (0.7, 1.0, 15, 25)}
    lo, hi, yawlo, yawhi = limits[difficulty]
    for _ in range(2000):
        direction = rng.normal(size=3)
        direction[2] *= 0.3
        direction /= np.linalg.norm(direction)
        offset = np.r_[direction * rng.uniform(lo, hi), math.radians(rng.uniform(yawlo, yawhi)) * rng.choice([-1, 1])]
        if valid_start(cfg, scene, frame, offset):
            return offset
    raise RuntimeError(f"Start rejection exhausted for {scene['id']}")


def make_evaluation_starts(cfg, manifest, split):
    env = legacy.module(cfg, "room_env")
    rng = np.random.default_rng(cfg["seed"] + (1000 if split == "val" else 2000))
    count = cfg["evaluation"]["validation_starts_per_goal_difficulty" if split == "val" else "test_starts_per_goal_difficulty"]
    starts = []
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        for index, ref in enumerate(scene["refs"]):
            frame = env.ReferenceFrame(np.asarray(ref["goal_matrix"]))
            for difficulty in ("small", "medium", "large"):
                for k in range(count):
                    starts.append({"scene_id": scene["id"], "goal_id": index, "difficulty": difficulty,
                                   "repeat": k, "offset": sample_start(cfg, scene, frame, rng, difficulty).tolist()})
    return starts


def task_error(cfg, frame, state):
    offset = np.r_[frame.body0.T @ (state["position"] - frame.goal[:3, 3]), state["yaw"]]
    offset[3] = (offset[3] + math.pi) % (2 * math.pi) - math.pi
    error = float((offset[:3] / cfg["cost"]["position_scale_m"]) @ (offset[:3] / cfg["cost"]["position_scale_m"]))
    error += (offset[3] / math.radians(cfg["cost"]["yaw_scale_deg"]))**2
    return offset.astype(np.float32), error


def pvs_action(cfg, rgb, goal, depth, rotation):
    vision = legacy.module(cfg, "vision")
    im, li = vision.observe_gray(calibrated_frame(cfg, rotation), rgb, depth)
    feature = vision.PVSFeature(vision.preprocess_gray(goal))
    error, jacobian = feature.error_jacobian(im, li)
    action, _ = vision.control_numpy(error, jacobian)
    return action.astype(np.float32)


def run_episode(cfg, scene, goal_index, start, autoencoder, actor, depth, device,
                rng, behavior="student", plant_name="P0", max_steps=None, exploration=0.0,
                instance=None, video_path=None):
    env = legacy.module(cfg, "room_env")
    own_instance = instance is None
    instance = instance or env.RoomInstance({k: scene[k] for k in ("id", "split", "family", "seed", "params")})
    frame = env.ReferenceFrame(np.asarray(scene["refs"][goal_index]["goal_matrix"]))
    plant = env.PLANTS[plant_name]
    state = {"position": frame.position_from_offset(start), "yaw": float(start[3]), "body0": frame.body0,
             **plant.initial_extra()}
    goal = instance.goal_rgb(frame)
    rotation = frame.camera_from_body.astype(np.float32)
    rgb = instance.render_rgb(frame, state["position"], state["yaw"])
    estimate = depth.predict_fresh([rgb])[0]
    offset, error = task_error(cfg, frame, state)
    frames, depths, offsets, errors = [rgb], [estimate.astype(np.float16)], [offset], [error]
    actions, costs, terminals, truncations, factors, values, vdot = [], [], [], [], [], [], []
    previous = [np.zeros(4, np.float32)]
    hold, status, no_descent_steps = 0, "timeout", 0
    started = time.perf_counter()
    max_steps = max_steps or cfg["collection"]["steps"]
    writer = None
    if video_path:
        import cv2
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 288))
        if not writer.isOpened():
            raise RuntimeError("Could not open MP4 writer")
    try:
        for step in range(max_steps):
            if behavior == "pvs":
                action = pvs_action(cfg, rgb, goal, estimate, rotation)
                value, flow, lower = float("nan"), float("nan"), np.eye(cfg["encoder"]["latent_dim"], dtype=np.float32)
            else:
                z, zg, j = frozen_features(cfg, autoencoder, [rgb], [goal], [estimate], [rotation], device)
                result = certificate_control(actor, z, zg, j, **cfg["controller"])
                if behavior == "student_hard":
                    chosen, feasible = hard_backtracking(actor, result, zg, j, cfg["plant"]["dt"],
                                                         cfg["execution"]["backtracking_scales"],
                                                         cfg["execution"]["residual_tolerance"])
                    result["action"] = chosen
                    no_descent_steps += int(not bool(feasible[0]))
                action = result["action"][0].detach().cpu().numpy()
                value, flow = float(result["V"][0].detach()), float(result["Vdot"][0].detach())
                lower = result["L"][0].detach().cpu().numpy()
            if exploration:
                noisy = torch.from_numpy(action + rng.normal(0, exploration, 4).astype(np.float32))
                action = saturate(noisy).numpy()
            if behavior != "pvs":
                flow = float((result["g"][0].detach().cpu() * torch.from_numpy(action)).sum())
            if not np.isfinite(action).all():
                raise FloatingPointError("Nonfinite command; refusing to write an invalid training transition")
            if writer:
                import cv2
                panel = np.zeros((288, 512, 3), np.uint8)
                panel[:256, :256], panel[:256, 256:] = rgb, goal
                text = f"{scene['id']} {plant_name} t={step*.1:.1f}s  GT(eval): {np.linalg.norm(offset[:3]):.3f}m"
                cv2.putText(panel, text, (5, 278), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            state = plant.step_state(state, action)
            offset, error = task_error(cfg, frame, state)
            collision = instance.collides(state["position"])
            outside = instance.outside_workspace(state["position"])
            good = (np.linalg.norm(offset[:3]) < cfg["collection"]["position_tolerance_m"] and
                    abs(offset[3]) < math.radians(cfg["collection"]["yaw_tolerance_deg"]))
            hold = hold + 1 if good else 0
            success = hold >= cfg["collection"]["required_hold_observations"] and not collision and not outside
            terminal = collision or outside or success
            truncated = step == max_steps - 1 and not terminal
            stage_cost = cfg["plant"]["dt"] * (error + cfg["cost"]["effort"] * float(action @ action))
            if collision or outside:
                stage_cost += cfg["cost"]["collision_terminal"]
            rgb = instance.render_rgb(frame, state["position"], state["yaw"])
            estimate = depth.predict_fresh([rgb])[0]
            frames.append(rgb)
            depths.append(estimate.astype(np.float16))
            offsets.append(offset)
            errors.append(error)
            actions.append(action)
            previous.append(action)
            costs.append(stage_cost)
            terminals.append(terminal)
            truncations.append(truncated)
            factors.append(lower.astype(np.float16))
            values.append(value)
            vdot.append(flow)
            if terminal or truncated:
                status = "collision" if collision else "outside" if outside else "success" if success else "timeout"
                break
    finally:
        if writer:
            writer.release()
        if own_instance:
            instance.close()
    arrays = {"rgb": np.stack(frames), "depth": np.stack(depths), "goal_rgb": goal,
              "action": np.asarray(actions, np.float32), "stage_cost": np.asarray(costs, np.float32),
              "terminal": np.asarray(terminals, bool), "truncated": np.asarray(truncations, bool),
              "error2": np.asarray(errors, np.float32), "offset": np.asarray(offsets, np.float32),
              "previous_action": np.asarray(previous, np.float32), "camera_from_body": rotation,
              "L": np.asarray(factors, np.float16)}
    if behavior != "pvs":
        from .vision import image_tensor
        with torch.no_grad():
            final_z = autoencoder.encode(image_tensor(cfg, [rgb], device))
            goal_z = autoencoder.encode(image_tensor(cfg, [goal], device))
            final_e = (final_z - goal_z) / math.sqrt(cfg["encoder"]["latent_dim"])
            final_value = float(actor.certificate(final_e, goal_z)[0][0])
        arrays["certificate"] = np.asarray([*values, final_value], np.float32)
        arrays["predicted_vdot"] = np.asarray(vdot, np.float32)
    differences = np.diff(arrays["action"], axis=0)
    summary = {"scene_id": scene["id"], "split": scene["split"], "goal_id": goal_index, "plant": plant_name,
               "status": status, "success": status == "success", "steps": len(actions),
               "final_position_m": float(np.linalg.norm(offset[:3])), "final_yaw_deg": float(abs(np.degrees(offset[3]))),
               "integrated_error": float(np.sum(errors[1:]) * cfg["plant"]["dt"]),
               "action_variation": float(np.sum(differences**2)), "wall_seconds": time.perf_counter() - started,
               "behavior": behavior}
    # PVS bootstrap has no latent certificate; no NaN is written to JSON.
    if behavior != "pvs":
        summary["predicted_positive_vdot_fraction"] = float(np.mean(np.asarray(vdot) > 1e-8))
        certificate = arrays["certificate"]
        tolerance = 1e-7 + 1e-4 * np.maximum(certificate[:-1], 0)
        summary["actual_V_increase_fraction"] = float(np.mean(np.diff(certificate) > tolerance))
        summary["actual_decay_violation_fraction"] = float(np.mean(
            np.diff(certificate) + cfg["loss"]["decay_per_second"] * cfg["plant"]["dt"] * certificate[:-1] > tolerance))
        summary["no_model_descent_steps"] = no_descent_steps
    return arrays, summary
