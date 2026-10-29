from __future__ import annotations

import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import legacy
from .config import digest, json_append, json_write, restore_rng, rng_state, seed_all, source_hashes
from .environment import make_evaluation_starts, make_manifest, run_episode, sample_start
from .learner import Learner
from .replay import Replay


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(cfg, output):
    output = Path(output).resolve()
    marker = output / "protocol.json"
    if marker.exists():
        protocol = read_json(marker)
        if protocol["config"] != cfg or protocol["source_hashes"] != source_hashes():
            raise ValueError("Existing run config/source differs; use a new run directory")
        if protocol["dependencies"] != legacy.dependency_hashes(cfg):
            raise ValueError("Read-only dependency changed since preparation")
        return protocol
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to adopt a nonempty directory without this experiment protocol")
    output.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(cfg)
    starts = {split: make_evaluation_starts(cfg, manifest, split) for split in ("val", "test")}
    protocol = {"schema": 1, "config": cfg, "source_hashes": source_hashes(),
                "dependencies": legacy.dependency_hashes(cfg), "created_unix": time.time(),
                "algorithm": "deterministic goal-conditioned LAC-style cost critic + AE + full-M certificate actor",
                "original_LAC_guarantee_inherited": False, "encoder_input": "RGB only",
                "control_depth": "frozen estimated metric depth, no true-depth alignment",
                "runtime_pose_labels": False}
    json_write(output / "scenes.json", manifest)
    json_write(output / "evaluation_starts.json", starts)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    json_write(marker, protocol)
    return protocol


class StopBoundary:
    def __init__(self, minutes=None):
        self.deadline = time.monotonic() + minutes * 60 if minutes else float("inf")
        self.requested = False
        self.previous = signal.signal(signal.SIGINT, self._request)

    def _request(self, *_):
        self.requested = True

    def stopped(self):
        return self.requested or time.monotonic() >= self.deadline

    def close(self):
        signal.signal(signal.SIGINT, self.previous)


def save_checkpoint(path, learner, state, replay, protocol):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    payload = {"schema": 1, "learner": learner.payload(), "state": state, "rng": rng_state(),
               "replay_entries": replay.entries, "replay_rng": replay.rng.bit_generator.state,
               "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, learner, replay, protocol, restore_random=True):
    payload = torch.load(path, map_location=learner.device, weights_only=False)
    expected = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if payload.get("protocol_sha256") != expected:
        raise ValueError("Checkpoint protocol mismatch")
    learner.restore(payload["learner"])
    if restore_random:
        restore_rng(payload["rng"])
        replay.rng.bit_generator.state = payload["replay_rng"]
        replay.entries = payload["replay_entries"]
        if any(not (replay.root / row["file"]).is_file() for row in replay.entries):
            raise FileNotFoundError("A checkpoint replay shard is missing")
        json_write(replay.index_path, replay.entries)
        replay.cache.clear()
    return payload["state"]


def control_fingerprint(learner, method):
    h = hashlib.sha256(method.encode())
    for module in (learner.ae.encoder, learner.actor):
        for key, value in module.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def aggregate(rows):
    if not rows:
        return {"episodes": 0, "complete": False}
    scenes = sorted({row["scene_id"] for row in rows})
    rates = {scene: float(np.mean([row["success"] for row in rows if row["scene_id"] == scene])) for scene in scenes}
    metrics = ("final_position_m", "final_yaw_deg", "integrated_error", "action_variation",
               "actual_V_increase_fraction", "actual_decay_violation_fraction", "no_model_descent_steps")
    summary = {"episodes": len(rows), "success_rate": float(np.mean([row["success"] for row in rows])),
               "macro_scene_success": float(np.mean(list(rates.values()))), "worst_scene_success": min(rates.values()),
               "scene_success": rates, "complete": True}
    for key in metrics:
        available = [row[key] for row in rows if key in row]
        if available:
            summary[f"mean_{key}"] = float(np.mean(available))
    summary["difficulty_success"] = {difficulty: float(np.mean([row["success"] for row in rows if row["difficulty"] == difficulty]))
                                     for difficulty in ("small", "medium", "large")
                                     if any(row["difficulty"] == difficulty for row in rows)}
    summary["status_counts"] = {status: sum(row["status"] == status for row in rows)
                                for status in sorted({row["status"] for row in rows})}
    plants = sorted({row["plant"] for row in rows})
    summary["plant"] = plants[0] if len(plants) == 1 else "mixed_see_plant_summary"
    if len(plants) > 1:
        summary["plant_summary"] = {plant: aggregate([row for row in rows if row["plant"] == plant]) for plant in plants}
    return summary


def select_validation_subset(starts, cap):
    if cap is None or len(starts) <= cap:
        return starts
    groups = {}
    for row in starts:
        groups.setdefault((row["scene_id"].split("_")[0], row["difficulty"]), []).append(row)
    selected, level = [], 0
    while len(selected) < cap:
        added = False
        for key in sorted(groups):
            if level < len(groups[key]):
                selected.append(groups[key][level])
                added = True
                if len(selected) == cap:
                    break
        if not added:
            break
        level += 1
    return selected


def evaluate(cfg, output, learner, depth, split="val", plants=None, cap=None, method="learned", stop=None):
    output = Path(output)
    manifest = read_json(output / "scenes.json")
    scenes = {row["id"]: row for row in manifest["scenes"]}
    starts = select_validation_subset(read_json(output / "evaluation_starts.json")[split], cap)
    plants = plants or cfg["plant"]["evaluation_plants"]
    fingerprint = control_fingerprint(learner, method)
    folder = output / "evaluation" / f"{split}_{method}_{fingerprint}_{'_'.join(plants)}_n{len(starts)}"
    folder.mkdir(parents=True, exist_ok=True)
    old_identity = learner.actor.identity
    if method == "identity_m":
        learner.actor.identity = True
    rows, current_instance, current_id = [], None, None
    env = legacy.module(cfg, "room_env")
    rng = np.random.default_rng(cfg["seed"] + 9999)
    complete = True
    try:
        for plant_name in plants:
            for index, start in enumerate(starts):
                row_path = folder / f"{plant_name}_{index:04d}.json"
                if row_path.exists():
                    rows.append(read_json(row_path))
                    continue
                if stop and stop.stopped():
                    complete = False
                    break
                scene = scenes[start["scene_id"]]
                if current_id != scene["id"]:
                    if current_instance:
                        current_instance.close()
                    current_instance = env.RoomInstance({k: scene[k] for k in ("id", "split", "family", "seed", "params")})
                    current_id = scene["id"]
                video = folder / f"{plant_name}_{index:04d}.mp4" if index < cfg["evaluation"]["video_episodes"] else None
                arrays, row = run_episode(cfg, scene, start["goal_id"], start["offset"], learner.ae, learner.actor,
                                           depth, learner.device, rng, behavior="pvs" if method == "pvs" else "student_hard",
                                           plant_name=plant_name, max_steps=cfg["evaluation"]["max_steps"],
                                           instance=current_instance, video_path=video)
                row.update(difficulty=start["difficulty"], start=start["offset"], controller_fingerprint=fingerprint)
                json_write(row_path, row)
                telemetry = {k: value for k, value in arrays.items() if k not in ("rgb", "depth", "goal_rgb", "L")}
                np.savez_compressed(folder / f"{plant_name}_{index:04d}_telemetry.npz", **telemetry)
                rows.append(row)
                print(json.dumps({"event": "evaluation_episode", "split": split, "plant": plant_name,
                                  "index": index, "status": row["status"]}, ensure_ascii=False), flush=True)
            if not complete:
                break
    finally:
        learner.actor.identity = old_identity
        if current_instance:
            current_instance.close()
    summary = aggregate(rows)
    summary.update(complete=complete and len(rows) == len(starts) * len(plants), split=split, method=method,
                   expected_episodes=len(starts) * len(plants), folder=str(folder), control_fingerprint=fingerprint,
                   selection_data=split == "val", latent_and_weights_fixed_during_episode=True)
    json_write(folder / "summary.json", summary)
    return summary


def train(cfg, output, resume=False, max_minutes=None):
    output = Path(output).resolve()
    protocol = prepare(cfg, output)
    seed_all(cfg["seed"])
    learner = Learner(cfg, cfg["device"])
    replay = Replay(output / "replay", cfg["collection"]["replay_transitions"],
                    cfg["collection"]["replay_episode_cache"], cfg["seed"])
    last = output / "last.pt"
    state = {"phase": "bootstrap", "index": 0, "round": 0, "best_score": None}
    if resume:
        if not last.exists():
            raise FileNotFoundError("--resume requires last.pt")
        state = load_checkpoint(last, learner, replay, protocol)
    elif last.exists():
        raise FileExistsError("Run already has last.pt; use --resume or a fresh output directory")
    elif replay.entries:
        raise FileExistsError("Existing replay without checkpoint cannot be silently adopted")
    depth = legacy.module(cfg, "depth_net").MetricDepth(str(learner.device))
    manifest = read_json(output / "scenes.json")
    scenes = [row for row in manifest["scenes"] if row["split"] == "train"]
    stopper = StopBoundary(max_minutes)
    save_checkpoint(last, learner, state, replay, protocol)
    try:
        while state["phase"] != "complete":
            if stopper.stopped():
                save_checkpoint(last, learner, state, replay, protocol)
                json_write(output / "status.json", {"state": "stopped_at_safe_boundary", **state, "counters": learner.counters})
                return
            phase = state["phase"]
            if phase in ("bootstrap", "collect"):
                bootstrap = phase == "bootstrap"
                per_scene = cfg["collection"]["bootstrap_episodes_per_scene" if bootstrap else "episodes_per_scene_per_round"]
                if state["index"] >= len(scenes) * per_scene:
                    state["phase"], state["index"] = ("warmup" if bootstrap else "validate"), 0
                    save_checkpoint(last, learner, state, replay, protocol)
                    continue
                index = state["index"]
                scene = scenes[index // per_scene]
                goal_index = (state["round"] + index % per_scene) % len(scene["refs"])
                difficulty = ("small", "medium", "large")[(index + state["round"]) % 3]
                rng = np.random.default_rng(cfg["seed"] * 1000000 + state["round"] * 10000 + index + (500000 if bootstrap else 0))
                frame = legacy.module(cfg, "room_env").ReferenceFrame(np.asarray(scene["refs"][goal_index]["goal_matrix"]))
                start = sample_start(cfg, scene, frame, rng, difficulty)
                arrays, metadata = run_episode(cfg, scene, goal_index, start, learner.ae, learner.actor, depth,
                                                learner.device, rng, behavior="pvs" if bootstrap else "student",
                                                max_steps=cfg["collection"]["bootstrap_steps" if bootstrap else "steps"],
                                                exploration=cfg["collection"]["exploration_sigma"])
                metadata.update(round=state["round"], difficulty=difficulty, start=start.tolist(),
                                encoder_version=learner.counters["representation"], actor_version=learner.counters["actor"])
                replay.add(arrays, metadata)
                state["index"] += 1
                json_append(output / "collection.jsonl", metadata)
                save_checkpoint(last, learner, state, replay, protocol)
                print(json.dumps({"event": phase, "scene": scene["id"], "episodes": state["index"],
                                  "transitions": replay.transitions}, ensure_ascii=False), flush=True)
                continue
            if phase in ("warmup", "representation", "actor"):
                budget = cfg["training"]["warmup_updates" if phase == "warmup" else f"{phase}_updates"]
                if state["index"] >= budget:
                    following = {"warmup": "representation", "representation": "actor", "actor": "collect"}[phase]
                    state["phase"], state["index"] = following, 0
                    save_checkpoint(last, learner, state, replay, protocol)
                    continue
                row = learner.update(replay, phase)
                state["index"] += 1
                row.update(phase=phase, round=state["round"], update=state["index"], replay_transitions=replay.transitions)
                json_append(output / "training.jsonl", row)
                if state["index"] % 20 == 0 or state["index"] == budget:
                    save_checkpoint(last, learner, state, replay, protocol)
                    print(json.dumps({"event": "update", **row}), flush=True)
                continue
            if phase == "validate":
                if (state["round"] + 1) % cfg["evaluation"]["validate_every"] == 0:
                    summary = evaluate(cfg, output, learner, depth, cap=cfg["evaluation"].get("training_validation_cap", 12),
                                       plants=["P0"], stop=stopper)
                    json_append(output / "validation.jsonl", {"round": state["round"], **summary})
                    if not summary["complete"]:
                        save_checkpoint(last, learner, state, replay, protocol)
                        json_write(output / "status.json", {"state": "validation_pending", **state})
                        return
                    score = [summary["macro_scene_success"], summary["worst_scene_success"], -summary["mean_integrated_error"]]
                    if state["best_score"] is None or tuple(score) > tuple(state["best_score"]):
                        state["best_score"] = score
                        save_checkpoint(output / "best.pt", learner, state, replay, protocol)
                state["round"] += 1
                state["phase"] = "complete" if state["round"] >= cfg["training"]["rounds"] else "representation"
                state["index"] = 0
                save_checkpoint(last, learner, state, replay, protocol)
            json_write(output / "status.json", {"state": "running", **state, "counters": learner.counters})
        json_write(output / "status.json", {"state": "complete", **state, "counters": learner.counters,
                                            "note": "Training completion is not evidence of stability or test success"})
    finally:
        stopper.close()
