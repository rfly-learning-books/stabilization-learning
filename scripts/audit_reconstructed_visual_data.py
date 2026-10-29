"""Independent CPU audit of reconstructed training RGB and unchanged labels.

Writes only INDEPENDENT_AUDIT.json. It never modifies DATA_MANIFEST or any
manifest entry, original files, learner state, or reference RGB.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "assets/original_visual_initialization"
LABEL_KEYS = ("offset", "action", "cost", "terminal", "truncated", "previous_action")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def clear(position, obstacles):
    return min(np.linalg.norm(np.maximum(np.abs(position - np.asarray(row["center"])) - row["halfsize"], 0.))
               for row in obstacles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-bundle", type=Path, required=True)
    args = parser.parse_args()
    began = time.time()
    data = args.data_bundle.resolve()
    check(data != ORIGINAL and not data.is_relative_to(ORIGINAL), "Audit destination must be the new bundle")
    manifest_path = data / "DATA_MANIFEST.json"
    before_manifest = sha(manifest_path)
    manifest = read(manifest_path)
    check(manifest["complete"] is True, "Data preparation is incomplete")
    check(manifest["protocol_kind"] == "reconstructed_environment" and manifest["original_rgb_equivalent"] is False, "Wrong data protocol")
    check(manifest["validation_test_RGB_used"] is False, "Held-out RGB use declared")
    entries = {entry["relative"]: entry for entry in manifest["entries"]}
    check(len(entries) == len(manifest["entries"]), "Duplicate manifest entries")
    for relative, entry in entries.items():
        path = (data / relative).resolve()
        check(path.is_relative_to(data), "Manifest path escapes bundle")
        check(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Data hash/size mismatch: " + relative)
    original_manifest = read(ROOT / "migration/visual_initialization_bundle.json")
    original_hashes = {entry["relative"]: entry["sha256"] for entry in original_manifest["entries"]}
    # Hash the original compact bundle, including weights, without loading
    # any evaluation RGB or using it to alter the reconstructed scenes.
    for relative, expected in original_hashes.items():
        check(sha(ORIGINAL / relative) == expected, "Original bundle modified: " + relative)
    check(manifest["source_initial_checkpoint_sha256"] == original_hashes["initial.pt"], "Wrong source initialization")
    scene_audit = ROOT / "migration/visual_scene_reconstruction/SCENE_AUDIT.json"
    check(sha(scene_audit) == manifest["scene_audit_sha256"], "Scene audit identity changed")
    audit = read(scene_audit)
    check(audit["passed"] and not audit["original_rgb_equivalent"], "Wrong scene audit")
    check(sha(audit["scene_manifest"]) == audit["scene_manifest_sha256"], "Renderer's actual manifest changed")
    for entry in audit["source_files"] + audit["asset_files"]:
        check(sha(entry["path"]) == entry["sha256"], "Reconstructed runtime source/asset changed")
    for name in ("scenes.json", "evaluation_starts.json", "training_anchors.json"):
        check(sha(data / name) == original_hashes[name], "Frozen metadata changed: " + name)
    scenes = {row["id"]: row for row in read(data / "scenes.json")["scenes"]}
    rows = read(data / "replay/index.json")
    check(len(rows) == 48 and sum(row["length"] for row in rows) == 2822, "Bootstrap size mismatch")
    import torch
    initial = torch.load(ORIGINAL / "initial.pt", map_location="cpu", weights_only=False)
    check(rows == initial["replay_rows"], "Original replay ordering or metadata changed")
    del initial
    progress = read(data / "preparation_progress.json")
    observed_refs = set()
    results = []
    maximum_physics, maximum_cost = 0., 0.
    for row in rows:
        scene = scenes[row["scene_id"]]
        check(row["split"] == "train" and scene["split"] == "train", "Non-training shard")
        new_path = data / "replay" / row["file"]
        old_path = ORIGINAL / "replay" / row["file"]
        check("replay/" + row["file"] in entries, "Unhashed replay")
        ref_rel = f'references/{row["scene_id"]}_{row["goal"]:02d}.npz'
        check(ref_rel in entries, "Unhashed reference")
        observed_refs.add(ref_rel)
        count = row["length"]
        frame_goal = np.asarray(scene["refs"][row["goal"]]["goal_matrix"])
        forward = frame_goal[:3, 2].copy()
        forward[2] = 0.
        forward /= np.linalg.norm(forward)
        body0 = np.column_stack([forward, np.cross([0., 0., 1.], forward), [0., 0., 1.]])
        rotation = frame_goal[:3, :3].T @ body0
        with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old, np.load(data / ref_rel, allow_pickle=False) as ref:
            for name in LABEL_KEYS:
                check(new[name].dtype == old[name].dtype and np.array_equal(new[name], old[name]), "Physical label altered: " + name)
            check(new["rgb"].shape == (count + 1, 256, 256, 3) and new["rgb"].dtype == np.uint8, "Malformed RGB array")
            check(new["goal_rgb"].shape == (256, 256, 3), "Malformed goal RGB")
            check(np.array_equal(new["goal_rgb"], ref["rgb"]) and np.array_equal(new["goal_depth"], ref["depth"]), "Replay/reference RGB-depth mismatch")
            check(np.array_equal(new["rotation"], rotation.astype(np.float32)) and np.array_equal(ref["camera_from_body"], rotation), "Reference extrinsics mismatch")
            check(np.array_equal(new["goal_depth"], new["collection_depth"]), "Bootstrap depth mismatch")
            check(np.isfinite(new["goal_depth"]).all() and new["goal_depth"].min() > 0., "Invalid depth estimate")
            check(np.allclose(np.linalg.norm(new["action"][:, :3], axis=-1).clip(max=1), np.linalg.norm(new["action"][:, :3], axis=-1), atol=1e-6)
                  and np.max(np.abs(new["action"][:, 3])) <= 1.000001, "Velocity bound exceeded")
            expected_previous = np.concatenate([np.zeros((1, 4), np.float32), new["action"][:-1]])
            check(np.array_equal(new["previous_action"], expected_previous), "Previous action mismatch")
            trunc = np.zeros(count, bool)
            trunc[-1] = not bool(new["terminal"][-1])
            check(np.array_equal(new["truncated"], trunc) and not new["terminal"][:-1].any(), "Invalid episode boundary")
            position = frame_goal[:3, 3]
            physical, costs, hold = [], [], 0
            scale = np.asarray([.4, .4, .4, .35], np.float32)
            for i, (offset, action, next_offset) in enumerate(zip(new["offset"][:-1], new["action"], new["offset"][1:])):
                p = position + body0 @ offset[:3].astype(float)
                command = scale * action.astype(float)
                p_next = p + rz(float(offset[3]) + .05 * command[3]) @ body0 @ command[:3] * .1
                expected = np.r_[body0.T @ (p_next - position), float(offset[3]) + command[3] * .1]
                physical.append(float(np.max(np.abs(expected - next_offset))))
                actual_p = position + body0 @ next_offset[:3].astype(float)
                d = scene["dims"]
                outside = not (-d["hw"] + .3 < actual_p[0] < d["hw"] - .3 and -d["hl"] + .3 < actual_p[1] < d["hl"] - .3 and .30 < actual_p[2] < d["height"] - .4)
                failed = clear(actual_p, scene["obstacles"]) <= .15 or outside
                E = float(np.sum(next_offset[:3].astype(float) ** 2) + (float(next_offset[3]) / np.radians(25)) ** 2)
                costs.append(abs(.1 * (E + .02 * float(action @ action)) + (20 if failed else 0) - float(new["cost"][i])))
                good = np.linalg.norm(next_offset[:3]) < .02 and abs(float(next_offset[3])) < np.radians(1)
                hold = hold + 1 if good else 0
                check(bool(new["terminal"][i]) == bool(failed or hold >= 11), "Terminal flag inconsistent with geometry and tolerance")
            maximum_physics = max(maximum_physics, max(physical))
            maximum_cost = max(maximum_cost, max(costs))
            check(max(physical) < 1e-6 and max(costs) < 2e-5, "P0/physical cost reconstruction failed")
            centers = new["smooth_center_index"]
            check(len(centers) == len(new["smooth_step"]) == len(new["smooth_delta"]), "Triplet bookkeeping mismatch")
            check(np.all((centers >= 0) & (centers < count)) and np.array_equal(new["smooth_center"], new["rgb"][centers]), "Triplet center RGB mismatch")
            check(new["smooth_minus"].shape == new["smooth_plus"].shape == new["smooth_center"].shape, "Triplet image shape mismatch")
            check(np.isfinite(new["smooth_step"]).all() and np.all(new["smooth_step"] > 0), "Invalid smoothing scale")
            done = progress["completed_shards"][row["file"]]
            check(done["sha256"] == entries["replay/" + row["file"]]["sha256"] and done["source_sha256"] == original_hashes["replay/" + row["file"]], "Preparation source/current provenance mismatch")
            results.append({"file": row["file"], "scene_id": row["scene_id"], "goal_id": row["goal"], "transitions": count,
                            "physical_labels_exact": True, "rgb_reference_equal": True, "p0_max_abs": max(physical), "cost_max_abs": max(costs)})
    actual_references = {name for name in entries if name.startswith("references/")}
    check(actual_references == observed_refs, "Unexpected or unused reference files")
    check(sha(manifest_path) == before_manifest, "DATA_MANIFEST changed during audit")
    result = dict(schema=1, passed=True, completed=datetime.now(timezone.utc).isoformat(),
        protocol_kind="reconstructed_environment", original_rgb_equivalent=False,
        data_manifest_sha256=before_manifest, audit_script_sha256=sha(Path(__file__)),
        independent=True, gpu_training_started=False, heldout_rgb_loaded=False,
        shards=48, transitions=2822, references=len(observed_refs),
        original_bundle_files_verified=len(original_hashes), original_bundle_unmodified=True,
        physical_labels_exact=True, frozen_metadata_preserved=True, train_only=True,
        per_shard_reference_consistent=True, all_data_manifest_entries_sha256_verified=True,
        data_manifest_and_entries_not_modified=True, p0_max_abs=maximum_physics, cost_max_abs=maximum_cost,
        checks=results, elapsed_seconds=time.time() - began)
    output = data / "INDEPENDENT_AUDIT.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "shards", "transitions", "references", "p0_max_abs", "cost_max_abs", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
