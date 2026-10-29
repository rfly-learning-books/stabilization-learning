"""Shared provenance, cases and loading for the explicitly authorized v5 diagnosis.

No tests, training or simulation start on import. The old test split is never
returned by this module. A new output root is required, and its budget cannot be
reset by reopening it. Old selected weights/replay are diagnostic inputs: this
work is not a new from-scratch performance claim.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

PROJECT = Path(__file__).resolve().parents[2]
OLD_ROOT = PROJECT / "runs/rgb_gpu_coverage_v4_seed81_20261027_141542"
for _p in (PROJECT, PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    """Create a new atomic artifact, rejecting replacement of existing results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows rename rejects an already-existing destination.
    tmp.rename(path)


def guard(root):
    root = Path(root)
    p = read_json(root / "PROTOCOL.json")
    if (root / "STOP").exists():
        raise RuntimeError("New diagnostic run STOP requested")
    if time.time() >= p["deadline_unix"]:
        raise RuntimeError("New diagnostic run budget exhausted; no automatic extension")


def event(root, stage, **payload):
    write_json(Path(root) / "events" / f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex[:8]}.json",
               dict(stage=stage, pid=os.getpid(), unix=time.time(), **payload))


@contextmanager
def gpu_lease(root, stage):
    """One GPU worker across cooperating agents; no stale-lock auto-removal."""
    root = Path(root)
    guard(root)
    lock = root / "GPU_LEASE.json"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), stage=stage, unix=time.time()), stream)
    event(root, "gpu_acquired", owner=stage)
    try:
        yield
    finally:
        if lock.exists() and read_json(lock).get("pid") == os.getpid():
            lock.unlink()
        event(root, "gpu_released", owner=stage)


def freeze_stage(root, name, settings, source_files=()):
    root = Path(root)
    guard(root)
    sources = [Path(__file__), *map(Path, source_files)]
    record = dict(name=name, created_unix=time.time(), settings=settings,
                  protocol_sha256=sha256(root / "PROTOCOL.json"),
                  sources={str(p.resolve()): sha256(p) for p in sources})
    write_json(root / "stage_protocols" / (name + ".json"), record)
    return record


def initialize(root, *, max_hours=2.0, seed=8127):
    import numpy as np
    from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
    from extensions.joint_certificate_v4.rollout import sample_offset
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    old = read_json(OLD_ROOT / "protocol.json")
    frozen = read_json(OLD_ROOT / "FROZEN.json")
    assert sha256(OLD_ROOT / frozen["checkpoint"]) == frozen["sha256"]
    _, cfg, scenes = load_runtime(PROJECT)
    train = sorted([s for s in scenes if s["split"] == "train"], key=lambda s: s["id"])
    chosen = [train[i] for i in np.linspace(0, len(train)-1, 8, dtype=int)]
    rng = np.random.default_rng(seed)
    cases = []
    for scene in chosen:
        for goal in range(4):
            for difficulty in ("near", "medium", "large"):
                offset = sample_offset(cfg, scene, goal, difficulty, rng)
                cases.append(dict(scene_id=scene["id"], goal_id=goal,
                                  difficulty=difficulty, repeat=0, offset=offset.tolist(),
                                  split="train", case_id=f"diag_{scene['id']}_g{goal}_{difficulty}"))
    assert len(cases) == 96
    val = old["config"]["evaluation_manifest"]["val"]
    assert len(val) == 48
    lookup = {s["id"]: s for s in scenes}
    assert all(lookup[c["scene_id"]]["split"] == "val" for c in val)
    write_json(root / "TRAIN_DIAG_CASES.json", cases)
    write_json(root / "VALIDATION_CASES.json", val)
    now = time.time()
    protocol = dict(schema="rgb_diagnostic_v5", started_unix=now,
                    deadline_unix=now + max_hours * 3600, max_hours=max_hours,
                    started_local=datetime.fromtimestamp(now, timezone(timedelta(hours=8))).isoformat(),
                    seed=seed, old_root=str(OLD_ROOT), old_protocol_sha256=sha256(OLD_ROOT/"protocol.json"),
                    checkpoint=frozen["checkpoint"], checkpoint_sha256=frozen["sha256"],
                    purpose="development diagnosis and controlled short repair, not final held-out evaluation",
                    scope="old v4 weights and archived training replay explicitly reused; no from-scratch claim",
                    original_run_unchanged=True, old_test_cases_used=False,
                    train_cases_sha256=sha256(root/"TRAIN_DIAG_CASES.json"),
                    validation_cases_sha256=sha256(root/"VALIDATION_CASES.json"),
                    stages=["derivative_geometry", "control_ablation", "gradient_audit", "conditional_training_ablation"],
                    control=dict(methods=["baseline", "target_J", "relative_damping", "identity_M", "FD_current_J"],
                                 plants=["P0", "P1"], cases=48, max_steps=200,
                                 relative_damping="0.02*trace(J.T M J)/4 + 1e-8; other terms unchanged",
                                 success=cfg["collection"],
                                 fd_step_selection="training-state numerical consistency only, never validation success"),
                    conditional_training=dict(arms=["original_alternating", "original_split", "cal_precision_alternating", "cal_precision_split"],
                                 updates_per_arm=dict(feature=600, metric=600),
                                 batch_size=16, initialization="same v4 selected weights and empty Adam",
                                 data="same fixed 600 training batches, same order in each module",
                                 only_loss_change="calibration denominator max(E,min(pos_tol^2,(yaw_tol/yaw_scale)^2)); original V bounds unchanged",
                                 split_semantics="all feature parameter updates then all metric parameter updates; M can vary with goal features while theta is frozen",
                                 evaluation="final checkpoint only; all 48 validation cases P0/P1 with learned and shared-feature identity M",
                                 activation="root records evidence-based decision before training; no training if core derivative/geometry error blocks interpretation"),
                    selection="development ranking: strict success, failure/tail errors, integrated E, input, and matched successful arrival times",
                    automatic_resume=False, no_old_stop_or_task_changes=True,
                    caveats=["one existing seed", "validation development, not untouched test", "44 prior rooms and public FastDepth",
                             "4DOF outer loop, no real robot", "true-depth and FD variants are privileged diagnostic controls",
                             "inference wall time not injected in dt", "no stability proof", "no reward threshold relaxation"])
    write_json(root / "PROTOCOL.json", protocol)
    event(root, "initialized", train_diagnostic_states=len(cases), validation_cases=len(val))
    return protocol


def bootstrap(root, device="cuda:0"):
    import torch
    from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
    from stabilization_gym.tasks_gpu_rgb.geometry import GPURGBGeometry
    from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
    from extensions.joint_certificate_v4.config import RGBConfig
    from extensions.joint_certificate_v3.rgb import _new_learner, _read_verified_checkpoint, _restore_for_evaluation
    from stabilization_learning.joint_v2.runtime import seed_all
    root = Path(root).resolve()
    guard(root)
    p = read_json(root / "PROTOCOL.json")
    old = read_json(OLD_ROOT / "protocol.json")
    if sha256(OLD_ROOT / "protocol.json") != p["old_protocol_sha256"]:
        raise RuntimeError("Original protocol changed")
    for category in ("sources", "assets"):
        for path, expected in old[category].items():
            if sha256(path) != expected:
                raise RuntimeError("Original frozen input changed: " + path)
    for name, key in (("TRAIN_DIAG_CASES.json", "train_cases_sha256"), ("VALIDATION_CASES.json", "validation_cases_sha256")):
        if sha256(root/name) != p[key]:
            raise RuntimeError("Diagnostic case manifest changed: " + name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(p["seed"])
    _, cfg, scenes = load_runtime(PROJECT)
    config = RGBConfig.from_dict(old["config"]["run"])
    if device != config.device:
        from dataclasses import replace
        config = replace(config, device=device)
    learner = _new_learner(cfg, config)
    ck = _read_verified_checkpoint(OLD_ROOT, p["checkpoint"], p["old_protocol_sha256"])
    _restore_for_evaluation(learner, ck)
    learner.features.eval(); learner.metric.eval()
    geometry = GPURGBGeometry(cfg, device)
    lookup = {s["id"]: s for s in scenes}
    def make_env(cases, plant="P0", max_steps=200):
        guard(root)
        chosen = [lookup[c["scene_id"]] for c in cases]
        if any(s["split"] not in {"train", "val"} for s in chosen):
            raise ValueError("Test-scene execution is forbidden in development diagnostics")
        return GPUParallelRGBEnv(cfg, chosen, plant=plant, max_steps=max_steps,
                                 device=device, cache_dir=PROJECT/".cache/rgb_gpu_warp", samples_per_axis=2)
    ctx = SimpleNamespace(project=PROJECT, root=root, old_root=OLD_ROOT, cfg=cfg, scenes=scenes,
                          learner=learner, geometry=geometry, config=config, device=device,
                          train_cases=read_json(root/"TRAIN_DIAG_CASES.json"),
                          validation_cases=read_json(root/"VALIDATION_CASES.json"),
                          protocol=p, checkpoint=ck, make_env=make_env,
                          provenance=dict(checkpoint=p["checkpoint"], checkpoint_sha256=p["checkpoint_sha256"],
                                          old_protocol_sha256=p["old_protocol_sha256"], sources_checked=len(old["sources"]),
                                          assets_checked=len(old["assets"]), old_test_used=False),
                          guard=lambda: guard(root), event=lambda stage, **kw: event(root,stage,**kw))
    return ctx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "check"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(initialize(args.root), ensure_ascii=False))
    else:
        with gpu_lease(args.root, "bootstrap_check"):
            ctx = bootstrap(args.root)
            print(json.dumps(ctx.provenance, ensure_ascii=False))
