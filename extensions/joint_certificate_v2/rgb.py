"""Fresh reconstructed-room RGB joint training and group-frozen evaluation.

The production path uses actual rendered RGB, the random legacy smooth RGB
backbone and fixed public FastDepth. No historical policy or replay is loaded.
Resume is deliberately unsupported: complete-update checkpoints preserve the
state needed for a separately reviewed recovery, never silently restart a run.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable
import json
import math
import os
import time
import traceback

import numpy as np
import torch

from stabilization_gym.tasks_v2.reconstructed_rgb import (
    load_runtime, ReconstructedRGBEnv, RGBGeometry, random_raw_autoencoder,
)
from stabilization_gym.tasks_v2.visual import VisualObservation, VisualTaskAdapter, quadratic_value
from stabilization_learning.joint_v2.visual import (
    NormalizedRGBFeatures, GoalConditionedMetric, JointVisualConfig,
    JointVisualLearner, VisualBatch,
)
from stabilization_learning.joint_v2.runtime import (
    RunStore, RunStopped, atomic_json, file_sha256, seed_all,
)


@dataclass(frozen=True)
class RGBRunConfig:
    seed: int = 81
    group_seeds: tuple[int, ...] = (81, 82, 83)
    device: str = "cuda"
    warmup_updates: int = 1000
    joint_pairs: int = 6500
    batch_size: int = 4
    bootstrap_episodes_per_scene: int = 1
    bootstrap_steps: int = 60
    collection_every_pairs: int = 100
    collection_episodes: int = 2
    collection_steps: int = 120
    validation_every_pairs: int = 500
    validation_max_steps: int = 200
    test_max_steps: int = 200
    train_scene_limit: int = 0
    validation_case_limit: int = 0
    test_case_limit: int = 0
    evaluation_seed: int = 2026102701
    calibration_samples: int = 32
    probe_count: int = 4
    replay_max_episodes: int = 96
    replay_cache_episodes: int = 2
    geometry_cache_samples: int = 16
    max_hours: float = 14.0
    minimum_free_gib: float = 20.0
    scope: str = "formal_training"
    learner: JointVisualConfig = field(default_factory=JointVisualConfig)

    def __post_init__(self):
        if self.seed not in self.group_seeds or len(set(self.group_seeds)) != len(self.group_seeds):
            raise ValueError("Declare a unique seed group containing this seed")
        for name in ("warmup_updates", "joint_pairs", "batch_size", "bootstrap_episodes_per_scene",
                     "bootstrap_steps", "collection_every_pairs", "collection_episodes", "collection_steps",
                     "validation_every_pairs", "validation_max_steps", "test_max_steps", "calibration_samples",
                     "probe_count", "replay_max_episodes", "replay_cache_episodes", "geometry_cache_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("train_scene_limit", "validation_case_limit", "test_case_limit"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not math.isfinite(self.max_hours) or self.max_hours <= 0:
            raise ValueError("A finite positive run budget is required")
        if self.scope not in {"formal_training", "engineering_smoke"}:
            raise ValueError("Declare formal_training or engineering_smoke scope")

    @classmethod
    def smoke(cls, seed: int = 81, *, device: str = "cuda"):
        return cls(seed=seed, group_seeds=(seed,), device=device, warmup_updates=1,
                   joint_pairs=1, batch_size=2, bootstrap_steps=8, collection_steps=8,
                   train_scene_limit=1, validation_case_limit=1, test_case_limit=1,
                   validation_max_steps=3, test_max_steps=3, calibration_samples=4,
                   probe_count=2, max_hours=0.25, minimum_free_gib=0., scope="engineering_smoke")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "group_seeds" in values:
            values["group_seeds"] = tuple(values["group_seeds"])
        if "learner" in values and not isinstance(values["learner"], JointVisualConfig):
            values["learner"] = JointVisualConfig(**values["learner"])
        return cls(**values)


def _guard(store, external):
    store.guard()
    if external is not None and external() is False:
        raise RunStopped("External original-run budget or STOP guard requested termination")


def _npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Never overwrite a recorded episode: " + str(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RawRGBReplay:
    """Fresh, scene-balanced raw-image replay with hash-checked bounded caches."""
    def __init__(self, root, rng, *, max_episodes=96, cache_episodes=2, geometry_cache=16):
        self.root, self.rng = Path(root), rng
        self.root.mkdir(parents=True, exist_ok=False)
        self.rows, self.cache, self.geometry_cache = [], OrderedDict(), OrderedDict()
        self.maximum, self.cache_limit, self.geometry_limit = max_episodes, cache_episodes, geometry_cache
        self.index = self.root / "index.json"
        atomic_json(self.index, {"schema": 1, "rows": []})

    def add(self, arrays, metadata, *, complete=True):
        identifier = metadata["episode_id"]
        path = self.root / f"episode_{identifier:06d}{'' if complete else '_partial'}.npz"
        _npz(path, arrays)
        row = dict(metadata, file=path.name, sha256=file_sha256(path),
                   transitions=len(arrays["action"]), complete=bool(complete))
        if complete and row["transitions"]:
            if row["split"] != "train":
                raise ValueError("Evaluation data cannot enter training replay")
            self.rows.append(row)
            if len(self.rows) > self.maximum:
                self.rows = self.rows[-self.maximum:]
            atomic_json(self.index, {"schema": 1, "rows": self.rows})
        return row

    def load(self, row):
        key = row["file"]
        if key not in self.cache:
            path = self.root / key
            if file_sha256(path) != row["sha256"]:
                raise RuntimeError("Raw replay shard hash mismatch")
            with np.load(path, allow_pickle=False) as saved:
                self.cache[key] = {name: saved[name] for name in saved.files}
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def sample_keys(self, size):
        if not self.rows:
            raise ValueError("Cannot sample empty fresh replay")
        by_scene = {}
        for row in self.rows:
            by_scene.setdefault(row["scene_id"], []).append(row)
        scenes = sorted(by_scene)
        keys = []
        for _ in range(size):
            pool = by_scene[scenes[int(self.rng.integers(len(scenes)))]]
            row = pool[int(self.rng.integers(len(pool)))]
            keys.append((row, int(self.rng.integers(row["transitions"]))))
        return keys

    def batch(self, keys, geometry, *, with_geometry=True, guard=None):
        images, goals, tangents, following = [], [], [], []
        offsets, next_offsets, action, previous, durations, mask, versions = [], [], [], [], [], [], []
        for row, index in keys:
            if guard is not None:
                guard()
            data = self.load(row)
            key = (row["file"], index)
            public = dict(rgb=data["rgb"][index], goal_rgb=data["goal_rgb"],
                          rotation=data["rotation"], previous_action=data["previous_action"][index])
            if with_geometry:
                if key not in self.geometry_cache:
                    self.geometry_cache[key] = tuple(t.detach().cpu() for t in geometry.observe(public))
                    while len(self.geometry_cache) > self.geometry_limit:
                        self.geometry_cache.popitem(last=False)
                self.geometry_cache.move_to_end(key)
                x, goal, tangent = (value.to(geometry.device) for value in self.geometry_cache[key])
            else:
                # AE warmup/calibration does not need depth or image tangents.
                x = geometry.vision.image_tensor(geometry.cfg, [public["rgb"]], geometry.device)
                goal = geometry.vision.image_tensor(geometry.cfg, [public["goal_rgb"]], geometry.device)
                tangent = x.new_zeros(1, 4, *x.shape[1:])
            images.append(x); goals.append(goal); tangents.append(tangent)
            following.append(geometry.vision.image_tensor(geometry.cfg, [data["rgb"][index + 1]], geometry.device))
            offsets.append(data["offset"][index]); next_offsets.append(data["offset"][index + 1])
            action.append(data["actual_normalized_action"][index]); previous.append(data["previous_action"][index])
            durations.append(data["dt"][index]); mask.append(row["policy"] == "current_controller")
            versions.append(row["behavior_version"])
        tensor = lambda value: torch.as_tensor(np.asarray(value), device=geometry.device, dtype=images[0].dtype)
        version = versions[0] if len(set(versions)) == 1 else -1
        return VisualBatch(VisualObservation(torch.cat(images), torch.cat(goals), torch.cat(tangents)),
                           torch.cat(following), tensor(offsets), tensor(next_offsets), tensor(action),
                           tensor(durations), torch.as_tensor(mask, device=geometry.device, dtype=torch.bool),
                           previous_action=tensor(previous), behavior_version=version, split="train")


def _balanced_cases(rows, limit, seed):
    """Deterministically balance rooms first, then difficulty and goal IDs.

    This is a pre-training metadata selection, never a performance selection.
    Every candidate start has already been independently generated in its split.
    """
    if limit == 0 or limit >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    remaining = [rows[int(i)] for i in rng.permutation(len(rows))]
    scene_counts, difficulty_counts, goal_counts, chosen = {}, {}, {}, []
    for _ in range(limit):
        def score(index):
            row = remaining[index]
            sc = scene_counts.get(row["scene_id"], 0)
            dc = difficulty_counts.get(row["difficulty"], 0)
            gc = goal_counts.get(row["goal_id"], 0)
            return sc, dc + gc, dc, gc, index
        row = remaining.pop(min(range(len(remaining)), key=score))
        chosen.append(row)
        for counts, name in ((scene_counts, "scene_id"), (difficulty_counts, "difficulty"), (goal_counts, "goal_id")):
            counts[row[name]] = counts.get(row[name], 0) + 1
    return chosen


def _manifest(cfg, scenes, config):
    from lac_vs.environment import sample_start
    from lac_vs import legacy
    room = legacy.module(cfg, "room_env")
    result = {"population_counts": {}, "subset_policy":
              "All starts predeclared; limits select rooms first then balance difficulty/goal, using fixed evaluation seed"}
    for split in ("val", "test"):
        rng = np.random.default_rng(config.evaluation_seed + (0 if split == "val" else 1))
        rows = []
        for scene in scenes:
            if scene["split"] != split:
                continue
            for goal_id, reference in enumerate(scene["refs"]):
                frame = room.ReferenceFrame(np.asarray(reference["goal_matrix"]))
                for difficulty in ("small", "medium", "large"):
                    rows.append(dict(case_id=f"{scene['id']}_g{goal_id}_{difficulty}",
                                     scene_id=scene["id"], goal_id=goal_id, difficulty=difficulty,
                                     offset=sample_start(cfg, scene, frame, rng, difficulty).tolist()))
        limit = config.validation_case_limit if split == "val" else config.test_case_limit
        result["population_counts"][split] = len(rows)
        result[split] = _balanced_cases(rows, limit, config.evaluation_seed + (911 if split == "val" else 912))
        if not result[split]:
            raise RuntimeError("Declared evaluation split is empty")
    return result


def _inputs(project, adapter, cfg):
    import inspect
    from stabilization_learning.joint_v2 import visual as learner_module
    from stabilization_gym.tasks_v2 import visual as task_module
    from stabilization_gym.tasks_v2 import reconstructed_rgb as bridge_module
    from stabilization_learning.joint_v2 import runtime as runtime_module
    import goal_fixed_core
    sources = {Path(__file__), Path(inspect.getfile(learner_module)), Path(inspect.getfile(task_module)),
               Path(inspect.getfile(bridge_module)), Path(inspect.getfile(runtime_module))}
    base = project / "migration/visual_scene_reconstruction"
    for row in adapter.scene_audit["source_files"]:
        path = Path(row["path"])
        sources.add(path if path.is_absolute() else base / path)
    # Freeze the reused RGB/backbone/image geometry sources as well as the bridge.
    sources.update((project / "third_party/visual_original/runtime").glob("*.py"))
    sources.update((project / "third_party/visual_original/lac_vs").glob("*.py"))
    depth_root = goal_fixed_core.ROOT / cfg["goal_fixed"]["depth_path"]
    sources.update(depth_root.glob("*.py"))
    assets = {base / "scenes.json", base / "SCENE_AUDIT.json", depth_root / "extracted_state_dict.pt"}
    for row in adapter.scene_audit["asset_files"]:
        path = Path(row["path"])
        assets.add(path if path.is_absolute() else base / path)
    return sorted(sources), sorted(assets)


def _new_learner(cfg, config):
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Declared CUDA device is unavailable; do not silently change protocol backend")
    features = NormalizedRGBFeatures(random_raw_autoencoder(cfg, config.device), 16).to(config.device)
    metric = GoalConditionedMetric(16, 32).to(config.device)
    return JointVisualLearner(features, metric, VisualTaskAdapter(), config.learner)


def _collect(cfg, scene, learner, geometry, replay, rng, episode_id, max_steps, guard,
             *, exploratory, store):
    guard()
    env = ReconstructedRGBEnv(cfg, scene, plant="P0", max_steps=max_steps)
    goal_id = int(rng.integers(len(scene["refs"])))
    difficulty = ("small", "medium", "large")[int(rng.integers(3))]
    metadata = dict(episode_id=episode_id, scene_id=scene["id"], split=scene["split"],
                    goal_id=goal_id, difficulty=difficulty,
                    policy="random_bounded" if exploratory else "current_controller",
                    behavior_version=learner.version)
    records = {"rgb": [], "offset": [], "action": [], "actual_normalized_action": [],
               "previous_action": [], "dt": []}
    initial_public = None

    def arrays():
        out = {key: np.asarray(value) for key, value in records.items()}
        out.update(goal_rgb=initial_public["goal_rgb"], rotation=initial_public["rotation"])
        return out

    try:
        public, info = env.reset(seed=int(rng.integers(2**31 - 1)), options=dict(goal_id=goal_id, difficulty=difficulty))
        initial_public = public
        records["rgb"].append(public["rgb"])
        records["offset"].append(info["supervision"]["offset"])
        for _ in range(max_steps):
            guard()
            if exploratory:
                action = rng.uniform(-1, 1, 4)
                action /= max(1., np.linalg.norm(action[:3]), abs(action[3]))
            else:
                with torch.no_grad():
                    observation = VisualObservation(*geometry.observe(public))
                    action = learner.policy(observation, create_graph=False).action[0].cpu().numpy()
            following, _, terminated, truncated, info = env.step(action)
            truth = info["supervision"]
            records["action"].append(action)
            records["actual_normalized_action"].append(truth["actual_velocity"] / np.array([.4, .4, .4, .35]))
            records["previous_action"].append(public["previous_action"])
            records["dt"].append(info["dt"])
            records["rgb"].append(following["rgb"])
            records["offset"].append(truth["next_offset"])
            public = following
            if terminated or truncated:
                break
        row = replay.add(arrays(), dict(metadata, reason=info["reason"]))
        store.event("replay_episode", **row)
        return row
    except BaseException:
        if initial_public is not None and records["action"]:
            row = replay.add(arrays(), dict(metadata, reason="interrupted"), complete=False)
            store.event("partial_replay_preserved", **row)
        raise
    finally:
        env.close()


def _evaluate(cfg, scenes_by_id, cases, learner, geometry, guard, *, plant="P0",
              method="learned", max_steps=200, output=None, store=None):
    rows = []
    for number, case in enumerate(cases):
        guard()
        env = ReconstructedRGBEnv(cfg, scenes_by_id[case["scene_id"]], plant=plant, max_steps=max_steps)
        before_version = learner.version
        trace = {"error": [], "V": [], "J": [], "action": [], "offset": [], "dt": [],
                 "actual_velocity": [], "physical_error": []}
        try:
            public, info = env.reset(seed=0, options={"goal_id": case["goal_id"], "offset": case["offset"]})
            trace["offset"].append(info["supervision"]["offset"])
            trace["physical_error"].append(info["supervision"]["physical_error"])
            elapsed, integrated, effort, physical_cost, variation, wall_times = 0., 0., 0., 0., 0., []
            previous = np.zeros(4)
            metric_saved = None
            for _ in range(max_steps):
                guard()
                begun = time.perf_counter()
                with torch.no_grad():
                    public_tensor = VisualObservation(*geometry.observe(public))
                    if method == "identity":
                        identity = lambda goal: torch.eye(goal.shape[-1], device=goal.device, dtype=goal.dtype).expand(len(goal), -1, -1)
                        out = learner.adapter.policy(learner.features, identity, public_tensor, create_graph=False)
                    else:
                        out = learner.policy(public_tensor, create_graph=False)
                wall_times.append(time.perf_counter() - begun)
                action = out.action[0].cpu().numpy()
                trace["error"].append(out.error[0].cpu().numpy())
                trace["V"].append(float(out.value[0]))
                trace["J"].append(out.interaction[0].cpu().numpy())
                current_metric = out.metric[0].cpu().numpy()
                if metric_saved is None:
                    metric_saved = current_metric
                elif not np.array_equal(current_metric, metric_saved):
                    raise RuntimeError("Goal-only metric changed during a frozen evaluation episode")
                public, _, terminated, truncated, info = env.step(action)
                truth = info["supervision"]
                elapsed += info["dt"]; integrated += truth["integrated_error"]
                effort += truth["normalized_input_squared_integral"]; physical_cost += truth["physical_cost"]
                variation += float(np.square(action - previous).sum()); previous = action
                trace["action"].append(action); trace["dt"].append(info["dt"])
                trace["actual_velocity"].append(truth["actual_velocity"])
                trace["offset"].append(truth["next_offset"])
                trace["physical_error"].append(truth["physical_error"])
                if terminated or truncated:
                    break
            # Complete successor V with the same frozen model and goal. No
            # second physical step and no fictitious renderer gradient.
            with torch.no_grad():
                x = geometry.vision.image_tensor(cfg, [public["rgb"]], geometry.device)
                goal = geometry.vision.image_tensor(cfg, [public["goal_rgb"]], geometry.device)
                error = learner.features.encode(x) - learner.features.encode(goal)
                metric = torch.as_tensor(metric_saved, device=geometry.device)[None]
                trace["error"].append(error[0].cpu().numpy())
                trace["V"].append(float(quadratic_value(error, metric)[0]))
            if learner.version != before_version:
                raise RuntimeError("Evaluation observed a changing learner version")
            values = np.asarray(trace["V"])
            tolerances = 1e-7 + 1e-4 * np.abs(values[:-1])
            physical_values = np.asarray(trace["physical_error"])
            actual_decay = np.diff(values) + learner.config.decrease_per_second * np.asarray(trace["dt"]) * physical_values[:-1]
            physical_tolerance = 1e-7 + 1e-4 * physical_values
            comparable = ((values >= learner.config.value_lower * physical_values - physical_tolerance)
                          & (values <= learner.config.value_upper * physical_values + physical_tolerance))
            row = dict(**case, plant=plant, method=method, success=bool(info["success"]),
                       reason=info["reason"], steps=len(trace["action"]), elapsed_s=elapsed,
                       integrated_error=integrated, physical_cost=physical_cost,
                       input_squared_integral=effort, action_variation=variation,
                       mean_policy_wall_s=float(np.mean(wall_times)),
                       final_physical_error=float(trace["physical_error"][-1]),
                       actual_V_increase_fraction=float(np.mean(np.diff(values) > tolerances)),
                       actual_decay_violation_fraction=float(np.mean(actual_decay > tolerances)),
                       certificate_comparable_fraction=float(np.mean(comparable)),
                       version=before_version)
            if output is not None:
                path = Path(output) / f"case_{number:04d}.npz"
                _npz(path, dict(**{key: np.asarray(value) for key, value in trace.items()}, M=metric_saved))
                row.update(trajectory=str(path), trajectory_sha256=file_sha256(path),
                           trajectory_scope="features,J,M,physical truth,actions; raw RGB replay is training-only")
            rows.append(row)
            if store is not None:
                store.event("evaluation_episode", **row)
        finally:
            env.close()
    return rows


def _summary(rows):
    return {"episodes": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_integrated_error": float(np.mean([row["integrated_error"] for row in rows])),
            "mean_input_squared_integral": float(np.mean([row["input_squared_integral"] for row in rows])),
            "mean_elapsed_s_all": float(np.mean([row["elapsed_s"] for row in rows])),
            "mean_policy_wall_s": float(np.mean([row["mean_policy_wall_s"] for row in rows])),
            "mean_actual_V_increase_fraction": float(np.mean([row["actual_V_increase_fraction"] for row in rows])),
            "mean_actual_decay_violation_fraction": float(np.mean([row["actual_decay_violation_fraction"] for row in rows])),
            "mean_certificate_comparable_fraction": float(np.mean([row["certificate_comparable_fraction"] for row in rows]))}


def _probe(learner, batch, store, *, phase):
    with torch.no_grad():
        goal = learner.features.encode(batch.observation.goal_image)
        error = learner.features.encode(batch.observation.image) - goal
        metric = learner.metric(goal)
        values = quadratic_value(error, metric)
        physical = learner.adapter.physical_error(batch.offset)
    store.event("fixed_training_probe", phase=phase, step=sum(learner.counts.values()),
                counts=learner.counts.copy(), version=learner.version,
                normalization_scale=float(learner.features.scale), calibrated=bool(learner.features.calibrated),
                M=metric.cpu().tolist(), V=values.cpu().tolist(), E=physical.cpu().tolist(),
                feature_error_norm=error.norm(dim=-1).cpu().tolist())


def _checkpoint(learner, replay, rng, store, *, name, next_pair, collector_episode):
    step = sum(learner.counts.values())
    index_name = name.removesuffix(".pt") + "_replay.json"
    index_path = store.root / "replay_indices" / index_name
    atomic_json(index_path, {"rows": replay.rows})
    runtime = dict(counts=learner.counts.copy(), version=learner.version, next_pair=next_pair,
                   collector_episode=collector_episode, replay_index=str(index_path),
                   replay_index_sha256=file_sha256(index_path), replay_rows=deepcopy(replay.rows),
                   sampler_rng=deepcopy(rng.bit_generator.state), phase="complete_update",
                   learner_config=asdict(learner.config), control_config=asdict(learner.adapter.config),
                   auto_resume_supported=False)
    return store.checkpoint(name, dict(models=dict(features=learner.features.state_dict(), metric=learner.metric.state_dict()),
                            optimizers=dict(features=learner.feature_optimizer.state_dict(), metric=learner.metric_optimizer.state_dict()),
                            runtime_state=runtime), completed_updates=step)


def train(output, config: RGBRunConfig | None = None, guard: Callable | None = None, *,
          project_root=None, original_deadline_unix=None):
    """Create and train a fresh run; refuses every existing output directory."""
    config = config or RGBRunConfig()
    project = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    adapter, cfg, scenes = load_runtime(project)
    manifest = _manifest(cfg, scenes, config)
    source_files, assets = _inputs(project, adapter, cfg)
    protocol_config = dict(**asdict(config), evaluation_manifest=manifest,
                           policy="current_image_J_shared_encoder_goal_only_M",
                           old_weights_loaded=False, old_replay_loaded=False,
                           perception_from_scratch="RGB encoder and M only; FastDepth fixed public weights",
                           physical_scope="P0/P1 visual velocity outer loop; no motor/attitude dynamics",
                           selection="validation success fraction, then lower physical error integral, then lower input",
                           physical_loss="one-step differentiable P0 label model; no renderer gradient",
                           original_deadline_unix=original_deadline_unix,
                           renderer_original_rgb_equivalent=False, auto_resume_supported=False)
    store = RunStore.create(output, protocol_config, max_hours=config.max_hours,
                            sources=source_files, assets=assets, minimum_free_gib=config.minimum_free_gib,
                            deadline_unix=original_deadline_unix)
    check = lambda: _guard(store, guard)
    with store:
        try:
            check(); seed_all(config.seed)
            atomic_json(store.root / "EVALUATION_MANIFEST.json", manifest)
            atomic_json(store.root / "RUNTIME_CONFIG.json", cfg)
            geometry = RGBGeometry(cfg, config.device)
            learner = _new_learner(cfg, config)
            rng = np.random.default_rng(config.seed + 17001)
            replay = RawRGBReplay(store.root / "replay", rng, max_episodes=config.replay_max_episodes,
                                  cache_episodes=config.replay_cache_episodes, geometry_cache=config.geometry_cache_samples)
            training_scenes = [scene for scene in scenes if scene["split"] == "train"]
            if config.train_scene_limit:
                training_scenes = training_scenes[:config.train_scene_limit]
            if not training_scenes:
                raise RuntimeError("No training scenes declared")
            store.event("initialized", seed=config.seed, scope=config.scope, depth=geometry.provenance,
                        train_scenes=[scene["id"] for scene in training_scenes], random_backbone=True, random_metric=True)
            episode_id = 0
            for scene in training_scenes:
                for _ in range(config.bootstrap_episodes_per_scene):
                    _collect(cfg, scene, learner, geometry, replay, rng, episode_id, config.bootstrap_steps,
                             check, exploratory=True, store=store)
                    episode_id += 1
            probe_keys = replay.sample_keys(config.probe_count)
            atomic_json(store.root / "PROBE_MANIFEST.json", [{"file": row["file"], "sha256": row["sha256"], "index": index}
                                                           for row, index in probe_keys])
            probe_batch = replay.batch(probe_keys, geometry, with_geometry=False, guard=check)
            _checkpoint(learner, replay, rng, store, name="initial.pt", next_pair=0, collector_episode=episode_id)
            _probe(learner, probe_batch, store, phase="random_initial")
            for _ in range(config.warmup_updates):
                check()
                batch = replay.batch(replay.sample_keys(config.batch_size), geometry, with_geometry=False, guard=check)
                metrics = learner.warmup(batch)
                store.event("update", block="warmup", step=sum(learner.counts.values()), metrics=metrics)
                _probe(learner, probe_batch, store, phase="warmup")
            calibration = replay.batch(replay.sample_keys(config.calibration_samples), geometry, with_geometry=False, guard=check)
            learner.calibrate(calibration)
            store.event("coordinate_calibrated", after_warmup=learner.counts["warmup"],
                        scale=float(learner.features.scale), coordinate_version=int(learner.features.coordinate_version),
                        samples=config.calibration_samples)
            _probe(learner, probe_batch, store, phase="post_warmup_calibration")
            best = None
            scenes_by_id = {scene["id"]: scene for scene in scenes}
            for pair in range(1, config.joint_pairs + 1):
                check()
                batch = replay.batch(replay.sample_keys(config.batch_size), geometry, guard=check)
                for block, operation in (("feature", learner.update_features), ("metric", learner.update_metric)):
                    check(); metrics = operation(batch)
                    store.event("update", block=block, pair=pair, step=sum(learner.counts.values()), metrics=metrics)
                    _probe(learner, probe_batch, store, phase=block)
                if pair % config.collection_every_pairs == 0 and pair < config.joint_pairs:
                    for _ in range(config.collection_episodes):
                        scene = training_scenes[int(rng.integers(len(training_scenes)))]
                        _collect(cfg, scene, learner, geometry, replay, rng, episode_id, config.collection_steps,
                                 check, exploratory=False, store=store)
                        episode_id += 1
                if pair % config.validation_every_pairs == 0 or pair == config.joint_pairs:
                    name = f"update_{sum(learner.counts.values()):06d}.pt"
                    _checkpoint(learner, replay, rng, store, name=name, next_pair=pair + 1, collector_episode=episode_id)
                    rows = _evaluate(cfg, scenes_by_id, manifest["val"], learner, geometry, check,
                                     max_steps=config.validation_max_steps,
                                     output=store.root / "validation" / f"pair_{pair:06d}", store=store)
                    summary = _summary(rows)
                    score = (summary["success_rate"], -summary["mean_integrated_error"], -summary["mean_input_squared_integral"])
                    store.event("validation", pair=pair, checkpoint=name, summary=summary, score=list(score))
                    if best is None or score > best["score"]:
                        best = dict(score=score, checkpoint=name, update=sum(learner.counts.values()), summary=summary)
            check(); store.verify_inputs()
            store.freeze(best["checkpoint"], selected_update=best["update"], selection=dict(rule=protocol_config["selection"], **best["summary"]))
            final = dict(status="complete", stage="training_frozen", scope=config.scope, seed=config.seed,
                         counts=learner.counts, version=learner.version, selected_checkpoint=best["checkpoint"],
                         selected_update=best["update"], validation=best["summary"],
                         final_checkpoint=f"update_{sum(learner.counts.values()):06d}.pt",
                         group_test_complete=False, group_seeds=list(config.group_seeds),
                         caveat="Engineering smoke is not a performance result" if config.scope == "engineering_smoke" else
                         "Validation only; held-out group testing has not yet run")
            atomic_json(store.root / "FINAL_STATE.json", final)
            store.event("training_complete", **final)
            return final
        except BaseException as error:
            final = dict(status="stopped" if isinstance(error, RunStopped) else "failed", stage="train",
                         error=repr(error), traceback=traceback.format_exc(), auto_resume_supported=False)
            atomic_json(store.root / "FINAL_STATE.json", final)
            store.event("training_stopped", **final)
            raise


def _read_verified_checkpoint(root, name, protocol_hash):
    path = Path(root) / name
    marker = json.loads(path.with_suffix(".COMMITTED.json").read_text(encoding="utf-8"))
    if marker["protocol_sha256"] != protocol_hash or file_sha256(path) != marker["sha256"]:
        raise RuntimeError("Evaluation checkpoint hash/protocol mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["schema"] != "joint_checkpoint_v2" or checkpoint["boundary"] != "complete_update":
        raise RuntimeError("Unsupported/incomplete evaluation checkpoint")
    return checkpoint


def _restore_for_evaluation(learner, checkpoint):
    payload = checkpoint["payload"]
    learner.features.load_state_dict(payload["models"]["features"])
    learner.metric.load_state_dict(payload["models"]["metric"])
    learner.counts = payload["runtime_state"]["counts"].copy()
    learner.version = payload["runtime_state"]["version"]


def test_group(roots, output, *, guard=None, project_root=None, config: RGBRunConfig | None = None,
               original_deadline_unix=None):
    """Test only after the entire declared seed group has frozen its selections."""
    roots = [Path(root).resolve() for root in roots]
    if not roots:
        raise ValueError("Declare the full frozen RGB group")
    protocols = [json.loads((root / "protocol.json").read_text(encoding="utf-8")) for root in roots]
    expected = tuple(protocols[0]["config"]["group_seeds"])
    if any(tuple(protocol["config"]["group_seeds"]) != expected for protocol in protocols):
        raise ValueError("Seed-group declarations differ")
    RunStore.require_frozen_group(roots, expected_seeds=expected)
    if any((root / "WRITER.lock").exists() for root in roots):
        raise RuntimeError("Training writer is still active; group testing cannot start")
    # Budgets/seeds may differ, scientific interfaces, evaluation scope and
    # objectives may not. Check actual source/asset bytes before evaluating.
    ignored = {"seed", "max_hours", "minimum_free_gib", "original_deadline_unix"}
    scientific = {name: value for name, value in protocols[0]["config"].items() if name not in ignored}
    for protocol in protocols:
        if {name: value for name, value in protocol["config"].items() if name not in ignored} != scientific:
            raise ValueError("Frozen group scientific configurations differ")
        for category in ("sources", "assets"):
            if protocol[category] != protocols[0][category]:
                raise ValueError("Frozen group source/asset declarations differ")
            for path, expected_hash in protocol[category].items():
                if file_sha256(path) != expected_hash:
                    raise RuntimeError("Frozen training source/asset changed before testing: " + path)
    manifest = protocols[0]["config"]["evaluation_manifest"]
    if any(protocol["config"]["evaluation_manifest"] != manifest for protocol in protocols):
        raise ValueError("Group comparisons require exactly shared declared test starts")
    fields = RGBRunConfig.__dataclass_fields__
    first_cfg = {name: value for name, value in protocols[0]["config"].items() if name in fields}
    config = config or RGBRunConfig.from_dict(first_cfg)
    if (config.test_max_steps != first_cfg["test_max_steps"] or config.scope != first_cfg["scope"]
            or tuple(config.group_seeds) != expected or config.device != first_cfg["device"]):
        raise ValueError("Group test cannot change frozen horizon, scope, seed group or execution backend")
    project = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    adapter, cfg, scenes = load_runtime(project)
    sources, assets = _inputs(project, adapter, cfg)
    store = RunStore.create(output, dict(kind="rgb_group_test", group_seeds=list(expected),
                            roots=[str(root) for root in roots], test_manifest=manifest["test"],
                            methods=["selected", "final", "identity", "random_initial"], plants=["P0", "P1"],
                            missing_baselines={"shadow_AE": "Not trained in this version; no same-data shadow-AE claim"},
                            initial_baseline="Update zero includes random features and original uncalibrated scale; training gains include later coordinate calibration",
                            max_steps=config.test_max_steps, scope=config.scope,
                            original_deadline_unix=original_deadline_unix),
                            max_hours=config.max_hours, sources=sources, assets=assets,
                            minimum_free_gib=config.minimum_free_gib, deadline_unix=original_deadline_unix)
    check = lambda: _guard(store, guard)
    with store:
        try:
            geometry = RGBGeometry(cfg, config.device)
            scenes_by_id = {scene["id"]: scene for scene in scenes}
            all_rows = []
            for root, protocol in zip(roots, protocols):
                frozen = json.loads((root / "FROZEN.json").read_text(encoding="utf-8"))
                selected = _read_verified_checkpoint(root, frozen["checkpoint"], file_sha256(root / "protocol.json"))
                initial = _read_verified_checkpoint(root, "initial.pt", file_sha256(root / "protocol.json"))
                training_final = json.loads((root / "FINAL_STATE.json").read_text(encoding="utf-8"))
                if training_final["status"] != "complete" or training_final["stage"] != "training_frozen":
                    raise ValueError("Group testing requires completed and frozen training runs")
                final = _read_verified_checkpoint(root, training_final["final_checkpoint"], file_sha256(root / "protocol.json"))
                cfg_values = {name: value for name, value in protocol["config"].items() if name in fields}
                learner = _new_learner(cfg, RGBRunConfig.from_dict(cfg_values))
                for plant in ("P0", "P1"):
                    for method in ("selected", "final", "identity", "random_initial"):
                        chosen = initial if method == "random_initial" else final if method == "final" else selected
                        _restore_for_evaluation(learner, chosen)
                        rows = _evaluate(cfg, scenes_by_id, manifest["test"], learner, geometry, check,
                                         plant=plant, method=method, max_steps=config.test_max_steps,
                                         output=store.root / f"seed_{protocol['config']['seed']}" / plant / method, store=store)
                        for row in rows:
                            row["seed"] = protocol["config"]["seed"]
                        all_rows.extend(rows)
                del learner
            grouped = {}
            paired = {}
            for plant in ("P0", "P1"):
                for method in ("selected", "final", "identity", "random_initial"):
                    grouped[f"{plant}/{method}"] = _summary([row for row in all_rows if row["plant"] == plant and row["method"] == method])
                selected_rows = {(row["seed"], row["case_id"]): row for row in all_rows
                                 if row["plant"] == plant and row["method"] == "selected"}
                for method in ("final", "identity", "random_initial"):
                    baseline = {(row["seed"], row["case_id"]): row for row in all_rows
                                if row["plant"] == plant and row["method"] == method}
                    common = [key for key in selected_rows if selected_rows[key]["success"] and baseline[key]["success"]]
                    paired[f"{plant}/selected_vs_{method}"] = dict(
                        common_successes=len(common),
                        selected_mean_time_s=float(np.mean([selected_rows[key]["elapsed_s"] for key in common])) if common else None,
                        baseline_mean_time_s=float(np.mean([baseline[key]["elapsed_s"] for key in common])) if common else None,
                        total_shared_cases=len(selected_rows))
            result = dict(status="complete", scope=config.scope, group_seeds=list(expected), summary=grouped,
                          rows=all_rows, paired_common_success_time=paired,
                          missing_baselines={"shadow_AE": "Not implemented/trained in this version"},
                          input_cost_note="Normalized command-square integral, not measured physical energy in joules",
                          initial_baseline_note="Random initial is before AE warmup and scale calibration; selected-encoder identity is the matched M ablation",
                          independent_start_note="Starts are shared across methods and seeds; repetitions are not new independent starts",
                          guarantee="Empirical reconstructed-room visual outer-loop evaluation only")
            store.verify_inputs()
            atomic_json(store.root / "RESULTS.json", result)
            atomic_json(store.root / "FINAL_STATE.json", {key: value for key, value in result.items() if key != "rows"})
            store.event("group_test_complete", summary=grouped, scope=config.scope)
            return result
        except BaseException as error:
            final = dict(status="stopped" if isinstance(error, RunStopped) else "failed", error=repr(error),
                         traceback=traceback.format_exc(), restart_allowed=False)
            atomic_json(store.root / "FINAL_STATE.json", final)
            store.event("group_test_stopped", **final)
            raise


__all__ = ["RGBRunConfig", "RawRGBReplay", "train", "test_group"]
