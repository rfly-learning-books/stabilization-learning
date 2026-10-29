"""Fresh RGB execution pipeline; v2 science and immutable v2 runs are preserved."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import json
import traceback
import numpy as np
import torch
from extensions.joint_certificate_v2.rgb import (
    RGBRunConfig as BaseRGBRunConfig, _inputs as reference_inputs,
    _guard, _manifest, _summary, _probe,
)
from stabilization_gym.tasks_v3.reconstructed_rgb import (
    load_runtime, BatchedRGBGeometry as RGBGeometry, random_raw_autoencoder,
)
from stabilization_gym.tasks_v3.visual import AcceleratedVisualTaskAdapter
from stabilization_learning.joint_v3.features import AcceleratedRGBFeatures
from stabilization_learning.joint_v3.visual import AcceleratedVisualLearner, GoalConditionedMetric
from stabilization_learning.joint_v2.runtime import RunStore, RunStopped, atomic_json, file_sha256, seed_all
from .replay import FastRawRGBReplay as RawRGBReplay
from .rollout import collect_batch, evaluate_batch

@dataclass(frozen=True)
class RGBRunConfig(BaseRGBRunConfig):
    implementation: str = 'joint_rgb_v3'
    interaction_backend: str = 'forward'
    inference_batch_size: int = 4
    depth_batch_size: int = 4
    replay_cache_episodes: int = 32
    geometry_cache_samples: int = 128
    replay_cache_bytes: int = 512 << 20
    geometry_cache_bytes: int = 128 << 20

    def __post_init__(self):
        super().__post_init__()
        if self.implementation != 'joint_rgb_v3':
            raise ValueError('Explicit joint_rgb_v3 implementation required')
        if self.interaction_backend not in {'forward', 'legacy', 'vmap'}:
            raise ValueError('Unknown interaction backend')
        for name in ('inference_batch_size', 'depth_batch_size', 'replay_cache_bytes', 'geometry_cache_bytes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + ' must be a positive integer')

def _inputs(project, adapter, cfg):
    sources, assets = reference_inputs(project, adapter, cfg)
    sources = set(sources)
    sources.add(project / 'extensions/joint_certificate_v2/rgb.py')
    for folder in ('extensions/joint_certificate_v3', 'stabilization-gym/stabilization_gym/tasks_v3',
                   'stabilization-learning-control/stabilization_learning/joint_v3'):
        sources.update((project / folder).glob('*.py'))
    sources.update((project / 'stabilization-gym/stabilization_gym/interfaces').glob('*.py'))
    sources.add(project / 'stabilization-gym/stabilization_gym/tasks_v2/contracts.py')
    sources.add(project / 'scripts/run_rgb_certificate_v3.py')
    return sorted(sources), assets

def _new_learner(cfg, config):
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Declared CUDA device is unavailable')
    features = AcceleratedRGBFeatures(random_raw_autoencoder(cfg, config.device), 16,
                                      backend=config.interaction_backend).to(config.device)
    metric = GoalConditionedMetric(16, 32).to(config.device)
    return AcceleratedVisualLearner(features, metric, AcceleratedVisualTaskAdapter(), config.learner)

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
                   execution_spec=learner.execution_spec, replay_schema="raw_rgb_archive_v3_cache",
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
                           data_pipeline="bounded_verified_replay_batched_public_geometry",
                           batch_rng="main thread episode seeds; per-episode action RNG, no background sampling",
                           objective="joint_visual_v2_unchanged",
                           execution_environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                               dtype="float32", threads=torch.get_num_threads(),
                               tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                               tf32_cudnn=torch.backends.cudnn.allow_tf32),
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
            geometry = RGBGeometry(cfg, config.device, depth_batch_size=config.depth_batch_size)
            learner = _new_learner(cfg, config)
            rng = np.random.default_rng(config.seed + 17001)
            replay = RawRGBReplay(store.root / "replay", rng, max_episodes=config.replay_max_episodes,
                                  cache_episodes=config.replay_cache_episodes, geometry_cache=config.geometry_cache_samples,
                                  replay_cache_bytes=config.replay_cache_bytes, geometry_cache_bytes=config.geometry_cache_bytes)
            training_scenes = [scene for scene in scenes if scene["split"] == "train"]
            if config.train_scene_limit:
                training_scenes = training_scenes[:config.train_scene_limit]
            if not training_scenes:
                raise RuntimeError("No training scenes declared")
            store.event("initialized", seed=config.seed, scope=config.scope, depth=geometry.provenance,
                        train_scenes=[scene["id"] for scene in training_scenes], random_backbone=True, random_metric=True)
            episode_id = 0
            bootstrap_scenes = [scene for scene in training_scenes for _ in range(config.bootstrap_episodes_per_scene)]
            for start in range(0, len(bootstrap_scenes), config.inference_batch_size):
                chunk = bootstrap_scenes[start:start + config.inference_batch_size]
                collect_batch(cfg, chunk, learner, geometry, replay, rng, episode_id,
                              config.bootstrap_steps, check, exploratory=True, store=store)
                episode_id += len(chunk)
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
                    for start in range(0, config.collection_episodes, config.inference_batch_size):
                        count = min(config.inference_batch_size, config.collection_episodes-start)
                        chosen = [training_scenes[int(rng.integers(len(training_scenes)))] for _ in range(count)]
                        collect_batch(cfg, chosen, learner, geometry, replay, rng, episode_id,
                                      config.collection_steps, check, exploratory=False, store=store)
                        episode_id += count
                if pair % config.validation_every_pairs == 0 or pair == config.joint_pairs:
                    name = f"update_{sum(learner.counts.values()):06d}.pt"
                    _checkpoint(learner, replay, rng, store, name=name, next_pair=pair + 1, collector_episode=episode_id)
                    rows = evaluate_batch(cfg, scenes_by_id, manifest["val"], learner, geometry, check,
                                     max_steps=config.validation_max_steps, batch_size=config.inference_batch_size,
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
    if payload["runtime_state"].get("execution_spec") != learner.execution_spec:
        raise ValueError("Checkpoint execution backend/feature provider mismatch")
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
    if any(p["config"].get("implementation") != "joint_rgb_v3" for p in protocols):
        raise ValueError("v3 group testing requires v3 frozen protocols")
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
    for name, value in asdict(config).items():
        if name not in {"seed", "max_hours", "minimum_free_gib"} and value != first_cfg[name]:
            # JSON represents group tuples as lists.
            if name != "group_seeds" or list(value) != first_cfg[name]:
                raise ValueError("Group test cannot change frozen execution/science configuration: " + name)
    project = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    adapter, cfg, scenes = load_runtime(project)
    sources, assets = _inputs(project, adapter, cfg)
    store = RunStore.create(output, dict(kind="rgb_group_test", implementation="joint_rgb_v3", interaction_backend=config.interaction_backend, group_seeds=list(expected),
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
            geometry = RGBGeometry(cfg, config.device, depth_batch_size=config.depth_batch_size)
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
                        rows = evaluate_batch(cfg, scenes_by_id, manifest["test"], learner, geometry, check,
                                         plant=plant, method=method, max_steps=config.test_max_steps, batch_size=config.inference_batch_size,
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
