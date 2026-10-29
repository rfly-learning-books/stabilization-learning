"""One fresh seed, one frozen protocol/deadline, full archive and held-out test."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import time
import traceback
import numpy as np
import torch
from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
from stabilization_gym.tasks_gpu_rgb.geometry import GPURGBGeometry
from stabilization_learning.joint_v2.runtime import RunStore, RunStopped, atomic_json, file_sha256, seed_all
from extensions.joint_certificate_v2.rgb import _probe, _summary, _balanced_cases
from extensions.joint_certificate_v3.rgb import _inputs as previous_inputs, _new_learner, _restore_for_evaluation
from extensions.rgb_gpu_parallel.benchmark import GPUUtilization
from .config import RGBConfig
from .replay import StratifiedRGBReplay
from .rollout import DIFFICULTIES, collect, evaluate


def manifest(config, cfg, scenes, project):
    path = project/config.evaluation_starts
    original = json.loads(path.read_text(encoding="utf-8"))
    result = dict(source=str(path), source_sha256=file_sha256(path), same_original_offsets=True,
                  renderer_pixels_equivalent=False, population_counts={})
    lookup = {s["id"]: s for s in scenes}
    from lac_vs.environment import valid_start
    from lac_vs import legacy
    room = legacy.module(cfg, "room_env")
    for split, limit, expected in (("val", config.validation_case_limit, 48), ("test", config.test_case_limit, 72)):
        rows = []
        for item in original[split]:
            scene = lookup[item["scene_id"]]
            if scene["split"] != split:
                raise ValueError("Evaluation split mismatch")
            frame = room.ReferenceFrame(np.asarray(scene["refs"][item["goal_id"]]["goal_matrix"]))
            if not valid_start(cfg, scene, frame, np.asarray(item["offset"])):
                raise ValueError("Legacy start invalid under current scene assets")
            rows.append(dict(**item, case_id=f"{item['scene_id']}_g{item['goal_id']}_{item['difficulty']}_r{item['repeat']}"))
        if len(rows) != expected or len({r["case_id"] for r in rows}) != expected:
            raise ValueError("Unexpected original evaluation population")
        result["population_counts"][split] = len(rows)
        result[split] = _balanced_cases(rows, limit, config.seed+901)
    return result


def inputs(project, adapter, cfg, config):
    sources, assets = previous_inputs(project, adapter, cfg)
    sources = set(sources)
    for folder in ("extensions/joint_certificate_v4", "stabilization-gym/stabilization_gym/tasks_gpu_rgb"):
        sources.update((project/folder).glob("*.py"))
    sources.update([project/"scripts/run_rgb_certificate_v4.py", project/"extensions/rgb_gpu_parallel/benchmark.py"])
    assets = set(assets) | {project/config.evaluation_starts, project/"requirements-rgb-gpu-parallel.txt"}
    return sorted(sources), sorted(assets)


def initialize(root, config, project, config_path=None):
    adapter, cfg, scenes = load_runtime(project)
    declared = manifest(config, cfg, scenes, project)
    train_scenes = [s for s in scenes if s["split"] == "train"]
    if config.train_scene_limit:
        train_scenes = train_scenes[:config.train_scene_limit]
    n_strata = sum(len(s["refs"])*len(DIFFICULTIES) for s in train_scenes)
    sources, assets = inputs(project, adapter, cfg, config)
    if config_path:
        assets.append(config_path.resolve())
    control = dict(runtime=cfg, training_scene_ids=[s["id"] for s in train_scenes],
                   training_strata=n_strata, minimum_new_transitions=n_strata*sum(config.transitions_per_stratum),
                   evaluation_manifest=declared, methods=["selected", "final", "identity", "random_initial"],
                   plants=["P0", "P1"], seed_group=[config.seed],
                   old_learned_weights_loaded=False, old_replay_loaded=False,
                   public_assets="44 existing reconstructed rooms and pretrained public FastDepth; encoder/M start random",
                   physical_scope="P0 4DOF velocity outer loop; P1 adds delay/filter; no motor/attitude dynamics",
                   objective="unchanged joint_v2 losses, current-image interaction matrix, fixed damping, forward JVP",
                   data_path="GPU render and online geometry; append-only uint8 disk archive; bounded mmap/pinned upload on replay",
                   sampling="every stratum quota includes failures; new starts on every explicit reset; no autoreset transitions",
                   near_start="radius and abs yaw uniform 0.5..2 times the unchanged success thresholds",
                   selection="validation success fraction, then lower integrated physical error, then lower input; positive updates only",
                   freeze="single declared seed is selected before every held-out test; no test-driven changes",
                   diagnostics=dict(J_effective_norm_threshold=1e-7, successor_value="same frozen phi/M/scale at both endpoints",
                                    value="0.5 e.T M e", M="goal-conditioned SPD trace16 floor0.1"),
                   caveats=["New renderer pixels differ from old", "One seed", "No shadow AE", "Identity uses selected encoder",
                            "Random initial precedes warmup/calibration", "Inference wall time not injected in dt; no realtime claim",
                            "No new global or nonlinear-domain stability proof"],
                   execution=dict(torch=torch.__version__, cuda=torch.version.cuda, dtype="float32",
                                  threads=torch.get_num_threads(), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                                  tf32_cudnn=torch.backends.cudnn.allow_tf32), automatic_resume=False)
    store = RunStore.create(root, dict(run=asdict(config), **control), max_hours=config.max_hours,
                            sources=sources, assets=assets, minimum_free_gib=config.minimum_free_gib)
    # Retain the exact source version even when later engineering work updates
    # this optional runner. External inputs still must match during execution.
    snapshot = store.root/"SOURCE_SNAPSHOT"
    snapshot.mkdir(exist_ok=False)
    for path, digest in store.protocol["sources"].items():
        target = snapshot/(digest+"_"+Path(path).name)
        if not target.exists():
            shutil.copyfile(path, target)
        if file_sha256(target) != digest:
            raise RuntimeError("Source snapshot mismatch")
    return store


def checkpoint(learner, replay, rng, store, name, next_pair):
    path = store.root/"replay_indices"/(name.removesuffix(".pt")+".json")
    atomic_json(path, replay.snapshot())
    payload = dict(models=dict(features=learner.features.state_dict(), metric=learner.metric.state_dict()),
                   optimizers=dict(features=learner.feature_optimizer.state_dict(), metric=learner.metric_optimizer.state_dict()),
                   runtime_state=dict(counts=learner.counts.copy(), version=learner.version,
                         next_pair=next_pair, replay_index=str(path), replay_index_sha256=file_sha256(path),
                         sampler_rng=deepcopy(rng.bit_generator.state), execution_spec=learner.execution_spec,
                         learner_config=asdict(learner.config), control_config=asdict(learner.adapter.config),
                         phase="complete_update", replay_schema="full_stratified_raw_rgb_v4", automatic_resume=False))
    return store.checkpoint(name, payload, completed_updates=sum(learner.counts.values()))


def audit(store, rows, replay, initial, final, config):
    """Independent NumPy recomputation from archived full successors, no model loss reuse."""
    errors = {k: 0. for k in ("value", "physical_error", "integrated_error", "input_cost", "physical_cost")}
    transitions = 0
    for row in rows:
        store.guard()
        if file_sha256(row["trajectory"]) != row["trajectory_sha256"]:
            raise RuntimeError("Trace hash changed")
        with np.load(row["trajectory"], allow_pickle=False) as a:
            e = a["error"].astype(float); m = a["M"].astype(float); offset = a["offset"].astype(float)
            if len(e) != row["steps"]+1 or len(offset) != len(e):
                raise RuntimeError("Missing full successor")
            v = .5*np.einsum("ti,ij,tj->t", e, m, e)
            E = np.square(offset[:, :3]).sum(-1)+np.square(offset[:, 3]/np.deg2rad(25))
            integral = E[1:]*a["dt"]
            effort = np.square(a["action"].astype(float)).sum(-1)*a["dt"]
            cost = integral + store.protocol["config"]["runtime"]["cost"]["effort"]*effort
            if row["reason"] in {"collision", "outside"}:
                cost[-1] += store.protocol["config"]["runtime"]["cost"]["collision_terminal"]
            for name, x, y in (("value", v, a["V"]), ("physical_error", E, a["physical_error"]),
                               ("integrated_error", integral, a["integrated_error"]), ("input_cost", effort, a["input_cost"]),
                               ("physical_cost", cost, a["physical_cost"])):
                difference = float(np.max(np.abs(x-y)))
                errors[name] = max(errors[name], difference)
                if not np.allclose(x, y, atol=2e-5, rtol=2e-5):
                    raise RuntimeError(f"Independent {name} recomputation failed: {difference}")
            if np.linalg.eigvalsh(m).min() < .0999 or not np.isclose(np.trace(m), 16, atol=1e-4):
                raise RuntimeError("M constraint failed")
            transitions += row["steps"]
    archive_files = 0
    for row in replay.rows:
        store.guard()
        for info in row["files"].values():
            if file_sha256(info["path"]) != info["sha256"]:
                raise RuntimeError("Raw archive hash changed: " + info["path"])
            archive_files += 1
    deltas = {}
    for module in ("features", "metric"):
        a, b = initial["payload"]["models"][module], final["payload"]["models"][module]
        # Named parameters, not mutable buffers, establish actual optimization.
        names = [k for k in a if (k.endswith("weight") or k.endswith("bias"))]
        deltas[module] = float(sum((a[k].double()-b[k].double()).square().sum() for k in names).sqrt())
    if not all(n > 0 and np.isfinite(n) for n in deltas.values()):
        raise RuntimeError("No actual parameter update")
    events = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((store.root/"events").glob("*.json"))]
    frozen_sequence = next(e["sequence"] for e in events if e["kind"] == "selection_frozen")
    test_sequence = next(e["sequence"] for e in events if e["kind"] == "heldout_test_started")
    if frozen_sequence >= test_sequence:
        raise RuntimeError("Held-out test started before selection freeze")
    result = dict(passed=True, test_transitions=transitions, maximum_absolute_difference=errors,
                  raw_archive_files_verified=archive_files, parameter_l2_change=deltas,
                  selected_before_heldout_test=True, independent_numpy=True,
                  update_counts=final["payload"]["runtime_state"]["counts"],
                  scientific_claim="engineering/numerical verification, not proof of performance or stability")
    atomic_json(store.root/"AUDIT.json", result)
    store.event("audit_complete", **result)
    return result


def run(root, project):
    store = RunStore(root)
    config = RGBConfig.from_dict(store.protocol["config"]["run"])
    declared = store.protocol["config"]
    timings = []; telemetry = None; replay = None
    with store:
        try:
            seed_all(config.seed)
            _, cfg, scenes = load_runtime(project)
            if cfg != declared["runtime"]:
                raise RuntimeError("Runtime task configuration changed")
            geometry = GPURGBGeometry(cfg, config.device)
            learner = _new_learner(cfg, config)
            if learner.feature_optimizer.state or learner.metric_optimizer.state:
                raise RuntimeError("Optimizers must start empty")
            telemetry = GPUUtilization()
            rng = np.random.default_rng(config.seed+17001)
            replay = StratifiedRGBReplay(store.root/"replay", rng, mmap_episodes=config.mmap_episodes)
            training = [s for s in scenes if s["id"] in declared["training_scene_ids"]]
            lookup = {s["id"]: s for s in scenes}
            store.event("initialized", seed=config.seed, random_features=True, random_metric=True, empty_optimizers=True,
                        depth=geometry.provenance, minimum_new_transitions=declared["minimum_new_transitions"])

            def phase(name, fn, **metadata):
                start = time.perf_counter()
                result = fn()
                torch.cuda.synchronize()
                stop = time.perf_counter()
                timing = dict(phase=name, seconds=stop-start, begin=start, end=stop,
                              gpu=telemetry.summarize(start, stop), **metadata)
                timings.append(timing); store.event("phase_complete", **timing)
                return result

            phase("collection", lambda: collect(cfg, training, learner, geometry, replay, rng, config, project, store,
                                                round_id=0, quota=config.transitions_per_stratum[0]), round_id=0)
            probe_keys = replay.sample_keys(config.probe_count)
            atomic_json(store.root/"PROBE_MANIFEST.json", dict(keys=probe_keys, split="train", fixed_for_every_update=True))
            probe_batch = replay.batch(probe_keys, geometry, with_geometry=False)
            checkpoint(learner, replay, rng, store, "initial.pt", 0)
            _probe(learner, probe_batch, store, phase="random_initial")

            def warmup():
                for _ in range(config.warmup_updates):
                    store.guard()
                    batch = replay.batch(replay.sample_keys(config.batch_size), geometry, with_geometry=False)
                    metrics = learner.warmup(batch)
                    store.event("update", block="warmup", step=sum(learner.counts.values()), metrics=metrics)
                    _probe(learner, probe_batch, store, phase="warmup")
            phase("warmup", warmup, updates=config.warmup_updates)
            calibration_keys = replay.sample_keys(config.calibration_samples)
            learner.calibrate(replay.batch(calibration_keys, geometry, with_geometry=False))
            store.event("coordinate_calibrated", scale=float(learner.features.scale), keys=calibration_keys,
                        version=learner.version, samples=config.calibration_samples)
            _probe(learner, probe_batch, store, phase="post_warmup_calibration")
            best = None
            segment_begin = time.perf_counter(); segment_pairs = 0
            for pair in range(1, config.joint_pairs+1):
                store.guard()
                batch = replay.batch(replay.sample_keys(config.batch_size), geometry)
                for block, operation in (("feature", learner.update_features), ("metric", learner.update_metric)):
                    store.guard(); metrics = operation(batch)
                    store.event("update", block=block, pair=pair, step=sum(learner.counts.values()), metrics=metrics)
                    _probe(learner, probe_batch, store, phase=block)
                segment_pairs += 1
                boundary = pair in config.collection_pairs or pair in config.validation_pairs
                if boundary:
                    end = time.perf_counter()
                    item = dict(phase="joint", seconds=end-segment_begin, pairs=segment_pairs,
                                begin=segment_begin, end=end, gpu=telemetry.summarize(segment_begin, end))
                    timings.append(item); store.event("phase_complete", **item)
                    if pair in config.validation_pairs:
                        name = f"update_{sum(learner.counts.values()):06d}.pt"
                        checkpoint(learner, replay, rng, store, name, pair+1)
                        rows = phase("validation", lambda: evaluate(cfg, lookup, declared["evaluation_manifest"]["val"],
                            learner, geometry, config, project, store, method="validation", output=store.root/"validation"/f"pair_{pair:06d}"),
                            cases=len(declared["evaluation_manifest"]["val"]), pair=pair)
                        summary = _summary(rows)
                        score = (summary["success_rate"], -summary["mean_integrated_error"], -summary["mean_input_squared_integral"])
                        store.event("validation", pair=pair, checkpoint=name, summary=summary, score=score)
                        if best is None or score > best["score"]:
                            best = dict(checkpoint=name, score=score, update=sum(learner.counts.values()), summary=summary)
                    if pair in config.collection_pairs:
                        index = config.collection_pairs.index(pair)
                        phase("collection", lambda: collect(cfg, training, learner, geometry, replay, rng, config, project, store,
                                            round_id=index, quota=config.transitions_per_stratum[index]), round_id=index)
                    segment_begin = time.perf_counter(); segment_pairs = 0
            if sum(r["transitions"] for r in replay.rows) < declared["minimum_new_transitions"]:
                raise RuntimeError("Full data quota not reached")
            store.verify_inputs()
            store.freeze(best["checkpoint"], selected_update=best["update"], selection=dict(rule=declared["selection"], **best["summary"]))
            final_name = f"update_{sum(learner.counts.values()):06d}.pt"
            atomic_json(store.root/"TRAINING_COMPLETE.json", dict(counts=learner.counts, selected=best,
                        final_checkpoint=final_name, fresh_transitions=sum(r["transitions"] for r in replay.rows)))
            initial = store.load_checkpoint("initial.pt")
            final = store.load_checkpoint(final_name)
            selected = store.load_checkpoint(best["checkpoint"])
            counts = learner.counts.copy()
            store.event("heldout_test_started", seed_group=[config.seed], selected_checkpoint=best["checkpoint"],
                        cases=len(declared["evaluation_manifest"]["test"]), frozen=True)
            rows = []
            for plant in declared["plants"]:
                for method in declared["methods"]:
                    chosen = initial if method == "random_initial" else final if method == "final" else selected
                    _restore_for_evaluation(learner, chosen)
                    result = phase("test", lambda: evaluate(cfg, lookup, declared["evaluation_manifest"]["test"],
                        learner, geometry, config, project, store, plant=plant, method=method, output=store.root/"test"/plant/method),
                        plant=plant, method=method, cases=len(declared["evaluation_manifest"]["test"]))
                    rows.extend(result)
            summary = {f"{p}/{m}": _summary([r for r in rows if r["plant"] == p and r["method"] == m])
                       for p in declared["plants"] for m in declared["methods"]}
            atomic_json(store.root/"RESULTS.json", dict(summary=summary, rows=rows))
            phase("audit", lambda: audit(store, rows, replay, initial, final, config))
            atomic_json(store.root/"TIMINGS.json", timings)
            atomic_json(store.root/"GPU_SAMPLES.json", telemetry.rows)
            store.verify_inputs()
            final_state = dict(status="complete", scope=config.scope, counts=counts, seed=config.seed,
                               selected_checkpoint=best["checkpoint"], final_checkpoint=final_name,
                               new_transitions=sum(r["transitions"] for r in replay.rows), test_episodes=len(rows),
                               summary=summary, scientific_analysis_complete=False,
                               report_needed=True, automatic_resume=False, deadline_unix=store.protocol["deadline_unix"])
            atomic_json(store.root/"FINAL_STATE.json", final_state)
            store.event("run_complete", **final_state)
            return final_state
        except BaseException as error:
            state = dict(status="stopped" if isinstance(error, RunStopped) else "failed", error=repr(error),
                         traceback=traceback.format_exc(), automatic_resume=False)
            atomic_json(store.root/"FINAL_STATE.json", state)
            atomic_json(store.root/"TIMINGS_PARTIAL.json", timings)
            if telemetry is not None:
                atomic_json(store.root/"GPU_SAMPLES_PARTIAL.json", telemetry.rows)
            store.event("run_stopped", **state)
            raise
        finally:
            if telemetry is not None:
                telemetry.close()
            if replay is not None:
                replay.close()
