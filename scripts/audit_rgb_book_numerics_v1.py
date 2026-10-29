"""Compare restored book algorithms and replay without launching training.

Runs one warmup, Critic, representation and M update on the same batch and
random states through the original class and the public algorithm factory.
All tensor checks are exact. Output is a new audit directory, not a run.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import MethodType, SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(a, b, path="root"):
    """Return exact differences and scalar/tensor coverage, including Adam."""
    import numpy as np
    import torch
    result = dict(tensors=0, arrays=0, scalars=0, mismatches=[])

    def visit(left, right, key):
        if torch.is_tensor(left):
            result["tensors"] += 1
            valid = (torch.is_tensor(right) and left.shape == right.shape and
                     left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu()))
            if not valid:
                detail = dict(path=key, kind="tensor")
                if torch.is_tensor(right) and left.shape == right.shape:
                    detail["maximum_absolute_difference"] = float(
                        (left.detach().cpu().double()-right.detach().cpu().double()).abs().max()) if left.numel() else 0.
                result["mismatches"].append(detail)
        elif isinstance(left, np.ndarray):
            result["arrays"] += 1
            if not (isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)):
                result["mismatches"].append(dict(path=key, kind="array"))
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                result["mismatches"].append(dict(path=key, kind="mapping_keys"))
            else:
                for name in left:
                    visit(left[name], right[name], key+"/"+str(name))
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                result["mismatches"].append(dict(path=key, kind="sequence"))
            else:
                for index, (lvalue, rvalue) in enumerate(zip(left, right)):
                    visit(lvalue, rvalue, key+"/"+str(index))
        else:
            result["scalars"] += 1
            if left != right:
                result["mismatches"].append(dict(path=key, kind="scalar", left=str(left), right=str(right)))
    visit(a, b, path)
    result["passed"] = not result["mismatches"]
    return result


def cached_replay_class(parent, cache_limit):
    """Execute the actual backend's nested cache class without duplicating data.

    That class has no dependency on a backend instance; its only closed-over
    names are parent, cache_limit and numpy. It is extracted from the current
    backend source, whose hash is recorded in the audit.
    """
    import numpy as np
    path = PROJECT / "extensions/rgb_book_reproduction_v1/backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "CachedReplay"]
    if len(matches) != 1:
        raise RuntimeError("Expected the single production CachedReplay definition")
    namespace = dict(parent=parent, cache_limit=cache_limit, np=np)
    module = ast.Module(body=[matches[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["CachedReplay"]


def audit(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    runtime_temp = PROJECT / ".cache/rgb_book_runtime_tmp"
    runtime_temp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(runtime_temp)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for folder in (PROJECT, PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control"):
        sys.path.insert(0, str(folder))
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from stabilization_gym.envs.tasks.original_visual_online import OriginalVisualOnlineAdapter
    from stabilization_learning.rgb_book import make_learner
    from extensions.unified_training_v1.backends.rgb import Backend
    from extensions.rgb_book_reproduction_v1.collection import book_collect

    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This audit requires the declared CUDA training device")
    backend_cfg = {}
    if args.config:
        backend_cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))["jobs"][0]["config"]
    reconstruction = PROJECT / "migration/visual_scene_reconstruction"
    adapter = OriginalVisualOnlineAdapter(PROJECT, reconstruction / "reconstructed_legacy_root",
                                         reconstruction / "SCENE_AUDIT.json")
    data = Path(backend_cfg.get("bootstrap_root", PROJECT / "assets/reconstructed_visual_bootstrap_20261023_v1"))
    cfg = adapter.training_config(data)
    cfg["goal_fixed"]["batch_size"] = backend_cfg.get("batch_size", args.batch_size)
    cfg["collection"]["steps"] = backend_cfg.get("collection_steps", 120)
    cfg["collection"]["episodes_per_scene_per_round"] = backend_cfg.get("trajectories_per_scene", 2)
    cfg["collection"]["replay_transitions"] = backend_cfg.get("replay_capacity", 20000)
    adapter.config_module.seed_all(41)
    torch.set_num_threads(backend_cfg.get("cpu_threads", 2))
    source = adapter.checkpoint("initial.pt")
    if any(source["learner"]["counts"].values()):
        raise ValueError("Audit must start before task optimization")
    started = time.time()
    checks = {}
    result = dict(schema="rgb_book_numerical_equivalence_v1", device=str(args.device),
                  batch_size=cfg["goal_fixed"]["batch_size"], scope="one update per stage; no formal training or evaluation",
                  tolerance="exact tensor/array/scalar equality", checks=checks,
                  source_hashes={str(path): sha(path) for path in (
                      Path(__file__), adapter.bundle / "initial.pt", data / "DATA_MANIFEST.json",
                      PROJECT / "extensions/rgb_book_reproduction_v1/backend.py",
                      PROJECT / "extensions/rgb_book_reproduction_v1/collection.py",
                      PROJECT / "stabilization-learning-control/stabilization_learning/rgb_book/learner.py")})
    try:
        # Mirror the migration RestoredLearner explicitly, then call the new
        # library factory. Each gets a separate initialization object.
        original = adapter.derivative.DerivativeLearner(deepcopy(cfg), deepcopy(source), args.device)
        original.restore(deepcopy(source["learner"]))
        original.initial.load_state_dict(source["learner"]["ae"], strict=True)
        from extensions.reproducible_training_v4.deterministic_rgb import install_deterministic_blur
        for name in ("ae", "shadow", "initial"):
            install_deterministic_blur(getattr(original, name))
        wrapped = make_learner(deepcopy(cfg), deepcopy(source), args.device)
        adapter.config_module.restore_rng(source["rng"])
        checks["initial_payload"] = compare(original.payload(), wrapped.payload())
        checks["initial_baseline"] = compare(original.initial.state_dict(), wrapped.initial.state_dict())
        parent = adapter.core.Replay
        original_replay = parent(data, cfg, 41)
        cached_replay = cached_replay_class(parent, backend_cfg.get("replay_cached_episodes", 16))(data, cfg, 41)
        for replay in (original_replay, cached_replay):
            replay.rng.bit_generator.state = deepcopy(source["replay_rng"])
        checks["replay_rows"] = compare(original_replay.rows, cached_replay.rows)
        for index in range(args.replay_samples):
            batch_a = original_replay.sample(cfg["goal_fixed"]["batch_size"])
            batch_b = cached_replay.sample(cfg["goal_fixed"]["batch_size"])
            checks[f"replay_batch_{index}"] = compare(batch_a, batch_b)
            checks[f"replay_rng_{index}"] = compare(original_replay.rng.bit_generator.state,
                                                     cached_replay.rng.bit_generator.state)

        # Exercise the actual import-stage probe. It must leave model,
        # optimizer, replay and global random states untouched.
        probe = SimpleNamespace(initialized=True, collection_id=0, replay=cached_replay,
            guard=lambda: None, event=lambda *a, **k: None, output=output,
            config=backend_cfg, learner=wrapped, adapter=adapter, fixed_probe=None)
        probe._optimizer_counts = MethodType(Backend._optimizer_counts, probe)
        probe._record_probe = MethodType(Backend._record_probe, probe)
        before_rng = deepcopy(adapter.config_module.rng_state())
        before_replay_rng = deepcopy(cached_replay.rng.bit_generator.state)
        book_collect(probe, initial=True)
        checks["initial_probe_payload_unchanged"] = compare(original.payload(), wrapped.payload())
        checks["initial_probe_global_rng_unchanged"] = compare(before_rng, adapter.config_module.rng_state())
        checks["initial_probe_replay_rng_unchanged"] = compare(before_replay_rng, cached_replay.rng.bit_generator.state)
        checks["initial_probe_caches_unchanged"] = compare(
            dict(cache=original.cache, tcache=original.tcache, inputs=original.inputs),
            dict(cache=wrapped.cache, tcache=wrapped.tcache, inputs=wrapped.inputs))

        def update(learner, phase, batch):
            if phase == "warmup":
                return learner.representation(batch, warmup=True)
            if phase == "critic_prefit":
                return learner.critic_prefit(batch)
            if phase == "representation":
                return learner.representation(batch)
            return learner.metric_update(batch)

        for phase in ("warmup", "critic_prefit", "representation", "metric"):
            state = deepcopy(adapter.config_module.rng_state())
            first = update(original, phase, batch_a)
            after = deepcopy(adapter.config_module.rng_state())
            adapter.config_module.restore_rng(state)
            second = update(wrapped, phase, batch_b)
            torch.cuda.synchronize()
            checks[phase+"_payload_and_optimizers"] = compare(original.payload(), wrapped.payload())
            checks[phase+"_logs"] = compare(first, second)
            checks[phase+"_rng"] = compare(after, adapter.config_module.rng_state())
            print(json.dumps(dict(phase=phase, passed=all(checks[k]["passed"] for k in
                (phase+"_payload_and_optimizers", phase+"_logs", phase+"_rng"))), ensure_ascii=False), flush=True)
        result["final_counts"] = wrapped.counts
        result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["passed"] = all(check["passed"] for check in checks.values())
    except BaseException as error:
        result.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["wall_seconds"] = time.time()-started
        (output / "NUMERICAL_EQUIVALENCE.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], report=str(output / "NUMERICAL_EQUIVALENCE.json"))))
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--replay-samples", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.replay_samples < 1:
        parser.error("batch size and replay samples must be positive")
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
