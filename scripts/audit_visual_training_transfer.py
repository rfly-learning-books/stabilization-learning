"""CPU-only offline checks of the retained original visual training pipeline.

This is a numerical migration audit, not online collection or a full retrain.
Original checkpoints, replay shards and source bundles are read-only.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import collections
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch


def emit(**row):
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def module_hashes(learner):
    return {name: tensor_hash(getattr(learner, name).state_dict())
            for name in ("ae", "shadow", "metric", "critics", "initial")}


def trees_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(trees_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(trees_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def all_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, float) else True


def batch_hash(batch):
    h = hashlib.sha256()
    for key, value in sorted(batch.items()):
        h.update(key.encode())
        if isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(value.tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def readonly_replay(adapter, payload):
    """Use the unchanged original sampler without invoking its writer init."""
    replay = object.__new__(adapter.core.Replay)
    replay.root = adapter.bundle / "replay"
    replay.cfg = adapter.cfg
    replay.rng = np.random.default_rng()
    replay.rng.bit_generator.state = copy.deepcopy(payload["replay_rng"])
    replay.path = None
    replay.rows = copy.deepcopy(payload["replay_rows"])
    replay.cache = collections.OrderedDict()
    replay.max = adapter.cfg["collection"]["replay_transitions"]
    if not all(row["split"] == "train" for row in replay.rows):
        raise AssertionError("Held-out data found in initialization replay")
    return replay


def atomic_json(path, value):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / "runs/visual_original_offline_migration_20261022_v1"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "TRAINING_AUDIT.json"
    if report_path.exists():
        raise FileExistsError("Training audit exists; retain it and use a new output for another audit")
    sys.path.insert(0, str(root / "stabilization-gym"))
    from stabilization_gym.envs.tasks.original_visual import OriginalVisualAdapter

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    began = time.monotonic()
    report = dict(
        schema=1, scope="CPU-only offline update/gradient/recovery audit; not online training or a full retrain",
        started=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        device="cpu", batch_size=4, torch_version=torch.__version__, numpy_version=np.__version__,
        cpu_threads=torch.get_num_threads(), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        script_sha256=sha_file(__file__),
        adapter_sha256_at_start=sha_file(root / "stabilization-gym/stabilization_gym/envs/tasks/original_visual.py"),
        source_manifest_sha256=sha_file(root / "migration/visual_initialization_bundle.json"),
        protocol_sha256=sha_file(out / "protocol.json"),
        checks={}, updates=[], completed=False,
    )
    atomic_json(report_path, report)
    try:
        assert not torch.cuda.is_available(), "CPU audit must not access GPU"
        protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
        report["checks"]["migration_protocol_fingerprints_match"] = all(
            sha_file(root / relative) == expected for relative, expected in protocol["fingerprints"].items())
        assert report["checks"]["migration_protocol_fingerprints_match"], "Migration protocol fingerprint mismatch"
        adapter = OriginalVisualAdapter(root)
        learner, payload = adapter.load_training_initial("cpu")
        replay = readonly_replay(adapter, payload)
        assert replay.transitions == 2822 and len(replay.rows) == 48
        assert int(adapter.cfg["goal_fixed"]["batch_size"]) == 4
        report["configuration"] = copy.deepcopy(adapter.cfg)
        report["initial_checkpoint_sha256"] = sha_file(adapter.bundle / "initial.pt")
        report["initial_module_hashes"] = module_hashes(learner)
        report["initial_counts"] = copy.deepcopy(learner.counts)
        report["checks"]["full_initial_learner_restore_exact"] = trees_equal(learner.payload(), payload["learner"])
        report["checks"]["initial_rng_restore_exact_cpu_fields"] = all(
            trees_equal(adapter.config_module.rng_state()[k], payload["rng"][k])
            for k in ("python", "numpy", "torch"))
        report["checks"]["bootstrap_2822_train_only"] = True
        assert all(report["checks"].values())
        del payload
        emit(event="loaded_initial", replay_transitions=replay.transitions, device="cpu")

        phases = [
            ("warmup", lambda b: learner.representation(b, warmup=True), {"ae", "shadow"}),
            ("critic_prefit", learner.critic_prefit, {"critics"}),
            ("representation", learner.representation, {"ae", "shadow", "critics"}),
            ("metric_update", learner.metric_update, {"metric"}),
        ]
        for name, update, expected_changes in phases:
            before = module_hashes(learner)
            batch = replay.sample(4)
            start = time.monotonic()
            emit(event="update_start", phase=name, batch_hash=batch_hash(batch))
            logs = update(batch)
            after = module_hashes(learner)
            changed = {key for key in before if before[key] != after[key]}
            result = dict(phase=name, batch_hash=batch_hash(batch), duration_s=time.monotonic() - start,
                          before_hashes=before, after_hashes=after, changed_modules=sorted(changed),
                          expected_changed_modules=sorted(expected_changes), logs=logs,
                          counts=copy.deepcopy(learner.counts))
            report["updates"].append(result)
            report["checks"][name + "_finite"] = all_finite(logs) and all_finite(learner.payload())
            report["checks"][name + "_parameter_isolation"] = changed == expected_changes
            if name == "representation":
                base = logs["ae_encoder_grad"]
                applied_critic = logs["critic_encoder_grad"] * logs["critic_encoder_scale"]
                applied_regular = logs["regular_encoder_grad"] * logs["regular_encoder_scale"]
                report["checks"]["critic_supervision_reaches_encoder"] = applied_critic > 0
                report["checks"]["critic_encoder_gradient_cap_25_percent"] = applied_critic <= .25 * base + 1e-10
                report["checks"]["regular_encoder_gradient_cap_5_percent"] = applied_regular <= .05 * base + 1e-10
                result["applied_gradient_norms"] = dict(ae=base, critic=applied_critic, regular=applied_regular)
            if name == "metric_update":
                report["checks"]["metric_update_no_ae_or_critic_gradient"] = all(
                    parameter.grad is None for module in (learner.ae, learner.critics)
                    for parameter in module.parameters())
                report["checks"]["metric_gradient_nonzero"] = logs["gradient_norm"] > 0
            atomic_json(report_path, report)
            emit(event="update_complete", phase=name, duration_s=result["duration_s"], changed_modules=sorted(changed))
            assert all(report["checks"].values()), "An update or gradient-isolation check failed"

        optimizer_entries = {name: len(getattr(learner, name).state_dict()["state"])
                             for name in ("aeopt", "shadowopt", "criticopt", "mopt")}
        assert all(value > 0 for value in optimizer_entries.values())
        checkpoint_dir = out / "training_checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / "after_four_updates.pt"
        checkpoint = dict(schema=1, role="offline_migration_audit_only", learner=learner.payload(),
                          rng=adapter.config_module.rng_state(), replay_rows=copy.deepcopy(replay.rows),
                          replay_rng=copy.deepcopy(replay.rng.bit_generator.state),
                          initial_model_hash=module_hashes(learner)["initial"],
                          config=copy.deepcopy(adapter.cfg))
        torch.save(checkpoint, checkpoint_path)
        del checkpoint
        report["checkpoint"] = dict(path=str(checkpoint_path), sha256=sha_file(checkpoint_path),
                                    optimizer_state_entries=optimizer_entries)
        report["checks"]["all_four_optimizers_nonempty_before_restore"] = True

        # The next representation step exercises random action generation plus
        # both nonempty AE optimizers and the nonempty derivative-Critic optimizer.
        next_batch = replay.sample(4)
        emit(event="resume_branch_a_start", phase="representation")
        first_logs = learner.representation(next_batch)
        first_payload = copy.deepcopy(learner.payload())
        first_rng = copy.deepcopy(adapter.config_module.rng_state())
        first_replay_rng = copy.deepcopy(replay.rng.bit_generator.state)
        first_batch_hash = batch_hash(next_batch)
        del next_batch

        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        learner.restore(saved["learner"])
        learner.zero_all()
        adapter.config_module.restore_rng(saved["rng"])
        replay.rows = copy.deepcopy(saved["replay_rows"])
        replay.rng.bit_generator.state = copy.deepcopy(saved["replay_rng"])
        replay.cache.clear()
        # Both these caches are ephemeral; rebuild them after restored weights.
        learner.inputs.clear()
        learner.clear_cache()
        report["checks"]["nonempty_optimizer_and_model_restore_exact"] = trees_equal(learner.payload(), saved["learner"])
        report["checks"]["restored_global_rng_exact"] = trees_equal(adapter.config_module.rng_state(), saved["rng"])
        report["checks"]["restored_replay_rng_exact"] = trees_equal(replay.rng.bit_generator.state, saved["replay_rng"])
        report["checks"]["initial_baseline_unchanged"] = module_hashes(learner)["initial"] == saved["initial_model_hash"]
        del saved
        next_batch = replay.sample(4)
        report["checks"]["restored_next_sample_exact"] = batch_hash(next_batch) == first_batch_hash
        emit(event="resume_branch_b_start", phase="representation")
        second_logs = learner.representation(next_batch)
        report["checks"]["restored_next_representation_logs_exact"] = trees_equal(first_logs, second_logs)
        report["checks"]["restored_next_representation_models_optimizers_counts_exact"] = trees_equal(first_payload, learner.payload())
        report["checks"]["restored_next_global_rng_exact"] = trees_equal(first_rng, adapter.config_module.rng_state())
        report["checks"]["restored_next_replay_rng_exact"] = trees_equal(first_replay_rng, replay.rng.bit_generator.state)
        report["recovery"] = dict(next_batch_sha256=first_batch_hash, first_logs=first_logs,
                                  second_logs=second_logs, final_counts=copy.deepcopy(learner.counts),
                                  final_module_hashes=module_hashes(learner))
        report["checks"]["bundle_index_not_created"] = not (adapter.bundle / "replay/index.json").exists()
        report["adapter_sha256_at_end"] = sha_file(root / "stabilization-gym/stabilization_gym/envs/tasks/original_visual.py")
        report["checks"]["adapter_unchanged_during_audit"] = report["adapter_sha256_at_end"] == report["adapter_sha256_at_start"]
        report["checks"]["protocol_unchanged_during_audit"] = sha_file(out / "protocol.json") == report["protocol_sha256"]
        report["completed"] = True
        report["passed"] = all(report["checks"].values())
        report["elapsed_s"] = time.monotonic() - began
        atomic_json(report_path, report)
        emit(event="training_audit_complete", passed=report["passed"], checks=len(report["checks"]), elapsed_s=report["elapsed_s"])
        if not report["passed"]:
            raise AssertionError("Offline migration audit did not pass every check")
    except Exception as error:
        report["passed"] = False
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        report["elapsed_s"] = time.monotonic() - began
        atomic_json(report_path, report)
        emit(event="training_audit_error", error=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
