"""Read-only gradient audit and bounded, explicitly warm-started RGB repairs.

This diagnostic extension never writes to the v4 run.  It reuses only that run's
training archive.  Every arm starts from the same selected weights, with empty
Adam states and the same immutable batch-key plan.  No test cases are selected
or evaluated here.  A split schedule freezes theta during the phi block, but
retains phi(goal) -> M input gradients: it is NOT a constant-matrix ablation.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import PROJECT  # Establish the same explicit local package imports.
from extensions.joint_certificate_v4.replay import StratifiedRGBReplay
from stabilization_learning.joint_v2.runtime import atomic_json, file_sha256
from stabilization_learning.joint_v2.visual import frozen_parameters
from stabilization_learning.joint_v3.visual import AcceleratedVisualLearner
from stabilization_gym.tasks_v2.visual import quadratic_value


LOSS_NAMES = ("physical", "calibration", "interaction", "decrease", "reconstruction", "geometry")
WEIGHT_NAMES = ("physical_weight", "calibration_weight", "interaction_weight", "decrease_weight",
                "reconstruction_weight", "geometry_weight")
ARMS = {
    "original_alternating": ("original", "alternating"),
    "cal_precision_alternating": ("cal_precision", "alternating"),
    "original_split": ("original", "split"),
    "cal_precision_split": ("cal_precision", "split"),
}


def write_new_json(path, value):
    path = Path(path)
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    atomic_json(path, value)


def precision_floor_from_task(cfg):
    """Fixed from the original tolerances, never selected using validation."""
    collection, cost = cfg["collection"], cfg["cost"]
    values = (float(collection["position_tolerance_m"]) ** 2,
              (float(collection["yaw_tolerance_deg"]) / float(cost["yaw_scale_deg"])) ** 2)
    result = min(values)
    if not 0 < result < .01:
        raise ValueError("Unexpected original task precision")
    return result


class CalibrationRepairLearner(AcceleratedVisualLearner):
    """Exactly one objective change: calibration denominator max(E, E_tol)."""

    def __init__(self, base, *, calibration_mode="original", precision_floor=.0004):
        if calibration_mode not in {"original", "cal_precision"}:
            raise ValueError(calibration_mode)
        super().__init__(base.features, base.metric, base.adapter, base.config)
        self.calibration_mode = calibration_mode
        self.precision_floor = float(precision_floor)
        self.version = base.version
        self.counts = dict(warmup=0, feature=0, metric=0)
        # Constructing this wrapper deliberately creates empty optimizer states.
        if self.feature_optimizer.state or self.metric_optimizer.state:
            raise RuntimeError("Short-repair optimizer state must start empty")

    def losses(self, batch, *, feature_graph=True):
        if self.calibration_mode == "original":
            return super().losses(batch, feature_graph=feature_graph)
        batch.validate()
        if not bool(self.features.calibrated):
            raise RuntimeError("Warm-start repair requires the selected calibrated coordinates")
        cfg = self.config
        out = self.policy(batch.observation, create_graph=feature_graph)
        physical = self.adapter.physical_error(batch.offset.detach())
        physical_next = self.adapter.physical_error(batch.next_offset.detach())
        dt = batch.dt.detach()
        predicted_error = self.adapter.physical_error(
            self.adapter.nominal_step(batch.offset.detach(), out.action, dt))
        previous = (torch.zeros_like(out.action) if batch.previous_action is None
                    else batch.previous_action.detach())
        task_loss = (predicted_error / (physical + cfg.physical_floor)
                     + cfg.input_weight * out.action.square().sum(-1) * dt
                     + cfg.action_change_weight * (out.action - previous).square().sum(-1)).mean()
        next_error = self.features.encode(batch.next_image.detach()) - out.goal
        next_value = quadratic_value(next_error, out.metric)

        def cal(value, truth):
            denominator = truth.clamp_min(self.precision_floor)
            return ((F.relu(cfg.value_lower * truth - value) / denominator).square()
                    + (F.relu(value - cfg.value_upper * truth) / denominator).square()).mean()

        calibration = .5 * (cal(out.value, physical) + cal(next_value, physical_next))
        observed_feature_change = next_error - out.error
        predicted_feature_change = (out.interaction @ batch.applied_action.detach().unsqueeze(-1)).squeeze(-1) * dt[:, None]
        movement_scale = batch.applied_action.detach().norm(dim=-1) * dt + cfg.motion_floor
        interaction = ((observed_feature_change - predicted_feature_change) / movement_scale[:, None]).square().mean()
        difference = next_value - out.value
        residual = difference + cfg.decrease_per_second * dt * physical
        mask = batch.controller_transition
        decrease = ((F.relu(residual[mask]) / (physical[mask] + cfg.physical_floor)).square().mean()
                    if mask.any() else out.value.sum() * 0.)
        images = torch.cat((batch.observation.image, batch.observation.goal_image))
        reconstruction = F.mse_loss(self.features.reconstruct(images), images.detach())
        spread = out.error.square().mean(0).add(1e-12).sqrt()
        geometry = (F.relu(cfg.feature_std_lower - spread).square()
                    + F.relu(spread - cfg.feature_std_upper).square()).mean()
        total = (cfg.physical_weight * task_loss + cfg.calibration_weight * calibration
                 + cfg.interaction_weight * interaction + cfg.decrease_weight * decrease
                 + cfg.reconstruction_weight * reconstruction + cfg.geometry_weight * geometry)
        feasible = .5 * self.metric.floor * out.error.square().sum(-1) <= cfg.value_upper * physical
        return dict(total=total, physical=task_loss, calibration=calibration, interaction=interaction,
                    decrease=decrease, reconstruction=reconstruction, geometry=geometry,
                    actual_V_difference_mean=difference.detach().mean(), actual_physical_error_mean=physical_next.mean(),
                    predicted_physical_error_mean=predicted_error.detach().mean(),
                    calibration_necessary_feasible_fraction=feasible.float().mean(),
                    controller_transition_fraction=mask.float().mean())


class ReadOnlyV4Replay(StratifiedRGBReplay):
    """Use v4's exact sampler/batch builder without its writing constructor."""

    def __init__(self, index_path, *, rng_seed=2581, mmap_episodes=64):
        self.index_path = Path(index_path).resolve()
        self.index_sha256 = file_sha256(self.index_path)
        saved = json.loads(self.index_path.read_text(encoding="utf-8-sig"))
        self.rows = saved["rows"]
        if not self.rows or any(row["split"] != "train" for row in self.rows):
            raise ValueError("The repair data must contain only declared training episodes")
        self.root = self.index_path.parent.parent / "replay"
        self.rng = np.random.default_rng(rng_seed)
        self.by_stratum = defaultdict(list)
        for episode, row in enumerate(self.rows):
            if row["episode"] != episode:
                raise ValueError("Noncontiguous replay index")
            self.by_stratum[row["stratum"]].extend((episode, i) for i in range(row["transitions"]))
            for info in row["files"].values():
                if not Path(info["path"]).resolve().is_relative_to(self.root.resolve()):
                    raise ValueError("Replay file is outside the declared original training archive")
        self.cache = OrderedDict()
        self.mmap_episodes = int(mmap_episodes)
        self.samples_drawn = 0
        self.unique_samples = set()
        self.draws_per_stratum = defaultdict(int)
        self.verified_files = {}

    def _load(self, episode):
        if episode not in self.cache:
            for info in self.rows[episode]["files"].values():
                path = info["path"]
                if path not in self.verified_files:
                    actual = file_sha256(path)
                    if actual != info["sha256"]:
                        raise RuntimeError("Original archived input changed: " + path)
                    self.verified_files[path] = actual
        return super()._load(episode)

    def add(self, *args, **kwargs):
        raise RuntimeError("The original archive is read-only")

    def source_summary(self):
        return dict(index=str(self.index_path), index_sha256=self.index_sha256,
                    training_transitions=sum(row["transitions"] for row in self.rows),
                    verified_files=self.verified_files, old_replay_reused=True, fresh_transitions=0)


def create_key_plan(replay, output, *, pairs=600, batch_size=16):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    keys = [replay.sample_keys(batch_size) for _ in range(pairs)]
    probes = replay.sample_keys(8)
    # The first fixed batch is balanced as in the unchanged original sampler.
    gradient_sets = {"balanced": keys[0]}
    for difficulty in ("near", "medium"):
        population = [(row["episode"], 0) for row in replay.rows if row["difficulty"] == difficulty]
        if len(population) < batch_size:
            raise ValueError("Not enough training states for gradient strata")
        indices = replay.rng.choice(len(population), size=batch_size, replace=False)
        gradient_sets[difficulty] = [population[int(i)] for i in indices]
    result = dict(schema="rgb_v5_shared_training_keys", pairs=pairs, batch_size=batch_size,
                  sampling="original uniform-stratum/uniform-transition sampler; identical keys for all arms",
                  replay_index=str(replay.index_path), replay_index_sha256=replay.index_sha256,
                  batch_keys=keys, probe_keys=probes, gradient_sets=gradient_sets,
                  evaluation="validation only; final update, no intermediate model selection",
                  warm_start=True, not_from_scratch=True)
    write_new_json(output, result)
    return result


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def gradient_flags(features, metric, block):
    modules = (features, metric)
    saved = [[p.requires_grad for p in module.parameters()] for module in modules]
    try:
        for module, enabled in zip(modules, (block == "feature", block == "metric")):
            for p in module.parameters():
                p.requires_grad_(enabled)
        yield
    finally:
        for module, flags in zip(modules, saved):
            for p, flag in zip(module.parameters(), flags):
                p.requires_grad_(flag)


def gradient_audit(learner, replay, geometry, plan, output, *, guard=lambda: None, event=None):
    """Weighted per-term gradients and pairwise cosine, without optimizer steps."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = {k: parameter_digest(getattr(learner, k)) for k in ("features", "metric")}
    all_rows = []
    previous_mode = getattr(learner, "calibration_mode", None)
    modes = ("original", "cal_precision") if previous_mode is not None else ("original",)
    for sample_name, keys in plan["gradient_sets"].items():
        guard()
        batch = replay.batch(keys, geometry)
        E = learner.adapter.physical_error(batch.offset.detach()).cpu().numpy()
        for mode in modes:
            if previous_mode is not None:
                learner.calibration_mode = mode
            for block in ("feature", "metric"):
                guard()
                vectors, losses, norms = {}, {}, {}
                module = learner.features if block == "feature" else learner.metric
                with gradient_flags(learner.features, learner.metric, block):
                    parameters = [p for p in module.parameters() if p.requires_grad]
                    terms = learner.losses(batch, feature_graph=(block == "feature"))
                    for name, weight_name in zip(LOSS_NAMES, WEIGHT_NAMES):
                        weighted = terms[name] * getattr(learner.config, weight_name)
                        losses[name] = dict(raw=float(terms[name].detach()), weight=getattr(learner.config, weight_name),
                                            weighted=float(weighted.detach()))
                        if weighted.requires_grad:
                            grads = torch.autograd.grad(weighted, parameters, allow_unused=True, retain_graph=True)
                        else:
                            grads = [None] * len(parameters)
                        # Keep comparison vectors on CPU to avoid accumulating GPU graphs.
                        vector = torch.cat([torch.zeros(p.numel(), dtype=torch.float32) if g is None
                                            else g.detach().reshape(-1).float().cpu()
                                            for p, g in zip(parameters, grads)])
                        vectors[name] = vector
                        norms[name] = float(torch.linalg.vector_norm(vector))
                        del grads
                    cosine = {}
                    for i, left in enumerate(LOSS_NAMES):
                        for right in LOSS_NAMES[i+1:]:
                            denominator = norms[left] * norms[right]
                            cosine[left + "/" + right] = (float(torch.dot(vectors[left], vectors[right])) / denominator
                                                           if denominator > 1e-20 else None)
                    total = sum(vectors.values())
                    row = dict(sample=sample_name, mode=mode, block=block, examples=len(keys),
                               physical_error=dict(min=float(E.min()), median=float(np.median(E)), max=float(E.max())),
                               losses=losses, weighted_gradient_norm=norms, cosine=cosine,
                               combined_gradient_norm=float(torch.linalg.vector_norm(total)))
                    all_rows.append(row)
                    write_new_json(output/f"{sample_name}_{mode}_{block}.json", row)
                    if event:
                        event("gradient_audit", sample=sample_name, mode=mode, block=block, norms=norms)
                    del terms, vectors, total
        del batch
    if previous_mode is not None:
        learner.calibration_mode = previous_mode
    after = {k: parameter_digest(getattr(learner, k)) for k in ("features", "metric")}
    if before != after:
        raise RuntimeError("Gradient-only audit changed model tensors")
    result = dict(schema="rgb_v5_loss_gradient_audit", parameter_hashes_before=before, parameter_hashes_after=after,
                  parameters_unchanged=True, optimizer_steps=0, rows=all_rows,
                  expected_zero_metric_terms=["interaction", "reconstruction", "geometry"],
                  source=replay.source_summary(),
                  limitation="Fixed training batches; gradient magnitude does not establish causal control improvement.")
    write_new_json(output/"GRADIENT_AUDIT.json", result)
    return result


@torch.no_grad()
def probe(learner, batch, local_step, block):
    goal = learner.features.encode(batch.observation.goal_image)
    e = learner.features.encode(batch.observation.image) - goal
    M = learner.metric(goal)
    V = quadratic_value(e, M)
    E = learner.adapter.physical_error(batch.offset)
    values = dict(step=local_step, block=block, M=M.cpu().numpy(), V=V.cpu().numpy(), E=E.cpu().numpy(),
                  feature_error=e.cpu().numpy(), lower_impossible=(7.25*e.square().sum(-1) < .1*E).cpu().numpy())
    return values


def train_arm(base, replay, geometry, plan, output, *, arm, precision_floor=.0004,
              guard=lambda: None, event=None, source=None):
    """600+600 by default; checkpoint is only frozen after all planned updates."""
    if arm not in ARMS:
        raise ValueError(arm)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"updates").mkdir()
    mode, schedule = ARMS[arm]
    learner = CalibrationRepairLearner(base, calibration_mode=mode, precision_floor=precision_floor)
    for module in (learner.features, learner.metric):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(True)
    initial_hashes = {k: parameter_digest(getattr(learner, k)) for k in ("features", "metric")}
    config = dict(arm=arm, calibration_mode=mode, schedule=schedule, pairs=plan["pairs"],
                  batch_size=plan["batch_size"], precision_floor=precision_floor, source=source,
                  initial_hashes=initial_hashes, original_optimizer_states_loaded=False,
                  expected_counts=dict(feature=plan["pairs"], metric=plan["pairs"]),
                  learner_config=asdict(learner.config), control_config=asdict(learner.adapter.config),
                  source_training_reused=True, source_weights_reused=True, fresh_transitions=0,
                  split_semantics="Freeze theta parameters in phi block while preserving phi(goal)->M derivatives; M values may change.",
                  selection="Final planned update only; no validation-directed early selection")
    write_new_json(output/"ARM_PROTOCOL.json", config)
    fixed_batch = replay.batch(plan["probe_keys"], geometry, with_geometry=False)
    probes = [probe(learner, fixed_batch, 0, "initial")]
    start = time.perf_counter()

    def update(block, pair, batch):
        guard()
        metrics = (learner.update_features(batch) if block == "feature" else learner.update_metric(batch))
        local_step = learner.counts["feature"] + learner.counts["metric"]
        row = dict(arm=arm, local_step=local_step, pair=pair, block=block, metrics=metrics,
                   elapsed_s=time.perf_counter()-start)
        write_new_json(output/"updates"/f"{local_step:06d}_{block}.json", row)
        probes.append(probe(learner, fixed_batch, local_step, block))
        if event and (local_step % 100 == 0 or local_step == 1):
            event("repair_progress", arm=arm, local_step=local_step, pairs=plan["pairs"], metrics=metrics)

    if schedule == "alternating":
        for pair, keys in enumerate(plan["batch_keys"], 1):
            guard()
            batch = replay.batch(keys, geometry)
            update("feature", pair, batch)
            update("metric", pair, batch)
    else:
        for block in ("feature", "metric"):
            for pair, keys in enumerate(plan["batch_keys"], 1):
                guard()
                batch = replay.batch(keys, geometry)
                update(block, pair, batch)
    guard()
    torch.cuda.synchronize() if next(learner.features.parameters()).is_cuda else None
    counts = learner.counts.copy()
    if counts["feature"] != plan["pairs"] or counts["metric"] != plan["pairs"]:
        raise RuntimeError("Incomplete planned repair updates")
    payload = dict(models=dict(features=learner.features.state_dict(), metric=learner.metric.state_dict()),
                   optimizers=dict(features=learner.feature_optimizer.state_dict(), metric=learner.metric_optimizer.state_dict()),
                   runtime_state=dict(counts=counts, version=learner.version, execution_spec=learner.execution_spec,
                                      learner_config=asdict(learner.config), control_config=asdict(learner.adapter.config),
                                      diagnostic_arm=arm, automatic_resume=False))
    checkpoint = output/"final.pt"
    temporary = checkpoint.with_name(checkpoint.name+".tmp")
    with temporary.open("xb") as stream:
        torch.save(dict(schema="rgb_v5_repair_checkpoint", boundary="complete_update", payload=payload,
                        completed_updates=counts["feature"]+counts["metric"]), stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, checkpoint)
    with (output/"STEP_PROBES.npz").open("xb") as stream:
        np.savez_compressed(stream, **{key: np.stack([r[key] for r in probes]) for key in probes[0]})
    final_hashes = {k: parameter_digest(getattr(learner, k)) for k in ("features", "metric")}
    result = dict(schema="rgb_v5_repair_frozen", arm=arm, checkpoint=str(checkpoint),
                  sha256=file_sha256(checkpoint), counts=counts, initial_hashes=initial_hashes,
                  final_hashes=final_hashes, elapsed_s=time.perf_counter()-start,
                  new_training=True, from_scratch=False, new_transitions=0,
                  source=replay.source_summary(), frozen_before_validation=True)
    write_new_json(output/"FROZEN.json", result)
    if event:
        event("repair_frozen", arm=arm, checkpoint=str(checkpoint), sha256=result["sha256"], counts=counts)
    return learner, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=("plan", "gradients", "train"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    # The parent owns GPU scheduling, authorization and the frozen overall budget.
    from .common import bootstrap, gpu_lease, freeze_stage, guard, read_json
    root = Path(args.root)
    protocol = read_json(root/"PROTOCOL.json")
    guard(root)
    replay = ReadOnlyV4Replay(Path(protocol["old_root"])/"replay_indices"/"update_014000.json")
    plan_path = root/"TRAINING_KEYS.json"
    try:
        if args.mode == "plan":
            pairs = protocol["conditional_training"]["updates_per_arm"]["feature"]
            if pairs != protocol["conditional_training"]["updates_per_arm"]["metric"]:
                raise ValueError("Matched module budgets required")
            result = create_key_plan(replay, plan_path, pairs=pairs,
                                     batch_size=protocol["conditional_training"]["batch_size"])
            freeze_stage(root, "training_key_plan", dict(path=str(plan_path), sha256=file_sha256(plan_path)),
                         [__file__])
            print(json.dumps({"plan": str(plan_path), "pairs": result["pairs"]}))
            return
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["replay_index_sha256"] != replay.index_sha256:
            raise RuntimeError("Training archive index changed")
        if args.mode == "train":
            if args.arm is None:
                raise ValueError("Training requires one explicit arm")
            authorization = read_json(root/"TRAINING_AUTHORIZATION.json")
            if not authorization.get("authorized") or args.arm not in authorization.get("arms", []):
                raise RuntimeError("Root's evidence-based training activation is required")
        stage = "gradient_audit" if args.mode == "gradients" else "training_"+args.arm
        with gpu_lease(root, stage):
            ctx = bootstrap(root)
            floor = precision_floor_from_task(ctx.cfg)
            freeze_stage(root, stage,
                         dict(key_plan_sha256=file_sha256(plan_path), precision_floor=floor,
                              mode=args.mode, arm=args.arm, source=ctx.provenance), [__file__])
            if args.mode == "gradients":
                learner = CalibrationRepairLearner(ctx.learner, precision_floor=floor)
                result = gradient_audit(learner, replay, ctx.geometry, plan, root/"gradient_audit",
                                       guard=ctx.guard, event=ctx.event)
                print(json.dumps({"gradient_rows": len(result["rows"]), "parameters_unchanged": True}))
            else:
                _, result = train_arm(ctx.learner, replay, ctx.geometry, plan, root/"training"/args.arm,
                                     arm=args.arm, precision_floor=floor, guard=ctx.guard, event=ctx.event,
                                     source=ctx.provenance)
                print(json.dumps({k:v for k,v in result.items() if k != "source"}, ensure_ascii=False))
    finally:
        replay.close()


if __name__ == "__main__":
    main()
