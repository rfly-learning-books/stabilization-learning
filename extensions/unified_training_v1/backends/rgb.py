"""Staged bridge for the preserved AE/derivative-Critic/goal-M RGB learner.

New GPU-rendered transitions; audited historical initial AE/M, empty optimizers.
This is a restoration pilot, not random-origin learning or original-pixel parity.
No historical source, checkpoint, run, threshold or optimizer is modified.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[3]
for _path in (PROJECT, PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _module_hash(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.rename(path)


class Backend:
    """Public stage API; all counts refer to real calls to retained optimizers."""
    name = "rgb"
    capabilities = dict(representation=True, evaluator=True, evaluator_kind="signed_derivative",
                        certificate=True, replay="scene_balanced_transitions", physical_action=True)

    def __init__(self, config, output, event=None, deadline=None):
        self.config = dict(config)
        self.output = Path(output).resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.event = event or (lambda kind, **payload: None)
        self.deadline = deadline
        self.device = self.config.get("device", "cuda:0")
        self.seed = int(self.config.get("seed", 81))
        self.batch_size = int(self.config.get("batch_size", 4))
        self.collection_id = self.evaluation_id = 0
        self.history = []
        self.fixed_probe = None
        self.initialized = False

    def guard(self):
        if getattr(self, "closed", False):
            raise RuntimeError("RGB backend is closed")
        if any((folder / "STOP").exists() for folder in
               (self.output, self.output.parent, self.output.parent.parent)):
            raise InterruptedError("RGB STOP requested")
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError("RGB deadline reached; no extension or automatic restart")

    def initialize(self):
        if self.initialized:
            raise RuntimeError("Already initialized")
        self.guard()
        from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
        from extensions.joint_certificate_v4.rollout import sample_offset
        self.adapter, _, self.scenes = load_runtime(PROJECT)
        self.cfg = deepcopy(self.adapter.cfg)
        self.cfg["goal_fixed"]["batch_size"] = self.batch_size
        self.cfg["collection"]["replay_transitions"] = int(self.config.get("replay_capacity", 20000))
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed); torch.set_num_threads(int(self.config.get("cpu_threads", 2)))
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        source = self.adapter.checkpoint("initial.pt")
        if any(source["learner"]["counts"].values()):
            raise ValueError("Historical initial must precede task updates")
        # Constructor imports only initial AE/M and anchors; all optimizers and
        # derivative Critic are initialized in this new run.
        self.learner = self.adapter.derivative.DerivativeLearner(self.cfg, source, self.device)
        if any(opt.state for opt in (self.learner.aeopt, self.learner.shadowopt,
                                    self.learner.criticopt, self.learner.mopt)):
            raise AssertionError("Optimizer is not empty")
        self.initial_state = deepcopy(self.learner.payload())
        self.initial_hash = _module_hash(self.learner.ae)
        self.initial_metric_hash = _module_hash(self.learner.metric)
        self.depth = self.adapter.core.FastDepth(self.cfg, self.device)
        parent = self.adapter.core.Replay
        cache_limit = int(self.config.get("replay_cached_episodes", 96))
        class MemoryCachedReplay(parent):
            def load(replay, row):
                if row["file"] not in replay.cache:
                    with np.load(replay.root / row["file"], allow_pickle=False) as archive:
                        replay.cache[row["file"]] = {key: archive[key] for key in archive.files}
                    while len(replay.cache) > cache_limit:
                        replay.cache.popitem(last=False)
                replay.cache.move_to_end(row["file"])
                return replay.cache[row["file"]]
        self.replay = MemoryCachedReplay(self.output, self.cfg, self.seed + 1901)
        if self.replay.rows:
            raise FileExistsError("New RGB backend requires empty replay")
        self.rng = np.random.default_rng(self.seed + 90281)
        self.lookup = {scene["id"]: scene for scene in self.scenes}
        self.train_scenes = sorted([s for s in self.scenes if s["split"] == "train"], key=lambda s: s["id"])
        limit = int(self.config.get("train_scene_count", len(self.train_scenes)))
        if not 1 <= limit <= len(self.train_scenes):
            raise ValueError("Invalid training scene count")
        self.train_scenes = [self.train_scenes[i] for i in np.linspace(0, len(self.train_scenes)-1, limit, dtype=int)]
        self.cases = {}
        for split, number, salt in (("val", int(self.config.get("validation_cases", 8)), 731000),
                                    ("test", int(self.config.get("test_cases", 16)), 941000)):
            scenes = sorted([s for s in self.scenes if s["split"] == split], key=lambda s: s["id"])
            rng = np.random.default_rng(self.seed + salt)
            rows = []
            for i in range(number):
                scene = scenes[i % len(scenes)]
                goal = (i // len(scenes)) % len(scene["refs"])
                difficulty = ("small", "medium", "large")[(i // len(scenes) + i) % 3]
                offset = sample_offset(self.cfg, scene, goal, difficulty, rng)
                rows.append(dict(case_id=f"unified_{split}_{i:04d}", scene_id=scene["id"],
                                 goal_id=goal, difficulty=difficulty, offset=offset.tolist(), split=split))
            self.cases[split] = rows
        _write(self.output / "RGB_CASES.json", self.cases)
        self.initialized = True
        self.save(self.output / "rgb_initial.pt")
        provenance = dict(backend="legacy_derivative_rgb_gpu_bridge", initialization="audited historical initial AE/M; fresh derivative Critic and empty Adam",
            random_origin=False, old_replay_loaded=False, initial_checkpoint=str(self.adapter.bundle / "initial.pt"),
            initial_checkpoint_sha256=_sha(self.adapter.bundle / "initial.pt"),
            old_task_counts=source["learner"]["counts"], initial_ae_sha256=self.initial_hash,
            initial_metric_sha256=self.initial_metric_hash, train_scenes=[s["id"] for s in self.train_scenes],
            representation_gradient_caps=dict(derivative=.25, regularization=.05),
            target_J=True, error_and_J_divisor=4, goal_metric_input_divisor=1, original_ridge=True,
            inference_preprocessing="original CPU image_tensor", sampling="fresh GPU rendered transitions; original scene-balanced replay sampling",
            replay_cache_limit=cache_limit, pixel_equivalent_to_original=False,
            prior_AE_pretraining_budget_fully_known=False, shadow_AE=True,
            module_updates=self.learner.counts.copy(), runtime=dict(python=sys.version, torch=torch.__version__))
        _write(self.output / "RGB_PROVENANCE.json", provenance)
        self.event("rgb_initialized", **provenance)
        return provenance

    def _env(self, cases, plant="P0", max_steps=None):
        from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
        self.guard()
        return GPUParallelRGBEnv(self.cfg, [self.lookup[c["scene_id"]] for c in cases], plant=plant,
            max_steps=int(max_steps or self.config.get("evaluation_steps", 200)), device=self.device,
            cache_dir=PROJECT / ".cache/rgb_gpu_warp", samples_per_axis=2)

    def _policy(self, public, variant="selected", depth=None):
        from extensions.rgb_diagnostic_v5.legacy_transfer import LegacyPolicy
        if variant == "initial":
            if not hasattr(self, "initial_models"):
                ae, metric = self.adapter.load_models("initial.pt", self.device)
                ae.requires_grad_(False); metric.requires_grad_(False)
                self.initial_models = (ae, metric)
            ae, metric = self.initial_models
        elif variant == "shadow":
            ae, metric = self.learner.shadow, self.learner.metric
        elif variant in ("selected", "identity"):
            ae, metric = self.learner.ae, self.learner.metric
        else:
            raise ValueError(variant)
        return LegacyPolicy(self.adapter, ae.eval(), metric.eval(), depth or self.depth, public,
                            identity=variant in ("identity", "shadow"))

    def _triplet(self, env, public):
        from lac_vs.environment import valid_start
        offset = env._offset().cpu().numpy()
        deltas = []
        for i, scene in enumerate(env.scenes):
            frame = env.room.ReferenceFrame(np.asarray(scene["refs"][env.goal_ids[i]]["goal_matrix"]))
            accepted = np.zeros(4)
            for _ in range(100):
                delta = self.rng.normal(size=4); delta[:3] *= .012; delta[3] *= math.radians(.6)
                if valid_start(self.cfg, scene, frame, offset[i]-delta) and valid_start(self.cfg, scene, frame, offset[i]+delta):
                    accepted = delta; break
            deltas.append(accepted)
        delta = env._tensor(np.asarray(deltas))
        displacement = torch.einsum("bij,bj->bi", env.body0, delta[:, :3])
        minus = env._render(env.position-displacement, env.yaw-delta[:, 3]).cpu().numpy()
        plus = env._render(env.position+displacement, env.yaw+delta[:, 3]).cpu().numpy()
        return dict(smooth_center=public["rgb"].cpu().numpy(), smooth_minus=minus,
                    smooth_plus=plus, smooth_step=np.linalg.norm(deltas, axis=1).astype(np.float32))

    def collect(self, initial=False):
        from extensions.joint_certificate_v4.rollout import sample_offset
        from train_goal_fixed import augment_depth
        started = time.perf_counter(); before = self.replay.transitions
        collection = self.collection_id; count = 0; coverage = []
        horizon = int(self.config.get("initial_collection_steps" if initial else "collection_steps", 16))
        lanes = int(self.config.get("parallel_envs", 4))
        if lanes < 1 or horizon < 1:
            raise ValueError("Positive collection dimensions required")
        for scene in self.train_scenes:
            self.guard()
            cases = []
            for i in range(lanes):
                goal = (collection * lanes + i) % len(scene["refs"])
                difficulty = ("small", "medium", "large")[(collection+i) % 3]
                offset = sample_offset(self.cfg, scene, goal, difficulty, self.rng)
                cases.append(dict(scene_id=scene["id"], goal_id=goal, offset=offset.tolist(), difficulty=difficulty))
            env = self._env(cases, max_steps=horizon)
            try:
                public, info = env.reset(options=dict(offsets=[c["offset"] for c in cases], goal_ids=[c["goal_id"] for c in cases]))
                goals = public["goal_rgb"].cpu().numpy(); rotations = public["rotation"].cpu().numpy()
                base_depth = np.stack([self.depth.predict(image) for image in goals])
                used_depth = base_depth if initial else np.stack([augment_depth(d.copy(), self.rng, self.cfg)[0] for d in base_depth])
                depth_map = {image.tobytes(): d for image, d in zip(goals, used_depth)}
                class DepthMap:
                    def predict(_, image): return depth_map[image.tobytes()]
                policy = self._policy(public, depth=DepthMap())
                gray = []
                if initial:
                    vision = self.adapter.core.legacy.module(self.cfg, "vision")
                    for i, case in enumerate(cases):
                        frame = env.room.ReferenceFrame(np.asarray(scene["refs"][case["goal_id"]]["goal_matrix"]))
                        target, interaction = vision.observe_gray(frame, goals[i], base_depth[i])
                        j = interaction.reshape(-1, 4).astype(np.float64); gram = j.T @ j
                        h = gram + .01*np.diag(np.diag(gram)) + max(1e-8, 1.01*np.trace(gram)/9999)*np.eye(4)
                        gray.append((target.ravel(), .7*np.linalg.solve(h, j.T)))
                ar = {k: [] for k in ("rgb", "offset", "action", "cost", "terminal", "truncated", "previous_action")}
                ar["rgb"].append(public["rgb"].cpu().numpy()); ar["offset"].append(info["supervision"]["offset"].cpu().numpy())
                triplets = []
                lengths = np.zeros(lanes, dtype=int); alive = np.ones(lanes, dtype=bool)
                for step in range(horizon):
                    self.guard()
                    if not alive.any(): break
                    if step % 10 == 0 and len(triplets) < self.cfg["goal_fixed"]["smooth_triplet_limit_per_episode"]:
                        triplets.append(self._triplet(env, public))
                    if initial:
                        commands = []
                        for image, (target, gain) in zip(public["rgb"].cpu().numpy(), gray):
                            raw = -gain @ (vision.preprocess_gray(image).ravel()-target)
                            commands.append(raw / max(1., np.linalg.norm(raw[:3]), abs(raw[3])))
                        action = torch.as_tensor(np.stack(commands), dtype=torch.float32, device=self.device)
                    else:
                        action = policy(public).action
                    noise = torch.as_tensor(self.rng.normal(0, .05 if initial else .06, (lanes, 4)), dtype=torch.float32, device=self.device)
                    action = self.adapter.core.saturate(action + noise)
                    following, _, terminated, truncated, info = env.step(action)
                    active = info["transition_mask"].cpu().numpy()
                    if not active[alive].all(): raise RuntimeError("Autoreset entered active training transition")
                    labels = info["supervision"]
                    for key, tensor in (("rgb", following["rgb"]), ("offset", labels["next_offset"]),
                        ("action", action), ("cost", labels["physical_cost"]), ("terminal", terminated),
                        ("truncated", truncated), ("previous_action", labels["previous_action"])):
                        ar[key].append(tensor.detach().cpu().numpy())
                    lengths += alive; alive &= ~(terminated | truncated).cpu().numpy(); public = following
                policy.check()
                arrays = {key: np.stack(values) for key, values in ar.items()}
                for lane, (case, length) in enumerate(zip(cases, lengths)):
                    if not length: raise RuntimeError("Empty initial trajectory")
                    item = {key: values[:length+(key in ("rgb", "offset")), lane] for key, values in arrays.items()}
                    item.update(goal_rgb=goals[lane], goal_depth=base_depth[lane], collection_depth=used_depth[lane], rotation=rotations[lane])
                    for key in triplets[0]: item[key] = np.stack([triplet[key][lane] for triplet in triplets])
                    row = dict(scene_id=scene["id"], split="train", goal_id=case["goal_id"])
                    self.replay.add(item, row, f"gpu_{collection:03d}_{scene['id']}_{lane:02d}")
                    count += int(length); coverage.append({**case, "transitions": int(length)})
                self.event("rgb_collection_scene", collection=collection, scene=scene["id"], transitions=int(lengths.sum()))
            finally:
                env.close()
        self.collection_id += 1
        if self.fixed_probe is None:
            state = deepcopy(self.replay.rng.bit_generator.state)
            self.fixed_probe = self.replay.sample(int(self.config.get("probe_batch_size", 4)))
            self.replay.rng.bit_generator.state = state
            self._record_probe("initial_data")
        result = dict(stage="initial_collection" if initial else "collection", fresh_transitions=count,
            replay_transitions=self.replay.transitions, old_replay_loaded=False, scene_count=len(self.train_scenes),
            prior_replay_transitions=before, wall_s=time.perf_counter()-started, coverage=coverage)
        _write(self.output / f"RGB_COLLECTION_{collection:03d}.json", result)
        return result

    def _optimizer_counts(self):
        c = self.learner.counts
        return dict(ae=c["warmup"]+c["representation"], shadow_ae=c["warmup"]+c["representation"],
                    derivative_critic=c["critic_prefit"]+c["representation"], metric=c["metric"])

    @torch.no_grad()
    def _record_probe(self, phase):
        if self.fixed_probe is None: return
        b = self.fixed_probe
        z = self.learner.encode(b["rgb"]); goal = self.learner.encode(b["goal_rgb"])
        _, matrix = self.learner.metric(goal)
        error = (z-goal)/4
        value = self.adapter.core.energy(error, matrix)
        physical = self.adapter.core.pose_error(self.learner.tensor(b["offset"]))
        row = dict(stage_update=sum(self.learner.counts.values()), phase=phase,
            phase_counts=self.learner.counts.copy(), optimizer_counts=self._optimizer_counts(),
            matrix=matrix.cpu().tolist(), value=value.cpu().tolist(), physical_error=physical.cpu().tolist(),
            feature_error=error.cpu().tolist(), ratio=(value/physical.clamp_min(1e-12)).cpu().tolist(),
            goal_keys=list(b["goal_key"]), fixed_training_probe=True)
        with (self.output / "rgb_step_probes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")

    def _updates(self, phase, updates):
        self.guard()
        if int(updates) != updates or updates < 0: raise ValueError("Nonnegative integer updates required")
        if updates and not self.replay.transitions: raise RuntimeError("Collect before training")
        begun = time.perf_counter(); rows = []
        for i in range(updates):
            self.guard(); batch = self.replay.sample(self.batch_size)
            if phase == "warmup": result = self.learner.representation(batch, warmup=True)
            elif phase == "critic_prefit": result = self.learner.critic_prefit(batch)
            elif phase == "representation": result = self.learner.representation(batch)
            elif phase == "metric": result = self.learner.metric_update(batch)
            else: raise ValueError(phase)
            row = dict(phase=phase, update=int(self.learner.counts[phase]), counts=self.learner.counts.copy(), **result)
            rows.append(row); self.history.append(row)
            with (self.output / "rgb_updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self._record_probe(phase)
            if (i+1) % 50 == 0 or i+1 == updates:
                self.event("rgb_update", phase=phase, finished=i+1, planned=updates, counts=self.learner.counts.copy())
        return dict(phase=phase, updates=updates, counts=self.learner.counts.copy(), wall_s=time.perf_counter()-begun,
                    optimizer_counts=self._optimizer_counts(), last=rows[-1] if rows else None)

    def warmup_representation(self, updates): return self._updates("warmup", updates)
    def pretrain_evaluator(self, updates): return self._updates("critic_prefit", updates)

    def train_epoch(self, epoch, representation_updates, evaluator_updates, certificate_updates):
        # Each original representation update also updates the derivative Critic.
        # Extra evaluator-only calls are explicit and never silently introduced.
        result = dict(epoch=epoch, representation=self._updates("representation", representation_updates),
            evaluator=self._updates("critic_prefit", evaluator_updates), certificate=self._updates("metric", certificate_updates))
        result["critic_updates_in_representation"] = representation_updates
        result["counts"] = self.learner.counts.copy()
        return result

    def save(self, path):
        self.guard(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        payload = dict(schema="unified_rgb_legacy_v1", learner=deepcopy(self.learner.payload()),
            initial_state=self.initial_state, replay_rows=deepcopy(self.replay.rows),
            replay_rng=deepcopy(self.replay.rng.bit_generator.state), collection_rng=deepcopy(self.rng.bit_generator.state),
            collection_id=self.collection_id, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), config=self.config,
            fixed_probe=self.fixed_probe,
            initial_ae_sha256=self.initial_hash, initial_metric_sha256=self.initial_metric_hash)
        tmp = path.with_suffix(path.suffix+".tmp"); torch.save(payload, tmp); tmp.rename(path)
        return dict(path=str(path), sha256=_sha(path), counts=self.learner.counts.copy())

    def load(self, path):
        self.guard(); payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload["schema"] != "unified_rgb_legacy_v1" or payload["config"] != self.config:
            raise ValueError("RGB checkpoint contract mismatch")
        self.learner.restore(payload["learner"]); self.learner.inputs.clear()
        self.fixed_probe = payload.get("fixed_probe")
        self.replay.rows = deepcopy(payload["replay_rows"]); self.replay.cache.clear()
        if any(not (self.replay.root/r["file"]).is_file() for r in self.replay.rows):
            raise FileNotFoundError("RGB replay shard absent")
        self.replay.rng.bit_generator.state = payload["replay_rng"]
        self.rng.bit_generator.state = payload["collection_rng"]; self.collection_id = payload["collection_id"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])
        np.random.set_state(payload["numpy_rng"]); random.setstate(payload["python_rng"])
        return dict(path=str(path), sha256=_sha(path), counts=self.learner.counts.copy())

    def validate(self):
        result = self.evaluate("val", ("selected",))
        result["selection_score"] = [result["summary"]["success_rate"], -result["summary"]["mean_integrated_error"]]
        return result

    def evaluate(self, split="test", variants=("selected", "initial", "identity")):
        self.guard(); split = "val" if split == "validation" else split
        if split not in self.cases: raise ValueError("Unknown evaluation split")
        if split == "test":
            frozen_path = self.output / "FROZEN.json"
            global_path = self.output.parent.parent / "GLOBAL_FROZEN.json"
            if not frozen_path.is_file() or not global_path.is_file():
                raise RuntimeError("Test requires job and global freeze barriers")
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            global_frozen = json.loads(global_path.read_text(encoding="utf-8"))
            if global_frozen.get("schema") != "unified_frozen_group_v1":
                raise RuntimeError("Unknown global freeze schema")
            if global_frozen["jobs"].get(self.output.name) != _sha(frozen_path):
                raise RuntimeError("RGB job freeze is absent from or changed after group freeze")
            job_protocol_path = self.output / "protocol.json"
            if _sha(job_protocol_path) != frozen["protocol_sha256"]:
                raise RuntimeError("RGB job protocol changed after freezing")
            job_protocol = json.loads(job_protocol_path.read_text(encoding="utf-8"))
            if global_frozen["protocol_sha256"] != job_protocol["suite_protocol_sha256"]:
                raise RuntimeError("RGB job does not belong to the frozen group protocol")
            if _sha(self.output / frozen["checkpoint"]) != frozen["sha256"]:
                raise RuntimeError("Frozen RGB checkpoint changed")
            saved = torch.load(self.output / frozen["checkpoint"], map_location="cpu", weights_only=False)
            for name in ("ae", "metric"):
                for key, value in getattr(self.learner, name).state_dict().items():
                    if not torch.equal(value.detach().cpu(), saved["learner"][name][key].cpu()):
                        raise RuntimeError("Active RGB model is not the frozen selected model")
        variants = tuple(variants)
        if len(set(variants)) != len(variants) or set(variants)-{"selected", "initial", "identity", "shadow"}:
            raise ValueError("Unknown or repeated evaluation variant")
        cases = self.cases[split]; folder = self.output / f"RGB_EVAL_{self.evaluation_id:03d}_{split}"
        folder.mkdir(exist_ok=False); self.evaluation_id += 1
        _write(folder / "FROZEN.json", dict(created_unix=time.time(), variants=variants, split=split,
            ae_sha256=_module_hash(self.learner.ae), metric_sha256=_module_hash(self.learner.metric),
            cases_sha256=_sha(self.output / "RGB_CASES.json"), counts=self.learner.counts.copy()))
        rows = []; started = time.perf_counter()
        for plant in self.config.get("evaluation_plants", ("P0", "P1")):
            for variant in variants:
                for start in range(0, len(cases), int(self.config.get("evaluation_batch", 4))):
                    chunk = cases[start:start+int(self.config.get("evaluation_batch", 4))]
                    env = self._env(chunk, plant)
                    try:
                        public, info = env.reset(options=dict(offsets=[c["offset"] for c in chunk], goal_ids=[c["goal_id"] for c in chunk]))
                        policy = self._policy(public, variant)
                        n = len(chunk); alive = np.ones(n, bool); lengths = np.zeros(n, int)
                        reason = np.array(["timeout"]*n, dtype=object)
                        trace = {k: [] for k in ("error", "V", "offset", "action", "applied_action", "dt", "integrated_error", "input_squared", "hold", "terminated", "truncated")}
                        first = policy(public); matrix = first.metric.detach().cpu().numpy()
                        trace["error"].append(first.error.cpu().numpy()); trace["V"].append(first.value.cpu().numpy())
                        trace["offset"].append(info["supervision"]["offset"].cpu().numpy())
                        wall = []
                        for step in range(env.max_steps):
                            self.guard()
                            if not alive.any(): break
                            torch.cuda.synchronize(); t = time.perf_counter(); out = policy(public)
                            torch.cuda.synchronize(); wall.append(time.perf_counter()-t)
                            following, _, terminal, truncated, info = env.step(out.action)
                            labels = info["supervision"]
                            if not info["transition_mask"].cpu().numpy()[alive].all(): raise RuntimeError("Evaluation autoreset contamination")
                            with torch.no_grad():
                                successor = policy.error(following["rgb"])
                                value = self.adapter.core.energy(successor, out.metric)
                            for key, tensor in (("error", successor), ("V", value), ("offset", labels["next_offset"]),
                                ("action", out.action), ("applied_action", labels["actual_velocity"]/env.scale), ("dt", info["dt"]),
                                ("integrated_error", labels["integrated_error"]), ("input_squared", labels["normalized_input_squared_integral"]),
                                ("hold", info["hold_observations"]), ("terminated", terminal), ("truncated", truncated)):
                                trace[key].append(tensor.detach().cpu().numpy())
                            ended = alive & (terminal | truncated).cpu().numpy()
                            for label in ("success", "collision", "outside"):
                                reason[ended & info[label].cpu().numpy()] = label
                            lengths += alive; alive &= ~(terminal | truncated).cpu().numpy(); public = following
                        policy.check()
                        host = {key: np.stack(values) for key, values in trace.items()}
                        for lane, (case, length) in enumerate(zip(chunk, lengths)):
                            arrays = {key: values[:length+(key in ("error", "V", "offset")), lane] for key, values in host.items()}
                            arrays.update(M=matrix[lane], J=first.interaction[lane].cpu().numpy())
                            audit = self._audit(arrays, reason[lane])
                            path = folder / f"{plant}_{variant}_{start+lane:04d}.npz"
                            with path.open("xb") as stream: np.savez_compressed(stream, **arrays)
                            q = arrays["offset"]; E = (q[:, :3]**2).sum(1)+(q[:, 3]/math.radians(25))**2
                            dv = np.diff(arrays["V"]); tol = 1e-7+1e-4*arrays["V"][:-1]
                            row = dict(**case, plant=plant, variant=variant, success=reason[lane]=="success", reason=str(reason[lane]), steps=int(length),
                                elapsed_s=float(arrays["dt"].sum()), integrated_error=float(arrays["integrated_error"].sum()),
                                input_squared_integral=float(arrays["input_squared"].sum()), final_error=float(E[-1]),
                                V_rise_fraction=float((dv>tol).mean()), V_lower_violation=float((arrays["V"]<.25*E-1e-7).mean()),
                                V_over_E_median=float(np.median(arrays["V"]/(E+1e-12))),
                                policy_wall_s=float(np.mean(wall[:length])), trajectory=str(path), trajectory_sha256=_sha(path), audit=audit)
                            rows.append(row)
                            self.event("rgb_evaluation_episode", split=split, plant=plant, variant=variant,
                                       case_id=case["case_id"], success=row["success"], reason=row["reason"])
                    finally:
                        env.close()
        groups = {}
        for plant in self.config.get("evaluation_plants", ("P0", "P1")):
            for variant in variants:
                group = [r for r in rows if r["plant"]==plant and r["variant"]==variant]
                groups[f"{plant}/{variant}"] = dict(episodes=len(group), successes=sum(r["success"] for r in group),
                    success_rate=float(np.mean([r["success"] for r in group])),
                    mean_integrated_error=float(np.mean([r["integrated_error"] for r in group])),
                    mean_input_squared=float(np.mean([r["input_squared_integral"] for r in group])),
                    failures={reason:sum(r["reason"]==reason for r in group) for reason in ("timeout", "collision", "outside")})
        selected = [r for r in rows if r["variant"]=="selected"]
        summary = dict(episodes=len(rows), unique_cases=len(cases), groups=groups,
            success_rate=float(np.mean([r["success"] for r in selected])) if selected else None,
            mean_integrated_error=float(np.mean([r["integrated_error"] for r in selected])) if selected else None,
            wall_s=time.perf_counter()-started, independent_successor_audit=True)
        result = dict(split=split, summary=summary, rows=rows, folder=str(folder))
        _write(folder / "RESULTS.json", result)
        return result

    def _audit(self, a, reason):
        e, m, j = (a[key].astype(np.float64) for key in ("error", "M", "J"))
        value = .5*np.einsum("bi,ij,bj->b", e, m, e)
        gram = j.T @ m @ j; ridge = max(1.01*np.trace(gram)/9999, 1e-8)
        h = gram+.01*np.diag(np.diag(gram))+ridge*np.eye(4)
        raw = -.7*np.linalg.solve(h, (j.T @ m @ e[:-1].T)).T
        action = raw / np.maximum(1., np.maximum(np.linalg.norm(raw[:, :3], axis=1), abs(raw[:, 3])))[:, None]
        if not np.allclose(action, a["action"], atol=5e-4, rtol=1e-4): raise AssertionError("Action regression")
        if not np.allclose(value, a["V"], atol=2e-5, rtol=1e-5): raise AssertionError("Complete successor V regression")
        q = a["offset"]; E = (q[:, :3]**2).sum(1)+(q[:, 3]/math.radians(25))**2
        if not np.allclose(E[1:]*a["dt"], a["integrated_error"], atol=1e-6, rtol=1e-6): raise AssertionError("Physical cost mismatch")
        good = (np.linalg.norm(q[1:, :3], axis=1)<.02) & (abs(q[1:, 3])<math.radians(1))
        counts = []; hold = 0
        for yes in good: hold = hold+1 if yes else 0; counts.append(hold)
        if not np.array_equal(counts, a["hold"]): raise AssertionError("Hold regression")
        if reason=="success" and counts[-1]<11: raise AssertionError("False success")
        if a["terminated"][:-1].any() or a["truncated"][:-1].any(): raise AssertionError("Transition after termination")
        return dict(passed=True, maximum_action_difference=float(np.max(abs(action-a["action"]))),
                    maximum_V_difference=float(np.max(abs(value-a["V"]))))

    def close(self):
        if hasattr(self, "replay"):
            self.replay.cache.clear()
        # empty_cache alone cannot release live model/optimizer tensors. The
        # runner retains this backend through frozen tests and closes it once.
        for name in ("learner", "depth", "initial_models", "initial_state", "fixed_probe"):
            if hasattr(self, name):
                delattr(self, name)
        self.closed = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


RGBBackend = Backend
