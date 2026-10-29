"""Preserved initialization, with an explicit fixed evaluation seed independent of training."""
from collections import OrderedDict
from copy import deepcopy
import math
import json
import random
import sys
import numpy as np
import torch
from extensions.unified_training_v1.backends.rgb import PROJECT, _module_hash, _sha, _write

def initialize_longroll(self):
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
        rng = np.random.default_rng(int(self.config.get("evaluation_seed", self.seed)) + salt)
        rows = []
        for i in range(number):
            scene = scenes[i % len(scenes)]
            goal = (i // len(scenes)) % len(scene["refs"])
            difficulty = ("small", "medium", "large")[(i // len(scenes) + i) % 3]
            offset = sample_offset(self.cfg, scene, goal, difficulty, rng)
            rows.append(dict(case_id=f"unified_{split}_{i:04d}", scene_id=scene["id"],
                             goal_id=goal, difficulty=difficulty, offset=offset.tolist(), split=split))
        self.cases[split] = rows
    if self.config.get("case_manifest"):
        # Used only by the training-room preflight; the formal runner rejects it.
        self.cases = json.loads(open(self.config["case_manifest"], encoding="utf-8").read())
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

