"""Book RGB protocol: physical room / derivative learner / staged runner."""
from copy import deepcopy
import importlib
import os
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from stabilization_learning.rgb_book import make_learner, inference_state, restore_models
from extensions.unified_training_v1.backends.rgb import Backend, PROJECT, _write, _sha, _module_hash


def save_tensor(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        raise FileExistsError(temp)
    torch.save(payload, temp)
    temp.rename(path)


class BookRGBBackend(Backend):
    """Preserve numerical updates and sampling; expose the common stage API."""
    def guard(self):
        super().guard()
        if shutil.disk_usage(self.output).free < float(self.config.get("minimum_free_gib", 20)) * 1024**3:
            raise InterruptedError("RGB disk reserve reached")

    def initialize(self):
        if self.initialized:
            raise RuntimeError("Already initialized")
        self.guard()
        from stabilization_gym.envs.tasks.original_visual_online import OriginalVisualOnlineAdapter
        reconstruction = PROJECT / "migration/visual_scene_reconstruction"
        self.adapter = OriginalVisualOnlineAdapter(PROJECT, reconstruction / "reconstructed_legacy_root",
                                                   reconstruction / "SCENE_AUDIT.json")
        self.adapter.joint = importlib.import_module("train_goal_fixed")
        data = Path(self.config["bootstrap_root"]).resolve()
        manifest = json.loads((data / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
        for entry in manifest["entries"]:
            target = (data / entry["relative"]).resolve()
            if not target.is_relative_to(data) or _sha(target) != entry["sha256"] or target.stat().st_size != entry["bytes"]:
                raise ValueError("Bootstrap asset changed: " + str(target))
        self.cfg = self.adapter.training_config(data)
        self.cfg["seed"] = self.seed
        self.cfg["goal_fixed"]["batch_size"] = self.batch_size
        self.cfg["collection"]["steps"] = self.config["collection_steps"]
        self.cfg["collection"]["episodes_per_scene_per_round"] = self.config["trajectories_per_scene"]
        self.cfg["collection"]["replay_transitions"] = self.config["replay_capacity"]
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(self.config.get("cpu_threads", 2))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        source = self.adapter.checkpoint("initial.pt")
        if any(source["learner"]["counts"].values()):
            raise ValueError("Historical zero-update initialization required")
        if source["learner"]["anchors"] != json.loads((data / "training_anchors.json").read_text(encoding="utf-8")):
            raise ValueError("Original normalization anchors changed")
        self.learner = make_learner(self.cfg, source, self.device)
        self.initial_hash = _module_hash(self.learner.ae)
        self.initial_metric_hash = _module_hash(self.learner.metric)
        self.depth = self.adapter.core.FastDepth(self.cfg, self.device)
        parent = self.adapter.core.Replay
        cache_limit = self.config.get("replay_cached_episodes", 16)
        class CachedReplay(parent):
            def load(replay, row):
                if row["file"] not in replay.cache:
                    with np.load(replay.root / row["file"], allow_pickle=False) as archive:
                        replay.cache[row["file"]] = {key: archive[key] for key in archive.files}
                    while len(replay.cache) > cache_limit:
                        replay.cache.popitem(last=False)
                replay.cache.move_to_end(row["file"])
                return replay.cache[row["file"]]
        self.replay = CachedReplay(self.output, self.cfg, self.seed)
        if self.replay.rows:
            raise FileExistsError("Fresh run required")
        rows = json.loads((data / "replay/index.json").read_text(encoding="utf-8"))
        fields = ("file", "length", "scene_id", "split", "goal")
        if ([tuple(r[k] for k in fields) for r in rows] !=
                [tuple(r[k] for k in fields) for r in source["replay_rows"]]):
            raise ValueError("Bootstrap order or metadata changed")
        if len(rows) != 48 or sum(r["length"] for r in rows) != 2822 or any(r["split"] != "train" for r in rows):
            raise ValueError("Expected original 48 shards / 2822 training transitions")
        for row in rows:
            shutil.copy2(data / "replay" / row["file"], self.replay.root / row["file"])
        self.replay.rows = deepcopy(rows)
        self.replay.rng.bit_generator.state = deepcopy(source["replay_rng"])
        _write(self.replay.path, rows)
        shutil.copytree(data / "references", self.output / "references")
        self.rng = np.random.default_rng(self.seed + 90017)
        self.scenes = json.loads((data / "scenes.json").read_text(encoding="utf-8"))["scenes"]
        self.lookup = {s["id"]: s for s in self.scenes}
        scenes = [s for s in self.scenes if s["split"] == "train"]
        self.train_scenes = scenes[:self.config.get("train_scene_count", len(scenes))]
        self.book_cases = json.loads(Path(self.config["book_cases"]).read_text(encoding="utf-8-sig"))
        for split, cases in self.book_cases.items():
            for index, case in enumerate(cases):
                case.setdefault("case_id", f"book_{split}_{index:04d}")
                case.setdefault("split", split)
                if self.lookup[case["scene_id"]]["split"] != split:
                    raise ValueError("Evaluation scene split mismatch")
        self.cases = self.book_cases
        self.reference_models = {}
        _write(self.output / "BOOK_CASES.json", self.book_cases)
        _write(self.output / "RESOLVED_ALGORITHM_CONFIG.json", self.cfg)
        self.adapter.config_module.restore_rng(source["rng"])
        self.initialized = True
        provenance = dict(schema="rgb_book_reproduction_v1", initialization="complete original initial learner and RNG",
            initial_checkpoint=str(self.adapter.bundle / "initial.pt"), initial_checkpoint_sha256=_sha(self.adapter.bundle / "initial.pt"),
            initial_ae_sha256=self.initial_hash, initial_metric_sha256=self.initial_metric_hash,
            random_origin=False, old_replay_loaded=True, bootstrap_shards=48, bootstrap_transitions=2822,
            bootstrap_manifest_sha256=_sha(data / "DATA_MANIFEST.json"), critic_reinitialized=False,
            full_initial_learner_restored=True, replay_rng_restored=True, global_rng_restored=True,
            train_scenes=[s["id"] for s in self.train_scenes], scene_order="original manifest order",
            scope="book training protocol in reconstructed SL rooms", original_rgb_equivalent=False,
            env_library="stabilization_gym.envs.tasks.rgb_book_physical",
            algo_library="stabilization_learning.rgb_book", renderer="audited reconstructed CPU renderer",
            neural_device=str(self.device), replay_cache_limit=cache_limit, fixed_goal_J=True,
            deterministic_cuda=True, reflection_backward="fixed ordered adjoint, same mathematical operator",
            error_and_J_divisor=4, goal_metric_input_divisor=1,
            prior_AE_pretraining_budget_fully_known=False)
        _write(self.output / "RGB_PROVENANCE.json", provenance)
        self.event("rgb_initialized", **provenance)
        return provenance

    def collect(self, initial=False):
        from .collection import book_collect
        return book_collect(self, initial=initial)

    def warmup_representation(self, updates):
        result = super().warmup_representation(updates)
        self.save(self.output / "pretraining/ae_warmup.pt", full=True)
        return result

    def save(self, path, full=False):
        self.guard(); path = Path(path)
        full = full or path.stem in ("initial", "ready", f"epoch_{self.config['epochs']:04d}")
        payload = dict(schema="rgb_book_checkpoint_v1", config=self.config,
            learner=deepcopy(self.learner.payload()) if full else inference_state(self.learner),
            optimizer_complete=full, initial_ae_sha256=self.initial_hash,
            initial_metric_sha256=self.initial_metric_hash,
            replay_rows=deepcopy(self.replay.rows) if full else None,
            replay_rng=deepcopy(self.replay.rng.bit_generator.state),
            collection_rng=deepcopy(self.rng.bit_generator.state), collection_id=self.collection_id,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate())
        save_tensor(path, payload)
        return dict(path=str(path), sha256=_sha(path), optimizer_complete=full, counts=self.learner.counts.copy())

    def load(self, path):
        self.guard()
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved.get("schema") != "rgb_book_checkpoint_v1" or saved["config"] != self.config:
            raise ValueError("RGB book checkpoint mismatch")
        restore_models(self.learner, saved["learner"])
        return dict(path=str(path), sha256=_sha(path), inference_only=True)

    def validate(self):
        result = self.evaluate("val", ("selected",))
        group = result["summary"]["groups"]["P0/selected"]
        rates = [v["successes"] / v["episodes"] for v in group["scene_id"].values()]
        result["selection_score"] = [float(np.mean(rates)), min(rates), -group["mean_integrated_error"]]
        return result

    def evaluate(self, split="test", variants=("selected", "identity", "shadow", "initial_identity")):
        from .evaluation import evaluate_cases, verify_frozen
        if split in ("val", "validation"):
            cases = self.book_cases["val"][:self.config.get("validation_cases", 48)]
            return evaluate_cases(self, cases, "val", variants, self.config["validation_plants"], self.config["validation_steps"])
        if split != "test":
            raise ValueError(split)
        verify_frozen(self)
        suites = []
        for suite in self.config["generalization_suites"]:
            verify_frozen(self)
            cases = self.book_cases[suite["cases"]]
            if suite.get("case_limit") is not None:
                cases = cases[:suite["case_limit"]]
            suites.append(evaluate_cases(self, cases, suite["name"], suite["variants"], suite["plants"],
                                         suite["steps"], suite.get("depth_scale", 1.)))
        return dict(complete=True, episodes=sum(s["summary"]["episodes"] for s in suites),
            generalization_suites=[{k: s[k] for k in ("split", "folder", "summary", "paired")} for s in suites])

    def close(self):
        if hasattr(self, "reference_models"):
            self.reference_models.clear()
        return super().close()
