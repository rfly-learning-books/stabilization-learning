"""Stage adapter for corrected episode-fixed M tasks, including old independent tasks."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import time
import uuid

import torch

from stabilization_learning.trainer import TrainConfig
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash
from stabilization_learning.unified_v1 import runner as evidence
from .independent import make_task
from .ppo import StableEpisodeLearner
from .validation import positive_int, seeds

TASKS = {"coupled_linear", "point_mass", "pendulum", "cartpole", "two_link_arm",
         "unicycle_tracking", "multirotor_tracking", "multirotor_hover", "multirotor_circle",
         "visual_servo", "push_any_shape"}


def validate_settings(config, schedule=None):
    known = {f.name for f in fields(TrainConfig)}
    allowed = known | {"task", "physical", "test_seeds", "cpu_threads", "total_updates"}
    if set(config) - allowed:
        raise ValueError("Unknown episode configuration: " + repr(sorted(set(config)-allowed)))
    if config.get("task", "two_link_arm") not in TASKS:
        raise ValueError("Unknown episode task")
    count = positive_int(config.get("total_updates", 1), "total_updates")
    if schedule and schedule["epochs"] * schedule["certificate_updates"] != count:
        raise ValueError("Episode schedule must exactly match total_updates")
    batch = positive_int(config.get("batch_episodes", 2), "batch_episodes")
    requested = count * batch
    if config.get("episodes", requested) != requested:
        raise ValueError("episodes must exactly equal total_updates * batch_episodes")
    train_seed = positive_int(config.get("seed", 81), "seed", 0)
    validation = seeds(config.get("validation_seeds", [810001, 810002]), "validation_seeds")
    test = seeds(config.get("test_seeds", [820001, 820002]), "test_seeds")
    train = set(range(train_seed, train_seed + requested))
    if train & set(validation) or train & set(test) or set(validation) & set(test):
        raise ValueError("Training, validation and test seed intervals must be disjoint")
    kwargs = {k: v for k, v in config.items() if k in known}
    kwargs.update(seed=train_seed, episodes=requested, batch_episodes=batch, validation_seeds=validation)
    return TrainConfig(**kwargs), test


def assert_test_gate(output, parameter_hash, checkpoint_key="actor"):
    output = Path(output)
    root = output.parent.parent
    protocol = evidence.read(root / "protocol.json")
    declared = [job["id"] for job in protocol["config"]["jobs"]]
    if output.name not in declared:
        raise RuntimeError("Job is absent from the declared experiment group")
    evidence.check_frozen(root, declared)
    gate = evidence.read(output / "FROZEN.json")
    path = output / gate["checkpoint"]
    if evidence.sha256(path) != gate["sha256"]:
        raise RuntimeError("Frozen checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    from .runtime import content_hash
    if content_hash(payload[checkpoint_key]) != parameter_hash:
        raise RuntimeError("Selected frozen parameters must be loaded before testing")


class EpisodeBackend:
    family = "episode_fixed_M"

    def __init__(self, config, output, event=None, deadline=None):
        self.config, self.output = deepcopy(config), Path(output)
        self.train_config, self.test_seeds = validate_settings(self.config)
        self.name = self.config.get("task", "two_link_arm")
        self.event = event or (lambda *_a, **_k: None)
        self.deadline, self.initialized, self.history = deadline, False, []

    def guard(self):
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError("Original training deadline reached")
        if any((p / "STOP").exists() for p in (self.output, self.output.parent, self.output.parent.parent)):
            raise InterruptedError("STOP requested")
        if self.initialized:
            self.learner.numerical.guard()

    def initialize(self):
        self.guard()
        if self.initialized:
            raise RuntimeError("Backend already initialized")
        torch.set_num_threads(1)
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = make_task(self.name, self.config.get("physical", {}))
        self.learner = StableEpisodeLearner(self.env, config=self.train_config, env_id=self.name,
                                           env_kwargs=self.config.get("physical", {}))
        self.initial_state = deepcopy(self.learner.actor.state_dict())
        self.initialized = True
        return dict(task=self.name, algorithm="episode_M_PPO_reproducible_v4", horizon_steps=self.env.max_steps,
                    dt=self.env.dt, reference=getattr(self.env, "reference_model_version", "versioned_v4"),
                    metric="episode-fixed; actual successor and same metric", initial_parameter_hash=module_hash(self.learner.actor),
                    training_seed_start=self.train_config.seed, training_episodes=self.train_config.episodes,
                    validation_seeds=list(self.train_config.validation_seeds), test_seeds=list(self.test_seeds))

    def collect(self, initial=False):
        self.guard()
        return dict(skipped=True, reason="fresh full episodes are collected inside each PPO update", initial=bool(initial))

    def warmup_representation(self, updates):
        if updates:
            raise ValueError("Episode PPO has no representation training stage")
        return dict(skipped=True, actual_updates=0)

    pretrain_evaluator = warmup_representation

    def train_epoch(self, epoch, representation_updates=0, evaluator_updates=0, certificate_updates=1):
        if representation_updates or evaluator_updates:
            raise ValueError("Unsupported independent update stage")
        positive_int(certificate_updates, "certificate_updates", 0)
        if self.learner.episodes + certificate_updates*self.train_config.batch_episodes > self.train_config.episodes:
            raise ValueError("Episode epoch exceeds the declared whole-group seed budget")
        rows = []
        for _ in range(certificate_updates):
            self.guard()
            batch = self.learner._collect(self.train_config.batch_episodes)
            losses = self.learner._update(*batch[:4])
            row = dict(epoch=epoch, update=self.learner.updates, **losses, episodes=batch[4],
                       parameter_hash=module_hash(self.learner.actor), physical_steps=self.learner.physical_steps)
            rows.append(row); self.history.append(row)
            self.guard(); self.event("certificate_update", task=self.name, **row)
        return dict(epoch=epoch, actual_updates=self.learner.updates, certificate_updates=len(rows), rows=rows)

    def validate(self):
        self.guard()
        result = self.learner.evaluate()
        if result["summary"]["objective"] < self.learner.best_score:
            self.learner.best_score = result["summary"]["objective"]
            self.learner.best_actor = deepcopy(self.learner.actor.state_dict())
        return dict(**result, selection_score=[-result["summary"]["objective"]],
                    active_parameter_hash=module_hash(self.learner.actor))

    def save(self, path):
        self.guard()
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        self.learner.save(temporary)
        payload = torch.load(temporary, map_location="cpu", weights_only=False)
        payload["v4_backend"] = dict(config=self.config, initial=self.initial_state, history=self.history)
        from extensions.unified_training_v3.numerical import require_finite
        require_finite(payload["v4_backend"], "episode backend checkpoint")
        torch.save(payload, temporary)
        if path.exists():
            raise FileExistsError(path)
        temporary.rename(path)
        return dict(path=str(path), updates=self.learner.updates)

    def load(self, path):
        self.guard()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        extra = payload["v4_backend"]
        if extra["config"] != self.config:
            raise ValueError("Checkpoint config mismatch")
        from extensions.unified_training_v3.numerical import require_finite
        require_finite(extra, "episode backend checkpoint")
        probe = deepcopy(self.learner.actor)
        probe.load_state_dict(extra["initial"])
        if not isinstance(extra["history"], list):
            raise ValueError("Invalid backend history")
        self.learner.restore(path)
        self.initial_state, self.history = extra["initial"], extra["history"]

    def evaluate(self, split="test", variants=("selected", "initial", "identity")):
        self.guard()
        from .runtime import content_hash
        if split == "test":
            assert_test_gate(self.output, content_hash(self.learner.actor.state_dict()))
        elif split != "validation":
            raise ValueError("Unknown split")
        chosen = deepcopy(self.learner.actor.state_dict())
        evaluation_seeds = self.test_seeds if split == "test" else self.train_config.validation_seeds
        groups = {}
        try:
            for variant in variants:
                if variant not in {"selected", "initial", "identity"}:
                    raise ValueError("Unknown variant")
                self.learner.actor.load_state_dict(self.initial_state if variant == "initial" else chosen)
                with evidence.preserve_rng():
                    groups[variant] = self.learner.evaluate(evaluation_seeds, identity=variant == "identity", trajectories=True)
                self.guard()
        finally:
            self.learner.actor.load_state_dict(chosen)
        return dict(task=self.name, split=split, variants=groups, candidate_certificate_only=True)

    def close(self):
        if self.initialized:
            self.learner.close()
            self.initialized = False
