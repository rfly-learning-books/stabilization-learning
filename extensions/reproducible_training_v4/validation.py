"""Validate the entire declared group before initialization or data collection."""
from copy import deepcopy
import math
import re

from stabilization_learning.unified_v1.runner import plain
from extensions.unified_training_v3 import runner as guarded

BACKENDS = {"state_model_gradient", "observer_model_gradient", "push_discrete_sac",
            "rgb_legacy", "episode_ppo", "dc_motor_gradient"}


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def seeds(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + " must be a nonempty seed list")
    result = tuple(positive_int(v, name, 0) for v in values)
    if len(set(result)) != len(result):
        raise ValueError(name + " contains duplicate seeds")
    return result


def validate_config(config):
    if config.get("schema") != "unified_training_v4":
        raise ValueError("Expected unified_training_v4 configuration; never reuse a historical run")
    cap = config.get("maximum_minutes")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("A finite positive maximum_minutes is required")
    deadline = config.get("absolute_deadline_unix")
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not math.isfinite(deadline)):
        raise ValueError("Invalid absolute deadline")
    free = config.get("minimum_free_gib", 5)
    if isinstance(free, bool) or not isinstance(free, (float, int)) or not math.isfinite(free) or free < 0:
        raise ValueError("minimum_free_gib must be finite and nonnegative")
    if config.get("deterministic", True) is not True:
        raise ValueError("v4 requires deterministic algorithms; unsupported kernels fail explicitly")
    jobs = config.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Declare the complete nonempty job group")
    seen = set()
    for job in jobs:
        name = job.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or name in seen:
            raise ValueError("Unique simple job IDs are required")
        seen.add(name)
        backend = job.get("backend")
        if backend not in BACKENDS:
            raise ValueError("Unsupported v4 backend: " + str(backend))
        cfg, schedule = job.get("config", {}), job.get("schedule", {})
        if not isinstance(cfg, dict) or not isinstance(schedule, dict):
            raise ValueError("config and schedule must be mappings")
        positive_int(cfg.get("seed", 81), "seed", 0)
        if cfg.get("cpu_threads", 1) != 1 or type(cfg.get("cpu_threads", 1)) is not int:
            raise ValueError("The audited reproducibility contract requires cpu_threads=1")
        for key in ("warmup_representation", "pretrain_evaluator", "epochs", "representation_updates",
                    "evaluator_updates", "certificate_updates"):
            positive_int(schedule.get(key, 0), key, 0)
        positive_int(schedule.get("epochs", 0), "epochs")
        positive_int(schedule.get("certificate_updates", 0), "certificate_updates")
        variants = job.get("test_variants", ["selected", "initial", "identity"])
        if not isinstance(variants, list) or not all(isinstance(v, str) for v in variants) or \
                "selected" not in variants or len(set(variants)) != len(variants):
            raise ValueError("Unique test variants including selected are required")
        if backend in {"episode_ppo", "dc_motor_gradient"}:
            if any(schedule.get(k, 0) for k in ("warmup_representation", "pretrain_evaluator",
                                               "representation_updates", "evaluator_updates")):
                raise ValueError("This backend has no independent representation or evaluator updates")
            if set(variants) - {"selected", "initial", "identity"}:
                raise ValueError("Unsupported independent task variant")
            if backend == "episode_ppo":
                from .independent_backend import validate_settings
            else:
                from .dc_motor import validate_settings
            validate_settings(cfg, schedule)
        else:
            if backend == "state_model_gradient":
                from dataclasses import fields
                from stabilization_learning.algos.pytorch.state_metric_full_horizon import FullHorizonConfig
                allowed = {f.name for f in fields(FullHorizonConfig)} | {"task", "physical", "total_updates",
                    "batch_size", "hidden", "cpu_threads", "validation_count", "test_count"}
            elif backend == "observer_model_gradient":
                allowed = {"task", "seed", "cpu_threads", "hidden", "metric_radius", "learning_rate", "scenarios",
                    "batch_size", "gradient_clip", "training_index_start", "validation_index_start", "test_index_start",
                    "validation_count", "test_count", "validation_scenarios", "test_scenarios"}
            elif backend == "rgb_legacy":
                allowed = {"device", "seed", "batch_size", "probe_batch_size", "replay_capacity", "cpu_threads",
                    "replay_cached_episodes", "train_scene_count", "validation_cases", "test_cases", "evaluation_steps",
                    "initial_collection_steps", "collection_steps", "parallel_envs", "evaluation_plants", "evaluation_batch",
                    "include_historical_reference"}
            else:
                from dataclasses import fields
                from extensions.unified_training_v1.backends.push import PushConfig
                allowed = {f.name for f in fields(PushConfig)}
            if set(cfg)-allowed:
                raise ValueError("Unknown backend configuration: " + repr(sorted(set(cfg)-allowed)))
            # Retain all existing backend-specific bounds, using a local copy.
            compatible = dict(schema="unified_training_v1", maximum_minutes=cap, jobs=[deepcopy(job)])
            guarded.validate_training_config(compatible)
    plain(config)
    return config

