"""Reproducible initial-state and task sampling for certificate experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    position_min: float = 0.02
    position_max: float = 0.08
    angle_min_rad: float = 0.0
    angle_max_rad: float = np.deg2rad(60.0)
    workspace_limit: float = 0.232
    pusher_radius: float = 0.018
    clearance: float = 0.002
    max_attempts: int = 10000

    def __post_init__(self):
        if min(self.position_min, self.position_max, self.workspace_limit, self.pusher_radius, self.clearance) <= 0:
            raise ValueError("sampling lengths must be positive")
        if self.position_min > self.position_max or self.angle_min_rad > self.angle_max_rad:
            raise ValueError("invalid task ranges")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")


def sample_free_pusher_positions(polygon, count, rng, config: SamplingConfig | None = None):
    """Sample non-overlapping pusher centers in the declared square workspace."""

    from shapely.geometry import Point

    cfg = config or SamplingConfig()
    if int(count) < 1:
        raise ValueError("count must be positive")
    obstacle = polygon.buffer(cfg.pusher_radius + cfg.clearance, join_style=2)
    values = []
    attempts = 0
    while len(values) < int(count) and attempts < cfg.max_attempts:
        attempts += 1
        candidate = rng.uniform(-cfg.workspace_limit, cfg.workspace_limit, size=2)
        if obstacle.contains(Point(candidate)):
            continue
        values.append(candidate)
    if len(values) < int(count):
        raise RuntimeError("could not sample enough collision-free pusher positions")
    return np.asarray(values, dtype=np.float64)


def sample_task(object_q, rng, config: SamplingConfig | None = None, orientation_period=2 * np.pi):
    """Sample a target pose without leaking a shape identity into the task."""

    cfg = config or SamplingConfig()
    object_q = np.asarray(object_q, dtype=np.float64)
    if object_q.shape != (3,) or not np.isfinite(object_q).all():
        raise ValueError("object_q must be finite shape (3,)")
    direction = float(rng.uniform(-np.pi, np.pi))
    distance = float(rng.uniform(cfg.position_min, cfg.position_max))
    target = object_q.copy()
    target[:2] += distance * np.array([math.cos(direction), math.sin(direction)])
    target[2] += float(rng.uniform(cfg.angle_min_rad, cfg.angle_max_rad)) * (1.0 if rng.random() < .5 else -1.0)
    return target


def make_seed_bank(count, seed=20260911):
    """Return independent deterministic seeds shared by every Q candidate."""

    if int(count) < 1:
        raise ValueError("seed-bank count must be positive")
    sequence = np.random.SeedSequence(int(seed))
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(int(count))]

