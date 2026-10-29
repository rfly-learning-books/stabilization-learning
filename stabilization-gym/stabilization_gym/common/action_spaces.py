"""Physical spaces whose samples obey mixed/disc constraints exactly.

These spaces describe physical interfaces, not the raw-M policy interface.
EpisodeMetricEnv exposes the ordinary Box expected by learning backends.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class ModeSpeedSpace(gym.Space):
    """Flat [integer mode, continuous nonnegative speed] physical action."""
    def __init__(self, modes=12, speed_cap=.6, *, seed=None):
        if modes < 1 or int(modes) != modes or not np.isfinite(speed_cap) or speed_cap <= 0:
            raise ValueError("Require a positive number of modes and finite positive speed cap")
        super().__init__((2,), np.float64, seed=seed)
        self.modes, self.speed_cap = int(modes), float(speed_cap)
        self.low = np.array([0., 0.], dtype=self.dtype)
        self.high = np.array([self.modes-1., self.speed_cap], dtype=self.dtype)

    @property
    def is_np_flattenable(self):
        return False  # A generic Box would lose the discrete-mode constraint.

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None:
            raise ValueError("ModeSpeedSpace does not support sampling masks")
        return np.array([self.np_random.integers(self.modes),
                         self.np_random.uniform(0., self.speed_cap)], dtype=self.dtype)

    def contains(self, x):
        try:
            if isinstance(x, dict):
                speed = np.asarray(x["u"]).reshape(-1)
                if speed.shape != (1,):
                    return False
                x = [x["mode"], speed[0]]
            a = np.asarray(x, dtype=float)
            if a.shape != self.shape or not np.isfinite(a).all():
                return False
            return bool(abs(a[0]-round(float(a[0]))) <= 1e-6 and
                        0 <= int(a[0]) < self.modes and 0 <= a[1] <= self.speed_cap)
        except (TypeError, ValueError, KeyError, OverflowError):
            return False

    def to_jsonable(self, sample_n):
        return [np.asarray(sample).tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return [np.asarray(sample, dtype=self.dtype) for sample in sample_n]

    def __repr__(self):
        return f"ModeSpeedSpace(modes={self.modes}, speed=[0, {self.speed_cap}])"


class BodyVelocitySpace(gym.Space):
    """A translation-speed ball times a bounded yaw-rate interval.

    Translation is sampled uniformly in the 3-D ball. The 1e-5 numerical
    acceptance tolerance mirrors the physical step's existing guard.
    """
    def __init__(self, translation_cap=.4, yaw_cap=.35, *, seed=None):
        if not np.isfinite([translation_cap, yaw_cap]).all() or min(translation_cap, yaw_cap) <= 0:
            raise ValueError("Require finite positive translation and yaw speed caps")
        super().__init__((4,), np.float64, seed=seed)
        self.translation_cap, self.yaw_cap = float(translation_cap), float(yaw_cap)
        self.high = np.array([translation_cap]*3 + [yaw_cap], dtype=self.dtype)
        self.low = -self.high

    @property
    def is_np_flattenable(self):
        return False  # Componentwise Box bounds cannot describe the speed ball.

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None:
            raise ValueError("BodyVelocitySpace does not support sampling masks")
        direction = self.np_random.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm == 0:
            direction, norm = np.array([1., 0., 0.]), 1.
        radius = self.translation_cap * self.np_random.random()**(1./3.)
        return np.r_[direction*(radius/norm), self.np_random.uniform(-self.yaw_cap, self.yaw_cap)]

    def contains(self, x):
        try:
            a = np.asarray(x, dtype=float)
            return bool(a.shape == self.shape and np.isfinite(a).all() and
                        np.linalg.norm(a[:3]) <= self.translation_cap+1e-5 and
                        abs(a[3]) <= self.yaw_cap+1e-5)
        except (TypeError, ValueError, OverflowError):
            return False

    def to_jsonable(self, sample_n):
        return [np.asarray(sample).tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return [np.asarray(sample, dtype=self.dtype) for sample in sample_n]

    def __repr__(self):
        return f"BodyVelocitySpace(translation_norm<={self.translation_cap}, abs_yaw<={self.yaw_cap})"
