"""Physical RGB rooms for the book-protocol training platform.

The room owns rendering, plant state, collision detection and termination.
Feature networks, the derivative Critic, M and optimizers belong to the
algorithm library.  The book collector uses the same room constructor as the
completed reconstructed-environment experiment.
"""
from __future__ import annotations

from ...tasks_v2.reconstructed_rgb import ReconstructedRGBEnv


# Public Gymnasium interface for applications composing their own runner.
BookRGBPhysicalEnv = ReconstructedRGBEnv


def make_book_env(cfg, scene, *, plant="P0", max_steps=120):
    """Build an RGB velocity environment without constructing a learner."""
    return BookRGBPhysicalEnv(cfg, scene, plant=plant, max_steps=max_steps)


def create_book_room(cfg, scene):
    """Create the audited physical room used by the original episode sampler.

    Keeping the room-level API allows that sampler to retain its exact order
    of observation, noise, auxiliary renders and termination checks.  The
    caller must close the returned room, including after an exception.
    """
    from train_goal_fixed import new_instance

    return new_instance(cfg, scene)


__all__ = ["BookRGBPhysicalEnv", "make_book_env", "create_book_room"]
