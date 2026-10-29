"""Composable learning APIs, separate from physical environment factories.

Pure model-gradient learners accept a task and a network supplied by the caller.
Historical RGB/Push experiments additionally require the audited research bundle;
their staged backend is exposed explicitly rather than loaded on package import.
"""
from .model_gradient import (
    ModelGradientLearner, StateModelGradientLearner, ObserverModelGradientLearner,
)


def create_training_backend(name, config, output, *, event=None, deadline=None):
    """Build a staged research backend without starting training.

    Initialize/collect/update are explicit operations. RGB and Push require the
    local audited assets and extensions; this function does not download them.
    For a full scientifically frozen experiment use ``run_experiment``.
    """
    from stabilization_learning.unified_v1.registry import make_backend
    return make_backend(name, config, output, event=event, deadline=deadline)


def run_experiment(config, output, *, project):
    """Run the declared complete group, with one budget and a freeze barrier."""
    from stabilization_learning.unified_v1.runner import run_suite
    return run_suite(config, output, project=project)


__all__ = ["ModelGradientLearner", "StateModelGradientLearner",
           "ObserverModelGradientLearner", "create_training_backend", "run_experiment"]
