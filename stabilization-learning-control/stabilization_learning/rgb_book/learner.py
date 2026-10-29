"""Retain the audited derivative objective and expose model-only snapshots.

The physical environment is supplied by the caller. No environment is created
and no physical state is read by this module.
"""
from copy import deepcopy
from pathlib import Path
import sys

MODELS = ("ae", "shadow", "metric", "critics")
OPTIMIZERS = ("aeopt", "shadowopt", "criticopt", "mopt")


def make_learner(config, initialization, device):
    runtime = Path(__file__).resolve().parents[3] / "third_party/visual_original/runtime"
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from derivative_critic_core import DerivativeLearner
    learner = DerivativeLearner(config, initialization, device)
    learner.restore(initialization["learner"])
    learner.initial.load_state_dict(initialization["learner"]["ae"], strict=True)
    from extensions.reproducible_training_v4.deterministic_rgb import install_deterministic_blur
    for name in ("ae", "shadow", "initial"):
        install_deterministic_blur(getattr(learner, name))
    return learner


def inference_state(learner):
    return {name: deepcopy(getattr(learner, name).state_dict()) for name in MODELS} | {
        "counts": deepcopy(learner.counts), "version": learner.version,
        "anchors": deepcopy(learner.anchors)}


def restore_models(learner, state):
    for name in MODELS:
        getattr(learner, name).load_state_dict(state[name], strict=True)
    learner.counts = deepcopy(state["counts"])
    learner.version = state["version"]
    learner.anchors = deepcopy(state["anchors"])
    learner.clear_cache()
    learner.inputs.clear()
