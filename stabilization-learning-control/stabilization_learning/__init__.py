"""Stabilization Learning: learned features and metrics, formal task controllers."""
from .core import MetricSpec
from .metrics import CholeskyMetric, MetricActor, quadratic_certificate
from .trainer import StabilizationLearner, TrainConfig
from .envs import make, available_envs

__version__ = "0.2.0"
__all__ = ["MetricSpec", "CholeskyMetric", "MetricActor", "quadratic_certificate",
           "StabilizationLearner", "TrainConfig", "make", "available_envs"]
