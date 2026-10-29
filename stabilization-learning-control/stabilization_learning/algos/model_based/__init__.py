"""Contracts for model-based stabilization research.

These are interfaces, not complete GMM-D or L-learning algorithms. They
support NumPy or tensor-backed implementations without coercing or detaching
the researcher's arrays.
"""
from .contracts import (
    DiscreteCertificate,
    DiscreteDynamicsModel,
    DiscreteTransitionBatch,
    GMMFitter,
    GMMTransitionModel,
    LagrangianDynamics,
    LagrangianFitter,
    LLearningEnergy,
    LLearningController,
    MechanicalTrainingBatch,
    MechanicalReference,
    RunnerBackend,
)
from .discrete import observed_discrete_change, predicted_discrete_change

__all__ = [
    "DiscreteCertificate", "DiscreteDynamicsModel", "DiscreteTransitionBatch",
    "GMMFitter", "GMMTransitionModel", "LagrangianDynamics", "LagrangianFitter", "LLearningEnergy",
    "LLearningController", "MechanicalTrainingBatch", "MechanicalReference",
    "RunnerBackend", "observed_discrete_change", "predicted_discrete_change",
]
