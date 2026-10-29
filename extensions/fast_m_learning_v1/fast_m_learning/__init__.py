"""Isolated fixed-M tensor sampling components; no learner is enabled.

Importing this package creates no environments, policies or optimizers.
Engineering coverage is recorded separately from the source implementation.
"""

from .tensor_kernel import TaskTensorSpec, TensorTaskKernel
from .batch_rollout import (
    BatchState,
    FullHorizonRunner,
    RunnerSnapshot,
    StepProposal,
    TensorAudit,
)

__all__ = [
    "TaskTensorSpec", "TensorTaskKernel", "BatchState", "FullHorizonRunner",
    "RunnerSnapshot", "StepProposal", "TensorAudit",
]
