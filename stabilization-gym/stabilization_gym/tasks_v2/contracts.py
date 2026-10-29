"""Small shared contracts, with no optimizer or training-loop ownership.

Public observations/references must contain only declared deployment information.
Training truth is deliberately absent from TaskContext and construct_action.
Python type checking cannot establish information provenance; adapters must test it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

from stabilization_gym.interfaces.certificate_v2 import VersionBinding


@dataclass(frozen=True)
class Capabilities:
    action_gradient: bool = False
    feature_gradient: bool = False
    interaction_parameter_gradient: bool = False
    candidate_probabilities: bool = False
    prediction_gradient: bool = False
    physical_step_gradient: bool = False
    action_kind: str = "physical"
    note: str = "Candidate certificate only; capabilities are not stability claims."


@dataclass
class TaskContext:
    error: torch.Tensor | None
    metric_context: Any = None
    model_context: Mapping[str, Any] = field(default_factory=dict)
    memory: Mapping[str, Any] = field(default_factory=dict)
    binding: VersionBinding | None = None


@dataclass
class ActionPlan:
    action: Any
    memory: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    distribution: Any = None


@runtime_checkable
class PhysicalEnv(Protocol):
    """Gymnasium convention: second step result is REWARD, not cost.

    Nonnegative physical cost (when defined) is a separately named info field.
    A pure estimator correction is not an action on this physical object.
    """
    def reset(self, *, seed: int | None = None, options: Mapping | None = None): ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]: ...


@runtime_checkable
class TaskAdapter(Protocol):
    capabilities: Capabilities

    def make_context(self, observation: Any, reference: Any, *, features: Any = None,
                     model: Any = None, memory: Mapping | None = None,
                     binding: VersionBinding | None = None) -> TaskContext: ...

    def construct_action(self, context: TaskContext, certificate: Any) -> ActionPlan: ...
