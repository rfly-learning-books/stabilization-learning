"""Structural interfaces for collaborating on model-based stabilization.

The leading dimension represents samples. Arrays can be NumPy arrays or
framework tensors; model and controller implementations own device and
automatic differentiation. No method here supplies default dynamics or control.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar

ArrayT = TypeVar("ArrayT")


def _shape2(value: Any, name: str) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 2 or any(int(d) <= 0 for d in shape):
        raise ValueError(f"{name} must have nonempty shape (samples, dimensions)")
    return int(shape[0]), int(shape[1])


@dataclass(frozen=True)
class DiscreteTransitionBatch(Generic[ArrayT]):
    """Paired observations and *applied* actions at one fixed sampling period."""

    state: ArrayT
    applied_action: ArrayT
    next_state: ArrayT
    dt_seconds: float

    def __post_init__(self) -> None:
        state_shape = _shape2(self.state, "state")
        if _shape2(self.next_state, "next_state") != state_shape:
            raise ValueError("next_state must match state shape")
        if _shape2(self.applied_action, "applied_action")[0] != state_shape[0]:
            raise ValueError("applied_action and state must have the same sample count")
        if (isinstance(self.dt_seconds, bool) or not isinstance(self.dt_seconds, Real)
                or not math.isfinite(self.dt_seconds) or self.dt_seconds <= 0):
            raise ValueError("dt_seconds must be a positive finite scalar")


@dataclass(frozen=True)
class MechanicalTrainingBatch(Generic[ArrayT]):
    """Mechanical coordinates, derivatives and actual generalized force.

    Commanded force and timestamps may be retained separately when available.
    They are not substituted for the force that acted on the plant.
    """

    q: ArrayT
    velocity: ArrayT
    acceleration: ArrayT
    applied_generalized_force: ArrayT
    commanded_generalized_force: ArrayT | None = None
    timestamps: ArrayT | None = None

    def __post_init__(self) -> None:
        expected = _shape2(self.q, "q")
        for name in ("velocity", "acceleration", "applied_generalized_force"):
            if _shape2(getattr(self, name), name) != expected:
                raise ValueError(f"{name} must match q shape")
        if self.commanded_generalized_force is not None:
            if _shape2(self.commanded_generalized_force, "commanded_generalized_force") != expected:
                raise ValueError("commanded_generalized_force must match q shape")
        if self.timestamps is not None:
            shape = getattr(self.timestamps, "shape", None)
            if shape is None or tuple(shape) != (expected[0],):
                raise ValueError("timestamps must have shape (samples,)")


@dataclass(frozen=True)
class MechanicalReference(Generic[ArrayT]):
    q: ArrayT
    velocity: ArrayT
    acceleration: ArrayT


class DiscreteDynamicsModel(Protocol[ArrayT]):
    def predict_next(self, state: ArrayT, action: ArrayT) -> ArrayT:
        """Return the full next state, at the sampling period used by the model."""
        ...


class DiscreteCertificate(Protocol[ArrayT]):
    def value(self, state: ArrayT) -> ArrayT:
        """Return one V value per state, shape (samples,)."""
        ...


class GMMTransitionModel(DiscreteDynamicsModel[ArrayT], Protocol[ArrayT]):
    def memberships(self, state: ArrayT, action: ArrayT) -> ArrayT:
        """Return mixture responsibilities, shape (samples, components)."""
        ...


class GMMFitter(Protocol[ArrayT]):
    def fit(
        self, batch: DiscreteTransitionBatch[ArrayT], *, seed: int
    ) -> tuple[GMMTransitionModel[ArrayT], Mapping[str, Any]]:
        """Fit only from the supplied training batch; return model and diagnostics."""
        ...


class LagrangianDynamics(Protocol[ArrayT]):
    """A single learned mechanics model; M_mass is not a certificate metric."""

    def mass_matrix(self, q: ArrayT) -> ArrayT:
        """Return symmetric positive-definite learned inertia M_mass(q)."""
        ...

    def coriolis_matrix(self, q: ArrayT, velocity: ArrayT) -> ArrayT:
        """Return C(q, qdot), consistent with the same learned inertia model."""
        ...

    def potential(self, q: ArrayT) -> ArrayT:
        """Return learned potential energy."""
        ...

    def gravity(self, q: ArrayT) -> ArrayT:
        """Return the potential gradient in the declared coordinates."""
        ...

    def inverse_dynamics(self, q: ArrayT, velocity: ArrayT, acceleration: ArrayT) -> ArrayT:
        """Return M_mass qddot + C qdot + gravity; model other forces explicitly."""
        ...


class LagrangianFitter(Protocol[ArrayT]):
    def fit(
        self, batch: MechanicalTrainingBatch[ArrayT], *, seed: int
    ) -> tuple[LagrangianDynamics[ArrayT], Mapping[str, Any]]:
        """Fit the mechanical model from the supplied training batch."""
        ...


class LLearningEnergy(Protocol[ArrayT]):
    def value(
        self, q: ArrayT, velocity: ArrayT, reference: MechanicalReference[ArrayT],
        controller_state: Any, *, audit_context: Mapping[str, Any] | None = None,
    ) -> ArrayT:
        """Evaluate tracking-error energy; audit context may include offline truth."""
        ...


class LLearningController(Protocol[ArrayT]):
    """Generate generalized force and explicit per-episode controller state."""

    def reset(self, q: ArrayT) -> Any:
        """Create new controller state for an episode."""
        ...

    def control(
        self, q: ArrayT, velocity: ArrayT, reference: MechanicalReference[ArrayT],
        controller_state: Any,
    ) -> tuple[ArrayT, Any]:
        """Return requested generalized force and the updated controller state."""
        ...


class RunnerBackend(Protocol):
    """Lifecycle used by v4; implementations still require validation/registration."""

    def initialize(self) -> Mapping[str, Any]: ...
    def collect(self, initial: bool = False) -> Mapping[str, Any]: ...
    def warmup_representation(self, updates: int) -> Mapping[str, Any]: ...
    def pretrain_evaluator(self, updates: int) -> Mapping[str, Any]: ...
    def train_epoch(
        self, epoch: int, representation_updates: int = 0,
        evaluator_updates: int = 0, certificate_updates: int = 0,
    ) -> Mapping[str, Any]: ...
    def validate(self) -> Mapping[str, Any]: ...
    def save(self, path: str | Path) -> Any: ...
    def load(self, path: str | Path) -> Any: ...
    def evaluate(
        self, split: str, variants: Sequence[str] = ("selected", "initial", "identity")
    ) -> Mapping[str, Any]: ...
