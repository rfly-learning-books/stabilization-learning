"""Episode-held-M execution for the seven original full-horizon tasks.

This module is a source-only implementation until the independent execution
gates are approved.  It neither constructs Gym environments nor creates an
actor, representation, optimizer, sampler, or training entry point.  The
supplied TensorTaskKernel owns pure mathematical queries; this runner owns
lane-local episode bookkeeping and the sole optional tensor audit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import math
from pathlib import Path
import struct
import threading
from typing import Any
import uuid

import numpy as np
import torch


SCHEMA_VERSION = "seven_task_batch_rollout_source_v1"
GRADIENT_MODE = "score_function_detached"
_SUPPORTED_ENV_IDS = {
    "coupled_linear", "point_mass", "pendulum", "cartpole", "two_link_arm",
    "unicycle_tracking", "multirotor_tracking",
}

_FLOAT_TRANSITION_FIELDS = (
    "state_before", "state_after", "error_before", "error_after",
    "action_nominal", "applied_action", "V_before", "V_after",
    "physical_error_before", "physical_error_after", "cost",
    "decay_residual", "normalized_decay_residual", "penalty",
)
_BOOL_TRANSITION_FIELDS = (
    "V_rise", "in_goal_before", "in_goal_after", "failure_before",
    "failure_after",
)


def _clone(value: Any) -> Any:
    """Own every mutable leaf, including the caller's serialized RNG payload."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not a supported snapshot format")
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Snapshot mappings must have string keys")
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(f"Unsupported snapshot leaf: {type(value).__name__}")


def _digest(value: Any) -> str:
    """Typed content fingerprint, for ownership checks rather than security."""
    digest = hashlib.sha256()

    def put(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            put(("tensor", str(tensor.dtype), tuple(tensor.shape)))
            put(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("Object arrays cannot be fingerprinted")
            put(("ndarray", item.dtype.str, tuple(item.shape)))
            put(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            put(item.item())
        elif isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("Fingerprint mappings must have string keys")
            digest.update(b"mapping{")
            for key in sorted(item):
                put(key)
                put(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"tuple[" if isinstance(item, tuple) else b"list[")
            for child in item:
                put(child)
            digest.update(b"]")
        elif item is None:
            digest.update(b"none;")
        elif isinstance(item, bool):
            digest.update(b"true;" if item else b"false;")
        elif isinstance(item, int):
            digest.update(b"int:" + str(item).encode("ascii") + b";")
        elif isinstance(item, float):
            digest.update(b"float:" + struct.pack("!d", item))
        elif isinstance(item, str):
            data = item.encode("utf-8")
            digest.update(b"str:" + str(len(data)).encode("ascii") + b":" + data)
        elif isinstance(item, bytes):
            digest.update(b"bytes:" + str(len(item)).encode("ascii") + b":" + item)
        else:
            raise TypeError(f"Unsupported fingerprint leaf: {type(item).__name__}")

    put(value)
    return digest.hexdigest()


def _hash_string(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be an explicit SHA256 string")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{name} must be hexadecimal")
    return value.lower()


def _tensor(value: Any, shape: tuple[int, ...], dtype: torch.dtype,
            device: torch.device, name: str, *, finite: bool = True,
            copy: bool = True) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor; implicit conversions are not allowed")
    if value.shape != shape or value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} requires shape={shape}, dtype={dtype}, device={device}")
    if finite and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}; no lane can be committed")
    return value.detach().clone() if copy else value


@dataclass(frozen=True)
class BatchState:
    """An owned snapshot of all lane bindings and committed execution state.

    Public copies may be modified by their owner without changing the runner.
    Phase/reference coordinates are in ``state`` and ``initial_state``.  The
    seven-task kernel has no additional controller memory or step RNG.
    NaN is used only for an absent hold/confirmation clock, never for physics.
    """

    episode_ids: tuple[str, ...]
    seeds: tuple[int | None, ...]
    initial_scales: torch.Tensor
    actor_versions: tuple[str, ...]
    phi_versions: tuple[str, ...]
    rng_protocol: str
    rng_state_by_episode: Mapping[str, Any]
    initial_state: torch.Tensor
    state: torch.Tensor
    initial_failure: torch.Tensor
    metric: torch.Tensor
    raw: torch.Tensor
    raw_logp: torch.Tensor
    context: torch.Tensor
    error: torch.Tensor
    physical_error: torch.Tensor
    in_goal: torch.Tensor
    observed_failure: torch.Tensor
    physical_steps: torch.Tensor
    active: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    in_goal_since: torch.Tensor
    first_hold_confirmation: torch.Tensor
    accumulated_cost: torch.Tensor
    certificate_penalty: torch.Tensor
    V_rises: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.episode_ids)

    def clone(self) -> "BatchState":
        return BatchState(**{field.name: _clone(getattr(self, field.name))
                             for field in fields(self)})


@dataclass(frozen=True)
class TensorAudit:
    """One tensor ledger: T+1 endpoints, T transitions, explicit valid masks.

    Per-lane physical step indexes the first dimension.  Failed-lane padding
    stays invalid; it never counts as a transition.  No Gym trace is retained.
    """

    records: Mapping[str, torch.Tensor]

    def clone(self) -> "TensorAudit":
        return TensorAudit(_clone(self.records))


@dataclass(frozen=True)
class StepProposal:
    runner_id: str
    generation: int
    ticket: int
    lane_indices: torch.Tensor
    transition: Mapping[str, Any]
    checksum: str

    def payload(self) -> dict[str, Any]:
        return dict(runner_id=self.runner_id, generation=self.generation,
                    ticket=self.ticket, lane_indices=self.lane_indices,
                    transition=self.transition)

    def clone(self) -> "StepProposal":
        return StepProposal(self.runner_id, self.generation, self.ticket,
                            _clone(self.lane_indices), _clone(self.transition), self.checksum)


@dataclass(frozen=True)
class RunnerSnapshot:
    identity: Mapping[str, Any]
    generation: int
    ticket_counter: int
    state: BatchState
    audit: TensorAudit | None
    checksum: str

    def payload(self) -> dict[str, Any]:
        return dict(identity=self.identity, generation=self.generation,
                    ticket_counter=self.ticket_counter,
                    state={field.name: getattr(self.state, field.name)
                           for field in fields(self.state)},
                    audit=None if self.audit is None else self.audit.records)


class FullHorizonRunner:
    """Single-owner, transactional, detached held-M physical rollout.

    ``kernel`` must be the pure TensorTaskKernel for one old-seven setting.
    Unsupported gradients fail explicitly; no actor/Phi changes occur here.
    Source/config manifests are required identities, not a claim that an
    external source or raw-to-M provenance audit has already been performed.
    """

    def __init__(self, kernel: Any, *, source_manifest: Mapping[str, str],
                 config_sha256: str, protocol_sha256: str,
                 certificate_weight: float = 0.0, failure_error: float = 25.0,
                 dense_audit: bool = False,
                 gradient_mode: str = GRADIENT_MODE) -> None:
        if gradient_mode != GRADIENT_MODE:
            raise NotImplementedError("Only score-function detached execution is implemented")
        if kernel.dtype != torch.float64 or kernel.device != torch.device("cpu"):
            raise NotImplementedError("First version is CPU float64 physics only")
        if not source_manifest:
            raise ValueError("An explicit frozen source manifest is required")
        sources = {str(path): _hash_string(sha, "source SHA256")
                   for path, sha in source_manifest.items()}
        spec = kernel.spec
        if spec.env_id not in _SUPPORTED_ENV_IDS:
            raise NotImplementedError("Only the seven original held-M physical task classes are supported")
        for path, sha in spec.source_sha256:
            if sources.get(str(path)) != str(sha).lower():
                raise ValueError("Source manifest must cover the kernel's frozen source identities")
        runner_path = Path(__file__).resolve()
        runner_sha = hashlib.sha256(runner_path.read_bytes()).hexdigest()
        if sources.get(str(runner_path)) != runner_sha:
            raise ValueError("Source manifest must also cover this exact runner implementation")
        if not math.isfinite(spec.decay) or spec.decay < 0 or kernel.decay != spec.decay:
            raise ValueError("Decay must be part of the immutable kernel specification")
        if not math.isfinite(certificate_weight) or certificate_weight < 0:
            raise ValueError("certificate_weight must be finite and nonnegative")
        if not math.isfinite(failure_error) or failure_error <= 0:
            raise ValueError("failure_error must be finite and positive")
        if spec.max_steps < 1 or not math.isfinite(spec.dt) or spec.dt <= 0:
            raise ValueError("Invalid physical horizon")
        if spec.required_hold_steps < 1 or not math.isclose(
                spec.required_hold_seconds, spec.required_hold_steps * spec.dt,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Hold clock must be derived from the declared physical dt")
        if spec.max_steps < spec.required_hold_steps + 1 and not spec.moving_reference:
            raise ValueError("Static full horizon is too short")
        if spec.moving_reference and (
                spec.horizon_seconds + 1e-10 < 2.0 * spec.reference_period or
                spec.required_hold_seconds + 1e-10 < spec.reference_period):
            raise ValueError("Moving reference needs two periods and one-period final hold")
        if not math.isclose(spec.horizon_seconds, spec.max_steps * spec.dt,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Inconsistent horizon seconds")
        self._kernel = kernel
        self._spec = spec
        self._dtype, self._device = kernel.dtype, kernel.device
        self._certificate_weight = float(certificate_weight)
        self._failure_error = float(failure_error)
        self._dense_audit = bool(dense_audit)
        self._identity = dict(
            schema_version=SCHEMA_VERSION, kernel_signature=spec.signature,
            runner_source_path=str(runner_path), runner_source_sha256=runner_sha,
            source_manifest=sources,
            config_sha256=_hash_string(config_sha256, "config_sha256"),
            protocol_sha256=_hash_string(protocol_sha256, "protocol_sha256"),
            gradient_mode=gradient_mode, certificate_weight=float(certificate_weight),
            failure_error=float(failure_error), dense_audit=bool(dense_audit),
            dt=float(spec.dt), max_steps=int(spec.max_steps),
            required_hold_seconds=float(spec.required_hold_seconds),
            decay=float(spec.decay),
        )
        self._lock = threading.RLock()
        # OS transaction identity does not consume Python/NumPy/Torch sampler RNG.
        self._runner_id = uuid.uuid4().hex
        self._generation = 0
        self._ticket_counter = 0
        self._state: BatchState | None = None
        self._audit: TensorAudit | None = None
        self._pending: StepProposal | None = None

    def _check_kernel(self) -> None:
        if (self._kernel.spec.signature != self._identity["kernel_signature"] or
                self._kernel.decay != self._identity["decay"]):
            raise RuntimeError("Kernel identity changed inside the runner")

    def _require_state(self) -> BatchState:
        if self._state is None:
            raise RuntimeError("start or restore is required")
        return self._state

    def _metric_binding(self, value: torch.Tensor, batch: int) -> torch.Tensor:
        checked = self._kernel.validate_metric(value, batch)
        # A tolerance on the spectral floor must never relax strict SPD.
        if bool((torch.linalg.cholesky_ex(checked).info != 0).any()):
            raise ValueError("Executed M must remain strictly positive definite")
        return checked

    @property
    def state(self) -> BatchState:
        with self._lock:
            return self._require_state().clone()

    @property
    def identity(self) -> dict[str, Any]:
        with self._lock:
            return _clone(self._identity)

    @property
    def complete(self) -> bool:
        with self._lock:
            return not bool(self._require_state().active.any())

    def start(self, initial_state: torch.Tensor, metric: torch.Tensor, *,
              episode_ids: Sequence[str], seeds: Sequence[int | None],
              initial_scales: torch.Tensor, actor_versions: Sequence[str],
              phi_versions: Sequence[str], raw: torch.Tensor,
              raw_logp: torch.Tensor, context: torch.Tensor,
              rng_protocol: str, rng_state_by_episode: Mapping[str, Any]) -> BatchState:
        """Bind caller-sampled decisions; this method samples nothing.

        Initial diagnostic failure is retained, but does not deactivate a lane:
        FullHorizonTask checks termination after its first physical successor.
        A completed runner can start a new whole batch; lane replacement during
        an active batch is deliberately unsupported.
        """
        with self._lock:
            self._check_kernel()
            if self._pending is not None or (self._state is not None and bool(self._state.active.any())):
                raise RuntimeError("Cannot replace an active or pending batch")
            ids = tuple(episode_ids)
            batch = len(ids)
            if batch < 1 or len(set(ids)) != batch or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError("episode_ids must be unique nonempty strings")
            seed_tuple = tuple(seeds)
            if len(seed_tuple) != batch or any(
                    seed is not None and (isinstance(seed, bool) or not isinstance(seed, int))
                    for seed in seed_tuple):
                raise ValueError("One explicit integer/None seed is required per lane")
            versions = []
            for name, values in (("actor", actor_versions), ("Phi", phi_versions)):
                values = tuple(values)
                if len(values) != batch or any(not isinstance(v, str) or not v for v in values):
                    raise ValueError(f"One nonempty {name} version is required per lane")
                if len(set(values)) != 1:
                    raise ValueError(f"A batch must bind one frozen old-policy {name} version")
                versions.append(values)
            if not isinstance(rng_protocol, str) or not rng_protocol:
                raise ValueError("Declare the external sampler RNG ownership protocol")
            if set(rng_state_by_episode) != set(ids):
                raise ValueError("Serialized external RNG binding must cover exactly these episodes")
            spec = self._spec
            initial = _tensor(initial_state, (batch, spec.state_dim), self._dtype,
                              self._device, "initial_state")
            bound_metric = _tensor(metric, (batch, spec.error_dim, spec.error_dim),
                                   self._dtype, self._device, "metric")
            # The old Cholesky action contract includes floor/trace/ceiling,
            # not only SPD.  Pure structural checking does not run physics.
            bound_metric = self._metric_binding(bound_metric, batch)
            scales = _tensor(initial_scales, (batch,), self._dtype, self._device, "initial_scales")
            if bool((scales <= 0).any()):
                raise ValueError("Initial scales must be positive provenance fields")
            bound_raw = _tensor(raw, (batch, spec.raw_dim), torch.float32, self._device, "raw")
            bound_logp = _tensor(raw_logp, (batch,), torch.float32, self._device, "raw_logp")
            bound_context = _tensor(context, (batch, spec.metric_context_dim),
                                     torch.float32, self._device, "context")
            if not torch.equal(bound_context, self._kernel.context(batch)):
                raise ValueError("Context differs from this frozen shared-goal task setting")
            rng = _clone(rng_state_by_episode)
            with torch.no_grad():
                measurement = self._kernel.measure(initial.clone())
            error, physical, good, initial_failure = self._measurement(measurement, batch)
            zeros = torch.zeros(batch, dtype=self._dtype, device=self._device)
            counters = torch.zeros(batch, dtype=torch.int64, device=self._device)
            flags = torch.zeros(batch, dtype=torch.bool, device=self._device)
            missing = torch.full_like(zeros, float("nan"))
            state = BatchState(
                ids, seed_tuple, scales, versions[0], versions[1], rng_protocol, rng,
                initial.clone(), initial.clone(), initial_failure.clone(), bound_metric,
                bound_raw, bound_logp, bound_context, error, physical, good,
                initial_failure.clone(), counters.clone(), ~flags, flags.clone(),
                flags.clone(), flags.clone(), flags.clone(),
                torch.where(good, zeros, missing), missing.clone(), zeros.clone(),
                zeros.clone(), counters.clone(),
            )
            self._validate_state(state)
            audit = self._new_audit(state) if self._dense_audit else None
            self._state, self._audit = state, audit
            self._generation += 1
            self._runner_id = uuid.uuid4().hex
            return state.clone()

    def _measurement(self, data: Mapping[str, Any], batch: int) -> tuple[torch.Tensor, ...]:
        return (
            _tensor(data["error"], (batch, self._spec.error_dim), self._dtype, self._device, "error"),
            _tensor(data["physical_error"], (batch,), self._dtype, self._device, "physical_error"),
            _tensor(data["in_goal"], (batch,), torch.bool, self._device, "in_goal"),
            _tensor(data["failure"], (batch,), torch.bool, self._device, "failure"),
        )

    def _indices(self, lane_indices: torch.Tensor | None) -> torch.Tensor:
        state = self._require_state()
        if lane_indices is None:
            indices = torch.nonzero(state.active, as_tuple=False).flatten()
        else:
            if (not isinstance(lane_indices, torch.Tensor) or lane_indices.ndim != 1 or
                    lane_indices.dtype != torch.int64 or lane_indices.device != self._device):
                raise ValueError("lane_indices requires a one-dimensional CPU int64 tensor")
            indices = lane_indices.detach().clone()
        if indices.numel() < 1:
            raise RuntimeError("No active physical lanes were requested")
        if (bool((indices < 0).any()) or bool((indices >= state.batch_size).any()) or
                indices.unique().numel() != indices.numel()):
            raise ValueError("Lane indexes must be distinct and within the batch")
        if not bool(state.active[indices].all()):
            raise RuntimeError("Inactive lanes cannot be sent to the physical model")
        return indices

    def _transition(self, data: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
        spec, batch = self._spec, indices.numel()
        shape_by_name = dict(
            state_before=(batch, spec.state_dim), state_after=(batch, spec.state_dim),
            error_before=(batch, spec.error_dim), error_after=(batch, spec.error_dim),
            action_nominal=(batch, spec.action_dim), applied_action=(batch, spec.action_dim),
        )
        result = {}
        for name in _FLOAT_TRANSITION_FIELDS:
            result[name] = _tensor(data[name], shape_by_name.get(name, (batch,)),
                                   self._dtype, self._device, name)
        for name in _BOOL_TRANSITION_FIELDS:
            result[name] = _tensor(data[name], (batch,), torch.bool, self._device, name)
        if data["kernel_signature"] != spec.signature or data["dt"] != spec.dt:
            raise ValueError("Transition belongs to a different task protocol")
        result.update(dt=float(data["dt"]), kernel_signature=data["kernel_signature"])
        state = self._require_state()
        if not torch.equal(result["state_before"], state.state[indices]):
            raise ValueError("Transition does not start at the committed lane states")
        if not torch.equal(result["error_before"], state.error[indices]):
            raise ValueError("Error cache disagrees with the pure kernel")
        if not torch.equal(result["in_goal_before"], state.in_goal[indices]):
            raise ValueError("Goal cache disagrees with the pure kernel")
        if not torch.equal(result["failure_before"], state.observed_failure[indices]):
            raise ValueError("Failure cache disagrees with the pure kernel")
        if not torch.equal(result["physical_error_before"], state.physical_error[indices]):
            raise ValueError("Physical-cost cache disagrees with the pure kernel")
        if bool((result["physical_error_before"] < 0).any()) or bool(
                (result["physical_error_after"] < 0).any()):
            raise ValueError("Physical squared-error cost cannot be negative")
        if bool((result["penalty"] < 0).any()):
            raise ValueError("Certificate penalty cannot be negative")
        return result

    def propose(self, lane_indices: torch.Tensor | None = None) -> StepProposal:
        """Pure physical query; repeating the same pending query advances nothing.

        The model receives only gathered active lanes.  Failure/finished lanes
        are never evaluated and subsequently masked.  All outputs are checked
        before a proposal exists; exceptions leave every lane untouched.
        """
        with self._lock:
            self._check_kernel()
            indices = self._indices(lane_indices)
            if self._pending is not None:
                if not torch.equal(indices, self._pending.lane_indices):
                    raise RuntimeError("Commit or cancel the pending proposal before querying other lanes")
                return self._pending.clone()
            state = self._require_state()
            with torch.no_grad():
                data = self._kernel.transition(state.state[indices].clone(),
                                               state.metric[indices].clone())
            transition = self._transition(data, indices)
            self._ticket_counter += 1
            payload = dict(runner_id=self._runner_id, generation=self._generation,
                           ticket=self._ticket_counter, lane_indices=indices,
                           transition=transition)
            self._pending = StepProposal(**payload, checksum=_digest(payload))
            return self._pending.clone()

    def _verified_proposal(self, proposal: StepProposal) -> StepProposal:
        if not isinstance(proposal, StepProposal) or self._pending is None:
            raise RuntimeError("There is no matching uncommitted proposal")
        pending = self._pending
        if (proposal.runner_id != self._runner_id or proposal.generation != self._generation or
                proposal.ticket != pending.ticket or proposal.checksum != pending.checksum or
                _digest(proposal.payload()) != pending.checksum):
            raise RuntimeError("Stale, foreign, modified, or already committed proposal")
        return pending

    def cancel(self, proposal: StepProposal) -> None:
        with self._lock:
            self._verified_proposal(proposal)
            self._pending = None

    def commit(self, proposal: StepProposal) -> BatchState:
        """Verify all selected lanes, then atomically install one new batch state."""
        with self._lock:
            self._check_kernel()
            pending = self._verified_proposal(proposal)
            old = self._require_state()
            indices, data = pending.lane_indices, pending.transition
            following = old.clone()
            steps = old.physical_steps[indices] + 1
            elapsed = steps.to(self._dtype) * self._spec.dt
            good, failed = data["in_goal_after"], data["failure_after"]
            since = old.in_goal_since[indices].clone()
            entered = good & (~data["in_goal_before"] | torch.isnan(since))
            since = torch.where(entered, elapsed, since)
            since = torch.where(good, since, torch.full_like(since, float("nan")))
            hold = torch.where(good, elapsed - since, torch.zeros_like(elapsed))
            hold_met = good & (hold + 1e-10 >= self._spec.required_hold_seconds)
            success = (steps >= self._spec.max_steps) & hold_met & ~failed
            terminated = failed | success
            truncated = (steps >= self._spec.max_steps) & ~terminated
            first = old.first_hold_confirmation[indices].clone()
            first = torch.where(hold_met & ~failed & torch.isnan(first), elapsed, first)
            following.state[indices] = data["state_after"]
            following.error[indices] = data["error_after"]
            following.physical_error[indices] = data["physical_error_after"]
            following.in_goal[indices] = good
            following.observed_failure[indices] = failed
            following.physical_steps[indices] = steps
            following.active[indices] = ~(terminated | truncated)
            following.success[indices] = success
            following.failure[indices] = failed
            following.terminated[indices] = terminated
            following.truncated[indices] = truncated
            following.in_goal_since[indices] = since
            following.first_hold_confirmation[indices] = first
            following.accumulated_cost[indices] += data["cost"]
            following.certificate_penalty[indices] += data["penalty"]
            following.V_rises[indices] += data["V_rise"].to(torch.int64)
            self._validate_state(following)
            # Audit changes are bounded slices, prepared after every lane has
            # passed validation.  A failed ledger write rolls those slices back.
            self._commit_audit(old, following, indices, data, hold)
            self._state = following
            self._generation += 1
            self._pending = None
            return following.clone()

    def advance(self, lane_indices: torch.Tensor | None = None) -> BatchState:
        """One query plus one commit; no actor, sampling, or learning is involved."""
        with self._lock:
            return self.commit(self.propose(lane_indices))

    def _validate_state(self, state: BatchState, *, validate_bindings: bool = False) -> None:
        batch, spec = state.batch_size, self._spec
        if batch < 1 or len(set(state.episode_ids)) != batch:
            raise ValueError("Invalid episode binding")
        shapes = dict(
            initial_scales=(batch,), initial_state=(batch, spec.state_dim),
            state=(batch, spec.state_dim), metric=(batch, spec.error_dim, spec.error_dim),
            raw=(batch, spec.raw_dim), raw_logp=(batch,),
            context=(batch, spec.metric_context_dim), error=(batch, spec.error_dim),
        )
        float32_names = {"raw", "raw_logp", "context"}
        integer_names = {"physical_steps", "V_rises"}
        boolean_names = {"initial_failure", "in_goal", "observed_failure", "active",
                         "success", "failure", "terminated", "truncated"}
        clocks = {"in_goal_since", "first_hold_confirmation"}
        metadata_names = {"episode_ids", "seeds", "actor_versions", "phi_versions",
                          "rng_protocol", "rng_state_by_episode"}
        for field in fields(state):
            value = getattr(state, field.name)
            if field.name in metadata_names:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{field.name} must remain an owned tensor")
            dtype = (torch.float32 if field.name in float32_names else
                     torch.int64 if field.name in integer_names else
                     torch.bool if field.name in boolean_names else self._dtype)
            _tensor(value, shapes.get(field.name, (batch,)), dtype, self._device,
                    field.name, finite=field.name not in clocks, copy=False)
            if field.name in clocks and bool(torch.isinf(value).any()):
                raise FloatingPointError("Hold clocks may be absent, never infinite")
        if any(len(values) != batch for values in (state.seeds, state.actor_versions, state.phi_versions)):
            raise ValueError("Lane provenance length mismatch")
        if any(not isinstance(values, tuple) for values in
               (state.episode_ids, state.seeds, state.actor_versions, state.phi_versions)):
            raise TypeError("Lane provenance sequences must be canonical tuples")
        if any(seed is not None and (isinstance(seed, bool) or not isinstance(seed, int))
               for seed in state.seeds):
            raise ValueError("Invalid lane seed provenance")
        if (len(set(state.actor_versions)) != 1 or len(set(state.phi_versions)) != 1 or
                any(not isinstance(value, str) or not value
                    for value in (*state.episode_ids, *state.actor_versions, *state.phi_versions))):
            raise ValueError("Each batch must preserve one nonempty Actor/Phi version")
        if not state.rng_protocol or not isinstance(state.rng_protocol, str):
            raise ValueError("Missing external RNG ownership protocol")
        if (not isinstance(state.rng_state_by_episode, Mapping) or
                set(state.rng_state_by_episode) != set(state.episode_ids)):
            raise ValueError("Lane RNG provenance mismatch")
        if bool((state.initial_scales <= 0).any()):
            raise ValueError("Initial scales must remain positive")
        if validate_bindings:
            self._metric_binding(state.metric, batch)
            if not torch.equal(state.context, self._kernel.context(batch)):
                raise ValueError("Restored context differs from the frozen task setting")
        if bool((state.physical_steps < 0).any()) or bool((state.physical_steps > spec.max_steps).any()):
            raise ValueError("Physical step counter outside the declared horizon")
        if bool((state.V_rises < 0).any()) or bool((state.V_rises > state.physical_steps).any()):
            raise ValueError("Certificate rise counter exceeds real physical steps")
        if bool((state.physical_error < 0).any()) or bool((state.accumulated_cost < 0).any()) or bool(
                (state.certificate_penalty < 0).any()):
            raise ValueError("Costs and penalties must be nonnegative")
        if not torch.equal(state.active, ~(state.terminated | state.truncated)):
            raise ValueError("Inconsistent active/done lane masks")
        if bool((state.terminated & state.truncated).any()) or bool((state.success & state.failure).any()):
            raise ValueError("Mutually exclusive final labels overlap")
        if not torch.equal(state.terminated, state.success | state.failure):
            raise ValueError("Termination must mean physical failure or full-horizon success")
        if not torch.equal(state.failure, state.observed_failure & (state.physical_steps > 0)):
            raise ValueError("Only successor failure terminates; initial failure is diagnostic")
        expected_truncated = ((state.physical_steps >= spec.max_steps) &
                              ~(state.success | state.failure))
        if not torch.equal(state.truncated, expected_truncated):
            raise ValueError("Timeout must occur exactly at a nonterminated horizon")
        if bool((state.success & (state.physical_steps != spec.max_steps)).any()):
            raise ValueError("Success cannot end a physical horizon early")
        if bool((state.active & (state.physical_steps >= spec.max_steps)).any()):
            raise ValueError("A lane cannot stay active beyond its horizon")
        if bool((state.active & state.failure).any()):
            raise ValueError("Failed lanes cannot remain active")
        expected_missing = ~state.in_goal
        if not torch.equal(torch.isnan(state.in_goal_since), expected_missing):
            raise ValueError("Current goal segment clock disagrees with goal membership")
        elapsed = state.physical_steps.to(self._dtype) * spec.dt
        if bool((state.in_goal_since[state.in_goal] < 0).any()) or bool(
                (state.in_goal_since[state.in_goal] > elapsed[state.in_goal] + 1e-10).any()):
            raise ValueError("Invalid continuous goal segment clock")
        finite_first = ~torch.isnan(state.first_hold_confirmation)
        if bool((state.first_hold_confirmation[finite_first] + 1e-10 < spec.required_hold_seconds).any()) or bool(
                (state.first_hold_confirmation[finite_first] > elapsed[finite_first] + 1e-10).any()):
            raise ValueError("Invalid first confirmation clock")
        hold = torch.where(state.in_goal, elapsed - state.in_goal_since, torch.zeros_like(elapsed))
        expected_success = ((state.physical_steps >= spec.max_steps) & state.in_goal &
                            (hold + 1e-10 >= spec.required_hold_seconds) & ~state.failure)
        if not torch.equal(state.success, expected_success):
            raise ValueError("Final success must use the uninterrupted terminal goal segment")

    def _new_audit(self, state: BatchState) -> TensorAudit:
        spec, batch = self._spec, state.batch_size
        horizon = spec.max_steps
        records: dict[str, torch.Tensor] = {}
        endpoint_dims = dict(state=spec.state_dim, error=spec.error_dim)
        for key, width in endpoint_dims.items():
            records["endpoint_" + key] = torch.full((horizon + 1, batch, width),
                float("nan"), dtype=self._dtype, device=self._device)
        for key in ("physical_error", "hold_seconds"):
            records["endpoint_" + key] = torch.full((horizon + 1, batch),
                float("nan"), dtype=self._dtype, device=self._device)
        for key in ("in_goal", "failure", "valid"):
            records["endpoint_" + key] = torch.zeros((horizon + 1, batch),
                dtype=torch.bool, device=self._device)
        records["endpoint_state"][0] = state.state
        records["endpoint_error"][0] = state.error
        records["endpoint_physical_error"][0] = state.physical_error
        records["endpoint_hold_seconds"][0] = 0.0
        records["endpoint_in_goal"][0] = state.in_goal
        records["endpoint_failure"][0] = state.initial_failure
        records["endpoint_valid"][0] = True
        # Before/after endpoints are represented once in the endpoint tensors.
        for key in ("action_nominal", "applied_action"):
            records[key] = torch.full((horizon, batch, spec.action_dim),
                float("nan"), dtype=self._dtype, device=self._device)
        for key in ("V_before", "V_after", "cost", "decay_residual",
                    "normalized_decay_residual", "penalty"):
            records[key] = torch.full((horizon, batch), float("nan"),
                dtype=self._dtype, device=self._device)
        for key in ("V_rise", "success", "failure", "terminated", "truncated", "valid"):
            records[key] = torch.zeros((horizon, batch), dtype=torch.bool, device=self._device)
        return TensorAudit(records)

    def _commit_audit(self, old: BatchState, following: BatchState,
                      indices: torch.Tensor, data: Mapping[str, Any], hold: torch.Tensor) -> None:
        if self._audit is None:
            return
        records = self._audit.records
        row = old.physical_steps[indices]
        successor = row + 1
        updates: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for key, source in (("endpoint_state", "state_after"),
                            ("endpoint_error", "error_after"),
                            ("endpoint_physical_error", "physical_error_after"),
                            ("endpoint_in_goal", "in_goal_after"),
                            ("endpoint_failure", "failure_after")):
            updates[key] = (successor, data[source])
        updates["endpoint_hold_seconds"] = (successor, hold)
        updates["endpoint_valid"] = (successor, torch.ones_like(indices, dtype=torch.bool))
        for key in ("action_nominal", "applied_action", "V_before", "V_after", "cost",
                    "decay_residual", "normalized_decay_residual", "penalty", "V_rise"):
            updates[key] = (row, data[key])
        for key in ("success", "failure", "terminated", "truncated"):
            updates[key] = (row, getattr(following, key)[indices])
        updates["valid"] = (row, torch.ones_like(indices, dtype=torch.bool))
        backups = {key: records[key][position, indices].clone()
                   for key, (position, _) in updates.items()}
        try:
            for key, (position, value) in updates.items():
                records[key][position, indices] = value
        except BaseException:
            for key, (position, _) in updates.items():
                records[key][position, indices] = backups[key]
            raise

    def audit(self) -> TensorAudit:
        with self._lock:
            self._require_state()
            if self._audit is None:
                raise RuntimeError("Dense audit was not enabled for this runner")
            return self._audit.clone()

    def summary(self) -> dict[str, Any]:
        """Return completed per-lane results; never reduce across batch lanes.

        ``arrival_time`` preserves rollout_metric's full executed horizon on
        success.  ``sustained_arrival_time`` instead reports terminal-segment
        entry plus required hold, matching FullHorizonTask's engineering info.
        """
        with self._lock:
            state = self._require_state()
            if bool(state.active.any()):
                raise RuntimeError("Incomplete lanes cannot be reported as completed episodes")
            spec = self._spec
            elapsed = state.physical_steps.to(self._dtype) * spec.dt
            horizon = torch.full_like(elapsed, spec.horizon_seconds)
            hold = torch.where(state.in_goal, elapsed - state.in_goal_since, torch.zeros_like(elapsed))
            tail_error = torch.where(state.failure,
                torch.clamp_min(state.physical_error, self._failure_error), state.physical_error)
            remaining = (spec.max_steps - state.physical_steps).to(self._dtype)
            fixed = state.accumulated_cost + remaining * spec.dt * tail_error
            objective = fixed + self._certificate_weight * state.certificate_penalty
            if not bool(torch.isfinite(torch.stack((fixed, objective, state.physical_error))).all()):
                raise FloatingPointError("Nonfinite completed rollout; refusing a training result")
            return _clone(dict(
                schema_version=SCHEMA_VERSION, episode_ids=state.episode_ids,
                fixed_horizon_cost=fixed, accumulated_cost=state.accumulated_cost,
                objective=objective, certificate_penalty=state.certificate_penalty,
                V_rise_fraction=state.V_rises.to(self._dtype) / state.physical_steps.clamp_min(1),
                success=state.success, failure=state.failure,
                initial_failure=state.initial_failure, physical_steps=state.physical_steps,
                metric_control_steps=state.physical_steps,
                arrival_time=torch.where(state.success, elapsed, horizon),
                sustained_arrival_time=torch.where(state.success,
                    state.in_goal_since + spec.required_hold_seconds, horizon),
                settling_time=torch.where(state.success, state.in_goal_since, horizon),
                first_hold_confirmation_time=state.first_hold_confirmation,
                hold_seconds=hold, required_hold_seconds=spec.required_hold_seconds,
                terminal_physical_error=state.physical_error,
                physical_terminated=state.terminated, physical_truncated=state.truncated,
                horizon_seconds=spec.horizon_seconds, moving_reference=spec.moving_reference,
                reference_period=spec.reference_period, metric_fixed_in_episode=True,
                gradient_mode=GRADIENT_MODE,
            ))

    def snapshot(self) -> RunnerSnapshot:
        """Own a complete commit-boundary snapshot; pending queries reject."""
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("Commit or cancel the pending query before taking a snapshot")
            state = self._require_state().clone()
            audit = None if self._audit is None else self._audit.clone()
            provisional = RunnerSnapshot(_clone(self._identity), self._generation,
                                         self._ticket_counter, state, audit, "")
            return RunnerSnapshot(provisional.identity, provisional.generation,
                                  provisional.ticket_counter, state, audit,
                                  _digest(provisional.payload()))

    def restore(self, snapshot: RunnerSnapshot) -> BatchState:
        """Restore owned lane state only, without reset, model calls, or RNG draws.

        External sampler RNG payloads are retained, not installed into shared
        process-global RNGs.  The sampler owner must restore them under its own
        declared ownership protocol before sampling later batches.
        """
        with self._lock:
            self._check_kernel()
            if self._pending is not None:
                raise RuntimeError("Cannot restore over a pending query")
            if not isinstance(snapshot, RunnerSnapshot):
                raise TypeError("Expected RunnerSnapshot")
            if (_digest(snapshot.identity) != _digest(self._identity) or
                    _digest(snapshot.payload()) != snapshot.checksum):
                raise ValueError("Snapshot identity/content changed or belongs to another protocol")
            if (isinstance(snapshot.generation, bool) or not isinstance(snapshot.generation, int) or
                    snapshot.generation < 0 or isinstance(snapshot.ticket_counter, bool) or
                    not isinstance(snapshot.ticket_counter, int) or snapshot.ticket_counter < 0):
                raise ValueError("Invalid transaction counters")
            state = snapshot.state.clone()
            self._validate_state(state, validate_bindings=True)
            audit = None if snapshot.audit is None else snapshot.audit.clone()
            if (audit is not None) != self._dense_audit:
                raise ValueError("Snapshot audit mode differs from this runner")
            if audit is not None:
                self._validate_audit(audit, state)
            self._state, self._audit = state, audit
            self._generation = snapshot.generation
            self._ticket_counter = snapshot.ticket_counter
            self._runner_id = uuid.uuid4().hex
            return state.clone()

    def _validate_audit(self, audit: TensorAudit, state: BatchState) -> None:
        expected = self._new_audit(state).records
        if set(audit.records) != set(expected):
            raise ValueError("Tensor audit schema mismatch")
        for key, value in audit.records.items():
            model = expected[key]
            _tensor(value, tuple(model.shape), model.dtype, model.device, key, finite=False, copy=False)
        valid = audit.records["valid"]
        endpoint_valid = audit.records["endpoint_valid"]
        slots = torch.arange(self._spec.max_steps, device=self._device)[:, None]
        expected_valid = slots < state.physical_steps[None, :]
        endpoint_slots = torch.arange(self._spec.max_steps + 1, device=self._device)[:, None]
        expected_endpoints = endpoint_slots <= state.physical_steps[None, :]
        if not torch.equal(valid, expected_valid) or not torch.equal(endpoint_valid, expected_endpoints):
            raise ValueError("Audit validity does not match actual lane physical steps")
        for key, value in audit.records.items():
            if value.dtype == torch.bool:
                mask = endpoint_valid if key.startswith("endpoint_") else valid
                if bool(value[~mask].any()):
                    raise ValueError("Boolean audit padding must remain false")
                continue
            mask = endpoint_valid if key.startswith("endpoint_") else valid
            if not bool(torch.isfinite(value[mask]).all()):
                raise FloatingPointError("Nonfinite valid physical audit entry")
            if not bool(torch.isnan(value[~mask]).all()):
                raise ValueError("Invalid audit padding must remain absent")
        batch_indices = torch.arange(state.batch_size, device=self._device)
        if not torch.equal(audit.records["endpoint_state"][0], state.initial_state):
            raise ValueError("Audit initial state disagrees with episode binding")
        if not torch.equal(audit.records["endpoint_state"][state.physical_steps, batch_indices], state.state):
            raise ValueError("Audit final endpoint disagrees with committed state")
        for key, terminal in (("endpoint_error", state.error),
                              ("endpoint_physical_error", state.physical_error),
                              ("endpoint_in_goal", state.in_goal),
                              ("endpoint_failure", state.observed_failure)):
            if not torch.equal(audit.records[key][state.physical_steps, batch_indices], terminal):
                raise ValueError("Audit final measurement disagrees with committed state")
        if not torch.equal(audit.records["endpoint_failure"][0], state.initial_failure):
            raise ValueError("Audit initial failure diagnostic disagrees with the binding")
        for key in ("success", "failure", "terminated", "truncated"):
            terminal = torch.zeros_like(state.active)
            stepped = state.physical_steps > 0
            terminal[stepped] = audit.records[key][state.physical_steps[stepped] - 1,
                                                 batch_indices[stepped]]
            if not torch.equal(terminal, getattr(state, key)):
                raise ValueError("Audit final labels disagree with committed state")
        accumulated = torch.zeros_like(state.accumulated_cost)
        penalty = torch.zeros_like(state.certificate_penalty)
        for row in range(self._spec.max_steps):
            accumulated += torch.where(valid[row], audit.records["cost"][row], 0.0)
            penalty += torch.where(valid[row], audit.records["penalty"][row], 0.0)
        if (not torch.equal(accumulated, state.accumulated_cost) or
                not torch.equal(penalty, state.certificate_penalty) or
                not torch.equal(audit.records["V_rise"].sum(dim=0), state.V_rises)):
            raise ValueError("Audit costs, penalties, or rise counts disagree with committed state")


__all__ = ["BatchState", "TensorAudit", "StepProposal", "RunnerSnapshot", "FullHorizonRunner"]
