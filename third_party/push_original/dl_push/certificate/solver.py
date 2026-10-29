"""Auditable sampled J/A/P CLF candidate solver."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .library import ModeLibrary
from .math import (
    certificate,
    exact_se2_step,
    minimum_linear_input,
    response_in_error_frame,
    stage_projection,
    validate_q,
)


@dataclass(frozen=True)
class CLFConfig:
    """Frozen controller constants for one experiment family."""

    feature_length: float = 0.10
    control_dt: float = 0.20
    input_upper_mps: float = 0.06
    clf_rate_joint: float = 0.50
    clf_rate_attitude: float = 0.80
    clf_rate_position: float = 1.00
    q_trace: float = 3.0
    q_eigenvalue_floor: float = 0.05
    attitude_enter_rad: float = np.deg2rad(2.0)
    attitude_exit_rad: float = np.deg2rad(3.0)
    attitude_hard_band_rad: float = np.deg2rad(5.0)
    attitude_barrier_rate: float = 1.0
    position_endpoint_tolerance: float = 1e-8
    endpoint_check: bool = True
    switch_cost: float = 0.01
    transfer_cost: float = 0.01
    response_error_margin: float = 0.0
    terminal_value_tolerance: float = 1e-12

    def __post_init__(self):
        values = (
            self.feature_length,
            self.control_dt,
            self.input_upper_mps,
            self.clf_rate_joint,
            self.clf_rate_attitude,
            self.clf_rate_position,
            self.q_trace,
            self.q_eigenvalue_floor,
            self.attitude_enter_rad,
            self.attitude_exit_rad,
            self.attitude_hard_band_rad,
            self.attitude_barrier_rate,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("CLF configuration values must be finite and positive")
        if self.q_eigenvalue_floor >= 1 or self.attitude_enter_rad > self.attitude_exit_rad:
            raise ValueError("invalid CLF angle thresholds")
        if self.attitude_hard_band_rad < self.attitude_exit_rad:
            raise ValueError("hard attitude band must contain the exit band")
        if self.position_endpoint_tolerance < 0 or self.response_error_margin < 0:
            raise ValueError("CLF tolerances must be nonnegative")

    @property
    def rates(self):
        return {"JOINT": self.clf_rate_joint, "ATTITUDE": self.clf_rate_attitude,
                "POSITION": self.clf_rate_position}


@dataclass
class ModeCandidate:
    mode: int
    stage: str
    beta: float
    value: float
    requested_drop: float
    lower: float
    upper: float
    speed: float | None
    feasible: bool
    status: str
    endpoint_value: float | None = None
    endpoint_error: np.ndarray | None = None
    cost: float = np.inf
    active_constraints: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "mode": int(self.mode),
            "stage": self.stage,
            "beta": float(self.beta),
            "value": float(self.value),
            "requested_drop": float(self.requested_drop),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "speed": None if self.speed is None else float(self.speed),
            "feasible": bool(self.feasible),
            "status": self.status,
            "endpoint_value": None if self.endpoint_value is None else float(self.endpoint_value),
            "endpoint_error": None if self.endpoint_error is None else self.endpoint_error.tolist(),
            "cost": float(self.cost),
            "active_constraints": list(self.active_constraints),
        }


@dataclass
class Decision:
    stage: str
    mode: int | None
    speed: float
    status: str
    candidates: list[ModeCandidate]
    q_hash: str | None = None

    @property
    def feasible(self):
        return self.mode is not None and self.status == "FEASIBLE"

    def to_dict(self):
        return {
            "stage": self.stage,
            "mode": None if self.mode is None else int(self.mode),
            "speed": float(self.speed),
            "status": self.status,
            "q_hash": self.q_hash,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class CLFSolver:
    """Generate and select one scalar nonnegative mode input.

    The input is a scalar normal speed for one contact mode.  A separate
    contact executor turns the selected mode and speed into a real XY pusher
    velocity.  No candidate is silently clipped into feasibility.
    """

    def __init__(self, library: ModeLibrary, Q, config: CLFConfig | None = None):
        self.library = library
        self.Q = validate_q(
            Q,
            eigenvalue_floor=(config or CLFConfig()).q_eigenvalue_floor,
            trace=(config or CLFConfig()).q_trace,
        ).copy()
        self.config = config or CLFConfig()
        self._q_hash = self._hash_q(self.Q)

    @staticmethod
    def _hash_q(Q):
        return __import__("hashlib").sha256(np.asarray(Q, dtype=np.float64).tobytes()).hexdigest()

    @property
    def q_hash(self):
        return self._q_hash

    def _stage_endpoint(self, D, response, speed, stage):
        next_state = exact_se2_step(
            np.array([D[0] * self.config.feature_length,
                      D[1] * self.config.feature_length, D[2]]),
            response,
            speed,
            self.config.control_dt,
        )
        endpoint = np.array([
            next_state[0] / self.config.feature_length,
            next_state[1] / self.config.feature_length,
            next_state[2],
        ])
        value, _, _ = certificate(endpoint, self.Q, stage)
        return endpoint, value

    def candidate(self, D, stage, mode):
        """Evaluate one mode and return all reasons, including rejection."""

        error = np.asarray(D, dtype=np.float64)
        if error.shape != (3,) or not np.isfinite(error).all():
            raise ValueError("D must be a finite 3-vector")
        if stage not in ("JOINT", "ATTITUDE", "POSITION"):
            raise ValueError(f"unknown stage {stage!r}")
        if not 0 <= int(mode) < self.library.count:
            raise ValueError("mode index out of range")
        index = int(mode)
        value, gradient, beta = certificate(
            error,
            self.Q,
            stage,
            self.library.c[:, index],
            self.config.feature_length,
        )
        rate = self.config.rates[stage]
        requested_drop = rate * value
        lower = float(self.library.u_min[index])
        upper = min(float(self.library.u_max[index]), self.config.input_upper_mps)
        reasons = []
        if not self.library.valid[index]:
            reasons.append("MODE_INVALID")
        if upper < lower:
            reasons.append("RATE_SATURATED")
        if value <= self.config.terminal_value_tolerance:
            lower = 0.0
            speed = 0.0
        else:
            margin = self.config.response_error_margin * float(np.linalg.norm(gradient))
            speed = minimum_linear_input(
                [beta], [-requested_drop - margin], lower, upper
            )
            if speed is None:
                if beta >= -1e-12:
                    reasons.append("NO_DESCENT")
                else:
                    reasons.append("RATE_SATURATED")
        endpoint_value = None
        endpoint_error = None
        if speed is not None and not reasons:
            if stage == "POSITION":
                barrier_value = self.config.attitude_hard_band_rad ** 2 - error[2] ** 2
                barrier_coefficient = 2.0 * error[2] * float(self.library.c[2, index])
                barrier_bound = self.config.attitude_barrier_rate * barrier_value
                constrained_speed = minimum_linear_input(
                    [beta, barrier_coefficient],
                    [-requested_drop, barrier_bound],
                    lower,
                    upper,
                )
                if constrained_speed is None:
                    reasons.append("CBF_CONFLICT")
                else:
                    speed = constrained_speed
            endpoint_error, endpoint_value = self._stage_endpoint(
                error, self.library.c[:, index], speed, stage
            )
            if self.config.endpoint_check:
                if endpoint_value > np.exp(-rate * self.config.control_dt) * value + self.config.position_endpoint_tolerance:
                    reasons.append("SAMPLE_HOLD_FAIL")
                if stage == "POSITION" and abs(endpoint_error[2]) > self.config.attitude_hard_band_rad + self.config.position_endpoint_tolerance:
                    reasons.append("CBF_CONFLICT")
        feasible = speed is not None and not reasons
        status = "FEASIBLE" if feasible else (reasons[0] if reasons else "INFEASIBLE")
        active = []
        if abs(beta * (speed or 0.0) + requested_drop) <= 1e-7:
            active.append("CLF")
        if speed is not None and abs(speed - lower) <= 1e-8:
            active.append("LOWER")
        if speed is not None and abs(speed - upper) <= 1e-8:
            active.append("UPPER")
        return ModeCandidate(
            mode=index,
            stage=stage,
            beta=float(beta),
            value=float(value),
            requested_drop=float(requested_drop),
            lower=float(lower),
            upper=float(max(upper, 0.0)),
            speed=None if speed is None else float(speed),
            feasible=feasible,
            status=status,
            endpoint_value=endpoint_value,
            endpoint_error=endpoint_error,
            active_constraints=active,
        )

    def candidates(self, D, stage):
        return [self.candidate(D, stage, index) for index in self.library.active_indices]

    def select(self, D, stage, current_mode=-1):
        """Select the lowest cost feasible candidate without fallback clipping."""

        candidates = self.candidates(D, stage)
        for candidate in candidates:
            switch = 0.0 if int(candidate.mode) == int(current_mode) else self.config.switch_cost
            transfer = 0.0
            if current_mode >= 0 and self.library.transition_time is not None:
                if not self.library.transition_valid[current_mode, candidate.mode]:
                    candidate.status = "TRANSFER_INVALID"
                    candidate.feasible = False
                else:
                    transfer = self.config.transfer_cost * float(self.library.transition_time[current_mode, candidate.mode])
            if candidate.feasible:
                candidate.cost = float((candidate.speed or 0.0) ** 2 + switch + transfer)
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            statuses = {candidate.status for candidate in candidates}
            if "NO_DESCENT" in statuses:
                status = "NO_DESCENT"
            elif "RATE_SATURATED" in statuses:
                status = "RATE_SATURATED"
            elif "TRANSFER_INVALID" in statuses:
                status = "TRANSFER_INVALID"
            elif "SAMPLE_HOLD_FAIL" in statuses:
                status = "SAMPLE_HOLD_FAIL"
            else:
                status = "NO_FEASIBLE_ACTION"
            return Decision(stage, None, 0.0, status, candidates, self.q_hash)
        chosen = min(feasible, key=lambda item: (item.cost, item.mode))
        return Decision(stage, chosen.mode, chosen.speed or 0.0, "FEASIBLE", candidates, self.q_hash)

    def select_stage(self, D, current_mode=-1, terminal=False):
        """Apply a deterministic J/A/P supervisor policy for one observation."""

        error = np.asarray(D, dtype=np.float64)
        if error.shape != (3,) or not np.isfinite(error).all():
            raise ValueError("D must be a finite 3-vector")
        if terminal:
            return Decision("HOLD", None, 0.0, "TERMINAL", [], self.q_hash)
        joint = self.select(error, "JOINT", current_mode)
        if joint.feasible:
            return joint
        if abs(error[2]) > self.config.attitude_enter_rad:
            attitude = self.select(error, "ATTITUDE", current_mode)
            if attitude.feasible:
                return attitude
            return attitude
        position = self.select(error, "POSITION", current_mode)
        if position.feasible:
            return position
        if abs(error[2]) > self.config.attitude_exit_rad:
            return self.select(error, "ATTITUDE", current_mode)
        # Preserve the most informative rejection, with joint evidence in the
        # candidate log supplied by the caller if needed.
        return position
