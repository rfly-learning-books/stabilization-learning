"""Real-physics runner for the sampling-first CLF controller."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from .calibration import CalibrationConfig, mode_command
from .evaluation import json_safe, success_from_metrics, summarize_trace, terminal_metrics
from .math import matrix_hash, rot2, shape_error
from .physics import MuJoCoAdapter, PhysicsState
from .solver import CLFConfig, CLFSolver


@dataclass(frozen=True)
class RolloutConfig:
    max_control_ticks: int = 100
    hold_ticks: int = 5
    position_tolerance_m: float = 0.005
    yaw_tolerance_rad: float = np.deg2rad(5.0)
    velocity_tolerance_mps: float = 0.005
    angular_velocity_tolerance_radps: float = 0.05
    coverage_min: float = 0.95
    pusher_speed_limit_mps: float = 0.20
    transfer_tolerance_m: float = 0.003
    engage_speed_mps: float = 0.01
    max_roll_pitch_rad: float = np.deg2rad(10.0)
    max_object_height_error_m: float = 0.03

    def __post_init__(self):
        values = (
            self.max_control_ticks,
            self.hold_ticks,
            self.position_tolerance_m,
            self.yaw_tolerance_rad,
            self.velocity_tolerance_mps,
            self.angular_velocity_tolerance_radps,
            self.coverage_min,
            self.pusher_speed_limit_mps,
            self.transfer_tolerance_m,
            self.engage_speed_mps,
            self.max_roll_pitch_rad,
            self.max_object_height_error_m,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("rollout settings must be finite and positive")


class CertificateRunner:
    """Execute one measured episode with explicit transfer and failure states."""

    def __init__(self, adapter: MuJoCoAdapter, geometry, library, Q,
                 solver_config: CLFConfig | None = None,
                 rollout_config: RolloutConfig | None = None,
                 orientation_period=None):
        self.adapter = adapter
        self.geometry = geometry
        self.library = library
        self.solver_config = solver_config or CLFConfig()
        self.rollout_config = rollout_config or RolloutConfig()
        self.orientation_period = float(orientation_period or getattr(geometry, "period", 2 * np.pi))
        self.solver = CLFSolver(library, Q, self.solver_config)
        self._calibration_config = CalibrationConfig(
            pusher_radius=float(library.metadata.get("pusher_radius", .018)),
            contact_clearance=float(library.metadata.get("clearance", .002)),
            max_command_mps=self.rollout_config.pusher_speed_limit_mps,
        )

    def _world_mode_target(self, state: PhysicsState, mode):
        local = self.library.contact_points[mode] + self.library.normals[mode] * float(
            self.library.contact_offset[mode]
        )
        return state.object_q[:2] + rot2(float(state.object_q[2])) @ local

    def _transfer_command(self, state, local_waypoint):
        target = state.object_q[:2] + rot2(float(state.object_q[2])) @ np.asarray(local_waypoint, dtype=float)
        delta = target - state.pusher_position[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= self.rollout_config.transfer_tolerance_m:
            return np.zeros(2, dtype=np.float64), True, distance
        command = delta / max(distance, 1e-12) * min(
            self.rollout_config.pusher_speed_limit_mps,
            distance / self.adapter.control_dt,
        )
        return command, False, distance

    def _push_command(self, state, mode, speed):
        return mode_command(
            state,
            self.library.contact_points[mode],
            self.library.normals[mode],
            self.library.tangents[mode],
            speed,
            self._calibration_config,
            self.library.contact_offset[mode],
        )

    def _terminal(self, measured):
        metrics = terminal_metrics(
            measured,
            self.target_q,
            self.solver_config.feature_length,
            self.orientation_period,
            self.geometry.polygon if self.geometry is not None else None,
        )
        return success_from_metrics(
            metrics,
            self.rollout_config.position_tolerance_m,
            self.rollout_config.yaw_tolerance_rad,
            self.rollout_config.velocity_tolerance_mps,
            self.rollout_config.angular_velocity_tolerance_radps,
            self.rollout_config.coverage_min,
            self.rollout_config.velocity_tolerance_mps,
        ), metrics

    def run(self, object_q, target_q, pusher_xy, seed=None):
        """Run one episode and return raw trace plus auditable summary."""

        self.target_q = np.asarray(target_q, dtype=np.float64)
        if self.target_q.shape != (3,) or not np.isfinite(self.target_q).all():
            raise ValueError("target_q must be finite shape (3,)")
        state = self.adapter.reset(object_q, pusher_xy)
        current_mode = -1
        requested_mode = None
        transfer_path = None
        transfer_index = 0
        hold_count = 0
        trace = []
        final_failure = None
        for tick in range(self.rollout_config.max_control_ticks):
            state_before = state.copy()
            D = shape_error(
                state.object_q,
                self.target_q,
                self.solver_config.feature_length,
                self.orientation_period,
            )
            terminal, before_metrics = self._terminal(state)
            if abs(state.roll) > self.rollout_config.max_roll_pitch_rad or abs(state.pitch) > self.rollout_config.max_roll_pitch_rad:
                final_failure = "TILT_LIMIT"
                command = np.zeros(2)
                execution_state = "FAIL"
                decision = None
            elif abs(state.object_position[2] - self.adapter.model.body_pos[self.adapter.object_body][2]) > self.rollout_config.max_object_height_error_m:
                final_failure = "HEIGHT_LIMIT"
                command = np.zeros(2)
                execution_state = "FAIL"
                decision = None
            elif terminal or hold_count > 0:
                decision = None
                command = np.zeros(2)
                execution_state = "HOLD"
                hold_count = hold_count + 1 if terminal else 0
                if hold_count >= self.rollout_config.hold_ticks:
                    final_failure = None
                    trace.append(self._record(tick, state_before, state, D, decision, command, execution_state, None, before_metrics))
                    return self._report(True, None, trace, state, before_metrics, current_mode)
            else:
                decision = self.solver.select_stage(D, current_mode)
                if not decision.feasible:
                    final_failure = decision.status
                    command = np.zeros(2)
                    execution_state = "FAIL"
                elif current_mode != decision.mode:
                    requested_mode = int(decision.mode)
                    if transfer_path is None:
                        if current_mode >= 0 and self.library.paths is not None:
                            transfer_path = self.library.paths[current_mode][requested_mode]
                        else:
                            target_local = self.library.contact_points[requested_mode] + self.library.normals[requested_mode] * float(
                                self.library.contact_offset[requested_mode]
                            )
                            transfer_path = [target_local.tolist()]
                        transfer_index = 0
                    if transfer_index < len(transfer_path):
                        command, reached, _ = self._transfer_command(state, transfer_path[transfer_index])
                        execution_state = "TRANSIT"
                        if reached:
                            transfer_index += 1
                    else:
                        local = self.library.contact_points[requested_mode] + self.library.normals[requested_mode] * float(
                            self.library.contact_offset[requested_mode]
                        )
                        _, reached, _ = self._transfer_command(state, local)
                        execution_state = "ENGAGE"
                        if state.contact.active:
                            current_mode = requested_mode
                            requested_mode = None
                            transfer_path = None
                            transfer_index = 0
                            command = self._push_command(
                                state, current_mode,
                                max(self.library.u_min[current_mode], self.rollout_config.engage_speed_mps),
                            )
                        else:
                            # Hold the pusher at the target while moving inward
                            # slowly.  A stationary point at the clearance
                            # shell cannot create contact by itself.
                            rotation = rot2(float(state.object_q[2]))
                            command = rotation @ (
                                -self.library.normals[requested_mode]
                                * self.rollout_config.engage_speed_mps
                            )
                else:
                    command = self._push_command(state, current_mode, decision.speed)
                    execution_state = "PUSH"
            command = np.asarray(command, dtype=np.float64)
            if not np.isfinite(command).all() or np.any(np.abs(command) > .300001):
                final_failure = "COMMAND_INVALID"
                command = np.zeros(2)
                execution_state = "FAIL"
            next_state = self.adapter.step(command)
            after_metrics = terminal_metrics(
                next_state,
                self.target_q,
                self.solver_config.feature_length,
                self.orientation_period,
                self.geometry.polygon if self.geometry is not None else None,
            )
            if execution_state == "HOLD" and not success_from_metrics(
                after_metrics,
                self.rollout_config.position_tolerance_m,
                self.rollout_config.yaw_tolerance_rad,
                self.rollout_config.velocity_tolerance_mps,
                self.rollout_config.angular_velocity_tolerance_radps,
                self.rollout_config.coverage_min,
                self.rollout_config.velocity_tolerance_mps,
            ):
                hold_count = 0
            trace.append(self._record(
                tick, state_before, next_state, D, decision, command,
                execution_state, final_failure, after_metrics,
            ))
            state = next_state
            if final_failure is not None:
                break
        if final_failure is None:
            final_failure = "TIMEOUT"
        return self._report(False, final_failure, trace, state, terminal_metrics(
            state, self.target_q, self.solver_config.feature_length,
            self.orientation_period, self.geometry.polygon if self.geometry is not None else None,
        ), current_mode)

    @staticmethod
    def _record(tick, before, after, D, decision, command, execution_state, failure_code, metrics):
        return {
            "tick": int(tick),
            "state_before": before.to_dict(),
            "state_after": after.to_dict(),
            "D_before": np.asarray(D).copy(),
            "command_mps": np.asarray(command).copy(),
            "stage": "HOLD" if execution_state == "HOLD" else (None if decision is None else decision.stage),
            "mode": None if decision is None else decision.mode,
            "decision_status": None if decision is None else decision.status,
            "decision": None if decision is None else decision.to_dict(),
            "execution_state": execution_state,
            "failure_code": failure_code,
            "metrics": metrics,
        }

    def _report(self, success, failure_code, trace, state, metrics, current_mode):
        return {
            "schema": 1,
            "success": bool(success),
            "failure_code": failure_code,
            "q_hash": matrix_hash(self.solver.Q),
            "mode_library_fingerprint": self.library.fingerprint,
            "ticks": len(trace),
            "current_mode": int(current_mode),
            "final_metrics": metrics,
            "trace_summary": summarize_trace(trace),
            "trace": trace,
        }

    @staticmethod
    def save_report(report, path):
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n")
        return path
