"""Sampled contact-mode calibration for the certificate controller."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from .library import ModeLibrary, fit_mode_response
from .math import rot2
from .physics import MuJoCoAdapter, PhysicsState
from .evaluation import json_safe


@dataclass(frozen=True)
class CalibrationConfig:
    speeds_mps: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.06)
    yaws_rad: tuple[float, ...] = tuple(np.deg2rad(np.arange(0.0, 360.0, 45.0)).tolist())
    repeats: int = 3
    hold_ticks: int = 3
    pusher_radius: float = 0.018
    contact_clearance: float = 0.002
    tangent_gain: float = 4.0
    tangent_speed_limit: float = 0.03
    max_command_mps: float = 0.20
    max_tilt_rad: float = np.deg2rad(5.0)
    minimum_hold_rate: float = 0.95
    maximum_fit_error: float = 0.15
    corner_force_min_n: float = 0.05
    corner_normal_alignment: float = 0.70
    object_xy: tuple[float, float] = (0.0, 0.0)
    object_z: float = 0.0205

    def __post_init__(self):
        if not self.speeds_mps or any(not np.isfinite(v) or v <= 0 or v > .30 for v in self.speeds_mps):
            raise ValueError("calibration speeds must lie in (0, .30]")
        if not self.yaws_rad or any(not np.isfinite(v) for v in self.yaws_rad):
            raise ValueError("calibration yaw grid must be finite and nonempty")
        if self.repeats < 1 or self.hold_ticks < 1:
            raise ValueError("calibration repeats and hold_ticks must be positive")
        if min(self.pusher_radius, self.contact_clearance, self.tangent_gain, self.tangent_speed_limit, self.max_command_mps, self.max_tilt_rad, self.corner_force_min_n) <= 0:
            raise ValueError("calibration geometry/controller settings must be positive")
        if not 0 < self.corner_normal_alignment <= 1:
            raise ValueError("corner normal alignment must be in (0,1]")


def mode_command(state: PhysicsState, point, normal, tangent, speed, config: CalibrationConfig, contact_offset=None):
    """Real XY velocity command for one normal-push calibration mode."""

    point = np.asarray(point, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    rotation = rot2(float(state.object_q[2]))
    local_pusher = rotation.T @ (state.pusher_position[:2] - state.object_q[:2])
    offset = config.pusher_radius + config.contact_clearance if contact_offset is None else float(contact_offset)
    if not np.isfinite(offset) or offset <= 0:
        raise ValueError("contact_offset must be positive")
    target = point + normal * offset
    tangent_error = float(tangent @ (local_pusher - target))
    body_velocity = rotation.T @ state.object_velocity[:2]
    twist_follow = float(tangent @ (body_velocity + state.object_angular_velocity[2] * np.array([-point[1], point[0]])))
    inward = -normal * float(speed)
    tangential_speed = np.clip(
        twist_follow - config.tangent_gain * tangent_error,
        -config.tangent_speed_limit,
        config.tangent_speed_limit,
    )
    tangential = tangent * tangential_speed
    command = rotation @ (inward + tangential)
    # The actuator limit is per axis, while the calibration contract also
    # records the total requested command for saturation diagnostics.
    command = np.clip(command, -config.max_command_mps, config.max_command_mps)
    return command


class ModeCalibrator:
    """Run real MuJoCo samples and fill a boundary-sampled mode library."""

    def __init__(self, adapter: MuJoCoAdapter, config: CalibrationConfig | None = None):
        self.adapter = adapter
        self.config = config or CalibrationConfig()
        self.velocity_rows = []

    def _contact_valid(self, state, library, mode):
        """Require one measured contact per declared active face."""

        contact = state.contact
        if not contact.active:
            return False
        expected = library.face_normals[mode][library.face_mask[mode]]
        if len(expected) <= 1:
            return True
        if contact.pair_count < len(expected):
            return False
        rotation = rot2(float(state.object_q[2]))
        expected_world = np.c_[expected @ rotation.T, np.zeros(len(expected))]
        actual_forces = np.linalg.norm(contact.pair_forces_world, axis=1)
        actual_normals = contact.pair_normals_world
        used = set()
        for expected_normal in expected_world:
            scores = np.abs(actual_normals @ expected_normal)
            order = np.argsort(-scores)
            selected = None
            for index in order:
                if int(index) in used:
                    continue
                if actual_forces[index] >= self.config.corner_force_min_n and scores[index] >= self.config.corner_normal_alignment:
                    selected = int(index)
                    break
            if selected is None:
                return False
            used.add(selected)
        return True

    def calibrate(self, library: ModeLibrary, output=None):
        if library.count < 1:
            raise ValueError("cannot calibrate an empty library")
        c = library.copy()
        samples = []
        self.velocity_rows = []
        for mode in range(c.count):
            twists, inputs, valid_samples = [], [], []
            run_records = []
            total = 0
            for yaw in self.config.yaws_rad:
                rotation = rot2(float(yaw))
                local_target = c.contact_points[mode] + c.normals[mode] * (
                    c.contact_offset[mode]
                )
                pusher = np.asarray(self.config.object_xy) + rotation @ local_target
                for speed in self.config.speeds_mps:
                    for repeat in range(self.config.repeats):
                        total += 1
                        try:
                            state = self.adapter.reset(
                                object_q=(*self.config.object_xy, float(yaw)),
                                pusher_xy=pusher,
                            )
                            run_twists = []
                            run_inputs = []
                            run_valid = []
                            tick_records = []
                            for tick in range(self.config.hold_ticks):
                                command = mode_command(
                                    state,
                                    c.contact_points[mode],
                                    c.normals[mode],
                                    c.tangents[mode],
                                    speed,
                                    self.config,
                                    c.contact_offset[mode],
                                )
                                next_state = self.adapter.step(command)
                                local_velocity = rot2(-float(next_state.object_q[2])) @ next_state.object_velocity[:2]
                                run_twists.append(np.r_[local_velocity, next_state.object_angular_velocity[2]])
                                run_inputs.append(float(speed))
                                run_valid.append(bool(
                                    self._contact_valid(next_state, c, mode)
                                    and abs(next_state.roll) <= self.config.max_tilt_rad
                                    and abs(next_state.pitch) <= self.config.max_tilt_rad
                                ))
                                valid_tick = bool(run_valid[-1])
                                body_velocity = rot2(-float(next_state.object_q[2])) @ next_state.object_velocity[:2]
                                tick_record = {
                                    "tick": tick,
                                    "object_q": next_state.object_q.copy().tolist(),
                                    "velocity_world_xy_mps": next_state.object_velocity[:2].copy().tolist(),
                                    "velocity_body_xy_mps": body_velocity.tolist(),
                                    "omega_radps": float(next_state.object_angular_velocity[2]),
                                    "command_world_xy_mps": command.copy().tolist(),
                                    "contact_active": bool(next_state.contact.active),
                                    "contact_pairs": int(next_state.contact.pair_count),
                                    "contact_force_norm_n": float(next_state.contact.force_norm),
                                    "roll_rad": float(next_state.roll),
                                    "pitch_rad": float(next_state.pitch),
                                    "valid": valid_tick,
                                }
                                tick_records.append(tick_record)
                                self.velocity_rows.append([
                                    mode, float(yaw), float(speed), repeat, tick,
                                    int(valid_tick),
                                    float(next_state.object_velocity[0]),
                                    float(next_state.object_velocity[1]),
                                    float(body_velocity[0]),
                                    float(body_velocity[1]),
                                    float(next_state.object_angular_velocity[2]),
                                    int(next_state.contact.pair_count),
                                    float(next_state.contact.force_norm),
                                    float(next_state.roll),
                                    float(next_state.pitch),
                                ])
                                state = next_state
                            valid_run = bool(
                                len(run_twists) == self.config.hold_ticks
                                and all(run_valid)
                                and all(np.isfinite(twist).all() for twist in run_twists)
                            )
                            if valid_run:
                                # Fit one steady response sample per run.  A
                                # per-tick fit would overweight solver
                                # transients and make the scalar c_i look
                                # nonlinear even when the run is stable.
                                twists.append(np.mean(np.asarray(run_twists), axis=0))
                                inputs.append(float(np.mean(run_inputs)))
                            valid_samples.append(valid_run)
                            run_records.append({
                                "yaw_rad": float(yaw),
                                "speed": float(speed),
                                "repeat": repeat,
                                "valid": bool(valid_run),
                                "twist": None if not valid_run else np.mean(np.asarray(run_twists), axis=0).tolist(),
                                "ticks": tick_records,
                            })
                        except (FloatingPointError, ValueError, RuntimeError):
                            # A failed sample is retained as an invalid sample
                            # in the summary; it never contributes to c_i.
                            valid_samples.append(False)
                            run_records.append({
                                "yaw_rad": float(yaw),
                                "speed": float(speed),
                                "repeat": repeat,
                                "valid": False,
                                "twist": None,
                                "ticks": [],
                                "error": "invalid_physics_sample",
                            })
                            continue
            twists_array = np.asarray(twists, dtype=np.float64)
            inputs_array = np.asarray(inputs, dtype=np.float64)
            valid_array = np.asarray(valid_samples, dtype=bool)
            hold_rate = float(valid_array.mean()) if len(valid_array) else 0.0
            fit_error = np.inf
            response = np.zeros(3, dtype=np.float64)
            by_speed = {}
            for record in run_records:
                by_speed.setdefault(record["speed"], []).append(record)
            stable_speeds = [
                speed for speed, records in by_speed.items()
                if records and np.mean([record["valid"] for record in records]) >= self.config.minimum_hold_rate
            ]
            selected_records = [
                record for record in run_records
                if record["valid"] and record["speed"] in stable_speeds
            ]
            if selected_records:
                twists_array = np.asarray([record["twist"] for record in selected_records], dtype=np.float64)
                inputs_array = np.asarray([record["speed"] for record in selected_records], dtype=np.float64)
                response, fit_error = fit_mode_response(
                    twists_array, inputs_array
                )
                hold_rate = float(len(selected_records) / max(sum(len(records) for speed, records in by_speed.items() if speed in stable_speeds), 1))
            c.c[:, mode] = response
            c.fit_error[mode] = fit_error
            c.hold_rate[mode] = hold_rate
            c.u_min[mode] = min(stable_speeds) if stable_speeds else 0.0
            c.u_max[mode] = max(stable_speeds) if stable_speeds else max(self.config.speeds_mps)
            c.valid[mode] = bool(
                hold_rate >= self.config.minimum_hold_rate
                and fit_error <= self.config.maximum_fit_error
                and np.linalg.norm(response) > 1e-10
            )
            valid_tick_records = [
                tick for record in selected_records for tick in record.get("ticks", [])
                if tick.get("valid", False)
            ]
            body_velocities = np.asarray([tick["velocity_body_xy_mps"] for tick in valid_tick_records], dtype=np.float64).reshape(-1, 2)
            world_velocities = np.asarray([tick["velocity_world_xy_mps"] for tick in valid_tick_records], dtype=np.float64).reshape(-1, 2)
            omegas = np.asarray([tick["omega_radps"] for tick in valid_tick_records], dtype=np.float64)
            velocity_summary = {
                "valid_tick_count": int(len(valid_tick_records)),
                "body_velocity_mean_mps": body_velocities.mean(axis=0).tolist() if len(body_velocities) else [0.0, 0.0],
                "body_velocity_std_mps": body_velocities.std(axis=0).tolist() if len(body_velocities) else [0.0, 0.0],
                "world_velocity_mean_mps": world_velocities.mean(axis=0).tolist() if len(world_velocities) else [0.0, 0.0],
                "world_velocity_std_mps": world_velocities.std(axis=0).tolist() if len(world_velocities) else [0.0, 0.0],
                "omega_mean_radps": float(omegas.mean()) if len(omegas) else 0.0,
                "omega_std_radps": float(omegas.std()) if len(omegas) else 0.0,
            }
            samples.append(
                dict(
                    mode=mode,
                    total_attempts=total,
                    valid_samples=int(valid_array.sum()),
                    hold_rate=hold_rate,
                    fit_error=fit_error,
                    response=response.tolist(),
                    velocity_summary=velocity_summary,
                    stable_speeds=stable_speeds,
                    speed_results={
                        str(speed): {
                            "attempts": len(records),
                            "valid": int(sum(record["valid"] for record in records)),
                            "hold_rate": float(np.mean([record["valid"] for record in records])),
                        }
                        for speed, records in by_speed.items()
                    },
                    runs=run_records,
                )
            )
        c.metadata.update(
            dict(
                calibration="MuJoCo named-index sampled contact",
                calibration_config={
                    "speeds_mps": list(self.config.speeds_mps),
                    "yaws_rad": list(self.config.yaws_rad),
                    "repeats": self.config.repeats,
                    "hold_ticks": self.config.hold_ticks,
                    "control_dt": self.adapter.control_dt,
                    "physics_dt": self.adapter.physics_dt,
                },
                mode_results=samples,
            )
        )
        if output is not None:
            self.save_samples(output, samples)
        return c

    def save_samples(self, output, samples):
        from pathlib import Path

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "calibration.json").write_text(
            json.dumps(json_safe({"schema": 1, "modes": samples}), indent=2, ensure_ascii=False) + "\n"
        )
        columns = [
            "mode", "yaw_rad", "command_speed_mps", "repeat", "tick", "valid",
            "velocity_world_x_mps", "velocity_world_y_mps",
            "velocity_body_x_mps", "velocity_body_y_mps", "omega_radps",
            "contact_pairs", "contact_force_norm_n", "roll_rad", "pitch_rad",
        ]
        rows = np.asarray(self.velocity_rows, dtype=np.float64)
        if rows.size:
            np.savez_compressed(output / "velocity_samples.npz", columns=np.asarray(columns), values=rows)
