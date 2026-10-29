"""Real MuJoCo mode demonstrations with measured-state overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibration import CalibrationConfig, mode_command
from .evaluation import json_safe
from .math import rot2


def _overlay(frame, lines):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    y = 8
    for line in lines:
        draw.rectangle((5, y - 2, width - 8, y + 17), fill=(245, 248, 250))
        draw.text((9, y), str(line), fill=(22, 45, 63), font=font)
        y += 18
    return np.asarray(image)


def _pixel(point, width, height, limit):
    point = np.asarray(point, dtype=float)
    return np.array([
        (point[0] + limit) / (2.0 * limit) * (width - 1),
        (limit - point[1]) / (2.0 * limit) * (height - 1),
    ])


def _arrow(draw, start, end, color, width=4):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    draw.line([tuple(start.astype(int)), tuple(end.astype(int))], fill=color, width=width)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return
    direction /= length
    side = np.array([-direction[1], direction[0]])
    head = end
    left = head - direction * 15.0 + side * 8.0
    right = head - direction * 15.0 - side * 8.0
    draw.polygon([tuple(head.astype(int)), tuple(left.astype(int)), tuple(right.astype(int))], fill=color)


def _motion_overlay(frame, state, limit=.232, velocity_horizon=.35):
    """Draw measured world/body velocity and signed angular velocity."""

    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    origin = _pixel(state.object_q[:2], width, height, limit)
    scale = width / (2.0 * limit)
    world_velocity = np.asarray(state.object_velocity[:2], dtype=float)
    body_velocity = rot2(-float(state.object_q[2])) @ world_velocity
    world_end = origin + np.array([world_velocity[0], -world_velocity[1]]) * velocity_horizon * scale
    body_world = rot2(float(state.object_q[2])) @ body_velocity
    body_end = origin + np.array([body_world[0], -body_world[1]]) * velocity_horizon * scale
    _arrow(draw, origin, world_end, (224, 76, 61), width=5)
    _arrow(draw, origin, body_end, (31, 142, 205), width=4)
    radius = max(20.0, .035 * scale)
    center = origin
    bbox = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    omega = float(state.object_angular_velocity[2])
    if abs(omega) > 1e-5:
        # PIL's screen y-axis is downward, so reverse the start/end angles for
        # the physical CCW/CW label shown next to the arc.
        if omega > 0:
            draw.arc(bbox, 210, 500, fill=(31, 142, 205), width=4)
            end = center + np.array([radius * .75, radius * .65])
        else:
            draw.arc(bbox, 40, 330, fill=(173, 108, 28), width=4)
            end = center + np.array([radius * .75, -radius * .65])
        radial = end - center
        radial /= max(np.linalg.norm(radial), 1e-8)
        tangent = np.array([-radial[1], radial[0]])
        _arrow(draw, end - radial * 12.0, end + tangent * 2.0, (31, 142, 205) if omega > 0 else (173, 108, 28), width=3)
    return np.asarray(image)


def render_mode_demo(
    adapter,
    geometry,
    library,
    mode,
    output,
    *,
    speed=0.02,
    yaw=0.0,
    control_ticks=10,
    fps=20,
    config: CalibrationConfig | None = None,
):
    """Render one fixed contact mode from a real MuJoCo trajectory."""

    import imageio.v2 as imageio
    import mujoco

    config = config or CalibrationConfig()
    mode = int(mode)
    if not 0 <= mode < library.count:
        raise ValueError("mode index out of range")
    local_center = library.contact_points[mode] + library.normals[mode] * library.contact_offset[mode]
    pusher_xy = rot2(yaw) @ local_center
    state = adapter.reset((0.0, 0.0, float(yaw)), pusher_xy)
    renderer = mujoco.Renderer(adapter.model, width=640, height=640)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    trace = []
    sample_every = max(1, int(round(adapter.substeps / (float(fps) * adapter.control_dt)))
                       )
    for tick in range(int(control_ticks)):
        command = mode_command(
            state,
            library.contact_points[mode],
            library.normals[mode],
            library.tangents[mode],
            float(speed),
            config,
            library.contact_offset[mode],
        )
        for substep in range(adapter.substeps):
            adapter.step_physics(command)
            if substep % sample_every == 0 or substep == adapter.substeps - 1:
                renderer.update_scene(adapter.data, camera="top")
                current = adapter.observe()
                frame = renderer.render()
                frame = _motion_overlay(frame, current)
                body = rot2(-float(current.object_q[2])) @ current.object_velocity[:2]
                omega = float(current.object_angular_velocity[2])
                rotation_label = "CCW" if omega > 1e-5 else ("CW" if omega < -1e-5 else "zero")
                frames.append(_overlay(frame, [
                    f"shape={library.shape}  mode={mode:02d}",
                    f"type={library.mode_type[mode]}  declared_faces={int(library.face_mask[mode].sum())}",
                    f"u={speed:.4f} m/s  t={adapter.data.time:.2f} s",
                    f"contact={current.contact.active} pairs={current.contact.pair_count} F={current.contact.force_norm:.2f} N",
                    f"yaw={current.object_q[2]:+.3f}  roll/pitch={current.roll:+.3f}/{current.pitch:+.3f}",
                    f"v_world=[{current.object_velocity[0]:+.4f}, {current.object_velocity[1]:+.4f}] m/s  red",
                    f"v_body=[{body[0]:+.4f}, {body[1]:+.4f}] m/s  blue",
                    f"omega_z={omega:+.4f} rad/s ({rotation_label})",
                    f"c=[{library.c[0,mode]:+.3f}, {library.c[1,mode]:+.3f}, {library.c[2,mode]:+.3f}]",
                ]))
        state = adapter.observe()
        trace.append({
            "tick": tick,
            "command_mps": command.copy(),
            "state": state.to_dict(),
        })
    with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)
    return {
        "path": str(output.resolve()),
        "shape": library.shape,
        "mode": mode,
        "speed_mps": float(speed),
        "control_ticks": int(control_ticks),
        "frames": len(frames),
        "trace": json_safe(trace),
    }


def render_steady_mode_demo(
    adapter,
    library,
    mode,
    output,
    *,
    speed=.08,
    distance_m=.03,
    max_ticks=40,
    fps=20,
    config=None,
):
    """Render a distance-based fixed-contact run with measured motion arrows."""

    import imageio.v2 as imageio
    import mujoco
    from .steady import (
        SteadyCalibrationConfig,
        _body_xy,
        _contact_valid,
        _fixed_contact_components,
        _fixed_contact_measurement,
        _normal_alignment,
        engage_contact_command,
    )

    config = config or SteadyCalibrationConfig(speeds_mps=(speed,))
    mode = int(mode)
    local_center = library.contact_points[mode] + library.normals[mode] * library.contact_offset[mode]
    state = adapter.reset((0., 0., 0.), local_center)
    renderer = mujoco.Renderer(adapter.model, width=640, height=640)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames, trace = [], []
    sample_every = max(1, int(round(adapter.substeps / (float(fps) * adapter.control_dt))))
    anchor_body = None
    contact_anchor_body = None
    for _ in range(config.engage_max_ticks * adapter.substeps):
        command = engage_contact_command(state, library, mode, speed, config)
        if not np.isfinite(command).all() or np.any(np.abs(command) > config.max_command_mps + 1e-12):
            break
        adapter.step_physics(command)
        state = adapter.observe()
        if _contact_valid(state, library, mode, config):
            anchor_body = _body_xy(state, state.pusher_position[:2])
            contact_anchor_body = _body_xy(state, state.contact.point_world[:2])
            break
    if anchor_body is None:
        anchor_body = _body_xy(state, state.pusher_position[:2])
        contact_anchor_body = _body_xy(state, state.contact.point_world[:2]) if state.contact.active else np.zeros(2)
    run_start_q = state.object_q.copy()
    for tick in range(int(max_ticks)):
        commands = []
        last_parts = None
        tick_valid = True
        tick_reason = None
        max_anchor_error = 0.0
        max_contact_error = 0.0
        min_alignment = 1.0
        for substep in range(adapter.substeps):
            last_parts = _fixed_contact_components(
                state,
                anchor_body,
                library.normals[mode],
                speed,
                tracking_gain=float(config.tracking_gain),
                tangent_body=library.tangents[mode],
                face_count=int(library.face_mask[mode].sum()),
            )
            command = last_parts["command_world"]
            if (
                not np.isfinite(command).all()
                or np.any(np.abs(command) > config.max_command_mps + 1e-12)
                or np.linalg.norm(last_parts["tracking_correction_body"]) > config.tracking_speed_limit + 1e-12
            ):
                tick_valid = False
                tick_reason = "command_saturated"
                break
            commands.append(command.copy())
            adapter.step_physics(command)
            current = adapter.observe()
            measurement = _fixed_contact_measurement(current, anchor_body, contact_anchor_body)
            anchor_error = float(np.linalg.norm(measurement["anchor_error_body"]))
            contact_error = float(np.linalg.norm(measurement["contact_error_body"]))
            max_anchor_error = max(max_anchor_error, anchor_error)
            max_contact_error = max(max_contact_error, contact_error)
            min_alignment = min(min_alignment, _normal_alignment(current, library, mode))
            substep_valid = bool(
                _contact_valid(current, library, mode, config)
                and anchor_error <= config.fixed_contact_tolerance_m
                and contact_error <= config.fixed_contact_point_tolerance_m
                and abs(current.roll) <= config.max_tilt_rad
                and abs(current.pitch) <= config.max_tilt_rad
            )
            if not substep_valid:
                tick_valid = False
                tick_reason = "fixed_contact_violation"
                state = current
                break
            state = current
            if substep % sample_every == 0 or substep == adapter.substeps - 1:
                renderer.update_scene(adapter.data, camera="top")
                frame = _motion_overlay(renderer.render(), current)
                body = rot2(-float(current.object_q[2])) @ current.object_velocity[:2]
                omega = float(current.object_angular_velocity[2])
                label = "CCW" if omega > 1e-5 else ("CW" if omega < -1e-5 else "zero")
                frames.append(_overlay(frame, [
                    f"shape={library.shape} mode={mode:02d} type={library.mode_type[mode]}",
                    f"strict fixed contact  drive u={speed:.4f} m/s  t={adapter.data.time:.2f} s",
                    f"distance={np.linalg.norm(current.object_q[:2]-run_start_q[:2]):.4f}/{distance_m:.4f} m  pairs={current.contact.pair_count}",
                    f"v_world=[{current.object_velocity[0]:+.4f}, {current.object_velocity[1]:+.4f}] m/s",
                    f"v_body=[{body[0]:+.4f}, {body[1]:+.4f}] m/s",
                    f"omega_z={omega:+.4f} rad/s ({label})  F={current.contact.force_norm:.2f} N",
                    f"anchor/contact error={anchor_error*1000:.2f}/{contact_error*1000:.2f} mm  align={min_alignment:.3f}",
                    f"c=[{library.c[0,mode]:+.3f}, {library.c[1,mode]:+.3f}, {library.c[2,mode]:+.3f}]",
                ]))
        if last_parts is None:
            break
        trace.append({
            "tick": tick,
            "command_mps": np.mean(commands, axis=0).tolist() if commands else last_parts["command_world"].tolist(),
            "command_body_mps": last_parts["command_body"].tolist(),
            "drive_command_body_mps": last_parts["drive_body"].tolist(),
            "push_direction_body": last_parts["push_direction_body"].tolist(),
            "drive_speed_mps": float(speed),
            "fixed_anchor_body": anchor_body.tolist(),
            "actual_anchor_body": _body_xy(state, state.pusher_position[:2]).tolist(),
            "fixed_anchor_error_norm_m": max_anchor_error,
            "fixed_contact_point_error_norm_m": max_contact_error,
            "normal_alignment": min_alignment,
            "valid": bool(tick_valid),
            "reason": tick_reason,
            "distance_from_start_m": float(np.linalg.norm(state.object_q[:2] - run_start_q[:2])),
            "state": state.to_dict(),
        })
        if not tick_valid or np.linalg.norm(state.object_q[:2] - run_start_q[:2]) >= distance_m:
            break
    with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)
    return {
        "path": str(output.resolve()),
        "shape": library.shape,
        "mode": mode,
        "speed_mps": float(speed),
        "distance_m": float(distance_m),
        "control_ticks": len(trace),
        "frames": len(frames),
        "trace": json_safe(trace),
    }
