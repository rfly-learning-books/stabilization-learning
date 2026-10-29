"""Independent physical metrics and certificate experiment reports."""

from __future__ import annotations

import copy
import json
import math

import numpy as np

from .math import shape_error


def _transformed_polygon(polygon, q):
    from shapely.affinity import affine_transform

    x, y, yaw = np.asarray(q, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return affine_transform(polygon, [c, s, -s, c, x, y])


def terminal_metrics(state, target_q, feature_length, orientation_period, polygon=None):
    """Compute success metrics only from the measured physical state."""

    D = shape_error(state.object_q, target_q, feature_length, orientation_period)
    position_error_m = float(np.linalg.norm(D[:2]) * feature_length)
    angle_error_rad = float(abs(D[2]))
    object_speed_mps = float(np.linalg.norm(state.object_velocity[:2]))
    angular_speed_radps = float(abs(state.object_angular_velocity[2]))
    pusher_speed_mps = float(np.linalg.norm(state.pusher_velocity[:2]))
    result = dict(
        D=D,
        position_error_m=position_error_m,
        angle_error_rad=angle_error_rad,
        object_speed_mps=object_speed_mps,
        angular_speed_radps=angular_speed_radps,
        pusher_speed_mps=pusher_speed_mps,
        roll_rad=float(state.roll),
        pitch_rad=float(state.pitch),
        contact=bool(state.contact.active),
    )
    if polygon is not None:
        actual = _transformed_polygon(polygon, state.object_q)
        goal = _transformed_polygon(polygon, target_q)
        intersection = actual.intersection(goal).area
        goal_area = max(goal.area, 1e-12)
        union = actual.union(goal).area
        result["coverage"] = float(intersection / goal_area)
        result["iou"] = float(intersection / max(union, 1e-12))
    return result


def success_from_metrics(metrics, position_tolerance=0.005, angle_tolerance=np.deg2rad(5.0),
                         velocity_tolerance=0.005, angular_velocity_tolerance=0.05,
                         coverage_min=0.95, pusher_velocity_tolerance=None):
    """Return the external success predicate; it never depends on Q."""

    return bool(
        metrics["position_error_m"] <= position_tolerance
        and metrics["angle_error_rad"] <= angle_tolerance
        and metrics["object_speed_mps"] <= velocity_tolerance
        and metrics["angular_speed_radps"] <= angular_velocity_tolerance
        and (
            pusher_velocity_tolerance is None
            or metrics["pusher_speed_mps"] <= pusher_velocity_tolerance
        )
        and metrics.get("coverage", coverage_min) >= coverage_min
    )


def summarize_trace(trace):
    """Create a JSON-safe summary while retaining every step in the trace."""

    if not trace:
        return {"ticks": 0, "failures": ["EMPTY_TRACE"]}
    failures = [item["failure_code"] for item in trace if item.get("failure_code")]
    return {
        "ticks": len(trace),
        "failures": failures,
        "stage_counts": {
            stage: sum(item.get("stage") == stage for item in trace)
            for stage in ("JOINT", "ATTITUDE", "POSITION", "HOLD")
        },
        "transfer_ticks": sum(item.get("execution_state") in {"RELEASE", "TRANSIT", "ENGAGE"} for item in trace),
        "feasible_ticks": sum(item.get("decision_status") == "FEASIBLE" for item in trace),
    }


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        scalar = value.item()
        return None if isinstance(scalar, float) and not math.isfinite(scalar) else scalar
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
