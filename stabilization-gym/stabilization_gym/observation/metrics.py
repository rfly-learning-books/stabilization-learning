"""Truth-based evaluation, deliberately outside deployment observation records."""
from __future__ import annotations
import numpy as np


def final_hold(times, within, horizon, hold_seconds, failed=False):
    times = np.asarray(times, dtype=float); within = np.asarray(within, dtype=bool)
    complete = bool(len(times) and abs(times[-1]-horizon) <= 1e-8 and not failed)
    if not len(times) or not within[-1]:
        return {"success": False, "arrival_time": None, "hold_confirmation_time": None,
                "final_hold_seconds": 0., "failure_capped_arrival_time": float(horizon)}
    bad = np.flatnonzero(~within)
    start = int(bad[-1]+1) if len(bad) else 0
    duration = float(times[-1]-times[start])
    success = bool(complete and duration+1e-10 >= hold_seconds)
    return {"success": success, "arrival_time": float(times[start]) if success else None,
            "hold_confirmation_time": float(times[start]+hold_seconds) if success else None,
            "final_hold_seconds": duration,
            "failure_capped_arrival_time": float(times[start]) if success else float(horizon)}


def information_windows(model, records, seconds=.5):
    """Non-overlapping finite-window linearized Gramians, estimated trajectory.

    Each window starts at its first posterior and predicts a nominal trajectory
    using known inputs, without further measurement corrections. Thus all
    transition Jacobians share consistent tangent frames. Actual visibility
    masks are retained. This is not a nonlinear observability proof.
    """
    n = model.spec.error_dim; scale = np.asarray(model.spec.error_scale)
    size = max(2, round(seconds/model.spec.dt)+1)
    result = []
    for start in range(0, len(records)-size+1, size-1):
        gramian = np.zeros((n, n)); transition = np.eye(n)
        reference = records[start]["estimate"].copy()
        for rec in records[start:start+size]:
            obs = rec["observation"]; mask = obs["measurement_mask"].astype(bool)
            try:
                h = model.measurement_jacobian(reference)[mask]*scale[None, :]
                weights = 1/np.asarray(model.spec.measurement_scale)[mask]**2
                gramian += model.spec.dt*transition.T@h.T@(weights[:, None]*h)@transition
                f = model.transition_jacobian(reference, obs["known_input"])*scale[None, :]/scale[:, None]
                reference = model.predict(reference, obs["known_input"])
                transition = f@transition
            except (ValueError, np.linalg.LinAlgError):
                result.append({"time": rec["time"], "valid": False, "reason": "model_domain_exit"})
                break
        else:
            eig = np.maximum(np.linalg.eigvalsh((gramian+gramian.T)/2), 0.)
            threshold = max(1e-12, float(eig.max())*1e-9)
            result.append({"time": records[start]["time"], "valid": True,
                "rank": int(np.sum(eig>threshold)), "dimension": n,
                "minimum_eigenvalue": float(eig.min()), "maximum_eigenvalue": float(eig.max()),
                "rank_tolerance": threshold})
    return result


def summarize_episode(model, records, certificate, *, certificate_version,
                      alpha_per_second=.1, noise_allowance_per_second=0., tolerance=1e-9,
                      failure_tail_cost_rate=25.):
    """certificate(truth, estimate, expected_version) is an explicit offline hook."""
    if not records:
        raise ValueError("Cannot score an empty episode")
    spec = model.spec; times = np.array([r["time"] for r in records])
    if np.any(np.diff(times) <= 0) or abs(times[0]) > 1e-10:
        raise ValueError("Expected ordered complete timestamp prefix from zero")
    errors = np.array([model.error(r["truth"], r["estimate"]) for r in records])
    normalized = errors/np.asarray(spec.error_scale)
    groups = np.array([model.groups(e) for e in errors])
    squared = np.sum(normalized**2, axis=1)
    failed = bool(any(r["failure"] for r in records))
    complete = bool(records[-1]["complete"] and not failed and abs(times[-1]-spec.horizon)<1e-8)
    physical_integral = float(np.sum(squared[:-1]*np.diff(times)))
    unobserved_tail = max(0., spec.horizon-float(times[-1]))
    tail = unobserved_tail*max(failure_tail_cost_rate, float(squared[-1])) if not complete else 0.
    hold = final_hold(times, np.all(groups<=np.asarray(spec.group_tolerances), axis=1),
                      spec.horizon, spec.hold_seconds, failed=not complete)
    values = np.array([certificate(r["truth"], r["estimate"], certificate_version) for r in records])
    deltas = np.diff(values); durations = np.diff(times)
    residual = deltas+alpha_per_second*durations*squared[:-1]
    robust_residual = residual-noise_allowance_per_second*durations
    count = len(deltas)
    masks = np.array([r["observation"]["measurement_mask"].astype(bool) for r in records])
    innovation_sum = 0.; innovation_count = 0
    raw_innovation = []
    for r, mask in zip(records, masks):
        innovation = r["observation"]["measurement"][mask]-model.measure(r["observation"]["estimate"])[mask]
        scaled = innovation/np.asarray(spec.measurement_scale)[mask]
        innovation_sum += float(scaled@scaled); innovation_count += len(scaled)
        raw_innovation.extend(innovation.tolist())
    recovery = []
    missing = ~masks.any(axis=1)
    for end in np.flatnonzero(missing[:-1] & ~missing[1:])+1:
        suffix = final_hold(times[end:]-times[end], np.all(groups[end:]<=np.asarray(spec.group_tolerances), axis=1),
                            spec.horizon-times[end], spec.hold_seconds, failed=not complete)
        recovery.append({"measurement_return_time": float(times[end]), "recovered": suffix["success"],
                         "recovery_seconds": suffix["arrival_time"], "hold_confirmation_seconds": suffix["hold_confirmation_time"]})
    window_steps = max(1, round(.5/spec.dt))
    window_residuals = [float(values[k+window_steps]-values[k]
        +alpha_per_second*np.sum(durations[k:k+window_steps]*squared[k:k+window_steps]))
        for k in range(max(0, len(values)-window_steps))]
    information = information_windows(model, records)
    return {**hold, "complete": complete, "failure": failed, "samples": len(records),
        "expected_samples": round(spec.horizon/spec.dt)+1, "recorded_duration": float(times[-1]),
        "failure_reason": next((r["failure_reason"] for r in records if r["failure"]), None),
        "component_rmse": dict(zip(spec.error_names, np.sqrt(np.mean(errors**2, axis=0)).tolist())),
        "component_units": dict(zip(spec.error_names, spec.error_units)),
        "group_peak": dict(zip(spec.group_names, np.max(groups, axis=0).tolist())),
        "group_final": dict(zip(spec.group_names, groups[-1].tolist())),
        "normalized_error_integral_recorded": physical_integral,
        "failure_tail_penalty": float(tail), "fixed_horizon_error_cost": physical_integral+tail,
        "fixed_horizon_cost_is_penalized_not_unobserved_physics": True,
        "correction_norm_mean": float(np.mean([np.linalg.norm(r["correction"]/np.asarray(spec.error_scale)) for r in records])),
        "innovation_normalized_rmse": float(np.sqrt(innovation_sum/innovation_count)) if innovation_count else None,
        "innovation_raw_rmse": float(np.sqrt(np.mean(np.asarray(raw_innovation)**2))) if raw_innovation else None,
        "innovation_channel_count": innovation_count,
        "missing_measurement_samples": int(missing.sum()), "occlusion_recovery": recovery,
        "certificate": {"parameter_version": certificate_version, "claim": "sampled_candidate_only",
            "alpha_per_second": alpha_per_second, "tolerance": tolerance,
            "noise_allowance_per_second": noise_allowance_per_second,
            "noise_allowance_is_proven_disturbance_bound": False,
            "initial_post_value": float(values[0]), "final_value": float(values[-1]),
            "transitions": count, "raw_increase_count": int(np.sum(deltas>tolerance)),
            "descent_violation_count": int(np.sum(residual>tolerance)),
            "descent_violation_fraction": float(np.mean(residual>tolerance)) if count else None,
            "allowance_adjusted_violation_count": int(np.sum(robust_residual>tolerance)),
            "maximum_descent_residual": float(residual.max()) if count else None,
            "window_seconds": .5, "window_count": len(window_residuals),
            "window_violation_count": sum(v>tolerance for v in window_residuals),
            "full_successor_metric_recomputed": True},
        "information_windows": information,
        "information_deficient_windows": sum(r.get("rank", 0)<spec.error_dim for r in information),
        "rotation_chart_margin_rad": float(np.pi-np.max(groups[:, 1])) if spec.name == "cooperative_pose" else None}
