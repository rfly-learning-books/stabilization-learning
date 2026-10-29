"""Rollout helpers for state-conditioned metric control/observation."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch

from .certificate import quadratic


@torch.no_grad()
def rollout_dc_motor(env, actor, *, seed: int, control: str = "analytic",
                     alpha: float = 0.05, clf_points: int = 21,
                     reset_options: dict[str, Any] | None = None) -> dict[str, Any]:
    if control not in {"analytic", "clf_candidate"}:
        raise ValueError("control must be 'analytic' or 'clf_candidate'")
    observation, info = env.reset(seed=int(seed), options=reset_options)
    total_cost = 0.0
    steps = 0
    records = []
    selected_feasible = 0
    selected_infeasible = 0
    while True:
        context = env.metric_context(observation)
        M = actor(context)
        error, control_context = env.build_error(observation)
        if control == "analytic":
            action = env.controller_action(M, error, control_context)
            audit = {"mode": "analytic"}
        else:
            action, audit = env.clf_candidate_action(actor, observation, M, alpha=alpha, points=clf_points)
            if audit["feasible_count"] > 0:
                selected_feasible += 1
            else:
                selected_infeasible += 1
        observation, cost, terminated, truncated, info = env.step(action)
        total_cost += float(cost)
        steps += 1
        records.append({"step": steps, "cost": float(cost), "audit": audit,
                        "info": {k: v for k, v in info.items() if isinstance(v, (bool, int, float, str))}})
        if terminated or truncated:
            break
    audit = env.get_episode_audit()
    trace = audit.get("metric_trace", [])
    residuals = [row["same_M_one_step_residual"] for row in trace if row.get("same_M_one_step_residual") is not None]
    rises = sum(1 for row in trace if row.get("V_after") is not None and row.get("V_before") is not None and row["V_after"] > row["V_before"] + 1e-8)
    return {
        "seed": int(seed),
        "control": control,
        "success": bool(info.get("success", False)),
        "failure": bool(info.get("failure", False)),
        "steps": steps,
        "total_cost": float(total_cost),
        "terminal_physical_error": float(info.get("physical_error", np.nan)),
        "hold_seconds": float(info.get("hold_seconds", 0.0)),
        "V_rise_steps": int(rises),
        "descent_residual_sum": float(sum(max(0.0, x) for x in residuals)),
        "descent_residual_min": float(min(residuals)) if residuals else None,
        "clf_feasible_selected": selected_feasible,
        "clf_infeasible_selected": selected_infeasible,
        "trace": trace if trace else records,
    }
