"""Explicit new aggregation revision, without changing the frozen v2 executor."""
from copy import deepcopy
import torch
from stabilization_learning.algos.pytorch.state_metric_model.learner import rollout_batch
from .costs import fixed_horizon_cost64


def rollout_batch64(task,policy,starts,**kwargs):
    rows,arrays=rollout_batch(task,policy,starts,**kwargs)
    updated=deepcopy(rows)
    fixed=fixed_horizon_cost64([r["accumulated_cost"] for r in rows],
        [r["physical_steps"] for r in rows],[r["terminal_physical_error"] for r in rows],
        [r["failure"] for r in rows],horizon=task.max_steps,dt=task.dt)
    coefficient=kwargs.get("certificate_weight",.1)
    for i,row in enumerate(updated):
        row["legacy_v2_summary"]=dict(fixed_horizon_cost=row["fixed_horizon_cost"],objective=row["objective"])
        row["fixed_horizon_cost"]=float(fixed[i])
        row["objective"]=float(fixed[i])+coefficient*row["certificate_penalty"]
        row["aggregation_version"]="float64_remaining_time_v3"
    return updated,arrays
