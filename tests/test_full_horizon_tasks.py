"""Independent physical/controller and sustained-success checks, CPU only."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m_learning_suite.task_runtime import TASKS, audit_task, make_task


def control(task, observation, M):
    error, context = task.build_error(observation)
    return task.calculate_action_from_M(M, error, context=context).detach().numpy()


def test_static_success_runs_full_horizon_and_holds_one_second():
    task = make_task("point_mass", {"max_steps": 40})
    obs, _ = task.reset(seed=1, options={"state": np.zeros(4)})
    M = torch.eye(4, dtype=torch.float64)
    for step in range(40):
        obs, cost, terminal, truncated, info = task.step(control(task, obs, M))
        assert cost == 0 and not truncated
        assert terminal == (step == 39)
        assert info["success"] == (step == 39)
    audit = task.get_episode_audit()
    assert audit["success"] and audit["hold_seconds"] == pytest.approx(1.6)
    assert audit["required_hold_seconds"] == pytest.approx(1.)
    assert audit["arrival_time"] == pytest.approx(1.)
    assert audit["settling_time"] == 0 and len(audit["trace"]) == 40
    assert audit["metric_control_steps"] == 40
    with pytest.raises(RuntimeError, match="Reset"):
        task.step([0., 0.])


def test_early_transient_hold_does_not_count_as_final_success():
    task = make_task("point_mass", {"max_steps": 40})
    task.reset(seed=1, options={"state": np.zeros(4)})
    for step in range(40):
        _, _, terminal, truncated, info = task.step([0., 0.] if step < 30 else [6., 0.])
        assert not terminal
        assert truncated == (step == 39)
    audit = task.get_episode_audit()
    assert audit["first_hold_confirmation_time"] == pytest.approx(1.)
    assert audit["hold_seconds"] == 0 and not audit["success"]
    assert audit["arrival_time"] == audit["horizon_seconds"]


@pytest.mark.parametrize("env_id,kwargs", [("unicycle_tracking", {}),
    ("multirotor_tracking", {"trajectory": "circle"})])
def test_tracking_horizon_covers_two_periods_and_requires_final_period(env_id, kwargs):
    task = make_task(env_id, kwargs)
    assert task.horizon_seconds >= 2 * task.reference_period - 1e-12
    assert task.required_hold_seconds >= task.reference_period - 1e-12
    assert task.required_hold_seconds < task.reference_period + task.dt + 1e-12


def test_metric_change_rejected_at_real_execution_not_differentiable_query():
    task = make_task("point_mass", {"max_steps": 40})
    obs, _ = task.reset(seed=1)
    identity = torch.eye(4, dtype=torch.float64)
    obs, *_ = task.step(control(task, obs, identity))
    changed = identity.clone(); changed[0, 0] += .1
    proposed = control(task, obs, changed)
    with pytest.raises(ValueError, match="M changed"):
        task.step(proposed)
    assert task.get_episode_audit()["physical_steps"] == 1


def test_physical_failure_remains_early_termination_with_capped_arrival():
    task = make_task("cartpole")
    task.reset(seed=1, options={"state": np.array([2.5, 0., 0., 0.])})
    _, _, terminal, truncated, info = task.step([0.])
    assert terminal and not truncated and info["failure"] and not info["success"]
    audit = task.get_episode_audit()
    assert audit["physical_steps"] == 1 and audit["arrival_time"] == task.horizon_seconds


@pytest.mark.parametrize("env_id,kwargs", [(name, {}) for name in TASKS] +
    [("multirotor_tracking", {"trajectory": "circle"})])
def test_real_M_gradient_numpy_physics_pairing_and_full_episode(env_id, kwargs):
    result = audit_task(env_id, kwargs)
    assert result["passed"], result
    assert result["action_change_norm"] > 1e-8
    assert result["independent_step_max_abs"] < 1e-10


def test_same_seed_fixed_M_produces_exact_paired_full_trace():
    left = make_task("point_mass", {"max_steps": 35})
    right = make_task("point_mass", {"max_steps": 35})
    M = torch.eye(4, dtype=torch.float64)
    obs_left, _ = left.reset(seed=731)
    obs_right, _ = right.reset(seed=731)
    for _ in range(left.max_steps):
        obs_left, *_ = left.step(control(left, obs_left, M))
        obs_right, *_ = right.step(control(right, obs_right, M))
    assert left.get_episode_audit() == right.get_episode_audit()
