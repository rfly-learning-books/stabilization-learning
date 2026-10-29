"""Small CPU checks of independent result auditing, not learning experiments."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_book_m_results import audit_episode, local_closed_loop, independent_errors
from m_learning_suite.task_runtime import make_task
from stabilization_learning.wrappers import rollout_metric


def recorded_point_mass():
    task = make_task("point_mass",{"max_steps":40,"initial_scale":0.001})
    initial = task.reset(seed=71)[0]
    M = torch.eye(4,dtype=torch.float64)
    row = rollout_metric(task,M,initial,certificate_weight=.1,decay=.05,save_trajectory=True)
    trace = task.trace
    arrays = {"M":M.numpy(),"states":np.array([x["state_before"] for x in trace]),
        "following":np.array([x["state_after"] for x in trace]),
        "actions":np.array([x["action"] for x in trace]),
        "errors":np.array([x["error_before"] for x in trace]),
        "errors_after":np.array([x["error_after"] for x in trace]),
        "V_before":np.array([x["V_before"] for x in row["trajectory"]]),
        "V_after":np.array([x["V_after"] for x in row["trajectory"]]),
        "costs":np.array([x["cost"] for x in trace]),
        "physical_error_before":np.array([x["physical_error_before"] for x in trace]),
        "physical_error_after":np.array([x["physical_error_after"] for x in trace]),
        "in_goal_before":np.array([x["in_goal_before"] for x in trace]),
        "in_goal_after":np.array([x["in_goal_after"] for x in trace]),
        "dt":np.array(task.dt),"goal":task.goal.copy(),
        "required_hold_seconds":np.array(task.required_hold_seconds),
        "times":np.arange(1,len(trace)+1)*task.dt}
    residual = arrays["V_after"]-arrays["V_before"]+.05*task.dt*np.sum(arrays["errors"]**2,axis=1)
    arrays["decay_residual"] = residual
    audit = task.get_episode_audit()
    for key in ("arrival_time","settling_time","hold_seconds","required_hold_seconds","horizon_seconds"):
        row[key] = audit[key]
    row.update(seed=71,method="identity",actual_decay_violation_fraction=float(np.mean(residual>1e-8)),
        actual_decay_violation_max=float(max(0.,residual.max())),
        action_total_variation=float(np.linalg.norm(np.diff(arrays["actions"],axis=0),axis=1).sum()),
        M_eigen_min=1.,M_eigen_max=1.,M_trace=4.)
    return task,arrays,row


def test_recorded_trajectory_and_sustained_arrival_recompute():
    task,arrays,row = recorded_point_mass()
    try:
        report = audit_episode(task,arrays,row,decay=.05,certificate_weight=.1)
        assert report["passed"],report
        assert report["measurements"]["physical_steps"] == 40
        assert report["measurements"]["arrival_time"] == 1.0
        assert report["measurements"]["arrival_time"] < task.horizon_seconds
    finally:
        task.close()


@pytest.mark.parametrize("field",["actions","following","V_after","physical_error_after"])
def test_deliberate_corruption_is_detected(field):
    task,arrays,row = recorded_point_mass()
    try:
        damaged = deepcopy(arrays)
        damaged[field].flat[0] += .01
        assert not audit_episode(task,damaged,row,decay=.05,certificate_weight=.1)["passed"]
    finally:
        task.close()


def test_point_mass_numerical_jacobian_matches_exact_discrete_map():
    task = make_task("point_mass")
    try:
        report = local_closed_loop(task,np.eye(4))
        dt = task.dt
        A = np.block([[np.eye(2),dt*np.eye(2)],[np.zeros((2,2)),np.eye(2)]])
        B = np.vstack((.5*dt**2*np.eye(2),dt*np.eye(2)))
        exact = A-B@task.inner_env._K.numpy()
        np.testing.assert_allclose(report["A_cl_discrete"],exact,atol=1e-10,rtol=1e-10)
        assert report["numerically_meaningful"]
        assert report["spectral_radius_below_one"]
        # Stable poles do not make Euclidean V a decreasing certificate.
        assert not report["quadratic_matrix_negative_definite"]
    finally:
        task.close()


@pytest.mark.parametrize("env,kwargs",[("unicycle_tracking",{}),("multirotor_tracking",{"trajectory":"circle"})])
def test_moving_reference_explicitly_excludes_static_equilibrium_check(env,kwargs):
    task = make_task(env,kwargs)
    try:
        report = local_closed_loop(task,np.eye(task.error_dim))
        assert not report["applicable"]
        state = task.reset(seed=42)[0]
        np.testing.assert_allclose(independent_errors(task,state),task.tensor_error(torch.as_tensor(state)).numpy(),atol=1e-12)
    finally:
        task.close()
