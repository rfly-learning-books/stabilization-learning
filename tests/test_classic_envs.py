"""Closed-loop and gradient checks for the small control tasks."""

import numpy as np
import pytest
import torch

from stabilization_learning.envs.classic import (
    CartPoleEnv,
    CoupledLinearEnv,
    PendulumEnv,
    PointMassEnv,
)


TASKS = (CoupledLinearEnv, PointMassEnv, PendulumEnv, CartPoleEnv)


def _control(env, state, M):
    e = env.tensor_error(state)
    context = {"state": state, "goal": torch.as_tensor(env.goal, dtype=state.dtype)}
    return env.calculate_action_from_M(M, e, context=context)


@pytest.mark.parametrize("task", TASKS)
def test_gym_reset_and_step(task):
    env = task()
    obs, info = env.reset(seed=17)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)
    env.close()


@pytest.mark.parametrize("task", TASKS)
def test_metric_changes_action_and_preserves_finite_difference_gradient(task):
    env = task()
    n = len(env.goal)
    state = torch.linspace(0.025, 0.07, n, dtype=torch.float64)
    identity = torch.eye(n, dtype=torch.float64)
    direction = torch.ones((n, n), dtype=torch.float64) / n
    scale = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    action = _control(env, state, identity + scale * direction)
    baseline = _control(env, state, identity)
    assert torch.linalg.vector_norm(action - baseline) > 1e-6
    # A scalar directional derivative tests the exact physical action, including
    # its final bounds, rather than just the unconstrained controller expression.
    weights = torch.arange(1, action.numel() + 1, dtype=torch.float64)
    value = (weights * action).sum()
    derivative, = torch.autograd.grad(value, scale)
    h = 1e-5
    plus = _control(env, state, identity + (scale.detach() + h) * direction)
    minus = _control(env, state, identity + (scale.detach() - h) * direction)
    finite_difference = (weights * (plus - minus)).sum() / (2 * h)
    assert torch.isfinite(derivative)
    assert abs(derivative) > 1e-6
    torch.testing.assert_close(derivative, finite_difference, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("task,start", [
    (CoupledLinearEnv, [0.5, -0.35, 0.1, -0.1]),
    (PointMassEnv, [0.5, -0.35, 0.1, -0.1]),
    (PendulumEnv, [0.2, -0.1]),
    (CartPoleEnv, [0.15, 0.0, 0.08, 0.0]),
])
def test_identity_metric_local_closed_loop_converges(task, start):
    env = task()
    state = torch.tensor(start, dtype=torch.float64)
    M = torch.eye(len(start), dtype=torch.float64)
    initial_error = torch.linalg.vector_norm(env.tensor_error(state)).item()
    with torch.no_grad():
        for _ in range(round(10.0 / env.dt)):
            state = env.torch_step(state, _control(env, state, M))
            assert torch.isfinite(state).all()
    final_error = torch.linalg.vector_norm(env.tensor_error(state)).item()
    assert final_error < 0.02
    assert final_error < 0.1 * initial_error


@pytest.mark.parametrize("task", TASKS)
def test_batched_dynamics_and_action_gradient(task):
    env = task()
    n = len(env.goal)
    state = torch.full((3, n), 0.03, dtype=torch.float64, requires_grad=True)
    M = torch.eye(n, dtype=torch.float64).repeat(3, 1, 1).requires_grad_()
    action = _control(env, state, M)
    next_state = env.torch_step(state, action)
    assert next_state.shape == state.shape
    assert action.shape == (3, *env.action_space.shape)
    dm, ds = torch.autograd.grad(next_state.square().sum(), (M, state))
    assert torch.isfinite(dm).all() and torch.isfinite(ds).all()
    assert dm.norm() > 0
    assert ds.norm() > 0


def test_point_mass_moving_reference_feedforward_is_exact_at_zero_error():
    env = PointMassEnv()
    env.set_reference([0.3, -0.2], [0.2, 0.1], [0.4, -0.3])
    state = torch.tensor(env.goal, dtype=torch.float64)
    M = torch.eye(4, dtype=torch.float64)
    action = _control(env, state, M)
    torch.testing.assert_close(action, torch.tensor([0.4, -0.3], dtype=torch.float64))
    next_state = env.torch_step(state, action)
    expected_position = state[:2] + env.dt * state[2:] + 0.5 * env.dt**2 * action
    torch.testing.assert_close(next_state[:2], expected_position)


@pytest.mark.parametrize("task,angle_index", [(PendulumEnv, 0), (CartPoleEnv, 2)])
def test_angle_error_wraps(task, angle_index):
    env = task()
    state = torch.zeros(len(env.goal), dtype=torch.float64)
    state[angle_index] = 2 * torch.pi + 0.1
    assert env.tensor_error(state)[angle_index].item() == pytest.approx(0.1)


@pytest.mark.parametrize("task", TASKS)
def test_success_requires_hold_and_returns_positive_integrated_cost(task):
    env = task()
    obs, info = env.reset(seed=1, options={"state": np.zeros(len(env.goal))})
    assert info["in_goal"] and not info["success"]
    for i in range(5):
        obs, cost, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert cost == pytest.approx(0.0)
        assert info["cost_convention"] == "positive_integral"
        assert terminated == (i == 4)
        assert not truncated
    with pytest.raises(RuntimeError):
        env.step(np.zeros(env.action_space.shape))


@pytest.mark.parametrize("task", TASKS)
def test_nonzero_cost_and_time_limit_are_not_success(task):
    env = task(max_steps=1)
    state = np.full(len(env.goal), 0.1)
    env.reset(seed=1, options={"state": state})
    _, cost, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
    assert cost == pytest.approx(env.dt * np.sum(state**2))
    assert cost > 0
    assert truncated and not terminated
    assert not info["success"]


@pytest.mark.parametrize("task", TASKS)
def test_nonfinite_state_is_failure(task):
    env = task()
    state = np.zeros(len(env.goal))
    state[0] = np.nan
    assert env.is_failure(state)
