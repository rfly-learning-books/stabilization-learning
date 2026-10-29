"""Physical and differentiable-controller checks for the robot environments."""

import numpy as np
import pytest
import torch

from stabilization_learning.envs.robotics import (
    MultirotorTrackingEnv,
    TwoLinkArmEnv,
    UnicycleTrackingEnv,
)


CASES = [
    (TwoLinkArmEnv, [0.40, -0.53, 0.02, -0.03], 4),
    (UnicycleTrackingEnv, [0.10, -0.08, 0.08, 0.0], 3),
    (MultirotorTrackingEnv, [0.10, 0.90, 0.01, -0.02, 0.03, 0.01, 0.0], 6),
]


def control(env, state, metric):
    error = env.tensor_error(state)
    return env.calculate_action_from_M(
        metric, error,
        context={"state": state, "goal": state.new_tensor(env.goal)},
    )


@pytest.mark.parametrize("factory,initial,error_dim", CASES)
def test_reset_physical_step_and_action_limits(factory, initial, error_dim):
    env = factory()
    observation, info = env.reset(seed=7)
    assert isinstance(info, dict)
    assert np.isfinite(env.state).all()
    state = torch.tensor(initial, dtype=torch.float64)
    metric = 100.0 * torch.eye(error_dim, dtype=state.dtype)
    action = control(env, state, metric)
    assert action.shape == (2,)
    assert torch.isfinite(action).all()
    assert np.all(action.detach().numpy() <= env.action_space.high + 1e-6)
    assert np.all(action.detach().numpy() >= env.action_space.low - 1e-6)
    result = env.step(action.detach().numpy())
    assert len(result) == 5
    assert np.isfinite(env.state).all()
    assert np.isfinite(result[1])
    env.close()


@pytest.mark.parametrize("factory,initial,error_dim", CASES)
def test_metric_changes_actions_with_correct_nonzero_gradient(factory, initial, error_dim):
    env = factory()
    state = torch.tensor(initial, dtype=torch.float64)
    metric = torch.eye(error_dim, dtype=torch.float64, requires_grad=True)
    action = control(env, state, metric)
    gradient = torch.autograd.grad(action.sum(), metric)[0]
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 1e-7
    direction = torch.zeros_like(metric)
    direction[0, 0] = 1.
    step = 1e-5
    plus = control(env, state, metric.detach() + step*direction)
    minus = control(env, state, metric.detach() - step*direction)
    derivative = (plus.sum()-minus.sum())/(2*step)
    torch.testing.assert_close(derivative, (gradient*direction).sum(), rtol=2e-4, atol=2e-6)
    changed = control(env, state, metric.detach() + .4*direction)
    assert torch.linalg.vector_norm(changed-action.detach()) > 1e-5


@pytest.mark.parametrize("factory,initial,error_dim", CASES)
def test_batched_physics_and_controller_are_differentiable(factory, initial, error_dim):
    env = factory()
    state = torch.tensor([initial, initial], dtype=torch.float64, requires_grad=True)
    metric = torch.eye(error_dim, dtype=torch.float64).repeat(2, 1, 1).requires_grad_()
    action = control(env, state, metric)
    next_state = env.torch_step(state, action)
    assert next_state.shape == state.shape
    state_grad, metric_grad = torch.autograd.grad(next_state.square().sum(), (state, metric))
    assert torch.isfinite(state_grad).all()
    assert torch.isfinite(metric_grad).all()
    assert torch.linalg.vector_norm(metric_grad) > 1e-7
    independent_action = action.detach().requires_grad_()
    action_grad = torch.autograd.grad(env.torch_step(state.detach(), independent_action).sum(),
                                      independent_action)[0]
    assert torch.isfinite(action_grad).all()
    assert torch.linalg.vector_norm(action_grad) > 1e-7


@pytest.mark.parametrize("factory,initial,error_dim", CASES)
def test_identity_metric_stabilizes_modest_initial_errors(factory, initial, error_dim):
    env = factory()
    state = torch.tensor(initial, dtype=torch.float64)
    metric = torch.eye(error_dim, dtype=state.dtype)
    initial_error = torch.linalg.vector_norm(env.tensor_error(state))
    with torch.no_grad():
        for _ in range(400):
            state = env.torch_step(state, control(env, state, metric))
            assert not env.is_failure(state.numpy())
    final_error = torch.linalg.vector_norm(env.tensor_error(state))
    assert final_error < .15 * initial_error
    assert env.is_success(state.numpy())


def test_arm_mechanics_are_coupled_positive_definite_and_balance_gravity():
    env = TwoLinkArmEnv()
    states = torch.tensor([[.3, -.45, 0., 0.], [.4, 1.4, .1, -.2]], dtype=torch.float64)
    inertia, coriolis, gravity = env.mechanics(states)
    torch.testing.assert_close(inertia, inertia.transpose(-1, -2))
    assert torch.all(torch.linalg.eigvalsh(inertia) > 0.)
    assert torch.all(inertia[..., 0, 1].abs() > .01)
    assert torch.linalg.vector_norm(coriolis[1]) > 0.
    next_state = env.torch_step(states[0], gravity[0])
    torch.testing.assert_close(next_state, states[0])


def test_planar_quadrotor_has_attitude_thrust_coupling_and_hover_equilibrium():
    env = MultirotorTrackingEnv()
    state = torch.tensor(env.goal, dtype=torch.float64)
    hover = torch.tensor([env.mass*env.gravity, 0.], dtype=state.dtype)
    torch.testing.assert_close(env.torch_step(state, hover), state)
    pitched = state.clone()
    pitched[4] = .1
    next_state = env.torch_step(pitched, hover)
    assert next_state[2] < 0.
    assert next_state[3] < 0.


@pytest.mark.parametrize("factory,phase_index", [
    (UnicycleTrackingEnv, 3),
    (lambda: MultirotorTrackingEnv(trajectory="circle"), 6),
])
def test_moving_reference_is_markov_and_has_no_hidden_clock(factory, phase_index):
    env = factory()
    env.reset(seed=4)
    state = torch.tensor(env.state, dtype=torch.float64)
    action = torch.tensor([.4, .1] if phase_index == 3 else [9.81, .01], dtype=state.dtype)
    first = env.torch_step(state, action)
    first_error = env.tensor_error(state)
    env.reset(seed=999)
    env.step(env.action_space.sample())
    torch.testing.assert_close(env.torch_step(state, action), first)
    torch.testing.assert_close(env.tensor_error(state), first_error)
    expected = state[phase_index] + env.dt*env.reference_omega
    expected = torch.atan2(torch.sin(expected), torch.cos(expected))
    torch.testing.assert_close(first[phase_index], expected)


def test_planar_quadrotor_tracks_slow_circle():
    env = MultirotorTrackingEnv(trajectory="circle")
    reference, _ = env.reference(torch.tensor(0., dtype=torch.float64))
    state = torch.cat((reference, reference.new_zeros(1)))
    state[0] += .1
    state[1] -= .1
    metric = torch.eye(6, dtype=state.dtype)
    initial_error = torch.linalg.vector_norm(env.tensor_error(state))
    with torch.no_grad():
        for _ in range(500):
            state = env.torch_step(state, control(env, state, metric))
            assert not env.is_failure(state.numpy())
    assert torch.linalg.vector_norm(env.tensor_error(state)) < .15*initial_error
    assert env.is_success(state.numpy())
