"""Behavioral equivalence, injected model use and online-information isolation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stabilization_gym.tasks_v2.state_observer import (
    make_state_control, ObservationReplayEnv, ObserverTaskAdapter)
from stabilization_gym.interfaces.certificate_v2 import CertificateEvaluator, VersionBinding
from stabilization_gym.observation import make_observation_task
from stabilization_learning.observation import make_observer


def binding():
    fields = {key: "test" for key in VersionBinding.__dataclass_fields__}
    fields["metric_scope"] = "state_dependent"
    return VersionBinding(**fields)


@pytest.mark.parametrize("name", ["coupled_linear", "point_mass", "pendulum", "cartpole"])
def test_state_action_and_physical_step_preserve_existing_model(name):
    env, adapter = make_state_control(name)
    obs, _ = env.reset(seed=17)
    state = torch.tensor(obs, dtype=torch.float64)
    e = adapter.model.error(state)
    M = torch.tensor(adapter.model.P0, dtype=torch.float64)
    certificate = CertificateEvaluator(lambda error, context: M, binding())
    context = adapter.make_context(obs, binding=certificate.binding)
    plan = adapter.construct_action(context, certificate)
    expected = adapter.model.control(M, e, state=state)
    assert torch.equal(plan.action, expected)
    predicted = adapter.model.step_model(state, expected).numpy()
    next_obs, reward, _, _, info = env.step(plan.action)
    np.testing.assert_allclose(next_obs, predicted, rtol=0, atol=1e-13)
    assert reward == -info["physical_cost"]
    assert info["dt"] == env.unwrapped.dt
    assert not hasattr(env, "optimizer")
    assert adapter.capabilities.prediction_gradient
    assert not adapter.capabilities.physical_step_gradient


def test_state_certificate_changes_action_and_retains_parameter_gradient():
    env, adapter = make_state_control("point_mass")
    obs, _ = env.reset(seed=2)
    scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    certificate = CertificateEvaluator(lambda error, context: scale*error.new_tensor(adapter.model.P0), binding())
    ctx = adapter.make_context(obs, binding=certificate.binding)
    action = adapter.construct_action(ctx, certificate).action
    gradient, = torch.autograd.grad(action.square().sum(), scale)
    assert gradient.abs() > 1e-8
    assert ctx.metric_context is None  # No duplicate independent copy of state error.


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
@pytest.mark.parametrize("scenario", ["nominal", "noise", "initial_offset"])
def test_observer_split_preserves_legacy_correction_prediction_and_sensor_trajectory(task, scenario):
    old = make_observation_task(task)
    old_obs, _ = old.reset(options={"scenario": scenario, "index": 4})
    metric_observer = make_observer("variable_M_probe", old.model)
    physical = ObservationReplayEnv(task)
    obs, _ = physical.reset(options={"scenario": scenario, "index": 4})
    adapter = ObserverTaskAdapter(physical.model)
    memory = adapter.initial_memory(scenario=scenario)
    for _ in range(6):
        assert "estimate" not in obs and "truth" not in obs
        assert physical.observation_space.contains(obs)
        ctx = adapter.make_context(obs, memory=memory)
        expected = metric_observer.correction(old_obs)
        plan = adapter.construct_action(ctx, metric_observer.metric)
        np.testing.assert_allclose(plan.action.detach().numpy(), expected, atol=1e-11, rtol=1e-11)
        labels = physical.supervision()
        scored = adapter.evaluate_supervision(labels, ctx, plan)
        np.testing.assert_array_equal(labels["truth"], old.supervision()["truth"])
        old_obs, old_reward, old_term, old_trunc, _ = old.step(expected)
        obs, _, _, _, _ = physical.step(plan.diagnostics["physical_action"])
        memory = plan.memory
        np.testing.assert_allclose(memory["estimate"], old_obs["estimate"], atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(obs["measurement"], old_obs["measurement"])
        assert scored["reward"] == pytest.approx(old_reward)
        assert not old_term and not old_trunc


def test_prediction_replacement_changes_estimate_without_changing_physics():
    physical = ObservationReplayEnv("motor_speed")
    obs, _ = physical.reset()
    normal = ObserverTaskAdapter(physical.model)
    calls = []
    def replacement(state, control, dt):
        calls.append((state.copy(), control.copy(), dt))
        return physical.model.predict(state, control, dt) + np.array([.02, 0.])
    changed = ObserverTaskAdapter(physical.model, predict=replacement)
    metric = make_observer("fixed_M", physical.model).metric
    a = normal.construct_action(normal.make_context(obs, memory=normal.initial_memory()), metric)
    b = changed.construct_action(changed.make_context(obs, memory=changed.initial_memory()), metric)
    assert len(calls) == 1
    np.testing.assert_allclose(b.memory["estimate"]-a.memory["estimate"], [.02, 0.], atol=1e-13)
    np.testing.assert_array_equal(a.action.detach().numpy(), b.action.detach().numpy())
    np.testing.assert_array_equal(a.diagnostics["physical_action"], b.diagnostics["physical_action"])


def test_measurement_model_replacement_changes_online_correction():
    physical = ObservationReplayEnv("motor_speed")
    obs, _ = physical.reset()
    metric = make_observer("fixed_M", physical.model).metric
    a = ObserverTaskAdapter(physical.model)
    b = ObserverTaskAdapter(physical.model, measure=lambda state: physical.model.measure(state)+.2)
    pa = a.construct_action(a.make_context(obs, memory=a.initial_memory()), metric)
    pb = b.construct_action(b.make_context(obs, memory=b.initial_memory()), metric)
    assert torch.linalg.vector_norm(pa.action-pb.action) > 1e-7


def test_truth_is_rejected_and_online_matrix_only_sees_estimate():
    physical = ObservationReplayEnv("motor_speed")
    obs, _ = physical.reset()
    adapter = ObserverTaskAdapter(physical.model)
    memory = adapter.initial_memory()
    with pytest.raises(ValueError, match="truth"):
        adapter.make_context(dict(obs, truth=physical.supervision()["truth"]), memory=memory)
    seen = []
    matrix = torch.eye(2, dtype=torch.float64, requires_grad=True)
    class SpyCertificate:
        def online_matrix(self, estimate, *, binding=None):
            seen.append(estimate.detach().numpy().copy())
            return matrix
        def evaluate(self, *args, **kwargs):
            raise AssertionError("Unknown state error must never be used in online V")
    ctx = adapter.make_context(obs, memory=memory)
    assert ctx.error is None
    plan = adapter.construct_action(ctx, SpyCertificate())
    np.testing.assert_array_equal(seen[0], memory["estimate"])
    assert not np.array_equal(seen[0], physical.supervision()["truth"])
    gradient, = torch.autograd.grad(plan.action.square().sum(), matrix)
    assert gradient.norm() > 0
    assert set(plan.memory) == {"estimate", "time"}


def test_missing_measurements_yield_zero_correction_and_still_predict():
    physical = ObservationReplayEnv("motor_speed")
    obs, _ = physical.reset()
    obs["measurement_mask"][:] = 0
    obs["confidence"][:] = 0
    obs["measurement"][:] = 0
    adapter = ObserverTaskAdapter(physical.model)
    metric = make_observer("fixed_M", physical.model).metric
    plan = adapter.construct_action(adapter.make_context(obs, memory=adapter.initial_memory()), metric)
    assert torch.equal(plan.action, torch.zeros(2, dtype=torch.float64))
    assert not np.array_equal(plan.memory["estimate"], adapter.initial_memory()["estimate"])


def test_replay_refuses_changed_forcing_or_estimator_correction():
    physical = ObservationReplayEnv("motor_speed")
    obs, _ = physical.reset()
    with pytest.raises(ValueError):
        physical.step(np.zeros(2))
    with pytest.raises(ValueError, match="declared known input"):
        physical.step(obs["known_input"]+.1)
    assert physical._index == 0


def test_complete_sensor_replay_keeps_final_endpoint_and_zero_duration_boundary():
    physical = ObservationReplayEnv("motor_speed")
    obs, _ = physical.reset()
    adapter = ObserverTaskAdapter(physical.model)
    metric = make_observer("fixed_M", physical.model).metric
    memory = adapter.initial_memory()
    recorded_times = []
    total_dt = 0.
    while True:
        recorded_times.append(float(obs["time"][0]))
        ctx = adapter.make_context(obs, memory=memory)
        plan = adapter.construct_action(ctx, metric)
        scored = adapter.evaluate_supervision(physical.supervision(), ctx, plan)
        obs, _, terminated, truncated, info = physical.step(plan.diagnostics["physical_action"])
        total_dt += info["dt"]
        memory = plan.memory
        if terminated or truncated:
            assert truncated and not terminated
            assert scored["complete"] and scored["physical_cost"] == 0.
            assert info["dt"] == 0.
            break
    assert recorded_times[-1] == physical.model.spec.horizon
    assert total_dt == pytest.approx(physical.model.spec.horizon)
    assert len(recorded_times) == round(physical.model.spec.horizon/physical.model.spec.dt)+1
    with pytest.raises(RuntimeError):
        physical.step(obs["known_input"])
