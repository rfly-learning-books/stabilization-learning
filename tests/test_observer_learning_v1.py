"""Parity and actual full-horizon gradient checks before authorizing long runs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pytest
import torch
from stabilization_gym.observation import make_observation_task
from stabilization_learning.observation import MetricObserver, value
from extensions.observer_learning_v1.tensor_model import TensorModel
from extensions.observer_learning_v1.learning import MetricNetwork, FrozenNetworkMetric, episode_data, batch_data, rollout


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_tensor_models_match_numpy_physics_output_jacobian_and_group_errors(task):
    model = make_observation_task(task).model; tensor = TensorModel(model)
    x = model.initial_truth(np.random.default_rng(42)); u = model.known_input(.2, .1)
    delta = np.linspace(-.01, .01, model.spec.error_dim)
    state = tensor.tensor(x).unsqueeze(0); control = tensor.tensor(u).unsqueeze(0)
    predicted = tensor.predict(state, control).squeeze(0).detach().numpy()
    assert np.max(np.abs(predicted-model.predict(x, u))) < 1e-10
    measurement, h = tensor.measure_and_jacobian(state)
    assert np.allclose(measurement[0].numpy(), model.measure(x), atol=1e-10)
    assert np.allclose(h[0].numpy(), model.measurement_jacobian(x), atol=1e-10)
    corrected = tensor.retract(state, tensor.tensor(delta).unsqueeze(0))
    assert np.allclose(corrected[0].numpy(), model.retract(x, delta), atol=1e-10)
    error = tensor.error(state, corrected)[0].numpy()
    assert np.allclose(error, model.error(x, model.retract(x, delta)), atol=1e-10)


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_full_rollout_matches_public_environment_and_has_finite_learning_gradient(task):
    torch.set_num_threads(1); torch.manual_seed(42)
    env = make_observation_task(task); obs, _ = env.reset(options={"index": 7, "scenario": "noise"})
    net = MetricNetwork(env.model)
    data = batch_data(env.model, [episode_data(task, "engineering", 7, "noise")])
    result = rollout(net, TensorModel(env.model), data, keep_trace=True)
    result["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert sum(p.grad.abs().sum().item() for p in net.parameters()) > 1e-12
    frozen = FrozenNetworkMetric(net); observer = MetricObserver(env.model, frozen)
    index = 0
    while True:
        obs, _, terminated, truncated, _ = env.step(observer.correction(obs))
        rec = env.evaluation_record()
        assert np.allclose(rec["estimate"], result["estimates"][index, 0].detach().numpy(), atol=2e-9)
        assert abs(value(env.model, rec["truth"], rec["estimate"], frozen)-result["values"][index, 0].item()) < 1e-8
        index += 1
        if terminated or truncated:
            break
    assert index == result["estimates"].shape[0]


def test_network_spectral_bounds_and_exact_initialization():
    model = make_observation_task("cooperative_pose").model
    net = MetricNetwork(model)
    states = torch.randn(8, model.spec.state_dim, dtype=torch.float64)
    initial = net(states)
    assert torch.allclose(initial, net.base_cholesky@net.base_cholesky.T)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 20)
    eig = torch.linalg.eigvalsh(net(states)).detach().numpy()
    assert eig.min() >= net.bounds[0]-1e-9
    assert eig.max() <= net.bounds[1]+1e-9


def test_zero_innovation_gives_zero_correction_for_nonconstant_network():
    env = make_observation_task("cooperative_pose"); obs, _ = env.reset()
    net = MetricNetwork(env.model)
    with torch.no_grad():
        net.net[-1].bias.uniform_(-.2, .2)
    obs["measurement"] = env.model.measure(obs["estimate"])
    observer = MetricObserver(env.model, FrozenNetworkMetric(net))
    assert np.array_equal(observer.correction(obs), np.zeros(12))
