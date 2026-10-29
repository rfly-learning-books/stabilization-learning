"""CPU reproducibility and scenario/domain parity for observer v4."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from stabilization_gym.observation import make_observation_task
from stabilization_gym.observation.models import PoseModel
from extensions.observer_learning_v1.learning import MetricNetwork, episode_data, batch_data, rollout
from extensions.observer_learning_v1.tensor_model import TensorModel
from extensions.unified_training_v1.backends.observer import _digest
from extensions.reproducible_training_v4.observer import CorrectedObserverBackend, SEMANTICS
from extensions.reproducible_training_v4.observer_model import ScenarioTensorModel, corrected_rollout


def _config(name):
    scenarios = ["nominal", "degenerate"] if name == "cooperative_pose" else ["nominal", "noise"]
    return dict(task=name, seed=89621, cpu_threads=1, batch_size=2, hidden=8,
        validation_count=2, test_count=2, training_index_start=89600,
        validation_index_start=89700, test_index_start=89800, scenarios=scenarios)


def test_mixed_geometry_matches_each_numpy_measurement_and_jacobian():
    model = PoseModel()
    tensor = ScenarioTensorModel(model, ["nominal", "degenerate"])
    states = np.stack([model.initial_estimate(), model.initial_estimate()])
    predicted, jacobian = tensor.measure_and_jacobian(torch.tensor(states))
    for lane, degenerate in enumerate((False, True)):
        actual = PoseModel(degenerate=degenerate)
        np.testing.assert_allclose(predicted[lane].numpy(), actual.measure(states[lane]), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(jacobian[lane].numpy(), actual.measurement_jacobian(states[lane]), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_exact_domain_matches_numpy_boundary_witnesses(name):
    model = make_observation_task(name).model
    states = [model.initial_estimate()]
    if name == "motor_speed":
        states.extend([np.array([6., 0.]), np.array([6.01, 0.])])
    elif name == "cartpole_output":
        states.extend([np.array([0., 21., 0., 0.]), np.array([0., 0., math.pi/4, 0.]), np.array([0., 0., math.pi/4+.001, 0.])])
    else:
        for index, value in ((2, .31), (2, .32), (7, 11.), (10, 11.)):
            state = model.initial_estimate()
            state[index] = value
            states.append(state)
        state = model.initial_estimate()
        state[6] = 1.01
        states.append(state)
    tensor = ScenarioTensorModel(model, ["nominal"]*len(states))
    expected = [model.domain_ok(state) for state in states]
    assert tensor.exact_domain_ok(torch.tensor(np.stack(states))).tolist() == expected


@pytest.mark.parametrize("name", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_healthy_nominal_rollout_preserves_loss_and_full_successor_value(name):
    torch.set_num_threads(1)
    torch.manual_seed(89631)
    model = make_observation_task(name).model
    network = MetricNetwork(model, hidden=8)
    episodes = [episode_data(name, "engineering", 89631, "nominal")]
    batch = batch_data(model, episodes)
    with torch.no_grad():
        old = rollout(network, TensorModel(model), batch, keep_trace=True)
        new = corrected_rollout(network, ScenarioTensorModel(model, ["nominal"]), batch, keep_trace=True)
    assert not bool(new["failed"].any())
    for key in ("loss", "cost", "terminal_mse", "certificate_loss", "errors", "values", "matrices", "estimates"):
        torch.testing.assert_close(old[key], new[key], rtol=1e-11, atol=1e-11)
    assert torch.equal(old["success"], new["success"])


def test_truth_failure_cannot_be_erased_by_later_successful_hold():
    model = make_observation_task("motor_speed").model
    torch.manual_seed(89641)
    network = MetricNetwork(model, hidden=8)
    batch = batch_data(model, [episode_data("motor_speed", "engineering", 89641, "nominal")])
    # Fault injection into an offline label: after the first point the normal
    # trajectory recovers, but a real complete-history failure must stay latched.
    batch["truth"][0, 0, 0] = 7.
    with torch.no_grad():
        old = rollout(network, TensorModel(model), batch)
        result = corrected_rollout(network, ScenarioTensorModel(model, ["nominal"]), batch)
    assert bool(old["success"][0])
    assert bool(result["failed"][0]) and not bool(result["success"][0])
    assert result["first_failure_index"].tolist() == [0]
    expected = model.spec.horizon*result["errors"][0].square().sum(-1).clamp_min(25.)
    torch.testing.assert_close(result["cost"], expected)


def test_pose_rotation_chart_failure_is_recorded():
    model = PoseModel()
    network = MetricNetwork(model, hidden=8)
    batch = batch_data(model, [episode_data("cooperative_pose", "engineering", 89643, "nominal")])
    # Isolate the chart predicate with truth rotated almost pi at one endpoint.
    batch["truth"][0, 0, 3:7] = torch.tensor([math.sin((math.pi-.001)/2), 0., 0., math.cos((math.pi-.001)/2)])
    with torch.no_grad():
        result = corrected_rollout(network, ScenarioTensorModel(model, ["nominal"]), batch)
    assert bool(result["failure_by_step"][0, 0]) and not bool(result["success"][0])


@pytest.mark.parametrize("name", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_same_seed_updates_checkpoint_restore_and_evaluation_reproduce(tmp_path, name):
    records = []
    for repeat in range(2):
        backend = CorrectedObserverBackend(_config(name), tmp_path / f"run_{repeat}")
        restored = None
        try:
            initialized = backend.initialize()
            initial = _digest(backend.network)
            collected = backend.collect(initial=True)
            trained = backend.train_epoch(1, certificate_updates=2)
            updated = _digest(backend.network)
            assert updated != initial and backend.updates == 2
            validation = backend.validate()
            evaluation = backend.evaluate("validation", ("selected",))
            deployed_rows = evaluation["groups"]["selected"]["rows"]
            for differentiable, deployed in zip(validation["rows"], deployed_rows):
                assert differentiable["success"] == deployed["success"]
                assert differentiable["cost"] == pytest.approx(deployed["fixed_horizon_error_cost"], rel=2e-10, abs=2e-10)
            checkpoint = tmp_path / f"checkpoint_{repeat}.pt"
            backend.save(checkpoint)
            restored = CorrectedObserverBackend(_config(name), tmp_path / f"restore_{repeat}")
            restored.initialize()
            restored.load(checkpoint)
            assert _digest(restored.network) == updated
            assert restored.validate() == validation
            assert restored.evaluate("validation", ("selected",)) == evaluation
            assert initialized["observer_semantics"] == SEMANTICS
            rows = [{key: value for key, value in row.items() if key != "wall_seconds"} for row in trained["rows"]]
            records.append(dict(initial=initial, updated=updated, collected=collected, rows=rows,
                                validation=validation, evaluation=evaluation))
        finally:
            backend.close()
            if restored is not None:
                restored.close()
    assert records[0] == records[1]
    evidence = dict(schema="observer_v4_same_seed_cpu_reproducibility", task=name, config=_config(name),
        identical_across_runs=True, parameters_changed=True, real_updates_per_run=2,
        checkpoint_restore_equal=True, deployment_evaluation_equal=True,
        validation_selection_matches_deployment=True, records=records,
        scope="Engineering regression only; no convergence, multi-seed robustness, or deployment guarantee")
    (tmp_path / "REPRODUCIBILITY.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def test_old_semantics_checkpoint_cannot_be_loaded_as_corrected(tmp_path):
    backend = CorrectedObserverBackend(_config("motor_speed"), tmp_path / "new")
    try:
        backend.initialize()
        checkpoint = tmp_path / "new.pt"
        backend.save(checkpoint)
        payload = torch.load(checkpoint, weights_only=False)
        payload.pop("observer_semantics")
        old = tmp_path / "old.pt"
        torch.save(payload, old)
        with pytest.raises(ValueError, match="different geometry/domain/loss protocol"):
            backend.load(old)
    finally:
        backend.close()
