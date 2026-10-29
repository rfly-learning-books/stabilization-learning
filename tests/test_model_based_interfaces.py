"""Behavioral checks for the reserved model-based interfaces."""
import numpy as np
import pytest
import torch

from extensions.book_disk_overnight_v1.algo import fit
from extensions.book_disk_overnight_v1.contract_adapter import DiskGMMFitter, DiskGMMTransitionModel
from extensions.book_disk_overnight_v1.env import DiskConfig, DiskEnv, transitions
from stabilization_learning.algos.model_based import (
    DiscreteTransitionBatch, MechanicalTrainingBatch,
    observed_discrete_change, predicted_discrete_change,
)


class StateDependentCertificate:
    def value(self, state):
        x = state[:, 0]
        return (1.0 + x*x) * x*x


class AdditiveModel:
    def predict_next(self, state, action):
        return state + action


def test_complete_successor_recomputes_state_dependent_certificate():
    state = np.array([[1.0], [2.0]])
    action = np.array([[-0.5], [-0.25]])
    successor = state + action
    expected = (1 + successor[:, 0]**2)*successor[:, 0]**2 - (1 + state[:, 0]**2)*state[:, 0]**2
    predicted = predicted_discrete_change(AdditiveModel(), StateDependentCertificate(), state, action)
    observed = observed_discrete_change(StateDependentCertificate(), state, successor)
    np.testing.assert_allclose(predicted, expected)
    np.testing.assert_allclose(observed, expected)
    frozen_start_metric = (1 + state[:, 0]**2) * (successor[:, 0]**2 - state[:, 0]**2)
    assert not np.allclose(predicted, frozen_start_metric)


def test_diagnostics_do_not_mutate_input_arrays():
    class InPlaceModel:
        def predict_next(self, state, action):
            state += action
            return state

    class InPlaceCertificate:
        def value(self, state):
            state *= 2
            return state[:, 0]**2

    state = np.array([[1.0], [2.0]])
    action = np.array([[-0.5], [-0.25]])
    initial, initial_action = state.copy(), action.copy()
    predicted = predicted_discrete_change(InPlaceModel(), InPlaceCertificate(), state, action)
    np.testing.assert_allclose(predicted, 4*((initial[:, 0] + action[:, 0])**2 - initial[:, 0]**2))
    np.testing.assert_array_equal(state, initial)
    np.testing.assert_array_equal(action, initial_action)


def test_discrete_diagnostics_reject_shape_and_nonfinite_values():
    class InvalidCertificate:
        def value(self, state):
            return np.full((len(state), 1), np.nan)

    with pytest.raises(ValueError, match="same sample count"):
        predicted_discrete_change(AdditiveModel(), StateDependentCertificate(), np.zeros((2, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="same shape"):
        observed_discrete_change(StateDependentCertificate(), np.zeros((2, 1)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        observed_discrete_change(InvalidCertificate(), np.zeros((2, 1)), np.zeros((2, 1)))


def test_transition_batch_rejects_wrong_shapes_and_period():
    with pytest.raises(ValueError, match="same sample count"):
        DiscreteTransitionBatch(np.zeros((2, 1)), np.zeros((3, 1)), np.zeros((2, 1)), .1)
    with pytest.raises(ValueError, match="positive finite"):
        DiscreteTransitionBatch(np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1)), float("nan"))
    with pytest.raises(ValueError, match="positive finite"):
        DiscreteTransitionBatch(np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1)), True)


def test_mechanical_batch_preserves_tensor_gradients_and_force_identity():
    q = torch.ones(3, 2, requires_grad=True)
    applied = torch.full_like(q, 2.0)
    commanded = torch.full_like(q, 3.0)
    data = MechanicalTrainingBatch(q, q*2, q*3, applied, commanded, torch.arange(3.0))
    assert data.q is q and data.q.requires_grad
    assert data.applied_generalized_force is applied
    assert data.commanded_generalized_force is commanded
    with pytest.raises(ValueError, match="timestamps"):
        MechanicalTrainingBatch(q, q, q, q, timestamps=torch.zeros(3, 1))
    with pytest.raises(ValueError, match="applied_generalized_force"):
        MechanicalTrainingBatch(q, q, q, torch.zeros(3, 1))


def test_existing_gmm_bridge_matches_original_disk_predictions():
    config = DiskConfig(response="varying")
    env = DiskEnv(config)
    train = transitions(env, 256, np.random.default_rng(27))
    batch = DiscreteTransitionBatch(
        train["observed_x"][:, None], train["action"][:, None],
        train["observed_next_x"][:, None], config.dt,
    )
    model, info = DiskGMMFitter(config, components=4).fit(batch, seed=41)
    replay, _ = DiskGMMFitter(config, components=4).fit(batch, seed=41)
    assert "EM_iterations" in info
    x, u = batch.state[:20], batch.applied_action[:20]
    weights = model.memberships(x, u)
    assert weights.shape[0] == len(x)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.predict_next(x, u), replay.predict_next(x, u))
    np.testing.assert_allclose(model.predict_next(x, np.zeros_like(u)), x)

    class DiskCertificate:
        def value(self, state):
            return 0.5*state[:, 0]**2

    predicted = predicted_discrete_change(model, DiskCertificate(), x, u)
    np.testing.assert_allclose(predicted, model.model.predict_D(x[:, 0], u[:, 0]), atol=1e-14)
    actual_next = env.step(x[:, 0], u[:, 0])[:, None]
    actual = observed_discrete_change(DiskCertificate(), x, actual_next)
    np.testing.assert_allclose(actual, train["true_D"][:20], atol=1e-14)
    other_period = DiscreteTransitionBatch(batch.state, batch.applied_action, batch.next_state, .2)
    with pytest.raises(ValueError, match="sampling period"):
        DiskGMMFitter(config).fit(other_period, seed=41)


def test_gmm_adapter_does_not_treat_direct_d_as_successor_model():
    config = DiskConfig()
    data = transitions(DiskEnv(config), 64, np.random.default_rng(1))
    direct, _ = fit("direct_D", data, config)
    with pytest.raises(ValueError, match="GMM route"):
        DiskGMMTransitionModel(direct)
