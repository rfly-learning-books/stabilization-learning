"""Independent mathematical and gradient checks for the shared learning core."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from stabilization_learning.core.types import MetricSpec
from stabilization_learning.envs.base import StateMetricEnv
from stabilization_learning.metrics import CholeskyMetric, MetricActor, quadratic_certificate
from stabilization_learning.representations import (
    DerivativeHead, IdentityRepresentation, RGBAutoEncoder, RepresentationLearner,
)


@pytest.mark.parametrize("dimension", [1, 3, 6, 16])
def test_cholesky_identity_trace_and_spectral_bounds(dimension):
    torch.manual_seed(82)
    spec = MetricSpec(dimension, eigen_floor=.1)
    mapping = CholeskyMetric(spec).double()
    identity = mapping(mapping.identity_raw(dtype=torch.float64))
    torch.testing.assert_close(identity, torch.eye(dimension, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    raw = torch.randn(5, spec.raw_dim, dtype=torch.float64)*4.
    matrices = mapping(raw)
    assert matrices.shape == (5, dimension, dimension)
    torch.testing.assert_close(matrices, matrices.transpose(-1, -2), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(matrices.diagonal(dim1=-2, dim2=-1).sum(-1),
                               torch.full((5,), float(dimension), dtype=torch.float64))
    eigenvalues = torch.linalg.eigvalsh(matrices)
    assert eigenvalues.min() >= spec.eigen_floor-1e-10
    assert eigenvalues.max() <= spec.eigen_ceiling+1e-10


def test_custom_trace_has_isotropic_initialization_and_documented_ceiling():
    spec = MetricSpec(3, eigen_floor=.2, trace=6.)
    mapping = CholeskyMetric(spec).double()
    matrix = mapping(mapping.identity_raw(dtype=torch.float64))
    torch.testing.assert_close(matrix, 2.*torch.eye(3, dtype=torch.float64))
    assert spec.raw_dim == 6
    assert spec.eigen_ceiling == pytest.approx(5.6)


@pytest.mark.parametrize("kwargs", [
    {"error_dim": 0}, {"error_dim": 1.5},
    {"error_dim": 3, "trace": float("inf")},
    {"error_dim": 3, "trace": float("nan")},
    {"error_dim": 3, "eigen_floor": -0.1},
    {"error_dim": 3, "eigen_floor": 1.0},
])
def test_metric_spec_rejects_nonphysical_bounds(kwargs):
    with pytest.raises(ValueError):
        MetricSpec(**kwargs)


@pytest.mark.parametrize("scale", [1e-30, 1e30, -1e30])
def test_cholesky_large_finite_raw_does_not_overflow(scale):
    mapping = CholeskyMetric(MetricSpec(4))
    raw = torch.full((2, mapping.spec.raw_dim), scale, requires_grad=True)
    matrix = mapping(raw)
    assert torch.isfinite(matrix).all()
    torch.testing.assert_close(matrix.diagonal(dim1=-2, dim2=-1).sum(-1), torch.full((2,), 4.))
    gradient = torch.autograd.grad(matrix.square().sum(), raw)[0]
    assert torch.isfinite(gradient).all()


def test_batched_cholesky_certificate_gradient_matches_finite_differences():
    torch.manual_seed(10)
    mapping = CholeskyMetric(MetricSpec(3)).double()
    raw = (.3*torch.randn(2, 6, dtype=torch.float64)).requires_grad_()
    error = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda r, e: quadratic_certificate(e, mapping(r)),
                                   (raw, error), eps=1e-6, atol=2e-5, rtol=2e-4)
    value = quadratic_certificate(error, mapping(raw))
    assert value.shape == (2,)
    assert torch.all(value > 0.)
    gradient = torch.autograd.grad(value.sum(), error)[0]
    torch.testing.assert_close(gradient, 2.*torch.matmul(mapping(raw), error.unsqueeze(-1)).squeeze(-1))


def test_metric_actor_starts_at_identity_and_samples_raw_parameter_distribution():
    torch.manual_seed(5)
    actor = MetricActor(4, MetricSpec(3), hidden=8)
    contexts = torch.randn(7, 4)
    torch.testing.assert_close(actor(contexts), torch.eye(3).expand(7, 3, 3))
    distribution = actor.distribution(contexts)
    sample = distribution.sample()
    assert sample.shape == (7, 6)
    assert distribution.log_prob(sample).shape == (7,)
    assert torch.isfinite(distribution.log_prob(sample)).all()
    loss = -distribution.log_prob(sample).mean()
    loss.backward()
    assert torch.linalg.vector_norm(actor.net[-1].weight.grad) > 0.


def test_derivative_head_exact_zero_velocity_linearity_and_action_gradient():
    torch.manual_seed(6)
    head = DerivativeHead(4, 2, hidden=8).double()
    z = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    goal = torch.randn_like(z)
    velocity = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    assert torch.equal(head.coefficients(z, z), torch.zeros_like(velocity))
    assert torch.equal(head(z, z, velocity), torch.zeros(5, dtype=z.dtype))
    assert torch.equal(head(z, goal, torch.zeros_like(velocity)), torch.zeros(5, dtype=z.dtype))
    torch.testing.assert_close(head(z, goal, -velocity), -head(z, goal, velocity))
    torch.testing.assert_close(head(z, goal, 2.3*velocity), 2.3*head(z, goal, velocity))
    output = head(z, goal, velocity)
    action_gradient, feature_gradient = torch.autograd.grad(output.sum(), (velocity, z))
    torch.testing.assert_close(action_gradient, head.coefficients(z, goal))
    assert torch.isfinite(feature_gradient).all()
    assert torch.linalg.vector_norm(feature_gradient) > 1e-7


def test_representation_learner_caps_only_auxiliary_encoder_gradient():
    torch.manual_seed(12)
    autoencoder = RGBAutoEncoder(latent_dim=4, image_size=16).double()
    head = DerivativeHead(4, 2, hidden=8).double()
    learner = RepresentationLearner(autoencoder, head, auxiliary_cap=.25)
    images = torch.rand(3, 3, 16, 16, dtype=torch.float64)
    goals = torch.rand_like(images)
    velocities = 20.*torch.randn(3, 2, dtype=torch.float64)
    targets = torch.full((3,), 20., dtype=torch.float64)
    encoder, decoder = list(autoencoder.encoder.parameters()), list(autoencoder.decoder.parameters())
    z = autoencoder(images)
    rec = nn.functional.mse_loss(autoencoder.decoder(z), images)
    rec_grads = torch.autograd.grad(rec, encoder+decoder, retain_graph=True)
    aux = nn.functional.smooth_l1_loss(head(z, autoencoder(goals), velocities), targets)
    aux_grads = torch.autograd.grad(aux, encoder+list(head.parameters()))
    encoder_before = [parameter.detach().clone() for parameter in encoder]
    result = learner.update(images, goals, velocities, targets)
    assert 0. < result["effective_aux_ratio"] <= .25+1e-10
    assert result["auxiliary_scale"] < 1.
    for parameter, rec_grad, aux_grad in zip(encoder, rec_grads, aux_grads):
        torch.testing.assert_close(parameter.grad, rec_grad+result["auxiliary_scale"]*aux_grad)
    for parameter, rec_grad in zip(decoder, rec_grads[len(encoder):]):
        torch.testing.assert_close(parameter.grad, rec_grad)
    for parameter, aux_grad in zip(head.parameters(), aux_grads[len(encoder):]):
        torch.testing.assert_close(parameter.grad, aux_grad)
    assert any(not torch.equal(before, after) for before, after in zip(encoder_before, encoder))
    # Removing labels on the next update must not apply stale head gradients.
    head_before = [parameter.detach().clone() for parameter in head.parameters()]
    learner.update(images)
    for before, after in zip(head_before, head.parameters()):
        torch.testing.assert_close(before, after)
        assert after.grad is None


class _FlatAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))
            self.decoder.weight.copy_(torch.eye(2))

    def forward(self, value):
        return self.encoder(value)


def test_zero_ae_gradient_disables_auxiliary_encoder_update():
    torch.manual_seed(51)
    autoencoder = _FlatAutoencoder()
    head = DerivativeHead(2, 1, hidden=8)
    learner = RepresentationLearner(autoencoder, head, auxiliary_cap=.25)
    images, goals = torch.zeros(4, 2), torch.ones(4, 2)
    before = autoencoder.encoder.weight.detach().clone()
    result = learner.update(images, goals, torch.ones(4, 1), torch.full((4,), 5.))
    assert result["encoder_ae_grad_norm"] == 0.
    assert result["encoder_aux_grad_norm"] > 0.
    assert result["auxiliary_scale"] == 0.
    torch.testing.assert_close(autoencoder.encoder.weight, before)


@pytest.mark.parametrize("cap", [-.2, float("nan"), float("inf")])
def test_auxiliary_cap_rejects_invalid_values(cap):
    with pytest.raises(ValueError):
        RepresentationLearner(_FlatAutoencoder(), auxiliary_cap=cap)


class _BoundaryEnv(StateMetricEnv):
    def __init__(self, **kwargs):
        super().__init__(state_dim=1, error_dim=1, action_low=[-1.], action_high=[1.],
                         dt=kwargs.pop("dt", .1), max_steps=kwargs.pop("max_steps", 10), **kwargs)

    def torch_step(self, state, action):
        return state+self.dt*action

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        return -(M@e.unsqueeze(-1)).squeeze(-1)


def test_physical_step_clips_actions_returns_positive_integral_and_copies_observation():
    env = _BoundaryEnv()
    input_state = np.array([2.])
    observation, _ = env.reset(options={"state": input_state})
    input_state[0] = 9.
    observation[0] = 8.
    assert env.state[0] == 2.
    observation, cost, terminated, truncated, info = env.step(np.array([5.]))
    assert observation[0] == pytest.approx(2.1)
    assert cost == pytest.approx(.4)
    assert info["applied_action"][0] == 1.
    assert not terminated and not truncated
    assert info["cost_convention"] == "positive_integral"
    observation[0] = 7.
    assert env.state[0] == pytest.approx(2.1)


def test_physical_boundary_success_hold_timeout_and_reset_guard():
    env = _BoundaryEnv()
    with pytest.raises(RuntimeError):
        env.step([0.])
    env.reset(options={"state": [0.]})
    for step in range(5):
        _, _, terminated, truncated, info = env.step([0.])
        assert terminated == (step == 4)
        assert not truncated
    assert info["success"]
    with pytest.raises(RuntimeError):
        env.step([0.])
    env = _BoundaryEnv(max_steps=1)
    env.reset(options={"state": [2.]})
    _, _, terminated, truncated, info = env.step([0.])
    assert truncated and not terminated and not info["success"]


@pytest.mark.parametrize("action", [[float("nan")], [float("inf")], [0., 1.]])
def test_physical_boundary_rejects_invalid_actions(action):
    env = _BoundaryEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.parametrize("kwargs", [{"dt": 0.}, {"dt": -1.}, {"dt": math.inf},
                                    {"max_steps": 0}, {"max_steps": 1.5}])
def test_physical_boundary_rejects_invalid_time_contract(kwargs):
    with pytest.raises(ValueError):
        _BoundaryEnv(**kwargs)


def test_state_error_preserves_observation_gradients_and_checks_reference_semantics():
    env = _BoundaryEnv()
    state = torch.tensor([.3], dtype=torch.float64, requires_grad=True)
    error, context = env.build_error(state, representation=IdentityRepresentation())
    gradient = torch.autograd.grad(error.square().sum(), state)[0]
    torch.testing.assert_close(gradient, 2.*state)
    assert context["state"] is state
    with pytest.raises(ValueError):
        env.build_error(state, goal=np.array([1.]))
    with pytest.raises(ValueError):
        env.build_error(state, representation=nn.Linear(1, 1))
