"""Numerical contract tests for the new, isolated shared-feature visual core."""
from dataclasses import replace
import math

import pytest
import torch

from stabilization_gym.tasks_v2.visual import (
    VisualControlConfig, VisualTaskAdapter, pinhole_image_tangents, quadratic_value,
)
from stabilization_learning.joint_v2.certificate import NeuralCertificate, make_binding
from stabilization_learning.joint_v2.visual import (
    JointVisualConfig, engineering_fixture, frozen_parameters,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def calibrated_fixture(seed=7):
    learner, batch = engineering_fixture(seed)
    learner.warmup(batch)
    learner.calibrate(batch)
    return learner, batch


def parameter_snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def unchanged(before, module):
    return all(torch.equal(before[name], value) for name, value in module.state_dict().items())


def test_image_tangent_translation_has_correct_sign_and_physical_units():
    height = width = 7
    image = torch.arange(width, dtype=torch.float64)[None, None, None].expand(2, 3, height, -1) / width
    depth = torch.full((2, 1, height, width), 2.0, dtype=image.dtype)
    k = torch.tensor([[10., 0., 3.], [0., 10., 3.], [0., 0., 1.]], dtype=image.dtype)
    tangents = pinhole_image_tangents(image, depth, k)
    expected = torch.full_like(image, 10.0 / 2.0 * 0.4 / width)
    torch.testing.assert_close(tangents[:, 0], expected)
    torch.testing.assert_close(tangents[:, 1], torch.zeros_like(image))
    assert tangents.shape == (2, 4, 3, 7, 7)


def test_mixed_jvp_parameter_gradient_matches_finite_difference():
    learner, batch = calibrated_fixture()
    features, adapter, public = learner.features, learner.adapter, batch.observation
    parameter = features.backbone.encoder[0].weight
    weights = torch.arange(1, 17, dtype=torch.float64).reshape(1, 4, 4)

    def objective(create_graph):
        j = adapter.interaction(features, public.image, public.image_tangents, create_graph=create_graph)
        return (j * weights).sum()

    analytic, = torch.autograd.grad(objective(True), parameter)
    index = int(analytic.abs().argmax())
    original = parameter.detach().clone()
    epsilon = 1e-5
    with torch.no_grad():
        parameter.view(-1)[index] += epsilon
    plus = objective(False).detach()
    with torch.no_grad():
        parameter.copy_(original)
        parameter.view(-1)[index] -= epsilon
    minus = objective(False).detach()
    with torch.no_grad():
        parameter.copy_(original)
    numerical = (plus - minus) / (2 * epsilon)
    assert analytic.view(-1)[index].abs() > 1e-6
    torch.testing.assert_close(analytic.view(-1)[index], numerical, rtol=2e-4, atol=1e-6)


def test_physical_loss_has_nonzero_encoder_gradient_via_j_alone():
    learner, batch = calibrated_fixture()
    out = learner.policy(batch.observation)
    action, _, _ = learner.adapter.solve(out.error.detach(), out.interaction, out.metric.detach())
    following = learner.adapter.nominal_step(batch.offset, action, batch.dt)
    loss = learner.adapter.physical_error(following).mean()
    grads = torch.autograd.grad(loss, tuple(learner.features.parameters()), allow_unused=True)
    assert sum(float(g.square().sum()) for g in grads if g is not None) > 1e-10


def test_frozen_metric_weights_preserve_gradient_through_goal_input():
    learner, batch = calibrated_fixture()
    learner._zero()
    before = parameter_snapshot(learner.metric)
    with frozen_parameters(learner.metric):
        goal = learner.features.encode(batch.observation.goal_image)
        metric = learner.metric(goal)
        weight = torch.arange(16, dtype=metric.dtype).reshape(1, 4, 4)
        loss = (metric * weight).sum()
        goal_gradient, = torch.autograd.grad(loss, goal, retain_graph=True)
        loss.backward()
    assert goal_gradient.norm() > 1e-6
    assert any(p.grad is not None and p.grad.norm() > 1e-6 for p in learner.features.parameters())
    assert all(p.grad is None for p in learner.metric.parameters())
    assert unchanged(before, learner.metric)


def test_calibration_phase_and_registered_coordinate_buffers():
    learner, batch = engineering_fixture()
    with pytest.raises(RuntimeError, match="AE warmup"):
        learner.calibrate(batch)
    with pytest.raises(RuntimeError, match="calibrated coordinates"):
        learner.update_features(batch)
    learner.warmup(batch)
    learner.calibrate(batch)
    assert {"center", "scale", "calibrated", "coordinate_version"} <= dict(learner.features.named_buffers()).keys()
    assert learner.features.scale > 0 and learner.features.coordinate_version == 1
    with pytest.raises(RuntimeError, match="already calibrated"):
        learner.calibrate(batch)
    scale = learner.features.scale.clone()
    learner.train_blocks(batch)
    torch.testing.assert_close(learner.features.scale, scale, rtol=0, atol=0)


def test_two_parameter_blocks_really_update_and_preserve_the_other_module():
    learner, batch = calibrated_fixture()
    feature_before = parameter_snapshot(learner.features)
    metric_before = parameter_snapshot(learner.metric)
    feature_report = learner.update_features(batch)
    assert feature_report["feature_gradient_norm"] > 0
    assert not unchanged(feature_before, learner.features)
    assert unchanged(metric_before, learner.metric)
    feature_before = parameter_snapshot(learner.features)
    action_before = learner.policy(batch.observation).action.detach()
    metric_report = learner.update_metric(batch)
    action_after = learner.policy(batch.observation).action.detach()
    assert metric_report["metric_gradient_norm"] > 0
    assert unchanged(feature_before, learner.features)
    assert not unchanged(metric_before, learner.metric)
    assert (action_after - action_before).norm() > 1e-8


def test_labels_do_not_enter_policy_and_only_detached_labels_enter_loss():
    learner, batch = calibrated_fixture()
    changed_offset = (batch.offset * 8).requires_grad_()
    changed = replace(batch, offset=changed_offset, next_offset=batch.next_offset * 3)
    # Both use exactly the public policy type. It has no offset/truth fields.
    assert not hasattr(batch.observation, "offset")
    torch.testing.assert_close(learner.policy(batch.observation).action,
                               learner.policy(changed.observation).action, rtol=0, atol=0)
    terms = learner.losses(changed)
    terms["total"].backward()
    assert changed_offset.grad is None
    assert not torch.isclose(terms["physical"], learner.losses(batch)["physical"])


def test_calibration_violation_gradients_reach_both_phi_and_metric():
    learner, batch = calibrated_fixture()
    # Deliberately break coordinates in a test to expose the formerly infeasible
    # objective. This is not a supported training normalization change.
    with torch.no_grad():
        learner.features.scale.mul_(0.03)
    calibration = learner.losses(batch)["calibration"]
    feature_grads = torch.autograd.grad(calibration, tuple(learner.features.parameters()),
                                        retain_graph=True, allow_unused=True)
    metric_grads = torch.autograd.grad(calibration, tuple(learner.metric.parameters()), allow_unused=True)
    assert calibration > 0
    assert sum(float(g.square().sum()) for g in feature_grads if g is not None) > 1e-6
    assert sum(float(g.square().sum()) for g in metric_grads if g is not None) > 1e-6


def test_same_version_difference_reencodes_both_endpoints_after_parameter_change():
    learner, batch = calibrated_fixture()
    earlier = learner.same_version_difference(batch).detach()
    learner.update_features(batch)
    learner.update_metric(batch)
    public = batch.observation
    goal = learner.features.encode(public.goal_image)
    e = learner.features.encode(public.image) - goal
    en = learner.features.encode(batch.next_image) - goal
    metric = learner.metric(goal)
    expected = quadratic_value(en, metric) - quadratic_value(e, metric)
    actual = learner.same_version_difference(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual - earlier).abs().max() > 1e-10
    assert learner.losses(batch)["decrease"] == 0  # exploratory actions need not decrease V
    with pytest.raises(ValueError, match="behavior version"):
        learner.losses(replace(batch, controller_transition=torch.ones_like(batch.controller_transition)))


def test_bound_frontend_checks_version_and_matches_native_policy():
    learner, batch = calibrated_fixture()
    binding = make_binding(task="visual", revision="v1", action_units="normalized_velocity",
                           metric_scope="goal_conditioned")
    certificate = NeuralCertificate(learner.metric, binding, factor=0.5)
    context = learner.adapter.make_context(batch.observation, features=learner.features, binding=binding)
    action = learner.adapter.construct_action(context, certificate)
    torch.testing.assert_close(action.action, learner.policy(batch.observation).action)
    stale = replace(binding, weights="old")
    context.binding = stale
    with pytest.raises(ValueError, match="Stale or incompatible"):
        learner.adapter.construct_action(context, certificate)


def test_goal_metric_spectrum_and_radial_action_direction():
    learner, batch = calibrated_fixture()
    out = learner.policy(batch.observation)
    eigenvalues = torch.linalg.eigvalsh(out.metric)
    assert eigenvalues.min() >= learner.metric.floor - 1e-10
    assert eigenvalues.max() <= 4 - 3 * learner.metric.floor + 1e-10
    torch.testing.assert_close(out.metric.diagonal(dim1=-2, dim2=-1).sum(-1),
                               torch.full((3,), 4.0, dtype=torch.float64))
    assert (out.action[:, :3].norm(dim=-1) <= 1 + 1e-12).all()
    assert (out.action[:, 3].abs() <= 1 + 1e-12).all()
    # Exact nominal J dynamics only: positive radial limiting preserves sign.
    nominal_vdot = torch.einsum("bi,bij,bj->b", out.error, out.metric,
                               (out.interaction @ out.action.unsqueeze(-1)).squeeze(-1))
    assert nominal_vdot.max() <= 1e-10


def test_checkpoint_restores_adam_and_the_next_update(tmp_path):
    learner, batch = calibrated_fixture()
    learner.train_blocks(batch)
    path = tmp_path / "joint.pt"
    learner.save(path)
    clone, _ = engineering_fixture(seed=123)
    clone.load(path)
    assert clone.counts == learner.counts and clone.version == learner.version
    assert clone.feature_optimizer.state and clone.metric_optimizer.state
    torch.testing.assert_close(clone.policy(batch.observation).action,
                               learner.policy(batch.observation).action, rtol=0, atol=0)
    learner.train_blocks(batch)
    clone.train_blocks(batch)
    assert unchanged(parameter_snapshot(learner.features), clone.features)
    assert unchanged(parameter_snapshot(learner.metric), clone.metric)


@pytest.mark.parametrize("kwargs", [{"gain": math.nan}, {"damping": 0}, {"yaw_speed": math.inf}])
def test_invalid_control_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        VisualControlConfig(**kwargs)


def test_invalid_loss_config_and_eval_data_cannot_train():
    with pytest.raises(ValueError):
        JointVisualConfig(calibration_weight=math.nan)
    learner, batch = calibrated_fixture()
    with pytest.raises(ValueError, match="training-split"):
        learner.update_features(replace(batch, split="test"))
