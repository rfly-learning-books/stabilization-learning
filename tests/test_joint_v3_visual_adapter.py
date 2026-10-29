"""Numerical and task-contract regressions for provider-owned accelerated JVPs."""
from copy import deepcopy

import pytest
import torch

from stabilization_gym.tasks_v2.visual import VisualTaskAdapter
from stabilization_gym.tasks_v3.visual import AcceleratedVisualTaskAdapter, InteractionJVPProvider
from stabilization_learning.joint_v2.visual import JointVisualLearner, engineering_fixture
from stabilization_learning.joint_v3.features import AcceleratedRGBFeatures


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(backend="forward"):
    reference, batch = engineering_fixture()
    reference.warmup(batch)
    reference.calibrate(batch)
    reference._zero()
    features = AcceleratedRGBFeatures(deepcopy(reference.features.backbone),
                                      reference.features.latent_dim, backend=backend).double()
    features.load_state_dict(reference.features.state_dict(), strict=True)
    learner = JointVisualLearner(features, deepcopy(reference.metric),
                                 AcceleratedVisualTaskAdapter(reference.adapter.config), reference.config)
    return reference, learner, batch


def gradient_vector(loss, parameters):
    parameters = tuple(parameters)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)])


@pytest.mark.parametrize("backend", ["forward", "legacy", "vmap"])
def test_j_actions_values_and_registered_scale_match_v2(backend):
    reference, learner, batch = fixture(backend)
    expected = reference.policy(batch.observation)
    actual = learner.policy(batch.observation)
    assert isinstance(learner.features, InteractionJVPProvider)
    for name in ("interaction", "action", "raw_action", "value", "metric", "error", "goal"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=2e-9, atol=2e-11)
    torch.testing.assert_close(learner.features.scale, reference.features.scale, rtol=0, atol=0)
    assert learner.features.scale != 1
    assert learner.features.execution_spec()["backend"] == backend
    assert learner.features.state_dict().keys() == reference.features.state_dict().keys()


@pytest.mark.parametrize("backend", ["forward", "legacy", "vmap"])
def test_j_only_mixed_parameter_gradient_matches_legacy(backend):
    reference, learner, batch = fixture(backend)

    def derivative(core):
        out = core.policy(batch.observation)
        action, _, _ = core.adapter.solve(out.error.detach(), out.interaction, out.metric.detach())
        physical = core.adapter.physical_error(core.adapter.nominal_step(batch.offset, action, batch.dt))
        return gradient_vector(physical.mean(), core.features.parameters())

    expected, actual = derivative(reference), derivative(learner)
    assert expected.norm() > 1e-6
    torch.testing.assert_close(actual, expected, rtol=3e-8, atol=2e-11)


@pytest.mark.parametrize("backend", ["forward", "legacy", "vmap"])
def test_complete_loss_backward_reaches_features_and_metric(backend):
    reference, learner, batch = fixture(backend)
    expected = reference.losses(batch)
    actual = learner.losses(batch)
    torch.testing.assert_close(actual["total"], expected["total"], rtol=2e-9, atol=2e-11)
    expected["total"].backward()
    actual["total"].backward()
    for old_module, new_module in ((reference.features, learner.features), (reference.metric, learner.metric)):
        assert any(p.grad is not None and p.grad.norm() > 1e-8 for p in new_module.parameters())
        for old, new in zip(old_module.parameters(), new_module.parameters()):
            old_grad = torch.zeros_like(old) if old.grad is None else old.grad
            new_grad = torch.zeros_like(new) if new.grad is None else new.grad
            torch.testing.assert_close(new_grad, old_grad, rtol=3e-8, atol=2e-10)


@pytest.mark.parametrize("backend", ["forward", "legacy", "vmap"])
def test_no_grad_inference_has_nonzero_J_and_create_graph_false_detaches(backend):
    reference, learner, batch = fixture(backend)
    with torch.no_grad():
        expected = reference.policy(batch.observation, create_graph=False)
        actual = learner.policy(batch.observation, create_graph=False)
    assert actual.interaction.norm() > 1e-6
    assert not actual.action.requires_grad
    torch.testing.assert_close(actual.action, expected.action, rtol=2e-9, atol=2e-11)
    public = batch.observation
    with torch.enable_grad():
        jvp = learner.features.interaction_jvp(public.image.requires_grad_(), public.image_tangents,
                                              create_graph=False)
    assert not jvp.requires_grad and jvp.grad_fn is None
    with torch.no_grad():
        differentiable = learner.features.interaction_jvp(public.image, public.image_tangents,
                                                         create_graph=True)
    assert differentiable.requires_grad


@pytest.mark.parametrize("backend", ["forward", "legacy", "vmap"])
def test_inference_mode_is_rejected_instead_of_silently_zeroing_J(backend):
    _, learner, batch = fixture(backend)
    public = batch.observation
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no_grad"):
        learner.features.interaction_jvp(public.image, public.image_tangents, create_graph=False)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no_grad"):
        learner.adapter.interaction(learner.features, public.image, public.image_tangents, create_graph=False)


def test_common_task_context_and_action_contract_matches_v2():
    reference, learner, batch = fixture()
    memory = {"case": "unchanged"}
    with torch.no_grad():
        old_context = reference.adapter.make_context(batch.observation, features=reference.features, memory=memory)
        context = learner.adapter.make_context(batch.observation, features=learner.features, memory=memory)
        expected = reference.adapter.construct_action(old_context, reference.metric)
        actual = learner.adapter.construct_action(context, learner.metric)
    assert context.memory is memory and actual.memory is memory
    torch.testing.assert_close(actual.action, expected.action, rtol=2e-9, atol=2e-11)
    for key in ("V", "M", "J", "raw_action", "hessian"):
        torch.testing.assert_close(actual.diagnostics[key], expected.diagnostics[key], rtol=2e-9, atol=2e-11)


def test_encode_only_provider_has_explicit_legacy_fallback():
    reference, _, batch = fixture()

    class EncodeOnly:
        def encode(self, image):
            return reference.features.encode(image)

    provider = EncodeOnly()
    public = batch.observation
    adapter = AcceleratedVisualTaskAdapter(encode_only_backend="legacy")
    expected = VisualTaskAdapter.interaction(provider, public.image, public.image_tangents)
    actual = adapter.interaction(provider, public.image, public.image_tangents)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(TypeError, match="interaction_jvp"):
        AcceleratedVisualTaskAdapter(encode_only_backend="error").interaction(
            provider, public.image, public.image_tangents)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no_grad"):
        adapter.interaction(provider, public.image, public.image_tangents)


def test_provider_capability_errors_are_not_silently_bypassed():
    reference, _, batch = fixture()

    class FailingProvider:
        def encode(self, image):
            return reference.features.encode(image)

        def interaction_jvp(self, *args, **kwargs):
            raise NotImplementedError("selected forward operation unsupported")

    with pytest.raises(NotImplementedError, match="unsupported"):
        AcceleratedVisualTaskAdapter().interaction(FailingProvider(), batch.observation.image,
                                                   batch.observation.image_tangents)


def test_invalid_backend_and_derivative_shapes_are_rejected():
    reference, learner, batch = fixture()
    with pytest.raises(ValueError, match="backend"):
        AcceleratedRGBFeatures(reference.features.backbone, 4, backend="auto")
    with pytest.raises(ValueError, match="encode_only_backend"):
        AcceleratedVisualTaskAdapter(encode_only_backend="forward")
    with pytest.raises(ValueError, match="B,A,C,H,W"):
        learner.features.interaction_jvp(batch.observation.image, batch.observation.image_tangents[:, :, :, :-1])
    with pytest.raises(TypeError, match="boolean"):
        learner.features.interaction_jvp(batch.observation.image, batch.observation.image_tangents, create_graph=1)
