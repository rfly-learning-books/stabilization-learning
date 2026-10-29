"""CPU conformance for the explicit interface; no original trainer is launched."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stabilization_gym import make
from stabilization_gym.interfaces.m_learning import (
    CapabilityError, LearningAssessment, MetricScope, OriginalPushMFacade,
    OriginalVisualMFacade, PortableMFacade,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("name", ["coupled_linear", "point_mass", "pendulum", "cartpole",
                                 "two_link_arm", "unicycle_tracking", "multirotor_tracking"])
def test_portable_control_parity_graph_and_single_step(name):
    env = make(name, max_steps=3)
    task = PortableMFacade(env, name=name)
    task.require_algorithm("episode_metric_ppo").require_capability("action_wrt_M")
    obs, _ = task.reset(seed=43)
    sample = task.error_context(obs)
    M = torch.diag(torch.linspace(.8, 1.2, env.error_dim, dtype=torch.float32)).requires_grad_()
    before = env.state.copy()
    result = task.control(M, sample, training=True)
    direct = env.calculate_action_from_M(M.to(sample.error), sample.error, context=sample.control_context)
    torch.testing.assert_close(result.action, direct, rtol=0, atol=0)
    np.testing.assert_array_equal(env.state, before)
    derivative, = torch.autograd.grad(result.action.sum(), M)
    assert torch.isfinite(derivative).all() and derivative.norm() > 0
    assert result.certificate.claim == "candidate_only"
    assert task.describe().cost_convention == "positive_integral"
    task.step(result.action.detach().numpy())
    assert env._elapsed_steps == 1
    with pytest.raises(CapabilityError):
        task.require_algorithm("state_dependent_ppo")
    env.close()


def test_actual_episode_learner_accepts_facade_and_keeps_phi_contract(tmp_path):
    from stabilization_learning import StabilizationLearner, TrainConfig
    task = PortableMFacade(make("point_mass", max_steps=2), name="point_mass")
    learner = StabilizationLearner(task, config=TrainConfig(
        episodes=2, batch_episodes=2, epochs=1, validation_seeds=(10001,)))
    # A genuine two-episode CPU optimizer update checks compatibility, without
    # opening any original/GPU run or modifying its protocol.
    batch = learner._collect(2)
    before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    losses = learner._update(*batch[:4])
    assert losses["epochs_run"] == 1
    assert any(not torch.equal(before[k], v) for k, v in learner.actor.state_dict().items())
    assert learner._protocol()["environment_signature"]["settings"]["source_signature_sha256"]
    task.close()


def test_original_push_real_cpu_control_and_full_certificate_derivative():
    from stabilization_gym.envs.tasks.original_protocols import OriginalPushTask
    root = Path(__file__).resolve().parents[1]
    if not (root / "assets/original_push_reference/manifest.json").exists():
        pytest.skip("Retained original Push bundle unavailable")
    native = OriginalPushTask(root=root)
    bank = native.bank("cpu")
    physical = native.new_environment(bank, shape_names=["T"], num_envs=1)
    actor = native.engine.FullMActor(native.cfg).cpu()
    facade = OriginalPushMFacade(native, bank, physical)
    data = physical.reset(seed=37)
    sample = facade.error_context(data)
    cert = facade.certificate(actor, sample)
    assert cert.M.shape == (1, 3, 3) and cert.full_error_gradient.shape == (1, 3)
    frozen_metric_gradient = 2 * (cert.M @ sample.error[..., None]).squeeze(-1)
    assert not torch.allclose(cert.full_error_gradient, frozen_metric_gradient, atol=1e-8)
    before = physical.env.q.copy()
    result = facade.control(actor, sample)
    decision, expected = native.calculate_action_from_certificate(actor, data, bank, physical)
    torch.testing.assert_close(result.action["mode"], decision[0], rtol=0, atol=0)
    torch.testing.assert_close(result.action["u"], decision[1], rtol=0, atol=0)
    torch.testing.assert_close(result.certificate.M, expected["M"], rtol=0, atol=0)
    np.testing.assert_array_equal(physical.env.q, before)
    soft = facade.control(actor, sample, training=True)
    assert soft.action is None and soft.policy_logits.shape == (1, 12 * native.cfg.train_speed_points)
    derivative = torch.autograd.grad(soft.policy_logits.sum(), tuple(actor.parameters()), allow_unused=True)
    assert any(x is not None and x.norm() > 0 for x in derivative)
    with pytest.raises(CapabilityError):
        facade.control(torch.eye(3), sample)
    with pytest.raises(CapabilityError):
        facade.require_algorithm("episode_metric_ppo")
    physical.env.close()


class _VisualAdapterDouble:
    """Only exercises facade cache/version semantics; not original parity."""
    def reference(self, ae, metric, rgb, rotation, **kwargs):
        return SimpleNamespace(j=torch.eye(16, 4)[None])

    def encode_error(self, ae, current, goal, device):
        a, b = ae(current), ae(goal)
        return (a-b)/4, b

    def calculate_action_from_M(self, M, error, reference):
        return (reference.j.transpose(-1, -2) @ M @ error[..., None]).squeeze(-1)


def test_visual_facade_discloses_partial_support_and_enforces_reference_version():
    ae, metric = torch.nn.Linear(16, 16), torch.nn.Linear(16, 136)
    facade = OriginalVisualMFacade(_VisualAdapterDouble(), ae, metric)
    goal, current = torch.zeros(1, 16), torch.ones(1, 16)
    with pytest.raises(CapabilityError):
        facade.error_context(current)
    facade.bind_reference(goal, np.eye(3))
    sample = facade.error_context(current, representation_grad=True)
    M = torch.eye(16)[None].requires_grad_()
    result = facade.control(M, sample, training=True)
    derivatives = torch.autograd.grad(result.action.sum(), (M, ae.weight))
    assert all(x.norm() > 0 for x in derivatives)
    assert facade.describe().metric_scope == MetricScope.GOAL_CONDITIONED
    assert facade.describe().support_level == "partial_control_facade"
    assert not facade.describe().gradients.reference_wrt_representation
    with pytest.raises(CapabilityError):
        facade.require_algorithm("episode_metric_ppo")
    facade.bind_reference(goal+1, np.eye(3))
    with pytest.raises(CapabilityError):
        facade.control(M, sample)
    with torch.no_grad():
        ae.weight.add_(.01)
    with pytest.raises(CapabilityError):
        facade.error_context(current)


def test_original_visual_real_cpu_control_parity_without_renderer_or_training():
    from stabilization_gym.envs.tasks.original_visual import OriginalVisualAdapter
    root = Path(__file__).resolve().parents[1]
    bundle = root / "assets/original_visual_initialization"
    if not (bundle / "initial.pt").exists():
        pytest.skip("Retained original visual bundle unavailable")
    adapter = OriginalVisualAdapter(root)
    ae, metric = adapter.load_models("initial.pt", device="cpu")
    shard = next((bundle / "replay").glob("bootstrap_*.npz"))
    with np.load(shard, allow_pickle=False) as data:
        current, goal, rotation, depth = data["rgb"][0], data["goal_rgb"], data["rotation"], data["goal_depth"]
    facade = OriginalVisualMFacade(adapter, ae, metric)
    reference = facade.bind_reference(goal, rotation, depth=depth, device="cpu")
    sample = facade.error_context(current)
    with torch.no_grad():
        _, original_M = metric(sample.metric_context["z_goal"])
    M = original_M.detach().requires_grad_()
    result = facade.control(M, sample, training=True)
    direct = adapter.calculate_action_from_M(M, sample.error, reference)
    torch.testing.assert_close(result.action, direct, rtol=0, atol=0)
    torch.testing.assert_close(result.certificate.value, adapter.certificate(M, sample.error), rtol=0, atol=0)
    derivative, = torch.autograd.grad(result.action.sum(), M)
    assert derivative.norm() > 0 and torch.isfinite(derivative).all()
    assert result.action.shape == (1, 4) and not reference.j.requires_grad


def test_missing_capability_and_unknown_assessment_are_not_success():
    with pytest.raises(CapabilityError):
        PortableMFacade(SimpleNamespace())
    assessment = LearningAssessment()
    assert not assessment.analytical_guarantee["established"]
    assert assessment.trajectory_decrease == {}
    assert assessment.task_performance == {}


def test_audit_wrapper_is_not_misreported_as_a_pure_control_query():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m_learning_suite.task_runtime import make_task
    env = make_task("point_mass", {"max_steps": 20})
    task = PortableMFacade(env)
    obs, _ = task.reset(seed=41)
    before = env.state.copy()
    sample = task.error_context(obs)
    task.control(torch.eye(task.error_dim), sample)
    np.testing.assert_array_equal(env.state, before)
    assert env._pending_metric is not None
    assert "audit_pending_metric" in task.describe().control_call_side_effects
    assert not task.describe().control_call_is_pure
    with pytest.raises(CapabilityError):
        task.require_capability("pure_control")
    task.close()
