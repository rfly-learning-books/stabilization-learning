"""Same rollout loop executes state inputs and internal observer updates."""
import numpy as np
import pytest
import torch

from stabilization_gym.interfaces.certificate_v2 import CertificateEvaluator, VersionBinding
from stabilization_gym.tasks_v2.state_observer import make_state_control, ObservationReplayEnv, ObserverTaskAdapter
from stabilization_learning.observation import make_observer
from stabilization_learning.joint_v2.rollout import PhysicalActionSession, ObserverSession, RolloutDriver
from stabilization_gym.tasks_v2.contracts import ActionPlan


def binding():
    fields = {key: "test" for key in VersionBinding.__dataclass_fields__}
    fields["metric_scope"] = "state_dependent"
    return VersionBinding(**fields)


def test_shared_driver_completes_real_state_episode_with_negative_cost():
    physical, adapter = make_state_control("point_mass", max_steps=2)
    certificate = CertificateEvaluator(lambda e, c: e.new_tensor(adapter.model.P0), binding())
    session = PhysicalActionSession(physical)
    start = session.reset(seed=51)
    driver = RolloutDriver(adapter, certificate, session.apply_plan, binding=certificate.binding)
    result = driver.run(start, max_steps=10)
    assert len(result.steps) == 2
    assert result.truncated and not result.terminated and not result.collector_cutoff
    assert result.total_reward == -sum(row.info["physical_cost"] for row in result.steps)
    assert all(row.supervision is None for row in result.steps)
    assert all(not row.plan.action.requires_grad for row in result.steps)
    start.observation[:] = 999
    assert np.max(np.abs(result.steps[0].observation)) < 999


def test_same_driver_completes_observer_episode_without_sending_correction_to_plant():
    physical = ObservationReplayEnv("motor_speed")
    adapter = ObserverTaskAdapter(physical.model)
    certificate = make_observer("fixed_M", physical.model).metric
    session = ObserverSession(physical, adapter, supervised=True)
    start = session.reset(options={"index": 3})
    received = []
    original_step = physical.step
    def monitored(action):
        value = action.detach().numpy() if torch.is_tensor(action) else action
        received.append(np.asarray(value).copy())
        assert np.asarray(value).shape == (physical.model.spec.input_dim,)
        return original_step(action)
    physical.step = monitored
    result = RolloutDriver(adapter, certificate, session.apply_plan).run(start, max_steps=250)
    assert len(result.steps) == 201 and len(received) == 201
    assert result.truncated and not result.collector_cutoff
    assert result.steps[-1].supervision["complete"]
    assert result.steps[-1].supervision["dt"] == 0.
    assert result.steps[-1].plan.action.shape == (2,)
    assert result.steps[-1].applied_action.shape == (1,)
    for row in result.steps:
        assert "truth" not in row.observation and "truth" not in row.next_observation
        assert "supervision" not in row.info and row.supervision is not None
        assert row.reward == -row.supervision["physical_cost"]


def test_inference_observer_session_never_reads_privileged_labels():
    physical = ObservationReplayEnv("motor_speed")
    adapter = ObserverTaskAdapter(physical.model)
    certificate = make_observer("fixed_M", physical.model).metric
    session = ObserverSession(physical, adapter, supervised=False)
    start = session.reset()
    def forbidden():
        raise AssertionError("Inference may not request hidden truth")
    physical.supervision = forbidden
    result = RolloutDriver(adapter, certificate, session.apply_plan).run(start, max_steps=3)
    assert result.collector_cutoff and not result.truncated and not result.terminated
    assert all(row.supervision is None for row in result.steps)
    assert result.total_reward == 0.


def test_collector_cutoff_does_not_invent_task_timeout():
    physical, adapter = make_state_control("point_mass", max_steps=200)
    certificate = CertificateEvaluator(lambda e, c: torch.eye(4, dtype=e.dtype), binding())
    session = PhysicalActionSession(physical)
    start = session.reset(seed=5)
    result = RolloutDriver(adapter, certificate, session.apply_plan).run(start, max_steps=1)
    assert result.collector_cutoff and not result.truncated
    assert not result.steps[0].truncated


def test_candidate_distribution_is_not_mislabeled_as_behavior_likelihood():
    physical, adapter = make_state_control("point_mass", max_steps=2)
    certificate = CertificateEvaluator(lambda e, c: torch.eye(4, dtype=e.dtype), binding())
    original = adapter.construct_action
    def with_candidates(context, cert):
        plan = original(context, cert)
        plan.distribution = torch.distributions.Categorical(logits=torch.tensor([1., 2.]))
        return plan
    adapter.construct_action = with_candidates
    session = PhysicalActionSession(physical)
    result = RolloutDriver(adapter, certificate, session.apply_plan).run(session.reset(seed=5), max_steps=1)
    recorded = result.steps[0].candidate_distribution
    assert torch.allclose(recorded["probabilities"], torch.softmax(torch.tensor([1., 2.]), 0))
    assert recorded["behavior_log_probability_available"] is False
    assert result.steps[0].plan.distribution is None


def test_missing_physical_candidate_requires_explicit_failure_session():
    physical, _ = make_state_control("point_mass")
    session = PhysicalActionSession(physical)
    with pytest.raises(ValueError, match="No physical action"):
        session.apply_plan(None, ActionPlan(action=None, diagnostics={"failure": "no_physical_candidate"}))
