"""End-to-end objective/optimizer equivalence, cache lifetime, shared contracts."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from stabilization_gym.tasks_v2.contracts import TaskAdapter
from stabilization_gym.tasks_v2.visual import VisualObservation
from stabilization_learning.joint_v2.visual import engineering_fixture
from stabilization_learning.joint_v3.features import AcceleratedRGBFeatures
from stabilization_learning.joint_v3.visual import AcceleratedVisualLearner
from stabilization_learning.joint_v3.inference import FrozenRGBPolicy

torch.set_num_threads(1)


def equivalent_pair():
    reference, batch = engineering_fixture()
    reference.warmup(batch)
    reference.calibrate(batch)
    reference._zero()
    features = AcceleratedRGBFeatures(deepcopy(reference.features.backbone), 4).double()
    features.load_state_dict(reference.features.state_dict())
    fast = AcceleratedVisualLearner(features, deepcopy(reference.metric), config=reference.config)
    fast.feature_optimizer.load_state_dict(deepcopy(reference.feature_optimizer.state_dict()))
    fast.metric_optimizer.load_state_dict(deepcopy(reference.metric_optimizer.state_dict()))
    fast.counts, fast.version = reference.counts.copy(), reference.version
    return reference, fast, batch


def test_full_loss_endpoints_and_adam_updates_match_reference():
    reference, fast, batch = equivalent_pair()
    a, b = reference.losses(batch), fast.losses(batch)
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(reference.same_version_difference(batch), fast.same_version_difference(batch))
    assert isinstance(fast.adapter, TaskAdapter)
    for method in ("update_features", "update_metric"):
        expected = getattr(reference, method)(batch)
        actual = getattr(fast, method)(batch)
        assert expected.keys() == actual.keys()
        for name in expected:
            assert actual[name] == pytest.approx(expected[name], rel=2e-7, abs=2e-9), name
        for original, accelerated in ((reference.features,fast.features), (reference.metric,fast.metric)):
            for x, y in zip(original.parameters(), accelerated.parameters()):
                torch.testing.assert_close(x, y, rtol=2e-7, atol=2e-9)
        assert fast.version == reference.version and fast.counts == reference.counts


def test_frozen_goal_cache_matches_live_policy_and_rejects_update():
    _, fast, batch = equivalent_pair()
    session = FrozenRGBPolicy(fast)
    with torch.no_grad():
        expected = fast.policy(batch.observation, create_graph=False)
    result = session(batch.observation, goal_keys=(1,2,3))
    for name in ("action", "interaction", "error", "metric", "value"):
        torch.testing.assert_close(getattr(expected,name), getattr(result,name))
    # New states get a new L even when the goals stay cached.
    changed = replace(batch.observation, image=batch.observation.image + .02)
    fresh = session(changed, goal_keys=(1,2,3))
    assert not torch.equal(result.interaction, fresh.interaction)
    fast.update_features(batch)
    with pytest.raises(RuntimeError, match="version changed"):
        session(batch.observation, goal_keys=(1,2,3))
    with torch.no_grad():
        post = fast.policy(batch.observation, create_graph=False)
    torch.testing.assert_close(FrozenRGBPolicy(fast)(batch.observation).action, post.action)


def test_cache_rejects_loading_or_coordinate_mutation_even_without_version_increment():
    _, fast, batch = equivalent_pair()
    session = FrozenRGBPolicy(fast)
    fast.features.load_state_dict(deepcopy(fast.features.state_dict()))
    with pytest.raises(RuntimeError, match="version changed"):
        session(batch.observation)
    session = FrozenRGBPolicy(fast)
    fast.features.scale.mul_(1.1)
    with pytest.raises(RuntimeError, match="version changed"):
        session(batch.observation)


def test_identity_goal_cache_and_bounded_eviction():
    _, fast, batch = equivalent_pair()
    session = FrozenRGBPolicy(fast, identity=True, max_goals=1)
    result = session(batch.observation, goal_keys=(1,2,3))
    torch.testing.assert_close(result.metric, torch.eye(4,dtype=torch.float64).expand(3,-1,-1))
    assert len(session._goals) == 1
    torch.testing.assert_close(result.action, session(batch.observation, goal_keys=(1,2,3)).action)


def test_standalone_checkpoint_backend_is_checked(tmp_path):
    _, fast, batch = equivalent_pair()
    path = tmp_path/"model.pt"
    fast.save(path)
    fast.load(path, restore_rng=False)
    legacy_features = AcceleratedRGBFeatures(deepcopy(fast.features.backbone), 4, backend="legacy").double()
    other = AcceleratedVisualLearner(legacy_features, deepcopy(fast.metric))
    with pytest.raises(ValueError, match="mismatch"):
        other.load(path, restore_rng=False)
