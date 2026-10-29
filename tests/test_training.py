"""Short real PPO runs, exact optimizer resume, and objective-boundary checks."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch

from stabilization_learning.core.types import MetricSpec
from stabilization_learning.envs.classic import PointMassEnv
from stabilization_learning.envs.visual_servo import VisualServoEnv
from stabilization_learning.metrics import CholeskyMetric
from stabilization_learning.representations import RGBAutoEncoder
from stabilization_learning.trainer import StabilizationLearner, TrainConfig, module_hash
from stabilization_learning.wrappers import CostToReward, EpisodeMetricEnv, rollout_metric


def _exact_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _exact_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _exact_equal(a, b)
    else:
        assert left == right


def _learner(config=None, **env_kwargs):
    return StabilizationLearner(
        PointMassEnv(max_steps=24, initial_scale=.4, **env_kwargs),
        config=config or TrainConfig(episodes=8, batch_episodes=4, epochs=2,
                                     learning_rate=1e-3, validation_seeds=(1001, 1002)),
    )


def test_real_ppo_changes_metric_and_resumes_nonempty_optimizers_exactly(tmp_path):
    uninterrupted = _learner()
    initial_actor = deepcopy(uninterrupted.actor.state_dict())
    initial_metric = uninterrupted.actor(torch.zeros(4)).detach().clone()
    uninterrupted.train(tmp_path / "uninterrupted")
    assert uninterrupted.episodes == 8
    assert uninterrupted.updates == 2
    assert uninterrupted.physical_steps > 0
    assert uninterrupted.optimizer.state_dict()["state"]
    assert uninterrupted.value_optimizer.state_dict()["state"]
    assert any(not torch.equal(initial_actor[key], value)
               for key, value in uninterrupted.actor.state_dict().items())
    assert torch.linalg.vector_norm(uninterrupted.actor(torch.zeros(4))-initial_metric) > 1e-7
    assert uninterrupted.history[-1]["train"]["fixed_horizon_cost"] > 0.
    assert all(row["validation"]["objective"] >= row["validation"]["fixed_horizon_cost"]
               for row in uninterrupted.history)

    interrupted = _learner()
    interrupted.train(tmp_path / "resumed", max_updates=1)
    assert interrupted.episodes == 4
    checkpoint = torch.load(tmp_path / "resumed/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["optimizer"]["state"] and checkpoint["value_optimizer"]["state"]
    continued = _learner()
    continued.train(tmp_path / "resumed", resume=True)
    for name in ("actor", "value", "optimizer", "value_optimizer"):
        _exact_equal(getattr(uninterrupted, name).state_dict(), getattr(continued, name).state_dict())
    _exact_equal(uninterrupted.history, continued.history)
    _exact_equal(uninterrupted.best_actor, continued.best_actor)
    assert uninterrupted.best_score == continued.best_score
    assert uninterrupted.physical_steps == continued.physical_steps


def test_validation_alone_selects_best_and_test_evaluation_does_not_mutate_it(tmp_path):
    learner = _learner(TrainConfig(episodes=4, batch_episodes=2, epochs=2,
                                  validation_seeds=(1001, 1002)))
    learner.train(tmp_path / "selection")
    best_before = deepcopy(learner.best_actor)
    training_actor_before = deepcopy(learner.actor.state_dict())
    score_before = learner.best_score
    assert score_before == min(row["validation"]["objective"] for row in learner.history)
    result = learner.evaluate((2001, 2002), best=True)
    assert [row["seed"] for row in result["episodes"]] == [2001, 2002]
    assert learner.best_score == score_before
    _exact_equal(learner.best_actor, best_before)
    _exact_equal(learner.actor.state_dict(), training_actor_before)
    saved = torch.load(tmp_path / "selection/best.pt", map_location="cpu", weights_only=False)
    _exact_equal(saved["actor"], best_before)
    assert saved["score"] == score_before


def test_checkpoint_rejects_changed_actual_environment_and_training_protocol(tmp_path):
    learner = _learner()
    path = tmp_path / "checkpoint.pt"
    learner.save(path)
    # No env_kwargs were provided to either learner: the actual physical
    # configuration still has to be part of the resume protocol.
    changed_dt = _learner(dt=.08)
    with pytest.raises(ValueError, match="protocol"):
        changed_dt.restore(path)
    changed_goal = _learner(goal=[.2, 0., 0., 0.])
    with pytest.raises(ValueError, match="protocol"):
        changed_goal.restore(path)
    changed_config = _learner(TrainConfig(episodes=10, batch_episodes=4, epochs=2,
                                          learning_rate=1e-3, validation_seeds=(1001, 1002)))
    with pytest.raises(ValueError, match="protocol"):
        changed_config.restore(path)


def test_training_seeds_cannot_overlap_declared_validation():
    learner = _learner(TrainConfig(episodes=2, batch_episodes=2, epochs=1,
                                  seed=41, validation_seeds=(41,)))
    with pytest.raises(ValueError, match="collides"):
        learner._collect(1)


@pytest.mark.parametrize("kwargs", [
    {"episodes": 1.5}, {"batch_episodes": 0}, {"epochs": -1},
    {"std": float("inf")}, {"learning_rate": float("nan")},
    {"clip": 0.}, {"clip": 1.}, {"target_kl": -.1},
    {"certificate_weight": -.1}, {"validation_seeds": ()},
])
def test_invalid_training_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


class _OneStepTerminalEnv(gym.Env):
    """Fixture isolating finite-horizon padding from controller dynamics."""
    def __init__(self, failure):
        self.failure = failure
        self.metric_spec = MetricSpec(2)
        self.metric_context_dim = self.error_dim = 2
        self.dt, self.max_steps = .1, 10
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([1., 0.], np.float32), {}

    def metric_context(self, observation=None):
        return torch.zeros(2)

    def build_error(self, observation, representation=None):
        return torch.as_tensor(observation), {}

    def calculate_action_from_M(self, M, e, *, context):
        return -(M@e)

    def step(self, action):
        return np.array([1., 0.], np.float32), .1, True, False, {
            "physical_error": 1., "failure": self.failure, "success": not self.failure,
        }


def test_fast_failure_has_full_horizon_cost_and_failure_capped_arrival_time():
    failed_env = _OneStepTerminalEnv(True)
    obs, _ = failed_env.reset()
    failed = rollout_metric(failed_env, torch.eye(2), obs, certificate_weight=2., save_trajectory=True)
    assert failed["physical_steps"] == 1
    assert failed["accumulated_cost"] == pytest.approx(.1)
    assert failed["fixed_horizon_cost"] == pytest.approx(.1+9*.1*25.)
    assert failed["arrival_time"] == 1.
    assert failed["failure"] and not failed["success"]
    assert failed["objective"] == pytest.approx(failed["fixed_horizon_cost"]+2*failed["certificate_penalty"])
    assert len(failed["trajectory"]) == 1
    success_env = _OneStepTerminalEnv(False)
    obs, _ = success_env.reset()
    success = rollout_metric(success_env, torch.eye(2), obs)
    assert success["fixed_horizon_cost"] == pytest.approx(1.)
    assert success["arrival_time"] == .1
    assert success["fixed_horizon_cost"] < failed["fixed_horizon_cost"]


class _RecordingPointMass(PointMassEnv):
    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        self.recorded_metrics.append(M.detach().clone())
        return super().calculate_action_from_M(M, e, context=context,
                                               controller_state=controller_state, certificate=certificate)


def test_episode_metric_boundary_holds_one_metric_and_reports_positive_cost():
    physical = _RecordingPointMass(max_steps=12, initial_scale=.5)
    physical.recorded_metrics = []
    wrapper = EpisodeMetricEnv(physical, certificate_weight=.1)
    obs, _ = wrapper.reset(seed=2)
    raw = wrapper.metric.identity_raw()+.1
    next_obs, cost, terminated, truncated, info = wrapper.step(raw.numpy())
    assert cost > 0. and terminated and not truncated
    assert info["physical_steps"] == len(physical.recorded_metrics)
    assert len(physical.recorded_metrics) > 1
    torch.testing.assert_close(torch.tensor(obs), torch.tensor(next_obs))
    for matrix in physical.recorded_metrics:
        torch.testing.assert_close(matrix, physical.recorded_metrics[0])
    with pytest.raises(RuntimeError):
        wrapper.step(raw.numpy())


def test_reward_adapter_negates_cost_explicitly():
    cost_env = EpisodeMetricEnv(_OneStepTerminalEnv(False))
    reward_env = CostToReward(EpisodeMetricEnv(_OneStepTerminalEnv(False)))
    cost_env.reset(seed=4)
    reward_env.reset(seed=4)
    raw = CholeskyMetric(MetricSpec(2)).identity_raw().numpy()
    cost = cost_env.step(raw)[1]
    reward = reward_env.step(raw)[1]
    assert cost > 0 and reward == -cost


def test_explicit_phi_is_bound_to_visual_context_and_frozen_during_policy_collection():
    phi = RGBAutoEncoder(latent_dim=16, image_size=16)
    env = VisualServoEnv(image_size=16, max_steps=2)
    learner = StabilizationLearner(env, representation=phi,
                                   config=TrainConfig(episodes=2, batch_episodes=2, epochs=1,
                                                      validation_seeds=(1001,)))
    assert env.representation is phi
    assert all(not parameter.requires_grad for parameter in phi.parameters())
    before = module_hash(phi)
    obs, _ = env.reset(seed=12)
    image = torch.from_numpy(obs["goal_rgb"].copy()).permute(2, 0, 1)[None].float()/255.
    torch.testing.assert_close(learner._context(obs), phi(image)[0])
    learner._collect(1)
    assert module_hash(phi) == before == learner.representation_hash


def test_metric_wrapper_binds_explicit_phi_for_shared_context_and_control():
    phi = RGBAutoEncoder(latent_dim=16, image_size=16)
    physical = VisualServoEnv(image_size=16, max_steps=2)
    wrapped = EpisodeMetricEnv(physical, representation=phi)
    assert physical.representation is phi
    assert all(not parameter.requires_grad for parameter in phi.parameters())
    context, _ = wrapped.reset(seed=4)
    goal_rgb = torch.from_numpy(physical.render()[:, 16:].copy()).permute(2, 0, 1)[None].float()/255.
    torch.testing.assert_close(torch.from_numpy(context), phi(goal_rgb)[0])
    before = module_hash(phi)
    result = wrapped.step(wrapped.metric.identity_raw().numpy())
    assert result[2] and not result[3]
    assert module_hash(phi) == before
