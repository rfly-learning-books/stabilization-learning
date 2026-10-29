from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from extensions.reproducible_training_v4.push import (
    SafePushAnyShapeEnv, source_files_for_config, make_timeaware_task, PROTOCOL,
)


@pytest.fixture
def env():
    value = SafePushAnyShapeEnv(max_steps=4, speed_points=3)
    yield value
    value.close()


def test_actual_library_and_sources_are_frozen(env):
    paths = source_files_for_config()
    assert len([p for p in paths if p.endswith("report.json")]) == 32
    assert len([p for p in paths if p.endswith("library.npz")]) == 32
    assert set(paths) == set(env.asset_manifest["sources"])
    assert all(Path(path).is_file() and len(sha) == 64 for path, sha in env.asset_manifest["sources"].items())
    assert env.asset_manifest["protocol"] == PROTOCOL


def test_seeded_actual_fixed_metric_rollout_repeats(env):
    traces = []
    for _ in range(2):
        observation, info = env.reset(seed=905)
        assert "no_SO101_transfer_time" in info["time_protocol"]
        trace = []
        for step in range(4):
            error, context = env.build_error(observation)
            matrix = torch.eye(3, requires_grad=True)
            candidates = env.candidate_scores(matrix, error, context=context)
            candidates["V"].sum().backward()
            assert torch.isfinite(matrix.grad).all()
            action = env.calculate_action_from_M(matrix, error, context=context)
            observation, cost, terminated, truncated, info = env.step(action)
            trace.append((observation["pose"].tolist(), action.tolist(), cost, terminated, truncated))
            if terminated or truncated:
                break
        traces.append(trace)
    assert traces[0] == traces[1]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_metric_fails_closed_without_step(env, value):
    observation, _ = env.reset(seed=9)
    error, context = env.build_error(observation)
    before = deepcopy(env.plant.vector.q)
    matrix = torch.eye(3)
    matrix[0, 0] = value
    with pytest.raises(FloatingPointError):
        env.calculate_action_from_M(matrix, error, context=context)
    assert np.array_equal(before, env.plant.vector.q)
    with pytest.raises(FloatingPointError, match="stopped"):
        env.reset(seed=9)


@pytest.mark.parametrize("matrix", [torch.zeros(3, 3), torch.diag(torch.tensor([-1., 1., 1.])),
                                    torch.tensor([[1., .2, 0], [0, 1., 0], [0, 0, 1.]])])
def test_invalid_metric_rejected(env, matrix):
    observation, _ = env.reset(seed=9)
    error, context = env.build_error(observation)
    with pytest.raises(ValueError):
        env.candidate_scores(matrix, error, context=context)


@pytest.mark.parametrize("action", [[.5, .1], [1, float("nan")], [1, float("inf")]])
def test_invalid_action_never_advances(env, action):
    env.reset(seed=4)
    before = env.plant.vector.q.copy()
    with pytest.raises((ValueError, FloatingPointError)):
        env.step(action)
    assert np.array_equal(before, env.plant.vector.q)


def test_nonfinite_successor_latches(env, monkeypatch):
    env.reset(seed=4)
    original = env.plant.step
    def corrupt(action):
        obs, reward, term, trunc, info = original(action)
        obs["pose"][0] = np.nan
        return obs, reward, term, trunc, info
    monkeypatch.setattr(env.plant, "step", corrupt)
    with pytest.raises(FloatingPointError, match="transition"):
        env.step([1., 0.])
    with pytest.raises(FloatingPointError, match="stopped"):
        env.step([1., 0.])


@pytest.mark.parametrize("option,value", [("max_steps", 2.5), ("max_steps", True), ("speed_points", 1)])
def test_invalid_integer_options_rejected(option, value):
    with pytest.raises(ValueError):
        SafePushAnyShapeEnv(**{option: value})


def test_timeaware_factory_selects_namespaced_v2(tmp_path):
    from test_unified_v3_environment_stability import _bank
    bank = _bank(tmp_path)
    task = make_timeaware_task(shapes=["T"], bank=bank, seed=4)
    assert type(task.env.plant).__module__.startswith("third_party.so101_timeaware_v2")
    with pytest.raises(ValueError):
        task.env.step(.9)
