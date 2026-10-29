"""CPU-only boundary and rolling-clock regressions for the isolated Push env."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from extensions.unified_training_v3.environment import JointPushEnv, PushTaskAdapter, vendor_modules


def _toy_data(**overrides):
    vendor_modules()
    from third_party.so101_timeaware_v2.so101_timeaware.timing import TimeConfig
    cfg = TimeConfig(max_actions=12, max_seconds=25., **overrides)
    angle = np.arange(12) * 2 * np.pi / 12
    shape = dict(polygon=[[-.02,-.02],[.02,-.02],[.02,.02],[-.02,.02],[-.02,-.02]],
        points=np.c_[.02*np.cos(angle), .02*np.sin(angle)].tolist(),
        normals=np.c_[np.cos(angle), np.sin(angle)].tolist(),
        C=np.c_[np.cos(angle), np.sin(angle), np.where(np.arange(12)%2,5.,-5.)].tolist(),
        valid=[True]*12, transfer_seconds=np.full((13,12), .2).tolist(),
        reachable=np.ones((13,12), dtype=bool).tolist())
    return dict(schema="so101_timebank_v1", config=asdict(cfg), shapes={"T":shape})


def _bank(tmp_path, data=None):
    path = tmp_path / "synthetic_time_bank.json"
    path.write_text(json.dumps(_toy_data() if data is None else data), encoding="utf-8")
    return vendor_modules()["Bank"](path)


@pytest.mark.parametrize("field,value", [
    ("encoder_omega_multiplier", float("inf")), ("success_bonus", float("nan")),
    ("failure_penalty", float("inf")), ("residual_tolerance", float("nan")),
    ("max_actions", float("inf")), ("max_actions", 2.5), ("max_actions", True),
    ("travels", [True]), ("travels", [.002, float("inf")]),
])
def test_nonfinite_or_malformed_config_rejected(tmp_path, field, value):
    data = _toy_data()
    data["config"][field] = value
    with pytest.raises(ValueError):
        _bank(tmp_path, data)


def test_no_valid_home_contact_rejected_before_sampling(tmp_path):
    data = _toy_data()
    data["shapes"]["T"]["reachable"][12] = [False]*12
    with pytest.raises(ValueError, match="initial/home"):
        _bank(tmp_path, data)


@pytest.mark.parametrize("pose", [[.01, .01], [.01, .01, float("nan")], [[.01, .01, .1]]])
def test_invalid_reset_pose_does_not_mutate_live_plant(tmp_path, pose):
    env = JointPushEnv(_bank(tmp_path), ["T"], seed=71321)
    before = env.plant.state_dict()
    with pytest.raises(ValueError, match="finite vector"):
        env.reset(pose=pose)
    assert env.plant.state_dict() == before


@pytest.mark.parametrize("action", [1.5, True, float("nan"), -1, 48])
def test_invalid_action_is_not_rounded_or_executed(tmp_path, action):
    env = JointPushEnv(_bank(tmp_path), ["T"], seed=71321)
    before = env.plant.state_dict()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.plant.state_dict() == before


@pytest.mark.parametrize("key,value", [("q", [float("nan"),0.,0.]), ("elapsed", -1.),
    ("mode", 12), ("actions", 2.5), ("contact", "False"), ("rng", {})])
def test_checkpoint_rejection_is_atomic(tmp_path, key, value):
    env = JointPushEnv(_bank(tmp_path), ["T"], seed=71321)
    before = env.plant.state_dict()
    bad = deepcopy(before)
    bad[key] = value
    with pytest.raises((ValueError, TypeError)):
        env.plant.load_state_dict(bad)
    assert env.plant.state_dict() == before


def test_checkpoint_roundtrip_and_rolling_clock_remain_deterministic(tmp_path):
    bank = _bank(tmp_path)
    env = JointPushEnv(bank, ["T"], seed=71321)
    twin = JointPushEnv(bank, ["T"], seed=71322)
    rng = np.random.default_rng(71323)
    transitions = 0
    for _ in range(40):
        env.reset()
        twin.plant.load_state_dict(env.plant.state_dict())
        while not env.plant.done:
            action = int(rng.integers(48))
            before = env.plant.elapsed
            _, reward, _, _, info = env.step(action)
            twin.plant.step(action)
            assert twin.plant.state_dict() == env.plant.state_dict()
            assert np.isfinite(reward)
            assert before <= env.plant.elapsed <= bank.config.max_seconds + 1e-8
            assert info["dt"] == pytest.approx(sum(info["stages"].values()))
            assert info["dt"] == pytest.approx(env.plant.elapsed-before)
            transitions += 1
        assert env.plant.reason in ("success", "time_limit", "action_limit", "workspace")
    assert transitions >= 300


def _adapter_plan(logits, physical):
    adapter = object.__new__(PushTaskAdapter)
    candidates = dict(logits=torch.tensor([logits]), physical=torch.tensor([physical]),
        V=torch.tensor([1.]), V_next=torch.ones(1,len(logits)))
    adapter.model = SimpleNamespace(candidates=lambda *_: candidates)
    context = SimpleNamespace(memory={}, model_context={"batch": {}})
    return adapter.construct_action(context, object())


def test_deployment_excludes_masked_actions_even_below_old_logit_sentinel():
    plan = _adapter_plan([-1e9,-2e9,-3e9], [False,True,True])
    assert plan.action == 1
    assert plan.distribution.probs[0,0].item() == 0.


def test_deployment_reports_no_physical_action_and_rejects_nan():
    assert _adapter_plan([-1e9,-1e9], [False,False]).action is None
    with pytest.raises(FloatingPointError):
        _adapter_plan([float("nan"),-1e9], [True,False])
