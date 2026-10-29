"""CPU integration tests of the real task/controller boundary."""
import math

import numpy as np
import pytest
import torch
from torch import nn

from stabilization_learning.envs.push_any_shape import PushAnyShapeEnv, DEFAULT_LIBRARY
from stabilization_learning.envs.visual_servo import VisualServoEnv, LegacyVisualServoEnv


def test_push_real_library_step_and_candidate_gradient():
    if not (DEFAULT_LIBRARY / "T/library/library.npz").is_file():
        pytest.skip("External calibrated shape library not installed")
    env = PushAnyShapeEnv(max_steps=3)
    try:
        obs, _ = env.reset(seed=41)
        e, context = env.build_error(obs)
        M = torch.eye(3, requires_grad=True)
        candidate = env.candidate_scores(M, e, context=context)
        derivative, = torch.autograd.grad(candidate["V_next"].sum(), M)
        assert torch.isfinite(derivative).all() and derivative.norm() > 0
        action = env.calculate_action_from_M(M, e, context=context)
        assert not action.requires_grad
        previous = obs["pose"].copy()
        obs, cost, terminated, truncated, info = env.step(action)
        assert cost >= 0 and np.isfinite(cost)
        assert not np.array_equal(previous, obs["pose"])
        assert info["extrapolated"] in (True, False)
        assert env.metric_context(obs).shape == (39,)
        assert env.render().shape == (256, 256, 3)
        assert len(env.asset_manifest["library_fingerprints"]) == 32
    finally:
        env.close()


def test_visual_M_gradient_matches_finite_difference_and_physical_step():
    torch.set_num_threads(2)
    env = VisualServoEnv(max_steps=5)
    obs, _ = env.reset(seed=41)
    e, context = env.build_error(obs)
    M = torch.eye(16, requires_grad=True)
    action = env.calculate_action_from_M(M, e, context=context)
    direction = torch.zeros_like(M)
    direction[1, 2] = direction[2, 1] = 1
    scalar = action @ torch.tensor([.4, -.2, .1, .3])
    analytic = (torch.autograd.grad(scalar, M)[0]*direction).sum()
    eps = .002
    def f(m):
        return env.calculate_action_from_M(m, e, context=context) @ torch.tensor([.4, -.2, .1, .3])
    numeric = (f(M.detach()+eps*direction)-f(M.detach()-eps*direction))/(2*eps)
    torch.testing.assert_close(analytic, numeric, rtol=.03, atol=1e-5)
    assert action[:3].norm() <= .4 and action[3].abs() <= .35
    expected = env.predict_next(torch.from_numpy(env.state.copy()), action.detach().double()).numpy()
    obs, cost, _, _, info = env.step(action)
    np.testing.assert_allclose(env.state, expected, atol=1e-7)
    assert cost >= 0 and info["depth_source"] == "constant"
    assert set(obs) == {"rgb", "goal_rgb"}
    assert not {"state", "pose", "offset", "depth"} & context.keys()
    assert env.metric_context(obs).shape == (16,)
    env.close()


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(48, 16))

    def forward(self, x):
        return self.net(x)


def test_shared_phi_gradient_and_episode_reference_version():
    torch.manual_seed(4)
    phi = SharedEncoder()
    env = VisualServoEnv(representation=phi, max_steps=2)
    obs, _ = env.reset(seed=41)
    e, context = env.build_error(obs, representation=phi)
    M = torch.diag(torch.linspace(.5, 1.5, 16)).requires_grad_()
    action = env.calculate_action_from_M(M, e, context=context)
    gradients = torch.autograd.grad(action[0] + e.square().sum(), tuple(phi.parameters()), allow_unused=True)
    assert any(g is not None and g.norm() > 0 for g in gradients)
    assert context["J"].requires_grad
    with torch.no_grad():
        phi.net[-1].weight.add_(.001)
    with pytest.raises(RuntimeError, match="Phi changed"):
        env.step(action)
    # A reset acknowledges the new representation and rebuilds goal/J*.
    obs, _ = env.reset(seed=41)
    assert env.build_error(obs)[0].shape == (16,)
    env.close()


def test_visual_split_and_terminal_semantics():
    train = VisualServoEnv(split="train", max_steps=1)
    test = VisualServoEnv(split="test", max_steps=1)
    o1, _ = train.reset(seed=7)
    o2, _ = test.reset(seed=7)
    assert not np.array_equal(o1["goal_rgb"], o2["goal_rgb"])
    _, _, terminated, truncated, info = train.step(torch.zeros(4))
    assert not terminated and truncated and not info["success"]
    with pytest.raises(RuntimeError, match="reset"):
        train.step(torch.zeros(4))
    train.close()
    test.close()


def test_missing_legacy_assets_are_never_silently_replaced(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required assets"):
        LegacyVisualServoEnv(legacy_root=tmp_path)


def test_invalid_visual_action_fails_before_state_changes():
    env = VisualServoEnv()
    env.reset(seed=41)
    state = env.state.copy()
    with pytest.raises(ValueError, match="physical limits"):
        env.step(torch.ones(4))
    np.testing.assert_array_equal(env.state, state)
    env.close()


def test_push_matches_original_fixed_M_candidates_hard_choice_and_step():
    """Independent upstream code supplies the baseline, including recovery."""
    if not (DEFAULT_LIBRARY / "T/library/library.npz").is_file():
        pytest.skip("External calibrated shape library not installed")
    env = PushAnyShapeEnv(max_steps=3)
    upstream = env._control
    M = torch.tensor([[1.7, .4, 0.], [.4, .8, .1], [0., .1, .5]])

    class FixedMetric(nn.Module):
        def encoder(self, C, valid, ell):
            return torch.zeros((len(C), 1), dtype=C.dtype, device=C.device)

        def conditioned(self, D, condition):
            m = M.to(D).expand(*D.shape[:-1], 3, 3)
            return torch.linalg.cholesky(m), m

        def value(self, D, condition):
            return torch.einsum("...i,ij,...j->...", D, M.to(D), D)

    original = upstream.FullMControlEnv(env.bank, env.cfg, shape_names=["T"], num_envs=1)
    try:
        obs, _ = env.reset(seed=4)
        data = original.reset(seed=4)
        for _ in range(3):
            e, context = env.build_error(obs)
            expected = upstream.candidates(FixedMetric(), data, env.bank, env.cfg,
                                           training=False, speed_points=env.speed_points)
            got = env.candidate_scores(M, e, context=context)
            for key in ("V", "V_next", "e_next", "flow", "endpoint"):
                torch.testing.assert_close(got[key], expected[key], rtol=1e-5, atol=2e-6)
            assert torch.equal(got["feasible"], expected["feasible"])
            assert torch.equal(got["physical"], expected["physical"])
            mode, speed, recovery, fail = upstream.hard_choice(expected, data, env.cfg,
                torch.as_tensor(original.recovery), torch.as_tensor(original.at_goal()))
            action = env.calculate_action_from_M(M, e, context=context)
            torch.testing.assert_close(action, torch.stack([mode[0].float(), speed[0]]))
            obs, cost, term, trunc, info = env.step(action)
            data, _, done, _ = original.advance(mode.numpy(), speed.numpy(),
                (~expected["feasible"].any(-1)).numpy(), recovery.numpy(), fail.numpy())
            np.testing.assert_allclose(obs["pose"], original.obs["pose"][0], atol=1e-10)
            assert (term or trunc) == bool(done[0])
            assert info["recovery"] == bool(original.recovery[0])
            assert cost == pytest.approx(env.dt*env.physical_cost(obs))
            if term or trunc:
                break
    finally:
        env.close()
        original.env.close()


def test_push_M_changes_actual_action_and_rejects_fractional_mode():
    if not (DEFAULT_LIBRARY / "T/library/library.npz").is_file():
        pytest.skip("External calibrated shape library not installed")
    env = PushAnyShapeEnv()
    try:
        obs, _ = env.reset(seed=0)
        e, context = env.build_error(obs)
        unit = env.calculate_action_from_M(torch.eye(3), e, context=context)
        shaped = env.calculate_action_from_M(torch.tensor([[1.7, .4, 0.], [.4, .8, .1], [0., .1, .5]]), e, context=context)
        assert not torch.equal(unit, shaped)
        old = obs["pose"].copy()
        with pytest.raises(ValueError, match="integer mode"):
            env.step({"mode": 1.2, "u": [0.]})
        np.testing.assert_array_equal(env._obs["pose"], old)
        import importlib
        numeric = importlib.import_module("dl_push.certificate.numeric_env")
        expected = numeric.exact_step(old, obs["C"][:, int(shaped[0])], float(shaped[1]), env.dt)
        following, _, _, _, info = env.step(shaped)
        np.testing.assert_allclose(following["pose"], expected, atol=1e-10)
        np.testing.assert_allclose(info["applied_action"], shaped.numpy())
    finally:
        env.close()


def test_push_success_timeout_and_grid_failure_fixed_horizon_cost():
    if not (DEFAULT_LIBRARY / "T/library/library.npz").is_file():
        pytest.skip("External calibrated shape library not installed")
    from stabilization_learning.wrappers import rollout_metric
    env = PushAnyShapeEnv(max_steps=10)
    try:
        target = np.array([.25, .25, 0.])
        obs, _ = env.reset(seed=41, options={"pose": target, "target": target})
        success = rollout_metric(env, torch.eye(3), obs)
        assert success["success"] and not success["failure"]
        assert success["physical_steps"] == 5 and success["physical_terminated"]
        assert not success["physical_truncated"] and success["fixed_horizon_cost"] == pytest.approx(0.)
        obs, _ = env.reset(seed=41)
        # Exercise a completely unavailable mode set without mocking physics,
        # hard_choice, termination, or fixed-horizon accounting.
        env.bank.valid[int(obs["shape_id"])].zero_()
        failed = rollout_metric(env, torch.eye(3), obs)
        assert failed["failure"] and not failed["success"]
        assert failed["physical_steps"] == 1 and failed["physical_terminated"]
        assert not failed["physical_truncated"]
        assert failed["fixed_horizon_cost"] == pytest.approx(
            failed["accumulated_cost"] + 9*env.dt*max(25., failed["terminal_physical_error"]))
        assert failed["arrival_time"] == env.max_steps*env.dt
    finally:
        env.close()
    timeout_env = PushAnyShapeEnv(max_steps=1)
    try:
        obs, _ = timeout_env.reset(seed=41)
        result = rollout_metric(timeout_env, torch.eye(3), obs)
        assert result["physical_truncated"] and not result["physical_terminated"]
        assert not result["success"] and not result["failure"]
    finally:
        timeout_env.close()


@pytest.mark.parametrize("kind", ["push", "visual"])
def test_physical_action_space_samples_are_executable(kind):
    if kind == "push" and not (DEFAULT_LIBRARY / "T/library/library.npz").is_file():
        pytest.skip("External calibrated shape library not installed")
    env = PushAnyShapeEnv(max_steps=50) if kind == "push" else VisualServoEnv(max_steps=50)
    try:
        env.reset(seed=41)
        env.action_space.seed(9)
        samples = [env.action_space.sample() for _ in range(50)]
        env.action_space.seed(9)
        np.testing.assert_array_equal(samples[0], env.action_space.sample())
        for action in samples:
            assert env.action_space.contains(action)
            assert action.shape == env.action_space.shape
            if kind == "push":
                assert action[0] == int(action[0]) and 0 <= action[1] <= .6
            else:
                assert np.linalg.norm(action[:3]) <= .4 and abs(action[3]) <= .35
            _, cost, terminated, truncated, _ = env.step(action)
            assert np.isfinite(cost) and cost >= 0
            if terminated or truncated:
                env.reset(seed=41)
        invalid = [1.2, .1] if kind == "push" else [.4, .4, .4, 0.]
        assert not env.action_space.contains(invalid)
        assert not env.action_space.contains(np.full(env.action_space.shape, np.nan))
        assert not env.action_space.contains(np.zeros(7))
        with pytest.raises(ValueError):
            env.step(invalid)
    finally:
        env.close()
