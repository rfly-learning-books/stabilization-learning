"""Behavioral checks for sensor isolation, manifold physics and honest metrics."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from stabilization_gym.observation import make_observation_task, episode_seed, MotorModel, CartPoleModel, PoseModel
from stabilization_gym.observation.metrics import final_hold, summarize_episode, information_windows
from stabilization_learning.observation import make_observer, StateMetric, reference_metric, difference, value, local_decay_matrix
from stabilization_learning.observation.certificate import checked_matrix


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_public_observation_never_contains_truth_and_does_not_alias_private_state(task):
    env = make_observation_task(task); obs, info = env.reset()
    assert set(obs) == {"measurement", "measurement_mask", "confidence", "estimate", "known_input", "time", "dt"}
    assert set(info) == {"task", "time", "phase"}
    assert env.observation_space.contains(obs)
    truth = env.supervision()["truth"]
    obs["estimate"][:] = 99; obs["measurement"][:] = -99
    assert np.array_equal(env.supervision()["truth"], truth)
    next_obs, _, _, _, _ = env.step(np.zeros(env.model.spec.error_dim))
    assert np.max(np.abs(next_obs["estimate"])) < 99
    assert env.observation_space.contains(next_obs)


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_methods_see_identical_object_trajectory_but_different_internal_estimates(task):
    first = make_observation_task(task); second = make_observation_task(task)
    first.reset(options={"scenario": "noise", "index": 3}); second.reset(options={"scenario": "noise", "index": 3})
    a, b = first.model.spec.error_dim, second.model.spec.error_dim
    for _ in range(5):
        oa, *_ = first.step(np.zeros(a)); ob, *_ = second.step(np.full(b, .001))
        assert np.array_equal(first.supervision()["truth"], second.supervision()["truth"])
        assert np.array_equal(oa["measurement"], ob["measurement"])
        assert np.array_equal(oa["known_input"], ob["known_input"])
    assert not np.array_equal(oa["estimate"], ob["estimate"])


def test_whole_trajectory_split_and_initial_guess_are_independent_of_truth():
    seeds = {s: {episode_seed(s, i) for i in range(20)} for s in ("engineering", "train", "validation", "test")}
    assert len(set.union(*seeds.values())) == 80
    env = make_observation_task("cooperative_pose")
    a, _ = env.reset(options={"index": 1}); truth_a = env.supervision()["truth"]
    b, _ = env.reset(options={"index": 2}); truth_b = env.supervision()["truth"]
    assert np.array_equal(a["estimate"], b["estimate"])
    assert not np.array_equal(truth_a, truth_b)
    with pytest.raises(ValueError):
        env.reset(seed=42)


def test_motor_exact_zoh_and_speed_only_observation():
    model = MotorModel(); x = np.array([.3, -.2]); u = np.array([1.4])
    block = np.array([[-100., -10., 100.], [10., -10., 0.], [0., 0., 0.]])
    assert np.allclose(model.predict(x, u), (expm(.01*block)@np.r_[x, u])[:2], atol=1e-14)
    assert np.array_equal(model.measure(x), [-.2])
    assert np.linalg.matrix_rank(np.vstack([model.C, model.C@model.A])) == 2


def test_cartpole_reuses_existing_physics_without_full_state_observation():
    import torch
    from stabilization_gym.envs.classic_control.classic import CartPoleEnv
    model = CartPoleModel(); original = CartPoleEnv(dt=model.spec.dt)
    x = np.array([.1, -.2, .05, .12]); u = np.array([.3])
    expected = original.torch_step(torch.tensor(x), torch.tensor(u)).numpy()
    assert np.allclose(model.predict(x, u), expected, atol=1e-13)
    assert np.array_equal(model.measure(x), x[[0, 2]])


def test_pose_geometry_jacobian_matches_right_multiplicative_finite_difference():
    model = PoseModel(); x = model.initial_truth(np.random.default_rng(41))
    h = model.measurement_jacobian(x)
    for j in range(12):
        delta = np.zeros(12); delta[j] = 1e-6
        numeric = (model.measure(model.retract(x, delta))-model.measure(model.retract(x, -delta)))/2e-6
        assert np.allclose(h[:, j], numeric, atol=1e-6, rtol=1e-7)
    assert np.linalg.matrix_rank(h[:, :6]) == 6
    assert np.linalg.matrix_rank(PoseModel(True).measurement_jacobian(x)[:, :6]) < 6


def test_pose_error_uses_geodesic_and_is_quaternion_sign_invariant():
    model = PoseModel(); x = model.initial_estimate()
    delta = np.zeros(12); delta[3:6] = [.2, -.1, .3]
    changed = model.retract(x, delta)
    assert np.allclose(model.error(changed, x), delta, atol=1e-12)
    changed[3:7] *= -1
    assert np.allclose(model.error(changed, x), delta, atol=1e-12)
    assert np.linalg.norm(Rotation.from_quat(changed[3:7]).as_matrix().T@Rotation.from_quat(changed[3:7]).as_matrix()-np.eye(3)) < 1e-12


def test_missing_points_have_no_hidden_values_and_zero_metric_correction():
    env = make_observation_task("cooperative_pose"); obs, _ = env.reset(options={"scenario": "occlusion"})
    observer = make_observer("fixed_M", env.model)
    while not 1.6 <= obs["time"][0] < 2.4:
        obs, *_ = env.step(np.zeros(12))
    assert not obs["measurement_mask"].any()
    assert not obs["measurement"].any() and not obs["confidence"].any()
    assert np.array_equal(observer.correction(obs), np.zeros(12))


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_perfect_estimate_has_zero_correction_and_nominal_invariance(task):
    env = make_observation_task(task); obs, _ = env.reset()
    truth = env.supervision()["truth"]; obs["estimate"] = truth
    for name in ("fixed_M", "variable_M_probe", "kalman_ekf"):
        observer = make_observer(name, env.model)
        correction = observer.correction(obs)
        assert np.max(np.abs(correction)) < 1e-12
        predicted = env.model.predict(env.model.retract(truth, correction), obs["known_input"])
        assert np.allclose(predicted, env.model.truth_step(truth, obs["known_input"], 0., "nominal"), atol=1e-12)


def test_metric_changes_actual_correction_and_successor_must_be_recomputed():
    env = make_observation_task("motor_speed"); obs, _ = env.reset()
    obs["estimate"] = np.array([.5, .4]); obs["measurement"] = np.array([.9])
    fixed = make_observer("fixed_M", env.model); variable = make_observer("variable_M_probe", env.model)
    assert np.linalg.norm(variable.correction(obs)-fixed.correction(obs)) > 1e-8
    metric = variable.metric; a, b = np.array([.5, .4]), np.array([.9, .8]); e = np.array([.1, .2])
    result = difference(env.model, a+e, a, b+e, b, metric)
    assert abs(result["delta"] - e@(metric.matrix(b)-metric.matrix(a))@e) < 1e-12
    assert abs(result["delta"]) > 1e-4  # Error is fixed; metric movement matters.


def test_state_metric_directional_derivative_is_included_in_local_condition():
    model = MotorModel(); estimate = np.array([.2, .4]); control = np.array([1.])
    metric = reference_metric(model, .15)
    check = local_decay_matrix(model, estimate, control, metric)
    direction = model.A@estimate+model.B@control
    weights = np.linspace(.7, 1.3, 2)
    analytic = metric.base*.15*(1-np.tanh(weights@estimate)**2)*(weights@direction)
    assert np.allclose(check["metric_flow_derivative"], analytic, rtol=1e-6, atol=1e-7)
    assert np.linalg.norm(analytic) > .1
    fixed = local_decay_matrix(model, estimate, control, reference_metric(model))
    assert fixed["max_eigenvalue"] < 0


def test_metric_version_change_invalidates_certificate():
    metric = StateMetric(np.eye(2), .1); version = metric.version
    checked_matrix(metric, np.zeros(2), version)
    metric.amplitude = .2
    with pytest.raises(ValueError, match="version"):
        checked_matrix(metric, np.zeros(2), version)
    with pytest.raises(ValueError):
        StateMetric(np.diag([1., -1.]))


def test_success_requires_final_hold_and_full_horizon():
    times = np.arange(11)*.1
    assert final_hold(times, [False]*5+[True]*6, 1., .4)["success"]
    assert not final_hold(times, [True]*10+[False], 1., .4)["success"]
    assert not final_hold(times[:6], [True]*6, 1., .4)["success"]
    assert not final_hold(times, [True]*11, 1., .4, failed=True)["success"]


def test_failure_keeps_denominator_and_pads_unknown_tail_separately():
    env = make_observation_task("motor_speed"); env.reset()
    env.step(np.array([20., 0.]))
    record = env.evaluation_record(); metric = reference_metric(env.model)
    summary = summarize_episode(env.model, [record], lambda x, z, version: value(env.model, x, z, metric, version),
                                certificate_version=metric.version)
    assert summary["failure"] and not summary["success"] and not summary["complete"]
    assert summary["failure_tail_penalty"] >= 50.
    assert summary["normalized_error_integral_recorded"] == 0.
    assert summary["certificate"]["transitions"] == 0
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2))


def test_low_innovation_is_not_small_state_error():
    model = MotorModel(); truth = np.array([1., 0.]); estimate = np.zeros(2)
    assert np.array_equal(model.measure(truth), model.measure(estimate))
    assert value(model, truth, estimate, StateMetric(np.eye(2))) == 1.


def test_scalar_book_gain_boundary_is_not_assumed_stable_for_any_gain():
    for gain, expected in ((.2, .64), (.8, .04), (1.5, .25), (2.1, 1.21)):
        assert np.isclose((1-gain)**2, expected)
    gain = 2.1; error = 1.
    for _ in range(20):
        error = (1-gain)*error
    assert abs(error) > 1.


def test_double_integrator_offdiagonal_metric_produces_unmeasured_velocity_correction():
    a = np.array([[0., 1.], [0., 0.]]); c = np.array([[1., 0.]])
    metric = np.array([[2., -1.], [-1., 1.]])
    gain = np.linalg.solve(metric, c.T*2.)
    assert np.allclose(gain.ravel(), [2., 2.])
    q = a.T@metric+metric@a-4*c.T@c
    assert np.linalg.eigvalsh(q).max() < 0.
    assert np.linalg.solve(np.eye(2), c.T)[1, 0] == 0.
