"""Stress-data isolation and held-out access checks for the overnight protocol."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pytest
from stabilization_gym.observation import make_observation_task
from stabilization_learning.observation.observers import GeometryFilterObserver
from extensions.observer_learning_v1 import pipeline


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_bounded_sensor_noise_stays_within_declared_support(task):
    env = make_observation_task(task)
    env.reset(options={"index": 37, "scenario": "bounded_noise"})
    trajectory = env.supervised_trajectory()
    ideal = np.array([env.model.measure(x) for x in trajectory["truth"]])
    residual = np.abs(trajectory["measurement"]-ideal)
    limits = np.broadcast_to(np.asarray(env.model.spec.sensor_std), residual.shape)
    assert np.all(residual[trajectory["mask"]] <= limits[trajectory["mask"]]+1e-10)


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_initialization_stress_changes_only_the_public_prior(task):
    normal, stress = make_observation_task(task), make_observation_task(task)
    a, _ = normal.reset(options={"index": 19})
    b, _ = stress.reset(options={"index": 19, "scenario": "initial_offset"})
    assert normal.supervision()["episode"]["replay_sha256"] == stress.supervision()["episode"]["replay_sha256"]
    assert not np.array_equal(a["estimate"], b["estimate"])
    for k in ("measurement", "measurement_mask", "known_input"):
        assert np.array_equal(a[k], b[k])


def test_wrong_match_is_a_measurement_corruption_with_same_physical_motion():
    clean, corrupt = make_observation_task("cooperative_pose"), make_observation_task("cooperative_pose")
    clean.reset(options={"index": 2, "scenario": "noise"})
    corrupt.reset(options={"index": 2, "scenario": "wrong_match"})
    a, b = clean.supervised_trajectory(), corrupt.supervised_trajectory()
    assert np.array_equal(a["truth"], b["truth"])
    assert np.array_equal(a["control"], b["control"])
    assert np.array_equal(a["measurement"][0], b["measurement"][0])
    assert not np.array_equal(a["measurement"], b["measurement"])


def test_geometry_baseline_requires_only_public_measurements_and_handles_missing_points():
    env = make_observation_task("cooperative_pose")
    public, _ = env.reset(options={"index": 3})
    baseline = GeometryFilterObserver(env.model)
    # Destroy privileged data: the observer has only the public model and mapping.
    env._truth[:] = np.nan
    assert np.isfinite(baseline.correction(public)).all()
    public["measurement_mask"][:] = 0
    public["measurement"][:] = 0
    assert np.array_equal(baseline.correction(public), np.zeros(12))


def test_no_formal_test_access_before_complete_unchanged_selection(tmp_path, monkeypatch):
    cfg = {"tasks": ["motor_speed"], "seeds": [41, 42]}
    def forbidden(*args, **kwargs):
        pytest.fail("Held-out task was accessed before selection verification")
    monkeypatch.setattr(pipeline, "make_observation_task", forbidden)
    with pytest.raises(RuntimeError, match="All observer selections"):
        pipeline.evaluate_policy(tmp_path, "motor_speed", "fixed_M", 0, cfg)
    pipeline.write(tmp_path/"GLOBAL_SELECTION_COMMITTED.json", {})
    with pytest.raises(RuntimeError, match="Incomplete global"):
        pipeline.evaluate_policy(tmp_path, "motor_speed", "fixed_M", 0, cfg)
    commitment = {}
    for seed in cfg["seeds"]:
        job = f"motor_speed_seed{seed}"
        checkpoint = tmp_path/"jobs"/job/"selected.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"frozen selected weights")
        commitment[job] = {"sha256": pipeline.sha(checkpoint)}
    pipeline.write(tmp_path/"GLOBAL_SELECTION_COMMITTED.json", commitment)
    checkpoint.write_bytes(b"tampered weights")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        pipeline.evaluate_policy(tmp_path, "motor_speed", "fixed_M", 0, cfg)
