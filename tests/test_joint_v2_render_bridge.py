"""Real audited-room RGB/plant checks; no policy optimization or old AE loads."""
from pathlib import Path

import numpy as np
import pytest
import torch

from stabilization_gym.tasks_v2.reconstructed_rgb import (
    load_runtime, ReconstructedRGBEnv, RGBGeometry, PUBLIC_KEYS, random_raw_autoencoder)


@pytest.fixture(scope="module")
def runtime():
    root = Path(__file__).resolve().parents[1]
    assert (root/"migration/visual_scene_reconstruction/SCENE_AUDIT.json").is_file()
    # Runtime construction is metadata/source audit, never checkpoint loading.
    original = torch.load
    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime construction may not load checkpoints")
    torch.load = forbidden
    try:
        adapter, cfg, scenes = load_runtime(root)
    finally:
        torch.load = original
    scene = next(s for s in scenes if s["split"] == "train")
    return adapter, cfg, scene


@pytest.mark.parametrize("plant", ["P0", "P1"])
def test_real_rgb_step_matches_model_and_keeps_truth_out_of_policy(runtime, plant):
    adapter, cfg, scene = runtime
    env = ReconstructedRGBEnv(cfg, scene, plant=plant, max_steps=2)
    try:
        obs, start = env.reset(seed=271, options={"difficulty": "small"})
        assert set(obs) == PUBLIC_KEYS
        assert env.observation_space.contains(obs)
        assert obs["rgb"].shape == (256, 256, 3)
        assert obs["rgb"].std() > 5
        command = np.array([.1, -.05, .03, .07])
        next_obs, reward, terminated, truncated, info = env.step(command)
        label = info["supervision"]
        assert set(next_obs) == PUBLIC_KEYS
        assert "actual_velocity" not in next_obs and "offset" not in next_obs and "depth" not in next_obs
        assert reward == -label["physical_cost"]
        assert not terminated and not truncated
        if plant == "P0":
            expected = adapter.core.p0_offset_step(torch.tensor(start["supervision"]["offset"]), torch.tensor(command)).numpy()
            # Frozen plant ACTION_SCALE stores float32 constants; the analytic
            # model uses Python .4/.35.  At this step their roundoff is <1e-9 m/rad.
            np.testing.assert_allclose(label["next_offset"], expected, atol=1e-9, rtol=0)
            np.testing.assert_allclose(label["actual_velocity"], np.array([.4, .4, .4, .35])*command)
        else:
            # First command is queued: P1 starts with zero actual velocity and prior command.
            np.testing.assert_array_equal(label["offset"], label["next_offset"])
            np.testing.assert_array_equal(label["actual_velocity"], np.zeros(4))
        obs2, _, terminated, truncated, info2 = env.step(command)
        assert truncated and not terminated
        assert info2["reason"] == "time_limit"
        if plant == "P1":
            assert np.linalg.norm(info2["supervision"]["actual_velocity"]) > 0
        with pytest.raises(RuntimeError):
            env.step(command)
    finally:
        env.close()


def test_invalid_commands_do_not_advance_physics(runtime):
    _, cfg, scene = runtime
    env = ReconstructedRGBEnv(cfg, scene)
    try:
        env.reset(seed=271)
        for bad in ([np.nan, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1.1], [0, 0]):
            with pytest.raises(ValueError):
                env.step(bad)
        assert env._steps == 0
    finally:
        env.close()


def test_public_depth_current_image_geometry_no_old_policy_load(runtime, monkeypatch):
    _, cfg, scene = runtime
    loaded = []
    original = torch.load
    def allowed_depth(path, *args, **kwargs):
        assert Path(path).name == "extracted_state_dict.pt"
        assert kwargs.get("weights_only") is True
        loaded.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", allowed_depth)
    env = ReconstructedRGBEnv(cfg, scene, max_steps=2)
    try:
        obs, info = env.reset(seed=271)
        geometry = RGBGeometry(cfg, "cpu")
        image, goal, tangents = geometry.observe(obs)
        assert image.shape == goal.shape == (1, 3, 128, 128)
        assert tangents.shape == (1, 4, 3, 128, 128)
        assert torch.isfinite(image).all() and torch.isfinite(tangents).all()
        assert tangents.abs().sum() > 0
        assert len(loaded) == 1
        assert geometry.provenance["simulator_depth_used"] is False
        with pytest.raises(ValueError, match="Only RGB"):
            geometry.observe(dict(obs, truth=info["supervision"]["offset"]))
        with pytest.raises(RuntimeError):
            env.instance.renderer.enable_depth_rendering()
    finally:
        env.close()


def test_success_requires_configured_hold(runtime):
    _, cfg, scene = runtime
    env = ReconstructedRGBEnv(cfg, scene, max_steps=20)
    try:
        env.reset(seed=271, options={"offset": np.zeros(4)})
        required = cfg["collection"]["required_hold_observations"]
        for index in range(required):
            _, _, terminated, truncated, info = env.step(np.zeros(4))
            assert terminated == (index == required-1)
            assert not truncated
        assert info["success"] and info["reason"] == "success"
    finally:
        env.close()


def test_random_legacy_backbone_does_not_load_pretrained_weights(runtime, monkeypatch):
    _, cfg, _ = runtime
    def forbidden(*args, **kwargs):
        raise AssertionError("Fresh AE construction may not read checkpoints")
    monkeypatch.setattr(torch, "load", forbidden)
    torch.manual_seed(271)
    model = random_raw_autoencoder(cfg, "cpu").eval()
    with torch.no_grad():
        code, reconstruction = model(torch.zeros(1, 3, 128, 128))
    assert code.shape == (1, cfg["encoder"]["latent_dim"])
    assert reconstruction.shape == (1, 3, 128, 128)
    assert torch.isfinite(code).all() and torch.isfinite(reconstruction).all()
