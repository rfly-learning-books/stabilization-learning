"""Independent environment-package installation and Gymnasium entry points."""
import subprocess
import sys

import gymnasium as gym
import pytest
import torch
import stabilization_gym


@pytest.mark.parametrize("env_id", tuple(stabilization_gym.GYM_ENVIRONMENTS))
def test_gymnasium_registration_exposes_real_controller_and_step(env_id):
    env = gym.make(env_id, max_steps=2)
    obs, _ = env.reset(seed=41)
    task = env.unwrapped
    e, context = task.build_error(obs)
    u = task.calculate_action_from_M(torch.eye(task.error_dim).to(e), e, context=context)
    _, cost, terminated, truncated, info = env.step(u.detach().cpu().numpy())
    assert cost >= 0
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "physical_error" in info
    env.close()


def test_environment_package_does_not_import_learning_package():
    code = "import sys; import stabilization_gym; e=stabilization_gym.make('point_mass'); e.reset(seed=41); assert not any(n=='stabilization_learning' or n.startswith('stabilization_learning.') for n in sys.modules)"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)


def test_asset_free_installation_does_not_require_legacy_workspace(tmp_path, monkeypatch):
    from stabilization_gym.common import paths
    monkeypatch.delenv("STABILIZATION_ASSET_CONFIG", raising=False)
    monkeypatch.setenv("STABILIZATION_ASSET_ROOT", str(tmp_path / "optional_assets"))
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "standalone" / "paths.py"))
    assert paths.asset_path("fastdepth") == tmp_path / "optional_assets" / "fastdepth"
    assert not paths.asset_path("fastdepth").exists()
