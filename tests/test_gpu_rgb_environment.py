"""Task-boundary regressions for the optional CUDA vector backend.

Run with the isolated benchmark interpreter; ordinary environments skip cleanly.
"""
from pathlib import Path
import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA environment test")

from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime, ReconstructedRGBEnv
from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def task():
    torch.set_num_threads(1)
    _, cfg, scenes = load_runtime(ROOT)
    return cfg, next(s for s in scenes if s["split"] == "train")


@pytest.mark.parametrize("plant", ["P0", "P1"])
def test_success_hold_and_next_step_reset_have_no_fake_transition(task, plant):
    cfg, scene = task
    gpu = GPUParallelRGBEnv(cfg, [scene], plant=plant, max_steps=200, cache_dir=ROOT/".cache/rgb_gpu_warp")
    old = ReconstructedRGBEnv(cfg, scene, plant=plant, max_steps=200)
    try:
        gpu.reset(options={"offsets": np.zeros((1, 4))})
        old.reset(options={"offset": np.zeros(4)})
        for i in range(cfg["collection"]["required_hold_observations"]):
            obs, r, t, tr, info = gpu.step(torch.zeros((1, 4), device="cuda"))
            _, r0, t0, tr0, info0 = old.step(np.zeros(4))
            assert t.item() == t0 and tr.item() == tr0
            assert info["success"].item() == info0["success"]
            assert abs(r.item()-r0) < 1e-10
        assert t.item() and info["success"].item()
        _, r, t, tr, info = gpu.step(torch.zeros((1, 4), device="cuda"))
        assert not info["transition_mask"].item()
        assert r.item() == 0 and not t.item() and not tr.item()
        assert obs["rgb"].is_cuda and obs["rgb"].dtype == torch.uint8
    finally:
        old.close(); gpu.close()


@pytest.mark.parametrize("plant", ["P0", "P1"])
def test_failure_and_cost_match_reference(task, plant):
    cfg, scene = task
    gpu = GPUParallelRGBEnv(cfg, [scene], plant=plant, max_steps=250, cache_dir=ROOT/".cache/rgb_gpu_warp")
    old = ReconstructedRGBEnv(cfg, scene, plant=plant, max_steps=250)
    try:
        old_obs, initial = old.reset(seed=388)
        gpu.reset(options={"offsets": initial["supervision"]["offset"][None]})
        command = np.array([0, 0, -1, 0], np.float32)
        for _ in range(250):
            _, r0, t0, tr0, info0 = old.step(command)
            _, r, t, tr, info = gpu.step(torch.as_tensor(command[None], device="cuda"))
            assert t.item() == t0 and tr.item() == tr0
            assert info["failure"].item() == info0["failure"]
            assert abs(r.item()-r0) < 1e-6
            np.testing.assert_allclose(info["supervision"]["next_offset"][0].cpu(),
                                       info0["supervision"]["next_offset"], atol=1e-7, rtol=0)
            if t0 or tr0:
                assert info0["failure"]
                break
    finally:
        old.close(); gpu.close()


def test_gpu_action_boundary_and_independent_lanes(task):
    cfg, scene = task
    gpu = GPUParallelRGBEnv(cfg, [scene]*2, max_steps=1, cache_dir=ROOT/".cache/rgb_gpu_warp")
    try:
        public, before = gpu.reset(seed=439)
        saved = public["rgb"].clone()
        with pytest.raises(ValueError):
            gpu.step(torch.zeros((2, 4)))
        with pytest.raises(ValueError):
            gpu.step(torch.full((2, 4), float("nan"), device="cuda"))
        with pytest.raises(ValueError):
            gpu.step(torch.ones((2, 4), device="cuda"))
        actions = torch.tensor([[.1, 0, 0, 0], [-.1, 0, 0, 0]], device="cuda")
        _, _, _, tr, info = gpu.step(actions)
        assert tr.all()
        assert info["supervision"]["next_offset"][0, 0] > before["supervision"]["offset"][0, 0]
        assert info["supervision"]["next_offset"][1, 0] < before["supervision"]["offset"][1, 0]
        assert torch.equal(public["rgb"], saved), "Renderer reused a previously returned observation"
        assert set(public) == {"rgb", "goal_rgb", "rotation", "previous_action"}
    finally:
        gpu.close()
