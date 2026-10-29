"""Public geometry batch equivalence on actual rendered training-room images."""
from pathlib import Path
import numpy as np
import pytest
import torch
from stabilization_gym.tasks_v2.reconstructed_rgb import RGBGeometry as ReferenceGeometry
from stabilization_gym.tasks_v3.reconstructed_rgb import BatchedRGBGeometry, ReconstructedRGBEnv, load_runtime


def test_real_training_room_geometry_matches_reference_and_excludes_truth():
    torch.set_num_threads(1)
    _, cfg, scenes = load_runtime(Path(__file__).resolve().parents[1])
    scene = next(s for s in scenes if s["split"] == "train")
    env = ReconstructedRGBEnv(cfg, scene, max_steps=2)
    try:
        first, _ = env.reset(seed=271)
        second, _, _, _, _ = env.step(np.array([.1, -.03, .05, .04]))
        geometry = BatchedRGBGeometry(cfg, "cpu", depth_batch_size=2)
        expected = [ReferenceGeometry.observe(geometry, public) for public in (first,second)]
        actual = geometry.observe_batch([first,second])
        for axis, value in enumerate(actual):
            torch.testing.assert_close(value, torch.cat([row[axis] for row in expected]), rtol=1e-5, atol=1e-6)
        assert len(geometry.cache_revision) == 64
        assert geometry.provenance["simulator_depth_used"] is False
        with pytest.raises(ValueError, match="Only RGB"):
            geometry.observe_batch([dict(first, truth=np.zeros(4))])
    finally:
        env.close()
