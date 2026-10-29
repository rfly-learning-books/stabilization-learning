from pathlib import Path

import numpy as np
import pytest
import torch

from stabilization_learning.envs.classic import PointMassEnv
from stabilization_learning.integrations.slc import load_slc_policy, load_upstream, train_slc
from stabilization_learning.wrappers import EpisodeMetricEnv


def _factory():
    return EpisodeMetricEnv(PointMassEnv(max_steps=16, initial_scale=0.2), raw_bound=1.0)


def test_upstream_lac_updates_real_actor_and_critic_and_reloads(tmp_path):
    result = train_slc(_factory, tmp_path / "slc", steps=8, batch_size=4,
                       start_steps=2, update_after=4, hidden_sizes=(16, 16), seed=7)
    assert result["updates"] == 5
    assert result["metric_decisions"] == 8
    assert result["physical_steps"] <= 8 * 16
    assert result["actor_initial_sha256"] != result["actor_final_sha256"]
    assert result["critic_initial_sha256"] != result["critic_final_sha256"]
    assert result["provenance"]["algorithm_modified"] is False
    assert "third_party" in Path(result["provenance"]["implementation"]).parts
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["optimizers"]["actor"]["state"]
    assert checkpoint["optimizers"]["critic"]["state"]
    assert np.all(checkpoint["replay"]["done"] == 1)
    assert np.all(checkpoint["replay"]["cost"] >= 0)
    assert np.isfinite(checkpoint["replay"]["cost"]).all()
    assert len(checkpoint["diagnostics"]) == 5
    restored = load_slc_policy(_factory, result["checkpoint"])
    actual = restored.get_action(np.asarray(result["probe_observation"], dtype=np.float32), deterministic=True)
    np.testing.assert_allclose(actual, result["probe_action"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="action bounds"):
        load_slc_policy(lambda: EpisodeMetricEnv(PointMassEnv(max_steps=16), raw_bound=2.0),
                        result["checkpoint"])


def test_refuses_to_overwrite_existing_result(tmp_path):
    (tmp_path / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        train_slc(_factory, tmp_path, steps=2)
    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_serial_adapter_refuses_distributed_launch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="one process"):
        load_upstream()
