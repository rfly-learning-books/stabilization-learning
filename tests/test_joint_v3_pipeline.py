"""CPU fixtures for the actual v3 runner, frozen inference and rollout pipeline.

These tiny synthetic plants test interfaces and archival invariants only. They
are not reconstructed-room performance or rendering benchmarks.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn

from extensions.joint_certificate_v3 import rgb, rollout
from extensions.joint_certificate_v3.replay import FastRawRGBReplay
from stabilization_learning.joint_v2.runtime import RunStore, RunStopped
from stabilization_learning.joint_v3.features import AcceleratedRGBFeatures
from stabilization_learning.joint_v3.visual import AcceleratedVisualLearner


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3 * 4 * 4, 16)
        self.decoder = nn.Linear(16, 3 * 4 * 4)

    def encode(self, image):
        return torch.tanh(self.encoder(image.flatten(1)))

    def decode(self, code):
        return torch.sigmoid(self.decoder(code)).reshape(-1, 3, 4, 4)


class FakeGeometry:
    def __init__(self, cfg=None, device="cpu", **kwargs):
        assert str(device) == "cpu"
        self.device, self.cfg = torch.device("cpu"), cfg or {}
        self.provenance = {"kind": "synthetic_cpu_fixture", "trainable": False}
        self.cache_revision = "fixture_v1"
        self.batch_sizes = []
        self.vision = SimpleNamespace(image_tensor=lambda cfg, xs, dev: self.encode_images(xs))

    def encode_images(self, images):
        return torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2) / 255.

    def observe_batch(self, publics):
        assert all(set(public) == {"rgb", "goal_rgb", "rotation", "previous_action"} for public in publics)
        self.batch_sizes.append(len(publics))
        images = self.encode_images([p["rgb"] for p in publics])
        goals = self.encode_images([p["goal_rgb"] for p in publics])
        tangent = images.new_zeros(len(publics), 4, 3, 4, 4)
        for axis in range(4):
            tangent[:, axis, :, axis, :] = .03
        return images, goals, tangent

    def observe(self, public):
        return self.observe_batch([public])


class FakeEnv:
    instances = []

    def __init__(self, cfg, scene, *, plant, max_steps):
        self.scene, self.plant, self.maximum = scene, plant, max_steps
        self.steps, self.closed, self.done = 0, False, False
        self.offset = None
        self.previous = np.zeros(4, dtype=np.float32)
        type(self).instances.append(self)

    def public(self):
        pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        rgb = pixels + int(self.scene.get("base", 20)) + 2 * self.steps
        return dict(rgb=rgb, goal_rgb=pixels + 5, rotation=np.eye(3, dtype=np.float32),
                    previous_action=self.previous.copy())

    def reset(self, *, seed, options):
        self.offset = np.asarray(options.get("offset", [.3, .2, .1, .04]), dtype=np.float64)
        return self.public(), {"supervision": {"offset": self.offset.copy(),
                                               "physical_error": float(self.offset @ self.offset)}}

    def step(self, action):
        assert not self.closed and not self.done
        if self.scene.get("fail_step") == self.steps + 1:
            raise RuntimeError("fixture physical step failed")
        before = self.offset.copy()
        self.offset = .8 * self.offset
        self.steps += 1
        self.previous = np.asarray(action).copy()
        terminated = self.steps >= self.scene["steps"]
        truncated = self.steps >= self.maximum and not terminated
        self.done = terminated or truncated
        error = float(self.offset @ self.offset)
        effort = .1 * float(np.square(action).sum())
        truth = dict(offset=before, next_offset=self.offset.copy(), physical_error=error,
                     integrated_error=.1 * error, normalized_input_squared_integral=effort,
                     physical_cost=.1 * error + .01 * effort,
                     actual_velocity=np.asarray(action) * np.array([.4, .4, .4, .35]))
        info = dict(supervision=truth, dt=.1, success=terminated,
                    reason="success" if terminated else "time_limit" if truncated else "running")
        return self.public(), -truth["physical_cost"], terminated, truncated, info

    def close(self):
        self.closed = True


@pytest.fixture
def fixture_runtime(monkeypatch):
    FakeEnv.instances = []
    monkeypatch.setattr(rollout, "ReconstructedRGBEnv", FakeEnv)
    monkeypatch.setattr(rgb, "random_raw_autoencoder", lambda cfg, device: TinyBackbone().to(device))
    monkeypatch.setattr(rgb, "RGBGeometry", FakeGeometry)
    torch.manual_seed(17)
    return rgb._new_learner({}, rgb.RGBRunConfig.smoke(device="cpu"))


def scenes(split="train"):
    return [dict(id="short", split=split, refs=[{}], steps=1, base=20),
            dict(id="long", split=split, refs=[{}], steps=3, base=35)]


def cases():
    return [dict(case_id=scene["id"] + "_case", scene_id=scene["id"], goal_id=0,
                 offset=[.3 + .1*i, .2, .1, .04], difficulty="small")
            for i, scene in enumerate(scenes())]


class EventStore:
    def __init__(self):
        self.events = []

    def event(self, kind, **data):
        self.events.append((kind, data))


def test_collect_keeps_per_env_termination_and_complete_successors(tmp_path, fixture_runtime):
    replay = FastRawRGBReplay(tmp_path / "replay", np.random.default_rng(4))
    geometry, store = FakeGeometry(), EventStore()
    result = rollout.collect_batch({}, scenes(), fixture_runtime, geometry, replay, np.random.default_rng(5),
                                   10, 5, lambda: None, exploratory=False, store=store)
    assert [row["transitions"] for row in result] == [1, 3]
    assert [row["episode_id"] for row in result] == [10, 11]
    assert geometry.batch_sizes == [2, 1, 1]
    assert all(env.closed for env in FakeEnv.instances)
    for row in result:
        data = replay.load(row)
        assert len(data["rgb"]) == len(data["offset"]) == row["transitions"] + 1
        assert len(data["action"]) == len(data["actual_normalized_action"]) == row["transitions"]
        np.testing.assert_allclose(data["actual_normalized_action"], data["action"])
        assert row["complete"] and row["policy"] == "current_controller"


def test_collect_stop_preserves_finished_and_partial_episodes_and_closes_all(tmp_path, fixture_runtime):
    replay = FastRawRGBReplay(tmp_path / "replay", np.random.default_rng(4))
    store = EventStore()
    def guard():
        if sum(env.steps for env in FakeEnv.instances) >= 2:
            raise RunStopped("fixture STOP")
    with pytest.raises(RunStopped, match="fixture STOP"):
        rollout.collect_batch({}, scenes(), fixture_runtime, FakeGeometry(), replay, np.random.default_rng(5),
                              0, 5, guard, exploratory=True, store=store)
    assert all(env.closed for env in FakeEnv.instances)
    assert [(row["episode_id"], row["transitions"]) for row in replay.rows] == [(0, 1)]
    assert (replay.root / "episode_000000.npz").exists()
    partial = replay.root / "episode_000001_partial.npz"
    assert partial.exists()
    with np.load(partial) as saved:
        assert len(saved["rgb"]) == len(saved["offset"]) == len(saved["action"]) + 1 == 2
    assert any(kind == "partial_replay_preserved" for kind, _ in store.events)


def test_evaluation_batch_size_preserves_actions_metrics_and_full_v(tmp_path, fixture_runtime):
    mapping = {scene["id"]: scene for scene in scenes("test")}
    one = rollout.evaluate_batch({}, mapping, cases(), fixture_runtime, FakeGeometry(), lambda: None,
                                 method="selected", max_steps=5, output=tmp_path / "one", batch_size=1)
    two = rollout.evaluate_batch({}, mapping, cases(), fixture_runtime, FakeGeometry(), lambda: None,
                                 method="selected", max_steps=5, output=tmp_path / "two", batch_size=2)
    assert [row["steps"] for row in one] == [1, 3]
    for a, b in zip(one, two):
        for field in ("steps", "success", "reason", "elapsed_s", "final_physical_error"):
            assert a[field] == b[field]
        for field in ("integrated_error", "input_squared_integral", "action_variation", "physical_cost"):
            assert a[field] == pytest.approx(b[field], rel=1e-5, abs=1e-7)
        with np.load(a["trajectory"]) as x, np.load(b["trajectory"]) as y:
            for field in ("error", "J", "M", "V", "action", "offset"):
                np.testing.assert_allclose(x[field], y[field], rtol=1e-5, atol=1e-7)
            n = len(y["action"])
            assert len(y["error"]) == len(y["V"]) == len(y["offset"]) == n + 1
            independent = .5 * np.einsum("ni,ij,nj->n", y["error"], y["M"], y["error"])
            np.testing.assert_allclose(y["V"], independent, rtol=2e-6, atol=1e-8)
            assert len(np.diff(y["V"])) == n
    assert all(env.closed for env in FakeEnv.instances)


def test_identity_uses_selected_features_and_same_initial_feature_coordinates(tmp_path, fixture_runtime):
    mapping = {scene["id"]: scene for scene in scenes("test")}
    versions = {key: value.clone() for key, value in fixture_runtime.features.state_dict().items()}
    selected = rollout.evaluate_batch({}, mapping, cases(), fixture_runtime, FakeGeometry(), lambda: None,
                                     method="selected", output=tmp_path / "selected", batch_size=2)
    identity = rollout.evaluate_batch({}, mapping, cases(), fixture_runtime, FakeGeometry(), lambda: None,
                                     method="identity", output=tmp_path / "identity", batch_size=2)
    for left, right in zip(selected, identity):
        with np.load(left["trajectory"]) as a, np.load(right["trajectory"]) as b:
            np.testing.assert_allclose(a["error"][0], b["error"][0], rtol=0, atol=0)
            np.testing.assert_allclose(a["J"][0], b["J"][0], rtol=0, atol=0)
            np.testing.assert_array_equal(b["M"], np.eye(16))
            np.testing.assert_allclose(b["V"], .5 * np.square(b["error"]).sum(-1), rtol=1e-6)
    for key, value in fixture_runtime.features.state_dict().items():
        torch.testing.assert_close(value, versions[key], rtol=0, atol=0)


def test_evaluation_failure_closes_envs_and_cannot_overwrite_saved_cases(tmp_path, fixture_runtime):
    mapping = {scene["id"]: scene for scene in scenes("test")}
    mapping["long"]["fail_step"] = 1
    output = tmp_path / "evaluation"
    with pytest.raises(RuntimeError, match="fixture physical step failed"):
        rollout.evaluate_batch({}, mapping, cases(), fixture_runtime, FakeGeometry(), lambda: None,
                               output=output, batch_size=1)
    saved = output / "case_0000.npz"
    before = saved.read_bytes()
    assert not (output / "case_0001.npz").exists()
    assert all(env.closed for env in FakeEnv.instances)
    with pytest.raises(FileExistsError):
        rollout.evaluate_batch({}, mapping, cases()[:1], fixture_runtime, FakeGeometry(), lambda: None,
                               output=output, batch_size=1)
    assert saved.read_bytes() == before
    assert all(env.closed for env in FakeEnv.instances)


def test_runner_new_factory_and_small_full_train_use_v3_and_freeze(tmp_path, monkeypatch, fixture_runtime):
    assert isinstance(fixture_runtime, AcceleratedVisualLearner)
    assert isinstance(fixture_runtime.features, AcceleratedRGBFeatures)
    assert fixture_runtime.execution_spec["implementation"] == "joint_rgb_v3"
    assert fixture_runtime.features.backend == "forward"
    fake_scenes = scenes()
    manifest = {"val": cases(), "test": cases(), "population_counts": {"val": 2, "test": 2}}
    monkeypatch.setattr(rgb, "load_runtime", lambda root: (None, {}, fake_scenes))
    monkeypatch.setattr(rgb, "_inputs", lambda *args: ([], []))
    monkeypatch.setattr(rgb, "_manifest", lambda *args: manifest)
    config = replace(rgb.RGBRunConfig.smoke(device="cpu"), bootstrap_steps=3,
                     validation_max_steps=3, test_max_steps=3, train_scene_limit=0,
                     inference_batch_size=2, calibration_samples=4)
    root = tmp_path / "training"
    result = rgb.train(root, config, project_root=tmp_path)
    assert result["status"] == "complete" and result["stage"] == "training_frozen"
    assert result["counts"] == {"warmup": 1, "feature": 1, "metric": 1}
    assert result["selected_update"] == 3
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["config"]["implementation"] == "joint_rgb_v3"
    assert not (root / "WRITER.lock").exists()
    marker = json.loads((root / "FROZEN.json").read_text())
    checkpoint = torch.load(root / marker["checkpoint"], map_location="cpu", weights_only=False)
    runtime = checkpoint["payload"]["runtime_state"]
    assert runtime["execution_spec"]["implementation"] == "joint_rgb_v3"
    assert runtime["replay_schema"] == "raw_rgb_archive_v3_cache"
    assert runtime["counts"]["feature"] == runtime["counts"]["metric"] == 1
    assert all(env.closed for env in FakeEnv.instances)
    with pytest.raises(FileExistsError):
        rgb.train(root, config, project_root=tmp_path)


def frozen_stub(path, *, implementation="joint_rgb_v3", seed=81, group=(81,)):
    config = dict(seed=seed, group_seeds=list(group), implementation=implementation)
    store = RunStore.create(path, config, max_hours=1, minimum_free_gib=0)
    with store:
        store.checkpoint("post.pt", {"models": {}, "optimizers": {}, "runtime_state": {}}, completed_updates=1)
        store.freeze("post.pt", selected_update=1, selection={"scope": "fixture"})
    return path


def test_group_refuses_unfrozen_or_incomplete_seed_groups(tmp_path):
    pending = tmp_path / "unfrozen"
    store = RunStore.create(pending, {"implementation": "joint_rgb_v3", "seed": 81,
                                      "group_seeds": [81]}, max_hours=1, minimum_free_gib=0)
    with pytest.raises(FileNotFoundError):
        rgb.test_group([pending], tmp_path / "forbidden_unfrozen")
    assert not (tmp_path / "forbidden_unfrozen").exists()
    one = frozen_stub(tmp_path / "seed81", group=(81, 82))
    with pytest.raises(RuntimeError, match="Entire predeclared seed group"):
        rgb.test_group([one], tmp_path / "forbidden_subset")
    assert not (tmp_path / "forbidden_subset").exists()


def test_group_refuses_old_v2_protocol_and_live_writer_before_initialization(tmp_path):
    old = frozen_stub(tmp_path / "old_v2", implementation="joint_rgb_v2")
    with pytest.raises(ValueError, match="requires v3 frozen protocols"):
        rgb.test_group([old], tmp_path / "forbidden_old")
    active = frozen_stub(tmp_path / "active")
    (active / "WRITER.lock").write_text("fixture writer")
    with pytest.raises(RuntimeError, match="writer is still active"):
        rgb.test_group([active], tmp_path / "forbidden_active")
    assert not (tmp_path / "forbidden_old").exists()
    assert not (tmp_path / "forbidden_active").exists()
