"""CPU-only replay equivalence, invalidation, bounded-memory and guard tests."""
from dataclasses import fields
from types import SimpleNamespace
import os

import numpy as np
import pytest
import torch

from extensions.joint_certificate_v2.rgb import RawRGBReplay
from extensions.joint_certificate_v3 import replay as module
from extensions.joint_certificate_v3.replay import FastRawRGBReplay


def episode(n=4, base=0):
    rgb = np.stack([np.full((8, 8, 3), base + i, dtype=np.uint8) for i in range(n + 1)])
    return dict(rgb=rgb, goal_rgb=np.full((8, 8, 3), base + 9, dtype=np.uint8),
                rotation=np.eye(3), offset=np.arange((n + 1) * 4).reshape(n + 1, 4) / 10,
                action=np.zeros((n, 4)), actual_normalized_action=np.full((n, 4), .2),
                previous_action=np.full((n, 4), .1), dt=np.full(n, .1))


def metadata(identifier, scene="room"):
    return dict(episode_id=identifier, scene_id=scene, split="train", goal_id=0,
                policy="random_bounded", behavior_version=2, difficulty="small")


class Geometry:
    def __init__(self, batched=True):
        self.cfg = {"scale": 1.}
        self.device = torch.device("cpu")
        self.cache_revision = "fixed_depth_1"
        self.provenance = {"sha256": "weights_a", "trainable": False}
        self.observed = []
        self.image_batches = []
        self.batch_calls = 0
        self.vision = SimpleNamespace(image_tensor=self.image_tensor)
        if not batched:
            self.observe_batch = None

    def image_tensor(self, cfg, images, device):
        self.image_batches.append(len(images))
        return torch.as_tensor(np.stack(images), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.

    def encode_images(self, images):
        return self.image_tensor(self.cfg, images, self.device)

    def observe(self, public):
        assert set(public) == {"rgb", "goal_rgb", "rotation", "previous_action"}
        assert not torch.is_grad_enabled()
        self.observed.append(int(public["rgb"][0, 0, 0]))
        image = self.encode_images([public["rgb"]])
        goal = self.encode_images([public["goal_rgb"]])
        return image, goal, image.unsqueeze(1).repeat(1, 4, 1, 1, 1) * self.cfg["scale"]

    def observe_batch(self, publics):
        self.batch_calls += 1
        outputs = [self.observe(public) for public in publics]
        return tuple(torch.cat([row[index] for row in outputs]) for index in range(3))


def make_replay(tmp_path, **kwargs):
    return FastRawRGBReplay(tmp_path / "replay", np.random.default_rng(21), **kwargs)


def assert_batch_equal(left, right):
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if field.name == "observation":
            for item in fields(a):
                torch.testing.assert_close(getattr(a, item.name), getattr(b, item.name), rtol=0, atol=0)
        elif torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            assert a == b


def test_hash_once_per_stat_and_cached_tampering_is_rejected(tmp_path, monkeypatch):
    replay = make_replay(tmp_path)
    row = replay.add(episode(), metadata(0))
    original = module.file_sha256
    seen = []
    monkeypatch.setattr(module, "file_sha256", lambda path: (seen.append(path), original(path))[1])
    first = replay.load(row)
    assert replay.load(row) is first
    assert len(seen) == 1
    path = replay.root / row["file"]
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
    replay.load(row)
    assert len(seen) == 2
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        replay.load(row)
    assert len(seen) == 3 and not replay.cache


def test_hash_is_not_repeated_when_only_decoded_cache_was_evicted(tmp_path, monkeypatch):
    replay = make_replay(tmp_path, cache_episodes=1)
    rows = [replay.add(episode(base=i), metadata(i)) for i in range(2)]
    original = module.file_sha256
    seen = []
    monkeypatch.setattr(module, "file_sha256", lambda path: (seen.append(path), original(path))[1])
    for row in (rows[0], rows[1], rows[0]):
        replay.load(row)
    assert len(seen) == 2


def test_replay_payload_is_read_only_and_bytes_bound_both_caches(tmp_path):
    arrays = episode()
    size = sum(value.nbytes for value in arrays.values())
    replay = make_replay(tmp_path, replay_cache_bytes=size, geometry_cache_bytes=4608)
    rows = [replay.add(arrays, metadata(i)) for i in range(3)]
    for row in rows:
        replay.load(row)
        assert replay.cache_bytes <= size
    assert len(replay.cache) == 1
    with pytest.raises(ValueError, match="read-only"):
        replay.load(rows[-1])["rgb"][0, 0, 0, 0] = 7
    geometry = Geometry()
    replay.batch([(rows[0], 0), (rows[1], 1)], geometry)
    assert len(replay.geometry_cache) == 1
    assert replay.geometry_bytes == 4608
    for values in replay.geometry_cache.values():
        assert sum(value.untyped_storage().nbytes() for value in values) == replay.geometry_bytes


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("with_geometry", [True, False])
def test_batch_matches_legacy_including_successors_and_supervision(tmp_path, batched, with_geometry):
    replay = make_replay(tmp_path)
    legacy = RawRGBReplay(tmp_path / "legacy", np.random.default_rng(21))
    rows, old_rows = [], []
    for i in range(2):
        meta = dict(metadata(i), policy="current_controller" if i else "random_bounded", behavior_version=i)
        rows.append(replay.add(episode(base=10*i), meta))
        old_rows.append(legacy.add(episode(base=10*i), meta))
    selection = [(1, 3), (0, 1), (1, 3), (0, 0)]
    geometry, old_geometry = Geometry(batched), Geometry(batched)
    keys = [(rows[row], index) for row, index in selection]
    actual = replay.batch(keys, geometry, with_geometry=with_geometry)
    # The old wrapper does not own no_grad; its fixed geometry does.
    with torch.no_grad():
        expected = legacy.batch([(old_rows[row], index) for row, index in selection], old_geometry,
                                with_geometry=with_geometry)
    assert_batch_equal(actual, expected)
    actual.validate()
    assert actual.behavior_version == -1
    assert not actual.observation.image.requires_grad
    assert not actual.observation.image_tangents.requires_grad
    if with_geometry:
        assert len(geometry.observed) == 3
        replay.batch(keys, geometry)
        assert len(geometry.observed) == 3
        assert geometry.batch_calls == int(batched)
    else:
        assert geometry.observed == []
        assert geometry.image_batches == [4, 4, 4]


def test_revision_configuration_and_source_keys_invalidate_geometry(tmp_path):
    replay = make_replay(tmp_path)
    rows = [replay.add(episode(base=i), metadata(i)) for i in range(2)]
    geometry = Geometry()
    replay.batch([(rows[0], 0)], geometry)
    replay.batch([(rows[1], 0)], geometry)
    assert len(geometry.observed) == 2  # equal index does not alias another shard
    geometry.cache_revision = "fixed_depth_2"
    replay.batch([(rows[0], 0)], geometry)
    assert len(geometry.observed) == 3 and len(replay.geometry_cache) == 1
    geometry.cfg["scale"] = 2.
    replay.batch([(rows[0], 0)], geometry)
    assert len(geometry.observed) == 4
    replacement = Geometry()
    replay.batch([(rows[0], 0)], replacement)
    assert len(replacement.observed) == 1


def test_original_archive_sampling_distribution_and_rng_are_unchanged(tmp_path):
    replay = make_replay(tmp_path, max_episodes=3)
    legacy = RawRGBReplay(tmp_path / "legacy", np.random.default_rng(21), max_episodes=3)
    for i in range(5):
        for target in (replay, legacy):
            target.add(episode(), metadata(i, scene=f"room_{i % 2}"))
    convert = lambda pairs: [(row["episode_id"], index) for row, index in pairs]
    assert convert(replay.sample_keys(500)) == convert(legacy.sample_keys(500))
    assert len(replay.rows) == 3
    assert len(list(replay.root.glob("episode_*.npz"))) == 5
    partial = replay.add(episode(), metadata(5), complete=False)
    assert len(replay.rows) == 3 and (replay.root / partial["file"]).exists()
    with pytest.raises(FileExistsError):
        replay.add(episode(), metadata(0))
    with pytest.raises(ValueError, match="Only complete training"):
        replay.batch([(partial, 0)], Geometry())


def test_guard_stops_before_geometry_and_eval_data_never_enters_training(tmp_path):
    replay = make_replay(tmp_path)
    row = replay.add(episode(), metadata(0))
    geometry = Geometry()
    def stop():
        raise RuntimeError("stopped")
    with pytest.raises(RuntimeError, match="stopped"):
        replay.batch([(row, 0)], geometry, guard=stop)
    assert geometry.observed == []
    with pytest.raises(ValueError, match="Evaluation data"):
        replay.add(episode(), dict(metadata(1), split="test"))
    assert len(replay.rows) == 1
    with pytest.raises(ValueError, match="Only complete training"):
        replay.batch([(dict(row, split="test"), 0)], geometry)


def test_disabled_caches_still_decode_each_distinct_shard_once_per_batch(tmp_path, monkeypatch):
    replay = make_replay(tmp_path, cache_episodes=0, geometry_cache=0)
    row = replay.add(episode(), metadata(0))
    original = module.np.load
    loads = []
    monkeypatch.setattr(module.np, "load", lambda path, **kw: (loads.append(path), original(path, **kw))[1])
    replay.batch([(row, 0), (row, 2), (row, 0)], Geometry())
    assert len(loads) == 1 and not replay.cache and not replay.geometry_cache


def test_mutable_geometry_needs_revision_and_incomplete_successor_is_rejected(tmp_path):
    replay = make_replay(tmp_path)
    arrays = episode()
    arrays["rgb"] = arrays["rgb"][:-1]
    row = replay.add(arrays, metadata(0))
    with pytest.raises(ValueError, match="every actual successor"):
        replay.batch([(row, 0)], Geometry())
    geometry = Geometry()
    geometry.provenance["trainable"] = True
    geometry.cache_revision = None
    with pytest.raises(ValueError, match="cache_revision"):
        replay.batch([(row, 0)], geometry)
