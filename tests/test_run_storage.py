"""Preserve optimizer/RNG/tensor semantics while bounding new run storage."""
import ast
from collections import OrderedDict
import collections
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from stabilization_learning.storage import (
    export_checkpoint, load_checkpoint, save_checkpoint, verify_checkpoint,
)
from stabilization_learning.storage.checkpoints import MAGIC
from stabilization_learning.storage.replay import ReplayRetention, bounded_replay_class


def equal(a, b):
    assert type(a) is type(b)
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_optimizer_next_update_rng_views_and_export_are_identical(tmp_path):
    torch.manual_seed(123)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    def step(m, opt):
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
    step(model, optimizer)
    base = torch.arange(24.).reshape(4, 6)
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        views=(base, base[:, 1::2]), rng=(torch.get_rng_state(), np.random.get_state(), random.getstate()),
        bf16=torch.arange(4, dtype=torch.bfloat16), empty=torch.empty(0), metadata=OrderedDict(a=1, b=(2, 3)))
    frozen = deepcopy(payload)
    target = tmp_path / "job/checkpoints/ready.pt"
    save_checkpoint(target, payload, store_root=tmp_path / "job/.checkpoint_objects")
    loaded = load_checkpoint(target)
    equal(frozen, loaded)
    assert loaded["views"][0].untyped_storage().data_ptr() == loaded["views"][1].untyped_storage().data_ptr()
    assert loaded["views"][1].storage_offset() == 1
    equal(payload, frozen)
    restored = torch.nn.Linear(4, 2)
    restored.load_state_dict(loaded["model"])
    restored_optimizer = torch.optim.Adam(restored.parameters())
    restored_optimizer.load_state_dict(loaded["optimizer"])
    step(model, optimizer)
    step(restored, restored_optimizer)
    equal(model.state_dict(), restored.state_dict())
    equal(optimizer.state_dict(), restored_optimizer.state_dict())
    output = tmp_path / "export.pt"
    export_checkpoint(target, output)
    equal(torch.load(output, weights_only=False), frozen)
    assert verify_checkpoint(target)["objects"] > 0
    # Moving the whole job keeps every reference valid; no source path is stored.
    shutil.move(str(tmp_path / "job"), str(tmp_path / "moved"))
    equal(load_checkpoint(tmp_path / "moved/checkpoints/ready.pt"), frozen)


def test_frozen_tensors_are_stored_once_and_no_checkpoint_is_overwritten(tmp_path):
    root = tmp_path / ".checkpoint_objects"
    fixed = torch.randn(512, 512, generator=torch.Generator().manual_seed(45))
    first = save_checkpoint(tmp_path / "a.pt", dict(a=fixed, b=fixed.clone(), count=1), store_root=root)
    second = save_checkpoint(tmp_path / "b.pt", dict(a=fixed, b=fixed.clone(), count=2), store_root=root)
    assert first["new_object_bytes"] < first["logical_bytes"] * .6
    assert second["new_object_bytes"] < 4096
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / "a.pt", {}, store_root=root)
    equal(load_checkpoint(tmp_path / "a.pt"), dict(a=fixed, b=fixed.clone(), count=1))


@pytest.mark.parametrize("damage", ["missing", "corrupt", "path"])
def test_damaged_objects_or_escaped_references_are_rejected(tmp_path, damage):
    path = tmp_path / "a.pt"
    save_checkpoint(path, dict(x=torch.ones(32)), store_root=tmp_path / ".checkpoint_objects")
    manifest = json.loads(path.read_bytes()[len(MAGIC):])
    if damage == "path":
        manifest["store"] = "../../../../elsewhere"
        path.write_bytes(MAGIC + json.dumps(manifest).encode())
    else:
        blob = next((tmp_path / ".checkpoint_objects").rglob("*.z"))
        if damage == "missing":
            blob.unlink()
        else:
            blob.write_bytes(b"corrupted")
    with pytest.raises((OSError, ValueError)):
        load_checkpoint(path)
    with pytest.raises((OSError, ValueError)):
        verify_checkpoint(path)


def test_native_checkpoints_still_load(tmp_path):
    path = tmp_path / "old.pt"
    state = {"x": torch.arange(3)}
    torch.save(state, path)
    equal(load_checkpoint(path), state)
    save_checkpoint(tmp_path / "new.pt", state)
    equal(torch.load(tmp_path / "new.pt", weights_only=False), state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not present")
def test_cuda_checkpoint_loads_on_cpu_and_cuda_without_casting(tmp_path):
    state = {"x": torch.arange(32, dtype=torch.float32, device="cuda:0") / 7}
    path = tmp_path / "cuda.pt"
    save_checkpoint(path, state, store_root=tmp_path / ".checkpoint_objects")
    assert torch.equal(load_checkpoint(path)["x"], state["x"].cpu())
    restored = load_checkpoint(path, map_location="cuda:0")["x"]
    assert restored.device == state["x"].device and torch.equal(restored, state["x"])


def test_cost_m_delta_can_reference_compact_ready(tmp_path):
    from extensions.rgb_cost_m_v2.checkpoint import load_checkpoint as load_delta, digest
    ready = tmp_path / "checkpoints/ready.pt"
    raw = tmp_path / "raw_candidates/epoch_0001.pt"
    root = tmp_path / ".checkpoint_objects"
    save_checkpoint(ready, {"learner": {"ae": torch.ones(4), "metric": torch.zeros(1)}, "count": 0}, store_root=root)
    payload = dict(schema="rgb_frozen_candidate_delta_v2", reference=dict(path="../checkpoints/ready.pt", sha256=digest(ready)),
        state={"count": 1}, learner_changes={"metric": torch.ones(1)})
    save_checkpoint(raw, payload, store_root=root)
    loaded = load_delta(raw)
    assert loaded["count"] == 1
    equal(loaded["learner"], {"ae": torch.ones(4), "metric": torch.ones(1)})


def test_retention_preserves_pinned_and_foreign_files_and_refuses_existing_runs(tmp_path):
    root = tmp_path / "replay"
    root.mkdir()
    retention = ReplayRetention(root, "active_and_restart")
    for name in ("pinned.npz", "active.npz", "old.npz", "foreign.npz"):
        (root / name).write_bytes(b"data")
    for name in ("pinned.npz", "active.npz", "old.npz"):
        retention.register(name)
    retention.pin([{"file": "pinned.npz"}], "checkpoints/ready.pt")
    cache = {"old.npz": "cached"}
    assert retention.prune([{"file": "active.npz"}], cache) == 4
    assert not cache and not (root / "old.npz").exists()
    assert all((root / name).exists() for name in ("pinned.npz", "active.npz", "foreign.npz"))
    with pytest.raises(FileExistsError):
        ReplayRetention(root, "active_and_restart")
    with pytest.raises(ValueError):
        retention.register("../outside.npz")


def test_bounded_real_replay_samples_match_archive_mode(tmp_path):
    # Load the actual replay class without initializing legacy rendering/models.
    path = Path(__file__).resolve().parents[1] / "third_party/visual_original/runtime/goal_fixed_core.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Replay")
    namespace = dict(Path=Path, json=json, np=np, collections=collections,
        json_write=lambda p, data: p.write_text(json.dumps(data), encoding="utf-8"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    replay_cls = bounded_replay_class(namespace["Replay"], "active_and_restart", 2)
    cfg = {"collection": {"replay_transitions": 4}}
    bounded = replay_cls(tmp_path / "bounded", cfg, seed=41)
    archive = namespace["Replay"](tmp_path / "archive", cfg, seed=41)
    for epoch in range(6):
        arrays = {key: np.full((2, 2), epoch, dtype=np.float32) for key in (
            "action", "cost", "terminal", "truncated", "previous_action", "smooth_center", "smooth_minus", "smooth_plus", "smooth_step")}
        arrays.update(rgb=np.full((3, 2, 2, 3), epoch, dtype=np.uint8), offset=np.full((3, 4), epoch, dtype=np.float32),
            goal_rgb=np.zeros((2, 2, 3), np.uint8), goal_depth=np.ones((2, 2), np.float32), rotation=np.eye(3))
        row = {"split": "train", "scene_id": "a", "goal_id": 0}
        for replay in (bounded, archive):
            replay.add(arrays, row, f"episode_{epoch}")
        if epoch == 0:
            bounded.retention.pin(bounded.rows, "checkpoints/ready.pt")
        equal(bounded.rows, archive.rows)
        equal(bounded.sample(8), archive.sample(8))
        equal(bounded.rng.bit_generator.state, archive.rng.bit_generator.state)
    assert len(list(bounded.root.glob("*.npz"))) == 3
    assert len(list(archive.root.glob("*.npz"))) == 6
