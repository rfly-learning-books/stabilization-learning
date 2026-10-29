"""Protocol/replay/guard tests; the separate recorded smoke uses real rooms."""
from dataclasses import replace
import json

import numpy as np
import pytest

from extensions.joint_certificate_v2 import rgb
from stabilization_learning.joint_v2.runtime import RunStore, RunStopped


def _episode(n=3):
    return dict(rgb=np.zeros((n + 1, 8, 8, 3), dtype=np.uint8),
                goal_rgb=np.zeros((8, 8, 3), dtype=np.uint8), rotation=np.eye(3),
                offset=np.zeros((n + 1, 4)), action=np.zeros((n, 4)),
                actual_normalized_action=np.zeros((n, 4)), previous_action=np.zeros((n, 4)),
                dt=np.full(n, 0.1))


def _metadata(identifier, scene="training_room"):
    return dict(episode_id=identifier, scene_id=scene, split="train", policy="random_bounded",
                behavior_version=0, goal_id=0, difficulty="small", reason="time_limit")


def test_fresh_replay_is_immutable_and_partial_episode_is_not_train_data(tmp_path):
    replay = rgb.RawRGBReplay(tmp_path / "replay", np.random.default_rng(10))
    complete = replay.add(_episode(), _metadata(0))
    partial = replay.add(_episode(1), _metadata(1), complete=False)
    assert len(replay.rows) == 1 and replay.rows[0]["transitions"] == 3
    assert (replay.root / partial["file"]).is_file()
    assert len(replay.sample_keys(7)) == 7
    with pytest.raises(FileExistsError):
        replay.add(_episode(), _metadata(0))
    with pytest.raises(FileExistsError):
        rgb.RawRGBReplay(replay.root, np.random.default_rng(11))
    assert replay.load(complete)["rgb"].shape[0] == 4


def test_replay_hash_and_evaluation_split_are_enforced(tmp_path):
    replay = rgb.RawRGBReplay(tmp_path / "replay", np.random.default_rng(10))
    row = replay.add(_episode(), _metadata(0))
    path = replay.root / row["file"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        replay.load(row)
    invalid = dict(_metadata(1), split="test")
    with pytest.raises(ValueError, match="Evaluation data"):
        replay.add(_episode(), invalid)
    assert len(replay.rows) == 1


def test_replay_sampling_preserves_declared_scene_balance_and_limits_active_rows(tmp_path):
    replay = rgb.RawRGBReplay(tmp_path / "replay", np.random.default_rng(10), max_episodes=2)
    for i in range(3):
        replay.add(_episode(), _metadata(i, scene=f"room_{i}"))
    assert [row["episode_id"] for row in replay.rows] == [1, 2]
    assert (replay.root / "episode_000000.npz").is_file()  # evict active data, preserve evidence
    seen = {row["scene_id"] for row, _ in replay.sample_keys(30)}
    assert seen == {"room_1", "room_2"}


def test_external_stop_records_final_state_before_initializing_models(tmp_path, monkeypatch):
    monkeypatch.setattr(rgb, "load_runtime", lambda project: (None, {}, []))
    monkeypatch.setattr(rgb, "_manifest", lambda *args: {"val": [], "test": []})
    monkeypatch.setattr(rgb, "_inputs", lambda *args: ([], []))
    monkeypatch.setattr(rgb, "_new_learner", lambda *args: pytest.fail("STOP must precede initialization"))
    output = tmp_path / "stopped"
    with pytest.raises(RunStopped):
        rgb.train(output, rgb.RGBRunConfig.smoke(device="cpu"), guard=lambda: False)
    final = json.loads((output / "FINAL_STATE.json").read_text(encoding="utf-8"))
    assert final["status"] == "stopped" and final["auto_resume_supported"] is False
    assert not (output / "WRITER.lock").exists()
    assert not (output / "initial.pt").exists()


def _frozen_fixture(path, seed, group):
    config = dict(seed=seed, group_seeds=group, evaluation_manifest={"val": [], "test": []})
    store = RunStore.create(path, config, max_hours=1, minimum_free_gib=0)
    with store:
        store.checkpoint("post.pt", {"models": {}, "optimizers": {}, "runtime_state": {}}, completed_updates=1)
        store.freeze("post.pt", selected_update=1, selection={"scope": "unit_fixture"})
    return path


def test_group_testing_rejects_a_subset_of_predeclared_seed_group(tmp_path):
    root = _frozen_fixture(tmp_path / "seed81", 81, [81, 82, 83])
    with pytest.raises(RuntimeError, match="Entire predeclared seed group"):
        rgb.test_group([root], tmp_path / "forbidden_test")
    assert not (tmp_path / "forbidden_test").exists()


def test_group_testing_rejects_still_active_writer(tmp_path):
    root = _frozen_fixture(tmp_path / "seed81", 81, [81])
    (root / "WRITER.lock").write_text("active", encoding="utf-8")
    with pytest.raises(RuntimeError, match="writer is still active"):
        rgb.test_group([root], tmp_path / "forbidden_test")


def test_formal_budget_counts_and_smoke_scope_are_explicit():
    formal = rgb.RGBRunConfig()
    assert formal.warmup_updates + 2 * formal.joint_pairs == 14000
    assert formal.group_seeds == (81, 82, 83)
    smoke = rgb.RGBRunConfig.smoke(seed=82, device="cpu")
    assert smoke.group_seeds == (82,) and smoke.scope == "engineering_smoke"
    assert smoke.warmup_updates + 2 * smoke.joint_pairs == 3
    with pytest.raises(ValueError):
        replace(formal, warmup_updates=0)
    assert rgb.RGBRunConfig.from_dict({}).learner == formal.learner
    assert rgb.RGBRunConfig.from_dict({"learner": {"metric_lr": .001}}).learner.metric_lr == .001


def test_declared_subset_balances_rooms_and_difficulties_before_any_learning():
    rows = [dict(scene_id=f"room{scene}", goal_id=goal, difficulty=difficulty,
                 case_id=f"{scene}/{goal}/{difficulty}")
            for scene in range(8) for goal in range(3) for difficulty in ("small", "medium", "large")]
    selected = rgb._balanced_cases(rows, 8, 90)
    assert len({row["scene_id"] for row in selected}) == 8
    difficulty_counts = [sum(row["difficulty"] == diff for row in selected) for diff in ("small", "medium", "large")]
    assert max(difficulty_counts) - min(difficulty_counts) <= 1
    assert selected == rgb._balanced_cases(rows, 8, 90)
    assert len({row["case_id"] for row in selected}) == 8
