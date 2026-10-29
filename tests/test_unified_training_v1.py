"""Integration safety tests use an injected learner, not a copied scientific loss."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from stabilization_learning.unified_v1.runner import (
    run_suite, validate_config, write_new, check_frozen, read)


def config():
    return dict(schema="unified_training_v1", maximum_minutes=2, minimum_free_gib=0,
        jobs=[dict(id=name, backend="state_model_gradient", config={},
                   schedule=dict(epochs=2, certificate_updates=1), test_variants=["selected", "initial"])
              for name in ("first", "second")])


class RecordingBackend:
    def __init__(self, cfg, output, event, deadline, calls):
        self.output, self.calls, self.updates = output, calls, 0

    def initialize(self):
        return {"representation": False, "critic": False}

    def collect(self, initial=False):
        return {"initial": initial}

    def warmup_representation(self, updates):
        return {"skipped": True, "updates": 0}

    def pretrain_evaluator(self, updates):
        return {"skipped": True, "updates": 0}

    def train_epoch(self, **kwargs):
        self.updates += 1
        self.calls.append((self.output.name, "train", self.updates))
        return {"actual_parameter_updates": self.updates}

    def validate(self):
        # Later updates degrade; selecting the final checkpoint is incorrect.
        return {"selection_score": [1, -abs(self.updates-1)]}

    def save(self, path):
        path.write_text(json.dumps({"updates": self.updates}))

    def load(self, path):
        self.updates = read(path)["updates"]

    def evaluate(self, split, variants):
        check_frozen(self.output.parent.parent, ["first", "second"])
        self.calls.append((self.output.name, "test", self.updates))
        assert self.updates == 1
        return {"variants": variants, "score": 0}  # Test cannot reselect a model.


def test_all_models_frozen_before_testing_and_validation_selects(tmp_path):
    calls = []
    def factory(name, cfg, out, *, event, deadline):
        return RecordingBackend(cfg, out, event, deadline, calls)
    root = tmp_path / "new_run"
    final = run_suite(config(), root, project=tmp_path, factory=factory, telemetry=False)
    assert final["status"] == "complete"
    assert calls == [("first", "train", 1), ("first", "train", 2),
                     ("second", "train", 1), ("second", "train", 2),
                     ("first", "test", 1), ("second", "test", 1)]
    assert not (root / "WRITER.lock").exists()
    assert read(root / "jobs/first/FROZEN.json")["selected_epoch"] == 1
    with pytest.raises(FileExistsError):
        run_suite(config(), root, project=tmp_path, factory=factory, telemetry=False)
    with (root / "jobs/first/checkpoints/epoch_0001.pt").open("a") as stream:
        stream.write("corruption")
    with pytest.raises(RuntimeError, match="hash"):
        check_frozen(root, ["first", "second"])


def test_failure_is_preserved_and_incomplete_group_cannot_test(tmp_path):
    calls = []
    class Fails(RecordingBackend):
        def train_epoch(self, **kwargs):
            raise RuntimeError("intentional backend failure")
    def factory(name, cfg, out, *, event, deadline):
        return Fails(cfg, out, event, deadline, calls)
    root = tmp_path / "failed_run"
    with pytest.raises(RuntimeError, match="intentional"):
        run_suite(config(), root, project=tmp_path, factory=factory, telemetry=False)
    assert read(root / "FINAL_STATE.json")["status"] == "failed"
    assert not (root / "GLOBAL_FROZEN.json").exists()
    assert not (root / "WRITER.lock").exists()
    assert not any(c[1] == "test" for c in calls)


def test_invalid_counts_and_overlapping_data_fail_before_creating_run(tmp_path):
    bad = deepcopy(config())
    bad["jobs"][0]["config"] = {"validation_seeds": [1], "test_seeds": [1]}
    with pytest.raises(ValueError, match="overlap"):
        validate_config(bad)
    bad = deepcopy(config())
    bad["jobs"][0]["schedule"]["epochs"] = -1
    with pytest.raises(ValueError):
        validate_config(bad)
    path = tmp_path / "immutable.json"
    write_new(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_new(path, {"a": 2})


def test_validation_rollback_is_saved_with_its_score(tmp_path):
    class Rollback(RecordingBackend):
        def validate(self):
            if self.updates == 2:
                self.updates = 1
            return {"selection_score": [self.updates]}
    calls = []
    def factory(name, cfg, out, *, event, deadline):
        return Rollback(cfg, out, event, deadline, calls)
    root = tmp_path / "rollback"
    run_suite(config(), root, project=tmp_path, factory=factory, telemetry=False)
    assert read(root / "jobs/first/checkpoints/epoch_0002.pt")["updates"] == 1


def test_real_state_and_observer_freeze_barrier(tmp_path):
    # Separate engineering starts, not the declared pilot holdout cases.
    cfg = dict(schema="unified_training_v1", maximum_minutes=5, minimum_free_gib=0,
        jobs=[dict(id="state_probe", backend="state_model_gradient",
                   config=dict(task="point_mass", seed=937, total_updates=1, batch_size=1,
                               hidden=8, scales=[1.], validation_seeds=[935001], test_seeds=[936001]),
                   schedule=dict(epochs=1, certificate_updates=1)),
              dict(id="observer_probe", backend="observer_model_gradient",
                   config=dict(task="motor_speed", seed=938, batch_size=1, hidden=8,
                               validation_count=1, test_count=1, training_index_start=93500,
                               validation_index_start=93600, test_index_start=93700, scenarios=["nominal"]),
                   schedule=dict(epochs=1, certificate_updates=1))])
    result = run_suite(cfg, tmp_path / "real_barrier", project=tmp_path, telemetry=False)
    assert result["status"] == "complete"
    state = read(tmp_path / "real_barrier/jobs/state_probe/TEST_RESULTS.json")["result"]
    observer = read(tmp_path / "real_barrier/jobs/observer_probe/TEST_RESULTS.json")["result"]
    assert all(value["passed"] for value in state["audit"].values())
    assert all(value["passed"] for value in observer["audit"].values())
