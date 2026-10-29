"""Lifecycle tests for book cadence, stage order and frozen testing."""
from copy import deepcopy
from pathlib import Path

import pytest

from extensions.rgb_book_reproduction_v1 import runner


def configuration():
    return dict(schema="unified_training_v1", maximum_minutes=2, minimum_free_gib=0,
        book_protocol=dict(mode="engineering", validation_every=2), jobs=[dict(id="rgb_book", backend="rgb_legacy",
        schedule=dict(warmup_representation=3, pretrain_evaluator=4, epochs=4,
            representation_updates=5, evaluator_updates=0, certificate_updates=6),
        config=dict(epochs=4, cross_epochs=None), test_variants=["selected", "identity"])])


class FakeBackend:
    def __init__(self, config, output, **kwargs):
        self.output, self.epoch, self.calls = Path(output), 0, []
        self.validations = 0

    def initialize(self):
        self.calls.append("initialize")
        return dict(initialized=True)

    def collect(self, initial=False):
        self.calls.append("bootstrap" if initial else "collect")
        return dict(initial=initial)

    def warmup_representation(self, updates):
        self.calls.append(("warmup", updates))
        return dict(updates=updates)

    def pretrain_evaluator(self, updates):
        self.calls.append(("prefit", updates))
        return dict(updates=updates)

    def train_epoch(self, epoch, **kwargs):
        self.epoch = epoch
        self.calls.append(("train", epoch, kwargs))
        return dict(epoch=epoch)

    def validate(self):
        self.calls.append(("validate", self.epoch))
        self.validations += 1
        # Initial and ready intentionally outperform all trained candidates.
        score = [99., 99., 0.] if not self.epoch else [float(6-self.epoch), 0., 0.]
        return dict(selection_score=score, summary={"success_rate": score[0]}, rows=[{"never_duplicate": True}])

    def save(self, path):
        self.calls.append(("save", Path(path).stem))
        Path(path).write_text(str(self.epoch))

    def load(self, path):
        self.calls.append(("load", Path(path).stem))
        self.epoch = int(Path(path).read_text())

    def evaluate(self, split, variants):
        runner.v1.check_frozen(self.output.parent.parent, [self.output.name])
        self.calls.append(("test", self.epoch))
        return dict(complete=True, split=split, summary={}, rows=[])

    def close(self):
        self.calls.append("close")


def launch(tmp_path, backend_class=FakeBackend):
    cfg = configuration()
    config_path = tmp_path / "config.json"
    runner.v1.write_new(config_path, cfg)
    box = {}

    def factory(name, config, output, **kwargs):
        box["backend"] = backend_class(config, output, **kwargs)
        return box["backend"]

    result = runner.run_suite(cfg, tmp_path / "run", project=tmp_path, config_path=config_path,
        telemetry=False, factory=factory,
        inventory_function=lambda project, config, path: {str(path): runner.v1.sha256(path)})
    return cfg, box["backend"], result


def test_book_epoch_order_cadence_and_freeze(tmp_path):
    _, backend, result = launch(tmp_path)
    assert result["status"] == "complete"
    calls = backend.calls
    assert [v for v in calls if isinstance(v, tuple) and v[0] == "validate"] == [
        ("validate", 0), ("validate", 0), ("validate", 2), ("validate", 4)]
    assert calls.index("bootstrap") < calls.index(("warmup", 3)) < calls.index(("prefit", 4))
    for epoch in range(1, 5):
        position = next(i for i, value in enumerate(calls) if isinstance(value, tuple) and value[:2] == ("train", epoch))
        assert calls[position-1] == "collect"
        assert calls[position][2] == dict(representation_updates=5, evaluator_updates=0, certificate_updates=6)
    frozen = runner.v1.read(tmp_path / "run/jobs/rgb_book/FROZEN.json")
    assert frozen["selected_epoch"] == 2
    assert ("test", 2) in calls
    assert calls[-1] == "close"
    assert not (tmp_path / "run/WRITER.lock").exists()
    assert len(list((tmp_path / "run/jobs/rgb_book/checkpoints").glob("*.COMMITTED.json"))) == 6
    assert "rows" not in runner.v1.read(tmp_path / "run/jobs/rgb_book/stages/validate_epoch_0002.json")["result"]


def test_failure_preserves_partial_state_and_never_tests(tmp_path):
    class Failing(FakeBackend):
        def train_epoch(self, epoch, **kwargs):
            if epoch == 3:
                raise InterruptedError("requested stop")
            return super().train_epoch(epoch, **kwargs)
    with pytest.raises(InterruptedError, match="requested stop"):
        launch(tmp_path, Failing)
    final = runner.v1.read(tmp_path / "run/FINAL_STATE.json")
    assert final["status"] == "stopped"
    assert final["jobs"]["rgb_book"]["completed_epochs"] == 2
    assert not (tmp_path / "run/GLOBAL_FROZEN.json").exists()
    assert not (tmp_path / "run/WRITER.lock").exists()


@pytest.mark.parametrize("key,value", [("select_initial", True), ("select_ready", True),
    ("validation_every", 0), ("validation_every", 3), ("mode", "unknown")])
def test_rejects_protocol_mutations(key, value):
    cfg = configuration()
    cfg["book_protocol"][key] = value
    with pytest.raises(ValueError):
        runner.validate(cfg)


def test_rejects_preselected_cross_checkpoint():
    cfg = configuration()
    cfg["jobs"][0]["config"]["cross_epochs"] = [20, 13]
    with pytest.raises(ValueError, match="cross-checkpoint"):
        runner.validate(cfg)


def test_formal_budget_is_exact(tmp_path):
    cfg = configuration()
    cfg["book_protocol"]["mode"] = "formal"
    cfg["minimum_free_gib"] = 20
    cfg["jobs"][0]["schedule"] = deepcopy(runner.BOOK_SCHEDULE)
    starts = tmp_path / "starts.json"
    runner.v1.write_new(starts, dict(val=list(range(48)), test=list(range(360))))
    methods = ["selected", "identity", "shadow", "initial_identity"]
    cfg["jobs"][0]["test_variants"] = methods
    cfg["jobs"][0]["config"].update(epochs=24, seed=41, batch_size=4, collection_steps=120,
        validation_plants=["P0"], validation_steps=300, trajectories_per_scene=2, bootstrap_transitions=2822,
        book_cases=str(starts), generalization_suites=[
            dict(name="book_full_val", cases="val", variants=methods, plants=["P0"], steps=300),
            dict(name="book_test", cases="test", variants=methods, plants=["P0", "P1"], steps=300),
            dict(name="depth_0_5", cases="test", variants=["selected", "identity"], plants=["P0"],
                 steps=300, case_limit=72, depth_scale=.5),
            dict(name="depth_2", cases="test", variants=["selected", "identity"], plants=["P0"],
                 steps=300, case_limit=72, depth_scale=2.)])
    assert runner.validate(cfg) is cfg
    cfg["jobs"][0]["schedule"]["certificate_updates"] = 201
    with pytest.raises(ValueError, match="1000/1000"):
        runner.validate(cfg)


def test_report_independently_checks_complete_successor_and_action(tmp_path):
    import numpy as np
    from extensions.rgb_book_reproduction_v1.reporting import trajectory_audit
    path = tmp_path / "trace.npz"
    arrays = dict(error=np.zeros((12, 16)), M=np.eye(16), J=np.eye(16, 4),
        V=np.zeros(12), offset=np.zeros((12, 4)), action=np.zeros((11, 4)),
        dt=np.full(11, .1), integrated_error=np.zeros(11), hold=np.arange(1, 12),
        terminated=np.array([False]*10+[True]), truncated=np.zeros(11, bool))
    np.savez_compressed(path, **arrays)
    result = trajectory_audit(path, runner.v1.sha256(path), dict(success=True, audit=dict(passed=True)))
    assert result["endpoints"] == 12
    arrays["V"][-1] = .1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Complete endpoint V"):
        trajectory_audit(path, runner.v1.sha256(path), dict(success=True, audit=dict(passed=True)))
