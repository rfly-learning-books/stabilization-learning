"""CPU fault injection at the real v3 lifecycle's publication boundaries."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from extensions.unified_training_v3 import lifecycle, runner


def config():
    return dict(schema="unified_training_v1", maximum_minutes=1, minimum_free_gib=0,
        absolute_deadline_unix=2000., initializer_preflight={"cutoff_unix": 1040.},
        jobs=[dict(id="probe", backend="push_discrete_sac", config={},
                   schedule=dict(epochs=1, certificate_updates=1),
                   test_variants=["selected", "initial", "identity"])])


class Backend:
    def __init__(self, clock, root, fault):
        self.clock, self.root, self.fault = clock, root, fault
        self.closed = False

    def initialize(self):
        if self.fault == "late_initialize":
            self.clock[0] = 1100.
        if self.fault == "stop_initialize":
            (self.root / "STOP").touch()
        return {}

    def validate(self):
        return {"selection_score": [1.]}

    def save(self, path):
        path.write_bytes(b"real test payload")
        if self.fault == "late_save":
            self.clock[0] = 1100.

    def load(self, path):
        assert path.is_file()

    def collect(self, initial=False): return {}
    def warmup_representation(self, updates): return {}
    def pretrain_evaluator(self, updates): return {}
    def train_epoch(self, **kwargs): return {}

    def evaluate(self, **kwargs):
        if self.fault == "late_test":
            self.clock[0] = 1100.
        if self.fault == "stop_test":
            (self.root / "jobs/probe/STOP").touch()
        return {"test": "completed by fake backend"}

    def close(self):
        self.closed = True
        if self.fault == "bad_close":
            raise RuntimeError("injected cleanup failure")


def run(tmp_path, monkeypatch, fault=None):
    clock = [1000.]
    monkeypatch.setattr(lifecycle.time, "time", lambda: clock[0])
    monkeypatch.setattr(lifecycle.v1, "preserve_rng", nullcontext)
    root = tmp_path / "fresh"
    backend = Backend(clock, root, fault)
    def inventory(_):
        if fault == "slow_inventory":
            clock[0] = 1039.
        return {}
    def invoke():
        return lifecycle.run_suite(config(), root, project=tmp_path,
            factory=lambda *args, **kwargs: backend, inventory=inventory, telemetry=False)
    return root, backend, invoke


@pytest.mark.parametrize("fault", ["late_initialize", "stop_initialize", "late_test", "stop_test", "late_save"])
def test_late_or_stopped_stage_cannot_publish_success(tmp_path, monkeypatch, fault):
    root, backend, invoke = run(tmp_path, monkeypatch, fault)
    with pytest.raises(InterruptedError):
        invoke()
    assert backend.closed
    final = lifecycle.v1.read(root / "FINAL_STATE.json")
    assert final["status"] == "stopped"
    assert not (root / "WRITER.lock").exists()
    assert not (root / "RESULTS.json").exists()
    assert not (root / "jobs/probe/TEST_RESULTS.json").exists()
    assert not list((root / "events").glob("*suite_completed*"))
    if fault.endswith("initialize"):
        assert not (root / "jobs/probe/stages/initialize.json").exists()
    if fault == "late_save":
        assert not (root / "jobs/probe/checkpoints/initial.pt").exists()
        assert list((root / "jobs/probe/checkpoints").glob("*.uncommitted"))


def test_original_cutoff_survives_slow_source_inventory(tmp_path, monkeypatch):
    root, backend, invoke = run(tmp_path, monkeypatch, "slow_inventory")
    assert invoke()["status"] == "complete"
    protocol = lifecycle.v1.read(root / "protocol.json")
    assert protocol["deadline_unix"] == 1040.
    assert protocol["started_unix"] == 1039.
    assert lifecycle.v1.check_frozen(root, ["probe"])
    assert backend.closed


def test_failed_cleanup_cannot_be_called_complete(tmp_path, monkeypatch):
    root, backend, invoke = run(tmp_path, monkeypatch, "bad_close")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        invoke()
    final = lifecycle.v1.read(root / "FINAL_STATE.json")
    assert final["status"] == "failed" and final["cleanup_errors"]
    assert not list((root / "events").glob("*suite_completed*"))
    assert not (root / "RESULTS.json").exists()
    assert not (root / "WRITER.lock").exists()


def test_partial_checkpoint_never_gets_published(tmp_path):
    def broken(path):
        path.write_bytes(b"partial")
        raise OSError("disk full")
    backend = SimpleNamespace(save=broken)
    with pytest.raises(OSError, match="disk full"):
        lifecycle.commit_checkpoint(backend, tmp_path, "initial", {}, lambda: None)
    assert not (tmp_path / "checkpoints/initial.pt").exists()
    assert not (tmp_path / "checkpoints/initial.COMMITTED.json").exists()
    assert list((tmp_path / "checkpoints").glob("*.uncommitted/initial.pt"))


def test_checkpoint_publication_never_replaces_racing_writer(tmp_path):
    destination = tmp_path / "checkpoints/initial.pt"
    def racing_save(path):
        path.write_bytes(b"candidate")
        destination.write_bytes(b"existing evidence")
    with pytest.raises(FileExistsError):
        lifecycle.commit_checkpoint(SimpleNamespace(save=racing_save), tmp_path, "initial", {}, lambda: None)
    assert destination.read_bytes() == b"existing evidence"
    assert not destination.with_suffix(".COMMITTED.json").exists()


def test_preflight_cancellation_cleans_up_owned_child(monkeypatch, tmp_path):
    child = SimpleNamespace(communicate=lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    cleaned = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(runner, "_finish_timed_out_child", cleaned.append)
    with pytest.raises(KeyboardInterrupt):
        runner._run_owned_child(["unused"], cwd=tmp_path, timeout=1)
    assert cleaned == [child]


@pytest.mark.parametrize("failure", ["final_state", "terminal_event"])
def test_incomplete_terminal_evidence_preserves_reporting_lock(tmp_path, monkeypatch, failure):
    root, backend, invoke = run(tmp_path, monkeypatch)
    original = lifecycle.v1.write_new
    def broken_write(path, value):
        if (failure == "final_state" and Path(path).name == "FINAL_STATE.json") or (
                failure == "terminal_event" and "suite_completed" in Path(path).name):
            raise OSError("injected evidence write failure")
        return original(path, value)
    monkeypatch.setattr(lifecycle.v1, "write_new", broken_write)
    with pytest.raises(OSError, match="evidence write failure"):
        invoke()
    assert backend.closed
    assert (root / "WRITER.lock").is_file()
    assert not list((root / "events").glob("*suite_completed*"))


@pytest.mark.parametrize("backend", ["push_discrete_sac", "rgb_legacy", "state_model_gradient", "observer_model_gradient"])
def test_unknown_test_variant_rejected_before_initializer(backend):
    cfg = config()
    cfg["jobs"][0].update(backend=backend, test_variants=["selected", "typo"])
    with pytest.raises(ValueError, match="Unsupported test variant"):
        runner.validate_training_config(cfg)


@pytest.mark.parametrize("reserve", [-1, float("nan"), float("inf"), True])
def test_invalid_disk_reserve_rejected_before_run(reserve):
    cfg = config()
    cfg["minimum_free_gib"] = reserve
    with pytest.raises(ValueError):
        runner.validate_training_config(cfg)


@pytest.mark.parametrize("stage", ["warmup_representation", "representation_updates"])
def test_unsupported_push_stage_is_rejected_before_collection(stage):
    cfg = config()
    cfg["jobs"][0]["schedule"][stage] = 1
    with pytest.raises(ValueError, match="independent representation"):
        runner.validate_training_config(cfg)


@pytest.mark.parametrize("key", ["replay_cached_episodes", "batch_size", "probe_batch_size",
    "evaluation_batch", "test_cases", "validation_cases", "collection_steps", "parallel_envs"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_rgb_invalid_counts_never_start_an_initializer(tmp_path, monkeypatch, key, value):
    cfg = config()
    cfg["jobs"][0].update(backend="rgb_legacy", config={key: value})
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid RGB config reached child initializer")
    monkeypatch.setattr(runner, "_run_owned_child", unexpected)
    with pytest.raises(ValueError, match="RGB requires a positive integer"):
        runner.preflight(tmp_path / "unused.json", cfg, tmp_path / "preflight.json",
                         project=tmp_path, deadline=2000.)
    assert not (tmp_path / "preflight.json").exists()


@pytest.mark.parametrize("plants", [[], ["P0", "P0"], ["P2"], "P0"])
def test_invalid_rgb_plants_rejected_before_run(plants):
    cfg = config()
    cfg["jobs"][0].update(backend="rgb_legacy", config={"evaluation_plants": plants})
    with pytest.raises(ValueError, match="evaluation_plants"):
        runner.validate_training_config(cfg)


def test_rgb_cpu_device_rejected_before_expensive_initialization():
    cfg = config()
    cfg["jobs"][0].update(backend="rgb_legacy", config={"device": "cpu"})
    with pytest.raises(ValueError, match="CUDA device"):
        runner.validate_training_config(cfg)


def test_valid_rgb_configuration_can_reach_preflight():
    cfg = config()
    cfg["jobs"][0].update(backend="rgb_legacy", config={"device": "cuda:0",
        "batch_size": 4, "probe_batch_size": 4, "replay_cached_episodes": 96,
        "evaluation_plants": ["P0", "P1"], "evaluation_batch": 4, "test_cases": 16})
    assert runner.validate_training_config(cfg) is cfg
