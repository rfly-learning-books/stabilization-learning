"""New-run task routing and pre-initialization schedule boundaries."""
import json
from pathlib import Path
import time

import pytest

from extensions.unified_training_v3 import runner


def config(backend, settings=None, **schedule):
    return dict(schema="unified_training_v1", maximum_minutes=1,
        jobs=[dict(id="engineering", backend=backend, config=settings or {},
                   schedule=dict(epochs=1, certificate_updates=1, **schedule))])


@pytest.mark.parametrize("backend,settings,expected", [
    ("state_model_gradient", {}, "StableStateBackend"),
    ("observer_model_gradient", {}, "StableObserverBackend"),
    ("rgb_legacy", {}, "StableRGBBackend"),
    ("rgb_legacy", {"include_historical_reference": True}, "StableHistoricalRGBBackend"),
])
def test_all_other_backends_route_to_guarded_classes_without_initializing(tmp_path, backend, settings, expected):
    instance = runner.make_backend(backend, settings, tmp_path / expected,
                                   event=lambda *args, **kwargs: None, deadline=time.time() + 60)
    try:
        assert type(instance).__name__ == expected
        assert not instance.initialized
    finally:
        instance.close()


def test_state_schedule_cannot_escape_checked_training_seed_interval():
    cfg = config("state_model_gradient", {"total_updates": 8})
    cfg["jobs"][0]["schedule"].update(epochs=3, certificate_updates=3)
    with pytest.raises(ValueError, match="total_updates"):
        runner.validate_training_config(cfg)


@pytest.mark.parametrize("updates", [1, 3])
def test_observer_budget_covers_initial_collection_and_extra_fresh_updates(updates):
    cfg = config("observer_model_gradient", {"batch_size": 2, "training_index_start": 99996})
    cfg["jobs"][0]["schedule"].update(epochs=2, certificate_updates=updates)
    with pytest.raises(ValueError, match="index namespace"):
        runner.validate_training_config(cfg)


def test_observer_exact_last_available_batch_is_allowed():
    cfg = config("observer_model_gradient", {"batch_size": 2, "training_index_start": 99996})
    assert runner.validate_training_config(cfg) is cfg


@pytest.mark.parametrize("backend", ["state_model_gradient", "observer_model_gradient"])
@pytest.mark.parametrize("stage", ["warmup_representation", "pretrain_evaluator",
                                   "representation_updates", "evaluator_updates"])
def test_unsupported_stages_rejected_before_preflight(backend, stage):
    cfg = config(backend, **{stage: 1})
    with pytest.raises(ValueError, match="no independent"):
        runner.validate_training_config(cfg)


def test_real_state_observer_group_preflight_freeze_restore_and_evaluate(tmp_path, monkeypatch):
    """Separate engineering seeds/short state horizon; no performance claim."""
    project = Path(__file__).resolve().parents[1]
    cfg = dict(schema="unified_training_v1", maximum_minutes=3, minimum_free_gib=0,
        absolute_deadline_unix=time.time() + 600, historical_reference={"source_paths": []},
        jobs=[
            dict(id="state_engineering", backend="state_model_gradient",
                config=dict(task="point_mass", seed=79231, total_updates=1, batch_size=1,
                    hidden=8, scales=[1.], validation_seeds=[9723101], test_seeds=[9723201],
                    probe_batch=4, attack_steps=1, checkpoint_stride=4, physical={"max_steps": 12}),
                schedule=dict(epochs=1, certificate_updates=1),
                test_variants=["selected", "initial", "identity"]),
            dict(id="observer_engineering", backend="observer_model_gradient",
                config=dict(task="motor_speed", seed=79232, batch_size=1, hidden=8,
                    validation_count=1, test_count=1, training_index_start=79300,
                    validation_index_start=79400, test_index_start=79500, scenarios=["nominal"]),
                schedule=dict(epochs=1, certificate_updates=1),
                test_variants=["selected", "initial", "identity"]),
        ])
    path = tmp_path / "engineering_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    # The production group still requires its two RGB arms; this test has no
    # GPU access and isolates the CPU adapters under the same lifecycle.
    monkeypatch.setattr(runner.v2, "validate_extra", lambda *_: None)
    monkeypatch.setattr(runner.v2, "visual_source_inventory", lambda *_: {})
    monkeypatch.setattr(runner.v1, "source_inventory", lambda *_: {})
    root = tmp_path / "cpu_engineering_group"
    final = runner.run_suite(cfg, root, project=project, config_path=path, telemetry=False)
    assert final["status"] == "complete"
    ids = [job["id"] for job in cfg["jobs"]]
    runner.v1.check_frozen(root, ids)
    events = [runner.v1.read(path) for path in sorted((root / "events").glob("*.json"))]
    barrier = next(e["sequence"] for e in events if e["kind"] == "all_models_frozen")
    assert all(e["sequence"] > barrier for e in events if e["kind"] == "test_started")
    assert sum(e["kind"] == "test_completed" for e in events) == 2
    for jid in ids:
        report = runner.v1.read(root / "jobs" / jid / "TEST_RESULTS.json")
        assert report["started_after_group_freeze"]
    assert not (root / "WRITER.lock").exists()
