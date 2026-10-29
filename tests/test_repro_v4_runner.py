"""Version/route/preflight contracts for new reproducible runs."""
from copy import deepcopy
from pathlib import Path
import time

import pytest
import torch

from extensions.reproducible_training_v4 import runner
from extensions.reproducible_training_v4.runtime import content_hash, configure
from extensions.reproducible_training_v4.validation import validate_config
from extensions.reproducible_training_v4.contracts import describe


def config():
    return dict(schema="unified_training_v4", maximum_minutes=2, minimum_free_gib=0,
        jobs=[dict(id="arm", backend="episode_ppo", config=dict(task="two_link_arm", seed=9120,
            total_updates=1, batch_episodes=1, epochs=1, validation_seeds=[99120], test_seeds=[99220]),
            schedule=dict(epochs=1, certificate_updates=1))])


def test_checkpoint_digest_frames_container_structure_and_tensor_identity():
    assert content_hash({'a': {'b': 1}, 'c': 2}) != content_hash({'a': {'b': 1, 'c': 2}})
    assert content_hash([1, 2]) != content_hash((1, 2))
    assert content_hash(torch.tensor([1, 2])) != content_hash(torch.tensor([[1, 2]]))
    assert content_hash(torch.tensor([1., 2.])) != content_hash(torch.tensor([1., 3.]))
    assert content_hash({'a': 1, 'b': 2}) == content_hash({'b': 2, 'a': 1})


@pytest.mark.parametrize("change", [
    {"schema": "unified_training_v1"}, {"maximum_minutes": float("nan")}, {"deterministic": False},
    {"absolute_deadline_unix": float("inf")}, {"minimum_free_gib": -1}, {"jobs": []},
])
def test_bad_outer_protocol_rejected(change):
    cfg = config(); cfg.update(change)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("settings", [
    {"test_seeds": [9120]}, {"validation_seeds": [9120]}, {"test_seeds": [99120]},
    {"test_seeds": [True]}, {"total_updates": 2}, {"episodes": 2}, {"cpu_threads": 2},
    {"unused_parameter": 3},
])
def test_independent_schedule_and_seed_contracts_rejected_before_initialization(settings):
    cfg = config(); cfg["jobs"][0]["config"].update(settings)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("backend,settings,cls", [
    ("state_model_gradient", {}, "StableStateBackend"),
    ("observer_model_gradient", {}, "CorrectedObserverBackend"),
    ("episode_ppo", {"task": "unicycle_tracking"}, "EpisodeBackend"),
    ("dc_motor_gradient", {}, "MotorBackend"),
    ("rgb_legacy", {}, "DeterministicRGBBackend"),
    ("rgb_legacy", {"include_historical_reference": True}, "DeterministicHistoricalRGBBackend"),
])
def test_explicit_new_routes(tmp_path, backend, settings, cls):
    instance = runner.make_backend(backend, settings, tmp_path/"job", event=lambda *a, **kw: None,
                                   deadline=time.time()+30)
    try:
        assert type(instance).__name__ == cls
    finally:
        instance.close()


def test_runtime_and_rgb_capability_claims_are_explicit():
    configure()
    assert torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled()
    contract = describe(dict(backend="rgb_legacy", config={}))
    assert contract["stability_guarantee"] == "none"
    assert "outside V" in contract["P1_certificate"]


def test_runtime_lock_survives_json_roundtrip_and_rejects_dependency_change(tmp_path):
    import json
    from extensions.reproducible_training_v4.runtime import manifest, check_lock
    configure()
    lock = tmp_path/"runtime.json"
    captured = manifest()
    lock.write_text(json.dumps(captured), encoding="utf-8")
    assert check_lock(tmp_path, {"runtime_lock": "runtime.json"}) == lock
    captured["packages"].append(["injected-different-version", "0"])
    lock.write_text(json.dumps(captured), encoding="utf-8")
    with pytest.raises(RuntimeError, match="packages"):
        check_lock(tmp_path, {"runtime_lock": "runtime.json"})


def test_hidden_windows_worker_has_same_canonical_os_identity():
    import os
    import subprocess
    import sys
    from extensions.reproducible_training_v4.runtime import system_identity
    if os.name != "nt":
        pytest.skip("Windows hidden-child version regression")
    child = subprocess.run([sys.executable, "-X", "utf8", "-c",
        "from extensions.reproducible_training_v4.runtime import system_identity; print(system_identity())"],
        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL, capture_output=True,
        text=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=30)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == system_identity()


def test_missing_historical_asset_rejected_by_standalone_preflight_before_child(tmp_path, monkeypatch):
    cfg = dict(schema="unified_training_v4", maximum_minutes=1, jobs=[dict(id="rgb", backend="rgb_legacy",
        config={"include_historical_reference": True}, schedule={"epochs": 1, "certificate_updates": 1},
        test_variants=["selected", "historical_trained_reference"])])
    from stabilization_learning.unified_v1.runner import write_new
    path = tmp_path/"config.json"; write_new(path, cfg)
    monkeypatch.setattr(runner, "_run_owned_child", lambda *a, **kw: pytest.fail("child must not start"))
    with pytest.raises(FileNotFoundError):
        runner.preflight(path, cfg, tmp_path/"preflight.json", project=tmp_path, deadline=time.time()+30)
