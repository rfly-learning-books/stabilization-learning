"""Engineering stability tests; short horizons are not performance evidence."""
from copy import deepcopy

import pytest
import torch

from extensions.unified_training_v1.backends.state import Backend, TASKS, _digest
from extensions.unified_training_v3.state import StableStateBackend, validate_config
from extensions.unified_training_v3 import state as stable


def _config(task="point_mass"):
    return dict(task=task, seed=73124, total_updates=2, batch_size=1, hidden=8,
        scales=[1.], validation_seeds=[9713101], test_seeds=[9713201],
        probe_batch=4, attack_steps=1, checkpoint_stride=4, physical={"max_steps":12})


@pytest.mark.parametrize("change", [
    {"total_updates":1.5}, {"batch_size":True}, {"scales":[]}, {"scales":[1.,1.]},
    {"validation_seeds":[1.5]}, {"validation_seeds":[True]},
    {"validation_seeds":[71100000]}, {"test_seeds":[71160000]},
    {"validation_seeds":[1],"test_seeds":[1]},
    {"physical":{"dt":float("nan")}}, {"physical":{"max_steps":1.5}},
    {"learning_rate":float("inf")}, {"batch_states":8}, {"train_seed_start":1},
])
def test_bad_configuration_rejected_before_initialization(change):
    config = _config(); config.update(change)
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("task", TASKS)
def test_four_tasks_healthy_update_matches_preserved_math(tmp_path, task):
    records = []
    for backend_type, name in ((Backend, "old"), (StableStateBackend, "new")):
        backend = backend_type(_config(task), tmp_path / name)
        try:
            backend.initialize()
            row = backend.learner.update()
            validation = backend.validate()
            records.append((_digest(backend.policy), row, validation["candidate_validation"],
                            deepcopy(backend.learner.replay)))
        finally:
            if hasattr(backend, "close"):
                backend.close()
    assert records[0][:3] == records[1][:3]
    torch.testing.assert_close(records[0][3], records[1][3], rtol=0, atol=0)


def _snapshot(backend):
    return (_digest(backend.policy), backend.learner.updates,
            deepcopy(backend.learner.optimizer.state_dict()), backend.learner.rng.get_state().clone(),
            backend.learner.replay.clone(), backend.learner.dual,
            deepcopy(backend.learner.history))


def _assert_atomic(before, backend):
    after = _snapshot(backend)
    assert before[0:2] == after[0:2]
    assert before[2]["param_groups"] == after[2]["param_groups"]
    assert before[2]["state"].keys() == after[2]["state"].keys()
    for key, values in before[2]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, after[2]["state"][key][name], rtol=0, atol=0)
    torch.testing.assert_close(before[3], after[3], rtol=0, atol=0)
    torch.testing.assert_close(before[4], after[4], rtol=0, atol=0)
    assert before[5:] == after[5:]
    assert not backend.learner.in_update


def _assert_stopped(backend, tmp_path):
    with pytest.raises(FloatingPointError, match="stopped"):
        backend.train_epoch(2, certificate_updates=1)
    with pytest.raises(FloatingPointError, match="stopped"):
        backend.save(tmp_path / "must_not_exist.pt")
    assert not (tmp_path / "must_not_exist.pt").exists()


def test_infinite_loss_stops_before_adam_and_restores_transaction(tmp_path, monkeypatch):
    backend = StableStateBackend(_config(), tmp_path / "run")
    backend.initialize()
    before = _snapshot(backend)
    original = stable.full_horizon_components
    def corrupt(*args, **kwargs):
        value = original(*args, **kwargs)
        value = value.clone(); value[:,0] = float("inf")
        return value
    monkeypatch.setattr(stable, "full_horizon_components", corrupt)
    with pytest.raises(FloatingPointError, match="state loss"):
        backend.train_epoch(1, certificate_updates=1)
    _assert_atomic(before, backend)
    _assert_stopped(backend, tmp_path)
    backend.close()


def test_finite_gradient_with_overflowing_norm_is_rejected(tmp_path):
    backend = StableStateBackend(_config(), tmp_path / "run")
    backend.initialize()
    before = _snapshot(backend)
    handles = [p.register_hook(lambda grad: torch.full_like(grad, 1e200)) for p in backend.policy.parameters()]
    with pytest.raises(RuntimeError, match="non-finite"):
        backend.train_epoch(1, certificate_updates=1)
    for handle in handles:
        handle.remove()
    _assert_atomic(before, backend)
    _assert_stopped(backend, tmp_path)
    backend.close()


def test_post_adam_corruption_rolls_back_parameters_optimizer_and_rng(tmp_path, monkeypatch):
    backend = StableStateBackend(_config(), tmp_path / "run")
    backend.initialize(); backend.learner.update()
    before = _snapshot(backend)
    original = backend.learner.optimizer.step
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        with torch.no_grad():
            next(backend.policy.parameters()).fill_(float("nan"))
        return result
    monkeypatch.setattr(backend.learner.optimizer, "step", corrupt)
    with pytest.raises(FloatingPointError):
        backend.train_epoch(2, certificate_updates=1)
    _assert_atomic(before, backend)
    _assert_stopped(backend, tmp_path)
    backend.close()


def test_schedule_cannot_escape_audited_seed_range(tmp_path):
    backend = StableStateBackend(_config(), tmp_path / "run")
    backend.initialize()
    before = _snapshot(backend)
    with pytest.raises(ValueError, match="seed budget"):
        backend.train_epoch(1, certificate_updates=3)
    _assert_atomic(before, backend)
    backend.close()


def test_corrupt_adam_moment_is_rejected_before_another_update(tmp_path):
    backend = StableStateBackend(_config(), tmp_path / "run")
    backend.initialize(); backend.learner.update()
    parameter = next(iter(backend.learner.optimizer.state))
    backend.learner.optimizer.state[parameter]["exp_avg_sq"].fill_(float("inf"))
    before = _digest(backend.policy)
    with pytest.raises(FloatingPointError, match="optimizer state"):
        backend.train_epoch(2, certificate_updates=1)
    assert _digest(backend.policy) == before
    assert backend.learner.updates == 1
    _assert_stopped(backend, tmp_path)
    backend.close()


def test_finite_validation_rejection_retains_original_rollback(tmp_path, monkeypatch):
    from stabilization_learning.algos.pytorch.state_metric_full_horizon import learner as original
    backend = StableStateBackend(_config(), tmp_path / "run")
    backend.initialize()
    initial_hash = _digest(backend.policy)
    backend.learner.update()
    learning_rate = backend.learner.optimizer.param_groups[0]["lr"]
    gate = original.selection_gate
    def rejected(*args, **kwargs):
        result = gate(*args, **kwargs)
        result.update(qualified=False, reasons=["synthetic finite rejection"])
        return result
    monkeypatch.setattr(original, "selection_gate", rejected)
    result = backend.validate()
    assert result["candidate_validation"]["rollback"]["to_update"] == 0
    assert _digest(backend.policy) == initial_hash
    assert backend.learner.optimizer.param_groups[0]["lr"] == learning_rate*.5
    assert backend.learner.updates == 1
    backend.save(tmp_path / "finite_rejection.pt")
    backend.close()


def test_checkpoint_mismatch_is_rejected_without_mutating_live_state(tmp_path):
    backend = StableStateBackend(_config(), tmp_path / "run")
    backend.initialize(); backend.learner.update()
    checkpoint = tmp_path / "good.pt"
    backend.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["unified_backend"]["config"]["seed"] += 1
    bad = tmp_path / "bad.pt"; torch.save(payload, bad)
    before = _snapshot(backend)
    with pytest.raises(ValueError, match="config mismatch"):
        backend.load(bad)
    _assert_atomic(before, backend)
    _assert_stopped(backend, tmp_path)
    backend.close()
