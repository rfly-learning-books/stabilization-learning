"""CPU observer failure injection and parity for all three retained tasks."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from extensions.unified_training_v1.backends import observer as legacy
from extensions.unified_training_v3 import observer


def config(task="motor_speed", **kwargs):
    return dict(task=task, seed=78301, cpu_threads=1, batch_size=1, hidden=8,
        validation_count=1, test_count=1, training_index_start=78300,
        validation_index_start=78400, test_index_start=78500,
        scenarios=["nominal"], **kwargs)


@pytest.fixture
def backend(tmp_path):
    result = observer.StableObserverBackend(config(), tmp_path / "observer")
    result.initialize()
    yield result
    result.close()


@pytest.mark.parametrize("change", [
    {"batch_size": 0}, {"batch_size": 1.5}, {"hidden": True}, {"cpu_threads": 0},
    {"gradient_clip": 0.}, {"gradient_clip": float("inf")}, {"learning_rate": float("nan")},
    {"metric_radius": 0.}, {"metric_radius": float("inf")}, {"metric_radius": 1000.},
    {"validation_count": 0}, {"test_count": 1.5}, {"training_index_start": -1},
    {"training_index_start": 99999, "batch_size": 2}, {"scenarios": []},
    {"scenarios": "nominal"}, {"scenarios": ["wrong_match"]}, {"device": "cuda"},
])
def test_invalid_configuration_rejected_without_output(tmp_path, change):
    cfg = config()
    cfg.update(change)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        observer.StableObserverBackend(cfg, output)
    assert not output.exists()


def test_same_indices_remain_disjoint_across_declared_split_namespaces():
    from stabilization_gym.observation.environment import episode_seed
    cfg = config()
    cfg.update(training_index_start=75000, validation_index_start=75000, test_index_start=75000)
    observer.validate_config(cfg)
    assert len({episode_seed(split, 75000) for split in ("train", "validation", "test")}) == 3


@pytest.mark.parametrize("task", legacy.TASKS)
def test_real_short_updates_preserve_old_results_for_every_observer(tmp_path, task):
    records = []
    for cls, label in ((legacy.Backend, "old"), (observer.StableObserverBackend, "stable")):
        backend = cls(config(task), tmp_path / label)
        try:
            initialized = backend.initialize()
            collected = backend.collect(initial=True)
            trained = backend.train_epoch(1, certificate_updates=2)
            validated = backend.validate()
            deployed = backend.evaluate("validation", ("selected",))
            assert deployed["audit"]["selected"]["passed"]
            rows = [{key: value for key, value in row.items() if key != "wall_seconds"} for row in trained["rows"]]
            records.append(dict(initialized=initialized, collected=collected, rows=rows,
                                validation=validated, deployed=deployed, parameters=legacy._digest(backend.network)))
            assert backend.updates == 2
            assert all(row["parameters_changed"] for row in rows)
        finally:
            close = getattr(backend, "close", None)
            if close:
                close()
    assert records[0] == records[1]


def test_exhausted_split_rejected_before_any_index_or_counter_mutation(backend):
    backend.config["batch_size"] = 2
    backend.train_index = 99999
    before = (backend.train_index, backend.collected_transitions, len(backend.pending))
    with pytest.raises(RuntimeError, match="exhausted"):
        backend.collect()
    assert (backend.train_index, backend.collected_transitions, len(backend.pending)) == before


def test_nonfinite_pending_truth_cannot_update_or_publish(backend):
    backend.collect()
    backend.pending[0]["truth"][0, 0] = np.nan
    before = legacy._digest(backend.network)
    with pytest.raises(FloatingPointError, match="episode data"):
        backend.train_epoch(1, certificate_updates=1)
    assert backend.updates == 0 and legacy._digest(backend.network) == before
    with pytest.raises(FloatingPointError, match="stopped after numerical failure"):
        backend.save(backend.output / "bad.pt")
    assert not (backend.output / "bad.pt").exists()


def test_wrong_split_pending_rejected(backend):
    backend.collect()
    backend.pending[0]["split"] = "test"
    with pytest.raises(ValueError, match="different data split"):
        backend.train_epoch(1, certificate_updates=1)
    assert backend.updates == 0


def test_finite_gradient_elements_with_infinite_global_norm_abort(backend, monkeypatch):
    backend.collect()
    def huge_gradient(network, *_args, **_kwargs):
        loss = sum(p.sum() for p in network.parameters()) * 1.e200
        return dict(loss=loss, cost=loss.detach().reshape(1), certificate_loss=loss.detach())
    monkeypatch.setattr(observer, "rollout", huge_gradient)
    before = legacy._digest(backend.network)
    with pytest.raises(FloatingPointError, match="total gradient norm"):
        backend.train_epoch(1, certificate_updates=1)
    assert backend.updates == 0 and legacy._digest(backend.network) == before


def test_nan_optimizer_moment_rejected_before_following_step(backend):
    backend.collect()
    backend.train_epoch(1, certificate_updates=1)
    before = legacy._digest(backend.network)
    first = next(iter(backend.optimizer.state.values()))
    first["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="optimizer state"):
        backend.train_epoch(2, certificate_updates=1)
    assert backend.updates == 1 and legacy._digest(backend.network) == before


def test_failed_optimizer_output_is_not_counted_as_success(backend):
    backend.collect()
    def inject(_optimizer, _args, _kwargs):
        with torch.no_grad():
            next(backend.network.parameters()).fill_(float("nan"))
    # Our injection is after the numerical pre-hook but before Adam; the post
    # hook must detect the corrupt output before counters and metrics advance.
    handle = backend.optimizer.register_step_pre_hook(inject)
    try:
        with pytest.raises(FloatingPointError, match="parameter"):
            backend.train_epoch(1, certificate_updates=1)
        assert backend.updates == 0 and not backend.history
    finally:
        handle.remove()


def test_validation_checks_auxiliary_metrics_not_only_loss(backend, monkeypatch):
    original = observer.rollout
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result["terminal_mse"][0] = float("nan")
        return result
    monkeypatch.setattr(observer, "rollout", corrupt)
    with pytest.raises(FloatingPointError, match="validation rollout.terminal_mse"):
        backend.validate()


def test_failed_metric_solve_latches_stopped_backend(backend, monkeypatch):
    def singular(*_args, **_kwargs):
        raise torch.linalg.LinAlgError("singular metric fault injection")
    monkeypatch.setattr(observer, "rollout", singular)
    with pytest.raises(torch.linalg.LinAlgError, match="singular metric"):
        backend.train_epoch(1, certificate_updates=1)
    with pytest.raises(FloatingPointError, match="stopped after numerical failure"):
        backend.save(backend.output / "bad_solve.pt")
    assert backend.updates == 0


def test_checkpoint_roundtrip_and_corruption_rejected_before_mutation(backend, tmp_path):
    backend.collect()
    backend.train_epoch(1, certificate_updates=1)
    path = tmp_path / "good.pt"
    backend.save(path)
    restored = observer.StableObserverBackend(config(), tmp_path / "restored")
    restored.initialize()
    try:
        restored.load(path)
        assert legacy._digest(restored.network) == legacy._digest(backend.network)
        assert restored.updates == backend.updates
        payload = torch.load(path, weights_only=False)
        payload["network"][next(iter(payload["network"]))] += .1
        corrupted = tmp_path / "corrupt.pt"
        torch.save(payload, corrupted)
        before = legacy._digest(restored.network)
        with pytest.raises(ValueError, match="digest mismatch"):
            restored.load(corrupted)
        assert legacy._digest(restored.network) == before
    finally:
        restored.close()


def test_checkpoint_preserves_existing_interrupted_temporary(backend, tmp_path):
    path = tmp_path / "checkpoint.pt"
    temporary = path.with_suffix(".pt.tmp")
    temporary.write_bytes(b"failure evidence")
    with pytest.raises(FileExistsError, match="interrupted checkpoint"):
        backend.save(path)
    assert temporary.read_bytes() == b"failure evidence" and not path.exists()
