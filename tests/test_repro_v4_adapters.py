"""CPU engineering regressions for v4 adapters; no formal training/result claims."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from extensions.reproducible_training_v4.independent_backend import EpisodeBackend, assert_test_gate
from extensions.reproducible_training_v4.dc_motor import MotorBackend, full_successor_loss
from extensions.reproducible_training_v4.runtime import content_hash
from stabilization_learning.unified_v1 import runner as evidence


def backend(kind, output):
    if kind == "episode":
        config = dict(task="two_link_arm", physical=dict(max_steps=60, initial_scale=.01),
                      total_updates=2, batch_episodes=1, epochs=1, seed=98431,
                      validation_seeds=[98531], test_seeds=[98631])
        obj = EpisodeBackend(config, output)
    else:
        config = dict(total_updates=2, batch_size=2, hidden=8, seed=98431,
                      validation_seeds=[98531], test_seeds=[98631])
        obj = MotorBackend(config, output)
    obj.initialize()
    return obj


def actor(obj):
    return obj.learner.actor if isinstance(obj, EpisodeBackend) else obj.actor


def active_state(obj):
    if isinstance(obj, EpisodeBackend):
        return obj.learner._snapshot()
    return dict(actor=deepcopy(obj.actor.state_dict()), optimizer=deepcopy(obj.optimizer.state_dict()),
                updates=obj.updates, history=deepcopy(obj.history), initial=deepcopy(obj.initial_state),
                torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate())


@pytest.mark.parametrize("kind", ["episode", "motor"])
def test_actual_initialize_update_validate_save_load_replay(kind, tmp_path):
    records = []
    for repeat in range(2):
        obj = backend(kind, tmp_path / str(repeat))
        original = content_hash(actor(obj).state_dict())
        first = obj.train_epoch(1)
        validation = obj.validate()
        assert content_hash(actor(obj).state_dict()) != original
        checkpoint = tmp_path / str(repeat) / "checkpoint.pt"
        obj.save(checkpoint)
        second = obj.train_epoch(2)
        expected = content_hash(actor(obj).state_dict())
        obj.close()
        restored = backend(kind, tmp_path / (str(repeat) + "restored"))
        restored.load(checkpoint)
        assert restored.train_epoch(2) == second
        assert content_hash(actor(restored).state_dict()) == expected
        records.append((first, validation, second, expected))
        restored.close()
    assert records[0] == records[1]


def test_motor_loss_backpropagates_through_successor_metric(tmp_path):
    obj = backend("motor", tmp_path)
    outputs = []
    def capture(_module, _inputs, output):
        output.retain_grad()
        outputs.append(output)
    handle = obj.actor.register_forward_hook(capture)
    loss, details = full_successor_loss(obj.actor, obj.env, [99431], decay=100.)
    loss.backward()
    handle.remove()
    assert len(outputs) == 2
    assert details["full_successor_residual"][0] > 0
    assert outputs[1].grad is not None
    assert torch.isfinite(outputs[1].grad).all() and outputs[1].grad.norm() > 0
    assert sum(p.grad.abs().sum() for p in obj.actor.parameters() if p.grad is not None) > 0
    obj.close()


@pytest.mark.parametrize("kind", ["episode", "motor"])
def test_whole_epoch_budget_rejected_before_any_update(kind, tmp_path):
    obj = backend(kind, tmp_path)
    before = content_hash(active_state(obj))
    with pytest.raises(ValueError, match="whole-group"):
        obj.train_epoch(1, certificate_updates=3)
    assert content_hash(active_state(obj)) == before
    obj.close()


@pytest.mark.parametrize("kind", ["episode", "motor"])
def test_checkpoint_publication_is_atomic_and_never_overwrites(kind, tmp_path, monkeypatch):
    obj = backend(kind, tmp_path)
    final = tmp_path / "final.pt"
    obj.save(final)
    digest = evidence.sha256(final)
    with pytest.raises(FileExistsError):
        obj.save(final)
    assert evidence.sha256(final) == digest
    original_save = torch.save
    def failed_save(payload, path, *args, **kwargs):
        # For episode PPO allow its first write, then fail augmenting adapter state.
        if kind == "motor" or "v4_backend" in payload:
            with open(path, "wb") as stream:
                stream.write(b"partial serialization")
            raise OSError("injected checkpoint serialization failure")
        return original_save(payload, path, *args, **kwargs)
    monkeypatch.setattr(torch, "save", failed_save)
    pending = tmp_path / "must_not_be_published.pt"
    with pytest.raises(OSError, match="serialization failure"):
        obj.save(pending)
    assert not pending.exists()
    assert evidence.sha256(final) == digest
    obj.close()


@pytest.mark.parametrize("fault", ["adam_shape", "late_rng"])
def test_motor_failed_load_restores_everything_and_latches_stop(tmp_path, fault):
    obj = backend("motor", tmp_path)
    obj.train_epoch(1)
    healthy = tmp_path / "healthy.pt"
    obj.save(healthy)
    payload = torch.load(healthy, weights_only=False)
    # Make a partial actor mutation observable before the later exception.
    next(iter(payload["actor"].values())).add_(.1)
    if fault == "adam_shape":
        next(iter(payload["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(777)
    else:
        payload["python_rng"] = ("invalid RNG state",)
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    before = content_hash(active_state(obj))
    with pytest.raises((ValueError, TypeError)):
        obj.load(bad)
    assert content_hash(active_state(obj)) == before
    with pytest.raises(FloatingPointError, match="stopped"):
        obj.save(tmp_path / "forbidden.pt")
    assert not (tmp_path / "forbidden.pt").exists()
    obj.close()


def freeze_group(root, objects):
    evidence.write_new(root / "protocol.json", dict(config=dict(jobs=[dict(id=k) for k in objects])))
    protocol_hash = evidence.sha256(root / "protocol.json")
    jobs = {}
    for name, obj in objects.items():
        evidence.write_new(obj.output / "protocol.json", dict(suite_protocol_sha256=protocol_hash))
        obj.save(obj.output / "selected.pt")
        evidence.write_new(obj.output / "FROZEN.json", dict(checkpoint="selected.pt",
            sha256=evidence.sha256(obj.output / "selected.pt"),
            protocol_sha256=evidence.sha256(obj.output / "protocol.json")))
        jobs[name] = evidence.sha256(obj.output / "FROZEN.json")
    evidence.write_new(root / "GLOBAL_FROZEN.json", dict(protocol_sha256=protocol_hash, jobs=jobs))


def test_complete_multijob_freeze_opens_real_test_and_checks_peer_evidence(tmp_path):
    objects = {kind: backend(kind, tmp_path / "jobs" / kind) for kind in ("episode", "motor")}
    for obj in objects.values():
        obj.train_epoch(1)
        with pytest.raises((FileNotFoundError, RuntimeError)):
            obj.evaluate()
    freeze_group(tmp_path, objects)
    for obj in objects.values():
        chosen = content_hash(actor(obj).state_dict())
        result = obj.evaluate()
        assert set(result["variants"]) == {"selected", "initial", "identity"}
        assert content_hash(actor(obj).state_dict()) == chosen
        with torch.no_grad():
            next(actor(obj).parameters()).add_(.01)
        with pytest.raises(RuntimeError, match="must be loaded"):
            obj.evaluate()
        obj.load(obj.output / "selected.pt")
    (objects["motor"].output / "FROZEN.json").write_text("{}", encoding="utf8")
    with pytest.raises((RuntimeError, KeyError)):
        assert_test_gate(objects["episode"].output, content_hash(actor(objects["episode"]).state_dict()))
    for obj in objects.values():
        obj.close()


def test_motor_validation_records_complete_successor_metric(tmp_path):
    obj = backend("motor", tmp_path)
    obj.train_epoch(1)
    result = obj.validate()
    for episode in result["rows"]:
        assert not (episode["failure"] and episode["success"])
        for row in episode["trace"]:
            error, next_error = np.asarray(row["error"]), np.asarray(row["error_next"])
            before = error @ np.asarray(row["M"]) @ error
            after = next_error @ np.asarray(row["M_next"]) @ next_error
            assert row["V"] == pytest.approx(before, abs=1e-12)
            assert row["V_next"] == pytest.approx(after, abs=1e-12)
            assert row["residual"] == pytest.approx(after-before+.05*obj.env.dt*(error@error), abs=1e-12)
            assert row["commanded_action"] == row["applied_action"]
    obj.close()


@pytest.mark.parametrize("method", ["validate", "evaluate"])
def test_motor_nonfinite_evaluation_latches_stop_before_any_save(tmp_path, monkeypatch, method):
    obj = backend("motor", tmp_path)
    obj.train_epoch(1)
    original = content_hash(obj.actor.state_dict())
    physical_step = obj.env.step
    def invalid_cost(action):
        observation, _, terminated, truncated, info = physical_step(action)
        return observation, float("nan"), terminated, truncated, info
    monkeypatch.setattr(obj.env, "step", invalid_cost)
    with pytest.raises(FloatingPointError):
        if method == "validate":
            obj.validate()
        else:
            obj.evaluate(split="validation")
    assert content_hash(obj.actor.state_dict()) == original
    with pytest.raises(FloatingPointError, match="stopped"):
        obj.save(tmp_path / "invalid_selection.pt")
    assert not (tmp_path / "invalid_selection.pt").exists()
    obj.close()
