import json
import random
import time
import numpy as np
import pytest
import torch
from torch import nn

from stabilization_learning.joint_v2.runtime import RunStore, RunStopped, capture_rng, restore_rng
from stabilization_learning.joint_v2.certificate import NeuralCertificate, make_binding


def test_exclusive_owner_stop_and_immutable_events(tmp_path):
    run = RunStore.create(tmp_path / "run", {}, max_hours=1, minimum_free_gib=0)
    with run:
        run.event("begin", update=0)
        with pytest.raises(FileExistsError):
            RunStore(run.root).acquire()
        (run.root / "STOP").touch()
        with pytest.raises(RunStopped):
            run.guard()
        run.event("stopped", reason="STOP")
    events = sorted((run.root / "events").glob("*.json"))
    assert len(events) == 2
    assert json.loads(events[0].read_text())["data"]["update"] == 0
    assert not (run.root / "WRITER.lock").exists()


def test_checkpoint_boundary_rng_and_freeze(tmp_path):
    run = RunStore.create(tmp_path / "run", {}, max_hours=1, minimum_free_gib=0)
    with run:
        state = dict(models={"x": torch.tensor([3.])}, optimizers={}, runtime_state={"next": 3})
        run.checkpoint("update_2.pt", state, completed_updates=2)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        ck = run.load_checkpoint("update_2.pt")
        restore_rng(ck["rng"])
        actual = (random.random(), np.random.rand(), torch.rand(2))
        assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
        with pytest.raises(FileExistsError):
            run.checkpoint("update_2.pt", state, completed_updates=2)
        run.freeze("update_2.pt", selected_update=2, selection="predeclared_validation")
        assert len(RunStore.require_frozen_group([run.root])) == 1
        with pytest.raises(FileNotFoundError):
            RunStore.require_frozen_group([run.root, tmp_path / "missing_seed"])


def test_frozen_source_and_checkpoint_corruption_rejected(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    run = RunStore.create(tmp_path / "run", {}, max_hours=1, sources=[source], minimum_free_gib=0)
    with run:
        run.checkpoint("u1.pt", dict(models={}, optimizers={}, runtime_state={}), completed_updates=1)
        with (run.root / "u1.pt").open("ab") as f:
            f.write(b"corrupt")
        with pytest.raises(RuntimeError, match="hash"):
            run.load_checkpoint("u1.pt")
        source.write_text("changed")
        with pytest.raises(RuntimeError, match="input changed"):
            run.verify_inputs()


def test_expired_deadline_not_extended_by_open(tmp_path):
    run = RunStore.create(tmp_path / "run", {}, max_hours=1e-9, minimum_free_gib=0)
    time.sleep(.001)
    with pytest.raises(RunStopped):
        RunStore(run.root).acquire()
    assert not (run.root / "WRITER.lock").exists()


class StateMetric(nn.Module):
    def forward(self, error, context):
        return torch.diag_embed(1 + error.square())


def test_certificate_complete_state_derivative_and_version_guard():
    binding = make_binding(task="test", revision="v1", action_units="SI")
    cert = NeuralCertificate(StateMetric(), binding, factor=.5)
    e = torch.tensor([[.2, .3]], dtype=torch.float64)
    grad = cert.gradient(e).full_error_gradient
    assert torch.allclose(grad, e + 2*e**3)
    en = e + .1
    expected = (.5 * (en.square()*(1+en.square()) - e.square()*(1+e.square()))).sum(-1)
    assert torch.allclose(cert.difference(e, en), expected)
    with pytest.raises(ValueError, match="Stale"):
        cert.evaluate(e, binding=make_binding(task="test", revision="v2", action_units="SI"))


def test_observer_online_matrix_does_not_require_true_error():
    cert = NeuralCertificate(StateMetric(), make_binding(task="observer", revision="1", action_units="internal"),
                             online_matrix=lambda estimate: torch.diag_embed(1+estimate.square()))
    assert torch.equal(cert.online_matrix(torch.zeros(2)), torch.eye(2))
    other = NeuralCertificate(StateMetric(), cert.binding)
    with pytest.raises(TypeError, match="online_matrix"):
        other.online_matrix(torch.zeros(2))
