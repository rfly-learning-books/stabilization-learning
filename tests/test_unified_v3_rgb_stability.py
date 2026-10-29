"""CPU-only RGB guard wiring tests; no rendering or GPU training claims."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from extensions.unified_training_v3.rgb import (
    _StableRGB, StableRGBBackend, StableHistoricalRGBBackend,
)
from extensions.unified_training_v1.backends.rgb import RGBBackend
from extensions.unified_training_v2.historical_rgb import HistoricalRGBBackend


class TinyLearner:
    def __init__(self, fault=None):
        torch.manual_seed(82741)
        self.ae = torch.nn.Linear(2, 1, bias=False)
        self.shadow = deepcopy(self.ae)
        self.initial = deepcopy(self.ae).requires_grad_(False)
        self.metric = torch.nn.Linear(2, 1, bias=False)
        self.critics = torch.nn.ModuleList([torch.nn.Linear(2, 1, bias=False)])
        for opt, module in (("aeopt", self.ae), ("shadowopt", self.shadow),
                            ("criticopt", self.critics), ("mopt", self.metric)):
            setattr(self, opt, torch.optim.Adam(module.parameters(), lr=.003))
        self.anchors = {"rec": 1., "curvature": 1., "latent_std": [1., 1.]}
        self.fault = fault
        self.updates = 0

    def supervised_loss(self, prediction):
        loss = prediction.square().mean()
        if self.fault == "nonfinite_loss_with_finite_gradient":
            loss = loss + torch.tensor(float("nan"))
        return loss, {"loss": float(loss.detach())}

    def model_update(self, model, optimizer, batch):
        optimizer.zero_grad(set_to_none=True)
        loss, result = self.supervised_loss(model(batch["x"]))
        loss.backward()
        optimizer.step()
        return result

    def representation(self, batch, warmup=False):
        result = self.model_update(self.ae, self.aeopt, batch)
        self.model_update(self.shadow, self.shadowopt, batch)
        self.updates += 1
        return result

    def critic_prefit(self, batch):
        return self.model_update(self.critics[0], self.criticopt, batch)

    def metric_update(self, batch):
        return self.model_update(self.metric, self.mopt, batch)


def episode():
    return dict(action=np.zeros((2, 4)), rgb=np.zeros((3, 2, 2, 3), dtype=np.uint8),
        offset=np.zeros((3, 4)), cost=np.zeros(2), terminal=np.array([False, True]),
        truncated=np.zeros(2, dtype=bool), previous_action=np.zeros((2, 4)),
        smooth_center=np.zeros((1, 2, 2, 3)), smooth_minus=np.zeros((1, 2, 2, 3)),
        smooth_plus=np.zeros((1, 2, 2, 3)), smooth_step=np.ones(1))


class TinyReplay:
    def __init__(self):
        self.adds = 0
        self.arrays = episode()
        self.batch = {"x": torch.tensor([[.1, .2], [-.2, .4]])}

    def add(self, arrays, row, identifier):
        self.adds += 1
        self.arrays = arrays

    def load(self, row): return self.arrays
    def sample(self, count): return self.batch


class LegacyCPUFixture:
    def __init__(self, output, fault=None):
        self.output, self.fault = Path(output), fault
        self.closed = False
        self.saved = 0

    def guard(self):
        if self.closed:
            raise RuntimeError("closed")

    def initialize(self):
        self.output.mkdir(parents=True)
        self.learner, self.replay = TinyLearner(self.fault), TinyReplay()
        self.initial_state = {"ae": deepcopy(self.learner.ae.state_dict())}
        self.fixed_probe = None
        if self.fault == "corrupt_initial":
            with torch.no_grad():
                self.learner.metric.weight.fill_(float("nan"))
        self.save(self.output / "initial.pt")
        return {"initialized": True}

    def _updates(self, phase, updates):
        result = None
        for _ in range(updates):
            batch = self.replay.sample(2)
            result = getattr(self.learner, phase)(batch)
        return result

    def save(self, path):
        self.saved += 1
        payload = {name: getattr(self.learner, name).state_dict() for name in
                   (*_StableRGB._module_names, *_StableRGB._optimizer_names)}
        torch.save(payload, path)
        return {}

    def load(self, path):
        payload = torch.load(path, weights_only=False)
        for name, value in payload.items():
            getattr(self.learner, name).load_state_dict(value)
        return {}

    def evaluate(self, *args, **kwargs): return {"metric": 1.}
    def close(self): self.closed = True


class CPUBackend(_StableRGB, LegacyCPUFixture):
    pass


@pytest.fixture
def backend(tmp_path):
    result = CPUBackend(tmp_path / "guarded")
    result.initialize()
    yield result
    result.close()


def test_wrappers_preserve_the_retained_rgb_base_classes():
    assert issubclass(StableRGBBackend, RGBBackend)
    assert issubclass(StableHistoricalRGBBackend, HistoricalRGBBackend)


def test_initial_save_is_guarded_before_initialize_returns(tmp_path):
    backend = CPUBackend(tmp_path / "bad_initial", fault="corrupt_initial")
    try:
        with pytest.raises(FloatingPointError):
            backend.initialize()
        assert backend.saved == 0
        assert not (backend.output / "initial.pt").exists()
    finally:
        backend.close()


def test_healthy_updates_preserve_adam_math_and_checkpoint_contract(backend, tmp_path):
    raw = TinyLearner()
    batch = backend.replay.batch
    for _ in range(5):
        raw.representation(batch)
        backend._updates("representation", 1)
    for name in ("ae", "shadow"):
        for actual, expected in zip(getattr(backend.learner, name).parameters(), getattr(raw, name).parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    optimizer = backend.learner.aeopt
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.state_dict().keys() == raw.aeopt.state_dict().keys()
    path = tmp_path / "candidate.pt"
    backend.save(path)
    assert set(torch.load(path, weights_only=False)) == set(
        (*_StableRGB._module_names, *_StableRGB._optimizer_names))


def test_nonfinite_loss_is_rejected_before_any_optimizer_step(tmp_path):
    backend = CPUBackend(tmp_path / "bad_loss", fault="nonfinite_loss_with_finite_gradient")
    backend.initialize()
    try:
        initial = backend.learner.ae.weight.detach().clone()
        with pytest.raises(FloatingPointError):
            backend._updates("representation", 1)
        torch.testing.assert_close(backend.learner.ae.weight, initial, rtol=0, atol=0)
        assert not backend.learner.aeopt.state
        assert backend.learner.updates == 0
        with pytest.raises(FloatingPointError):
            backend.save(backend.output / "forbidden.pt")
    finally:
        backend.close()


@pytest.mark.parametrize("corruption", ["gradient", "parameter", "optimizer_state"])
def test_optimizer_input_fault_stops_steps_and_latches(backend, corruption):
    backend._updates("representation", 1)
    optimizer = backend.learner.aeopt
    parameter = next(backend.learner.ae.parameters())
    parameter.grad = torch.ones_like(parameter)
    before = parameter.detach().clone()
    if corruption == "gradient":
        parameter.grad.fill_(float("nan"))
    elif corruption == "parameter":
        with torch.no_grad(): parameter.fill_(float("nan"))
    else:
        optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        optimizer.step()
    if corruption != "parameter":
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    with pytest.raises(FloatingPointError):
        backend.evaluate("test")
    with pytest.raises(FloatingPointError):
        backend.save(backend.output / "forbidden.pt")


def test_finite_gradient_overflow_is_detected_after_adam_step(backend):
    parameter = next(backend.learner.ae.parameters())
    parameter.grad = torch.full_like(parameter, 1.e30)
    with pytest.raises(FloatingPointError):
        backend.learner.aeopt.step()
    with pytest.raises(FloatingPointError):
        backend._updates("representation", 1)


def test_optimizer_guards_survive_load_state_dict(backend, tmp_path):
    backend._updates("representation", 1)
    path = tmp_path / "snapshot.pt"
    backend.save(path)
    optimizer = backend.learner.aeopt
    backend.load(path)
    assert backend.learner.aeopt is optimizer
    parameter = next(backend.learner.ae.parameters())
    optimizer.state[parameter]["exp_avg_sq"].fill_(float("inf"))
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(FloatingPointError):
        optimizer.step()


def test_loaded_nonfinite_optimizer_state_cannot_be_used(backend, tmp_path):
    backend._updates("representation", 1)
    path = tmp_path / "snapshot.pt"
    backend.save(path)
    payload = torch.load(path, weights_only=False)
    next(iter(payload["aeopt"]["state"].values()))["exp_avg"].fill_(float("nan"))
    corrupt = tmp_path / "corrupt.pt"
    torch.save(payload, corrupt)
    with pytest.raises(FloatingPointError):
        backend.load(corrupt)
    with pytest.raises(FloatingPointError):
        backend._updates("representation", 1)


def test_loading_a_healthy_snapshot_cannot_hide_corrupt_live_state(backend):
    with torch.no_grad():
        backend.learner.ae.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        backend.load(backend.output / "initial.pt")
    assert torch.isnan(backend.learner.ae.weight).all()


@pytest.mark.parametrize("corruption", ["nonfinite", "successor_length", "terminal", "empty_triplet"])
def test_invalid_episode_is_rejected_before_replay_write(backend, corruption):
    arrays = episode()
    if corruption == "nonfinite": arrays["cost"][0] = np.nan
    elif corruption == "successor_length": arrays["rgb"] = arrays["rgb"][:-1]
    elif corruption == "terminal": arrays["terminal"][0] = True
    else: arrays["smooth_center"] = arrays["smooth_center"][:0]
    with pytest.raises(FloatingPointError):
        backend.replay.add(arrays, {"split": "train"}, "probe")
    assert backend.replay.adds == 0
    with pytest.raises(FloatingPointError):
        backend.save(backend.output / "forbidden.pt")


def test_corrupt_cached_episode_and_batch_are_rechecked(backend):
    backend.replay.arrays["cost"][0] = np.inf
    with pytest.raises(FloatingPointError):
        backend.replay.load({})


def test_corrupt_sample_is_rejected_before_update(backend):
    backend.replay.batch["x"][0, 0] = float("nan")
    initial = backend.learner.ae.weight.detach().clone()
    with pytest.raises(FloatingPointError):
        backend._updates("representation", 1)
    torch.testing.assert_close(backend.learner.ae.weight, initial, rtol=0, atol=0)


def test_healthy_episode_still_enters_and_leaves_replay(backend):
    arrays = episode()
    backend.replay.add(arrays, {"split": "train"}, "probe")
    assert backend.replay.adds == 1
    assert backend.replay.load({}) is arrays
    assert backend.replay.sample(2) is backend.replay.batch


def test_original_probe_overflow_latches_before_checkpoint(backend, monkeypatch):
    class Metric(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.eye(2))

        def forward(self, goal):
            matrix = self.matrix.expand(len(goal), 2, 2)
            return matrix, matrix

    monkeypatch.setattr(LegacyCPUFixture, "_record_probe", RGBBackend._record_probe, raising=False)
    monkeypatch.setattr(LegacyCPUFixture, "_optimizer_counts", RGBBackend._optimizer_counts, raising=False)
    backend.learner.metric = Metric()
    backend.learner.counts = dict(warmup=0, representation=0, critic_prefit=0, metric=0)
    backend.learner.encode = lambda array: torch.as_tensor(array, dtype=torch.float32)
    backend.learner.tensor = lambda array: torch.as_tensor(array, dtype=torch.float32)
    backend.adapter = SimpleNamespace(core=SimpleNamespace(
        energy=lambda error, matrix: torch.einsum("bi,bij,bj->b", error, matrix, error),
        pose_error=lambda offset: offset.square().sum(-1)))
    backend.fixed_probe = dict(rgb=np.array([[1.e30, 0.]], dtype=np.float32),
        goal_rgb=np.zeros((1, 2), dtype=np.float32), offset=np.ones((1, 4)), goal_key=["probe"])
    with pytest.raises(FloatingPointError, match="Nonfinite RGB training probe"):
        backend._record_probe("overflow_fixture")
    assert (backend.output / "rgb_step_probes.jsonl").read_text() == ""
    with pytest.raises(FloatingPointError):
        backend.save(backend.output / "forbidden.pt")
