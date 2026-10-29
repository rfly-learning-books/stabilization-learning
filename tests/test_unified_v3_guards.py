"""CPU checks for non-Push guards on unchanged PyTorch Adam instances."""
from copy import deepcopy
import gc
import weakref

import pytest
import torch

from extensions.unified_training_v3.guards import NumericalGuard


def test_hooks_preserve_adam_updates_state_and_identity_after_restore():
    torch.manual_seed(75219)
    baseline = torch.nn.Linear(3, 2).double()
    guarded = deepcopy(baseline)
    expected = torch.optim.Adam(baseline.parameters(), lr=.001)
    actual = torch.optim.Adam(guarded.parameters(), lr=.001)
    guard = NumericalGuard("test")
    guard.watch_module(guarded, "network")
    guard.watch_optimizer(actual, "Adam")
    guard.watch_optimizer(actual, "Adam")  # Idempotent, even after state loads.
    assert len(actual._optimizer_step_pre_hooks) == 1
    for index in range(5):
        data = torch.tensor([[.1, .3, -.7]], dtype=torch.float64)
        for model, optimizer in ((baseline, expected), (guarded, actual)):
            optimizer.zero_grad()
            model(data).square().sum().backward()
            optimizer.step()
        if index == 2:
            actual.load_state_dict(deepcopy(actual.state_dict()))
        for left, right in zip(baseline.parameters(), guarded.parameters()):
            assert torch.equal(left, right)
        for left, right in zip(expected.state.values(), actual.state.values()):
            assert all(torch.equal(left[key], right[key]) for key in left)
        guard.check_all()
    assert type(actual) is torch.optim.Adam
    guard.close()
    assert not actual._optimizer_step_pre_hooks and not actual._optimizer_step_post_hooks


@pytest.mark.parametrize("fault", ["gradient", "moment", "learning_rate"])
def test_optimizer_corruption_stops_before_step_and_latches(fault):
    param = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.Adam([param], lr=.01)
    guard = NumericalGuard("test")
    guard.watch_optimizer(optimizer, "Adam")
    param.grad = torch.tensor([.1])
    optimizer.step()
    before = param.detach().clone()
    if fault == "gradient":
        param.grad.fill_(torch.inf)
    elif fault == "moment":
        optimizer.state[param]["exp_avg"].fill_(torch.nan)
    else:
        optimizer.param_groups[0]["lr"] = float("nan")
    with pytest.raises(FloatingPointError):
        optimizer.step()
    assert torch.equal(before, param)
    with pytest.raises(FloatingPointError, match="stopped after numerical failure"):
        guard.guard()
    guard.close()


def test_finite_gradient_that_overflows_adam_state_is_detected_after_step():
    param = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.Adam([param], lr=.01)
    guard = NumericalGuard("test")
    guard.watch_optimizer(optimizer, "Adam")
    param.grad = torch.tensor([1.e30])
    with pytest.raises(FloatingPointError, match="optimizer state"):
        optimizer.step()
    guard.close()


def test_close_releases_watched_objects_and_is_idempotent():
    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(module.parameters())
    guard = NumericalGuard("test")
    guard.watch_module(module, "module")
    guard.watch_optimizer(optimizer, "optimizer")
    refs = [weakref.ref(module), weakref.ref(optimizer)]
    del module, optimizer
    guard.close()
    guard.close()
    gc.collect()
    assert all(reference() is None for reference in refs)
    with pytest.raises(RuntimeError, match="closed"):
        guard.guard()
