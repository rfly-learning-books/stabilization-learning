"""Equivalent reflection math and real strict-mode CUDA RGB learning."""
import ast
from copy import deepcopy
import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from extensions.reproducible_training_v4.deterministic_rgb import (
    deterministic_reflect_pad2d, install_deterministic_blur, _DeterministicRGB,
    DeterministicRGBBackend, DeterministicHistoricalRGBBackend, REVISION)
from extensions.reproducible_training_v4.runtime import configure, content_hash
from extensions.unified_training_v3.rgb import StableRGBBackend, StableHistoricalRGBBackend

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def strict_runtime():
    configure(83017)
    assert torch.are_deterministic_algorithms_enabled()


def retained_blur(channels=2, stride=1):
    # Execute only the exact retained class definition, avoiding scene loading
    # and legacy configuration imports in the CPU mathematical tests.
    path = ROOT/"third_party/visual_original/runtime/cae_vs_common.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianBlur")
    namespace = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["GaussianBlur"](channels, stride=stride)


@pytest.mark.parametrize("shape", [(1, 1, 2, 2), (1, 2, 2, 3), (2, 3, 4, 5), (3, 4, 2)])
def test_forward_and_adjoint_match_retained_reflection_on_cpu(shape):
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    reference = F.pad(x, (1, 1, 1, 1), mode="reflect")
    actual = deterministic_reflect_pad2d(x)
    assert torch.equal(actual, reference)
    weights = torch.randn_like(reference)
    first = torch.autograd.grad(actual, x, weights, retain_graph=True)[0]
    second = torch.autograd.grad(reference, x, weights)[0]
    torch.testing.assert_close(first, second, rtol=1e-14, atol=1e-14)


def test_double_gradcheck_and_gradgradcheck():
    x = torch.randn(1, 1, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(deterministic_reflect_pad2d, (x,))
    assert torch.autograd.gradgradcheck(deterministic_reflect_pad2d, (x,))


def test_jvp_vmap_and_reverse_over_forward_match_reflection():
    x = torch.randn(2, 1, 3, 4, dtype=torch.float64)
    tangent = torch.randn_like(x, requires_grad=True)
    def transformed(function):
        return torch.vmap(lambda value, direction: torch.func.jvp(function, (value,), (direction,))[1])(x, tangent)
    actual = transformed(deterministic_reflect_pad2d)
    expected = transformed(lambda value: F.pad(value, (1, 1, 1, 1), mode="reflect"))
    assert torch.equal(actual, expected)
    weights = torch.randn_like(actual)
    first = torch.autograd.grad(actual, tangent, weights, retain_graph=True)[0]
    second = torch.autograd.grad(expected, tangent, weights)[0]
    torch.testing.assert_close(first, second, rtol=1e-14, atol=1e-14)


def test_patch_preserves_parameters_buffers_optimizer_and_other_instances():
    blur = retained_blur()
    other = deepcopy(blur)
    model = nn.Sequential(nn.Conv2d(2, 2, 1), blur).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    original_forward = type(blur).forward
    before = deepcopy(model.state_dict())
    parameter_ids = [id(p) for p in model.parameters()]
    buffer_ids = [id(b) for b in model.buffers()]
    x = torch.randn(2, 2, 5, 4, dtype=torch.float64)
    expected = model(x)
    assert install_deterministic_blur(model) == ["1"]
    assert install_deterministic_blur(model) == ["1"]
    assert type(blur).forward is original_forward and other.forward.__func__ is original_forward
    assert parameter_ids == [id(p) for p in model.parameters()]
    assert buffer_ids == [id(b) for b in model.buffers()]
    assert parameter_ids == [id(p) for p in optimizer.param_groups[0]["params"]]
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    clone = deepcopy(model)
    torch.testing.assert_close(clone(x), expected, rtol=0, atol=0)


def test_backend_installation_is_local_and_covers_lazy_models():
    class Fixture:
        def initialize(self):
            self.learner = SimpleNamespace(**{name: nn.Sequential(retained_blur()) for name in ("ae", "shadow", "initial")})
            self.adapter = SimpleNamespace(load_models=lambda *a, **k: (nn.Sequential(retained_blur()), nn.Linear(1, 1)))
            self.events = []
            self.event = lambda *a, **k: self.events.append((a, k))
            return {"initialized": True}
    class Backend(_DeterministicRGB, Fixture):
        pass
    backend = Backend()
    result = backend.initialize()
    assert result["deterministic_reflection_padding"]["revision"] == REVISION
    assert set(result["deterministic_reflection_padding"]["modules"]) == {"ae", "shadow", "initial"}
    for name in ("initial.pt", "best.pt"):
        ae, _ = backend.adapter.load_models(name)
        assert ae[0]._v4_deterministic_reflection
    assert issubclass(DeterministicRGBBackend, StableRGBBackend)
    assert issubclass(DeterministicHistoricalRGBBackend, StableHistoricalRGBBackend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Strict CUDA unit test requires a GPU")
def test_cuda_backward_is_strict_and_bitwise_repeatable():
    base = torch.randn(2, 3, 8, 7, dtype=torch.float64, device="cuda")
    weight = torch.randn(2, 3, 10, 9, dtype=torch.float64, device="cuda")
    results = []
    for _ in range(2):
        x = base.clone().requires_grad_(True)
        output = deterministic_reflect_pad2d(x)
        gradient = torch.autograd.grad(output, x, weight)[0]
        results.append((output.detach(), gradient.detach()))
    assert all(torch.equal(a, b) for a, b in zip(*results))
    cpu = base.cpu().requires_grad_(True)
    expected = torch.autograd.grad(F.pad(cpu, (1, 1, 1, 1), mode="reflect"), cpu, weight.cpu())[0]
    torch.testing.assert_close(results[0][1].cpu(), expected, rtol=1e-14, atol=1e-14)
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual retained AE model_update requires a GPU")
def test_actual_retained_ae_and_shadow_model_update_under_strict_cuda(tmp_path):
    from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
    adapter, _, _ = load_runtime(ROOT)
    cfg = deepcopy(adapter.cfg)
    cfg["goal_fixed"]["batch_size"] = 1
    source = adapter.checkpoint("initial.pt")
    learner = adapter.derivative.DerivativeLearner(cfg, source, "cuda:0")
    try:
        installed = {name: install_deterministic_blur(getattr(learner, name)) for name in ("ae", "shadow", "initial")}
        before = {name: content_hash(getattr(learner, name).state_dict()) for name in ("ae", "shadow")}
        rng = np.random.default_rng(83017)
        rgb = lambda: rng.integers(0, 256, (1, 128, 128, 3), dtype=np.uint8)
        batch = dict(rgb=rgb(), goal_rgb=rgb(), action=np.zeros((1, 4), dtype=np.float32),
            offset=np.array([[.1, -.1, .05, .03]], dtype=np.float32),
            smooth_minus=rgb(), smooth_center=rgb(), smooth_plus=rgb())
        outcome = learner.representation(batch, warmup=False)
        after = {name: content_hash(getattr(learner, name).state_dict()) for name in ("ae", "shadow")}
        assert all(before[name] != after[name] for name in before)
        assert learner.counts["representation"] == 1
        assert learner.aeopt.state and learner.shadowopt.state and learner.criticopt.state
        from lac_vs.vision import encoder_jvp, image_tensor
        image = image_tensor(cfg, batch["rgb"], "cuda:0")
        tangents = torch.ones((1, 4, *image.shape[1:]), device="cuda", dtype=image.dtype)
        jacobian = encoder_jvp(learner.ae, image, tangents)
        assert jacobian.shape == (1, 4, 16) and torch.isfinite(jacobian).all()
        assert torch.are_deterministic_algorithms_enabled()
        import json
        (tmp_path/"GPU_MODEL_UPDATE.json").write_text(json.dumps(dict(passed=True, revision=REVISION,
            torch=torch.__version__, gpu=torch.cuda.get_device_name(), strict_deterministic=True,
            installed=installed, before=before, after=after, outcome=outcome,
            jacobian_shape=list(jacobian.shape), scope="one actual AE and shadow representation update; no formal training"), indent=2), encoding="utf-8")
    finally:
        del learner, source
        gc.collect()
        torch.cuda.empty_cache()
