"""Deterministic adjoint for the retained RGB encoder's one-pixel reflection.

The convolution, kernel, learned tensors and optimizer objects are unchanged.
Only already-created GaussianBlur instances receive this equivalent padding.
"""
from functools import wraps
from types import MethodType

import torch
# PyTorch 2.3 lazily imports this inside custom_function_call. Loading it before
# entering a jvp transform avoids its NestedTensor initialization under a dual.
import torch._dynamo
from torch.nn import functional as F

from extensions.unified_training_v3.rgb import StableRGBBackend, StableHistoricalRGBBackend


REVISION = "reflect_pad1_fixed_gather_ordered_adjoint_v4"


def _gather_reflection(x):
    height, width = x.shape[-2:]
    if x.ndim not in (3, 4) or height <= 1 or width <= 1:
        raise ValueError("One-pixel reflection requires a 3D/4D tensor with both spatial dimensions > 1")
    rows = torch.cat((torch.tensor([1], device=x.device), torch.arange(height, device=x.device),
                      torch.tensor([height-2], device=x.device)))
    columns = torch.cat((torch.tensor([1], device=x.device), torch.arange(width, device=x.device),
                         torch.tensor([width-2], device=x.device)))
    return x.index_select(-2, rows).index_select(-1, columns)


class _ReflectPad1(torch.autograd.Function):
    # The legacy encoder uses torch.func.jvp under vmap for visual Jacobians.
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return _gather_reflection(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.height, ctx.width = inputs[0].shape[-2:]

    @staticmethod
    def backward(ctx, gradient):
        height, width = ctx.height, ctx.width
        # First apply the transpose of the column gather. A column receives its
        # central entry and at most two reflected entries, in this fixed order.
        middle = gradient[..., 1:width+1]
        left, right = gradient[..., :1], gradient[..., -1:]
        narrow_zero = torch.zeros_like(left)
        wide_zero = gradient.new_zeros((*gradient.shape[:-1], width-2))
        columns = middle + torch.cat((narrow_zero, left, wide_zero), dim=-1)
        columns = columns + torch.cat((wide_zero, right, narrow_zero), dim=-1)
        # Then apply the row transpose. All operations are gather, concatenate
        # and pointwise sum; there is no overlapping write or atomic reduction.
        middle = columns[..., 1:height+1, :]
        top, bottom = columns[..., :1, :], columns[..., -1:, :]
        narrow_zero = torch.zeros_like(top)
        wide_zero = columns.new_zeros((*columns.shape[:-2], height-2, width))
        result = middle + torch.cat((narrow_zero, top, wide_zero), dim=-2)
        return result + torch.cat((wide_zero, bottom, narrow_zero), dim=-2)

    @staticmethod
    def jvp(ctx, tangent):
        # Reuse the guarded primitive so reverse-over-forward differentiation
        # does not reintroduce index_select's accumulation backward on CUDA.
        return _ReflectPad1.apply(tangent)


def deterministic_reflect_pad2d(x):
    """Equivalent to F.pad(x, (1,1,1,1), mode='reflect'), including its adjoint."""
    return _ReflectPad1.apply(x)


def _blur_forward(self, x):
    return F.conv2d(deterministic_reflect_pad2d(x), self.kernel,
                    stride=self.stride, groups=x.shape[1])


def install_deterministic_blur(model):
    """Patch only this module graph's GaussianBlur methods; retain all objects."""
    names = []
    for name, module in model.named_modules():
        if type(module).__name__ != "GaussianBlur":
            continue
        kernel = getattr(module, "kernel", None)
        if not torch.is_tensor(kernel) or kernel.ndim != 4 or kernel.shape[1:] != (1, 3, 3) or not hasattr(module, "stride"):
            raise TypeError("Unexpected GaussianBlur contract")
        if not getattr(module, "_v4_deterministic_reflection", False):
            module.forward = MethodType(_blur_forward, module)
            module._v4_deterministic_reflection = True
        names.append(name)
    if not names:
        raise RuntimeError("The declared smooth RGB model contains no GaussianBlur instances")
    return names


class _DeterministicRGB:
    def initialize(self):
        result = super().initialize()
        installed = {}
        for name in ("ae", "shadow", "initial"):
            model = getattr(self.learner, name, None)
            if model is not None:
                installed[name] = install_deterministic_blur(model)
        # Initial/historical test models are loaded lazily. Bind this adapter
        # instance only, before any visual reference computes its encoder JVP.
        original_load = self.adapter.load_models
        @wraps(original_load)
        def load_models(*args, **kwargs):
            ae, metric = original_load(*args, **kwargs)
            install_deterministic_blur(ae)
            return ae, metric
        self.adapter.load_models = load_models
        result = dict(result, deterministic_reflection_padding=dict(revision=REVISION,
            modules=installed, scope="same reflection forward and mathematical adjoint; fixed reduction order"))
        self.event("deterministic_rgb_padding_installed", revision=REVISION, modules=installed)
        return result


class DeterministicRGBBackend(_DeterministicRGB, StableRGBBackend):
    """Strict deterministic CUDA backend with the retained RGB training math."""


class DeterministicHistoricalRGBBackend(_DeterministicRGB, StableHistoricalRGBBackend):
    """The same local deterministic implementation with frozen historical tests."""
