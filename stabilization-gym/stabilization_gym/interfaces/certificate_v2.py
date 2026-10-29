"""Full candidate-certificate queries with explicit model/representation identity.

No optimizer dependency. Existing frozen research trainers remain untouched.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable
import torch
from .m_learning import CertificateResult


@dataclass(frozen=True)
class VersionBinding:
    task: str
    model: str
    representation: str
    normalization: str
    reference: str
    goal: str
    certificate: str
    controller: str
    protocol: str
    weights: str
    action_units: str
    metric_scope: str

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in asdict(self).values()):
            raise ValueError("Every identity component must be explicitly declared")
        if self.metric_scope not in {"episode_fixed", "state_dependent", "goal_conditioned"}:
            raise ValueError("Unknown metric scope")

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def require(self, other):
        if not isinstance(other, VersionBinding) or self != other:
            raise ValueError("Stale or incompatible model/representation/certificate/action-unit binding")


@dataclass
class CompleteCertificateResult(CertificateResult):
    context_gradient: torch.Tensor | None = None
    version_digest: str = ""


class CertificateEvaluator:
    """matrix(error, context) is re-evaluated at every requested endpoint.

    Error and independent context are separate coordinates. A state occurring
    in error must not also be supplied as an independent context coordinate
    when taking its directional derivative.
    """
    def __init__(self, matrix: Callable, binding: VersionBinding, *,
                 factor=1., live_version: Callable | None = None):
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Certificate factor must be positive")
        self.matrix, self.binding = matrix, binding
        self.factor, self.live_version = float(factor), live_version

    def _check(self, binding):
        self.binding.require(binding or self.binding)
        if self.live_version is not None:
            self.binding.require(self.live_version())

    def evaluate(self, error, context=None, *, derivative=False, binding=None):
        self._check(binding)
        if not torch.is_tensor(error) or not error.is_floating_point() or not torch.isfinite(error).all():
            raise ValueError("Finite floating error coordinates required")
        if derivative and not error.requires_grad:
            error = error.detach().clone().requires_grad_(True)
        if derivative and torch.is_tensor(context) and not context.requires_grad:
            context = context.detach().clone().requires_grad_(True)
        with torch.set_grad_enabled(derivative or torch.is_grad_enabled()):
            M = self.matrix(error, context)
            if M.shape[-2:] != (error.shape[-1], error.shape[-1]) or not torch.isfinite(M).all():
                raise ValueError("Matrix does not match certificate coordinates")
            if not torch.allclose(M, M.transpose(-1, -2), rtol=1e-6, atol=1e-8):
                raise ValueError("Certificate matrix is not symmetric")
            if torch.any(torch.linalg.eigvalsh(M) <= 0):
                raise ValueError("Certificate matrix must be positive definite")
            value = self.factor * torch.einsum("...i,...ij,...j->...", error, M, error)
            grad, context_grad = None, None
            if derivative:
                inputs = (error, context) if torch.is_tensor(context) and context.requires_grad else (error,)
                derivatives = torch.autograd.grad(value.sum(), inputs, create_graph=True, allow_unused=True)
                grad = derivatives[0]
                if len(derivatives) > 1:
                    context_grad = derivatives[1]
        return CompleteCertificateResult(M=M, value=value, full_error_gradient=grad,
            context_gradient=context_grad, version_digest=self.binding.digest, claim="candidate_only")

    def difference(self, error, following_error, context=None, following_context=None, *, binding=None):
        before = self.evaluate(error, context, binding=binding)
        after = self.evaluate(following_error, following_context, binding=binding)
        return after.value - before.value

    def directional_derivative(self, error, error_flow, context=None, context_flow=None, *, binding=None):
        result = self.evaluate(error, context, derivative=True, binding=binding)
        derivative = (result.full_error_gradient * error_flow).sum(-1)
        if context_flow is not None and result.context_gradient is not None:
            derivative = derivative + (result.context_gradient * context_flow).sum(-1)
        return derivative
