"""Algorithm-owned candidate certificate using the verified query evaluator."""
from __future__ import annotations
import torch
from torch import nn
from stabilization_gym.interfaces.certificate_v2 import CertificateEvaluator, VersionBinding


def make_binding(*, task: str, revision: str, action_units: str,
                 metric_scope: str = "state_dependent", **overrides) -> VersionBinding:
    fields = dict(task=task, model=revision, representation=revision,
                  normalization=revision, reference=revision, goal=revision,
                  certificate=revision, controller=revision, protocol=revision,
                  weights=revision, action_units=action_units, metric_scope=metric_scope)
    fields.update(overrides)
    return VersionBinding(**fields)


class NeuralCertificate(nn.Module):
    """matrix(error, context) must return the actual SPD matrix used by control.

    This wrapper owns the matrix module. The evaluator computes complete
    endpoint values and gradients including error-dependent matrix terms.
    """
    def __init__(self, matrix: nn.Module, binding: VersionBinding, factor: float = 1.,
                 online_matrix=None):
        super().__init__()
        self.matrix = matrix
        self.binding = binding
        self.factor = factor
        self._online_matrix = online_matrix
        self.evaluator = CertificateEvaluator(self.matrix, binding, factor=factor,
                                             live_version=lambda: self.binding)

    def evaluate(self, error, context=None, **kwargs):
        return self.evaluator.evaluate(error, context, **kwargs)

    def online_matrix(self, metric_context, *, binding=None):
        """Observer gain queries must not require an unavailable true error."""
        self.evaluator._check(binding)
        query = self._online_matrix or getattr(self.matrix, "online_matrix", None)
        if query is None:
            raise TypeError("Declare an online_matrix(context) provider for observer control")
        matrix = query(metric_context)
        if (matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]
                or not torch.isfinite(matrix).all()
                or not torch.allclose(matrix, matrix.transpose(-1, -2), rtol=1e-6, atol=1e-8)
                or torch.any(torch.linalg.eigvalsh(matrix) <= 0)):
            raise ValueError("Online matrix must be finite, symmetric and positive definite")
        return matrix

    def gradient(self, error, context=None, **kwargs):
        return self.evaluate(error, context, derivative=True, **kwargs)

    def difference(self, error, following_error, context=None, following_context=None, **kwargs):
        return self.evaluator.difference(error, following_error, context, following_context, **kwargs)

    def directional_derivative(self, *args, **kwargs):
        return self.evaluator.directional_derivative(*args, **kwargs)

    def rebind(self, binding: VersionBinding):
        self.binding = binding
        self.evaluator = CertificateEvaluator(self.matrix, binding, factor=self.factor,
                                             live_version=lambda: self.binding)
