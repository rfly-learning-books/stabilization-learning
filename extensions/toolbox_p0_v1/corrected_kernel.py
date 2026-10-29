"""Reuse the original tensor executor with an explicitly new objective identity."""
from __future__ import annotations
from dataclasses import replace
import hashlib
from extensions.fast_m_learning_v1.fast_m_learning.tensor_kernel import TensorTaskKernel
from stabilization_learning.objectives_v2 import descent_terms, OBJECTIVE_VERSION


class CorrectedTensorKernel(TensorTaskKernel):
    def __init__(self, task, **kwargs):
        super().__init__(task,**kwargs)
        signature=hashlib.sha256((self.spec.signature+OBJECTIVE_VERSION).encode()).hexdigest()
        self._spec=replace(self.spec,signature=signature)

    def transition(self, state, M):
        result=super().transition(state,M)
        terms=descent_terms(result["V_after"]-result["V_before"],result["physical_error_before"],
            dt=self.spec.dt,decay=self.decay)
        result.update(decay_residual=terms["residual"],normalized_decay_residual=terms["normalized_residual"],
                      penalty=terms["penalty"])
        return result
