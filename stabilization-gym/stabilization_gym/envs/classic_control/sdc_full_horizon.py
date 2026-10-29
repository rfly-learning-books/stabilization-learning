"""Explicit capability adapter; the original four physical models are unchanged."""
from dataclasses import replace
import numpy as np
import torch
from .sdc_metric import SDCMetricTask


class FullHorizonSDCTask(SDCMetricTask):
    def __init__(self, physical, name):
        super().__init__(physical, name)
        self.description = replace(self.description,
            backend=type(self).__module__,
            support_level="full_horizon_state_M_model_gradient",
            supported_algorithms=("state_metric_full_horizon_v3",))

    def boundary_excess(self, state):
        """Differentiable excess; does not replace the actual failure flags."""
        parts = [(torch.linalg.vector_norm(state, dim=-1) / 100 - 1).relu().square()]
        e = self.error(state)
        if self.name == "pendulum":
            parts += [(e[..., 0].abs() / (np.pi / 2) - 1).relu().square(),
                      (state[..., 1].abs() / 12 - 1).relu().square()]
        elif self.name == "cartpole":
            parts += [(state[..., 0].abs() / 2.4 - 1).relu().square(),
                      (e[..., 2].abs() / (np.pi / 4) - 1).relu().square()]
        return torch.stack(parts).sum(0)
