"""Fail-closed numerical boundaries for new Push runs; no repair or tuning."""
import math
from numbers import Real

import numpy as np
import torch


def require_finite(value, label):
    """Inspect numerical leaves without detaching or changing their values."""
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite {label}")
    elif isinstance(value, np.ndarray):
        # Integer RGB pixels and boolean masks cannot encode NaN/Inf. Avoid a
        # full scan of those large arrays at every replay/learner boundary.
        if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
    elif isinstance(value, Real) and not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite {label}")


class FiniteAdam(torch.optim.Adam):
    """Keep Adam mathematics, rejecting corrupt inputs and failed updates."""
    def __init__(self, params, *, failure, label, **kwargs):
        self.failure, self.label = failure, label
        super().__init__(params, **kwargs)

    def check(self, *, gradients=False):
        for group in self.param_groups:
            require_finite({key: value for key, value in group.items() if key != "params"},
                           self.label + " optimizer settings")
            for parameter in group["params"]:
                require_finite(parameter, self.label + " parameter")
                require_finite(self.state.get(parameter, {}), self.label + " optimizer state")
                if gradients and parameter.grad is not None:
                    require_finite(parameter.grad, self.label + " gradient")

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Push numerical guard expects explicit loss/backward, not an optimizer closure")
        try:
            self.check(gradients=True)
            result = super().step()
            self.check()
            return result
        except FloatingPointError as error:
            self.failure(error)
            raise


class CheckedReplay(list):
    def __init__(self, *, failure):
        super().__init__()
        self.failure = failure

    def append(self, row):
        try:
            require_finite(row, "replay transition")
            if row["dt"] < 0:
                raise FloatingPointError("Negative replay transition duration")
        except FloatingPointError as error:
            self.failure(error)
            raise
        super().append(row)
