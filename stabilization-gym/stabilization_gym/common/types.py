"""Small contracts shared by environments and learners."""
from dataclasses import dataclass
import math
from numbers import Integral


@dataclass(frozen=True)
class MetricSpec:
    error_dim: int
    eigen_floor: float = 0.05
    trace: float | None = None

    def __post_init__(self):
        trace = float(self.error_dim) if self.trace is None else float(self.trace)
        if not isinstance(self.error_dim, Integral) or isinstance(self.error_dim, bool) or self.error_dim < 1 or not math.isfinite(trace) or not 0 < self.eigen_floor < trace / self.error_dim:
            raise ValueError("Require dimension > 0 and 0 < eigen_floor < trace / dimension")
        object.__setattr__(self, "trace", trace)

    @property
    def raw_dim(self):
        return self.error_dim * (self.error_dim + 1) // 2

    @property
    def eigen_ceiling(self):
        return self.trace - (self.error_dim - 1) * self.eigen_floor
