"""Small linear observer toy used only to test the observer contract."""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .core import ObserverContext, ObserverTruth, MetricObserver


@dataclass
class LinearObserverToy:
    A: torch.Tensor
    H: torch.Tensor
    R: torch.Tensor
    Q: torch.Tensor

    def __post_init__(self):
        self.A = self.A.to(torch.float64)
        self.H = self.H.to(torch.float64)
        self.R = self.R.to(torch.float64)
        self.Q = self.Q.to(torch.float64)
        self.true_state = None
        self.estimate = None
        self.prior_metric = None
        self.generator = None

    def reset(self, *, seed: int):
        self.generator = torch.Generator().manual_seed(int(seed))
        self.true_state = torch.randn(self.A.shape[0], generator=self.generator, dtype=torch.float64)
        self.estimate = self.true_state + .2 * torch.randn(self.true_state.shape, generator=self.generator, dtype=torch.float64)
        self.prior_metric = torch.eye(self.A.shape[0], dtype=torch.float64)
        return self.context()

    def context(self):
        if self.true_state is None:
            raise RuntimeError("reset required")
        innovation = self.H @ self.true_state + .01 * torch.randn(self.H.shape[0], generator=self.generator, dtype=torch.float64) - self.H @ self.estimate
        return ObserverContext(self.estimate.clone(), innovation, self.H.clone(),
            torch.ones(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            self.prior_metric.clone())

    def step(self, observer: MetricObserver):
        update = observer.update(self.context())
        self.estimate = update.posterior_estimate
        # One-step hidden transition; actions/M do not change the plant.
        self.true_state = self.A @ self.true_state + .01 * torch.randn(self.true_state.shape, generator=self.generator, dtype=torch.float64)
        self.prior_metric = update.metric
        return self.context(), update

