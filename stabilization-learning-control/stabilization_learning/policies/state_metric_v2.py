"""State-conditioned physical M; fixed trace belongs to S, never to M."""
from __future__ import annotations
import torch
from torch import nn
from .metric import CholeskyMetric


class StateMetricPolicy(nn.Module):
    def __init__(self, spec, basis, *, goal, context_scale, hidden=32):
        super().__init__()
        self.shape_metric=CholeskyMetric(spec).double()
        self.net=nn.Sequential(nn.Linear(2*spec.error_dim,hidden),nn.Tanh(),
            nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,spec.raw_dim)).double()
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("center",self.shape_metric.identity_raw(dtype=torch.float64))
        self.register_buffer("basis",torch.as_tensor(basis,dtype=torch.float64).clone())
        self.register_buffer("goal",torch.as_tensor(goal,dtype=torch.float64).clone())
        scale=torch.as_tensor(context_scale,dtype=torch.float64)
        self.register_buffer("context_scale",torch.broadcast_to(scale,(spec.error_dim,)).clone())
        if (self.context_scale<=0).any():
            raise ValueError("Context normalization must be fixed and positive")

    def raw(self,error):
        goal=self.goal.expand_as(error)
        context=torch.cat((error/self.context_scale,goal/self.context_scale),dim=-1)
        return self.center+self.net(context)

    def shape(self,error):
        return self.shape_metric(self.raw(error))

    def forward(self,error):
        S=self.shape(error)
        return self.basis@S@self.basis.T
