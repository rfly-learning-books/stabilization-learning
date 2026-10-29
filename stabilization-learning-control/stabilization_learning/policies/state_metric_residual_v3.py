"""State-dependent metric with an exact nominal metric at the target.

The gate is O(||e||^2), so M(0)=P0 and DM(0)=0 for every parameter value.
This preserves the nominal *linearization*, not a global stability theorem.
"""
import torch
from .state_metric_v2 import StateMetricPolicy


class AnchoredStateMetricPolicy(StateMetricPolicy):
    def __init__(self, spec, basis, *, goal, context_scale, hidden=16,
                 residual_limit=1.0):
        super().__init__(spec, basis, goal=goal, context_scale=context_scale, hidden=hidden)
        if not 0 < residual_limit <= 2:
            raise ValueError("Require a positive, bounded raw residual")
        self.register_buffer("residual_limit", torch.tensor(residual_limit, dtype=torch.float64))

    def raw(self, error):
        z = error / self.context_scale
        radius2 = z.square().sum(-1, keepdim=True)
        gate = radius2 / (1 + radius2)
        context = torch.cat((z, self.goal.expand_as(error) / self.context_scale), -1)
        return self.center + gate * self.residual_limit * torch.tanh(self.net(context))
