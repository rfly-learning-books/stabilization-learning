"""Explicit-precision fixed-horizon bookkeeping for a future executor revision."""
import torch


def fixed_horizon_cost64(accumulated,steps,terminal_error,failure,*,horizon,dt,failure_floor=25.):
    if dt<=0 or horizon<1:
        raise ValueError("Positive physical clock and horizon required")
    cost=torch.as_tensor(accumulated,dtype=torch.float64)
    count=torch.as_tensor(steps,dtype=torch.int64)
    terminal=torch.as_tensor(terminal_error,dtype=torch.float64)
    failed=torch.as_tensor(failure,dtype=torch.bool)
    if (count<0).any() or (count>horizon).any():
        raise ValueError("Executed physical steps outside declared horizon")
    tail=torch.where(failed,terminal.clamp_min(failure_floor),terminal)
    seconds=(horizon-count).to(torch.float64)*dt
    return cost+seconds*tail
