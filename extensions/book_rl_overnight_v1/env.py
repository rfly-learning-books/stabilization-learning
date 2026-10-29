"""Physical 2D bounded integrator and a fixed discrete Lyapunov filter."""
from __future__ import annotations
import math
import torch


def bounded(action):
    return action / action.norm(dim=-1, keepdim=True).clamp_min(1.)


def value(state):
    return .5*state.square().sum(-1)


def analytic_action(state, gain=1.5):
    return bounded(-gain*state)


def certificate_filter(state, proposal, dt=.1, decay=.25):
    """Closest action in unit disk intersected with the exact next-V disk.

    The constraint is V(x+dt*u)-V(x) <= -decay*dt*||x||^2.
    No slack is optimized. Empty sets or numerical failures explicitly return
    the minimum-next-V bounded action and a fallback flag.
    """
    if not 0 < 2*decay*dt < 1:
        raise ValueError("Discrete filter requires 0 < 2*decay*dt < 1")
    original_dtype=state.dtype
    x,p=state.double(),proposal.double()
    center=-x/dt
    distance=center.norm(dim=-1)
    radius=math.sqrt(1-2*decay*dt)*x.norm(dim=-1)/dt
    feasible=distance <= 1+radius+1e-10
    candidate=bounded(p)
    inside=(candidate-center).norm(dim=-1) <= radius+1e-10
    direction=(p-center)/(p-center).norm(dim=-1,keepdim=True).clamp_min(1e-20)
    disk_projection=center+radius[:,None]*direction
    disk_projection=torch.where((distance<1e-12)[:,None],torch.zeros_like(p),disk_projection)
    projected_in_unit=disk_projection.norm(dim=-1) <= 1+1e-10
    unit_center=center/distance[:,None].clamp_min(1e-20)
    chord=(1-radius.square()+distance.square())/(2*distance.clamp_min(1e-20))
    height=(1-chord.square()).clamp_min(0).sqrt()
    perpendicular=torch.stack([-unit_center[:,1],unit_center[:,0]],-1)
    a=chord[:,None]*unit_center+height[:,None]*perpendicular
    b=chord[:,None]*unit_center-height[:,None]*perpendicular
    boundary=torch.where(((a-p).square().sum(-1)<=(b-p).square().sum(-1))[:,None],a,b)
    result=torch.where(inside[:,None],candidate,
                       torch.where(projected_in_unit[:,None],disk_projection,boundary))
    residual=value(x+dt*result)-value(x)+decay*dt*x.square().sum(-1)
    numerical=feasible & ((result.norm(dim=-1)>1+1e-7)|(residual>1e-7)|~torch.isfinite(result).all(-1))
    fallback=~feasible|numerical
    result=torch.where(fallback[:,None],bounded(-x/dt),result).to(original_dtype)
    info=dict(feasible=feasible,fallback=fallback,solver_failure=numerical,
              correction=(result-proposal).norm(dim=-1),
              nominal_residual=value(state+dt*result)-value(state)+decay*dt*state.square().sum(-1))
    return result,info


class BoundedIntegrator:
    """Owns x, physical transitions, finite-horizon reset and goal hold count.

    First arrival never terminates collection. Boundary exit is a true failure;
    horizon truncation keeps the Bellman bootstrap in this continuing task.
    """
    def __init__(self,starts,lanes,*,device="cpu",dt=.1,horizon=128,radius=3.,tolerance=.1,hold_steps=20):
        self.starts=torch.as_tensor(starts,dtype=torch.float32,device=device)
        self.device=device;self.lanes=lanes;self.dt=dt;self.horizon=horizon
        self.radius=radius;self.tolerance=tolerance;self.hold_steps=hold_steps
        self.state=torch.zeros(lanes,2,device=device)
        self.steps=torch.zeros(lanes,dtype=torch.long,device=device)
        self.hold=torch.zeros_like(self.steps);self.episode=torch.zeros_like(self.steps)
        self.cursor=0;self.reset()

    def reset(self,indices=None):
        indices=torch.arange(self.lanes,device=self.device) if indices is None else indices
        n=len(indices)
        ids=(torch.arange(n,device=self.device)+self.cursor)%len(self.starts)
        self.state[indices]=self.starts[ids];self.steps[indices]=0;self.hold[indices]=0
        self.episode[indices]+=1;self.cursor+=n
        return self.state.clone()

    def step(self,executed,disturbance=None):
        if not torch.isfinite(executed).all() or bool((executed.norm(dim=-1)>1.00001).any()):
            raise ValueError("Executed action must be finite and in the unit disk")
        current=self.state.clone()
        nominal=current+self.dt*executed
        next_state=nominal if disturbance is None else nominal+disturbance
        failure=next_state.norm(dim=-1)>self.radius
        inside=next_state.norm(dim=-1)<=self.tolerance
        self.hold=torch.where(inside,self.hold+1,torch.zeros_like(self.hold))
        self.steps+=1;self.state=next_state
        truncated=self.steps>=self.horizon
        cost=self.dt*(next_state.square().sum(-1)+.01*executed.square().sum(-1))+20*failure.float()
        return dict(state=current,next_state=next_state.clone(),executed=executed.clone(),
            nominal_delta=value(nominal)-value(current),true_delta=value(next_state)-value(current),
            base_cost=cost,base_reward=-cost,failure=failure,truncated=truncated,
            hold=self.hold.clone(),step=self.steps.clone(),episode=self.episode.clone())
