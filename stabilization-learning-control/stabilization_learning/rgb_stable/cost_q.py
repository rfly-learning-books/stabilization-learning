"""Bounded physical-action cost critics with stationary visual coordinates."""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .objectives import component_energy


def visual_state(error,goal,interaction,previous):
    """100 observable coordinates; no physical pose, M parameter or reward input."""
    return torch.cat((error,goal,interaction.flatten(-2),previous),-1)


def transition_cost(offset,next_offset,action,failure,dt=.1):
    rho=component_energy(next_offset)
    outside=(rho>=1).any(-1)
    effort=action.square().sum(-1)
    running=dt*(outside.float()+.05*torch.log1p(rho.sum(-1)).clamp(max=8.)+.01*effort)
    cost=running+20.*failure.float()
    inside=(component_energy(offset)<1).all(-1)
    leaves=inside & (outside | failure.bool())
    return cost,inside,leaves


def bellman_cost(cost,failure,next_cost,gamma=.99):
    # Collection horizons and successful-arrival cutoffs are NOT absorbing.
    # Only physical failure removes the continuing cost-to-go.
    return cost+gamma*(~failure.bool()).to(cost.dtype)*next_cost


class BoundedCostHead(nn.Module):
    def __init__(self,cap=32.):
        super().__init__(); self.cap=float(cap)
        self.net=nn.Sequential(nn.Linear(104,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1))
        nn.init.normal_(self.net[-1].weight,std=.001)
        nn.init.constant_(self.net[-1].bias,math.log(1./(cap-1.)))

    def forward(self,state,action):
        return self.cap*torch.sigmoid(self.net(torch.cat((state,action),-1)).squeeze(-1))


class PhysicalActionCostQ(nn.Module):
    def __init__(self,cap=32.):
        super().__init__()
        self.heads=nn.ModuleList([BoundedCostHead(cap),BoundedCostHead(cap)])
        self.hold=nn.Sequential(nn.Linear(104,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.constant_(self.hold[-1].bias,-4.)
        self.register_buffer('center',torch.zeros(100))
        self.register_buffer('scale',torch.ones(100))

    @torch.no_grad()
    def calibrate(self,state):
        self.center.copy_(state.mean(0)); self.scale.copy_(state.std(0,unbiased=False).clamp_min(.03))

    def normalize(self,state):
        return torch.tanh((state-self.center)/(3*self.scale))

    def forward(self,state,action):
        x=self.normalize(state)
        values=torch.stack([q(x,action) for q in self.heads],-1)
        hold_logit=self.hold(torch.cat((x,action),-1)).squeeze(-1)
        return values,hold_logit


class FrozenFeatureReplay:
    """GPU resident transitions; room/available-phase balanced sampling."""
    def __init__(self,capacity,device):
        self.capacity=int(capacity); self.device=device; self.data={}; self.weights=None
        self.source_files=set(); self.room_ids={}; self.seen=0

    def __len__(self):
        return len(self.data['e']) if self.data else 0

    def add(self,batch,room,file):
        if file in self.source_files: raise ValueError('Feature source ingested twice')
        room_id=self.room_ids.setdefault(room,len(self.room_ids))
        n=len(batch['e']); batch={k:v.detach().to(self.device) for k,v in batch.items()}
        batch['room']=torch.full((n,),room_id,device=self.device,dtype=torch.long)
        rho=component_energy(batch['offset']).amax(-1)
        batch['phase']=torch.where(rho<=1,0,torch.where(rho<=6.25,1,2))
        if self.data and self.data.keys()!=batch.keys(): raise ValueError('Feature replay schema mismatch')
        self.data={k:torch.cat((self.data[k],v),0)[-self.capacity:] if self.data else v[-self.capacity:] for k,v in batch.items()}
        self.source_files.add(file); self.seen+=n; self.weights=None

    def sample(self,size):
        if not self.data: raise RuntimeError('Empty feature replay')
        if self.weights is None:
            room,phase=self.data['room'],self.data['phase']; w=torch.zeros(len(self),device=self.device)
            rooms=room.unique()
            for sid in rooms:
                masks=[(room==sid)&(phase==p) for p in range(3)]
                count=sum(bool(m.any()) for m in masks)
                for mask in masks:
                    if bool(mask.any()): w[mask]=1./(len(rooms)*count*int(mask.sum()))
            # CUDA multinomial uses a nondeterministic cumsum on this pinned
            # PyTorch build. Only sample indices on CPU; all features and
            # network computations remain resident on the GPU.
            self.weights=(w/w.sum()).cpu()
        indices=torch.multinomial(self.weights,size,replacement=True).to(self.device)
        return {k:v[indices] for k,v in self.data.items() if k not in ('room','phase')}
