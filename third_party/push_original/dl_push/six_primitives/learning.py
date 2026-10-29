"""Direct actor DQ prototype: D supervision + Q cost critic + actor penalties.

No candidate set, argmin action search, CLF filter, MJX or pretrained prior.
Learned penalties do not constitute a hard closed-loop stability certificate.
"""
from copy import deepcopy
import torch
from torch import nn


def shaped_cost(cost,reach,next_reach,terminated,weight,gamma=.99):
    return cost+weight*(gamma*(1-terminated)*next_reach-reach)


def feature_input(obs):
    turn=obs['task'][:,4:].sum(-1,keepdim=True)
    scale=.1+.9*turn
    return torch.cat((obs['s'],obs['g']/scale,obs['task']),-1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image=nn.Sequential(nn.Conv2d(1,16,5,2),nn.ReLU(),nn.Conv2d(16,32,3,2),nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(89,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
    def forward(self,obs):return self.fusion(torch.cat((self.image(obs['shape']),feature_input(obs)),-1))


class Actor(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.head=nn.Linear(128,2)
        nn.init.uniform_(self.head.weight,-.003,.003);nn.init.zeros_(self.head.bias)
    def forward(self,obs):return torch.tanh(self.head(self.encoder(obs)))


class Critics(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder()
        self.q1=nn.Sequential(nn.Linear(130,128),nn.ReLU(),nn.Linear(128,1))
        self.q2=nn.Sequential(nn.Linear(130,128),nn.ReLU(),nn.Linear(128,1))
        self.d_return=deepcopy(self.q1)
        # D predicts measured energy difference; auxiliary head predicts progress.
        self.d=nn.Sequential(nn.Linear(130,128),nn.ReLU(),nn.Linear(128,2))
    def forward(self,obs,action):
        z=torch.cat((self.encoder(obs),action),-1)
        return self.q1(z).squeeze(-1),self.q2(z).squeeze(-1),self.d(z),self.d_return(z).squeeze(-1)


class Replay:
    def __init__(self,capacity=100000):
        if capacity<1:raise ValueError('Invalid replay capacity')
        self.capacity=capacity;self.size=0;self.cursor=0;self.storage=None
    def add(self,obs,action,nxt,info,terminated):
        fields={**{'o_'+k:v for k,v in obs.items()},**{'n_'+k:v for k,v in nxt.items()},
                'action':action,'cost':info['q_cost'],'d':info['d_target'],
                'progress':info['progress_target'],'terminated':terminated.float(),
                'q_discount':info.get('q_discount',torch.full_like(info['q_cost'],.99)),
                'q_terminal':info.get('q_terminal',terminated).float(),
                'energy':info['energy_before'],
                'reach':info.get('reach_before',torch.zeros_like(info['energy_before'])),
                'next_reach':info.get('reach_after',torch.zeros_like(info['energy_before'])),
                'd_h':info.get('d_h',info['d_target']),
                'p_h':info.get('p_h',info['progress_target']),
                'valid_h':info.get('valid_h',torch.ones_like(info['d_target'])),
                'prep':info.get('prep_before',torch.zeros_like(info['energy_before'])),
                'next_prep':info.get('prep_after',torch.zeros_like(info['energy_before']))}
        if self.storage is None:
            self.storage={k:torch.empty((self.capacity,*v.shape[1:]),dtype=torch.uint8 if k.endswith('shape') else v.dtype)
                          for k,v in fields.items()}
        n=len(action)
        if n>self.capacity:raise ValueError('Batch exceeds replay capacity')
        ids=(torch.arange(n)+self.cursor)%self.capacity
        for k,v in fields.items():self.storage[k][ids]=v.detach().cpu().to(self.storage[k].dtype)
        self.cursor=(self.cursor+n)%self.capacity;self.size=min(self.capacity,self.size+n)
    def sample(self,n,device,balanced=False):
        ids=torch.randint(self.size,(n,))
        if balanced:
            moving=self.storage['action'][:self.size].norm(dim=-1)>.2
            active=moving.nonzero().flatten();stopped=(~moving).nonzero().flatten()
            if len(active) and len(stopped):
                ids=torch.cat((active[torch.randint(len(active),(n//2,))],stopped[torch.randint(len(stopped),(n-n//2,))]))
        b={k:v[ids].to(device).float() for k,v in self.storage.items()}
        obs={k[2:]:v for k,v in b.items() if k.startswith('o_')}
        nxt={k[2:]:v for k,v in b.items() if k.startswith('n_')}
        return obs,nxt,b


class DirectDQ:
    def __init__(self,device='cuda',penalty=10.,contraction=.02,dt=.05,behavior_weight=0.,reach_shaping=0.,progress_shaping=0.,prep_shaping=0.,horizon_steps=1):
        self.device=device;self.actor=Actor().to(device);self.critics=Critics().to(device)
        self.actor_target=deepcopy(self.actor);self.critic_target=deepcopy(self.critics)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.critic_opt=torch.optim.Adam(self.critics.parameters(),lr=3e-4)
        self.penalty=penalty;self.rate=1-(1-contraction*dt)**max(1,horizon_steps);self.rate_1=contraction*dt;self.updates=0
        self.behavior_weight=behavior_weight;self.reach_shaping=reach_shaping
        self.progress_shaping=progress_shaping;self.prep_shaping=prep_shaping
        self.demonstrations=None

    def update(self,replay,batch_size):
        obs,nxt,b=replay.sample(batch_size,self.device)
        with torch.no_grad():
            next_action=(self.actor_target(nxt)+(torch.randn_like(b['action'])*.1).clamp(-.2,.2)).clamp(-1,1)
            q1,q2,_,next_d_return=self.critic_target(nxt,next_action)
            # Cost minimization: conservative upper twin estimate, true terminal only.
            q_discount=b.get('q_discount',torch.full_like(b['cost'],.99))
            q_terminal=b.get('q_terminal',b['terminated'])
            cost=shaped_cost(b['cost'],b['reach'],b['next_reach'],q_terminal,self.reach_shaping,q_discount)
            if self.progress_shaping:
                scale=.1+.9*obs['task'][:,4:].sum(-1)
                phi=obs['g'][:,0].abs()/scale
                phi_next=nxt['g'][:,0].abs()/scale
                cost=cost+self.progress_shaping*(q_discount*(1-q_terminal)*phi_next-phi)
            if self.prep_shaping:
                cost=cost+self.prep_shaping*(q_discount*(1-q_terminal)*b['next_prep']-b['prep'])
            target=cost+q_discount*(1-q_terminal)*torch.maximum(q1,q2)
            remaining=obs['g'][:,0]/(.1+.9*obs['task'][:,4:].sum(-1))
            actual_violation=torch.relu(b['d']+self.rate_1*b['energy'])+torch.relu(-b['progress'])+torch.relu(b['progress']-remaining.clamp_min(0))
            d_return_target=actual_violation+.99*(1-b['terminated'])*next_d_return
        q1,q2,d,d_return=self.critics(obs,b['action'])
        q_loss=nn.functional.smooth_l1_loss(q1,target)+nn.functional.smooth_l1_loss(q2,target)
        valid=b['valid_h']
        d_loss=((nn.functional.smooth_l1_loss(100*d[:,0],100*b['d_h'],reduction='none')
                 +nn.functional.smooth_l1_loss(100*d[:,1],100*b['p_h'],reduction='none'))*valid).sum()/valid.sum().clamp_min(1.)
        d_return_loss=nn.functional.smooth_l1_loss(100*d_return,100*d_return_target)
        loss=q_loss+d_loss+d_return_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite critic loss')
        self.critic_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.critics.parameters(),1.);self.critic_opt.step()
        self.updates+=1;actor_loss=None
        if self.updates%2==0:
            for p in self.critics.parameters():p.requires_grad_(False)
            action=self.actor(obs);q1,q2,d,d_return=self.critics(obs,action)
            turn=obs['task'][:,4:].sum(-1);scale=.1+.9*turn
            remaining=obs['g'][:,0]/scale
            violation=torch.relu(d[:,0]+self.rate*b['energy'])+torch.relu(-d[:,1])+torch.relu(d[:,1]-remaining.clamp_min(0))
            objective=torch.maximum(q1,q2).mean()+self.penalty*(violation.mean()+d_return.mean())
            if self.behavior_weight:
                if self.demonstrations is not None:
                    anchor_obs,_,anchor=self.demonstrations.sample(batch_size,self.device,balanced=True)
                    anchor_action=self.actor(anchor_obs)
                    weight=torch.exp(-anchor['cost'].clamp_min(0))
                    regression=(weight*(anchor_action-anchor['action']).square().sum(-1)).sum()/weight.sum().clamp_min(1.)
                else:
                    good_progress=(torch.relu(-b['d'])/(b['energy']+.01)).clamp(0,1)
                    settled_goal=torch.exp(-(remaining/.04).square())
                    weight=torch.exp(-b['cost'].clamp_min(0))*(good_progress+settled_goal)
                    regression=(weight*(action-b['action']).square().sum(-1)).sum()/weight.sum().clamp_min(1.)
                objective=objective+self.behavior_weight*regression
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite actor loss')
            self.actor_opt.zero_grad();objective.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1.);self.actor_opt.step()
            for p in self.critics.parameters():p.requires_grad_(True)
            actor_loss=float(objective.detach())
            with torch.no_grad():
                for network,target_network in ((self.actor,self.actor_target),(self.critics,self.critic_target)):
                    for p,tp in zip(network.parameters(),target_network.parameters()):tp.lerp_(p,.005)
        return {'q_loss':float(q_loss.detach()),'d_loss':float(d_loss.detach()),'d_return_loss':float(d_return_loss.detach()),'actor_loss':actor_loss,'updates':self.updates}

    def bootstrap(self,replay,updates=800,batch_size=128):
        # Supervised initialization from actual exploration actions and feedback.
        # No scripted controller is installed in the inference path.
        final=0.
        for _ in range(updates):
            obs,_,b=replay.sample(batch_size,self.device,balanced=True)
            action=self.actor(obs)
            weight=torch.exp(-b['cost'].clamp_min(0))
            loss=(weight*(action-b['action']).square().sum(-1)).sum()/weight.sum().clamp_min(1.)
            self.actor_opt.zero_grad();loss.backward();self.actor_opt.step()
            final=float(loss.detach())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.demonstrations=Replay(replay.size)
        self.demonstrations.size=replay.size
        self.demonstrations.storage={k:v[:replay.size].clone() for k,v in replay.storage.items()}
        return final

    def checkpoint(self):
        return {'actor':self.actor.state_dict(),'critics':self.critics.state_dict(),
                'actor_target':self.actor_target.state_dict(),'critic_target':self.critic_target.state_dict(),
                'actor_optimizer':self.actor_opt.state_dict(),'critic_optimizer':self.critic_opt.state_dict(),
                'updates':self.updates,
                'demonstrations':None if self.demonstrations is None else {'size':self.demonstrations.size,'storage':self.demonstrations.storage},
                'method':'direct-actor DQ penalty prototype; no action filtering','certified':False}
