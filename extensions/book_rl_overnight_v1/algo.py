"""Off-policy entropy-regularized SAC; Q always evaluates proposal actions."""
from __future__ import annotations
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F


def network(inputs,outputs,width):
    return nn.Sequential(nn.Linear(inputs,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,outputs))


class Actor(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=network(2,4,width)

    def sample(self,state,deterministic=False):
        mean,logstd=self.net(state).chunk(2,-1)
        logstd=logstd.clamp(-5,1)
        if deterministic:
            return mean.tanh(),None
        normal=torch.distributions.Normal(mean,logstd.exp())
        latent=normal.rsample();action=latent.tanh()
        logp=(normal.log_prob(latent)-2*(math.log(2)-latent-F.softplus(-2*latent))).sum(-1)
        return action,logp


class Replay:
    def __init__(self,capacity,device):
        self.capacity=capacity;self.size=0;self.cursor=0
        self.data={k:torch.empty((capacity,*shape),device=device) for k,shape in
            {"state":(2,),"proposal":(2,),"executed":(2,),"next_state":(2,),
             "reward":(),"base_reward":(),"penalty":(),"failure":()}.items()}

    def add(self,transition):
        n=len(transition["state"])
        ids=(torch.arange(n,device=self.data["state"].device)+self.cursor)%self.capacity
        for key in self.data:self.data[key][ids]=transition[key]
        self.cursor=(self.cursor+n)%self.capacity;self.size=min(self.capacity,self.size+n)

    def sample(self,size):
        ids=torch.randint(self.size,(size,),device=self.data["state"].device)
        return {key:values[ids] for key,values in self.data.items()}


class SAC:
    def __init__(self,cfg,device):
        self.cfg=cfg;self.device=device
        self.actor=Actor(cfg["hidden"]).to(device)
        self.q1=network(4,1,cfg["hidden"]).to(device);self.q2=network(4,1,cfg["hidden"]).to(device)
        self.t1=deepcopy(self.q1).requires_grad_(False);self.t2=deepcopy(self.q2).requires_grad_(False)
        self.log_alpha=torch.tensor(math.log(cfg["initial_alpha"]),device=device,requires_grad=True)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=cfg["lr"])
        self.q_opt=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=cfg["lr"])
        self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=cfg["lr"])
        self.updates=0

    def action(self,state,deterministic=False):
        with torch.no_grad():return self.actor.sample(state,deterministic)[0]

    def update(self,batch):
        alpha=self.log_alpha.exp().detach()
        with torch.no_grad():
            proposal,logp=self.actor.sample(batch["next_state"])
            pair=torch.cat([batch["next_state"],proposal],-1)
            future=torch.minimum(self.t1(pair),self.t2(pair)).squeeze(-1)-alpha*logp
            target=batch["reward"]+self.cfg["gamma"]*(1-batch["failure"])*future
        pair=torch.cat([batch["state"],batch["proposal"]],-1)
        q1,q2=self.q1(pair).squeeze(-1),self.q2(pair).squeeze(-1)
        qloss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        self.q_opt.zero_grad(set_to_none=True);qloss.backward()
        qnorm=torch.nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()),10.)
        self.q_opt.step()
        for q in (self.q1,self.q2):q.requires_grad_(False)
        proposal,logp=self.actor.sample(batch["state"])
        pair=torch.cat([batch["state"],proposal],-1)
        loss=(alpha*logp-torch.minimum(self.q1(pair),self.q2(pair)).squeeze(-1)).mean()
        self.actor_opt.zero_grad(set_to_none=True);loss.backward()
        pnorm=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),10.);self.actor_opt.step()
        for q in (self.q1,self.q2):q.requires_grad_(True)
        aloss=-(self.log_alpha*(logp.detach()+self.cfg["target_entropy"])).mean()
        self.alpha_opt.zero_grad(set_to_none=True);aloss.backward();self.alpha_opt.step()
        with torch.no_grad():
            for target_q,q in ((self.t1,self.q1),(self.t2,self.q2)):
                for a,b in zip(target_q.parameters(),q.parameters()):a.lerp_(b,self.cfg["tau"])
        self.updates+=1
        return dict(update=self.updates,q_loss=float(qloss),actor_loss=float(loss),alpha_loss=float(aloss),
            alpha=float(self.log_alpha.exp()),q_gradient=float(qnorm),actor_gradient=float(pnorm),
            mean_target=float(target.mean()),mean_q=float(torch.minimum(q1,q2).mean()))

    def payload(self):
        return dict(actor=self.actor.state_dict(),q1=self.q1.state_dict(),q2=self.q2.state_dict(),
            target_q1=self.t1.state_dict(),target_q2=self.t2.state_dict(),log_alpha=self.log_alpha.detach(),
            actor_optimizer=self.actor_opt.state_dict(),critic_optimizer=self.q_opt.state_dict(),
            alpha_optimizer=self.alpha_opt.state_dict(),updates=self.updates)

    def load_policy(self,payload):
        self.actor.load_state_dict(payload["actor"],strict=True)
