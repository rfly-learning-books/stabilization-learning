"""The original full-Cholesky metric, time-aware soft PPO policy and hard gate."""
import numpy as np
import torch
from torch import nn
import so101_multishape as m
from .timing import TimeConfig

def load_actor(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=True)
    actor=m.base.FullMActor(m.base.FullMConfig(**ck['cfg'])).to(device);actor.load_state_dict(ck['actor'])
    return actor,ck

class CandidateModel:
    def __init__(self,bank,device='cpu'):
        self.bank=bank;self.cfg=bank.config;self.device=torch.device(device)
        self.C=torch.tensor(np.array([bank.specs[n]['C'] for n in bank.names]),dtype=torch.float32,device=device)
        self.valid=torch.tensor(np.array([bank.specs[n]['valid'] for n in bank.names]),device=device)
        self.C_encoded=self.C.clone();self.C_encoded[:,:,2]*=self.cfg.encoder_omega_multiplier
    def batch(self,observations):
        keys=['shape','q','pusher','mode','contact','remaining','tau','reachable','reposition'];out={}
        for k in keys:
            dtype=torch.long if k in ['shape','mode'] else torch.bool if k in ['contact','reachable','reposition'] else torch.float32
            out[k]=torch.tensor(np.array([o[k] for o in observations]),dtype=dtype,device=self.device)
        return out
    def coordinates(self,q):
        e=torch.cat([q[...,:2]/self.cfg.length_scale,q[...,2:3]],-1)
        return m.base.coordinates(e)
    def candidates(self,actor,data,training=False):
        cfg=self.cfg;q=data['q'];C=self.C[data['shape']];K=len(cfg.travels)
        travel=torch.tensor(cfg.travels,device=q.device,dtype=q.dtype)
        a=q[:,2,None];world=torch.stack([torch.cos(a)*C[:,:,0]-torch.sin(a)*C[:,:,1],torch.sin(a)*C[:,:,0]+torch.cos(a)*C[:,:,1],C[:,:,2]],-1)
        pred=(q[:,None,None,:]+world[:,:,None,:]*travel[None,None,:,None]).reshape(len(q),12*K,3)
        with torch.enable_grad():
            condition=actor.encoder(self.C_encoded[data['shape']],self.valid[data['shape']],ell=cfg.length_scale)
            D=self.coordinates(q).detach().requires_grad_(True)
            L,M=actor.conditioned(D,condition);V=torch.einsum('bi,bij,bj->b',D,M,D)
            gradient,=torch.autograd.grad(V.sum(),D,create_graph=training,retain_graph=training)
            Vp=actor.value(self.coordinates(pred),condition)
            tangent=torch.cat([world[:,:,:2]/cfg.length_scale,(torch.cos(a/2)*C[:,:,2]/np.sqrt(2))[:,:,None]],-1)
            norm=D.detach().square().sum(-1).clamp_min(1e-6)
            flow=((gradient[:,None,:]*tangent).sum(-1)[:,:,None]*travel).reshape(len(q),-1)/norm[:,None]
            endpoint=(Vp-V[:,None])/norm[:,None];both=torch.maximum(flow,endpoint)
            duration=(data['tau'][:,:,None]+data['reposition'][:,:,None]*(cfg.settle_before_transfer+cfg.engage_seconds)+cfg.settle_after_push+travel[None,None,:]/cfg.push_speed).reshape(len(q),-1)
            physical=data['reachable'][:,:,None].expand(-1,-1,K).reshape(len(q),-1)&(torch.linalg.vector_norm(pred[:,:,:2],dim=-1)<.06)&(pred[:,:,2].abs()<np.pi-.05)
            feasible=physical&(flow< -cfg.residual_tolerance)&(endpoint< -cfg.residual_tolerance)
            logits=torch.nn.functional.logsigmoid(-both/cfg.residual_temperature)-endpoint/cfg.progress_temperature-cfg.logit_time_weight*duration/cfg.time_reference
            logits=logits.masked_fill(~physical,-1e9)
        return dict(logits=logits,V=V,V_next=Vp,D=D,condition=condition,M=M,both=both,physical=physical,feasible=feasible,pred=pred,duration=duration)

class TimeCritic(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        from dl_push.certificate.full_m_control import ModeEncoder
        self.encoder=ModeEncoder(hidden)
        self.net=nn.Sequential(nn.Linear(2*hidden+44,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
    def forward(self,data,model):
        cfg=model.cfg;C=model.C_encoded[data['shape']];valid=model.valid[data['shape']]
        state=torch.cat([model.coordinates(data['q']),data['pusher']/cfg.length_scale,torch.nn.functional.one_hot(data['mode']+1,13).float(),data['tau'].clamp(0,cfg.max_seconds)/cfg.max_seconds,data['reachable'].float(),data['remaining'][:,None]/cfg.max_seconds,data['contact'][:,None].float()],-1)
        return self.net(torch.cat([self.encoder(C,valid,ell=cfg.length_scale),state],-1)).squeeze(-1)

class Selector(m.c.Policy):
    """A delegates to the unchanged selector; B/C use the same time-aware gate."""
    def __init__(self,bank,shape,actor,variant):
        self.bank=bank;self.name=shape;self.variant=variant;self.actor=actor.eval();self.metric='finetuned'
        self.model=CandidateModel(bank,next(actor.parameters()).device);self.C=bank.specs[shape]['C']
        self.Ct=torch.tensor(self.C[None],dtype=torch.float32,device=self.model.device);self.Ct[:,:,2]*=bank.config.encoder_omega_multiplier
        self.valid=torch.tensor([bank.specs[shape]['valid'].tolist()],device=self.model.device)
        with torch.no_grad():self.cond=self.actor.encoder(self.Ct,self.valid,ell=bank.config.length_scale).detach()
    def value(self,q):
        q=torch.tensor(np.atleast_2d(q),dtype=torch.float32,device=self.model.device)
        with torch.no_grad():return self.actor.value(self.model.coordinates(q),self.cond.expand(len(q),-1)).cpu().numpy()
    def choose(self,q,previous,hold,pusher):
        if self.variant=='A':
            if self.model.device.type!='cpu':raise ValueError('Unchanged A selector requires CPU evaluation')
            return super().choose(q,previous,hold,pusher)
        previous=-1 if previous is None else previous;contact=self.bank.contact_usable(self.name,q,previous,pusher)
        tau,reachable,reposition=self.bank.row(self.name,previous,contact)
        obs=dict(shape=self.bank.names.index(self.name),q=q,pusher=pusher,mode=previous,contact=contact,remaining=self.bank.config.max_seconds,tau=tau,reachable=reachable,reposition=reposition)
        cand=self.model.candidates(self.actor,self.model.batch([obs]))
        V=cand['V'].detach();score=cand['V_next'].detach()/(V[:,None]+1e-6)+self.bank.config.selection_time_weight*cand['duration']/self.bank.config.time_reference
        feasible=cand['feasible'];recovery=False
        if not bool(feasible.any()):
            recovery=True;angles=cand['pred'][:,:,2].abs();feasible=cand['physical']&(angles<abs(q[2])-1e-8)
            score=angles/max(abs(q[2]),1e-6)+self.bank.config.selection_time_weight*cand['duration']/self.bank.config.time_reference
        if not bool(feasible.any()):raise RuntimeError('no_descending_mode')
        index=int(score.masked_fill(~feasible,torch.inf).argmin());mode,k=divmod(index,len(self.bank.config.travels))
        return mode,float(self.bank.config.travels[k]),float(V[0]),recovery
