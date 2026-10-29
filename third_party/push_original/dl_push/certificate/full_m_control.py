"""Full-Cholesky certificate actor and numeric-only control adapter.

The actor is deterministic M(D,C). PPO explores the induced distribution over
real (mode,speed) candidates. Hard evaluation uses mode persistence and an
independent attitude recovery certificate; its actions are NOT PPO samples.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .numeric_env import NumericPushVectorEnv, NumericConfig, DEFAULT_LIBRARY
from .library import ModeLibrary

CODEC = 'target_frame_sqrt2_sin_half_v1'


@dataclass(frozen=True)
class FullMConfig:
    codec: str = CODEC
    length_scale: float = .1
    dt: float = .2
    u_max: float = .6
    speed_min_search: float = .0001
    train_speed_points: int = 17
    eval_speed_points: int = 65
    workspace_m: float = .5
    max_steps: int = 100
    hold_steps: int = 5
    position_tol: float = .005
    angle_tol: float = math.radians(2)
    coverage_tol: float = .95
    angle_chart_limit: float = math.pi - .05
    eigen_floor: float = .05
    cholesky_floor: float = .001
    hidden: int = 64
    residual_tol: float = 1e-6
    training_margin: float = .001
    residual_temperature: float = .05
    progress_temperature: float = .1
    stay_bias: float = 2.
    input_cost: float = .02
    error_weight: float = 1.
    switch_weight: float = 1.
    no_action_weight: float = .5
    overlap_weight: float = 5.
    success_bonus: float = 200.
    failure_penalty: float = 600.
    error_cap: float = 20.
    rl_reward_scale: float = .01

    def __post_init__(self):
        if self.codec != CODEC:
            raise ValueError('Codec mismatch; old raw-yaw/sin-squared checkpoints are incompatible')
        positive=[self.length_scale,self.dt,self.u_max,self.speed_min_search,self.workspace_m,
                  self.max_steps,self.hold_steps,self.position_tol,self.angle_tol,self.hidden,
                  self.train_speed_points,self.eval_speed_points,self.residual_temperature,
                  self.progress_temperature,self.rl_reward_scale]
        if not np.isfinite(positive).all() or min(positive)<=0:raise ValueError('Invalid scales')
        if not 0<self.eigen_floor<1 or not self.speed_min_search<self.u_max:raise ValueError('Invalid matrix/speed bounds')


def coordinates(e):
    """e=[target-frame position/.1, continuous delta]; no silent yaw wrapping."""
    return torch.cat([e[..., :2], math.sqrt(2)*torch.sin(e[..., 2:3]/2)], -1)


def torch_step(e,c,u,h,ell=.1):
    s=u*h;a=c[...,2]*s;heading=e[...,2]+a/2
    amp=s*torch.sinc(a/(2*math.pi))/ell
    return e+torch.stack([amp*(torch.cos(heading)*c[...,0]-torch.sin(heading)*c[...,1]),
        amp*(torch.sin(heading)*c[...,0]+torch.cos(heading)*c[...,1]),a],-1)


def cholesky_metric(raw, floor=.05, diag_floor=.001):
    d=F.softplus(raw[..., :3])+diag_floor;z=torch.zeros_like(d[...,0])
    L=torch.stack([d[...,0],z,z,raw[...,3],d[...,1],z,raw[...,4],raw[...,5],d[...,2]],-1).reshape(*raw.shape[:-1],3,3)
    A=L@L.transpose(-1,-2)
    M=floor*torch.eye(3,device=raw.device,dtype=raw.dtype)+(3-3*floor)*A/A.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]
    return L,M


class ModeEncoder(nn.Module):
    def __init__(self,h=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(3,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU())
    def forward(self,C,valid,ell=.1):
        if C.ndim!=3 or C.shape[-1]!=3 or valid.shape!=C.shape[:2]:raise ValueError('C[B,N,3], valid[B,N] required')
        if not bool(valid.any(-1).all()):raise ValueError('Empty mode set')
        values=torch.where(valid[...,None],C*C.new_tensor([1.,1.,ell]),0.)
        H=self.net(values)
        return torch.cat([(H*valid[...,None]).sum(1)/valid.sum(1,keepdim=True),
            H.masked_fill(~valid[...,None],-torch.inf).amax(1)],-1)


class FullMActor(nn.Module):
    def __init__(self,cfg=None):
        super().__init__();self.cfg=cfg or FullMConfig();h=self.cfg.hidden
        self.encoder=ModeEncoder(h)
        self.state=nn.Sequential(nn.Linear(3,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(3*h,2*h),nn.SiLU(),nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,6))
        # Small, nonzero last-layer initialization preserves full M and its
        # input gradients from the first update (no Q=I preflight).
        nn.init.normal_(self.head[-1].weight,std=.01);nn.init.zeros_(self.head[-1].bias)
    def conditioned(self,D,condition):
        if D.ndim==3:condition=condition[:,None].expand(-1,D.shape[1],-1)
        raw=self.head(torch.cat([self.state(D),condition],-1))
        L,M=cholesky_metric(raw,self.cfg.eigen_floor,self.cfg.cholesky_floor)
        return L,M
    def forward(self,D,C,valid):
        return self.conditioned(D,self.encoder(C,valid,self.cfg.length_scale))
    def value(self,D,condition):
        _,M=self.conditioned(D,condition)
        return torch.einsum('...i,...ij,...j->...',D,M,D)


class ModeBank:
    """Immutable numeric libraries and geometry; no simulator import."""
    def __init__(self,root=DEFAULT_LIBRARY,device='cpu',dtype=torch.float32):
        self.root=Path(root).resolve();self.names=sorted(p.parent.name for p in self.root.glob('*/report.json'))
        if len(self.names)!=32:raise ValueError(f'Expected 32 final shape libraries, got {len(self.names)}')
        libs=[ModeLibrary.load(self.root/n/'library') for n in self.names]
        if any(x.count!=12 or not x.valid.any() for x in libs):raise ValueError('Invalid 12-mode bank')
        self.fingerprints={x.shape:x.fingerprint for x in libs}
        self.device=torch.device(device);self.dtype=dtype
        self.C=torch.as_tensor(np.stack([x.c.T for x in libs]),device=device,dtype=dtype)
        self.valid=torch.as_tensor(np.stack([x.valid for x in libs]),device=device)
        polygons=[json.loads((self.root/n/'polygon.json').read_text()) for n in self.names]
        rings=[np.asarray(p['coordinates'][0][:-1]) for p in polygons]
        width=max(map(len,rings));verts=np.zeros((32,width,2));mask=np.zeros((32,width),bool)
        for i,r in enumerate(rings):verts[i,:len(r)]=r;mask[i,:len(r)]=True
        self.vertices=torch.as_tensor(verts,device=device,dtype=dtype)
        self.vertex_mask=torch.as_tensor(mask,device=device)

    def batch(self,obs,recovery=None,hold=None,hold_steps=5):
        n=len(obs['pose']);dev=self.device;dtype=self.dtype
        return dict(e=torch.as_tensor(obs['error'],device=dev,dtype=dtype),
            q=torch.as_tensor(obs['pose'],device=dev,dtype=dtype),
            target=torch.as_tensor(obs['target'],device=dev,dtype=dtype),
            shape=torch.as_tensor(obs['shape_id'],device=dev,dtype=torch.long),
            mode=torch.as_tensor(obs['current_mode']-1,device=dev,dtype=torch.long),
            remaining=torch.as_tensor(obs['remaining_steps'],device=dev,dtype=dtype),
            recovery=torch.as_tensor(np.zeros(n) if recovery is None else recovery,device=dev,dtype=dtype),
            hold=torch.as_tensor(np.zeros(n) if hold is None else hold,device=dev,dtype=dtype)/hold_steps)


def candidate_geometry(e_next,target,ids,bank,cfg):
    """Endpoint geometry only. The numerical plant has no contact/wall model."""
    a=target[:,2:3];x=e_next[...,0]*cfg.length_scale;y=e_next[...,1]*cfg.length_scale
    px=target[:,0:1]+torch.cos(a)*x-torch.sin(a)*y
    py=target[:,1:2]+torch.sin(a)*x+torch.cos(a)*y
    yaw=a+e_next[...,2];v=bank.vertices[ids];mask=bank.vertex_mask[ids][:,None,:]
    wx=px[...,None]+torch.cos(yaw)[...,None]*v[:,None,:,0]-torch.sin(yaw)[...,None]*v[:,None,:,1]
    wy=py[...,None]+torch.sin(yaw)[...,None]*v[:,None,:,0]+torch.cos(yaw)[...,None]*v[:,None,:,1]
    inside=(wx.masked_fill(~mask,torch.inf).amin(-1)>=0)&(wy.masked_fill(~mask,torch.inf).amin(-1)>=0)
    inside&=(wx.masked_fill(~mask,-torch.inf).amax(-1)<=cfg.workspace_m)&(wy.masked_fill(~mask,-torch.inf).amax(-1)<=cfg.workspace_m)
    return inside&(e_next[...,2].abs()<cfg.angle_chart_limit)


def candidates(actor,data,bank,cfg,training=False,speed_points=None):
    """Complete dM/dD gradient. Return immutable action ordering mode*K+k."""
    K=speed_points or cfg.train_speed_points;e=data['e'];ids=data['shape'];B=len(e)
    speeds=torch.logspace(math.log10(cfg.speed_min_search),math.log10(cfg.u_max),K,device=e.device,dtype=e.dtype)
    C=bank.C[ids];valid=bank.valid[ids]
    # Encoded conditions are reused inside THIS call, never cached across SGD.
    condition=actor.encoder(bank.C,bank.valid,cfg.length_scale)[ids]
    with torch.enable_grad():
        D=coordinates(e).detach().requires_grad_(True)
        L,M=actor.conditioned(D,condition);V=torch.einsum('bi,bij,bj->b',D,M,D)
        gradient,=torch.autograd.grad(V.sum(),D,create_graph=training,retain_graph=training)
    a=e[:,2,None];b=torch.stack([(torch.cos(a)*C[...,0]-torch.sin(a)*C[...,1])/cfg.length_scale,
        (torch.sin(a)*C[...,0]+torch.cos(a)*C[...,1])/cfg.length_scale,
        torch.cos(a/2)*C[...,2]/math.sqrt(2)],-1)
    beta=torch.einsum('bi,bni->bn',gradient,b)
    ep=torch_step(e[:,None,None,:],C[:,:,None,:],speeds[None,None,:],cfg.dt,cfg.length_scale).reshape(B,12*K,3)
    Vp=actor.value(coordinates(ep),condition)
    norm=(D.detach().square().sum(-1)+1e-6)[:,None]
    flow=(cfg.dt*beta[:,:,None]*speeds).reshape(B,-1)/norm
    endpoint=(Vp-V[:,None])/norm
    both=torch.maximum(flow,endpoint)
    physical=valid[:,:,None].expand(-1,-1,K).reshape(B,-1)&candidate_geometry(ep,data['target'],ids,bank,cfg)
    feasible=physical&(flow< -cfg.residual_tol)&(endpoint< -cfg.residual_tol)
    mode_ids=torch.arange(12,device=e.device).repeat_interleave(K)
    input_values=speeds.repeat(12)
    cost=cfg.input_cost*(input_values/cfg.u_max).square()
    # A sigmoid gate alone saturates for all descending actions, making the
    # many tiny speeds nearly as likely as useful finite progress. This extra
    # endpoint preference aligns soft exploration with the hard speed ranking.
    logits=F.logsigmoid(-both/cfg.residual_temperature)-endpoint/cfg.progress_temperature-cost[None,:]-cfg.stay_bias*(mode_ids[None,:]!=data['mode'][:,None])
    logits=logits.masked_fill(~physical,-1e9)
    return dict(M=M,L=L,V=V,V_next=Vp,gradient=gradient,beta=beta,e_next=ep,
        flow=flow,endpoint=endpoint,both=both,physical=physical,feasible=feasible,
        logits=logits,modes=mode_ids,speeds=input_values,condition=condition)


def smoothness_loss(actor,data,condition,noise=None):
    e=data['e'];noise=.01*torch.randn_like(e) if noise is None else noise
    D=coordinates(e);Dn=coordinates(e+noise)
    _,M=actor.conditioned(D,condition);_,Mn=actor.conditioned(Dn,condition)
    return ((Mn-M).square().sum((-2,-1))/(noise.square().sum(-1)+1e-6)).mean()


def hard_choice(cand,data,cfg,recovery,at_goal):
    """No phase masks on M. Independent V_R only after normal-grid failure."""
    feasible=cand['feasible'];has=feasible.any(-1);angle=data['e'][:,2]
    recovery=recovery & (angle.abs()>cfg.angle_tol)
    continuing_recovery=recovery.clone()
    recovery=recovery | (~has & (angle.abs()>cfg.angle_tol) & ~at_goal)
    modes=cand['modes'];us=cand['speeds'];B=len(angle)
    current=modes[None,:]==data['mode'][:,None]
    normal_pool=feasible & torch.where((feasible&current).any(-1)[:,None],current,torch.ones_like(current))
    normal_cost=cand['V_next']/(cand['V'].detach()[:,None]+1e-6)+cfg.input_cost*(us[None,:]/cfg.u_max).square()
    normal_index=normal_cost.masked_fill(~normal_pool,torch.inf).argmin(-1)
    next_angle=cand['e_next'][...,2]
    omega=(next_angle-angle[:,None])/cfg.dt
    recover_pool=cand['physical']&(angle[:,None]*omega<0)&(next_angle.abs()<angle.abs()[:,None]-1e-8)
    recover_pool&=((next_angle*angle[:,None]>=0)|(next_angle.abs()<=cfg.angle_tol))
    keep_recovery=continuing_recovery & (recover_pool&current).any(-1)
    recover_pool&=torch.where(keep_recovery[:,None],current,torch.ones_like(current))
    recover_score=-angle.sign()[:,None]*omega
    recover_index=recover_score.masked_fill(~recover_pool,-torch.inf).argmax(-1)
    index=torch.where(recovery,recover_index,normal_index)
    fail=torch.where(recovery,~recover_pool.any(-1),~has)&~at_goal
    u=us[index];u=torch.where(at_goal|fail,torch.zeros_like(u),u)
    return modes[index],u,recovery&~at_goal,fail


class FullMControlEnv:
    """Pure integrator plus task bookkeeping. No simulator or pusher state."""
    def __init__(self,bank,cfg=None,shape_names=None,num_envs=32):
        self.cfg=cfg or FullMConfig();self.bank=bank;c=self.cfg
        self.env=NumericPushVectorEnv(num_envs,shape_names,bank.root,NumericConfig(
            dt=c.dt,u_max=c.u_max,max_steps=c.max_steps,workspace_m=c.workspace_m,
            length_scale=c.length_scale,success_mode='none',terminate_on_outside=True))
        self.recovery=np.zeros(num_envs,bool);self.hold=np.zeros(num_envs,int)
        self.no_action_steps=np.zeros(num_envs,int);self.recovery_steps=np.zeros(num_envs,int)
        self.cost_error=np.zeros(num_envs);self.max_position=np.zeros(num_envs)

    def reset(self,seed=None,options=None):
        self.obs,self.info=self.env.reset(seed=seed,options=options)
        for name in ('recovery','hold','no_action_steps','recovery_steps','cost_error','max_position'):getattr(self,name)[:]=0
        return self.batch()

    def batch(self):return self.bank.batch(self.obs,self.recovery,self.hold,self.cfg.hold_steps)

    def at_goal(self):
        e=self.obs['error'];c=self.cfg
        return (np.linalg.norm(e[:,:2],axis=1)*c.length_scale<=c.position_tol)&(np.abs(e[:,2])<=c.angle_tol)&(self.info['coverage']>=c.coverage_tol)

    def advance(self,mode,u,no_action,recovery=None,forced_failure=None):
        c=self.cfg;active=~self.env.done.copy();old_cov=self.info['coverage'].copy();old_switch=self.env.switches.copy()
        if recovery is not None:self.recovery=np.asarray(recovery,bool)
        modes=np.asarray(mode,dtype=np.int64);u=np.asarray(u,dtype=float)
        bad=np.asarray(no_action,bool)&(np.abs(self.obs['error'][:,2])>c.angle_tol)&active
        self.no_action_steps+=bad;self.recovery_steps+=self.recovery&active
        self.obs,_,_,_,self.info=self.env.step({'mode':modes,'u':u[:,None]})
        e=self.obs['error'];E=np.minimum((e[:,:2]**2).sum(-1)+(1-np.cos(e[:,2])),c.error_cap)
        self.cost_error+=active*c.dt*E
        self.max_position=np.maximum(self.max_position,np.linalg.norm(e[:,:2],axis=1)*c.length_scale)
        self.hold=np.where(active & self.at_goal() & (u==0),self.hold+1,np.where(active,0,self.hold))
        success=active&(self.hold>=c.hold_steps)&~self.env.outside
        if forced_failure is not None:
            f=np.asarray(forced_failure,bool)&active&~success;self.env.term[f]=True;self.env.reason[f]='GRID_NO_ACTION'
        chart=(np.abs(e[:,2])>=c.angle_chart_limit)&active&~success
        self.env.term[chart]=True;self.env.reason[chart]='ANGLE_CHART_EXIT'
        self.env.term[success]=True;self.env.trunc[success]=False;self.env.reason[success]='SUCCESS'
        self.env.done=self.env.term|self.env.trunc
        new_done=active&self.env.done;failed=new_done&~success
        switched=self.env.switches-old_switch
        reward=active*(-c.error_weight*c.dt*E-c.switch_weight*switched-c.no_action_weight*bad
            +c.overlap_weight*(self.info['coverage']-old_cov)-c.input_cost*c.dt*(u/c.u_max)**2)
        reward+=c.success_bonus*success-c.failure_penalty*failed
        extrapolated=self.info['extrapolated'].copy()
        self.obs=self.env._obs();self.info=self.env._info(extrapolated)
        self.info.update(no_action_steps=self.no_action_steps.copy(),recovery_steps=self.recovery_steps.copy(),
                         cumulative_error=self.cost_error.copy(),max_position_error_m=self.max_position.copy())
        return self.batch(),reward,self.env.done.copy(),active
