"""Signed, action-linear pose-error derivative supervision for RGB features."""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_goal_fixed as joint
from goal_fixed_core import (
    GoalMetric, make_model, energy, policy, pose_error, p0_offset_step,
    calibration_loss, actual_descent_loss,
)
from lac_vs.models import frozen_parameters
from lac_vs.vision import image_tensor, dct_reconstruction


def physical_scales(tensor):
    return tensor.new_tensor([.4,.4,.4,.35])


def derivative_coefficients(offset):
    """dE/dt = a_v^T v for physical body velocity, static target and P0.

    Offsets use the inherited reference frame. Position is in metres, yaw in
    radians, E=||p||^2+(yaw/25deg)^2. No pose/depth is a network input.
    """
    x,y,z,yaw=offset.unbind(-1)
    c,s=torch.cos(yaw),torch.sin(yaw)
    return torch.stack([2*(c*x+s*y),2*(-s*x+c*y),2*z,
                        2*yaw/(math.radians(25)**2)],-1)


class DerivativeCritic(nn.Module):
    def __init__(self, latent=16, hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3*latent,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,4))
        nn.init.normal_(self.net[-1].weight,std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def coefficients(self,z,goal):
        e=(z-goal)/4
        base=torch.cat([goal,goal,torch.zeros_like(goal)],-1)
        return self.net(torch.cat([z,goal,e],-1))-self.net(base)

    def forward(self,z,goal,physical_velocity):
        return (self.coefficients(z,goal)*physical_velocity).sum(-1)


def gradient_norm(grads,prototype):
    values=[g.square().sum() for g in grads if g is not None]
    return torch.stack(values).sum().sqrt() if values else prototype.new_zeros(())


class DerivativeLearner(joint.Learner):
    def __init__(self,cfg,source,device='cuda'):
        self.cfg=cfg; self.p=cfg['goal_fixed']; self.d=cfg['derivative']; self.device=device
        self.ae=make_model(cfg).to(device).eval()
        self.ae.load_state_dict(source['learner']['ae'])
        self.shadow=copy.deepcopy(self.ae).eval()
        self.initial=copy.deepcopy(self.ae).eval().requires_grad_(False)
        self.metric=GoalMetric().to(device)
        self.metric.load_state_dict(source['learner']['metric'])
        self.critics=nn.ModuleList([DerivativeCritic().to(device)])
        self.aeopt=torch.optim.Adam(self.ae.parameters(),lr=self.p['encoder_lr'])
        self.shadowopt=torch.optim.Adam(self.shadow.parameters(),lr=self.p['encoder_lr'])
        self.criticopt=torch.optim.Adam(self.critics.parameters(),lr=self.p['critic_lr'])
        self.mopt=torch.optim.Adam(self.metric.parameters(),lr=self.p['metric_lr'])
        self.anchors=copy.deepcopy(source['learner']['anchors'])
        self.counts=dict(warmup=0,critic_prefit=0,representation=0,metric=0)
        self.version=0; self.target_version=0; self.cache={}; self.tcache={}; self.inputs={}

    def target(self,*_args,**_kwargs):
        raise RuntimeError('Bellman/TD targets are forbidden for the derivative Critic')

    def zero_all(self):
        for module in (self.ae,self.shadow,self.metric,self.critics):
            module.zero_grad(set_to_none=True)

    def candidate_actions(self,b):
        recorded=self.tensor(b['action']); n=len(recorded)
        basis=torch.cat([torch.eye(4,device=self.device),-torch.eye(4,device=self.device)])
        random=torch.rand(n,self.d['random_actions_per_pair'],4,device=self.device)*2-1
        den=torch.maximum(torch.ones_like(random[...,0]),random[...,:3].norm(dim=-1))
        random=torch.cat([random[...,:3]/den[...,None],random[...,3:]],-1)
        return torch.cat([recorded[:,None],basis[None].expand(n,-1,-1),random],1)

    def supervised_loss(self,z,zg,offset,actions):
        true=derivative_coefficients(offset).detach()
        pred=self.critics[0].coefficients(z,zg)
        velocity=actions*physical_scales(actions)
        labels=torch.einsum('bi,bki->bk',true,velocity)
        predictions=torch.einsum('bi,bki->bk',pred,velocity)
        scale=(true*physical_scales(true)).norm(dim=-1).detach()+self.d['label_scale_floor']
        loss=((predictions-labels)/scale[:,None]).square().mean()
        a=pred*physical_scales(pred); target=true*physical_scales(true)
        cosine=(a*target).sum(-1)/(a.norm(dim=-1)*target.norm(dim=-1)+1e-8)
        mask=labels.abs()>1e-4
        sign=float(((predictions[mask]*labels[mask])>0).float().mean()) if mask.any() else 1.
        return loss,dict(derivative_loss=float(loss),derivative_rmse=float((predictions-labels).square().mean().sqrt()),
                         derivative_sign_accuracy=sign,derivative_gradient_cosine=float(cosine.mean()))

    def regularizer(self,model,b,z):
        sm=z.new_zeros(())
        if self.counts['representation']%self.p['smooth_triplet_every']==0:
            tri=np.concatenate([b['smooth_minus'][:2],b['smooth_center'][:2],b['smooth_plus'][:2]])
            zm,zc,zp=self.encode(tri,model).chunk(3)
            sm=(zp-2*zc+zm).square().mean()/self.anchors['curvature']
        wanted=z.new_tensor(self.anchors['latent_std'])
        var=(F.relu(.3*wanted-z.std(0,unbiased=False))/wanted.clamp_min(.05)).square().mean()
        return self.p['smoothness_weight']*sm+self.p['latent_variance_weight']*var+1e-6*z.square().mean(),sm,var

    def model_update(self,model,optimizer,b,actions,with_critic,warmup=False):
        model.zero_grad(set_to_none=True)
        x=image_tensor(self.cfg,np.concatenate([b['rgb'],b['goal_rgb']]),self.device)
        z,y=model(x); n=len(b['rgb'])
        rgb=F.mse_loss(y,x); dct=dct_reconstruction(y,x)
        rec=(rgb+.25*dct)/self.anchors['rec']
        params=list(model.parameters()); enc=list(model.base.encoder.parameters())
        enc_ids={id(p) for p in enc}
        rec_grads=torch.autograd.grad(rec,params,retain_graph=not warmup)
        base_norm=gradient_norm([g for p,g in zip(params,rec_grads) if id(p) in enc_ids],rec)
        dgrads=[None]*len(enc); cgrads=[]; dnorm=rec.new_zeros(()); derivative=rec.new_zeros(()); logs={}
        regular=rec.new_zeros(()); sm=rec.new_zeros(()); var=rec.new_zeros(()); rgrads=[None]*len(enc)
        if not warmup:
            regular,sm,var=self.regularizer(model,b,z)
            if with_critic:
                derivative,logs=self.supervised_loss(z[:n],z[n:],self.tensor(b['offset']),actions)
                critic_params=list(self.critics.parameters())
                grads=torch.autograd.grad(self.d['supervision_weight']*derivative,enc+critic_params,
                                           retain_graph=True,allow_unused=True)
                dgrads=list(grads[:len(enc)]); cgrads=list(grads[len(enc):])
                dnorm=gradient_norm(dgrads,rec)
            rgrads=list(torch.autograd.grad(regular,enc,allow_unused=True))
        rnorm=gradient_norm(rgrads,rec)
        ds=min(1.,self.d['critic_encoder_gradient_cap']*float(base_norm)/max(float(dnorm),1e-12))
        rs=min(1.,self.d['regular_encoder_gradient_cap']*float(base_norm)/max(float(rnorm),1e-12))
        by_id={id(p):g for p,g in zip(params,rec_grads)}
        for p,g in zip(params,rec_grads):
            p.grad=g.detach().clone()
        for p,dg,rg in zip(enc,dgrads,rgrads):
            if dg is not None:p.grad.add_(dg.detach(),alpha=ds)
            if rg is not None:p.grad.add_(rg.detach(),alpha=rs)
        if with_critic:
            for p,g in zip(self.critics.parameters(),cgrads):
                p.grad=None if g is None else g.detach()
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        cnorm=torch.nn.utils.clip_grad_norm_(self.critics.parameters(),1.) if with_critic else rec.new_zeros(())
        if not torch.isfinite(norm) or not torch.isfinite(rec+regular+derivative) or not torch.isfinite(cnorm):
            raise FloatingPointError('Nonfinite AE/derivative update')
        optimizer.step()
        if with_critic:self.criticopt.step()
        return dict(total=float(rec+regular+derivative),reconstruction=float(rgb),dct=float(dct),
            gradient_norm=float(norm),ae_encoder_grad=float(base_norm),critic_encoder_grad=float(dnorm),
            regular_encoder_grad=float(rnorm),critic_encoder_scale=ds,regular_encoder_scale=rs,
            critic_gradient_norm=float(cnorm),curvature=float(sm),variance=float(var),**logs)

    def representation(self,b,warmup=False):
        self.zero_all(); actions=None if warmup else self.candidate_actions(b)
        main=self.model_update(self.ae,self.aeopt,b,actions,with_critic=not warmup,warmup=warmup)
        self.shadow.zero_grad(set_to_none=True)
        shadow=self.model_update(self.shadow,self.shadowopt,b,actions,with_critic=False,warmup=warmup)
        if any(p.grad is not None for p in self.metric.parameters()):
            raise RuntimeError('Representation update leaked into M')
        self.counts['warmup' if warmup else 'representation']+=1
        self.version+=1; self.clear_cache()
        return dict(**main,shadow_reconstruction=shadow['reconstruction'],shadow_gradient_norm=shadow['gradient_norm'],
                    metric_gradient_norm=0.,encoder_gradient_cap=self.d['critic_encoder_gradient_cap'])

    def critic_prefit(self,b):
        self.zero_all()
        with torch.no_grad():
            z,zg=self.encode(np.concatenate([b['rgb'],b['goal_rgb']])).chunk(2)
        loss,logs=self.supervised_loss(z,zg,self.tensor(b['offset']),self.candidate_actions(b))
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(self.critics.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite derivative prefit')
        if any(p.grad is not None for p in self.ae.parameters()):raise RuntimeError('Prefit changed encoder gradients')
        self.criticopt.step(); self.counts['critic_prefit']+=1
        return dict(total=float(loss),gradient_norm=float(norm),**logs)

    def metric_update(self,b):
        self.zero_all()
        with torch.no_grad():z,zn=self.encode(b['rgb']),self.encode(b['next_rgb'])
        zg,j=self.goal_features(b,perturb=True)
        _,m=self.metric(zg); e,en=(z-zg)/4,(zn-zg)/4
        offset=self.tensor(b['offset']); E=pose_error(offset); En=pose_error(self.tensor(b['next_offset']))
        cal=.5*(calibration_loss(energy(e,m),E,self.cfg)+calibration_loss(energy(en,m),En,self.cfg))
        dec=actual_descent_loss(e,en,m,E,self.cfg)
        u,h=policy(e,j,m); nxt=p0_offset_step(offset,u); Eplus=pose_error(nxt)
        floor=self.p['objective_floor']; rate=self.p['decay_per_second']*.1
        physical=(F.relu(Eplus-E+rate*E)/(E+floor)).square().mean()
        p=offset[:,:3].square().sum(-1); pn=nxt[:,:3].square().sum(-1)
        position=(F.relu(pn-p+rate*p)/(p+floor)).square().mean()
        scale=offset.new_tensor([1.,1.,1.,math.radians(25)])
        r=offset/scale; dr=(nxt-offset)/scale
        cosine=(r*dr).sum(-1)/(r.norm(dim=-1)*dr.norm(dim=-1)+1e-6)
        direction=F.relu(cosine+.15).square().mean()
        smooth=(u-self.tensor(b['previous_action'])).square().mean()
        regular=(m-torch.eye(16,device=self.device)).square().mean()
        true_coeff=derivative_coefficients(offset).detach()
        denom=(true_coeff*physical_scales(true_coeff)).norm(dim=-1)+self.d['label_scale_floor']
        with frozen_parameters(self.critics):
            pred=self.critics[0](z,zg,u*physical_scales(u))
            cost=(pred/denom).mean()
            loss=(self.p['metric_cost_weight']*cost+self.p['metric_calibration_weight']*cal+
                  self.p['metric_real_descent_weight']*dec+self.p['metric_position_descent_weight']*(physical+position)+
                  self.p['metric_direction_weight']*direction+self.p['action_smoothness_weight']*smooth+
                  self.p['metric_condition_weight']*regular)
            loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.metric.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite M update')
        if any(p.grad is not None for p in self.ae.parameters()) or any(p.grad is not None for p in self.critics.parameters()):
            raise RuntimeError('M update leaked into encoder or derivative Critic')
        self.mopt.step(); self.counts['metric']+=1
        true_rate=(true_coeff*u.detach()*physical_scales(u)).sum(-1)
        return dict(total=float(loss),cost=float(cost),calibration=float(cal),real_descent=float(dec),
            physical_descent=float(physical),position_descent=float(position),direction=float(direction),
            action_smoothness=float(smooth),gradient_norm=float(norm),encoder_leak=0.,critic_leak=0.,
            predicted_derivative=float(pred.mean()),true_derivative=float(true_rate.mean()),
            derivative_error=float((pred.detach()-true_rate).square().mean().sqrt()),
            predicted_physical_delta=float((Eplus-E).mean()),V=float(energy(e,m).mean()),
            M_eigen_min=float(torch.linalg.eigvalsh(m.detach()).min()),
            M_eigen_max=float(torch.linalg.eigvalsh(m.detach()).max()),H_condition=float(torch.linalg.cond(h).max()))

    def payload(self):
        names=('ae','shadow','metric','critics','aeopt','shadowopt','criticopt','mopt')
        return {name:getattr(self,name).state_dict() for name in names}|dict(
            counts=self.counts,version=self.version,anchors=self.anchors)

    def restore(self,payload):
        for name in ('ae','shadow','metric','critics','aeopt','shadowopt','criticopt','mopt'):
            getattr(self,name).load_state_dict(payload[name])
        self.counts=copy.deepcopy(payload['counts']); self.version=payload['version']; self.anchors=payload['anchors']
        self.clear_cache()
