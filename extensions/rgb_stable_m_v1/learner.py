"""Fresh AE/derivative-Critic/M learning with checked M proposals."""
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from stabilization_learning.rgb_stable.objectives import (
    progress_risk, priority_gradient, trust_statistics, group_risks)
from extensions.reproducible_training_v4.deterministic_rgb import install_deterministic_blur


class ComponentCritic(nn.Module):
    """Action-linear d(rho_position,rho_yaw)/dt, zero anchored at the goal."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(48,128), nn.SiLU(), nn.Linear(128,128),
                                 nn.SiLU(), nn.Linear(128,4))
        nn.init.normal_(self.net[-1].weight,std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def coefficients(self,z,goal):
        base = torch.cat((goal,goal,torch.zeros_like(goal)),-1)
        raw = self.net(torch.cat((z,goal,(z-goal)/4),-1))-self.net(base)
        zero = torch.zeros_like(raw)
        p = torch.cat((raw[:,:3],zero[:,3:]),-1)/.02**2
        yaw = torch.cat((zero[:,:3],raw[:,3:]),-1)/math.radians(1.)**2
        return torch.stack((p,yaw),1)

    def forward(self,z,goal,velocity):
        return (self.coefficients(z,goal)*velocity[:,None,:]).sum(-1)


def build_learner(adapter,cfg,device,settings):
    # Runtime import comes after the audited task adapter establishes its paths.
    core, parent = adapter.core, adapter.derivative.DerivativeLearner
    from lac_vs.vision import image_tensor, dct_reconstruction
    from lac_vs.models import frozen_parameters

    class Learner(parent):
        def __init__(self):
            self.cfg,self.p,self.d,self.device = cfg,cfg['goal_fixed'],cfg['derivative'],device
            self.settings = dict(settings)
            self.ae = core.make_model(cfg,initialize=False).to(device).eval()
            install_deterministic_blur(self.ae)
            self.shadow = deepcopy(self.ae).eval()
            self.initial = deepcopy(self.ae).eval().requires_grad_(False)
            self.metric = core.GoalMetric().to(device).eval()
            self.critics = nn.ModuleList([ComponentCritic().to(device)])
            self.aeopt = torch.optim.Adam(self.ae.parameters(),lr=settings['encoder_lr'])
            self.shadowopt = torch.optim.Adam(self.shadow.parameters(),lr=settings['encoder_lr'])
            self.criticopt = torch.optim.Adam(self.critics.parameters(),lr=settings['critic_lr'])
            self.mopt = torch.optim.Adam(self.metric.parameters(),lr=settings['metric_lr'])
            self.anchors = dict(rec=1.,curvature=1.,latent_std=[1.]*16)
            self.counts = dict(warmup=0,critic_prefit=0,representation=0,metric=0)
            self.version = self.target_version = 0
            self.cache,self.tcache,self.inputs = {},{},{}
            self.guard_batch = None
            self.metric_accepted = 0

        def supervised_loss(self,z,zg,offset,actions):
            true = adapter.derivative.derivative_coefficients(offset).detach().clone()
            # Undo the legacy yaw-energy scaling, then apply success tolerances.
            true[:,3] *= math.radians(25.)**2
            zeros = torch.zeros_like(true)
            target = torch.stack((torch.cat((true[:,:3],zeros[:,3:]),-1)/.02**2,
                torch.cat((zeros[:,:3],true[:,3:]),-1)/math.radians(1.)**2),1)
            predicted = self.critics[0].coefficients(z,zg)
            physical = actions*actions.new_tensor([.4,.4,.4,.35])
            labels = torch.einsum('bci,bki->bck',target,physical)
            values = torch.einsum('bci,bki->bck',predicted,physical)
            # A per-component floor avoids blowing up near exactly-zero labels.
            factors = target.new_tensor([.4,.4,.4,.35])
            floors = target.new_tensor([.05/.02**2,.02/math.radians(1.)**2])
            denom = (target*factors).norm(dim=-1)+floors
            loss = ((values-labels)/denom[:,:,None]).square().mean()
            cos = F.cosine_similarity(predicted*factors,target*factors,dim=-1,eps=1e-8)
            active = (target*factors).norm(dim=-1)>1e-5
            direction = cos[active].mean() if bool(active.any()) else loss.new_tensor(1.)
            return loss,dict(derivative_loss=float(loss),derivative_gradient_cosine=float(direction),
                position_gradient_cosine=float(cos[:,0].mean()),yaw_gradient_cosine=float(cos[:,1].mean()))

        @torch.no_grad()
        def calibrate(self,replay,batches=32):
            raw,goals,energies,recs,curves = [],[],[],[],[]
            for _ in range(batches):
                b = replay.sample(16)
                z = self.encode(b['rgb']); zg = self.encode(b['goal_rgb'])
                raw.append(z); goals.append(zg); energies.append(core.pose_error(self.tensor(b['offset'])))
            z,g,e = torch.cat(raw),torch.cat(goals),torch.cat(energies)
            scale = torch.sqrt((.5*((z-g)/4).square().sum(-1).mean()/e.mean().clamp_min(1e-8)).clamp_min(1e-8))
            center = z.mean(0)
            for model in (self.ae,self.shadow):
                model.center.copy_(center); model.scale.fill_(float(scale))
            for _ in range(8):
                b = replay.sample(8)
                x = image_tensor(self.cfg,np.concatenate((b['rgb'],b['goal_rgb'])),self.device)
                _,y = self.ae(x); recs.append(float(F.mse_loss(y,x)+.25*dct_reconstruction(y,x)))
                a,c,d = self.encode(np.concatenate((b['smooth_minus'],b['smooth_center'],b['smooth_plus']))).chunk(3)
                curves.append(float((a-2*c+d).square().mean()))
            self.anchors = dict(rec=max(np.mean(recs),1e-5),curvature=max(np.mean(curves),1e-5),
                latent_std=((z-center)/scale).std(0).clamp_min(.1).cpu().tolist(),
                scale=float(scale),center=center.cpu().tolist(),samples=len(z))
            self.clear_cache(); self.version += 1
            return self.anchors

        @torch.no_grad()
        def prepared(self,b):
            z,zn = self.encode(b['rgb']),self.encode(b['next_rgb'])
            goal,j = self.goal_features(b,perturb=False)
            return dict(z=z,goal=goal,j=j,e=(z-goal)/4,en=(zn-goal)/4,
                offset=self.tensor(b['offset']),next_offset=self.tensor(b['next_offset']),
                previous=self.tensor(b['previous_action']))

        def objectives(self,b):
            m = self.metric(b['goal'])[1]
            u,_ = core.policy(b['e'],b['j'],m)
            successor = core.p0_offset_step(b['offset'],u)
            risk,components,residual = progress_risk(b['offset'],successor,
                self.settings['progress_fraction'],self.settings['interior_rho'])
            physical = core.pose_error(b['offset']); pn = core.pose_error(b['next_offset'])
            cal = .5*(core.calibration_loss(core.energy(b['e'],m),physical,self.cfg)+
                core.calibration_loss(core.energy(b['en'],m),pn,self.cfg))
            replay = core.actual_descent_loss(b['e'],b['en'],m,physical,self.cfg)
            with frozen_parameters(self.critics):
                coeff = self.critics[0].coefficients(b['z'],b['goal']).detach()
            scales = u.new_tensor([.4,.4,.4,.35])
            pred = (coeff*u[:,None,:]*scales).sum(-1)
            normalizer = (coeff*scales).norm(dim=-1).detach()+1.
            surrogate = (pred/normalizer).mean()
            aux = .1*cal+.05*replay+.02*surrogate+.001*(u-b['previous']).square().mean()
            return risk,aux,components,u,m,dict(calibration=float(cal),replay_descent=float(replay),
                critic_surrogate=float(surrogate),progress_violation_fraction=float((residual>0).float().mean()))

        def begin_metric_block(self,b):
            self.guard_batch = self.prepared(b)

        def metric_update(self,b):
            self.zero_all()
            data = self.prepared(b)
            if self.guard_batch is not None:
                data = {k:torch.cat((v,self.guard_batch[k]),0) for k,v in data.items()}
            risk,aux,components,old_u,old_m,logs = self.objectives(data)
            params = list(self.metric.parameters())
            gp = torch.cat([g.reshape(-1) for g in torch.autograd.grad(risk,params,retain_graph=True)])
            ga = torch.cat([g.reshape(-1) for g in torch.autograd.grad(aux,params)])
            combined,gradlogs = priority_gradient(gp,ga,self.settings['aux_gradient_cap'])
            cursor = 0
            for p in params:
                p.grad = combined[cursor:cursor+p.numel()].view_as(p).clone(); cursor += p.numel()
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if not bool(torch.isfinite(risk+aux)): raise FloatingPointError('Nonfinite metric objective')
            saved = deepcopy(self.metric.state_dict()); optimizer = deepcopy(self.mopt.state_dict())
            before = group_risks(components.detach(),data['offset'])
            old_u,old_m = old_u.detach(),old_m.detach()
            self.mopt.step()
            proposed = deepcopy(self.metric.state_dict())
            accepted = False; ratio = 0.; stats = {}
            # Adam moments are retained only when its proposed direction is used.
            for ratio in (1.,.5,.25,.125,.0625):
                with torch.no_grad():
                    for key,value in self.metric.state_dict().items():
                        value.copy_(saved[key]+ratio*(proposed[key]-saved[key]) if value.is_floating_point() else saved[key])
                    new_risk,_,pc,new_u,new_m,_ = self.objectives(data)
                    stats = trust_statistics(old_u,new_u,old_m,new_m)
                    after = group_risks(pc,data['offset'])
                    accepted = bool(torch.isfinite(new_risk) and new_risk <= risk.detach()+1e-7 and
                        (after <= before+1e-6).all() and
                        stats['action_change_mean'] <= self.settings['action_trust_mean'] and
                        stats['action_change_max'] <= self.settings['action_trust_max'])
                if accepted: break
            if not accepted:
                self.metric.load_state_dict(saved); self.mopt.load_state_dict(optimizer); ratio=0.
            self.metric_accepted += int(accepted); self.counts['metric'] += 1
            if any(p.grad is not None for p in self.ae.parameters()) or any(p.grad is not None for p in self.critics.parameters()):
                raise RuntimeError('Metric gradient crossed module boundary')
            return dict(total=float(risk+aux),primary=float(risk),auxiliary=float(aux),
                candidate_primary=float(new_risk),metric_step_accepted=accepted,line_search_scale=ratio,
                applied_action_change_mean=stats['action_change_mean'] if accepted else 0.,
                applied_action_change_max=stats['action_change_max'] if accepted else 0.,
                gradient_norm=float(norm),**logs,**gradlogs,**stats)

    return Learner()
