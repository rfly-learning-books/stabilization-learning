"""Sequence D/Q regression and best-of-feasible-samples policy learning."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dl_push.six_primitives.learning import Actor,Encoder


class SequenceDQ(nn.Module):
    def __init__(self,knots):
        super().__init__();self.knots=knots;self.encoder=Encoder()
        self.head=nn.Sequential(nn.Linear(128+2*knots,128),nn.ReLU(),nn.Linear(128,5))
        self.register_buffer('label_mean',torch.zeros(5))
        self.register_buffer('label_scale',torch.ones(5))

    def forward(self,obs,knots_normalized):
        z=self.encoder(obs)
        if knots_normalized.ndim==4:
            z=z[:,None,:].expand(-1,knots_normalized.shape[1],-1)
        return self.head(torch.cat((z,knots_normalized.flatten(-2)),-1))

    def predict(self,obs,knots_normalized):
        return self(obs,knots_normalized)*self.label_scale+self.label_mean


def best_sample_loss(prediction,candidate_actions,valid):
    """Match a feasible mode, never a potentially infeasible average of modes."""
    distance=(prediction[:,None]-candidate_actions).square().sum(-1)
    usable=valid.any(-1)
    if not usable.any():return prediction.sum()*0,usable
    values=distance[usable].masked_fill(~valid[usable],float('inf')).min(-1).values
    return values.mean(),usable


def load_shards(path,successful_only):
    files=sorted(Path(path).glob('episode_*.npz'))
    if not files:raise ValueError('No collected episode shards')
    rows={};episode_ids=[]
    for i,p in enumerate(files):
        with np.load(p,allow_pickle=False) as archive:
            data={key:archive[key] for key in archive.files}
        n=len(data['s'])
        if not n:continue
        for key,value in data.items():
            if key in ('episode_success','shape_name'):continue
            rows.setdefault(key,[]).append(value)
        eligible=data['real_accepted'].astype(bool)
        if successful_only:eligible &= bool(data['episode_success'])
        rows.setdefault('policy_eligible',[]).append(eligible)
        episode_ids.extend([i]*n)
    if not episode_ids:raise ValueError('Dataset contains no transitions')
    return {key:np.concatenate(value) for key,value in rows.items()},np.asarray(episode_ids),files


def obs_tensor(data,ids,device):
    return {key:torch.as_tensor(data[key][ids],device=device).float() for key in ('s','g','shape','task')}


class DQRefiner:
    """Learned sequence gradients propose controls; Planner re-simulates every proposal."""
    def __init__(self,path,device='cpu',steps=8,lr=.04):
        saved=torch.load(path,map_location=device,weights_only=False)
        self.model=SequenceDQ(saved['knots']).to(device)
        self.model.load_state_dict(saved['dq']);self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.steps=steps;self.lr=lr;self.device=device

    def __call__(self,obs,knots,qpos,userdata,task,cfg):
        if knots.shape[1]!=self.model.knots:raise ValueError('Surrogate/planner knot count mismatch')
        o={k:torch.as_tensor(v,device=self.device).float() for k,v in obs.items()}
        import jax
        prep=np.asarray(jax.vmap(task.prep)(qpos))
        g=userdata[:,0]-qpos[:,0]
        phase=(prep>cfg.prep_switch)&(np.abs(g)>cfg.goal_tolerance)
        root=.5*(np.where(phase,prep,g)/.1)**2
        tolerance=.5*(np.where(phase,cfg.prep_switch,cfg.goal_tolerance)/.1)**2
        target=torch.as_tensor(cfg.contraction*np.maximum(root-tolerance,0),device=self.device)
        phase=torch.as_tensor(phase,device=self.device)
        x=torch.tensor(knots/.3,device=self.device,requires_grad=True)
        for _ in range(self.steps):
            pred=self.model.predict(o,x)
            d=torch.where(phase,pred[:,0],pred[:,1])
            loss=(pred[:,2]+cfg.progress_weight*pred[:,4]
                  +cfg.violation_weight*(torch.relu(d+target)+torch.relu(pred[:,3]))).mean()
            grad,=torch.autograd.grad(loss,x)
            if not torch.isfinite(grad).all():raise FloatingPointError('Non-finite DQ sequence gradient')
            with torch.no_grad():
                x-=self.lr*grad/grad.norm(dim=(1,2),keepdim=True).clamp_min(1.)
                x.clamp_(-1,1)
        return x.detach().cpu().numpy()*.3
