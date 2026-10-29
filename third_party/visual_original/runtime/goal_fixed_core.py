"""Goal-conditioned fixed-J visual stabilization. Original experiments read-only."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import ast
import collections
import copy
import hashlib
import importlib
import json
import math
import pickle
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lac_vs import legacy
from lac_vs.config import load_config, digest, json_write, json_append, seed_all
from lac_vs.models import CostCritic, frozen_parameters, ema
from lac_vs.vision import image_tensor, images_and_tangents, encoder_jvp, dct_reconstruction
from lac_vs.environment import task_error, sample_start, valid_start
from cae_vs_common import CAEModel

def emit(**row):
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

class InertMetadata:
    """Metadata objects in a remote checkpoint are never executed."""
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

def metadata_getattr(obj, name):
    if isinstance(obj, InertMetadata) and name in ('train_transform','val_transform','h5_loader'):
        return None
    raise pickle.UnpicklingError('Checkpoint attempted a non-metadata getattr')

class RestrictedCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__builtin__' and name == 'getattr':
            return metadata_getattr
        if module == 'collections' and name in ('OrderedDict', 'defaultdict'):
            return getattr(collections, name)
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor', '_rebuild_parameter'):
            return getattr(importlib.import_module(module), name)
        if module == 'torch' and name.endswith('Storage'):
            return getattr(torch, name)
        if module in ('builtins', '__builtin__') and name in ('set', 'frozenset', 'slice', 'list', 'dict', 'tuple'):
            import builtins
            return getattr(builtins, name)
        if module.startswith(('nyudepthv2', 'torch.utils.data', 'torchvision.transforms', 'argparse')):
            return InertMetadata
        raise pickle.UnpicklingError(f'Refusing remote checkpoint global: {module}.{name}')

def depth_definition(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    # Import the network declarations, not upstream training/evaluation code.
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    space = dict(torch=torch, nn=nn, F=F, math=math, time=time)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), space)
    return space['FastDepth']

class FastDepth:
    def __init__(self, cfg, device='cuda'):
        self.cfg, self.device = cfg, torch.device(device)
        root = ROOT / cfg['goal_fixed']['depth_path']
        self.model = depth_definition(root/'models.py')()
        extracted = root/'extracted_state_dict.pt'
        if not extracted.exists():
            restricted = types.ModuleType('restricted_fastdepth_checkpoint')
            restricted.Unpickler = RestrictedCheckpointUnpickler
            restricted.load = lambda stream, **kwargs: RestrictedCheckpointUnpickler(stream, **kwargs).load()
            payload = torch.load(root/'FastDepth_L1_Best.pth', map_location='cpu', pickle_module=restricted,
                                 weights_only=False)
            state = payload['model_state_dict']
            if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
                raise ValueError('Depth state is not a pure tensor state dictionary')
            temp = extracted.with_suffix('.tmp')
            torch.save(state, temp)
            temp.replace(extracted)
        state = torch.load(extracted, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.hash = digest(extracted)

    @torch.no_grad()
    def predict(self, rgb):
        # Upstream model uses RGB/255, no ImageNet mean/std. Preserve the full
        # square camera FOV; resizing to 224 is reversed for the 256 image grid.
        x = cv2.resize(rgb, (224,224), interpolation=cv2.INTER_AREA).astype(np.float32)/255.
        x = torch.from_numpy(x.transpose(2,0,1).copy())[None].to(self.device)
        z = self.model(x)[0,0].cpu().numpy()
        if not np.isfinite(z).all():
            raise FloatingPointError('Nonfinite FastDepth output')
        z = cv2.resize(z, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.clip(z, self.cfg['goal_fixed']['min_depth'], self.cfg['goal_fixed']['max_depth']).astype(np.float32)

class GoalMetric(nn.Module):
    """Only the fixed goal latent enters M. All eigenvalue bounds are structural."""
    def __init__(self, dim=16, hidden=128, floor=.1):
        super().__init__()
        self.dim, self.floor = dim, floor
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),
                               nn.Linear(hidden,dim*(dim+1)//2))
        rows, cols=torch.tril_indices(dim,dim)
        self.register_buffer('rows',rows);self.register_buffer('cols',cols)
        self.register_buffer('diagonal',rows==cols)
        nn.init.normal_(self.net[-1].weight,std=.001);nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[self.diagonal]=math.log(math.expm1(1.-1e-3))

    def forward(self, goal):
        raw=self.net(goal)
        entries=torch.where(self.diagonal,F.softplus(raw)+1e-3,raw)
        lower=entries.new_zeros(*entries.shape[:-1],self.dim,self.dim)
        lower[...,self.rows,self.cols]=entries
        gram=lower@lower.transpose(-1,-2)
        trace=gram.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]
        eye=torch.eye(self.dim,device=goal.device,dtype=goal.dtype)
        m=self.floor*eye+self.dim*(1-self.floor)*gram/trace
        return lower,m

def energy(e,m):
    return .5*torch.einsum('...i,...ij,...j->...',e,m,e)

def saturate(u):
    scale=torch.maximum(torch.ones_like(u[...,0]),torch.linalg.vector_norm(u[...,:3],dim=-1))
    scale=torch.maximum(scale,u[...,3].abs())
    return u/scale[...,None]

def policy(e,j,m,gain=.7):
    jt=j.transpose(-1,-2)
    b=jt@m@j
    tr=b.diagonal(dim1=-2,dim2=-1).sum(-1)
    eta=torch.maximum(tr*1.01/(1e4-1),tr.new_full(tr.shape,1e-8))
    h=b+.01*torch.diag_embed(b.diagonal(dim1=-2,dim2=-1))+eta[...,None,None]*torch.eye(4,device=j.device,dtype=j.dtype)
    g=(jt@m@e.unsqueeze(-1)).squeeze(-1)
    raw=-gain*torch.linalg.solve(h,g.unsqueeze(-1)).squeeze(-1)
    return saturate(raw),h

def p0_offset_step(offset,u,dt=.1):
    """Exact differentiable task-coordinate equivalent of the inherited P0."""
    yaw=offset[...,3]+.5*.35*u[...,3]*dt
    c,s=torch.cos(yaw),torch.sin(yaw)
    dx=.4*dt*(c*u[...,0]-s*u[...,1])
    dy=.4*dt*(s*u[...,0]+c*u[...,1])
    return offset+torch.stack([dx,dy,.4*dt*u[...,2],.35*dt*u[...,3]],-1)

def pose_error(offset):
    scale=offset.new_tensor([1.,1.,1.,math.radians(25)])
    return (offset/scale).square().sum(-1)

def calibration_loss(v,E,cfg):
    p=cfg['goal_fixed'];den=(E.detach()+p['objective_floor'])
    return ((F.relu(p['value_lower']*E-v)/den).square()+
            (F.relu(v-p['value_upper']*E)/den).square()).mean()

def actual_descent_loss(e,en,m,E,cfg):
    v,vn=energy(e,m),energy(en,m)
    residual=vn-v+cfg['goal_fixed']['decay_per_second']*.1*E
    return (F.relu(residual)/(E.detach()+cfg['goal_fixed']['objective_floor'])).square().mean()

def resource(out,cfg,measure_size=False):
    disk=shutil.disk_usage(out)
    if disk.free/2**30<cfg['goal_fixed']['minimum_free_gib']:
        raise RuntimeError('Safe storage stop: free space below configured reserve')
    result=dict(free_gib=disk.free/2**30,cuda_allocated_mib=torch.cuda.memory_allocated()/2**20,
                cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20)
    try:
        result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=5).strip()
    except Exception:
        result['gpu']='unavailable'
    if measure_size:
        size=sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())/2**30
        result['run_gib']=size
        if size>cfg['goal_fixed']['maximum_run_gib']:
            raise RuntimeError('Safe storage stop: run has reached its configured maximum size')
    return result

class Replay:
    """Scene-balanced original-image replay; latent/J caches are versioned separately."""
    def __init__(self,out,cfg,seed=41):
        self.root=Path(out)/'replay';self.root.mkdir(parents=True,exist_ok=True)
        self.cfg=cfg;self.rng=np.random.default_rng(seed)
        self.path=self.root/'index.json'
        self.rows=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else []
        self.cache=collections.OrderedDict()
        self.max=cfg['collection']['replay_transitions']
        if any(r['split']!='train' for r in self.rows):
            raise ValueError('Evaluation data found in replay')
    @property
    def transitions(self):
        return sum(r['length'] for r in self.rows)
    def add(self,arrays,row,identifier):
        if row['split']!='train':raise ValueError('Only train can enter replay')
        path=self.root/f'{identifier}.npz'
        # A crash can leave a shard written after the last saved active index.
        # Keep that evidence and give the retried episode a distinct filename.
        attempt=0
        while path.exists():
            attempt+=1;path=self.root/f'{identifier}_attempt{attempt:02d}.npz'
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,**arrays)
        temp.replace(path)
        self.rows.append(dict(file=path.name,length=len(arrays['action']),scene_id=row['scene_id'],split='train',goal=row['goal_id']))
        while self.transitions>self.max and len(self.rows)>1:self.rows.pop(0)
        json_write(self.path,self.rows)
    def load(self,row):
        if row['file'] not in self.cache:
            with np.load(self.root/row['file'],allow_pickle=False) as a:self.cache[row['file']]={k:a[k] for k in a.files}
            while len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(row['file']);return self.cache[row['file']]
    def sample(self,size):
        scenes=sorted({r['scene_id'] for r in self.rows});samples=[]
        for _ in range(size):
            scene=scenes[int(self.rng.integers(len(scenes)))];pool=[r for r in self.rows if r['scene_id']==scene]
            prob=np.array([r['length'] for r in pool],float);prob/=prob.sum()
            row=pool[int(self.rng.choice(len(pool),p=prob))];a=self.load(row);t=int(self.rng.integers(row['length']))
            n=int(self.rng.integers(len(a['smooth_center'])))
            samples.append(dict(rgb=a['rgb'][t],next_rgb=a['rgb'][t+1],goal_rgb=a['goal_rgb'],
                goal_depth=a['goal_depth'],rotation=a['rotation'],offset=a['offset'][t],next_offset=a['offset'][t+1],
                cost=a['cost'][t],terminal=a['terminal'][t],truncated=a['truncated'][t],action=a['action'][t],
                previous_action=a['previous_action'][t],smooth_center=a['smooth_center'][n],
                smooth_minus=a['smooth_minus'][n],smooth_plus=a['smooth_plus'][n],
                smooth_step=a['smooth_step'][n],goal_key=f'{row["scene_id"]}/{row["goal"]}'))
        return {k:[r[k] for r in samples] if k=='goal_key' else np.stack([r[k] for r in samples]) for k in samples[0]}

class Reference:
    def __init__(self,cfg,model,metric,rgb,depth,rotation,device='cuda',identity=False,scale=1.):
        self.cfg,self.model,self.device=cfg,model,device
        x,q=images_and_tangents(cfg,[rgb],[depth*scale],[rotation],device)
        with frozen_parameters(model),torch.no_grad():
            self.goal=model.encode(x).detach()
            self.j=encoder_jvp(model,x,q).transpose(1,2).detach()
            self.lower,self.m=metric(self.goal)
            if identity:self.m=torch.eye(16,device=device)[None]
            # Cache full gain, since J and M are constant for the complete episode.
            eye=torch.eye(16,device=device)
            e=eye[None]
            jt=self.j.transpose(-1,-2);b=jt@self.m@self.j
            tr=b.diagonal(dim1=-2,dim2=-1).sum(-1)
            ridge=torch.maximum(tr*1.01/(1e4-1),tr.new_full(tr.shape,1e-8))
            h=b+.01*torch.diag_embed(b.diagonal(dim1=-2,dim2=-1))+ridge[...,None,None]*torch.eye(4,device=device)
            self.k=.7*torch.linalg.solve(h,jt@self.m)
            self.h=h
    @torch.no_grad()
    def observe(self,rgb):
        z=self.model.encode(image_tensor(self.cfg,[rgb],self.device))
        e=(z-self.goal)/4.
        u=saturate(-(self.k@e.unsqueeze(-1)).squeeze(-1))
        return u[0].cpu().numpy(),e[0].cpu().numpy(),float(energy(e,self.m)[0])

def make_model(cfg,initialize=True):
    return CAEModel(cfg,cfg['goal_fixed']['ae_variant'],initialize=initialize)

def source_fingerprints(cfg):
    files=[Path(__file__),Path(__file__).with_name('train_goal_fixed.py'),Path(__file__).with_name('report_goal_fixed.py'),
           Path(__file__).with_name('cae_vs_common.py')]
    files += [ROOT/'lac_vs'/n for n in ('config.py','environment.py','legacy.py','vision.py','locking.py','models.py')]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}
