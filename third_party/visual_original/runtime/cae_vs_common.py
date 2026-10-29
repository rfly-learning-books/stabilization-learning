"""RGB-only, pose-label-free CAE-VS experiment helpers.

All original project and legacy modules are imported read-only. New artifacts
belong to their own protocol. Pose may generate RGB views and score control;
the representation loader never reads positions, offsets, cost or scene labels.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lac_vs import legacy
from lac_vs.config import load_config, digest, json_write, source_hashes, seed_all
from lac_vs.vision import images_and_tangents, encoder_jvp, image_tensor

VARIANTS = ('ae_relu', 'ae_smooth', 'cae_smooth')

def emit(**row):
    print(json.dumps(row, ensure_ascii=False), flush=True)

def resources(path):
    free = shutil.disk_usage(path).free / 2**30
    if free < 30:
        raise RuntimeError(f'Safe storage stop: only {free:.2f} GiB free')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    return dict(free_gib=free, gpu=gpu, cuda_allocated_mib=torch.cuda.memory_allocated()/2**20,
                cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20)

def setup(seed=2026101647):
    seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.65)

def save(path, payload):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)

class GaussianBlur(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        self.register_buffer('kernel', (k[:, None]*k[None, :]/16.)[None, None].repeat(channels, 1, 1, 1))
        self.stride = stride
    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), self.kernel,
                        stride=self.stride, groups=x.shape[1])

def smooth_encoder(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.Softplus(beta=20., threshold=30.))
        elif isinstance(child, nn.MaxPool2d):
            setattr(module, name, GaussianBlur(64, stride=2))
        elif isinstance(child, nn.Conv2d) and child.stride != (1, 1):
            # Keep all learned convolution weights. Low-pass before decimation,
            # including the residual projection branch, with fixed calibration.
            setattr(module, name, nn.Sequential(GaussianBlur(child.in_channels), child))
        else:
            smooth_encoder(child)

class CAEModel(nn.Module):
    def __init__(self, cfg, variant, initialize=False):
        super().__init__()
        self.variant = variant
        self.base = legacy.autoencoder(cfg, 'cpu', initialize=initialize)
        if variant != 'ae_relu':
            smooth_encoder(self.base.encoder)
        self.register_buffer('center', torch.zeros(16))
        self.register_buffer('scale', torch.ones(16))
    def encode(self, x):
        return (self.base.encode(x)-self.center)/self.scale
    def forward(self, x):
        z = self.encode(x)
        internal = z*self.scale+self.center
        y = self.base.expand_spatial(self.base.expand(internal)[..., None, None])
        return z, self.base.decoder(y)

def load_model(out, variant, device='cuda'):
    cfg = load_config(ROOT/'configs/full.yaml')
    payload = torch.load(Path(out)/'models'/variant/'best.pt', map_location='cpu', weights_only=False)
    model = CAEModel(cfg, variant)
    model.load_state_dict(payload['model'])
    return model.to(device).eval().requires_grad_(False), payload

def assert_training_integrity(out):
    out = Path(out)
    protocol = json.loads((out/'training_protocol.json').read_text(encoding='utf-8'))
    for relative, expected in protocol['training_sources'].items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError(f'Training source changed: {relative}; create a new protocol')
    cfg = load_config(ROOT/'configs/full.yaml')
    if protocol['original_sources'] != source_hashes() or protocol['dependencies'] != legacy.dependency_hashes(cfg):
        raise RuntimeError('Read-only original source/dependency mismatch')
    if digest(out/'dataset_index.json') != protocol['dataset_index_sha256']:
        raise RuntimeError('Frozen dataset index changed')
    return protocol

def load_pairs(out, split, limit=None):
    """Load only RGB, frozen estimated depth, camera calibration and action.

    No pose, reward, termination, target coordinates or environment label is
    passed to the loss. Scene ids stay outside tensors, for split auditing only.
    """
    out = Path(out)
    cfg = load_config(ROOT/'configs/full.yaml')
    rows = [r for r in json.loads((out/'dataset_index.json').read_text(encoding='utf-8')) if r['split']==split]
    if limit is not None:
        # Index is scene-major; stride is fixed before any model is evaluated.
        take = np.linspace(0, len(rows)-1, min(limit, len(rows))).astype(int)
        rows = [rows[i] for i in take]
    xs0=[];xs1=[];qs=[];actions=[]
    for row in rows:
        path = out/row['file']
        if digest(path) != row['sha256']:
            raise RuntimeError(f'Dataset shard changed: {path}')
        with np.load(path) as a:
            x0, q = images_and_tangents(cfg, [a['rgb'][0]], [a['estimated_depth'][0]],
                                       [a['camera_from_body']], 'cpu')
            x1 = image_tensor(cfg, [a['rgb'][1]], 'cpu')
            xs0.append(x0[0].numpy());xs1.append(x1[0].numpy())
            qs.append(q[0].numpy());actions.append(a['action'])
    result = dict(x0=np.stack(xs0), x1=np.stack(xs1), q=np.stack(qs), u=np.stack(actions))
    emit(event='loaded_pairs', split=split, pairs=len(rows), arrays_mib=sum(v.nbytes for v in result.values())/2**20)
    return result, rows

def tensors(data, indices, device='cuda'):
    return {k: torch.from_numpy(np.ascontiguousarray(v[indices])).to(device) for k,v in data.items()}

def direction(model, x, q):
    return torch.func.jvp(lambda value: model.encode(value)/4., (x,), (q,))[1]

def identity_control(e, j):
    """Same conditioned four-axis LM controller for every representation."""
    b=j.T@j
    eta=max(1e-8, 1.01*np.trace(b)/(1e4-1))
    raw=-.7*np.linalg.solve(b+.01*np.diag(np.diag(b))+eta*np.eye(4),j.T@e)
    return raw/max(1.,np.linalg.norm(raw[:3]),abs(raw[3]))

def stats(values):
    values=np.asarray(values,dtype=float)
    v=values[np.isfinite(values)]
    if not len(v):
        return dict(count=0, missing=len(values))
    return dict(count=len(v),missing=len(values)-len(v),mean=float(v.mean()),median=float(np.median(v)),
                p95=float(np.percentile(v,95)),maximum=float(v.max()))
