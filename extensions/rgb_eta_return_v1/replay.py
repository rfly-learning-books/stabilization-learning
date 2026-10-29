"""One fixed-encoder replay on the GPU, with actual metric-action provenance."""
from pathlib import Path
import numpy as np
import torch


class EtaReplay:
    def __init__(self, capacity, device, feature_version):
        self.capacity, self.device = capacity, torch.device(device)
        self.feature_version = feature_version
        self.arrays = {}; self.size = self.cursor = self.total = 0

    def append(self, data, feature_version):
        if feature_version != self.feature_version: raise ValueError('Replay coordinate version mismatch')
        if not bool(data['eta_valid'].all()): raise ValueError('Behavior did not execute the logged eta controller')
        n = len(data['eta'])
        if not n: return
        if n > self.capacity: raise ValueError('Append exceeds capacity')
        if not self.arrays:
            self.arrays = {k:torch.empty((self.capacity,*v.shape[1:]),dtype=v.dtype,device=self.device) for k,v in data.items()}
        if set(data) != set(self.arrays): raise ValueError('Replay schema changed')
        ids = (torch.arange(n,device=self.device)+self.cursor)%self.capacity
        for k,v in data.items(): self.arrays[k][ids] = v.detach()
        self.cursor = (self.cursor+n)%self.capacity; self.size = min(self.capacity,self.size+n); self.total += n

    def batch(self, count):
        if not self.size: raise ValueError('Empty eta replay')
        ids = torch.randint(self.size,(count,),device=self.device)
        return dict(**{k:v[ids] for k,v in self.arrays.items()},feature_version=self.feature_version)

    def save(self, folder):
        folder = Path(folder); folder.mkdir(exist_ok=False)
        for k,v in self.arrays.items(): np.save(folder/(k+'.npy'),v[:self.size].cpu().numpy(),allow_pickle=False)
        return dict(size=self.size,total=self.total,cursor=self.cursor,capacity=self.capacity,
            feature_version=self.feature_version,bytes=sum(v[:self.size].numel()*v.element_size() for v in self.arrays.values()),
            representation='frozen coordinate transitions, eta, physical command and supervision; no duplicate camera-frame shards')


class ImageBank:
    """AE-only image reservoir; random physical actions never enter eta replay."""
    def __init__(self, capacity, seed):
        self.capacity = capacity; self.rng = np.random.default_rng(seed)
        self.images = None; self.seen = self.size = 0

    def add(self, images):
        images = images.detach().cpu().numpy()
        if self.images is None: self.images = np.empty((self.capacity,*images.shape[1:]),dtype=images.dtype)
        for image in images:
            self.seen += 1
            slot = self.seen-1 if self.seen<=self.capacity else int(self.rng.integers(self.seen))
            if slot<self.capacity: self.images[slot] = image
        self.size = min(self.capacity,self.seen)

    def batch(self, count, device):
        ids = self.rng.integers(self.size,size=count)
        return torch.as_tensor(self.images[ids],device=device)
