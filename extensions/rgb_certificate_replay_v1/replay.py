"""Bounded in-memory uint8 replay; targets are stored once per room and goal."""
from pathlib import Path
import numpy as np
import torch


class Replay:
    def __init__(self, capacity, seed, device='cuda:0'):
        self.capacity, self.device = int(capacity), device
        self.rng = np.random.default_rng(seed)
        self.size = self.cursor = self.total = 0
        self.arrays = {}
        self.goals, self.goal_keys = [], {}
        self.goal_latents = None
        self.features = None

    def register_goals(self, keys, images):
        result = []
        images = images.detach().cpu().numpy()
        for key, image in zip(keys, images):
            if key not in self.goal_keys:
                self.goal_keys[key] = len(self.goals)
                self.goals.append(image.copy())
                self.goal_latents = None
            result.append(self.goal_keys[key])
        return np.asarray(result, dtype=np.int32)

    def append(self, data):
        converted = {k:(v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k,v in data.items()}
        n = len(converted['action'])
        if not n: return
        if not self.arrays:
            self.arrays = {k:np.empty((self.capacity,*v.shape[1:]),dtype=v.dtype) for k,v in converted.items()}
        if set(converted) != set(self.arrays): raise ValueError('Replay schema changed')
        indices = (np.arange(n)+self.cursor)%self.capacity
        for key, array in converted.items(): self.arrays[key][indices] = array
        self.cursor = int((self.cursor+n)%self.capacity)
        self.size = min(self.capacity,self.size+n); self.total += n
        if self.features is not None:
            with torch.no_grad():
                x = torch.as_tensor(converted['image'],device=self.device)
                xn = torch.as_tensor(converted['next_image'],device=self.device)
                self.z[indices] = self.features(x).cpu().numpy()
                self.zn[indices] = self.features(xn).cpu().numpy()

    def _indices(self, count):
        if not self.size: raise RuntimeError('Empty fresh replay')
        return self.rng.integers(0,self.size,size=count)

    def raw_batch(self, count):
        ids = self._indices(count)
        batch = {k:torch.as_tensor(self.arrays[k][ids],device=self.device) for k in ('image','next_image','offset','energy')}
        batch['goal_image'] = torch.as_tensor(np.stack([self.goals[i] for i in self.arrays['goal_id'][ids]]),device=self.device)
        return batch

    @torch.no_grad()
    def bind_features(self, features, guard=lambda:None):
        self.features = features
        d = len(features.center)
        self.z = np.empty((self.capacity,d),dtype=np.float32)
        self.zn = np.empty_like(self.z)
        for start in range(0,self.size,512):
            guard(); end = min(self.size,start+512)
            self.z[start:end] = features(torch.as_tensor(self.arrays['image'][start:end],device=self.device)).cpu().numpy()
            self.zn[start:end] = features(torch.as_tensor(self.arrays['next_image'][start:end],device=self.device)).cpu().numpy()
        self._goals()

    @torch.no_grad()
    def _goals(self):
        if self.goal_latents is None:
            self.goal_latents = self.features(torch.as_tensor(np.stack(self.goals),device=self.device)).cpu().numpy()

    def batch(self, count):
        self._goals(); ids = self._indices(count)
        goal = self.goal_latents[self.arrays['goal_id'][ids]]
        result = dict(error=self.z[ids]-goal,next_error=self.zn[ids]-goal,goal=goal)
        for k in ('action','previous','remaining','next_remaining','reward','terminal','energy'): result[k]=self.arrays[k][ids]
        return {k:torch.as_tensor(v,device=self.device) for k,v in result.items()}

    def save(self, folder):
        folder=Path(folder);folder.mkdir(exist_ok=False)
        # Raw camera frames are already downsampled to 64px. No duplicate
        # per-update archives; this one bounded snapshot enables later audits.
        for k, data in self.arrays.items(): np.save(folder/(k+'.npy'),data[:self.size],allow_pickle=False)
        np.save(folder/'goals.npy',np.stack(self.goals),allow_pickle=False)
        return dict(size=self.size,total=self.total,cursor=self.cursor,capacity=self.capacity,goal_keys=self.goal_keys,
                    bytes=sum(a[:self.size].nbytes for a in self.arrays.values()))
