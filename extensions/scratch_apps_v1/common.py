from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def paths():
    for p in (ROOT / 'third_party/push_original', ROOT / 'third_party/so101_timeaware_v1',
              ROOT / 'stabilization-gym', ROOT / 'stabilization-learning-control'):
        sys.path.insert(0, str(p))
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def save_torch(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(value, tmp); tmp.replace(path)


class Stopped(RuntimeError):
    pass


class Run:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.protocol = read(self.root / 'protocol.json')
        self.config = self.protocol['config']
        for name, expected in self.protocol['sources'].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError('Frozen source changed: ' + name)
        for name, expected in self.protocol.get('assets', {}).items():
            if sha(ROOT / name) != expected:
                raise RuntimeError('Frozen environment asset changed: ' + name)
        self.event_no = len(list((self.root / 'events').glob('*.json')))

    def guard(self):
        if (self.root / 'STOP').exists():
            raise Stopped('STOP marker is present')
        if time.time() >= self.protocol['deadline_unix']:
            raise Stopped('Frozen wall-clock budget exhausted')
        import shutil
        if shutil.disk_usage(self.root).free < 20 * 1024**3:
            raise Stopped('Less than 20 GiB free; preserve completed evidence')

    def event(self, kind, **data):
        # Stages run sequentially in child processes; the supervisor also emits.
        self.event_no = len(list((self.root / 'events').glob('*.json'))) + 1
        record = dict(kind=kind, unix=time.time(), **data)
        write(self.root / 'events' / f'{self.event_no:07d}_{kind}.json', record)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    def new_stage(self, name, details):
        self.guard()
        out = self.root / name
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'protocol.json', dict(parent_sha256=sha(self.root / 'protocol.json'), **details))
        self.event('stage_started', stage=name)
        return out

    def frozen_group(self, group):
        for seed in self.config[group]['seeds']:
            p = self.root / f'{group}_seed{seed}' / 'FROZEN.json'
            if not p.exists():
                raise RuntimeError(f'Cannot test before every {group} model is frozen: {p}')
            marker = read(p)
            if sha(p.parent / marker['checkpoint']) != marker['sha256']:
                raise RuntimeError('Frozen checkpoint was changed')


def source_files():
    files = []
    for name in ('extensions/scratch_apps_v1', 'third_party/so101_timeaware_v1',
                 'third_party/push_original/dl_push', 'third_party/visual_original/runtime',
                 'third_party/visual_original/lac_vs',
                 'migration/visual_scene_reconstruction/reconstructed_legacy_root/src'):
        files.extend((ROOT / name).rglob('*.py'))
    files += [ROOT / 'scripts/run_scratch_applications_v1.py']
    files += [ROOT / 'stabilization-gym/stabilization_gym/envs/tasks' / n
              for n in ('original_visual.py','original_visual_online.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}
