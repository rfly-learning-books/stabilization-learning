"""Bounded timing of the historical RGB schedule, never a new policy experiment.

Old training archives are read-only timing fixtures. Optimizer updates affect
disposable copies. Equal-budget comparisons retain batch 4 and full probe logs.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT/'stabilization-gym', ROOT/'stabilization-learning-control'):
    sys.path.insert(0, str(folder))

import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch

from extensions.rgb_gpu_parallel.benchmark import (
    restore_workload, tensor_state_hash, sha, write_json, GPUUtilization, measure,
)
from extensions.joint_certificate_v2 import rgb as v2
from extensions.joint_certificate_v3 import rgb as v3
from extensions.joint_certificate_v3.replay import FastRawRGBReplay
from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime, RGBGeometry
from stabilization_gym.tasks_v3.reconstructed_rgb import BatchedRGBGeometry
from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
from stabilization_gym.tasks_gpu_rgb.geometry import GPURGBGeometry
from stabilization_gym.tasks_v2.visual import VisualObservation, quadratic_value
from stabilization_learning.joint_v2.visual import VisualBatch
from stabilization_learning.joint_v2.runtime import atomic_json
from stabilization_learning.joint_v3.inference import FrozenRGBPolicy


OLD_ROOT = ROOT/'runs/joint_certificate_v2_night_20261027_015308/launch_retry_20261027_015623'


class EventStore:
    """Same atomic fsync event payload path, in a new timing-only directory."""
    def __init__(self, root):
        self.root = root
        self.sequence = 0

    def event(self, kind, **data):
        self.sequence += 1
        atomic_json(self.root/f'{self.sequence:06d}_{kind}.json',
                    dict(sequence=self.sequence, kind=kind, unix=time.time(),
                         scope='timing_only_not_formal_training', data=data))


class GPUGeometryBridge:
    """Bounded host replay -> batched upload -> identical GPU geometry provider."""
    def __init__(self, geometry):
        self.geometry = geometry
        for key in ('device', 'cfg', 'cache_revision', 'provenance'):
            setattr(self, key, getattr(geometry, key))

    def observe_batch(self, public):
        packed = {key: torch.as_tensor(np.stack([p[key] for p in public]), device=self.device)
                  for key in public[0]}
        return self.geometry.observe(packed)

    def encode_images(self, images):
        return self.geometry.encode_images(torch.as_tensor(np.stack(images), device=self.device))


class ResidentReplay:
    """Full immutable historical training population on GPU for a timing fixture.

    Represents keeping newly GPU-collected uint8 frames resident while archival
    remains separate. It is not a production replay/collector integration.
    """
    def __init__(self, archive, rows, geometry):
        self.rows, self.geometry = rows, geometry
        self.rng = np.random.default_rng(811)
        self.by_file = {r['file']: i for i, r in enumerate(rows)}
        image_starts, action_starts, payload = [], [], []
        ni = na = 0
        for row in rows:
            path = archive/row['file']
            if sha(path) != row['sha256']:
                raise RuntimeError('Timing fixture archive hash mismatch')
            with np.load(path, allow_pickle=False) as data:
                value = {k: data[k] for k in data.files}
            image_starts.append(ni); action_starts.append(na)
            ni += len(value['rgb']); na += len(value['action'])
            payload.append(value)
        self.image_starts, self.action_starts = np.array(image_starts), np.array(action_starts)
        self.values = {}
        for key in ('rgb', 'offset', 'actual_normalized_action', 'previous_action', 'dt'):
            self.values[key] = torch.as_tensor(np.concatenate([d[key] for d in payload]), device='cuda')
        for key in ('goal_rgb', 'rotation'):
            self.values[key] = torch.as_tensor(np.stack([d[key] for d in payload]), device='cuda')
        self.controller = torch.tensor([r['policy']=='current_controller' for r in rows], device='cuda')
        self.resident_bytes = sum(v.numel()*v.element_size() for v in self.values.values())

    sample_keys = v2.RawRGBReplay.sample_keys

    def batch(self, keys, geometry=None, *, with_geometry=True):
        versions = [row['behavior_version'] for row, _ in keys]
        lane = np.array([self.by_file[r['file']] for r, _ in keys])
        step = np.array([i for _, i in keys])
        images = torch.as_tensor(self.image_starts[lane]+step, device='cuda')
        actions = torch.as_tensor(self.action_starts[lane]+step, device='cuda')
        lane = torch.as_tensor(lane, device='cuda')
        d = self.values; g = self.geometry
        public = dict(rgb=d['rgb'][images], goal_rgb=d['goal_rgb'][lane],
                      rotation=d['rotation'][lane], previous_action=d['previous_action'][actions])
        if with_geometry:
            x, goal, tangent = g.observe(public)
        else:
            x, goal = g.encode_images(public['rgb']), g.encode_images(public['goal_rgb'])
            tangent = x.new_zeros(len(keys), 4, *x.shape[1:])
        return VisualBatch(VisualObservation(x, goal, tangent), g.encode_images(d['rgb'][images+1]),
                           d['offset'][images].float(), d['offset'][images+1].float(),
                           d['actual_normalized_action'][actions].float(), d['dt'][actions].float(),
                           self.controller[lane], d['previous_action'][actions].float(),
                           behavior_version=versions[0] if len(set(versions))==1 else -1, split='train')


def host_reader(cls, output, archive, rows, **kwargs):
    # Constructor writes its new index under output. Only read methods are used
    # after redirecting archive reads; no original archive/index is ever written.
    reader = cls(output, np.random.default_rng(811), **kwargs)
    reader.root, reader.rows = archive, rows
    return reader


def timed_learning(output, name, learner, payload, warmup_payload, reader, geometry, seconds, repeats, telemetry):
    rows = []
    original = tensor_state_hash(learner)
    probe = reader.batch(reader.sample_keys(4), geometry, with_geometry=False)
    for mode in ('warmup', 'joint'):
        for repeat in range(repeats):
            v2._restore_for_evaluation(learner, warmup_payload if mode=='warmup' else payload)
            learner.feature_optimizer.state.clear(); learner.metric_optimizer.state.clear()
            reader.rng = np.random.default_rng(829)
            store = EventStore(output/'events'/name/mode/str(repeat))
            before = tensor_state_hash(learner)
            def operation():
                batch = reader.batch(reader.sample_keys(4), geometry, with_geometry=mode=='joint')
                blocks = [('warmup', learner.warmup)] if mode=='warmup' else [
                    ('feature', learner.update_features), ('metric', learner.update_metric)]
                for block, fn in blocks:
                    metrics = fn(batch)
                    store.event('update', block=block, step=sum(learner.counts.values()), metrics=metrics)
                    v2._probe(learner, probe, store, phase=block)
                return 4
            row = measure(operation, seconds, telemetry, 'same-batch training transitions/s', warmups=2)
            row.update(pipeline=name, mode=mode, repeat=repeat, batch_size=4,
                       seconds_per_operation=row['seconds']/row['iterations'],
                       parameters_changed=tensor_state_hash(learner)!=before,
                       includes='replay sampling/read, geometry if joint, full optimizer(s), atomic update/probe logs')
            if not row['parameters_changed']:
                raise AssertionError('Timing operation did not actually update the model')
            rows.append(row); print(json.dumps(row), flush=True)
    v2._restore_for_evaluation(learner, payload)
    if tensor_state_hash(learner) != original:
        raise AssertionError('Disposable workload failed to restore')
    return rows


def timed_evaluation(output, cfg, scenes, learner, geometry, size, seconds, repeats, telemetry):
    # Unique TRAIN rooms model the heterogeneous val/test scheduling cost;
    # no held-out policy evaluation or selection is performed here.
    families = ('classroom', 'office', 'meeting', 'library')
    selected = [next(s for s in scenes if s['split']=='train' and s['family']==families[i%4]
                     and s['id'].endswith(f'{i//4:02d}')) for i in range(size)]
    begin = time.perf_counter()
    env = GPUParallelRGBEnv(cfg, selected, max_steps=120, cache_dir=ROOT/'.cache/rgb_gpu_warp')
    public, _ = env.reset(seed=1207)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-begin
    results = []
    try:
        for repeat in range(repeats):
            public, _ = env.reset(seed=1207)
            state = dict(public=public)
            policy = FrozenRGBPolicy(learner)
            keys = [(s['id'], 0) for s in selected]
            traces = []
            def operation():
                with torch.no_grad():
                    out = policy(VisualObservation(*geometry.observe(state['public'])), goal_keys=keys)
                    following, _, terminal, truncated, info = env.step(out.action)
                    successor_error = learner.features.encode(geometry.encode_images(following['rgb']))-out.goal
                    successor_value = quadratic_value(successor_error, out.metric)
                    packed = torch.cat((out.error, out.interaction.flatten(1), out.action,
                                        info['supervision']['next_offset'].float(),
                                        info['supervision']['actual_velocity'].float(),
                                        out.value[:,None], successor_value[:,None],
                                        info['supervision']['physical_error'][:,None].float(),
                                        terminal[:,None], truncated[:,None], info['transition_mask'][:,None]), 1)
                    traces.append(packed.cpu().numpy())
                    state['public'] = following
                    return info['transition_mask'].sum()
            row = measure(operation, seconds, telemetry, 'valid full-trace evaluation transitions/s')
            start = time.perf_counter()
            v2._npz(output/f'BENCH_TRACE_B{size}_{repeat}.npz',
                    dict(rows=np.stack(traces), timing_only=np.array(True)))
            archive_seconds = time.perf_counter()-start
            row.update(mode='evaluation', pipeline='gpu', batch_size=size, repeat=repeat,
                       seconds_with_archive=row['seconds']+archive_seconds,
                       samples_per_second_with_archive=row['samples']/(row['seconds']+archive_seconds),
                       setup_seconds=setup_seconds, archive_seconds=archive_seconds,
                       distinct_train_rooms=size,
                       scope='throughput fixture with explicit reset masks; not policy evaluation results')
            results.append(row); print(json.dumps(row), flush=True)
    finally:
        env.close(); del env; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--pipelines', nargs='+',
                        choices=('v2_reference','v3_cpu','gpu_host_replay','gpu_resident'),
                        default=['v2_reference','v3_cpu','gpu_host_replay','gpu_resident'])
    parser.add_argument('--evaluation-batches', nargs='*', type=int, default=[8,16])
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    telemetry = None
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _, cfg, scenes = load_runtime(ROOT)
        learner, payload, workload = restore_workload(ROOT, cfg)
        warmup_path = Path(workload['path']).parent/'initial.pt'
        warmup_marker = json.loads(warmup_path.with_suffix('.COMMITTED.json').read_text(encoding='utf8'))
        if sha(warmup_path) != warmup_marker['sha256']:
            raise RuntimeError('Unwarmed workload checkpoint hash changed')
        warmup_payload = torch.load(warmup_path, map_location='cpu', weights_only=False)
        archive = OLD_ROOT/'rgb_seed81/replay'
        rows = json.loads((archive/'index.json').read_text(encoding='utf8'))['rows']
        source_hash = sha(__file__)
        protocol = dict(scope='bounded performance experiment, no formal training/selection',
                        created_unix=time.time(), config=vars(args), source=__file__, source_hash=source_hash,
                        workload=workload, historical_root=str(OLD_ROOT),
                        warmup_workload=dict(path=str(warmup_path),sha256=warmup_marker['sha256']),
                        archive_episodes=len(rows), archive_transitions=sum(r['transitions'] for r in rows),
                        historical_archive_readonly=True, batch_size=4,
                        tf32_matmul=False, tf32_cudnn=False, torch=torch.__version__,
                        notes=['GPU-resident replay is a capacity/timing fixture, not a production runner',
                               'Original v2/v3/GPU files remain frozen; no prior output is overwritten',
                               'No disk cache flushing; host replay uses normal OS file cache',
                               'Evaluation timing uses training rooms and full successor V/trace transfer'])
        write_json(output/'PROTOCOL.json', protocol)
        report = dict(protocol=protocol, phases=[])
        geometry = GPURGBGeometry(cfg)
        start = time.perf_counter()
        resident = ResidentReplay(archive, rows, geometry)
        torch.cuda.synchronize()
        report['fixture_load_seconds'] = time.perf_counter()-start
        report['resident_raw_bytes'] = resident.resident_bytes
        telemetry = GPUUtilization()
        start=time.perf_counter(); time.sleep(3); end=time.perf_counter()
        report['idle'] = telemetry.summarize(start,end)
        for name in args.pipelines:
            if name=='v2_reference':
                active = v2._new_learner(cfg, v2.RGBRunConfig(seed=81, group_seeds=(81,)))
                v2._restore_for_evaluation(active,payload)
                g = RGBGeometry(cfg, 'cuda')
                reader = host_reader(v2.RawRGBReplay,output/name,archive,rows,
                                     cache_episodes=2,geometry_cache=16)
            else:
                active = learner
                if name=='gpu_resident':
                    g, reader = geometry, resident
                else:
                    g = BatchedRGBGeometry(cfg,'cuda',depth_batch_size=4) if name=='v3_cpu' else GPUGeometryBridge(geometry)
                    reader = host_reader(FastRawRGBReplay,output/name,archive,rows,cache_episodes=32,
                                         geometry_cache=128,replay_cache_bytes=512<<20,geometry_cache_bytes=128<<20)
            report['phases'].extend(timed_learning(output,name,active,payload,warmup_payload,reader,g,args.seconds,args.repeats,telemetry))
            write_json(output/'PROGRESS.json',report)
            if name=='v2_reference':
                del active
            if reader is not resident:
                del reader, g
            gc.collect(); torch.cuda.empty_cache()
        for size in args.evaluation_batches:
            report['phases'].extend(timed_evaluation(output,cfg,scenes,learner,geometry,size,args.seconds,args.repeats,telemetry))
            write_json(output/'PROGRESS.json',report)
        telemetry.close(); report['telemetry']=telemetry.rows; telemetry=None
        if sha(__file__) != source_hash:
            raise RuntimeError('Timing script changed during measurement')
        frozen_model = workload['path']
        report['frozen_model_unchanged'] = sha(frozen_model)==workload['sha256']
        report['model_restored_hash'] = tensor_state_hash(learner)
        write_json(output/'PROFILE.json', report)
        write_json(output/'FINAL_STATE.json',dict(status='complete',phases=len(report['phases'])))
    except BaseException as error:
        write_json(output/'FINAL_STATE.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if telemetry is not None:
            telemetry.close(); write_json(output/'PARTIAL_TELEMETRY.json',telemetry.rows)


if __name__=='__main__':
    main()
