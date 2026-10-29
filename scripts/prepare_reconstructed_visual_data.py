"""Re-render inherited training transitions for the declared rebuilt rooms.

Only training RGB is read/generated here. Actions and pose transitions remain
off-policy observations from the old bootstrap. This is not fresh PVS rollout.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stabilization-gym'))
from stabilization_gym.envs.tasks.original_visual_online import load_reconstructed_runtime
from stabilization_gym.envs.tasks.original_visual import file_sha

TRIPLET_SEED = 20261023041


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def emit(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def resources(out):
    import subprocess
    if shutil.disk_usage(out).free / 2**30 < 50:
        raise RuntimeError('Free disk below 50 GiB')
    value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    used, total = map(float, value.splitlines()[0].split(','))
    if used / total > .65:
        raise RuntimeError('GPU total usage exceeds 65 percent')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-root', required=True)
    parser.add_argument('--scene-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    resources(out)
    adapter = load_reconstructed_runtime(ROOT, args.legacy_root, args.scene_audit)
    source = adapter.bundle
    payload = adapter.checkpoint('initial.pt')
    rows = copy.deepcopy(payload['replay_rows'])
    assert len(rows) == 48 and sum(row['length'] for row in rows) == 2822
    assert all(row['split'] == 'train' for row in rows)
    source_manifest = json.loads((ROOT / 'migration/visual_initialization_bundle.json').read_text(encoding='utf-8'))
    source_hashes = {row['relative']: row['sha256'] for row in source_manifest['entries']}
    fingerprint = dict(script=file_sha(Path(__file__)), scene_audit=file_sha(Path(args.scene_audit)),
                       online_adapter=file_sha(ROOT / 'stabilization-gym/stabilization_gym/envs/tasks/original_visual_online.py'),
                       source_initial=file_sha(source / 'initial.pt'), triplet_seed=TRIPLET_SEED)
    progress_path = out / 'preparation_progress.json'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if progress['fingerprint'] != fingerprint:
            raise RuntimeError('Preparation inputs changed; use a new output directory')
    else:
        if any(out.iterdir()):
            raise RuntimeError('Output must be empty on first preparation')
        progress = dict(fingerprint=fingerprint, completed_shards={})
        write_json(progress_path, progress)
    for name in ('scenes.json', 'evaluation_starts.json', 'training_anchors.json'):
        if file_sha(source / name) != source_hashes[name]:
            raise RuntimeError('Source metadata changed: ' + name)
        shutil.copy2(source / name, out / name)
    scenes = {row['id']: row for row in json.loads((out / 'scenes.json').read_text(encoding='utf-8'))['scenes']}
    (out / 'replay').mkdir(exist_ok=True)
    (out / 'references').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.50)
    depth_model = adapter.core.FastDepth(adapter.cfg, 'cuda')
    room_module = adapter.core.legacy.module(adapter.cfg, 'room_env')
    from lac_vs.environment import valid_start
    diagnostics = []
    began = time.monotonic()
    for index, row in enumerate(rows):
        name = row['file']
        source_path, target = source / 'replay' / name, out / 'replay' / name
        if file_sha(source_path) != source_hashes['replay/' + name]:
            raise RuntimeError('Source bootstrap changed: ' + name)
        if name in progress['completed_shards']:
            done = progress['completed_shards'][name]
            if file_sha(target) != done['sha256']:
                raise RuntimeError('Prepared shard changed: ' + name)
            if file_sha(out / done['reference']) != done['reference_sha256']:
                raise RuntimeError('Prepared reference changed: ' + name)
            continue
        if (out / 'STOP').exists():
            emit(event='data_preparation_stopped', completed=len(progress['completed_shards']))
            return
        scene = scenes[row['scene_id']]
        assert scene['split'] == 'train'
        goal_id = row['goal']
        frame = room_module.ReferenceFrame(np.asarray(scene['refs'][goal_id]['goal_matrix']))
        with np.load(source_path, allow_pickle=False) as saved:
            original = {key: saved[key] for key in saved.files}
        count = row['length']
        if original['offset'].shape != (count + 1, 4) or original['rgb'].shape[0] != count + 1:
            raise RuntimeError('Inherited state array length mismatch')
        for key in ('action', 'cost', 'terminal', 'truncated', 'previous_action'):
            if len(original[key]) != count:
                raise RuntimeError('Inherited transition array length mismatch: ' + key)
        expected_previous = np.concatenate([np.zeros((1, 4), np.float32), original['action'][:-1]])
        if not np.array_equal(original['previous_action'], expected_previous):
            raise RuntimeError('Inherited previous actions are inconsistent')
        expected_truncated = np.zeros(count, bool)
        expected_truncated[-1] = not bool(original['terminal'][-1])
        if not np.array_equal(original['truncated'], expected_truncated) or original['terminal'][:-1].any():
            raise RuntimeError('Inherited terminal/truncation boundary mismatch')
        step_errors = []
        for offset, command, successor in zip(original['offset'][:-1], original['action'], original['offset'][1:]):
            state = dict(position=frame.position_from_offset(offset), yaw=float(offset[3]), body0=frame.body0)
            next_state = room_module.PlantP0.step_state(state, command)
            reconstructed = np.r_[frame.body0.T @ (next_state['position'] - frame.goal[:3, 3]), next_state['yaw']]
            step_errors.append(float(np.max(np.abs(reconstructed - successor))))
        physics_error = max(step_errors)
        if physics_error > 1e-6:
            raise RuntimeError('Inherited P0 transition does not match preserved dynamics')
        instance = adapter.new_environment(scene)
        try:
            if instance.obstacles != scene['obstacles'] or instance.dims != scene['dims']:
                raise RuntimeError('Rebuilt collision geometry changed')
            goal = instance.goal_rgb(frame)
            goal_depth = depth_model.predict(goal)
            offsets = original['offset']
            images = np.stack([instance.render_rgb(frame, frame.position_from_offset(offset), float(offset[3])) for offset in offsets])
            states = [frame.position_from_offset(offset) for offset in offsets]
            failed = np.array([instance.collides(position) or instance.outside_workspace(position) for position in states[1:]])
            errors = (offsets[:, :3] ** 2).sum(1) + (offsets[:, 3] / np.radians(25)) ** 2
            expected_cost = .1 * (errors[1:] + .02 * (original['action'] ** 2).sum(1)) + failed * 20
            cost_error = float(np.max(np.abs(expected_cost - original['cost'])))
            if not np.allclose(expected_cost, original['cost'], atol=2e-5, rtol=1e-5):
                raise RuntimeError('Inherited physical costs disagree with reconstructed scene')
            hold = 0
            for step, offset in enumerate(offsets[1:]):
                good = np.linalg.norm(offset[:3]) < .02 and abs(offset[3]) < np.radians(1)
                hold = hold + 1 if good else 0
                if bool(original['terminal'][step]) != bool(failed[step] or hold >= 11):
                    raise RuntimeError('Inherited terminal flags disagree with reconstructed geometry')
            rng = np.random.default_rng(np.random.SeedSequence([TRIPLET_SEED, index]))
            center, minus, plus, magnitudes, delta_values, center_indices = [], [], [], [], [], []
            for step in range(0, len(original['action']), 10):
                if len(center) >= adapter.cfg['goal_fixed']['smooth_triplet_limit_per_episode']:
                    break
                delta = rng.normal(size=4)
                delta[:3] *= .012
                delta[3] *= np.radians(.6)
                low, high = offsets[step] - delta, offsets[step] + delta
                if valid_start(adapter.cfg, scene, frame, low) and valid_start(adapter.cfg, scene, frame, high):
                    center.append(images[step])
                    minus.append(instance.render_rgb(frame, frame.position_from_offset(low), low[3]))
                    plus.append(instance.render_rgb(frame, frame.position_from_offset(high), high[3]))
                    magnitudes.append(np.linalg.norm(delta))
                    delta_values.append(delta)
                    center_indices.append(step)
            if not center:
                center = minus = plus = [images[0]]
                magnitudes, delta_values, center_indices = [.01], [np.zeros(4)], [0]
            # Unused old image-derived telemetry is discarded; physical labels
            # and every field consumed by the unchanged Replay.sample remain.
            arrays = {key: original[key] for key in ('offset', 'action', 'cost', 'terminal', 'truncated', 'previous_action')}
            arrays.update(rgb=images, goal_rgb=goal, goal_depth=goal_depth, collection_depth=goal_depth.copy(),
                          rotation=frame.camera_from_body.astype(np.float32),
                          smooth_center=np.asarray(center, np.uint8), smooth_minus=np.asarray(minus, np.uint8),
                          smooth_plus=np.asarray(plus, np.uint8), smooth_step=np.asarray(magnitudes, np.float32),
                          smooth_delta=np.asarray(delta_values, np.float64), smooth_center_index=np.asarray(center_indices, np.int32))
            temp = target.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            temp.replace(target)
            reference = out / 'references' / f'{scene["id"]}_{goal_id:02d}.npz'
            with reference.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, rgb=goal, depth=goal_depth, camera_from_body=frame.camera_from_body)
            reference.with_suffix('.tmp').replace(reference)
            progress['completed_shards'][name] = dict(sha256=file_sha(target), source_sha256=file_sha(source_path),
                transitions=row['length'], triplets=len(center), cost_max_abs=cost_error,
                physics_max_abs=physics_error, reference=reference.relative_to(out).as_posix(),
                reference_sha256=file_sha(reference),
                train_rgb_mean_abs_change=float(np.abs(images.astype(np.float32) - original['rgb']).mean()))
            write_json(progress_path, progress)
            emit(event='bootstrap_rerendered', index=index + 1, total=len(rows), file=name, transitions=row['length'])
        finally:
            instance.close()
        resources(out)
    write_json(out / 'replay/index.json', rows)
    for row in rows:
        done = progress['completed_shards'][row['file']]
        with np.load(out / 'replay' / row['file'], allow_pickle=False) as data, np.load(out / done['reference'], allow_pickle=False) as reference:
            if not np.array_equal(data['goal_rgb'], reference['rgb']) or not np.array_equal(data['goal_depth'], reference['depth']):
                raise RuntimeError('Prepared replay/reference inconsistency')
    # No validation/test RGB was generated or used to fit the renderer/data.
    entries = []
    for path in sorted(out.rglob('*')):
        if path.is_file() and (path.name in ('scenes.json', 'evaluation_starts.json', 'training_anchors.json')
                              or path.parent.name in ('replay', 'references')):
            entries.append(dict(relative=path.relative_to(out).as_posix(), sha256=file_sha(path), bytes=path.stat().st_size))
    record = dict(schema=1, protocol_kind='reconstructed_environment', original_rgb_equivalent=False,
        complete=True, created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_initial_checkpoint_sha256=fingerprint['source_initial'], scene_audit_sha256=fingerprint['scene_audit'],
        preparation_fingerprint=fingerprint, bootstrap=dict(shards=48, transitions=2822, train_only=True),
        trajectory_origin='Inherited original off-policy P0 pose/action transitions, not newly collected PVS actions',
        image_changes='Current/next/goal RGB and FastDepth estimates regenerated with frozen reconstructed renderer',
        anchors='Original initial model center/scale and original training anchors unchanged',
        smooth_triplets=dict(seed=TRIPLET_SEED, original_delta_unavailable=True, generated_from_training_poses=True,
                             original_sigma_position_m=.012, original_sigma_yaw_deg=.6),
        discarded_old_telemetry=['features', 'V', 'M', 'L', 'J_reference', 'H'],
        validation_test_RGB_used=False, entries=entries, elapsed_seconds=time.monotonic() - began)
    write_json(out / 'DATA_MANIFEST.json', record)
    emit(event='data_preparation_complete', shards=48, transitions=2822, output=str(out), elapsed_seconds=record['elapsed_seconds'])


if __name__ == '__main__':
    main()
