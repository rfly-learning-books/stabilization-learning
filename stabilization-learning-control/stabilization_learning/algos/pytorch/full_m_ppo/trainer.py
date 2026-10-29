"""Original per-step certificate-induced PPO, with independent run safeguards.

Collection, original shaped rewards, GAE, PPO, auxiliary losses and validation
selection are delegated to the hash-verified original implementation. This is
separate from the episode-fixed metric-PPO baseline.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from stabilization_gym.envs.tasks.original_protocols import make_original_task, digest


def write_json(path, record):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(p)


@contextmanager
def writer_lock(path):
    import msvcrt
    f = Path(path).open('a+b')
    f.seek(0)
    if not f.read(1):
        f.write(b'1'); f.flush()
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        raise RuntimeError(f'Another writer holds {path}')
    try:
        yield
    finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()


def check_resources(out, config, *, scan_size=False):
    free = shutil.disk_usage(out).free / 2**30
    if free < config['minimum_free_gib']:
        return f'disk_free_below_limit:{free:.3f}GiB'
    if scan_size:
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 2**30
        if size > config['maximum_run_gib']:
            return f'run_size_exceeds_limit:{size:.3f}GiB'
    if config['device'] == 'cuda':
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        used, total = map(float, result.stdout.splitlines()[0].split(','))
        if used / total > config['gpu_memory_fraction']:
            return f'gpu_memory_exceeds_limit:{used / total:.5f}'
    return None


def train_original_push(config, output, *, resume=False, stop_after=None):
    task = make_original_task('push_any_shape')
    native = task.engine
    out = Path(output).resolve()
    if str(torch.__version__) != config['torch_version']:
        raise RuntimeError(f"Original runtime required: {config['torch_version']}; got {torch.__version__}")
    audit_path = task.root / config['preflight_audit']
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['passed'] or audit['torch'] != str(torch.__version__):
        raise RuntimeError('Original-source/control/update parity must pass on this runtime first')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if config['device'] == 'cuda':
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    if out.exists() and not resume:
        raise FileExistsError(out)
    if resume and not (out / 'last.pt').is_file():
        raise FileNotFoundError('Resume requires last.pt in the same output directory')
    out.mkdir(parents=True, exist_ok=True)
    # Hash the actual adapter and learner, independent of installation depth.
    from stabilization_gym.envs.tasks import original_protocols
    source_files = [Path(__file__), Path(original_protocols.__file__)]
    protocol = dict(schema=1, config=config, original_manifest=task.manifest,
                    preflight_audit_sha256=digest(audit_path),
                    initial_sha256=digest(task.reference / 'initial.pt'),
                    adapter_sources={str(p.relative_to(task.root)): digest(p) for p in source_files},
                    certificate_contract=asdict(task.contract),
                    training='native soft candidate PPO; full state-dependent M gradient; original reward and losses',
                    selection='original fixed development bank only; no test selection',
                    historical_torch=task.manifest['torch'], current_torch=str(torch.__version__),
                    note='Same Torch/CUDA wheel; hardware, Python and numerical libraries are separately recorded, not claimed bit-identical to historical host')
    # Compare the serialized representation, including tuple -> JSON list.
    # All source hashes and configuration values remain part of the check.
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    marker = out / 'protocol.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != protocol:
            raise RuntimeError('Protocol/source mismatch: create a new protocol, never bypass this check')
    else:
        write_json(marker, protocol)
    runtime = dict(python=sys.version, executable=sys.executable,
                   packages={p: importlib.metadata.version(p) for p in
                             ('torch', 'sympy', 'numpy', 'shapely', 'gymnasium', 'Pillow', 'imageio')},
                   gpu=torch.cuda.get_device_name(0) if config['device']=='cuda' else None)
    if (out / 'runtime.json').exists():
        if json.loads((out / 'runtime.json').read_text(encoding='utf-8')) != runtime:
            raise RuntimeError('Resume runtime changed')
    else:
        write_json(out / 'runtime.json', runtime)
    for name in ('rollouts', 'evaluations'):
        (out / name).mkdir(exist_ok=True)
    for p in source_files:
        target = out / 'provenance' / p.relative_to(task.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(p):
            raise RuntimeError('Snapshot mismatch')
        if not target.exists():
            shutil.copy2(p, target)
    with writer_lock(task.root / 'runs/.original_push_gpu.lock'), writer_lock(out / '.writer.lock'):
        bank = task.bank(config['device'])
        checkpoint = out / 'last.pt' if resume else task.reference / 'initial.pt'
        actor, saved = task.load_actor(checkpoint, config['device'])
        if saved['run_config'] != task.run_config:
            raise RuntimeError('Original training configuration mismatch')
        if not resume and (saved['next_update'] != 0 or saved['optimizer']['state']):
            raise RuntimeError('Retraining must start before any optimizer update')
        critic = native.Critic(task.cfg).to(config['device'])
        critic.load_state_dict(saved['critic'])
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                                     lr=task.run_config['learning_rate'])
        optimizer.load_state_dict(saved['optimizer'])
        native.restore_rng(saved['rng'])
        next_update, best_key = saved['next_update'], saved['best_key']
        control = task.new_environment(bank)
        started = time.perf_counter()
        safety_reason = None
        try:
            if not resume:
                native.save_checkpoint(out / 'initial.pt', actor, critic, optimizer, bank,
                                       task.cfg, 0, task.run_config, None)
                native.save_checkpoint(out / 'last.pt', actor, critic, optimizer, bank,
                                       task.cfg, 0, task.run_config, None)
            for update in range(next_update, config['updates']):
                if (out / 'STOP').exists():
                    safety_reason = 'user_STOP'; break
                if time.perf_counter() - started > config['max_seconds']:
                    safety_reason = 'time_slice'; break
                safety_reason = check_resources(out, config, scan_size=(update % 50 == 0))
                if safety_reason:
                    break
                rollout, statistics = native.collect(actor, critic, control, config['seed'] * 100000 + update)
                native.save_rollout(out / 'rollouts' / f'update_{update:05d}.npz', rollout)
                losses = native.update_ppo(actor, critic, optimizer, rollout, bank, task.cfg,
                                           task.run_config['epochs'], task.run_config['minibatch'])
                next_update = update + 1
                record = dict(update=next_update, collection=statistics, losses=losses,
                              elapsed_seconds=time.perf_counter() - started)
                if next_update % config['eval_every'] == 0 or next_update == config['updates']:
                    report = task.evaluate(actor, bank, output=out / 'evaluations' / f'update_{next_update:05d}')
                    record['hard_eval'] = {k: report[k] for k in
                                          ('success_rate', 'failure_error', 'mean_switches', 'mean_no_action_steps', 'mean_cumulative_error')}
                    key = native.selection_key(report)
                    if best_key is None or tuple(key) > tuple(best_key):
                        best_key = key
                        native.save_checkpoint(out / 'best.pt', actor, critic, optimizer, bank,
                                               task.cfg, next_update, task.run_config, best_key)
                native.save_checkpoint(out / 'last.pt', actor, critic, optimizer, bank,
                                       task.cfg, next_update, task.run_config, best_key)
                with (out / 'history.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, allow_nan=False) + '\n')
                write_json(out / 'status.json', dict(status='TRAINING', next_update=next_update,
                                                     target_updates=config['updates'], best_key=best_key,
                                                     elapsed_seconds=time.perf_counter() - started))
                print(json.dumps(record), flush=True)
                if stop_after is not None and next_update >= stop_after:
                    safety_reason = 'requested_test_boundary'; break
            if next_update >= config['updates']:
                write_json(out / 'training_complete.json', dict(updates=next_update,
                            last_sha256=digest(out / 'last.pt'), best_sha256=digest(out / 'best.pt')))
                write_json(out / 'status.json', dict(status='TRAINING_COMPLETE', next_update=next_update,
                                                     evaluation='pending', best_key=best_key))
            else:
                write_json(out / 'status.json', dict(status='SAFE_STOP', next_update=next_update,
                                                     reason=safety_reason, best_key=best_key))
        except BaseException as exc:
            write_json(out / 'error.json', dict(type=type(exc).__name__, message=str(exc),
                                               traceback=traceback.format_exc(), next_update=next_update))
            write_json(out / 'status.json', dict(status='FAILED', next_update=next_update))
            raise
        finally:
            control.env.close()
    return out
