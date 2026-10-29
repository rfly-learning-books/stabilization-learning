"""Read-only verification of migrated training and fixed-test evidence.

Uses CPU only. Physical motion is independently integrated by Gauss quadrature,
not by the training plant's sinc integrator. Never changes trained checkpoints.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'stabilization-gym'), str(ROOT / 'stabilization-learning-control')]
import numpy as np
import torch
from audit_original_push_migration import exact
from stabilization_gym.envs.tasks.original_protocols import make_original_task, digest
from stabilization_learning.algos.pytorch.full_m_ppo.trainer import write_json


def history(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        result[row['update']] = row
    return result


def compare_history(original, migrated):
    differences = []
    for update, row in migrated.items():
        for group in ('collection', 'losses', 'hard_eval'):
            for key, value in row.get(group, {}).items():
                if key == 'seconds' or key not in original[update].get(group, {}):
                    continue
                if value != original[update][group][key]:
                    differences.append(dict(update=update, group=group, key=key,
                                            old=original[update][group][key], new=value))
    return differences


def physical_trace(task, folder, rows):
    report = json.loads((folder / 'report.json').read_text())
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as f:
        tr = {k: f[k] for k in f.files}
    order = report['shapes']
    row_by_shape = {r['shape']: r for r in rows}
    # The T regression is excluded from fresh-start tests by the declared protocol.
    ids = [i for i, name in enumerate(order) if name != 'T']
    initial = np.array([row_by_shape[order[i]]['initial_pose'] for i in ids])
    target = np.array([row_by_shape[order[i]]['target_pose'] for i in ids])
    poses = tr['pose_next'][:, ids]
    before = np.concatenate([initial[None], poses[:-1]])
    active = tr['active'][:, ids].astype(bool)
    modes = tr['action'][:, ids, 0].astype(int)
    speed = np.minimum(tr['action'][:, ids, 1], task.cfg.u_max)
    assert np.all((modes >= 0) & (modes < 12))
    assert np.all(speed >= 0) and np.all(tr['action'][:, ids, 1] <= task.cfg.u_max + 1e-7)
    libraries = []
    validity = []
    for i in ids:
        with np.load(task.library / order[i] / 'library/library.npz') as f:
            libraries.append(f['c'].T)
            validity.append(f['valid'])
    C = np.asarray(libraries)
    selected = C[np.arange(len(ids))[None], modes]
    valid = np.asarray(validity)[np.arange(len(ids))[None], modes]
    assert valid[active].all()
    vx, vy, omega = (selected * speed[..., None]).transpose(2, 0, 1)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    moments = (nodes + 1) * task.cfg.dt / 2
    theta = before[..., 2, None] + omega[..., None] * moments
    dx = (np.cos(theta) * vx[..., None] - np.sin(theta) * vy[..., None]) @ weights * task.cfg.dt / 2
    dy = (np.sin(theta) * vx[..., None] + np.cos(theta) * vy[..., None]) @ weights * task.cfg.dt / 2
    predicted = before + np.stack([dx, dy, omega * task.cfg.dt], -1) * active[..., None]
    max_step_error = float(np.abs(predicted - poses).max())
    assert max_step_error < 1e-10, max_step_error
    M = tr['M'][:, ids].astype(float)
    L = tr['L'][:, ids].astype(float)
    eigenvalues = np.linalg.eigvalsh(M[active])
    assert eigenvalues.min() >= task.cfg.eigen_floor - 1e-6
    assert eigenvalues.max() <= 3 - 2 * task.cfg.eigen_floor + 1e-6
    assert np.allclose(np.trace(M, axis1=-2, axis2=-1), 3, atol=1e-6)
    A = L @ np.swapaxes(L, -1, -2)
    reconstructed = task.cfg.eigen_floor * np.eye(3) + (3 - 3 * task.cfg.eigen_floor) * A / np.trace(A, axis1=-2, axis2=-1)[..., None, None]
    matrix_error = float(np.abs(reconstructed - M).max())
    assert matrix_error < 1e-6
    e = tr['e'][:, ids].astype(float)
    D = e.copy()
    D[..., 2] = np.sqrt(2) * np.sin(e[..., 2] / 2)
    value = np.einsum('...i,...ij,...j->...', D, M, D)
    assert np.allclose(value, tr['V'][:, ids], rtol=3e-6, atol=2e-6)
    integrated = np.minimum(np.sum(((poses[..., :2] - target[None, :, :2]) / task.cfg.length_scale)**2, -1)
                            + 1 - np.cos(poses[..., 2] - target[None, :, 2]), task.cfg.error_cap)
    costs = (integrated * active * task.cfg.dt).sum(0)
    expected = np.array([row_by_shape[order[i]]['cumulative_error'] for i in ids])
    assert np.allclose(costs, expected, rtol=1e-9, atol=1e-10)
    for j, i in enumerate(ids):
        ep = row_by_shape[order[i]]
        assert int(active[:, j].sum()) == ep['physical_steps'] == ep['steps']
        if ep['success']:
            used = np.flatnonzero(active[:, j])[-task.cfg.hold_steps:]
            assert len(used) == task.cfg.hold_steps
            assert np.all(speed[used, j] == 0)
            assert np.all(np.linalg.norm(poses[used, j, :2] - target[j, :2], axis=-1) <= task.cfg.position_tol)
            assert np.all(np.abs(poses[used, j, 2] - target[j, 2]) <= task.cfg.angle_tol)
            assert np.all(tr['coverage'][used, i] >= task.cfg.coverage_tol)
    return dict(physical_steps=int(active.sum()), max_step_error=max_step_error,
                max_cholesky_reconstruction_error=matrix_error,
                eigen_min=float(eigenvalues.min()), eigen_max=float(eigenvalues.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    out = Path(args.run).resolve()
    if not (out / 'complete.json').is_file():
        raise RuntimeError('Wait for the fixed evaluation to complete')
    torch.set_num_threads(1)
    task = make_original_task('push_any_shape')
    config = json.loads((out / 'protocol.json').read_text())['config']
    result = json.loads((out / 'analysis/RESULTS.json').read_text(encoding='utf-8'))
    old_history = history(task.reference / 'history.jsonl')
    new_history = history(out / 'history.jsonl')
    checks = {'all_1000_updates_present': set(new_history) == set(range(1, config['updates'] + 1)),
              'transition_count_matches_logs': result['training']['transitions'] == sum(r['collection']['transitions'] for r in new_history.values()),
              'sources_unchanged': task.engine.source_hashes() == task.manifest['source_sha256']}
    differences = compare_history(old_history, new_history)
    reproduction = {'history_difference_count': len(differences), 'history_difference_examples': differences[:10]}
    checkpoint_hashes_before = {n: digest(out / n) for n in ('initial.pt', 'best.pt', 'last.pt')}
    for name in ('best.pt', 'last.pt'):
        a = torch.load(task.reference / name, map_location='cpu', weights_only=True)
        b = torch.load(out / name, map_location='cpu', weights_only=True)
        reproduction[name] = {k: exact(a[k], b[k]) for k in ('actor', 'critic', 'optimizer', 'rng', 'next_update', 'best_key')}
        reproduction[name]['best_key_max_absolute_difference'] = float(np.max(np.abs(np.asarray(a['best_key']) - np.asarray(b['best_key']))))
        reproduction[name]['best_key_equal_within_1e_12'] = bool(np.allclose(a['best_key'], b['best_key'], rtol=0, atol=1e-12))
    asset_manifest = json.loads((ROOT / 'migration/original_protocol_assets.json').read_text(encoding='utf-8'))
    checks['all_copied_assets_match_provenance'] = all(digest(ROOT / r['destination']) == r['sha256'] for r in asset_manifest['copied_files'])
    verification = {}
    for method, rows in result['episodes'].items():
        keys = {(r['seed'], r['shape']) for r in rows}
        checks[method + '_310_unique_cases'] = len(keys) == len(rows) == 310
        traces = []
        for seed in config['test_seeds']:
            selected = [r for r in rows if r['seed'] == seed]
            traces.append(physical_trace(task, out / f'comparison/{method}/seed_{seed}', selected))
        verification[method] = dict(physical_steps=sum(t['physical_steps'] for t in traces),
                                    max_physical_step_error=max(t['max_step_error'] for t in traces),
                                    min_eigenvalue=min(t['eigen_min'] for t in traces),
                                    max_eigenvalue=max(t['eigen_max'] for t in traces))
    trajectory_equal = []
    for seed in config['test_seeds']:
        with np.load(out / f'comparison/migrated_best/seed_{seed}/trajectory.npz') as a, np.load(out / f'comparison/historical_best/seed_{seed}/trajectory.npz') as b:
            trajectory_equal.append(set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files))
    reproduction['all_new_start_trajectories_equal_to_old_best'] = all(trajectory_equal)
    checks['verification_did_not_mutate_models'] = all(digest(out / n) == v for n, v in checkpoint_hashes_before.items())
    report = dict(passed=all(checks.values()), checks=checks, historical_reproduction=reproduction,
                  independent_physics=verification, checkpoint_hashes=checkpoint_hashes_before,
                  notes=['Gauss quadrature checks physical transitions without calling the original integrator.',
                         'No GPU use, optimization, model selection or checkpoint changes.',
                         'Checkpoint containers have different file hashes. Tensor equality is checked directly; development-score metadata is also reported with its exact numerical difference.'])
    write_json(out / 'analysis/FINAL_AUDIT.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit('Final verification failed')


if __name__ == '__main__':
    main()
