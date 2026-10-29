"""Independent old/new execution and optimizer parity, before any new training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
import importlib.metadata
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'stabilization-gym'), str(ROOT / 'stabilization-learning-control')]
import numpy as np
import torch
from stabilization_gym.envs.tasks.original_protocols import make_original_task, load_push_source, digest
from stabilization_learning.algos.pytorch.full_m_ppo.trainer import write_json


def exact(a, b):
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def initialized(engine, task, device, checkpoint=None):
    saved = torch.load(checkpoint or task.reference / 'initial.pt', map_location=device, weights_only=True)
    actor = engine.FullMActor(task.cfg).to(device); actor.load_state_dict(saved['actor'])
    critic = engine.Critic(task.cfg).to(device); critic.load_state_dict(saved['critic'])
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=2e-4)
    optimizer.load_state_dict(saved['optimizer'])
    engine.restore_rng(saved['rng'])
    return actor, critic, optimizer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default='runs/original_push_parity_20261022')
    args = ap.parse_args(); out = ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    task = make_original_task('push_any_shape')
    legacy_root = Path('D:/书稿论文/push_any_shape_compact_package/scripts/so101_vendor/dl_push')
    old = load_push_source(legacy_root, '_audit_historical_push')
    old_bank = old.ModeBank('D:/书稿论文/_依赖/DL-Push/runs/mode_library_geometric32_20260913', device=args.device)
    bank = task.bank(args.device)
    checks = {'source_hashes_identical': old.source_hashes() == task.engine.source_hashes(),
              'calibration_identical': exact(old_bank.C, bank.C) and exact(old_bank.valid, bank.valid),
              'fingerprints_identical': old_bank.fingerprints == bank.fingerprints}

    # One old and one migrated complete hard-controller trajectory bank.
    saved = torch.load(task.reference / 'best.pt', map_location=args.device, weights_only=True)
    old_actor = old.FullMActor(task.cfg).to(args.device); old_actor.load_state_dict(saved['actor'])
    actor, _ = task.load_actor(task.reference / 'best.pt', args.device)
    old_report = old.evaluate(old_actor, old_bank, task.cfg, task.run_config['eval_shapes'],
                              output=out / 'old_execution')
    new_report = task.evaluate(actor, bank, output=out / 'migrated_execution')
    with np.load(out / 'old_execution/trajectory.npz') as a, np.load(out / 'migrated_execution/trajectory.npz') as b:
        checks['all_trajectory_fields_identical'] = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
        trace = {k: a[k].copy() for k in a.files}
    checks['all_episode_outcomes_identical'] = old_report['episodes'] == new_report['episodes']
    historic = json.loads((task.reference / 'final_best/report.json').read_text())
    historical_comparison = dict(old_success=historic['success_rate'], migrated_success=new_report['success_rate'],
                                old_cumulative_error=historic['mean_cumulative_error'],
                                migrated_cumulative_error=new_report['mean_cumulative_error'],
                                old_mean_switches=historic['mean_switches'], new_mean_switches=new_report['mean_switches'])
    checks['historical_32_case_success_reproduced'] = historic['success_rate'] == new_report['success_rate']

    # Public environment interface must produce the same actions/gradient.
    control = task.new_environment(bank)
    data = control.reset(seed=20261022)
    (mode, speed, recovery, fail), cand = task.calculate_action_from_certificate(actor, data, bank, control)
    direct = task.engine.candidates(actor, data, bank, task.cfg, speed_points=task.cfg.eval_speed_points)
    checks['public_candidate_contract_identical'] = all(exact(cand[k], direct[k]) for k in cand)
    ctx = task.metric_context(data, bank)
    D = ctx['D'].detach().clone().requires_grad_(True)
    condition = actor.encoder(bank.C, bank.valid, task.cfg.length_scale)[data['shape']]
    V = actor.value(D, condition); grad = torch.autograd.grad(V.sum(), D)[0]
    eps = 1e-3
    fd = torch.stack([(actor.value(D.detach() + eps * torch.eye(3, device=args.device)[j], condition)
                       - actor.value(D.detach() - eps * torch.eye(3, device=args.device)[j], condition)) / (2 * eps)
                      for j in range(3)], -1)
    checks['full_state_gradient_finite_difference'] = bool(torch.allclose(grad, fd, atol=3e-3, rtol=2e-3))
    full_minus_fixed = float((grad - 2 * torch.einsum('bij,bj->bi', cand['M'], D)).abs().max())
    checks['state_dependence_is_nonzero'] = full_minus_fixed > 1e-6
    control.env.close()

    # Same sampled transition batch, gradients, optimizer and learned weights.
    products = []
    for engine, b in [(old, old_bank), (task.engine, bank)]:
        a, c, opt = initialized(engine, task, args.device)
        ctrl = engine.FullMControlEnv(b, task.cfg, task.run_config['train_shapes'], 32)
        rollout, stats = engine.collect(a, c, ctrl, 100000)
        losses = engine.update_ppo(a, c, opt, rollout, b, task.cfg, 2, 128)
        products.append(dict(rollout=rollout, losses=losses, actor=copy.deepcopy(a.state_dict()),
                             critic=copy.deepcopy(c.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
                             rng=copy.deepcopy(engine.rng_state())))
        if engine is task.engine:
            task.engine.save_checkpoint(out / 'nonempty_optimizer.pt', a, c, opt, bank, task.cfg,
                                        1, task.run_config)
        ctrl.env.close()
    for key in ('rollout', 'losses', 'actor', 'critic', 'optimizer', 'rng'):
        checks['one_update_' + key + '_identical'] = exact(products[0][key], products[1][key])

    # Non-empty optimizer resume: two independent loads must produce the same
    # next collection AND update, not merely the same model output.
    continued = []
    for engine, b in [(old, old_bank), (task.engine, bank)]:
        a, c, opt = initialized(engine, task, args.device, out / 'nonempty_optimizer.pt')
        ctrl = engine.FullMControlEnv(b, task.cfg, task.run_config['train_shapes'], 32)
        roll, stats = engine.collect(a, c, ctrl, 100001)
        loss = engine.update_ppo(a, c, opt, roll, b, task.cfg, 2, 128)
        continued.append(dict(rollout=roll, losses=loss, actor=copy.deepcopy(a.state_dict()),
                              critic=copy.deepcopy(c.state_dict()), optimizer=copy.deepcopy(opt.state_dict())))
        ctrl.env.close()
    checks['nonempty_optimizer_next_update_identical'] = exact(*continued)
    checks['historical_sources_unchanged'] = old.source_hashes() == task.manifest['source_sha256']
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'sympy', 'numpy', 'shapely', 'gymnasium', 'Pillow', 'imageio')}
    result = dict(passed=all(checks.values()), checks=checks, torch=str(torch.__version__), device=args.device,
                  versions=versions, python=sys.version, gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                  full_gradient_minus_fixed_max=full_minus_fixed, historical_comparison=historical_comparison,
                  original_weights_training_update=saved['next_update'],
                  note='Old-source vs copied-source equality is tested on this host; historical host is compared separately.')
    write_json(out / 'AUDIT.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit('Migration parity failed; no production training is authorized by this audit')


if __name__ == '__main__':
    main()
