"""Verify the actual migrated trainer across a non-empty resume boundary."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'stabilization-gym'), str(ROOT / 'stabilization-learning-control'), str(ROOT / 'scripts')]
import torch
from audit_original_push_migration import exact, initialized
from stabilization_gym.envs.tasks.original_protocols import make_original_task, load_push_source, digest
from stabilization_learning.algos.pytorch.full_m_ppo.trainer import train_original_push, write_json


def main():
    config = json.loads((ROOT / 'configs/push_any_shape_original.json').read_text())
    output = ROOT / 'runs/push_original_migration_seed1_20261022_v2'
    if output.exists():
        raise FileExistsError(output)
    train_original_push(config, output, stop_after=2)
    before = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
    train_original_push(config, output, resume=True, stop_after=3)
    after = torch.load(output / 'last.pt', map_location=config['device'], weights_only=True)
    task = make_original_task('push_any_shape')
    old = load_push_source(Path('D:/书稿论文/push_any_shape_compact_package/scripts/so101_vendor/dl_push'),
                           '_independent_old_push_resume')
    bank = old.ModeBank('D:/书稿论文/_依赖/DL-Push/runs/mode_library_geometric32_20260913', device=config['device'])
    actor, critic, optimizer = initialized(old, task, config['device'])
    control = old.FullMControlEnv(bank, task.cfg, task.run_config['train_shapes'], 32)
    for update in range(3):
        rollout, stats = old.collect(actor, critic, control, 100000 + update)
        old.update_ppo(actor, critic, optimizer, rollout, bank, task.cfg, 2, 128)
    control.env.close()
    checks = {
        'resume_started_with_nonempty_optimizer': bool(before['optimizer']['state']),
        'resume_advanced_2_to_3': before['next_update'] == 2 and after['next_update'] == 3,
        'actor_matches_uninterrupted_original': exact(after['actor'], actor.state_dict()),
        'critic_matches_uninterrupted_original': exact(after['critic'], critic.state_dict()),
        'optimizer_matches_uninterrupted_original': exact(after['optimizer'], optimizer.state_dict()),
        'rng_matches_uninterrupted_original': exact(after['rng'], old.rng_state()),
        'original_source_hashes_unchanged': old.source_hashes() == task.engine.source_hashes(),
    }
    # A real changed configuration must still be rejected before any update.
    digest_before = digest(output / 'last.pt')
    changed = copy.deepcopy(config)
    changed['seed'] += 1
    try:
        train_original_push(changed, output, resume=True, stop_after=4)
    except RuntimeError as exc:
        checks['changed_protocol_still_rejected'] = 'Protocol/source mismatch' in str(exc)
    else:
        checks['changed_protocol_still_rejected'] = False
    checks['rejected_resume_leaves_checkpoint_unchanged'] = digest_before == digest(output / 'last.pt')
    report = dict(passed=all(checks.values()), checks=checks, production_output=str(output),
                  checkpoint_update=3, checkpoint_sha256=digest_before,
                  previous_failed_run='push_original_migration_seed1_20261022_v1',
                  change='Canonical JSON comparison only; original training/controller source untouched')
    write_json(output / 'trainer_resume_audit.json', report)
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('Resume equivalence failed')


if __name__ == '__main__':
    main()
