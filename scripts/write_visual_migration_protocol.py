"""Record the independent offline migration, never relax the original protocol."""
import datetime
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/visual_original_offline_migration_20261022_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [
        'stabilization-gym/stabilization_gym/envs/tasks/original_protocols.py',
        'stabilization-gym/stabilization_gym/envs/tasks/original_visual.py',
        'migration/visual_source_bundle.json',
        'migration/visual_initialization_bundle.json',
        'migration/visual_recovery_candidates/recovery_report.json',
        'assets/original_visual_initialization/protocol.json',
        'scripts/write_visual_migration_protocol.py',
    ]
    record = dict(
        schema=1, kind='offline_original_visual_migration_audit',
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        independent_output=str(OUT),
        original_experiment='ae_derivative_critic_seed41_20261019_v1',
        complete_training=False, closed_loop_evaluation=False,
        original_protocol_preserved=True, original_files_modified=False,
        fingerprints={name: digest(ROOT / name) for name in files},
        initialization='Import complete original derivative initial.pt learner, all optimizers, counters and RNG. Build architecture without deleted preliminary AE. Initial baseline copied from retained initial AE.',
        path_changes={
            'legacy_root': 'assets/recovered_visual_legacy',
            'initial_autoencoder': None,
            'scene_assets': 'third_party/visual_original/assets/scene_dependencies',
            'old_depth_audit': 'third_party/visual_original/assets/old_depth_audit/model',
            'fastdepth': 'third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1',
        },
        preserved_computations=['shared 16D RGB AE', 'goal-only Cholesky metric',
            'original fixed reference normalized-command JVP', 'original weighted damped policy and saturation',
            'original derivative Critic and AE/M optimizer updates'],
        online_inputs='RGB-derived features; goal FastDepth estimate and camera calibration. No true pose or true depth.',
        blocked_dependency=dict(file='legacy/scene_builders.py', missing_lines=[131, 285],
            expected_sha256='5c68541538a836abb17373de99445e42c2c3bd742aedb59471b8fc53a43bf317'),
        prohibited_claims=['new closed-loop success rate', 'complete training reproduction',
            'pixel-identical original renderer', 'global Lyapunov stability guarantee'],
        runtime=dict(python=platform.python_version(), platform=platform.platform()),
    )
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old['fingerprints'] != record['fingerprints']:
            raise RuntimeError('Offline protocol sources changed; preserve old evidence and use a new run')
        return
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'protocol': str(path), 'sha256': digest(path)}))


if __name__ == '__main__':
    main()
