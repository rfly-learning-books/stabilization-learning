"""Train the preserved task protocol from the independent project directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'stabilization-learning-control'), str(ROOT / 'stabilization-gym')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/push_any_shape_original.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Only for checkpoint equivalence checks; does not change training protocol')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config['task'] == 'visual_servo':
        from stabilization_gym.envs.tasks.original_protocols import make_original_task
        make_original_task('visual_servo').require_assets()
    elif config['task'] == 'push_any_shape':
        from stabilization_learning.algos.pytorch.full_m_ppo import train_original_push
        train_original_push(config, args.output, resume=args.resume, stop_after=args.stop_after)
    else:
        raise ValueError(config['task'])


if __name__ == '__main__':
    main()
