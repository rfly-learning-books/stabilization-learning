"""Fresh replay-based RGB certificate run. No resume or overwrite operation."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'stabilization-gym',ROOT/'stabilization-learning-control'):
    sys.path.insert(0,str(p))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

if __name__=='__main__':
    raise SystemExit('Prototype disabled: its physical-action Q and learned-response candidate controller differ from the requested eta-conditioned return Critic. No formal training is authorized through this prototype.')
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('train',))
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from extensions.rgb_certificate_replay_v1.runner import train
    sys.exit(train(args.output,json.loads(Path(args.config).read_text(encoding='utf-8-sig'))))
