"""Fresh RGB eta-return training. No overwrite or resume operation."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'stabilization-gym',ROOT/'stabilization-learning-control'): sys.path.insert(0,str(p))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('MKL_NUM_THREADS','2')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('train','supervise','report')); p.add_argument('--config'); p.add_argument('--output',required=True)
    p.add_argument('--root'); p.add_argument('--process-snapshot')
    args=p.parse_args()
    if args.command=='report':
        if not args.root or not args.process_snapshot: p.error('report requires --root and --process-snapshot')
        from extensions.rgb_eta_return_v1.report import report
        report(args.root,args.output,args.process_snapshot)
    elif args.command=='supervise':
        if not args.config: p.error('supervise requires --config')
        from extensions.rgb_eta_return_v1.supervise import supervise
        sys.exit(supervise(args.config,args.output))
    else:
        if not args.config: p.error('train requires --config')
        from extensions.rgb_eta_return_v1.runner import train
        sys.exit(train(args.output,json.loads(Path(args.config).read_text(encoding='utf-8-sig'))))
