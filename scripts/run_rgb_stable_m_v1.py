"""Fresh RGB representation, component Critic and controlled metric updates."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'stabilization-gym',ROOT/'stabilization-learning-control'):
    sys.path.insert(0,str(path))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('MKL_NUM_THREADS','2')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['supervise','worker','report'])
    p.add_argument('--config',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path); p.add_argument('--process-snapshot',type=Path)
    args=p.parse_args()
    from extensions.rgb_stable_m_v1 import runner
    if args.command=='report':
        if args.root is None or args.process_snapshot is None: p.error('report requires root and fresh external process snapshot')
        result=runner.report(args.root,args.output,args.process_snapshot)
    else:
        if args.config is None: p.error('config required')
        if args.command=='supervise': sys.exit(runner.supervise(args.config,args.output))
        result=runner.worker(args.config,args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
