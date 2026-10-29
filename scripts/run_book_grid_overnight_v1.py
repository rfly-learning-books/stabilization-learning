"""Run the Reviewed 95-state grid in a new result directory."""
from pathlib import Path
import argparse
import sys
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--episodes',type=int,default=10000)
    parser.add_argument('--max-minutes',type=float,default=15)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    from extensions.book_grid_overnight_v1.runner import run
    run(args.output,seed=args.seed,episodes=200 if args.smoke else args.episodes,max_minutes=args.max_minutes,smoke=args.smoke)
