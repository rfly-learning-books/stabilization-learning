"""Run the isolated CPU chapter-8 disk experiment in a fresh output directory."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
runtime_tmp=ROOT/'.cache/book_disk_runtime_tmp'
runtime_tmp.mkdir(parents=True,exist_ok=True)
tempfile.tempdir=str(runtime_tmp)
os.environ.setdefault('MPLCONFIGDIR',str(runtime_tmp/'matplotlib'))
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--max-minutes',type=float,default=30.)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    from extensions.book_disk_overnight_v1.runner import run
    result=run(args.output,args.seed,args.max_minutes,args.smoke)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
