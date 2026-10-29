"""Fresh-output CPU checks for source-linked E01/E02/E08 examples."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
temporary=ROOT/'.cache/book_foundations_runtime_tmp';temporary.mkdir(parents=True,exist_ok=True)
tempfile.tempdir=str(temporary);os.environ.setdefault('MPLCONFIGDIR',str(temporary/'matplotlib'))
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--max-minutes',type=float,default=10.)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    from extensions.book_foundations_overnight_v1.runner import run
    result=run(args.output,args.max_minutes,args.smoke)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
