"""Run the frozen-model generation comparison in its isolated environment."""
import argparse
import json
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(key,"1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from extensions.book_generation_overnight_v1.runner import run
    result=run(args.config,args.output);print(json.dumps(result),flush=True)
    raise SystemExit(0 if result["status"]=="complete" else 1)
