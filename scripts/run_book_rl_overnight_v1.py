"""Run the E13 equal-budget off-policy SAC integrator subexperiment."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["train","validate","smoke"])
    parser.add_argument("--config",type=Path,default=ROOT/"extensions/book_rl_overnight_v1/seed41.json")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    from extensions.book_rl_overnight_v1.runner import read,validate,run
    cfg=read(args.config)
    if args.command=="smoke":
        cfg=deepcopy(cfg)
        cfg.update(mode="engineering",device="cpu",maximum_minutes=5,parallel_envs=4,epochs=2,
            transitions_per_method=128,random_transitions=32,updates_per_vector_step=1,
            batch_size=16,train_starts=16,validation_starts=4,test_starts=4,evaluation_horizon=12)
        cfg["environment"].update(horizon=16,hold_steps=3)
        cfg["impulse_step"]=4;cfg["sac"]["hidden"]=32
    validate(cfg)
    if args.command=="validate":result=dict(valid=True,config=cfg)
    else:
        if args.output is None:parser.error("--output is required for a new run")
        result=run(cfg,args.output)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
