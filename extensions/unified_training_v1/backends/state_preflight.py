"""Bounded one-update integration checks; never reads the held-out test split."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT/"stabilization-gym", ROOT/"stabilization-learning-control"):
    sys.path.insert(0, str(folder))

from extensions.unified_training_v1.backends.state import Backend as StateBackend, TASKS as STATE_TASKS
from extensions.unified_training_v1.backends.observer import Backend as ObserverBackend, TASKS as OBS_TASKS


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tasks",nargs="*",default=[*STATE_TASKS,*OBS_TASKS])
    parser.add_argument("--evaluation",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for task in args.tasks:
        start=time.perf_counter();row=dict(task=task)
        try:
            cls=StateBackend if task in STATE_TASKS else ObserverBackend
            backend=cls(dict(task=task,seed=81,batch_size=4 if task in STATE_TASKS else 2,
                             validation_count=4,test_count=8,total_updates=8),args.output/task,
                        deadline=time.time()+900)
            clock=time.perf_counter();row["initialize"]=backend.initialize();row["initialize_seconds"]=time.perf_counter()-clock
            backend.collect(initial=True);backend.warmup_representation(0);backend.pretrain_evaluator(0)
            backend.collect(initial=False)
            clock=time.perf_counter();row["training"]=backend.train_epoch(1,0,0,1);row["update_seconds"]=time.perf_counter()-clock
            clock=time.perf_counter();row["validation"]=backend.validate();row["validation_seconds"]=time.perf_counter()-clock
            checkpoint=args.output/task/"preflight.pt";saved=backend.save(checkpoint);loaded=backend.load(checkpoint)
            assert saved["parameter_hash"]==loaded["parameter_hash"]
            row["checkpoint_roundtrip"]=True
            if args.evaluation:
                clock=time.perf_counter();row["evaluation"]=backend.evaluate("validation",["selected","initial","identity"])
                row["evaluation_seconds"]=time.perf_counter()-clock
                assert all(v["passed"] for v in row["evaluation"]["audit"].values())
            row["passed"]=True
        except Exception as error:
            row.update(passed=False,error=repr(error),traceback=traceback.format_exc())
        row["wall_seconds"]=time.perf_counter()-start;rows.append(row)
        (args.output/"RESULTS.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({key:value for key,value in row.items() if key in
            ("task","passed","error","initialize_seconds","update_seconds","validation_seconds","evaluation_seconds","wall_seconds")}),flush=True)
    if not all(r["passed"] for r in rows):raise SystemExit(1)


if __name__=="__main__":main()
