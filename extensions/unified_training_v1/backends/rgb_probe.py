"""Bounded integration probe; never a training-performance claim."""
from pathlib import Path
import argparse
import json
import os
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def expired():
        (args.output / "TIMEOUT.json").write_text(json.dumps({"timeout": True}), encoding="utf-8")
        os._exit(124)
    timer = threading.Timer(args.seconds, expired); timer.daemon = True; timer.start()
    from extensions.unified_training_v1.backends.rgb import Backend
    b = Backend(dict(seed=81, train_scene_count=1, parallel_envs=2,
        initial_collection_steps=3, collection_steps=3, validation_cases=2, test_cases=2,
        evaluation_steps=12, evaluation_batch=2, evaluation_plants=["P0"], batch_size=4),
        args.output, event=lambda kind, **data: print(json.dumps({"event": kind, **data}, ensure_ascii=False), flush=True),
        deadline=time.time()+args.seconds)
    results = {}
    for name, function in (("initialize", b.initialize), ("collect", lambda: b.collect(initial=True)),
        ("warmup", lambda: b.warmup_representation(1)), ("critic", lambda: b.pretrain_evaluator(1)),
        ("train", lambda: b.train_epoch(1, 1, 0, 1)), ("validation", b.validate)):
        start = time.perf_counter(); results[name] = function(); results[name]["probe_wall_s"] = time.perf_counter()-start
        print(json.dumps({"completed": name, "wall_s": results[name]["probe_wall_s"]}), flush=True)
    path = args.output / "probe_saved.pt"; b.save(path); b.load(path)
    b.close(); timer.cancel()
    (args.output / "PROBE_RESULT.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"probe_passed": True}), flush=True)


if __name__ == "__main__":
    main()
