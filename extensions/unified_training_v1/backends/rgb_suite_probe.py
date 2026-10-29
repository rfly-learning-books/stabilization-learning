"""Bounded RGB integration probe through the actual unified runner."""
from pathlib import Path
import argparse
import json
import os
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "stabilization-learning-control", ROOT / "stabilization-gym"):
    sys.path.insert(0, str(folder))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    def expired():
        if args.output.exists():
            (args.output / "PROBE_TIMEOUT.json").write_text(json.dumps({"timeout": True}), encoding="utf-8")
        os._exit(124)
    timer = threading.Timer(args.seconds, expired); timer.daemon = True; timer.start()
    from stabilization_learning.unified_v1.runner import run_suite
    config = dict(schema="unified_training_v1", maximum_minutes=args.seconds / 60,
        jobs=[dict(id="rgb_probe", backend="rgb_legacy",
            config=dict(seed=81, train_scene_count=1, parallel_envs=2,
                initial_collection_steps=3, collection_steps=3, validation_cases=2, test_cases=2,
                evaluation_steps=12, evaluation_batch=2, evaluation_plants=["P0"], batch_size=4),
            schedule=dict(warmup_representation=1, pretrain_evaluator=1, epochs=1,
                representation_updates=1, evaluator_updates=0, certificate_updates=1),
            test_variants=["selected", "initial", "identity", "shadow"])])
    result = run_suite(config, args.output, project=ROOT, telemetry=False)
    import torch
    timer.cancel()
    result["torch_memory_allocated_after_close"] = torch.cuda.memory_allocated()
    (args.output / "PROBE_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"probe_passed": True, "torch_memory_allocated_after_close": result["torch_memory_allocated_after_close"]}))


if __name__ == "__main__":
    main()
