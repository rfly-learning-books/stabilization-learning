"""Explicit RGB pipeline v3 commands; no work is launched on import."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/"stabilization-gym", ROOT/"stabilization-learning-control"):
    sys.path.insert(0, str(directory))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "test-group"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--roots", type=Path, nargs="+")
    args = parser.parse_args(argv)
    import torch
    from extensions.joint_certificate_v3.rgb import RGBRunConfig, train, test_group
    torch.set_num_threads(1)
    if args.output.exists():
        parser.error("Output must be a new directory; never overwrite or restart a recorded run")
    if args.command == "smoke":
        if args.config or args.roots:
            parser.error("smoke uses its explicit short engineering configuration")
        hours = .25 if args.max_hours is None else args.max_hours
        if not 0 < hours <= .25:
            parser.error("Engineering smoke is capped at 15 minutes")
        cfg = replace(RGBRunConfig.smoke(device=args.device or "cuda"), max_hours=hours,
                      validation_case_limit=2, test_case_limit=2, inference_batch_size=2,
                      bootstrap_episodes_per_scene=2)
        # One shared budget covers BOTH stages; no renewed budget after train.
        deadline = time.time() + hours*3600
        args.output.mkdir(parents=True, exist_ok=False)
        guard = lambda: not (args.output/"STOP").exists()
        train(args.output/"rgb_seed81", cfg, guard=guard, original_deadline_unix=deadline)
        result = test_group([args.output/"rgb_seed81"], args.output/"group_test", config=cfg,
                            guard=guard, original_deadline_unix=deadline)
    elif args.command == "train":
        if args.config is None or args.max_hours is None or args.roots:
            parser.error("train requires --config and --max-hours; it never resumes old runs")
        cfg = RGBRunConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        cfg = replace(cfg, max_hours=args.max_hours, **({"device":args.device} if args.device else {}))
        result = train(args.output, cfg)
    else:
        if not args.roots or args.max_hours is None or args.config or args.device:
            parser.error("test-group requires --roots and --max-hours; scientific config comes from frozen roots")
        frozen = json.loads((args.roots[0]/"protocol.json").read_text(encoding="utf-8"))["config"]
        cfg = RGBRunConfig.from_dict({k:v for k,v in frozen.items() if k in RGBRunConfig.__dataclass_fields__})
        result = test_group(args.roots, args.output, config=replace(cfg, max_hours=args.max_hours))
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
