"""New bounded RGB long-trajectory experiment; separate from all prior runs."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/"stabilization-gym", ROOT/"stabilization-learning-control"):
    sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "worker"):
        item = commands.add_parser(name)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    item = commands.add_parser("report")
    item.add_argument("--root", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--process-snapshot", type=Path, required=True)
    args = parser.parse_args()
    from extensions.rgb_longroll_v1 import runner
    if args.command == "train":
        result = runner.supervise(args.config, args.output, project=ROOT)
    elif args.command == "worker":
        import torch
        torch.set_num_threads(1)
        result = runner.run_suite(runner.v1.read(args.config), args.output, project=ROOT, config_path=args.config)
    else:
        result = runner.report_suite(args.root, args.output, args.process_snapshot)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
