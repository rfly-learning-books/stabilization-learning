"""Immutable entry point for the historical-reference suite; no resume."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "stabilization-gym", ROOT / "stabilization-learning-control"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--process-snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        import torch
        torch.set_num_threads(1)
        from stabilization_learning.unified_v1.runner import read
        from extensions.unified_training_v2.runner import run_suite
        result = run_suite(read(args.config), args.output, project=ROOT, config_path=args.config)
    else:
        from extensions.unified_training_v2.reporting import report_suite
        result = report_suite(args.root, args.output, args.process_snapshot)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
