"""Explicit entry point for new unified integration runs. No automatic resume."""
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
    sub.add_parser("list")
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--process-snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        from stabilization_learning.unified_v1.registry import BACKENDS
        print(json.dumps(BACKENDS, indent=2))
    elif args.command == "train":
        import torch
        torch.set_num_threads(1)
        from stabilization_learning.unified_v1.runner import read, run_suite
        result = run_suite(read(args.config), args.output, project=ROOT)
        print(json.dumps(result, ensure_ascii=False))
    else:
        from stabilization_learning.unified_v1.reporting import report_suite
        print(json.dumps(report_suite(args.root, args.output, args.process_snapshot), ensure_ascii=False))


if __name__ == "__main__":
    main()
