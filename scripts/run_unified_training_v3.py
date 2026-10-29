"""New-run entry point with isolated preflight; never resumes historical runs."""
from pathlib import Path
import argparse
import json
import sys
import time

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
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--process-snapshot", type=Path, required=True)
    worker = sub.add_parser("_preflight_job", help=argparse.SUPPRESS)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--job", required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--result", type=Path, required=True)
    worker.add_argument("--config-sha256", required=True)
    worker.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    from stabilization_learning.unified_v1.runner import read, validate_config
    if args.command == "_preflight_job":
        from extensions.unified_training_v3.runner import preflight_job
        preflight_job(args.config, args.job, args.output, args.result, args.config_sha256, args.deadline)
        return
    if args.command == "train":
        from extensions.unified_training_v3.runner import run_suite
        result = run_suite(read(args.config), args.output, project=ROOT, config_path=args.config)
    elif args.command == "preflight":
        from extensions.unified_training_v3.runner import preflight as run_preflight
        config = validate_config(read(args.config))
        deadline = min(time.time() + 60 * config["maximum_minutes"],
                       config.get("absolute_deadline_unix", float("inf")))
        result = run_preflight(args.config, config, args.output, project=ROOT, deadline=deadline)
    else:
        from extensions.unified_training_v2.reporting import report_suite
        result = report_suite(args.root, args.output, args.process_snapshot)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
