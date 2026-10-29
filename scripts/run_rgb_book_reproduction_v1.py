"""Train and report the book RGB protocol through Env/Algo/Runner."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "stabilization-gym", ROOT / "stabilization-learning-control"):
    sys.path.insert(0, str(folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "supervise", "worker"):
        item = commands.add_parser(name)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        if name == "worker":
            item.add_argument("--deadline", type=float)
    item = commands.add_parser("validate")
    item.add_argument("--config", type=Path, required=True)
    item = commands.add_parser("report")
    item.add_argument("--root", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--process-snapshot", type=Path, required=True)
    args = parser.parse_args()
    from extensions.rgb_book_reproduction_v1 import runner
    if args.command in ("train", "supervise"):
        result = runner.supervise(args.config, args.output, project=ROOT)
    elif args.command == "worker":
        result = runner.run_suite(runner.v1.read(args.config), args.output, project=ROOT,
                                  config_path=args.config, deadline=args.deadline)
    elif args.command == "validate":
        cfg = runner.validate(runner.v1.read(args.config))
        result = dict(valid=True, mode=cfg.get("book_protocol", {}).get("mode", "formal"), jobs=[j["id"] for j in cfg["jobs"]])
    else:
        from extensions.rgb_book_reproduction_v1.reporting import report_suite
        result = report_suite(args.root, args.output, args.process_snapshot)
    print(json.dumps(result, ensure_ascii=False))
    if isinstance(result, dict) and result.get("exit_code", 0) != 0:
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
