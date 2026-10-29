"""Prepare or execute the sequential Reviewed-book experiment queue."""
from pathlib import Path
import argparse
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
for path in (PROJECT, PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control"):
    sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--root", type=Path, required=True)
    observers = sub.add_parser("observers")
    observers.add_argument("--config", type=Path, required=True)
    observers.add_argument("--output", type=Path, required=True)
    observers.add_argument("--parent-stop", type=Path)
    e20 = sub.add_parser("e20")
    e20.add_argument("--config", type=Path, required=True)
    e20.add_argument("--output", type=Path, required=True)
    e09 = sub.add_parser("e09")
    e09.add_argument("--output", type=Path, required=True)
    e09.add_argument("--seed", type=int, default=41)
    catalog = sub.add_parser("catalog")
    catalog.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from extensions.book_overnight_v1.queue import prepare, run, read
    if args.command == "prepare":
        result = prepare(args.plan, args.output)
    elif args.command == "worker":
        result = run(args.root)
    elif args.command == "observers":
        from extensions.book_overnight_v1.applications import observers
        result = observers(read(args.config), args.output, parent_stop=args.parent_stop)
    elif args.command == "e20":
        from extensions.book_overnight_v1.applications import e20
        result = e20(read(args.config), args.output)
    elif args.command == "e09":
        from extensions.book_overnight_v1.applications import e09
        result = e09(args.output, args.seed)
    else:
        from extensions.book_overnight_v1.catalog import build
        result = build(args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result.get("status") in ("failed", "stopped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
