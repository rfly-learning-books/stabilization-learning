from __future__ import annotations

import argparse
import json
from pathlib import Path

from lac_vs.config import ROOT, load_config, seed_all


def main():
    parser = argparse.ArgumentParser(description="RGB-only goal-conditioned LAC+AE and full-M certificate control")
    parser.add_argument("command", choices=["doctor", "audit", "git-cache", "prepare", "train", "smoke", "evaluate", "report"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "full.yaml"))
    parser.add_argument("--output", help="Run directory; artifacts are separate from source and legacy experiments")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--plant", choices=["P0", "P1"])
    parser.add_argument("--method", choices=["learned", "identity_m", "pvs"], default="learned")
    parser.add_argument("--apply", action="store_true", help="git-cache: untrack generated ignored files; preserve disk files")
    args = parser.parse_args()
    if args.command == "smoke" and args.config == str(ROOT / "configs" / "full.yaml"):
        args.config = str(ROOT / "configs" / "smoke.yaml")
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    output = Path(args.output).resolve() if args.output else ROOT / "runs" / ("smoke" if args.command == "smoke" else "seed41")
    if args.command in ("doctor", "audit", "git-cache"):
        from lac_vs.audit import doctor, git_cache_cleanup, run_audit
        if args.command == "audit":
            report = run_audit(ROOT / "runtime" / "gradient_audit.json")
            if not report["passed"]:
                raise SystemExit(1)
        elif args.command == "doctor":
            report = doctor(cfg, ROOT / "runtime" / "environment.json")
        else:
            report = git_cache_cleanup([ROOT, Path(cfg["paths"]["legacy_root"]), ROOT.parent],
                                       ROOT / "runtime" / "git_cache_cleanup.json", args.apply)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    from lac_vs.locking import writer_lock
    with writer_lock(output):
        execute(args, cfg, output)


def execute(args, cfg, output):
    from lac_vs.runner import prepare, train
    if args.command == "prepare":
        protocol = prepare(cfg, output)
        print(json.dumps({"prepared": str(output), "algorithm": protocol["algorithm"]}, ensure_ascii=False))
    elif args.command in ("train", "smoke"):
        train(cfg, output, resume=args.resume, max_minutes=args.max_minutes)
    elif args.command == "evaluate":
        from lac_vs import legacy
        from lac_vs.learner import Learner
        from lac_vs.replay import Replay
        from lac_vs.runner import evaluate, load_checkpoint
        protocol = prepare(cfg, output)
        seed_all(cfg["seed"])
        learner = Learner(cfg, cfg["device"])
        replay = Replay(output / "replay")
        path = Path(args.checkpoint)
        path = path if path.is_absolute() else output / path
        load_checkpoint(path, learner, replay, protocol, restore_random=False)
        depth = legacy.module(cfg, "depth_net").MetricDepth(str(learner.device))
        summary = evaluate(cfg, output, learner, depth, split=args.split, method=args.method,
                           plants=[args.plant] if args.plant else None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        from lac_vs.report import make_report
        print(make_report(output))


if __name__ == "__main__":
    main()
