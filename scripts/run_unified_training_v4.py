"""Versioned reproducible training; fresh directories and full-group test barrier."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "stabilization-gym", ROOT / "stabilization-learning-control"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "preflight", "reproduce"):
        entry = sub.add_parser(name)
        entry.add_argument("--config", type=Path, required=True)
        entry.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    capture = sub.add_parser("capture-runtime")
    capture.add_argument("--output", type=Path, required=True)
    child = sub.add_parser("_preflight_job", help=argparse.SUPPRESS)
    child.add_argument("--config", type=Path, required=True)
    child.add_argument("--job", required=True)
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--result", type=Path, required=True)
    child.add_argument("--config-sha256", required=True)
    child.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    # Python hash randomization is chosen at process creation, never at runtime.
    required_environment = dict(PYTHONHASHSEED="0", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.command in {"train", "preflight", "reproduce", "_preflight_job"}:
        raw_config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not any(job.get("backend") == "rgb_legacy" or
                   str(job.get("config", {}).get("device", "cpu")).startswith("cuda")
                   for job in raw_config.get("jobs", [])):
            required_environment["CUDA_VISIBLE_DEVICES"] = "-1"
    if any(os.environ.get(k) != v for k, v in required_environment.items()):
        import subprocess
        child_env = os.environ.copy()
        child_env.update(required_environment)
        raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                                          *sys.argv[1:]], env=child_env))
    from extensions.reproducible_training_v4.runtime import configure
    configure()
    from extensions.reproducible_training_v4 import runner
    from stabilization_learning.unified_v1.runner import read
    if args.command == "capture-runtime":
        from extensions.reproducible_training_v4.runtime import manifest
        from stabilization_learning.unified_v1.runner import write_new
        result = manifest()
        write_new(args.output, result)
        print(json.dumps(dict(runtime_lock=str(args.output.resolve()), packages=len(result["packages"]))))
        return
    if args.command == "_preflight_job":
        runner.preflight_job(args.config, args.job, args.output, args.result, args.config_sha256, args.deadline)
        return
    if args.command == "train":
        result = runner.run_suite(read(args.config), args.output, project=ROOT, config_path=args.config,
                                  telemetry=any(j["backend"] == "rgb_legacy" for j in read(args.config)["jobs"]))
    elif args.command == "preflight":
        config = runner.validate_training_config(read(args.config))
        deadline = min(time.time() + config["maximum_minutes"] * 60,
                       config.get("absolute_deadline_unix", float("inf")))
        result = runner.preflight(args.config, config, args.output, project=ROOT, deadline=deadline)
    elif args.command == "reproduce":
        from extensions.reproducible_training_v4.reproduction import reproduce
        result = reproduce(args.config, args.output, project=ROOT)
    else:
        from extensions.reproducible_training_v4.reproduction import compare
        result = compare(args.first, args.second, args.output)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
