"""Observer training with canonical import paths and the retained v4 algorithm."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def register_project_paths(project=ROOT):
    """Register only owned source roots, preserving all other search entries."""
    def key(value):
        return os.path.normcase(os.path.realpath(value or os.curdir))

    paths = tuple(str(Path(project).resolve() / suffix) if suffix else str(Path(project).resolve())
                  for suffix in ("", "stabilization-gym", "stabilization-learning-control"))
    owned = {key(path): path for path in paths}
    seen, result = set(), []
    for value in sys.path:
        identity = key(value)
        if identity in owned:
            if identity not in seen:
                result.append(owned[identity])
                seen.add(identity)
        else:
            result.append(value)
    for path in paths:
        if key(path) not in seen:
            result.insert(0, path)
            seen.add(key(path))
    sys.path[:] = result
    return paths


register_project_paths()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "preflight", "doctor"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("compare")
    command.add_argument("--first", type=Path, required=True)
    command.add_argument("--second", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = dict(PYTHONHASHSEED="0", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                    CUDA_VISIBLE_DEVICES="-1")
    if any(os.environ.get(name) != value for name, value in required.items()):
        import subprocess
        env = os.environ.copy()
        env.update(required)
        raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                                          *sys.argv[1:]], env=env))
    from extensions.reproducible_training_v4.runtime import configure
    from extensions.observer_training_v5 import runner
    from stabilization_learning.unified_v1.runner import read, write_new
    configure()
    if args.command == "compare":
        from extensions.reproducible_training_v4.reproduction import compare
        result = compare(args.first, args.second, args.output)
    else:
        config = runner.validate_config(read(args.config))
        if args.command == "doctor":
            result = runner.diagnose(ROOT, config, args.config)
            write_new(args.output, result)
        elif args.command == "preflight":
            deadline = min(time.time() + config["maximum_minutes"] * 60,
                           config.get("absolute_deadline_unix", float("inf")))
            result = runner.preflight(args.config, config, args.output, project=ROOT, deadline=deadline)
        else:
            result = runner.run_suite(config, args.output, project=ROOT, config_path=args.config)
    if args.command == "doctor":
        summary = {key: result[key] for key in ("runtime_lock_passed", "package_records", "modules")}
    elif args.command == "train":
        summary = dict(status=result.get("status", "returned"))
    else:
        summary = result
    print(json.dumps(dict(command=args.command, output=str(args.output.resolve()), result=summary),
                     ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
