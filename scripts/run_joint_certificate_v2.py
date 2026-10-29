"""Explicit v2 build, training and frozen-test commands. Nothing runs on import.

init-night records ONE deadline for every later stage. Use --night-root on all
formal stage commands; standalone commands require an explicit local hour cap.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import platform
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "stabilization-gym", ROOT / "stabilization-learning-control"):
    sys.path.insert(0, str(directory))

from stabilization_learning.joint_v2.runtime import atomic_json, file_sha256, RunStopped


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_fingerprint():
    import torch
    versions = {}
    for name in ("torch", "numpy", "gymnasium", "mujoco", "matplotlib", "scipy", "pyrender"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                platform=platform.platform(), packages=versions, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def sources():
    files = [Path(__file__)] + list((ROOT / "scripts").glob("run_joint_certificate*_v2.py"))
    for folder in (ROOT / "stabilization-gym/stabilization_gym/tasks_v2",
                   ROOT / "stabilization-learning-control/stabilization_learning/joint_v2",
                   ROOT / "extensions/joint_certificate_v2"):
        files += list(folder.glob("*.py"))
    # Legacy physics/backbones are reused read-only; freeze those dependencies too.
    for name in ("push_original", "so101_timeaware_v1", "visual_original"):
        files += list((ROOT / "third_party" / name).rglob("*.py"))
    files += list((ROOT / "extensions/visual_reconstructed_v1").rglob("*.py"))
    files += list((ROOT / "stabilization-gym/stabilization_gym").rglob("*.py"))
    files += list((ROOT / "stabilization-learning-control/stabilization_learning").rglob("*.py"))
    return {str(p.resolve()): file_sha256(p) for p in sorted(set(files))}


def stage_guard(args):
    started = time.time()
    if args.night_root:
        root = Path(args.night_root).resolve()
        protocol_path = root / "NIGHT_PROTOCOL.json"
        protocol = read(protocol_path)
        protocol_hash = file_sha256(protocol_path)
        if (root / "NIGHT_PROTOCOL.sha256").read_text().strip() != protocol_hash:
            raise RuntimeError("Night protocol seal mismatch")
        deadline = protocol["deadline_unix"]
        origin, hours = protocol["started_unix"], protocol["max_hours"]
        if (not all(math.isfinite(x) for x in (deadline, origin, hours))
                or not 0 < hours <= 14 or abs(deadline-origin-hours*3600) > 1e-4):
            raise ValueError("Invalid original shared budget")
        if deadline <= started:
            raise RunStopped("Original night deadline has already passed")
        for path, expected in protocol["sources"].items():
            if file_sha256(path) != expected:
                raise RuntimeError("Night source changed: " + path)
        for path, expected in protocol["configs"].items():
            if file_sha256(path) != expected:
                raise RuntimeError("Night config changed: " + path)
        if file_sha256(protocol["time_bank"]) != protocol["time_bank_sha256"]:
            raise RuntimeError("Night environment bank changed")
        if protocol.get("runtime") is not None and runtime_fingerprint() != protocol["runtime"]:
            raise RuntimeError("Night Python/library/device runtime changed")
        if getattr(args, "output", None) and not Path(args.output).resolve().is_relative_to(root):
            raise ValueError("Formal stage outputs must be inside their night root")
        stop_paths = [root / "STOP"]
    else:
        if args.max_hours is None or not math.isfinite(args.max_hours) or args.max_hours <= 0:
            raise ValueError("Supply --night-root or an explicit positive --max-hours")
        deadline = started + args.max_hours * 3600
        stop_paths = []
    if getattr(args, "output", None):
        stop_paths.append(Path(args.output) / "STOP")
    def guard():
        if args.night_root and file_sha256(protocol_path) != protocol_hash:
            raise RunStopped("Frozen night protocol changed")
        if time.time() >= deadline:
            raise RunStopped("Original shared deadline reached")
        if any(path.exists() for path in stop_paths):
            raise RunStopped("STOP present")
        if shutil.disk_usage(ROOT).free < 20 * 1024**3:
            raise RunStopped("Disk reserve below 20 GiB")
    guard()
    return guard, dict(started_unix=started, deadline_unix=deadline,
                       original_deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat())


def init_night(args):
    from stabilization_learning.joint_v2.push import PushTrainConfig
    from stabilization_gym.tasks_v2.push import JointPushCosts
    from extensions.joint_certificate_v2.rgb import RGBRunConfig
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 14:
        raise ValueError("Positive finite max-hours no greater than 14 required")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Declare unique nonempty seeds")
    bank = Path(args.time_bank).resolve()
    digest = file_sha256(bank)
    rgb_template = read(args.rgb_config) if args.rgb_config else asdict(RGBRunConfig())
    push_template = read(args.push_config) if args.push_config else asdict(PushTrainConfig())
    cost_config = asdict(JointPushCosts(**(read(args.costs) if args.costs else {})))
    configs = {}
    for seed in args.seeds:
        rgb = dict(rgb_template, seed=seed, group_seeds=args.seeds)
        configs[f"rgb_seed{seed}.json"] = asdict(RGBRunConfig.from_dict(rgb))
        configs[f"push_seed{seed}.json"] = asdict(PushTrainConfig(**dict(push_template, seed=seed)))
    configs["push_costs.json"] = cost_config
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = dict(schema="joint_certificate_night_v2", started_unix=started,
        started_utc=datetime.fromtimestamp(started, timezone.utc).isoformat(),
        deadline_unix=started+args.max_hours*3600, max_hours=args.max_hours,
        seeds=args.seeds, time_bank=str(bank), time_bank_sha256=digest,
        runtime=runtime_fingerprint(),
        sources=sources(), sequence=["verify", "rgb_all_train", "rgb_group_test",
                                     "push_all_train", "push_group_test", "so101_actuated", "analysis"],
        scope="joint RGB and unified pushing; state/observer interface regression only",
        independent_training_seeds=True, shared_test_starts=True,
        resume_policy="no automatic resume; preserve partial evidence",
        calibration_origin="declared existing environment asset; no policy initialization")
    for name, value in configs.items():
        atomic_json(root / "configs" / name, value)
    protocol["configs"] = {str((root / "configs" / name).resolve()): file_sha256(root / "configs" / name) for name in configs}
    atomic_json(root / "NIGHT_PROTOCOL.json", protocol)
    (root / "NIGHT_PROTOCOL.sha256").write_text(file_sha256(root / "NIGHT_PROTOCOL.json") + "\n")
    return protocol


def require_planned_config(args, name):
    if args.night_root:
        planned = Path(args.night_root).resolve() / "configs" / name
        actual = args.costs if name == "push_costs.json" else args.config
        if actual is None or file_sha256(actual) != file_sha256(planned):
            raise ValueError("Stage must use its frozen night config: " + name)


def require_planned_group(args, roots):
    if args.night_root:
        night = Path(args.night_root).resolve()
        expected = read(night / "NIGHT_PROTOCOL.json")["seeds"]
        paths = [Path(path).resolve() for path in roots]
        if any(not path.is_relative_to(night) for path in paths):
            raise ValueError("Frozen run is outside this night")
        actual = [read(path / "protocol.json")["config"]["seed"] for path in paths]
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError("Frozen group differs from the original night seed set")
        plan = read(night / "NIGHT_PROTOCOL.json")
        for path in paths:
            protocol = read(path / "protocol.json")
            config = protocol["config"]
            kind = "rgb" if "group_seeds" in config else "push"
            planned = read(night / "configs" / f"{kind}_seed{config['seed']}.json")
            if any(config.get(key) != value for key, value in planned.items()):
                raise ValueError("Group member scientific config differs from the night plan")
            if kind == "push":
                if (protocol.get("costs") != read(night / "configs/push_costs.json")
                    or protocol.get("time_bank_sha256") != plan["time_bank_sha256"]):
                    raise ValueError("Group member costs/bank differ from night plan")
                deadline = protocol.get("runtime_limits", {}).get("deadline_unix")
            else:
                deadline = protocol.get("deadline_unix")
            if deadline is None or not math.isfinite(deadline) or deadline > plan["deadline_unix"]:
                raise ValueError("Group member deadline is missing or exceeds original deadline")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify", help="Run actual v2 contract/gradient/renderer regression checks")
    p.add_argument("--junit")
    p = sub.add_parser("analyze", help="Verify writers exited, then plot actual saved updates into a new directory")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--process-snapshot", help="Fresh external Windows CIM process witness for formal night analysis")
    p = sub.add_parser("init-night", help="Create a new immutable shared deadline/config plan; does not start training")
    p.add_argument("--output", required=True)
    p.add_argument("--time-bank", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[81, 82, 83])
    p.add_argument("--max-hours", type=float, default=14.)
    p.add_argument("--rgb-config", help="Complete preflight-selected RGB template, frozen on creation")
    p.add_argument("--push-config", help="Complete preflight-selected Push template")
    p.add_argument("--costs", help="Preflight-selected joint Push costs")
    for name in ("rgb-smoke", "rgb-train", "rgb-test", "push-train", "push-prepare", "push-test", "push-arm"):
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
        p.add_argument("--config")
        p.add_argument("--night-root")
        p.add_argument("--max-hours", type=float)
        if name == "rgb-smoke":
            p.add_argument("--seed", type=int, default=81)
        if name == "rgb-test":
            p.add_argument("--runs", nargs="+", required=True)
        if name == "push-train":
            p.add_argument("--time-bank", required=True)
            p.add_argument("--costs")
        if name in ("push-test", "push-arm"):
            p.add_argument("--run", required=True)
            p.add_argument("--group-manifest", required=True)
            p.add_argument("--repeats", type=int, default=16 if name == "push-test" else 4)
    p = sub.add_parser("push-freeze")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--night-root")
    p.add_argument("--max-hours", type=float)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        tests = sorted(str(p) for p in (ROOT / "tests").glob("test_joint_v2_*.py"))
        command = [sys.executable, "-m", "pytest", "-q", *tests]
        if args.junit:
            command += ["--junitxml", str(Path(args.junit).resolve())]
        return subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL).returncode
    if args.command == "analyze":
        from extensions.joint_certificate_v2.analysis import analyze
        print(json.dumps(analyze(args.root, args.output, process_snapshot=args.process_snapshot), ensure_ascii=False, indent=2))
        return 0
    if args.command == "init-night":
        print(json.dumps(init_night(args), ensure_ascii=False, indent=2)); return 0
    if args.command == "push-freeze":
        from stabilization_learning.joint_v2.push import freeze_group
        guard, _ = stage_guard(args)
        require_planned_group(args, args.runs)
        if args.night_root and set(args.seeds) != set(read(Path(args.night_root)/"NIGHT_PROTOCOL.json")["seeds"]):
            raise ValueError("Freeze barrier cannot shrink the planned seed set")
        guard()
        print(json.dumps(freeze_group(args.runs, args.output, args.seeds), indent=2)); return 0
    guard, limits = stage_guard(args)
    if args.command in ("push-test", "push-arm") and args.night_root:
        barrier = read(args.group_manifest)
        require_planned_group(args, [row["path"] for row in barrier["runs"]])
        if set(barrier["expected_seeds"]) != set(read(Path(args.night_root)/"NIGHT_PROTOCOL.json")["seeds"]):
            raise ValueError("Test barrier differs from original night seed group")
    if args.command.startswith("rgb-"):
        from extensions.joint_certificate_v2.rgb import RGBRunConfig, train, test_group
        if args.command == "rgb-test":
            require_planned_group(args, args.runs)
            result = test_group(args.runs, args.output, guard=guard, project_root=ROOT,
                                original_deadline_unix=limits["deadline_unix"])
        else:
            config = (RGBRunConfig.smoke(seed=args.seed) if args.command == "rgb-smoke"
                      else RGBRunConfig.from_dict(read(args.config)) if args.config else RGBRunConfig())
            require_planned_config(args, f"rgb_seed{config.seed}.json")
            if args.night_root:
                expected = read(Path(args.night_root) / "NIGHT_PROTOCOL.json")["seeds"]
                if config.seed not in expected or set(config.group_seeds) != set(expected):
                    raise ValueError("RGB seed group differs from the immutable night plan")
            result = train(args.output, config, guard=guard, project_root=ROOT,
                           original_deadline_unix=limits["deadline_unix"])
    elif args.command == "push-prepare":
        from stabilization_gym.tasks_v2.push import prepare_time_bank
        result = str(prepare_time_bank(args.output, guard=guard))
    elif args.command == "push-train":
        from stabilization_learning.joint_v2.push import PushTrainConfig, train
        from stabilization_gym.tasks_v2.push import JointPushCosts
        config = PushTrainConfig(**(read(args.config) if args.config else {}))
        costs = JointPushCosts(**(read(args.costs) if args.costs else {}))
        require_planned_config(args, f"push_seed{config.seed}.json")
        require_planned_config(args, "push_costs.json")
        if args.night_root:
            protocol = read(Path(args.night_root) / "NIGHT_PROTOCOL.json")
            if config.seed not in protocol["seeds"] or file_sha256(args.time_bank) != protocol["time_bank_sha256"]:
                raise ValueError("Push seed or environment bank differs from the immutable night plan")
        result = train(args.time_bank, args.output, config, costs, guard=guard, runtime_limits=limits)
    elif args.command == "push-test":
        from stabilization_learning.joint_v2.push import test_frozen
        result = test_frozen(args.run, args.group_manifest, args.output, args.repeats, guard=guard)
    elif args.command == "push-arm":
        from extensions.joint_certificate_v2.arm import evaluate_frozen_arm
        result = evaluate_frozen_arm(args.run, args.group_manifest, args.output,
                                     repeats=args.repeats, guard=guard)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Modules retain their own partial artifacts. Never start another run here.
        print(f"joint-v2 stopped: {type(error).__name__}: {error}", file=sys.stderr)
        raise
