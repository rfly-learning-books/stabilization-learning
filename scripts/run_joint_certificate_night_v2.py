"""Run an already-frozen v2 night plan, with one watchdog and no auto-resume.

This launcher never creates a plan or extends its deadline. Its children are
explicit commands; failed branches keep all evidence and cannot enter tests.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from run_joint_certificate_v2 import ROOT, read, stage_guard
from stabilization_learning.joint_v2.runtime import atomic_json, file_sha256, RunStopped


def command_plan(root, seeds):
    root = Path(root).resolve()
    common = ["--night-root", str(root)]
    cli = [sys.executable, str(ROOT / "scripts/run_joint_certificate_v2.py")]
    def stage(name, arguments):
        return name, cli + arguments
    rgb = [root / f"rgb_seed{seed}" for seed in seeds]
    push = [root / f"push_seed{seed}" for seed in seeds]
    protocol = read(root / "NIGHT_PROTOCOL.json")
    return dict(
        verify=[stage("verify", ["verify", "--junit", str(root / "engineering_tests.xml")])],
        rgb_train=[stage(f"rgb_train_seed{seed}", ["rgb-train", "--config", str(root / f"configs/rgb_seed{seed}.json"),
                   "--output", str(path), *common]) for seed, path in zip(seeds, rgb)],
        rgb_test=[stage("rgb_group_test", ["rgb-test", "--runs", *map(str, rgb), "--output", str(root / "rgb_test"), *common])],
        push_train=[stage(f"push_train_seed{seed}", ["push-train", "--config", str(root / f"configs/push_seed{seed}.json"),
                    "--costs", str(root / "configs/push_costs.json"), "--time-bank", protocol["time_bank"],
                    "--output", str(path), *common]) for seed, path in zip(seeds, push)],
        push_freeze=[stage("push_group_freeze", ["push-freeze", "--runs", *map(str, push), "--seeds", *map(str, seeds),
                     "--output", str(root / "PUSH_GROUP_FROZEN.json"), *common])],
        push_test=[stage(f"push_test_seed{seed}", ["push-test", "--run", str(path),
                   "--group-manifest", str(root / "PUSH_GROUP_FROZEN.json"), "--repeats", "16",
                   "--output", str(root / f"push_test_seed{seed}"), *common]) for seed, path in zip(seeds, push)],
        arm=[stage(f"arm_seed{seed}", ["push-arm", "--run", str(path), "--group-manifest", str(root / "PUSH_GROUP_FROZEN.json"),
             "--repeats", "4", "--output", str(root / f"arm_seed{seed}"), *common]) for seed, path in zip(seeds, push)])


def stop_owned_child(process):
    """Terminate only the still-live child launched by this supervisor."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
                       check=False)
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def supervise(root):
    root = Path(root).resolve()
    guard, limits = stage_guard(argparse.Namespace(night_root=str(root), max_hours=None, output=None))
    if (root / "FINAL_STATE.json").exists() or (root / "night_events").exists():
        raise FileExistsError("Preserve completed/interrupted night; automatic resume is unsupported")
    protocol = read(root / "NIGHT_PROTOCOL.json")
    token = uuid.uuid4().hex
    lock = root / "NIGHT_WRITER.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), token=token), stream)
    sequence = 0
    outcomes = []
    child = None
    def event(kind, **data):
        nonlocal sequence
        sequence += 1
        atomic_json(root / "night_events" / f"{sequence:08d}_{kind}.json",
                    dict(sequence=sequence, kind=kind, unix=time.time(),
                         protocol_sha256=file_sha256(root / "NIGHT_PROTOCOL.json"), data=data))
    def run(name, command):
        nonlocal child
        guard()
        outpath = root / "supervisor_logs" / f"{name}.log"
        outpath.parent.mkdir(exist_ok=True)
        with outpath.open("xb") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            event("stage_started", name=name, pid=child.pid, command=command)
            try:
                while child.poll() is None:
                    guard()
                    # Child guards check STOP too; watchdog covers uninterruptible rollouts.
                    if any(path.exists() for path in root.glob("*/STOP")):
                        raise RunStopped("A stage STOP is present")
                    time.sleep(0.5)
                code = child.returncode
                guard()  # A final stage may exit between the last poll and deadline.
            except BaseException:
                stop_owned_child(child)
                raise
            finally:
                child = None
        row = dict(stage=name, exit_code=code)
        outcomes.append(row)
        event("stage_exited", **row)
        return code == 0
    def group(stages):
        ok = True
        for name, command in stages:
            if not run(name, command):
                ok = False
                break  # Preserve incomplete group; no retrain or reduced group test.
        return ok
    status, detail = "failed", None
    try:
        event("supervisor_started", pid=os.getpid(), **limits)
        plan = command_plan(root, protocol["seeds"])
        if not group(plan["verify"]):
            raise RuntimeError("Engineering verification failed; no training was launched")
        rgb_ok = group(plan["rgb_train"])
        rgb_test_ok = group(plan["rgb_test"]) if rgb_ok else False
        push_ok = group(plan["push_train"])
        frozen_ok = group(plan["push_freeze"]) if push_ok else False
        push_test_ok = group(plan["push_test"]) if frozen_ok else False
        arm_ok = group(plan["arm"]) if frozen_ok else False
        status = "complete" if all((rgb_test_ok, push_test_ok, arm_ok)) else "partial_failed"
    except RunStopped as error:
        status, detail = "stopped", str(error)
    except BaseException as error:
        status, detail = "failed", f"{type(error).__name__}: {error}"
    finally:
        if child is not None:
            stop_owned_child(child)
        result = dict(status=status, detail=detail, stages=outcomes, **limits,
                      finished_unix=time.time(), scientific_analysis_complete=False,
                      scope="training/test execution status; performance conclusions require post-analysis")
        atomic_json(root / "FINAL_STATE.json", result)
        event("supervisor_finished", **result)
        if read(lock).get("token") == token:
            lock.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--night-root", required=True)
    args = parser.parse_args()
    result = supervise(args.night_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
