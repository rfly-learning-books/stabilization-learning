"""Fresh RGB GPU run, with a separate supervisor enforcing its original deadline."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT/"stabilization-gym", ROOT/"stabilization-learning-control"):
    sys.path.insert(0, str(folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "profile", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from extensions.joint_certificate_v4.config import RGBConfig
    from extensions.joint_certificate_v4.runner import initialize, run
    from extensions.joint_certificate_v4.process_control import terminate_owned_tree
    from stabilization_learning.joint_v2.runtime import atomic_json
    if args.command == "worker":
        result = run(args.output.resolve(), ROOT)
        print(json.dumps({k:v for k,v in result.items() if k != "summary"}, ensure_ascii=False))
        return
    if args.command == "run":
        if args.config is None:
            parser.error("Formal run requires a frozen config file")
        config = RGBConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        if config.scope != "formal_training":
            parser.error("Formal command requires formal_training scope")
    else:
        if args.config:
            parser.error("Smoke/profile have explicit engineering configurations")
        config = RGBConfig.smoke() if args.command == "smoke" else RGBConfig.profile()
    store = initialize(args.output.resolve(), config, ROOT, args.config)
    command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "worker", "--output", str(store.root)]
    log = (store.root/"worker.log").open("xb")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=flags, start_new_session=os.name != "nt")
    atomic_json(store.root/"LAUNCH.json", dict(pid=process.pid, supervisor_pid=os.getpid(), command=command,
                started_unix=store.protocol["started_unix"], deadline_unix=store.protocol["deadline_unix"],
                protocol_sha256=store.protocol_hash, automatic_resume=False))
    print(json.dumps(dict(root=str(store.root), pid=process.pid, deadline_unix=store.protocol["deadline_unix"])), flush=True)
    stopped = None
    try:
        while process.poll() is None:
            if (store.root/"STOP").exists():
                stopped = "STOP"
            elif time.time() >= store.protocol["deadline_unix"]:
                stopped = "original_deadline"
            if stopped:
                terminate_owned_tree(process)
                break
            time.sleep(1.)
    except BaseException:
        terminate_owned_tree(process)
        raise
    finally:
        log.close()
    atomic_json(store.root/"SUPERVISOR_FINAL.json", dict(returncode=process.returncode, stopped=stopped, unix=time.time(),
                worker_exited=process.poll() is not None, automatic_resume=False))
    if stopped:
        atomic_json(store.root/"FINAL_STATE.json", dict(status="stopped", reason=stopped, writer_exited=True, automatic_resume=False))
    elif process.returncode and not (store.root/"FINAL_STATE.json").exists():
        atomic_json(store.root/"FINAL_STATE.json", dict(status="failed", reason="worker_exited_before_final_state",
                    returncode=process.returncode, worker_exited=True, automatic_resume=False))
    raise SystemExit(process.returncode if process.returncode else (1 if stopped else 0))


if __name__ == "__main__":
    main()
