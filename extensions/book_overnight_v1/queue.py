"""Source-frozen, single-writer experiment queue with an absolute deadline."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid


PROJECT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.rename(path)


def deadline(plan):
    value = datetime.fromisoformat(plan["deadline_iso"])
    if value.tzinfo is None:
        raise ValueError("The absolute deadline must include its timezone")
    return value.timestamp()


def validate(plan):
    if plan.get("schema") != "book_overnight_queue_v1":
        raise ValueError("Expected book_overnight_queue_v1")
    deadline(plan)
    checklist = read(plan["checklist"])
    known = {t["id"] for t in checklist["tasks"]}
    if len(known) != 32:
        raise ValueError("Use the complete Reviewed 30+2 task checklist")
    identifiers = set()
    for step in plan["steps"]:
        if step["id"] in identifiers or not step["id"].replace("_", "").isalnum():
            raise ValueError("Each queue step needs a unique simple identifier")
        if not set(step["task_ids"]) <= known or not step["task_ids"]:
            raise ValueError("Unknown or empty task mapping")
        if not step.get("scope") or step.get("maximum_seconds", 0) <= 0:
            raise ValueError("Declare each step's scientific scope and time budget")
        if not set(step.get("depends_on", [])) <= identifiers:
            raise ValueError("Dependencies must precede their dependent step")
        identifiers.add(step["id"])
        if not step.get("commands"):
            raise ValueError("Each executable step requires commands")
        for command in step["commands"]:
            if not command.get("argv") or not isinstance(command["argv"], list):
                raise ValueError("Use structured argument arrays, never shell command strings")
            if not Path(command["argv"][0]).is_file():
                raise FileNotFoundError(command["argv"][0])
            if not Path(command.get("cwd", PROJECT)).is_dir():
                raise FileNotFoundError(command["cwd"])
        for output in step.get("output_dirs", []):
            path = Path(output).resolve()
            if not path.is_relative_to((PROJECT/"runs").resolve()) or path == (PROJECT/"runs").resolve():
                raise ValueError("Experiment outputs must be new folders under project runs")
    return plan


def matrix(plan, results=None):
    records = {row["id"]: row for row in (results or [])}
    rows = []
    for task in read(plan["checklist"])["tasks"]:
        related = [s for s in plan["steps"] if task["id"] in s["task_ids"]]
        completed = [s["id"] for s in related if records.get(s["id"], {}).get("status") == "complete"]
        rows.append(dict(id=task["id"], title=task["title"], kind=task["kind"], priority=task["priority"],
            completion_criterion=task["completion_criterion"], planned_steps=[s["id"] for s in related],
            completed_steps=completed, state="new_evidence_available" if completed else
                "planned" if related and not results else "unfinished" if related else "not_implemented_this_queue",
            mapped_scopes=[s["scope"] for s in related], full_task_acceptance=False,
            note=plan.get("task_notes", {}).get(task["id"], "Task acceptance is assessed against all original checklist requirements.")))
    return rows


def prepare(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = validate(read(plan_path))
    if output.exists():
        raise FileExistsError("Queue preparation requires a new directory")
    files = {plan_path, Path(plan["checklist"])}
    files.update(Path(p) for p in plan.get("source_files", []))
    files.update((PROJECT / "extensions/book_overnight_v1").glob("*.py"))
    files.add(PROJECT / "scripts/run_book_overnight_v1.py")
    for step in plan["steps"]:
        files.update(Path(p) for p in step.get("source_files", []))
        for path in step.get("output_dirs", []):
            if Path(path).exists():
                raise FileExistsError(path)
    sources = {str(p.resolve()): sha(p) for p in sorted(files)}
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema="book_overnight_queue_protocol_v1", plan=plan, sources=sources,
        prepared_utc=datetime.now(timezone.utc).isoformat(), deadline_unix=deadline(plan),
        execution="one sequential child writer; no resume; independent step failures retained",
        task_acceptance="Completed commands are evidence, not automatic acceptance of an entire checklist task")
    write(output / "protocol.json", protocol)
    write(output / "TASK_STATUS_PREPARED.json", matrix(plan))
    return dict(prepared=str(output), steps=len(plan["steps"]), deadline=plan["deadline_iso"])


def verify_sources(protocol):
    for path, expected in protocol["sources"].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError("Frozen queue source changed: " + path)


def prerequisite(plan):
    """Never launch a Python waiting process while the RGB report is pending."""
    spec = plan["after_rgb"]
    root = Path(spec["run"])
    final = read(root / "FINAL_STATE.json")
    if final["status"] not in spec.get("acceptable_statuses", ["complete"]):
        raise RuntimeError("RGB final status is outside the declared handoff conditions")
    if (root / "WRITER.lock").exists():
        raise RuntimeError("RGB writer lock remains")
    report = Path(spec["report"])
    report_file = report / "REPORT.json" if report.is_dir() else report
    reported = read(report_file)
    if Path(reported["root"]).resolve() != root.resolve():
        raise RuntimeError("The RGB report belongs to a different run")
    curves = Path(spec["curves_marker"])
    if not curves.is_file():
        raise RuntimeError("RGB curves have not been delivered")
    return dict(rgb_final_sha256=sha(root / "FINAL_STATE.json"), report_sha256=sha(report_file),
                curves_marker_sha256=sha(curves))


def stop_owned_child(child):
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True,
            stdin=subprocess.DEVNULL, timeout=20, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        child.terminate()
    child.wait(timeout=25)


def run(root):
    root = Path(root).resolve()
    protocol = read(root / "protocol.json")
    plan = validate(protocol["plan"])
    if (root / "FINAL_STATE.json").exists() or (root / "STOP").exists():
        raise RuntimeError("Finalized or stopped queues are not resumed")
    verify_sources(protocol)
    handoff = prerequisite(plan)
    hard_deadline = min(protocol["deadline_unix"], deadline(plan))
    if time.time() >= hard_deadline:
        raise InterruptedError("Absolute overnight deadline already reached")
    lease = root / "WRITER.lock"
    token = uuid.uuid4().hex
    write(lease, dict(pid=os.getpid(), token=token))
    sequence = 0
    results = []
    gpu_samples = []
    gpu_last = 0.
    final = dict(schema="book_overnight_final_v1", status="running", started_unix=time.time(), handoff=handoff,
                 deadline_unix=hard_deadline, automatic_resume=False)

    def event(kind, **data):
        nonlocal sequence
        sequence += 1
        row = dict(sequence=sequence, kind=kind, unix=time.time(), pid=os.getpid(), **data)
        write(root / "events" / f"{sequence:06d}_{kind}.json", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def guard():
        if read(lease)["token"] != token:
            raise RuntimeError("Queue writer lease changed")
        if (root / "STOP").exists() or time.time() >= hard_deadline-30:
            raise InterruptedError("Queue STOP or absolute deadline reached")
        if shutil.disk_usage(root).free < plan.get("minimum_free_gib", 20) * 1024**3:
            raise InterruptedError("Queue disk reserve reached")

    def sample_gpu(step):
        nonlocal gpu_last
        if time.time()-gpu_last < 10:
            return
        gpu_last = time.time()
        row = dict(unix=gpu_last, step=step, scope="whole GPU, including display and other applications")
        try:
            completed = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            fields = completed.stdout.splitlines()[0].split(",")
            row.update(utilization_percent=float(fields[0]), memory_mib=float(fields[1]), power_watts=float(fields[2]))
        except Exception as error:
            row["error"] = repr(error)
        gpu_samples.append(row)
        with (root/"GPU_SAMPLES.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")

    try:
        event("queue_started", step_count=len(plan["steps"]))
        for step in plan["steps"]:
            guard(); verify_sources(protocol)
            prior = {r["id"]: r["status"] for r in results}
            if any(prior.get(d) != "complete" for d in step.get("depends_on", [])):
                record = dict(id=step["id"], status="dependency_unfinished", task_ids=step["task_ids"])
                results.append(record); event("step_skipped", **record); continue
            if hard_deadline-time.time() < step.get("minimum_start_seconds", 60)+30:
                results.append(dict(id=step["id"], status="insufficient_remaining_time", task_ids=step["task_ids"]))
                continue
            for output in step.get("output_dirs", []):
                if Path(output).exists():
                    raise FileExistsError("Never overwrite or resume step outputs: " + output)
            step_begin = time.time()
            step_end = min(hard_deadline-30, step_begin + step["maximum_seconds"])
            record = dict(id=step["id"], task_ids=step["task_ids"], scope=step["scope"],
                status="running", started_unix=step_begin, commands=[])
            event("step_started", step=step["id"], task_ids=step["task_ids"], deadline_unix=step_end)
            try:
                for number, command in enumerate(step["commands"]):
                    guard(); verify_sources(protocol)
                    logs = root / "logs"; logs.mkdir(exist_ok=True)
                    log_path = logs / f"{step['id']}_{number:02d}.log"
                    env = os.environ.copy(); env.update(command.get("environment", {}))
                    env.setdefault("PYTHONUTF8", "1"); env.setdefault("PYTHONHASHSEED", "0")
                    cmd_end = min(step_end, time.time()+command.get("timeout_seconds", step["maximum_seconds"]))
                    started = time.time()
                    with log_path.open("xb") as log:
                        child = subprocess.Popen(command["argv"], cwd=command.get("cwd", str(PROJECT)),
                            env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        event("child_started", step=step["id"], command=number, child_pid=child.pid,
                              argv=command["argv"], log=str(log_path))
                        reason = None
                        try:
                            while child.poll() is None:
                                guard()
                                sample_gpu(step["id"])
                                if time.time() >= cmd_end:
                                    raise TimeoutError("Step/command budget reached")
                                time.sleep(.5)
                        except BaseException as error:
                            reason = repr(error)
                            for path in step.get("output_dirs", []):
                                directory = Path(path)
                                if directory.is_dir() and not (directory / "STOP").exists():
                                    write(directory / "STOP", dict(reason=reason, owning_queue=str(root)))
                            grace = min(15, max(0, hard_deadline-time.time()-15))
                            try:
                                child.wait(timeout=grace)
                            except subprocess.TimeoutExpired:
                                stop_owned_child(child)
                            raise
                        finally:
                            record["commands"].append(dict(command=number, pid=child.pid, exit_code=child.poll(),
                                started_unix=started, finished_unix=time.time(), stop_reason=reason, log=str(log_path)))
                    if child.returncode:
                        raise RuntimeError(f"Command exited with code {child.returncode}")
                    for path in command.get("expected_files", []):
                        if not Path(path).is_file():
                            raise RuntimeError("Required evidence absent after successful process: " + path)
                record["status"] = "complete"
            except InterruptedError:
                record["status"] = "stopped"
                raise
            except Exception as error:
                record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            finally:
                record["finished_unix"] = time.time(); results.append(record)
                write(root / "steps" / (step["id"] + ".json"), record)
                event("step_finished", step=step["id"], status=record["status"])
        final["status"] = "complete" if all(r["status"] == "complete" for r in results) else "partial"
    except BaseException as error:
        final.update(status="stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                     error=repr(error), traceback=traceback.format_exc())
    finally:
        final.update(finished_unix=time.time(), steps=results)
        valid = [r for r in gpu_samples if "utilization_percent" in r]
        pairs = [(a,b) for a,b in zip(valid,valid[1:]) if 0 < b["unix"]-a["unix"] <= 30]
        seconds = sum(b["unix"]-a["unix"] for a,b in pairs)
        write(root / "GPU_SUMMARY.json", dict(scope="whole GPU, not process attribution", samples=len(gpu_samples),
            valid_samples=len(valid), integrated_seconds=seconds,
            weighted_utilization_percent=sum(a["utilization_percent"]*(b["unix"]-a["unix"]) for a,b in pairs)/seconds if seconds else None,
            maximum_memory_mib=max((r["memory_mib"] for r in valid), default=None)))
        write(root / "TASK_STATUS_FINAL.json", matrix(plan, results))
        write(root / "FINAL_STATE.json", final)
        event("queue_finished", status=final["status"])
        if lease.exists() and read(lease).get("token") == token:
            lease.unlink()
    return final
