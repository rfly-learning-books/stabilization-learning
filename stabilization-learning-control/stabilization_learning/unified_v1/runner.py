"""One owner, explicit optional stages, validation selection and a group test barrier.

This is a new integration protocol. It never resumes or changes old experiments.
Each backend retains its own representation, objective, action and data semantics.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid

from .registry import BACKENDS, make_backend


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite result must be diagnosed, not written as a metric")
    return value


def write_new(path, value):
    path = Path(path)
    encoded = json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows rename fails if destination exists, preserving earlier evidence.
    temporary.rename(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_config(config):
    if config.get("schema") != "unified_training_v1":
        raise ValueError("Expected unified_training_v1 configuration")
    cap = config.get("maximum_minutes")
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("A positive finite new-run budget is required")
    jobs = config.get("jobs", [])
    if not jobs or not isinstance(jobs, list):
        raise ValueError("Declare the entire job group before training")
    names = set()
    for job in jobs:
        name = job.get("id", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or name in names:
            raise ValueError("Unique simple job IDs required")
        names.add(name)
        if job.get("backend") not in BACKENDS:
            raise ValueError("Unsupported backend")
        schedule = job.get("schedule", {})
        for key in ("warmup_representation", "pretrain_evaluator", "epochs", "representation_updates",
                    "evaluator_updates", "certificate_updates"):
            number = schedule.get(key, 0)
            if type(number) is not int or number < 0:
                raise ValueError("Schedule counts must be nonnegative integers")
        if schedule.get("epochs", 0) < 1 or schedule.get("certificate_updates", 0) < 1:
            raise ValueError("Integration training must include actual certificate updates")
        variants = job.get("test_variants", ["selected", "initial", "identity"])
        if "selected" not in variants or len(variants) != len(set(variants)):
            raise ValueError("Unique test variants including selected are required")
        # Data split definitions live in each task protocol; shared scalar IDs are
        # checked here when provided, in addition to backend-specific checks.
        cfg = job.get("config", {})
        for first, second in (("train_seeds", "validation_seeds"), ("train_seeds", "test_seeds"),
                              ("validation_seeds", "test_seeds")):
            if set(cfg.get(first, [])) & set(cfg.get(second, [])):
                raise ValueError("Training/validation/test seeds overlap")
    plain(config)
    return config


def source_inventory(project):
    project = Path(project)
    folders = [
        "stabilization-learning-control/stabilization_learning",
        "stabilization-gym/stabilization_gym",
        "stabilization-learning-control/stabilization_learning/unified_v1",
        "extensions/unified_training_v1", "extensions/rgb_diagnostic_v5",
        "extensions/rgb_gpu_parallel", "extensions/joint_certificate_v4",
        "extensions/observer_learning_v1",
        "stabilization-gym/stabilization_gym/tasks_v2",
        "stabilization-gym/stabilization_gym/tasks_gpu_rgb",
        "stabilization-gym/stabilization_gym/observation",
        "stabilization-gym/stabilization_gym/envs/classic_control",
        "stabilization-learning-control/stabilization_learning/algos/pytorch/state_metric_full_horizon",
        "stabilization-learning-control/stabilization_learning/policies",
        "third_party/visual_original/runtime",
        "third_party/push_original/dl_push/certificate",
        "third_party/so101_timeaware_v1",
        "assets/recovered_visual_legacy/src",
        "migration/visual_scene_reconstruction/reconstructed_legacy_root",
    ]
    paths = {p.resolve() for folder in folders for p in (project / folder).rglob("*.py")}
    entry = project / "scripts/run_unified_training_v1.py"
    if entry.exists():
        paths.add(entry.resolve())
    for relative in (
        "assets/original_visual_initialization/initial.pt",
        "assets/original_visual_initialization/protocol.json",
        "migration/visual_scene_reconstruction/scenes.json",
        "migration/visual_scene_reconstruction/SCENE_AUDIT.json",
        "migration/visual_initialization_bundle.json",
        "migration/visual_source_bundle.json",
        "third_party/visual_original/assets/old_depth_audit/model/model.safetensors",
        "runs/scratch_apps_so101_full_v1_20261026_184356/so101_environment/time_bank.json",
    ):
        path = project / relative
        if path.exists():
            paths.add(path.resolve())
    return {str(p): sha256(p) for p in sorted(paths)}


class Evidence:
    def __init__(self, root, protocol):
        self.root, self.protocol = Path(root), protocol
        self.protocol_hash = sha256(self.root / "protocol.json")
        self.sequence = 0
        self.token = uuid.uuid4().hex
        self.lease = self.root / "WRITER.lock"
        with self.lease.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "token": self.token}, stream)

    def guard(self):
        if read(self.lease).get("token") != self.token:
            raise RuntimeError("Writer ownership changed")
        if sha256(self.root / "protocol.json") != self.protocol_hash:
            raise RuntimeError("Protocol changed after launch")
        if (self.root / "STOP").exists() or time.time() >= self.protocol["deadline_unix"]:
            raise InterruptedError("STOP or original run deadline reached")
        if shutil.disk_usage(self.root).free < self.protocol.get("minimum_free_gib", 5) * 1024**3:
            raise InterruptedError("Disk reserve reached")

    def verify_sources(self):
        self.guard()
        for path, expected in self.protocol["sources"].items():
            if sha256(path) != expected:
                raise RuntimeError("Frozen scientific source changed: " + path)

    def event(self, kind, **data):
        self.sequence += 1
        record = dict(sequence=self.sequence, kind=kind, unix=time.time(), pid=os.getpid(),
                      protocol_sha256=self.protocol_hash, data=plain(data))
        write_new(self.root / "events" / f"{self.sequence:08d}_{kind}.json", record)
        # A bounded live progress channel; checkpoints/rollouts need not be read.
        print(json.dumps({"event": kind, "sequence": self.sequence,
                          **{k: v for k, v in data.items() if k in ("job", "epoch", "stage", "status")}},
                         ensure_ascii=False), flush=True)

    def close(self):
        if self.lease.exists() and read(self.lease).get("token") == self.token:
            self.lease.unlink()


class GPUTelemetry:
    """Full-run samples, including CPU phases; no best-window selection."""
    def __init__(self, root):
        self.path = Path(root) / "GPU_SAMPLES.jsonl"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self.path.open("x", encoding="utf-8") as stream:
            while not self.stop.is_set():
                item = {"unix": time.time()}
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4,
                        stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr.strip())
                    item["gpus"] = [dict(zip(("utilization_percent", "memory_mib", "power_watts"),
                                             map(float, line.split(","))))
                                     for line in result.stdout.strip().splitlines()]
                except Exception as error:
                    item["error"] = str(error)
                stream.write(json.dumps(item) + "\n")
                stream.flush()
                self.stop.wait(5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


@contextmanager
def preserve_rng():
    from stabilization_learning.joint_v2.runtime import capture_rng, restore_rng
    saved = capture_rng()
    try:
        yield
    finally:
        restore_rng(saved)


def selection_score(result):
    score = result.get("selection_score")
    if score is None:
        raise ValueError("Backend must supply its predeclared validation score")
    if not isinstance(score, (list, tuple)):
        score = [score]
    if not score or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in score):
        raise ValueError("Validation score must be finite, ordered and explicit")
    return tuple(score)


def commit_checkpoint(backend, folder, name, metadata):
    path = folder / "checkpoints" / (name + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    backend.save(path)
    if not path.is_file():
        raise RuntimeError("Backend save did not produce its checkpoint")
    marker = dict(checkpoint=str(path.relative_to(folder)), sha256=sha256(path),
                  created_unix=time.time(), **metadata)
    write_new(path.with_suffix(".COMMITTED.json"), marker)
    return marker


def check_frozen(root, expected_jobs):
    root = Path(root)
    global_record = read(root / "GLOBAL_FROZEN.json")
    if global_record["protocol_sha256"] != sha256(root / "protocol.json"):
        raise RuntimeError("Suite protocol changed")
    if set(global_record["jobs"]) != set(expected_jobs):
        raise RuntimeError("Entire declared group must be frozen")
    for job in expected_jobs:
        folder = root / "jobs" / job
        marker = read(folder / "FROZEN.json")
        path = (folder / marker["checkpoint"]).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != marker["sha256"]:
            raise RuntimeError("Frozen checkpoint hash/path mismatch")
        if sha256(folder / "FROZEN.json") != global_record["jobs"][job]:
            raise RuntimeError("Frozen group manifest changed")
        if sha256(folder / "protocol.json") != marker["protocol_sha256"]:
            raise RuntimeError("Job protocol changed")
        if read(folder / "protocol.json")["suite_protocol_sha256"] != global_record["protocol_sha256"]:
            raise RuntimeError("Job protocol belongs to another suite")
    return global_record


def run_suite(config, root, *, project, factory=make_backend, telemetry=True):
    config = validate_config(config)
    root, project = Path(root).resolve(), Path(project).resolve()
    sources = source_inventory(project)
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = dict(schema="unified_training_run_v1", config=config, sources=sources,
                    started_unix=started, deadline_unix=started+60*config["maximum_minutes"],
                    created_utc=datetime.now(timezone.utc).isoformat(), minimum_free_gib=config.get("minimum_free_gib", 5),
                    automatic_resume=False, scope="bounded integration training; not a full reproduction or stability proof")
    write_new(root / "protocol.json", protocol)
    owner = Evidence(root, protocol)
    backends = {}
    jobs = {}
    final = dict(status="running", started_unix=started, jobs={})
    resource = GPUTelemetry(root) if telemetry else None
    try:
        if resource:
            resource.__enter__()
        owner.event("suite_started", jobs=[j["id"] for j in config["jobs"]])
        for job in config["jobs"]:
            owner.verify_sources()
            jid = job["id"]
            folder = root / "jobs" / jid
            folder.mkdir(parents=True, exist_ok=False)
            write_new(folder / "protocol.json", dict(schema="unified_job_v1", job=job,
                suite_protocol_sha256=owner.protocol_hash, deadline_unix=protocol["deadline_unix"]))
            event = lambda kind, _jid=jid, **payload: owner.event(kind, job=_jid, **payload)
            backend = factory(job["backend"], job.get("config", {}), folder,
                              event=event, deadline=protocol["deadline_unix"])
            backends[jid] = backend
            stages = []
            def phase(name, function):
                owner.guard()
                if (folder / "STOP").exists():
                    raise InterruptedError("Job STOP requested")
                owner.event("stage_started", job=jid, stage=name)
                begin = time.time()
                outcome = plain(function())
                result = dict(stage=name, started_unix=begin, finished_unix=time.time(), result=outcome)
                stages.append(result)
                write_new(folder / "stages" / (name + ".json"), result)
                owner.event("stage_completed", job=jid, stage=name, result=outcome)
                return outcome
            phase("initialize", backend.initialize)
            with preserve_rng():
                initial_validation = phase("validate_initial", backend.validate)
            initial = commit_checkpoint(backend, folder, "initial", {"epoch": 0})
            selected = dict(initial, selected_epoch=0, candidate_stage="initial",
                            selection_score=selection_score(initial_validation))
            phase("collect_initial", lambda: backend.collect(initial=True))
            sched = job["schedule"]
            phase("warmup_representation", lambda: backend.warmup_representation(sched.get("warmup_representation", 0)))
            phase("pretrain_evaluator", lambda: backend.pretrain_evaluator(sched.get("pretrain_evaluator", 0)))
            with preserve_rng():
                first_validation = phase("validate_ready", backend.validate)
            ready = commit_checkpoint(backend, folder, "ready", {"epoch": 0, "after_pretraining": True})
            if selection_score(first_validation) > tuple(selected["selection_score"]):
                selected = dict(ready, selected_epoch=0, candidate_stage="ready",
                                selection_score=selection_score(first_validation))
            # Retaining initial/ready is a valid non-improvement result; reports
            # must distinguish this from selecting a main-training checkpoint.
            for epoch in range(1, sched["epochs"] + 1):
                phase(f"collect_epoch_{epoch:04d}", lambda: backend.collect(initial=False))
                phase(f"train_epoch_{epoch:04d}", lambda: backend.train_epoch(epoch=epoch,
                    representation_updates=sched.get("representation_updates", 0),
                    evaluator_updates=sched.get("evaluator_updates", 0),
                    certificate_updates=sched["certificate_updates"]))
                with preserve_rng():
                    validation = phase(f"validate_epoch_{epoch:04d}", backend.validate)
                # The preserved state learner can roll back during validation;
                # snapshot the active model to which its returned score applies.
                current = commit_checkpoint(backend, folder, f"epoch_{epoch:04d}", {"epoch": epoch})
                score = selection_score(validation)
                if score > tuple(selected["selection_score"]):
                    selected = dict(current, selected_epoch=epoch, candidate_stage="epoch", selection_score=score)
                owner.event("epoch_completed", job=jid, epoch=epoch,
                            selection_score=score, selected_epoch=selected["selected_epoch"])
            owner.verify_sources()
            frozen = dict(selected, schema="unified_frozen_v1", initial=initial,
                          protocol_sha256=sha256(folder / "protocol.json"), frozen_unix=time.time(),
                          selection="backend validation score; initial and ready candidates eligible; test never consulted")
            write_new(folder / "FROZEN.json", frozen)
            jobs[jid] = dict(backend=job["backend"], stages=stages, frozen=frozen)
            final["jobs"][jid] = {"status": "trained_frozen", "selected_epoch": selected["selected_epoch"]}
            owner.event("job_frozen", job=jid, selected_epoch=selected["selected_epoch"])
        write_new(root / "GLOBAL_FROZEN.json", dict(schema="unified_frozen_group_v1", created_unix=time.time(),
            protocol_sha256=owner.protocol_hash, jobs={jid: sha256(root/"jobs"/jid/"FROZEN.json") for jid in jobs}))
        owner.event("all_models_frozen", jobs=list(jobs))
        for job in config["jobs"]:
            owner.verify_sources()
            check_frozen(root, jobs)
            jid, backend = job["id"], backends[job["id"]]
            folder = root / "jobs" / jid
            frozen = jobs[jid]["frozen"]
            backend.load(folder / frozen["checkpoint"])
            owner.event("test_started", job=jid)
            results = plain(backend.evaluate(split="test", variants=job.get("test_variants", ["selected", "initial", "identity"])))
            write_new(folder / "TEST_RESULTS.json", dict(started_after_group_freeze=True,
                group_freeze_sha256=sha256(root/"GLOBAL_FROZEN.json"), result=results, finished_unix=time.time()))
            check_frozen(root, jobs)
            jobs[jid]["test"] = results
            final["jobs"][jid]["status"] = "complete"
            owner.event("test_completed", job=jid, result=results)
        write_new(root / "RESULTS.json", dict(schema="unified_suite_results_v1", jobs=jobs,
            scientific_scope=protocol["scope"], new_performance_guarantee=False))
        final["status"] = "complete"
        owner.event("suite_completed", status="complete")
    except BaseException as error:
        final.update(status="stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                     error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        owner.event("suite_stopped", status=final["status"], error=repr(error))
        raise
    finally:
        for backend in backends.values():
            close = getattr(backend, "close", None)
            if close:
                try:
                    close()
                except Exception as error:
                    final.setdefault("cleanup_errors", []).append(str(error))
        if resource:
            resource.__exit__()
        final.update(finished_unix=time.time(), wall_seconds=time.time()-started)
        write_new(root / "FINAL_STATE.json", final)
        owner.close()
    return final
