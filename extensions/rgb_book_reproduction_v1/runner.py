"""Book RGB training stages in the shared Env/Algo/Runner architecture."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import time
import traceback

from stabilization_learning.unified_v1 import runner as v1
from extensions.unified_training_v2.runner import visual_source_inventory


BOOK_SCHEDULE = dict(warmup_representation=1000, pretrain_evaluator=1000,
                     epochs=24, representation_updates=300,
                     evaluator_updates=0, certificate_updates=200)


def validate(config):
    v1.validate_config(config)
    if len(config["jobs"]) != 1 or config["jobs"][0]["backend"] != "rgb_legacy":
        raise ValueError("Book reproduction declares exactly one RGB job")
    book = config.get("book_protocol", {})
    mode = book.get("mode", "formal")
    if mode not in ("formal", "engineering"):
        raise ValueError("book_protocol.mode must be formal or engineering")
    cadence = book.get("validation_every", 2)
    if type(cadence) is not int or cadence < 1:
        raise ValueError("validation_every must be a positive integer")
    if book.get("select_initial", False) or book.get("select_ready", False):
        raise ValueError("Book model selection uses training-round validation only")
    job = config["jobs"][0]
    cfg, schedule = job["config"], job["schedule"]
    if cfg.get("epochs") != schedule["epochs"]:
        raise ValueError("Backend and Runner epoch counts must agree")
    if schedule["epochs"] < cadence or schedule["epochs"] % cadence:
        raise ValueError("Final epoch must have a scheduled validation")
    if cfg.get("cross_epochs") is not None:
        raise ValueError("A historical cross-checkpoint diagnostic is not a training selection rule")
    if mode == "formal":
        if schedule != BOOK_SCHEDULE or cadence != 2:
            raise ValueError("Formal book reproduction preserves 1000/1000/24*(300+200) and 2-round validation")
        if cfg.get("seed") != 41 or cfg.get("batch_size") != 4 or cfg.get("collection_steps") != 120:
            raise ValueError("Formal book reproduction uses seed41, batch4 and 120-step trajectories")
        if cfg.get("validation_plants") != ["P0"] or cfg.get("validation_steps") != 300:
            raise ValueError("Book selection uses P0 validation with a 300-step horizon")
        if cfg.get("trajectories_per_scene") != 2 or cfg.get("bootstrap_transitions") != 2822:
            raise ValueError("Book collection uses 2822 bootstrap transitions and two trajectories per training scene")
        variants = ["selected", "identity", "shadow", "initial_identity"]
        if job.get("test_variants") != variants:
            raise ValueError("Preserve the four original book evaluation methods")
        cases = v1.read(cfg["book_cases"])
        if len(cases.get("val", [])) != 48 or len(cases.get("test", [])) != 360:
            raise ValueError("The book start manifest must contain 48 validation and 360 held-out test starts")
        expected_suites = [
            ("book_full_val", "val", variants, ["P0"], 48, 1.),
            ("book_test", "test", variants, ["P0", "P1"], 360, 1.),
            ("depth_0_5", "test", ["selected", "identity"], ["P0"], 72, .5),
            ("depth_2", "test", ["selected", "identity"], ["P0"], 72, 2.)]
        suites = cfg.get("generalization_suites", [])
        if len(suites) != len(expected_suites):
            raise ValueError("Declare all four book evaluation suites")
        for suite, (name, case_key, methods, plants, count, depth) in zip(suites, expected_suites):
            observed = (suite.get("name"), suite.get("cases"), suite.get("variants"), suite.get("plants"),
                suite.get("case_limit", len(cases[case_key])), suite.get("depth_scale", 1.), suite.get("steps"))
            if observed != (name, case_key, methods, plants, count, depth, 300):
                raise ValueError("Book evaluation suite differs from the frozen 3360-episode protocol: " + name)
        if config["maximum_minutes"] > 360:
            raise ValueError("The authorized maximum run budget is 360 minutes")
        if config.get("minimum_free_gib", 20) < 20:
            raise ValueError("Formal runs retain a 20 GiB disk reserve")
    deadline = config.get("absolute_deadline_unix")
    if deadline is not None and (not isinstance(deadline, (int, float)) or
                                isinstance(deadline, bool) or not math.isfinite(deadline)):
        raise ValueError("Invalid absolute deadline")
    return config


def source_inventory(project, config, config_path):
    project = Path(project).resolve()
    sources = v1.source_inventory(project)
    sources.update(visual_source_inventory(project))
    paths = list((project / "extensions/rgb_book_reproduction_v1").glob("*.py"))
    paths += list((project / "extensions/rgb_longroll_v1").glob("*.py"))
    paths += [project / "scripts/run_rgb_book_reproduction_v1.py", Path(config_path),
              project / "extensions/unified_training_v2/runner.py",
              project / "extensions/unified_training_v2/historical_rgb.py"]
    paths += [Path(p) for p in config.get("source_files", [])]
    paths += [Path(p) for p in config.get("comparative_sources", {}).values()]
    cfg = config["jobs"][0]["config"]
    for key in ("sl_cases", "book_cases", "book_validation_cases", "initial_checkpoint",
                "bootstrap_manifest", "bootstrap_provenance"):
        if cfg.get(key):
            paths.append(Path(cfg[key]))
    for ref in cfg.get("references", {}).values():
        paths += [Path(ref[k]) for k in ("encoder", "metric") if ref.get(k)]
    for key in ("bootstrap_replay", "bootstrap_root", "bootstrap_directory"):
        if cfg.get(key):
            paths += [p for p in Path(cfg[key]).rglob("*") if p.is_file()]
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[str(path)] = v1.sha256(path)
    return sources


def compact_result(value):
    """Stage references keep trace rows in their original evaluation artifacts."""
    if isinstance(value, dict):
        return {k: compact_result(v) for k, v in value.items()
                if k not in ("rows", "episodes", "trace", "matrices", "information_windows")}
    if isinstance(value, list):
        return [compact_result(v) for v in value]
    return value


def run_suite(config, root, *, project, config_path, telemetry=True, factory=None,
              deadline=None, inventory_function=None):
    validate(config)
    project, root = Path(project).resolve(), Path(root).resolve()
    if root.exists():
        raise FileExistsError("A new run directory is required")
    started = time.time()
    limits = [started + 60 * config["maximum_minutes"]]
    limits += [float(v) for v in (config.get("absolute_deadline_unix"), deadline) if v is not None]
    hard_deadline = min(limits)
    sources = (inventory_function or source_inventory)(project, config, config_path)
    if time.time() >= hard_deadline:
        raise TimeoutError("Source verification consumed the launch budget")
    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema="unified_training_run_v1", implementation="rgb_book_reproduction_v1",
        config=config, sources=sources, started_unix=started, deadline_unix=hard_deadline,
        created_utc=datetime.now(timezone.utc).isoformat(), minimum_free_gib=config.get("minimum_free_gib", 20),
        automatic_resume=False, scope="Book RGB stage/sampling/selection reproduction in SL with CUDA network training")
    v1.write_new(root / "protocol.json", protocol)
    owner = v1.Evidence(root, protocol)
    resource = v1.GPUTelemetry(root) if telemetry else None
    backend = None
    final = dict(status="running", started_unix=started, jobs={})
    job = config["jobs"][0]
    jid, cadence = job["id"], config.get("book_protocol", {}).get("validation_every", 2)
    folder = root / "jobs" / jid
    try:
        if resource:
            resource.__enter__()
        owner.verify_sources()
        owner.event("suite_started", jobs=[jid], protocol_mode=config.get("book_protocol", {}).get("mode", "formal"))
        folder.mkdir(parents=True, exist_ok=False)
        v1.write_new(folder / "protocol.json", dict(schema="unified_job_v1", job=job,
            suite_protocol_sha256=owner.protocol_hash, deadline_unix=hard_deadline))
        if factory is None:
            from .backend import BookRGBBackend
            factory = lambda name, cfg, output, **kwargs: BookRGBBackend(cfg, output, **kwargs)
        backend = factory(job["backend"], job["config"], folder,
            event=lambda kind, **payload: owner.event(kind, job=jid, **payload), deadline=hard_deadline)
        stages = []

        def phase(name, function):
            owner.guard()
            if (folder / "STOP").exists():
                raise InterruptedError("Job STOP requested")
            owner.event("stage_started", job=jid, stage=name)
            begin = time.time()
            outcome = v1.plain(function())
            record = dict(stage=name, started_unix=begin, finished_unix=time.time(), result=compact_result(outcome))
            path = folder / "stages" / (name + ".json")
            v1.write_new(path, record)
            stages.append(dict(stage=name, path=str(path.relative_to(folder)), sha256=v1.sha256(path)))
            owner.event("stage_completed", job=jid, **record)
            return outcome

        phase("initialize", backend.initialize)
        initial = v1.commit_checkpoint(backend, folder, "initial", {"epoch": 0, "selection_eligible": False})
        if config.get("book_protocol", {}).get("diagnostic_initial", True):
            with v1.preserve_rng():
                phase("validate_initial", backend.validate)
        phase("collect_initial", lambda: backend.collect(initial=True))
        schedule = job["schedule"]
        phase("warmup_representation", lambda: backend.warmup_representation(schedule["warmup_representation"]))
        phase("pretrain_evaluator", lambda: backend.pretrain_evaluator(schedule["pretrain_evaluator"]))
        ready = v1.commit_checkpoint(backend, folder, "ready", {"epoch": 0, "after_pretraining": True, "selection_eligible": False})
        if config.get("book_protocol", {}).get("diagnostic_ready", True):
            with v1.preserve_rng():
                phase("validate_ready", backend.validate)
        selected = None
        for epoch in range(1, schedule["epochs"] + 1):
            phase(f"collect_epoch_{epoch:04d}", lambda: backend.collect(initial=False))
            phase(f"train_epoch_{epoch:04d}", lambda: backend.train_epoch(epoch=epoch,
                representation_updates=schedule["representation_updates"],
                evaluator_updates=schedule["evaluator_updates"],
                certificate_updates=schedule["certificate_updates"]))
            validation = None
            if epoch % cadence == 0:
                with v1.preserve_rng():
                    validation = phase(f"validate_epoch_{epoch:04d}", backend.validate)
            marker = v1.commit_checkpoint(backend, folder, f"epoch_{epoch:04d}",
                {"epoch": epoch, "selection_eligible": validation is not None})
            score = v1.selection_score(validation) if validation is not None else None
            if score is not None and (selected is None or score > tuple(selected["selection_score"])):
                selected = dict(marker, selected_epoch=epoch, candidate_stage="epoch", selection_score=score)
            final["jobs"][jid] = dict(status="training", completed_epochs=epoch,
                selected_epoch=selected["selected_epoch"] if selected else None)
            owner.event("epoch_completed", job=jid, epoch=epoch, selection_score=score,
                selected_epoch=selected["selected_epoch"] if selected else None,
                validation=compact_result(validation) if validation is not None else None)
        if selected is None:
            raise RuntimeError("No eligible book validation checkpoint")
        owner.verify_sources()
        frozen = dict(selected, schema="unified_frozen_v1", initial=initial, ready=ready,
            protocol_sha256=v1.sha256(folder / "protocol.json"), frozen_unix=time.time(),
            selection="Every second training epoch, P0 macro success / worst-scene success / negative mean integrated error; strict lexicographic improvement")
        v1.write_new(folder / "FROZEN.json", frozen)
        owner.event("job_frozen", job=jid, selected_epoch=selected["selected_epoch"])
        v1.write_new(root / "GLOBAL_FROZEN.json", dict(schema="unified_frozen_group_v1", created_unix=time.time(),
            protocol_sha256=owner.protocol_hash, jobs={jid: v1.sha256(folder / "FROZEN.json")}))
        owner.event("all_models_frozen", jobs=[jid])
        owner.verify_sources()
        v1.check_frozen(root, [jid])
        backend.load(folder / frozen["checkpoint"])
        final["jobs"][jid]["status"] = "trained_frozen"
        owner.event("test_started", job=jid)
        results = v1.plain(backend.evaluate(split="test", variants=job.get("test_variants", ["selected", "initial", "identity"])))
        test = dict(started_after_group_freeze=True, group_freeze_sha256=v1.sha256(root / "GLOBAL_FROZEN.json"),
            result=compact_result(results), finished_unix=time.time())
        v1.write_new(folder / "TEST_RESULTS.json", test)
        v1.check_frozen(root, [jid])
        owner.verify_sources()
        owner.event("test_completed", job=jid, result=compact_result(results))
        v1.write_new(root / "RESULTS.json", dict(schema="rgb_book_suite_results_v1", jobs={jid: dict(
            backend=job["backend"], stages=stages, frozen=frozen, test=compact_result(results))}, scientific_scope=protocol["scope"]))
        final["jobs"][jid]["status"] = "complete"
        final["status"] = "complete"
        owner.event("suite_completed", status="complete")
    except BaseException as error:
        final.update(status="stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        owner.event("suite_stopped", status=final["status"], error=repr(error))
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                final.setdefault("cleanup_errors", []).append(str(error))
        if resource:
            resource.__exit__()
        final.update(finished_unix=time.time(), wall_seconds=time.time()-started)
        v1.write_new(root / "FINAL_STATE.json", final)
        owner.close()
    return final


def supervise(config_path, root, *, project):
    config = validate(v1.read(config_path))
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("A new run directory is required")
    started = time.time()
    deadline = min(started+60*config["maximum_minutes"], config.get("absolute_deadline_unix", float("inf")))
    if started >= deadline:
        raise TimeoutError("Declared deadline has expired")
    command = [sys.executable, "-X", "utf8", "-u", str(Path(project)/"scripts/run_rgb_book_reproduction_v1.py"),
        "worker", "--config", str(Path(config_path).resolve()), "--output", str(root), "--deadline", str(deadline)]
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(json.dumps(dict(event="supervisor_started", pid=os.getpid(), child_pid=child.pid,
        root=str(root), deadline_unix=deadline)), flush=True)
    stop_seen, forced = None, False
    while child.poll() is None:
        now = time.time()
        if ((root/"STOP").exists() or now >= deadline) and stop_seen is None:
            stop_seen = now
        if stop_seen is not None and now-stop_seen >= 15:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                    capture_output=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
            else:
                child.terminate()
            forced = True
            break
        time.sleep(1)
    code = child.wait(timeout=20)
    result = dict(schema="rgb_book_supervisor_v1", parent_pid=os.getpid(), child_pid=child.pid,
        started_unix=started, finished_unix=time.time(), exit_code=code, deadline_unix=deadline,
        forced_child_tree_stop=forced, automatic_resume=False)
    if root.exists():
        v1.write_new(root/"SUPERVISOR_FINAL.json", result)
    return result
