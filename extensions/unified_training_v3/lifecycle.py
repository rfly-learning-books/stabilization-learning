"""Fresh-run lifecycle with guarded completion and immutable checkpoint commits.

Keep the original stage order, validation selection and complete-group barrier.
Guards are cooperative: they reject late results, not interrupt native kernels.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import traceback
import uuid

from stabilization_learning.unified_v1 import runner as v1


def commit_checkpoint(backend, folder, name, metadata, guard):
    """Publish a completed file without exposing partial saves or replacing one."""
    folder = Path(folder)
    path = folder / "checkpoints" / (name + ".pt")
    marker_path = path.with_suffix(".COMMITTED.json")
    guard()
    if path.exists() or marker_path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / ("." + name + "-" + uuid.uuid4().hex + ".uncommitted")
    staging.mkdir()
    staged = staging / path.name
    # Failed saves stay in a distinctly uncommitted directory for diagnosis.
    backend.save(staged)
    guard()
    if not staged.is_file() or staged.stat().st_size == 0:
        raise RuntimeError("Backend save did not produce a nonempty checkpoint")
    # Windows FlushFileBuffers requires a writable handle.
    with staged.open("r+b") as stream:
        os.fsync(stream.fileno())
    digest = v1.sha256(staged)
    marker = dict(checkpoint=str(path.relative_to(folder)), sha256=digest,
                  created_unix=time.time(), **metadata)
    v1.plain(marker)
    guard()
    # link is atomic and refuses an existing destination on Windows and POSIX.
    # A filesystem without hard links fails closed; it must not silently replace.
    os.link(staged, path)
    guard()
    v1.write_new(marker_path, marker)
    staged.unlink()
    staging.rmdir()
    return marker


def run_suite(config, root, *, project, factory, inventory, telemetry=True):
    """Run the declared group using the preflight's original absolute cutoff."""
    v1.validate_config(config)
    root, project = Path(root).resolve(), Path(project).resolve()
    sources = inventory(project)
    started = time.time()
    deadline = min(started + 60 * config["maximum_minutes"],
                   config["absolute_deadline_unix"],
                   config["initializer_preflight"]["cutoff_unix"])
    if started >= deadline:
        raise TimeoutError("Original budget consumed before suite creation")
    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema="unified_training_run_v1", runner_revision="v3_stability",
        config=config, sources=sources, started_unix=started, deadline_unix=deadline,
        created_utc=datetime.now(timezone.utc).isoformat(),
        minimum_free_gib=config.get("minimum_free_gib", 5), automatic_resume=False,
        scope="bounded integration training; not a full reproduction or stability proof")
    v1.write_new(root / "protocol.json", protocol)
    owner = v1.Evidence(root, protocol)
    backends, jobs = {}, {}
    final = dict(status="running", started_unix=started, jobs={})
    resource = v1.GPUTelemetry(root) if telemetry else None
    primary_error = None
    try:
        if resource:
            resource.__enter__()
        owner.event("suite_started", jobs=[j["id"] for j in config["jobs"]])
        for job in config["jobs"]:
            owner.verify_sources()
            jid = job["id"]
            folder = root / "jobs" / jid
            folder.mkdir(parents=True, exist_ok=False)
            v1.write_new(folder / "protocol.json", dict(schema="unified_job_v1", job=job,
                suite_protocol_sha256=owner.protocol_hash, deadline_unix=deadline))
            event = lambda kind, _jid=jid, **payload: owner.event(kind, job=_jid, **payload)
            backend = factory(job["backend"], job.get("config", {}), folder,
                              event=event, deadline=deadline)
            backends[jid] = backend
            stages = []

            def guard():
                owner.guard()
                if (folder / "STOP").exists():
                    raise InterruptedError("Job STOP requested")

            def phase(name, function):
                guard()
                owner.event("stage_started", job=jid, stage=name)
                begin = time.time()
                outcome = v1.plain(function())
                guard()
                result = dict(stage=name, started_unix=begin, finished_unix=time.time(), result=outcome)
                v1.write_new(folder / "stages" / (name + ".json"), result)
                stages.append(result)
                owner.event("stage_completed", job=jid, stage=name, result=outcome)
                return outcome

            def checkpoint(name, metadata):
                return commit_checkpoint(backend, folder, name, metadata, guard)

            phase("initialize", backend.initialize)
            with v1.preserve_rng():
                initial_validation = phase("validate_initial", backend.validate)
            score = v1.selection_score(initial_validation)
            initial = checkpoint("initial", {"epoch": 0})
            selected = dict(initial, selected_epoch=0, candidate_stage="initial", selection_score=score)
            phase("collect_initial", lambda: backend.collect(initial=True))
            sched = job["schedule"]
            phase("warmup_representation", lambda: backend.warmup_representation(sched.get("warmup_representation", 0)))
            phase("pretrain_evaluator", lambda: backend.pretrain_evaluator(sched.get("pretrain_evaluator", 0)))
            with v1.preserve_rng():
                first_validation = phase("validate_ready", backend.validate)
            score = v1.selection_score(first_validation)
            ready = checkpoint("ready", {"epoch": 0, "after_pretraining": True})
            if score > tuple(selected["selection_score"]):
                selected = dict(ready, selected_epoch=0, candidate_stage="ready", selection_score=score)
            for epoch in range(1, sched["epochs"] + 1):
                phase(f"collect_epoch_{epoch:04d}", lambda: backend.collect(initial=False))
                phase(f"train_epoch_{epoch:04d}", lambda: backend.train_epoch(epoch=epoch,
                    representation_updates=sched.get("representation_updates", 0),
                    evaluator_updates=sched.get("evaluator_updates", 0),
                    certificate_updates=sched["certificate_updates"]))
                with v1.preserve_rng():
                    validation = phase(f"validate_epoch_{epoch:04d}", backend.validate)
                score = v1.selection_score(validation)
                current = checkpoint(f"epoch_{epoch:04d}", {"epoch": epoch})
                if score > tuple(selected["selection_score"]):
                    selected = dict(current, selected_epoch=epoch, candidate_stage="epoch", selection_score=score)
                owner.event("epoch_completed", job=jid, epoch=epoch,
                            selection_score=score, selected_epoch=selected["selected_epoch"])
            owner.verify_sources()
            guard()
            frozen = dict(selected, schema="unified_frozen_v1", initial=initial,
                protocol_sha256=v1.sha256(folder / "protocol.json"), frozen_unix=time.time(),
                selection="backend validation score; initial and ready candidates eligible; test never consulted")
            v1.write_new(folder / "FROZEN.json", frozen)
            jobs[jid] = dict(backend=job["backend"], stages=stages, frozen=frozen)
            final["jobs"][jid] = {"status": "trained_frozen", "selected_epoch": selected["selected_epoch"]}
            owner.event("job_frozen", job=jid, selected_epoch=selected["selected_epoch"])
        owner.guard()
        v1.write_new(root / "GLOBAL_FROZEN.json", dict(schema="unified_frozen_group_v1", created_unix=time.time(),
            protocol_sha256=owner.protocol_hash, jobs={jid: v1.sha256(root/"jobs"/jid/"FROZEN.json") for jid in jobs}))
        owner.event("all_models_frozen", jobs=list(jobs))
        for job in config["jobs"]:
            owner.verify_sources()
            v1.check_frozen(root, jobs)
            jid, backend = job["id"], backends[job["id"]]
            folder = root / "jobs" / jid
            guard()
            backend.load(folder / jobs[jid]["frozen"]["checkpoint"])
            guard()
            owner.event("test_started", job=jid)
            results = v1.plain(backend.evaluate(split="test", variants=job.get("test_variants", ["selected", "initial", "identity"])))
            guard()
            owner.verify_sources()
            v1.check_frozen(root, jobs)
            guard()
            v1.write_new(folder / "TEST_RESULTS.json", dict(started_after_group_freeze=True,
                group_freeze_sha256=v1.sha256(root/"GLOBAL_FROZEN.json"), result=results, finished_unix=time.time()))
            jobs[jid]["test"] = results
            final["jobs"][jid]["status"] = "complete"
            owner.event("test_completed", job=jid, result=results)
        owner.guard()
        final["status"] = "complete"
    except BaseException as error:
        primary_error = error
        final.update(status="stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError)) else "failed",
                     error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        try:
            owner.event("suite_stopped", status=final["status"], error=repr(error))
        except Exception as event_error:
            final["event_write_error"] = str(event_error)
        raise
    finally:
        for backend in backends.values():
            close = getattr(backend, "close", None)
            if close:
                try:
                    close()
                except BaseException as error:
                    final.setdefault("cleanup_errors", []).append(repr(error))
        if resource:
            try:
                resource.__exit__()
            except BaseException as error:
                final.setdefault("cleanup_errors", []).append(repr(error))
        completion_error = None
        if final["status"] == "complete":
            try:
                if final.get("cleanup_errors"):
                    raise RuntimeError("Backend cleanup failed; inspect FINAL_STATE.json")
                owner.guard()
                v1.write_new(root / "RESULTS.json", dict(schema="unified_suite_results_v1", jobs=jobs,
                    scientific_scope=protocol["scope"], new_performance_guarantee=False))
                owner.guard()
            except BaseException as error:
                completion_error = error
                final.update(status="stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError)) else "failed",
                             error_type=type(error).__name__, error=str(error))
        final.update(finished_unix=time.time(), wall_seconds=time.time()-started)
        # Failure to persist final evidence deliberately keeps the lock: a later
        # report must diagnose the incomplete writer instead of assuming success.
        v1.write_new(root / "FINAL_STATE.json", final)
        if final["status"] == "complete":
            owner.event("suite_completed", status="complete")
        elif primary_error is None:
            owner.event("suite_stopped", status=final["status"], error=final.get("error"))
        # An event-write failure must also leave the reporting interlock intact.
        owner.close()
        if completion_error is not None:
            raise completion_error
    return final
