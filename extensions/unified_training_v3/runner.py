"""New-run wrapper with process-isolated initialization and complete provenance."""
from copy import deepcopy
import json
import locale
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from stabilization_learning.unified_v1 import runner as v1
from stabilization_learning.unified_v1.registry import make_backend as legacy_factory
from extensions.unified_training_v2 import runner as v2
from . import lifecycle


def _validate_rgb_config(config):
    # These values are otherwise truncated by int(), or first used after
    # expensive initialization/collection, including held-out evaluation.
    counts = ("batch_size", "probe_batch_size", "replay_capacity", "replay_cached_episodes",
              "cpu_threads", "train_scene_count", "validation_cases", "test_cases",
              "initial_collection_steps", "collection_steps", "parallel_envs",
              "evaluation_batch", "evaluation_steps")
    for key in counts:
        if key in config and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError("RGB requires a positive integer: " + key)
    plants = config.get("evaluation_plants", ("P0", "P1"))
    if (not isinstance(plants, (list, tuple)) or not plants or
            any(not isinstance(p, str) or p not in ("P0", "P1") for p in plants) or
            len(set(plants)) != len(plants)):
        raise ValueError("RGB evaluation_plants must be a nonempty unique P0/P1 list")
    device = config.get("device", "cuda:0")
    if not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("RGB reconstructed environment requires a CUDA device declaration")


def validate_training_config(config):
    """Reject late-stage contract errors before any initializer or training."""
    v1.validate_config(config)
    reserve = config.get("minimum_free_gib", 5)
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("minimum_free_gib must be finite and nonnegative")
    if "absolute_deadline_unix" in config:
        cutoff = config["absolute_deadline_unix"]
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff):
            raise ValueError("absolute_deadline_unix must be finite")
    allowed = {
        "push_discrete_sac": {"selected", "initial", "identity", "random_initial", "identity_m", "learned", "final"},
        "state_model_gradient": {"selected", "initial", "identity", "fixed_M"},
        "observer_model_gradient": {"selected", "initial", "identity", "fixed_M"},
        "rgb_legacy": {"selected", "initial", "identity", "shadow"},
    }
    for job in config["jobs"]:
        variants = job.get("test_variants", ["selected", "initial", "identity"])
        options = allowed[job["backend"]].copy()
        if job["backend"] == "rgb_legacy" and job.get("config", {}).get("include_historical_reference"):
            options.update({"historical_trained_reference", "historical_trained_identity"})
        if not isinstance(variants, (list, tuple)) or any(v not in options for v in variants):
            raise ValueError(f"Unsupported test variant for {job['id']}: {variants}")
        if job["backend"] == "push_discrete_sac" and any(
                job.get("schedule", {}).get(key, 0) for key in ("warmup_representation", "representation_updates")):
            raise ValueError("Push representation is updated inside SAC; independent representation updates must be zero")
        if job["backend"] == "rgb_legacy":
            _validate_rgb_config(job.get("config", {}))
        if job["backend"] in ("state_model_gradient", "observer_model_gradient"):
            schedule = job.get("schedule", {})
            if any(schedule.get(key, 0) for key in ("warmup_representation", "pretrain_evaluator",
                                                    "representation_updates", "evaluator_updates")):
                raise ValueError("State/observer backends have no independent representation or evaluator updates")
            requested = schedule["epochs"] * schedule["certificate_updates"]
            if job["backend"] == "state_model_gradient":
                from .state import validate_config as validate_state
                effective = validate_state(job.get("config", {}))
                if requested > effective.updates:
                    raise ValueError("State schedule exceeds declared total_updates and its audited training seed interval")
            else:
                from .observer import validate_config as validate_observer
                effective = validate_observer(job.get("config", {}))
                # Initial collection plus one batch per epoch may collect more
                # than the update count; extra updates request fresh batches.
                batches = max(1 + schedule["epochs"], requested)
                count = effective.get("batch_size", 2) * batches
                if effective.get("training_index_start", 60000) + count > 100000:
                    raise ValueError("Observer schedule exhausts its declared training index namespace")
    return config


def make_backend(name, config, output, *, event, deadline):
    if name == "push_discrete_sac":
        from .push import FixedPushBackend
        return FixedPushBackend(config, output, event=event, deadline=deadline)
    if name == "rgb_legacy":
        from .rgb import StableRGBBackend, StableHistoricalRGBBackend
        cls = StableHistoricalRGBBackend if config.get("include_historical_reference") else StableRGBBackend
        return cls(config, output, event=event, deadline=deadline)
    if name == "state_model_gradient":
        from .state import StableStateBackend
        return StableStateBackend(config, output, event=event, deadline=deadline)
    if name == "observer_model_gradient":
        from .observer import StableObserverBackend
        return StableObserverBackend(config, output, event=event, deadline=deadline)
    return legacy_factory(name, config, output, event=event, deadline=deadline)


def _bank_path(project, job):
    from extensions.unified_training_v1.backends.push import PushConfig
    path = Path(job.get("config", {}).get("time_bank", PushConfig().time_bank)).expanduser()
    return (path if path.is_absolute() else Path(project) / path).resolve()


def extra_sources(project, config, config_path):
    project = Path(project).resolve()
    paths = {Path(config_path).resolve(), project / "scripts/run_unified_training_v3.py"}
    for folder in (project / "extensions/unified_training_v3", project / "third_party/so101_timeaware_v2"):
        paths.update(p.resolve() for p in folder.rglob("*.py"))
    for job in config["jobs"]:
        if job["backend"] == "push_discrete_sac":
            paths.add(_bank_path(project, job))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing new-run source or configured bank: {path}")
    return {str(path): v1.sha256(path) for path in sorted(paths)}


def preflight_sources(project, config, config_path, *, base_inventory=None):
    """Bind both retained scientific sources and the actual configured bank."""
    project = Path(project).resolve()
    sources = (base_inventory or v1.source_inventory)(project)
    sources.update(v2.visual_source_inventory(project))
    sources.update(extra_sources(project, config, config_path))
    paths = [project / "extensions/unified_training_v2/runner.py",
        project / "extensions/unified_training_v2/historical_rgb.py",
        project / "extensions/unified_training_v2/push_curves.py",
        project / "extensions/unified_training_v2/reporting.py",
        project / "scripts/run_unified_training_v2.py"]
    paths.extend(Path(p) for p in config.get("historical_reference", {}).get("source_paths", []))
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[str(path)] = v1.sha256(path)
    return sources


def _finish_timed_out_child(process):
    """Terminate only the owned preflight process tree and confirm its exit."""
    if os.name == "nt":
        # venv Python can create a second Python process; killing only the
        # launcher would leave an unowned initializer running on the machine.
        result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0:
            raise RuntimeError("Could not confirm termination of the owned preflight process tree")
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=10)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Preflight process tree exit could not be confirmed") from error


def _run_owned_child(command, *, cwd, timeout):
    child_env = os.environ.copy()
    # Only pipe encoding changes; -X utf8 and the default file encoding remain
    # identical to the parent so the original Windows failure stays testable.
    child_env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
    kwargs = dict(cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE, env=child_env,
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if os.name != "nt":
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _finish_timed_out_child(process)
        raise TimeoutError(f"Preflight exceeded {timeout:.1f}s for its owned process tree") from error
    except BaseException:
        # A cancelled parent must not abandon a still-running initializer.
        _finish_timed_out_child(process)
        raise
    return process.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


def preflight_job(config_path, job_id, output, result_path, expected_config_sha, deadline):
    """Run an actual initializer in this process without collection or updates."""
    if v1.sha256(config_path) != expected_config_sha:
        raise RuntimeError("Configuration changed before preflight initialization")
    config = v1.read(config_path)
    matches = [job for job in config["jobs"] if job["id"] == job_id]
    if len(matches) != 1:
        raise ValueError("Preflight requires exactly one declared job")
    job = matches[0]
    backend = make_backend(job["backend"], job.get("config", {}), Path(output),
        event=lambda *_args, **_kwargs: None, deadline=deadline)
    try:
        result = backend.initialize()
        evidence = dict(schema="unified_initializer_preflight_job_v3", job=job_id,
            backend=job["backend"], python_executable=sys.executable,
            preferred_encoding=locale.getpreferredencoding(False), utf8_mode=sys.flags.utf8_mode,
            initialized=True, bank_sha256=result.get("bank_sha256"),
            scope="initializer only; no collection, optimizer update, validation or test")
        v1.write_new(result_path, evidence)
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()


def preflight(config_path, config, output, *, project, deadline, timeout_per_job=180):
    """Check every declared initializer in a fresh process before any training."""
    validate_training_config(config)
    if not math.isfinite(deadline) or not math.isfinite(timeout_per_job) or timeout_per_job <= 0:
        raise ValueError("Preflight requires finite deadline and positive finite timeout")
    config_path, output, project = Path(config_path).resolve(), Path(output).resolve(), Path(project).resolve()
    if output.exists():
        raise FileExistsError(output)
    config_sha = v1.sha256(config_path)
    if v1.read(config_path) != config:
        raise ValueError("In-memory configuration does not match the declared file")
    result = dict(schema="unified_initializer_preflight_v3", status="running",
        config_path=str(config_path), config_sha256=config_sha, started_unix=time.time(),
        python_executable=sys.executable, preferred_encoding=locale.getpreferredencoding(False),
        utf8_mode=sys.flags.utf8_mode, jobs=[], scope="initialization only; no training or evaluation")
    try:
        with tempfile.TemporaryDirectory(prefix="unified_v3_preflight_") as temp:
            temp = Path(temp)
            for job in config["jobs"]:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Original preflight/run deadline reached")
                job_result = temp / (job["id"] + "_result.json")
                job_output = temp / "jobs" / job["id"]
                command = [sys.executable, "-X", f"utf8={sys.flags.utf8_mode}",
                    str(project / "scripts/run_unified_training_v3.py"), "_preflight_job",
                    "--config", str(config_path), "--job", job["id"], "--output", str(job_output),
                    "--result", str(job_result), "--config-sha256", config_sha, "--deadline", str(deadline)]
                timeout = min(float(timeout_per_job), remaining)
                returncode, stdout, stderr = _run_owned_child(command, cwd=project, timeout=timeout)
                if returncode != 0 or not job_result.is_file():
                    result["jobs"].append(dict(job=job["id"], backend=job["backend"],
                        initialized=False, returncode=returncode, stdout_tail=stdout[-3000:], stderr_tail=stderr[-6000:]))
                    raise RuntimeError(f"Initializer preflight failed for {job['id']}; inspect {output}")
                item = v1.read(job_result)
                if item.get("job") != job["id"] or item.get("backend") != job["backend"] or not item.get("initialized"):
                    raise RuntimeError("Preflight child returned invalid identity or status")
                result["jobs"].append(item)
                if v1.sha256(config_path) != config_sha:
                    raise RuntimeError("Configuration changed during preflight")
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["finished_unix"] = time.time()
        v1.write_new(output, result)
    return result


def run_suite(config, root, *, project, config_path, telemetry=True):
    """Preflight and then execute a fresh suite; never resume or replace a run."""
    project, root, config_path = Path(project).resolve(), Path(root).resolve(), Path(config_path).resolve()
    started = time.time()
    if root.exists():
        raise FileExistsError("Choose a fresh run directory; old runs cannot be resumed")
    preflight_path = root.with_name(root.name + ".PREFLIGHT.json")
    if preflight_path.exists():
        raise FileExistsError(preflight_path)
    v2.validate_extra(config, project)
    validate_training_config(config)
    if v1.read(config_path) != config:
        raise ValueError("In-memory configuration differs from the frozen file")
    cutoff = min(started + float(config["maximum_minutes"]) * 60., float(config["absolute_deadline_unix"]))
    original_inventory = v1.source_inventory
    before = preflight_sources(project, config, config_path, base_inventory=original_inventory)
    checks = preflight(config_path, config, preflight_path, project=project, deadline=cutoff)
    after = preflight_sources(project, config, config_path, base_inventory=original_inventory)
    if before != after:
        raise RuntimeError("A scientific source or configured time bank changed during preflight")
    for job in config["jobs"]:
        if job["backend"] == "push_discrete_sac":
            item = next(x for x in checks["jobs"] if x["job"] == job["id"])
            if item.get("bank_sha256") != after[str(_bank_path(project, job))]:
                raise RuntimeError("The Push bank loaded during preflight changed")
    effective = deepcopy(config)
    # Source hashing and suite setup consume time. Keep a bounded reserve;
    # never add time to either the user budget or absolute deadline.
    remaining_minutes = (cutoff - time.time() - 30.) / 60.
    if remaining_minutes <= 0:
        raise TimeoutError("No training budget remains after preflight and setup reserve")
    effective["maximum_minutes"] = min(float(config["maximum_minutes"]), remaining_minutes)
    effective["initializer_preflight"] = dict(path=str(preflight_path), sha256=v1.sha256(preflight_path),
        original_maximum_minutes=config["maximum_minutes"], total_budget_started_unix=started,
        cutoff_unix=cutoff, setup_reserve_seconds=30.)
    def inventory(root_project):
        if preflight_sources(project, config, config_path, base_inventory=original_inventory) != after:
            raise RuntimeError("A preflighted source or configured time bank changed before launch")
        sources = original_inventory(root_project)
        sources.update(after)
        sources[str(preflight_path)] = v1.sha256(preflight_path)
        return sources
    return lifecycle.run_suite(effective, root, project=project, inventory=inventory,
                               telemetry=telemetry, factory=make_backend)
