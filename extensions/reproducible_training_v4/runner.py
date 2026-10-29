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


from .validation import validate_config as validate_training_config
from .runtime import configure, check_lock
from .assets import historical_sources


def make_backend(name, config, output, *, event, deadline):
    config = deepcopy(config)
    if name == "rgb_legacy":
        config.setdefault("cpu_threads", 1)
        from .deterministic_rgb import DeterministicRGBBackend, DeterministicHistoricalRGBBackend
        backend = DeterministicHistoricalRGBBackend if config.get("include_historical_reference") else DeterministicRGBBackend
        return backend(config, output, event=event, deadline=deadline)
    if name == "observer_model_gradient":
        from .observer import CorrectedObserverBackend
        return CorrectedObserverBackend(config, output, event=event, deadline=deadline)
    if name == "episode_ppo":
        from .independent_backend import EpisodeBackend
        return EpisodeBackend(config, output, event=event, deadline=deadline)
    if name == "dc_motor_gradient":
        from .dc_motor import MotorBackend
        return MotorBackend(config, output, event=event, deadline=deadline)
    from extensions.unified_training_v3.runner import make_backend as guarded_factory
    return guarded_factory(name, config, output, event=event, deadline=deadline)


def _bank_path(project, job):
    from extensions.unified_training_v1.backends.push import PushConfig
    path = Path(job.get("config", {}).get("time_bank", PushConfig().time_bank)).expanduser()
    return (path if path.is_absolute() else Path(project) / path).resolve()


def extra_sources(project, config, config_path):
    project = Path(project).resolve()
    paths = {Path(config_path).resolve(), project / "scripts/run_unified_training_v4.py"}
    for asset_config in (project/"assets.json", Path(os.environ["STABILIZATION_ASSET_CONFIG"]).expanduser()
                         if os.environ.get("STABILIZATION_ASSET_CONFIG") else project/"assets.json"):
        if asset_config.is_file():
            paths.add(asset_config.resolve())
    locked_runtime = check_lock(project, config)
    if locked_runtime is not None:
        paths.add(locked_runtime)
    for folder in (project / "extensions/unified_training_v3", project / "third_party/so101_timeaware_v2",
                   project / "extensions/reproducible_training_v4", project / "extensions/fast_m_learning_v2",
                   project / "extensions/state_conditioned_m_v1", project / "scripts/m_learning_suite",
                   project / "migration/next_envs_v1", project / "third_party/push_original/dl_push"):
        paths.update(p.resolve() for p in folder.rglob("*.py"))
    for job in config["jobs"]:
        if job["backend"] == "push_discrete_sac":
            paths.add(_bank_path(project, job))
        elif job["backend"] == "episode_ppo" and job.get("config", {}).get("task") == "push_any_shape":
            from .push import source_files_for_config
            paths.update(Path(p) for p in source_files_for_config(job["config"].get("physical", {})))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing new-run source or configured bank: {path}")
    return {str(path): v1.sha256(path) for path in sorted(paths)}


def preflight_sources(project, config, config_path, *, base_inventory=None):
    """Bind both retained scientific sources and the actual configured bank."""
    project = Path(project).resolve()
    sources = (base_inventory or v1.source_inventory)(project)
    if any(job["backend"] == "rgb_legacy" for job in config["jobs"]):
        sources.update(v2.visual_source_inventory(project))
    sources.update(extra_sources(project, config, config_path))
    sources.update(historical_sources(project, config))
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
        evidence = dict(schema="unified_initializer_preflight_job_v4", job=job_id,
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
    check_lock(project, config)
    historical_sources(project, config)
    if output.exists():
        raise FileExistsError(output)
    config_sha = v1.sha256(config_path)
    if v1.read(config_path) != config:
        raise ValueError("In-memory configuration does not match the declared file")
    result = dict(schema="unified_initializer_preflight_v4", status="running",
        config_path=str(config_path), config_sha256=config_sha, started_unix=time.time(),
        python_executable=sys.executable, preferred_encoding=locale.getpreferredencoding(False),
        utf8_mode=sys.flags.utf8_mode, jobs=[], scope="initialization only; no training or evaluation")
    try:
        with tempfile.TemporaryDirectory(prefix="unified_v4_preflight_") as temp:
            temp = Path(temp)
            for job in config["jobs"]:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Original preflight/run deadline reached")
                job_result = temp / (job["id"] + "_result.json")
                job_output = temp / "jobs" / job["id"]
                command = [sys.executable, "-X", f"utf8={sys.flags.utf8_mode}",
                    str(project / "scripts/run_unified_training_v4.py"), "_preflight_job",
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
    configure()
    validate_training_config(config)
    if v1.read(config_path) != config:
        raise ValueError("In-memory configuration differs from the frozen file")
    cutoff = min(started + float(config["maximum_minutes"]) * 60., float(config.get("absolute_deadline_unix", float("inf"))))
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
    effective["absolute_deadline_unix"] = cutoff
    effective["reproducibility_contract"] = dict(declared_config_path=str(config_path),
        declared_config_sha256=v1.sha256(config_path), revision="v4", exact_runtime_required=True)
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
