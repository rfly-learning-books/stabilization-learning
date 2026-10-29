"""Observer-only launch contract, with strict runtime and complete provenance."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import time

from stabilization_learning.unified_v1 import runner as files
from extensions.reproducible_training_v4 import runner as retained
from extensions.reproducible_training_v4 import lifecycle
from extensions.reproducible_training_v4.runtime import check_lock, configure, manifest


def validate_config(config):
    if any(job.get("backend") != "observer_model_gradient" for job in config.get("jobs", [])):
        raise ValueError("This entry accepts observer_model_gradient jobs only")
    retained.validate_training_config(config)
    return config


def sources(project, config, config_path):
    project = Path(project).resolve()
    result = retained.preflight_sources(project, config, config_path)
    paths = [project / "scripts/run_observer_training_v5.py",
             project / "scripts/launch_observer_training_v5.ps1"]
    paths.extend(sorted((project / "extensions/observer_training_v5").glob("*.py")))
    for path in paths:
        result[str(path.resolve())] = files.sha256(path)
    return result


def verify_module_origins(project):
    """Check executed modules, not just the files expected by the inventory."""
    project = Path(project).resolve()
    locations = {}
    for name, expected in (("stabilization_gym", project / "stabilization-gym/stabilization_gym/__init__.py"),
                           ("stabilization_learning", project / "stabilization-learning-control/stabilization_learning/__init__.py"),
                           ("extensions.observer_training_v5.runner", project / "extensions/observer_training_v5/runner.py"),
                           ("extensions.reproducible_training_v4.runner", project / "extensions/reproducible_training_v4/runner.py"),
                           ("extensions.reproducible_training_v4.observer", project / "extensions/reproducible_training_v4/observer.py")):
        path = Path(importlib.import_module(name).__file__).resolve()
        if path != expected.resolve():
            raise RuntimeError("Unexpected module import location: " + str(path))
        locations[name] = dict(path=str(path), sha256=files.sha256(path))
    return locations


def diagnose(project, config, config_path):
    validate_config(config)
    locations = verify_module_origins(project)
    lock = check_lock(project, config)
    return dict(schema="observer_launch_diagnosis_v5", runtime_lock_passed=lock is not None,
                runtime_lock=str(lock), python_executable=sys.executable,
                package_records=len(manifest()["packages"]), modules=locations,
                sources=sources(project, config, config_path))


def preflight(config_path, config, output, *, project, deadline):
    validate_config(config)
    verify_module_origins(project)
    return retained.preflight(config_path, config, output, project=project, deadline=deadline)


def run_suite(config, root, *, project, config_path):
    """Start a fresh v4 lifecycle with the v5 launch sources also frozen."""
    project, root, config_path = Path(project).resolve(), Path(root).resolve(), Path(config_path).resolve()
    started = time.time()
    if root.exists():
        raise FileExistsError("Choose a fresh output directory; previous runs are immutable")
    preflight_path = root.with_name(root.name + ".PREFLIGHT.json")
    if preflight_path.exists():
        raise FileExistsError(preflight_path)
    configure()
    validate_config(config)
    verify_module_origins(project)
    if files.read(config_path) != config:
        raise ValueError("In-memory configuration differs from the declared file")
    cutoff = min(started + config["maximum_minutes"] * 60., config.get("absolute_deadline_unix", float("inf")))
    before = sources(project, config, config_path)
    preflight(config_path, config, preflight_path, project=project, deadline=cutoff)
    after = sources(project, config, config_path)
    if before != after:
        raise RuntimeError("A source changed during preflight")
    effective = deepcopy(config)
    effective["absolute_deadline_unix"] = cutoff
    effective["reproducibility_contract"] = dict(declared_config_path=str(config_path),
        declared_config_sha256=files.sha256(config_path), revision="observer_launch_v5", exact_runtime_required=True)
    remaining_minutes = (cutoff - time.time() - 30.) / 60.
    if remaining_minutes <= 0:
        raise TimeoutError("No training budget remains after preflight and setup reserve")
    effective["maximum_minutes"] = min(config["maximum_minutes"], remaining_minutes)
    effective["initializer_preflight"] = dict(path=str(preflight_path), sha256=files.sha256(preflight_path),
        original_maximum_minutes=config["maximum_minutes"], total_budget_started_unix=started,
        cutoff_unix=cutoff, setup_reserve_seconds=30.)

    def inventory(_project):
        if sources(project, config, config_path) != after:
            raise RuntimeError("A preflighted source changed before launch")
        result = dict(after)
        result[str(preflight_path)] = files.sha256(preflight_path)
        return result

    return lifecycle.run_suite(effective, root, project=project, inventory=inventory,
                               telemetry=False, factory=retained.make_backend)
