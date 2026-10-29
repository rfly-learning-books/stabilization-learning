"""Observer entry-point isolation and pre-collection protocol boundaries."""
from copy import deepcopy
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts" / "run_observer_training_v5.py"


def _key(path):
    return os.path.normcase(os.path.realpath(os.fspath(path)))


@pytest.fixture
def register_paths(monkeypatch):
    # Importing an entry point may register paths, but may not dispatch a job.
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = runpy.run_path(str(ENTRY), run_name="_observer_v5_import_test")
    return namespace["register_project_paths"]


def _project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for name in ("stabilization-gym", "stabilization-learning-control"):
        (project / name).mkdir()
    return project


def _paths(project):
    return tuple(str(path.resolve()) for path in (
        project, project / "stabilization-gym", project / "stabilization-learning-control"))


def test_path_registration_is_idempotent_and_retains_nonproject_order(
        tmp_path, monkeypatch, register_paths):
    project = _project(tmp_path)
    paths = _paths(project)
    foreign = str(tmp_path / "foreign_package_root")
    # A child directory is not a duplicate of the registered project root.
    child = str(project / "documentation")
    original = [foreign, paths[1], child, paths[0], foreign, paths[2], paths[1], *sys.path]
    monkeypatch.setattr(sys, "path", original)
    keys = {_key(path) for path in paths}
    expected_foreign = [path for path in original if _key(path) not in keys]

    result = register_paths(project=project)
    once = list(sys.path)
    assert isinstance(result, tuple) and result == paths
    assert [path for path in sys.path if _key(path) not in keys] == expected_foreign
    for path in paths:
        assert sum(_key(item) == _key(path) for item in sys.path) == 1
        assert path in sys.path

    assert register_paths(project=project) == result
    assert sys.path == once


def test_relative_and_dot_segment_aliases_collapse_to_one_project_entry(
        tmp_path, monkeypatch, register_paths):
    project = _project(tmp_path)
    paths = _paths(project)
    monkeypatch.chdir(tmp_path)
    aliases = [os.path.relpath(path, tmp_path) for path in paths]
    aliases += [str(Path(path) / ".." / Path(path).name) for path in paths]
    foreign = ["", str(tmp_path / "site-packages"), str(tmp_path / "site-packages")]
    monkeypatch.setattr(sys, "path", [*aliases, *foreign])

    assert register_paths(project=Path("project")) == paths
    keys = {_key(path) for path in paths}
    assert [path for path in sys.path if _key(path) not in keys] == foreign
    for path in paths:
        assert sum(_key(item) == _key(path) for item in sys.path) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows path equivalence is platform-specific")
def test_windows_case_and_separator_aliases_do_not_duplicate_packages(
        tmp_path, monkeypatch, register_paths):
    project = _project(tmp_path)
    paths = _paths(project)
    monkeypatch.chdir(tmp_path)
    aliases = []
    for path in paths:
        aliases.extend([path.upper(), path.replace("\\", "/"),
                        os.path.relpath(path, tmp_path).upper().replace("\\", "/")])
    foreign = [str(tmp_path / "external"), str(tmp_path / "external").upper()]
    monkeypatch.setattr(sys, "path", [*aliases, *foreign])

    assert register_paths(project=project) == paths
    keys = {_key(path) for path in paths}
    assert [path for path in sys.path if _key(path) not in keys] == foreign
    assert len(sys.path) == len(paths) + len(foreign)
    assert all(path in sys.path for path in paths)


def test_runpy_import_does_not_parse_arguments_or_start_training(tmp_path, monkeypatch):
    output = tmp_path / "must_not_be_created"
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [str(ENTRY), "train", "--config", "absent.json",
                                      "--output", str(output)])

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("Importing the observer entry point dispatched a command")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", unexpected_dispatch)
    monkeypatch.setattr(subprocess, "Popen", unexpected_dispatch)
    namespace = runpy.run_path(str(ENTRY), run_name="_observer_v5_not_main")
    assert callable(namespace["main"])
    assert callable(namespace["register_project_paths"])
    assert not output.exists()


@pytest.mark.parametrize("launcher_pythonpath", [False, True])
def test_entry_import_resolves_the_declared_environment_and_algorithm_packages(
        tmp_path, launcher_pythonpath):
    # A new interpreter avoids cached imports hiding a changed search order.
    code = (
        "import json, runpy; "
        f"runpy.run_path({str(ENTRY)!r}, run_name='_observer_v5_probe'); "
        "import stabilization_gym, stabilization_learning; "
        "print(json.dumps([stabilization_gym.__file__, stabilization_learning.__file__]))"
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if launcher_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(_paths(ROOT))
    else:
        env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=45, check=True)
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    expected = [ROOT / "stabilization-gym" / "stabilization_gym" / "__init__.py",
                ROOT / "stabilization-learning-control" / "stabilization_learning" / "__init__.py"]
    assert [_key(path) for path in actual] == [_key(path) for path in expected]


def _config(task="motor_speed"):
    return dict(schema="unified_training_v4", maximum_minutes=1,
        minimum_free_gib=0, deterministic=True, jobs=[dict(
            id="observer_" + task, backend="observer_model_gradient",
            config=dict(task=task, seed=41, cpu_threads=1, batch_size=2,
                        training_index_start=1000, validation_index_start=72000,
                        test_index_start=80000, validation_count=4, test_count=4,
                        scenarios=["nominal", "noise"]),
            schedule=dict(warmup_representation=0, pretrain_evaluator=0, epochs=3,
                          representation_updates=0, evaluator_updates=0, certificate_updates=1),
            test_variants=["selected", "initial", "identity", "fixed_M"])])


def test_original_formal_observer_configuration_remains_valid_and_unchanged():
    from extensions.observer_training_v5.runner import validate_config

    config = json.loads((ROOT / "extensions/book_overnight_v1/observers_seed41.json").read_text(
        encoding="utf-8-sig"))
    original = deepcopy(config)
    assert validate_config(config) == original
    assert config == original


@pytest.mark.parametrize("task", ["motor_speed", "cartpole_output", "cooperative_pose"])
def test_all_three_observer_tasks_are_admitted(task):
    from extensions.observer_training_v5.runner import validate_config

    config = _config(task)
    assert validate_config(config) == config


@pytest.mark.parametrize("backend", ["rgb_legacy", "state_model_gradient", "push_discrete_sac",
                                     "episode_ppo", "dc_motor_gradient", "unknown_backend"])
def test_nonobserver_backend_is_rejected_before_backend_specific_configuration(backend):
    from extensions.observer_training_v5.runner import validate_config

    config = _config()
    config["jobs"][0]["backend"] = backend
    config["jobs"][0]["config"] = {"task": "invalid_for_every_backend"}
    with pytest.raises(ValueError, match="(?i)observer"):
        validate_config(config)


def test_unrecognized_observer_task_is_rejected():
    from extensions.observer_training_v5.runner import validate_config

    config = _config()
    config["jobs"][0]["config"]["task"] = "unknown_scene_motion"
    with pytest.raises(ValueError, match="(?i)observer|task"):
        validate_config(config)


@pytest.mark.parametrize("updates,exact_start", [(1, 99992), (2, 99988), (4, 99976)])
def test_training_namespace_accounts_for_collection_and_pending_batches(updates, exact_start):
    from extensions.observer_training_v5.runner import validate_config

    config = _config()
    config["jobs"][0]["schedule"]["certificate_updates"] = updates
    config["jobs"][0]["config"]["training_index_start"] = exact_start
    assert validate_config(config) == config
    # Each individual batch fits. Only the complete planned run exceeds the
    # split namespace, so the rejection must account for the whole schedule.
    config["jobs"][0]["config"]["training_index_start"] += 1
    with pytest.raises(ValueError, match="(?i)index|indices|namespace|capacity|exhaust"):
        validate_config(config)


@pytest.mark.parametrize("field,value", [("batch_size", 0), ("cpu_threads", 2),
                                         ("gradient_clip", 0), ("validation_count", 0)])
def test_existing_observer_parameter_bounds_are_still_enforced(field, value):
    from extensions.observer_training_v5.runner import validate_config

    config = _config()
    config["jobs"][0]["config"][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("field", ["warmup_representation", "pretrain_evaluator",
                                  "representation_updates", "evaluator_updates"])
def test_observer_rejects_stages_without_a_feature_learner_or_critic(field):
    from extensions.observer_training_v5.runner import validate_config

    config = _config()
    config["jobs"][0]["schedule"][field] = 1
    with pytest.raises(ValueError):
        validate_config(config)


def test_module_origins_reject_a_different_declared_checkout(tmp_path):
    from extensions.observer_training_v5 import runner

    # The modules remain genuinely imported from ROOT. Merely declaring a
    # different checkout may not cause its expected inventory to stand in for
    # the source that is already executing.
    different_project = tmp_path / "other_checkout"
    different_project.mkdir()
    with pytest.raises(RuntimeError, match="Unexpected module import location"):
        runner.verify_module_origins(different_project)


@pytest.mark.parametrize("module_name", ["stabilization_gym", "stabilization_learning",
    "extensions.observer_training_v5.runner", "extensions.reproducible_training_v4.runner",
    "extensions.reproducible_training_v4.observer"])
def test_identical_version_and_source_from_another_location_is_rejected(
        tmp_path, monkeypatch, module_name):
    from extensions.observer_training_v5 import runner

    module = importlib.import_module(module_name)
    original = Path(module.__file__).resolve()
    original_version = getattr(module, "__version__", None)
    duplicate = tmp_path / "other_checkout" / original.relative_to(ROOT)
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(original.read_bytes())
    assert hashlib.sha256(duplicate.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest()
    monkeypatch.setattr(module, "__file__", str(duplicate))
    assert getattr(module, "__version__", None) == original_version

    with pytest.raises(RuntimeError, match="Unexpected module import location"):
        runner.verify_module_origins(ROOT)


@pytest.mark.parametrize("command", ["train", "preflight", "doctor"])
def test_wrong_import_is_rejected_before_inventory_initialization_or_child_process(
        tmp_path, monkeypatch, command):
    from extensions.observer_training_v5 import runner

    observer = importlib.import_module("extensions.reproducible_training_v4.observer")
    monkeypatch.setattr(observer, "__file__", str(tmp_path / "different_observer.py"))
    monkeypatch.setattr(runner, "configure", lambda: None)

    def unexpected_work(*args, **kwargs):
        pytest.fail("The entry proceeded past a mismatched imported module")

    monkeypatch.setattr(runner, "sources", unexpected_work)
    monkeypatch.setattr(runner, "check_lock", unexpected_work)
    monkeypatch.setattr(runner.files, "read", unexpected_work)
    monkeypatch.setattr(runner.retained, "preflight", unexpected_work)
    monkeypatch.setattr(runner.lifecycle, "run_suite", unexpected_work)
    monkeypatch.setattr(subprocess, "Popen", unexpected_work)
    config, config_path, output = _config(), tmp_path / "config.json", tmp_path / "run"

    with pytest.raises(RuntimeError, match="Unexpected module import location"):
        if command == "train":
            runner.run_suite(config, output, project=ROOT, config_path=config_path)
        elif command == "preflight":
            runner.preflight(config_path, config, output, project=ROOT, deadline=1.)
        else:
            runner.diagnose(ROOT, config, config_path)
    assert not output.exists()
    assert not output.with_name(output.name + ".PREFLIGHT.json").exists()
