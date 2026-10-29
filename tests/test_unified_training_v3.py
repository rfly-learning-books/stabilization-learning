"""Regression tests for encoding, provenance, and pre-training initialization."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "runs/scratch_apps_so101_full_v1_20261026_184356/so101_environment/time_bank.json"


def _bank_data():
    return json.loads(BANK.read_text(encoding="utf-8"))


def _write_bank(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def _config(bank):
    return dict(schema="unified_training_v1", maximum_minutes=3, minimum_free_gib=0,
        jobs=[dict(id="push_probe", backend="push_discrete_sac",
            config=dict(time_bank=str(bank), seed=71963, num_envs=2, initial_steps=4,
                        collection_steps=2, batch_size=8, validation_repeats=1, test_repeats=1,
                        validation_seed_start=6371000, test_seed_start=6372000),
            schedule=dict(epochs=1, certificate_updates=1), test_variants=["selected", "initial", "identity"])])


def test_gbk_process_loads_unicode_bank_and_steps(tmp_path):
    data = _bank_data()
    data["notes"] = "中文标定说明 🧪"
    bank = tmp_path / "unicode_bank.json"
    _write_bank(bank, data)
    code = """
import locale, sys
from extensions.unified_training_v3.environment import make_push
bundle = make_push(shapes=['T'], time_bank=sys.argv[1], tensor_observations=True)
obs, _ = bundle.reset(seed=71963)
following, reward, terminated, truncated, info = bundle.step(0)
assert obs['q'].shape == (3,) and following['q'].shape == (3,)
assert info['dt'] >= 0 and not truncated
print(locale.getpreferredencoding(False), 'utf8_mode=' + str(sys.flags.utf8_mode), bundle.spec.name)
assert not any(name == 'stabilization_learning' or name.startswith('stabilization_learning.') for name in sys.modules)
bundle.close()
"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(map(str, (ROOT, ROOT / "stabilization-gym", ROOT / "stabilization-learning-control")))
    result = subprocess.run([sys.executable, "-X", "utf8=0", "-c", code, str(bank)], cwd=ROOT,
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert result.returncode == 0, result.stderr
    assert "cross_shape_calibrated" in result.stdout
    assert "utf8_mode=0" in result.stdout


@pytest.mark.parametrize("corruption", ["nonfinite_transfer", "wrong_reachability", "nonfinite_travel", "wrong_boolean"])
def test_malformed_time_bank_rejected_before_training(tmp_path, corruption):
    from third_party.so101_timeaware_v2.so101_timeaware.timing import TimeBank
    data = _bank_data()
    if corruption == "nonfinite_transfer":
        data["shapes"]["T"]["transfer_seconds"][0][0] = float("nan")
    elif corruption == "wrong_reachability":
        data["shapes"]["T"]["reachable"].pop()
    elif corruption == "nonfinite_travel":
        data["config"]["travels"][0] = float("nan")
    else:
        data["shapes"]["T"]["valid"][0] = "False"
    bank = tmp_path / "invalid.json"
    _write_bank(bank, data)
    with pytest.raises(ValueError):
        TimeBank(bank)


def test_vendor_import_does_not_change_caller_threads(tmp_path):
    code = """
import sys, torch
torch.set_num_threads(3)
from extensions.unified_training_v3.environment import make_push
bundle = make_push(shapes=['T'], time_bank=sys.argv[1])
assert torch.get_num_threads() == 3
bundle.close()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(map(str, (ROOT, ROOT / "stabilization-gym", ROOT / "stabilization-learning-control")))
    proc = subprocess.run([sys.executable, "-X", "utf8=0", "-c", code, str(BANK)], cwd=ROOT,
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert proc.returncode == 0, proc.stderr


def test_unknown_loaded_vendor_is_rejected(monkeypatch):
    from extensions.unified_training_v3.environment import vendor_modules
    vendor_modules()
    from dl_push.certificate import full_m_control
    monkeypatch.setattr(full_m_control, "__file__", str(ROOT / "not_the_recorded_vendor.py"))
    with pytest.raises(RuntimeError, match="Unexpected previously imported vendor"):
        vendor_modules()


def test_preflight_timeout_stops_its_python_process_tree():
    psutil = pytest.importorskip("psutil")
    from extensions.unified_training_v3.runner import _run_owned_child
    marker = "unified_v3_owned_probe_" + uuid.uuid4().hex
    code = ("import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',sys.argv[1]]); "
            "time.sleep(30)")
    with pytest.raises(TimeoutError, match="owned process tree"):
        _run_owned_child([sys.executable, "-c", code, marker], cwd=ROOT, timeout=2.5)
    survivors = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info.get("cmdline") or []
            if marker in command:
                survivors.append(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    assert not survivors


def test_same_bank_and_updates_match_preserved_backend(tmp_path):
    from extensions.unified_training_v1.backends.push import _module_hash
    from extensions.unified_training_v2.push_curves import RecordedPushBackend
    from extensions.unified_training_v3.push import FixedPushBackend
    cfg = _config(BANK)["jobs"][0]["config"]
    records = []
    for cls, name in ((RecordedPushBackend, "old"), (FixedPushBackend, "fixed")):
        backend = cls(cfg, tmp_path / name)
        try:
            initialized = backend.initialize()
            # Hold execution settings constant: the historical vendor import
            # has a one-time process-wide thread side effect.
            torch.set_num_threads(1)
            collection = backend.collect(initial=True)
            pretrain = backend.pretrain_evaluator(2)
            update = backend.train_epoch(epoch=1, representation_updates=0, evaluator_updates=2, certificate_updates=2)
            validation = backend.validate()
            records.append(dict(initial=initialized["initial_actor_sha256"],
                bank=initialized["bank_sha256"], transitions=collection["new_transitions"],
                pretrain_actor_unchanged=pretrain["actor_unchanged"], counts=update["counts"],
                actor=_module_hash(backend.actor), q1=_module_hash(backend.q1),
                score=validation["selection_score"], episodes=validation["episodes"]))
        finally:
            backend.close()
    assert records[0] == records[1]


def test_actual_gbk_preflight_precedes_training_and_detects_bad_bank(tmp_path):
    valid = tmp_path / "valid_bank.json"
    data = _bank_data()
    data["notes"] = "UTF-8 中文测试 🧪"
    _write_bank(valid, data)
    config = _config(valid)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(map(str, (ROOT, ROOT / "stabilization-gym", ROOT / "stabilization-learning-control")))
    preflight = tmp_path / "passed.json"
    command = [sys.executable, "-X", "utf8=0", str(ROOT / "scripts/run_unified_training_v3.py"),
               "preflight", "--config", str(config_file), "--output", str(preflight)]
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(preflight.read_text(encoding="utf-8"))
    assert result["status"] == "passed" and result["jobs"][0]["initialized"]
    assert result["jobs"][0]["utf8_mode"] == 0
    assert result["jobs"][0]["bank_sha256"]
    data["shapes"]["T"]["transfer_seconds"][0][0] = float("nan")
    _write_bank(valid, data)
    failed = tmp_path / "failed.json"
    command[-1] = str(failed)
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert proc.returncode != 0
    result = json.loads(failed.read_text(encoding="utf-8"))
    assert result["status"] == "failed" and not result["jobs"][0]["initialized"]
    assert "Invalid numerical bank" in result["jobs"][0]["stderr_tail"]
    assert not (tmp_path / "run").exists()


def test_new_runner_binds_custom_bank_preflight_and_original_budget(tmp_path, monkeypatch):
    from extensions.unified_training_v3 import runner
    bank = tmp_path / "custom_bank.json"
    _write_bank(bank, _bank_data())
    config = _config(bank)
    config["absolute_deadline_unix"] = time.time() + 600
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner.v2, "validate_extra", lambda *_: None)
    monkeypatch.setattr(runner.v2, "visual_source_inventory", lambda *_: {})
    monkeypatch.setattr(runner.v1, "source_inventory", lambda *_: {})
    def fake_preflight(path, cfg, output, **_kwargs):
        result = dict(status="passed", jobs=[dict(job="push_probe", bank_sha256=runner.v1.sha256(bank))])
        runner.v1.write_new(output, result)
        return result
    monkeypatch.setattr(runner, "preflight", fake_preflight)
    def fake_run(effective, root, **_kwargs):
        sources = _kwargs["inventory"](ROOT)
        assert sources[str(bank.resolve())] == runner.v1.sha256(bank)
        assert sources[effective["initializer_preflight"]["path"]] == effective["initializer_preflight"]["sha256"]
        assert 0 < effective["maximum_minutes"] < config["maximum_minutes"]
        assert effective["initializer_preflight"]["cutoff_unix"] <= config["absolute_deadline_unix"]
        return {"status": "preflight_test_only"}
    monkeypatch.setattr(runner.lifecycle, "run_suite", fake_run)
    assert runner.run_suite(config, tmp_path / "run", project=ROOT, config_path=config_file)["status"] == "preflight_test_only"
    assert not (tmp_path / "run").exists()


def test_new_runner_refuses_bank_changed_during_preflight(tmp_path, monkeypatch):
    from extensions.unified_training_v3 import runner
    bank = tmp_path / "bank.json"
    _write_bank(bank, _bank_data())
    config = _config(bank)
    config["absolute_deadline_unix"] = time.time() + 600
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner.v2, "validate_extra", lambda *_: None)
    monkeypatch.setattr(runner.v2, "visual_source_inventory", lambda *_: {})
    monkeypatch.setattr(runner.v1, "source_inventory", lambda *_: {})
    def change_bank(_path, _cfg, output, **_kwargs):
        result = dict(status="passed", jobs=[dict(job="push_probe", bank_sha256=runner.v1.sha256(bank))])
        runner.v1.write_new(output, result)
        bank.write_text("{}", encoding="utf-8")
        return result
    monkeypatch.setattr(runner, "preflight", change_bank)
    with pytest.raises(RuntimeError, match="changed during preflight"):
        runner.run_suite(config, tmp_path / "run", project=ROOT, config_path=config_file)
    assert not (tmp_path / "run").exists()


def test_real_runner_preserves_preflight_and_group_barrier(tmp_path, monkeypatch):
    from extensions.unified_training_v3 import runner
    bank = tmp_path / "bank.json"
    _write_bank(bank, _bank_data())
    config = _config(bank)
    config["absolute_deadline_unix"] = time.time() + 600
    config["historical_reference"] = {"source_paths": []}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner.v2, "validate_extra", lambda *_: None)
    monkeypatch.setattr(runner.v2, "visual_source_inventory", lambda *_: {})
    monkeypatch.setattr(runner.v1, "source_inventory", lambda *_: {})
    def fake_preflight(path, cfg, output, **_kwargs):
        result = dict(status="passed", jobs=[dict(job="push_probe", bank_sha256=runner.v1.sha256(bank))])
        runner.v1.write_new(output, result)
        return result
    monkeypatch.setattr(runner, "preflight", fake_preflight)
    class FakeBackend:
        def __init__(self, output): self.output = output
        def initialize(self): return {}
        def validate(self): return {"selection_score": [1.]}
        def save(self, path): path.write_text("test checkpoint", encoding="utf-8")
        def collect(self, initial=False): return {}
        def warmup_representation(self, updates): return {}
        def pretrain_evaluator(self, updates): return {}
        def train_epoch(self, **kwargs): return {}
        def load(self, path): assert path.is_file()
        def evaluate(self, split, variants):
            runner.v1.check_frozen(self.output.parents[1], ["push_probe"])
            return {"split": split, "variants": list(variants)}
    monkeypatch.setattr(runner, "make_backend", lambda _name, _cfg, output, **_kwargs: FakeBackend(output))
    root = tmp_path / "engineering_run"
    result = runner.run_suite(config, root, project=ROOT, config_path=config_file, telemetry=False)
    assert result["status"] == "complete"
    protocol = runner.v1.read(root / "protocol.json")
    assert str(bank.resolve()) in protocol["sources"]
    assert protocol["config"]["initializer_preflight"]["path"] in protocol["sources"]
    assert (root / "GLOBAL_FROZEN.json").is_file()
    assert (root / "jobs/push_probe/TEST_RESULTS.json").is_file()
    assert not (root / "WRITER.lock").exists()


def test_cpu_push_full_engineering_lifecycle(tmp_path, monkeypatch):
    """Use distinct engineering seeds and a fresh temp root, not formal cases."""
    from extensions.unified_training_v3 import runner
    config = _config(BANK)
    config["absolute_deadline_unix"] = time.time() + 600
    config["historical_reference"] = {"source_paths": []}
    config["jobs"][0]["schedule"] = dict(epochs=1, warmup_representation=0,
        pretrain_evaluator=2, representation_updates=0, evaluator_updates=2, certificate_updates=2)
    config_file = tmp_path / "engineering_config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    # The production v2 suite requires two RGB reference arms. This CPU-only
    # engineering test isolates the actual Push lifecycle instead.
    monkeypatch.setattr(runner.v2, "validate_extra", lambda *_: None)
    monkeypatch.setattr(runner.v2, "visual_source_inventory", lambda *_: {})
    monkeypatch.setattr(runner.v1, "source_inventory", lambda *_: {})
    root = tmp_path / "engineering_only"
    result = runner.run_suite(config, root, project=ROOT, config_path=config_file, telemetry=False)
    assert result["status"] == "complete"
    protocol = runner.v1.read(root / "protocol.json")
    preflight = runner.v1.read(protocol["config"]["initializer_preflight"]["path"])
    assert preflight["status"] == "passed" and preflight["jobs"][0]["initialized"]
    assert runner.v1.check_frozen(root, ["push_probe"])
    tested = runner.v1.read(root / "jobs/push_probe/TEST_RESULTS.json")["result"]
    assert set(tested["variants"]) == {"selected", "initial", "identity"}
    assert all(v["cases"] == 2 for v in tested["variants"].values())
    assert not (root / "WRITER.lock").exists()
