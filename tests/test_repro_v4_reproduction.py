"""Fault injection for immutable two-run comparison, without formal training."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from extensions.reproducible_training_v4 import reproduction as repro


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(processes=None):
    return dict(schema="unified_training_process_snapshot", query_success=True,
        created_unix=time.time(), processes=processes if processes is not None else [
            dict(ProcessId=os.getpid(), ParentProcessId=0, Name="python.exe", CommandLine="python -m pytest")])


@pytest.fixture
def external(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(snapshot()).encode(), stderr=b"")
    monkeypatch.setattr(repro.subprocess, "run", run)
    return calls


def refresh(root):
    folder = root/"jobs/probe"
    frozen = read(folder/"FROZEN.json")
    marker = read((folder/frozen["checkpoint"]).with_suffix(".COMMITTED.json"))
    frozen.update(marker)
    write(folder/"FROZEN.json", frozen)
    group = read(root/"GLOBAL_FROZEN.json")
    group["jobs"]["probe"] = repro.files.sha256(folder/"FROZEN.json")
    write(root/"GLOBAL_FROZEN.json", group)
    test = read(folder/"TEST_RESULTS.json")
    test["group_freeze_sha256"] = repro.files.sha256(root/"GLOBAL_FROZEN.json")
    write(folder/"TEST_RESULTS.json", test)
    result = read(root/"RESULTS.json")
    result["jobs"]["probe"].update(frozen=frozen, test=test["result"],
        stages=[read(path) for path in sorted((folder/"stages").glob("*.json"))])
    write(root/"RESULTS.json", result)


def rebind_protocol(root, protocol):
    write(root/"protocol.json", protocol)
    folder = root/"jobs/probe"
    job = read(folder/"protocol.json")
    job.update(job=protocol["config"]["jobs"][0], suite_protocol_sha256=repro.files.sha256(root/"protocol.json"))
    write(folder/"protocol.json", job)
    frozen = read(folder/"FROZEN.json")
    frozen["protocol_sha256"] = repro.files.sha256(folder/"protocol.json")
    write(folder/"FROZEN.json", frozen)
    group = read(root/"GLOBAL_FROZEN.json")
    group["protocol_sha256"] = repro.files.sha256(root/"protocol.json")
    write(root/"GLOBAL_FROZEN.json", group)
    refresh(root)


def alter_payload(root, change, name="epoch_0001"):
    path = root/"jobs/probe/checkpoints"/(name+".pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    change(payload)
    torch.save(payload, path)
    marker_path = path.with_suffix(".COMMITTED.json")
    marker = read(marker_path)
    marker["sha256"] = repro.files.sha256(path)
    write(marker_path, marker)
    refresh(root)


@pytest.fixture
def pair(tmp_path):
    cfg = dict(schema="unified_training_v4", maximum_minutes=10, minimum_free_gib=0,
        jobs=[dict(id="probe", backend="episode_ppo", config=dict(task="two_link_arm"),
                   schedule=dict(epochs=1, certificate_updates=1), test_variants=["selected"])])
    declared = tmp_path/"declared.json"
    write(declared, cfg)
    source = tmp_path/"scientific_source.py"
    source.write_text("scientific_equation = 1\n", encoding="utf-8")
    for side in ("run_a", "run_b"):
        root = tmp_path/side
        folder = root/"jobs/probe"
        preflight = tmp_path/(side+".PREFLIGHT.json")
        write(preflight, dict(status="passed", publication=side))
        effective = deepcopy(cfg)
        effective.update(maximum_minutes=9, absolute_deadline_unix=time.time()+600,
            reproducibility_contract=dict(declared_config_path=str(declared),
                declared_config_sha256=repro.files.sha256(declared), revision="v4", exact_runtime_required=True),
            initializer_preflight=dict(path=str(preflight), sha256=repro.files.sha256(preflight),
                original_maximum_minutes=10, cutoff_unix=time.time()+600))
        protocol = dict(schema="unified_training_run_v1", runner_revision="v4_reproducible_contracts",
            config=effective, runtime={"runtime": "same CPU fixture"}, deadline_unix=time.time()+500,
            sources={str(path): repro.files.sha256(path) for path in (declared, source, preflight)})
        write(root/"protocol.json", protocol)
        write(folder/"protocol.json", dict(schema="unified_job_v1", job=cfg["jobs"][0],
            suite_protocol_sha256=repro.files.sha256(root/"protocol.json")))
        (folder/"replay").mkdir(parents=True)
        np.savez(folder/"replay/shard.npz", state=np.array([1., 2.]))
        for name, epoch in (("initial", 0), ("ready", 0), ("epoch_0001", 1)):
            path = folder/"checkpoints"/(name+".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(actor={"weight": torch.tensor([1.+epoch])},
                optimizer={"state": {0: {"exp_avg": torch.tensor([.1])}}, "param_groups": [{"lr": .001}]},
                torch_rng=torch.tensor([1, 2], dtype=torch.uint8), updates=epoch, best_score=float("inf"), history=[],
                replay_rows=[dict(file="shard.npz", split="train")])
            torch.save(payload, path)
            write(path.with_suffix(".COMMITTED.json"), dict(checkpoint=str(path.relative_to(folder)),
                sha256=repro.files.sha256(path), epoch=epoch, created_unix=time.time()))
        for name in repro._expected_stages(1):
            outcome = dict(loss=.25, physical_steps=3)
            if name.startswith("validate"):
                outcome["selection_score"] = [0.]
            write(folder/"stages"/(name+".json"), dict(stage=name, result=outcome,
                started_unix=time.time(), finished_unix=time.time()))
        frozen = read(folder/"checkpoints/initial.COMMITTED.json")
        frozen.update(schema="unified_frozen_v1", selected_epoch=0, candidate_stage="initial", selection_score=[0.],
            protocol_sha256=repro.files.sha256(folder/"protocol.json"))
        write(folder/"FROZEN.json", frozen)
        write(root/"GLOBAL_FROZEN.json", dict(schema="unified_frozen_group_v1",
            protocol_sha256=repro.files.sha256(root/"protocol.json"), jobs={"probe": repro.files.sha256(folder/"FROZEN.json")}))
        trace = folder/"trajectories/test/selected/case.npz"
        trace.parent.mkdir(parents=True)
        np.savez(trace, error=np.array([1., .5]), M=np.eye(2))
        test = dict(started_after_group_freeze=True, group_freeze_sha256=repro.files.sha256(root/"GLOBAL_FROZEN.json"),
            result=dict(successes=1, rows=[dict(trajectory=str(trace), trajectory_sha256=repro.files.sha256(trace),
                terminal_error=.5, policy_wall_s=time.perf_counter())], wall_s=time.perf_counter()))
        write(folder/"TEST_RESULTS.json", test)
        write(root/"RESULTS.json", dict(schema="unified_suite_results_v1", jobs={"probe": dict(backend="episode_ppo",
            frozen=frozen, test=test["result"], stages=[read(path) for path in sorted((folder/"stages").glob("*.json"))])}))
        write(root/"FINAL_STATE.json", dict(status="complete", jobs={"probe": {"status": "complete"}}))
    return tmp_path/"run_a", tmp_path/"run_b", tmp_path/"comparison.json"


def test_equivalent_runs_compare_real_tensor_and_archive_contents(pair, external):
    first, second, output = pair
    result = repro.compare(first, second, output)
    assert result["passed"] and result["checkpoints"] == 3
    assert len(external) == 2 and all(command[0] == "powershell" for command in external)
    assert result["process_snapshot_after_read"]["query_success"]


@pytest.mark.parametrize("field", ["actor", "optimizer", "torch_rng", "optimizer_sequence_type", "model_named_path"])
def test_any_model_optimizer_or_rng_difference_cannot_pass(pair, external, field):
    first, second, output = pair
    def change(payload):
        if field == "actor":
            payload[field]["weight"] += 1
        elif field == "optimizer":
            payload[field]["state"][0]["exp_avg"] += 1
        elif field == "torch_rng":
            payload[field][0] += 1
        elif field == "optimizer_sequence_type":
            payload["optimizer"]["param_groups"] = tuple(payload["optimizer"]["param_groups"])
        else:
            payload["actor"]["path"] = torch.tensor([1.])
    alter_payload(second, change)
    with pytest.raises(RuntimeError, match="mismatch"):
        repro.compare(first, second, output)
    assert not read(output)["passed"]
    expected_field = {"optimizer_sequence_type": "optimizer", "model_named_path": "actor"}.get(field, field)
    assert expected_field in next(x for x in read(output)["differences"] if x["component"] == "checkpoint")["differing_fields"]


@pytest.mark.parametrize("component", ["training", "validation", "selection", "evaluation", "replay", "trajectory"])
def test_any_scientific_stage_or_array_difference_cannot_pass(pair, external, component):
    first, second, output = pair
    folder = second/"jobs/probe"
    if component in {"training", "validation"}:
        name = "train_epoch_0001" if component == "training" else "validate_epoch_0001"
        path = folder/"stages"/(name+".json")
        data = read(path)
        data["result"]["loss"] += .1
        write(path, data)
    elif component == "selection":
        path = folder/"FROZEN.json"
        data = read(path)
        data["selection_score"] = [1.]
        write(path, data)
    elif component == "evaluation":
        path = folder/"TEST_RESULTS.json"
        data = read(path)
        data["result"]["successes"] = 0
        write(path, data)
    elif component == "replay":
        np.savez(folder/"replay/shard.npz", state=np.array([3., 4.]))
    else:
        path = folder/"trajectories/test/selected/case.npz"
        np.savez(path, error=np.array([1., .6]), M=np.eye(2))
        data = read(folder/"TEST_RESULTS.json")
        data["result"]["rows"][0]["trajectory_sha256"] = repro.files.sha256(path)
        write(folder/"TEST_RESULTS.json", data)
    refresh(second)
    with pytest.raises(RuntimeError, match="mismatch"):
        repro.compare(first, second, output)
    assert not read(output)["passed"]


@pytest.mark.parametrize("target", ["FINAL_STATE.json", "RESULTS.json", "jobs/probe/TEST_RESULTS.json",
    "jobs/probe/stages/train_epoch_0001.json", "jobs/probe/checkpoints/ready.pt",
    "jobs/probe/checkpoints/ready.COMMITTED.json"])
def test_missing_evidence_in_either_run_rejected_before_unpickling(pair, external, monkeypatch, target):
    first, second, output = pair
    (second/target).unlink()
    monkeypatch.setattr(repro.torch, "load", lambda *a, **k: pytest.fail("Incomplete run was unpickled"))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        repro.compare(first, second, output)
    assert not output.exists()


def test_writer_lock_rejected_before_any_run_evidence_read(pair, external, monkeypatch):
    first, second, output = pair
    (first/"WRITER.lock").write_text("active", encoding="utf-8")
    monkeypatch.setattr(repro.files, "read", lambda *a, **k: pytest.fail("Active run evidence was read"))
    with pytest.raises(RuntimeError, match="Writer lock"):
        repro.compare(first, second, output)


@pytest.mark.parametrize("kind", ["model_nan", "optimizer_inf", "best_score_nan"])
def test_equal_nonfinite_payloads_never_establish_reproducibility(pair, external, kind):
    first, second, output = pair
    def change(payload):
        if kind == "model_nan":
            payload["actor"]["weight"][0] = float("nan")
        elif kind == "optimizer_inf":
            payload["optimizer"]["state"][0]["exp_avg"][0] = float("inf")
        else:
            payload["best_score"] = float("nan")
    for root in (first, second):
        alter_payload(root, change)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        repro.compare(first, second, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["changed_source", "changed_declared_config", "actual_config_differs", "barrier_hash", "trajectory_hash"])
def test_invalid_provenance_or_gate_cannot_pass(pair, external, kind):
    first, second, output = pair
    if kind == "changed_source":
        (first.parent/"scientific_source.py").write_text("scientific_equation = 2\n", encoding="utf-8")
    elif kind == "changed_declared_config":
        declared = first.parent/"declared.json"
        value = read(declared); value["jobs"][0]["config"]["seed"] = 82
        write(declared, value)
    elif kind == "actual_config_differs":
        protocol = read(second/"protocol.json")
        protocol["config"]["jobs"][0]["config"]["seed"] = 82
        write(second/"protocol.json", protocol)
        group = read(second/"GLOBAL_FROZEN.json")
        group["protocol_sha256"] = repro.files.sha256(second/"protocol.json")
        write(second/"GLOBAL_FROZEN.json", group)
        job = read(second/"jobs/probe/protocol.json")
        job["suite_protocol_sha256"] = group["protocol_sha256"]
        write(second/"jobs/probe/protocol.json", job)
        frozen = read(second/"jobs/probe/FROZEN.json")
        frozen["protocol_sha256"] = repro.files.sha256(second/"jobs/probe/protocol.json")
        write(second/"jobs/probe/FROZEN.json", frozen)
        refresh(second)
    else:
        path = second/"jobs/probe/TEST_RESULTS.json"
        value = read(path)
        if kind == "barrier_hash":
            value["group_freeze_sha256"] = "0"*64
        else:
            value["result"]["rows"][0]["trajectory_sha256"] = "0"*64
        write(path, value)
    with pytest.raises(RuntimeError):
        repro.compare(first, second, output)
    assert not output.exists()


def test_mutation_during_comparison_is_detected_even_for_nonselected_checkpoint(pair, external, monkeypatch):
    first, second, output = pair
    original = repro.process_snapshot
    calls = 0
    def check(roots):
        nonlocal calls
        calls += 1
        if calls == 2:
            path = first/"jobs/probe/checkpoints/epoch_0001.pt"
            path.write_bytes(b"changed after comparison")
        return original(roots)
    monkeypatch.setattr(repro, "process_snapshot", check)
    with pytest.raises(RuntimeError, match="Evidence changed"):
        repro.compare(first, second, output)
    assert not output.exists()


@pytest.mark.parametrize("part", ["runtime", "sources", "config_sha256"])
def test_individually_valid_but_different_contracts_never_pass(pair, external, part):
    first, second, output = pair
    protocol = read(second/"protocol.json")
    if part == "runtime":
        protocol["runtime"]["runtime"] = "different CPU fixture"
    elif part == "sources":
        source = second.parent/"additional_source.py"
        source.write_text("additional_equation = 1", encoding="utf-8")
        protocol["sources"][str(source)] = repro.files.sha256(source)
    else:
        contract = protocol["config"]["reproducibility_contract"]
        declared = read(Path(contract["declared_config_path"]))
        declared["jobs"][0]["config"]["seed"] = 82
        path = second.parent/"second_declared.json"
        write(path, declared)
        protocol["sources"].pop(contract["declared_config_path"])
        contract.update(declared_config_path=str(path), declared_config_sha256=repro.files.sha256(path))
        protocol["sources"][str(path)] = contract["declared_config_sha256"]
        protocol["config"]["jobs"] = declared["jobs"]
    rebind_protocol(second, protocol)
    with pytest.raises(RuntimeError, match="mismatch"):
        repro.compare(first, second, output)
    assert any(row["component"] == part for row in read(output)["differences"])


@pytest.mark.parametrize("problem", ["query_failure", "stale", "empty_command", "missing_parent", "missing_self", "duplicate_pid", "target_writer"])
def test_external_snapshot_fails_closed(monkeypatch, tmp_path, problem):
    root = tmp_path/"run"
    value = snapshot()
    if problem == "query_failure":
        value["query_success"] = False
    elif problem == "stale":
        value["created_unix"] -= 61
    elif problem == "empty_command":
        value["processes"][0]["CommandLine"] = ""
    elif problem == "missing_parent":
        value["processes"][0].pop("ParentProcessId")
    elif problem == "missing_self":
        value["processes"] = []
    elif problem == "duplicate_pid":
        value["processes"].append(deepcopy(value["processes"][0]))
    else:
        value["processes"].append(dict(ProcessId=os.getpid()+100000, ParentProcessId=0, Name="python.exe",
            CommandLine=f'python scripts/run_unified_training_v4.py train --output "{root}"'))
    monkeypatch.setattr(repro.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0,
        stdout=json.dumps(value).encode(), stderr=b""))
    with pytest.raises(RuntimeError):
        repro.process_snapshot([root])


def test_injected_snapshot_is_not_a_public_bypass(pair):
    with pytest.raises(TypeError, match="snapshot"):
        repro.compare(*pair, snapshot=snapshot())


def test_relative_target_writer_is_rejected_even_as_python_ancestor(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    value = snapshot()
    ancestor = os.getpid()+100000
    value["processes"][0]["ParentProcessId"] = ancestor
    value["processes"].append(dict(ProcessId=ancestor, ParentProcessId=0, Name="python.exe",
        CommandLine="python scripts/run_unified_training_v4.py train --output run"))
    monkeypatch.setattr(repro.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0,
        stdout=json.dumps(value).encode(), stderr=b""))
    with pytest.raises(RuntimeError, match="still has a Python"):
        repro.process_snapshot([tmp_path/"run"])
