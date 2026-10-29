"""Evidence gate and scientific-report regression checks."""
import json
import time

import numpy as np
import pytest

from stabilization_learning.unified_v1.reporting import (
    check_process_snapshot, _evaluate, _gpu, report_suite)
from stabilization_learning.unified_v1.runner import write_new


def snapshot(path, processes=(), age=0):
    path.write_text(json.dumps(dict(schema="unified_training_process_snapshot",query_success=True,
        created_unix=time.time()-age,processes=list(processes))),encoding="utf-8")
    return path


def test_external_gate_rejects_unknown_stale_and_incomplete(tmp_path):
    path=tmp_path/"process.json"
    assert check_process_snapshot(snapshot(path))["passed"]
    process=dict(ProcessId=1,ParentProcessId=2,Name="python.exe",CommandLine="python scripts/train.py")
    with pytest.raises(RuntimeError,match="unknown Python"):
        check_process_snapshot(snapshot(path,[process]))
    process["CommandLine"]=None
    with pytest.raises(RuntimeError,match="Incomplete"):
        check_process_snapshot(snapshot(path,[process]))
    with pytest.raises(RuntimeError,match="60 seconds"):
        check_process_snapshot(snapshot(path,age=61))


def test_gpu_uses_full_time_and_preserves_missing_samples(tmp_path):
    rows=[dict(unix=0,gpus=[dict(utilization_percent=20)]),
          dict(unix=5,error="query failed"),dict(unix=10,gpus=[dict(utilization_percent=80)])]
    (tmp_path/"GPU_SAMPLES.jsonl").write_text("\n".join(map(json.dumps,rows)),encoding="utf-8")
    result=_gpu(tmp_path,0,20)
    assert result["error_samples"]==1
    assert result["devices"][0]["time_weighted_percent"]==60
    assert result["devices"][0]["coverage_fraction"]==.75


def test_push_recomputes_both_successor_metrics(tmp_path):
    transition=dict(error=[1.,2.],error_next=[.5,1.],M=[[1.,0.],[0.,2.]],
                    M_next=[[2.,0.],[0.,3.]],V=9.,V_next=3.5)
    test=dict(variants=dict(selected=dict(num_episodes=1,episodes=[dict(trace=[transition])])) )
    result=_evaluate(tmp_path,dict(backend="push_discrete_sac"),test)
    assert result["postprocess_numeric_audit"]["passed"]
    transition["V_next"]=2.25  # Incorrect use of current M at next state.
    assert not _evaluate(tmp_path,dict(backend="push_discrete_sac"),test)["postprocess_numeric_audit"]["passed"]


def test_failed_partial_run_is_reported_without_fabricating_tests(tmp_path):
    root=tmp_path/"run";root.mkdir()
    start=time.time()-2
    job=dict(id="state",backend="state_model_gradient",schedule=dict(epochs=1,certificate_updates=1))
    write_new(root/"protocol.json",dict(started_unix=start,config=dict(jobs=[job]),sources={}))
    write_new(root/"FINAL_STATE.json",dict(status="failed",finished_unix=start+1,error="intentional"))
    output=tmp_path/"report"
    result=report_suite(root,output,snapshot(tmp_path/"process.json"))
    assert result["status"]=="failed"
    assert not result["numerical_audit_passed"]
    saved=json.loads((output/"REPORT.json").read_text(encoding="utf-8"))
    assert saved["jobs"]["state"]["evaluation"]["status"].startswith("not completed")
    assert (output/"REPORT.html").is_file()
    with pytest.raises(FileExistsError):
        report_suite(root,output,snapshot(tmp_path/"process.json"))


def test_writer_lock_rejects_before_opening_any_run_content(tmp_path):
    root=tmp_path/"run";root.mkdir()
    (root/"WRITER.lock").write_text("a possibly active owner",encoding="utf-8")
    # Deliberately omit protocol and final state. Lock rejection must precede
    # any attempts to read run content, including seemingly final artifacts.
    with pytest.raises(RuntimeError,match="Writer lock"):
        report_suite(root,tmp_path/"report",snapshot(tmp_path/"process.json"))
