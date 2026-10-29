"""Post-analysis must preserve raw updates and fail closed around active writers."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import os
import subprocess
import time
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from extensions.joint_certificate_v2.analysis import analyze,assert_run_quiescent,matrix_stats,_snapshot_witness


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding="utf-8")


def test_process_identity_fail_closed(tmp_path):
    root=tmp_path/"night_unique_run"
    base=dict(Name="python.exe",ProcessId=2,ParentProcessId=1,CommandLine='python train.py --output "'+str(root)+'"')
    with pytest.raises(RuntimeError):assert_run_quiescent(root,processes=[base],current_pid=99)
    with pytest.raises(RuntimeError):assert_run_quiescent(root,processes=[dict(base,CommandLine=None)],current_pid=99)
    own=dict(base,ProcessId=99,ParentProcessId=98,CommandLine='python analysis.py --root "'+str(root)+'"')
    launcher=dict(own,ProcessId=98,ParentProcessId=1)
    witness=assert_run_quiescent(root,processes=[own,launcher],current_pid=99)
    assert witness["excluded_analysis_pids"]==[98,99]


def test_matrix_statistics_are_actual_frobenius_changes():
    initial=np.eye(3)[None];current=initial*2
    stats=matrix_stats(current,initial,initial)
    assert stats["relative_to_initial"]==pytest.approx(1.)
    assert stats["relative_step_change"]==pytest.approx(1.)
    assert stats["eigen_min"]==stats["eigen_max"]==2.
    with pytest.raises(ValueError):matrix_stats(np.eye(4),initial,initial)


@pytest.mark.skipif(os.name!="nt",reason="Windows process-query integration check")
def test_real_cim_blocks_another_run_process(tmp_path):
    root=tmp_path/"joint_v2_guard_active_writer"
    process=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)",str(root)],
                             creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        with pytest.raises(RuntimeError,match="writers still active"):
            assert_run_quiescent(root)
    finally:
        process.terminate();process.wait(timeout=10)


def test_push_and_rgb_reports_keep_steps_calibration_and_inputs(tmp_path):
    root=tmp_path/"suite";push=root/"push_seed81";rgb=root/"rgb_seed81"
    write(push/"protocol.json",dict(config=dict(seed=81)))
    write(push/"FINAL_STATE.json",dict(status="complete"));write(push/"FROZEN.json",dict(selected_update=2))
    write(push/"TRAINING.json",[dict(update=1,loss=.3,episodes=[]),
        dict(update=2,loss=.2,episodes=[dict(reward=-3.,success=False)])])
    write(push/"M_PROBES.json",[dict(update=i,M=(np.eye(3)[None]*(1+i*.1)).tolist()) for i in range(3)])
    write(push/"events/001_validation.json",dict(kind="validation_complete",data=dict(update=2,success_rate=0.,mean_combined_cost=3.,mean_seconds=2.)))
    write(rgb/"protocol.json",dict(config=dict(seed=81)))
    write(rgb/"FINAL_STATE.json",dict(status="stopped"))
    for seq,step,phase,scale in ((1,0,"random_initial",1.),(3,1,"warmup",1.),(4,1,"post_warmup_calibration",.1)):
        write(rgb/f"events/{seq:03d}_probe.json",dict(sequence=seq,kind="fixed_training_probe",data=dict(
            step=step,phase=phase,M=np.eye(4)[None].tolist(),normalization_scale=scale,calibrated=scale<1,
            version=seq,V=[1.],E=[2.],feature_error_norm=[.2])))
    write(rgb/"events/002_update.json",dict(sequence=2,kind="update",data=dict(step=1,block="warmup",metrics=dict(total=.2,gradient_norm=.1))))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json")}
    out=tmp_path/"analysis";summary=analyze(root,out,verify_writers=False)
    assert len(summary["branches"])==2 and summary["training_modified"] is False
    assert any("stopped" in s for s in summary["warnings"])
    assert len(list(out.glob("*.png")))==4
    rgb_branch=next(b for b in summary["branches"] if b["kind"]=="rgb")
    with (out/(rgb_branch["prefix"]+"_metrics.csv")).open(encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
    assert [int(row["step"]) for row in rows]==[0,1,1]
    assert rows[-1]["phase"]=="post_warmup_calibration"
    push_branch=next(b for b in summary["branches"] if b["kind"]=="push")
    with (out/(push_branch["prefix"]+"_updates.csv")).open(encoding="utf-8-sig") as f:updates=list(csv.DictReader(f))
    assert updates[0]["completed_episode_reward_mean"]=="" and updates[1]["completed_episode_reward_mean"]=="-3.0"
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json")}
    assert (out/"PROVENANCE.json").exists()
    assert "SD 不是置信区间" in (out/"ANALYSIS_ZH.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):analyze(root,out,verify_writers=False)


def test_missing_failed_branch_does_not_invent_results(tmp_path):
    root=tmp_path/"failed";write(root/"protocol.json",dict(config={}))
    write(root/"FINAL_STATE.json",dict(status="failed"))
    report=analyze(root,tmp_path/"report",verify_writers=False)
    assert report["overall_status"]=="failed" and report["branches"]==[]
    assert report["warnings"] and report["performance_inferred"] is False


def test_formal_analysis_requires_fresh_external_inventory(tmp_path,monkeypatch):
    root=tmp_path/"formal_night";write(root/"NIGHT_PROTOCOL.json",{})
    with pytest.raises(RuntimeError,match="external complete process snapshot"):
        analyze(root,tmp_path/"forbidden",verify_writers=False)
    path=tmp_path/"processes.json"
    with pytest.raises(FileNotFoundError):_snapshot_witness(root,path)
    write(path,dict(schema="joint_v2_process_snapshot",query_success=True,created_unix=time.time(),processes=[]))
    witness=_snapshot_witness(root,path)
    assert witness["snapshot_path"]==str(path.resolve())
    assert witness["snapshot_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="must be verified"):
        analyze(root,tmp_path/"bypass",verify_writers=False,process_snapshot=path)
    monkeypatch.setattr("extensions.joint_certificate_v2.analysis.assert_run_quiescent",lambda *a,**k:dict(method="test supplemental query"))
    result=analyze(root,tmp_path/"valid",process_snapshot=path)
    assert result["writer_exit"]["snapshot_sha256"]==witness["snapshot_sha256"]
    assert "supplemental" in result["writer_exit"]["final_check"]["scope"]


@pytest.mark.parametrize("problem",["schema","failed_query","missing_time","stale","future","unknown_identity","unknown_name","writer"])
def test_external_snapshot_rejects_incomplete_stale_or_active(tmp_path,problem):
    root=tmp_path/"formal_night";path=tmp_path/"processes.json"
    data=dict(schema="joint_v2_process_snapshot",query_success=True,created_unix=time.time(),processes=[])
    if problem=="schema":data["schema"]="unrecognized"
    elif problem=="failed_query":data["query_success"]=False
    elif problem=="missing_time":data.pop("created_unix")
    elif problem=="stale":data["created_unix"]-=61
    elif problem=="future":data["created_unix"]+=61
    else:
        row=dict(ProcessId=123,ParentProcessId=1,Name="python.exe",CommandLine="python unrelated.py")
        if problem=="unknown_identity":row["CommandLine"]=None
        elif problem=="unknown_name":row["Name"]="unknown.exe"
        else:row["CommandLine"]="python train.py --run "+str(root)
        data["processes"]=[row]
    write(path,data)
    with pytest.raises(RuntimeError):_snapshot_witness(root,path)
