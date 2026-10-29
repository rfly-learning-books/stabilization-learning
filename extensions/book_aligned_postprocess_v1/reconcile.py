"""Re-audit saved evidence, preserving the original producer's failed status."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/"stabilization-gym",ROOT/"stabilization-learning-control"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import numpy as np
from extensions.book_aligned_m_v2 import audit as original_audit
from extensions.lqr_metric_direct_v1.audit import errors,model,failure_flags


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def require_no_writer(run):
    cmd="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
    process=subprocess.run(["powershell","-NoProfile","-Command",cmd],capture_output=True,text=True,encoding="utf-8",errors="replace",check=True)
    entries=json.loads(process.stdout) if process.stdout.strip() else []
    if isinstance(entries,dict):entries=[entries]
    if any("run_book_aligned_night_v2.py" in (p.get("CommandLine") or "") and str(run).lower() in (p.get("CommandLine") or "").lower() for p in entries):
        raise RuntimeError("Original training/evaluation/audit writer still active")


def main(run,destination):
    run=Path(run).resolve();destination=Path(destination).resolve()
    require_no_writer(run)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    protocol=read(run/"protocol.json")
    for name,digest in protocol["sources"].items():
        if sha(ROOT/name)!=digest:raise RuntimeError("Frozen source changed: "+name)
    if read(run/"FINAL_STATE.json")["status"]!="failed":
        raise RuntimeError("This reconciliation only handles the preserved audit-failed run")
    original_identity=dict(final_state=sha(run/"FINAL_STATE.json"),protocol=sha(run/"protocol.json"),
        original_audit_source=sha(ROOT/"extensions/book_aligned_m_v2/audit.py"))
    reports=[];differences=[];groups_processed=[]
    for task in protocol["tasks"]:
        manifest=read(run/task/"EVALUATION_MANIFEST.json")
        old_audits=read(run/task/"AUDIT.json")["groups"] if (run/task/"AUDIT.json").exists() else read(run/task/"AUDIT_FAILED.json")
        by={item["group"]:item for item in old_audits}
        task_reports=[]
        for entry in manifest["groups"]:
            group=run/entry["group"]
            if sha(group/"trajectories.npz")!=entry["trajectory_sha256"] or sha(group/"episodes.json")!=entry["rows_sha256"]:
                raise RuntimeError("Sealed evaluation evidence changed")
            data=read(group/"episodes.json")
            old=by.get(entry["group"])
            if old is not None and old["passed"]:
                report=dict(old,reused_original_pass=True)
                task_reports.append(report);reports.append(report);groups_processed.append(entry["group"])
                continue
            with np.load(group/"trajectories.npz",allow_pickle=False) as trace:
                arrays=dict(trace)
            corrected=deepcopy(data["rows"])
            key=original_audit.ALIASES.get(task,task);cfg=model(key)
            for lane,row in enumerate(data["rows"]):
                valid=arrays["active"][:,lane].astype(bool)
                n=int(valid.sum())
                following=arrays["following"][valid,lane]
                e=errors(key,following[-1])
                terminal=float(e@e)
                failure=bool(failure_flags(key,following[-1:])[0])
                cost=float(arrays["costs"][valid,lane].sum())
                tail=max(terminal,25.) if failure else terminal
                canonical=cost+(cfg["horizon"]-n)*cfg["dt"]*tail
                # Reproduce the exact producer defect independently, rather than
                # accepting an arbitrary difference or relaxing an audit tolerance.
                legacy_seconds=float(np.float32(cfg["horizon"]-n)*np.float32(cfg["dt"]))
                legacy=cost+legacy_seconds*tail
                if not np.isclose(row["fixed_horizon_cost"],legacy,atol=1e-8,rtol=1e-8):
                    raise RuntimeError("Mismatch is not explained by the declared v2 float32 tail clock")
                if not np.isclose(row["objective"],legacy+protocol["training"]["certificate_weight"]*row["certificate_penalty"],atol=1e-8,rtol=1e-8):
                    raise RuntimeError("Original objective contains another unexplained discrepancy")
                corrected[lane]["fixed_horizon_cost"]=canonical
                corrected[lane]["objective"]=canonical+protocol["training"]["certificate_weight"]*row["certificate_penalty"]
                if n<cfg["horizon"]:
                    differences.append(dict(group=entry["group"],lane=lane,steps=n,
                        original_fixed_horizon_cost=row["fixed_horizon_cost"],float64_recomputed_cost=canonical,
                        difference=row["fixed_horizon_cost"]-canonical,
                        producer_float32_clock_reproduced=legacy,
                        original_within_declared_audit_tolerance=bool(np.isclose(row["fixed_horizon_cost"],canonical,atol=1e-8,rtol=1e-8))))
            weights=None if data["weights"] is None else {k:np.asarray(v) for k,v in data["weights"].items()}
            audit=original_audit.audit_group(task,arrays,corrected,role=data["role"],weights=weights,
                decay=protocol["training"]["decay"],certificate_weight=protocol["training"]["certificate_weight"])
            report=dict(group=entry["group"],**audit,reused_original_pass=False,
                original_producer_summary_replaced=False,supplemental_float64_summary=True)
            task_reports.append(report);reports.append(report);groups_processed.append(entry["group"])
            folder=destination/task/group.name
            write(folder/"EPISODES_FLOAT64_RECOMPUTATION.json",dict(original_group=entry["group"],rows=corrected,
                note="Supplement only; original episodes, objectives, weights and model selection are unchanged"))
            write(folder/"AUDIT.json",report)
            if not audit["passed"]:raise AssertionError("Unresolved physical/certificate/summary issue: "+entry["group"]+" "+repr(audit["checks"]))
            print(json.dumps(dict(task=task,group=group.name,passed=True),ensure_ascii=False),flush=True)
        write(destination/task/"AUDIT_RECONCILED.json",dict(passed=all(x["passed"] for x in task_reports),groups=task_reports,
            original_pipeline_status="failed",original_failure_preserved=True))
    if sha(run/"FINAL_STATE.json")!=original_identity["final_state"]:
        raise RuntimeError("Original final status was modified")
    result=dict(schema="postprocess_float64_tail_reconciliation_v1",at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        passed=all(r["passed"] for r in reports),original_pipeline_status="failed",
        original_failure_preserved=True,original_summary_audit_passed=False,
        source_and_weight_changes=False,training_or_evaluation_rerun=False,selected_models_changed=False,
        groups=len(reports),episodes=sum(r["episodes"] for r in reports),original_identity=original_identity,
        correction="Future aggregation helper casts remaining integer steps to float64 before multiplying dt. Original producer remains unchanged.",
        failing_tail_rows=len(differences),maximum_abs_cost_difference=max([abs(r["difference"]) for r in differences] or [0]),
        rows=differences,postprocess_sources={str(Path(__file__).resolve()):sha(Path(__file__)),
            str(Path(__file__).with_name("costs.py").resolve()):sha(Path(__file__).with_name("costs.py"))})
    write(destination/"RECONCILIATION.json",result)
    (destination/"RECONCILIATION.md").write_text("# 审计失败后的独立补充复核\n\n"
        "原训练与终验均已完成；原流水线在小车倒立摆最终策略的失败尾项汇总审计处停止，原FINAL_STATE仍保留failed。\n\n"
        "原因是剩余步数×dt先以float32相乘，再进入float64成本。审计正确地捕获了真实聚合误差；未放宽任何容差。\n\n"
        f"共{len(reports)}组、{result['episodes']}回合的控制/物理/证书及独立float64成本复核完成；"
        f"尾项行{len(differences)}条，最大成本差{result['maximum_abs_cost_difference']:.12g}。\n\n"
        "本目录只增加独立重算附录。原始成绩、权重、轨迹、协议、失败记录和选模不变；没有重新训练或重跑终验。"
        "原始汇总精度检查并未被改写为通过，独立重算通过与原生产者缺陷分别登记。\n",encoding="utf-8")
    print(json.dumps(dict(destination=str(destination),groups=result["groups"],episodes=result["episodes"],
        max_cost_difference=result["maximum_abs_cost_difference"],passed=result["passed"]),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main(sys.argv[1],sys.argv[2])
