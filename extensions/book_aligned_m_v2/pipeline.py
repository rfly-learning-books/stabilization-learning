"""Single-supervisor, separate train/evaluate/audit subprocess night pipeline."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/"scripts",ROOT/"stabilization-gym",ROOT/"stabilization-learning-control"):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
import numpy as np
import torch
from torch import nn
from stabilization_gym import make
from stabilization_gym.envs.classic_control.sdc_metric import SDCMetricTask
from stabilization_learning.policies.state_metric_v2 import StateMetricPolicy
from stabilization_learning.algos.pytorch.state_metric_model.learner import ModelMetricConfig,ModelMetricLearner,rollout_batch
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash
from stabilization_learning.common.contracts_v2 import identity
from train_book_m_suite import WriterLock
from . import audit

TASKS=("coupled_linear","point_mass","pendulum","cartpole")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()


def emit(out,**value):
    value=dict(at=now(),**value)
    folder=Path(out)/"events";folder.mkdir(parents=True,exist_ok=True)
    with (folder/(str(time.time_ns())+"_"+value.get("stage","event")+".json")).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


def sources():
    files=[]
    for parent in ("stabilization-gym/stabilization_gym","stabilization-learning-control/stabilization_learning",
                   "extensions/book_aligned_m_v2","extensions/toolbox_p0_v1"):
        files.extend((ROOT/parent).rglob("*.py"))
    files.extend([ROOT/"scripts/train_book_m_suite.py",ROOT/"scripts/analysis/run_book_aligned_night_v2.py",
                  ROOT/"extensions/lqr_metric_direct_v1/audit.py"])
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}


def check_sources(protocol):
    changed=[p for p,h in protocol["sources"].items() if sha(ROOT/p)!=h]
    if changed:raise RuntimeError("Sources changed after protocol freeze: "+repr(changed))


def construct(name,protocol):
    torch.manual_seed(41)
    task=SDCMetricTask(make(name),name)
    policy=StateMetricPolicy(task.metric_spec,task.L0,goal=task.goal,
        context_scale=max(float(task.physical.initial_scale),.1),hidden=protocol["hidden"])
    cfg=ModelMetricConfig(**{**protocol["training"],"validation_seeds":tuple(protocol["training"]["validation_seeds"]),
        "test_seeds":tuple(protocol["training"]["test_seeds"])})
    return task,policy,cfg


class FixedPolicy(nn.Module):
    def __init__(self,M):super().__init__();self.register_buffer("M",torch.as_tensor(M,dtype=torch.float64))
    def forward(self,e):return self.M.expand(*e.shape[:-1],*self.M.shape)


class SDREPolicy(nn.Module):
    def __init__(self,task):super().__init__();self.task=task
    def forward(self,e):return self.task.classical_sdre_metric(e)


def stage_train(out,name,protocol):
    task,policy,cfg=construct(name,protocol)
    learner=ModelMetricLearner(task,policy,cfg,source_digest=identity(protocol["sources"]))
    def should_stop():
        if (out/"STOP").exists():return True
        if time.time()>protocol["deadline_unix"]:
            try:
                with (out/"STOP").open("x",encoding="utf-8") as stream:
                    stream.write("Declared night budget reached; no automatic resume.\n")
            except FileExistsError:pass
            return True
        return False
    learner.should_stop=should_stop
    destination=out/name
    complete=learner.train(destination,emit=lambda **v:emit(out,**v),resume=(destination/"last.pt").exists())
    if not complete:raise InterruptedError("Run STOP honored at whole-update boundary")
    emit(out,stage="task_training_complete",task=name,updates=learner.updates,
        best_update=learner.best_update,checkpoint_sha256=sha(destination/"last.pt"))


def stage_evaluate(out,name,protocol):
    task,policy,cfg=construct(name,protocol)
    payload=torch.load(out/name/"last.pt",map_location="cpu",weights_only=False)
    policy.load_state_dict(payload["modules"]["policy"])
    final=deepcopy(policy);policy.load_state_dict(payload["counters"]["best_actor"])
    models={"fixed_P0":FixedPolicy(task.P0),"learned":policy,"final":final}
    aliases={"initial":"fixed_P0"}
    if name in ("coupled_linear","point_mass"):
        aliases["classical_SDRE"]="fixed_P0"
    else:models["classical_SDRE"]=SDREPolicy(task)
    index=[];test_count=len(cfg.test_seeds)
    for scale in protocol["scales"]:
        starts=task.initial_states(cfg.test_seeds,scale=scale)
        for role,model_ in models.items():
            if (out/"STOP").exists():raise InterruptedError("STOP during evaluation")
            rows,arrays=rollout_batch(task,model_,starts,record=True,decay=cfg.decay,
                certificate_weight=cfg.certificate_weight,should_stop=lambda:(out/"STOP").exists())
            group=out/name/"evaluation"/("scale_"+str(scale)+"_"+role)
            group.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(group/"trajectories.npz",**arrays)
            write(group/"episodes.json",dict(task=name,role=role,scale=scale,seeds=list(cfg.test_seeds),rows=rows,
                weights={k:v.detach().cpu().numpy().tolist() for k,v in model_.state_dict().items()} if role in ("learned","final") else None))
            index.append(dict(group=str(group.relative_to(out)),episodes=test_count,trajectory_sha256=sha(group/"trajectories.npz"),
                rows_sha256=sha(group/"episodes.json"),role=role,scale=scale))
            emit(out,stage="evaluation_group_complete",task=name,role=role,scale=scale,episodes=test_count,
                successes=sum(r["success"] for r in rows),mean_cost=float(np.mean([r["fixed_horizon_cost"] for r in rows])))
    write(out/name/"EVALUATION_MANIFEST.json",dict(groups=index,aliases=aliases,test_count=test_count))


def stage_audit(out,name,protocol):
    manifest=read(out/name/"EVALUATION_MANIFEST.json");reports=[]
    for item in manifest["groups"]:
        group=out/item["group"]
        if sha(group/"trajectories.npz")!=item["trajectory_sha256"] or sha(group/"episodes.json")!=item["rows_sha256"]:
            raise RuntimeError("Evaluation evidence changed")
        data=read(group/"episodes.json")
        weights=None if data["weights"] is None else {k:np.asarray(v) for k,v in data["weights"].items()}
        with np.load(group/"trajectories.npz",allow_pickle=False) as z:
            report=audit.audit_group(name,dict(z),data["rows"],role=data["role"],weights=weights,
                decay=protocol["training"]["decay"],certificate_weight=protocol["training"]["certificate_weight"])
        reports.append(dict(group=item["group"],**report))
        if not report["passed"]:
            write(out/name/"AUDIT_FAILED.json",reports)
            raise AssertionError("Independent audit failed: "+item["group"]+" "+repr(report["checks"]))
    write(out/name/"AUDIT.json",dict(passed=all(r["passed"] for r in reports),groups=reports))
    emit(out,stage="task_audit_complete",task=name,groups=len(reports),episodes=sum(r["episodes"] for r in reports),passed=True)


def final_report(out,protocol):
    comparisons=[];audited_episodes=0
    lines=["# 书稿接口对齐：状态相关 M / SDC 夜训报告","",
        "学习控制器为 u=sat(u_ff−R⁻¹B(e)ᵀM(e,g)e)，M=L₀S(e,g)L₀ᵀ。",
        "经典在线SDRE单列为对照，学习M不冒称精确Riccati解。初始化/固定P₀只作为基线。",
        "采用完整后继M计算ΔV，真实误差只在分母做数值保护。","",
        "|任务|尺度|学习相对P₀成本|基线成功|学习成功|学习实际下降违例|","|---|---:|---:|---:|---:|---:|"]
    for name in TASKS:
        manifest=read(out/name/"EVALUATION_MANIFEST.json");a=read(out/name/"AUDIT.json")
        if not a["passed"]:raise RuntimeError("Cannot complete a failed audit")
        audited_episodes+=sum(r["episodes"] for r in a["groups"])
        by={}
        for item in manifest["groups"]:
            d=read(out/item["group"]/"episodes.json");by[(d["scale"],d["role"])]=d["rows"]
        for scale in protocol["scales"]:
            base=by[scale,"fixed_P0"];learned=by[scale,"learned"]
            b=np.asarray([x["fixed_horizon_cost"] for x in base]);l=np.asarray([x["fixed_horizon_cost"] for x in learned])
            delta=float((l.mean()/b.mean()-1)*100) if b.mean()!=0 else None
            row=dict(task=name,scale=scale,cost_percent=delta,baseline_success=sum(x["success"] for x in base),
                learned_success=sum(x["success"] for x in learned),episodes=len(base),
                paired_cost_delta=(l-b).tolist(),decay_violation=float(np.mean([x["decay_violation_fraction"] for x in learned])))
            comparisons.append(row)
            lines.append("|"+name+"|"+str(scale)+"|"+("n/a" if delta is None else f"{delta:+.3f}%")+"|"+
                str(row["baseline_success"])+"/"+str(row["episodes"])+"|"+str(row["learned_success"])+"/"+
                str(row["episodes"])+"|"+f'{row["decay_violation"]:.3%}'+"|")
    improved=sum(r["cost_percent"] is not None and r["cost_percent"]<0 for r in comparisons)
    lines+=["",f"成本降低 {improved}/{len(comparisons)} 组；独立复算 {audited_episodes} 条终验。",
        "", "## 对齐结论与限制","",
        "接口完成状态相关模型、学习M、真实控制、完整证书差分及独立物理评价的闭环。"+
        "实际收益以表内成功/保持/成本和全部不利结果为准，不能只凭M正定或训练损失下降认定镇定。",
        "新策略仅具有限仿真与数值证据；旧固定M线性证书不能直接移植为本状态相关策略的严格非线性域证明。",
        "Push、RGB、积分Disk及其他工程任务未被错误接入本后端；其原完成成果保留，后续后端适配仍单列。",
        "本轮新任务seed41，未启动seed43；测试从未用于选模或调参。",
        "", "## 后续训练决策","",
        "本批预算完成后停止。不会自动增加种子、扩大任务或续跑旧实验；seed43未因本批结果而获准。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write(out/"RESULTS.json",dict(comparisons=comparisons,improved_groups=improved,audited_episodes=audited_episodes,
        stable_claim="no new mathematical stability guarantee",seed43_started=False))
    write(out/"FINAL_STATE.json",dict(at=now(),status="completed_audited_stopped",seed43_started=False,
        audited_episodes=audited_episodes,report=str(out/"REPORT.md")))
    emit(out,stage="pipeline_complete",audited_episodes=audited_episodes,improved_groups=improved,total_groups=len(comparisons))


def build_protocol(smoke,p0):
    cfg=ModelMetricConfig()
    if smoke:
        cfg=ModelMetricConfig(updates=3,batch_states=8,model_horizon=3,validation_every=3,checkpoint_every=1,
            validation_seeds=(210001,210002),test_seeds=(310001,310002))
    return dict(schema="book_aligned_state_M_night_v2",created_at=now(),P0_report=str(p0.resolve()),
        engineering_smoke=smoke,tasks=list(TASKS),training=asdict(cfg),hidden=32,scales=[1.] if smoke else [1.,1.5,2.],
        controller="learned state M + exact declared SDC A(e),B(e); u=sat(u_ff-R^-1 B(e)^T M(e,g)e)",
        certificate="V(e)=e^T M(e,g)e; full successor M; trace constraints on S only",
        baselines=["fixed_P0_with_B_e","classical_online_SDRE","validation_best_learned","final"],
        aliases="initial==fixed_P0; classical_SDRE==fixed_P0 for the two linear plants",
        selection="validation non-success count first, then fixed-horizon physical cost + .1 corrected full-M descent penalty",
        exclusions=["seed42/43 in this new scope","restart of original Push/RGB/base/direct-M runs",
                    "unqualified state-M Disk/water/flexible/fixed-wing/swarm training","book manuscript edits"],
        max_wall_seconds=8*3600,deadline_unix=time.time()+8*3600,
        sources=sources(),original_STOP_sha256=sha(ROOT/"runs/book_m_night_suite_20261023_v1/STOP"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--p0",type=Path)
    parser.add_argument("--engineering-smoke",action="store_true")
    parser.add_argument("--stage",choices=["train","evaluate","audit"])
    parser.add_argument("--task",choices=TASKS)
    args=parser.parse_args();out=args.output.resolve();torch.set_num_threads(1)
    if args.stage:
        protocol=read(out/"protocol.json");check_sources(protocol)
        {"train":stage_train,"evaluate":stage_evaluate,"audit":stage_audit}[args.stage](out,args.task,protocol)
        return
    if out.exists():raise FileExistsError("New pipeline requires a new directory")
    if args.p0 is None:parser.error("--p0 is required")
    if not args.engineering_smoke and not read(args.p0/"VALIDATION.json").get("admitted_for_state_M_night"):
        raise RuntimeError("P0 and end-to-end engineering acceptance must pass before formal training")
    out.mkdir(parents=True);protocol=build_protocol(args.engineering_smoke,args.p0)
    write(out/"protocol.json",protocol)
    write(out/"USER_AUTHORIZATION.json",dict(at=now(),instruction="完成P0验收后启动正式训练；对齐书稿接口，在当前空间进行夜间长训；自主决策，不再提问。",old_STOP_preserved=True,seed43=False))
    emit(out,stage="pipeline_started",engineering_smoke=args.engineering_smoke,tasks=list(TASKS),
        updates_per_task=protocol["training"]["updates"],model_queries_budget=4*protocol["training"]["updates"]*
        protocol["training"]["batch_states"]*protocol["training"]["model_horizon"])
    started=time.monotonic()
    try:
        with WriterLock(ROOT/"runs/book_m_suite_slot.lock"),WriterLock(out/"writer.lock"):
            if os.name=="nt":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            for name in TASKS:
                for stage in ("train","evaluate","audit"):
                    if (out/"STOP").exists():raise InterruptedError("STOP preserved; no automatic resume")
                    if time.monotonic()-started>protocol["max_wall_seconds"]:
                        with (out/"STOP").open("x",encoding="utf-8") as stream:stream.write("Versioned eight-hour budget reached; explicit authorization required to resume.\n")
                        raise InterruptedError("Night budget reached")
                    check_sources(protocol)
                    emit(out,stage="stage_started",task=name,substage=stage)
                    subprocess.run([sys.executable,str(ROOT/"scripts/analysis/run_book_aligned_night_v2.py"),
                        "--output",str(out),"--stage",stage,"--task",name],check=True,cwd=ROOT)
            check_sources(protocol)
            if sha(ROOT/"runs/book_m_night_suite_20261023_v1/STOP")!=protocol["original_STOP_sha256"]:
                raise RuntimeError("Original STOP changed")
            final_report(out,protocol)
    except Exception as exc:
        status="stopped" if isinstance(exc,InterruptedError) or (out/"STOP").exists() else "failed"
        write(out/"FINAL_STATE.json",dict(at=now(),status=status,error=repr(exc),traceback=traceback.format_exc(),seed43_started=False))
        emit(out,stage="pipeline_"+status,error=repr(exc))
        raise
    finally:
        if os.name=="nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=="__main__":main()
