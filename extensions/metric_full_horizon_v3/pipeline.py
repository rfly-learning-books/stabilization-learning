"""Bounded train/commit/test/audit workflow for the corrected mechanism."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from stabilization_gym import make
from stabilization_gym.envs.classic_control.sdc_full_horizon import FullHorizonSDCTask
from stabilization_learning.policies.state_metric_residual_v3 import AnchoredStateMetricPolicy
from stabilization_learning.algos.pytorch.state_metric_full_horizon import FullHorizonConfig, FullHorizonMetricLearner
from stabilization_learning.algos.pytorch.state_metric_full_horizon.evaluation import evaluate, local_check, selection_gate
from stabilization_learning.algos.pytorch.state_metric_full_horizon.objectives import falsify
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash
from stabilization_learning.common.contracts_v2 import identity
from .audit import audit_group

ROOT=Path(__file__).resolve().parents[2]
TASKS=("coupled_linear","point_mass","pendulum","cartpole")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def emit(out,stage,**value):
    value={"at":now(),"stage":stage,**value}
    folder=out/"events";folder.mkdir(exist_ok=True)
    name=str(time.time_ns())+"_"+stage
    tmp=folder/(name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    os.rename(tmp,folder/(name+".json"))
    print(json.dumps(value,ensure_ascii=False),flush=True)


def manifest():
    files=[]
    for base in ("stabilization-gym/stabilization_gym","stabilization-learning-control/stabilization_learning",
                 "extensions/metric_full_horizon_v3"):
        files.extend((ROOT/base).rglob("*.py"))
    files += [ROOT/"extensions/book_aligned_m_v2/audit.py",ROOT/"extensions/lqr_metric_direct_v1/audit.py",
              ROOT/"scripts/analysis/run_metric_full_horizon_v3.py"]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}


def check_sources(protocol):
    changed=[p for p,h in protocol["sources"].items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    if changed:raise RuntimeError("Frozen source changed: "+repr(changed))


@contextmanager
def writer_lock():
    import msvcrt
    import psutil
    blockers=[]
    for proc in psutil.process_iter(["pid","name","cmdline"]):
        if proc.info["pid"]==os.getpid() or "python" not in (proc.info["name"] or "").lower():continue
        command=" ".join(proc.info["cmdline"] or [])
        if not command:raise RuntimeError("Python process identity unknown; writer status cannot be established")
        if any(s in command for s in ("run_metric_full_horizon_v3.py","run_book_aligned_night_v2.py","train_book_m_suite.py")):
            blockers.append(dict(pid=proc.pid,command=command))
    if blockers:raise RuntimeError("Another training/evaluation writer is active: "+repr(blockers))
    path=ROOT/"runs/FULL_HORIZON_WRITER.lock"
    with path.open("a+b") as stream:
        if stream.tell()==0:stream.write(b"0");stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def config_from(value):
    value=dict(value)
    for key in ("scales","validation_seeds","test_seeds"):
        if key in value:value[key]=tuple(value[key])
    return FullHorizonConfig(**value)


def construct(name,protocol):
    torch.manual_seed(protocol["training"]["seed"])
    task=FullHorizonSDCTask(make(name),name)
    policy=AnchoredStateMetricPolicy(task.metric_spec,task.L0,goal=task.goal,
        context_scale=max(float(task.physical.initial_scale),.1),hidden=protocol["hidden"],
        residual_limit=protocol["residual_limit"])
    learner=FullHorizonMetricLearner(task,policy,config_from(protocol["training"]),
        source_digest=identity(protocol["sources"]))
    return task,policy,learner


class FixedPolicy(nn.Module):
    def __init__(self,M):
        super().__init__();self.register_buffer("M",torch.tensor(M,dtype=torch.float64))
    def forward(self,e):return self.M.expand(*e.shape[:-1],*self.M.shape)


class ClassicalSDRE(nn.Module):
    def __init__(self,task):super().__init__();self.task=task
    def forward(self,e):return self.task.classical_sdre_metric(e)


def train_all(out,protocol,stop):
    selections={}
    for name in protocol["tasks"]:
        check_sources(protocol)
        folder=out/name;folder.mkdir(exist_ok=True)
        if (folder/"TRAINING_COMPLETE.json").exists():
            selections[name]=read(folder/"TRAINING_COMPLETE.json")
            if sha(folder/"last.pt")!=selections[name]["checkpoint_sha256"]:
                raise ValueError("Completed training checkpoint changed")
            continue
        task,policy,learner=construct(name,protocol)
        learner.should_stop=stop
        if (folder/"last.pt").exists():learner.restore(folder/"last.pt")
        else:
            learner.initialize();learner.save(folder/"last.pt")
            emit(out,"baseline_qualified",task=name,cost_scale=learner.cost_scale)
        while learner.updates<learner.config.updates and learner.stale<learner.config.patience:
            if stop():raise InterruptedError("STOP at whole-update boundary")
            started=time.perf_counter();row=learner.update()
            validation=None
            if learner.updates%learner.config.validation_every==0 or learner.updates==learner.config.updates:
                validation=learner.validate()
            learner.save(folder/"last.pt")
            emit(out,"update_committed",task=name,**row,wall_seconds=time.perf_counter()-started,
                best_update=learner.best_update,validation=validation)
        policy.load_state_dict(learner.best_actor)
        done=dict(task=name,updates=learner.updates,best_update=learner.best_update,
            best_validation_ratio=learner.best_score,best_actor_sha256=module_hash(policy),
            checkpoint_sha256=sha(folder/"last.pt"),model_transitions=learner.model_transitions,
            stopped_by="validation_patience" if learner.stale>=learner.config.patience else "update_budget")
        write(folder/"TRAINING_COMPLETE.json",done)
        write(folder/"TRAINING_HISTORY.json",learner.history)
        selections[name]=done
        emit(out,"task_training_complete",**done)
    check_sources(protocol)
    committed=out/"SELECTION_COMMITTED.json"
    value=dict(tasks=selections,protocol_sha256=sha(out/"protocol.json"),test_access_at_selection=False)
    if committed.exists():
        if read(committed)!=value:raise ValueError("Model selection commitment changed")
    else:write(committed,value)
    return selections


def evaluate_all(out,protocol,stop):
    cfg=config_from(protocol["training"])
    selected=read(out/"SELECTION_COMMITTED.json")
    reports={};total=0
    for name in protocol["tasks"]:
        check_sources(protocol)
        folder=out/name
        if sha(folder/"last.pt")!=selected["tasks"][name]["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after test selection commitment")
        task,policy,learner=construct(name,protocol);learner.restore(folder/"last.pt")
        policy.load_state_dict(learner.best_actor)
        if module_hash(policy)!=selected["tasks"][name]["best_actor_sha256"]:
            raise ValueError("Test weights differ from committed validation selection")
        models={"fixed_P0":FixedPolicy(task.P0),"learned":policy}
        candidate=deepcopy(policy);candidate.load_state_dict(learner.last_candidate)
        aliases={"initial":"fixed_P0"}
        if learner.best_update==0:
            del models["learned"]
            aliases["learned"]="fixed_P0"
        if module_hash(candidate)!=module_hash(policy):models["last_candidate"]=candidate
        else:aliases["last_candidate"]="learned"
        if name in ("coupled_linear","point_mass"):aliases["classical_SDRE"]="fixed_P0"
        else:models["classical_SDRE"]=ClassicalSDRE(task)
        data={};audits=[]
        for scale in cfg.scales:
            for role,model_ in models.items():
                if stop():raise InterruptedError("STOP before test group")
                dest=folder/"test"/(str(scale)+"_"+role)
                if (dest/"AUDIT.json").exists():
                    meta=read(dest/"GROUP.json");audit=read(dest/"AUDIT.json")
                    if not audit["passed"] or sha(dest/"trace.npz")!=meta["trace_sha256"]:
                        raise ValueError("Existing test group is failed or changed")
                    rows=meta["rows"]
                else:
                    if dest.exists():raise ValueError("Incomplete test group retained; explicit diagnosis required: "+str(dest))
                    dest.mkdir(parents=True)
                    rows,trace=evaluate(task,model_,task.initial_states(cfg.test_seeds,scale=scale),
                        decay=cfg.decay,record=True,should_stop=stop)
                    np.savez_compressed(dest/"trace.npz",**trace)
                    weights={k:v.detach().numpy() for k,v in model_.state_dict().items()} if role in ("learned","last_candidate") else None
                    meta=dict(role=role,task=name,scale=scale,seeds=list(cfg.test_seeds),rows=rows,
                        trace_sha256=sha(dest/"trace.npz"),weights=None if weights is None else {k:v.tolist() for k,v in weights.items()})
                    write(dest/"GROUP.json",meta)
                    audit=audit_group(name,trace,rows,role=role,weights=weights,seeds=cfg.test_seeds,scale=scale,decay=cfg.decay)
                    write(dest/"AUDIT.json",audit)
                    if not audit["passed"]:raise AssertionError("Independent audit failed: "+str(dest))
                data[scale,role]=rows;audits.append(audit);total+=len(rows)
                emit(out,"test_audited",task=name,role=role,scale=scale,episodes=len(rows),
                    successes=sum(r["success"] for r in rows),mean_cost=float(np.mean([r["fixed_horizon_cost"] for r in rows])))
        for scale in cfg.scales:
            for alias,target in aliases.items():
                data[scale,alias]=data[scale,target]
        generator=torch.Generator().manual_seed(protocol["test_probe_seed"])
        probes=learner.probes(512,generator)
        _,rate=falsify(task,policy,probes,decay=cfg.decay,steps=cfg.attack_steps,
            lower=learner.lower,upper=learner.upper)
        gate=selection_gate({s:data[s,"learned"] for s in cfg.scales},
            {s:data[s,"fixed_P0"] for s in cfg.scales},local=local_check(task,policy,decay=cfg.decay),
            probe_rate=float(rate.max()),rate_tolerance=cfg.rate_tolerance,
            max_arrival_delay=cfg.max_arrival_delay,max_energy_ratio=cfg.max_energy_ratio)
        comparisons=[]
        for scale in cfg.scales:
            b=data[scale,"fixed_P0"];l=data[scale,"learned"]
            bc=np.array([r["fixed_horizon_cost"] for r in b]);lc=np.array([r["fixed_horizon_cost"] for r in l])
            rng=np.random.default_rng(7300001)
            indices=rng.integers(0,len(b),(2000,len(b)))
            bootstrap=(lc[indices].mean(1)/bc[indices].mean(1)-1)*100
            classical=data.get((scale,"classical_SDRE"),b)
            comparisons.append(dict(scale=scale,cost_percent=float((lc.mean()/bc.mean()-1)*100),
                paired_bootstrap_95_percent=np.quantile(bootstrap,[.025,.975]).tolist(),
                baseline_success=sum(r["success"] for r in b),learned_success=sum(r["success"] for r in l),
                episodes=len(b),arrival_delay=float(np.mean([a["arrival_time"]-z["arrival_time"] for a,z in zip(l,b)])),
                energy_ratio=float(np.mean([r["control_energy"] for r in l])/np.mean([r["control_energy"] for r in b])),
                max_normalized_decay_rate=max(r["max_normalized_decay_rate"] for r in l),
                cost_vs_classical_SDRE_percent=float((lc.mean()/np.mean([r["fixed_horizon_cost"] for r in classical])-1)*100)))
        with torch.no_grad():
            ms=policy(task.error(probes));us=task.control(ms,task.error(probes),state=probes)
            baseline_u=task.control(torch.tensor(task.P0),task.error(probes),state=probes)
            variation=float((ms-ms[0]).abs().max());coupling=float((us-baseline_u).abs().max())
        reports[name]=dict(selection=selected["tasks"][name],aliases=aliases,comparisons=comparisons,
            test_gate=gate,M_state_variation=variation,action_difference_from_P0=coupling,
            audit_passed=all(a["passed"] for a in audits))
    return dict(tasks=reports,audited_episodes=total,
        scope="one training seed; paired intervals describe test initial states only",formal_stability_proof=False)


def make_report(out,result):
    lines=["# 完整回合镇定学习机制验收", "",
        "同一实际 M(e) 生成控制与候选证书，完整重算后继 M。采用完整时域梯度、近目标锚定、训练反例回放和独立验证门槛。",
        "物理模型、输入上限、成功保持、原物理成本均沿用四基座协议。学习方法仍为模型梯度，不是 PPO 或每步精确求解 SDRE。", "",
        "|任务|尺度|相对P0成本|配对95%区间|成功（学习/基线）|到达差s|能量比|",
        "|---|---:|---:|---|---|---:|---:|"]
    for name,task in result["tasks"].items():
        for r in task["comparisons"]:
            lo,hi=r["paired_bootstrap_95_percent"]
            lines.append(f'|{name}|{r["scale"]}|{r["cost_percent"]:+.2f}%|[{lo:+.2f}%, {hi:+.2f}%]|'
                f'{r["learned_success"]}/{r["episodes"]}；{r["baseline_success"]}/{r["episodes"]}|'
                f'{r["arrival_delay"]:+.3f}|{r["energy_ratio"]:.3f}|')
    lines += ["",f'独立NumPy审计共 {result["audited_episodes"]} 回合。以下验收失败或无收益仍保留：',""]
    for name,t in result["tasks"].items():
        lines.append(f'- {name}：验证选择更新 {t["selection"]["best_update"]}；独立测试门槛 '
            f'{"通过" if t["test_gate"]["qualified"] else "未通过："+repr(t["test_gate"]["reasons"])}；'
            f'状态M变化 {t["M_state_variation"]:.6g}，控制变化 {t["action_difference_from_P0"]:.6g}。')
    lines += ["", "测试在所有任务选模锁定后才读取，未用于重新选择模型。last_candidate 与选中策略相同则记别名，线性任务经典SDRE与P0相同。",
        "配对区间只描述单个训练种子的新初态；本实验不建立跨训练种子的可靠性结论。成功率基线已接近天花板时，成本改善不能改称成功率提升。",
        "局部锚定和浮点线性化检查不构成非线性区域或全局稳定证明。PGD未发现反例也不是形式验证。",
        "本次覆盖四个现有模型；书稿原加速度驱动倒立摆、小车原模型以及其他未接入任务仍未在本次补齐。",
        "", "研究依据与实现说明见 docs/METRIC_FULL_HORIZON_V3.md。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/metric_full_horizon_v3_pilot.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--development-only",action="store_true")
    parser.add_argument("--updates",type=int)
    parser.add_argument("--tasks",nargs="+",choices=TASKS)
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=args.output.resolve()
    if out.parent!=ROOT/"runs":raise ValueError("Use a new immediate child of this workspace's runs directory")
    with writer_lock():
        if args.resume:
            protocol=read(out/"protocol.json");check_sources(protocol)
            if (out/"STOP").exists() or (out/"FINAL_STATE.json").exists():
                raise ValueError("Stopped or finalized output cannot resume")
        else:
            out.mkdir(parents=False,exist_ok=False)
            setting=read(args.config);cfg=config_from(setting["training"])
            if cfg.seed!=41:raise ValueError("This authorized pilot uses seed41 only")
            if args.updates:
                if not args.development_only:raise ValueError("Budget override only belongs to explicit development runs")
                cfg=replace(cfg,updates=args.updates)
            protocol=dict(schema="metric_full_horizon_v3_acceptance",created_at=now(),
                tasks=args.tasks or setting["tasks"],training=asdict(cfg),hidden=setting["hidden"],
                residual_limit=setting["residual_limit"],sources=manifest(),
                config_sha256=sha(args.config),config=setting,development_only=args.development_only,
                deadline_unix=time.time()+setting["maximum_seconds"],test_probe_seed=6200001,
                authorization="调研资料，修改环境或者算法设计，使得能够真正提升；在这个窗口补全训练机制",
                existing_runs="sealed; no old STOP is removed and no old training is resumed",
                selection_rule=setting["selection"],test_policy=setting["test_policy"])
            write(out/"protocol.json",protocol)
            for p in protocol["sources"]:
                dest=out/"source_snapshot"/p;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/p,dest)
        def stop():
            if (out/"STOP").exists():return True
            if time.time()>protocol["deadline_unix"]:
                try:
                    with (out/"STOP").open("x",encoding="utf-8") as f:f.write("Declared mechanism acceptance time budget reached.\n")
                except FileExistsError:pass
                return True
            return False
        try:
            emit(out,"pipeline_started",pid=os.getpid(),tasks=protocol["tasks"],development_only=protocol["development_only"])
            selections=train_all(out,protocol,stop)
            if protocol["development_only"]:
                final=dict(status="development_complete_no_test_access",tasks=selections,at=now())
            else:
                result=evaluate_all(out,protocol,stop)
                check_sources(protocol)
                write(out/"RESULTS.json",result);make_report(out,result)
                final=dict(status="mechanism_acceptance_complete",at=now(),audited_episodes=result["audited_episodes"],
                    all_test_gates_passed=all(t["test_gate"]["qualified"] for t in result["tasks"].values()),
                    formal_stability_proof=False,seed43_started=False)
            write(out/"FINAL_STATE.json",final);emit(out,"pipeline_complete",**final)
        except BaseException as error:
            status="stopped" if isinstance(error,(InterruptedError,KeyboardInterrupt)) else "failed"
            write(out/"DIAGNOSTIC.json",dict(status=status,at=now(),error=repr(error),traceback=traceback.format_exc()))
            emit(out,"pipeline_"+status,error=repr(error))
            raise
