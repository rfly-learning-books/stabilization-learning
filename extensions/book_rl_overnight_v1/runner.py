"""E13 integrator subexperiment: equal-budget training and frozen held-out tests."""
from __future__ import annotations
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import uuid

import numpy as np
import torch
from .env import BoundedIntegrator,analytic_action,bounded,certificate_filter,value
from .algo import SAC,Replay


METHODS=("sac","sac_soft","sac_filter")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temporary.rename(path)


def save_model(path,learner):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temp=path.with_suffix(".tmp")
    torch.save(dict(learner=learner.payload(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if str(learner.device).startswith("cuda") else None),temp)
    temp.rename(path)


def csv_rows(path,rows):
    if not rows:return
    with path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader();writer.writerows(rows)


def validate(cfg):
    if cfg.get("schema")!="book_e13_integrator_sac_v1" or cfg.get("methods")!=list(METHODS):raise ValueError("Declare all three E13 methods")
    if cfg.get("mode") not in ("formal","engineering"):raise ValueError("Declare formal or engineering mode")
    for key in ("parallel_envs","epochs","transitions_per_method","random_transitions","updates_per_vector_step","batch_size","train_starts","validation_starts","test_starts","evaluation_horizon"):
        if type(cfg[key]) is not int or cfg[key]<=0:raise ValueError("Invalid positive count: "+key)
    total,lanes,epochs=cfg["transitions_per_method"],cfg["parallel_envs"],cfg["epochs"]
    if total%(lanes*epochs) or cfg["random_transitions"]%lanes or cfg["random_transitions"]>=total:raise ValueError("Use exact vector-step transition budgets")
    if cfg["batch_size"]>cfg["random_transitions"]:raise ValueError("Replay prefill must cover batch")
    if not 0<cfg["maximum_minutes"]<=40:raise ValueError("Maximum E13 budget is 40 minutes")
    if cfg["mode"]=="formal" and (cfg["seed"]!=41 or cfg["device"]!="cuda:0" or total!=32768 or lanes!=64 or cfg["batch_size"]!=256):raise ValueError("Formal E13 uses declared seed41 CUDA 32768-transition/64-lane/batch256 design")
    return cfg


def starts_manifest(cfg):
    result={}
    for split,seed,rlo,rhi in (("train",14041,.15,1.5),("validation",24041,.2,1.7),("test",34041,.2,2.0)):
        n=cfg[split+"_starts"]
        rng=np.random.default_rng(seed)
        angle=rng.uniform(-math.pi,math.pi,n)
        radius=np.sqrt(rng.uniform(rlo*rlo,rhi*rhi,n))
        result[split]=np.stack([radius*np.cos(angle),radius*np.sin(angle)],-1).astype(np.float32)
    tuples=[{tuple(v) for v in data} for data in result.values()]
    if any(tuples[i]&tuples[j] for i in range(3) for j in range(i+1,3)):raise RuntimeError("Split initial states overlap")
    return result


def action_execution(state,proposal,method,cfg):
    saturated=bounded(proposal)
    if method=="sac_filter":
        executed,info=certificate_filter(state,proposal,cfg["environment"]["dt"],cfg["decay"])
    else:
        executed=saturated
        flags=torch.ones(len(state),dtype=torch.bool,device=state.device)
        info=dict(feasible=flags,fallback=~flags,solver_failure=~flags,
            correction=(executed-proposal).norm(dim=-1),
            nominal_residual=value(state+cfg["environment"]["dt"]*executed)-value(state)+cfg["decay"]*cfg["environment"]["dt"]*state.square().sum(-1))
    info.update(saturation_correction=(saturated-proposal).norm(dim=-1),
                certificate_correction=(executed-saturated).norm(dim=-1))
    return executed,info


def physical_audit(data,cfg):
    x,nxt,u,w=(data[k].astype(float) for k in ("state","next_state","executed","disturbance"))
    dt=cfg["environment"]["dt"]
    delta=.5*(np.sum(nxt*nxt,axis=-1)-np.sum(x*x,axis=-1))
    transition=float(np.max(np.abs(nxt-x-dt*u-w)))
    v_error=float(np.max(np.abs(delta-data["true_delta"])))
    cost=dt*(np.sum(nxt*nxt,axis=-1)+.01*np.sum(u*u,axis=-1))+20*data["failure"]
    cost_error=float(np.max(np.abs(cost-data["base_cost"])))
    if transition>1e-6 or v_error>2e-6 or cost_error>4e-6 or np.max(np.linalg.norm(u,axis=-1))>1.00001:
        raise AssertionError("Physical transition, V, cost or bounded-action audit failed")
    return dict(passed=True,max_transition_error=transition,max_V_error=v_error,max_cost_error=cost_error)


def evaluate(learner,method,starts,cfg,output,label,guard,disturbed=False):
    started=time.time()
    output.mkdir(parents=True,exist_ok=False)
    ecfg=deepcopy(cfg["environment"]);ecfg["horizon"]=cfg["evaluation_horizon"]
    device=cfg["device"];env=BoundedIntegrator(starts,len(starts),device=device,**ecfg)
    n=len(starts);alive=torch.ones(n,dtype=torch.bool,device=device)
    length=torch.zeros(n,dtype=torch.long,device=device)
    arrived=torch.full((n,),-1,dtype=torch.long,device=device)
    hold_reached=torch.full_like(arrived,-1)
    rng=np.random.default_rng(44041)
    disturbances=np.zeros((ecfg["horizon"],n,2),np.float32)
    if disturbed:
        disturbances=rng.uniform(-cfg["disturbance_uniform_half_width"],cfg["disturbance_uniform_half_width"],disturbances.shape).astype(np.float32)
        if cfg["impulse_step"]<ecfg["horizon"]:
            angles=np.linspace(0,2*math.pi,n,endpoint=False)
            disturbances[cfg["impulse_step"]]+=cfg["impulse_magnitude"]*np.stack([np.cos(angles),np.sin(angles)],-1)
    records=[];latencies=[]
    for t in range(ecfg["horizon"]):
        guard()
        if not bool(alive.any()):break
        if str(device).startswith("cuda"):torch.cuda.synchronize()
        tick=time.perf_counter()
        proposal=analytic_action(env.state) if method=="analytic" else learner.action(env.state,deterministic=True)
        executed,info=action_execution(env.state,proposal,method,cfg)
        if str(device).startswith("cuda"):torch.cuda.synchronize()
        latencies.append(time.perf_counter()-tick)
        executed=torch.where(alive[:,None],executed,torch.zeros_like(executed))
        disturbance=torch.as_tensor(disturbances[t],device=device)*alive[:,None]
        result=env.step(executed,disturbance)
        true_residual=result["true_delta"]+cfg["decay"]*ecfg["dt"]*result["state"].square().sum(-1)
        arrived=torch.where(alive&(arrived<0)&(env.state.norm(dim=-1)<=ecfg["tolerance"]),env.steps,arrived)
        hold_reached=torch.where(alive&(hold_reached<0)&(env.hold>=ecfg["hold_steps"]),env.steps,hold_reached)
        records.append(dict(**result,proposal=proposal,disturbance=disturbance,active=alive.clone(),
            true_residual=true_residual,**info))
        length+=alive;alive &= ~result["failure"]
    keys=records[0].keys()
    arrays={key:torch.stack([row[key] for row in records]).detach().cpu().numpy() for key in keys}
    audit=physical_audit(arrays,cfg)
    with (output/"trajectories.npz").open("xb") as f:np.savez_compressed(f,**arrays)
    lengths=length.cpu().numpy();first=arrived.cpu().numpy();first_hold=hold_reached.cpu().numpy()
    rows=[]
    for i,L in enumerate(lengths):
        result={key:values[:L,i] for key,values in arrays.items()}
        failure=bool(result["failure"][-1]);final_hold=int(result["hold"][-1])
        success=bool(not failure and L==ecfg["horizon"] and final_hold>=ecfg["hold_steps"])
        cost=result["base_cost"]
        rows.append(dict(case_id=f"{label}_{i:04d}",method=method,condition="disturbed" if disturbed else "nominal",
            success=success,reason="boundary_failure" if failure else "success" if success else "hold_not_met",
            steps=int(L),physical_cost=float(cost.sum()),discounted_base_return=float(-(cost*cfg["sac"]["gamma"]**np.arange(L)).sum()),
            final_radius=float(np.linalg.norm(result["next_state"][-1])),final_hold=final_hold,
            first_arrival_step=int(first[i]),first_hold_step=int(first_hold[i]),
            arrival_hold_censored=bool(first[i]>0 and ecfg["horizon"]-first[i]+1<ecfg["hold_steps"]),
            true_descent_violations=int((result["true_residual"]>1e-6).sum()),
            nominal_descent_violations=int((result["nominal_residual"]>1e-6).sum()),
            filter_interventions=int((result["certificate_correction"]>1e-6).sum()),
            saturation_events=int((result["saturation_correction"]>1e-6).sum()),
            mean_total_correction=float(result["correction"].mean()),
            mean_certificate_correction=float(result["certificate_correction"].mean()),fallbacks=int(result["fallback"].sum()),
            infeasible=int((~result["feasible"]).sum()),solver_failures=int(result["solver_failure"].sum()),
            batch_decision_ms=float(1000*np.mean(latencies[:L])),batch_size=n))
    summary=dict(episodes=n,successes=sum(r["success"] for r in rows),success_rate=float(np.mean([r["success"] for r in rows])),
        mean_physical_cost=float(np.mean([r["physical_cost"] for r in rows])),
        cost_quantile_05=float(np.quantile([r["physical_cost"] for r in rows],.05)),cost_quantile_95=float(np.quantile([r["physical_cost"] for r in rows],.95)),
        failures=sum(r["reason"]=="boundary_failure" for r in rows),
        true_violation_fraction=sum(r["true_descent_violations"] for r in rows)/sum(r["steps"] for r in rows),
        fallback_count=sum(r["fallbacks"] for r in rows),mean_batch_decision_ms=float(np.mean(latencies)*1000))
    record=dict(label=label,method=method,disturbed=disturbed,summary=summary,rows=rows,started_unix=started,finished_unix=time.time(),
        audit=audit,trajectory_sha256=sha(output/"trajectories.npz"),independent_test=label.startswith("test"))
    write(output/"RESULTS.json",record);csv_rows(output/"episodes.csv",rows)
    return record


def train_method(method,cfg,starts,root,guard,event):
    seed=cfg["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if cfg["device"].startswith("cuda"):torch.cuda.manual_seed_all(seed)
    output=root/method;output.mkdir()
    learner=SAC(cfg["sac"],cfg["device"])
    save_model(output/"checkpoints/initial.pt",learner)
    env=BoundedIntegrator(starts["train"],cfg["parallel_envs"],device=cfg["device"],**cfg["environment"])
    replay=Replay(cfg["transitions_per_method"],cfg["device"])
    per_epoch=cfg["transitions_per_method"]//cfg["epochs"]//cfg["parallel_envs"]
    count=0;best=None;validations=[];losses=[];epoch_rows=[]
    for epoch in range(1,cfg["epochs"]+1):
        history=[];begin=time.perf_counter()
        for _ in range(per_epoch):
            guard()
            proposal=torch.rand_like(env.state)*2-1 if count<cfg["random_transitions"] else learner.action(env.state)
            executed,info=action_execution(env.state,proposal,method,cfg)
            result=env.step(executed)
            residual=result["true_delta"]+cfg["decay"]*cfg["environment"]["dt"]*result["state"].square().sum(-1)
            penalty=cfg["soft_weight"]*residual.clamp_min(0) if method=="sac_soft" else torch.zeros_like(residual)
            reward=result["base_reward"]-penalty
            replay.add(dict(state=result["state"],proposal=proposal,executed=executed,next_state=result["next_state"],
                reward=reward,base_reward=result["base_reward"],penalty=penalty,failure=result["failure"].float()))
            history.append(dict(**result,proposal=proposal,disturbance=torch.zeros_like(env.state),
                training_reward=reward,soft_penalty=penalty,true_residual=residual,
                lane=torch.arange(cfg["parallel_envs"],device=cfg["device"]),**info))
            count+=cfg["parallel_envs"]
            if count>cfg["random_transitions"]:
                for _ in range(cfg["updates_per_vector_step"]):
                    guard();logs=learner.update(replay.sample(cfg["batch_size"]))
                    losses.append(dict(method=method,epoch=epoch,transitions=count,**logs))
                    if not all(math.isfinite(v) for v in logs.values()):raise FloatingPointError("Nonfinite SAC update")
                    with (output/"updates_stream.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(losses[-1],allow_nan=False)+"\n")
            done=torch.nonzero(result["failure"]|result["truncated"],as_tuple=False).flatten()
            if len(done):env.reset(done)
        arrays={k:torch.cat([row[k] for row in history]).detach().cpu().numpy() for k in history[0]}
        audit=physical_audit(arrays,cfg)
        with (output/f"transitions_epoch_{epoch:02d}.npz").open("xb") as f:np.savez_compressed(f,**arrays)
        validation=evaluate(learner,method,starts["validation"],cfg,output/f"validation_epoch_{epoch:02d}","validation",guard)
        summary=validation["summary"];score=(summary["success_rate"],-summary["mean_physical_cost"])
        checkpoint=output/f"checkpoints/epoch_{epoch:02d}.pt";save_model(checkpoint,learner)
        if best is None or score>tuple(best["score"]):best=dict(epoch=epoch,score=score,checkpoint=str(checkpoint.relative_to(root)),sha256=sha(checkpoint))
        validations.append(dict(method=method,epoch=epoch,transitions=count,**summary))
        row=dict(method=method,epoch=epoch,transitions=count,updates=learner.updates,
            epoch_seconds=time.perf_counter()-begin,mean_training_base_cost=float(arrays["base_cost"].mean()),
            mean_soft_penalty=float(arrays["soft_penalty"].mean()),training_failures=int(arrays["failure"].sum()),
            true_violation_fraction=float((arrays["true_residual"]>1e-6).mean()),
            filter_correction_fraction=float((arrays["certificate_correction"]>1e-6).mean()),fallbacks=int(arrays["fallback"].sum()),
            selected_epoch=best["epoch"])
        epoch_rows.append(row);event("epoch_completed",**row,validation=summary)
        write(output/f"epoch_{epoch:02d}.json",dict(training=row,validation=summary,physics_audit=audit,selected=best))
        csv_rows(output/f"updates_epoch_{epoch:02d}.csv",[r for r in losses if r["epoch"]==epoch])
    csv_rows(output/"training_epochs.csv",epoch_rows);csv_rows(output/"validation_curves.csv",validations)
    csv_rows(output/"all_updates.csv",losses)
    write(output/"FROZEN.json",dict(method=method,**best,transitions=count,updates=learner.updates,
        selection="validation success rate, then negative common physical cost; strict lexicographic improvement"))
    return best,validations


def figures(root,validations,tests,mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory=root/"figures";directory.mkdir()
    plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.2,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for method in METHODS:
        rows=[r for r in validations if r["method"]==method]
        for axis,key in zip(axes,("mean_physical_cost","success_rate","true_violation_fraction")):
            axis.plot([r["transitions"] for r in rows],[r[key] for r in rows],"o-",label=method)
            axis.set(xlabel="Real training transitions",ylabel=key)
    for axis in axes:axis.legend(fontsize=8)
    fig.suptitle("E13 integrator: all validation epochs, equal transition budgets")
    names=[]
    def save(fig,name):
        if mode=="engineering":fig.text(.99,-.015,"ENGINEERING SMOKE: reduced schedule and horizon",ha="right",fontsize=8)
        fig.savefig(directory/(name+".png"),dpi=300,bbox_inches="tight");fig.savefig(directory/(name+".pdf"),bbox_inches="tight")
        plt.close(fig);names.append(name)
    save(fig,"learning_curves")
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for col,disturbed in enumerate((False,True)):
        selected=[r for r in tests if r["disturbed"]==disturbed]
        labels=[r["method"] for r in selected]
        axes[0,col].boxplot([[row["physical_cost"] for row in r["rows"]] for r in selected],labels=labels,showfliers=True)
        axes[0,col].set(title="Disturbed test" if disturbed else "Nominal independent test",ylabel="Physical cost across initial states")
        for r in selected:
            s=r["summary"];axes[1,col].scatter(s["true_violation_fraction"],s["mean_physical_cost"],s=65)
            axes[1,col].annotate(f'{r["method"]}\n{s["successes"]}/{s["episodes"]}',(s["true_violation_fraction"],s["mean_physical_cost"]),fontsize=8)
        axes[1,col].set(xlabel="True decrease violation fraction",ylabel="Mean physical cost")
    fig.suptitle("Single seed; test-state distributions are not cross-seed confidence intervals")
    save(fig,"test_performance_and_constraints")
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for r in tests:
        if r["method"]!="sac_filter":continue
        mode="disturbed" if r["disturbed"] else "nominal"
        with np.load(root/"tests"/f"sac_filter_{mode}"/"trajectories.npz") as a:
            mask=a["active"][:,0];n=int(mask.sum());time_axis=np.arange(n)*.1
            column=int(r["disturbed"])
            axes[0,column].plot(time_axis,a["proposal"][:n,0,0],label="proposal x")
            axes[0,column].plot(time_axis,a["executed"][:n,0,0],label="executed x")
            axes[0,column].plot(time_axis,a["proposal"][:n,0,1],ls="--",label="proposal y")
            axes[0,column].plot(time_axis,a["executed"][:n,0,1],ls="--",label="executed y")
            axes[1,column].plot(time_axis,a["true_residual"][:n,0],label="true D + margin")
            axes[1,column].plot(time_axis,a["nominal_residual"][:n,0],label="nominal D + margin")
            axes[1,column].axhline(0,color="black",lw=.8)
            axes[0,column].set_title("First fixed test start: "+mode)
    for axis in axes.flat:axis.legend(fontsize=8);axis.set_xlabel("Physical time (s)")
    save(fig,"proposal_executed_and_true_descent")
    write(root/"FIGURES_COMPLETE.json",dict(complete=True,mode=mode,figures=names,dpi=300,formats=["PNG","PDF"],scope="E13 analytic small-system subexperiment"))


def run(cfg,root):
    validate(cfg);root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA required by formal E13")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root.mkdir(parents=True);start=time.time();deadline=start+60*cfg["maximum_minutes"]
    source=[*Path(__file__).parent.glob("*.py"),Path(__file__).resolve().parents[2]/"scripts/run_book_rl_overnight_v1.py"]
    starts=starts_manifest(cfg)
    protocol=dict(schema=cfg["schema"],config=cfg,started_unix=start,deadline_unix=deadline,
        sources={str(p.resolve()):sha(p) for p in source},scope="Reviewed E13 analytic 2D integrator subexperiment only; existing application experiment remains separate",
        observation="2D physical state, identical for all methods",proposal="tanh Gaussian action in [-1,1]^2",
        execution="unit-disk saturation for all; sac_filter additionally projects into exact nominal discrete-V constraint",
        q_semantics="Q(state, proposal); replay action and actor/target proposals are identical coordinates; deterministic wrapper defines the MDP",
        certificate=f"V=.5||x||^2; D+{cfg['decay']}*dt*||x||^2 <= 0 nominally",discount=cfg["sac"]["gamma"],
        common_cost="dt*(||next_state||^2+.01||executed||^2)+20*boundary_failure",
        soft_training_reward="negative common cost minus soft_weight*relu(true D + decay*dt*||x||^2); extra term logged separately",
        reset="no first-arrival termination; failure terminates; horizon truncates and bootstraps",
        success=f"final {cfg['environment']['hold_steps']} consecutive in-tolerance successor states after full evaluation horizon",
        tuning_budget="one prespecified configuration per learned method, no test-driven changes",
        starts={key:value.tolist() for key,value in starts.items()})
    write(root/"protocol.json",protocol)
    for key,data in starts.items():np.save(root/(key+"_starts.npy"),data)
    event_counter=0
    def event(kind,**fields):
        nonlocal event_counter
        write(root/"events"/f"{event_counter:06d}_{kind}.json",dict(kind=kind,unix=time.time(),**fields));event_counter+=1
    def guard():
        if (root/"STOP").exists():raise InterruptedError("E13 STOP requested")
        if time.time()>=deadline:raise TimeoutError("E13 absolute budget reached")
        if shutil.disk_usage(root).free<cfg.get("minimum_free_gib",5)*1024**3:raise InterruptedError("E13 disk reserve reached")
    final=dict(status="running",scope=protocol["scope"],started_unix=start)
    try:
        diagnostic_states=torch.tensor([[0.,0.],[1.,0.],[3.,0.],[5.,0.]],device=cfg["device"])
        diagnostic_u,diagnostic=certificate_filter(diagnostic_states,torch.ones_like(diagnostic_states),cfg["environment"]["dt"],cfg["decay"])
        write(root/"FILTER_FEASIBILITY_DIAGNOSTIC.json",dict(states=diagnostic_states.cpu().tolist(),executed=diagnostic_u.cpu().tolist(),
            outside_domain_stress_indices=[3],**{k:v.cpu().tolist() for k,v in diagnostic.items()}))
        frozen={};validations=[]
        for method in METHODS:
            event("method_started",method=method)
            best,history=train_method(method,cfg,starts,root,guard,event)
            frozen[method]=dict(**best,marker_sha256=sha(root/method/"FROZEN.json"));validations+=history
        for path,digest in protocol["sources"].items():
            if sha(path)!=digest:raise RuntimeError("E13 source changed during training")
        write(root/"GLOBAL_FROZEN.json",dict(methods=frozen,protocol_sha256=sha(root/"protocol.json"),frozen_unix=time.time()))
        event("all_models_frozen",methods=list(METHODS))
        tests=[]
        for method in (*METHODS,"analytic"):
            learner=None
            if method!="analytic":
                selected=root/frozen[method]["checkpoint"]
                if sha(selected)!=frozen[method]["sha256"]:raise RuntimeError("Frozen E13 checkpoint changed")
                learner=SAC(cfg["sac"],cfg["device"])
                learner.load_policy(torch.load(selected,map_location=cfg["device"],weights_only=False)["learner"])
            for disturbed in (False,True):
                mode="disturbed" if disturbed else "nominal"
                result=evaluate(learner,method,starts["test"],cfg,root/"tests"/f"{method}_{mode}","test",guard,disturbed)
                tests.append(result);event("test_completed",method=method,condition=mode,summary=result["summary"])
        csv_rows(root/"test_results.csv",[dict(method=r["method"],condition="disturbed" if r["disturbed"] else "nominal",**r["summary"]) for r in tests])
        write(root/"RESULTS.json",dict(scope=protocol["scope"],frozen=frozen,tests=tests,validations=validations,
            fairness=dict(real_transitions_per_method=cfg["transitions_per_method"],same_network_initialization_seed=cfg["seed"],same_observation=True,same_base_cost=True,all_methods_frozen_before_test=True),single_seed=True))
        figures(root,validations,tests,cfg["mode"])
        lines=["# E13：解析小系统强化学习与固定证书对照","",protocol["scope"],"",f"运行模式：{cfg['mode']}。", "",
            "Q输入候选动作，回放同时保存候选与执行动作。所有评价使用同一物理成本，软惩罚仅改变该方法训练奖励。",
            "训练不在首次到达时终止；评价要求完整时域末端持续保持。未见初态和共同扰动序列在全部方法冻结后使用。","",
            "| 方法 | 工况 | 成功/起点 | 平均物理成本 | 真实下降违例率 | 回退次数 |","|---|---|---:|---:|---:|---:|"]
        for r in tests:
            s=r["summary"];lines.append(f'| {r["method"]} | {"扰动" if r["disturbed"] else "名义"} | {s["successes"]}/{s["episodes"]} | {s["mean_physical_cost"]:.5g} | {s["true_violation_fraction"]:.4f} | {s["fallback_count"]} |')
        lines += ["","单seed结果；成本箱线图表示独立测试起点分布，不表示跨seed置信区间。固定证书名义约束与扰动后的真实D分别记录。", "", "本次仅覆盖E13解析小系统部分，已有闭环应用三方法对照仍需独立实验。"]
        (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        final["status"]="complete";event("completed")
    except BaseException as error:
        final.update(status="stopped" if isinstance(error,(InterruptedError,TimeoutError,KeyboardInterrupt)) else "failed",error=repr(error),traceback=traceback.format_exc())
        event("stopped",status=final["status"],error=repr(error));raise
    finally:
        final.update(finished_unix=time.time(),wall_seconds=time.time()-start)
        write(root/"FINAL_STATE.json",final)
    return final
