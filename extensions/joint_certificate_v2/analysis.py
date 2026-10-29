"""Read-only post-analysis of stopped v2 runs; creates a separate new artifact dir.

No checkpoint loads, optimization, fresh rollouts, smoothing or test selection.
Windows process enumeration is mandatory by default and fails closed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np


def _sha_bytes(data):return hashlib.sha256(data).hexdigest()


def _command_tail(command):
    match=re.match(r'^(?:"[^"]+"|\S+)\s*(.*)$',command.strip(),re.S)
    return match.group(1) if match else command


def assert_run_quiescent(root, *, processes=None, current_pid=None):
    """Reject matching Python writers or incomplete/failed process queries.

Matching is deliberately conservative: any other Python command naming this run
is considered a writer. Only this analysis PID and its identical-command Python
launcher ancestors are excluded. An unrelated meetings MCP is not a run writer.
"""
    root=Path(root).resolve();current_pid=os.getpid() if current_pid is None else current_pid
    automatic_query=processes is None
    if processes is None:
        if os.name!="nt":raise RuntimeError("Automatic quiescence verification currently requires Windows CIM")
        script="$ErrorActionPreference='Stop'; $items=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('python.exe','pythonw.exe','py.exe','python','pythonw','py') } | Select-Object ProcessId,ParentProcessId,Name,CommandLine); ConvertTo-Json -InputObject $items -Depth 3 -Compress"
        result=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",script],capture_output=True,text=True,timeout=30,check=False)
        if result.returncode!=0 or not result.stdout.strip():raise RuntimeError("Python process query failed; writer exit is unverified")
        try:processes=json.loads(result.stdout)
        except json.JSONDecodeError as error:raise RuntimeError("Process query was not valid JSON") from error
    if isinstance(processes,dict):processes=[processes]
    if not isinstance(processes,list):raise RuntimeError("Process query returned an unknown structure")
    for row in processes:
        if not isinstance(row,dict) or not row.get("CommandLine") or not row.get("ProcessId") or not row.get("Name"):
            raise RuntimeError("A Python process has missing identity/command line; cannot infer exit")
    by_pid={int(p["ProcessId"]):p for p in processes};excluded={current_pid}
    if automatic_query and current_pid not in by_pid:
        raise RuntimeError("CIM did not enumerate the current Python interpreter; query completeness is unverified")
    # Windows venv launchers sometimes remain alive while the actual interpreter runs.
    child=by_pid.get(current_pid)
    while child is not None:
        parent=by_pid.get(int(child.get("ParentProcessId",-1)))
        if parent is None or _command_tail(parent["CommandLine"])!=_command_tail(child["CommandLine"]):break
        excluded.add(int(parent["ProcessId"]));child=parent
    normalized=str(root).replace("\\","/").lower();name=root.name.lower()
    matches=[]
    for row in processes:
        command=row["CommandLine"].replace("\\","/").lower()
        if int(row["ProcessId"]) in excluded:continue
        if normalized in command or (len(name)>=7 and name in command):matches.append(int(row["ProcessId"]))
    if matches:raise RuntimeError("Run-related Python writers still active: "+", ".join(map(str,matches)))
    return dict(method="successful Windows CIM Python enumeration",checked_utc=datetime.now(timezone.utc).isoformat(),
                python_processes=len(processes),excluded_analysis_pids=sorted(excluded),matching_writer_pids=[])


def _snapshot_witness(root,path):
    """Validate a fresh complete inventory saved atomically by external PowerShell.

The producer must query every Python process successfully before committing this
file. Its path/hash are retained; this is an external observation, not a claim
that this interpreter's subprocess CIM view is necessarily system-wide.
"""
    path=Path(path).resolve();raw=path.read_bytes()
    try:data=json.loads(raw.decode("utf-8-sig"))
    except (ValueError,UnicodeError) as error:raise RuntimeError("Invalid external process snapshot") from error
    if not isinstance(data,dict) or data.get("schema")!="joint_v2_process_snapshot" or data.get("query_success") is not True:
        raise RuntimeError("External process snapshot schema/query success is unverified")
    created=data.get("created_unix")
    if isinstance(created,bool) or not isinstance(created,(int,float)) or not np.isfinite(created):
        raise RuntimeError("External process snapshot has no valid creation time")
    age=time.time()-created
    if age < -2 or age > 60:raise RuntimeError("External process snapshot must be no more than 60 seconds old and not future-dated")
    processes=data.get("processes")
    if not isinstance(processes,list):raise RuntimeError("External process snapshot must contain the complete Python process list")
    seen=set()
    for row in processes:
        if not isinstance(row,dict) or not isinstance(row.get("CommandLine"),str) or not row["CommandLine"].strip():
            raise RuntimeError("External process snapshot has an unknown/missing process identity")
        if row.get("Name","").lower() not in {"python.exe","pythonw.exe","py.exe","python","pythonw","py"}:
            raise RuntimeError("External process snapshot has an unknown process name")
        pid=row.get("ProcessId");parent=row.get("ParentProcessId")
        if isinstance(pid,bool) or not isinstance(pid,int) or pid<=0 or pid in seen or isinstance(parent,bool) or not isinstance(parent,int) or parent<0:
            raise RuntimeError("External process snapshot has an invalid/duplicate process identity")
        seen.add(pid)
    witness=assert_run_quiescent(root,processes=processes)
    witness.update(method="fresh external complete Windows CIM snapshot",snapshot_path=str(path),snapshot_sha256=_sha_bytes(raw),
                   snapshot_created_unix=created,snapshot_age_seconds=age,
                   automatic_query_limitation="Python subprocess and external PowerShell CIM inventories have differed; formal runs require the external snapshot")
    return witness


class Inputs:
    def __init__(self):self.hashes={}
    def json(self,path):
        path=Path(path).resolve();data=path.read_bytes();digest=_sha_bytes(data)
        if str(path) in self.hashes and self.hashes[str(path)]!=digest:raise RuntimeError("Input changed during analysis: "+str(path))
        self.hashes[str(path)]=digest
        return json.loads(data.decode("utf-8-sig"))


def matrix_stats(matrix,initial,previous):
    value=np.asarray(matrix,dtype=float)
    if value.ndim<2 or value.shape[-1]!=value.shape[-2] or not np.isfinite(value).all():
        raise ValueError("Invalid saved metric probe")
    if value.shape!=initial.shape or value.shape!=previous.shape:
        raise ValueError("Probe layout changed; matrix distances would be incomparable")
    flat=value.reshape(-1,value.shape[-1],value.shape[-1]);eigen=np.linalg.eigvalsh((flat+flat.transpose(0,2,1))/2)
    return dict(matrix_dimension=value.shape[-1],probe_count=len(flat),eigen_min=float(eigen.min()),eigen_max=float(eigen.max()),
        eigen_min_mean=float(eigen[:,0].mean()),eigen_max_mean=float(eigen[:,-1].mean()),
        trace_mean=float(np.trace(flat,axis1=1,axis2=2).mean()),
        relative_to_initial=float(np.linalg.norm(value-initial)/max(np.linalg.norm(initial),1e-12)),
        relative_step_change=float(np.linalg.norm(value-previous)/max(np.linalg.norm(previous),1e-12)),
        symmetry_error=float(np.max(abs(flat-flat.transpose(0,2,1)))))


def _csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def _scalar_fields(data):
    return {key:value for key,value in data.items() if value is None or isinstance(value,(str,int,float,bool))}


def _directories(root):
    """Avoid image replay, detailed trajectories and prior analysis products."""
    for directory,children,files in os.walk(root):
        relative=Path(directory).relative_to(root)
        if len(relative.parts)>3:children[:]=[];continue
        children[:]=[name for name in children if name not in {"replay","episodes","validation","source_snapshots","checkpoints","events"}
                     and not name.startswith(("deployment","POST_ANALYSIS","analysis_"))]
        if "protocol.json" in files:yield Path(directory)


def _event_rows(directory,reader,protocol_hash):
    for path in sorted((directory/"events").glob("*.json")):
        event=reader.json(path)
        if event.get("protocol_sha256") not in (None,protocol_hash):
            raise ValueError("Committed event belongs to a different protocol: "+str(path))
        yield event


def _push(directory,reader,events):
    training=reader.json(directory/"TRAINING.json") if (directory/"TRAINING.json").exists() else []
    raw=reader.json(directory/"M_PROBES.json") if (directory/"M_PROBES.json").exists() else []
    metrics=[];initial=previous=None
    for record in raw:
        matrix=np.asarray(record["M"],float)
        if initial is None:initial=matrix.copy();previous=matrix.copy()
        metrics.append(dict(step=record["update"],phase="joint",**matrix_stats(matrix,initial,previous)))
        previous=matrix
    updates=[]
    for record in training:
        row=dict(step=record["update"],**{k:v for k,v in _scalar_fields(record).items() if k!="update"})
        episodes=record.get("episodes",[])
        row["completed_episode_reward_mean"]=float(np.mean([e["reward"] for e in episodes])) if episodes else None
        row["completed_episode_success_rate"]=float(np.mean([e["success"] for e in episodes])) if episodes else None
        updates.append(row)
    validation=[dict(step=e["data"]["update"],**{k:v for k,v in _scalar_fields(e["data"]).items() if k!="update"})
                for e in events if e.get("kind")=="validation_complete"]
    return metrics,updates,validation


def _rgb(directory,reader,events):
    metrics=[];updates=[];validation=[];initial=previous=None;last_step=0
    for event in events:
        kind=event.get("kind");data=event.get("data",{})
        if kind=="fixed_training_probe":
            matrix=np.asarray(data["M"],float)
            if initial is None:initial=matrix.copy();previous=matrix.copy()
            row=dict(step=data["step"],sequence=event["sequence"],phase=data["phase"],
                normalization_scale=data["normalization_scale"],calibrated=data["calibrated"],version=data["version"],
                feature_error_norm_median=float(np.median(data["feature_error_norm"])),**matrix_stats(matrix,initial,previous))
            values=np.asarray(data["V"],float);physical=np.asarray(data["E"],float);valid=physical>1e-12
            row["V_over_E_median"]=float(np.median(values[valid]/physical[valid])) if valid.any() else None
            metrics.append(row);previous=matrix
        elif kind=="update" and {"step","block","metrics"}.issubset(data):
            last_step=data["step"]
            updates.append(dict(step=last_step,block=data["block"],pair=data.get("pair"),**_scalar_fields(data["metrics"])))
        elif kind=="validation":
            match=re.search(r"update_(\d+)",data.get("checkpoint",""))
            validation.append(dict(step=int(match.group(1)) if match else last_step,**_scalar_fields(data["summary"])))
    return metrics,updates,validation


def _plot(out,prefix,kind,metrics,updates,validation):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    artifacts=[]
    def line(ax,rows,key,label=None,**kwargs):
        used=[row for row in rows if isinstance(row.get(key),(int,float)) and row.get(key) is not None and np.isfinite(row[key])]
        if used:ax.plot([r["step"] for r in used],[r[key] for r in used],label=label or key,linewidth=1.,marker=".",markersize=3,**kwargs)
        return bool(used)
    if metrics:
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
        line(axes[0,0],metrics,"eigen_min",label="min across fixed probes")
        line(axes[0,0],metrics,"eigen_max",label="max across fixed probes")
        axes[0,0].set_title("M eigenvalue range");axes[0,0].legend(fontsize=8)
        line(axes[0,1],metrics,"relative_to_initial");axes[0,1].set_title("Frobenius distance / initial norm")
        line(axes[1,0],metrics,"relative_step_change");axes[1,0].set_title("Adjacent saved-probe change / previous norm")
        if kind=="rgb":
            line(axes[1,1],metrics,"normalization_scale",label="feature normalization scale")
            axes[1,1].set_yscale("log");axes[1,1].set_title("Coordinate calibration (same step may repeat)")
        else:
            line(axes[1,1],metrics,"trace_mean");axes[1,1].set_title("Mean trace of M")
            # Do not amplify floating-point drift of the trace-normalized matrix.
            center=float(metrics[0]["matrix_dimension"])
            traces=[row["trace_mean"] for row in metrics]
            axes[1,1].set_ylim(min(center-.01,min(traces)),max(center+.01,max(traces)))
            axes[1,1].ticklabel_format(axis="y",style="plain",useOffset=False)
        for axis in axes.flat:axis.set_xlabel("Actual optimizer update step");axis.grid(alpha=.2)
        fig.suptitle(prefix+": fixed input probes; raw, unsmoothed")
        name=prefix+"_M_vs_step.png";fig.savefig(out/name,dpi=150);plt.close(fig);artifacts.append(name)
    if updates or validation:
        fig,axes=plt.subplots(2,3,figsize=(14,7),layout="constrained")
        if kind=="rgb":
            for block in ("warmup","feature","metric"):
                subset=[r for r in updates if r["block"]==block]
                line(axes[0,0],subset,"total",label=block)
            axes[0,0].set_title("Block objectives (different losses)")
            for key in ("physical","calibration","interaction","reconstruction"):
                line(axes[0,1],updates,key)
            axes[0,1].set_title("Unweighted logged loss components")
            for key in ("feature_gradient_norm","metric_gradient_norm"):
                line(axes[0,2],updates,key)
            axes[0,2].set_title("Logged gradient norms")
            line(axes[1,0],metrics,"feature_error_norm_median");axes[1,0].set_title("Median feature-error norm on fixed images")
            line(axes[1,1],metrics,"V_over_E_median");axes[1,1].set_title("Median V / physical E on fixed images")
            line(axes[1,2],validation,"mean_integrated_error");axes[1,2].set_title("Validation physical-error integral")
        else:
            line(axes[0,0],updates,"loss");axes[0,0].set_title("PPO training loss")
            line(axes[0,1],updates,"completed_episode_reward_mean");axes[0,1].set_title("Completed-episode reward (gaps = no episodes)")
            line(axes[0,2],updates,"completed_episode_success_rate");axes[0,2].set_title("Completed training-episode success")
            line(axes[1,0],validation,"mean_combined_cost");axes[1,0].set_title("Validation complete joint cost")
            line(axes[1,1],validation,"success_rate");axes[1,1].set_title("Validation success")
            line(axes[1,2],validation,"mean_seconds");axes[1,2].set_title("Validation physical execution seconds")
        for axis in axes.flat:
            axis.set_xlabel("Actual optimizer update step");axis.grid(alpha=.2)
            handles,labels=axis.get_legend_handles_labels()
            if len(handles)>1:axis.legend(fontsize=7)
            if not handles:axis.text(.5,.5,"No logged data",ha="center",va="center",transform=axis.transAxes)
        fig.suptitle(prefix+": raw updates; validation is not held-out performance")
        name=prefix+"_training.png";fig.savefig(out/name,dpi=150);plt.close(fig);artifacts.append(name)
    return artifacts


def analyze(root,output,*,verify_writers=True,process_snapshot=None):
    root=Path(root).resolve();out=Path(output).resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    if out==root or out.exists():raise FileExistsError("Analysis needs a new output directory; inputs are never overwritten")
    if (root/"NIGHT_PROTOCOL.json").exists() and process_snapshot is None:
        raise RuntimeError("Formal night-run analysis requires a fresh external complete process snapshot")
    if process_snapshot is not None and not verify_writers:
        raise ValueError("A supplied process snapshot must be verified")
    witness=(_snapshot_witness(root,process_snapshot) if process_snapshot is not None else
             assert_run_quiescent(root) if verify_writers else dict(method="caller explicitly verified writer exit; automatic check disabled"))
    reader=Inputs();branches=[];warnings=[]
    overall_state=reader.json(root/"FINAL_STATE.json") if (root/"FINAL_STATE.json").exists() else {}
    # Stream large RGB events rather than retaining every 16x16 probe matrix.
    for directory in _directories(root):
        protocol=reader.json(directory/"protocol.json");digest=reader.hashes[str((directory/"protocol.json").resolve())]
        state=reader.json(directory/"FINAL_STATE.json") if (directory/"FINAL_STATE.json").exists() else {"status":"missing_final_state"}
        if state.get("status")!="complete":warnings.append(f"{directory.relative_to(root).as_posix()} 最终状态为 {state.get('status','unknown')}；只展示已有记录，不称完整训练。")
        event_iterator=_event_rows(directory,reader,digest)
        if (directory/"M_PROBES.json").exists() or (directory/"TRAINING.json").exists():
            kind="push";metrics,updates,validation=_push(directory,reader,event_iterator)
        else:
            kind="rgb";metrics,updates,validation=_rgb(directory,reader,event_iterator)
            if not metrics and not updates:continue
        frozen=reader.json(directory/"FROZEN.json") if (directory/"FROZEN.json").exists() else {}
        config=protocol.get("config",{});seed=config.get("seed","unknown")
        relative=directory.relative_to(root).as_posix();prefix=f"{kind}_seed{seed}_"+hashlib.sha256(relative.encode()).hexdigest()[:6]
        status=state.get("status","unknown")
        if not metrics:warnings.append(f"{relative} 缺少 M 探针，无法画出 M 随 step 的变化。")
        branch=dict(path=str(directory),relative_path=relative,kind=kind,seed=seed,status=status,
                    selected_update=frozen.get("selected_update"),probe_records=len(metrics),update_records=len(updates),
                    first_step=metrics[0]["step"] if metrics else None,last_step=metrics[-1]["step"] if metrics else None,
                    final_relative_M_change=metrics[-1]["relative_to_initial"] if metrics else None,
                    prefix=prefix,metrics=metrics,updates=updates,validation=validation,figures=[])
        branches.append(branch)
    if not branches:warnings.append("未找到可识别的 Push/RGB 训练曲线；不据此推断任务完成或性能。")
    out.mkdir(parents=True,exist_ok=False)
    for branch in branches:
        prefix=branch["prefix"]
        for name in ("metrics","updates","validation"):_csv(out/f"{prefix}_{name}.csv",branch[name])
        branch["figures"]=_plot(out,prefix,branch["kind"],branch["metrics"],branch["updates"],branch["validation"])
    notes=["横轴是实际优化更新计数，不是闭环物理时间。曲线未平滑，未补造缺失点，也未重训或改选模型。",
           "M 的谱、相邻变化和距初始值是固定探针上的数值；矩阵不要求逐元素单调变化，曲线也不证明收敛或稳定性。",
           "RGB 编码器更新会改变目标特征，故 M 探针的变化同时包含条件输入变化；它不等于 M 网络参数变化量。",
           "RGB 校准可在同一 optimizer step 改变坐标尺度：同一步的两个探针须保留，不能伪装成新增训练步。",
           "random_initial 是 AE 预热与坐标标定之前的基线；比较改善包含标定效应。隔离 M 效果应使用同一已训练编码器的 identity 对照。",
           "训练回报只在批次内有已完成回合时绘出；没有完成回合的更新不是零回报。验证曲线不能当作留出测试成绩。",
           "不同训练种子的共享测试起点不会增加独立测试初态；跨种子 SD 不是置信区间。本报告逐种子展示，不制造 CI。",
           "Push 时间来自降阶响应模型；真实关节驱动 MuJoCo 的验收需另看 arm 结果。两者均不等于真机，位姿输入不等于 RGB 定位。",
           "RGB 当前是四自由度速度外环；固定公共深度模块与既有房间几何不属于从随机初始化学习的编码器/M。"]
    report=["# 联合镇定学习：真实训练步骤曲线", "",f"输入：`{root}`", "", "本报告只读取已有训练记录，另存分析产物。", ""]
    for branch in branches:
        report += [f"## {branch['kind'].upper()}，seed {branch['seed']}", "",
            f"目录：`{branch['relative_path']}`。最终状态：**{branch['status']}**。记录 {branch['update_records']} 次更新、{branch['probe_records']} 个 M 探针；末端 step={branch['last_step']}，验证所选 step={branch['selected_update']}。", ""]
        if branch["metrics"]:
            report += [f"最后探针相对初始 M 的 Frobenius 变化为 {branch['final_relative_M_change']:.6g}；仅是该探针集合的数值变化。", ""]
        for name in branch["figures"]:report += [f"![{name}]({(out/name).as_posix()})", ""]
    report += ["## 解释边界", ""]+["- "+note for note in notes]+["", "## 缺失与失败", ""]
    report += ["- "+item for item in warnings] if warnings else ["未发现输入记录标记的失败；这不代替性能或证书验收。"]
    (out/"ANALYSIS_ZH.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    if verify_writers:
        witness["final_check"]=assert_run_quiescent(root)
        if process_snapshot is not None:
            if _sha_bytes(Path(process_snapshot).read_bytes())!=witness["snapshot_sha256"]:
                raise RuntimeError("External process snapshot changed during analysis")
            witness["final_check"]["scope"]="supplemental subprocess CIM check; initial external snapshot is the complete inventory witness"
    summary=dict(schema="joint_v2_post_analysis",root=str(root),overall_status=overall_state.get("status","not_recorded"),branches=[{k:v for k,v in b.items() if k not in ("metrics","updates","validation")} for b in branches],
                 warnings=warnings,notes=notes,writer_exit=witness,training_modified=False,performance_inferred=False)
    (out/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    provenance=dict(inputs=reader.hashes,analysis_source={str(Path(__file__).resolve()):_sha_bytes(Path(__file__).read_bytes())},
        generated_utc=datetime.now(timezone.utc).isoformat(),writer_exit=witness,
        outputs={p.name:_sha_bytes(p.read_bytes()) for p in out.iterdir() if p.is_file()})
    (out/"PROVENANCE.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--process-snapshot")
    args=parser.parse_args();print(json.dumps(analyze(args.root,args.output,process_snapshot=args.process_snapshot),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
