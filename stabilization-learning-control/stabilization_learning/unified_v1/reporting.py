"""Read-only, post-writer scientific report for the bounded integration suite.

No checkpoint is unpickled, model reselected, or original artifact modified.
The process inventory must be produced by external PowerShell immediately
before this command. Unknown Python processes are conservatively rejected.
"""
from __future__ import annotations

from collections import Counter
from html import escape
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .runner import read, sha256, write_new


def check_process_snapshot(path, now=None):
    snapshot = read(path)
    now = time.time() if now is None else now
    if snapshot.get("schema") != "unified_training_process_snapshot" or snapshot.get("query_success") is not True:
        raise RuntimeError("A successful external unified_training_process_snapshot is required")
    created = snapshot.get("created_unix")
    if not isinstance(created, (int, float)) or isinstance(created, bool) or not -5 <= now-created <= 60:
        raise RuntimeError("External process snapshot must be less than 60 seconds old")
    processes = snapshot.get("processes")
    if not isinstance(processes, list):
        raise RuntimeError("Complete processes array is required, even when empty")
    ignored = []
    for process in processes:
        for key in ("ProcessId", "ParentProcessId", "Name", "CommandLine"):
            if key not in process or process[key] is None:
                raise RuntimeError("Incomplete process identity: " + repr(process))
        cmd = str(process["CommandLine"]).strip()
        if not cmd or not re.fullmatch(r"(?:pythonw?|py)(?:\d+(?:\.\d+)*)?\.exe", str(process["Name"]), re.I):
            raise RuntimeError("Unknown or incomplete Python process identity")
        normalized = cmd.replace("\\", "/").lower()
        meetings = "/chatgpt-meetings/" in normalized and "/scripts/meetings_mcp_entrypoint.py" in normalized
        conda_hook = "/conda" in normalized and bool(re.search(r"shell\.(?:powershell|cmd\.exe|posix)\s+hook(?:\s|$)", normalized))
        if not (meetings or conda_hook):
            raise RuntimeError("Active writer or unknown Python process: " + cmd)
        ignored.append(dict(pid=process["ProcessId"], command=cmd, reason="meetings MCP" if meetings else "conda shell hook"))
    return dict(snapshot_sha256=sha256(path), created_unix=created, checked_unix=now,
                python_processes=len(processes), classified_nonwriters=ignored, passed=True)


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    values = [v for v in values if _number(v)]
    return float(np.mean(values)) if values else None


def _compact(value):
    return {k: v for k, v in value.items() if k not in ("rows", "episodes", "trace", "information_windows")}


def _safe_path(root, value):
    path = Path(value)
    path = (root/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("Artifact path escapes its declared job: " + str(path))
    return path


def _integrity(root, protocol, final):
    from stabilization_learning.storage import verify_checkpoint
    problems, inventory = [], {}
    verified_objects = set()
    phash = sha256(root/"protocol.json")
    for path, expected in protocol.get("sources", {}).items():
        actual = sha256(path) if Path(path).is_file() else None
        if actual != expected:
            problems.append("source changed/missing: " + path)
    events = [read(p) for p in sorted((root/"events").glob("*.json"))]
    for e in events:
        if e.get("protocol_sha256") != phash:
            problems.append("event protocol hash mismatch")
    frozen_path = root/"GLOBAL_FROZEN.json"
    frozen = read(frozen_path) if frozen_path.exists() else None
    declared = {j["id"] for j in protocol["config"]["jobs"]}
    if frozen:
        if set(frozen.get("jobs", {})) != declared or frozen.get("protocol_sha256") != phash:
            problems.append("global freeze membership/protocol mismatch")
    elif final.get("status") == "complete":
        problems.append("completed run lacks group freeze")
    for job in protocol["config"]["jobs"]:
        folder = root/"jobs"/job["id"]
        for path in sorted(folder.glob("checkpoints/*.COMMITTED.json")):
            marker = read(path)
            target = _safe_path(folder, marker["checkpoint"])
            if not target.is_file() or sha256(target) != marker["sha256"]:
                problems.append("committed checkpoint hash mismatch: " + str(target))
            elif target.is_file():
                try:
                    verify_checkpoint(target, verified=verified_objects)
                except (ValueError, OSError) as error:
                    problems.append("checkpoint objects invalid: " + str(error))
        local = folder/"FROZEN.json"
        if local.exists():
            marker = read(local)
            target = _safe_path(folder, marker["checkpoint"])
            if not target.is_file() or sha256(target) != marker["sha256"]:
                problems.append("selected checkpoint hash mismatch: " + job["id"])
            if sha256(folder/"protocol.json") != marker["protocol_sha256"]:
                problems.append("job protocol hash mismatch: " + job["id"])
            if read(folder/"protocol.json").get("suite_protocol_sha256") != phash:
                problems.append("job parent protocol mismatch: " + job["id"])
            if frozen and frozen["jobs"].get(job["id"]) != sha256(local):
                problems.append("global/local freeze mismatch: " + job["id"])
            if frozen and marker["frozen_unix"] > frozen["created_unix"]:
                problems.append("job frozen after group barrier: " + job["id"])
        test_path = folder/"TEST_RESULTS.json"
        if test_path.exists():
            record = read(test_path)
            if not frozen or record.get("group_freeze_sha256") != sha256(frozen_path):
                problems.append("test group barrier hash mismatch: " + job["id"])
        elif final.get("status") == "complete":
            problems.append("completed run lacks test result: " + job["id"])
    for event in events:
        if event["kind"] == "test_started" and (not frozen or event["unix"] < frozen["created_unix"]):
            problems.append("test started before all model freezes")
    if (root/"WRITER.lock").exists():
        problems.append("writer lock remains after process inventory; diagnostic report only")
    for path in root.rglob("*.json"):
        if not any(part in ("replay",) for part in path.relative_to(root).parts):
            inventory[str(path.relative_to(root))] = sha256(path)
    return dict(passed=not problems, problems=problems, source_files=len(protocol.get("sources", {})),
                protocol_sha256=phash, events=len(events), global_freeze=frozen,
                hashes=inventory), events


def _gpu(root, started, finished):
    rows = sorted(_jsonl(root/"GPU_SAMPLES.jsonl"), key=lambda r: r["unix"])
    devices = []
    for device in range(max([len(r.get("gpus", [])) for r in rows]+[0])):
        total = weighted = 0.
        values = []
        for i, row in enumerate(rows):
            following = rows[i+1]["unix"] if i+1 < len(rows) else finished
            width = max(0., min(following, finished)-max(row["unix"], started))
            if device < len(row.get("gpus", [])):
                value = row["gpus"][device]["utilization_percent"]
                if _number(value):
                    values.append(value); total += width; weighted += width*value
        devices.append(dict(device=device, sample_mean_percent=_mean(values),
            time_weighted_percent=weighted/total if total else None,
            covered_seconds=total, coverage_fraction=total/max(finished-started, 1e-9),
            valid_samples=len(values)))
    return dict(devices=devices, total_samples=len(rows), error_samples=sum("error" in r for r in rows),
                scope="full suite wall clock, including CPU stages; whole-card samples, not process-exclusive", rows=rows)


def _curves(folder, backend, stages):
    training, probes, counts = [], [], {}
    if backend == "rgb_legacy":
        training = _jsonl(folder/"rgb_updates.jsonl")
        probes = _jsonl(folder/"rgb_step_probes.jsonl")
        counts = training[-1].get("counts", {}) if training else {}
        for i, row in enumerate(training): row["actual_step"] = i+1
        if counts:
            counts = dict(phase_counts=counts, optimizer_updates=dict(
                ae=counts.get("warmup", 0)+counts.get("representation", 0),
                shadow_ae=counts.get("warmup", 0)+counts.get("representation", 0),
                derivative_critic=counts.get("critic_prefit", 0)+counts.get("representation", 0),
                M=counts.get("metric", 0)))
    elif backend == "push_discrete_sac":
        probes = _jsonl(folder/"push_fixed_step_probes.jsonl")
        for stage in stages:
            result = stage["result"]
            if "counts" in result: counts = result["counts"]
            if stage["stage"].startswith("train_epoch"):
                training.append(dict(actual_step=result.get("counts", {}).get("certificate_updates"),
                    **result.get("actor_statistics", {}), **result.get("q_statistics", {})))
    else:
        for stage in stages:
            result = stage["result"]
            if stage["stage"] == "initialize" and "fixed_training_probe_M" in result:
                probes.append(dict(step=0, matrix=result["fixed_training_probe_M"]))
            if stage["stage"].startswith("train_epoch"):
                training.extend(dict(actual_step=r["update"], **r) for r in result.get("rows", []))
                counts = dict(certificate_updates=result.get("actual_updates", 0),
                    representation_updates=0, evaluator_updates=0,
                    model_transitions=result.get("model_transitions"))
        for row in training:
            matrix = row.get("fixed_training_probe_M", row.get("fixed_public_estimate_M"))
            if matrix is not None: probes.append(dict(step=row["update"], matrix=matrix))
    statistics = []
    for probe in probes:
        matrix = np.asarray(probe.get("matrix", probe.get("M")), dtype=float)
        if matrix.ndim < 2: continue
        matrices = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1])
        eig = np.linalg.eigvalsh(matrices)
        statistics.append(dict(step=probe.get("stage_update", probe.get("step")),
            min_eigenvalue=float(eig.min()), max_eigenvalue=float(eig.max()),
            mean_trace=float(np.trace(matrices, axis1=-2, axis2=-1).mean()),
            mean_frobenius=float(np.linalg.norm(matrices, axis=(-2, -1)).mean())))
    return dict(actual_counts=counts, training=training, fixed_probes=probes, probe_statistics=statistics,
                M_curve_available=bool(statistics), M_curve_scope="fixed training inputs, actual update index; no monotonicity assumption",
                rollback_events=[s for s in stages if s["stage"].startswith("validate") and s["result"].get("candidate_validation", {}).get("rollback")])


def _evaluate(folder, job, test):
    backend = job["backend"]
    groups, pairs, errors, numeric = {}, {}, [], []
    if backend == "rgb_legacy":
        for name, summary in test.get("summary", {}).get("groups", {}).items():
            plant, variant = name.split("/", 1)
            rows = [r for r in test["rows"] if r["plant"] == plant and r["variant"] == variant]
            groups[name] = dict(summary, mean_terminal_error=_mean([r.get("final_error") for r in rows]),
                mean_V_rise_fraction=_mean([r.get("V_rise_fraction") for r in rows]),
                mean_V_lower_violation=_mean([r.get("V_lower_violation") for r in rows]),
                median_episode_V_over_E=float(np.median([r["V_over_E_median"] for r in rows])))
        for row in test.get("rows", []):
            path = _safe_path(folder, row["trajectory"])
            if sha256(path) != row["trajectory_sha256"]: errors.append("RGB trace hash mismatch")
            with np.load(path, allow_pickle=False) as z:
                e, m = z["error"].astype(float), z["M"].astype(float)
                v = .5*np.einsum("ni,ij,nj->n", e, m, e)
                numeric.append(float(np.max(np.abs(v-z["V"]))))
                if not np.allclose(v, z["V"], atol=2e-5, rtol=1e-5): errors.append("RGB complete V differs")
            if not row.get("audit", {}).get("passed"): errors.append("RGB action/hold backend audit failed")
        selected = {(r["plant"], r["case_id"]): r for r in test.get("rows", []) if r["variant"] == "selected"}
        for name in groups:
            plant, variant = name.split("/", 1)
            if variant == "selected": continue
            matched = [(selected[(plant, r["case_id"])], r) for r in test["rows"]
                       if r["plant"] == plant and r["variant"] == variant and r["success"]
                       and selected[(plant, r["case_id"])]["success"]]
            pairs[name] = dict(common_successes=len(matched), selected_minus_baseline_seconds=
                _mean([a["elapsed_s"]-b["elapsed_s"] for a,b in matched]))
    elif backend == "push_discrete_sac":
        for name, value in test.get("variants", {}).items():
            groups[name] = _compact(value)
            groups[name]["episodes"] = value.get("num_episodes")
            for episode in value.get("episodes", []):
                for t in episode.get("trace", []):
                    for suffix, vkey in (("", "V"), ("_next", "V_next")):
                        e, m = np.asarray(t["error"+suffix]), np.asarray(t["M"+suffix])
                        difference = abs(float(e@m@e)-t[vkey]); numeric.append(difference)
                        if difference > 2e-5+1e-5*abs(t[vkey]): errors.append("Push successor V differs")
        pairs = test.get("paired_time_on_common_success", {})
    else:
        for name, value in test.get("groups", {}).items():
            groups[name] = value["summary"]
            if not test.get("audit", {}).get(name, {}).get("passed"): errors.append("backend independent audit failed: "+name)
            for path in (folder/"trajectories"/"test"/name).glob("*.npz"):
                with np.load(path, allow_pickle=False) as z:
                    if backend == "observer_model_gradient":
                        e, m = z["normalized_error"], z["matrix"]
                        v = np.einsum("...i,...ij,...j->...", e, m, e)
                        difference = float(np.max(np.abs(v-z["value"]))); numeric.append(difference)
                        if difference > 1e-7: errors.append("Observer full V differs")
                    else:
                        cfg = job.get("config", {}); task = cfg.get("task", "point_mass")
                        for key, mkey, vkey in (("states", "M_current", "V_before"), ("following", "M_next", "V_after")):
                            state = z[key].astype(float)
                            goal = np.asarray(cfg.get("physical", {}).get("goal", np.zeros(state.shape[-1])))
                            e = state-goal
                            if task in ("pendulum", "cartpole"):
                                axis = 0 if task == "pendulum" else 2
                                e[..., axis] = np.arctan2(np.sin(state[..., axis]), np.cos(state[..., axis]))
                            v = np.einsum("...i,...ij,...j->...", e, z[mkey], e)
                            difference = float(np.max(np.abs(v-z[vkey]))); numeric.append(difference)
                            if difference > 1e-7: errors.append("State full successor V differs")
        pairs = test.get("paired_time", {})
    return dict(groups=groups, paired_time=pairs,
                postprocess_numeric_audit=dict(passed=bool(numeric) and not errors,
                    checked_values_or_trajectories=len(numeric), maximum_V_difference=max(numeric) if numeric else None,
                    errors=sorted(set(errors)), scope="independent NumPy quadratic arithmetic at both recorded endpoints; backend audits separately verify networks/actions"))


def _plots(output, jobs, gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = {}
    folder = output/"figures"; folder.mkdir()
    for name, job in jobs.items():
        curves = job["curves"]; rows = curves["training"]; probes = curves["probe_statistics"]
        figure, axes = plt.subplots(1, 2, figsize=(11, 3.7))
        for key in ("loss", "total", "actor_loss", "q_loss", "physical_training_cost"):
            points = [(r["actual_step"], r[key]) for r in rows if _number(r.get("actual_step")) and _number(r.get(key))]
            if points: axes[0].plot(*zip(*points), label=key)
        for key in ("min_eigenvalue", "max_eigenvalue", "mean_trace"):
            points = [(r["step"], r[key]) for r in probes if _number(r.get("step"))]
            if points: axes[1].plot(*zip(*points), label=key)
        for axis in axes:
            axis.set_xlabel("Actual update step"); axis.grid(alpha=.25)
            if axis.lines: axis.legend(fontsize=7)
            else: axis.text(.5, .5, "No recorded curve", ha="center", transform=axis.transAxes)
        axes[0].set_title("Recorded training objectives")
        axes[1].set_title("M on fixed training probes")
        figure.suptitle(name); figure.tight_layout()
        target=folder/(name+".png"); figure.savefig(target,dpi=140); plt.close(figure)
        result[name]=str(target.relative_to(output))
    if gpu["rows"]:
        figure, axis=plt.subplots(figsize=(11,3))
        origin=gpu["rows"][0]["unix"]
        for device in gpu["devices"]:
            i=device["device"]
            points=[((r["unix"]-origin)/60,r["gpus"][i]["utilization_percent"]) for r in gpu["rows"] if len(r.get("gpus",[]))>i]
            if points: axis.plot(*zip(*points), label=f"GPU {i}")
        axis.set(xlabel="Full-run elapsed minutes",ylabel="GPU utilization (%)",ylim=(0,105));axis.grid(alpha=.25)
        if axis.lines:axis.legend()
        figure.tight_layout();figure.savefig(folder/"gpu.png",dpi=140);plt.close(figure)
        result["gpu"]="figures/gpu.png"
    return result


def _html(data, figures):
    def table(mapping):
        return "<table><tr><th>项目</th><th>实测/记录</th></tr>"+"".join(
            "<tr><td>"+escape(str(k))+"</td><td><pre>"+escape(json.dumps(v,ensure_ascii=False,indent=2))+"</pre></td></tr>" for k,v in mapping.items())+"</table>"
    text = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>统一训练架构与有界实测</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#182334}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccd4df;padding:10px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace;margin:0}img{max-width:100%}.note{padding:16px;background:#eef3fa}a{color:#174f8f}</style><body>",
        "<h1>统一训练架构与有界实测</h1>",
        "<p class='note'>本轮状态："+escape(data["run_status"])+"。这是一次统一调度与真实更新的有界集成实验；不能由接口接通推断原有性能无损，也不能把不同起点、预算的历史成功率当作因果对照。</p>",
        "<p>统一流程：初始化与来源冻结 → 新采集 → 可选表示预热 → 可选 Critic 预训练 → 多轮采集及任务专属更新 → 验证选模 → 全组冻结 → 独立留出测试与数值复算。环境执行物理过程，TaskAdapter 构造任务控制，算法持有网络与优化器，Runner 管理阶段和证据。</p>",
        table(dict(运行目录=data["root"],墙钟秒=data["wall_seconds"],完整性通过=data["integrity"]["passed"],完整性问题=data["integrity"]["problems"],GPU=data["gpu_summary"])),
        "<p><a href='REPORT.json'>完整机器可读报告与来源哈希</a> · <a href='CURVES.json'>全部真实更新与固定 M 探针</a></p>"]
    if "gpu" in figures:text.append("<img src='"+figures["gpu"]+"' alt='全程 GPU 利用率'>")
    for name, job in data["jobs"].items():
        text += ["<h2>"+escape(name)+"</h2>",table(dict(后端=job["backend"],实际更新=job["curves"]["actual_counts"],选模=job.get("frozen"),来源与初始化=job.get("initialization"))),
                 "<p>以下结果按任务自身判据统计；输入平方是速度或估计修正的代理，不能跨任务当作能量比较。共同成功起点之外不比较成功到达时间。</p>",
                 table(job.get("evaluation",{})),"<img src='"+figures[name]+"' alt='真实更新曲线和固定探针 M 曲线'>",
                 "<p>保留所有 M 变化、损失回升及验证回滚。M 正定、loss 下降或梯度非零均不构成物理稳定性证明。</p>"]
    text += ["<h2>解释范围</h2><ul>"+"".join("<li>"+escape(item)+"</li>" for item in data["limitations"])+"</ul></body></html>"]
    return "".join(text)


def report_suite(root, output, process_snapshot):
    """Produce a new report only after external process verification succeeds."""
    gate=check_process_snapshot(process_snapshot)
    root,output=Path(root).resolve(),Path(output).resolve()
    # A lock is conservative evidence of possible ownership. Even a stale
    # lock must be diagnosed separately; reporting never opens mutable state.
    if (root/"WRITER.lock").exists():
        raise RuntimeError("Writer lock remains; refuse to read run artifacts")
    if output.exists():raise FileExistsError("Report outputs are immutable: "+str(output))
    protocol=read(root/"protocol.json")
    final=read(root/"FINAL_STATE.json") if (root/"FINAL_STATE.json").exists() else dict(status="incomplete_without_final_state")
    integrity,events=_integrity(root,protocol,final)
    started=protocol["started_unix"];finished=final.get("finished_unix",max([e["unix"] for e in events]+[started]))
    gpu=_gpu(root,started,finished);jobs={}
    for job in protocol["config"]["jobs"]:
        folder=root/"jobs"/job["id"]
        stages=sorted([read(p) for p in (folder/"stages").glob("*.json")],key=lambda s:s["started_unix"])
        item=dict(backend=job["backend"],declared_schedule=job["schedule"],curves=_curves(folder,job["backend"],stages),
                  initialization=next((s["result"] for s in stages if s["stage"]=="initialize"),None),
                  stages_completed=[s["stage"] for s in stages],
                  frozen=read(folder/"FROZEN.json") if (folder/"FROZEN.json").exists() else None)
        if (folder/"TEST_RESULTS.json").exists():
            item["evaluation"]=_evaluate(folder,job,read(folder/"TEST_RESULTS.json")["result"])
        else:item["evaluation"]=dict(status="not completed; no result fabricated")
        jobs[job["id"]]=item
    output.mkdir(parents=True,exist_ok=False)
    figures=_plots(output,jobs,gpu)
    limitations=[
        "单开发种子和有界更新预算；该轮未完成对历史完整训练的同预算、多种子性能等价证明。",
        "RGB 保留书稿目标处 J、误差与 J 同除以 4、原 ridge、导数 Critic 与梯度预算及 shadow AE；GPU 渲染像素与历史渲染不完全等价。",
        "RGB 使用有来源审计的历史 initial AE/M 与新 Critic、空优化器和新采集；不能称所有感知模块随机从零。公共 FastDepth、既有房间及历史 AE 预训练来源须以初始化记录为准。",
        "RGB P0/P1 是 4DOF 速度外环与延迟/滤波工况；推理墙钟未加入仿真 dt，不称实时或真机定位。",
        "Push 新离散 SAC 是 M 诱导候选策略的独立算法对照，未替换旧 PPO 基线；使用既有标定响应与时间银行，不等同 SO101 关节驱动 MuJoCo 或真机。",
        "状态控制保留完整时域模型梯度与原可行性门槛/回滚；观测保留公开传感序列与估计状态，真值仅用于训练监督和评价。二者无需虚构 AE 或价值 Critic。",
        "identity 与 selected 共享对应已选特征；initial 同时包含初始化与未预热差异。更新数按真实计数报告，表示更新内的 Critic/shadow 更新单独列出。",
        "数值复算只验证给定记录端点的计算及冻结来源；M 正定和采样下降都不提供新增全局或非线性域稳定证明。",
        "GPU 统计覆盖整个本轮墙钟和整卡采样，缺失时段单列，不用最好窗口代替全程；不同任务的成本不可直接平均比较。"]
    data=dict(schema="unified_training_report_v1",root=str(root),run_status=final["status"],final_state=final,
        wall_seconds=finished-started,process_gate=gate,integrity=integrity,jobs=jobs,
        gpu_summary={k:v for k,v in gpu.items() if k!="rows"},limitations=limitations,
        numerical_audit_passed=all(j.get("evaluation",{}).get("postprocess_numeric_audit",{}).get("passed",False) for j in jobs.values()),
        performance_equivalence_proven=False)
    write_new(output/"CURVES.json",dict(jobs={k:v["curves"] for k,v in jobs.items()},gpu_samples=gpu["rows"]))
    write_new(output/"REPORT.json",data)
    (output/"REPORT.html").write_text(_html(data,figures),encoding="utf-8")
    return dict(output=str(output),html=str(output/"REPORT.html"),status=final["status"],
                integrity_passed=integrity["passed"],numerical_audit_passed=data["numerical_audit_passed"],
                performance_equivalence_proven=False)
