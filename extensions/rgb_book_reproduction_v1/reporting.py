"""Compact post-writer report with independent endpoint and control audits."""
from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
import json
import math

import numpy as np

from stabilization_learning.unified_v1.runner import read, sha256, write_new
from stabilization_learning.unified_v1.reporting import check_process_snapshot, _integrity, _gpu


def json_rows(path):
    if path.exists():
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def scalars(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(scalars(item, name + "."))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[name] = item
    return result


def curves(folder):
    updates = [dict(actual_step=i+1, phase=row.get("phase"), **scalars(row))
               for i, row in enumerate(json_rows(folder / "rgb_updates.jsonl"))]
    probes = []
    for i, row in enumerate(json_rows(folder / "rgb_step_probes.jsonl")):
        matrices = np.asarray(row.get("matrix", row.get("M")), dtype=float)
        if matrices.ndim < 2:
            continue
        matrices = matrices.reshape(-1, matrices.shape[-2], matrices.shape[-1])
        eigenvalues = np.linalg.eigvalsh(matrices)
        probes.append(dict(actual_probe=i+1, phase=row.get("phase"),
            stage_update=row.get("stage_update", row.get("step")),
            min_eigenvalue=float(eigenvalues.min()), max_eigenvalue=float(eigenvalues.max()),
            mean_trace=float(np.trace(matrices, axis1=-2, axis2=-1).mean()),
            mean_frobenius=float(np.linalg.norm(matrices, axis=(-2, -1)).mean()),
            maximum_condition=float((eigenvalues[:, -1] / eigenvalues[:, 0]).max()),
            **{k: v for k, v in scalars(row).items() if k.startswith(("phase_counts.", "optimizer_counts."))}))
    validation = []
    for path in sorted((folder / "stages").glob("validate_*.json")):
        row = read(path)
        epoch = int(path.stem.rsplit("_", 1)[-1]) if path.stem.startswith("validate_epoch_") else 0
        validation.append(dict(stage=row["stage"], epoch=epoch,
            selection_eligible=path.stem.startswith("validate_epoch_"),
            selection_score=row["result"].get("selection_score"),
            summary=row["result"].get("summary"), folder=row["result"].get("folder")))
    validations = sorted(validation, key=lambda row: (row["epoch"], row["stage"]))
    return dict(training_updates=updates, matrix_probes=probes, validations=validations,
        raw_training=str(folder / "rgb_updates.jsonl"), raw_matrix_probes=str(folder / "rgb_step_probes.jsonl"),
        update_count=len(updates), probe_count=len(probes),
        last_counts={k: v for k, v in updates[-1].items() if k.startswith("counts.")} if updates else {})


def trajectory_audit(path, expected, row):
    if sha256(path) != expected:
        raise ValueError("Trajectory hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        e, matrix, interaction = [data[k].astype(np.float64) for k in ("error", "M", "J")]
        value = .5 * np.einsum("ni,ij,nj->n", e, matrix, e)
        if not np.allclose(value, data["V"], atol=2e-5, rtol=1e-5):
            raise ValueError("Complete endpoint V recomputation mismatch")
        gram = interaction.T @ matrix @ interaction
        ridge = max(1.01 * np.trace(gram) / 9999, 1e-8)
        hessian = gram + .01 * np.diag(np.diag(gram)) + ridge * np.eye(4)
        raw = -.7 * np.linalg.solve(hessian, interaction.T @ matrix @ e[:-1].T).T
        action = raw / np.maximum(1., np.maximum(np.linalg.norm(raw[:, :3], axis=1), abs(raw[:, 3])))[:, None]
        if not np.allclose(action, data["action"], atol=5e-4, rtol=1e-4):
            raise ValueError("Independent visual-servo action mismatch")
        offset = data["offset"].astype(float)
        physical = np.sum(offset[:, :3]**2, axis=1) + (offset[:, 3]/math.radians(25))**2
        if not np.allclose(physical[1:] * data["dt"], data["integrated_error"], atol=1e-6, rtol=1e-6):
            raise ValueError("Physical integrated-error mismatch")
        good = (np.linalg.norm(offset[1:, :3], axis=1) < .02) & (np.abs(offset[1:, 3]) < math.radians(1))
        hold, holds = 0, []
        for yes in good:
            hold = hold+1 if yes else 0
            holds.append(hold)
        if not np.array_equal(holds, data["hold"]):
            raise ValueError("Success hold counter mismatch")
        if row["success"] and (not holds or holds[-1] < 11):
            raise ValueError("Reported success lacks 11 consecutive in-tolerance observations")
        if data["terminated"][:-1].any() or data["truncated"][:-1].any():
            raise ValueError("Recorded transition after termination")
        if not row.get("audit", {}).get("passed"):
            raise ValueError("Backend trajectory audit did not pass")
        return dict(endpoints=len(value), maximum_V_difference=float(np.max(abs(value-data["V"]))),
            maximum_action_difference=float(np.max(abs(action-data["action"]))) if len(action) else 0.)


def paired(rows):
    result = []
    for plant in sorted({r["plant"] for r in rows}):
        selected = {r["case_id"]: r for r in rows if r["plant"] == plant and r["variant"] == "selected"}
        if not selected:
            continue
        for variant in sorted({r["variant"] for r in rows if r["plant"] == plant} - {"selected"}):
            baseline = {r["case_id"]: r for r in rows if r["plant"] == plant and r["variant"] == variant}
            if baseline.keys() != selected.keys():
                result.append(dict(plant=plant, baseline=variant, paired=False, reason="different or incomplete start lists"))
                continue
            common = [key for key in baseline if baseline[key]["success"] and selected[key]["success"]]
            result.append(dict(plant=plant, baseline=variant, paired=True, episodes=len(selected),
                gained=sum(selected[k]["success"] and not baseline[k]["success"] for k in selected),
                lost=sum(baseline[k]["success"] and not selected[k]["success"] for k in selected),
                common_successes=len(common), selected_minus_baseline_seconds=float(np.mean([
                    selected[k]["elapsed_s"]-baseline[k]["elapsed_s"] for k in common])) if common else None))
    return result


def evaluations(folder):
    suites, problems = [], []
    endpoints, trajectories, vmax, amax = 0, 0, 0., 0.
    for directory in sorted(folder.glob("RGB_EVAL_*")):
        result_path = directory / "RESULTS.json"
        complete = result_path.exists()
        if not complete:
            result_path = directory / "PARTIAL.json"
        if not result_path.exists():
            suites.append(dict(folder=str(directory), complete=False, recorded_episodes=0))
            continue
        result = read(result_path)
        rows = result.get("rows", [])
        for row in rows:
            path = Path(row["trajectory"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                problems.append("Trajectory path outside job: " + str(path))
                continue
            try:
                audit = trajectory_audit(path, row["trajectory_sha256"], row)
                endpoints += audit["endpoints"]
                trajectories += 1
                vmax = max(vmax, audit["maximum_V_difference"])
                amax = max(amax, audit["maximum_action_difference"])
            except Exception as error:
                problems.append(str(path) + ": " + str(error))
        suites.append(dict(split=result.get("split", directory.name), folder=str(directory),
            complete=complete, recorded_episodes=len(rows), summary=result.get("summary", {}),
            paired=paired(rows), failure_counts=dict(Counter(r["reason"] for r in rows if not r["success"]))))
    return dict(suites=suites, audit=dict(passed=bool(trajectories) and not problems,
        checked_trajectories=trajectories, checked_endpoints=endpoints, maximum_V_difference=vmax,
        maximum_action_difference=amax, problems=problems,
        tolerances=dict(V_atol=2e-5, V_rtol=1e-5, action_atol=5e-4, action_rtol=1e-4),
        scope="All recorded endpoints use the same trajectory M; actions, physical costs and hold counters are independently recomputed"))


def plots(output, learning, gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = output / "figures"
    destination.mkdir()
    paths = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for phase in ("warmup", "critic_prefit", "representation", "metric"):
        rows = [r for r in learning["training_updates"] if r["phase"] == phase]
        keys = [k for k in ("loss", "total", "critic_loss", "metric_loss", "calibration", "calibration_loss") if any(k in r for r in rows)]
        if not keys:
            keys = sorted({k for r in rows for k in r if "loss" in k and "." not in k})[:2]
        for key in keys[:2]:
            points = [(r["actual_step"], r[key]) for r in rows if key in r]
            if points:
                axes[0].plot(*zip(*points), label=phase+":"+key, linewidth=.7)
    for key in ("min_eigenvalue", "max_eigenvalue", "mean_trace"):
        points = [(r["actual_probe"], r[key]) for r in learning["matrix_probes"]]
        if points:
            axes[1].plot(*zip(*points), label=key)
    for axis in axes:
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=6)
    axes[0].set(xlabel="Actual update", title="Recorded objectives")
    axes[1].set(xlabel="Fixed training probe index", title="M spectrum / trace")
    figure.tight_layout(); figure.savefig(destination/"learning.png", dpi=140); plt.close(figure)
    paths.append("figures/learning.png")
    figure, axis = plt.subplots(figsize=(8, 4))
    for plant in ("P0", "P1"):
        points = [(r["epoch"], r["summary"]["groups"][plant+"/selected"]["successes"])
            for r in learning["validations"] if r["selection_eligible"] and r.get("summary") and plant+"/selected" in r["summary"].get("groups", {})]
        if points:
            axis.plot(*zip(*points), "o-", label=plant)
    axis.set(xlabel="Training epoch", ylabel="Successful validation starts", title="Raw epoch validation (no smoothing)")
    axis.grid(alpha=.25)
    if axis.lines:
        axis.legend()
    figure.tight_layout(); figure.savefig(destination/"validation.png", dpi=140); plt.close(figure)
    paths.append("figures/validation.png")
    if gpu["rows"]:
        figure, axis = plt.subplots(figsize=(10, 3))
        origin = gpu["rows"][0]["unix"]
        for device in gpu["devices"]:
            index = device["device"]
            points = [((r["unix"]-origin)/60, r["gpus"][index]["utilization_percent"])
                for r in gpu["rows"] if len(r.get("gpus", [])) > index]
            if points:
                axis.plot(*zip(*points), label=f"GPU {index}")
        axis.set(xlabel="Full-run minutes", ylabel="GPU utilization (%)", ylim=(0, 105))
        axis.grid(alpha=.25)
        figure.tight_layout(); figure.savefig(destination/"gpu.png", dpi=140); plt.close(figure)
        paths.append("figures/gpu.png")
    return paths


def report_suite(root, output, snapshot):
    gate = check_process_snapshot(snapshot)
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Report must be written outside the frozen run")
    if output.exists():
        raise FileExistsError("Use a fresh report directory")
    protocol, final = read(root / "protocol.json"), read(root / "FINAL_STATE.json")
    integrity, _ = _integrity(root, protocol, final)
    job = protocol["config"]["jobs"][0]
    folder = root / "jobs" / job["id"]
    learning, evaluation = curves(folder), evaluations(folder)
    gpu = _gpu(root, protocol["started_unix"], final["finished_unix"])
    frozen = read(folder / "FROZEN.json") if (folder / "FROZEN.json").exists() else None
    output.mkdir(parents=True, exist_ok=False)
    figures = plots(output, learning, gpu)
    write_new(output / "CURVES.json", learning)
    write_new(output / "EVALUATION_AUDIT.json", evaluation)
    write_new(output / "INTEGRITY.json", integrity)
    write_new(output / "GPU.json", gpu)
    data = dict(schema="rgb_book_report_v1", root=str(root), status=final["status"], final=final,
        writer_check=gate, integrity_passed=integrity["passed"], evaluation_audit=evaluation["audit"],
        frozen=frozen, schedule=job["schedule"], mode=protocol["config"].get("book_protocol", {}).get("mode", "formal"),
        GPU={k: v for k, v in gpu.items() if k != "rows"}, suites=evaluation["suites"],
        figures=figures, curve_files={"training_and_M":"CURVES.json", "raw_updates":learning["raw_training"], "raw_M":learning["raw_matrix_probes"]})
    write_new(output / "REPORT.json", data)
    lines = ["# RGB书稿流程复刻：训练与评价", "", f"运行状态：**{final['status']}**。",
        f"运行目录：`{root}`。实验模式：`{data['mode']}`。",
        "", "初始化 → 历史训练回放 → AE预热 → 导数Critic预训练 → 采样/表示更新/M更新 → P0验证选模 → 全部冻结 → 实验评价。",
        "", "主训练只在预先声明的轮次按验证集选模；initial和ready为诊断。训练目标、验证成功率与正式测试分别记录。",
        "", f"实际记录更新数：{learning['update_count']}；固定训练输入M探针数：{learning['probe_count']}。",
        f"选择轮次：{frozen['selected_epoch'] if frozen else '尚未冻结'}。",
        f"独立复算：{evaluation['audit']['checked_trajectories']}条轨迹、{evaluation['audit']['checked_endpoints']}个端点，最大V误差{evaluation['audit']['maximum_V_difference']:.6g}。",
        f"来源完整性通过：{integrity['passed']}；轨迹复算通过：{evaluation['audit']['passed']}。", "",
        "| 评价套件 | 工况/模型 | 成功/总数 | 平均累计物理误差 |", "|---|---|---:|---:|"]
    for suite in evaluation["suites"]:
        for name, group in suite.get("summary", {}).get("groups", {}).items():
            lines.append(f"| {suite['split']} | {name} | {group['successes']}/{group['episodes']} | {group['mean_integrated_error']:.6g} |")
    lines += ["", "配对增减例数及共同成功起点到达时间见REPORT.json；全部失败原因与逐条复算诊断见EVALUATION_AUDIT.json。",
        "", "原始各轮结果全部保留，最终选模不替代训练曲线。书稿360测试、SL72测试、训练场景诊断和48起点验证分别统计。",
        "", "本次为单seed41；迁移后的重建像素、既有房间、FastDepth和历史AE初始化沿用声明来源，网络训练使用CUDA。4DOF目标处J与P1延迟按配置执行，计算墙钟未加入dt，输入平方积分为速度代理量。稳定性结论限于已完成的数值与闭环实验。"]
    if final["status"] != "complete":
        lines += ["", "本次未完成整个冻结与实验流程；仅报告已落盘结果。停止/错误：" + final.get("error", "见FINAL_STATE.json")]
    for figure in figures:
        lines += ["", f"![{Path(figure).stem}]({figure})"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>RGB书稿流程复刻</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.7 system-ui}pre{white-space:pre-wrap}img{max-width:100%}</style><body>"
    html += "<pre>" + escape("\n".join(line for line in lines if not line.startswith("!["))) + "</pre>"
    html += "".join("<img src='"+escape(path)+"'>" for path in figures) + "</body></html>"
    (output / "REPORT.html").write_text(html, encoding="utf-8")
    return dict(output=str(output), status=final["status"], integrity_passed=integrity["passed"],
        trajectory_audit_passed=evaluation["audit"]["passed"], checked_trajectories=evaluation["audit"]["checked_trajectories"])
