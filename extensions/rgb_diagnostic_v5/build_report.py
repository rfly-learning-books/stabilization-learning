"""Read completed diagnostic artifacts and create a new, source-linked report.

No training or simulation is performed. A fresh external process snapshot is
required; a report directory is never reused. Scientific claims are supplied
in a separately reviewed narrative rather than inferred from loss reduction.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import time
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def verify_snapshot(path):
    snapshot = read(path)
    if not snapshot.get("query_success") or not 0 <= time.time() - snapshot["created_unix"] <= 60:
        raise RuntimeError("A successful external process query within 60 seconds is required")
    for process in snapshot["processes"]:
        command = process.get("CommandLine", process.get("command_line"))
        if not command:
            raise RuntimeError("Unidentified Python process")
        if "meetings_mcp_entrypoint.py" not in command:
            raise RuntimeError("Python process requires explicit identification before analysis: " + command)
    return snapshot


def report(root, output, snapshot, narrative):
    verify_snapshot(snapshot)
    root, output = Path(root).resolve(), Path(output).resolve()
    if (root / "GPU_LEASE.json").exists():
        raise RuntimeError("GPU writer lease still exists")
    if output.exists():
        raise FileExistsError(output)
    story = read(narrative)
    required = ["PROTOCOL.json", "control/RESULTS.json", "control/CONTROL_SCIENTIFIC_AUDIT.json",
                "derivative_geometry_v2/SUMMARY.json", "derivative_geometry_v2/ANALYSIS.json",
                "derivative_quantization/SUMMARY.json", "gradient_audit/GRADIENT_AUDIT.json",
                "TRAINING_AUTHORIZATION.json", "TRAINING_GROUP_FROZEN.json",
                "TRAINING_SUITE_COMPLETE.json", "TRAINING_INDEPENDENT_AUDIT.json",
                "FOLLOWUP_TRAJECTORY_AUDIT.json"]
    required += story.get("additional_required_artifacts", [])
    data = {name: read(root / name) for name in required}
    if data["TRAINING_SUITE_COMPLETE.json"]["completed_evaluation_episodes"] != 768:
        raise RuntimeError("Incomplete training comparison")
    if not data["control/CONTROL_SCIENTIFIC_AUDIT.json"]["passed"]:
        raise RuntimeError("Control numerical audit failed")
    if not data["TRAINING_INDEPENDENT_AUDIT.json"]["passed"]:
        raise RuntimeError("Training provenance/parameter audit failed")
    followup = data["FOLLOWUP_TRAJECTORY_AUDIT.json"]
    if (not followup["passed"] or followup["counts"]["training_episodes"] != 768
            or followup["counts"]["legacy_episodes"] != 480):
        raise RuntimeError("Incomplete or failed independent trajectory audit")
    if not data["gradient_audit/GRADIENT_AUDIT.json"]["parameters_unchanged"]:
        raise RuntimeError("Gradient-only audit changed weights")
    output.mkdir(parents=True)
    figures = output / "figures"
    figures.mkdir()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    sources = {str(root / name): sha(root / name) for name in required}
    sources.update({str(Path(snapshot).resolve()): sha(snapshot), str(Path(narrative).resolve()): sha(narrative),
                    str(Path(__file__).resolve()): sha(__file__)})
    gallery = []
    table_records = []
    episode_rows = []

    def figure_done(fig, name, title, caption):
        fig.tight_layout()
        fig.savefig(figures / (name + ".png"), dpi=170)
        fig.savefig(figures / (name + ".svg"))
        plt.close(fig)
        gallery.append(dict(name=name, title=title, caption=caption))

    def compare_chart(summaries, methods, name, title, labels=None):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        labels = labels or methods
        x = np.arange(len(methods))
        for i, plant in enumerate(("P0", "P1")):
            values = [summaries[plant + "/" + m] for m in methods]
            for col, key in enumerate(("successes", "mean_integrated_error", "mean_input_squared_integral")):
                ax[col].bar(x + (i - .5) * .36, [v[key] for v in values], .36,
                            label=plant, color=("#087f8c", "#dc8a2f")[i])
            for method, value in zip(methods, values):
                table_records.append(dict(experiment=name, plant=plant, method=method, **value))
        for a, ylabel in zip(ax, ("严格成功回合 / 48", "平均 ∫E dt（归一化误差·秒）", "平均 ∫‖u‖²dt（归一化命令·秒）")):
            a.set_xticks(x, labels, rotation=23, ha="right")
            a.set_ylabel(ylabel)
            a.grid(axis="y", alpha=.15)
        ax[0].set_ylim(0, 48)
        ax[0].legend()
        fig.suptitle(title)
        figure_done(fig, name, title, "全部 48 个开发验证起点、200 步；保留失败。误差积分和输入代理分别报告，不互相替代。")

    control = data["control/RESULTS.json"]
    for group in control["summaries"]:
        path = root / "control" / group / "RESULTS.json"
        sources[str(path)] = sha(path)
        rows = read(path)["rows"]
        if len(rows) != 48:
            raise RuntimeError("Incomplete control group")
        episode_rows.extend(dict(experiment="control", **row) for row in rows)
    compare_chart(control["summaries"], ["baseline", "target_J", "relative_damping", "identity_M", "FD_current_J"],
                  "control_comparison", "冻结 v4 模型：单因素控制对照",
                  ["原 v4", "目标 J", "尺度阻尼", "同特征 I", "渲染差分 J"])

    geometry = data["derivative_geometry_v2/SUMMARY.json"]
    save_json(output / "GEOMETRY_ALL_SUMMARIES.json", geometry)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for source, label, color in (("fast", "FastDepth", "#087f8c"), ("true", "仿真真深度", "#dc8a2f")):
        rows = [r for r in geometry["summary"] if r["source"] == source and r["stratum"] == "all"
                and r["kind"] == "feature_motion"]
        if rows:
            rows.sort(key=lambda r: r["size"])
            axes[0].plot([r["size"] * 400 for r in rows], [r["cosine"]["mean"] for r in rows], "o-", label=label, color=color)
            axes[1].plot([r["size"] * 400 for r in rows], [100 * r["opposite_fraction"] for r in rows], "o-", label=label, color=color)
    axes[0].set_ylabel("J 与实际特征差分的方向余弦均值")
    axes[1].set_ylabel("反向比例 %")
    for a in axes:
        a.set_xlabel("平移轴位移尺度 mm（偏航按同一 h）")
        a.grid(alpha=.15)
    axes[0].legend()
    figure_done(fig, "geometry_scale", "解析 J 的准确性依赖微位移尺度", "96 个训练诊断状态×4 轴；各幅值都保留。横轴为平移0.4h米，同组偏航为0.35h弧度；真深度仅用于诊断。")

    arms = data["PROTOCOL.json"]["conditional_training"]["arms"]
    training = {}
    all_updates = {}
    cross_comparisons = {}

    def paired(base, changed):
        b, c = ({r["case_id"]: r for r in rows} for rows in (base, changed))
        if set(b) != set(c) or len(b) != 48:
            raise RuntimeError("Cross-comparison requires all 48 identical starts")
        common = [key for key in b if b[key]["success"] and c[key]["success"]]
        diffs = [c[key]["elapsed_s"] - b[key]["elapsed_s"] for key in common]
        return dict(common_success_cases=common, common_success_count=len(common),
                    common_success_time_differences_s=diffs,
                    mean_common_success_time_difference_s=float(np.mean(diffs)) if diffs else None,
                    success_difference=sum(r["success"] for r in c.values())-sum(r["success"] for r in b.values()),
                    mean_integrated_error_ratio=float(np.mean([r["integrated_error"] for r in c.values()]) /
                                                      np.mean([r["integrated_error"] for r in b.values()])),
                    mean_input_squared_integral_ratio=float(np.mean([r["input_squared_integral"] for r in c.values()]) /
                                                            np.mean([r["input_squared_integral"] for r in b.values()])))

    for arm in arms:
        rel = "training_evaluation/" + arm + "/RESULTS.json"
        results = read(root / rel)
        sources[str(root / rel)] = sha(root / rel)
        training[arm] = results
        for plant in ("P0", "P1"):
            base_path = root / "control" / plant / "baseline/RESULTS.json"
            learned_path = root / "training_evaluation" / arm / plant / "baseline/RESULTS.json"
            identity_path = root / "training_evaluation" / arm / plant / "identity_M/RESULTS.json"
            for path in (base_path, learned_path, identity_path):
                sources[str(path)] = sha(path)
            baseline_rows, learned_rows, identity_rows = (read(p)["rows"] for p in (base_path, learned_path, identity_path))
            episode_rows.extend(dict(experiment="training", arm=arm, **row) for row in learned_rows + identity_rows)
            cross_comparisons[arm + "/" + plant] = dict(
                repair_vs_frozen_v4=paired(baseline_rows, learned_rows),
                learned_M_vs_shared_feature_identity=paired(identity_rows, learned_rows))
        for key, value in results["summaries"].items():
            plant, method = key.split("/")
            table_records.append(dict(experiment="training", arm=arm, plant=plant, method=method, **value))
        probe_path = root / "training" / arm / "STEP_PROBES.npz"
        sources[str(probe_path)] = sha(probe_path)
        with np.load(probe_path, allow_pickle=False) as probes:
            step, matrix = probes["step"], probes["M"].astype(float)
            if len(step) != 1201 or not np.array_equal(step, np.arange(1201)):
                raise RuntimeError("Incomplete fixed-probe module steps")
            value = probes["V"].astype(float)
            physical = probes["E"].astype(float)
            fig, ax = plt.subplots(2, 2, figsize=(12, 7))
            ax[0, 0].plot(step, np.linalg.norm(matrix - matrix[0], axis=(-2, -1)), lw=.8)
            ax[0, 0].set_ylabel("各固定探针 ‖M(step)−M(0)‖F")
            eig = np.linalg.eigvalsh(matrix)
            ax[0, 1].plot(step, eig[..., 0], lw=.7)
            ax[0, 1].set_ylabel("各固定探针 M 最小特征值")
            ax[1, 0].plot(step, value / np.maximum(physical, 1e-20), lw=.8)
            ax[1, 0].axhline(.1, color="black", ls="--", lw=.7)
            ax[1, 0].set_ylabel("各固定探针 V / E")
            ax[1, 1].plot(step, probes["lower_impossible"].mean(axis=-1), color="#ac3647")
            ax[1, 1].set_ylabel("固定探针中任意允许 M 均不可达下界的比例")
            for a in ax.flat:
                a.set_xlabel("本次修复新增模块更新（0–1200）")
                a.grid(alpha=.12)
            fig.suptitle(arm)
            figure_done(fig, "probes_" + arm, arm + "：实际更新与固定探针", "8 个固定旧训练探针；每个新增模块更新均记录，不平滑、不插值。φ 更新也可能通过目标特征改变 M 数值。")
        update_files = sorted((root / "training" / arm / "updates").glob("*.json"))
        updates = [read(p) for p in update_files]
        if [row["local_step"] for row in updates] != list(range(1, 1201)):
            raise RuntimeError("Missing actual module update logs")
        all_updates[arm] = dict(rows=updates, source_hashes={str(p): sha(p) for p in update_files})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for col, block in enumerate(("feature", "metric")):
            chosen = [row for row in updates if row["block"] == block]
            for key in ("physical", "calibration", "interaction", "decrease", "reconstruction", "geometry"):
                ax[col].plot([row["local_step"] for row in chosen], [row["metrics"][key] for row in chosen],
                             lw=.6, label=key)
            ax[col].set_yscale("symlog", linthresh=1e-9)
            ax[col].set_xlabel("本次修复新增模块更新")
            ax[col].set_title(block + " 块：全部 600 次更新")
            ax[col].set_ylabel("原始分项损失（未乘权重）")
            ax[col].grid(alpha=.12)
        ax[1].legend(fontsize=8)
        figure_done(fig, "loss_" + arm, arm + "：全部实际损失记录", "所有更新均绘制，无平滑或挑选窗口；损失数值不是梯度贡献，也不是闭环成功的替代指标。")
    save_json(output / "ALL_MODULE_UPDATES.json", all_updates)
    save_json(output / "PAIRED_TRAINING_COMPARISONS.json", cross_comparisons)

    legacy_rel = story.get("legacy_results")
    if legacy_rel:
        legacy = read(root / legacy_rel)
        if not legacy["audit"]["passed"] or legacy["audit"]["episodes"] != 480:
            raise RuntimeError("Historical transfer group audit incomplete")
        sources[str(root / legacy_rel)] = sha(root / legacy_rel)
        compare_chart(legacy["summaries"], story["legacy_methods"], "legacy_transfer", "原书稿组合与精确旧阻尼在当前环境的对照", story.get("legacy_labels"))
        for group in legacy["summaries"]:
            path = (root / legacy_rel).parent / group / "RESULTS.json"
            sources[str(path)] = sha(path)
            rows = read(path)["rows"]
            if len(rows) != 48:
                raise RuntimeError("Incomplete historical transfer group")
            episode_rows.extend(dict(experiment="legacy_transfer", **row) for row in rows)

    # Complete episode-level evidence and a deterministic full-step example.
    save_json(output / "COMPARISONS.json", dict(control=control, training=training,
              legacy=legacy if legacy_rel else None, group_rows=table_records, episode_rows=episode_rows,
              distinct_validation_starts=48, caveat="Repeated methods and plants do not add independent starting states or training seeds."))
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    example_sources = [("v4", root / "control/P0/baseline/case_0000.npz")]
    example_sources += [(arm, root / "training_evaluation" / arm / "P0/baseline/case_0000.npz") for arm in arms]
    for label, path in example_sources:
        sources[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as tr:
            n = len(tr["V"])
            t = np.concatenate(([0.], np.cumsum(tr["dt"].astype(float))))
            if len(t) != n:
                raise RuntimeError("Successor value and transition time lengths differ")
            V = .5 * np.einsum("ni,ij,nj->n", tr["error"].astype(float), tr["M"].astype(float), tr["error"].astype(float))
            E = tr["physical_error"].astype(float)
            ax[0, 0].plot(t, E, label=label, lw=.9)
            ax[0, 1].plot(t, V, lw=.9)
            ax[1, 0].plot(t, V / np.maximum(E, 1e-20), lw=.9)
            ax[1, 1].plot(t[:-1], np.diff(V), lw=.9)
    for a, label in zip(ax.flat, ("物理 E", "同参数完整 V", "V / E", "实际完整 V(t+1)−V(t)")):
        a.set_ylabel(label)
        a.set_xlabel("环境时间 s")
        a.grid(alpha=.15)
    ax[1, 1].axhline(0, color="black", lw=.7)
    ax[0, 0].legend(fontsize=7)
    figure_done(fig, "fixed_first_case", "固定展示：清单第一个 P0 验证起点", "按清单顺序选取，与结果无关；所有实际 step 和 V 回升均保留。不同模型的 V 数值不用于性能排名。全部起点的轨迹位于来源运行目录。")

    save_json(output / "SOURCE_MANIFEST.json", dict(created_unix=time.time(), root=str(root), sources=sources,
              scope="completed development diagnostics and short repairs; not held-out testing", narrative=str(narrative)))
    esc = lambda s: html.escape(str(s))
    sections = []
    for section in story["sections"]:
        chunk = "<section><h2>" + esc(section["title"]) + "</h2>"
        chunk += "".join("<p>" + esc(p) + "</p>" for p in section.get("paragraphs", []))
        if "table" in section:
            tab = section["table"]
            chunk += "<div class='scroll'><table><thead><tr>" + "".join("<th>" + esc(c) + "</th>" for c in tab["headers"]) + "</tr></thead><tbody>"
            chunk += "".join("<tr>" + "".join("<td>" + esc(c) + "</td>" for c in row) + "</tr>" for row in tab["rows"])
            chunk += "</tbody></table></div>"
        sections.append(chunk + "</section>")
    rows_html = "".join("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in [r["experiment"], r.get("arm", r["method"]), r["plant"], r["method"], f'{r["successes"]}/{r["episodes"]}', f'{r["mean_integrated_error"]:.5f}', f'{r["median_integrated_error"]:.5f}', f'{r["p90_integrated_error"]:.5f}', f'{r["mean_input_squared_integral"]:.5f}', json.dumps(r["reasons"], ensure_ascii=False)]) + "</tr>" for r in table_records)
    full_table = "<section><h2>全部组别结果</h2><div class='scroll'><table><thead><tr><th>实验</th><th>组</th><th>工况</th><th>M/控制</th><th>成功</th><th>平均累计 E</th><th>中位</th><th>P90</th><th>输入代理</th><th>终止原因</th></tr></thead><tbody>" + rows_html + "</tbody></table></div></section>"
    images = "".join("<section><h2>" + esc(g["title"]) + "</h2><img src='figures/" + g["name"] + ".png' alt='" + esc(g["title"]) + "'><p class='caption'>" + esc(g["caption"]) + " <a href='figures/" + g["name"] + ".svg'>矢量图</a></p></section>" for g in gallery)
    links = "".join("<li><a href='" + html.escape(Path(p).as_uri(), quote=True) + "'>" + esc(p) + "</a><code>" + digest + "</code></li>" for p, digest in sources.items())
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>RGB视觉伺服退化诊断</title><style>
body{font-family:'Microsoft YaHei',sans-serif;color:#182c35;background:#eef3f4;margin:0;line-height:1.8}main{max-width:1180px;margin:auto;padding:36px 24px}header,section{background:white;border-radius:12px;padding:28px;margin-bottom:20px}header{border-top:7px solid #087f8c}h1{font-size:30px;line-height:1.5}h2{font-size:21px;color:#126773}p{max-width:1000px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce5e7;vertical-align:top}th{background:#e8f2f3}.scroll{overflow:auto}img{width:100%;height:auto}.caption{font-size:13px;color:#526a72}a{color:#087f8c}code{display:block;font-size:10px;overflow-wrap:anywhere}li{overflow-wrap:anywhere}.tag{font-size:12px;letter-spacing:1px;color:#087f8c}@media print{body{background:white}main{padding:0}section{break-inside:avoid}header,section{padding:16px}code{font-size:8px}}
</style><main><header><div class='tag'>2026-10-27 · RGB DIAGNOSTIC V5</div><h1>为什么书稿里的 RGB 视觉伺服好，当前联合训练却退化？</h1><p>""" + esc(story["lead"]) + "</p></header>" + "".join(sections) + full_table + images + "<section><h2>可追溯数据与脚本</h2><p><a href='COMPARISONS.json'>完整组别与回合摘要</a> · <a href='SOURCE_MANIFEST.json'>来源哈希</a> · <a href='GEOMETRY_ALL_SUMMARIES.json'>全部几何诊断切片</a></p><details><summary>展开全部来源</summary><ul>" + links + "</ul></details></section></main></html>"
    (output / "REPORT.html").write_text(page, encoding="utf-8")
    save_json(output / "DELIVERY.json", dict(report=str(output / "REPORT.html"), figures=gallery,
              table_rows=len(table_records), source_files=len(sources), completed_unix=time.time()))
    print(json.dumps(dict(report=str(output / "REPORT.html"), figures=len(gallery), groups=len(table_records)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--process-snapshot", required=True)
    parser.add_argument("--narrative", required=True)
    args = parser.parse_args()
    report(args.root, args.output, args.process_snapshot, args.narrative)
