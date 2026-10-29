from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import json_write


def make_report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output).resolve()
    report_dir = output / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    training = [json.loads(line) for line in (output / "training.jsonl").read_text(encoding="utf-8").splitlines()] if (output / "training.jsonl").exists() else []
    evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "evaluation").glob("*/summary.json"))]
    if training:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        plots = [("critic_fit", "Cost critic prediction error"), ("reconstruction", "RGB reconstruction error"),
                 ("lac_drift", "Replay LAC drift estimate"), ("actor_gradient_norm", "Certificate actor gradient norm")]
        for ax, (key, title) in zip(axes.flat, plots):
            rows = [(i, row[key]) for i, row in enumerate(training) if key in row]
            if rows:
                ax.plot([r[0] for r in rows], [r[1] for r in rows])
            ax.set_title(title)
            ax.set_xlabel("logged optimizer update")
            ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(report_dir / "training.png", dpi=160)
        plt.close(fig)
    telemetry = list((output / "evaluation").glob("*/*_telemetry.npz"))
    if telemetry:
        with np.load(telemetry[0], allow_pickle=False) as data:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            axes[0].plot(np.linalg.norm(data["offset"][:, :3], axis=-1))
            axes[0].set_title("Position error (m), evaluation truth")
            if "certificate" in data:
                axes[1].plot(data["certificate"])
            axes[1].set_title("Certificate on actual RGB observations")
            axes[2].plot(data["action"])
            axes[2].set_title("Normalized commanded velocities")
            for ax in axes:
                ax.set_xlabel("step")
                ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(report_dir / "example_trajectory.png", dpi=160)
            plt.close(fig)
    lines = ["# LAC+AE 证书视觉伺服执行报告", "", "本报告根据已有日志生成。训练完成、预测下降和数学验收均不等同于区域稳定性证明。", "",
             f"训练日志行数：{len(training)}。评估汇总数：{len(evaluations)}。", "",
             "| 划分 | 方法 | 对象 | 回合数 | 成功率 | 最差场景成功率 | 实际证书上升比例 | 完整评估 |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for row in evaluations:
        if not row.get("episodes"):
            continue
        parts = row.get("plant_summary", {row.get("plant", "see raw episodes"): row})
        for plant, part in parts.items():
            increase = f"{part['mean_actual_V_increase_fraction']:.3f}" if "mean_actual_V_increase_fraction" in part else "N/A"
            lines.append(f"| {row['split']} | {row['method']} | {plant} | {part['episodes']} | {part['success_rate']:.3f} | "
                         f"{part['worst_scene_success']:.3f} | {increase} | {row['complete']} |")
    if training:
        lines.extend(["", "![训练曲线](training.png)"])
    if telemetry:
        lines.extend(["", "![一个实际渲染轨迹](example_trajectory.png)"])
    lines.extend(["", "smoke 配置只验证工程链路，不用于比较方法性能。P1 存在未观测执行器状态，其结果仅作经验压力测试。",
                  "目标条件 Critic 的非负性不自动保证严格正定、全状态可观测或 Lyapunov 下降。"])
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_write(report_dir / "summary.json", {"training_rows": len(training), "evaluations": evaluations})
    return str(report_dir / "REPORT.md")
