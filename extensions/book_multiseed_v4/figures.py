"""Post-selection scientific figures for the frozen multi-seed experiment.

The authorized pipeline calls this after every test/audit group is sealed. An
external observer must not call it while a writer is active: only committed
events are suitable for live observation. No model is selected by this module.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


TASK_LABELS = {
    "coupled_linear": "Coupled linear system",
    "point_mass": "2-D point mass",
    "pendulum": "Torque-driven pendulum",
    "cartpole": "Force-driven CartPole",
}
COLORS = ("#1764a0", "#d98421", "#23866e", "#ac4670", "#7257a5")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed_events(run_dir):
    """Read only the event files already committed by atomic rename."""
    return [_read(p) for p in sorted((Path(run_dir) / "events").glob("*.json"))]


def aggregate_curves(series):
    """Exact-update mean/sample-SD across independent training seeds.

    Missing/early-stopped observations stay missing. No interpolation, forward
    filling or episode-as-seed pseudo-replication is performed.
    """
    points = defaultdict(list)
    for seed, values in sorted(series.items()):
        observed = set()
        for update, value in values:
            if update in observed:
                raise ValueError(f"Duplicate update for training seed {seed}: {update}")
            observed.add(update)
            if math.isfinite(float(value)):
                points[int(update)].append(float(value))
    return [dict(update=x, mean=float(np.mean(y)),
                 std=float(np.std(y, ddof=1)) if len(y) > 1 else None,
                 n=len(y)) for x, y in sorted(points.items())]


def training_series(events, task, value):
    """Extract one declared observable without dropping rejected candidates."""
    result = defaultdict(list)
    for event in events:
        if event.get("stage") != "update_committed" or event.get("task") != task:
            continue
        y = value(event)
        if y is not None:
            result[int(event["seed"])].append((int(event["update"]), float(y)))
    for values in result.values():
        values.sort()
    return dict(result)


def _group_value(event, scale, key):
    for row in (event.get("validation") or {}).get("groups", []):
        if float(row["scale"]) == float(scale):
            if key == "success_percent":
                return 100.0 * row["successes"] / row["episodes"]
            if key == "cost_percent":
                return 100.0 * (row["cost_ratio"] - 1)
            if key == "energy_percent":
                return 100.0 * (row["energy_ratio"] - 1)
            return row.get(key)
    return None


def result_summary(result):
    """Each training seed contributes once, including failed test gates."""
    summary = {}
    for task, task_result in result["tasks"].items():
        seeds = task_result["seeds"]
        scales = sorted({float(r["scale"]) for s in seeds.values()
                         for r in s["comparisons"]})
        groups = []
        for scale in scales:
            rows = [(int(seed), row) for seed, s in seeds.items()
                    for row in s["comparisons"] if float(row["scale"]) == scale]
            metrics = {}
            for key in ("cost_percent", "arrival_delay", "energy_ratio",
                        "max_normalized_decay_rate", "cost_vs_classical_SDRE_percent",
                        "learned_success_percent", "baseline_success_percent"):
                values = []
                for _, row in rows:
                    if key.endswith("success_percent"):
                        source = "learned_success" if key.startswith("learned") else "baseline_success"
                        values.append(100.0 * row[source] / row["episodes"])
                    elif key in row:
                        values.append(float(row[key]))
                if values:
                    metrics[key] = dict(mean=float(np.mean(values)),
                        std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        min=float(min(values)), max=float(max(values)), n=len(values),
                        values=values)
            groups.append(dict(scale=scale, seeds=[seed for seed, _ in rows], metrics=metrics))
        summary[task] = dict(groups=groups,
            seeds={str(seed): dict(test_passed=bool(s["test_gate"]["qualified"]),
                reasons=s["test_gate"].get("reasons", []),
                audit_passed=bool(s.get("audit_passed", False)),
                selection=s["selection"], aliases=s.get("aliases", {}),
                M_state_variation=s.get("M_state_variation"),
                action_difference_from_P0=s.get("action_difference_from_P0"))
                for seed, s in sorted(seeds.items(), key=lambda pair: int(pair[0]))})
    return summary


def _plot_api():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _save(fig, folder, name, manifest, *, caption, data):
    files = {}
    for ext in ("png", "pdf", "svg"):
        dest = folder / f"{name}.{ext}"
        fig.savefig(dest, dpi=180, bbox_inches="tight")
        files[ext] = dict(path=dest.relative_to(folder.parent).as_posix(), sha256=_sha(dest))
    manifest.append(dict(name=name, caption=caption, files=files, data=data))


def _draw_curve(ax, series, *, color, label, raw=True):
    if raw:
        for points in series.values():
            if points:
                x, y = zip(*points)
                ax.plot(x, y, color=color, lw=.65, alpha=.22)
    aggregate = aggregate_curves(series)
    if aggregate:
        x = np.asarray([r["update"] for r in aggregate])
        y = np.asarray([r["mean"] for r in aggregate])
        sd = np.asarray([r["std"] if r["std"] is not None else np.nan for r in aggregate])
        ns = sorted({r["n"] for r in aggregate})
        count = str(ns[0]) if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
        ax.plot(x, y, color=color, lw=1.7, label=f"{label}; n={count}")
        ax.fill_between(x, y-sd, y+sd, color=color, alpha=.14, linewidth=0)
    return aggregate


def _training_figures(plt, folder, events, result, manifest):
    for task, task_result in result["tasks"].items():
        scales = sorted({float(r["scale"]) for s in task_result["seeds"].values()
                         for r in s["comparisons"]})
        fig, axes = plt.subplots(3, 2, figsize=(13.4, 11.2), constrained_layout=True)
        axes = axes.ravel()
        panels = ((0, "cost_percent", "Validation physical cost vs P0 (%)"),
                  (2, "success_percent", "Validation success (%)"),
                  (3, "arrival_delay", "Arrival difference vs P0 (s)"),
                  (4, "energy_percent", "Control energy difference vs P0 (%)"))
        data = {}
        for index, key, title in panels:
            ax = axes[index]
            for j, scale in enumerate(scales):
                series = training_series(events, task,
                    lambda event, scale=scale, key=key: _group_value(event, scale, key))
                data[f"{key}_scale_{scale}"] = _draw_curve(ax, series,
                    color=COLORS[j], label=f"scale {scale:g}")
            ax.set_title(title)
            if key != "success_percent":
                ax.axhline(0, color="#555555", lw=.8, ls="--")
            else:
                ax.set_ylim(-2, 102)
            ax.legend(fontsize=8, loc="best")
        feasible = training_series(events, task, lambda e:
            float(e["validation"]["qualified"]) if e.get("validation") else None)
        data["validation_gate_pass_fraction"] = _draw_curve(axes[1], feasible,
            color=COLORS[0], label="all gates pass", raw=False)
        axes[1].set(title="Candidate validation gate pass fraction", ylim=(-.05, 1.05))
        axes[1].legend(fontsize=8)
        trajectory_available = any(_group_value(e, s, "max_normalized_decay_rate") is not None
                                   for e in events if e.get("task") == task for s in scales)
        if trajectory_available:
            for j, scale in enumerate(scales):
                series = training_series(events, task,
                    lambda e, scale=scale: _group_value(e, scale, "max_normalized_decay_rate"))
                data[f"full_successor_trajectory_rate_scale_{scale}"] = _draw_curve(
                    axes[5], series, color=COLORS[j], label=f"scale {scale:g}")
            axes[5].set_title("Full-successor normalized decay residual (trajectory max)")
        else:
            series = training_series(events, task, lambda e:
                (e.get("validation") or {}).get("probe_max_normalized_decay_rate"))
            data["full_successor_probe_rate"] = _draw_curve(axes[5], series,
                color=COLORS[0], label="empirical probe maximum")
            axes[5].set_title("Full-successor normalized decay residual (probe max)")
        axes[5].axhline(1e-6, color="#ad322a", lw=.8, ls="--", label="gate 1e-6")
        axes[5].set_yscale("symlog", linthresh=1e-6)
        axes[5].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("Committed optimizer updates")
        fig.suptitle(f"{TASK_LABELS.get(task, task)}: every evaluated candidate\n"
                     "Solid = training-seed mean; band = sample SD; faint = each seed (no smoothing)",
                     fontsize=14)
        _save(fig, folder, task+"_validation", manifest,
            caption="各训练种子独立曲线及均值±样本标准差；包含未通过门槛的候选。没有平滑、补齐或用测试重选。",
            data=data)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        diagnostics = (("loss", "Training surrogate objective"),
                       ("physical_training_cost", "Physical cost on changing training batches"),
                       ("constraint", "Training constraint penalty"),
                       ("gradient_norm", "Gradient norm before clipping"),
                       ("model_transitions", "Main-actor logical model transitions"),
                       ("wall_seconds", "Committed update wall time incl. validation (s)"))
        data = {}
        for j, (key, title) in enumerate(diagnostics):
            ax = axes.ravel()[j]
            data[key] = _draw_curve(ax, training_series(events, task, lambda e, key=key:e.get(key)),
                color=COLORS[j % len(COLORS)], label="training seeds")
            ax.set(title=title, xlabel="Committed optimizer updates")
            if key in ("constraint", "gradient_norm"):
                ax.set_yscale("symlog", linthresh=1e-8)
            ax.legend(fontsize=8)
        fig.suptitle(TASK_LABELS.get(task, task)+": optimization diagnostics", fontsize=14)
        _save(fig, folder, task+"_training", manifest,
            caption="训练批次随更新变化，训练成本不等于固定验证集成绩；model_transitions只统计主学习策略逻辑前向，不含参考策略、PGD、重算和测试。",
            data=data)
        plt.close(fig)


def _test_figure(plt, folder, summary, manifest):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    entries = [(task, group) for task, s in summary.items() for group in s["groups"]]
    labels = [f"{task}\nscale {group['scale']:g}" for task, group in entries]
    panels = (("cost_percent", "Test physical cost vs fixed P0 (%)", 0),
              ("cost_vs_classical_SDRE_percent", "Test physical cost vs classical online SDRE (%)", 0),
              ("learned_success_percent", "Test success (%)", 100),
              ("arrival_delay", "Test arrival difference vs P0 (s)", 0),
              ("energy_ratio", "Test control energy / P0", 1),
              ("max_normalized_decay_rate", "Test full-successor normalized decay residual (max)", 1e-6))
    for ax, (key, title, reference) in zip(axes.ravel(), panels):
        for x, (task, group) in enumerate(entries):
            row = group["metrics"].get(key)
            if row is None:
                continue
            color = COLORS[list(summary).index(task) % len(COLORS)]
            values = row["values"]
            dx = np.linspace(-.13, .13, len(values)) if len(values) > 1 else [0.]
            ax.scatter(x+np.asarray(dx), values, color=color, s=17, alpha=.6, zorder=3)
            ax.errorbar(x, row["mean"], yerr=row["std"] or 0., color=color,
                        fmt="_", markersize=14, capsize=3, lw=1.5, zorder=4)
        ax.axhline(reference, color="#555555", lw=.8, ls="--")
        ax.set(title=title, xticks=np.arange(len(entries)), xticklabels=labels)
        ax.tick_params(axis="x", labelsize=7, rotation=40)
        if key == "max_normalized_decay_rate":
            ax.set_yscale("symlog", linthresh=1e-6)
        if key == "learned_success_percent":
            ax.set_ylim(-2, 102)
    fig.suptitle("Held-out test results: each dot is one independently trained seed\n"
                 "Center and error bar = seed mean and sample SD; failed test gates remain included",
                 fontsize=14)
    _save(fig, folder, "multiseed_test_comparison", manifest,
        caption="测试集只在全部选模锁定后使用。每点是一枚独立训练种子，误差棒为训练种子样本标准差；不是初态配对自助区间，也不是稳定性证明。",
        data=summary)
    plt.close(fig)


def _resolve_role(role, aliases):
    visited = set()
    while role in aliases:
        if role in visited:
            raise ValueError("Cyclic method aliases")
        visited.add(role)
        role = aliases[role]
    return role


def _safe_job(run_dir, relative):
    job = (run_dir / relative).resolve()
    if not job.is_relative_to(run_dir.resolve()):
        raise ValueError("Job path leaves run directory")
    return job


def _trajectory_figures(plt, folder, run_dir, result, manifest):
    """Predeclared illustration: minimum declared model seed, largest scale, lane 0."""
    illustrations = []
    for task, task_result in result["tasks"].items():
        seed = min(int(s) for s in task_result["seeds"])
        one = task_result["seeds"][str(seed)]
        if not one.get("job_path"):
            illustrations.append(dict(task=task, available=False, reason="No sealed trace path declared"))
            continue
        scale = max(float(r["scale"]) for r in one["comparisons"])
        aliases = one.get("aliases", {})
        job = _safe_job(run_dir, one["job_path"])
        physical = one.get("physical", {})
        dt = float(physical["dt"])
        roles = []
        for requested in ("fixed_P0", "learned", "classical_SDRE"):
            actual = _resolve_role(requested, aliases)
            if any(row[0] == actual for row in roles):
                continue
            roles.append((actual, requested))
        fig, axes = plt.subplots(3, 2, figsize=(13.2, 10.8), constrained_layout=True)
        data = dict(training_seed=seed, scale=scale, lane=0, aliases=aliases, groups=[])
        for j, (role, _) in enumerate(roles):
            dest = job / task / "test" / f"{scale}_{role}"
            metadata, audit = _read(dest/"GROUP.json"), _read(dest/"AUDIT.json")
            if not audit.get("passed") or _sha(dest/"trace.npz") != metadata["trace_sha256"]:
                raise ValueError(f"Unsealed or invalid trace cannot be plotted: {dest}")
            with np.load(dest/"trace.npz", allow_pickle=False) as trace:
                valid = trace["active"][:, 0].astype(bool)
                states = trace["states"][valid, 0]
                goal = np.asarray(physical.get("goal", np.zeros(states.shape[-1])))
                errors = states - goal
                before, after = trace["V_before"][valid, 0], trace["V_after"][valid, 0]
                actions = trace["actions"][valid, 0]
                metric = trace["M_current"][valid, 0]
                following_metric = trace["M_next"][valid, 0]
                following = trace["following"][valid, 0] - goal
                delta = after-before
                # Recompute the actual successor quadratic, not frozen-current-M.
                recomputed = np.einsum("ti,tij,tj->t", following, following_metric, following) - before
                if not np.allclose(delta, recomputed, atol=1e-8, rtol=1e-8):
                    raise ValueError("Figure successor V disagrees with audited metric")
                time = np.arange(len(states))*dt
                color = COLORS[j]
                same = [alias for alias in ("fixed_P0", "learned", "classical_SDRE")
                        if _resolve_role(alias, aliases) == role]
                label = " = ".join(same)
                axes[0, 0].plot(time, np.linalg.norm(errors, axis=-1), color=color, label=label)
                axes[0, 1].plot(time, before/max(abs(float(before[0])), 1e-18), color=color, label=label)
                axes[1, 0].plot(time, delta, color=color, label=label)
                axes[1, 1].plot(time, np.linalg.norm(actions, axis=-1), color=color, label=label)
                axes[2, 0].plot(states[:, 0], states[:, 1], color=color, label=label)
                axes[2, 0].scatter(states[0, 0], states[0, 1], s=22, color=color, marker="o")
                axes[2, 0].scatter(states[-1, 0], states[-1, 1], s=30, color=color, marker="x")
                change = np.linalg.norm(metric-metric[0], axis=(-2, -1))
                axes[2, 1].plot(time, change, color=color, label=label)
                data["groups"].append(dict(role=role, aliases=same, test_initial_seed=metadata["seeds"][0],
                    trace_sha256=metadata["trace_sha256"], steps=len(states),
                    max_raw_successor_delta_V=float(delta.max()),
                    metric_change_frobenius_max=float(change.max())))
        titles = ("Physical state error norm", "Candidate V(t) / V(0)",
                  "Full successor V(e+, M(e+)) - V(e, M(e))", "Applied control norm",
                  "First two physical state coordinates", "Actual M change from its first value (Frobenius)")
        for i, ax in enumerate(axes.ravel()):
            ax.set_title(titles[i])
            if i != 4:
                ax.set_xlabel("Physical time (s)")
            else:
                ax.set(xlabel="state coordinate 0", ylabel="state coordinate 1")
            ax.legend(fontsize=7.5)
        axes[0, 1].set_yscale("symlog", linthresh=1e-8)
        axes[1, 0].set_yscale("symlog", linthresh=1e-8)
        axes[1, 0].axhline(0, color="#555555", lw=.7, ls="--")
        fig.suptitle(f"{TASK_LABELS.get(task, task)}: fixed illustration, seed {seed}, scale {scale:g}, test lane 0\n"
                     "Chosen by protocol order, not by outcome; candidate V differs between methods",
                     fontsize=13)
        _save(fig, folder, task+"_trajectory", manifest,
            caption="固定展示最小训练种子、最大评价尺度、首个测试初态；不是最优轨迹。各方法的V本身不同，不能用V绝对值横向排名。",
            data=data)
        plt.close(fig)
        illustrations.append(dict(task=task, available=True, **data))
    return illustrations


def _mean_sd(row, digits=3):
    if row is None:
        return "未记录"
    text = f"{row['mean']:.{digits}f}"
    return text + (f" ± {row['std']:.{digits}f}" if row["std"] is not None else "（仅1种子）")


def _report(run_dir, result, summary, figures, illustrations):
    seeds = sorted({int(seed) for task in summary.values() for seed in task["seeds"]})
    completed = sum(len(task["seeds"]) for task in summary.values())
    passed = sum(s["test_passed"] for task in summary.values() for s in task["seeds"].values())
    audited = all(s["audit_passed"] for task in summary.values() for s in task["seeds"].values())
    lines = ["# 书稿接口多种子训练与独立终验", "",
        f"本轮包含 {len(summary)} 个现有 SDC 基座任务、训练种子 {seeds}，共 {completed} 个任务/种子组合。",
        f"独立测试门槛通过 {passed}/{completed}；已声明的全部轨迹审计{'通过' if audited else '存在未通过项'}。",
        "结果图和统计包括失败或退化的种子；未利用测试成绩重选模型，也未重新训练失败种子以替换结果。", "",
        "## 结果与训练种子波动", "",
        "下表为独立训练种子均值 ± 样本标准差。每枚种子权重相同；不是把测试初态当作训练种子，也不是初态配对自助法区间。", "",
        "|任务|尺度|相对P0成本 %|相对经典SDRE成本 %|学习成功率 %|到达差 s|能量比|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    if result.get("development_only",False):
        lines[2:2] = ["**工程流程试跑：独立开发随机流、缩减预算；不属于正式多种子性能结论。**", ""]
    for task, task_summary in summary.items():
        for group in task_summary["groups"]:
            m = group["metrics"]
            lines.append(f"|{task}|{group['scale']:g}|{_mean_sd(m.get('cost_percent'))}|"
                         f"{_mean_sd(m.get('cost_vs_classical_SDRE_percent'))}|"
                         f"{_mean_sd(m.get('learned_success_percent'), 2)}|"
                         f"{_mean_sd(m.get('arrival_delay'))}|{_mean_sd(m.get('energy_ratio'))}|")
    lines += ["", "成本降低只表示固定时域物理误差积分改善；能量、到达时间和成功保持分别报告。若基线成功率已经为100%，不得将成本下降称为成功率提升。",
        "五枚训练种子可检验初步重复性，但不足以支持强分布推断。图中阴影和误差棒是样本标准差，不是置信区间。", "",
        "## 每个种子的门槛、选模与不利结果", "",
        "|任务|训练种子|实际更新|选中更新|独立测试门槛|原因|实际M状态变化|控制差|",
        "|---|---:|---:|---:|---|---|---:|---:|"]
    for task, task_summary in summary.items():
        for seed, s in task_summary["seeds"].items():
            selection = s["selection"]
            lines.append(f"|{task}|{seed}|{selection.get('updates', '未记录')}|{selection.get('best_update', '未记录')}|"
                         f"{'通过' if s['test_passed'] else '未通过'}|{'; '.join(s['reasons']) or '—'}|"
                         f"{s['M_state_variation']}|{s['action_difference_from_P0']}|")
            if not s["audit_passed"]:
                lines.append(f"\n**{task} seed{seed} 的独立审计未通过，不能视作通过验收。**\n")
    lines += ["", "initial 与固定 P0 控制器相同；线性任务的经典在线 SDRE 也与固定 P0 相同。这些别名不构成额外独立方法。",
        "选择的模型是满足验证门槛后的最低验证成本候选；训练最后一个候选可能被拒绝，不能与选中的模型混称。", "",
        "## 图件", "",
        "训练图保留各独立种子的原始曲线、均值和标准差，包含验证失败候选，不平滑、不插值、不补齐提前停止的曲线。图例的 n 是该更新实际有记录的种子数。",
        "训练批次物理成本对应不断变化的训练初态，固定验证集曲线更适合评价训练进展。model_transitions 仅计主策略逻辑前向，不包含参考策略、PGD、反向重算、验证及独立测试。", ""]
    for figure in figures:
        lines += [f"### {figure['name']}", "", figure["caption"], "",
                  f"![{figure['name']}]({figure['files']['png']['path']})", "",
                  f"[矢量PDF]({figure['files']['pdf']['path']}) · [SVG]({figure['files']['svg']['path']})", ""]
    lines += ["## 机制、接口与书稿对齐边界", "",
        "环境层提供物理状态、误差、SDC系数、输入限幅、物理步进、成功保持和失败条件；策略层将当前误差映射为正定 M(e)。同一个实际 M 同时产生控制与候选证书：", "",
        r"\[u=\operatorname{sat}(u_{\rm ff}-R^{-1}B(e)^\top M(e)e),\quad V(e)=e^\top M(e)e.\]", "",
        "算法层采用完整物理回合模型梯度、训练反例搜索与回放、验证门槛和整次更新断点；不是 PPO，也不是每步在线求解 Riccati 方程。固定 P0 与经典在线 SDRE 分别作为对照。",
        "评测层在所有任务/种子选模锁定后运行相同初态的三尺度测试；独立 NumPy 审计重建状态转移、控制、M 和完整后继 V 差分；报告与图件层只读取封存结果。",
        "传统 Gym 基线任务和历史算法仍属于工具箱，但本轮正式训练仅覆盖表中四个 SDC 模型。扭矩驱动倒立摆、二维质点等模型与书稿可能存在任务定义差异，不能将接口对齐写成书稿全部任务复现。",
        "近目标锚定、局部浮点线性化检查、有限轨迹下降和 PGD 未找到反例都没有新增全局或非线性区域稳定性证明。", ""]
    for candidate in ("BOOK_ALIGNMENT.json", "BOOK_ALIGNMENT.md", "ARCHITECTURE.md"):
        if (run_dir/candidate).exists():
            lines.append(f"书稿任务映射与架构补充：[{candidate}]({candidate})。")
    lines += ["", "图件数据、各文件 SHA256、固定样例规则见 [FIGURES.json](FIGURES.json)。",
              "所有任务和种子的测试选择门禁见 [GLOBAL_SELECTION_COMMITTED.json](GLOBAL_SELECTION_COMMITTED.json)。", ""]
    (run_dir/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_figures(run_dir, result=None):
    """Called by the authorized writer after all selected models are audited.

    This is not a live reader. A selection commitment is necessary but is not by
    itself a claim that the writer is idle; process checks belong to the caller.
    """
    run_dir = Path(run_dir).resolve()
    if not (run_dir/"GLOBAL_SELECTION_COMMITTED.json").is_file():
        raise ValueError("All model selections must be committed before result figures")
    if result is None:
        result = _read(run_dir/"RESULTS.json")
    events = committed_events(run_dir)
    summary = result_summary(result)
    folder = run_dir/"figures"
    folder.mkdir(exist_ok=True)
    plt = _plot_api()
    manifest = []
    _training_figures(plt, folder, events, result, manifest)
    _test_figure(plt, folder, summary, manifest)
    illustrations = _trajectory_figures(plt, folder, run_dir, result, manifest)
    value = dict(schema="book_multiseed_v4_figures", figures=manifest, summary=summary,
        illustration_rule="minimum declared training seed; largest declared scale; first declared test initial seed",
        illustrations=illustrations,
        uncertainty="sample SD across independent training seeds, not a confidence interval",
        all_test_gate_failures_retained=True, formal_stability_proof=False,
        selection_commitment_sha256=_sha(run_dir/"GLOBAL_SELECTION_COMMITTED.json"))
    _write(run_dir/"FIGURES.json", value)
    _report(run_dir, result, summary, manifest, illustrations)
    return dict(manifest="FIGURES.json", report="REPORT.md", figure_count=len(manifest),
        manifest_sha256=_sha(run_dir/"FIGURES.json"), report_sha256=_sha(run_dir/"REPORT.md"),
        all_test_gate_failures_retained=True, formal_stability_proof=False)
