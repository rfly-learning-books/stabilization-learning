"""Read only after the owning pipeline has finished the relevant writes."""
from pathlib import Path
import json
import numpy as np


def create_report(root, config, rows, selections):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(root); figures=root/"figures"; figures.mkdir(exist_ok=True)
    files=[]
    for task in config["tasks"]:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for seed in config["seeds"]:
            history=json.loads((root/"jobs"/f"{task}_seed{seed}"/"TRAINING.json").read_text(encoding="utf-8"))
            train=[r for r in history["training"] if "loss" in r]
            axes[0].plot([r["update"] for r in train],[r["mean_cost"] for r in train],alpha=.65,label=str(seed))
            val=history["validation"]
            axes[1].plot([r["update"] for r in val],[r["successes"]/r["episodes"] for r in val],marker=".",label=str(seed))
        axes[0].set(title="Sampled training error integral",xlabel="Gradient update",ylabel="Normalized error integral")
        axes[1].set(title="Fixed validation final-hold success",xlabel="Gradient update",ylim=(-.02,1.02))
        for ax in axes: ax.grid(alpha=.25);ax.legend(title="Training seed")
        fig.suptitle(task+(" - engineering development" if config.get("development_only") else ""));fig.tight_layout()
        for suffix in ("png","pdf","svg"):
            path=figures/f"{task}_learning.{suffix}";fig.savefig(path,dpi=160);files.append(path.relative_to(root).as_posix())
        plt.close(fig)
    lines=["# 可变证书观测器多种子实验", "", "学习对象为内部校正律中的状态相关 M；网络只接收估计状态，真值只参与离线监督与独立评价。", "",
        f"任务数 {len(config['tasks'])}，训练种子 {config['seeds']}，每组最多 {config['updates']} 次完整回合梯度更新。", "",
        "同一任务的不同训练种子使用共同测试轨迹。跨种子差异不能增加独立初态数，标准差不是置信区间。", "",
        "|任务|工况|方法|种子|成功保持/回合|平均误差成本|终端误差最大值|候选下降违反率|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for task in config["tasks"]:
        scenarios=config["test_scenarios"]+(config["pose_test_stress"] if task=="cooperative_pose" else [])
        for scenario in scenarios:
            methods=config["baselines"]+(config["pose_baselines"] if task=="cooperative_pose" else [])+["learned_M"]
            for method in methods:
                for seed in config["seeds"] if method=="learned_M" else [0]:
                    subset=[r for r in rows if (r["task"],r["scenario"],r["method"],r["seed"])==(task,scenario,method,seed)]
                    violation=[r["certificate"]["descent_violation_fraction"] for r in subset if r["certificate"]["transitions"]]
                    lines.append(f"|{task}|{scenario}|{method}|{seed}|{sum(r['success'] for r in subset)}/{len(subset)}|{np.mean([r['fixed_horizon_error_cost'] for r in subset]):.6g}|{max(r['empirical_terminal_error_max'] for r in subset):.6g}|{np.mean(violation) if violation else 0:.3%}|")
    lines += ["", "固定 M 与学习 M 的候选证书不同；误差成本、成功保持与真实物理误差用于跨方法比较。", "",
        "持续噪声下报告末段误差分位数和最大值，它们是有限轨迹经验范围。Gaussian 噪声无有限确定性全局幅值界，本实验未建立学习观测器的全局或非线性域稳定证明。", "",
        "遮挡、标定偏差、错误关联与退化几何中的失败均保留。合成关键点前端不能代替真实图像检测或飞行实验。", "",
        "CartPole 此处验证短时已知输入下的观测，并未证明估计器接入控制器后的非线性闭环稳定。", "",
        "## 学习曲线", ""]
    for filename in files:
        if filename.endswith(".png"): lines += [f"![{Path(filename).stem}]({filename})", ""]
    (root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    (root/"FIGURES.json").write_text(json.dumps(files,indent=2),encoding="utf-8")
