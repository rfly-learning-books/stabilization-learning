"""Summarize a completed GPU environment benchmark without changing its inputs."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    args = p.parse_args()
    root = Path(args.root).resolve()
    if (root/"REPORT.html").exists():
        raise FileExistsError("Preserve the existing benchmark report")
    final = json.loads((root/"FINAL_STATE.json").read_text(encoding="utf8"))
    if final["status"] != "complete":
        raise RuntimeError("Only summarize a completed benchmark")
    profile = json.loads((root/"PROFILE.json").read_text(encoding="utf8"))
    audit = json.loads((root/"AUDIT.json").read_text(encoding="utf8"))
    if not audit["passed"] or not profile["rollout_model_unchanged"] or not profile["model_restored"]:
        raise RuntimeError("Audit or model preservation failed; inspect the original evidence")
    grouped = defaultdict(list)
    for row in profile["phases"]:
        grouped[(row["mode"], row["plant"], row["batch_size"], row["kind"])].append(row)
    summary = []
    for (mode, plant, batch, kind), rows in grouped.items():
        samples = sum(r["samples"] for r in rows)
        seconds = sum(r["seconds"] for r in rows)
        n = sum(r["telemetry"]["samples"] for r in rows)
        item = dict(mode=mode, plant=plant, batch_size=batch, kind=kind, repeats=len(rows),
                    samples=samples, seconds=seconds, samples_per_second=samples/seconds,
                    repeat_sample_rate_sd=statistics.stdev(r["samples_per_second"] for r in rows) if len(rows)>1 else 0,
                    gpu_mean_percent=sum(r["telemetry"]["gpu"]*r["telemetry"]["samples"] for r in rows)/n if n else None,
                    telemetry_samples=n, unit=rows[0]["unit"],
                    torch_peak_mib=max(r["peak_torch_mib"] for r in rows),
                    device_mean_mib=statistics.mean(r["telemetry"]["memory_mib"] for r in rows))
        summary.append(item)
    index = {(r["mode"],r["plant"],r["batch_size"],r["kind"]): r for r in summary}
    for row in summary:
        reference = index.get((row["mode"], row["plant"], row["batch_size"], "serial"))
        if row["kind"] == "gpu" and reference:
            row["speedup_vs_same_batch_serial"] = row["samples_per_second"]/reference["samples_per_second"]
    (root/"SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"serial":"#777f8d", "gpu":"#087f8c"}
    for mode, ax in (("environment", axes[0,0]), ("closed_loop", axes[0,1])):
        for kind in ("serial", "gpu"):
            rows = sorted((r for r in summary if r["mode"]==mode and r["plant"]=="P0" and r["kind"]==kind),key=lambda r:r["batch_size"])
            ax.errorbar([r["batch_size"] for r in rows], [r["samples_per_second"] for r in rows],
                        yerr=[r["repeat_sample_rate_sd"] for r in rows], label=kind, color=colors[kind], marker="o", capsize=3)
        ax.set_xscale("log", base=2); ax.set_xlabel("Parallel environments")
        ax.set_ylabel("Valid transitions / second"); ax.legend(); ax.grid(alpha=.18)
        ax.set_title("P0: physics + RGB rendering" if mode=="environment" else "P0: complete visual control loop")
    ax=axes[1,0]
    for kind in ("serial", "gpu"):
        rows=sorted((r for r in summary if r["mode"]=="closed_loop" and r["plant"]=="P0" and r["kind"]==kind),key=lambda r:r["batch_size"])
        ax.plot([r["batch_size"] for r in rows], [r["gpu_mean_percent"] for r in rows], color=colors[kind],marker="o",label=kind)
    ax.set_xscale("log", base=2); ax.set_ylim(0,100); ax.set_xlabel("Parallel environments")
    ax.set_ylabel("Whole-device GPU utilization (%)"); ax.set_title("P0: complete-loop GPU activity"); ax.legend(); ax.grid(alpha=.18)
    ax=axes[1,1]
    rows=sorted((r for r in summary if r["mode"]=="learning"),key=lambda r:r["batch_size"])
    bars=ax.bar([str(r["batch_size"]) for r in rows],[r["samples_per_second"] for r in rows],color="#be6b35")
    for bar,r in zip(bars,rows):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{r["gpu_mean_percent"]:.1f}% GPU',ha="center",va="bottom")
    ax.set_xlabel("Training batch size"); ax.set_ylabel("Training transitions / second")
    ax.set_title("GPU raw buffer + geometry + full optimizer pair")
    config=profile["protocol"]["config"]
    fig.suptitle(f'{profile["protocol"]["gpu"]} | 256x256 RGB | {config["repeats"]} x {config["seconds"]:g}s per condition',fontsize=13)
    fig.savefig(root/"throughput.png",dpi=160); plt.close(fig)

    titles={"environment":"环境推进与 RGB 渲染", "closed_loop":"完整视觉闭环", "learning":"特征与 M 学习更新"}
    parts=[]
    for mode in titles:
        rows=sorted((r for r in summary if r["mode"]==mode),key=lambda r:(r["plant"],r["batch_size"],r["kind"]=="gpu"))
        cells=[]
        for r in rows:
            ratio=r.get("speedup_vs_same_batch_serial")
            ratio="—" if ratio is None else f"{ratio:.2f}×"
            cells.append(f'<tr><td>{r["plant"]}</td><td>{r["batch_size"]}</td><td>{"GPU 并行" if r["kind"]=="gpu" else "原串行环境"}</td>'
                         f'<td>{r["samples_per_second"]:.1f}</td><td>±{r["repeat_sample_rate_sd"]:.1f}</td>'
                         f'<td>{r["gpu_mean_percent"]:.1f}%</td><td>{ratio}</td><td>{r["device_mean_mib"]/1024:.2f}</td></tr>')
        parts.append(f'<h2>{titles[mode]}</h2><div class="table"><table><thead><tr><th>工况</th><th>并行数 / 批量</th><th>管路</th><th>样本/秒</th><th>计时重复 SD</th><th>平均 GPU</th><th>同批量速度比</th><th>整卡显存 GiB</th></tr></thead><tbody>{"".join(cells)}</tbody></table></div>')
    physical_max=max(max(r["max_abs"].values()) for r in audit["physical"])
    depth=audit["geometry"]["estimated_depth"]
    action=audit["geometry"]["controller_action"]
    protocol=profile["protocol"]
    main_row=index.get(("closed_loop", "P0", 16, "gpu"))
    headline=""
    if main_row:
        ref=index[("closed_loop", "P0", 16, "serial")]
        headline=(f'<p><strong>16 路 P0 完整闭环：{main_row["samples_per_second"]:.1f} 个有效样本/秒，'
                  f'原管路为 {ref["samples_per_second"]:.1f} 个/秒，速度比 {main_row["speedup_vs_same_batch_serial"]:.2f}×；'
                  f'平均 GPU 活动 {main_row["gpu_mean_percent"]:.1f}%。</strong></p>')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RGB GPU 并行环境实测</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#233140;background:#f6f8fa;margin:0;line-height:1.7}}
main{{max-width:1100px;margin:auto;padding:36px 24px}}h1{{font-size:30px;line-height:1.3}}h2{{margin-top:36px;font-size:21px}}
.note{{background:#e9f2f3;padding:18px 22px;border-left:4px solid #087f8c;border-radius:5px}}img{{max-width:100%;height:auto;background:white}}
table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid #e4e9ef;text-align:right}}
th:first-child,td:first-child{{text-align:left}}.table{{overflow-x:auto}}a{{color:#087f8c}}code{{background:#e9edf2;padding:2px 5px}}.muted{{color:#687785}}
</style><main><p class="muted">2026-10-27 · 原生 Windows · RTX 4070 Ti · 性能工程实验</p>
<h1>同一 RGB 视觉伺服任务的 GPU 并行管路</h1>
<div class="note">已实现批量 GPU P0/P1 状态推进、MuJoCo Warp RGB 渲染、GPU FastDepth / 图像几何，并复用原特征、M 和控制构造。以下数据为实测吞吐和整卡活动，不是训练成功率或完整长训加速承诺。</div>
{headline}
<p>对照双方使用同一套任务定义、256×256 相机、128×128 网络输入、四个训练房间族、同一冻结工程模型；共同使用 MuJoCo {protocol["mujoco"]}、PyTorch {protocol["torch"]}。原正式训练使用的旧运行环境没有修改。这里的串行对照不是重新运行旧正式训练。</p>
<img src="throughput.png" alt="吞吐和 GPU 活动随批量变化">
{"".join(parts)}
<h2>结果应怎样解读</h2>
<p>环境采样计入物理推进、碰撞/成功判定、RGB 渲染和回合重置；完整视觉闭环还计入 FastDepth、图像几何、当前特征交互矩阵 L、M 和动作求解。自动重置不算有效转移。学习计时从这次新采的 GPU 原始缓冲读取，执行一对完整的特征/M 反传和 Adam 更新；同一批被两次优化使用，样本数只计一次。</p>
<p>串行对照已经使用现有的批量网络计算；环境步进和图像几何仍经过原来的 CPU 管路。这次速度比反映批量环境、渲染后端和 GPU 几何处理的综合变化，不能全部归因于省去了 NumPy 到 GPU 的复制。小批量纯环境测量中，经典渲染路径更快；当前特征交互矩阵 L 在两条完整闭环管路里都逐步重算。</p>
<p>每种条件预热后重复 {protocol["config"]["repeats"]} 次、每次约 {protocol["config"]["seconds"]:g} 秒。SD 是计时波动，不是跨种子标准差或置信区间。GPU 数值来自每 200 ms 的整卡采样，裁去窗口边界；空闲基线为 {profile["idle"]["gpu"]:.1f}%。显存是整卡窗口平均，包含图形、Warp、PyTorch 和桌面使用。</p>
<p>该物理任务仍是四自由度速度外环，并非完整电机/姿态多旋翼仿真。学习计时不含磁盘回放、归档、验证和出图；闭环计算使用短验收模型，各重复之间恢复参数，不能据此判断学到的控制效果。多房间按组提交 GPU 渲染，不同组尚未融合为一个内核。</p>
<h2>一致性与图像差异</h2>
<p>44 个房间资产均完成静态 GPU 渲染检查；验证/测试房间只检查加载和图像，不用于学习、选模或吞吐调参。四房间 × P0/P1 的物理对照最大数值差为 {physical_max:.3g}；成功保持、失败边界、独立环境与图像存储的额外 5 项回归通过。</p>
<p>对相同 RGB 输入，GPU 深度的相对 L2 差为 {depth["relative_l2"]:.3g}，控制动作最大绝对差为 {action["max_abs"]:.3g}，主要来自预处理数值/量化差异。固定深度的像素交互计算另行对照。未知真值输入被拒绝，模拟器真值深度未进入控制。</p>
<p>下图左为经典渲染、右为 GPU 批量渲染。几何、相机和目标相同，但光照、抗锯齿及纹理过滤有差异，不能称逐像素复现；闭环轨迹因此也可能不同。新图像分布下的实际学习效果仍需独立验证。</p>
<img src="renderer_comparison.png" style="max-width:640px" alt="四种房间经典渲染和 GPU 渲染对照">
<h2>交付与原始数据</h2><p><a href="PROFILE.json">原始计时与 GPU 采样</a> · <a href="SUMMARY.json">分组汇总</a> · <a href="AUDIT.json">独立一致性检查</a> · <a href="PROTOCOL.json">冻结测量协议</a></p>
<p>代码入口：<code>scripts/benchmark_rgb_gpu_parallel.py</code>；环境：<code>stabilization_gym/tasks_gpu_rgb</code>；说明：<code>docs/RGB_GPU_PARALLEL.md</code>。仅在 Windows 本机实测，Linux 尚未在另一台机器验收。</p>
</main></html>'''
    (root/"REPORT.html").write_text(body,encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
