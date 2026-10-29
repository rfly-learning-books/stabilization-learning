"""Read a completed v4 run after an external process witness; write new reports."""
from pathlib import Path
import argparse
import datetime
import html
import json
import math
import time
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def witness(path):
    state = read(path)
    if state.get("schema") != "rgb_v4_process_snapshot" or state.get("query_success") is not True:
        raise RuntimeError("A successful external PowerShell process snapshot is required")
    age = time.time()-state["created_unix"]
    if not 0 <= age <= 60:
        raise RuntimeError("Process witness must be at most 60 seconds old")
    for process in state["processes"]:
        command = process.get("CommandLine")
        if not command:
            raise RuntimeError("Missing Python command line is not an exit")
        if "meetings_mcp_entrypoint.py" in command or "report_rgb_v4.py" in command:
            continue
        raise RuntimeError("Unclassified or active Python writer: " + command)


def estimate(root):
    protocol, timings, outcome = read(root/"protocol.json"), read(root/"TIMINGS.json"), read(root/"RESULTS.json")
    cfg = protocol["config"]["run"]
    collections = [read(p) for p in sorted(root.glob("COLLECTION_*.json"))]
    actual_initial = collections[0]["added_transitions"]
    actual_followup = sum(r["added_transitions"] for r in collections[1:])
    t0 = sum(t["seconds"] for t in timings if t["phase"] == "collection" and t["round_id"] == 0)
    t1 = sum(t["seconds"] for t in timings if t["phase"] == "collection" and t["round_id"] != 0)
    warm = sum(t["seconds"] for t in timings if t["phase"] == "warmup")/cfg["warmup_updates"]
    joint = sum(t["seconds"] for t in timings if t["phase"] == "joint")/sum(t["pairs"] for t in timings if t["phase"] == "joint")
    # Full aligned budget. Linear per-case scaling is conservative relative to
    # the larger formal batches, but remains an extrapolation, not a guarantee.
    validation = sum(t["seconds"] for t in timings if t["phase"] == "validation") / sum(t["cases"] for t in timings if t["phase"] == "validation")
    testing = sum(t["seconds"] for t in timings if t["phase"] == "test") / sum(t["cases"] for t in timings if t["phase"] == "test")
    audit = sum(t["seconds"] for t in timings if t["phase"] == "audit")
    formal_strata = 24*4*4
    stages = dict(initial_collection=t0/actual_initial*(formal_strata*80),
                  refresh_collection=t1/actual_followup*(formal_strata*64*4),
                  warmup=1000*warm, joint=6500*joint, validation=5*48*validation,
                  heldout_test=4*2*72*testing,
                  audit_and_setup=max(120., audit*129024/(actual_initial+actual_followup)))
    total = sum(stages.values())
    return dict(stage_seconds=stages, base_minutes=total/60,
                planning_range_minutes=[math.floor(total/60*.95/5)*5, math.ceil(total/60*1.3/5)*5],
                source_profile=str(root), measured_pairs=cfg["joint_pairs"],
                measured_fresh_transitions=actual_initial+actual_followup,
                warmup_seconds_per_update=warm, joint_seconds_per_pair=joint,
                joint_sample_throughput=cfg["batch_size"]/joint,
                formal_new_transitions_lower_bound=129024, formal_max_test_steps=115200,
                assumptions=["disk/CPU contention may change", "full archive remains uniformly sampleable",
                             "later features can change episode lengths", "physical evaluation is still 4DOF",
                             "not extrapolated from pure renderer FPS", "GPU is device-wide telemetry"])


def curves(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    updates, probes, matrices = [], [], []
    for path in sorted((root/"events").glob("*.json")):
        event = read(path); d = event["data"]
        if event["kind"] == "update":
            updates.append(d)
        elif event["kind"] == "fixed_training_probe":
            m = np.asarray(d["M"])
            eig = np.linalg.eigvalsh(m)
            probes.append(dict(step=d["step"], phase=d["phase"], scale=d["normalization_scale"],
                               eig_min=eig.min(), eig_max=eig.max(), V=np.mean(d["V"]), E=np.mean(d["E"])))
            matrices.append(m)
    if not probes:
        return
    steps = np.array([p["step"] for p in probes]); matrices = np.stack(matrices)
    np.savez_compressed(output/"FULL_M_PROBES.npz", step=steps, M=matrices,
                        phase=np.array([p["phase"] for p in probes]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for block in ("warmup", "feature", "metric"):
        items = [u for u in updates if u["block"] == block]
        if items:
            axes[0, 0].plot([u["step"] for u in items], [u["metrics"].get("loss", u["metrics"].get("total", np.nan)) for u in items], lw=.7, label=block)
    axes[0, 0].set_title("Actual per-update losses (no smoothing)"); axes[0, 0].legend()
    axes[0, 1].plot(steps, [p["eig_min"] for p in probes], label="min eigenvalue")
    axes[0, 1].plot(steps, [p["eig_max"] for p in probes], label="max eigenvalue")
    axes[0, 1].set_title("M spectrum across fixed training probes"); axes[0, 1].legend()
    axes[1, 0].plot(steps, np.linalg.norm((matrices-matrices[:1]).reshape(len(steps), -1), axis=1), label="distance to initial")
    axes[1, 0].plot(steps[1:], np.linalg.norm(np.diff(matrices, axis=0).reshape(len(steps)-1, -1), axis=1), label="change since previous probe")
    axes[1, 0].set_title("Real M movement; calibration boundary retained"); axes[1, 0].legend()
    axes[1, 1].plot(steps, [p["V"] for p in probes], label="mean V")
    axes[1, 1].plot(steps, [p["E"] for p in probes], label="mean physical E")
    axes[1, 1].set_title("Fixed training probes, not test trajectories"); axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Completed module updates"); ax.grid(alpha=.2)
    fig.savefig(output/"TRAINING_AND_M.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    values = matrices[:, 0].reshape(len(steps), -1).T
    view = ax.imshow(values, aspect="auto", origin="lower", extent=[steps[0], steps[-1] or 1, 0, values.shape[0]])
    ax.set(xlabel="Completed module updates", ylabel="All 256 entries of probe-0 M", title="Full matrix history (original samples; no monotonicity imposed)")
    fig.colorbar(view, ax=ax); fig.savefig(output/"M_ALL_ENTRIES.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-snapshot", type=Path, required=True)
    args = parser.parse_args()
    witness(args.process_snapshot)
    root, output = args.root.resolve(), args.output.resolve()
    final, audit = read(root/"FINAL_STATE.json"), read(root/"AUDIT.json")
    if final["status"] != "complete" or audit["passed"] is not True or (root/"WRITER.lock").exists():
        raise RuntimeError("Complete closed run with numerical audit required")
    output.mkdir(parents=True, exist_ok=False)
    result, protocol = read(root/"RESULTS.json"), read(root/"protocol.json")
    text = ["# RGB v4 GPU 管线结果", "", f"范围：`{final['scope']}`；来源：`{root}`。", "",
            "下列短验收或计时结果不构成正式性能结论；正式结果也只支持重建房间内的经验评价。", ""]
    if final["scope"] == "performance_profile":
        eta = estimate(root)
        (output/"ESTIMATE.json").write_text(json.dumps(eta, ensure_ascii=False, indent=2), encoding="utf8")
        text += [f"正式单 seed 的分项外推合计约 **{eta['base_minutes']:.1f} 分钟**，规划范围 **{eta['planning_range_minutes'][0]}–{eta['planning_range_minutes'][1]} 分钟**。",
                 "包含至少 129024 条新转移、14000 模块更新和扩大后的验证/测试；它不是纯渲染 FPS 换算。", "",
                 "|阶段|预计分钟|", "|---|---:|"]
        text += [f"|{key}|{value/60:.1f}|" for key, value in eta["stage_seconds"].items()]
    text += ["", "|工况/模型|成功数/总数|平均误差积分|平均输入平方积分|", "|---|---:|---:|---:|"]
    for key, summary in result["summary"].items():
        text.append(f"|{key}|{summary['successes']}/{summary['episodes']}|{summary['mean_integrated_error']:.6g}|{summary['mean_input_squared_integral']:.6g}|")
    text += ["", "数值复算通过；真实参数更新及冻结先于测试通过。保留真实 V 回升、失败和不利结果。", "",
             "公开 FastDepth、既有 44 房间、单种子、4DOF 外环、渲染像素变化等限制仍成立。输入平方不是实测能量；推理墙钟未注入物理 dt。", "",
             "[训练与 M 曲线](TRAINING_AND_M.png) · [全部 M 元素曲线](M_ALL_ENTRIES.png) · [完整探针数据](FULL_M_PROBES.npz)", "",
             "此自动表格不替代完整中文科学分析：还需比较同起点 selected/identity、共同成功时间、J 方向及 V/E 泛化、物理成功保持和失败原因。"]
    curves(root, output)
    content = "\n".join(text)
    (output/"REPORT.md").write_text(content, encoding="utf8")
    import markdown
    page = "<!doctype html><meta charset='utf-8'><style>body{font:17px/1.65 system-ui;max-width:1100px;margin:40px auto;color:#15263b}table{border-collapse:collapse}td,th{border:1px solid #ccd5df;padding:8px}img{max-width:100%}</style>"+markdown.markdown(content, extensions=["tables"])
    page += "<p><img src='TRAINING_AND_M.png'></p><p><img src='M_ALL_ENTRIES.png'></p>"
    (output/"REPORT.html").write_text(page, encoding="utf8")
    print(str(output/"REPORT.html"))


if __name__ == "__main__":
    main()
