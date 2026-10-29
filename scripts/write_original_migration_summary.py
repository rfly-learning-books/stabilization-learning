"""Summarize completed evidence without changing the frozen evaluation."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    a = run / 'analysis'
    data = json.loads((a / 'RESULTS.json').read_text(encoding='utf-8'))
    audit = json.loads((a / 'FINAL_AUDIT.json').read_text(encoding='utf-8'))
    assert audit['passed'] and (run / 'complete.json').is_file()
    labels = {'migrated_best': '新目录重新训练：验证最佳', 'historical_best': '历史最佳 M',
              'identity': '单位 M', 'initial': '原始未训练 M', 'migrated_final': '重新训练：最终 M'}
    summary = data['new_start_summary']
    # Preserve the evaluator's original image and produce a readable supplement.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    records = {}
    for line in (run / 'history.jsonl').read_text().splitlines():
        row = json.loads(line)
        records[row['update']] = row
    collected = 0
    points = []
    for update in sorted(records):
        row = records[update]
        collected += row['collection']['transitions']
        if 'hard_eval' in row:
            points.append((collected / 1000, update, row['hard_eval']))
    selected = next(p for p in points if p[1] == data['training']['best_update'])
    initial = data['development_summary']['initial']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    specs = [('success_rate', 'Development success (%)', 100),
             ('mean_switches', 'Mean mode switches', 1),
             ('mean_cumulative_error', 'Mean integrated pose error', 1)]
    for ax, (key, label, scale) in zip(axes, specs):
        ax.plot([p[0] for p in points], [p[2][key]*scale for p in points], lw=1.4, color='#246e8e', label='Each development evaluation')
        ax.axhline(initial[key]*scale, color='#92999f', ls='--', lw=1, label='Untrained initialization')
        ax.scatter([selected[0]], [selected[2][key]*scale], color='#b05328', s=42, zorder=4, label='Selected by original rule')
        ax.set_xlabel('Collected physical transitions (thousands)')
        ax.set_ylabel(label)
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.grid(alpha=.2)
    axes[0].set_ylim(89, 101.5)
    axes[0].set_yticks([90, 95, 100])
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(a / 'learning_verified.png', dpi=160)
    plt.close(fig)
    paragraphs = [
        '结论：Push Any Shape 的原 Full-M 降阶数值协议已成功迁移，完整重新训练保留了原效果。',
        f"训练完成 {data['training']['updates']} 轮、{data['training']['transitions']:,} 条有效物理转移。按原开发集规则选出的最佳模型仍为第 {data['training']['best_update']} 轮。",
        '1000 轮日志的可比采样、损失和验证字段没有差异；最佳及最终 Actor、Critic、优化器和随机状态与历史检查点逐张量一致。310 个新起点的全部轨迹字段与历史最佳模型一致。检查点容器文件哈希不相同；选模分数元数据仅有 5.55×10⁻¹⁷ 的末位差异，未改变所选模型。',
    ]
    headers = ['方法', '新起点成功', '累计误差', '失败封顶到达秒', '平均切换', '终端位置 mm', '终端偏航 °']
    rows = []
    for method, name in labels.items():
        s = summary[method]
        rows.append([name, f"{s['successes']}/{s['episodes']} ({s['success_rate']:.2%})",
                     f"{s['mean_cumulative_error']:.6f}", f"{s['mean_capped_arrival_s']:.3f}",
                     f"{s['mean_switches']:.3f}", f"{s['mean_position_error_m']*1000:.3f}",
                     f"{s['mean_abs_yaw_error_deg']:.3f}"])
    findings = [
        '相对单位 M：37 个起点新增成功，没有反向丢失成功的起点；270 个共同成功配对中平均快 4.602 秒。平均累计误差降低约 19.8%。',
        '相对原始未训练 M：成功数从 304 增至 307，共同成功配对平均快 0.441 秒，平均切换减少；累计误差从 0.272676 增至 0.328873。学习不是所有指标都更优。原选模规则依次优先成功率、失败误差、切换、无动作和累计误差，不能解释为只优化累计误差。',
        '单位 M 的终端位置误差较小，但偏航误差较大且超时更多，不能仅凭位置误差判断伺服质量。',
        '迁移最佳 M 有 3 个回合在 20 秒时限内未成功，单位 M 有 40 个超时；初始 M 有 6 个超时。失败均按 20 秒计入到达时间，不把提前结束当作快速收敛。',
        '迁移最佳 M 的 4,653 个有效物理步未发现超过 10⁻⁷ 的实际 V 回升；该统计包含目标保持步。所有方法在本批测试中的回升计数均为零。证书下降本身不等同于有限时间成功或全局收敛保证。',
        '独立积分核验的最大位姿步进误差为 2.22×10⁻¹⁶；矩阵正定、trace、Cholesky 构造、成功保持、累计误差和视频解码均通过检查。',
        '原始未训练 M 在 32 个开发案例上已经全部成功，因此成功率训练曲线保持平坦。切换数和累计误差随训练波动；下图展示原始验证点和原规则选出的第 340 轮，不用测试结果选择曲线或模型。',
    ]
    scope = [
        '新起点测试为 31 个已训练、已标定形状 × 10 个固定新种子，每种方法 310 回合。T 形由原程序固定为同一个回归案例，排除重复计数并保留在 32 案例开发回归中。五种方法共计 1,550 个有效新起点测试，另各有 32 个开发案例。',
        '这是同一标定库内的新起点测试、单训练种子和降阶数值推动模型；不包含未见形状、SO101 关节接触及时间感知执行层。',
        '原多旋翼室内视觉伺服严格复现仍缺 6 个原源码模块和最早 AE 文件，清单及预期哈希保存在 migration/original_protocol_assets.json。本次结论限于 Push。',
        '工程记录：v1 的协议序列化恢复问题保留在原目录；v2 第 835 轮发生状态文件占用错误，检查点和日志已提交，核验后按原协议恢复。未改动原训练核心、未删除历史实验、未根据测试选择模型。',
    ]
    md = ['# 原协议迁移：Push Any Shape 训练与测试', '']
    for p in paragraphs: md += [p, '']
    md += ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    md += ['| ' + ' | '.join(row) + ' |' for row in rows]
    md += ['', '到达时间包含成功保持；失败按 20 秒封顶。累计误差为原归一化位姿误差沿有效物理步的积分。', '', '## 性能与证书', '']
    for p in findings: md += ['- ' + p, '']
    md += ['## 适用范围与记录', '']
    for p in scope: md += ['- ' + p, '']
    md += ['## 训练曲线', '', '![开发集训练曲线](learning_verified.png)', '',
           '[统计与配对数据](RESULTS.json) · [独立最终核验](FINAL_AUDIT.json) · [开发回归视频](../final_best/hard_numeric.mp4)', '']
    (a / 'MIGRATION_SUMMARY.md').write_text('\n'.join(md), encoding='utf-8')
    esc = html.escape
    table = '<table><thead><tr>' + ''.join('<th>' + esc(v) + '</th>' for v in headers) + '</tr></thead><tbody>'
    table += ''.join('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'
    body = '<h1>原协议迁移：Push Any Shape</h1>' + ''.join('<p>' + esc(p) + '</p>' for p in paragraphs)
    body += '<div class="scroll">' + table + '</div><p class="note">到达时间含成功保持，失败按 20 秒封顶。累计误差为原归一化位姿误差沿有效物理步的积分。</p>'
    for title, content in [('性能与证书', findings), ('适用范围与记录', scope)]:
        body += '<h2>' + title + '</h2><ul>' + ''.join('<li>' + esc(p) + '</li>' for p in content) + '</ul>'
    body += '<h2>训练曲线</h2><img src="learning_verified.png" alt="随实际采集步数变化的开发集成功率、切换和累计误差"><h2>32 个开发回归案例</h2><video controls preload="metadata" poster="../final_best/overview.jpg" src="../final_best/hard_numeric.mp4"></video><p><a href="RESULTS.json">配对数据</a> · <a href="FINAL_AUDIT.json">独立最终核验</a> · <a href="MIGRATION_SUMMARY.md">Markdown 文档</a></p>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Push 原协议迁移结果</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#24384c;background:#fafbfd}h1{font-size:30px}h2{margin-top:34px;color:#175b73}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:12px;border-bottom:1px solid #dce4eb;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}tr:first-child td{background:#eaf4f3}.scroll{overflow:auto}li{margin:12px 0}img,video{max-width:100%;border:1px solid #dce4eb;border-radius:8px;background:white}.note{color:#526679;font-size:14px}a{color:#126880}</style><main>' + body + '</main></html>'
    (a / 'MIGRATION_SUMMARY.html').write_text(page, encoding='utf-8')
    print(a / 'MIGRATION_SUMMARY.html')


if __name__ == '__main__':
    main()
