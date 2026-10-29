"""Project the fixed historical single-seed schedule from bounded timings."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import math
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root')
    args=parser.parse_args()
    root=Path(args.root).resolve()
    if (root/'REPORT.html').exists():
        raise FileExistsError('Preserve the existing estimate')
    final=json.loads((root/'FINAL_STATE.json').read_text(encoding='utf8'))
    if final['status']!='complete' or final['phases']!=30:
        raise RuntimeError('Require all thirty complete timing windows')
    profile=json.loads((root/'PROFILE.json').read_text(encoding='utf8'))
    old=json.loads((root/'HISTORICAL_TIMING.json').read_text(encoding='utf8'))
    groups=defaultdict(list)
    for row in profile['phases']:
        groups[(row['pipeline'],row['mode'],row['batch_size'])].append(row)
    summary=[]
    for (pipeline,mode,batch),rows in groups.items():
        seconds=sum(r.get('seconds_with_archive',r['seconds']) for r in rows)
        samples=sum(r['samples'] for r in rows)
        telemetry_count=sum(r['telemetry']['samples'] for r in rows)
        rates=[r.get('samples_per_second_with_archive',r['samples_per_second']) for r in rows]
        summary.append(dict(pipeline=pipeline,mode=mode,batch_size=batch,
            repeats=len(rows),seconds_per_operation=seconds/sum(r['iterations'] for r in rows),
            samples_per_second=samples/seconds,repeat_sample_rate_sd=statistics.stdev(rates),
            gpu_mean_percent=sum(r['telemetry']['gpu']*r['telemetry']['samples'] for r in rows)/telemetry_count,
            setup_seconds=rows[0].get('setup_seconds',0.)))
    index={(r['pipeline'],r['mode'],r['batch_size']):r for r in summary}
    times=old['training']['event_gap_seconds']
    update_old=times['warmup']+times['feature']+times['metric']+times['fixed_training_probe']
    other_old=old['training']['total_seconds']-update_old-times['replay_episode']-times['evaluation_episode']
    old_total=old['test']['end']-old['training']['start']
    between=old['test']['start']-old['training']['end']
    eval8=index[('gpu','evaluation',8)]
    eval16=index[('gpu','evaluation',16)]
    # Full declared horizons, even though old trajectories sometimes ended early.
    validation=5*(8*100/eval8['samples_per_second']+eval8['setup_seconds'])
    testing=8*(16*120/eval16['samples_per_second']+eval16['setup_seconds'])
    predictions=[]
    for pipeline in ('gpu_host_replay','gpu_resident'):
        components=dict(
            warmup=1000*index[(pipeline,'warmup',4)]['seconds_per_operation'],
            joint=6500*index[(pipeline,'joint',4)]['seconds_per_operation'],
            collection_and_archive=times['replay_episode'],
            validation=validation,testing=testing,other=other_old+between)
        seconds=sum(components.values())
        # An explicit scheduling allowance, not a statistical confidence interval.
        low=5*math.floor((seconds/60+5)/5)
        high=5*math.ceil((seconds/60+10)/5)
        predictions.append(dict(pipeline=pipeline,components_seconds=components,
            component_sum_minutes=seconds/60,planning_range_minutes=[low,high],
            reduction_percent_range=[100*(1-high*60/old_total),100*(1-low*60/old_total)],
            speedup_range=[old_total/(high*60),old_total/(low*60)]))
    result=dict(historical_minutes=old_total/60,historical_training_minutes=old['training']['total_seconds']/60,
                historical_testing_minutes=old['test']['total_seconds']/60,
                measured_windows=summary,predictions=predictions,
                fixed_budget=dict(batch_size=4,warmup=1000,feature_updates=6500,metric_updates=6500,
                                  validations=5,validation_cases=8,validation_horizon=100,
                                  test_models=4,test_plants=2,test_cases=16,test_horizon=120),
                planning_allowance='Add 5-10 minutes to component projection and round outwards to five-minute boundaries; not a confidence interval',
                assumptions=['Measured optimizer/probe/replay work; no reduction in update counts',
                    'Keep historical collection/archive wall time unchanged',
                    'Use measured heterogeneous training-room full-trace evaluation throughput at original maximum horizons',
                    'GPU resident replay requires production collector/replay/schedule integration',
                    'Predictions are fixed-workload estimates, not time to reach equal success or error'])
    (root/'ESTIMATE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    previous=[update_old/60,times['replay_episode']/60,times['evaluation_episode']/60,
              old['test']['total_seconds']/60,(other_old+between)/60]
    columns=[previous]
    for prediction in predictions:
        c=prediction['components_seconds']
        columns.append([(c['warmup']+c['joint'])/60,c['collection_and_archive']/60,
                        c['validation']/60,c['testing']/60,c['other']/60])
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    bottom=np.zeros(3)
    for i,(label,color) in enumerate(zip(('Updates + replay + probe logs','Collection + archive','Validation','Four-model testing','Other'),
                                       ('#087f8c','#b0bac5','#d99b55','#795889','#666666'))):
        values=[column[i] for column in columns]
        ax.bar(range(3),values,bottom=bottom,color=color,label=label)
        bottom+=values
    for i,total in enumerate(bottom):
        ax.text(i,total+1.5,f'{total:.1f} min',ha='center')
    ax.set_xticks(range(3),['Historical v2\nmeasured','GPU + host replay\ncomponent estimate',
                           'GPU + resident replay\ncomponent estimate'])
    ax.set_ylabel('Wall-clock minutes per seed'); ax.set_ylim(0,max(bottom)*1.2)
    ax.set_title('Same budget: batch 4, 1,000 warmups + 6,500 feature/M pairs')
    ax.legend(loc='upper right',fontsize=8); ax.spines[['top','right']].set_visible(False)
    fig.savefig(root/'time_estimate.png',dpi=150); plt.close(fig)
    names={'v2_reference':'原 v2 管路复测','v3_cpu':'v3 + CPU 图像几何',
           'gpu_host_replay':'GPU 几何 + CPU 有界回放','gpu_resident':'GPU 几何 + 常驻原始回放'}
    cells=[]
    for pipeline in names:
        w=index[(pipeline,'warmup',4)]; j=index[(pipeline,'joint',4)]
        cells.append(f'<tr><td>{names[pipeline]}</td><td>{w["seconds_per_operation"]:.4f}</td>'
                     f'<td>{j["seconds_per_operation"]:.4f}</td><td>{j["samples_per_second"]:.1f}</td>'
                     f'<td>{j["gpu_mean_percent"]:.1f}%</td></tr>')
    forecast=[]
    for p in predictions:
        lo,hi=p['planning_range_minutes']; reduction=p['reduction_percent_range']
        forecast.append(f'<li><strong>{names[p["pipeline"]]}：建议按 {lo}–{hi} 分钟/seed 预留</strong>；'
                        f'分项相加为 {p["component_sum_minutes"]:.1f} 分钟，较历史总时间减少约 '
                        f'{reduction[0]:.0f}%–{reduction[1]:.0f}%。</li>')
    resident_mib=profile['resident_raw_bytes']/1024**2
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RGB 原训练流程的 GPU 耗时估算</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.7;background:#f6f8fa;color:#233140;margin:0}}
main{{max-width:1000px;margin:auto;padding:32px 22px}}h1{{line-height:1.3;font-size:28px}}h2{{font-size:21px;margin-top:30px}}
.lead{{background:#e5f0f1;padding:15px 20px;border-left:4px solid #087f8c}}table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border-bottom:1px solid #dde4e8;padding:10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}a{{color:#087f8c}}
</style><main><h1>按上一轮完整预算，RGB 训练能缩短多少</h1>
<p>2026-10-27 · Windows / RTX 4070 Ti · 单 seed 固定工作量预测</p>
<div class="lead">历史 RGB 训练与冻结用了 {result['historical_training_minutes']:.1f} 分钟，四模型 P0/P1 测试用了 {result['historical_testing_minutes']:.1f} 分钟；包含阶段间切换，总计 <strong>{old_total/60:.1f} 分钟</strong>。下方预测来自新增分项实测，尚非 GPU 后端完成整轮长训的实测。</div>
<ul>{''.join(forecast)}</ul><img src="time_estimate.png" alt="历史实际时间与固定预算预测时间分解">
<h2>保持了哪些工作量</h2><p>batch=4；1,000 次 AE 预热；6,500 对特征/M 更新，即 14,000 次模块更新；5 次验证，每次 8 个起点、最多 100 步；最终四模型 × P0/P1 × 16 个起点，每回合最多 120 步。回放沿用上一轮 49 个训练回合、2,926 个转移作为只读计时输入，没有把旧数据用于新的正式训练。</p>
<h2>补测的学习链路</h2><p>每项预热后计时 3 次，每次约 10 秒。包含场景均衡抽样、回放访问、必要的图像几何、完整反传/Adam，以及每次更新后的原子日志和固定 M 探针。联合更新两次使用同一批，样本只计一次。</p>
<table><thead><tr><th>管路</th><th>预热秒/次</th><th>联合秒/对</th><th>联合样本/秒</th><th>联合平均 GPU</th></tr></thead><tbody>{''.join(cells)}</tbody></table>
<p>GPU 常驻原始回放占 {resident_mib:.1f} MiB，保留 uint8 RGB 和真实后继，不缓存随 phi 更新的特征/L。CPU 有界回放保留分块读取、解压与原有缓存限制。操作系统文件缓存未清空，因此不是冷盘随机读取极限测量。</p>
<h2>如何推算总时间</h2><p>总时间 = 1,000 × 实测预热秒数 + 6,500 × 实测联合更新秒数 + 采样归档 + 验证 + 最终测试 + 初始化/检查点等开销。</p>
<p>采样归档保守保留历史 {times['replay_episode']/60:.1f} 分钟，未假设每次仅一个新回合也能获得 16 路加速。验证和测试补测了完整后继 V、诊断传回与压缩轨迹保存；分别使用 8/16 个不同训练房间模拟同规模异构调度，吞吐为 {eval8['samples_per_second']:.1f}/{eval16['samples_per_second']:.1f} 个有效转移/秒。未重跑留出策略评价。预测采用完整回合上限，并计入每批场景构建时间。此项新测量使用 P0；P1 采用相同计算量估计，依据是此前 P0/P1 完整闭环吞吐接近。</p>
<p>建议预留时间在分项结果上增加 5–10 分钟，再向外取整到 5 分钟，用于完整 Runner 接入后的调度、变长回合和归档波动。这是明确的工程余量，不是统计置信区间。GPU 数字是整卡活动采样，不是理论算力占用。</p>
<h2>范围与限制</h2><p>这些是完成相同更新/评价预算的时间，不能承诺达到相同成功率所需的时间。GPU 渲染存在图像分布差异；当前任务仍是 4DOF 速度外环。批量 16 的样本吞吐更高，但保持 6,500 对更新会使用四倍训练样本，不能把它算作本次等预算加速。</p>
<p>本次使用临时模型副本；预热用未标定的 initial，联合更新用已标定的工程模型；完成的计时阶段均还原模型。首次计时选用已标定模型，触发预热保护；第二次完成 18 个窗口后，常驻回放适配器缺少行为版本字段，触发批次校验。失败记录和当时脚本保留在同级 *_a、*_b 目录。修正只发生在独立计时脚本，*_c 只补测缺少的 12 个窗口；本报告核验并合并共 30 个完整窗口，不重复已完成对照，不改旧失败状态。</p>
<p><a href="PROFILE.json">完整新计时与 GPU 采样</a> · <a href="ESTIMATE.json">计算明细</a> · <a href="HISTORICAL_TIMING.json">历史事件耗时</a> · <a href="PROTOCOL.json">计时协议</a></p></main></html>'''
    (root/'REPORT.html').write_text(body,encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
