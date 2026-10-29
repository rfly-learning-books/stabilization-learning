"""Read committed completed stages only, plot unsmoothed training/validation series."""
from __future__ import annotations
import html
from pathlib import Path
import numpy as np
from .common import read,write,sha


def build(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.grid':True,'grid.alpha':.2})
    dest=run.root/'figures';dest.mkdir(exist_ok=True);figures=[];summaries={}
    for group in ('push','so101'):
        fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for seed in run.config[group]['seeds']:
            stage=run.root/f'{group}_seed{seed}';rows=read(stage/'TRAINING.json')
            xx=[r['update'] for r in rows]
            ax[0,0].plot(xx,[r.get('mean_return',r.get('success_rate')) for r in rows],label=f'seed {seed}')
            ax[0,1].plot(xx,[r.get('existence',0.) for r in rows],label=f'seed {seed}')
            events=[read(p) for p in sorted((run.root/'events').glob(f'*_{group}_validation.json'))]
            ev=[r for r in events if r['seed']==seed]
            ax[1,0].plot([r['update'] for r in ev],[r['success_rate'] for r in ev],label=f'seed {seed}')
            ax[1,1].plot([r['update'] for r in ev],[r['mean_cumulative_error'] for r in ev],label=f'seed {seed}')
            summaries[f'{group}/{seed}']=read(stage/'FROZEN.json')
        for a,title in zip(ax.flat,['New rollout return / episode success','Existence penalty','Fixed validation success','Fixed validation error integral']):
            a.set(title=title,xlabel='PPO update');a.legend()
        fig.suptitle(f'{group}: raw complete training (no smoothing, no monotonicity imposed)')
        path=dest/f'{group}_training.png';fig.savefig(path,dpi=160);fig.savefig(path.with_suffix('.pdf'));plt.close(fig);figures.append(path)
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for seed in run.config['push']['seeds']:
        stage=run.root/f'push_seed{seed}'
        values=[(r['update'],np.array(r['M_probe_mean'])) for r in read(stage/'M_PROBES.json')]
        for a,(i,j) in zip(ax.flat,[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
            a.plot([v[0] for v in values],[v[1][i,j] for v in values],label=f'seed {seed}')
            a.set(title=f'M[{i+1},{j+1}]',xlabel='PPO update');a.legend()
    fig.suptitle('M at EVERY update: fixed three states per training shape, mean (not a stability guarantee)')
    path=dest/'push_M_vs_step.png';fig.savefig(path,dpi=160);fig.savefig(path.with_suffix('.pdf'));plt.close(fig);figures.append(path)
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for seed in run.config['push']['seeds']:
        rows=read(run.root/f'push_seed{seed}'/'M_PROBES.json')
        ax[0].plot([r['update'] for r in rows],[r['M_probe_distance'] for r in rows],label=f'seed {seed}')
        ax[1].plot([r['update'] for r in rows],[r['M_probe_step_change'] for r in rows],label=f'seed {seed}')
    for a,title in zip(ax,['Relative distance from random M','Relative M change since previous update']):
        a.set(title=title,xlabel='PPO update',ylabel='Frobenius ratio');a.legend()
    path=dest/'push_M_convergence.png';fig.savefig(path,dpi=160);fig.savefig(path.with_suffix('.pdf'));plt.close(fig);figures.append(path)
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for seed in run.config['rgb']['seeds']:
        rows=read(run.root/f'rgb_seed{seed}'/'TRAINING.json')
        for a,phases,key,title in [(ax[0,0],['warmup','representation'],'reconstruction','RGB reconstruction'),
            (ax[0,1],['critic_prefit','representation'],'derivative_loss','Signed derivative supervision'),
            (ax[1,0],['metric'],'total','M objective'),(ax[1,1],['metric'],'real_descent','Recorded successor V penalty')]:
            rs=[r for r in rows if r['phase'] in phases and key in r]
            a.plot([r['update'] for r in rs],[r[key] for r in rs],label=f'seed {seed}')
            a.set(title=title,xlabel='Cumulative stage gradient update');a.legend()
        summaries[f'rgb/{seed}']=read(run.root/f'rgb_seed{seed}'/'FROZEN.json')
    path=dest/'rgb_training.png';fig.savefig(path,dpi=160);fig.savefig(path.with_suffix('.pdf'));plt.close(fig);figures.append(path)
    table=[];timing=[];failures=[];rgbtable=[]
    for seed in run.config['push']['seeds']:
        result=read(run.root/f'push_seed{seed}/test/RESULTS.json')
        for method,r in result.items():
            table.append(f'<tr><td>Push/{seed}</td><td>{method}</td><td>{r["success_rate"]:.1%}</td><td>{r["mean_cumulative_error"]:.5g}</td></tr>')
    for seed in run.config['so101']['seeds']:
        r=read(run.root/f'so101_seed{seed}/test/RESULTS.json')
        for method in sorted(set(x['variant'] for x in r['arm'])):
            subset=[x for x in r['arm'] if x['variant']==method]
            table.append(f'<tr><td>SO101 actuated/{seed}</td><td>{method}</td><td>{np.mean([x["success"] for x in subset]):.1%}</td><td>{np.mean([x["integrated_error"] for x in subset]):.5g}</td></tr>')
            good=[x['sim_time_s'] for x in subset if x['success']]
            goodtext='—' if not good else f'{np.mean(good):.3f}'
            timing.append(f'<tr><td>{seed}</td><td>{method}</td><td>{np.mean([x["sim_time_s"] for x in subset]):.3f}</td><td>{goodtext}</td></tr>')
            reasons={reason:sum(x['reason']==reason for x in subset) for reason in sorted({x['reason'] for x in subset})}
            failures.append(f'<p>SO101 {seed} {method}：{html.escape(str(reasons))}</p>')
            novel=[x for x in subset if x['heldout']]
            if novel:
                table.append(f'<tr><td>SO101 留出几何/{seed}</td><td>{method}</td><td>{np.mean([x["success"] for x in novel]):.1%}</td><td>{np.mean([x["integrated_error"] for x in novel]):.5g}</td></tr>')
        # Compare duration on exactly the same successful initial states.
        baseline={(x['shape'],x['seed']):x for x in r['arm'] if x['variant']=='A_frozen_original'}
        for method in ('B_frozen_time','C_time_selected'):
            pairs=[(baseline[(x['shape'],x['seed'])],x) for x in r['arm'] if x['variant']==method
                   and x['success'] and baseline[(x['shape'],x['seed'])]['success']]
            if pairs:
                delta=float(np.mean([b['sim_time_s']-a['sim_time_s'] for a,b in pairs]))
                failures.append(f'<p>种子 {seed}：{method} 相对 A，共同成功 {len(pairs)} 对，平均时间差 {delta:+.3f} 秒（负值为更快）。其余失败样本仍包含在上表。</p>')
            else:failures.append(f'<p>种子 {seed}：{method} 与 A 没有共同成功样本，不能给出成功任务的配对加速结论。</p>')
    for seed in run.config['rgb']['seeds']:
        result=read(run.root/f'rgb_seed{seed}/RESULTS.json')
        for method,r in result['summaries'].items():
            rgbtable.append(f'<tr><td>{seed}</td><td>{method}</td><td>{r["success_rate"]:.1%}</td><td>{r["mean_integrated_error"]:.5g}</td><td>{r["mean_actual_decay_violation_fraction"]:.1%}</td></tr>')
    images=''.join(f'<figure><img src="figures/{p.name}"></figure>' for p in figures)
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>从零训练与 SO101 仿真</title>
    <style>body{{max-width:1150px;margin:40px auto;font:17px/1.75 system-ui;color:#18232d;padding:20px}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px}}</style>
    <h1>从零训练与 SO101 仿真</h1><p>Push 的 Actor/Critic 和 RGB 的共享编码器/M/Critic 在本运行内随机初始化，优化器为空，训练轨迹新采样。
    SO101 的初始权重来自对应种子本轮 Push 验证选模。所有方法都保留失败和不利结果。</p>
    <p>Push 是已标定能力矩阵条件下的留出形状测试。SO101 留出几何不用于 M 优化，但提供新采的接触能力标定。
    RGB 使用既有 44 个重建房间几何、新初态和新图像转移；FastDepth 是固定公共预训练网络。</p>
    <p>时间优化先在降阶响应模型中完成，SO101 部署评价由关节驱动 MuJoCo 执行，不是真机部署。曲线来自原始记录，未强制单调下降。
    同起点评测用于配对比较；重复训练种子不增加独立初态数；跨种子标准差不是置信区间。没有新增全局或非线性域稳定证明。</p>
    <table><tr><th>任务/种子</th><th>方法</th><th>成功保持率</th><th>误差积分</th></tr>{''.join(table)}</table>
    <h2>SO101 实测仿真时间（秒）</h2><p>“成功样本时间”受各方法成功集合影响，不能单凭这一列认定配对加速；原始 JSON 保留同起点配对记录。</p>
    <table><tr><th>种子</th><th>方法</th><th>全部样本</th><th>仅成功样本</th></tr>{''.join(timing)}</table>{''.join(failures)}
    <h2>RGB 独立测试</h2><table><tr><th>种子</th><th>方法/对象</th><th>成功率</th><th>误差积分</th><th>下降违反率</th></tr>{''.join(rgbtable)}</table>{images}</html>'''
    (run.root/'REPORT.html').write_text(body,encoding='utf-8')
    write(run.root/'FIGURES.json',[dict(file=str(p.relative_to(run.root)),sha256=sha(p)) for p in figures])
    write(run.root/'SUMMARY.json',summaries)
