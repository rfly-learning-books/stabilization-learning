"""Generate a factual HTML training/evaluation report from this run's evidence."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def lines(path):
    if not path.exists():return []
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:out.append(json.loads(line))
        except json.JSONDecodeError:continue
    return out

def report(out):
    out=Path(out);dest=out/'report';dest.mkdir(parents=True,exist_ok=True)
    protocol=read(out/'protocol.json',{});status=read(out/'status.json',{})
    # A crash may leave log lines beyond the checkpoint: de-duplicate by update key.
    updates={}
    for row in lines(out/'training.jsonl'):updates[(row['phase'],row['round'],row['update'])]=row
    data=list(updates.values());summaries=[]
    for path in sorted((out/'evaluation').glob('*/summary.json')):
        summaries.append(read(path))
    blocks=[]
    if data:
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for phase in ('warmup','representation','metric'):
            rows=[r for r in data if r['phase']==phase]
            if not rows:continue
            x=np.arange(len(rows))*25
            for ax,key in zip(axes.ravel(),('total','reconstruction','gradient_norm','real_descent')):
                v=[r.get(key,np.nan) for r in rows]
                ax.plot(x,v,label=phase);ax.set_title(key);ax.grid(alpha=.25);ax.legend()
                if np.any(np.asarray(v)>0):ax.set_yscale('symlog',linthresh=1e-4)
        fig.tight_layout();fig.savefig(dest/'training.png',dpi=150);plt.close(fig)
        blocks.append('<h2>Training</h2><img src="training.png">')
    table=[]
    for result in summaries:
        for method,agg in result.get('summaries',{}).items():
            table.append(dict(tag=result['tag'],method=method,complete=result['complete'],episodes=agg['episodes'],
                success_rate=agg.get('success_rate'),macro=agg.get('macro_scene_success'),worst=agg.get('worst_scene_success'),
                final_cm=100*agg.get('mean_final_position_m',0),yaw_deg=agg.get('mean_final_yaw_deg'),
                V_up=agg.get('mean_actual_V_increase_fraction'),action_variation=agg.get('mean_action_variation'),
                difficulty=agg.get('difficulty_success',{}),status=agg.get('status_counts',{})))
    if table:
        header='<tr><th>Evaluation</th><th>Method/plant</th><th>n</th><th>Success</th><th>Scene macro</th><th>Worst</th><th>Final cm</th><th>Final yaw deg</th><th>V rises</th></tr>'
        rows=[]
        for t in table:
            fmt=lambda v:f'{100*v:.1f}%' if v is not None else 'N/A'
            rows.append(f'<tr><td>{html.escape(t["tag"])} {"complete" if t["complete"] else "PARTIAL"}</td><td>{html.escape(t["method"])}</td>'
                f'<td>{t["episodes"]}</td><td>{fmt(t["success_rate"])}</td><td>{fmt(t["macro"])}</td><td>{fmt(t["worst"])}</td>'
                f'<td>{t["final_cm"]:.2f}</td><td>{t["yaw_deg"]:.2f}</td><td>{fmt(t["V_up"])}</td></tr>')
        blocks.append('<h2>Observed closed-loop results</h2><table>'+header+''.join(rows)+'</table>')
    tests=[r for r in summaries if r['tag']=='fixed_test']
    message='固定测试集：未测试。'
    if tests:message=f'固定测试集：{"已完成" if tests[0]["complete"] else "尚未完成"}，已记录 {tests[0]["episodes"]}/{tests[0]["planned"]} 回合。'
    representative=[];keys=set()
    for path in sorted((out/'evaluation').glob('**/*.json')):
        if path.name=='summary.json':continue
        row=read(path,{})
        if not row.get('video'):continue
        key=(row.get('tag'),row.get('method'),row.get('difficulty'),row.get('success'))
        if key in keys:continue
        keys.add(key)
        if row.get('tag') not in ('fixed_test','full_val'):continue
        representative.append((path,row))
    if representative:
        blocks.append('<h2>Representative successes and failures</h2>')
        for path,row in representative[:18]:
            arpath=path.with_suffix('.npz')
            if arpath.exists():
                with np.load(arpath) as a:
                    offset=a['offset'];V=a['V'];e=a['features'];m=a['M']
                E=np.sum(offset[:,:3]**2,axis=1)+(offset[:,3]/np.deg2rad(25))**2
                t=np.arange(len(V))*.1
                fig,ax=plt.subplots(1,3,figsize=(13,3))
                ax[0].plot(t,100*np.linalg.norm(offset[:,:3],axis=1));ax[0].set_title('Position error (cm)')
                ax[1].plot(t,V,label='V');ax[1].plot(t,E,label='physical E');ax[1].legend();ax[1].set_title('Fixed-network energy / physical error')
                ax[2].plot(t,np.linalg.norm(e,axis=1),label='||e||');ax[2].plot(t,np.sqrt(np.maximum(2*V,0)),label='||M^1/2 e||');ax[2].legend()
                for a0 in ax:a0.set_xlabel('time (s)');a0.grid(alpha=.25)
                fig.tight_layout();image=dest/f'{path.parent.name}_{path.stem}.png';fig.savefig(image,dpi=120);plt.close(fig)
                title=html.escape(f'{row["scene_id"]} {row["method"]} {row["plant"]} {row["difficulty"]}: {row["status"]}')
                blocks.append(f'<p>{title}</p><img src="{image.name}"><video controls preload="none" src="../{row["video"]}"></video>')
    complete=read(out/'complete.json')
    audit=read(out/'engineering_audit.json',{})
    text=f'''<!doctype html><meta charset="utf-8"><title>Goal-fixed Phi / M training</title>
<style>body{{max-width:1300px;margin:32px auto;font:16px/1.6 system-ui;padding:20px;color:#172132}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccd3de}}img{{max-width:100%}}video{{width:650px;max-width:100%}}pre{{white-space:pre-wrap;background:#f2f5fa;padding:15px}}</style>
<h1>多环境固定参考 J*：AE 主导的 Phi 与目标条件 M 联合训练</h1>
<p>{message}</p><p>运行状态：{html.escape(str(status.get('stage','preparing')))}。工程验收：{audit.get('passed','尚未完成')}。</p>
<p>16维RGB；冻结FastDepth MobileNetV1/NNConv5公开预训练复现版。M只依赖参考特征，回合内固定；Cholesky参数化，trace=16，0.1I ≤ M ≤14.5I。位置/偏航真值只参与训练标签、终止和评价。</p>
<p>本报告只呈现已记录的结果。结构约束及估计模型下降不等于真实系统的Lyapunov证明；泛化范围是同类生成器下未见布局和目标。</p>
{''.join(blocks)}
<h2>Limits and interpretation</h2><p>AE为已有多场景权重压缩到16维后继续训练，包含平滑编码器修改，属于热启动。FastDepth为公开复现权重，原作者下载站当前不可达；来源和哈希见protocol.json。初始AE、单位M与联合方法使用相同固定测试起点。P1含隐藏执行器状态，仅作压力测试。</p>
<p>如果成功率提高但真实V仍频繁上升，应描述为性能改善和候选证书，不能称为已证明镇定证书。数据中出现的碰撞、越界、超时均保留；测试结果不用于模型选择。</p>
<h2>Latest saved state</h2><pre>{html.escape(json.dumps(status,ensure_ascii=False,indent=2))}</pre>
<p><a href="../protocol.json">配置、来源与哈希</a> · <a href="../engineering_audit.json">工程验收</a> · <a href="summary.json">机器可读结果</a></p>'''
    (dest/'REPORT.html').write_text(text,encoding='utf-8')
    (dest/'summary.json').write_text(json.dumps(dict(complete=bool(complete),test_status=message,results=table,
        engineering_audit_passed=audit.get('passed'),saved_counts=status.get('counts')),ensure_ascii=False,indent=2),encoding='utf-8')
    return dest/'REPORT.html'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();print(report(args.output))
