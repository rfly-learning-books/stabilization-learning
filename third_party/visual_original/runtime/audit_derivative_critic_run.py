"""Independent, CPU-only audit of immutable derivative-Critic run evidence."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'2')
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def make_report(out,result,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'analysis'
    left=records['fixed_test','learned','P0']
    right=records['fixed_test','identity_m','P0']
    cases=[]
    for title,condition in [('M succeeds; identity fails',lambda a,b:a['success'] and not b['success']),('M fails; identity succeeds',lambda a,b:b['success'] and not a['success']),('Both succeed',lambda a,b:a['success'] and b['success'])]:
        candidates=[i for i in sorted(left) if condition(left[i],right[i])]
        if candidates:
            i=next((i for i in candidates if left[i]['difficulty']=='medium'),candidates[0])
            cases.append((title,i))
    fig,axes=plt.subplots(len(cases),4,figsize=(16,3.1*len(cases)),squeeze=False)
    for row,(title,i) in enumerate(cases):
        for method,label,color in [('learned','Learned M','tab:blue'),('identity_m','Identity M','tab:orange')]:
            r=records['fixed_test',method,'P0'][i]
            with np.load(out/'evaluation/fixed_test'/f'{method}_P0_{i:04d}.npz') as a:
                off=a['offset'];v=a['V'];u=a['action'];t=np.arange(len(v))*.1
                axes[row,0].plot(t,100*np.linalg.norm(off[:,:3],axis=1),label=label,color=color)
                axes[row,1].plot(t,np.abs(np.degrees(off[:,3])),color=color)
                axes[row,2].plot(t,v,color=color)
                axes[row,3].plot(np.arange(len(u))*.1,np.linalg.norm(u,axis=1),color=color)
        for col,name in enumerate(['Position error (cm)','Yaw error (deg)','Actual RGB certificate V','Normalized command norm']):
            axes[row,col].set_title(name);axes[row,col].set_xlabel('Time (s)');axes[row,col].grid(alpha=.25)
        axes[row,0].set_ylabel(title+'\n'+left[i]['scene_id']+f' / start {i}')
        axes[row,0].legend()
    fig.tight_layout();fig.savefig(dest/'paired_trajectories.png',dpi=150);plt.close(fig)
    table=[]
    for method in ['learned','identity_m','shadow_ae_identity','initial_ae_identity']:
        for plant in ['P0','P1']:
            rs=list(records['fixed_test',method,plant].values())
            n=sum(r['success'] for r in rs)
            table.append(f'<tr><td>{method}</td><td>{plant}</td><td>{n}/{len(rs)} ({100*n/len(rs):.1f}%)</td><td>{np.mean([r["integrated_error"] for r in rs]):.3f}</td><td>{100*np.mean([r["actual_V_increase_fraction"] for r in rs]):.1f}%</td></tr>')
    pairtable=[]
    for r in result['paired']:
        if r['tag']!='fixed_test':continue
        pairtable.append(f'<tr><td>{r["plant"]}</td><td>{r["method"]} / {r["baseline"]}</td><td>{r["method_only"]}</td><td>{r["baseline_only"]}</td><td>{r["common_successes"]}</td></tr>')
    videohtml=[]
    for success in [True,False]:
        choices=[r for r in left.values() if r.get('video') and r['success']==success]
        if choices:
            r=next((r for r in choices if r['difficulty']=='large'),choices[0])
            source='../'+r['video'].replace('\\','/')
            videohtml.append(f'<p>{r["scene_id"]}, {r["difficulty"]}, {r["status"]}</p><video controls preload="none" src="{source}"></video>')
    doc='''<!doctype html><meta charset="utf-8"><title>Derivative-Critic independent evaluation</title><style>body{font:16px/1.65 system-ui;max-width:1320px;margin:35px auto;padding:18px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}img{max-width:100%}video{width:650px;max-width:100%}</style>'''
    doc+='<h1>物理下降率 Critic＋AE＋目标 M：固定测试与独立复核</h1><p>24轮训练、192回合完整验证、2880回合固定测试、288回合深度压力测试均完成。检查点仅由验证集选择。本页使用原始轨迹独立重算动作、能量、物理误差、终止、起点与配对统计。</p>'
    doc+='<table><tr><th>方法</th><th>Plant</th><th>成功率</th><th>累计物理误差</th><th>V回升比例（回合均值）</th></tr>'+''.join(table)+'</table>'
    doc+='<h2>同起点配对得失</h2><table><tr><th>Plant</th><th>方法 / 基线</th><th>仅方法成功</th><th>仅基线成功</th><th>共同成功</th></tr>'+''.join(pairtable)+'</table><img src="paired_trajectories.png">'
    doc+='<h2>成功与失败录像</h2>'+''.join(videohtml)
    doc+='<p>新Phi＋单位M与同预算AE＋单位M比较表示贡献；同Phi的学习M与单位M比较度量贡献。AE对照共享主策略采集的数据，未独立采样，也没有为纯AE另外训练M。因此本实验未独立检验证书Critic与M之间的交互效应。单训练种子，仅四类生成器的新布局和目标，不代表任意真实场景泛化。</p>'
    doc+='<p>正定界及预测下降不构成真实Lyapunov证明。V回升仍存在；即使平均累计误差较小，也不能替代到达率、共同成功轨迹的精度与时间。P1为执行器滞后压力测试，不能将P0解析指令导数当作P1真实导数。</p>'
    doc+='<p><a href="independent_audit.json">独立复核与配对精度数据</a> · <a href="RESULTS.json">原始最终汇总与视频验收</a> · <a href="../report/REPORT.html">完整训练与评估报告</a></p>'
    (dest/'RESULTS.html').write_text(doc,encoding='utf-8')


def audit(out):
    if not (out / 'complete.json').exists():
        raise RuntimeError('Wait for the existing run to finish; do not relaunch it.')
    protocol = read(out / 'protocol.json')
    cfg = protocol['config']
    starts = read(out / 'evaluation_starts.json')
    errors = []
    maxima = defaultdict(float)
    records = defaultdict(dict)
    cert = defaultdict(lambda: dict(steps=0,strict_rises=0,tolerant_rises=0,monotonic_successes=0,successes=0))
    ratios = defaultdict(list)
    refs = {}
    shapes = {}
    expected = {'full_val':192,'fixed_test':2880,'depth_scale_0.5':144,'depth_scale_2.0':144}

    def check(ok, message):
        if not ok and len(errors) < 100:
            errors.append(message)

    project = out.parents[1]
    for name, fingerprint in protocol['sources'].items():
        check(sha(project / name) == fingerprint, 'source hash: ' + name)
    complete = read(out / 'complete.json')
    check(sha(out/'protocol.json') == complete['protocol_sha256'], 'protocol SHA256')
    check(sha(out/'best.pt') == complete['best_sha256'], 'best checkpoint SHA256')
    initial_counts = read(out/'training_complete.json')['counts']
    check(initial_counts == dict(warmup=1000,critic_prefit=1000,representation=7200,metric=4800), 'training counts')
    replay = read(out/'replay/index.json')
    check(all(r['split']=='train' for r in replay), 'held-out replay leak')
    check(sum(r['length'] for r in replay) <= 20000, 'active replay capacity')

    for tag, count in expected.items():
        folder = out/'evaluation'/tag
        summary = read(folder/'summary.json')
        check(summary['complete'] and summary['episodes']==summary['planned']==count, tag+': completeness')
        paths = [p for p in sorted(folder.glob('*.json')) if p.name!='summary.json']
        check(len(paths)==count, tag+': record count')
        for path in paths:
            r = read(path)
            key = (tag,r['method'],r['plant'])
            records[key][r['index']] = r
            label = tag+'/'+path.name
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                u = a['action'].astype(np.float64)
                off = a['offset'].astype(np.float64)
                e = a['features'].astype(np.float64)
                v = a['V'].astype(np.float64)
                m = a['M'].astype(np.float64)
                j = a['J_reference'].astype(np.float64)
                saved_h = a['H'].astype(np.float64)
                n = len(u)
                check(n==r['steps'] and off.shape==(n+1,4) and e.shape==(n+1,16) and v.shape==(n+1,), label+': trajectory shapes')
                check(all(np.isfinite(x).all() for x in (u,off,e,v,m,j,saved_h)), label+': nonfinite values')
                ev = np.linalg.eigvalsh(m)
                check(ev.min()>=.1-2e-5 and ev.max()<=14.5+2e-4, label+': SPD bounds')
                check(abs(np.trace(m)-16)<5e-5 and np.max(abs(m-m.T))<1e-6, label+': trace/symmetry')
                shapes.setdefault(r['method'],[]).extend([float(ev.min()),float(ev.max())])
                if r['method']=='learned':
                    lower = a['L'].astype(np.float64)
                    gram = lower@lower.T
                    check(np.allclose(m,.1*np.eye(16)+14.4*gram/np.trace(gram),atol=3e-5,rtol=3e-5),label+': Cholesky construction')
                else:
                    check(np.allclose(m,np.eye(16),atol=1e-7),label+': identity baseline')
                b = j.T@m@j
                ridge = max(1e-8,1.01*np.trace(b)/(1e4-1))
                h = b+.01*np.diag(np.diag(b))+ridge*np.eye(4)
                check(np.allclose(h,saved_h,atol=2e-6,rtol=1e-4),label+': H recomputation')
                raw = (-.7*np.linalg.solve(h,j.T@m@e[:-1].T)).T
                scale = np.maximum(1,np.maximum(np.linalg.norm(raw[:,:3],axis=1),np.abs(raw[:,3])))
                prediction = raw/scale[:,None]
                delta = float(np.max(np.abs(prediction-u)))
                maxima['action_recompute_abs'] = max(maxima['action_recompute_abs'],delta)
                check(delta<2e-3,label+': action recomputation')
                check(np.max(np.linalg.norm(u[:,:3],axis=1))<=1.00001 and np.max(abs(u[:,3]))<=1.00001,label+': action limits')
                cond = float(np.linalg.cond(h))
                maxima['H_condition'] = max(maxima['H_condition'],cond)
                check(cond<=10000.1,label+': H conditioning')
                energy = .5*np.einsum('ti,ij,tj->t',e,m,e)
                delta_v = float(np.max(abs(energy-v)))
                maxima['V_recompute_abs'] = max(maxima['V_recompute_abs'],delta_v)
                check(np.allclose(energy,v,atol=2e-5,rtol=5e-5),label+': V recomputation')
                E = np.sum(off[:,:3]**2,axis=1)+(off[:,3]/math.radians(25))**2
                check(abs(.1*E[1:].sum()-r['integrated_error'])<2e-4*(1+abs(r['integrated_error'])),label+': physical integral')
                check(abs(np.linalg.norm(off[-1,:3])-r['final_position_m'])<2e-6,label+': terminal position')
                check(abs(abs(math.degrees(off[-1,3]))-r['final_yaw_deg'])<1e-4,label+': terminal yaw')
                check(np.allclose(off[0],r['offset_start'],atol=1e-6),label+': initial offset')
                if tag in ('full_val','fixed_test'):
                    start = starts['val' if tag=='full_val' else 'test'][r['index']]
                    check(start['scene_id']==r['scene_id'] and start['goal_id']==r['goal_id'] and np.allclose(start['offset'],off[0],atol=1e-6),label+': frozen start')
                good = (np.linalg.norm(off[1:,:3],axis=1)<.02)&(np.abs(off[1:,3])<math.radians(1))
                check(r['success']==(r['status']=='success'),label+': success label')
                if r['success']:
                    check(n>=11 and bool(good[-11:].all()),label+': success hold/precision')
                check(bool(a['terminal'][-1])==(r['status']!='timeout'),label+': terminal flag')
                check(bool(a['truncated'][-1])==(r['status']=='timeout'),label+': truncation flag')
                expected_cost = .1*(E[1:]+.02*np.sum(u*u,axis=1))
                if r['status'] in ('collision','outside'):
                    expected_cost[-1] += 20
                check(np.allclose(expected_cost,a['cost'],atol=2e-5,rtol=2e-5),label+': successor cost label')
                du = np.diff(u,axis=0)
                check(abs(np.sum(du*du)-r['action_variation'])<2e-5*(1+r['action_variation']),label+': action variation')
                if r['plant']=='P0':
                    yaw = off[:-1,3]+.5*.35*u[:,3]*.1
                    move = np.stack([.04*(np.cos(yaw)*u[:,0]-np.sin(yaw)*u[:,1]),.04*(np.sin(yaw)*u[:,0]+np.cos(yaw)*u[:,1]),.04*u[:,2],.035*u[:,3]],axis=1)
                    residual = off[:-1]+move-off[1:]
                    # The simulator wraps yaw at +/-pi; compare on the circle.
                    residual[:,3] = (residual[:,3]+math.pi)%(2*math.pi)-math.pi
                    delta_p = float(np.max(abs(residual)))
                    maxima['P0_step_abs'] = max(maxima['P0_step_abs'],delta_p)
                    check(delta_p<2e-5,label+': P0 physics')
                if 'critic_true_coeff' in a.files:
                    zoff=off[a['critic_indices']]
                    x,y,z,yaw=zoff.T
                    truth=np.stack([2*(np.cos(yaw)*x+np.sin(yaw)*y),2*(-np.sin(yaw)*x+np.cos(yaw)*y),2*z,2*yaw/(math.radians(25)**2)],axis=1)
                    check(np.allclose(truth,a['critic_true_coeff'],atol=1e-5,rtol=1e-5),label+': physical derivative labels')
                difference = np.diff(v)
                tolerance = 1e-7+1e-4*v[:-1]
                g = cert[key]
                g['steps']+=n
                g['strict_rises']+=int(np.sum(difference>1e-10))
                g['tolerant_rises']+=int(np.sum(difference>tolerance))
                g['successes']+=int(r['success'])
                g['monotonic_successes']+=int(r['success'] and not np.any(difference>tolerance))
                ratios[key].append(v/(E+cfg['goal_fixed']['objective_floor']))
                refkey=(tag,r['method'],r['scene_id'],r['goal_id'])
                if refkey in refs:
                    check(np.allclose(m,refs[refkey][0],atol=1e-6,rtol=1e-6) and np.allclose(j,refs[refkey][1],atol=1e-6,rtol=1e-6),label+': fixed goal M/J')
                else:
                    refs[refkey]=(m.copy(),j.copy())
        print('Audited',tag,len(paths),flush=True)

    comparisons=[]
    for tag in expected:
        for plant in ('P0','P1'):
            for method,baseline in [('learned','identity_m'),('identity_m','shadow_ae_identity'),('identity_m','initial_ae_identity')]:
                a=records.get((tag,method,plant),{})
                b=records.get((tag,baseline,plant),{})
                if not a or not b:continue
                check(a.keys()==b.keys(),tag+': paired indices')
                pairs=[(a[i],b[i]) for i in sorted(a.keys()&b.keys())]
                both=[(x,y) for x,y in pairs if x['success'] and y['success']]
                row=dict(tag=tag,plant=plant,method=method,baseline=baseline,episodes=len(pairs),method_success=sum(x['success'] for x,y in pairs),baseline_success=sum(y['success'] for x,y in pairs),method_only=sum(x['success'] and not y['success'] for x,y in pairs),baseline_only=sum(y['success'] and not x['success'] for x,y in pairs),common_successes=len(both))
                for field in ('integrated_error','action_variation','final_position_m','final_yaw_deg'):
                    row[field]=[float(np.mean([x[field] for x,y in pairs])),float(np.mean([y[field] for x,y in pairs]))]
                    if both:row['common_'+field]=[float(np.mean([x[field] for x,y in both])),float(np.mean([y[field] for x,y in both]))]
                if both:row['common_time_s']=[float(np.mean([x['steps']*.1 for x,y in both])),float(np.mean([y['steps']*.1 for x,y in both]))]
                comparisons.append(row)
    certificates={}
    for key,g in cert.items():
        g['strict_rise_fraction']=g['strict_rises']/g['steps']
        g['tolerant_rise_fraction']=g['tolerant_rises']/g['steps']
        g['V_over_E_plus_floor_quantiles']=np.quantile(np.concatenate(ratios[key]),[.05,.5,.95]).tolist()
        certificates['/'.join(key)]=g
    existing=read(out/'analysis/RESULTS.json')
    check(existing['audit']['passed'],'original final video/energy audit')
    result=dict(passed=not errors,errors=errors,completed=datetime.now(timezone(timedelta(hours=8))).isoformat(),audited_episodes=sum(len(v) for v in records.values()),maxima=dict(maxima),training_counts=initial_counts,paired=comparisons,certificate=certificates,original_video_audit=existing['audit'],limitation='Single training seed; AE-only control shares student replay. Validation selected the checkpoint. P1 derivatives are not P0 ground truth. No uniform true Lyapunov guarantee.')
    dest=out/'analysis';dest.mkdir(exist_ok=True)
    target=dest/'independent_audit.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if result['passed']:
        make_report(out,result,records)
    print(json.dumps(dict(passed=result['passed'],episodes=result['audited_episodes'],errors=errors,maxima=result['maxima']),ensure_ascii=False),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    raise SystemExit(audit(parser.parse_args().output.resolve()))
