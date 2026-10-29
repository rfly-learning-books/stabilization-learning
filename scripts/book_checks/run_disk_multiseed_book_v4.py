"""Five-seed Chapter 9 D regression and separately labelled Chapter 8 checks.

New isolated outputs only. No book source or prior run is changed. The two-
parameter least-squares method follows the book's original example; it learns
D with V fixed, not M, a neural policy, contact physics, or image features.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BOOK = Path("E:/rfly-books/Rfly-Stable-Learning-Book_Reviewed_20261022")
SEEDS = (41, 42, 43, 44, 45)
T, RADIUS, A_MAX, K_MAX = .1, .05, .05, 1.2
TRUE_BETA = np.array([.1, .5 * .1**2 * (76 / 75)])
PREFIXES = np.unique(np.rint(np.geomspace(20, 8000, 40)).astype(int))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generator(seed, phase):
    return np.random.default_rng(np.random.SeedSequence([20261025, seed, phase]))


def training_data(seed):
    rng = generator(seed, 1)
    xi = rng.uniform(-RADIUS, -.005, 8000)
    h = rng.uniform(.2, 1., 8000)
    a = -h * xi
    gains = rng.uniform(.8, 1.2, 8000)
    following = xi + T * gains * a
    return np.column_stack((xi, a, following, .5 * (following**2 - xi**2)))


def fit(data):
    xi, a, following, y = data.T
    h = -a / xi
    features = np.column_stack((-h, h**2))
    beta, _, rank, _ = np.linalg.lstsq(features, y / xi**2, rcond=None)
    grid = np.linspace(.1, 1., 91)
    q = 1 + 2 * (-beta[0] * grid + beta[1] * grid**2)
    index = int(np.argmin(q))
    # This analytic diagnostic is reported after fitting; never used to select K.
    diagnostic_h = np.linspace(.2, 1., 1001)
    diagnostic_features = np.column_stack((-diagnostic_h, diagnostic_h**2))
    return dict(beta=beta.tolist(), selected_K=float(grid[index]), q_fit=float(q[index]),
                feature_rank=int(rank), training_normalized_mse=float(np.mean((features @ beta-y/xi**2)**2)),
                analytic_D_normalized_rmse=float(np.sqrt(np.mean((diagnostic_features @ (beta-TRUE_BETA))**2))))


def validate(seed, K):
    rng = generator(seed, 2)
    xi = rng.uniform(-RADIUS, -.005, 4000)
    a = -K * xi
    following = xi + T * rng.uniform(.8, 1.2, 4000) * a
    ratio = (following / xi)**2
    lower = (1-T*K_MAX*K)**2
    mean = float(ratio.mean())
    margin = (1-lower) * math.sqrt(math.log(100) / (2*len(xi)))
    family_margin = (1-lower) * math.sqrt(math.log(500) / (2*len(xi)))
    return np.column_stack((xi, a, following, ratio)), dict(q_validation=mean,
        q_upper_99=min(1.,mean+margin), q_upper_family_99=min(1.,mean+family_margin),
        hoeffding_range=[lower,1.], hoeffding_margin=margin,
        confidence_note="Per-seed 99%; additional delta=.01/5 bounds give simultaneous >=99% by union bound.")


def trajectories(seed, K, count=256, steps=120):
    gains = generator(seed, 3).uniform(.8, 1.2, (steps, count))
    xi = np.full((steps+1,count), -RADIUS)
    action = np.empty((steps,count))
    for k in range(steps):
        action[k] = -K*xi[k]
        xi[k+1] = xi[k]+T*gains[k]*action[k]
    return dict(xi=xi, actions=action, gains=gains, V=.5*xi**2)


def audit(data, validation, trace, selected):
    """Reconstruct labels, gain support, fit, selection and physical recurrences."""
    x,a,xn,y=data.T
    vx,va,vn,z=validation.T
    beta = np.linalg.solve(np.column_stack((x*a,a*a)).T @ (np.column_stack((x*a,a*a))/x[:,None]**4),
        np.column_stack((x*a,a*a)).T @ (y/x**4))
    # Weighted normal equations independently check the normalized least squares.
    inferred=(xn-x)/(T*a)
    inferred_validation=(vn-vx)/(T*va)
    q_analytic=1-2*T*selected['selected_K']+T*T*selected['selected_K']**2*76/75
    checks = dict(training_labels=float(np.max(np.abs(y-.5*(xn*xn-x*x))))<1e-14,
        fitted_coefficients=bool(np.allclose(beta,selected['beta'],rtol=1e-9,atol=1e-12)),
        training_gain_support=bool(np.all((inferred>=.8-1e-12)&(inferred<=1.2+1e-12))),
        validation_gain_support=bool(np.all((inferred_validation>=.8-1e-12)&(inferred_validation<=1.2+1e-12))),
        validation_ratios=bool(np.allclose(z,(vn/vx)**2,rtol=0,atol=1e-14)),
        validation_actions=bool(np.allclose(va,-selected['selected_K']*vx,rtol=0,atol=1e-14)),
        trajectory_dynamics=bool(np.allclose(trace['xi'][1:],trace['xi'][:-1]+T*trace['gains']*trace['actions'],rtol=0,atol=1e-14)),
        trajectory_actions=bool(np.allclose(trace['actions'],-selected['selected_K']*trace['xi'][:-1],rtol=0,atol=1e-14)),
        trajectory_V=bool(np.allclose(trace['V'],.5*trace['xi']**2,rtol=0,atol=1e-14)),
        domain=all(bool(np.all((v>=-RADIUS-1e-14)&(v<=1e-14))) for v in (x,xn,vx,vn,trace['xi'])),
        action_limits=all(bool(np.all((v>=-1e-14)&(v<=A_MAX+1e-14))) for v in (a,va,trace['actions'])),
        gain_feasible=0<selected['selected_K']<=min(A_MAX/RADIUS,1/(T*K_MAX)),
        rank=selected['feature_rank']==2)
    return dict(passed=all(checks.values()),checks=checks,q_analytic=q_analytic,
                known_model_controller_identical=selected['selected_K']==1.)


def figures(out, results, histories, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    figdir=out/'figures';figdir.mkdir()
    files=[]
    def save(fig,name):
        for suffix in ('png','pdf','svg'):
            path=figdir/(name+'.'+suffix)
            fig.savefig(path,dpi=200,bbox_inches='tight')
            files.append(dict(path=path.relative_to(out).as_posix(),sha256=sha(path)))
        plt.close(fig)
    colors=plt.cm.viridis(np.linspace(.12,.88,len(SEEDS)))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for s,color in zip(SEEDS,colors):
        rows=histories[s]
        for ax,key in zip(axes.flat,('analytic_D_normalized_rmse','training_normalized_mse','q_fit','selected_K')):
            ax.plot(PREFIXES,[r[key] for r in rows],color=color,alpha=.55,lw=1,label=f'seed {s}')
            ax.set_xscale('log');ax.set_xlabel('Training transitions (nested prefixes)');ax.grid(alpha=.18)
    for ax,key,title in zip(axes.flat,('analytic_D_normalized_rmse','training_normalized_mse','q_fit','selected_K'),
        ('D-fit error against known analytic model','Normalized noisy training residual','Fitted second-moment factor','Feedback gain selected from fixed grid')):
        ys=np.array([[r[key] for r in histories[s]] for s in SEEDS]);m=ys.mean(0);sd=ys.std(0,ddof=1)
        ax.plot(PREFIXES,m,color='#182d4d',lw=2,label='5-seed mean')
        ax.fill_between(PREFIXES,m-sd,m+sd,color='#182d4d',alpha=.12,label='between-seed +/- 1 SD')
        ax.set_title(title)
    axes[0,0].set_yscale('log');axes[0,0].set_ylabel('RMSE of normalized expected D')
    axes[0,1].set_ylabel('MSE of normalized noisy labels')
    axes[1,0].axhline(1519/1875,ls='--',color='#b04b34',label='analytic q for K=1')
    axes[1,0].set_ylabel('q');axes[1,1].set_ylabel('K [1/s]');axes[1,1].set_ylim(.8,1.2)
    axes[0,0].legend(fontsize=7,ncol=2);axes[1,0].legend(fontsize=7)
    fig.suptitle('Book Chapter 9 | 5-seed D regression with fixed V = error² / 2\nAnalytic error is a diagnostic; independent validation is accessed only after all final gains are frozen.',fontsize=12)
    save(fig,'disk_D_training_curves')

    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    t=np.arange(121)*T;tk=t[:-1]
    mean_e=np.stack([traces[s]['xi'].mean(1) for s in SEEDS]);mean_v=np.stack([traces[s]['V'].mean(1) for s in SEEDS]);mean_u=np.stack([traces[s]['actions'].mean(1) for s in SEEDS])
    for ax,arr,title,ylabel,xx in [(axes[0,0],mean_e,'Position error','error [m]',t),
            (axes[0,1],mean_v,'Fixed Lyapunov energy','V [m²]',t),
            (axes[0,2],mean_u,'Executed push velocity','a [m/s]',tk)]:
        for i,s in enumerate(SEEDS):ax.plot(xx,arr[i],color=colors[i],alpha=.5,lw=1)
        m=arr.mean(0);sd=arr.std(0,ddof=1);ax.plot(xx,m,color='#182d4d',lw=2,label='mean of 5 run means')
        ax.fill_between(xx,m-sd,m+sd,color='#182d4d',alpha=.13,label='between-run +/-1 SD')
        ax.set(xlabel='Time [s]',ylabel=ylabel,title=title);ax.grid(alpha=.18)
    axes[0,0].plot(t,-RADIUS*.9**np.arange(121),'--',color='#b04b34',label='analytic mean = Ch8 deterministic path')
    axes[0,0].legend(fontsize=7)
    axes[0,1].semilogy(t,.5*RADIUS**2*(1519/1875)**np.arange(121),'--',color='#b04b34',label='analytic expectation')
    qu=max(results[str(s)]['q_upper_family_99'] for s in SEEDS)
    axes[0,1].semilogy(t,.5*RADIUS**2*qu**np.arange(121),':',color='#6f3b9c',label='99% simultaneous bound (formula)')
    axes[0,1].legend(fontsize=7);axes[0,2].legend(fontsize=7)
    labels=[str(s) for s in SEEDS];pos=np.arange(5)
    axes[1,0].plot(pos,[results[str(s)]['q_validation'] for s in SEEDS],'o',label='independent validation mean')
    axes[1,0].plot(pos,[results[str(s)]['q_upper_99'] for s in SEEDS],'^',label='per-run 99% upper bound')
    axes[1,0].plot(pos,[results[str(s)]['q_upper_family_99'] for s in SEEDS],'s',label='simultaneous 99% upper bound')
    axes[1,0].axhline(1519/1875,ls='--',color='#b04b34',label='analytic q')
    axes[1,0].set(xticks=pos,xticklabels=labels,xlabel='Training seed',ylabel='Second-moment factor q',title='Frozen-policy independent validation')
    axes[1,0].legend(fontsize=7);axes[1,0].grid(alpha=.18)
    sample=traces[41];axes[1,1].plot(t,sample['xi'][:,0],label='seed41, first fixed rollout')
    quant=np.quantile(np.concatenate([traces[s]['xi'] for s in SEEDS],axis=1),[.05,.95],axis=1)
    axes[1,1].fill_between(t,*quant,alpha=.2,label='5–95% of 1280 physical paths')
    axes[1,1].set(xlabel='Time [s]',ylabel='error [m]',title='Actual simulated trajectories (not formulas)');axes[1,1].legend(fontsize=7);axes[1,1].grid(alpha=.18)
    ax=axes[1,2]
    for row,k in enumerate((0,5,10,20,40)):
        x=sample['xi'][k,0]*100
        ax.plot([-5.5,.5],[row,row],color='#d6dde7',lw=1)
        ax.add_patch(Circle((x,row),.13,facecolor=colors[0],edgecolor='#182d4d'))
        ax.annotate('',xy=(x-.18,row),xytext=(x-.65,row),arrowprops=dict(arrowstyle='->',color='#b04b34'))
    ax.axvline(0,color='#22825c',ls='--');ax.set(xlim=(-5.8,.6),ylim=(-.5,4.5),yticks=np.arange(5),yticklabels=['0 s','.5 s','1 s','2 s','4 s'],xlabel='Position relative to target [cm]',title='One-sided disk positions: actual first path')
    ax.text(.5,-.22,'Circle size is symbolic; only 1D position is modeled.',transform=ax.transAxes,ha='center',fontsize=8)
    fig.suptitle('Book disk result | 5 × 256 independent 120-step rollouts\nAll gains K=1: learned D and known-model feedback are the same controller; no performance improvement is claimed.',fontsize=12)
    save(fig,'disk_closed_loop_results')
    return files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve()
    if out.parent != ROOT/'runs':raise ValueError('Use a new immediate child of runs/')
    started=time.perf_counter();out.mkdir(exist_ok=False)
    sources=[Path(__file__),BOOK/'ch8/chapter8.tex',BOOK/'ch9/chapter9.tex',BOOK/'ch9/examples/stochastic_disk/run_stochastic_disk.py']
    source_hashes={str(p):sha(p) for p in sources}
    protocol=dict(schema='book_disk_D_multiseed_v4',seeds=list(SEEDS),training_count=8000,validation_count=4000,
        rollout_count_per_seed=256,rollout_steps=120,random_streams={str(s):{p:[20261025,s,i] for p,i in [('train',1),('validation',2),('rollout',3)]} for s in SEEDS},
        scope='Chapter9 scalar fixed-contact D-learning, V fixed; Chapter8 deterministic reference; not neural M or vision',
        parameters=dict(T=T,r=RADIUS,a_max=A_MAX,kappa_support=[.8,1.2],candidate_K=[.1,1.,.01]),
        prefixes=PREFIXES.tolist(),selection='final 8000 training transitions only; all gains frozen before any validation',
        figure_policy='all seeds; mean +/- sample SD; first declared trajectory; no cherry-picking',sources=source_hashes)
    write(out/'protocol.json',protocol)
    snap=out/'source_snapshot';snap.mkdir()
    for i,p in enumerate(sources):shutil.copyfile(p,snap/f'{i}_{p.name}')
    histories={};selected={};datasets={}
    for seed in SEEDS:
        folder=out/f'seed{seed}';folder.mkdir()
        data=training_data(seed);datasets[seed]=data
        histories[seed]=[dict(training_count=int(n),**fit(data[:n])) for n in PREFIXES]
        selected[str(seed)]=fit(data)
        np.save(folder/'training_transitions.npy',data)
        write(folder/'TRAINING_HISTORY.json',histories[seed])
    write(out/'SELECTION_COMMITTED.json',dict(selections=selected,test_access=False,protocol_sha256=sha(out/'protocol.json')))
    results={};traces={}
    for seed in SEEDS:
        entry=selected[str(seed)];data,val=validate(seed,entry['selected_K']);trace=trajectories(seed,entry['selected_K'])
        check=audit(datasets[seed],data,trace,entry);traces[seed]=trace
        folder=out/f'seed{seed}';np.save(folder/'validation_transitions.npy',data);np.savez_compressed(folder/'trajectories.npz',**trace)
        write(folder/'AUDIT.json',check)
        results[str(seed)]={**entry,**val,**check}
    deterministic=np.empty((201,101));deterministic[0]=np.linspace(-RADIUS,0,101)
    for k in range(200):deterministic[k+1]=deterministic[k]+T*(-deterministic[k])
    np.save(out/'CH8_DETERMINISTIC_TRAJECTORIES.npy',deterministic)
    deterministic_check=dict(scope='Chapter8 fixed known gain kappa=1, K=1; no fitting',
        states=101,steps=200,max_energy_equality_residual=float(np.max(np.abs(.5*deterministic[1:]**2-.81*.5*deterministic[:-1]**2))),
        domain_preserved=bool(np.all((deterministic>=-RADIUS)&(deterministic<=0))))
    write(out/'CH8_DETERMINISTIC_CHECK.json',deterministic_check)
    files=figures(out,results,histories,traces)
    unchanged=all(sha(p)==h for p,h in source_hashes.items())
    summary=dict(scope=protocol['scope'],seeds=results,all_audits_passed=all(v['passed'] for v in results.values()),
        all_bounds_below_one=all(v['q_upper_family_99']<1 for v in results.values()),sources_unchanged=unchanged,
        elapsed_seconds=time.perf_counter()-started,training_transitions=40000,independent_validation_transitions=20000,
        independent_closed_loop_transitions=5*256*120,learned_M=False,vision_training=False,
        deterministic_reference=deterministic_check)
    write(out/'RESULTS.json',summary);write(out/'FIGURES.json',files)
    rows=['# 书稿随机圆盘：五种子 D 学习与闭环结果','',
        '按 Reviewed_20261022 书稿第9章的同一单侧、固定接触、标量模型运行。每种子 8000 条训练转移、4000 条独立验证转移；五个反馈增益全部冻结后才访问验证数据。',
        '固定 V(ξ)=ξ²/2，学习两个 D 回归系数。本实验不是神经 M 训练、二维接触仿真或视觉圆盘复现。',
        '', '|种子|K|拟合q|独立验证q|单次99%上界|五次同时99%上界|审计|','|---|---:|---:|---:|---:|---:|---|']
    for seed in SEEDS:
        x=results[str(seed)];rows.append(f"|{seed}|{x['selected_K']:.2f}|{x['q_fit']:.7f}|{x['q_validation']:.7f}|{x['q_upper_99']:.7f}|{x['q_upper_family_99']:.7f}|{'通过' if x['passed'] else '失败'}|")
    rows += ['', '全部种子都选择 K=1，与已知模型下的约束最优反馈相同；因此此处证明的是数值辨识与独立证书检查的一致性，不能称为优于已知模型反馈。',
        '训练图横轴是累计转移样本数，不是神经网络 epoch；曲线来自同一训练集的预先规定前缀。解析 D 误差只作诊断，不参与反馈选择。阴影为五次独立训练的样本标准差，不是95%置信区间。',
        '结果图含每个种子独立的 256 条、120 步真实仿真轨迹。解析均方曲线与 Hoeffding 界用不同线型标明；界曲线不是新增实验数据。单条位置图固定选择 seed41 的第一个轨迹，圆盘尺寸仅作符号展示。',
        '已知增益支撑与固定接触、独立同分布假设给出该理想单侧工作域的推导基础。有限样本上界是概率事件上的结论，不扩展为视觉、多接触、真实机器人或神经 M 的全局稳定证明。',
        '', '书稿对应：ch8/chapter8.tex:2344–2416；ch9/chapter9.tex:1065–1240；原始算法 ch9/examples/stochastic_disk/run_stochastic_disk.py。书稿及旧结果未改动。',
        '', f"总耗时：{summary['elapsed_seconds']:.2f} 秒。",'',
        '![训练曲线](figures/disk_D_training_curves.png)','', '![闭环结果](figures/disk_closed_loop_results.png)']
    (out/'REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    final=dict(status='complete' if summary['all_audits_passed'] and unchanged else 'failed_audit',completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    write(out/'FINAL_STATE.json',final);print(json.dumps(dict(output=str(out),**final),ensure_ascii=False))
    return 0 if final['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
