"""Data, fitting, independent validation, freeze and paired physical tests."""
from dataclasses import replace
from contextlib import nullcontext
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .env import DiskConfig, DiskEnv, transitions, fixed_starts, concatenate, value
from .algo import ROUTES, fit, calibrate, choose_action, prediction_with_margin


ROOT=Path(__file__).resolve().parents[2]
BOOK=Path('E:/rfly-books/Rfly-Stable-Learning-Book_Reviewed_20261022/ch8/chapter8.tex')
TASKS=Path('E:/rfly-books/output/reports/全书实验验证任务清单_20261028/全书实验验证任务清单.md')
MODES=('mean','margin','counterexample')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.rename(path)


def csv_rows(path,rows):
    if not rows:return
    names=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader();writer.writerows(rows)


def save_data(path,data):
    np.savez_compressed(path,**data)
    csv_rows(path.with_suffix('.csv'),[dict(index=i,**{k:float(v[i]) for k,v in data.items()}) for i in range(len(data['x']))])


def scenarios(smoke=False):
    if smoke:
        c=DiskConfig(horizon_seconds=2.)
        return [dict(name='engineering_constant',train=c,test=c,count=64,excitation='random',noise=0.)]
    entries=[]
    for response in ('constant','varying'):
        for count in (128,512,2048):
            c=DiskConfig(response=response)
            entries.append(dict(name=f'{response}_n{count}',train=c,test=c,count=count,excitation='random',noise=0.))
    for gain in (.6,1.,1.4):
        for dt in (.05,.1,.2):
            if gain==1. and dt==.1:continue
            c=DiskConfig(kappa=gain,dt=dt)
            entries.append(dict(name=f'gain{gain:g}_dt{dt:g}',train=c,test=c,count=512,excitation='random',noise=0.))
    c=DiskConfig()
    entries.append(dict(name='bounded_response_transfer',train=c,test=replace(c,response='varying'),count=512,excitation='random',noise=0.))
    for excitation in ('feedback','multigain'):
        entries.append(dict(name='excitation_'+excitation,train=c,test=c,count=512,excitation=excitation,noise=0.))
    entries.append(dict(name='training_measurement_noise',train=c,test=c,count=512,excitation='random',noise=.0002))
    return entries


def rollout(model,calibration,env,starts,mode,guard,candidates):
    n=len(starts);c=env.config;steps=int(np.ceil(c.horizon_seconds/c.dt))
    state=np.full((steps+1,n),np.nan);state[0]=starts
    keys=('action','predicted_D','margin','true_D','required_decrease','predicted_lower','predicted_upper','gain')
    arrays={key:np.full((steps,n),np.nan) for key in keys}
    arrays['feasible']=np.zeros((steps,n),bool);arrays['valid']=np.zeros((steps,n),bool)
    hold=np.zeros(n,int);arrival=np.full(n,np.nan);failure=np.zeros(n,bool)
    required_hold=int(np.ceil(c.hold_seconds/c.dt))
    for step in range(steps):
        guard();active=np.flatnonzero(~failure)
        if not len(active):break
        x=state[step,active]
        choice=choose_action(model,x,env,calibration,mode,candidates)
        a=choice['action'];following=env.step(x,a)
        state[step+1,active]=following
        for key in ('action','predicted_D','margin','feasible','required_decrease','predicted_lower','predicted_upper'):
            arrays[key][step,active]=choice[key]
        arrays['true_D'][step,active]=value(following)-value(x)
        arrays['gain'][step,active]=env.gain(x,a);arrays['valid'][step,active]=True
        bad=~env.in_domain(following)
        failure[active[bad]]=True
        hold[active]=np.where(np.abs(following)<=c.tolerance,hold[active]+1,0)
        reached=(hold[active]>=required_hold)&np.isnan(arrival[active])&~bad
        arrival[active[reached]]=(step+1)*c.dt
    arrays.update(x=state,V=value(state),arrival_time=arrival,domain_failure=failure,
                  success=(~failure)&(hold>=required_hold),dt=np.asarray(c.dt))
    valid=arrays['valid'];actual=arrays['true_D'][valid];pred=arrays['predicted_D'][valid];margin=arrays['margin'][valid]
    residual=actual+arrays['required_decrease'][valid]
    epsilon=1e-12+1e-8*arrays['V'][:-1][valid]
    final=np.array([state[np.flatnonzero(np.isfinite(state[:,i]))[-1],i] for i in range(n)])
    summary=dict(episodes=n,successes=int(arrays['success'].sum()),success_rate=float(arrays['success'].mean()),
        domain_exit_rate=float(failure.mean()),final_mean_absolute_error=float(np.abs(final).mean()),
        final_maximum_absolute_error=float(np.abs(final).max()),input_peak=float(np.max(arrays['action'][valid])),
        integrated_absolute_error=float(np.nansum(np.abs(state[1:]))*c.dt/n),
        predicted_D_rmse=float(np.sqrt(np.mean((actual-pred)**2))),
        actual_descent_violation_rate=float(np.mean(residual>epsilon)),
        actual_descent_max_violation=float(max(0.,residual.max())),
        V_increase_rate=float(np.mean(actual>epsilon)),
        empty_estimated_action_set_rate=float(np.mean(~arrays['feasible'][valid])),
        empirical_margin_coverage=float(np.mean(np.abs(actual-pred)<=margin+epsilon)),
        valid_transitions=int(valid.sum()),hold_seconds=c.hold_seconds,
        contact_model='fixed contact by quasi-static assumption; interval/input feasibility checked')
    return arrays,summary


def save_rollout(folder,trace,summary):
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'trajectories.npz',**trace)
    rows=[]
    for t,e in zip(*np.nonzero(trace['valid'])):
        rows.append(dict(episode=int(e),step=int(t),time=float(t*trace['dt']),x=float(trace['x'][t,e]),
            next_x=float(trace['x'][t+1,e]),V=float(trace['V'][t,e]),next_V=float(trace['V'][t+1,e]),
            **{k:(bool(trace[k][t,e]) if k=='feasible' else float(trace[k][t,e])) for k in
               ('action','predicted_D','margin','true_D','required_decrease','predicted_lower','predicted_upper','gain','feasible')}))
    csv_rows(folder/'transitions.csv',rows)
    episodes=[]
    for episode in range(trace['x'].shape[1]):
        valid_steps=np.flatnonzero(trace['valid'][:,episode])
        last=int(valid_steps[-1]+1) if len(valid_steps) else 0
        success=bool(trace['success'][episode]);failed=bool(trace['domain_failure'][episode])
        arrival=trace['arrival_time'][episode]
        episodes.append(dict(episode=episode,shared_boundary_anchor=episode<4,
            start_x=float(trace['x'][0,episode]),final_x=float(trace['x'][last,episode]),
            transitions=len(valid_steps),success=success,
            status='domain_exit' if failed else 'success' if success else 'time_limit',
            first_completed_hold_time=float(arrival) if np.isfinite(arrival) else None,
            input_peak=float(np.nanmax(trace['action'][:,episode])) if len(valid_steps) else 0.,
            empty_action_set_steps=int(np.sum(~trace['feasible'][valid_steps,episode]))))
    csv_rows(folder/'episodes.csv',episodes)
    # Reopen the persisted endpoints independently of the controller model.
    with np.load(folder/'trajectories.npz') as raw:
        mask=raw['valid'];recomputed=value(raw['x'][1:])-value(raw['x'][:-1])
        delta=float(np.max(np.abs(recomputed[mask]-raw['true_D'][mask])))
        successor=float(np.max(np.abs((raw['x'][:-1]+raw['dt']*raw['gain']*raw['action'])[mask]-raw['x'][1:][mask])))
    audit=dict(full_successor_endpoints=True,V_delta_max_abs_error=delta,
               physical_successor_max_abs_error=successor,passed=delta<=1e-14 and successor<=1e-14)
    write(folder/'summary.json',summary);write(folder/'endpoint_audit.json',audit)
    if not audit['passed']:raise AssertionError('Persisted trajectory endpoint audit failed')
    return audit


def region_audit(model,calibration,env,mode,folder):
    c=env.config;x=np.linspace(-c.radius,0.,81)[:,None]
    a=env.safe_upper(x)*np.linspace(0.,1.,65)[None,:]
    x=np.broadcast_to(x,a.shape)
    predicted,margin,lower,upper=prediction_with_margin(model,x,a,env,calibration,mode!='mean')
    actual=value(env.step(x,a))-value(x)
    rate=.5*(1.-max(0.,1.-c.dt)**2)
    estimated=(predicted+margin<=-rate*x*x+1e-14)&(lower>=-c.radius-1e-12)&(upper<=1e-12)
    violation=np.maximum(0.,actual+rate*x*x)
    covered=np.abs(actual-predicted)<=margin+1e-12+1e-8*value(x)
    np.savez_compressed(folder/'region_grid.npz',x=x,action=a,predicted_D=predicted,margin=margin,
                        actual_D=actual,estimated_feasible=estimated,violation=violation,covered=covered)
    return dict(grid_points=x.size,coverage=float(covered.mean()),
                accepted_points=int(estimated.sum()),
                accepted_actual_violation_rate=float(np.mean(violation[estimated]>1e-12)) if estimated.any() else None,
                maximum_accepted_violation=float(violation[estimated].max()) if estimated.any() else None,
                evidence='finite deterministic grid; no between-grid derivative bound or regional certification')


def make_plots(output,curves,scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'figures';figures.mkdir(exist_ok=True)
    base='constant_n512' if any(r['scenario']=='constant_n512' for r in scores) else scores[0]['scenario']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for route in ROUTES:
        path=output/'test'/base/'margin'/route/'trajectories.npz'
        with np.load(path) as a:
            time_axis=np.arange(len(a['x']))*float(a['dt']);e=min(3,a['x'].shape[1]-1)
            axes[0].plot(time_axis,a['x'][:,e],label=route)
            axes[1].semilogy(time_axis,np.maximum(a['V'][:,e],1e-20),label=route)
    axes[0].set(xlabel='Time (s)',ylabel='Position error (m)',title='Paired far-boundary start')
    axes[1].set(xlabel='Time (s)',ylabel='V = x^2 / 2',title='Fixed candidate scale')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'paired_position_V.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for route in ROUTES:
        rs=[r for r in curves if r['scenario']==base and r['dataset']=='ordinary' and r['route']==route]
        axes[0].loglog([r['samples'] for r in rs],[max(r['training_label_D_rmse'],1e-20) for r in rs],marker='o',label=route)
        axes[1].loglog([r['samples'] for r in rs],[max(r['validation_true_D_rmse'],1e-20) for r in rs],marker='o',label=route)
    axes[0].set(xlabel='Labeled training samples',ylabel='D RMSE',title='Training fit')
    axes[1].set(xlabel='Labeled training samples',ylabel='D RMSE',title='Independent validation transitions')
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'training_validation_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for route in ROUTES:
        rs=[r for r in scores if r['scenario'].startswith('constant_n') and r['mode']=='margin' and r['route']==route]
        rs.sort(key=lambda r:r['samples'])
        if rs:axes[0].plot([r['samples'] for r in rs],[r['success_rate'] for r in rs],marker='o',label=route)
    axes[0].set(xlabel='Training samples',ylabel='Held-out success rate',title='Data budget vs closed-loop performance',ylim=(-.02,1.02))
    with np.load(output/'test'/base/'margin'/'direct_D'/'region_grid.npz') as a:
        axes[1].scatter(a['predicted_D'].ravel(),a['actual_D'].ravel(),s=2,alpha=.2)
        low=min(a['predicted_D'].min(),a['actual_D'].min());high=max(a['predicted_D'].max(),a['actual_D'].max())
        axes[1].plot([low,high],[low,high],color='k',lw=.8)
    axes[1].set(xlabel='Predicted D',ylabel='Actual V(next)-V(current)',title='Direct D, independent region audit')
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/'data_budget_and_D_audit.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    target='bounded_response_transfer' if any(r['scenario']=='bounded_response_transfer' for r in scores) else base
    for ax,mode in zip(axes,MODES):
        with np.load(output/'test'/target/mode/'direct_D'/'region_grid.npz') as a:
            field=np.where(a['estimated_feasible'],a['violation'],np.nan)
            image=ax.imshow(field,origin='lower',aspect='auto',extent=[0,1,-.05,0],vmin=0)
            fig.colorbar(image,ax=ax,shrink=.7)
        ax.set(xlabel='Fraction of safe action cap',ylabel='Position error (m)',title=mode+' accepted violations')
    fig.tight_layout();fig.savefig(figures/'accepted_violation_heatmap.png',dpi=150);plt.close(fig)
    # Exact feasibility boundary of the printed constant-gain K feedback.
    gain=np.linspace(.1,3.,81);dt=np.linspace(.01,1.,81);kk,tt=np.meshgrid(gain,dt)
    feasible=(kk*tt<=1.) # printed K=1, Kr=a_max
    np.savez_compressed(output/'printed_gain_sampling_feasibility.npz',kappa=kk,dt=tt,K=1.,feasible=feasible)
    fig,ax=plt.subplots(figsize=(6,4));ax.pcolormesh(kk,tt,feasible,shading='auto',vmin=0,vmax=1)
    ax.set(xlabel='Response gain kappa',ylabel='Sampling period (s)',title='Exact printed-model domain condition: 0 < kappa K Ts <= 1')
    fig.tight_layout();fig.savefig(figures/'printed_gain_sampling_feasibility.png',dpi=150);plt.close(fig)


def run(output,seed=41,max_minutes=30.,smoke=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('A fresh output directory is required')
    if seed!=41:raise ValueError('This protocol currently declares only seed41')
    if not np.isfinite(max_minutes) or max_minutes<=0:raise ValueError('Positive finite budget required')
    output.mkdir(parents=True);begun=time.time();deadline=begun+60*max_minutes
    planned=scenarios(smoke);candidates=17 if smoke else 65
    def guard():
        if (output/'STOP').exists():raise InterruptedError('STOP requested')
        if time.time()>=deadline:raise TimeoutError('Declared disk-experiment budget expired')
    source_paths=[BOOK,TASKS,*Path(__file__).parent.glob('*.py'),ROOT/'scripts/run_book_disk_overnight_v1.py']
    protocol=dict(schema='book_disk_overnight_v1',created_utc=datetime.now(timezone.utc).isoformat(),seed=seed,
        mode='engineering' if smoke else 'formal',maximum_minutes=max_minutes,deadline_unix=deadline,
        source_hashes={str(p):sha(p) for p in source_paths},CPU_threads=1,GPU_used=False,
        printed_task='Reviewed chapter8 2353-2467; fixed-side 1D quasi-static disk',
        excluded='Not chapter10 visual disk; no images, free-plane contact, rigid-body contact or hardware validation',
        candidate='V(x)=0.5*x^2, fixed for every route',routes=list(ROUTES),modes=list(MODES),
        information='Known-model route receives the evaluation dynamics; learned routes receive equal labeled training budgets plus the same physical gain bound',
        stages='initialize fixed splits -> common data -> fitting/validation -> freeze all models -> paired independent tests',
        margin='Empirical residual maximum on separate calibration data, not a proven regional error bound',
        error_components=dict(perception='no visual observation; one scenario corrupts training positions only',
            model='learned D/increment prediction residual, independently audited',
            sampling='exact declared ZOH transition, including quadratic term in complete V difference',
            execution='exact bounded gain and command cap, no unmodeled delay or actuator dynamics'),
        active_data='First half pilot; second half locally samples worst pilot increment residual states; exactly N labeled transitions; same resulting dataset for all learned routes',
        split='Independent RNG for train/calibration/validation/test transitions and validation/test starts; exact goal/near/boundary anchors shared and separately flagged',
        success='Final consecutive 1 second within 1mm after entire horizon; domain exit is failure',
        E05_scope='Finite regression information and fixed/multigain/random excitation; training-only observation-noise case. No mechanical-object or window-integral study',
        scenarios=[dict(**{k:v for k,v in s.items() if k not in ('train','test')},train=s['train'].asdict(),test=s['test'].asdict()) for s in planned])
    write(output/'protocol.json',protocol)
    curves=[];scores=[];validation_scores=[];fit_records=[];models={};frozen=[];datasets={};final={}
    try:
        # CLI sets the BLAS/OpenMP limits before importing NumPy/SciPy.
        with nullcontext():
            for si,s in enumerate(planned):
                guard();name=s['name'];folder=output/'data'/name;folder.mkdir(parents=True)
                train_env=DiskEnv(s['train']);test_env=DiskEnv(s['test']);n=s['count']
                rng=np.random.default_rng(np.random.SeedSequence([seed,si,0]));pilot=transitions(train_env,n//2,rng,s['excitation'],noise=s['noise'])
                ordinary=concatenate(pilot,transitions(train_env,n-n//2,rng,s['excitation'],noise=s['noise']))
                preliminary,_=fit('identified_model',pilot,s['train'],seed)
                residual=np.abs(preliminary.increment(pilot['x'],pilot['action'])-(pilot['next_x']-pilot['x']))
                centers=pilot['x'][np.argsort(residual)[-min(16,len(residual)):]]
                counter=concatenate(pilot,transitions(train_env,n-n//2,np.random.default_rng(np.random.SeedSequence([seed,si,1])),
                    s['excitation'],center=centers,noise=s['noise']))
                independent={key:transitions(train_env if key=='calibration' else test_env,64 if smoke else 512,
                    np.random.default_rng(np.random.SeedSequence([seed,si,10+j]))) for j,key in enumerate(('calibration','validation','test'))}
                for key,data in dict(ordinary=ordinary,counterexample=counter,**independent).items():save_data(folder/(key+'.npz'),data)
                starts={split:fixed_starts(s['test'],6 if smoke else (12 if split=='validation' else 32),
                    np.random.default_rng(np.random.SeedSequence([seed,si,30+j]))) for j,split in enumerate(('validation','test'))}
                np.savez_compressed(folder/'starts.npz',**starts,shared_anchor_count=np.asarray(4))
                datasets[name]=(s,starts,independent)
                for dataset,data in (('ordinary',ordinary),('counterexample',counter)):
                    for route in ROUTES:
                        last=None
                        for fraction in ((1.,) if smoke else (.25,.5,1.)):
                            guard();size=max(16,int(n*fraction));subset={k:v[:size] for k,v in data.items()}
                            t0=time.perf_counter();model,diagnostics=fit(route,subset,s['train'],seed)
                            elapsed=time.perf_counter()-t0
                            prediction=model.predict_D(subset['observed_x'],subset['action'],train_env)
                            label=value(subset['observed_next_x'])-value(subset['observed_x'])
                            val=independent['validation'];vp=model.predict_D(val['x'],val['action'],test_env)
                            curves.append(dict(scenario=name,dataset=dataset,route=route,samples=size,
                                training_label_D_rmse=float(np.sqrt(np.mean((prediction-label)**2))),
                                validation_true_D_rmse=float(np.sqrt(np.mean((vp-val['true_D'])**2))),fit_seconds=elapsed))
                            last=(model,diagnostics)
                        model,diagnostics=last;calibration=calibrate(model,independent['calibration'],train_env)
                        dest=output/'checkpoints'/name/dataset;dest.mkdir(parents=True,exist_ok=True)
                        path=dest/(route+'.npz');model.save(path)
                        fit_records.append(dict(scenario=name,dataset=dataset,route=route,samples=n,diagnostics=diagnostics,calibration=calibration,
                            model_file=str(path.relative_to(output)),sha256=sha(path)))
                        models[name,dataset,route]=(model,calibration)
                        frozen.append(dict(path=str(path.relative_to(output)),sha256=sha(path)))
                for mode in MODES:
                    dataset='counterexample' if mode=='counterexample' else 'ordinary'
                    for route in ROUTES:
                        model,calibration=models[name,dataset,route]
                        trace,summary=rollout(model,calibration,test_env,starts['validation'],mode,guard,candidates)
                        save_rollout(output/'validation'/name/mode/route,trace,summary)
                        validation_scores.append(dict(scenario=name,mode=mode,route=route,samples=n,**summary))
                print(json.dumps(dict(event='disk_scenario_fitted',index=si+1,planned=len(planned),scenario=name)),flush=True)
            csv_rows(output/'training_curves.csv',curves);csv_rows(output/'validation_scores.csv',validation_scores)
            write(output/'fit_diagnostics.json',fit_records)
            write(output/'FROZEN.json',dict(schema='book_disk_frozen_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                selection='Predeclared final data budget; validation is diagnostic and never changes hyperparameters',models=frozen))
            for si,(name,(s,starts,independent)) in enumerate(datasets.items()):
                guard();env=DiskEnv(s['test'])
                for mode in MODES:
                    dataset='counterexample' if mode=='counterexample' else 'ordinary'
                    for route in ROUTES:
                        model,calibration=models[name,dataset,route]
                        trace,summary=rollout(model,calibration,env,starts['test'],mode,guard,candidates)
                        folder=output/'test'/name/mode/route
                        audit=save_rollout(folder,trace,summary)
                        region=region_audit(model,calibration,env,mode,folder);write(folder/'region_audit.json',region)
                        pred=model.predict_D(independent['test']['x'],independent['test']['action'],env)
                        transition_rmse=float(np.sqrt(np.mean((pred-independent['test']['true_D'])**2)))
                        scores.append(dict(scenario=name,mode=mode,route=route,samples=s['count'],kappa=s['test'].kappa,
                            dt=s['test'].dt,response=s['test'].response,**summary,heldout_transition_D_rmse=transition_rmse,
                            grid_margin_coverage=region['coverage'],endpoint_audit_passed=audit['passed']))
                print(json.dumps(dict(event='disk_scenario_tested',index=si+1,planned=len(planned),scenario=name)),flush=True)
            csv_rows(output/'test_scores.csv',scores)
            paired=[]
            for name in datasets:
                for mode in MODES:
                    with np.load(output/'test'/name/mode/'known_model'/'trajectories.npz') as base:
                        base_ok=base['success'];base_time=base['arrival_time']
                        for route in ROUTES[1:]:
                            with np.load(output/'test'/name/mode/route/'trajectories.npz') as sample:
                                common=sample['success']&base_ok
                                paired.append(dict(scenario=name,mode=mode,route=route,baseline='known_model',
                                    gained=int((sample['success']&~base_ok).sum()),lost=int((~sample['success']&base_ok).sum()),
                                    common_successes=int(common.sum()),
                                    paired_arrival_difference_seconds=float(np.mean(sample['arrival_time'][common]-base_time[common])) if common.any() else None))
            csv_rows(output/'paired_comparisons.csv',paired)
            for item in frozen:
                if sha(output/item['path'])!=item['sha256']:raise RuntimeError('Frozen model changed during testing')
            counter_env=DiskEnv(DiskConfig())
            counter_x=-.004;counter_action=.05;counter_next=float(counter_env.step(counter_x,counter_action))
            write(output/'domain_counterexample.json',dict(x=counter_x,action=counter_action,next_x=counter_next,
                actual_D=float(value(counter_next)-value(counter_x)),input_limit_satisfied=True,
                work_domain_preserved=bool(counter_env.in_domain(counter_next)),
                safe_action_upper=float(counter_env.safe_upper(counter_x)),
                interpretation='V decreases while the successor crosses the one-sided target; D descent alone does not prove domain preservation'))
            make_plots(output,curves,scores)
            lines=['# 第8章一维圆盘四路线实验','',
                '对象为固定接触侧、受约束的一维准静态圆盘。V=ξ²/2，四路线共享动作限幅和响应上界；已知模型路线另有真实动力学信息。该实验不包含图像表示、平面自由接触或硬件。','',
                f'运行类型：{"工程短测" if smoke else "正式单种子实验"}；seed={seed}；场景设置 {len(planned)} 组。训练、误差标定、验证与测试转移分别采样；验证与测试的四个边界锚点共用并已标记，其余起点独立。','',
                '| 场景 | 模式 | 路线 | 成功/总数 | 域退出率 | 最大真实下降违例 | 估计可行动作空集率 |',
                '|---|---|---|---:|---:|---:|---:|']
            for r in scores:
                lines.append(f"| {r['scenario']} | {r['mode']} | {r['route']} | {r['successes']}/{r['episodes']} | {r['domain_exit_rate']:.3g} | {r['actual_descent_max_violation']:.3g} | {r['empty_estimated_action_set_rate']:.3g} |")
            lines+=['','校准裕度是独立样本残差的最大值；测试中的覆盖率和反例均保留。网格审计没有提供网格间误差界，因此不是整个区域的形式认证。已知响应上界则给出单向域保持的解析条件。',
                    '','域反例另存 domain_counterexample.json：ξ=-4 mm、a=0.05 m/s、Ts=0.1 s、κ=1 时，后继为+1 mm，V下降但越过单侧工作域。正式控制器保留域约束，因此该反例不混入训练或测试成功统计。',
                    '','E05附带结果限于有限窗口信息矩阵、固定反馈/多增益/随机动作和训练测量噪声。没有实施机械对象、频率扫描或窗口积分辨识，未标记E05整体完成。',
                    '','输入平方和是命令代理；保持时间是物理采样时间，不含拟合与求解墙钟。时间比较仅统计共同成功的配对起点。','',
                    '文件：protocol.json；FROZEN.json；fit_diagnostics.json；training_curves.csv；validation_scores.csv；test_scores.csv；paired_comparisons.csv；data/；checkpoints/；validation/；test/；figures/。']
            (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            final=dict(status='complete',formal=not smoke,scenarios=len(planned),controller_tests=len(scores),
                all_endpoint_audits_passed=all(r['endpoint_audit_passed'] for r in scores),report=str(output/'REPORT.md'))
    except BaseException as error:
        final=dict(status='stopped' if isinstance(error,(TimeoutError,InterruptedError,KeyboardInterrupt)) else 'failed',
                   error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        final.update(wall_seconds=time.time()-begun,finished_utc=datetime.now(timezone.utc).isoformat())
        write(output/'FINAL_STATE.json',final)
    return final
