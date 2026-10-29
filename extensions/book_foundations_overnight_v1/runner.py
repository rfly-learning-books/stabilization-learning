"""Fixed-source textbook numerical reproduction; no learning claims."""
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from zipfile import ZipFile
import numpy as np
from scipy.linalg import eigvalsh
from .env import simulate,exact_affine,zoh,scalar_integrator
from .algo import pi_action,virtual_output_action,resonance_controller

ROOT=Path(__file__).resolve().parents[2]
BOOK=Path('E:/rfly-books/Rfly-Stable-Learning-Book_Reviewed_20261022')
MATERIALS=BOOK/'docs/section37_materials'
SOURCE_LINES=dict(dc_motor=['ch2/chapter2.tex',219,245],sampled_integrator=['ch8/chapter8.tex',112,112],
    PI=['ch8/chapter8.tex',2317,2341],PI_conditions=['ch6/chapter6.tex',1282,1418],
    virtual_output=['ch6/chapter6.tex',1047,1075],hidden_state=['ch8/chapter8.tex',2314,2314],
    discounted_counterexample=['ch8/chapter8.tex',2270,2270],resonance=['ch3/section37_same_energy.tex',126,340])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    def numpy_scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f'Unsupported JSON value {type(value).__name__}')
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=numpy_scalar),encoding='utf-8');temp.rename(path)


def save_trace(folder,name,data):
    np.savez_compressed(folder/(name+'.npz'),**data)
    length=len(data['time']);names=[];columns=[]
    for key,a in data.items():
        a=np.asarray(a)
        if a.ndim and len(a)==length:
            if a.ndim==1:names.append(key);columns.append(a)
            elif a.ndim==2:
                for i in range(a.shape[1]):names.append(f'{key}_{i}');columns.append(a[:,i])
    with (folder/(name+'.csv')).open('x',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(names);writer.writerows(zip(*columns))


def save_figure(fig,folder,name):
    fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=150);fig.savefig(folder/(name+'.pdf'))


def foundation_cases(output,guard,smoke):
    import matplotlib.pyplot as plt
    raw=output/'raw';figures=output/'figures';results={}
    A=np.array([[-100.,-10.],[10.,-10.]]);B=np.array([100.,0.]);Ad,Bd=zoh(A,B,.01)
    printed_A=np.array([[.365340,-.059564],[.059564,.901415]]);printed_B=np.array([[.631113],[.035473]])
    errors=(float(np.max(abs(Ad-printed_A))),float(np.max(abs(Bd-printed_B))))
    results['dc_motor_ZOH']=dict(Ad=Ad.tolist(),Bd=Bd.tolist(),printed_rounding_max_error=max(errors),
        tolerance=5.01e-7,passed=max(errors)<=5.01e-7,source=SOURCE_LINES['dc_motor'])
    np.savez_compressed(raw/'dc_motor_ZOH.npz',A=A,B=B,Ad=Ad,Bd=Bd)
    fig,axes=plt.subplots(1,2,figsize=(11,4));integrator=[]
    for period,limit in ((.2,None),(1.1,None),(.2,1.),(1.1,1.)):
        guard();t,x,u=scalar_integrator(.4,period,12 if smoke else 30,limit)
        expected=.4*(1-2*period)**np.arange(len(t)) if limit is None else None
        label=f'T={period}, '+('unlimited' if limit is None else '|u|<=1')
        axes[0].plot(t,x,label=label);axes[1].semilogy(t,np.maximum(x*x,1e-20),label=label)
        name=f'integrator_T{period}_limit{limit}'
        save_trace(raw,name,dict(time=t,state=x,action=np.r_[u,np.nan],V=x*x))
        integrator.append(dict(period=period,limit=limit,initial=.4,final=float(x[-1]),input_peak=float(abs(u).max()),
            actual_successor_max_error=float(np.max(abs(x[1:]-x[:-1]-period*u))),
            analytic_max_error=float(np.max(abs(x-expected))) if expected is not None else None,
            bounded_sample_period_condition=0<2*period<2))
    axes[0].set(xlabel='Physical time (s)',ylabel='State',title='Executed sampled feedback')
    axes[1].set(xlabel='Physical time (s)',ylabel='V=x^2',title='Sampling and saturation boundary')
    axes[0].legend(fontsize=7);save_figure(fig,figures,'sampled_integrator');plt.close(fig)
    results['sampled_integrator']=dict(source=SOURCE_LINES['sampled_integrator'],cases=integrator,
        passed=all(c['actual_successor_max_error']<1e-12 for c in integrator))

    time_axis=np.linspace(0.,8.,401);virtual=[];fig,axes=plt.subplots(1,3,figsize=(13,4))
    A=np.array([[0.,1.],[-1.,1.]]);B=np.array([0.,1.]);reference=1.;gain=2.
    for route in ('original_output','virtual'):
        def rhs(t,s):return A@s+B*virtual_output_action(s,reference,gain,route)
        trajectory=simulate(rhs,[0.,0.],time_axis,guard)
        ctrl=np.array([virtual_output_action(s,reference,gain,route) for s in trajectory])
        feedback=np.array([1-gain,-2-gain]) if route=='virtual' else np.array([1+gain,-gain])
        offset=-gain*reference if route=='virtual' else gain*reference
        truth=exact_affine(A+np.outer(B,feedback),B*offset,[0.,0.],time_axis)
        output_y=-trajectory[:,0]+trajectory[:,1]
        error=float(np.max(abs(trajectory-truth)))
        save_trace(raw,'virtual_'+route,dict(time=time_axis,state=trajectory,truth=truth,output=output_y,action=ctrl))
        axes[0].plot(time_axis,output_y,label=route);axes[1].plot(time_axis,trajectory[:,0],label=route)
        axes[2].semilogy(time_axis,np.maximum(np.linalg.norm(trajectory-np.array([-1.,0.]),axis=1),1e-15),label=route)
        virtual.append(dict(route=route,maximum_truth_error=error,final_output_error=float(abs(output_y[-1]-1)),
            internal_state_peak=float(np.max(abs(trajectory))),input_peak=float(abs(ctrl).max()),
            complete_state_success=bool(np.linalg.norm(trajectory[-1]-np.array([-1.,0.]))<.01),
            eigenvalues_real=np.linalg.eigvals(A+np.outer(B,feedback)).real.tolist()))
    axes[0].axhline(1,color='k',ls='--');axes[0].set(xlabel='Time (s)',ylabel='Original output y',title='Shared target y=1')
    axes[1].set(xlabel='Time (s)',ylabel='Internal state s1',title='Internal dynamics retained')
    axes[2].set(xlabel='Time (s)',ylabel='Full-state error norm',title='Output tracking alone is insufficient')
    axes[0].legend(fontsize=8);save_figure(fig,figures,'virtual_output_comparison');plt.close(fig)
    results['virtual_output']=dict(source=SOURCE_LINES['virtual_output'],new_parameters=dict(k=2.,r=1.,initial=[0.,0.]),
        cases=virtual,passed=all(c['maximum_truth_error']<1e-5 for c in virtual),
        interpretation='Numerically integrated physical ODE compared to independent affine matrix exponential')

    hidden=simulate(lambda t,s:np.array([-s[0],s[1],s[0],s[0]**2]),[0.,.1,0.,0.],time_axis,guard)
    hidden_truth=.1*np.exp(time_axis)
    save_trace(raw,'hidden_state',dict(time=time_axis,state=hidden,hidden_truth=hidden_truth))
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(time_axis,hidden[:,0],label='output');axes[0].plot(time_axis,hidden[:,2],ls='--',label='integral of output')
    axes[1].semilogy(time_axis,hidden[:,1],label='unobservable internal state')
    for ax in axes:ax.set_xlabel('Time (s)');ax.legend(fontsize=8)
    axes[0].set_title('Zero output history');axes[1].set_title('Hidden state diverges')
    save_figure(fig,figures,'output_history_counterexample');plt.close(fig)
    results['hidden_state']=dict(source=SOURCE_LINES['hidden_state'],maximum_output=float(abs(hidden[:,0]).max()),
        maximum_error_integral=float(abs(hidden[:,2:]).max()),final_hidden_state=float(hidden[-1,1]),
        truth_max_error=float(np.max(abs(hidden[:,1]-hidden_truth))),passed=np.max(abs(hidden[:,1]-hidden_truth))<1e-6)

    P=np.array([[.5,.5],[.5,1.5]]);Acl=np.array([[-2.,-1.],[1.,0.]])
    residual=Acl.T@P+P@Acl+np.eye(2)
    results['PI_certificate']=dict(source=SOURCE_LINES['PI'],P=P.tolist(),A=Acl.tolist(),
        eigenvalues_P=np.linalg.eigvalsh(P).tolist(),Lyapunov_residual=float(abs(residual).max()),passed=abs(residual).max()<1e-14)
    pi_results=[];t=np.linspace(0.,20.,1001)
    cases=(('constant',.3,None),('bounded_feasible',.3,.5),('bounded_infeasible',1.2,.5),('sinusoidal',.3,None))
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for index,(name,amp,limit) in enumerate(cases):
        for route in ('P','PI','known_feedforward'):
            disturbance=(lambda t:amp*np.sin(t)) if name=='sinusoidal' else (lambda t:amp)
            def rhs(t,state):
                actual,_=pi_action(state,disturbance(t),route,limit)
                return np.array([actual+disturbance(t),state[0] if route=='PI' else 0.])
            initial=[1.,0.];trajectory=simulate(rhs,initial,t,guard)
            actions=np.array([pi_action(s,disturbance(ti),route,limit) for ti,s in zip(t,trajectory)])
            exact_error=None
            if name=='constant':
                Am=Acl if route=='PI' else np.array([[-2.,0.],[0.,0.]])
                b=np.array([amp if route!='known_feedforward' else 0.,0.])
                truth=exact_affine(Am,b,initial,t);exact_error=float(abs(trajectory-truth).max())
            centered=trajectory-np.array([0.,amp if route=='PI' and name!='sinusoidal' else 0.])
            V=np.einsum('bi,ij,bj->b',centered,P,centered) if route=='PI' and name!='sinusoidal' else trajectory[:,0]**2/2
            save_trace(raw,f'PI_{name}_{route}',dict(time=t,state=trajectory,actual_action=actions[:,0],requested_action=actions[:,1],V=V,
                disturbance=np.array([disturbance(ti) for ti in t])))
            tail=t>=18.;steady_rmse=float(np.sqrt(np.mean(trajectory[tail,0]**2)))
            pi_results.append(dict(case=name,route=route,disturbance_amplitude=amp,limit=limit,
                final_error=float(abs(trajectory[-1,0])),tail_RMSE=steady_rmse,
                state_peak=float(abs(trajectory).max()),integrator_peak=float(abs(trajectory[:,1]).max()),
                input_peak=float(abs(actions[:,0]).max()),saturation_fraction=float(np.mean(abs(actions[:,0]-actions[:,1])>1e-10)),
                equilibrium_input_feasible=bool(limit is None or amp<=limit),
                terminal_tail_success=bool(steady_rmse<.01 and np.max(abs(trajectory))<100.),
                analytic_max_error=exact_error))
            axes.flat[index].plot(t,trajectory[:,0],label=route)
        axes.flat[index].set(xlabel='Time (s)',ylabel='Physical state/error',title=name);axes.flat[index].legend(fontsize=8)
    save_figure(fig,figures,'PI_applicability');plt.close(fig)
    results['PI_tracking']=dict(source=SOURCE_LINES['PI_conditions'],fixed_gains=dict(kp=2.,ki=1.),cases=pi_results,
        added_experimental_parameters='disturbance .3 or 1.2, limit .5, sinusoid angular frequency 1; declared extensions, not printed numerical values',
        passed=all(c['analytic_max_error'] is None or c['analytic_max_error']<1e-8 for c in pi_results))
    k=np.arange(16);x=2.**k;vg=5*x*x/3
    series=np.sum((.4)**np.arange(80));identity=.1*(5*(2*x)**2/3)-vg+x*x
    save_trace(raw,'discounted_counterexample',dict(time=k.astype(float),state=x,discounted_value=vg,ordinary_difference=np.r_[np.diff(vg),np.nan]))
    results['discounted_counterexample']=dict(source=SOURCE_LINES['discounted_counterexample'],
        partial_sum=series,exact_at_x1=5/3,ordinary_value_multiplier=4.,
        discounted_identity_relative_error=float(np.max(abs(identity)/x**2)),
        passed=abs(series-5/3)<1e-14,interpretation='Finite discounted value does not imply ordinary Lyapunov descent')
    return results


def import_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def resonance(output,guard,smoke):
    import matplotlib.pyplot as plt
    raw=output/'raw';figures=output/'figures'
    needed=['compute_example_Q.py','example_Qw.csv','example_closed_loop_F.csv','example_controller_report.json','resonance_safe_proof_bundle.zip']
    missing=[str(MATERIALS/n) for n in needed if not (MATERIALS/n).is_file()]
    if missing:return dict(status='dependency_missing',missing=missing,completed=False,passed=False)
    source=output/'source_vendor';source.mkdir()
    qmodule=import_source('foundations_retained_Q',MATERIALS/'compute_example_Q.py')
    guard();q8=qmodule.make_Q(8);guard();q12=qmodule.make_Q(12) if not smoke else q8.copy()
    stored=np.loadtxt(MATERIALS/'example_Qw.csv',delimiter=',',skiprows=1)
    Q=q12;F0,G,H,b,K,F=resonance_controller(Q,qmodule.A,qmodule.B)
    Fstored=np.loadtxt(MATERIALS/'example_closed_loop_F.csv',delimiter=',')
    power=np.linalg.matrix_power(F,256);D=.5*Q-power.T@Q@power
    numerical=dict(Q_order8_vs12_max_abs=float(abs(q8-q12).max()) if not smoke else None,
        Q_vs_retained_max_abs=float(abs(Q-stored).max()),F_vs_retained_max_abs=float(abs(F-Fstored).max()),
        minimum_Q_eigenvalue=float(eigvalsh(Q)[0]),minimum_contraction_matrix_eigenvalue=float(eigvalsh(D)[0]),
        maximum_256_step_energy_ratio=float(eigvalsh(power.T@Q@power,Q)[-1]),
        spectral_radius=float(max(abs(np.linalg.eigvals(F)))),H=float(H),
        floating_point_recheck=True,interval_certification_reexecuted=False)
    np.savez_compressed(raw/'resonance_matrices.npz',Q=Q,Q8=q8,F0=F0,G=G,H=H,b=b,K=K,F=F,D=D)
    initial=np.zeros(36);initial[0]=1.;trajectory=[initial]
    for k in range(512):guard();trajectory.append(F@trajectory[-1])
    trajectory=np.array(trajectory);V=np.einsum('bi,ij,bj->b',trajectory,Q,trajectory)
    save_trace(raw,'resonance_unforced',dict(time=np.arange(len(V)),state=trajectory,V=V,action=-trajectory@K))
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].plot(np.arange(len(V)),trajectory[:,0],label='actual y');axes[0].plot(np.arange(len(V)),trajectory[:,1],label='actual velocity')
    axes[1].semilogy(np.arange(len(V)),V,label='same fixed Q energy');axes[0].legend();axes[1].legend()
    axes[0].set(xlabel='Time (s)',ylabel='State',title='Executed unforced augmented dynamics');axes[1].set(xlabel='Time (s)',ylabel='V',title='256-step contraction recheck')
    save_figure(fig,figures,'resonance_unforced');plt.close(fig)
    names=('same_energy_resonance_model.py','same_energy_integrated_notch_model.py','expanded_window_search.py')
    with ZipFile(MATERIALS/'resonance_safe_proof_bundle.zip') as archive:
        for name in names:
            dest=(source/name).resolve()
            if not dest.is_relative_to(source.resolve()):raise ValueError('Invalid retained-source path')
            dest.write_bytes(archive.read('outputs/'+name))
    sys.path.insert(0,str(source));import_source(names[0][:-3],source/names[0]);geometry=import_source(names[1][:-3],source/names[1]);response=import_source(names[2][:-3],source/names[2]).response
    g=geometry.Geometry(1,geometry.T,24).configure(1.025,0.,1.,0.)
    frequencies=np.array([.5,.997,1.09509144,2.]);amps=np.array([.004,.02,.004,.004]);phase=np.array([.37,-.71,.53,1.13])
    forced=[];traces=[]
    for h in ((.02,) if smoke else (.02,.01)):
        guard();end=64 if smoke else 2048;count=round(end/h);hold=round(1/h);lag=round(33/h)
        t=np.arange(count+1)*h;hist=np.zeros((lag+count+1,2));uncontrolled=np.zeros((count+1,2));u=np.zeros(count+1)
        q=np.arange(-lag,1)*h;weights=np.full(lag+1,h);weights[[0,-1]]*=.5
        cw=g.cfun(q-1)*weights[:,None];bx=np.einsum('q,qi,qij->j',g.wp,g.cf,g.Efuture)
        E=geometry.ev(h);gh=geometry.gamma(h);inc=np.zeros((count,2));held=0.
        for w,amp,p in zip(frequencies,amps,phase):
            rr=np.linalg.solve(1j*w*np.eye(2)-geometry.A,geometry.B)
            inc+=amp*np.imag(np.exp(1j*(w*t[:-1]+p))[:,None]*((np.exp(1j*w*h)*np.eye(2)-E)@rr))
        for k in range(count+1):
            if k%hold==0:
                guard();held=-float((cw*hist[k:k+lag+1]).sum()+bx@hist[k+lag])/(g.H+120.)
            u[k]=held
            if k<count:
                hist[k+lag+1]=E@hist[k+lag]+gh*held+inc[k]
                uncontrolled[k+1]=E@uncontrolled[k]+inc[k]
        states=hist[lag:];force=sum(a*np.sin(w*t+p) for w,a,p in zip(frequencies,amps,phase))
        save_trace(raw,f'resonance_forced_h{h}',dict(time=t,state=states,uncontrolled=uncontrolled,action=u,disturbance=force))
        outcome=dict(h=h,duration=end,actual_history_used=True,steady_state_evaluated=not smoke)
        if not smoke:
            mask=(t>=1536)&(t<2048);tail=t[mask];y=states[mask,0]
            side=np.array([w+2*np.pi*l for l in range(-3,4) for w in frequencies])
            design=np.column_stack([np.ones(len(tail))]+[fn(w*tail) for w in side for fn in (np.sin,np.cos)])
            coeff=np.linalg.lstsq(design,y,rcond=None)[0];amplitude=np.hypot(coeff[1::2],coeff[2::2]).reshape(7,4)
            baseline=amps/abs(1-frequencies**2+.02j*frequencies);expected=response(g,120.,frequencies)
            ratio_db=10*np.log10((amplitude**2).sum(0)/baseline**2)
            outcome.update(frequencies=frequencies.tolist(),measured_same_frequency_amplitudes=amplitude[3].tolist(),
                baseline_amplitudes=baseline.tolist(),all_fitted_sidebands_RMS_ratio_db=ratio_db.tolist(),
                fitted_sideband_orders=list(range(-3,4)),analytic_same_frequency_relative_error=float(np.max(abs(amplitude[3]/(amps*expected[0])-1))),
                fit_residual_rms=float(np.sqrt(np.mean((y-design@coeff)**2))),actual_control_rms=float(np.sqrt(np.mean(u[mask]**2))),
                target_meets_minus7db=bool(ratio_db[1]<=-7),four_frequencies_below_plus1db=bool(ratio_db.max()<=1.001),
                physical_harmonic_agreement_passed=bool(np.max(abs(amplitude[3]/(amps*expected[0])-1))<.002),
                physical_harmonic_relative_tolerance=.002,
                no_continuous_band_certification_claim=True)
        forced.append(outcome);traces.append((t,states,uncontrolled))
    t,controlled,uncontrolled=traces[-1];mask=t>t[-1]-min(40.,t[-1])
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(t[mask],uncontrolled[mask,0],label='actual unforced-control baseline u=0');ax.plot(t[mask],controlled[mask,0],label='actual history-energy feedback')
    ax.set(xlabel='Time (s)',ylabel='Physical displacement',title='Four simultaneous forces; numerically propagated trajectories');ax.legend(fontsize=8)
    save_figure(fig,figures,'resonance_forced_actual');plt.close(fig)
    matrix_passed=numerical['Q_vs_retained_max_abs']<1e-8 and numerical['F_vs_retained_max_abs']<1e-9 and numerical['minimum_Q_eigenvalue']>0 and numerical['maximum_256_step_energy_ratio']<.5
    return dict(completed=True,status='complete',source=SOURCE_LINES['resonance'],numerical=numerical,forced=forced,
        passed=matrix_passed and (smoke or all(c['physical_harmonic_agreement_passed'] for c in forced)),
        physical_performance_requirements_met=None if smoke else all(c['target_meets_minus7db'] and c['four_frequencies_below_plus1db'] for c in forced),
        evidence='Independent floating-point reconstruction and actual physical simulation. Archived interval proof provenance retained, not rerun.')


def run(output,max_minutes=10.,smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if max_minutes<=0 or not np.isfinite(max_minutes):raise ValueError('Finite positive budget required')
    output.mkdir(parents=True);(output/'raw').mkdir();(output/'figures').mkdir()
    begun=time.time();deadline=begun+60*max_minutes
    def guard():
        if (output/'STOP').exists():raise InterruptedError('STOP requested')
        if time.time()>=deadline:raise TimeoutError('Declared foundations budget expired')
    sources=[BOOK/p[0] for p in SOURCE_LINES.values()]
    sources+=list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/run_book_foundations_overnight_v1.py']
    sources+=list(MATERIALS.glob('compute_example_*.py'))+[MATERIALS/n for n in ('example_Qw.csv','example_closed_loop_F.csv','example_controller_report.json','resonance_safe_proof_bundle.zip')]
    protocol=dict(schema='book_foundations_overnight_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        mode='engineering' if smoke else 'formal_numerical_recheck',maximum_minutes=max_minutes,deadline_unix=deadline,
        CPU_threads=1,GPU_used=False,source_lines=SOURCE_LINES,
        source_hashes={str(p):sha(p) for p in sorted(set(sources)) if p.exists()},
        scope='Partial E01/E02/E08: specified low-dimensional examples, not complete methods benchmark or learned model training',
        numerical_thresholds=dict(motor_rounding_abs=5.01e-7,virtual_affine_abs=1e-5,hidden_analytic_abs=1e-6,
            PI_affine_abs=1e-8,resonance_Q_abs=1e-8,resonance_F_abs=1e-9,resonance_harmonic_relative=.002),
        performance_criteria=dict(virtual_terminal_full_state_norm=.01,PI_tail_interval=[18.,20.],
            PI_tail_RMSE=.01,PI_entire_augmented_state_peak=100.,
            resonance_target_total_fitted_sideband_db=-7.,resonance_four_frequency_total_fitted_sideband_max_db=1.001),
        simulated='ODE solve_ivp trajectories, exact held-input updates, actual measured-history resonance control',
        theory_comparison='Affine matrix exponentials and printed rounded matrices are separate reference calculations',
        omitted='No hardware, general nonlinear proof, full tracking frequency/mismatch benchmark, window-length/noisy observability sweep, or fresh interval certification')
    write(output/'protocol.json',protocol);final={}
    try:
        results=foundation_cases(output,guard,smoke);print(json.dumps(dict(event='foundations_ODE_cases_complete')),flush=True)
        results['resonance']=resonance(output,guard,smoke)
        write(output/'RESULTS.json',results)
        checks=[bool(v['passed']) for v in results.values() if 'passed' in v]
        lines=['# Reviewed 书稿基础数值包','',
            '覆盖 E01、E02、E08 中已明确给出方程的部分算例。控制器没有训练过程；图中运动来自数值积分或真实保持输入状态传播，解析解仅作为独立真值。','',
            f'本轮为 {protocol["mode"]}；数值一致性检查通过 {sum(checks)}/{len(checks)}。',
            '','| 算例 | 来源 | 内容 |','|---|---|---|',
            '| 电机采样模型 | 第2章219–245行 | 精确ZOH与正文六位小数核对 |',
            '| 积分器 | 第8章112行 | T=0.2/1.1，连续反馈与采样、输入限幅边界 |',
            '| 虚拟输出 | 第6章1047–1075行 | 原输出与虚拟输出控制；完整内部状态均保留 |',
            '| 输出历史 | 第8章2314行 | 输出和积分为零而隐藏状态发散 |',
            '| PI | 第8章2317–2341行、第6章1282后 | P/PI/真实扰动补偿、可实现与不可实现输入、正弦失配 |',
            '| 折扣反例 | 第8章2270行 | 有限折扣价值与普通下降条件分开 |',
            '| 共振 | 第3章3.7随书材料 | 原Q/F数值复算、256步收缩、真实历史受迫仿真 |','',
            '共振采用原36维增广矩阵与原窗函数。这里重新计算的是浮点数值证据；随书区间证明未重新认证。工程短测仅跑64秒受迫过程，不据此计算稳态抑振结论；正式计算2048秒并用最后512秒拟合。',
            '','PI增加的扰动0.3/1.2、限幅0.5、正弦频率1为本包预先指定工况。常值参考或扰动需要可实现稳态输入且增广闭环稳定；不能把误差输出较小与全状态稳定等同。',
            '','E01/E02/E08均标为部分覆盖：未完成一般任务方法排名、完整鲁棒性与频率扫描、多模型学习消融或噪声下历史窗口研究。所有原始轨迹在raw/，图为figures/下PNG与PDF，数值核验在RESULTS.json。']
        lines+=['','## 本次实测','',f"电机ZOH与书稿六位小数最大差 {results['dc_motor_ZOH']['printed_rounding_max_error']:.3g}。",'',
            '| 输出控制方式 | 终端输出误差 | 内部状态峰值 | 终端完整状态到达 |','|---|---:|---:|---|']
        for c in results['virtual_output']['cases']:
            lines.append(f"| {c['route']} | {c['final_output_error']:.6g} | {c['internal_state_peak']:.6g} | {c['complete_state_success']} |")
        lines+=['','| PI工况 | 控制器 | 最后2秒误差RMSE | 输入峰值 | 积分状态峰值 | 尾段判据 |','|---|---|---:|---:|---:|---|']
        for c in results['PI_tracking']['cases']:
            lines.append(f"| {c['case']} | {c['route']} | {c['tail_RMSE']:.6g} | {c['input_peak']:.6g} | {c['integrator_peak']:.6g} | {c['terminal_tail_success']} |")
        lines+=['','尾段判据：18至20秒误差RMSE小于0.01，且整个过程增广状态绝对值峰值小于100；不是从起点开始持续满足误差容差。数值一致性检查与控制成功结果分别记录。']
        resonance_result=results['resonance']
        if resonance_result.get('completed'):
            rn=resonance_result['numerical']
            lines+=['',f"共振Q与原表最大差 {rn['Q_vs_retained_max_abs']:.3g}，F与原表最大差 {rn['F_vs_retained_max_abs']:.3g}；256步最大广义能量比 {rn['maximum_256_step_energy_ratio']:.8g}。"]
            if not smoke:
                lines+=['','| 物理传播步长 | 目标总边带dB | 四频最大总边带dB | 同频幅值最大相对误差 | 物理要求 |','|---|---:|---:|---:|---|']
                for c in resonance_result['forced']:
                    lines.append(f"| {c['h']} | {c['all_fitted_sidebands_RMS_ratio_db'][1]:.5f} | {max(c['all_fitted_sidebands_RMS_ratio_db']):.5f} | {c['analytic_same_frequency_relative_error']:.6g} | {c['target_meets_minus7db'] and c['four_frequencies_below_plus1db']} |")
                lines+=['','这里总边带仅计入预先选定的−3至3阶；四个固定频率的数值结果不代替连续频带证明。']
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        final=dict(status='complete' if all(checks) else 'failed_numerical_checks',passed=all(checks),
            partial_work_packages=['E01','E02','E08'],checks_passed=sum(checks),checks_total=len(checks),report=str(output/'REPORT.md'))
    except BaseException as error:
        final=dict(status='stopped' if isinstance(error,(TimeoutError,InterruptedError,KeyboardInterrupt)) else 'failed',
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        final.update(wall_seconds=time.time()-begun,finished_utc=datetime.now(timezone.utc).isoformat());write(output/'FINAL_STATE.json',final)
    return final
