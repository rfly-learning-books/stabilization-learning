"""Post-exit scientific summary of fixed-coordinate eta-return RGB training."""
from pathlib import Path
import json
import time
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def writer_check(snapshot, root):
    s = read(snapshot)
    if s.get('schema')!='unified_training_process_snapshot' or s.get('query_success') is not True:
        raise ValueError('Successful external process enumeration required')
    age = time.time()-s['created_unix']
    if age<0 or age>60: raise ValueError('Process snapshot must be no more than 60 seconds old')
    if not isinstance(s.get('processes'),list): raise ValueError('Complete Python process list required')
    norm = str(Path(root).resolve()).replace('\\','/').lower()
    for p in s['processes']:
        command = p.get('CommandLine') or p.get('command_line')
        if not command or not p.get('ProcessId') or p.get('ParentProcessId') is None:
            raise ValueError('Unknown Python identity is not evidence of exit')
        cmd = command.replace('\\','/').lower()
        if norm in cmd or ('run_rgb_eta_return_v1.py' in cmd and ' train ' in cmd):
            raise RuntimeError('Training/evaluation/report writer still present: '+command)
    return s


def report(root, output, snapshot):
    root,output = Path(root),Path(output)
    writer_check(snapshot,root)
    if (root/'FINAL_STATE.json').exists():
        state=read(root/'FINAL_STATE.json')
    else:
        supervisor=read(Path(__file__).resolve().parents[2]/'handoff'/(root.name+'_SUPERVISOR.json'))
        if supervisor['status']=='running': raise RuntimeError('No completed supervisor state')
        completed=[read(p) for p in sorted((root/'events').glob('*_epoch_complete.json'))]
        last=completed[-1] if completed else {}
        state=dict(status=supervisor['status'],error='Worker exited without FINAL_STATE; inspect supervisor and stderr',
            completed_epochs=last.get('epoch',0),selected_epoch=last.get('selected_epoch'),gpu={})
    protocol = read(root/'protocol.json')
    output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    updates = [json.loads(s) for s in (root/'UPDATES.jsonl').read_text(encoding='utf-8').splitlines()] if (root/'UPDATES.jsonl').exists() else []
    figure,axes = plt.subplots(3,2,figsize=(13,11))
    for ax,key in zip(axes.flat,('ae_loss','q_loss','performance','m_loss','calibration','replay_descent')):
        points = [(i,r[key]) for i,r in enumerate(updates) if key in r]
        if points: ax.plot(*zip(*points),lw=.65)
        ax.set_title(key); ax.set_xlabel('all logged optimizer calls'); ax.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(output/'all_updates.png',dpi=160); plt.close(figure)
    gradient_rows=[r for r in updates if 'gradient_performance' in r]
    if gradient_rows:
        figure,axes=plt.subplots(1,2,figsize=(13,4))
        for key in ('performance','calibration','replay_descent','physical_descent','direction'):
            axes[0].plot([r['gradient_'+key] for r in gradient_rows],label=key)
            if key!='performance': axes[1].plot([r['cosine_to_return_'+key] for r in gradient_rows],label=key)
        axes[0].set_yscale('symlog',linthresh=1e-6); axes[0].set_title('Weighted M-gradient norms')
        axes[1].set_ylim(-1.05,1.05); axes[1].set_title('Gradient cosine to return objective')
        for ax in axes: ax.set_xlabel('recorded common-batch gradient audits'); ax.legend(fontsize=7); ax.grid(alpha=.2)
        figure.tight_layout(); figure.savefig(output/'metric_gradients.png',dpi=160); plt.close(figure)
    val = [(int(p.stem.split('_')[1]),read(p)) for p in sorted(root.glob('VAL_*.json'))]
    figure,axes = plt.subplots(1,2,figsize=(12,4))
    for plant in protocol['config']['evaluation_plants']:
        points = [(e,v['groups']['selected/'+plant]) for e,v in val]
        if points:
            axes[0].plot([e for e,_ in points],[r['successes']/r['episodes'] for _,r in points],label=plant+' arrival')
            axes[0].plot([e for e,_ in points],[r['hold_successes']/r['episodes'] for _,r in points],'--',label=plant+' +hold')
            axes[1].plot([e for e,_ in points],[r['mean_integrated_error'] for _,r in points],label=plant)
    for ax in axes: ax.legend(); ax.set_xlabel('epoch (0=initial M)'); ax.grid(alpha=.2)
    axes[0].set_title('Validation only: arrival and additional hold'); axes[1].set_title('Validation integrated physical error')
    figure.tight_layout(); figure.savefig(output/'validation.png',dpi=160); plt.close(figure)
    probes = [(int(p.stem.split('_')[-1]),read(p)) for p in sorted(root.glob('M_PROBE_*.json'))]
    if probes:
        figure,axes = plt.subplots(1,3,figsize=(13,4))
        for e,key in ((0,'eigen_min'),(1,'eigen_max')): axes[0].plot([n for n,_ in probes],[r[key] for _,r in probes],label=key)
        axes[0].legend(); axes[0].set_title('Fixed training goal metric eigenvalues')
        im=axes[1].imshow(probes[0][1]['mean_matrix']); axes[1].set_title('Initial mean M'); figure.colorbar(im,ax=axes[1])
        difference=np.asarray(probes[-1][1]['mean_matrix'])-np.asarray(probes[0][1]['mean_matrix'])
        bound=max(float(np.abs(difference).max()),1e-8)
        im=axes[2].imshow(difference,vmin=-bound,vmax=bound,cmap='coolwarm'); axes[2].set_title('Final - initial mean M'); figure.colorbar(im,ax=axes[2])
        figure.tight_layout(); figure.savefig(output/'metric.png',dpi=160); plt.close(figure)
    tests = read(root/'TEST.json') if (root/'TEST.json').exists() else None
    diagnostics = {}; paired = {}
    if tests:
        for key in tests['groups']:
            variant,plant = key.split('/'); ratios=[]; deltas=[]; cosines=[]; predicted_down_actual_up=[]
            for path in (root/'trajectories').glob(f'TEST_{variant}_{plant}_*.npz'):
                with np.load(path,allow_pickle=False) as a:
                    mask=a['active']; ratios.extend((a['V']/(a['energy']+1e-8))[mask].tolist())
                    change=a['V_next']-a['V']; deltas.extend(change[mask].tolist())
                    if 'applied_action' in a:
                        prediction=.1*np.einsum('tbdi,tbi->tbd',a['interaction'],a['applied_action'])
                        actual=a['next_error']-a['error']; denom=np.linalg.norm(prediction,axis=-1)*np.linalg.norm(actual,axis=-1)
                        valid=mask&(denom>1e-10)
                        cos=(prediction*actual).sum(-1)/(denom+1e-12); cosines.extend(cos[valid].tolist())
                        predicted_v=.5*np.einsum('tbi,tbij,tbj->tb',a['error']+prediction,a['matrix'],a['error']+prediction)
                        predicted_down_actual_up.extend(((predicted_v<a['V'])&(change>0))[valid].tolist())
            diagnostics[key]=dict(transitions=len(ratios),VE_median=float(np.median(ratios)) if ratios else None,
                V_increase_fraction=float(np.mean(np.asarray(deltas)>1e-7)) if deltas else None,
                VE_below_training_lower=float(np.mean(np.asarray(ratios)<.25)) if ratios else None,
                J_mean_cosine=float(np.mean(cosines)) if cosines else None,
                J_opposite_fraction=float(np.mean(np.asarray(cosines)<0)) if cosines else None,
                predicted_down_actual_up_fraction=float(np.mean(predicted_down_actual_up)) if predicted_down_actual_up else None)
        for plant in protocol['config']['evaluation_plants']:
            selected={r['case_id']:r for r in tests['episodes'] if r['plant']==plant and r['variant']=='selected'}
            for variant in ('identity','initial_metric'):
                other={r['case_id']:r for r in tests['episodes'] if r['plant']==plant and r['variant']==variant}
                common=[k for k in selected if selected[k]['success'] and other[k]['success']]
                paired[plant+'/'+variant]=dict(common_successes=len(common),
                    arrival_time_difference_selected_minus_reference=float(np.mean([.1*(selected[k]['first_success_step']-other[k]['first_success_step']) for k in common])) if common else None,
                    selected_only=sum(selected[k]['success'] and not other[k]['success'] for k in selected),
                    reference_only=sum(other[k]['success'] and not selected[k]['success'] for k in selected))
    result=dict(state=state,validation=[dict(epoch=e,groups=v['groups']) for e,v in val],
        test=tests['groups'] if tests else None,certificate_and_J=diagnostics,paired_arrival_time=paired)
    (output/'ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# RGB：η 回报 Critic 与 Lyapunov M 实验', '',
        f"状态：{state['status']}；完成 {state['completed_epochs']} 轮；所选轮次 {state['selected_epoch']}。", '',
        'Critic 输入为视觉上下文与 136 维 η，学习折扣累计物理奖励。η 构造 Cholesky M；M 仍用于 V 和原目标 J 视觉伺服控制器。', '',
        '本轮使用随机初始化的原 CAE 架构进行预热，随后固定编码器及坐标标定。回报 Critic 和 M 都从新参数、新回放及空优化器开始。采用双 Q、目标网络与确定性 M 策略，没有 SAC 熵项。', '',
        '## 独立测试', '', '| 模型/工况 | 到达 | 到达后持续保持 | 物理失败 | 误差积分 | 输入平方 |', '|---|---:|---:|---:|---:|---:|']
    if tests:
        for key,r in tests['groups'].items():
            lines.append(f"| {key} | {r['successes']}/{r['episodes']} | {r['hold_successes']}/{r['episodes']} | {r['failures']} | {r['mean_integrated_error']:.6g} | {r['mean_input_squared']:.6g} |")
    else: lines += ['未完成正式测试；现有验证结果不能作为独立测试增益。']
    gpu=state.get('gpu',{}).get('time_weighted_utilization')
    lines += ['', '## 训练与证书检查', '',
        '验证曲线包含第 0 轮和全部已完成轮次；M 图使用相同训练目标上下文。保留回升、退化和失败结果。', '',
        '![全部更新](all_updates.png)', '', '![验证曲线](validation.png)', '', '![M 曲线](metric.png)', '',
        '![M 损失梯度](metric_gradients.png)', '',
        f"全程整卡 GPU 时间加权利用率：{gpu:.2f}%。" if gpu is not None else 'GPU 利用率数据不可用。', '',
        'ANALYSIS.json 保留完整 V/E、实际 V 回升、目标 J 方向误差和共同成功起点的配对到达时间。AUDIT.json 保留同参数完整后继 V 和控制指令的独立数值复算。', '',
        '回放 V 下降项约束的是行为数据；不能把它直接当作当前新 η 动作的反事实后继。保留的 P0 物理/方向约束与累计回报项共同优化 M，二者的梯度范数与夹角已记录，不能将全部收益独归因于 Critic。', '',
        '## 解释范围', '',
        '单 seed41；44 个既有重建房间；公共 FastDepth；GPU 渲染像素。P0 为四自由度速度外环，P1 增加延迟和滤波。推理墙钟没有计入 dt，输入平方是速度代价代理。M 正定及软下降损失不构成新的稳定性证明。', '',
        '本轮特征初始化、固定特征阶段及评价起点均与历史书稿实验不同，因此不能用总成功率差直接判断累计回报 Critic 的因果收益，也不能声称复现了书稿结论。', '',
        f'原始运行：{root.as_posix()}']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
