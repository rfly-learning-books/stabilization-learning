"""A serial train/evaluate/audit pipeline for the versioned four-task benchmark."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .core import (ROOT,TASKS,DirectLearner,SparseValidationConfig,initial_states,batch_rollout)
from . import audit
import numpy as np
import torch
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash,atomic_save
from train_book_m_suite import WriterLock

OLD=ROOT/'runs/book_m_night_suite_20261023_v1'
METHODS=('lqr_baseline','learned','final')
TITLE={'coupled_linear':'耦合线性系统','point_mass':'平面质点',
       'pendulum_local':'局部倒立单摆','cartpole_local':'小车倒立摆'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def now():return datetime.now().astimezone().isoformat()


def emit(out,stage,**values):
    event=dict(at=now(),stage=stage,**values)
    # A committed immutable snapshot can be inspected during a writer's lifetime.
    destination=out/'events'/f'{stage}_{time.time_ns()}.json'
    destination.parent.mkdir(exist_ok=True)
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(event,stream,ensure_ascii=False,allow_nan=False)
    print(json.dumps(event,ensure_ascii=False,allow_nan=False),flush=True)


def config():
    old=read(ROOT/'configs/book_m_night_suite.json')
    return dict(schema='direct_lqr_metric_benchmark_v1',tasks=list(TASKS),seeds=[41,42],
        training=dict(old['training'],validation_every_updates=5),
        test_seeds=list(range(20001,20097)),scales=[1.,1.5,2.],methods=list(METHODS),
        learning_scope='Per-episode fixed M, goal-only context; not state-dependent M or SDRE',
        controller='u=sat(u_ff-R^-1 B0^T M e)',metric='M=L0 S L0.T, P0=L0 L0.T; S has original trace/spectral bounds',
        initialization='S=I recovers P0 and the frozen nominal LQR controller',
        objective='unchanged fixed-horizon physical cost + 0.1 original normalized empirical descent penalty',
        selection='Initial, every-five-update and final validation only; test never selects or changes a policy',
        initial_is_baseline=True,initial_not_duplicate_evaluation=True,seed43_started=False,
        physical_protocol='Original dt, integrator, input limits, reset distribution, horizons and final 1s hold',
        expected_training_runs=8,expected_training_episodes=9600,expected_evaluation_episodes=6912,
        original_run_untouched=True)


def parameters(cfg,seed):
    values=dict(cfg['training'],seed=seed)
    values['validation_seeds']=tuple(values['validation_seeds'])
    return SparseValidationConfig(**values)


def source_inventory():
    previous=read(OLD/'protocol.json')['sources']
    for relative,digest in previous.items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Original frozen source changed: '+relative)
    sources=dict(previous)
    for path in sorted((ROOT/'extensions/lqr_metric_direct_v1').glob('*.py')):
        sources[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    for relative in ('extensions/fast_m_learning_v2/learner.py','scripts/analysis/run_lqr_metric_benchmark_v1.py'):
        sources[relative]=sha(ROOT/relative)
    return sources


def guard_no_worker():
    query=("$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '^python(w)?\\.exe$' -and (( $_.CommandLine -match "
        "'run_lqr_metric_benchmark_v1.py' -and $_.CommandLine -match '--mode (train|evaluate)') "
        "-or $_.CommandLine -match 'train_book_m|train_sparse_m') } | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Depth 3")
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],
        capture_output=True,text=True,check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.stdout.strip() and json.loads(result.stdout) not in (None,[]):
        raise RuntimeError('An active formal writer must exit before setup/audit')


def check_sources(out):
    protocol=read(out/'protocol.json')
    for relative,digest in protocol['sources'].items():
        if sha(ROOT/relative)!=digest or sha(out/'provenance'/relative)!=digest:
            raise RuntimeError('Version-bound source changed: '+relative)
    if sha(OLD/'STOP')!=protocol['original_STOP_sha256']:
        raise RuntimeError('Original completed-run STOP changed')
    for relative,digest in protocol['original_training_artifacts'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Original completed training artifact changed')
    return protocol


def setup(out):
    guard_no_worker()
    if (out/'protocol.json').exists():return check_sources(out)
    cfg=config();sources=source_inventory()
    from .preflight import run_checks
    preflight=run_checks(out/'preflight')
    if not preflight['passed']:raise RuntimeError('Preflight did not pass')
    if source_inventory()!=sources:raise RuntimeError('Source changed during preflight')
    artifacts={}
    for task in TASKS:
        for seed in cfg['seeds']:
            folder=OLD/f'{task}_seed{seed}'
            for name in ('initial_policy.pt','best.pt','final_policy.pt','last.pt'):
                path=folder/name;artifacts[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    protocol=dict(at=now(),user_instruction='基于这个取代原有的LQR组，再做一次基准训练',
        config=cfg,sources=sources,preflight=preflight,
        original_protocol_sha256=sha(OLD/'protocol.json'),original_STOP_sha256=sha(OLD/'STOP'),
        original_training_artifacts=artifacts)
    for relative in sources:
        destination=out/'provenance'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,destination)
    write(out/'protocol.json',protocol)
    write(out/'README_STATE.json',{'state':'declared_preflight_passed','at':now(),'config':cfg})
    emit(out,'declared',training_runs=8,training_episodes=9600,test_episodes=6912,
         preflight_checks=len(preflight['checks']))
    return protocol


def stopper(out):
    last_check=[0.]
    def requested():
        if (out/'STOP').exists():return True
        if time.monotonic()-last_check[0]>15:
            if shutil.disk_usage(ROOT).free<50*1024**3:raise RuntimeError('Disk free below 50GiB')
            last_check[0]=time.monotonic()
        return False
    return requested


def train(out):
    cfg=check_sources(out)['config'];stop=stopper(out)
    with WriterLock(ROOT/'runs/book_m_suite_slot.lock'),WriterLock(out/'writer.lock'):
        for task in TASKS:
            for seed in cfg['seeds']:
                folder=out/f'{task}_seed{seed}';folder.mkdir(exist_ok=True)
                if (folder/'training_complete.json').exists():
                    completed=read(folder/'training_complete.json')
                    if any(sha(folder/name)!=digest for name,digest in completed['artifacts'].items()):
                        raise RuntimeError('Completed training bundle changed')
                    continue
                learner=DirectLearner(task,config=parameters(cfg,seed));learner.should_stop=stop
                initial=folder/'initial_policy.pt'
                if not initial.exists():atomic_save(deepcopy(learner.actor.state_dict()),initial)
                resume=(folder/'last.pt').exists()
                if resume:learner.restore(folder/'last.pt')
                emit(out,'training_started',task=task,seed=seed,resumed=resume,episodes=learner.episodes)
                started=time.perf_counter()
                # train(max_updates=1) keeps the tested sparse validation/resume machinery.
                while learner.episodes<learner.config.episodes:
                    learner.train(folder,resume=(folder/'last.pt').exists(),max_updates=1)
                    if learner.updates%5==0 or learner.episodes>=learner.config.episodes:
                        selected=learner._selected()
                        emit(out,'training_progress',task=task,seed=seed,updates=learner.updates,
                            episodes=learner.episodes,physical_steps=learner.physical_steps,
                            selected_update=selected['updates'],best_validation_objective=learner.best_score,
                            wall_seconds=time.perf_counter()-started)
                atomic_save(deepcopy(learner.actor.state_dict()),folder/'final_policy.pt')
                complete=dict(at=now(),task=task,seed=seed,episodes=learner.episodes,updates=learner.updates,
                    physical_steps=learner.physical_steps,best_selected_update=learner._selected()['updates'],
                    initialization='S=I, physical M=P0',P0=learner.physical.inner_env.P0.tolist(),
                    actor_metric_scope='episode_fixed',optimizer_nonempty=bool(learner.optimizer.state_dict()['state']),
                    artifacts={name:sha(folder/name) for name in ('initial_policy.pt','best.pt','final_policy.pt','last.pt')},
                    wall_seconds=time.perf_counter()-started)
                write(folder/'training_complete.json',complete)
                emit(out,'training_complete',task=task,seed=seed,selected_update=complete['best_selected_update'],
                     wall_seconds=complete['wall_seconds'])
                learner.physical.close()
    emit(out,'all_training_complete',training_runs=8)


def load_policy(learner,folder,method):
    name={'lqr_baseline':'initial_policy.pt','learned':'best.pt','final':'final_policy.pt'}[method]
    payload=torch.load(folder/name,map_location='cpu',weights_only=False)
    state=payload['actor'] if method=='learned' else payload
    learner.actor.load_state_dict(state)
    return module_hash(learner.actor),sha(folder/name)


def evaluate(out):
    cfg=check_sources(out)['config'];stop=stopper(out)
    with WriterLock(ROOT/'runs/book_m_suite_slot.lock'),WriterLock(out/'writer.lock'):
        for task in TASKS:
            for seed in cfg['seeds']:
                folder=out/f'{task}_seed{seed}'
                completion=read(folder/'training_complete.json')
                if any(sha(folder/name)!=digest for name,digest in completion['artifacts'].items()):
                    raise RuntimeError('Training artifacts changed before evaluation')
                for factor in cfg['scales']:
                    destination=folder/f'test_scale_{factor:.1f}';destination.mkdir(exist_ok=True)
                    learner=DirectLearner(task,config=parameters(cfg,seed),scale=factor)
                    seeds=cfg['test_seeds'];starts=initial_states(learner.physical,seeds)
                    contexts=[learner._context(start) for start in starts]
                    for method in METHODS:
                        seal=destination/f'{method}_complete.json'
                        if seal.exists():
                            previous=read(seal)
                            if any(sha(destination/name)!=digest for name,digest in previous['artifacts'].items()):
                                raise RuntimeError('Sealed evaluation artifacts changed')
                            continue
                        policy_hash,checkpoint_hash=load_policy(learner,folder,method)
                        with torch.no_grad():
                            raw=(learner.actor.center if method=='lqr_baseline' else learner.actor.mean(contexts[0]))
                            matrix=learner.actor.metric(raw).numpy()
                            shape=learner.actor.metric.shape(raw).numpy()
                        rows,arrays=batch_rollout(learner.physical,matrix,starts,
                            certificate_weight=cfg['training']['certificate_weight'],decay=cfg['training']['decay'],
                            record=True,should_stop=stop)
                        artifacts={}
                        for i,(test_seed,row) in enumerate(zip(seeds,rows)):
                            n=row['physical_steps'];stem=f'{method}_{test_seed}'
                            z={key:value[:n,i].copy() for key,value in arrays.items()}
                            z.update(M=matrix,S=shape,P0=learner.physical.inner_env.P0,
                                raw_policy=raw.detach().numpy(),goal=learner.physical.goal,
                                dt=np.asarray(learner.physical.dt),times=np.arange(1,n+1)*learner.physical.dt,
                                required_hold_seconds=np.asarray(1.))
                            recomputed,checked=audit.measurements(task,z,
                                decay=cfg['training']['decay'],certificate_weight=cfg['training']['certificate_weight'])
                            if not checked['passed']:raise RuntimeError('Evaluation data failed independent numerical recomputation: '+str(checked))
                            for key,value in recomputed.items():
                                if not math.isclose(float(value),float(row[key]),abs_tol=1e-8,rel_tol=1e-8):
                                    raise RuntimeError('Batch and independent measurement differ: '+key)
                            temporary=destination/(stem+'.tmp.npz')
                            np.savez_compressed(temporary,**z)
                            target=destination/(stem+'.npz');os.replace(temporary,target)
                            row.update(task=task,method=method,seed=test_seed,training_seed=seed,scale=factor,
                                policy_hash=policy_hash,checkpoint_sha256=checkpoint_hash,
                                trajectory=target.name,trajectory_sha256=sha(target),
                                independent_episode_audit=checked,metric_role='physical_M_not_normalized_S')
                            write(destination/(stem+'.json'),row)
                            for name in (stem+'.npz',stem+'.json'):artifacts[name]=sha(destination/name)
                        write(seal,dict(at=now(),method=method,episodes=len(rows),artifacts=artifacts))
                        emit(out,'evaluation_complete',task=task,seed=seed,scale=factor,method=method,
                            episodes=len(rows),successes=sum(r['success'] for r in rows),
                            mean_cost=float(np.mean([r['fixed_horizon_cost'] for r in rows])))
                    learner.physical.close()
    emit(out,'all_evaluation_complete',episodes=cfg['expected_evaluation_episodes'])


def audit_all(out):
    guard_no_worker();protocol=check_sources(out);cfg=protocol['config']
    groups=[];count=0;passed=0;definition=[];training=[]
    with WriterLock(ROOT/'runs/book_m_suite_slot.lock'),WriterLock(out/'writer.lock'):
        for task in TASKS:
            for seed in cfg['seeds']:
                folder=out/f'{task}_seed{seed}'
                complete=read(folder/'training_complete.json');training.append(complete)
                if any(sha(folder/name)!=digest for name,digest in complete['artifacts'].items()):
                    raise RuntimeError('Frozen training bundle changed during audit')
                checkpoint=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
                best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
                validated=[row for row in checkpoint['history'] if row.get('validated')]
                selected=min(validated,key=lambda row:row['validation']['objective'])
                if (checkpoint['episodes']!=1200 or checkpoint['updates']!=50 or len(checkpoint['history'])!=51
                    or [row['updates'] for row in validated]!=list(range(0,51,5))
                    or best['selected_update']!=selected['updates'] or best['score']!=selected['validation']['objective']
                    or not checkpoint['optimizer']['state']):
                    raise RuntimeError('Training budget, optimizer or validation selection differs')
                for factor in cfg['scales']:
                    learner=DirectLearner(task,config=parameters(cfg,seed),scale=factor)
                    destination=folder/f'test_scale_{factor:.1f}';by_method={}
                    initial_by_method={}
                    for method in METHODS:
                        policy_hash,checkpoint_hash=load_policy(learner,folder,method)
                        rows=[];initial=[]
                        for test_seed in cfg['test_seeds']:
                            row=read(destination/f'{method}_{test_seed}.json')
                            path=destination/row['trajectory']
                            if sha(path)!=row['trajectory_sha256'] or row['policy_hash']!=policy_hash or row['checkpoint_sha256']!=checkpoint_hash:
                                raise RuntimeError('Policy/checkpoint/trajectory provenance differs')
                            with np.load(path,allow_pickle=False) as z:
                                expected_start=initial_states(learner.physical,[test_seed])[0]
                                context=learner._context(expected_start)
                                with torch.no_grad():
                                    raw=learner.actor.center if method=='lqr_baseline' else learner.actor.mean(context)
                                    expected_M=learner.actor.metric(raw).numpy()
                                if not np.array_equal(expected_start,z['states'][0]) or not np.array_equal(expected_M,z['M']):
                                    raise RuntimeError('Seed/reset/policy-to-M provenance differs')
                                measured,checked=audit.measurements(task,z,
                                    decay=cfg['training']['decay'],certificate_weight=cfg['training']['certificate_weight'])
                                if not checked['passed']:raise RuntimeError('Post-writer numerical audit failed')
                                for key,value in measured.items():
                                    if not math.isclose(float(value),float(row[key]),abs_tol=1e-8,rel_tol=1e-8):
                                        raise RuntimeError('Saved measurement differs: '+key)
                                initial.append(z['states'][0].copy())
                                if factor==1. and test_seed==20001:
                                    definition.append(dict(task=task,training_seed=seed,method=method,
                                        **audit.local_certificate(task,z['M'])))
                            rows.append(row);count+=1;passed+=1
                        by_method[method]=rows;initial_by_method[method]=np.stack(initial)
                    if not all(np.array_equal(initial_by_method['lqr_baseline'],initial_by_method[m]) for m in METHODS):
                        raise RuntimeError('Unpaired comparison initial states')
                    old_folder=OLD/f'{task}_seed{seed}'/f'test_scale_{factor:.1f}'
                    old_learned=[read(old_folder/f'learned_{s}.json') for s in cfg['test_seeds']]
                    old_baseline=[read(old_folder/f'identity_{s}.json') for s in cfg['test_seeds']]
                    def mean(method,key):return float(np.mean([r[key] for r in by_method[method]]))
                    summary={m:{key:mean(m,key) for key in ('fixed_horizon_cost','success','arrival_time',
                        'V_rise_fraction','actual_decay_violation_fraction','terminal_physical_error','action_total_variation')}
                        for m in METHODS}
                    base=np.array([r['fixed_horizon_cost'] for r in by_method['lqr_baseline']])
                    learned=np.array([r['fixed_horizon_cost'] for r in by_method['learned']])
                    previous=np.array([r['fixed_horizon_cost'] for r in old_learned])
                    old_base=np.array([r['fixed_horizon_cost'] for r in old_baseline])
                    if not np.allclose(base,old_base,atol=1e-9,rtol=1e-8):
                        raise RuntimeError('New P0 baseline differs from old LQR baseline physics')
                    delta=learned-base
                    groups.append(dict(task=task,seed=seed,scale=factor,n=96,summary=summary,
                        learned_cost_change_percent=100*(learned.mean()/base.mean()-1),
                        learned_vs_previous_learned_cost_percent=100*(learned.mean()/previous.mean()-1),
                        paired_mean_cost_delta=float(delta.mean()),
                        paired_delta_95_normal_approx_halfwidth=float(1.96*delta.std(ddof=1)/math.sqrt(96)),
                        confidence_scope='96 test initial states conditional on one trained policy, not 96 training replicates',
                        previous_baseline_matched=True,previous_learned_cost=float(previous.mean()),
                        policy_M_distinctness=distinct_matrices(destination)))
                    learner.physical.close()
                    emit(out,'audit_group_complete',task=task,seed=seed,scale=factor,verified_episodes=count)
    result=dict(at=now(),passed=count==cfg['expected_evaluation_episodes'] and passed==count,
        training_runs=len(training),evaluation_episodes=count,comparison_groups=len(groups),
        expected_evaluation_episodes=cfg['expected_evaluation_episodes'],training=training,groups=groups,
        local_definition_evidence=definition,original_frozen_sources_unchanged=51,
        original_training_artifacts_unchanged=len(protocol['original_training_artifacts']),
        candidate_certificate_only=True,seed43_started=False,
        interpretation='Audit passage establishes coverage, provenance and numerical measurements; performance and mathematical stabilization are separate findings')
    write(out/'AUDIT.json',result)
    write_report(out,result)
    emit(out,'audit_complete',passed=result['passed'],episodes=count,groups=len(groups))
    return result


def distinct_matrices(folder):
    matrices=[]
    for method in METHODS:
        with np.load(folder/f'{method}_20001.npz',allow_pickle=False) as z:matrices.append(z['M'].copy())
    unique=[]
    for M in matrices:
        if not any(np.array_equal(M,other) for other in unique):unique.append(M)
    return len(unique)


def write_report(out,result):
    lines=['# 直接M增益：四任务重新训练与基准评价','',
        '控制器为 `u=sat(u_ff-R^-1 B0^T M e)`，证书为 `V=e^T M e`。',
        '实际M采用 `M=L0 S L0^T`；S单位初始化恢复原LQR的P0。B0沿用原局部设计输入矩阵，M每回合固定；本轮不是状态相关M或SDRE实验。','',
        f"完成 {result['training_runs']} 组训练、{result['comparison_groups']} 组尺度对照、{result['evaluation_episodes']} 条评价；种子41/42，每组1200回合/50次更新。",'',
        '训练仍优化物理成本加原下降罚项，按验证集选模型，未用测试结果调参。初始策略与P0基线相同，不重复计作独立评价臂。','',
        '|任务|训练种子|尺度|学习M成本变化|相对旧学习策略成本变化|基线成功|学习成功|学习下降违例|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        base=g['summary']['lqr_baseline'];learned=g['summary']['learned']
        lines.append(f"|{TITLE[g['task']]}|{g['seed']}|{g['scale']:.1f}|{g['learned_cost_change_percent']:+.2f}%|{g['learned_vs_previous_learned_cost_percent']:+.2f}%|{round(base['success']*96)}/96|{round(learned['success']*96)}/96|{learned['actual_decay_violation_fraction']*100:.3f}%|")
    improved=sum(g['learned_cost_change_percent']<0 for g in result['groups'])
    successes={m:sum(round(g['summary'][m]['success']*96) for g in result['groups']) for m in METHODS}
    numerical_same=sum(d['same_M_discrete_descent'] for d in result['local_definition_evidence'] if d['method']=='learned')
    lines+=['',f'学习M相对P0基线成本降低 {improved}/24组；成功数：{successes}。',
        f'8个学习策略中，局部名义离散线性模型的同M严格下降数值检查通过 {numerical_same}/8。',
        '该局部数值检查不代替饱和域、非线性吸引域、扰动鲁棒性或浮点严格证明。',
        '归一化S的正定约束只保证实际M正定，不单独保证闭环下降或收敛。',
        '本轮保留所有不利结果，未启动seed43，未改动旧51份冻结源码和原训练权重。','',
        '完整来源、初态、策略到M绑定、独立NumPy物理/证书/成本/保持复算见 [AUDIT.json](AUDIT.json)。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(out):
    protocol=setup(out)
    if (out/'STOP').exists():raise RuntimeError('New-run STOP exists; preserve it until explicit resume')
    script=ROOT/'scripts/analysis/run_lqr_metric_benchmark_v1.py'
    started=time.perf_counter()
    for mode in ('train','evaluate','audit'):
        guard_no_worker()
        args=[sys.executable,'-I',str(script),'--mode',mode,'--output',str(out)]
        process=subprocess.Popen(args,cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',bufsize=1)
        write(out/f'{mode}.START.json',dict(at=now(),pid=process.pid,arguments=args))
        log=out/f'{mode}.{time.time_ns()}.log'
        with log.open('x',encoding='utf-8') as stream:
            for line in process.stdout:
                stream.write(line);stream.flush()
                print(line,end='',flush=True)
        code=process.wait()
        write(out/f'{mode}.EXIT.json',dict(at=now(),returncode=code))
        if code:raise RuntimeError(mode+' stage exited '+str(code))
    result=read(out/'AUDIT.json')
    if not result['passed']:raise RuntimeError('Final audit failed')
    check_sources(out)
    write(out/'FINAL_STATE.json',dict(at=now(),complete=True,formal_training_active=False,
        training_runs=8,training_episodes=9600,evaluation_episodes=6912,audit_passed=True,
        seed43_started=False,original_STOP_preserved=True,total_wall_seconds=time.perf_counter()-started))
    if not (out/'STOP').exists():write(out/'STOP',dict(at=now(),reason='completed_new_four_task_benchmark',automatic_resume=False))
    emit(out,'complete',training_runs=8,evaluation_episodes=6912,wall_seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('preflight','run','train','evaluate','audit'),default='run')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out==ROOT/'runs' or out.is_relative_to(OLD):
        raise ValueError('Use a separate versioned directory under runs')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.mode=='preflight':setup(out)
    elif args.mode=='run':run(out)
    elif args.mode=='train':train(out)
    elif args.mode=='evaluate':evaluate(out)
    else:audit_all(out)

