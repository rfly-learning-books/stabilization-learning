"""Parallel compute, one formal artifact writer; use frozen numerical functions."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/book_m_night_suite_20261023_v1'
for path in (ROOT/'stabilization-gym', ROOT/'stabilization-learning-control', ROOT/'scripts', ROOT/'scripts/analysis'):
    sys.path.insert(0,str(path))

import numpy as np
import torch
import train_book_m_suite as legacy
import audit_book_m_results as scientific
import audit_book_m_requested_scope as requested

_cfg = None
_identity = None
_scratch = None
_evaluators = {}
_states = {}


def now():
    return datetime.now().astimezone().isoformat()


def initialize(cfg, identity, scratch, run_path=None):
    global _cfg,_identity,_scratch,RUN
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _cfg,_identity,_scratch=cfg,identity,Path(scratch)
    if run_path is not None:
        RUN=Path(run_path)


def source_gate():
    protocol=legacy.read(RUN/'protocol.json')
    for relative,expected in protocol['sources'].items():
        if legacy.sha(ROOT/relative)!=expected or legacy.sha(RUN/'provenance'/relative)!=expected:
            raise RuntimeError('Frozen source mismatch: '+relative)
    if len(protocol['sources'])!=51:
        raise RuntimeError('Unexpected source inventory')
    identity=requested.original_cache_identity(protocol['sources']['scripts/audit_book_m_results.py'],protocol['sources'])
    return protocol,identity


def load_evaluator(job):
    setting,seed,factor=job['setting'],job['train_seed'],job['factor']
    key=(setting['id'],seed,factor)
    if key not in _evaluators:
        _evaluators[key]=legacy.make_learner(setting,_cfg,seed,scale=factor)
    evaluator,physical=_evaluators[key]
    folder=RUN/(setting['id']+f'_seed{seed}')
    if (setting['id'],seed) not in _states:
        _states[(setting['id'],seed)]={
            'identity':torch.load(folder/'initial_policy.pt',map_location='cpu',weights_only=False),
            'initial':torch.load(folder/'initial_policy.pt',map_location='cpu',weights_only=False),
            'learned':torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['actor'],
            'final':torch.load(folder/'final_policy.pt',map_location='cpu',weights_only=False)['actor']}
    evaluator.actor.load_state_dict(_states[(setting['id'],seed)][job['method']])
    return evaluator,physical,folder


def audit_cached(job):
    """Pure numerical computation on an already sealed row/trajectory pair."""
    started=time.perf_counter()
    directory=RUN/(job['setting']['id']+f"_seed{job['train_seed']}")/('test_scale_'+str(job['factor']))
    path=directory/f"{job['method']}_{job['seed']}.json"
    row=legacy.read(path)
    row_sha=legacy.sha(path)
    trajectory=directory/f"{job['method']}_{job['seed']}.npz"
    if row.get('trajectory')!=trajectory.name or legacy.sha(trajectory)!=row['trajectory_sha256']:
        raise RuntimeError('Sealed trajectory changed: '+str(path))
    key=hashlib.sha256((_identity+row_sha+row['trajectory_sha256']).encode()).hexdigest()
    cache=RUN/'independent_audit_cache'/(key+'.json')
    audited=legacy.read(cache) if cache.exists() else None
    hit=audited is not None
    if audited is not None and not requested.cache_valid(audited,row,scientific):
        raise RuntimeError('Existing content-keyed audit cache is invalid: '+str(cache))
    if audited is None or job.get('verify_policy'):
        evaluator,physical,_=load_evaluator(job)
        with np.load(trajectory,allow_pickle=False) as arrays:
            if job.get('verify_policy'):
                obs,_=evaluator.env.reset(seed=job['seed'])
                with torch.no_grad():
                    metric=(torch.eye(evaluator.env.error_dim) if job['method']=='identity'
                            else evaluator.actor(evaluator._context(obs)))
                if (row['policy_hash']!=legacy.module_hash(evaluator.actor)
                        or not np.array_equal(obs,arrays['states'][0]) or not np.array_equal(metric.numpy(),arrays['M'])):
                    raise RuntimeError('Policy/M/initial-state binding differs: '+str(path))
            if audited is None:
                audited=scientific.audit_episode(physical,arrays,row,decay=_cfg['training']['decay'],
                    certificate_weight=_cfg['training']['certificate_weight'])
    if not requested.cache_valid(audited,row,scientific):
        raise RuntimeError('Independent episode computation failed: '+str(path))
    if row_sha!=legacy.sha(path) or row['trajectory_sha256']!=legacy.sha(trajectory):
        raise RuntimeError('Sealed inputs changed during computation')
    return {'row':row,'audit':audited,'cache_key':key,'cache_hit':hit,
            'seconds':time.perf_counter()-started,'new_episode':False,'job':job}


def generate_episode(job):
    """Original evaluator writes only private scratch; parent commits the run."""
    started=time.perf_counter()
    evaluator,physical,_=load_evaluator(job)
    # Each persistent evaluator gets a single safe-stop wrapper.
    if not getattr(evaluator,'_fast_stop_guard',False):
        legacy.guard_physical_boundaries(evaluator,lambda:(RUN/'STOP').exists())
        evaluator._fast_stop_guard=True
    directory=_scratch/(job['setting']['id']+f"_seed{job['train_seed']}")/str(job['factor'])/f"{job['method']}_{job['seed']}"
    directory.mkdir(parents=True,exist_ok=True)
    row=legacy.evaluate_one(evaluator,physical,job['method'],job['seed'],directory)
    row_path=directory/f"{job['method']}_{job['seed']}.json"
    trajectory=directory/row['trajectory']
    with np.load(trajectory,allow_pickle=False) as arrays:
        audited=scientific.audit_episode(physical,arrays,row,decay=_cfg['training']['decay'],
            certificate_weight=_cfg['training']['certificate_weight'])
    if not requested.cache_valid(audited,row,scientific):
        raise RuntimeError('New episode failed independent numerical checks')
    key=hashlib.sha256((_identity+legacy.sha(row_path)+row['trajectory_sha256']).encode()).hexdigest()
    return {'row':row,'audit':audited,'cache_key':key,'cache_hit':False,
        'row_bytes':row_path.read_bytes(),'trajectory_bytes':trajectory.read_bytes(),
        'seconds':time.perf_counter()-started,'new_episode':True,'job':job}


def compute(job):
    return generate_episode(job) if job.get('generate') else audit_cached(job)


def atomic_bytes(path,data):
    path=Path(path)
    temporary=path.with_name(path.name+f'.fast_v2_{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def commit(result,evidence):
    job,row=result['job'],result['row']
    if result['new_episode']:
        directory=RUN/(job['setting']['id']+f"_seed{job['train_seed']}")/('test_scale_'+str(job['factor']))
        target=directory/row['trajectory'];row_path=target.with_suffix('.json')
        if row_path.exists():
            raise RuntimeError('Refusing to overwrite a sealed episode row')
        if target.exists():
            orphan=evidence/'uncommitted_before_recovery';orphan.mkdir(exist_ok=True)
            shutil.copy2(target,orphan/(str(job['factor'])+'_'+target.name))
        atomic_bytes(target,result['trajectory_bytes'])
        if legacy.sha(target)!=row['trajectory_sha256']:
            raise RuntimeError('Trajectory commit hash differs')
        atomic_bytes(row_path,result['row_bytes'])
    cache=RUN/'independent_audit_cache'/(result['cache_key']+'.json')
    if not cache.exists():
        requested.write_new(cache,result['audit'])


def check_policy_artifacts(folder):
    frozen=legacy.read(folder/'training_complete.json')
    for name,expected in frozen['immutable_sha256'].items():
        if legacy.sha(folder/name)!=expected:
            raise RuntimeError('Immutable policy changed: '+str(folder/name))
    if legacy.sha(folder/'last.pt')!=frozen['last_sha256']:
        raise RuntimeError('Frozen last checkpoint changed')
    return frozen


def run_evaluation(workers=4):
    protocol,identity=source_gate();cfg=protocol['config']
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    evidence=RUN/'monitoring'/('fast_v2_eval_'+stamp);evidence.mkdir()
    requested.write_new(evidence/'EXECUTION_DECLARATION.json',{
        'at':now(),'schema':'book_m_fast_evaluation_execution_v2','parent_pid':os.getpid(),
        'workers':workers,'formal_artifact_writers':1,'allowed_seeds':[41,42],
        'user_instruction':'那就继续训练对齐','sources':protocol['sources'],
        'extension_sha256':legacy.sha(__file__),'numerical_functions':'Unmodified original evaluate_one and audit_episode',
        'new_optimizer_updates':0,'original_protocol_unchanged':True,
        'declaration':'Parallel execution is an external execution amendment, not a claim of an unchanged process topology.'})
    if (RUN/'STOP').exists():
        raise RuntimeError('STOP must be explicitly archived/removed by the authorized supervisor first')
    with legacy.WriterLock(ROOT/'runs/book_m_suite_slot.lock'),legacy.WriterLock(RUN/'writer.lock'):
        if any(RUN.glob('*_seed43')):
            raise RuntimeError('Seed43 is excluded before its later decision')
        setting=next(task for task in cfg['tasks'] if task['id']=='multirotor_circle')
        folder=RUN/'multirotor_circle_seed42'
        frozen=check_policy_artifacts(folder)
        jobs=[]
        for factor in cfg['test_initial_scale_factors']:
            directory=folder/('test_scale_'+str(factor));directory.mkdir(exist_ok=True)
            expected={'factor':factor,'task':setting,'training_seed':42,
                'test_seeds':list(range(20001,20097)),'frozen_training':frozen,'protocol_sha256':legacy.sha(RUN/'protocol.json')}
            if (directory/'evaluation_protocol.json').exists():
                if legacy.read(directory/'evaluation_protocol.json')!=expected:
                    raise RuntimeError('Evaluation protocol differs')
            else:
                legacy.write(directory/'evaluation_protocol.json',expected)
            for method in ('identity','initial','learned','final'):
                for seed in range(20001,20097):
                    jobs.append({'setting':setting,'train_seed':42,'factor':factor,'method':method,'seed':seed,
                        'generate':not (directory/f'{method}_{seed}.json').exists(),'verify_policy':True})
        missing=sum(job['generate'] for job in jobs)
        requested.write_new(evidence/'COVERAGE_AT_START.json',{'at':now(),'existing':len(jobs)-missing,'missing':missing})
        records={factor:[] for factor in cfg['test_initial_scale_factors']}
        starts={};last_resource=0.;started=time.monotonic();generated=0
        cache_dir=RUN/'independent_audit_cache';cache_dir.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(cfg,identity,str(evidence/'scratch'),str(RUN))) as pool:
            for count,result in enumerate(pool.map(compute,jobs,chunksize=1),1):
                if (RUN/'STOP').exists():
                    raise RuntimeError('STOP requested during accelerated evaluation')
                if time.monotonic()-last_resource>=15:
                    reason,snapshot=legacy.resources(RUN,cfg);last_resource=time.monotonic()
                    if reason:
                        raise RuntimeError('Resource gate: '+reason)
                    legacy.write(RUN/'status.json',{'stage':'multirotor_circle_seed42:evaluation','stop_reason':None,
                        'timestamp':now(),'execution_backend':'fast_v2_four_compute_one_writer',
                        'verified_rows':count,'new_episodes':generated,'resources':snapshot})
                commit(result,evidence)
                job=result['job'];records[job['factor']].append(result['row'])
                starts[(job['factor'],job['method'],job['seed'])]=result['audit']['initial_state']
                generated+=int(result['new_episode'])
                if count%32==0 or count==len(jobs):
                    progress={'at':now(),'verified':count,'total':len(jobs),'new_episodes':generated,
                        'new_total':missing,'wall_seconds':time.monotonic()-started,'last_job_seconds':result['seconds']}
                    with (evidence/'PROGRESS.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(progress)+'\n')
                    print(json.dumps(progress),flush=True)
        for factor,rows in records.items():
            for seed in range(20001,20097):
                base=starts[(factor,'identity',seed)]
                if any(starts[(factor,method,seed)]!=base for method in ('initial','learned','final')):
                    raise RuntimeError('Paired initial states differ')
            legacy.write(folder/('test_scale_'+str(factor))/'SUMMARY.json',legacy.paired_summary(rows))
        best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        initial=torch.load(folder/'initial_policy.pt',map_location='cpu',weights_only=False)
        final=torch.load(folder/'final_policy.pt',map_location='cpu',weights_only=False)['actor']
        completed={'task':'multirotor_circle','seed':42,'best_selected_update':best['selected_update'],
            'best_is_initial':best['selected_update']==0,
            'evaluations':{str(factor):legacy.paired_summary(rows) for factor,rows in records.items()},
            'final_training_sha256':legacy.sha(folder/'final_training.pt'),'best_sha256':legacy.sha(folder/'best.pt'),
            'actor_parameters_changed':any(not torch.equal(final[key],initial[key]) for key in final),
            'policy_effect_probes':legacy.read(folder/'POLICY_EFFECT_PROBES.json'),
            'candidate_certificate_only':True,'artifact_sha256':legacy.completion_hashes(folder),
            'independent_final_audit':'pending; completion means evaluation coverage only'}
        legacy.write(folder/'complete.json',completed)
        legacy.report(RUN,cfg)
        legacy.write(RUN/'status.json',{'stage':'multirotor_circle_seed42:evaluation','stop_reason':'user_STOP',
            'timestamp':now(),'requested_scope_evaluation_complete':True,'candidate_certificate_only':True})
        if not (RUN/'STOP').exists():
            requested.write_new(RUN/'STOP',{'at':now(),'schema':'book_m_fast_v2_requested_scope_complete',
                'scope':[41,42],'reason':'Complete requested scope; independent audit and seed43 value decision required'})
        requested.write_new(evidence/'COMPLETE.json',{'at':now(),'new_episodes':generated,'total':1152,
            'wall_seconds':time.monotonic()-started,'STOP_preserved':True})
        return evidence


def run_audit_cache(workers=4):
    requested.guard_stopped(RUN)
    protocol,identity=source_gate();cfg=protocol['config']
    evidence=RUN/'monitoring'/('fast_v2_cache_'+datetime.now().strftime('%Y%m%d_%H%M%S'));evidence.mkdir()
    with requested.StoppedAuditLocks(ROOT,RUN):
        requested.guard_stopped(RUN)
        if any(RUN.glob('*_seed43')):
            raise RuntimeError('Excluded seed directory')
        jobs=[]
        for setting in cfg['tasks']:
            for seed in (41,42):
                folder=RUN/(setting['id']+f'_seed{seed}')
                if not (folder/'complete.json').exists():
                    raise RuntimeError('Incomplete requested run: '+str(folder))
                for factor in cfg['test_initial_scale_factors']:
                    for method in ('identity','initial','learned','final'):
                        for test_seed in range(20001,20097):
                            jobs.append({'setting':setting,'train_seed':seed,'factor':factor,'method':method,'seed':test_seed})
        requested.write_new(evidence/'START.json',{'at':now(),'episodes':len(jobs),'workers':workers,
            'artifact_writers':1,'extension_sha256':legacy.sha(__file__),'original_cache_identity':identity})
        started=time.monotonic();hits=0
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(cfg,identity,str(evidence/'scratch'),str(RUN))) as pool:
            for count,result in enumerate(pool.map(compute,jobs,chunksize=4),1):
                if not (RUN/'STOP').exists():
                    raise RuntimeError('Persistent STOP disappeared during cache audit')
                commit(result,evidence);hits+=int(result['cache_hit'])
                if count%256==0 or count==len(jobs):
                    progress={'at':now(),'audited':count,'total':len(jobs),'cache_hits':hits,
                        'wall_seconds':time.monotonic()-started}
                    with (evidence/'PROGRESS.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(progress)+'\n')
                    print(json.dumps(progress),flush=True)
        requested.write_new(evidence/'COMPLETE.json',{'at':now(),'episodes':len(jobs),
            'cache_hits':hits,'wall_seconds':time.monotonic()-started})
    return evidence
