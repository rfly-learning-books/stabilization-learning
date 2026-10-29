"""Bounded RGB-only experiment using the shared initialization/epoch/test runner."""
from pathlib import Path
from copy import deepcopy
import json
import os
import shutil
import subprocess
import sys
import time

from stabilization_learning.unified_v1 import runner as shared


ROOT=Path(__file__).resolve().parents[2]


def validate(c):
    shared.validate_config(c)
    if len(c['jobs'])!=1 or c['jobs'][0]['backend']!='rgb_legacy': raise ValueError('One RGB-only experiment required')
    b=c['jobs'][0]['config']
    if not b.get('engineering'):
        if b['validation_cases']!=48 or b['test_cases']!=72 or b['evaluation_steps']!=200:
            raise ValueError('Preserve all48 validation/all72 test starts and200-step limit')
        if c['jobs'][0]['schedule']['epochs']!=20: raise ValueError('Declare all20 rounds before training')
        if b['parallel_envs']!=16 or b['collection_steps']!=120: raise ValueError('Declared GPU/long-trajectory protocol')
    if c['jobs'][0]['schedule']['representation_updates']!=0: raise ValueError('Frozen encoder during Q/M epochs')
    if not 0<b['stability']['gamma']<=.99: raise ValueError('The cost bound requires gamma<=.99')
    if len(b['regular_collection_quotas'])!=c['jobs'][0]['schedule']['epochs']: raise ValueError('Quota/epoch mismatch')
    if b['stability']['aux_gradient_cap']>1 or b['stability']['aux_gradient_cap']<0: raise ValueError('Auxiliary gradient cap')
    return c


def worker(config_path,output):
    c=validate(shared.read(config_path)); deadline=c.get('absolute_deadline_unix',time.time()+60*c['maximum_minutes'])
    c=deepcopy(c); c['maximum_minutes']=min(c['maximum_minutes'],(deadline-time.time())/60)
    if c['maximum_minutes']<=0: raise TimeoutError('Original deadline expired')
    original=shared.source_inventory
    def inventory(project):
        result=original(project)
        folders=('extensions/rgb_cost_m_v2','extensions/rgb_stable_m_v1','extensions/rgb_longroll_v1',
            'extensions/reproducible_training_v4','extensions/unified_training_v2','extensions/unified_training_v3',
            'third_party/visual_original/lac_vs')
        paths=[p for folder in folders for p in (ROOT/folder).rglob('*.py')]
        paths += [ROOT/'scripts/run_rgb_cost_m_v2.py',Path(config_path),Path(c['jobs'][0]['config']['case_manifest'])]
        paths += list((ROOT/'third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1').glob('*state_dict.pt'))
        result.update({str(p.resolve()):shared.sha256(p) for p in paths})
        return result
    def factory(name,cfg,folder,*,event,deadline):
        from .backend import CostMetricRGBBackend
        # Retain the exact new implementation with this run, in addition to the
        # shared runner's hashes of all dependencies.
        archive=Path(folder).parents[1]/'provenance'
        for subdir in ('extensions/rgb_cost_m_v2','extensions/rgb_stable_m_v1','stabilization-learning-control/stabilization_learning/rgb_stable'):
            for source in (ROOT/subdir).glob('*.py'):
                target=archive/source.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
        return CostMetricRGBBackend(cfg,folder,event=event,deadline=deadline)
    shared.source_inventory=inventory
    try: return shared.run_suite(c,output,project=ROOT,factory=factory)
    finally: shared.source_inventory=original


def supervise(config_path,output):
    c=validate(shared.read(config_path)); output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    start=time.time(); c['absolute_deadline_unix']=min(start+60*c['maximum_minutes'],c.get('absolute_deadline_unix',float('inf')))
    frozen=ROOT/'handoff'/f'{output.name}_CONFIG.json'; shared.write_new(frozen,c)
    args=[sys.executable,'-X','utf8','-u',str(ROOT/'scripts/run_rgb_cost_m_v2.py'),'worker',
          '--config',str(frozen),'--output',str(output)]
    with (ROOT/'handoff'/f'{output.name}_STDOUT.log').open('x',encoding='utf-8') as stdout, \
         (ROOT/'handoff'/f'{output.name}_STDERR.log').open('x',encoding='utf-8') as stderr:
        child=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        shared.write_new(ROOT/'handoff'/f'{output.name}_SUPERVISOR_STARTED.json',dict(pid=os.getpid(),
            child_pid=child.pid,started_unix=start,deadline_unix=c['absolute_deadline_unix'],command=args,root=str(output)))
        stop_seen=None; forced=False
        while child.poll() is None:
            now=time.time()
            if now>=c['absolute_deadline_unix'] or (output/'STOP').exists():
                if stop_seen is None: stop_seen=now
                if now-stop_seen>=15:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=True,
                        timeout=15,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    forced=True; break
            time.sleep(1)
        code=child.wait(timeout=20)
        shared.write_new(ROOT/'handoff'/f'{output.name}_SUPERVISOR_FINAL.json',dict(pid=os.getpid(),child_pid=child.pid,
            exit_code=code,finished_unix=time.time(),forced_child_tree_stop=forced,automatic_resume=False))
    return code


def report(root,output,snapshot):
    from stabilization_learning.unified_v1 import reporting
    original_curves=reporting._curves
    def curves(folder,backend,stages):
        result=original_curves(folder,backend,stages)
        history=result['training']
        if history and backend=='rgb_legacy':
            counts=history[-1].get('actual_calls',{})
            result['actual_counts']=dict(stage_calls_including_rejected_epochs=counts,
                optimizer_updates=dict(ae=counts.get('warmup',0)+counts.get('representation',0),
                    shadow_ae=counts.get('warmup',0)+counts.get('representation',0),
                    derivative_critic=counts.get('critic_prefit',0)+counts.get('representation',0),
                    cumulative_cost_q=counts.get('q',0),
                    M_proposals=counts.get('metric',0),M_applied_before_epoch_acceptance=counts.get('metric_applied',0)),
                retained_model_counts=history[-1].get('counts',{}))
            result['rollback_events']=[s for s in stages if s['stage'].startswith('validate') and
                not s['result'].get('raw_candidate',{}).get('accepted',True)]
        return result
    reporting._curves=curves
    try: result=reporting.report_suite(root,output,snapshot)
    finally: reporting._curves=original_curves
    root,output=Path(root),Path(output)
    protocol=shared.read(root/'protocol.json'); job=root/'jobs'/protocol['config']['jobs'][0]['id']
    records=[shared.read(p) for p in sorted((job/'acceptance').glob('*.json'))]
    epochs=[r for r in records if r['epoch']>0]
    threshold=protocol['config']['jobs'][0]['config'].get('target_successes',38)
    ready=[r for r in records if r['epoch']==0]
    raw_counts=([ready[-1]['raw_successes']] if ready else [])+[r['raw_successes'] for r in epochs]
    observed_no_decrease=all(all(b>=a for a,b in zip(before,after)) for before,after in zip(raw_counts,raw_counts[1:])) if len(raw_counts)>1 else None
    details=dict(schema='rgb_frozen_features_cost_metric_report_v2',raw_candidates=records,raw_success_counts_non_decreasing=observed_no_decrease,
        candidates_meeting_target=[r['epoch'] for r in epochs if min(r['raw_successes'])>=threshold],
        rejected_epochs=[r['epoch'] for r in epochs if not r['accepted']],
        target=dict(successes=threshold,denominator=48,plants=['P0','P1']),
        random_origin=True,old_policy_weights_loaded=False,
        Q_input='observable visual state and actual4D command',Q_target='discounted physical cost with failure-only absorbing mask',
        feature_freeze=shared.read(job/'FEATURES_FROZEN.json') if (job/'FEATURES_FROZEN.json').exists() else None,
        caveat='Accepted validation models are protected by selection; raw candidate monotonicity is measured separately.')
    # The five-second retention audit also uses the complete successor feature.
    retention=job/'RGB_RETENTION_TEST/RESULTS.json'
    if retention.exists():
        import numpy as np
        max_error=0.; endpoints=0
        for row in shared.read(retention)['rows']:
            path=Path(row['trajectory'])
            if shared.sha256(path)!=row['trajectory_sha256']: raise RuntimeError('Retention hash mismatch')
            with np.load(path,allow_pickle=False) as a:
                e=a['error'].astype(float); m=a['M'].astype(float)
                value=.5*np.einsum('bi,ij,bj->b',e,m,e)
                if not np.allclose(value,a['V'],atol=2e-5,rtol=1e-5): raise AssertionError('Retention V mismatch')
                max_error=max(max_error,float(np.max(abs(value-a['V'])))); endpoints+=len(value)
        details['retention_independent_V_audit']=dict(passed=True,endpoints=endpoints,maximum_absolute_error=max_error)
    shared.write_new(output/'STABILITY_ANALYSIS.json',details)
    return dict(report=result,stability=details)
