"""CPU-only independent final audit of reconstructed visual-servo telemetry.

No training code is imported, no environment rendered, and no original output
is changed. ``passed`` means complete and reproducible recorded measurements,
not improved performance, exact historical RGB, or a global Lyapunov proof.
An incomplete run, bounded development sample, or failed check never passes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT/'third_party/visual_original'
METHODS = ('learned', 'identity_m', 'shadow_ae_identity', 'initial_ae_identity')
TAGS = ('full_val', 'fixed_test', 'depth_scale_0.5', 'depth_scale_2.0')
COUNTS = {'warmup':1000, 'critic_prefit':1000, 'representation':7200, 'metric':4800}
DT = .1
SCALES = np.asarray([.4,.4,.4,.35], dtype=np.float32).astype(float)
# Action/absolute-V/physical bounds are copied from the passed historical
# runs/visual_original_offline_migration_20261022_v1/CONTROL_AUDIT.json.
# Relative, matrix-structure and scalar tolerances are newly declared in this
# audit, before its first 12-episode sample; they are not historical measurements.
LIMITS = {'normalized_action_abs':5e-5, 'V_abs':1e-5, 'V_rel':5e-6,
          'physical_step_abs':1e-6, 'H_abs':2e-6, 'H_rel':5e-6,
          'M_abs':3e-6, 'metric_trace_abs':2e-5, 'scalar_abs':2e-5, 'scalar_rel':5e-6}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''): h.update(part)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def require(value,message):
    if not value: raise ValueError(message)


def within(root,relative):
    path=(root/str(relative).replace('\\','/')).resolve()
    require(path.is_relative_to(root.resolve()),'Path escapes declared root: '+str(relative))
    return path


def close(a,b,atol=None,rtol=None):
    return bool(np.allclose(a,b,atol=LIMITS['scalar_abs'] if atol is None else atol,
                            rtol=LIMITS['scalar_rel'] if rtol is None else rtol))


def maximum(a,b):
    return float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))


def same_numbers(expected,actual,path='root'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and expected.keys()==actual.keys(),'Mapping keys differ: '+path)
        for key in expected: same_numbers(expected[key],actual[key],path+'.'+key)
    elif isinstance(expected,(list,tuple)):
        require(isinstance(actual,(list,tuple)) and len(expected)==len(actual),'Sequence differs: '+path)
        for i,(a,b) in enumerate(zip(expected,actual)): same_numbers(a,b,path+f'[{i}]')
    elif isinstance(expected,(bool,str)) or expected is None:
        require(type(expected) is type(actual) and expected==actual,'Value differs: '+path)
    else:
        require(math.isclose(float(expected),float(actual),abs_tol=1e-9,rel_tol=1e-7),'Number differs: '+path)


def subset(starts,cap):
    groups=defaultdict(list)
    for row in starts: groups[row['scene_id'].split('_')[0],row['difficulty']].append(row)
    return [groups[key][level] for level in range(max(map(len,groups.values())))
            for key in sorted(groups) if level<len(groups[key])][:cap]


def schedules(starts):
    return {'full_val':('val',starts['val'],METHODS,('P0',),1.),
            'fixed_test':('test',starts['test'],METHODS,('P0','P1'),1.),
            'depth_scale_0.5':('test',subset(starts['test'],72),METHODS[:2],('P0',),.5),
            'depth_scale_2.0':('test',subset(starts['test'],72),METHODS[:2],('P0',),2.)}


def episode_identity(run,relative,start):
    paths=[run/relative,(run/relative).with_suffix('.npz'),
           run/'references'/f'{start["scene_id"]}_{start["goal_id"]:02d}.npz']
    digests={p.relative_to(run).as_posix():sha(p) for p in paths}
    return fingerprint(digests),digests


def final_passable(report,max_episodes):
    return bool(max_episodes==0 and report['coverage_complete'] and report['audited_episodes']==3360
                and not report['errors'] and not report['pending'] and report.get('reports',{}).get('passed') is True)


def protocol_audit(run,protocol):
    cfg=protocol['config']; p=cfg['goal_fixed']; deps=protocol['dependencies']
    require(protocol['migration']['protocol_kind']=='reconstructed_environment'
            and protocol['migration']['original_rgb_equivalent'] is False,'Not the reconstructed appearance protocol')
    require(protocol['migration']['smoke'] is False,'Smoke cannot pass formal final gate')
    require(p['rounds']==24 and p['warmup_updates']==1000 and p['representation_updates']==300
            and p['metric_updates']==200 and cfg['derivative']['critic_prefit_updates']==1000,'Training budget changed')
    require(tuple(p['evaluation_methods'])==METHODS and p['evaluation_plants']==['P0','P1']
            and p['stress_depth_scales']==[.5,2.] and p['stress_test_cap']==72
            and cfg['evaluation']['max_steps']==300,'Final evaluation design changed')
    require(cfg['plant']['dt']==DT and cfg['collection']['required_hold_observations']==11
            and p['metric_eigen_floor']==.1 and p['metric_trace']==16
            and p['bounded_metric_eigen_max']==14.5,'Control/hold/metric protocol changed')
    require(protocol['fixed_reference_J'] is True and protocol['fixed_networks_within_episode'] is True,'Episode freeze declaration missing')
    checked={}
    def verify(path,digest):
        require(path.is_file() and sha(path)==digest,'Source/provenance hash mismatch: '+str(path))
        checked[str(path)]=digest
    for relative,digest in protocol['sources'].items():
        verify(within(VENDOR,relative),digest)
        if Path(relative).suffix in ('.py','.yaml'):verify(within(run/'provenance/vendor',relative),digest)
    for relative,digest in deps['migration_sources'].items():
        verify(within(ROOT,relative),digest);verify(within(run/'provenance/migration',relative),digest)
    for item in deps['scene_sources']:verify(Path(item['path']),item['sha256'])
    verify(run/'SCENE_AUDIT.json',deps['scene_audit_sha256'])
    verify(run/'DATA_MANIFEST.json',deps['data_manifest_sha256'])
    scene_audit=read(run/'SCENE_AUDIT.json');data_manifest=read(run/'DATA_MANIFEST.json')
    require(scene_audit['passed'] is True and scene_audit['original_rgb_equivalent'] is False,'Scene audit failed')
    require(data_manifest['complete'] is True and data_manifest['validation_test_RGB_used'] is False,'Data leakage/completion declaration')
    for item in scene_audit['asset_files']:verify(Path(item['path']),item['sha256'])
    data=Path(cfg['derivative']['source_run'])
    for relative,digest in deps['data_files'].items():verify(within(data,relative),digest)
    for name in ('scenes.json','evaluation_starts.json','training_anchors.json'):
        verify(run/name,deps['data_files'][name])
    for name,digest in protocol['initialization'].items():verify(ROOT/'assets/original_visual_initialization'/name,digest)
    import yaml
    require(yaml.safe_load((run/'resolved_config.yaml').read_text(encoding='utf-8'))==cfg,'Resolved config differs from protocol')
    for name in ('engineering_audit.json','migration_initialization_audit.json'):
        require(read(run/name)['passed'] is True,'Required engineering audit failed: '+name)
    scenes=read(run/'scenes.json')['scenes']; by_scene={s['id']:s for s in scenes}
    require(len(by_scene)==44 and Counter(s['split'] for s in scenes)==Counter(train=24,val=8,test=12),'Scene splits differ')
    starts=read(run/'evaluation_starts.json')
    require(len(starts['val'])==48 and len(starts['test'])==360,'Frozen starting-state counts differ')
    for split in ('val','test'):
        keys=[]
        for row in starts[split]:
            require(by_scene[row['scene_id']]['split']==split,'Start crosses split')
            keys.append((row['scene_id'],row['goal_id'],row['difficulty'],row['repeat']))
        require(len(keys)==len(set(keys)),'Duplicate frozen starting point identity')
    for row in read(run/'replay/index.json'):
        require(row['split']=='train' and by_scene[row['scene_id']]['split']=='train','Held-out sample entered replay')
    training=read(run/'training_complete.json')
    require(training['counts']==COUNTS and training['rounds']==24,'Training did not finish the declared budget')
    validations={row['round']:row for row in map(json.loads,(run/'validation.jsonl').read_text(encoding='utf-8').splitlines())}
    require(set(validations)==set(range(2,25,2)),'Validation schedule incomplete')
    selected=max(sorted(validations),key=lambda number:validations[number]['score'])
    require(training['best_round']==selected,'Selected best is not validation optimum')
    for number,row in validations.items():
        summary=read(run/f'evaluation/round_{number:02d}/summary.json')
        require(summary['complete'] is True and summary['episodes']==48 and summary['split']=='val','Training validation incomplete')
        v=summary['summaries']['learned/P0']
        same_numbers([v['macro_scene_success'],v['worst_scene_success'],-v['mean_integrated_error']],row['score'])
    # Check serialization identities on CPU without building neural modules or
    # calling CUDA. The last checkpoint retains final training, best is separate.
    import torch
    torch.set_num_threads(1)
    checkpoint_rows={}
    for name in ('initial.pt','last.pt','best.pt'):
        digest=sha(run/name)
        payload=torch.load(run/name,map_location='cpu',weights_only=False)
        require(payload['protocol_hash']==fingerprint(protocol),'Checkpoint protocol mismatch: '+name)
        counts=payload['learner']['counts']
        expected=({k:0 for k in COUNTS} if name=='initial.pt' else COUNTS if name=='last.pt' else
                  dict(warmup=1000,critic_prefit=1000,representation=selected*300,metric=selected*200))
        require(counts==expected,'Checkpoint counters differ: '+name)
        if name=='best.pt':require(payload['state']['best_round']==selected,'Best checkpoint round differs')
        checkpoint_rows[name]={'sha256':digest,'counts':counts,'version':payload['learner']['version']}
        del payload
        require(sha(run/name)==digest,'Checkpoint changed while auditing: '+name)
    return {'passed':True,'verified_source_data_files':len(checked),'source_identity':fingerprint(checked),
            'best_round':selected,'checkpoints':checkpoint_rows,'training_counts':COUNTS},by_scene,starts


def control(features,M,J):
    b=J.T@M@J
    h=b+.01*np.diag(np.diag(b))+max(float(np.trace(b))*1.01/(1e4-1),1e-8)*np.eye(4)
    gain=.7*np.linalg.solve(h,J.T@M)
    u=-features@gain.T
    return u/np.maximum(1.,np.maximum(np.linalg.norm(u[:,:3],axis=1),np.abs(u[:,3])))[:,None],h


def physical_rollout(start,actions,plant,recorded):
    pose=np.asarray(start,dtype=float).copy();actual=np.zeros(4);previous=np.zeros(4)
    results=[];locals_=[];mix=1-math.exp(-DT/.15)
    for i,command in enumerate(actions):
        if plant=='P1':actual=(1-mix)*actual+mix*SCALES*previous
        else:actual=SCALES*command
        def step(state):
            yaw=state[3]+.5*actual[3]*DT;c,s=math.cos(yaw),math.sin(yaw)
            return state+DT*np.array([c*actual[0]-s*actual[1],s*actual[0]+c*actual[1],actual[2],actual[3]])
        locals_.append(step(recorded[i]));pose=step(pose);results.append(pose.copy());previous=command
    results,locals_=np.asarray(results),np.asarray(locals_)
    for values in (results,locals_):values[:,3]=(values[:,3]+math.pi)%(2*math.pi)-math.pi
    return results,locals_


def geometry_flags(offset,scene,goal):
    matrix=np.asarray(scene['refs'][goal]['goal_matrix'],dtype=float)
    forward=matrix[:3,2].copy();forward[2]=0;forward/=np.linalg.norm(forward)
    body=np.column_stack((forward,np.cross([0.,0.,1.],forward),[0.,0.,1.]))
    world=matrix[:3,3]+offset[:,:3]@body.T
    collision=np.zeros(len(world),bool)
    for ob in scene['obstacles']:
        delta=np.maximum(np.abs(world-np.asarray(ob['center']))-ob['halfsize'],0.)
        collision|=np.sum(delta*delta,axis=1)<=.15**2
    d=scene['dims'];x,y,z=world.T
    outside=~((x>-d['hw']+.3)&(x<d['hw']-.3)&(y>-d['hl']+.3)&(y<d['hl']-.3)&(z>.30)&(z<d['height']-.4))
    return collision,outside


def critic_audit(a,row):
    expected=row['plant']=='P0' and row['method'] in METHODS[:2]
    require(('derivative_critic' in row)==expected,'P0 derivative diagnostic scope differs')
    if not expected:
        require(not any(k.startswith('critic_') for k in a),'P1/shadow diagnostic incorrectly labeled')
        return
    indices=np.linspace(0,len(a['features'])-1,min(16,len(a['features']))).astype(int)
    require(np.array_equal(indices,a['critic_indices']),'Derivative diagnostic states differ')
    x,y,z,yaw=a['offset'][indices].astype(float).T;c,s=np.cos(yaw),np.sin(yaw)
    true=np.stack([2*(c*x+s*y),2*(-s*x+c*y),2*z,2*yaw/math.radians(25)**2],axis=1)
    require(close(true,a['critic_true_coeff'],atol=2e-5,rtol=2e-6),'P0 derivative labels disagree with physical pose')
    pred=np.asarray(a['critic_pred_coeff'],dtype=float)*SCALES
    target=np.asarray(a['critic_true_coeff'],dtype=float)*SCALES
    active=np.abs(target)>1e-4;mask=np.linalg.norm(target,axis=1)>1e-5
    cosine=np.sum(pred*target,axis=1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(target,axis=1)+1e-8)
    recomputed={'states':len(indices),'rmse':float(np.sqrt(np.mean((pred-target)**2))),
        'sign_accuracy':float(np.mean((pred*target)[active]>0)) if active.any() else 1.,
        'action_gradient_cosine':float(cosine[mask].mean()) if mask.any() else 1.}
    require(all(close(value,row['derivative_critic'][key],atol=2e-6,rtol=2e-6) for key,value in recomputed.items()),'Derivative statistics mismatch')


def episode_audit(path,scene,start,identity,cfg):
    row=read(path);tag,method,plant,index,split,depth_scale=identity
    require(all(row.get(k)==v for k,v in {'tag':tag,'method':method,'plant':plant,'index':index,'split':split,
        'scene_id':start['scene_id'],'goal_id':start['goal_id'],'difficulty':start['difficulty']}.items()),'Episode identity/start split differs')
    require(row['offset_start']==start['offset'] and row['depth_scale']==depth_scale,'Frozen start or depth scale differs')
    require(row['networks_frozen_in_episode'] is True and row['checkpoint']=='best.pt','Episode freeze/checkpoint declaration differs')
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:a={k:data[k] for k in data.files}
    n=row['steps'];require(isinstance(n,int) and 1<=n<=300,'Invalid episode length')
    shapes={'offset':(n+1,4),'features':(n+1,16),'V':(n+1,),'action':(n,4),'cost':(n,),
            'terminal':(n,),'truncated':(n,),'previous_action':(n,4),'M':(16,16),'L':(16,16),'J_reference':(16,4),'H':(4,4)}
    for key,shape in shapes.items():require(a[key].shape==shape and np.isfinite(a[key]).all(),'Invalid/nonfinite telemetry '+key)
    require(maximum(a['offset'][0],start['offset'])<=1e-6,'Initial state differs')
    require(np.array_equal(a['previous_action'][0],np.zeros(4)) and np.array_equal(a['previous_action'][1:],a['action'][:-1]),'Previous action trace differs')
    M=a['M'].astype(float);L=a['L'].astype(float);J=a['J_reference'].astype(float);e=a['features'].astype(float)
    eig=np.linalg.eigvalsh(M)
    require(close(M,M.T,atol=1e-7,rtol=0.) and eig[0]>=.1-3e-6 and eig[-1]<=14.5+3e-6
            and abs(np.trace(M)-16)<=LIMITS['metric_trace_abs'],'M SPD/trace bounds')
    require(np.max(np.abs(np.triu(L,1)))==0 and np.all(np.diag(L)>0),'Invalid Cholesky factor')
    if method=='learned':
        gram=L@L.T;expected_M=.1*np.eye(16)+14.4*gram/np.trace(gram)
        require(close(M,expected_M,atol=LIMITS['M_abs'],rtol=1e-6),'M does not match normalized Cholesky factor')
    else:require(np.array_equal(a['M'],np.eye(16,dtype=np.float32)),'Identity ablation has nonidentity M')
    command,h=control(e[:-1],M,J);V=.5*np.einsum('ti,ij,tj->t',e,M,e)
    require(maximum(command,a['action'])<=LIMITS['normalized_action_abs'],'M/J/features do not reproduce commands')
    require(close(h,a['H'],atol=LIMITS['H_abs'],rtol=LIMITS['H_rel']),'Saved damped matrix differs')
    require(close(V,a['V'],atol=LIMITS['V_abs'],rtol=LIMITS['V_rel']),'Certificate values do not match 0.5 e^T M e')
    u=a['action'].astype(float);offset=a['offset'].astype(float)
    require(np.max(np.linalg.norm(u[:,:3],axis=1))<=1.00001 and np.max(np.abs(u[:,3]))<=1.00001,'Velocity limits violated')
    replay,one=physical_rollout(start['offset'],u,plant,offset)
    require(maximum(replay,offset[1:])<=LIMITS['physical_step_abs'] and maximum(one,offset[1:])<=LIMITS['physical_step_abs'],'Physical step replay differs')
    collision,outside=geometry_flags(offset[1:],scene,start['goal_id'])
    good=(np.linalg.norm(offset[1:,:3],axis=1)<.02)&(np.abs(offset[1:,3])<math.radians(1))
    held=np.zeros(n,int);count=0
    for i,value in enumerate(good):count=count+1 if value else 0;held[i]=count
    terminal=collision|outside|(held>=11)
    require(not terminal[:-1].any(),'Trajectory continued after terminal event')
    status='collision' if collision[-1] else 'outside' if outside[-1] else 'success' if held[-1]>=11 else 'timeout'
    require(row['status']==status and row['success']==(status=='success'),'Success/failure/status differs from physical trajectory')
    require(np.array_equal(terminal,a['terminal']),'Terminal flags differ')
    truncated=np.zeros(n,bool);truncated[-1]=n==300 and not terminal[-1]
    require(np.array_equal(truncated,a['truncated']) and (terminal[-1] or n==300),'Time horizon or truncation flags differ')
    E=np.sum(offset[:,:3]**2,axis=1)+(offset[:,3]/math.radians(25))**2
    cost=.1*(E[1:]+.02*np.sum(u*u,axis=1))+20*(collision|outside)
    require(close(cost,a['cost']),'Recorded physical cost differs')
    # Match published float32 threshold arithmetic, separately verified against
    # the independent float64 physical energy and quadratic certificate above.
    off=a['offset'];E32=np.sum(off[:,:3]**2,axis=1)+(off[:,3]/math.radians(25))**2
    dv=np.diff(a['V']);tol=1e-7+1e-4*a['V'][:-1];p=cfg['goal_fixed']
    measured={'final_position_m':float(np.linalg.norm(offset[-1,:3])),
        'final_yaw_deg':float(abs(np.degrees(offset[-1,3]))),'integrated_error':float(.1*E32[1:].sum()),
        'action_variation':float(np.square(np.diff(a['action'],axis=0)).sum()),
        'actual_V_increase_fraction':float(np.mean(dv>tol)),
        'actual_decay_violation_fraction':float(np.mean(dv+p['decay_per_second']*.1*E32[:-1]>tol)),
        'certificate_comparable_fraction':float(np.mean((a['V']>=p['value_lower']*E32-tol.mean())&(a['V']<=p['value_upper']*E32+tol.mean()))),
        'V_over_E_quantiles':np.quantile(a['V']/(E32+p['objective_floor']),[.05,.5,.95]).tolist(),
        'M_eigen_min':float(eig[0]),'M_eigen_max':float(eig[-1]),'M_trace':float(np.trace(M))}
    for key,value in measured.items():require(close(value,row[key]),'Episode statistic differs: '+key)
    critic_audit(a,row)
    reference=path.parent.parent.parent/'references'/f'{scene["id"]}_{start["goal_id"]:02d}.npz'
    with np.load(reference,allow_pickle=False) as ref:
        require(np.array_equal(a['goal_depth'],ref['depth']),'Episode depth differs from frozen goal reference')
        require(np.array_equal(a['collection_depth'],ref['depth']),'Evaluation inherited a training depth perturbation')
        require(close(a['rotation'],ref['camera_from_body'],atol=1e-7,rtol=0.),'Reference rotation differs')
    raw_rises=int(np.sum(np.diff(a['V'].astype(float))>1e-10))
    arrival=n*DT if row['success'] else 30.
    return {'passed':True,'identity':list(identity),'row':row,'initial_offset':offset[0].tolist(),
        'reference_fingerprints':{k:hashlib.sha256(np.ascontiguousarray(a[k]).tobytes()).hexdigest() for k in ('M','J_reference','goal_depth','rotation')},
        'max_errors':{'normalized_action_abs':maximum(command,u),'V_abs':maximum(V,a['V']),
            'H_abs':maximum(h,a['H']),'physical_rollout_abs':maximum(replay,offset[1:]),
            'physical_one_step_abs':maximum(one,offset[1:]),'cost_abs':maximum(cost,a['cost'])},
        'recomputed':{'success':row['success'],'status':status,'arrival_time':arrival,'steps':n,
            'V_raw_rises':raw_rises,'monotonic_success':bool(row['success'] and raw_rises==0),
            'hold_observations':int(held[-1]),'integrated_error_float64':float(.1*E[1:].sum()),
            'actual_decay_violation_fraction':measured['actual_decay_violation_fraction']}}


def aggregate(rows):
    rates={s:float(np.mean([r['success'] for r in rows if r['scene_id']==s])) for s in sorted({r['scene_id'] for r in rows})}
    out={'episodes':len(rows),'success_rate':float(np.mean([r['success'] for r in rows])),
         'macro_scene_success':float(np.mean(list(rates.values()))),'worst_scene_success':min(rates.values()),
         'scene_success':rates,'complete':True}
    for key in ('final_position_m','final_yaw_deg','integrated_error','action_variation','actual_V_increase_fraction',
                'actual_decay_violation_fraction','certificate_comparable_fraction','latency_ms'):
        out['mean_'+key]=float(np.mean([r[key] for r in rows]))
    out['difficulty_success']={d:float(np.mean([r['success'] for r in rows if r['difficulty']==d])) for d in ('small','medium','large') if any(r['difficulty']==d for r in rows)}
    out['status_counts']=dict(sorted(Counter(r['status'] for r in rows).items()));out['plant']=rows[0]['plant']
    diags=[r['derivative_critic'] for r in rows if 'derivative_critic' in r]
    if diags:out['derivative_critic']={k:float(np.mean([d[k] for d in diags])) for k in ('rmse','sign_accuracy','action_gradient_cosine')}
    return out


def summaries_and_pairs(run,records,plan):
    computed={};pairs=[];cert={};report_text=(run/'report/REPORT.html').read_text(encoding='utf-8')
    require('完整测试已完成' in report_text,'HTML report still labels evaluation incomplete')
    for tag,(split,starts,methods,plants,scale) in plan.items():
        rows=[r['row'] for r in records if r['row']['tag']==tag]
        summary=read(run/f'evaluation/{tag}/summary.json')
        group={f'{method}/{plant}':aggregate([r for r in rows if r['method']==method and r['plant']==plant])
               for method in methods for plant in plants}
        expected={'tag':tag,'split':split,'complete':True,'episodes':len(rows),'planned':len(starts)*len(methods)*len(plants),'summaries':group}
        same_numbers(expected,summary,'summary.'+tag);computed[tag]=summary
        for key,v in group.items():
            needle=f'<tr><td>{tag}</td><td>{key}</td><td>{v["episodes"]}</td><td>{100*v["success_rate"]:.1f}%</td><td>{v["mean_integrated_error"]:.4g}</td><td>{100*v["mean_actual_V_increase_fraction"]:.1f}%</td>'
            require(needle in report_text,'HTML report statistic differs: '+tag+'/'+key)
    for link in re.findall(r'(?:src|href)="([^"]+)"',report_text):
        if '://' not in link:require(within(run,Path('report')/link).exists(),'Missing report link: '+link)
    fixed={ (r['row']['method'],r['row']['plant'],r['row']['index']):r for r in records if r['row']['tag']=='fixed_test'}
    original_pairs=[]
    for plant in ('P0','P1'):
        for base in METHODS[1:]:
            ab=[(fixed['learned',plant,i],fixed[base,plant,i]) for i in range(360)]
            basic={'plant':plant,'baseline':base,'episodes':360,
                'learned_only':sum(a['row']['success'] and not b['row']['success'] for a,b in ab),
                'baseline_only':sum(b['row']['success'] and not a['row']['success'] for a,b in ab)}
            original_pairs.append(basic)
            jointly=[(a,b) for a,b in ab if a['row']['success'] and b['row']['success']]
            pairs.append(basic|{'joint_successes':len(jointly),'joint_success_arrival_delta':float(np.mean([a['recomputed']['arrival_time']-b['recomputed']['arrival_time'] for a,b in jointly])) if jointly else None,
                'all_start_30s_capped_arrival_delta':float(np.mean([a['recomputed']['arrival_time']-b['recomputed']['arrival_time'] for a,b in ab])),
                'paired_integrated_error_delta':float(np.mean([a['row']['integrated_error']-b['row']['integrated_error'] for a,b in ab]))})
    for method in METHODS:
        for plant in ('P0','P1'):
            part=[fixed[method,plant,i] for i in range(360)];n=sum(r['recomputed']['steps'] for r in part)
            rises=sum(r['recomputed']['V_raw_rises'] for r in part)
            cert[f'{method}/{plant}']={'transitions':n,'V_rises':rises,
                'monotonic_successes':sum(r['recomputed']['monotonic_success'] for r in part),
                'transition_weighted_V_rise_fraction':rises/n}
    analysis=read(run/'analysis/RESULTS.json')
    same_numbers(computed['fixed_test']['summaries'],analysis['test'],'analysis.test')
    same_numbers(cert,analysis['certificate'],'analysis.certificate');same_numbers(original_pairs,analysis['paired'],'analysis.paired')
    require(analysis['audit']['passed'] is True and not analysis['audit']['errors'],'Original final analysis failed')
    complete=read(run/'complete.json')
    same_numbers(computed['fixed_test'],complete['test'],'complete.test');same_numbers(computed['full_val'],complete['validation'],'complete.validation')
    require(complete['protocol_sha256']==sha(run/'protocol.json') and complete['best_sha256']==sha(run/'best.pt'),'Completion model/protocol hashes differ')
    require(complete['training_counts']==COUNTS,'Completion counts differ')
    return {'passed':True,'groups':computed,'paired':pairs,'certificate':cert},analysis


def decode(path,expected):
    import cv2
    cv2.setNumThreads(1)
    cap=cv2.VideoCapture(str(path));require(cap.isOpened(),'Video failed to open: '+str(path))
    n=0;shape=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            require(frame is not None and frame.ndim==3,'Invalid decoded frame')
            if shape is None:shape=list(frame.shape)
            require(list(frame.shape)==shape,'Video frame shape changed');n+=1
    finally:cap.release()
    require(n==expected,'Video frame count differs: '+str(path))
    return {'passed':True,'frames':n,'shape':shape,'sha256':sha(path)}


def run_audit(args):
    run=args.run.resolve();require(run.is_relative_to(ROOT/'runs'),'Run must be inside project runs')
    require(math.isfinite(args.max_minutes) and args.max_minutes>0 and args.max_episodes>=0,'Invalid audit budget')
    started=time.monotonic();deadline=started+args.max_minutes*60
    script_hash=sha(__file__);protocol_hash=sha(run/'protocol.json');protocol=read(run/'protocol.json')
    report={'schema':1,'passed':False,'meaning_of_passed':'complete_coverage_and_independently_reproduced_recorded_measurements_not_performance_or_global_stability',
        'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':script_hash,'protocol_sha256':protocol_hash,
        'device':'cpu','expected_episodes':3360,'audited_episodes':0,'coverage_complete':False,'errors':[],'pending':[],
        'limits':LIMITS,'modified_training_or_result_files':False,
        'limitations':['Reconstructed appearance is not historical RGB reproduction.',
            'Current RGB is omitted from saved NPZ: feature encoding, image JVP and FastDepth inference are not independently rerun here.',
            'This audit checks saved goal depth consistency, fixed-reference telemetry and the actual M/J/e-to-command chain.',
            'P1 replay preserves hidden lag and previous command; P0 derivative labels are not asserted as P1 truth.',
            'Finite sampled trajectories and SPD M do not prove global stability; failures and V rises are reported.',
            'Single seed and shared-policy collection limit causal and generalization conclusions.']}
    prior=ROOT/'runs/visual_original_offline_migration_20261022_v1/CONTROL_AUDIT.json'
    report['limits_provenance']={'historical_evidence':str(prior),'sha256':sha(prior),
        'copied_bounds':['normalized_action_abs=5e-5','V_abs=1e-5','physical_step_abs=1e-6'],
        'newly_declared_before_first_sample':['V_rel','H_abs/H_rel','M_abs','metric_trace_abs','scalar_abs/scalar_rel'],
        'tuned_using_new_results':False}
    if args.max_episodes:report['pending'].append('Development max-episodes sampling cannot open the final gate')
    cache_dir=run/'independent_final_audit_cache';cache_dir.mkdir(exist_ok=True)
    cache_key=fingerprint({'script':script_hash,'protocol':protocol_hash})
    cache_path=cache_dir/'records.json';cache=read(cache_path) if cache_path.exists() else {}
    if cache.get('identity')!=cache_key:cache={'identity':cache_key,'episodes':{},'videos':{}}
    records=[];video_results={};artifact_hashes={};time_ended=False
    try:
        report['protocol'],scenes,starts=protocol_audit(run,protocol)
        plan=schedules(starts)
        expected_paths={}
        for tag,(split,points,methods,plants,scale) in plan.items():
            for method in methods:
                for plant in plants:
                    for i,start in enumerate(points):expected_paths[f'evaluation/{tag}/{method}_{plant}_{i:04d}.json']=((tag,method,plant,i,split,scale),start)
        require(len(expected_paths)==3360,'Audit schedule must cover exactly 3360 episodes')
        present={p.relative_to(run).as_posix() for tag in TAGS for p in (run/'evaluation'/tag).glob('*.json') if p.name!='summary.json'}
        require(not present-set(expected_paths),'Unexpected extra final evaluation episode files')
        missing=sorted(set(expected_paths)-present);report['missing_episode_count']=len(missing);report['missing_episode_examples']=missing[:10]
        for relative,(identity,start) in expected_paths.items():
            path=run/relative
            if not path.exists():continue
            if time.monotonic()>=deadline or (args.max_episodes and len(records)>=args.max_episodes):time_ended=True;break
            npz=path.with_suffix('.npz')
            if not npz.exists():report['pending'].append('Uncommitted/missing telemetry: '+relative);continue
            key,input_hashes=episode_identity(run,relative,start)
            saved=cache['episodes'].get(relative)
            try:
                if saved and saved['identity']==key:result=saved['result']
                else:
                    result=episode_audit(path,scenes[start['scene_id']],start,identity,protocol['config'])
                    require(episode_identity(run,relative,start)[0]==key,'Episode/reference changed during audit')
                    cache['episodes'][relative]={'identity':key,'result':result}
                require(result['row']['encoder_version']==report['protocol']['checkpoints']['best.pt']['version'],'Mixed encoder versions in final evaluation')
                records.append(result);artifact_hashes.update(input_hashes)
                if result['row'].get('video'):
                    video=within(run,result['row']['video']);vrel=video.relative_to(run).as_posix();vdigest=sha(video)
                    item=cache['videos'].get(vrel)
                    if not item or item['sha256']!=vdigest:item=decode(video,result['row']['steps']+1);cache['videos'][vrel]=item
                    require(item['frames']==result['row']['steps']+1,'Cached video length changed')
                    video_results[vrel]=item;artifact_hashes[vrel]=vdigest
            except Exception as error:report['errors'].append(relative+': '+str(error))
            if len(records)%100==0:write(cache_path,cache)
        report['audited_episodes']=len(records);report['coverage_complete']=len(records)==3360 and not missing
        report['time_or_sample_budget_ended']=time_ended
        if not report['coverage_complete']:report['pending'].append('All 3360 final episodes have not been audited')
        # Compare the same goal under all starting points and all plants. J/M
        # may differ by representation or depth stress, but never by start.
        references={};starts_seen={}
        for result in records:
            r=result['row'];key=(r['tag'],r['method'],r['scene_id'],r['goal_id'])
            require(key not in references or references[key]==result['reference_fingerprints'],'Reference M/J changed with start or plant')
            references[key]=result['reference_fingerprints']
            key=(r['tag'],r['index'])
            require(key not in starts_seen or starts_seen[key]==result['initial_offset'],'Methods/plants did not use identical starts')
            starts_seen[key]=result['initial_offset']
        report['paired_start_and_fixed_reference_checks']={'passed':True,'starting_points_observed':len(starts_seen),'reference_groups_observed':len(references)}
        report['max_errors']={key:max((r['max_errors'][key] for r in records),default=0.) for key in ('normalized_action_abs','V_abs','H_abs','physical_rollout_abs','physical_one_step_abs','cost_abs')}
        report['videos']={'decoded_videos':len(video_results),'decoded_frames':sum(x['frames'] for x in video_results.values()),'records':video_results}
        required=('complete.json','analysis/RESULTS.json','report/REPORT.html')
        for name in required:
            if not (run/name).exists():report['pending'].append('Final artifact absent: '+name)
        if report['coverage_complete'] and not report['pending'] and not report['errors']:
            report['reports'],analysis=summaries_and_pairs(run,records,plan)
            videos_on_disk={p.relative_to(run).as_posix() for tag in TAGS for p in (run/'evaluation'/tag).glob('*.mp4')}
            require(videos_on_disk==set(video_results),'Unreferenced or undecoded final videos')
            require(analysis['audit']['decoded_videos']==len(video_results) and analysis['audit']['decoded_frames']==report['videos']['decoded_frames'],'Original video audit counts differ')
            for name in (*required,'training_complete.json','initial.pt','last.pt','best.pt','scenes.json','evaluation_starts.json','training_anchors.json','protocol.json','SCENE_AUDIT.json','DATA_MANIFEST.json'):
                artifact_hashes[name]=sha(run/name)
            for tag in TAGS:artifact_hashes[f'evaluation/{tag}/summary.json']=sha(run/f'evaluation/{tag}/summary.json')
            require(sha(run/'protocol.json')==protocol_hash,'Protocol changed during audit')
            report['passed']=final_passable(report,args.max_episodes)
        report['artifact_sha256']=artifact_hashes
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error));report['traceback']=traceback.format_exc()
    finally:
        write(cache_path,cache);report['elapsed_seconds']=time.monotonic()-started
        write(run/'INDEPENDENT_FINAL_AUDIT.json',report)
    print(json.dumps({k:report[k] for k in ('passed','audited_episodes','expected_episodes','coverage_complete','errors','pending')},ensure_ascii=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--max-minutes',type=float,default=120.)
    parser.add_argument('--max-episodes',type=int,default=0,help='Development read-only sample; nonzero bounded coverage never passes')
    args=parser.parse_args()
    # A separate audit lock never acquires, changes or impersonates training locks.
    import msvcrt
    lock=args.run.resolve()/'independent_final_audit.lock'
    with lock.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:result=run_audit(args)
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
    if result['errors']:raise SystemExit(2)


if __name__=='__main__':main()
