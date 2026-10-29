"""Post-run provenance and NumPy full-successor audit; no model reselection."""
from pathlib import Path
import numpy as np
import torch
from .common import read,write,sha,tensor_hash


def actor_hash(state):
    from .push import FullMActor,FullMConfig
    model=FullMActor(FullMConfig(**state['cfg']));model.load_state_dict(state['actor'])
    return tensor_hash(model)


def audit(run):
    checks={};details={};maximum=0.;count=0;initial_hashes=[]
    # Frozen training metadata and optimizer evidence precede every test.
    events=[read(p) for p in sorted((run.root/'events').glob('*.json'))]
    for group in ('push','so101','rgb'):
        frozen=[e['unix'] for e in events if e['kind']==f'{group}_frozen']
        tests=[e['unix'] for e in events if e['kind']=='child_launch' and e['stage']==f'{group}-test']
        checks[f'{group}_all_models_frozen_before_test']=len(frozen)==len(run.config[group]['seeds']) and bool(tests) and max(frozen)<min(tests)
    for seed in run.config['push']['seeds']:
        p=run.root/f'push_seed{seed}';initial=torch.load(p/'initial.pt',map_location='cpu',weights_only=True)
        final=torch.load(p/'final.pt',map_location='cpu',weights_only=True)
        frozen=read(p/'FROZEN.json');ih=actor_hash(initial);initial_hashes.append(ih)
        checks[f'push_{seed}_random_empty_optimizer']=initial['origin']=='random' and initial['next_update']==0 and not initial['optimizer']['state']
        checks[f'push_{seed}_parameters_updated']=ih!=actor_hash(final) and final['next_update']==run.config['push']['updates']
        checks[f'push_{seed}_selected_after_update']=frozen['selected_update']>0 and sha(p/'selected.pt')==frozen['sha256']
        probe=read(p/'M_PROBES.json')
        checks[f'push_{seed}_every_M_step_present']=[r['update'] for r in probe]==list(range(run.config['push']['updates']+1))
        for file in (p/'test').glob('*/trajectory_*.npz'):
            with np.load(file,allow_pickle=False) as a:
                active=a['active'];V=[]
                for suffix in ('','_next'):
                    e=a['e'+suffix].astype(float);D=e.copy();D[...,2]=np.sqrt(2)*np.sin(e[...,2]/2)
                    m=a['M'+suffix].astype(float)
                    vv=np.einsum('...i,...ij,...j->...',D,m,D)
                    err=float(np.max(abs(vv-a['V'+suffix])[active]));maximum=max(maximum,err)
                    if not np.allclose(vv[active],a['V'+suffix][active],atol=3e-5,rtol=3e-5):
                        raise RuntimeError('Independent full-M value mismatch: '+str(file))
                    if np.linalg.eigvalsh(m[active]).min()<.05-1e-5:
                        raise RuntimeError('Metric SPD floor violated: '+str(file))
                    V.append(vv)
                count+=int(active.sum())
        so=run.root/f'so101_seed{seed}';start=torch.load(so/'initial.pt',map_location='cpu',weights_only=True)
        selected=torch.load(p/'selected.pt',map_location='cpu',weights_only=True)
        end=torch.load(so/'final.pt',map_location='cpu',weights_only=True)
        checks[f'so101_{seed}_exact_new_push_parent']=actor_hash(start)==actor_hash(selected) and start['initial_checkpoint_sha256']==sha(p/'selected.pt')
        checks[f'so101_{seed}_time_parameters_updated']=actor_hash(start)!=actor_hash(end)
        arm=read(so/'test/RESULTS.json')['arm']
        for row in arm:
            # Recompute time integral independently, including repositions and holds.
            folder=so/'test/arm'/row['variant']/row['shape']
            matches=[f for f in folder.glob('*.npz') if sha(f)==row['trace_sha256']]
            if len(matches)!=1:raise RuntimeError('SO101 telemetry hash mismatch')
            with np.load(matches[0]) as a:
                poses=a['pose'].astype(float);t=a['time'];err=(poses[:,:2]**2).sum(1)/.05**2+2*np.sin(poses[:,2]/2)**2
                integral=float(np.sum(.5*(err[:-1]+err[1:])*np.diff(t)))
                if not np.isclose(integral,row['integrated_error'],rtol=1e-9,atol=1e-9):
                    raise RuntimeError('SO101 measured error integral mismatch')
                count+=max(0,len(t)-1)
        details[f'so101_{seed}_failure_reasons']={reason:sum(r['reason']==reason for r in arm) for reason in sorted({r['reason'] for r in arm})}
    checks['independent_push_initializations']=len(set(initial_hashes))==len(initial_hashes)
    for seed in run.config['rgb']['seeds']:
        out=run.root/f'rgb_seed{seed}';frozen=read(out/'FROZEN.json')
        raw=torch.load(out/'random_raw.pt',map_location='cpu',weights_only=False)
        checks[f'rgb_{seed}_empty_initial_replay']=not raw['replay_rows'] and raw['transitions_sampled']==0
        checks[f'rgb_{seed}_no_pretrained_AE']=raw['config']['paths']['initial_autoencoder'] is None and raw['origin']=='random'
        checks[f'rgb_{seed}_every_module_updated']=all(frozen['raw_initial_hashes'][k]!=frozen['final_hashes'][k] for k in ('ae','metric','critics'))
        for file in (out/'evaluation/heldout_test').glob('*.npz'):
            with np.load(file) as a:
                d=a['features'].astype(float);m=a['M'].astype(float)
                values=.5*np.einsum('ti,ij,tj->t',d,m,d);delta=float(np.max(abs(values-a['V'])))
                maximum=max(maximum,delta);count+=len(a['action'])
                if len(values)!=len(a['action'])+1 or not np.allclose(values,a['V'],atol=3e-5,rtol=3e-5):
                    raise RuntimeError('Independent RGB complete successor audit failed')
    result=dict(passed=all(checks.values()),checks=checks,details=details,
        audited_transitions=count,numpy_V_max_absolute_difference=maximum,
        scope='engineering provenance and empirical trajectory arithmetic; no performance or stability theorem')
    write(run.root/'AUDIT.json',result)
    if not result['passed']:raise RuntimeError('Post-run audit failed: '+str([k for k,v in checks.items() if not v]))
    return result
