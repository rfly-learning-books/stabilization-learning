"""Independent CPU provenance/update/probe audit after all training writers exit."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import torch

from .common import read_json, write_json, sha256


def digest_state(state):
    digest=hashlib.sha256()
    for name,tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_close(actual,expected,atol=2e-6,rtol=2e-5):
    difference=float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual,expected,atol=atol,rtol=rtol):
        raise AssertionError(difference)
    return difference


def run(root, snapshot):
    root=Path(root).resolve()
    snap=read_json(snapshot)
    if snap.get('query_success') is not True:
        raise RuntimeError('Successful external process enumeration is required')
    import time
    if abs(time.time()-float(snap['created_unix']))>60:
        raise RuntimeError('Process snapshot must be less than 60 seconds old at audit start')
    unrelated=[]
    for process in snap['processes']:
        command=process.get('CommandLine') or process.get('command_line')
        if not command:
            raise RuntimeError('Unknown Python process identity')
        lower=command.lower()
        if 'meetings_mcp_entrypoint.py' in lower:
            unrelated.append(process)
        elif 'training_audit' in lower:
            # A parent snapshot may be made before dispatch, but never ignore other writers.
            unrelated.append(process)
        else:
            raise RuntimeError('Other Python process needs explicit classification: '+command)
    if (root/'GPU_LEASE.json').exists():
        raise RuntimeError('A GPU writer still owns the diagnostic root')
    protocol=read_json(root/'PROTOCOL.json')
    complete=read_json(root/'TRAINING_SUITE_COMPLETE.json')
    group=read_json(root/'TRAINING_GROUP_FROZEN.json')
    plan=read_json(root/'TRAINING_KEYS.json')
    assert complete['status']=='complete' and complete['completed_evaluation_episodes']==768
    assert complete['group_frozen_before_validation'] and complete['no_intermediate_selection']
    assert len(group['arms'])==4 and group['validation_not_started']
    source=Path(protocol['old_root'])
    old_protocol=read_json(source/'protocol.json')
    assert sha256(source/'protocol.json')==protocol['old_protocol_sha256']
    source_inputs=0
    for category in ('sources','assets'):
        for path,expected in old_protocol[category].items():
            assert sha256(path)==expected,path
            source_inputs+=1
    assert sha256(source/protocol['checkpoint'])==protocol['checkpoint_sha256']
    initial=torch.load(source/protocol['checkpoint'],map_location='cpu',weights_only=False)['payload']['models']
    initial_hashes={key:digest_state(initial[key]) for key in ('features','metric')}
    assert initial_hashes==group['initial_hashes']
    assert sha256(root/'TRAINING_KEYS.json')==complete['key_plan_sha256']==group['key_plan_sha256']
    assert sha256(plan['replay_index'])==plan['replay_index_sha256']
    events=[read_json(path) for path in (root/'events').glob('*.json')]
    group_event=[e for e in events if e['stage']=='training_group_frozen']
    evaluation_events=[e for e in events if e['stage']=='repair_validation_started']
    assert len(group_event)==1 and len(evaluation_events)==4
    assert group_event[0]['unix']<min(e['unix'] for e in evaluation_events)
    replay_index=read_json(plan['replay_index'])
    expected_E=[]
    for episode,index in plan['probe_keys']:
        row=replay_index['rows'][episode]
        assert row['split']=='train'
        with np.load(row['files']['labels']['path'],allow_pickle=False) as data:
            q=np.asarray(data['offset'][index],dtype=float)
        expected_E.append(np.square(q[:3]).sum()+(q[3]/math.radians(25))**2)
    expected_E=np.asarray(expected_E)
    results=[]
    for arm in complete['arms']:
        folder=root/'training'/arm['arm']
        marker=read_json(folder/'FROZEN.json')
        arm_protocol=read_json(folder/'ARM_PROTOCOL.json')
        assert not arm_protocol['original_optimizer_states_loaded']
        assert marker['sha256']==arm['sha256']==sha256(arm['checkpoint'])
        assert marker['initial_hashes']==initial_hashes
        payload=torch.load(arm['checkpoint'],map_location='cpu',weights_only=False)['payload']
        final=payload['models']
        assert payload['runtime_state']['counts']==dict(warmup=0,feature=600,metric=600)
        parameter_changes={}
        for module in ('features','metric'):
            assert digest_state(final[module])==marker['final_hashes'][module]
            named=[name for name in initial[module] if name.endswith(('.weight','.bias','.weight_g','.weight_v'))]
            if not named:
                raise RuntimeError('No identified learned weights or biases')
            squared=sum((final[module][name].double()-initial[module][name].double()).square().sum().item()
                        for name in named)
            parameter_changes[module]=math.sqrt(squared)
            assert parameter_changes[module]>0 and math.isfinite(parameter_changes[module])
        for key in ('center','scale','calibrated','coordinate_version'):
            assert torch.equal(initial['features'][key],final['features'][key]),key
        updates=[read_json(path) for path in sorted((folder/'updates').glob('*.json'))]
        assert len(updates)==1200
        assert [row['local_step'] for row in updates]==list(range(1,1201))
        for block in ('feature','metric'):
            selected=[row for row in updates if row['block']==block]
            assert [row['pair'] for row in selected]==list(range(1,601))
            assert all(math.isfinite(row['metrics']['loss'] if 'loss' in row['metrics']
                                     else row['metrics']['total']) for row in selected)
        if arm['arm'].endswith('_alternating'):
            assert [row['block'] for row in updates]==['feature','metric']*600
        else:
            assert [row['block'] for row in updates]==['feature']*600+['metric']*600
        with np.load(folder/'STEP_PROBES.npz',allow_pickle=False) as p:
            step=np.asarray(p['step']);M=np.asarray(p['M'],dtype=float)
            e=np.asarray(p['feature_error'],dtype=float);V=np.asarray(p['V'],dtype=float)
            E=np.asarray(p['E'],dtype=float)
            assert np.array_equal(step,np.arange(1201)) and M.shape==(1201,8,16,16)
            recomputed=.5*np.einsum('tbi,tbij,tbj->tb',e,M,e)
            v_error=check_close(recomputed,V)
            e_error=check_close(E,np.broadcast_to(expected_E,E.shape))
            eigen=np.linalg.eigvalsh(M)
            trace=np.trace(M,axis1=-2,axis2=-1)
            assert eigen.min()>=.0999 and np.max(np.abs(trace-16))<1e-4
            final_lower_violation=float(np.mean(V[-1]<.1*E[-1]))
        results.append(dict(arm=arm['arm'],updates=1200,counts=dict(feature=600,metric=600),
                            named_weight_bias_l2_change=parameter_changes,normalization_unchanged=True,
                            probe_points=1201,probe_count=8,probe_V_recomputation_max_error=v_error,
                            probe_E_recomputation_max_error=e_error,min_M_eigen=float(eigen.min()),
                            max_M_trace_error=float(np.max(np.abs(trace-16))),
                            final_training_probe_lower_violation=final_lower_violation,
                            checkpoint_sha256=arm['sha256']))
        del payload,final,e,M,V,E
    verified_archive=0
    for path,expected in complete['source']['verified_files'].items():
        assert sha256(path)==expected,path
        verified_archive+=1
    result=dict(schema='rgb_v5_training_independent_audit',passed=True,
                external_process_snapshot=str(Path(snapshot).resolve()),process_snapshot_sha256=sha256(snapshot),
                source_protocol_sha256=protocol['old_protocol_sha256'],source_inputs_verified=source_inputs,
                old_archive_files_reverified=verified_archive,initial_parameter_hashes=initial_hashes,
                all_arms_same_initial_weights=True,all_arms_empty_new_optimizer_states_declared=True,
                all_four_frozen_before_first_validation=True,complete_module_update_sequences=True,
                results=results,scope='CPU recomputation of provenance, learned weight changes, update sequences, and fixed training probes; control trace audit is separate.')
    write_json(root/'TRAINING_INDEPENDENT_AUDIT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--process-snapshot',required=True)
    args=parser.parse_args()
    result=run(args.root,args.process_snapshot)
    print(json.dumps(result,ensure_ascii=False))
