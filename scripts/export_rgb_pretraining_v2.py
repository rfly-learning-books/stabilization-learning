"""Export reusable RGB pretraining only after every writer has exited.

The source ready checkpoint stays intact. Feature and derivative states can
be reused with a freshly initialized cost Q when changing its objective.
The second package retains the original cost-Q prefit and its optimizer.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stabilization-learning-control'))
from stabilization_learning.unified_v1.reporting import check_process_snapshot
from stabilization_learning.unified_v1.runner import read,sha256,write_new


def packages(payload,provenance):
    learner=payload['learner']
    if payload['schema']!='unified_rgb_legacy_v1' or not learner.get('features_frozen'):
        raise ValueError('Expected frozen visual coordinates from the ready checkpoint')
    if learner['counts']['metric']!=0:
        raise ValueError('A trained or selected M is not a pretraining-only checkpoint')
    common=dict(provenance=deepcopy(provenance),training_counts=deepcopy(learner['counts']))
    visual=dict(schema='rgb_visual_derivative_pretraining_v2',**common)
    for key in ('ae','shadow','critics','anchors','version'):
        visual[key]=deepcopy(learner[key])
    visual['reuse']='Freeze these exact visual coordinates; initialize a new cost Q when changing its learning target.'
    cost=dict(schema='rgb_cost_q_prefit_v2',**common)
    for key in ('q','target_q','qopt','metric','target_metric','mopt'):
        cost[key]=deepcopy(learner[key])
    cost['requires_identical_feature_version']=provenance['feature_version']
    cost['reuse']='Continue only with the same physical-cost definition, Bellman mask and visual feature version.'
    return visual,cost


def save_tensor(path,payload):
    if path.exists(): raise FileExistsError(path)
    temp=path.with_suffix('.pt.tmp')
    if temp.exists(): raise FileExistsError(temp)
    torch.save(payload,temp); temp.rename(path)
    return dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size)


def export(root,output,snapshot):
    gate=check_process_snapshot(snapshot)
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    final=read(root/'FINAL_STATE.json')
    if final['status'] not in ('complete','failed','stopped','timeout','timed_out'):
        raise ValueError('Run has no final state')
    protocol=read(root/'protocol.json'); jobs=protocol['config']['jobs']
    if len(jobs)!=1 or jobs[0]['backend']!='rgb_legacy': raise ValueError('One RGB job required')
    job=root/'jobs'/jobs[0]['id']; checkpoint=job/'checkpoints/ready.pt'
    frozen=read(job/'FEATURES_FROZEN.json')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['config']!=jobs[0]['config']: raise ValueError('Source configuration mismatch')
    provenance=dict(source_root=str(root),source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=sha256(checkpoint),protocol_sha256=sha256(root/'protocol.json'),
        feature_version=frozen['feature_version'],configuration=payload['config'],
        source_inventory=protocol.get('sources',protocol.get('source_files',{})),
        physical_state_used_for_supervision_only=True,pretrained_initialization=True,
        source_final_status=final['status'])
    visual,cost=packages(payload,provenance)
    output.mkdir(parents=True,exist_ok=False)
    files=[save_tensor(output/'visual_derivative_pretraining.pt',visual),
           save_tensor(output/'cost_q_prefit.pt',cost)]
    validations=[]
    for path in sorted((root/'events').glob('*.json')):
        event=read(path); data=event.get('data',{})
        if event['kind']=='candidate_validation' and data.get('epoch')==0:
            validations.append(dict(raw_successes=data['raw_successes'],
                actual_calls=data.get('actual_calls',{}),raw_folder=data['raw_folder']))
    replay=dict(source_root=str(job/'replay'),entries=payload['replay_rows'],
        copies_created=False,training_only=True,
        note='Keep the referenced raw training shards if reproducing prefit; no image replay is duplicated here.')
    write_new(output/'REPLAY_REFERENCE.json',replay)
    manifest=dict(schema='rgb_pretraining_reuse_manifest_v2',process_gate=gate,provenance=provenance,
        files=files,training_counts=payload['learner']['counts'],pretraining_validation=validations,
        stages=dict(visual='AE4000 warmup +3000 representation updates; derivative1000 prefit +3000 joint representation updates',
                    cost_q='2000 physical-action cumulative-cost updates, before any M update'),
        exact_continuation_checkpoint=str(checkpoint),
        exact_continuation_includes='original optimizers, RNG states and replay indices in the full ready.pt',
        pure_AE4000_or_derivative1000_checkpoint_available=False,
        future_experiment_rule='New run and provenance; preserve split and tests; pretrained reuse is not a fully random AE restart.',
        quality='Completed pretraining is not evidence of improved control; retain initial and ready validation scores.')
    write_new(output/'MANIFEST.json',manifest)
    return dict(output=str(output),files=files,training_counts=manifest['training_counts'],
        pretraining_validation=validations)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','process-snapshot'): p.add_argument('--'+name,required=True)
    args=p.parse_args()
    print(json.dumps(export(args.root,args.output,args.process_snapshot),ensure_ascii=False))
