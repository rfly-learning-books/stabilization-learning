"""Independent post-writer repeatability and rollback check on engineering runs."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stabilization-learning-control'))
from stabilization_learning.unified_v1.reporting import check_process_snapshot


def equal(a,b,path='root'):
    if torch.is_tensor(a):
        assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a: equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,path+'/'+str(i))
    else: assert a==b,(path,a,b)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main(args):
    gate=check_process_snapshot(args.process_snapshot)
    first,second=Path(args.first),Path(args.second)
    for root in (first,second): assert read(root/'FINAL_STATE.json')['status']=='complete'
    a=first/'jobs/rgb_stable_m_engineering'; b=second/'jobs/rgb_stable_m_engineering'
    checkpoint_pairs=0
    for p in sorted(a.rglob('*.pt')):
        q=b/p.relative_to(a)
        aa=torch.load(p,map_location='cpu',weights_only=False)
        bb=torch.load(q,map_location='cpu',weights_only=False)
        equal(aa,bb,str(p.relative_to(a))); checkpoint_pairs+=1
    history_a=[json.loads(x) for x in (a/'rgb_updates.jsonl').read_text(encoding='utf-8').splitlines()]
    history_b=[json.loads(x) for x in (b/'rgb_updates.jsonl').read_text(encoding='utf-8').splitlines()]
    equal(history_a,history_b,'all_training_metrics')
    trajectories=0
    for p in sorted(a.glob('RGB_*/*.npz')):
        with np.load(p,allow_pickle=False) as x, np.load(b/p.relative_to(a),allow_pickle=False) as y:
            equal({k:x[k] for k in x.files},{k:y[k] for k in y.files},str(p.relative_to(a)))
        trajectories+=1
    rejected=[]; previous=b/'checkpoints/ready.pt'
    for entry in sorted((b/'acceptance').glob('*.json')):
        record=read(entry); epoch=record['epoch']
        if epoch==0: continue
        checkpoint=b/'checkpoints'/f'epoch_{epoch:04d}.pt'
        if record['accepted']: previous=checkpoint
        else:
            equal(torch.load(previous,map_location='cpu',weights_only=False)['learner'],
                torch.load(checkpoint,map_location='cpu',weights_only=False)['learner'],'rejected_epoch_rollback')
            rejected.append(epoch)
    assert rejected,'The engineering case must exercise rollback'
    result=dict(passed=True,scope='short training on training-room engineering starts, not performance validation',
        process_gate=gate,checkpoint_pairs=checkpoint_pairs,training_rows=len(history_a),
        trajectory_pairs=trajectories,exact_checkpoint_and_optimizer_equality=True,
        rejected_epochs_restored_exactly=rejected,
        metric_proposals=sum(r['phase']=='metric' for r in history_a),
        applied_metric_proposals=sum(r.get('metric_step_accepted',False) for r in history_a))
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--first',required=True); p.add_argument('--second',required=True)
    p.add_argument('--process-snapshot',required=True); p.add_argument('--output',required=True)
    main(p.parse_args())
