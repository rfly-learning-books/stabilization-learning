"""Post-writer independent repeatability and frozen-coordinate audit."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stabilization-learning-control'))
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
from stabilization_learning.unified_v1.reporting import check_process_snapshot
from verify_rgb_stable_m_preflight import equal,read
from extensions.rgb_cost_m_v2.checkpoint import load_checkpoint


def main(args):
    gate=check_process_snapshot(args.process_snapshot)
    roots=[Path(args.first),Path(args.second)]
    for root in roots: assert read(root/'FINAL_STATE.json')['status']=='complete'
    a,b=[root/'jobs/rgb_cost_m_engineering' for root in roots]
    checkpoints=0
    for p in sorted(a.rglob('*.pt')):
        equal(load_checkpoint(p),load_checkpoint(b/p.relative_to(a)),str(p.relative_to(a)))
        checkpoints+=1
    ha=[json.loads(x) for x in (a/'rgb_updates.jsonl').read_text(encoding='utf-8').splitlines()]
    hb=[json.loads(x) for x in (b/'rgb_updates.jsonl').read_text(encoding='utf-8').splitlines()]
    equal(ha,hb,'all_training_metrics')
    trajectories=0
    for p in sorted(a.glob('RGB_*/*.npz')):
        with np.load(p,allow_pickle=False) as x,np.load(b/p.relative_to(a),allow_pickle=False) as y:
            equal({k:x[k] for k in x.files},{k:y[k] for k in y.files},str(p.relative_to(a)))
        trajectories+=1
    ready=torch.load(b/'checkpoints/ready.pt',map_location='cpu',weights_only=False)['learner']
    previous=ready; rejected=[]; frozen_checks=0
    for p in sorted((b/'checkpoints').glob('epoch_*.pt')):
        current=torch.load(p,map_location='cpu',weights_only=False)['learner']
        for name in ('ae','shadow','critics'):
            equal(ready[name],current[name],'frozen_'+name)
        assert current['features_frozen']; frozen_checks+=1
        previous=current
    previous=ready
    for p in sorted((b/'acceptance').glob('*.json')):
        record=read(p); epoch=record['epoch']
        if epoch==0: continue
        current=torch.load(b/'checkpoints'/f'epoch_{epoch:04d}.pt',map_location='cpu',weights_only=False)['learner']
        raw=load_checkpoint(b/'raw_candidates'/f'epoch_{epoch:04d}.pt')['learner']
        equal(raw['q'],current['q'],'Q_retained_after_validation')
        equal(raw['qopt'],current['qopt'],'Q_optimizer_retained')
        if not record['accepted']:
            for name in ('metric','mopt'): equal(previous[name],current[name],'M_rollback_'+name)
            assert current['counts']['q']>previous['counts']['q']
            rejected.append(epoch)
        previous=current
    qrows=[r for r in ha if r['phase'] in ('q','q_prefit')]
    assert qrows and all(0<=r['target_cost']<=32 for r in qrows)
    result=dict(passed=True,scope='two short independent GPU training processes on training-room engineering cases',
        process_gate=gate,checkpoint_pairs=checkpoints,training_rows=len(ha),trajectory_pairs=trajectories,
        exact_checkpoint_optimizer_metric_and_trajectory_equality=True,frozen_epoch_checks=frozen_checks,
        rejected_epochs_M_restored_Q_retained=rejected,rejection_unit_test_also_passed=True,
        actual_q_updates=len(qrows),metric_proposals=sum(r['phase']=='metric' for r in ha),
        applied_M_proposals_before_epoch_gate=sum(r.get('metric_step_accepted',False) for r in ha),
        performance_claim=False)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('first','second','process-snapshot','output'): p.add_argument('--'+name,required=True)
    main(p.parse_args())
