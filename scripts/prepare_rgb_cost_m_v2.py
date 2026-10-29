"""Materialize the versioned RGB cost experiment before source freezing."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

def write_new(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: f.write(text)

def main():
    source=(ROOT/'extensions/rgb_stable_m_v1/runner.py').read_text(encoding='utf-8')
    source=source.replace("!=24: raise ValueError('Declare all24 rounds before training')", "!=20: raise ValueError('Declare all20 rounds before training')")
    source=source.replace("    if b['stability']['aux_gradient_cap']", "    if c['jobs'][0]['schedule']['representation_updates']!=0: raise ValueError('Frozen encoder during Q/M epochs')\n    if not 0<b['stability']['gamma']<=.99: raise ValueError('The cost bound requires gamma<=.99')\n    if len(b['regular_collection_quotas'])!=c['jobs'][0]['schedule']['epochs']: raise ValueError('Quota/epoch mismatch')\n    if b['stability']['aux_gradient_cap']",1)
    source=source.replace("folders=('extensions/rgb_stable_m_v1'", "folders=('extensions/rgb_cost_m_v2','extensions/rgb_stable_m_v1'")
    source=source.replace('scripts/run_rgb_stable_m_v1.py','scripts/run_rgb_cost_m_v2.py')
    source=source.replace('from .backend import StableMetricRGBBackend','from .backend import CostMetricRGBBackend')
    source=source.replace('return StableMetricRGBBackend(', 'return CostMetricRGBBackend(')
    source=source.replace("for subdir in ('extensions/rgb_stable_m_v1'", "for subdir in ('extensions/rgb_cost_m_v2','extensions/rgb_stable_m_v1'")
    source=source.replace("derivative_critic=counts.get('critic_prefit',0)+counts.get('representation',0),", "derivative_critic=counts.get('critic_prefit',0)+counts.get('representation',0),\n                    cumulative_cost_q=counts.get('q',0),")
    source=source.replace("schema='rgb_stable_metric_report_v1'", "schema='rgb_frozen_features_cost_metric_report_v2'")
    source=source.replace("random_origin=True,old_policy_weights_loaded=False,", "random_origin=True,old_policy_weights_loaded=False,\n        Q_input='observable visual state and actual4D command',Q_target='discounted physical cost with failure-only absorbing mask',\n        feature_freeze=shared.read(job/'FEATURES_FROZEN.json') if (job/'FEATURES_FROZEN.json').exists() else None,")
    write_new(ROOT/'extensions/rgb_cost_m_v2/runner.py',source)
    cli=(ROOT/'scripts/run_rgb_stable_m_v1.py').read_text(encoding='utf-8')
    cli=cli.replace('extensions.rgb_stable_m_v1','extensions.rgb_cost_m_v2')
    cli=cli.replace('Fresh RGB representation, component Critic and controlled metric updates.','Frozen visual coordinates, physical-action cost Q and controlled metric updates.')
    write_new(ROOT/'scripts/run_rgb_cost_m_v2.py',cli)
    folder=ROOT/'configs/rgb_cost_m_v2'
    for label,base in [('seed41','seed41'),('engineering','engineering_final')]:
        c=json.loads((ROOT/f'configs/rgb_stable_m_v1/{base}.json').read_text(encoding='utf-8-sig'))
        c['description']='Random AE and supervised feature pretraining, frozen coordinates, double physical-action cost Q and constrained M updates. No old model or replay. Independent raw and accepted validation curves.'
        job=c['jobs'][0]; b=job['config']; s=job['schedule']
        job['id']='rgb_cost_m_'+('seed41' if label=='seed41' else 'engineering')
        b.update(q_batch_size=512,metric_batch_size=128,representation_prefit_updates=3000,q_prefit_updates=2000)
        b['stability'].update(q_lr=3e-4,gamma=.99,target_tau=.01)
        s.update(pretrain_evaluator=1000,epochs=20,representation_updates=0,evaluator_updates=600,certificate_updates=200)
        if label=='engineering':
            b.update(q_batch_size=32,metric_batch_size=16,representation_prefit_updates=4,q_prefit_updates=8)
            s.update(warmup_representation=16,pretrain_evaluator=8,epochs=2,evaluator_updates=4,certificate_updates=8)
        b['regular_collection_quotas']=b['regular_collection_quotas'][:s['epochs']]
        cases=Path(b['case_manifest']).read_text(encoding='utf-8-sig')
        b['case_manifest']=str(folder/f'{label}_cases.json')
        write_new(Path(b['case_manifest']),cases)
        write_new(folder/f'{label}.json',json.dumps(c,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
