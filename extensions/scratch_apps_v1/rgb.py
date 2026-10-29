"""Random shared RGB encoder, derivative critic and goal-conditioned M.

The renderer and fixed public FastDepth are explicit environment dependencies.
Policy/AE weights and RGB replay are never imported from earlier experiments.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
import math
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, Run, paths, read, write, sha, save_torch, seed_all, tensor_hash


def runtime(run, seed):
    paths()
    from stabilization_gym.envs.tasks.original_visual_online import OriginalVisualOnlineAdapter
    adapter = OriginalVisualOnlineAdapter(ROOT,
        ROOT/'migration/visual_scene_reconstruction/reconstructed_legacy_root',
        ROOT/'migration/visual_scene_reconstruction/SCENE_AUDIT.json')
    cfg=copy.deepcopy(adapter.cfg); c=run.config['rgb']
    cfg['seed']=seed; cfg['device']=run.config['device']
    cfg['paths']['initial_autoencoder']=None
    cfg['encoder']['initialization']='random in this run; no previous AE or SVD checkpoint'
    cfg['derivative'].pop('source_run',None);cfg['derivative'].pop('initialization_checkpoint',None)
    cfg['goal_fixed'].update(warmup_updates=c['warmup_updates'],rounds=c['rounds'],
        representation_updates=c['representation_updates'],metric_updates=c['metric_updates'])
    cfg['derivative']['critic_prefit_updates']=c['critic_prefit_updates']
    cfg['collection']['steps']=c['episode_steps'];cfg['evaluation']['max_steps']=c['evaluation_steps']
    cfg['evaluation']['validation_starts_per_goal_difficulty']=c['validation_repeats']
    cfg['evaluation']['test_starts_per_goal_difficulty']=c['test_repeats']
    return adapter,cfg


@contextmanager
def fresh_initialization_only():
    """Make any accidental torch checkpoint read during initialization fatal."""
    original=torch.load
    def forbidden(*args,**kwargs):
        raise RuntimeError('No checkpoint reads are allowed during scratch model initialization')
    torch.load=forbidden
    try:yield
    finally:torch.load=original


def create_learner(cfg,device):
    from goal_fixed_core import make_model,GoalMetric
    from derivative_critic_core import DerivativeLearner
    with fresh_initialization_only():
        ae=make_model(cfg,initialize=False);metric=GoalMetric()
        source={'learner':dict(ae=ae.state_dict(),metric=metric.state_dict(),
            anchors=dict(rec=1.,curvature=1.,latent_std=[1.]*16))}
        learner=DerivativeLearner(cfg,source,device)
    return learner


@contextmanager
def random_behavior(rng):
    """Random bounded commands, independent of pose and policy parameters."""
    import train_goal_fixed as joint
    from goal_fixed_core import saturate
    original=joint.Reference
    class RandomReference(original):
        def observe(self,rgb):
            _,e,v=super().observe(rgb)
            u=saturate(torch.tensor(rng.uniform(-.5,.5,4),dtype=torch.float32)).numpy()
            return u,e,v
    joint.Reference=RandomReference
    try:yield
    finally:joint.Reference=original


def evaluation(run,cfg,out,learner,depth,scenes,starts,tag,methods,plants):
    import train_goal_fixed as joint
    byid={s['id']:s for s in scenes};folder=out/'evaluation'/tag
    if folder.exists():raise FileExistsError('Partial evaluation is preserved: '+str(folder))
    folder.mkdir(parents=True);rows=[];inst=None;sid=None
    try:
        for method in methods:
            main=learner.ae
            if method=='shadow_ae_identity':learner.ae=learner.shadow
            if method=='initial_ae_identity':learner.ae=learner.initial
            try:
                for plant in plants:
                    for i,start in enumerate(starts):
                        run.guard();scene=byid[start['scene_id']]
                        if sid!=scene['id']:
                            if inst:inst.close()
                            inst=joint.new_instance(cfg,scene);sid=scene['id']
                        arrays,row=joint.episode(cfg,out,scene,start['goal_id'],start['offset'],learner,
                            depth,np.random.default_rng(run.config['sampling_seed']+8000+i),
                            steps=cfg['evaluation']['max_steps'],mode='learned' if method=='learned' else 'identity_m',
                            plant_name=plant,inst=inst)
                        file=folder/f'{method}_{plant}_{i:04d}.npz'
                        np.savez_compressed(file,**{k:v for k,v in arrays.items()
                            if k not in ('rgb','goal_rgb','smooth_center','smooth_minus','smooth_plus')})
                        # Complete T+1 V, including the successor after the final command.
                        D=arrays['features'];M=arrays['M'];vn=.5*np.einsum('ti,ij,tj->t',D,M,D)
                        mismatch=float(np.max(abs(vn-arrays['V'])))
                        if len(arrays['V'])!=len(arrays['action'])+1 or mismatch>1e-3*max(1.,float(abs(vn).max())):
                            raise RuntimeError('RGB successor certificate audit failed')
                        row.update(method=method,plant=plant,tag=tag,trace=file.name,trace_sha256=sha(file),
                            numpy_V_max_error=mismatch,difficulty=start['difficulty'])
                        write(file.with_suffix('.json'),row);rows.append(row)
            finally:learner.ae=main
    finally:
        if inst:inst.close()
    summaries={}
    for method in methods:
        for plant in plants:
            subset=[r for r in rows if r['method']==method and r['plant']==plant]
            summaries[f'{method}/{plant}']=joint.aggregate(subset)
    report=dict(episodes=rows,summaries=summaries)
    write(folder/'RESULTS.json',report)
    return report


def train(run:Run,seed):
    seed_all(seed);adapter,cfg=runtime(run,seed);c=run.config['rgb'];device=run.config['device']
    from goal_fixed_core import Replay,FastDepth
    from lac_vs import legacy
    from lac_vs.environment import sample_start,make_evaluation_starts
    from lac_vs.config import rng_state,restore_rng
    import train_goal_fixed as joint
    out=run.new_stage(f'rgb_seed{seed}',dict(config=cfg,scratch_config=c,
        model_initialization='random AE/goal M/derivative critic; checkpoint reads prohibited during init',
        replay_initialization='empty; new random bounded commands and new initial poses',
        scene_scope='44 existing reconstructed room geometries; freshly sampled starts and transitions',
        fixed_pretrained_dependency='public FastDepth only; not a learned control/AE initialization'))
    learner=create_learner(cfg,device);depth=FastDepth(cfg,device)
    replay=Replay(out,cfg,seed=seed+14000)
    if replay.rows:raise RuntimeError('Scratch replay must begin empty')
    manifest=read(ROOT/'migration/visual_scene_reconstruction/scenes.json')
    scenes=manifest['scenes']
    if c.get('smoke'):
        scenes=[next(s for s in scenes if s['split']==split) for split in ('train','val','test')]
    manifest=dict(schema=1,scenes=scenes,new_scene_seeds=False,new_state_and_action_samples=True)
    write(out/'scenes.json',manifest)
    evalcfg=copy.deepcopy(cfg);evalcfg['seed']=run.config['sampling_seed']+9000
    starts={split:make_evaluation_starts(evalcfg,manifest,split) for split in ('val','test')}
    if c.get('smoke'):starts={k:v[:1] for k,v in starts.items()}
    write(out/'evaluation_starts.json',starts)
    rng=np.random.default_rng(seed*100000+27);history=[];sample_count=0
    hashes0={k:tensor_hash(getattr(learner,k)) for k in ('ae','metric','critics')}
    def save(name,phase,step):
        save_torch(out/name,dict(schema='scratch_rgb_v1',learner=learner.payload(),
            initial_ae=learner.initial.state_dict(),rng=rng_state(),collection_rng=rng.bit_generator.state,
            replay_rng=replay.rng.bit_generator.state,replay_rows=replay.rows,phase=phase,step=step,
            config=cfg,origin='random',raw_initial_hashes=hashes0,transitions_sampled=sample_count,
            protocol_sha256=sha(run.root/'protocol.json')))
    save('random_raw.pt','before_sampling',0)
    envmod=legacy.module(cfg,'room_env');train_scenes=[s for s in scenes if s['split']=='train']
    def collect(round_index,episodes_per_scene,random_actions=False):
        nonlocal sample_count
        for scene in train_scenes:
            inst=joint.new_instance(cfg,scene)
            try:
                for ep in range(episodes_per_scene):
                    run.guard();gi=int(rng.integers(len(scene['refs'])))
                    frame=envmod.ReferenceFrame(np.asarray(scene['refs'][gi]['goal_matrix']))
                    difficulty=('small','medium','large')[int(rng.integers(3))]
                    start=sample_start(cfg,scene,frame,rng,difficulty)
                    kwargs=dict(steps=c['bootstrap_steps'] if random_actions else c['episode_steps'],collect=True,inst=inst)
                    if random_actions:
                        with random_behavior(rng):
                            arrays,row=joint.episode(cfg,out,scene,gi,start,learner,depth,rng,**kwargs)
                    else:arrays,row=joint.episode(cfg,out,scene,gi,start,learner,depth,rng,**kwargs)
                    replay.add(arrays,row,f'r{round_index:03d}_{scene["id"]}_{ep:03d}')
                    sample_count+=len(arrays['action'])
                    run.event('rgb_collection',seed=seed,round=round_index,scene=scene['id'],episode=ep,
                        random_actions=random_actions,new_transitions=len(arrays['action']),total_sampled=sample_count)
            finally:inst.close()
    collect(0,c['bootstrap_episodes_per_scene'],True)
    # Fit normalization only on the new training images; no pose input at deployment.
    learner.ta=copy.deepcopy(learner.ae)
    joint.calibrate(learner,replay,out)
    del learner.ta
    for model in (learner.shadow,learner.initial):model.load_state_dict(learner.ae.state_dict())
    save('initial.pt','calibrated_random_before_gradient',0)
    global_step=0;best=None;best_step=0
    def validate(round_index):
        nonlocal best,best_step
        state=rng_state()
        report=evaluation(run,cfg,out,learner,depth,scenes,starts['val'],f'validation_{round_index:03d}',
                          ['learned'],['P0'])
        restore_rng(state)
        rows=report['episodes'];sr=float(np.mean([r['success'] for r in rows]))
        worst=min(float(np.mean([r['success'] for r in rows if r['scene_id']==s])) for s in set(r['scene_id'] for r in rows))
        err=float(np.mean([r['integrated_error'] for r in rows]));key=(sr,worst,-err)
        if global_step>0 and (best is None or key>best):
            best=key;best_step=global_step;save('selected.pt','validation_selected',global_step)
        run.event('rgb_validation',seed=seed,round=round_index,update=global_step,
                  selected_update=best_step,success_rate=sr,mean_cumulative_error=err)
    validate(0)
    def updates(phase,count,round_index):
        nonlocal global_step
        for i in range(count):
            run.guard();b=replay.sample(cfg['goal_fixed']['batch_size'])
            if phase=='warmup':stats=learner.representation(b,warmup=True)
            elif phase=='critic_prefit':stats=learner.critic_prefit(b)
            elif phase=='representation':stats=learner.representation(b)
            else:stats=learner.metric_update(b)
            global_step+=1
            stats.update(phase=phase,update=global_step,phase_step=i+1,round=round_index,seed=seed)
            history.append(stats)
            if (i+1)%c['log_every']==0 or i+1==count:
                save('last.pt',phase,global_step);run.event('rgb_update',**stats)
    updates('warmup',c['warmup_updates'],0)
    updates('critic_prefit',c['critic_prefit_updates'],0)
    for round_index in range(1,c['rounds']+1):
        collect(round_index,c['episodes_per_scene'])
        updates('representation',c['representation_updates'],round_index)
        updates('metric',c['metric_updates'],round_index)
        if round_index%c['eval_every']==0 or round_index==c['rounds']:validate(round_index)
    save('final.pt','complete',global_step);write(out/'TRAINING.json',history)
    write(out/'FROZEN.json',dict(checkpoint='selected.pt',sha256=sha(out/'selected.pt'),
        selected_update=best_step,gradient_updates=global_step,transitions_sampled=sample_count,
        raw_initial_hashes=hashes0,final_hashes={k:tensor_hash(getattr(learner,k)) for k in hashes0},
        fixed_fastdepth_sha256=depth.hash,old_replay_imported=False))
    run.event('rgb_frozen',seed=seed,gradient_updates=global_step,selected_update=best_step)


def test(run:Run,seed):
    run.frozen_group('rgb');adapter,cfg=runtime(run,seed);out=run.root/f'rgb_seed{seed}'
    from goal_fixed_core import FastDepth
    ck=torch.load(out/'selected.pt',map_location=run.config['device'],weights_only=False)
    learner=create_learner(cfg,run.config['device']);learner.restore(ck['learner'])
    learner.initial.load_state_dict(ck['initial_ae']);depth=FastDepth(cfg,run.config['device'])
    report=evaluation(run,cfg,out,learner,depth,read(out/'scenes.json')['scenes'],
        read(out/'evaluation_starts.json')['test'],'heldout_test',
        ['learned','identity_m','shadow_ae_identity','initial_ae_identity'],['P0','P1'])
    write(out/'RESULTS.json',report)
    write(out/'FINAL_STATE.json',dict(status='complete',checkpoint_sha256=sha(out/'selected.pt'),
        original_rgb_equivalent=False,new_room_geometries=False,new_samples=True,selection_uses_test=False))
    run.event('rgb_test_complete',seed=seed,summaries=report['summaries'])
