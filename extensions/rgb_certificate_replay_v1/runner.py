"""Bounded fresh run, atomic progress, frozen validation selection and test."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F

from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
from stabilization_gym.tasks_gpu_rgb.certificate_control import value
from stabilization_learning.certificate_rgb import RGBCertificateLearner, LearnerConfig
from extensions.joint_certificate_v4.rollout import sample_offset
from .replay import Replay

ROOT=Path(__file__).resolve().parents[2]


def atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def state_hash(module):
    h=hashlib.sha256()
    for name,p in sorted(module.state_dict().items()):h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def small(images):
    return F.interpolate(images.permute(0,3,1,2).float(),size=(64,64),mode='area').round().clamp(0,255).to(torch.uint8)


def random_actions(n,device):
    x=torch.randn(n,4,device=device)
    x[:,:3]=F.normalize(x[:,:3],dim=-1)*torch.rand(n,1,device=device).pow(1/3)
    x[:,3]=torch.rand(n,device=device)*2-1
    return x


class GPUMonitor:
    def __init__(self,path):
        self.path=Path(path);self.rows=[];self.begin=time.time()
        self.proc=subprocess.Popen(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-lms','1000'],
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
    def read(self):
        with self.path.open('w',encoding='utf-8') as f:
            for line in self.proc.stdout:
                try:
                    gpu,mem,power=map(float,line.split(','));row=dict(unix=time.time(),gpu=gpu,memory_mib=mem,watts=power)
                    self.rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                except ValueError:continue
    def close(self):
        self.proc.terminate();self.proc.wait(timeout=10);self.thread.join(timeout=10)
        end=time.time()
        if not self.rows:return dict(samples=0,mean_utilization=None)
        times=np.asarray([self.begin]+[r['unix'] for r in self.rows]+[end])
        # Nearest endpoint extension includes the entire measured interval.
        weights=np.diff(times);values=np.asarray([self.rows[0]['gpu']]+[r['gpu'] for r in self.rows])
        return dict(samples=len(self.rows),measured_seconds=end-self.begin,time_weighted_utilization=float(np.dot(weights,values)/weights.sum()),
                    peak_memory_mib=max(r['memory_mib'] for r in self.rows),scope='whole device, entire formal run; includes collection, updates and evaluation')


class Run:
    def __init__(self,output,config):
        self.root=Path(output).resolve()
        self.root.mkdir(parents=True,exist_ok=False)
        self.config=config;self.started=time.time();self.deadline=self.started+60*config['maximum_minutes']
        self.event_id=0;self.logs=[];self.best_score=None;self.selected_epoch=None
        self.mon=GPUMonitor(self.root/'GPU_SAMPLES.jsonl')

    def guard(self):
        for p in (self.root/'STOP',ROOT/'STOP',ROOT/'runs/STOP'):
            if p.exists():raise InterruptedError('STOP requested: '+str(p))
        if time.time()>=self.deadline:raise TimeoutError('Absolute run deadline reached')
        if shutil.disk_usage(self.root).free<self.config.get('minimum_free_gib',8)*2**30:raise OSError('Minimum free disk threshold')

    def event(self,kind,**data):
        self.event_id+=1
        row=dict(kind=kind,unix=time.time(),elapsed_seconds=time.time()-self.started,**data)
        atomic(self.root/'events'/f'{self.event_id:06d}_{kind}.json',row)

    def checkpoint(self,name,learner,**metadata):
        self.guard();p=self.root/name;tmp=p.with_suffix('.tmp')
        torch.save(dict(**learner.payload(),metadata=metadata),tmp);os.replace(tmp,p)
        atomic(p.with_suffix('.json'),dict(file=str(p),sha256=sha(p),counts=learner.counts,**metadata))

    def record_updates(self,phase,count,fn):
        for i in range(count):
            self.guard();row=fn();row=dict(phase=phase,index=i+1,unix=time.time(),**row)
            with (self.root/'UPDATES.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if i==0 or (i+1)%50==0 or i+1==count:self.event('updates',phase=phase,completed=i+1,planned=count,metrics=row)


def make_cases(cfg,scenes,split,count,seed):
    rng=np.random.default_rng(seed)
    subset=sorted([s for s in scenes if s['split']==split],key=lambda s:s['id'])
    result=[]
    for i in range(count):
        scene=subset[i%len(subset)];goal=(i//len(subset))%len(scene['refs'])
        difficulty=('near','small','medium','large')[(i//len(subset)+i)%4]
        result.append(dict(case_id=f'{split}_{seed}_{i:04d}',scene_id=scene['id'],goal_id=goal,difficulty=difficulty,
                           offset=sample_offset(cfg,scene,goal,difficulty,rng).tolist()))
    return result


def execute_cases(run,cfg,lookup,learner,cases,*,plant='P0',replay=None,bootstrap=False,variant='selected',trace_name=None):
    c=run.config;n=len(cases);device=c['device']
    horizon=c['reach_steps']+c['post_success_steps']
    runtime=deepcopy(cfg);runtime['collection']['required_hold_observations']=horizon+1
    env=GPUParallelRGBEnv(runtime,[lookup[x['scene_id']] for x in cases],plant=plant,max_steps=horizon,device=device,
        cache_dir=ROOT/'.cache/rgb_certificate_replay_warp',samples_per_axis=2)
    rows=[];trace=[]
    try:
        public,info=env.reset(options=dict(offsets=[x['offset'] for x in cases],goal_ids=[x['goal_id'] for x in cases]))
        goals=small(public['goal_rgb']);images=small(public['rgb'])
        ids=replay.register_goals([f"{x['scene_id']}/g{x['goal_id']}" for x in cases],goals) if replay else None
        alive=torch.ones(n,device=device,dtype=torch.bool)
        first=torch.full((n,),-1,device=device,dtype=torch.long)
        success=torch.zeros_like(alive);post_bad=torch.zeros_like(first)
        cumulative=torch.zeros(n,device=device);effort=cumulative.clone();length=torch.zeros_like(first)
        predicted_good=torch.zeros_like(first);actual_bad=torch.zeros_like(first);total_cert=torch.zeros_like(first)
        reason=['timeout']*n
        offset=info['supervision']['offset'];previous=torch.zeros(n,4,device=device)
        for step in range(horizon):
            run.guard()
            if not bool(alive.any()):break
            if bootstrap:
                # Collection-only teacher is explicitly recorded; no truth is
                # passed to the learned controller, representation or Q inputs.
                if step%4==0:exploration=random_actions(n,device)*.35
                oracle=(-offset.float()*offset.new_tensor([2.5,2.5,2.5,2.86]).float()).clamp(-.6,.6)
                oracle[:,:3]/=oracle[:,:3].norm(dim=-1).clamp_min(1)[:,None]
                action=torch.where((torch.arange(n,device=device)%2==0)[:,None],oracle,exploration)
                diag=None
            else:
                action,diag=learner.act(images,goals,explore=replay is not None,variant=variant)
            action=torch.where(alive[:,None],action,0.)
            following,_,terminal,truncated,info=env.step(action)
            if not bool(info['transition_mask'][alive].all()):raise AssertionError('Autoreset entered replay')
            truth=info['supervision'];xn=small(following['rgb'])
            bad=info['failure'];strict=alive&(info['hold_observations']>=11)&~bad
            new_success=strict&(first<0)&(step+1<=c['reach_steps'])
            first=torch.where(new_success,step+1,first);success|=new_success
            qn=truth['next_offset']
            inside=(qn[:,:3].norm(dim=-1)<cfg['collection']['position_tolerance_m'])&(qn[:,3].abs()<math.radians(cfg['collection']['yaw_tolerance_deg']))
            post=alive&(first>=0)&(step+1>first)
            post_bad+= (post&~inside).long()
            post_end=(first>=0)&(step+1>=first+c['post_success_steps'])
            time_end=(first<0)&(step+1>=c['reach_steps'])
            end=alive&(bad|post_end|time_end|truncated)
            held=post_end&(post_bad==0)&~bad
            # Reward has physical coordinates and never contains the learned V.
            reward=-.1*(4*truth['physical_error'].clamp_max(4).float()+.1*(~inside).float()+.005*action.square().sum(-1)+.005*(action-previous).square().sum(-1))
            reward+=10*new_success.float()+20*(held&end).float()-50*bad.float()
            reward/=20.
            lengths_before=length.clone();length+=alive.long()
            cumulative+=alive*truth['integrated_error'].float();effort+=alive*truth['normalized_input_squared_integral'].float()
            if replay is not None:
                mask=alive.cpu().numpy()
                replay.append(dict(image=images[alive],next_image=xn[alive],goal_id=ids[mask],
                    offset=truth['offset'][alive].float(),energy=truth['physical_error_before'][alive].float(),
                    action=action[alive],previous=previous[alive],remaining=torch.full((int(alive.sum()),),1-step/horizon,device=device),
                    next_remaining=torch.full((int(alive.sum()),),max(0,1-(step+1)/horizon),device=device),
                    reward=reward[alive],terminal=end[alive],bootstrap=torch.full((int(alive.sum()),),bootstrap,device=device,dtype=torch.bool)))
            if diag is not None:
                with torch.no_grad():
                    en=learner.features(xn)-diag['goal'];actual=value(en,diag['matrix'])-diag['value']
                    residual=actual+.05*.1*diag['error'].square().sum(-1)
                predicted_good+=(alive&diag['model_feasible']).long();actual_bad+=(alive&(residual>1e-7)).long();total_cert+=alive.long()
                if trace_name:
                    trace.append(dict(error=diag['error'].cpu().numpy(),next_error=en.cpu().numpy(),matrix=diag['matrix'].cpu().numpy(),
                        action=action.cpu().numpy(),offset=truth['offset'].cpu().numpy(),next_offset=qn.cpu().numpy(),
                        V=diag['value'].cpu().numpy(),V_next=value(en,diag['matrix']).cpu().numpy(),
                        predicted_delta=diag['model_delta'].cpu().numpy(),model_feasible=diag['model_feasible'].cpu().numpy(),
                        active=alive.cpu().numpy()))
            for lane in torch.nonzero(end).flatten().cpu().tolist():
                reason[lane]='collision' if bool(info['collision'][lane]) else 'outside' if bool(info['outside'][lane]) else 'post_success_complete' if bool(post_end[lane]) else 'timeout'
            alive&=~end;images=xn;offset=qn;previous=action
        for i,case in enumerate(cases):
            rows.append(dict(**case,plant=plant,variant=variant,success=bool(success[i]),hold_success=bool(success[i] and post_bad[i]==0 and length[i]>=first[i]+c['post_success_steps']),
                first_success_step=int(first[i]),steps=int(length[i]),reason=reason[i],integrated_error=float(cumulative[i]),input_squared=float(effort[i]),
                post_outside_steps=int(post_bad[i]),model_feasible_steps=int(predicted_good[i]),actual_descent_violations=int(actual_bad[i]),certificate_steps=int(total_cert[i])))
        if trace_name and trace:
            path=run.root/'trajectories'/trace_name;path.parent.mkdir(exist_ok=True)
            arrays={k:np.stack([t[k] for t in trace]) for k in trace[0]}
            np.savez_compressed(path,**arrays)
        return rows
    finally:
        env.close()


def summary(rows):
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),hold_successes=sum(r['hold_success'] for r in rows),
        success_rate=float(np.mean([r['success'] for r in rows])),hold_rate=float(np.mean([r['hold_success'] for r in rows])),
        mean_integrated_error=float(np.mean([r['integrated_error'] for r in rows])),mean_input=float(np.mean([r['input_squared'] for r in rows])),
        actual_descent_violation_fraction=sum(r['actual_descent_violations'] for r in rows)/max(1,sum(r['certificate_steps'] for r in rows)),
        model_feasible_fraction=sum(r['model_feasible_steps'] for r in rows)/max(1,sum(r['certificate_steps'] for r in rows)))


def evaluate(run,cfg,lookup,learner,cases,label,variants=('selected',)):
    rows=[]
    for variant in variants:
        for plant in run.config['evaluation_plants']:
            # Group room instances to keep the render working set bounded.
            for room in sorted(set(x['scene_id'] for x in cases)):
                room_cases=[x for x in cases if x['scene_id']==room]
                for start in range(0,len(room_cases),run.config['parallel_envs']):
                    batch=room_cases[start:start+run.config['parallel_envs']]
                    rows+=execute_cases(run,cfg,lookup,learner,batch,plant=plant,variant=variant,
                        trace_name=f'{label}_{variant}_{plant}_{room}_{start}.npz')
    groups={f'{v}/{p}':summary([r for r in rows if r['variant']==v and r['plant']==p]) for v in variants for p in run.config['evaluation_plants']}
    atomic(run.root/f'{label}.json',dict(groups=groups,episodes=rows))
    run.event('evaluation_complete',label=label,groups=groups)
    return rows,groups


def collect_round(run,cfg,scenes,lookup,learner,replay,rng,round_id,bootstrap=False):
    c=run.config;rows=[];before=replay.total
    for room_i,scene in enumerate(scenes):
        cases=[]
        for i in range(c['parallel_envs']):
            goal=i%len(scene['refs']);difficulty=('near','small','medium','large')[(i//len(scene['refs'])+round_id)%4]
            cases.append(dict(case_id=f'train_{round_id}_{scene["id"]}_{i}',scene_id=scene['id'],goal_id=goal,difficulty=difficulty,
                offset=sample_offset(cfg,scene,goal,difficulty,rng).tolist()))
        rows+=execute_cases(run,cfg,lookup,learner,cases,replay=replay,bootstrap=bootstrap)
        run.event('collection_room',round=round_id,room=scene['id'],room_index=room_i+1,rooms=len(scenes),transitions=replay.total-before,bootstrap=bootstrap)
        if not bootstrap:
            run.record_updates(f'epoch_{round_id}_room_{room_i}_response',c['response_updates_per_room'],lambda:learner.response_update(replay.batch(c['batch_size'])))
            run.record_updates(f'epoch_{round_id}_room_{room_i}_q',c['q_updates_per_room'],lambda:learner.critic_update(replay.batch(c['batch_size'])))
            run.record_updates(f'epoch_{round_id}_room_{room_i}_M',c['m_updates_per_room'],lambda:learner.metric_update(replay.batch(c['batch_size'])))
    atomic(run.root/f'COLLECTION_{round_id:03d}.json',dict(new_transitions=replay.total-before,episodes=rows))


def audit_trajectories(root):
    maximum=0.;count=0
    for path in (Path(root)/'trajectories').glob('*.npz'):
        with np.load(path,allow_pickle=False) as a:
            mask=a['active'];m=a['matrix'].astype(np.float64)
            for key,logged in (('error','V'),('next_error','V_next')):
                e=a[key].astype(np.float64);v=.5*np.einsum('tbi,tbij,tbj->tb',e,m,e)
                delta=np.abs(v-a[logged]);maximum=max(maximum,float(delta[mask].max(initial=0.)))
                if not np.allclose(v[mask],a[logged][mask],rtol=2e-5,atol=2e-5):raise AssertionError('Independent full V audit failed: '+str(path))
            count+=int(mask.sum())
    return dict(passed=True,transitions=count,max_absolute_error=maximum,formula='0.5 e^T M e; both endpoints same frozen parameter version')


def train(output,config):
    run=Run(output,config);state='failed';error=None
    try:
        c=config;random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        adapter,cfg,scenes=load_runtime(ROOT)
        lookup={s['id']:s for s in scenes};train_scenes=sorted([s for s in scenes if s['split']=='train'],key=lambda x:x['id'])
        if c.get('train_scene_limit'):train_scenes=train_scenes[:c['train_scene_limit']]
        cases={s:make_cases(cfg,scenes,s,c[s+'_cases'],c['evaluation_seed']+(1 if s=='val' else 2)) for s in ('val','test')}
        sources=list((ROOT/'extensions/rgb_certificate_replay_v1').glob('*.py'))+list((ROOT/'stabilization-learning-control/stabilization_learning/certificate_rgb').glob('*.py'))
        sources += [ROOT/'scripts/run_rgb_certificate_replay_v1.py',ROOT/'stabilization-gym/stabilization_gym/tasks_gpu_rgb/certificate_control.py',ROOT/'stabilization-gym/stabilization_gym/tasks_gpu_rgb/environment.py']
        source_hashes={str(p):sha(p) for p in sources};snapshot=run.root/'SOURCE_SNAPSHOT';snapshot.mkdir()
        for p in sources:shutil.copyfile(p,snapshot/(sha(p)+'_'+p.name))
        protocol=dict(schema='rgb_certificate_replay_v1',config=c,started_unix=run.started,deadline_unix=run.deadline,
            sources=source_hashes,scene_manifest_sha256=sha(ROOT/'migration/visual_scene_reconstruction/scenes.json'),
            train_rooms=[s['id'] for s in train_scenes],cases=cases,initialization='random AE, response model, twin Q, goal M; empty Adam; no historical learned checkpoint or replay reads',
            bootstrap='half bounded random commands, half physical pose teacher only during initial data collection; truth absent from learned control and Q inputs',
            certificate='goal-conditioned SPD trace16 floor0.1; exact finite-step value under learned action-linear feature model',
            learning='AE reconstruction + training pose supervision + feature calibration; then frozen features; replay response fitting, reward TD twin Q and M-induced constrained categorical policy',
            enforcement='hard predicted discrete descent mask; infeasible minimum-residual fallback explicitly recorded; actual descent independently measured',
            reward='dt physical error/outside-tolerance/input/smoothness cost; first strict arrival +10, full extra hold +20, physical failure -50; divide all rewards by20; no learned V in reward',
            physical='existing 44 reconstructed rooms; GPU pixels; P0 train; P1 delayed/filter transfer evaluation; wall time excluded from dt',
            performance_scope='single seed; fresh independently sampled evaluation starts; validation only for selection; no new formal stability theorem',
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),automatic_resume=False)
        atomic(run.root/'protocol.json',protocol)
        run.event('initialized',deadline_unix=run.deadline,train_rooms=len(train_scenes),config=c)
        learner=RGBCertificateLearner(LearnerConfig(),c['device'])
        initial_hash={k:state_hash(getattr(learner,k)) for k in ('features','metric','response','q1','q2')}
        run.checkpoint('random_initial.pt',learner,epoch=-1,initial_hash=initial_hash)
        replay=Replay(c['replay_capacity'],c['seed']+71,c['device']);rng=np.random.default_rng(c['seed']+92)
        collect_round(run,cfg,train_scenes,lookup,learner,replay,rng,0,True)
        run.record_updates('representation_warmup',c['warmup_updates'],lambda:learner.warmup(replay.raw_batch(c['representation_batch'])))
        learner.freeze_features(replay.raw_batch(min(4096,replay.size)));replay.bind_features(learner.features,run.guard)
        feature_hash=state_hash(learner.features)
        run.event('features_frozen',feature_sha256=feature_hash,replay_size=replay.size,scale=float(learner.features.scale))
        run.record_updates('response_pretrain',c['response_pretrain'],lambda:learner.response_update(replay.batch(c['batch_size'])))
        run.record_updates('critic_pretrain',c['critic_pretrain'],lambda:learner.critic_update(replay.batch(c['batch_size'])))
        run.checkpoint('ready.pt',learner,epoch=0)
        evaluate(run,cfg,lookup,learner,cases['val'],'VAL_000')
        for epoch in range(1,c['epochs']+1):
            collect_round(run,cfg,train_scenes,lookup,learner,replay,rng,epoch)
            if state_hash(learner.features)!=feature_hash:raise AssertionError('Frozen features changed')
            rows,groups=evaluate(run,cfg,lookup,learner,cases['val'],f'VAL_{epoch:03d}')
            score=(sum(x['success'] for x in rows),sum(x['hold_success'] for x in rows),-sum(x['integrated_error'] for x in rows))
            run.checkpoint(f'epoch_{epoch:03d}.pt',learner,epoch=epoch,validation_score=score)
            if run.best_score is None or score>run.best_score:
                run.best_score=score;run.selected_epoch=epoch
                run.checkpoint('selected.pt',learner,epoch=epoch,validation_score=score)
            run.event('epoch_complete',epoch=epoch,planned_epochs=c['epochs'],counts=learner.counts,selected_epoch=run.selected_epoch,total_transitions=replay.total)
        run.checkpoint('final.pt',learner,epoch=c['epochs'])
        atomic(run.root/'FROZEN.json',dict(selected_epoch=run.selected_epoch,selected_sha256=sha(run.root/'selected.pt'),final_sha256=sha(run.root/'final.pt'),feature_sha256=feature_hash,freeze_unix=time.time()))
        run.event('frozen',selected_epoch=run.selected_epoch)
        learner.restore_models(torch.load(run.root/'selected.pt',map_location=c['device'],weights_only=False))
        rows,groups=evaluate(run,cfg,lookup,learner,cases['test'],'TEST',variants=('selected','identity','initial_metric'))
        audit=audit_trajectories(run.root);atomic(run.root/'AUDIT.json',audit)
        replay_meta=replay.save(run.root/'replay');atomic(run.root/'REPLAY.json',replay_meta)
        atomic(run.root/'RESULTS.json',dict(groups=groups,selected_epoch=run.selected_epoch,counts=learner.counts,actual_new_transitions=replay.total,initial_hash=initial_hash,audit=audit))
        state='complete'
    except BaseException as ex:
        error=repr(ex);state='stopped' if isinstance(ex,InterruptedError) else 'timeout' if isinstance(ex,TimeoutError) else 'failed'
        (run.root/'FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        run.event('run_failed',state=state,error=error)
    finally:
        gpu=run.mon.close();atomic(run.root/'GPU_SUMMARY.json',gpu)
        atomic(run.root/'FINAL_STATE.json',dict(status=state,error=error,finished_unix=time.time(),elapsed_seconds=time.time()-run.started,selected_epoch=run.selected_epoch,gpu=gpu))
    return 0 if state=='complete' else 1
