import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time
import torch
import numpy as np
from .environment import SixPrimitiveEnv,SixConfig,TASKS,SPLITS,RESET_MODES
from .learning import DirectDQ,Replay,Actor
from ..runtime import write_json


def serial(info,lane):
    return {k:v[lane].detach().cpu().tolist() for k,v in info.items()}


def validate_scratch(args, checkpoint=None):
    if not args.from_scratch:
        return
    if args.warmstart or args.bootstrap_steps or args.behavior_weight or args.guided_exploration:
        raise ValueError('--from-scratch forbids warmstart, behavior cloning and guided actions')
    if checkpoint is not None and (not checkpoint.get('from_scratch',False) or checkpoint.get('demonstrations') is not None):
        raise ValueError('Scratch continuation accepts only checkpoints from this scratch protocol, without demonstrations')


def save_checkpoint(learner, config, args, seen, path):
    value=dict(**learner.checkpoint(),config=asdict(config),transitions=seen,
               from_scratch=args.from_scratch,arguments=vars(args))
    temp=path.with_suffix('.pt.tmp')
    torch.save(value,temp)
    temp.replace(path)


def train(args):
    previous=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if previous is not None and previous.get('from_scratch',False):
        args.from_scratch=True
    validate_scratch(args,previous)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    c=SixConfig(num_envs=args.num_envs,horizon=args.horizon,reset_mode=args.reset_mode,
                fixed_task=args.task,fixed_shape=args.shape,contraction=args.contraction,
                distance_min=args.distance_min,distance_max=args.distance_max,angle_min=args.angle_min,angle_max=args.angle_max,reach_shaping=args.reach_shaping,contact_state=args.contact_state,
                task_ids=(0,1,2,3) if args.task_set=='translations' else (0,1,2,3,4,5))
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(2)
    env=SixPrimitiveEnv(c)
    try:
        write_json(out/'manifest.json',dict(**env.provenance(),arguments=vars(args)))
        obs,_=env.reset(seed=args.seed)
        learner=DirectDQ(env.device,penalty=args.penalty,contraction=c.contraction,behavior_weight=args.behavior_weight,reach_shaping=args.reach_shaping,progress_shaping=args.progress_shaping,prep_shaping=args.prep_shaping,horizon_steps=args.finite_horizon)
        if args.warmstart:
            previous=torch.load(args.warmstart,map_location=env.device,weights_only=False)
            # Encoder transfer only: goal/task-dependent action heads learn anew.
            learner.actor.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in previous['actor'].items() if k.startswith('encoder.')})
            learner.actor_target.load_state_dict(learner.actor.state_dict())
        if args.resume:
            if previous['config']['fixed_task']!=c.fixed_task:raise ValueError('Resume task mismatch')
            if (c.fixed_task is None or c.fixed_task>=4) and previous['config'].get('rotation_objective')!=c.rotation_objective:
                raise ValueError('Rotation objective changed: old critics/replay cannot be resumed; use a fresh run or encoder warmstart')
            for name in ('actor','critics','actor_target'):
                getattr(learner,name).load_state_dict(previous[name])
            learner.critic_target.load_state_dict(previous['critic_target'])
            learner.actor_opt.load_state_dict(previous['actor_optimizer'])
            learner.critic_opt.load_state_dict(previous['critic_optimizer'])
            learner.updates=previous['updates']
            if previous.get('demonstrations'):
                demo=previous['demonstrations'];learner.demonstrations=Replay(demo['size'])
                learner.demonstrations.size=demo['size']
                learner.demonstrations.storage={k:v.cpu() for k,v in demo['storage'].items()}
        replay=Replay(args.capacity);episodes=[];violations=0;seen=0;losses={}
        trajectories=[[] for _ in range(c.num_envs)];her_count=0
        windowed=args.finite_horizon>1 and not args.her
        pending=[];last_terminal=torch.full((c.num_envs,),float('nan'),device=env.device)
        noise=torch.zeros(c.num_envs,2,device=env.device)
        started=time.perf_counter()
        for step in range(args.steps):
            if step==args.warmup and args.bootstrap_steps:
                fit=learner.bootstrap(replay,args.bootstrap_steps,args.batch_size)
                torch.save(dict(**learner.checkpoint(),config=asdict(c),transitions=seen,
                                initialization='real-guided-exploration behavior cloning; before online DQ updates'),out/'bootstrap.pt')
                print(json.dumps({'bootstrap_mse':fit,'bootstrap_updates':args.bootstrap_steps}),flush=True)
            with torch.no_grad():
                action=learner.actor(obs)
                if step%args.exploration_hold==0:
                    noise=2*torch.rand_like(action)-1 if args.from_scratch and step<args.warmup else torch.randn_like(action)
                # Scratch warmup/noise is task-independent; the other branch is
                # retained for historical guided experiments only.
                if args.from_scratch:
                    action=noise.clamp(-1,1) if step<args.warmup else (action+args.noise*noise).clamp(-1,1)
                else:
                    axes=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[0.,0.],[0.,0.]],device=env.device)
                    radial=obs['s'][:,11:13]
                    radial=radial/radial.norm(dim=-1,keepdim=True).clamp_min(.01)
                    tangent=torch.stack((-radial[:,1],radial[:,0]),-1)
                    signed=torch.where((env.task==4)[:,None],-tangent,tangent)
                    direction=torch.where((env.task>=4)[:,None],signed-.5*radial,axes[env.task])
                if not args.from_scratch and step<args.warmup:
                    action=(.25*direction+.12*noise).clamp(-1,1)
                    if args.guided_exploration:
                        # Training-only randomized pulses above measured stiction range.
                        amplitude=torch.where(env.task>=4,.3,.58)[:,None]
                        action=(amplitude*direction+.07*noise).clamp(-1,1)
                        stop=obs['g'][:,0]<=torch.where(env.task>=4,.02,.003)
                        action[stop]=0
                elif not args.from_scratch:action=(action+args.noise*(.5*direction+noise)).clamp(-1,1)
            nxt,reward,term,trunc,info=env.step(action)
            if windowed:
                scale=.1+.9*obs['task'][:,4:].sum(-1)
                v_after=.5*(info['g']/scale).square()
                done_now=term|trunc
                if done_now.any():last_terminal[done_now]=v_after[done_now]
                pending.append((obs,action,nxt,info,term,env.steps.clone(),info['energy_before'].clone()))
                if len(pending)>args.finite_horizon:
                    o0,a0,n0,i0,t0,s0,e0=pending.pop(0)
                    valid=env.steps==s0+args.finite_horizon
                    fallback=torch.where(torch.isnan(last_terminal),e0,last_terminal)
                    i0=dict(i0)
                    i0['d_h']=torch.where(valid,v_after-e0,fallback-e0)
                    sc0=.1+.9*o0['task'][:,4:].sum(-1)
                    ph=(o0['g'][:,0]-info['g'])/sc0
                    i0['p_h']=torch.where(valid,ph,torch.zeros_like(ph))
                    i0['valid_h']=valid.float()
                    replay.add(o0,a0,n0,i0,t0)
            else:
                replay.add(obs,action,nxt,info,term)
            violations+=int((info['d_violation']>1e-6).sum());seen+=c.num_envs
            if replay.size>=args.batch_size and step>=args.warmup:
                for _ in range(args.updates_per_step):
                    losses=learner.update(replay,args.batch_size)
            done=term|trunc
            if args.her:
                from .relabel import relabel_episode
                for lane in range(c.num_envs):
                    one=lambda data:{k:v[lane:lane+1].detach().cpu().clone() for k,v in data.items()}
                    trajectories[lane].append((one(obs),action[lane:lane+1].detach().cpu().clone(),one(nxt),one(info),term[lane:lane+1].cpu()))
                    if done[lane]:
                        for o,a,n,i,t in relabel_episode(trajectories[lane],c):
                            replay.add(o,a,n,i,t);her_count+=1
                        trajectories[lane]=[]
            for lane in done.nonzero().flatten().tolist():
                episodes.append(dict(step=step+1,lane=lane,**serial(info,lane)))
            obs=env.reset(env_idx=done.nonzero().flatten())[0] if done.any() else nxt
            if args.save_every and (step+1)%args.save_every==0:
                torch.save({'actor':learner.actor.state_dict(),'config':asdict(c),'transitions':seen,
                            'method':'direct DQ actor snapshot','certified':False},out/f'policy_step_{step+1:06d}.pt')
                save_checkpoint(learner,c,args,seen,out/f'checkpoint_step_{step+1:06d}.pt')
            if (step+1)%50==0 or step+1==args.steps:
                record=dict(vector_steps=step+1,transitions=seen,mean_abs_g=float(info['g'].abs().mean()),
                            mean_prep_error=float(info['prep'].mean()),
                            actual_d_violation_fraction=violations/seen,episodes=len(episodes),
                            successes=sum(e['success'] for e in episodes),her_transitions=her_count,
                            elapsed_s=time.perf_counter()-started,
                            transitions_per_s=seen/(time.perf_counter()-started),
                            torch_peak_cuda_mib=torch.cuda.max_memory_allocated(env.device)/2**20,**losses)
                print(json.dumps(record),flush=True)
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        save_checkpoint(learner,c,args,seen,out/'last.pt')
        write_json(out/'episodes.json',episodes)
    finally:env.close()


def evaluate(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed);torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    selected_task=saved['config']['fixed_task'] if args.task is None else args.task
    requested=[selected_task] if selected_task is not None else list(saved['config'].get('task_ids',range(6)))
    if args.num_envs%len(requested):raise ValueError('Evaluation num-envs must be divisible by the number of tasks')
    config={**saved['config'],'num_envs':args.num_envs,'split':args.split,'fixed_task':None,'fixed_shape':None,'reset_mode':args.reset_mode,'task_ids':tuple(requested)}
    if args.horizon is not None:config['horizon']=args.horizon
    env=SixPrimitiveEnv(SixConfig(**config),render_mode='rgb_array' if args.video else None)
    actor=Actor().to(env.device);actor.load_state_dict(saved['actor']);actor.eval()
    task_ids=torch.tensor(requested,device=env.device,dtype=torch.long).repeat(args.num_envs//len(requested))
    records=[];planned=len(SPLITS[args.split])*args.num_envs*args.episodes
    try:
        write_json(out/'manifest.json',dict(**env.provenance(),arguments=vars(args)))
        # fixed_shape changes reset selection within the already-built split catalogue.
        for shape in SPLITS[args.split]:
            env.raw.fixed_shape=shape
            for rep in range(args.episodes):
                obs,_=env.reset(seed=args.seed+rep+100*SPLITS[args.split].index(shape),task=task_ids)
                finished=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
                trajectories=[[] for _ in range(args.num_envs)];frames=[]
                while not finished.all():
                    with torch.no_grad():action=actor(obs)
                    nxt,_,term,trunc,info=env.step(action)
                    if args.video and len(trajectories[0])%4==0:
                        frame=env.render()
                        if isinstance(frame,torch.Tensor):frame=frame.detach().cpu().numpy()
                        if isinstance(frame,np.ndarray):
                            if frame.ndim==4:frame=np.concatenate(list(frame),axis=1)
                            frames.append(frame.astype(np.uint8))
                    for lane in range(args.num_envs):
                        if not finished[lane]:trajectories[lane].append(serial(info,lane))
                    done=term|trunc
                    for lane in (done&~finished).nonzero().flatten().tolist():
                        records.append(dict(shape=shape,task_name=TASKS[int(task_ids[lane])],repeat=rep,
                                            steps=len(trajectories[lane]),**serial(info,lane)))
                    finished|=done
                    # Reset completed lanes to keep physics API valid, discard their later records.
                    obs=env.reset(env_idx=done.nonzero().flatten())[0] if done.any() else nxt
                for lane,trajectory in enumerate(trajectories):write_json(out/f'{shape}_{TASKS[int(task_ids[lane])]}_lane{lane}_{rep}.json',trajectory)
                if frames:
                    import imageio.v2 as imageio
                    imageio.mimsave(out/f'{shape}_{rep}.mp4',frames,fps=5)
        groups={}
        for r in records:
            group=groups.setdefault(f"{r['shape']}/{r['task_name']}",{'episodes':0,'successes':0})
            group['episodes']+=1;group['successes']+=int(r['success'])
        report={'planned_episodes':planned,'completed_episodes':len(records),
                'successes':sum(r['success'] for r in records),'strict_successes':sum(r['strict_success'] for r in records),'groups':groups,'episodes':records,'certified':False}
        rotations=[r for r in records if r['task']>=4]
        if rotations:
            completed=[r for r in rotations if r['success']]
            report['rotation_translation_error']={
                'objective':'sum of distance from initial position; no hard drift tolerance',
                'mean_integral_m_s':float(np.mean([r['translation_error_integral'] for r in rotations])),
                'successful_mean_integral_m_s':float(np.mean([r['translation_error_integral'] for r in completed])) if completed else None,
                'mean_max_drift_m':float(np.mean([r['max_side_error'] for r in rotations]))}
        write_json(out/'report.json',report)
        print(json.dumps({k:report[k] for k in ('planned_episodes','completed_episodes','successes','strict_successes')}),flush=True)
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description='Six SAPIEN primitives: direct pi(a|s,g,shape,task)')
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    t.add_argument('--steps',type=int,default=20000,help='vector steps')
    t.add_argument('--save-every',type=int,default=0)
    t.add_argument('--num-envs',type=int,default=12)
    t.add_argument('--from-scratch',action='store_true',help='no reference/teacher, cloning or external checkpoint; unbiased random exploration')
    t.add_argument('--task-set',choices=['translations','all'],default='all')
    t.add_argument('--updates-per-step',type=int,default=1)
    t.add_argument('--exploration-hold',type=int,default=8)
    t.add_argument('--horizon',type=int,default=240)
    t.add_argument('--task',type=int,choices=range(6))
    t.add_argument('--shape',choices=SPLITS['train'])
    t.add_argument('--warmup',type=int,default=200)
    t.add_argument('--batch-size',type=int,default=128)
    t.add_argument('--capacity',type=int,default=20000)
    t.add_argument('--noise',type=float,default=.15)
    t.add_argument('--penalty',type=float,default=100.)
    t.add_argument('--contraction',type=float,default=1.)
    t.add_argument('--behavior-weight',type=float,default=0.)
    t.add_argument('--bootstrap-steps',type=int,default=0)
    t.add_argument('--reach-shaping',type=float,default=0.)
    t.add_argument('--progress-shaping',type=float,default=0.,help='fixed potential shaping on normalized primary error, Q TD target only')
    t.add_argument('--prep-shaping',type=float,default=0.,help='fixed potential shaping on contour-based contact-preparation error')
    t.add_argument('--finite-horizon',type=int,default=1,help='D/progress label horizon H in vector steps (1 = per-step)')
    t.add_argument('--contact-state',action='store_true',help='use measured contact force instead of remaining time as s[17]')
    t.add_argument('--distance-min',type=float,default=.02)
    t.add_argument('--distance-max',type=float,default=.08)
    t.add_argument('--angle-min',type=float,default=.15)
    t.add_argument('--angle-max',type=float,default=1.57)
    t.add_argument('--guided-exploration',action='store_true',help='training-only randomized contact pulses with goal-dependent stops')
    t.add_argument('--her',action='store_true',help='relabel achieved scalar goals within real episodes')
    t.add_argument('--warmstart')
    t.add_argument('--resume',help='continue actor/critics and optimizers in a fresh output directory')
    e=sub.add_parser('evaluate')
    e.add_argument('--checkpoint',required=True)
    e.add_argument('--num-envs',type=int,default=24)
    e.add_argument('--task',type=int,choices=range(6),help='defaults to checkpoint task set; parallel lanes sample different states')
    e.add_argument('--split',choices=list(SPLITS),default='validation')
    e.add_argument('--episodes',type=int,default=3)
    e.add_argument('--horizon',type=int)
    e.add_argument('--video',action='store_true')
    for p in (t,e):
        p.add_argument('--output',required=True)
        p.add_argument('--seed',type=int,default=20260910)
        p.add_argument('--reset-mode',choices=RESET_MODES,default='fixed_origin')
    args=parser.parse_args()
    if args.command=='train':
        if min(args.steps,args.num_envs,args.horizon,args.batch_size,args.updates_per_step,args.exploration_hold)<1 or args.capacity<max(args.num_envs,args.batch_size) or args.warmup<0:
            parser.error('Invalid training sizes')
        if not np.isfinite(args.penalty) or args.penalty<0 or not np.isfinite(args.noise) or args.noise<0:parser.error('Invalid learning weights')
        if args.save_every<0:parser.error('save-every must be nonnegative')
        if args.bootstrap_steps<0 or (args.bootstrap_steps and args.warmup<1):parser.error('Bootstrap needs positive warmup')
        if not np.isfinite(args.behavior_weight) or args.behavior_weight<0:parser.error('Invalid behavior weight')
        if not np.isfinite(args.progress_shaping) or args.progress_shaping<0:parser.error('Invalid progress shaping weight')
        if not np.isfinite(args.prep_shaping) or args.prep_shaping<0:parser.error('Invalid prep shaping weight')
        if args.finite_horizon<1:parser.error('finite-horizon must be >= 1')
        if args.resume and args.warmstart:parser.error('Choose resume or warmstart')
        train(args)
    else:
        if args.episodes<1 or args.num_envs<1:parser.error('episodes and num-envs must be positive')
        evaluate(args)


if __name__=='__main__':main()
