"""Reproducible PPO for a full-M-induced categorical policy.

Exploration uses soft candidate logits; hard persistence/recovery is evaluated
separately. Forced stop/failure samples are excluded from the PPO policy loss.
No-action shaping is frozen at collection and paired with explicit feasibility
loss; this composite objective is not an unbiased fixed-reward policy gradient.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import copy
import hashlib
import json
import random
import signal
import time
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .full_m_control import (FullMConfig,FullMActor,ModeEncoder,ModeBank,FullMControlEnv,
                             coordinates,candidates,smoothness_loss,hard_choice)
from .numeric_env import DEFAULT_LIBRARY


def write_json(path,record):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


class Critic(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;h=cfg.hidden;self.encoder=ModeEncoder(h)
        self.head=nn.Sequential(nn.Linear(2*h+25,2*h),nn.SiLU(),nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,1))
    def forward(self,data,bank):
        ctx=torch.cat([data['e'],data['q'][:,:2]/self.cfg.workspace_m,
            data['target'][:,:2]/self.cfg.workspace_m,
            torch.sin(data['target'][:,2:3]),torch.cos(data['target'][:,2:3]),
            torch.nn.functional.one_hot(data['mode']+1,13).to(data['e'].dtype),
            data['remaining'][:,None]/self.cfg.max_steps,data['recovery'][:,None],data['hold'][:,None]],-1)
        cond=self.encoder(bank.C,bank.valid,self.cfg.length_scale)[data['shape']]
        return self.head(torch.cat([ctx,cond],-1)).squeeze(-1)


def advantages(reward,values,done,active,gamma=1.,gae_lambda=.95):
    """Finite 100-step TASK deadline: no bootstrap beyond deadline or failure."""
    adv=torch.zeros_like(reward);next_value=torch.zeros_like(values[0]);last=torch.zeros_like(next_value)
    for t in range(len(reward)-1,-1,-1):
        continuation=(~done[t]).to(values.dtype)
        delta=reward[t]+gamma*next_value*continuation-values[t]
        last=(delta+gamma*gae_lambda*continuation*last)*active[t]
        adv[t]=last;next_value=values[t]
    return adv,adv+values


def collect(actor,critic,control,seed):
    cfg=control.cfg;bank=control.bank;data=control.reset(seed=seed);records=[]
    actor.eval();critic.eval();started=time.perf_counter()
    for _ in range(cfg.max_steps):
        active=~control.env.done.copy()
        if not active.any():break
        with torch.no_grad():
            cand=candidates(actor,data,bank,cfg)
            dist=Categorical(logits=cand['logits']);index=dist.sample();logp=dist.log_prob(index)
            value=critic(data,bank)
        has_physical=cand['physical'].any(-1).cpu().numpy()
        no_action=~cand['feasible'].any(-1).cpu().numpy()
        goal=control.at_goal();policy_mask=active&~goal&has_physical
        mode=cand['modes'][index].cpu().numpy()
        # Canonical .6 endpoint removes only the float32 representation error.
        speed=np.minimum(cand['speeds'][index].cpu().numpy().astype(float),cfg.u_max)
        speed[~policy_mask]=0.
        old_data={k:v.detach() for k,v in data.items()}
        data,reward,done,used=control.advance(mode,speed,no_action,forced_failure=~has_physical&~goal)
        records.append(dict(data=old_data,index=index.detach(),logp=logp.detach(),value=value.detach(),
            reward=torch.as_tensor(reward,device=bank.device,dtype=bank.dtype),
            done=torch.as_tensor(done,device=bank.device),active=torch.as_tensor(used,device=bank.device),
            policy=torch.as_tensor(policy_mask,device=bank.device),
            M=cand['M'].detach(),L=cand['L'].detach(),
            action=torch.as_tensor(np.c_[mode,speed],device=bank.device,dtype=bank.dtype),
            no_action=torch.as_tensor(no_action&used,device=bank.device),
            selected_residual=cand['both'].detach().gather(1,index[:,None]).squeeze(1)))
    arrays={k:torch.stack([r[k] for r in records]) for k in records[0] if k!='data'}
    arrays['data']={k:torch.stack([r['data'][k] for r in records]) for k in data}
    arrays['advantage'],arrays['return']=advantages(cfg.rl_reward_scale*arrays['reward'],arrays['value'],arrays['done'],arrays['active'])
    stats=dict(transitions=int(arrays['active'].sum()),sampled_policy_steps=int(arrays['policy'].sum()),
        seconds=time.perf_counter()-started,mean_return=float(arrays['reward'].sum(0).mean()),
        success_rate=float((control.env.reason=='SUCCESS').mean()))
    return arrays,stats


def update_ppo(actor,critic,optimizer,rollout,bank,cfg,epochs=2,minibatch=128):
    flat={k:v.flatten(0,1) for k,v in rollout.items() if k!='data'}
    data={k:v.flatten(0,1) for k,v in rollout['data'].items()}
    active=torch.nonzero(flat['active'],as_tuple=False).squeeze(1)
    pm=flat['policy'];adv=flat['advantage'].clone()
    if pm.any():adv=(adv-adv[pm].mean())/adv[pm].std(unbiased=False).clamp_min(1e-6)
    else:adv.zero_()
    rows=[];actor.train();critic.train()
    for _ in range(epochs):
        order=active[torch.randperm(len(active),device=active.device)]
        for ids in order.split(minibatch):
            batch={k:v[ids] for k,v in data.items()};mask=flat['policy'][ids]
            value=critic(batch,bank);critic_loss=(value-flat['return'][ids]).square().mean()
            if mask.any():
                policy_data={k:v[mask] for k,v in batch.items()}
                cand=candidates(actor,policy_data,bank,cfg,training=True)
                distribution=Categorical(logits=cand['logits'])
                new_logp=distribution.log_prob(flat['index'][ids][mask])
                logratio=new_logp-flat['logp'][ids][mask];ratio=logratio.exp()
                A=adv[ids][mask]
                policy_loss=-torch.minimum(ratio*A,ratio.clamp(.8,1.2)*A).mean()
                best=cand['both'].masked_fill(~cand['physical'],torch.inf).amin(-1)
                exist=torch.relu(best+cfg.training_margin).square().mean()
                smooth=smoothness_loss(actor,policy_data,cand['condition'])
                entropy=distribution.entropy().mean()
                kl=((ratio-1)-logratio).mean()
            else:
                policy_loss=exist=smooth=entropy=kl=value.sum()*0
            loss=policy_loss+.5*critic_loss+.5*exist+.001*smooth-.002*entropy
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss; refusing to mark run complete')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            parameters=list(actor.parameters())+list(critic.parameters())
            norm=nn.utils.clip_grad_norm_(parameters,1.)
            if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            rows.append(dict(loss=float(loss.detach()),policy_loss=float(policy_loss.detach()),
                critic_loss=float(critic_loss.detach()),existence=float(exist.detach()),
                smoothness=float(smooth.detach()),entropy=float(entropy.detach()),kl=float(kl.detach())))
        if rows and rows[-1]['kl']>.03:break
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}


def save_rollout(path,rollout):
    values={k:v.detach().cpu().numpy() for k,v in rollout.items() if k!='data'}
    values.update({k:v.cpu().numpy() for k,v in rollout['data'].items()})
    np.savez_compressed(path,**values)


def evaluate(actor,bank,cfg,shape_names,seed=20260913,output=None,video=False):
    """Deterministic hard controller, same reset bank across model seeds."""
    ctrl=FullMControlEnv(bank,cfg,shape_names,len(shape_names));data=ctrl.reset(seed=seed)
    # Always preserve a fixed near-COM / 10-degree T regression case.
    if 'T' in shape_names:
        i=shape_names.index('T');q=ctrl.env.q.copy();target=ctrl.env.target.copy()
        target[i]=[.25,.25,0.];q[i]=[.251959141309,.249951905780,np.deg2rad(10.)]
        data=ctrl.reset(seed=seed,options={'pose':q,'target':target})
    actor.eval();trace=[];frames=[];started=time.perf_counter()
    if output is not None:Path(output).mkdir(parents=True,exist_ok=True)
    for _ in range(cfg.max_steps):
        if ctrl.env.done.all():break
        with torch.no_grad():
            cand=candidates(actor,data,bank,cfg,speed_points=cfg.eval_speed_points)
            goal=torch.as_tensor(ctrl.at_goal(),device=bank.device)
            m,u,recovery,fail=hard_choice(cand,data,cfg,torch.as_tensor(ctrl.recovery,device=bank.device),goal)
        no_action=~cand['feasible'].any(-1).cpu().numpy()
        row={k:v.detach().cpu().numpy() for k,v in data.items()}
        row.update({k:cand[k].detach().cpu().numpy() for k in ['M','L','V','gradient','beta','flow','endpoint','feasible']})
        row.update(action=np.c_[m.cpu().numpy(),u.cpu().numpy()],active=~ctrl.env.done.copy(),
                   recovery_active=recovery.cpu().numpy(),grid_no_action=no_action)
        speed=np.minimum(u.cpu().numpy().astype(float),cfg.u_max)
        data,r,done,active=ctrl.advance(m.cpu().numpy(),speed,no_action,recovery.cpu().numpy(),fail.cpu().numpy())
        row.update(reward=r,pose_next=ctrl.env.q.copy(),coverage=ctrl.info['coverage'].copy())
        trace.append(row)
        if video:
            from PIL import Image,ImageDraw
            ims=ctrl.env.render();cols=min(8,len(ims));rows=(len(ims)+cols-1)//cols
            canvas=Image.new('RGB',(cols*256,rows*256),'white')
            for i,im in enumerate(ims):
                tile=Image.fromarray(im);draw=ImageDraw.Draw(tile)
                draw.text((4,235),f'm={ctrl.env.mode[i]} R={int(ctrl.recovery[i])} u={speed[i]:.3f}',fill='black')
                canvas.paste(tile,((i%cols)*256,(i//cols)*256))
            frames.append(np.asarray(canvas))
    success=ctrl.env.reason=='SUCCESS';err=ctrl.obs['error'];E=(err[:,:2]**2).sum(-1)+(1-np.cos(err[:,2]))
    report=dict(scope='hard numeric evaluation, no contact simulation',codec=cfg.codec,shapes=shape_names,
        seed=seed,success_rate=float(success.mean()),failure_error=float(np.mean(E*~success)),
        mean_switches=float(ctrl.env.switches.mean()),mean_no_action_steps=float(ctrl.no_action_steps.mean()),
        mean_cumulative_error=float(ctrl.cost_error.mean()),elapsed_seconds=time.perf_counter()-started,
        episodes=[dict(shape=n,success=bool(success[i]),reason=str(ctrl.env.reason[i]),steps=int(ctrl.env.steps[i]),
            position_error_m=float(np.linalg.norm(err[i,:2])*cfg.length_scale),yaw_error_deg=float(np.rad2deg(err[i,2])),
            coverage=float(ctrl.info['coverage'][i]),switches=int(ctrl.env.switches[i]),
            no_action_steps=int(ctrl.no_action_steps[i]),recovery_steps=int(ctrl.recovery_steps[i]),
            max_position_error_m=float(ctrl.max_position[i])) for i,n in enumerate(shape_names)])
    if output is not None:
        path=Path(output);write_json(path/'report.json',report)
        np.savez_compressed(path/'trajectory.npz',**{k:np.stack([r[k] for r in trace]) for k in trace[0]})
        if video and frames:
            import imageio.v2 as imageio
            Image.fromarray(frames[0]).save(path/'overview.jpg',quality=95)
            with imageio.get_writer(path/'hard_numeric.mp4',fps=max(1,round(1/cfg.dt)),codec='libx264',quality=7) as w:
                for frame in frames:w.append_data(frame)
    ctrl.env.close();return report


def selection_key(report):
    return (report['success_rate'],-report['failure_error'],-report['mean_switches'],
            -report['mean_no_action_steps'],-report['mean_cumulative_error'])


def rng_state():
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=[state[0],state[1].tolist(),state[2],state[3],state[4]],
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path,actor,critic,optimizer,bank,cfg,update,run_config,best_key=None):
    record=dict(schema=1,codec=cfg.codec,cfg=asdict(cfg),actor=actor.state_dict(),critic=critic.state_dict(),
        optimizer=optimizer.state_dict(),next_update=update,run_config=run_config,
        fingerprints=bank.fingerprints,rng=rng_state(),best_key=best_key,source_sha256=source_hashes())
    temp=Path(path).with_suffix('.tmp');torch.save(record,temp);temp.replace(path)


STOP=False
def stop_signal(signum,frame):
    global STOP
    STOP=True


def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('full_m_control.py'),Path(__file__).with_name('numeric_env.py')]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train');train.add_argument('--output',required=True);train.add_argument('--updates',type=int,default=200)
    train.add_argument('--seed',type=int,default=0);train.add_argument('--device',default='cuda')
    train.add_argument('--num-envs',type=int,default=32);train.add_argument('--epochs',type=int,default=2)
    train.add_argument('--minibatch',type=int,default=128);train.add_argument('--eval-every',type=int,default=10)
    train.add_argument('--max-seconds',type=float,default=21600);train.add_argument('--resume')
    train.add_argument('--train-shapes',nargs='+');train.add_argument('--eval-shapes',nargs='+')
    train.add_argument('--no-final-video',action='store_true')
    test=sub.add_parser('evaluate');test.add_argument('--checkpoint',required=True);test.add_argument('--output',required=True)
    test.add_argument('--device',default='cpu');test.add_argument('--shapes',nargs='+');test.add_argument('--seed',type=int,default=20260913)
    test.add_argument('--video',action='store_true')
    for p in (train,test):p.add_argument('--library-root',default=str(DEFAULT_LIBRARY))
    args=parser.parse_args();torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; explicitly use --device cpu')
    bank=ModeBank(args.library_root,args.device)
    if args.command=='evaluate':
        ck=torch.load(args.checkpoint,map_location=args.device,weights_only=True);cfg=FullMConfig(**ck['cfg'])
        if ck['fingerprints']!=bank.fingerprints:raise ValueError('Library fingerprint mismatch')
        if ck.get('source_sha256')!=source_hashes():raise ValueError('Source changed since checkpoint; reproduce with its original code')
        if Path(args.output).exists():raise FileExistsError(args.output)
        actor=FullMActor(cfg).to(args.device);actor.load_state_dict(ck['actor'])
        report=evaluate(actor,bank,cfg,args.shapes or ck['run_config']['eval_shapes'],args.seed,args.output,args.video)
        print(json.dumps(report,ensure_ascii=False));return
    if min(args.updates,args.num_envs,args.epochs,args.minibatch,args.eval_every,args.max_seconds)<=0:raise ValueError('Invalid training count/budget')
    out=Path(args.output).resolve()
    if out.exists() and not args.resume:raise FileExistsError(out)
    if args.resume and (Path(args.resume).resolve()!=out/'last.pt' or not (out/'manifest.json').exists()):
        raise ValueError('Resume requires the original run directory and its last.pt; use a new run for other changes')
    out.mkdir(parents=True,exist_ok=True)
    for p in ['rollouts','evaluations']:(out/p).mkdir(exist_ok=True)
    cfg=FullMConfig();train_shapes=args.train_shapes or bank.names;eval_shapes=args.eval_shapes or sorted(set(train_shapes))
    if not set(train_shapes+eval_shapes)<=set(bank.names):raise ValueError('Unknown shape')
    lanes=[train_shapes[i%len(train_shapes)] for i in range(args.num_envs)]
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    actor=FullMActor(cfg).to(args.device);critic=Critic(cfg).to(args.device)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=2e-4)
    run_config=dict(seed=args.seed,train_shapes=train_shapes,eval_shapes=eval_shapes,num_envs=args.num_envs,
                    epochs=args.epochs,minibatch=args.minibatch,learning_rate=2e-4,gamma=1.,gae_lambda=.95)
    start_update=0;best_key=None
    if args.resume:
        ck=torch.load(args.resume,map_location=args.device,weights_only=True)
        if ck['cfg']!=asdict(cfg) or ck['fingerprints']!=bank.fingerprints or ck['run_config']!=run_config:
            raise ValueError('Resume config/split/library mismatch')
        if ck.get('source_sha256')!=source_hashes():raise ValueError('Resume source mismatch')
        actor.load_state_dict(ck['actor']);critic.load_state_dict(ck['critic']);optimizer.load_state_dict(ck['optimizer'])
        start_update=ck['next_update'];best_key=ck['best_key'];restore_rng(ck['rng'])
        if args.updates<=start_update:raise ValueError('--updates must exceed the resumed next_update; use evaluate for inspection')
    manifest=dict(config=asdict(cfg),run_config=run_config,torch=str(torch.__version__),device=args.device,
        source_sha256=source_hashes(),fingerprints=bank.fingerprints,
        protocol='soft PPO collection; separate hard persistence/recovery evaluation; CPU geometry, torch batched candidates')
    write_json(out/'manifest.json',manifest);write_json(out/'status.json',dict(status='RUNNING',next_update=start_update))
    ctrl=FullMControlEnv(bank,cfg,lanes,args.num_envs);start=time.perf_counter();next_update=start_update
    try:
        for update in range(start_update,args.updates):
            if STOP or time.perf_counter()-start>=args.max_seconds:break
            rollout,stats=collect(actor,critic,ctrl,args.seed*100000+update)
            save_rollout(out/'rollouts'/f'update_{update:05d}.npz',rollout)
            losses=update_ppo(actor,critic,optimizer,rollout,bank,cfg,args.epochs,args.minibatch)
            next_update=update+1;record=dict(update=next_update,collection=stats,losses=losses,elapsed_seconds=time.perf_counter()-start)
            if next_update%args.eval_every==0 or next_update==args.updates:
                report=evaluate(actor,bank,cfg,eval_shapes,output=out/'evaluations'/f'update_{next_update:05d}')
                record['hard_eval']={k:report[k] for k in ('success_rate','failure_error','mean_switches','mean_no_action_steps')}
                key=selection_key(report)
                if best_key is None or tuple(key)>tuple(best_key):
                    best_key=key;save_checkpoint(out/'best.pt',actor,critic,optimizer,bank,cfg,next_update,run_config,best_key)
            save_checkpoint(out/'last.pt',actor,critic,optimizer,bank,cfg,next_update,run_config,best_key)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            write_json(out/'status.json',dict(status='RUNNING',next_update=next_update,elapsed_seconds=time.perf_counter()-start))
            print(json.dumps(record),flush=True)
        status='COMPLETE' if next_update>=args.updates else ('INTERRUPTED' if STOP else 'BUDGET_STOP')
        if not (out/'last.pt').exists():save_checkpoint(out/'last.pt',actor,critic,optimizer,bank,cfg,next_update,run_config,best_key)
        if status=='COMPLETE' and not args.no_final_video:
            ck=torch.load(out/'best.pt',map_location=args.device,weights_only=True)
            actor.load_state_dict(ck['actor']);evaluate(actor,bank,cfg,eval_shapes,output=out/'final_best',video=True)
        write_json(out/'status.json',dict(status=status,next_update=next_update,elapsed_seconds=time.perf_counter()-start))
    except BaseException as exc:
        write_json(out/'status.json',dict(status='FAILED',next_update=next_update,error=repr(exc)))
        raise
    finally:ctrl.env.close()


if __name__=='__main__':
    signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal);main()
