"""PPO over full-M-induced candidates; semi-Markov returns use elapsed seconds."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from .environment import TimedResponseEnv,reached
from .policy import CandidateModel,TimeCritic,load_actor
from .timing import write_json,digest

def advantages(rewards,values,next_values,dones,dt,cfg):
    discount=cfg.gamma**(dt/cfg.discount_reference_seconds)
    trace_discount=cfg.gae_lambda**(dt/cfg.discount_reference_seconds)
    adv=torch.zeros_like(rewards);last=torch.zeros_like(rewards[0])
    for t in range(len(rewards)-1,-1,-1):
        alive=(~dones[t]).float()
        delta=rewards[t]+discount[t]*next_values[t]*alive-values[t]
        last=delta+discount[t]*trace_discount[t]*alive*last;adv[t]=last
    return adv,adv+values

def collect(actor,critic,model,envs,horizon):
    rows=[];episodes=[];actor.eval();critic.eval()
    for _ in range(horizon):
        obs=[e.observation() for e in envs];data=model.batch(obs)
        cand=model.candidates(actor,data)
        dist=Categorical(logits=cand['logits'].detach());index=dist.sample()
        with torch.no_grad():value=critic(data,model)
        rewards=[];done=[];dt=[];mask=[]
        for i,env in enumerate(envs):
            mask.append(bool(cand['physical'][i].any()) and not reached(env.spec,env.q))
            if not bool(cand['physical'][i].any()):
                env.done=True;env.reason='no_physical_candidate';r=-model.cfg.failure_penalty;d=True;info={'dt':0.};env.total_reward+=r
            else:_,r,d,info=env.step(int(index[i]))
            rewards.append(r);done.append(d);dt.append(info['dt'])
            if d:
                episodes.append(dict(shape=env.name,success=env.reason=='success',reason=env.reason,seconds=env.elapsed,reward=env.total_reward,repositions=env.repositions))
                env.reset()
        with torch.no_grad():next_value=critic(model.batch([e.observation() for e in envs]),model)
        rows.append(dict(data={k:v.detach() for k,v in data.items()},index=index,logp=dist.log_prob(index),value=value,next_value=next_value,reward=torch.tensor(rewards,device=model.device)*model.cfg.reward_scale,done=torch.tensor(done,device=model.device),dt=torch.tensor(dt,device=model.device),policy=torch.tensor(mask,device=model.device)))
    result={k:torch.stack([r[k] for r in rows]) for k in rows[0] if k!='data'}
    result['data']={k:torch.stack([r['data'][k] for r in rows]) for k in rows[0]['data']}
    result['advantage'],result['return']=advantages(result['reward'],result['value'],result['next_value'],result['done'],result['dt'],model.cfg)
    return result,episodes

def update(actor,critic,optimizer,model,rollout,epochs,minibatch):
    flat={k:v.flatten(0,1) for k,v in rollout.items() if k!='data'};data={k:v.flatten(0,1) for k,v in rollout['data'].items()}
    advantage=flat['advantage'].clone();mask=flat['policy']
    if mask.any():advantage=(advantage-advantage[mask].mean())/advantage[mask].std(unbiased=False).clamp_min(1e-6)
    stats=[];actor.train();critic.train()
    for _ in range(epochs):
        for ids in torch.randperm(len(advantage),device=model.device).split(minibatch):
            batch={k:v[ids] for k,v in data.items()};values=critic(batch,model);critic_loss=(values-flat['return'][ids]).square().mean()
            used=flat['policy'][ids]
            if used.any():
                b={k:v[used] for k,v in batch.items()};cand=model.candidates(actor,b,training=True)
                dist=Categorical(logits=cand['logits']);logratio=dist.log_prob(flat['index'][ids][used])-flat['logp'][ids][used]
                ratio=logratio.exp();adv=advantage[ids][used]
                ppo=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
                best=cand['both'].masked_fill(~cand['physical'],torch.inf).amin(-1)
                existence=torch.relu(best+.001).square().mean()
                _,nearby=actor.conditioned(cand['D']+.01*torch.randn_like(cand['D']),cand['condition'])
                smooth=(nearby-cand['M']).square().mean();entropy=dist.entropy().mean();kl=((ratio-1)-logratio).mean()
            else:ppo=existence=smooth=entropy=kl=values.sum()*0
            loss=ppo+.5*critic_loss+.5*existence+.001*smooth-.002*entropy
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite training gradient')
            optimizer.step();stats.append({k:float(v.detach()) for k,v in dict(loss=loss,ppo=ppo,critic=critic_loss,existence=existence,smoothness=smooth,entropy=entropy,kl=kl).items()})
        if stats and stats[-1]['kl']>.03:break
    return {k:float(np.mean([s[k] for s in stats])) for k in stats[0]}

def train(args,bank):
    out=Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('Use a new output directory, or --resume explicitly')
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if args.device.startswith('cuda'):torch.cuda.manual_seed_all(args.seed)
    model=CandidateModel(bank,args.device);actor,ck=load_actor(args.resume or args.checkpoint,args.device)
    for p in actor.parameters():p.requires_grad_(True)
    critic=TimeCritic(actor.cfg.hidden).to(args.device)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=args.learning_rate)
    envs=[TimedResponseEnv(bank,args.shapes,args.seed+i) for i in range(args.num_envs)];start=0
    run=dict(seed=args.seed,shapes=args.shapes,num_envs=args.num_envs,horizon=args.horizon,epochs=args.epochs,minibatch=args.minibatch,learning_rate=args.learning_rate,device=args.device)
    source_sha={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    if args.resume:
        if ck.get('schema')!='so101_timeaware_ppo_v1' or ck['time_bank_sha256']!=bank.sha256:raise ValueError('Incompatible resume bank/schema')
        if ck['run']!=run or ck['source_sha256']!=source_sha:raise ValueError('Resume must preserve configuration, seed, device, and source; --updates is total target')
        critic.load_state_dict(ck['critic']);optimizer.load_state_dict(ck['optimizer']);start=ck['update']
        for e,s in zip(envs,ck['environments']):e.load_state_dict(s)
        torch.set_rng_state(ck['torch_rng'].cpu())
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
    write_json(out/'run.json',dict(run=run,time_config=asdict(bank.config),time_bank=str(bank.path.resolve()),time_bank_sha256=bank.sha256,initial_checkpoint=str(args.checkpoint),source_sha256=source_sha,scope='C: PPO in timed reduced response model. Robot simulation evaluation is separate.'))
    initial={k:v.detach().clone() for k,v in actor.state_dict().items()}
    for step in range(start,args.updates):
        rollout,episodes=collect(actor,critic,model,envs,args.horizon)
        stats=update(actor,critic,optimizer,model,rollout,args.epochs,args.minibatch)
        stats.update(update=step+1,transitions=args.num_envs*args.horizon,simulated_seconds=float(rollout['dt'].sum()),completed_episodes=len(episodes),success_rate=None if not episodes else sum(e['success'] for e in episodes)/len(episodes))
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
        with (out/'episodes.jsonl').open('a') as f:
            for e in episodes:f.write(json.dumps(dict(update=step+1,**e))+'\n')
        state=dict(schema='so101_timeaware_ppo_v1',actor=actor.state_dict(),cfg=asdict(actor.cfg),critic=critic.state_dict(),optimizer=optimizer.state_dict(),time_config=asdict(bank.config),time_bank_sha256=bank.sha256,run=run,source_sha256=source_sha,update=step+1,environments=[e.state_dict() for e in envs],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [])
        torch.save(state,out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt')
        print(json.dumps(stats),flush=True)
    changed=max(float((v-initial[k]).abs().max()) for k,v in actor.state_dict().items())
    write_json(out/'verification.json',dict(updates_executed=max(0,args.updates-start),max_actor_parameter_change=changed,note='Parameter updates prove the training path runs, not policy quality.'))
