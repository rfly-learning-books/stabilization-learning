"""Unified prepare/train/evaluate entry point for A/B/C ablations."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
import so101_multishape as m
from .timing import TimeBank,prepare,load_config,write_json,digest,validate_arm_timing
from .environment import TimedResponseEnv,reached
from .policy import load_actor,Selector

def evaluate(args,bank):
    if args.backend=='arm':validate_arm_timing(bank.config)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new evaluation output directory')
    actor,ck=load_actor(args.checkpoint)
    if args.variant=='C' and ck.get('schema')!='so101_timeaware_ppo_v1':raise ValueError('C requires a time-aware trained checkpoint')
    if args.variant in ['A','B'] and ck.get('schema')=='so101_timeaware_ppo_v1':raise ValueError('A/B must use the original frozen checkpoint, not trained C weights')
    if args.variant=='C' and ck['time_bank_sha256']!=bank.sha256:raise ValueError('C evaluation time bank differs from training')
    if args.variant=='A' and (bank.config.length_scale!=.05 or bank.config.encoder_omega_multiplier!=2 or tuple(bank.config.travels)!=(.0004,.001,.002,.0035)):
        raise ValueError('Unchanged A requires the original coordinate/action conventions')
    for p in actor.parameters():p.requires_grad_(False)
    out.mkdir(parents=True,exist_ok=True);allrows=[]
    initials=m.c.cases(args.seed,args.episodes,extent=.012,angle=.35)
    write_json(out/'protocol.json',dict(variant=args.variant,backend=args.backend,checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=digest(args.checkpoint),time_bank_sha256=bank.sha256,config=asdict(bank.config),seed=args.seed,initials=[q.tolist() for q in initials],shapes=args.shapes))
    for name in args.shapes:
        rows=[]
        with m.install(name):
            policy=Selector(bank,name,actor,args.variant)
            for i,q in enumerate(initials):
                if args.backend=='arm':
                    cfg=m.c.Config(plant='arm',speed=bank.config.transit_speed,max_time=bank.config.max_seconds,max_actions=bank.config.max_actions)
                    video=out/f'{name}_{i:03d}.mp4' if args.video else None
                    r=m.c.rollout(cfg,q,args.seed+i,policy=policy,video=video)
                else:
                    env=TimedResponseEnv(bank,[name],args.seed+i);env.reset(name,q);trace=[]
                    while not env.done:
                        if reached(env.spec,env.q):
                            _,reward,done,info=env.step(0);trace.append(dict(mode=None,travel=0.,reward=reward,**info));continue
                        try:mode,travel,V,recovery=policy.choose(env.q,None if env.mode<0 else env.mode,True,env.pusher)
                        except RuntimeError as ex:env.done=True;env.reason=str(ex);break
                        k=int(np.argmin(abs(np.array(bank.config.travels)-travel)))
                        _,reward,done,info=env.step(mode*len(bank.config.travels)+k)
                        trace.append(dict(mode=mode,travel=travel,reward=reward,**info))
                    r=dict(success=env.reason=='success',reason=env.reason,sim_time_s=env.elapsed,actions=env.actions,repositions=env.repositions,mode_changes=env.switches,transfer_seconds=env.total_transfer,final=env.q.tolist(),trace=trace)
                r.update(shape=name,episode=i,variant=args.variant,backend=args.backend);rows.append(r);allrows.append(r)
                write_json(out/f'{name}.json',dict(initials=[q.tolist() for q in initials],episodes=rows))
                print(name,args.variant,i,r['success'],r['reason'],round(r['sim_time_s'],2),flush=True)
    write_json(out/'summary.json',dict(variant=args.variant,backend=args.backend,episodes=len(allrows),success_rate=sum(r['success'] for r in allrows)/len(allrows),mean_seconds=float(np.mean([r['sim_time_s'] for r in allrows])),note='numeric uses predicted elapsed time; arm uses measured MuJoCo execution time. Do not merge their success rates.'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Build shape transition time tables; optionally calibrate with fresh robot transfers')
    p.add_argument('--shapes',nargs='+',choices=list(m.PARTS),default=list(m.PARTS));p.add_argument('--calibration-root',type=Path,default=m.ROOT/'media/so101_multishape');p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--measure-pairs',type=int,default=0);p.add_argument('--seed',type=int,default=20261005)
    t=sub.add_parser('train',help='C: time-aware PPO on reduced response model')
    t.add_argument('--time-bank',type=Path,required=True);t.add_argument('--shapes',nargs='+');t.add_argument('--checkpoint',type=Path,default=m.CHECKPOINT);t.add_argument('--resume',type=Path);t.add_argument('--output',type=Path,required=True);t.add_argument('--updates',type=int,default=100);t.add_argument('--num-envs',type=int,default=32);t.add_argument('--horizon',type=int,default=64);t.add_argument('--epochs',type=int,default=2);t.add_argument('--minibatch',type=int,default=128);t.add_argument('--learning-rate',type=float,default=1e-4);t.add_argument('--seed',type=int,default=20261006);t.add_argument('--device',default='cpu')
    e=sub.add_parser('evaluate',help='Compare A original / B frozen time-aware / C trained time-aware')
    e.add_argument('--variant',choices=['A','B','C'],required=True);e.add_argument('--backend',choices=['numeric','arm'],default='numeric');e.add_argument('--time-bank',type=Path,required=True);e.add_argument('--shapes',nargs='+');e.add_argument('--checkpoint',type=Path,default=m.CHECKPOINT);e.add_argument('--output',type=Path,required=True);e.add_argument('--episodes',type=int,default=20);e.add_argument('--seed',type=int,default=20261007);e.add_argument('--video',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        if args.measure_pairs<0:parser.error('--measure-pairs must be nonnegative')
        prepare(args.output,args.shapes,args.calibration_root,load_config(args.config),args.measure_pairs,args.seed);return
    bank=TimeBank(args.time_bank);args.shapes=args.shapes or bank.names
    if any(n not in bank.names for n in args.shapes):parser.error('Requested shape absent from time bank')
    if args.command=='train':
        if min(args.updates,args.num_envs,args.horizon,args.epochs,args.minibatch)<1:parser.error('Training counts must be positive')
        from .training import train
        train(args,bank)
    else:
        if args.episodes<1:parser.error('--episodes must be positive')
        if args.video and args.backend!='arm':parser.error('--video requires --backend arm')
        evaluate(args,bank)

if __name__=='__main__':main()
