"""Continuous-contact MPPI recordings; live PhysX scene is never used for search."""
import argparse
import json
import time
from pathlib import Path
import torch
import imageio.v2 as imageio
from .demo import frame, measured
from .planner import snapshot,restore
from .contact_planner import ContactMPPI
from .__main__ import write_json
from dl_push.six_primitives.environment import SixConfig,SixPrimitiveEnv,SPLITS,TASKS


def physical12(env):
    actor=env.raw.tees[int(env.raw._active_shape_id[0])]
    _,_,_,omega,pusher,pusher_velocity=env._physical()
    return torch.cat((actor.pose.p[0],env.lifted_yaw[:1],actor.get_linear_velocity()[0],
                      omega[:1],pusher[0],pusher_velocity[0])).detach().cpu().clone()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--shape',default='Square')
    p.add_argument('--task',type=int,choices=range(6),default=0,help='0:+x, 1:-x, 2:+y, 3:-y, 4:CW, 5:CCW')
    p.add_argument('--samples',type=int,default=64);p.add_argument('--horizon',type=int,default=12)
    p.add_argument('--iterations',type=int,default=2);p.add_argument('--steps',type=int,default=180)
    p.add_argument('--push-min-forward',type=float,default=.35,
                   help='Minimum normalized forward speed in push candidates; use a separate run for stiction ablation')
    p.add_argument('--rotation-inward',type=float,default=.10,
                   help='Normalized inward component for rotation contact; only used by tasks 4/5')
    p.add_argument('--rotation-drift-weight',type=float,default=100.,
                   help='Weight on cumulative position drift in rotation search')
    p.add_argument('--seed',type=int,default=20260910);p.add_argument('--amount',type=float)
    p.add_argument('--save-transitions',action='store_true',help='Save measured live transitions for route-B datasets')
    p.add_argument('--no-video',action='store_true',help='Skip MP4 encoding during dataset collection')
    args=p.parse_args()
    if args.amount is None:
        args.amount=.60 if args.task>=4 else .04
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    split=next(k for k,v in SPLITS.items() if args.shape in v)
    config=dict(fixed_shape=args.shape,split=split,fixed_task=args.task,task_ids=(args.task,),reset_mode='fixed_origin')
    search=SixPrimitiveEnv(SixConfig(num_envs=args.samples,**config));live=None
    try:
        live=SixPrimitiveEnv(SixConfig(num_envs=1,**config))
        live.reset(seed=args.seed,amount=args.amount);search.reset(seed=args.seed,amount=args.amount)
        root=snapshot(search,lane=0)
        def first(value):
            if isinstance(value,dict):return {k:first(v) for k,v in value.items()}
            if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==args.samples:return value[:1].clone()
            return value
        restore(live,first(root))
        initial=measured(live,0.,live.metrics())
        initial['gap']=float(live.contact_gap()[0]*.1)
        write_json(out/'initial.json',dict(shape=args.shape,seed=args.seed,**initial))
        controller=ContactMPPI(live,search,args.horizon,args.iterations,
                                push_min=args.push_min_forward,
                                rotation_inward=args.rotation_inward,
                                rotation_drift_weight=args.rotation_drift_weight)
        info=live.metrics();records=[measured(live,0,info)];actions=[];logs=[];failure=None
        records[0].update(phase='approach',contact_force=0.,gap=float(live.contact_gap()[0]*.1))
        spec=live.raw._shape_specs[int(live.raw._active_shape_id[0])];start=time.monotonic();transitions=[]
        for step in range(args.steps):
            before=live.observe() if args.save_transitions else None
            state_before=physical12(live) if args.save_transitions else None
            previous=controller.previous.clone()
            try:action,diagnostic=controller()
            except RuntimeError as exc:
                failure=str(exc);print(failure,flush=True);break
            nxt,_,term,trunc,info=live.step(action[None])
            if args.save_transitions:
                transitions.append(dict(obs={k:v[0].cpu() for k,v in before.items() if k!='shape'},
                    next_obs={k:v[0].cpu() for k,v in nxt.items() if k!='shape'},action=action.cpu(),
                    previous_action=previous.cpu(),phase=diagnostic['phase'],
                    state12=state_before,next_state12=physical12(live),
                    d=info['d_target'][0].cpu(),q=info['q_cost'][0].cpu(),
                    energy_before=info['energy_before'][0].cpu(),energy_after=info['energy'][0].cpu(),
                    contact_force=info['contact_force'][0].cpu(),gap=(live.contact_gap()[0]*.1).cpu(),
                    terminated=bool(term[0]),truncated=bool(trunc[0]),accepted=diagnostic['accepted']))
            row=measured(live,(step+1)*live.dt,info)
            row.update(phase=diagnostic['phase'],contact_force=float(info['contact_force'][0]),gap=float(live.contact_gap()[0]*.1),omega=float(info['omega'][0]))
            records.append(row);actions.append(action.cpu().tolist());logs.append(diagnostic)
            with (out/f'{args.shape}.jsonl').open('a') as stream:stream.write(json.dumps(dict(**row,action=actions[-1],planner=diagnostic))+'\n')
            if step%10==0 or bool((term|trunc)[0]):print(json.dumps(dict(step=step+1,elapsed=time.monotonic()-start,**row)),flush=True)
            if bool((term|trunc)[0]):break
        result=dict(shape=args.shape,success=bool(info.get('success',torch.tensor([False]))[0]),
            strict_success=bool(info.get('strict_success',torch.tensor([False]))[0]),steps=len(actions),
            wall_seconds=time.monotonic()-start,arguments=vars(args),provenance=live.provenance(),
            task=args.task,task_name=TASKS[args.task],
            objective=('position_progress_with_attitude_hold' if args.task<4 else
                       'attitude_progress_with_cumulative_position_drift_minimization'),
            frames=len(records),video='measured PhysX state top-down rendering',final=records[-1],
            failure=failure,method='continuous_contact_mppi_v1',approach_path=[v.tolist() for v in controller.path])
        write_json(out/f'{args.shape}.json',dict(**result,trajectory=records,actions=actions,planner=logs))
        if args.save_transitions:
            torch.save(dict(schema='physx_continuous_live_v1',route='planner_guided',from_scratch=False,
                shape=live.observe()['shape'][0].cpu().to(torch.uint8),transitions=transitions,
                config=result['provenance']['config'],seed=args.seed,success=result['success'],
                strict_success=result['strict_success'],failure=failure,
                control_dt=.05,velocity_scale_m_s=.30,
                state12_names=['x','y','z','yaw_lifted','vx','vy','vz','omega','px','py','pvx','pvy'],
                supervision='Only executed live one-step labels; no H-step credit assigned to first action'),out/'transitions.pt')
        if not args.no_video:
            with imageio.get_writer(out/f'{args.shape}.mp4',fps=20,codec='libx264',quality=8) as writer:
                for row in records:writer.append_data(frame(spec,row,records[0],args.amount,args.shape,'fixed_origin',result,args.task))
        write_json(out/'report.json',[result])
    finally:
        if live is not None:live.close()
        search.close()


if __name__=='__main__':main()
