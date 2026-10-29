"""Record full, measured PhysX MPPI episodes as annotated top-down MP4s."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .planner import plan
from .__main__ import write_json
from dl_push.six_primitives.environment import SixConfig, SixPrimitiveEnv, SPLITS


def frame(spec, record, initial, amount, shape, mode, result=None, task=0):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new('RGB', (1280, 720), '#f4f6fa')
    d = ImageDraw.Draw(im)
    fontpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(fontpath, 22)
    title = ImageFont.truetype(fontpath, 30)
    def point(xy):
        return (int(360+1200*xy[0]), int(370-1200*xy[1]))
    for g in np.arange(-.25, .251, .05):
        d.line([point((g,-.25)), point((g,.25))], fill='#e0e5ed')
        d.line([point((-.25,g)), point((.25,g))], fill='#e0e5ed')
    def body(xy, yaw, color, outline):
        co, si = np.cos(yaw), np.sin(yaw)
        rot = np.array([[co, -si], [si, co]])
        for (cx,cy),(hx,hy) in spec.boxes:
            corners = np.array([[cx-hx,cy-hy],[cx+hx,cy-hy],[cx+hx,cy+hy],[cx-hx,cy+hy]])@rot.T+xy
            d.polygon([point(p) for p in corners], fill=color, outline=outline, width=2)
    directions = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    names = ['+x','-x','+y','-y','CW','CCW']
    label = names[int(task)]
    rotation = int(task) >= 4
    direction = directions[int(task)] if not rotation else np.array([0.,0.])
    target = np.array(initial['xy'])+amount*direction
    target_yaw = initial['yaw'] + ((-amount if int(task)==4 else amount) if rotation else 0.)
    body(target, target_yaw, '#c8eadb', '#419c7c')
    body(record['xy'], record['yaw'], '#526582', '#2c3d56')
    px,py = point(record['pusher'])
    r = .018*1200
    d.ellipse((px-r,py-r,px+r,py+r), fill='#e99b3a', outline='#925817', width=2)
    origin=np.array([.10,-.24]);tip=origin+.07*direction
    p0=point(origin);p1=point(tip)
    d.line([p0,p1], fill='#3b6dce', width=4)
    d.text((p1[0]-18,p1[1]-28), label if not rotation else 'rotation', font=font, fill='#3b6dce')
    d.text((40,22), f'{shape}  |  MPPI + PhysX GPU  |  {label}', font=title, fill='#24344c')
    lines = [f'Time: {record["t"]:.2f} s', f'Reset: {mode}',
             (f'Goal: {label}{amount*1000:.1f} mm' if not rotation
              else f'Goal: {label}{np.degrees(amount):.1f} deg'),
             f'Remaining: {record["g"]*1000:.2f} ' + ('mm' if not rotation else 'rad'),
             f'Side error: {record["side"]*1000:.2f} mm', f'Yaw change: {np.degrees(record["angle"]):.2f} deg',
             f'Object speed: {record["speed"]*1000:.2f} mm/s',
             f'Cumulative position error: {record.get("drift_integral",0.)*1000:.2f} mm s',
             f'Hold: {record["hold"]}/10', 'Green: target | Orange: pusher',
             'Measured physics state rendering', 'No learned actor / no MJX']
    if result is not None:
        lines += [f'Episode success: {result["success"]}', f'Strict success: {result["strict_success"]}']
    for i,line in enumerate(lines):
        d.text((715,95+41*i), line, font=font, fill='#24344c')
    return np.asarray(im)


def measured(env, t, info):
    xy,yaw,_,_,pusher,_ = env._physical()
    drift_sum = float(info.get('translation_error_sum', torch.zeros_like(info['g']))[0])
    return dict(t=t, xy=xy[0].cpu().tolist(), yaw=float(yaw[0]), pusher=pusher[0].cpu().tolist(),
        g=float(info['g'][0]), side=float(info['transverse_error'][0]), angle=float(info['angle_change'][0]),
        speed=float(info['speed'][0]), hold=int(env.hold[0]), drift_sum=drift_sum,
        drift_integral=drift_sum*env.dt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--shapes', nargs='+', default=['T','Square','RectWide','Plus','H'])
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--iterations', type=int, default=2)
    p.add_argument('--steps', type=int, default=240)
    p.add_argument('--seed', type=int, default=20260910)
    p.add_argument('--reset-mode', choices=['fixed_origin','primitive_contact'], default='fixed_origin')
    p.add_argument('--amount', type=float, default=.04)
    p.add_argument('--energy-weight', type=float, default=200.)
    p.add_argument('--q-weight', type=float, default=10.)
    p.add_argument('--q-limit', type=float, default=.25)
    p.add_argument('--warmstart', action='store_true', help='Shift MPPI sequence only; no learned checkpoint')
    p.add_argument('--replan-every', type=int, default=1)
    args=p.parse_args()
    if min(args.samples,args.horizon,args.iterations,args.steps,args.replan_every)<1 or args.samples<2:
        p.error('Positive sizes and samples >= 2 required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    reports=[]
    import imageio.v2 as imageio
    for shape in args.shapes:
        split=next(k for k,v in SPLITS.items() if shape in v)
        env=SixPrimitiveEnv(SixConfig(num_envs=args.samples, fixed_shape=shape, split=split,
            fixed_task=0, task_ids=(0,), reset_mode=args.reset_mode))
        try:
            _,info=env.reset(seed=args.seed, amount=args.amount)
            spec=env.raw._shape_specs[int(env.raw._active_shape_id[0])]
            records=[measured(env,0.,info)];logs=[];actions=[];start=time.monotonic()
            mean=torch.zeros(args.horizon,2,device=env.device) if args.warmstart else None
            selected=None;offset=0
            for step in range(args.steps):
                if selected is None or offset>=min(args.replan_every,args.horizon) or abs(records[-1]['g'])<.01:
                    if args.warmstart and selected is not None:
                        mean=torch.cat((selected[offset:],torch.zeros_like(selected[:offset])))
                    _,selected,_,diagnostic=plan(env,args.horizon,args.iterations,initial_mean=mean,
                        energy_weight=args.energy_weight,q_weight=args.q_weight,q_limit=args.q_limit)
                    offset=0
                env.done[1:]=False  # Only lane 0 is the executed demonstration.
                _,_,term,trunc,info=env.step(selected[offset].expand(env.num_envs,-1))
                records.append(measured(env,(step+1)*env.dt,info))
                actions.append(selected[offset].cpu().tolist());logs.append(dict(**diagnostic,sequence_offset=offset))
                with (out/f'{shape}.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(**records[-1],action=actions[-1],planner=logs[-1]))+'\n')
                offset+=1
                if step%10==0 or bool((term|trunc)[0]):
                    print(json.dumps(dict(shape=shape,step=step+1,elapsed=time.monotonic()-start,**records[-1])),flush=True)
                if bool((term|trunc)[0]):break
            result=dict(shape=shape,success=bool(info['success'][0]),strict_success=bool(info['strict_success'][0]),
                steps=len(actions),wall_seconds=time.monotonic()-start,arguments=vars(args),
                provenance=env.provenance(),frames=len(records),video='measured PhysX state top-down rendering',
                final=records[-1])
            write_json(out/f'{shape}.json',dict(**result,trajectory=records,actions=actions,planner=logs))
            with imageio.get_writer(out/f'{shape}.mp4',fps=20,codec='libx264',quality=8) as writer:
                for record in records:
                    writer.append_data(frame(spec,record,records[0],args.amount,shape,args.reset_mode,result))
            reports.append(result)
            write_json(out/'report.json',reports)
        finally:env.close()


if __name__=='__main__':main()
