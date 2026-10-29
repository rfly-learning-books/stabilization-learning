"""Diagnostic-only separation of MJWarp RGB quantization and motion aliasing.

get_rgb() returns unpacked 8-bit colors, so renderer.rgb_float is NOT an
unquantized render.  Read aa_accum before _aa_resolve's uint32 packing instead.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import json
import sys
import time
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[2]
for folder in (PROJECT,PROJECT/'stabilization-gym',PROJECT/'stabilization-learning-control'):
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))

from stabilization_gym.tasks_gpu_rgb.environment import rotation_z
from stabilization_gym.tasks_gpu_rgb.geometry import GPURGBGeometry
from extensions.rgb_diagnostic_v5.derivative_geometry import vector_metrics,describe,json_ready


class FloatGeometry:
    """Same downsampling/Sobel/geometry formulas, with diagnostic float input."""
    def __init__(self, original): self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def _raw(self,images):
        if not images.is_floating_point() or images.shape[1:]!=(self.size,self.size,3):
            raise ValueError('Expected diagnostic B,H,W,3 float image')
        return images.permute(0,3,1,2)
    def encode_images(self,images):return self._down(self._raw(images))
    def tangents(self,images,depth,rotation):return GPURGBGeometry.tangents(self,images,depth,rotation)


@torch.no_grad()
def render_pair(env,position,yaw):
    rgb=env._render(position,yaw)
    size=env.common.SIZE
    continuous=torch.empty((env.num_envs,size,size,3),device=env.device)
    for indices,renderer in env._groups:
        rc=renderer.context
        n=rc.samples_per_pixel**2
        if n<=1:raise ValueError('Pre-quantization accumulator requires supersampling')
        address=int(rc.rgb_adr.numpy()[0])
        accum=renderer.wp.to_torch(rc.aa_accum)
        continuous[indices]=((accum[:,address:address+size*size]/n)/255.).reshape(-1,size,size,3).clamp(0,1)
    return rgb,continuous


def perturb_pair(env,axis,h):
    position,yaw=env.position,env.yaw
    if axis<3:
        delta=(rotation_z(yaw)@env.body0)[:,:,axis]*env.scale[axis]*h
        return render_pair(env,position+delta,yaw),render_pair(env,position-delta,yaw)
    return render_pair(env,position,yaw+env.scale[3]*h),render_pair(env,position,yaw-env.scale[3]*h)


@torch.no_grad()
def repack_accumulator(env):
    """Repeat _aa_resolve before division by 255, avoiding round-trip error."""
    size=env.common.SIZE
    packed=torch.empty((env.num_envs,size,size,3),device=env.device,dtype=torch.uint8)
    for indices,renderer in env._groups:
        rc=renderer.context;address=int(rc.rgb_adr.numpy()[0])
        accum=renderer.wp.to_torch(rc.aa_accum)
        packed[indices]=(accum[:,address:address+size*size]*(1./(rc.samples_per_pixel**2))).clamp(0,255).to(torch.uint8).reshape(-1,size,size,3)
    return packed


def run(ctx,folder):
    from extensions.rgb_diagnostic_v5.common import write_json
    folder.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter();steps=(.00075,.0025,.01,.04)
    by_scene=defaultdict(list)
    for case in ctx.train_cases:by_scene[case['scene_id']].append(case)
    cases=[]
    # Predetermined by scene and goal indices; no measurement-based selection.
    for scene_index,(scene_id,population) in enumerate(by_scene.items()):
        for goal in range(4):
            difficulty=('near','medium','large')[(scene_index+goal)%3]
            cases.append(next(c for c in population if c['goal_id']==goal and c['difficulty']==difficulty))
    assert len(cases)==32
    chosen=defaultdict(list)
    for case in cases:chosen[case['scene_id']].append(case)
    geometry=ctx.geometry;float_geometry=FloatGeometry(geometry);features=ctx.learner.features
    features.eval()
    for parameter in features.parameters():parameter.requires_grad_(False)
    rows=[];repack=[];raw=defaultdict(list)
    for scene_id,batch in chosen.items():
        ctx.guard();env=ctx.make_env(batch)
        try:
            env.reset(seed=91082,options=dict(offsets=np.array([c['offset'] for c in batch]),goal_ids=[c['goal_id'] for c in batch]))
            u0,f0=render_pair(env,env.position,env.yaw)
            # _aa_resolve converts clamped color to int, hence floor not round.
            packed=repack_accumulator(env)
            repack.append(dict(scene_id=scene_id,max_uint8_difference=int((packed.int()-u0.int()).abs().max()),
                               unequal_fraction=float((packed!=u0).float().mean())))
            depth=geometry.predict_depth(u0)
            xu,bu=geometry.tangents(u0,depth,env.rotation)
            xf,bf=float_geometry.tangents(f0,depth,env.rotation)
            ju=features.interaction_jvp(xu,bu,create_graph=False).transpose(1,2)
            jf=features.interaction_jvp(xf,bf,create_graph=False).transpose(1,2)
            jdu=[];jdf=[]
            for h in steps:
                cu=[];cf=[]
                for axis in range(4):
                    (up,fp),(um,fm)=perturb_pair(env,axis,h)
                    xup,xum=geometry.encode_images(up),geometry.encode_images(um)
                    xfp,xfm=float_geometry.encode_images(fp),float_geometry.encode_images(fm)
                    with torch.no_grad():
                        zup,zum,zfp,zfm=features.encode(torch.cat((xup,xum,xfp,xfm))).chunk(4)
                    du=(zup-zum)/(2*h);df=(zfp-zfm)/(2*h)
                    cu.append(du);cf.append(df)
                    for name,a,b in [('J_uint_vs_FD_uint',ju[:,:,axis],du),('J_uint_vs_FD_float',ju[:,:,axis],df),
                                      ('J_float_vs_FD_float',jf[:,:,axis],df),('FD_float_vs_FD_uint',df,du),
                                      ('B_uint_vs_FD_uint',bu[:,axis],(xup-xum)/(2*h)),
                                      ('B_float_vs_FD_float',bf[:,axis],(xfp-xfm)/(2*h))]:
                        stats=vector_metrics(a.cpu().numpy(),b.cpu().numpy())
                        for lane,case in enumerate(batch):
                            rows.append(dict(case_id=case['case_id'],difficulty=case['difficulty'],comparison=name,step=h,axis=axis,
                                **{k:(bool(v[lane]) if k=='valid' else float(v[lane])) for k,v in stats.items()}))
                jdu.append(torch.stack(cu,-1));jdf.append(torch.stack(cf,-1))
            raw['J_uint'].append(ju.cpu().numpy());raw['J_float'].append(jf.cpu().numpy())
            raw['FD_uint'].append(torch.stack(jdu,1).cpu().numpy());raw['FD_float'].append(torch.stack(jdf,1).cpu().numpy())
            ctx.event('quantization_scene_complete',scene_id=scene_id,seconds=time.perf_counter()-began)
        finally:env.close()
    grouped=defaultdict(list)
    for row in rows:
        for stratum in ('all',row['difficulty'],'axis_'+str(row['axis'])):
            grouped[(row['comparison'],row['step'],stratum)].append(row)
    summary=[]
    for (comparison,h,stratum),group in grouped.items():
        valid=[r for r in group if r['valid']]
        summary.append(dict(comparison=comparison,step=h,stratum=stratum,n=len(group),valid=len(valid),
                            cosine=describe([r['cosine'] for r in valid]),relative_error=describe([r['relative_error'] for r in group]),
                            opposite_fraction=float(np.mean([r['cosine']<0 for r in valid])) if valid else None))
    merged={k:np.concatenate(v) for k,v in raw.items()}
    stability=[]
    for key in ('FD_uint','FD_float'):
        for i in range(len(steps)-1):
            stats=vector_metrics(merged[key][:,i].transpose(0,2,1).reshape(-1,16),merged[key][:,i+1].transpose(0,2,1).reshape(-1,16))
            stability.append(dict(path=key,steps=[steps[i],steps[i+1]],**{k:describe(v) for k,v in stats.items()}))
    result=dict(schema='rgb_v5_quantization_diagnostic',cases=32,split='train',updates=0,steps=list(steps),
        seconds=time.perf_counter()-began,repack=repack,summary=summary,finite_difference_stability=stability,
        float_source='raw context.aa_accum/(samples_per_pixel**2*255), clamped before integer packing',
        fixed_depth='FastDepth on original base uint8 image is shared to isolate quantization',
        caveats=['float images are diagnostic and not an evaluated replacement policy',
                 'Remaining error includes texture sampling, shading, occlusion, finite-step approximation and feature weighting',
                 'rgb_float from get_rgb already contains quantization; it was not used as unquantized ground truth'])
    write_json(folder/'SUMMARY.json',json_ready(result));write_json(folder/'ROWS.json',json_ready(rows));write_json(folder/'CASES.json',cases)
    with (folder/'ARRAYS.npz').open('xb') as f:np.savez_compressed(f,**merged)
    ctx.event('quantization_complete',cases=32,seconds=result['seconds'])
    print(json.dumps(dict(seconds=result['seconds'],repack=repack,all=[r for r in summary if r['stratum']=='all']),ensure_ascii=False),flush=True)


def main():
    from extensions.rgb_diagnostic_v5.common import bootstrap,gpu_lease,freeze_stage
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    freeze_stage(args.root,'derivative_quantization',dict(cases=32,steps=[.00075,.0025,.01,.04],selection='scene-index plus goal-index difficulty cycling',updates=0),[__file__])
    with gpu_lease(args.root,'derivative_quantization'):
        run(bootstrap(args.root),args.root/'derivative_quantization')


if __name__=='__main__':main()
