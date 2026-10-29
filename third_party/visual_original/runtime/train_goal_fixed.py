"""Resumable multi-environment AE/Critic/goal-M training and held-out evaluation.

Training uses pose labels; deployed policy only uses RGB-derived quantities.
Every rollout freezes the encoder, goal J and goal metric. Never resume an old
experiment with this different controller. All outputs have their own protocol.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from goal_fixed_core import *
from lac_vs.config import rng_state, restore_rng
from lac_vs.environment import make_manifest, make_evaluation_starts
from lac_vs.locking import writer_lock
from lac_vs.runner import aggregate, select_validation_subset

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')

class Stop:
    def __init__(self,out,seconds):
        self.out=Path(out);self.deadline=time.monotonic()+seconds;self.requested=False
        signal.signal(signal.SIGINT,lambda *_:setattr(self,'requested',True))
        if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,lambda *_:setattr(self,'requested',True))
    def stopped(self):
        return self.requested or (self.out/'STOP').exists() or time.monotonic()>=self.deadline

def dependencies(cfg):
    paths=ROOT/cfg['goal_fixed']['depth_path']
    files=[*paths.glob('*.py'),paths/'provenance.json',paths/'extracted_state_dict.pt']
    result={str(p.relative_to(ROOT)):digest(p) for p in files}
    old=legacy.dependency_hashes(cfg)
    # The old depth model is never run, but preserve the full read-only dependency audit.
    details=dict(fastdepth=result,read_only=old)
    repair=cfg['goal_fixed'].get('repair_source_run')
    if repair:
        original=ROOT/repair
        details['repair_source']={name:digest(original/name) for name in
            ('protocol.json','initial.pt','scenes.json','evaluation_starts.json')}
    return details

def prepare(out,cfg):
    out=Path(out);marker=out/'protocol.json'
    expected=dict(config=cfg,sources=source_fingerprints(cfg),dependencies=dependencies(cfg))
    if marker.exists():
        protocol=json.loads(marker.read_text(encoding='utf-8'))
        for key in expected:
            if protocol[key]!=expected[key]:raise RuntimeError(f'Protocol {key} changed: create a new output run')
        return protocol
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing nonempty output without protocol')
    out.mkdir(parents=True,exist_ok=True)
    manifest=make_manifest(cfg)
    starts={s:make_evaluation_starts(cfg,manifest,s) for s in ('val','test')}
    protocol=dict(schema=1,created=now(),**expected,
        algorithm='deterministic fixed-reference J*, goal-only Cholesky M, AE-dominant encoder + twin cost critic',
        encoder_input='one RGB image, shared current/goal encoder; 16 dimensions',
        online_pose=False,online_simulator_depth=False,metric_goal_only=True,
        certificate='V=.5 e^T M(goal) e; .1I <= M <=14.5I, trace M=16',
        guarantee='structural bounds and nominal-model descent only; real Lyapunov inequalities tested empirically',
        representation_update='AE gradient plus auxiliary encoder gradient clipped to 0.3 of AE encoder gradient',
        metric_update='encoder/critic parameters frozen, action derivative through SPD solve retained',
        actual_descent='real replay successors train the certificate; they are not counterfactual current-policy successors',
        action_descent='exact differentiable P0 kinematics with training-only pose labels plus Critic action gradients',
        selection='validation scene macro success, worst scene success, negative integrated physical error',
        dataset_scope='new layouts/goals from four shared room generators, not arbitrary real-world transfer',
        depth_provenance=json.loads((ROOT/cfg['goal_fixed']['depth_path']/'provenance.json').read_text(encoding='utf-8')))
    json_write(out/'protocol.json',protocol);json_write(out/'scenes.json',manifest);json_write(out/'evaluation_starts.json',starts)
    import yaml
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    src=out/'provenance';src.mkdir(exist_ok=True)
    for rel in expected['sources']:
        p=src/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    return protocol

def save(path,learner,state,replay,protocol):
    path=Path(path);temp=path.with_suffix('.tmp')
    payload=dict(schema=1,learner=learner.payload(),state=copy.deepcopy(state),rng=rng_state(),
        replay_rows=copy.deepcopy(replay.rows),replay_rng=replay.rng.bit_generator.state,
        protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest())
    torch.save(payload,temp);temp.replace(path)

def restore(path,learner,replay,protocol,random_state=True):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    expected=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if payload['protocol_hash']!=expected:raise RuntimeError('Checkpoint protocol mismatch')
    learner.restore(payload['learner'])
    if random_state:
        restore_rng(payload['rng']);replay.rows=payload['replay_rows'];replay.rng.bit_generator.state=payload['replay_rng']
        if any(not (replay.root/r['file']).exists() for r in replay.rows):raise FileNotFoundError('Missing replay shard')
        json_write(replay.path,replay.rows);replay.cache.clear()
    return payload['state']

def import_repair_bootstrap(out,cfg,learner,replay,protocol):
    """Explicit new protocol after a logging repair; never bypass resume hashes."""
    original=ROOT/cfg['goal_fixed']['repair_source_run']
    old_protocol=json.loads((original/'protocol.json').read_text(encoding='utf-8'))
    payload=torch.load(original/'initial.pt',map_location='cpu',weights_only=False)
    expected=hashlib.sha256(json.dumps(old_protocol,sort_keys=True).encode()).hexdigest()
    if payload['protocol_hash']!=expected:raise RuntimeError('Repair source checkpoint integrity failed')
    old_config=copy.deepcopy(old_protocol['config']);new_config=copy.deepcopy(cfg)
    new_config['goal_fixed'].pop('repair_source_run')
    if new_config!=old_config:raise RuntimeError('Repair import changes more than the declared logging fix')
    for name in ('scenes.json','evaluation_starts.json'):
        if json.loads((Path(out)/name).read_text(encoding='utf-8'))!=json.loads((original/name).read_text(encoding='utf-8')):
            raise RuntimeError('Repair split/starts differ from the original frozen protocol')
    if any(payload['learner']['counts'].values()):raise RuntimeError('Repair import must be the pre-training initial snapshot')
    learner.restore(payload['learner']);restore_rng(payload['rng'])
    replay.rows=payload['replay_rows'];replay.rng.bit_generator.state=payload['replay_rng']
    records=[]
    for row in replay.rows:
        if row['split']!='train':raise RuntimeError('Repair source replay split leakage')
        src=original/'replay'/row['file'];target=replay.root/row['file']
        if not target.exists():shutil.copy2(src,target)
        if digest(src)!=digest(target):raise RuntimeError('Repair replay copy mismatch')
        records.append(dict(file=row['file'],sha256=digest(target)))
    shutil.copytree(original/'references',Path(out)/'references',dirs_exist_ok=True)
    json_write(replay.path,replay.rows)
    shutil.copy2(original/'training_anchors.json',Path(out)/'training_anchors.json')
    state=payload['state'];state.update(stage='warmup',index=0,phase_update=0,
        repaired_from=str(original),repair_reason='progress log total keyword duplicated loss total; no optimizer math changed')
    state.pop('training_counts',None)
    json_write(Path(out)/'repair_provenance.json',dict(source=str(original),source_protocol_sha256=digest(original/'protocol.json'),
        source_initial_checkpoint_sha256=digest(original/'initial.pt'),reused_transitions=replay.transitions,
        reused_shards=records,abandoned_uncheckpointed_updates=25,
        change='warmup progress field total -> planned_updates; restart at the saved pre-warmup snapshot',
        resuming_old_protocol=False,counts_start=learner.counts))
    unit_audit(out,cfg,learner,replay)
    save(Path(out)/'initial.pt',learner,state,replay,protocol)
    save(Path(out)/'last.pt',learner,state,replay,protocol)
    emit(event='repair_bootstrap_imported',transitions=replay.transitions,counts=learner.counts,source=str(original))
    return state

class Learner:
    def __init__(self,cfg,device='cuda'):
        self.cfg=cfg;self.p=cfg['goal_fixed'];self.device=device
        self.ae=make_model(cfg).to(device).eval()
        self.metric=GoalMetric().to(device)
        self.critics=torch.nn.ModuleList([CostCritic(16),CostCritic(16)]).to(device)
        self.ta=copy.deepcopy(self.ae).eval().requires_grad_(False)
        self.tm=copy.deepcopy(self.metric).eval().requires_grad_(False)
        self.tc=copy.deepcopy(self.critics).eval().requires_grad_(False)
        self.opt=torch.optim.Adam([dict(params=self.ae.parameters(),lr=self.p['encoder_lr']),
                                   dict(params=self.critics.parameters(),lr=self.p['critic_lr'])])
        self.mopt=torch.optim.Adam(self.metric.parameters(),lr=self.p['metric_lr'])
        self.counts=dict(warmup=0,representation=0,metric=0)
        self.version=0;self.target_version=0;self.cache={};self.tcache={};self.inputs={}
        self.anchors=dict(rec=1.,curvature=1.,latent_std=[1.]*16)
    def tensor(self,a):return torch.as_tensor(a,dtype=torch.float32,device=self.device)
    def clear_cache(self):self.cache.clear();self.tcache.clear()
    def goal_features(self,b,target=False,perturb=False):
        model=self.ta if target else self.ae
        cache=self.tcache if target else self.cache
        zgs=[];js=[]
        for i,key in enumerate(b['goal_key']):
            if key not in self.inputs:
                x,q=images_and_tangents(self.cfg,[b['goal_rgb'][i]],[b['goal_depth'][i].astype(np.float32)],
                                       [b['rotation'][i]],'cpu')
                self.inputs[key]=(x,q)
            if key not in cache:
                x,q=(a.to(self.device) for a in self.inputs[key])
                with frozen_parameters(model),torch.no_grad():
                    zg=model.encode(x);j=encoder_jvp(model,x,q).transpose(1,2)
                cache[key]=(zg.detach(),j.detach())
            zg,j=cache[key];zgs.append(zg);js.append(j)
        zg,j=torch.cat(zgs),torch.cat(js)
        if perturb:
            # Per-axis translational scale uncertainty from approximate depth.
            lo,hi=self.p['depth_scale_train']
            scale=torch.exp(torch.empty(len(j),1,1,device=self.device).uniform_(math.log(lo),math.log(hi)))
            j=torch.cat([j[...,:3]/scale,j[...,3:]],-1)
            # Full spatial depth perturbations are used by fresh student rollouts.
        return zg,j
    def encode(self,images,model=None):
        return (model or self.ae).encode(image_tensor(self.cfg,images,self.device))
    def target(self,b):
        zg,j=self.goal_features(b,target=True)
        with torch.no_grad():
            zn=self.encode(b['next_rgb'],self.ta);m=self.tm(zg)[1]
            u=policy((zn-zg)/4,j,m)[0]
            tail=torch.stack([c(zn,zg,u) for c in self.tc]).amax(0)
            return self.tensor(b['cost'])+.99*(1-self.tensor(b['terminal']))*tail
    def representation(self,b,warmup=False):
        for m in (self.ae,self.metric,self.critics):m.zero_grad(set_to_none=True)
        x=image_tensor(self.cfg,b['rgb'],self.device);g=image_tensor(self.cfg,b['goal_rgb'],self.device)
        z,y=self.ae(torch.cat([x,g]));n=len(x);zt,zg=z[:n],z[n:]
        rgb=F.mse_loss(y,torch.cat([x,g]));dct=dct_reconstruction(y,torch.cat([x,g]))
        rec=(rgb+.25*dct)/self.anchors['rec']
        params=list(self.ae.parameters());enc=list(self.ae.base.encoder.parameters())
        if warmup:
            rec.backward();ratio=0.;aux=rec.new_zeros(());logs={}
        else:
            target=self.target(b)
            pred=[c(zt,zg,self.tensor(b['action'])) for c in self.critics]
            td=sum(F.smooth_l1_loss(v,target) for v in pred)/2
            anchor=sum(c(zg,zg,torch.zeros(n,4,device=self.device)).square().mean() for c in self.critics)/2
            zn=self.encode(b['next_rgb'])
            # Fixed metric values in representation updates isolate the two optimizers.
            with torch.no_grad():m=self.metric(zg.detach())[1]
            e,en=(zt-zg)/4,(zn-zg)/4
            E,En=pose_error(self.tensor(b['offset'])),pose_error(self.tensor(b['next_offset']))
            cal=.5*(calibration_loss(energy(e,m),E,self.cfg)+calibration_loss(energy(en,m),En,self.cfg))
            dec=actual_descent_loss(e,en,m,E,self.cfg)
            sm=rec.new_zeros(())
            if self.counts['representation']%self.p['smooth_triplet_every']==0:
                tri=np.concatenate([b['smooth_minus'][:2],b['smooth_center'][:2],b['smooth_plus'][:2]])
                zm,zc,zp=self.encode(tri).chunk(3)
                sm=(zp-2*zc+zm).square().mean()/self.anchors['curvature']
            wanted=z.new_tensor(self.anchors['latent_std'])
            var=(F.relu(.3*wanted-z.std(0,unbiased=False))/wanted.clamp_min(.05)).square().mean()
            aux=(self.p['critic_weight']*(td+.1*anchor)+self.p['calibration_weight']*cal+
                 self.p['real_descent_weight']*dec+self.p['smoothness_weight']*sm+
                 self.p['latent_variance_weight']*var+1e-6*z.square().mean())
            rec.backward(retain_graph=True)
            base={id(p):p.grad.detach().clone() for p in params if p.grad is not None}
            base_norm=torch.stack([base[id(p)].square().sum() for p in enc if id(p) in base]).sum().sqrt()
            self.ae.zero_grad(set_to_none=True)
            aux.backward()
            terms=[p.grad.square().sum() for p in enc if p.grad is not None]
            aux_norm=torch.stack(terms).sum().sqrt() if terms else base_norm.new_zeros(())
            ratio=min(1.,self.p['representation_aux_gradient_cap']*float(base_norm)/max(float(aux_norm),1e-12))
            for p in params:
                ag=p.grad
                if id(p) in base:p.grad=base[id(p)]+(ratio*ag if ag is not None else 0.)
                elif ag is not None:p.grad=ratio*ag
            logs=dict(td=float(td),calibration=float(cal),real_descent=float(dec),curvature=float(sm),
                variance=float(var),ae_encoder_grad=float(base_norm),aux_encoder_grad=float(aux_norm),
                critic_prediction=float(torch.stack(pred).mean()),critic_target=float(target.mean()))
        norm=torch.nn.utils.clip_grad_norm_(list(self.ae.parameters())+list(self.critics.parameters()),1.)
        if not torch.isfinite(norm) or not torch.isfinite(rec+aux):raise FloatingPointError('Nonfinite representation update')
        if any(p.grad is not None for p in self.metric.parameters()):raise RuntimeError('Encoder update leaked into M')
        self.opt.step();self.version+=1;self.target_version+=1;self.clear_cache()
        ema(self.ta,self.ae,.01)
        if not warmup:ema(self.tc,self.critics,.01)
        self.counts['warmup' if warmup else 'representation']+=1
        module_grad=lambda module:float(torch.stack([p.grad.detach().square().sum() for p in module.parameters()
             if p.grad is not None]).sum().sqrt()) if any(p.grad is not None for p in module.parameters()) else 0.
        return dict(total=float(rec+aux),reconstruction=float(rgb),dct=float(dct),ae_normalized=float(rec),
            aux_scale=ratio,gradient_norm=float(norm),encoder_gradient_norm=module_grad(self.ae.base.encoder),
            critic_gradient_norm=module_grad(self.critics),metric_gradient_norm=0.,**logs)
    def metric_update(self,b):
        for m in (self.ae,self.metric,self.critics):m.zero_grad(set_to_none=True)
        self.ae.eval()
        with torch.no_grad():
            z,zn=self.encode(b['rgb']),self.encode(b['next_rgb'])
        zg,j=self.goal_features(b,perturb=True)
        lower,m=self.metric(zg);e,en=(z-zg)/4,(zn-zg)/4
        E,En=pose_error(self.tensor(b['offset'])),pose_error(self.tensor(b['next_offset']))
        v=energy(e,m);cal=.5*(calibration_loss(v,E,self.cfg)+calibration_loss(energy(en,m),En,self.cfg))
        dec=actual_descent_loss(e,en,m,E,self.cfg)
        u,h=policy(e,j,m)
        offset=self.tensor(b['offset']);next_offset=p0_offset_step(offset,u)
        predicted_E=pose_error(next_offset)
        floor=self.p['objective_floor'];rate=self.p['decay_per_second']*.1
        physical=(F.relu(predicted_E-E+rate*E)/(E+floor)).square().mean()
        pos=offset[...,:3].square().sum(-1);posn=next_offset[...,:3].square().sum(-1)
        position=(F.relu(posn-pos+rate*pos)/(pos+floor)).square().mean()
        # Angle to the true physical negative-gradient half-space, not a velocity imitation label.
        dr=(next_offset-offset)/offset.new_tensor([1.,1.,1.,math.radians(25)])
        r=offset/offset.new_tensor([1.,1.,1.,math.radians(25)])
        cosine=(r*dr).sum(-1)/(r.norm(dim=-1)*dr.norm(dim=-1)+1e-6)
        direction=F.relu(cosine+.15).square().mean()
        smooth=(u-self.tensor(b['previous_action'])).square().mean()
        regular=(m-torch.eye(16,device=self.device)).square().mean()
        with frozen_parameters(self.critics):
            cost=torch.stack([c(z,zg,u) for c in self.critics]).amax(0).mean()
            loss=(self.p['metric_cost_weight']*cost+self.p['metric_calibration_weight']*cal+
                  self.p['metric_real_descent_weight']*dec+self.p['metric_position_descent_weight']*(physical+position)+
                  self.p['metric_direction_weight']*direction+self.p['action_smoothness_weight']*smooth+
                  self.p['metric_condition_weight']*regular)
            loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.metric.parameters(),1.)
        if not torch.isfinite(norm) or not torch.isfinite(loss):raise FloatingPointError('Nonfinite metric update')
        if any(p.grad is not None for p in self.ae.parameters()) or any(p.grad is not None for p in self.critics.parameters()):
            raise RuntimeError('Metric gradient leaked into encoder or critics')
        self.mopt.step();ema(self.tm,self.metric,.01);self.counts['metric']+=1
        return dict(total=float(loss),cost=float(cost),calibration=float(cal),real_descent=float(dec),
            physical_descent=float(physical),position_descent=float(position),direction=float(direction),
            action_smoothness=float(smooth),gradient_norm=float(norm),encoder_leak=0.,critic_leak=0.,
            V=float(v.mean()),predicted_physical_delta=float((predicted_E-E).mean()),H_condition=float(torch.linalg.cond(h).max()),
            M_eigen_min=float(torch.linalg.eigvalsh(m.detach()).min()),M_eigen_max=float(torch.linalg.eigvalsh(m.detach()).max()),
            M_trace=float(m.detach().diagonal(dim1=-2,dim2=-1).sum(-1).mean()))
    def payload(self):
        return {n:getattr(self,n).state_dict() for n in ('ae','metric','critics','ta','tm','tc','opt','mopt')}|dict(
            counts=self.counts,version=self.version,target_version=self.target_version,anchors=self.anchors)
    def restore(self,payload):
        for n in ('ae','metric','critics','ta','tm','tc','opt','mopt'):getattr(self,n).load_state_dict(payload[n])
        self.counts=payload['counts'];self.version=payload['version'];self.target_version=payload['target_version'];self.anchors=payload['anchors']
        self.clear_cache()

def new_instance(cfg,scene):
    inst=legacy.module(cfg,'room_env').RoomInstance({k:scene[k] for k in ('id','split','family','seed','params')})
    def forbidden(*_a,**_kw):raise RuntimeError('Simulator true-depth rendering is prohibited')
    inst.renderer.enable_depth_rendering=forbidden
    return inst

def goal_data(cfg,out,scene,gi,inst,depth):
    key=f'{scene["id"]}_{gi:02d}';path=Path(out)/'references'/f'{key}.npz'
    frame=legacy.module(cfg,'room_env').ReferenceFrame(np.array(scene['refs'][gi]['goal_matrix']))
    if path.exists():
        with np.load(path) as a:rgb=a['rgb'];d=a['depth']
    else:
        rgb=inst.goal_rgb(frame);d=depth.predict(rgb)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as f:np.savez_compressed(f,rgb=rgb,depth=d,camera_from_body=frame.camera_from_body)
    return frame,rgb,d.astype(np.float32)

def augment_depth(d,rng,cfg):
    p=cfg['goal_fixed'];lo,hi=p['depth_scale_train']
    scale=math.exp(rng.uniform(math.log(lo),math.log(hi)))
    field=cv2.resize(rng.normal(0,p['depth_spatial_log_sigma'],(8,8)).astype(np.float32),d.shape[::-1],interpolation=cv2.INTER_CUBIC)
    return np.clip(d*scale*np.exp(field),p['min_depth'],p['max_depth']),scale

def episode(cfg,out,scene,gi,start,learner,depth,rng,steps=120,mode='learned',plant_name='P0',
            collect=False,inst=None,video=None,depth_scale=1.):
    own=inst is None;inst=inst or new_instance(cfg,scene)
    env=legacy.module(cfg,'room_env');plant=env.PLANTS[plant_name]
    frame,goal,d=goal_data(cfg,out,scene,gi,inst,depth)
    base_depth=d.copy()
    if collect and mode!='bootstrap':d,depth_scale=augment_depth(d,rng,cfg)
    ref=Reference(cfg,learner.ae,learner.metric,goal,d,frame.camera_from_body,
                  identity=mode in ('identity_m','initial_ae_identity'),scale=depth_scale if not collect else 1.)
    state=dict(position=frame.position_from_offset(start),yaw=float(start[3]),body0=frame.body0,**plant.initial_extra())
    current=inst.render_rgb(frame,state['position'],state['yaw']);offset,E=task_error(cfg,frame,state)
    xs=[current];offs=[offset];es=[];Vs=[];actions=[];costs=[];term=[];trunc=[];previous=[np.zeros(4,np.float32)]
    sm0=[];smm=[];smp=[];sms=[];hold=0;status='timeout';begin=time.perf_counter();latency=[]
    writer=None
    if mode=='bootstrap':
        vmod=legacy.module(cfg,'vision');xg,li=vmod.observe_gray(frame,goal,d)
        jpg=li.reshape(-1,4).astype(np.float64);b=jpg.T@jpg
        h=b+.01*np.diag(np.diag(b))+max(1e-8,1.01*np.trace(b)/(1e4-1))*np.eye(4)
        kp=.7*np.linalg.solve(h,jpg.T)
    if video:
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(512,304))
        if not writer.isOpened():raise RuntimeError('Could not create MP4')
    try:
        for t in range(steps+1):
            t0=time.perf_counter();u,e,V=ref.observe(current);latency.append((time.perf_counter()-t0)*1000)
            es.append(e);Vs.append(V)
            if writer:
                panel=np.zeros((304,512,3),np.uint8);panel[:256,:256]=current;panel[:256,256:]=goal
                cv2.putText(panel,f'{scene["id"]} {mode} {plant_name}',(5,274),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                cv2.putText(panel,f'{t*.1:.1f}s pos={100*np.linalg.norm(offset[:3]):.2f}cm yaw={abs(np.degrees(offset[3])):.2f} V={V:.4g}',
                            (5,295),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                writer.write(cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
            if t==steps:break
            if mode=='bootstrap':
                err=vmod.preprocess_gray(current).ravel()-xg.ravel()
                raw=-kp@err;u=(raw/max(1.,np.linalg.norm(raw[:3]),abs(raw[3]))).astype(np.float32)
            if collect:
                u=saturate(torch.from_numpy(u+rng.normal(0,.05 if mode=='bootstrap' else .06,4).astype(np.float32))).numpy()
                if len(sm0)<cfg['goal_fixed']['smooth_triplet_limit_per_episode'] and t%10==0:
                    delta=rng.normal(size=4);delta[:3]*=.012;delta[3]*=math.radians(.6)
                    minus=offset-delta;plus=offset+delta
                    if valid_start(cfg,scene,frame,minus) and valid_start(cfg,scene,frame,plus):
                        sm0.append(current);smm.append(inst.render_rgb(frame,frame.position_from_offset(minus),minus[3]))
                        smp.append(inst.render_rgb(frame,frame.position_from_offset(plus),plus[3]));sms.append(float(np.linalg.norm(delta)))
            if not np.isfinite(u).all() or np.linalg.norm(u[:3])>1.00001 or abs(u[3])>1.00001:
                raise FloatingPointError('Invalid physical command')
            actions.append(u.astype(np.float32));previous.append(u.astype(np.float32));state=plant.step_state(state,u)
            offset,E=task_error(cfg,frame,state);offs.append(offset)
            collision=inst.collides(state['position']);outside=inst.outside_workspace(state['position'])
            good=np.linalg.norm(offset[:3])<.02 and abs(offset[3])<math.radians(1.)
            hold=hold+1 if good else 0
            terminal=collision or outside or hold>=11;time_limit=t==steps-1 and not terminal
            term.append(terminal);trunc.append(time_limit)
            costs.append(.1*(E+.02*float(u@u))+(20 if collision or outside else 0))
            current=inst.render_rgb(frame,state['position'],state['yaw']);xs.append(current)
            if terminal:
                status='collision' if collision else 'outside' if outside else 'success'
                _,en,Vn=ref.observe(current);es.append(en);Vs.append(Vn)
                if writer:
                    panel=np.zeros((304,512,3),np.uint8);panel[:256,:256]=current;panel[:256,256:]=goal
                    writer.write(cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
                break
    finally:
        if writer:writer.release()
        if own:inst.close()
    if collect and not sm0:sm0=[xs[0]];smm=[xs[0]];smp=[xs[0]];sms=[.01]
    ar=dict(rgb=np.asarray(xs,np.uint8),goal_rgb=goal,goal_depth=base_depth,collection_depth=d,
        rotation=frame.camera_from_body.astype(np.float32),
        offset=np.asarray(offs,np.float32),action=np.asarray(actions,np.float32),cost=np.asarray(costs,np.float32),
        terminal=np.asarray(term,bool),truncated=np.asarray(trunc,bool),previous_action=np.asarray(previous[:-1],np.float32),
        features=np.asarray(es,np.float32),V=np.asarray(Vs,np.float32),M=ref.m[0].cpu().numpy(),
        L=ref.lower[0].cpu().numpy(),J_reference=ref.j[0].cpu().numpy(),H=ref.h[0].cpu().numpy(),
        smooth_center=np.asarray(sm0,np.uint8),smooth_minus=np.asarray(smm,np.uint8),smooth_plus=np.asarray(smp,np.uint8),
        smooth_step=np.asarray(sms,np.float32))
    errors=np.sum(ar['offset'][:,:3]**2,axis=1)+(ar['offset'][:,3]/math.radians(25))**2
    dv=np.diff(ar['V']);tol=1e-7+1e-4*ar['V'][:-1]
    vals=ar['V']/(errors+cfg['goal_fixed']['objective_floor'])
    du=np.diff(ar['action'],axis=0)
    row=dict(scene_id=scene['id'],split=scene['split'],goal_id=gi,plant=plant_name,method=mode,status=status,
        success=status=='success',steps=len(actions),final_position_m=float(np.linalg.norm(offset[:3])),
        final_yaw_deg=float(abs(np.degrees(offset[3]))),integrated_error=float(.1*errors[1:].sum()),
        action_variation=float((du*du).sum()),actual_V_increase_fraction=float(np.mean(dv>tol)),
        actual_decay_violation_fraction=float(np.mean(dv+cfg['goal_fixed']['decay_per_second']*.1*errors[:-1]>tol)),
        certificate_comparable_fraction=float(np.mean((ar['V']>=cfg['goal_fixed']['value_lower']*errors-tol.mean())&
            (ar['V']<=cfg['goal_fixed']['value_upper']*errors+tol.mean()))),
        V_over_E_quantiles=np.quantile(vals,[.05,.5,.95]).tolist(),elapsed_s=time.perf_counter()-begin,
        depth_scale=float(depth_scale),latency_ms=float(np.mean(latency[2:] or latency)),
        M_eigen_min=float(np.linalg.eigvalsh(ar['M'])[0]),M_eigen_max=float(np.linalg.eigvalsh(ar['M'])[-1]),
        M_trace=float(np.trace(ar['M'])),J_rank=int(np.linalg.matrix_rank(ar['J_reference'])),
        offset_start=list(map(float,start)),encoder_version=learner.version,networks_frozen_in_episode=True)
    return ar,row

def calibrate(learner,replay,out):
    raw=[];goals=[];Es=[]
    with torch.no_grad():
        for _ in range(16):
            b=replay.sample(4);raw.append(learner.encode(b['rgb']).cpu());goals.append(learner.encode(b['goal_rgb']).cpu())
            Es.append(pose_error(torch.from_numpy(b['offset'])))
        raw=torch.cat(raw);goals=torch.cat(goals);E=torch.cat(Es)
        s=math.sqrt(max(float(((raw-goals)/4).square().sum(-1).mean()*.5/E.mean()),1e-8))
        learner.ae.center.copy_(raw.mean(0).to(learner.device));learner.ae.scale.fill_(s)
        learner.ta.load_state_dict(learner.ae.state_dict());z=(raw-raw.mean(0))/s
        recs=[];curves=[]
        for _ in range(8):
            b=replay.sample(4);both=np.concatenate([b['rgb'],b['goal_rgb']]);x=image_tensor(learner.cfg,both,learner.device)
            _,y=learner.ae(x);recs.append(float(F.mse_loss(y,x)+.25*dct_reconstruction(y,x)))
            a,c,d=learner.encode(np.concatenate([b['smooth_minus'],b['smooth_center'],b['smooth_plus']])).chunk(3)
            curves.append(float((a-2*c+d).square().mean()))
    learner.anchors=dict(rec=max(float(np.mean(recs)),1e-5),curvature=max(float(np.mean(curves)),1e-5),
                         latent_std=z.std(0).clamp_min(.1).tolist(),scale=s,center=raw.mean(0).tolist())
    learner.version+=1;learner.target_version+=1;learner.clear_cache()
    json_write(Path(out)/'training_anchors.json',learner.anchors)
    emit(event='training_calibration',**learner.anchors)

def unit_audit(out,cfg,learner,replay):
    checks={}
    metric=GoalMetric().double();g=torch.randn(8,16,dtype=torch.double);L,M=metric(g)
    eig=torch.linalg.eigvalsh(M);checks['SPD_trace_bounds']=bool(eig.min()>=.1-1e-10 and eig.max()<=14.5+1e-10 and
        torch.allclose(M.diagonal(dim1=-2,dim2=-1).sum(-1),torch.full((8,),16.,dtype=torch.double),atol=1e-10))
    e=torch.randn(8,16,dtype=torch.double,requires_grad=True);V=energy(e,M);grad=torch.autograd.grad(V.sum(),e)[0]
    checks['goal_only_full_state_gradient']=torch.allclose(grad,(M@e.unsqueeze(-1)).squeeze(-1),atol=1e-10)
    j=torch.randn(8,16,4,dtype=torch.double);u,h=policy(e.detach(),j,M)
    checks['speed_constraint']=bool(u[:,:3].norm(dim=-1).max()<=1+1e-12 and u[:,3].abs().max()<=1+1e-12)
    checks['nominal_nonincrease']=bool(torch.einsum('bi,bij,bj->b',e.detach(),M,(j@u.unsqueeze(-1)).squeeze(-1)).max()<1e-9)
    checks['condition_bound']=bool(torch.linalg.cond(h).max()<1e4+1e-6)
    metric.zero_grad();scalar=u[:,0].sum();scalar.backward();p=metric.net[-1].weight
    flat=p.grad.abs().reshape(-1).argmax();idx=np.unravel_index(int(flat),tuple(p.shape));analytic=float(p.grad[idx]);old=float(p[idx]);eps=1e-5
    with torch.no_grad():
        p[idx]=old+eps;plus=float(policy(e.detach(),j,metric(g)[1])[0][:,0].sum())
        p[idx]=old-eps;minus=float(policy(e.detach(),j,metric(g)[1])[0][:,0].sum());p[idx]=old
    numeric=(plus-minus)/(2*eps)
    checks['M_to_controller_finite_difference']=abs(analytic-numeric)<1e-5+1e-3*abs(numeric)
    b=replay.sample(2)
    x=image_tensor(cfg,b['goal_rgb'],learner.device);_,q=images_and_tangents(cfg,b['goal_rgb'],b['goal_depth'],b['rotation'],learner.device)
    with torch.no_grad():
        automatic=encoder_jvp(learner.ae,x[:1],q[:1])[:,0]
        eps=.0005;fd=(learner.ae.encode(x[:1]+eps*q[:1,0])-learner.ae.encode(x[:1]-eps*q[:1,0]))/(2*eps*4)
        rel=float((automatic-fd).norm()/fd.norm().clamp_min(1e-8))
    checks['encoder_JVP_input_finite_difference']=rel<.05
    before=copy.deepcopy(learner.payload())
    ml=learner.metric_update(b);checks['M_nonzero_grad_and_isolation']=ml['gradient_norm']>0 and ml['encoder_leak']==0 and ml['critic_leak']==0
    rl=learner.representation(b);checks['AE_dominant_encoder_gradient']=rl['aux_scale']*rl['aux_encoder_grad']<=.3001*rl['ae_encoder_grad']+1e-8
    learner.ae.zero_grad(set_to_none=True)
    z=learner.encode(b['rgb']);zg=learner.encode(b['goal_rgb'])
    critic_value=learner.critics[0](z,zg,learner.tensor(b['action'])).sum()
    cgrad=torch.autograd.grad(critic_value,list(learner.ae.base.encoder.parameters()),allow_unused=True)
    critic_encoder_norm=float(torch.stack([a.square().sum() for a in cgrad if a is not None]).sum().sqrt())
    checks['critic_encoder_gradient_nonzero']=critic_encoder_norm>0
    learner.restore(before)
    off=torch.tensor([[.1,.2,-.05,.3]],dtype=torch.float64);cmd=torch.tensor([[.2,-.3,.1,.4]],dtype=torch.float64)
    sim=legacy.module(cfg,'room_env').PLANTS['P0'];state=dict(position=off[0,:3].numpy(),yaw=.3,body0=np.eye(3));n=sim.step_state(state,cmd[0].numpy())
    expected=torch.tensor(np.r_[n['position'],n['yaw']])[None]
    checks['P0_pose_gradient_model_matches_simulator']=bool(torch.allclose(p0_offset_step(off,cmd),expected,atol=1e-10))
    checks['train_only_replay']=all(r['split']=='train' for r in replay.rows)
    checks['terminal_vs_timeout']=all(not np.any(replay.load(r)['terminal']&replay.load(r)['truncated']) for r in replay.rows[:2])
    try:
        with writer_lock(out):pass
        checks['single_writer_lock']=False
    except RuntimeError:checks['single_writer_lock']=True
    result=dict(passed=all(checks.values()),checks=checks,jvp_relative_error=rel,M_gradient_analytic=analytic,
        M_gradient_finite_difference=numeric,critic_encoder_gradient_norm=critic_encoder_norm,
        metric_smoke=ml,representation_smoke=rl,resources=resource(out,cfg))
    json_write(Path(out)/'engineering_audit.json',result);emit(event='engineering_audit',**result)
    if not result['passed']:raise RuntimeError('Engineering acceptance failed')

def evaluation(cfg,out,learner,depth,split,tag,cap=None,methods=None,plants=None,stop=None,stress=False):
    out=Path(out);manifest=json.loads((out/'scenes.json').read_text(encoding='utf-8'));scenes={s['id']:s for s in manifest['scenes']}
    starts=json.loads((out/'evaluation_starts.json').read_text(encoding='utf-8'))[split]
    if cap:starts=select_validation_subset(starts,cap)
    methods=methods or ['learned'];plants=plants or ['P0'];rng=np.random.default_rng(cfg['seed']+77001)
    folder=out/'evaluation'/tag;folder.mkdir(parents=True,exist_ok=True)
    rows=[];inst=None;scene_id=None;complete=True;videos=set()
    initial=None
    try:
        for method in methods:
            if method=='initial_ae_identity':
                initial=make_model(cfg).to(learner.device).eval().requires_grad_(False)
                anchors=json.loads((out/'training_anchors.json').read_text(encoding='utf-8'))
                initial.center.copy_(torch.tensor(anchors['center'],device=learner.device));initial.scale.fill_(anchors['scale'])
                old=learner.ae;learner.ae=initial
            try:
                for plant in plants:
                    for i,start in enumerate(starts):
                        name=f'{method}_{plant}_{i:04d}';path=folder/f'{name}.json'
                        if path.exists():rows.append(json.loads(path.read_text(encoding='utf-8')));continue
                        if stop and stop.stopped():complete=False;break
                        scene=scenes[start['scene_id']]
                        if scene_id!=scene['id']:
                            if inst:inst.close()
                            inst=new_instance(cfg,scene);scene_id=scene['id']
                        video_key=(method,plant,start['difficulty'])
                        video=folder/f'{name}.mp4' if video_key not in videos else None
                        arrays,row=episode(cfg,out,scene,start['goal_id'],start['offset'],learner,depth,rng,
                            steps=cfg['evaluation']['max_steps'],mode=method,plant_name=plant,inst=inst,video=video,
                            depth_scale=float(tag.rsplit('_',1)[-1]) if stress else 1.)
                        row.update(difficulty=start['difficulty'],index=i,tag=tag,checkpoint='best.pt',timestamp=now())
                        # Retain one success and one failure video for each difficulty where encountered.
                        outcome_key=video_key+(row['status']=='success',)
                        if video:
                            row['video']=str(video.relative_to(out));videos.add(video_key)
                        elif outcome_key not in videos:
                            # The trajectory's RGB evidence can be rendered later without re-running control.
                            video=folder/f'{name}.mp4';write_video(video,arrays,row)
                            row['video']=str(video.relative_to(out))
                        videos.add(outcome_key)
                        telemetry={k:v for k,v in arrays.items() if k not in ('rgb','goal_rgb','smooth_center','smooth_minus','smooth_plus')}
                        with (folder/f'{name}.npz').open('wb') as f:np.savez_compressed(f,**telemetry)
                        json_write(path,row);rows.append(row)
                        if (i+1)%12==0:
                            json_write(out/'status.json',dict(stage='evaluation',tag=tag,method=method,plant=plant,index=i+1,
                                total=len(starts),timestamp=now(),resources=resource(out,cfg)))
                            emit(event='evaluation_progress',tag=tag,method=method,plant=plant,index=i+1,total=len(starts))
                    if not complete:break
            finally:
                if initial is not None:
                    learner.ae=old;del initial;initial=None
            if not complete:break
    finally:
        if inst:inst.close()
    summaries={f'{m}/{p}':aggregate([r for r in rows if r['method']==m and r['plant']==p]) for m in methods for p in plants
               if any(r['method']==m and r['plant']==p for r in rows)}
    for key,summary in summaries.items():
        method,plant=key.split('/')
        selected=[r for r in rows if r['method']==method and r['plant']==plant]
        summary['mean_certificate_comparable_fraction']=float(np.mean([r['certificate_comparable_fraction'] for r in selected]))
        summary['M_eigen_min']=min(r['M_eigen_min'] for r in selected)
        summary['M_eigen_max']=max(r['M_eigen_max'] for r in selected)
        summary['mean_controller_latency_ms']=float(np.mean([r['latency_ms'] for r in selected]))
    result=dict(tag=tag,split=split,complete=complete,episodes=len(rows),planned=len(starts)*len(methods)*len(plants),summaries=summaries)
    json_write(folder/'summary.json',result);emit(event='evaluation_summary',**result)
    return result

def write_video(path,arrays,row):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(512,304))
    if not writer.isOpened():raise RuntimeError('MP4 writer failure')
    try:
        for i,rgb in enumerate(arrays['rgb']):
            panel=np.zeros((304,512,3),np.uint8);panel[:256,:256]=rgb;panel[:256,256:]=arrays['goal_rgb']
            cv2.putText(panel,f'{row["scene_id"]} {row["method"]} {i*.1:.1f}s',(5,278),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            writer.write(cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
    finally:writer.release()

def run(out,cfg,args):
    seed_all(cfg['seed']);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(cfg['goal_fixed']['memory_fraction'])
    depth=FastDepth(cfg);protocol=prepare(out,cfg);out=Path(out)
    if (out/'complete.json').exists():
        emit(event='already_complete',output=str(out));return
    replay=Replay(out,cfg);learner=Learner(cfg)
    stop=Stop(out,args.max_minutes*60 if args.max_minutes else cfg['goal_fixed']['wall_budget_hours']*3600)
    state=dict(stage='bootstrap',round=0,index=0,phase_update=0,best=[-1.,-1.,-1e30],created=now(),
        wall_deadline_unix=time.time()+cfg['goal_fixed']['wall_budget_hours']*3600)
    if (out/'last.pt').exists():
        if not args.resume:raise RuntimeError('Checkpoint exists: explicit --resume is required')
        state=restore(out/'last.pt',learner,replay,protocol)
        emit(event='resumed',**state,counts=learner.counts)
    elif args.resume:raise FileNotFoundError('Resume requested without checkpoint')
    elif cfg['goal_fixed'].get('repair_source_run'):
        state=import_repair_bootstrap(out,cfg,learner,replay,protocol)
    if state['stage']!='trained':
        if time.time()>state['wall_deadline_unix'] and (out/'best.pt').exists():
            state.update(stage='trained',training_stop_reason='wall_budget_expired_before_resume',training_counts=learner.counts.copy())
        else:
            stop.deadline=min(stop.deadline,time.monotonic()+max(0.,state['wall_deadline_unix']-time.time()))
    manifest=json.loads((out/'scenes.json').read_text(encoding='utf-8'));training=[s for s in manifest['scenes'] if s['split']=='train']
    rng=np.random.default_rng(cfg['seed']+90017)
    # Collection randomness also lives in the checkpoint state.
    if 'collection_rng' in state:rng.bit_generator.state=state['collection_rng']
    p=cfg['goal_fixed'];start_time=time.monotonic()
    def checkpoint():
        state['collection_rng']=rng.bit_generator.state;state['updated']=now()
        save(out/'last.pt',learner,state,replay,protocol)
    def status(**extra):
        json_write(out/'status.json',dict(**state,counts=state.get('training_counts',learner.counts),
            active_model_counts=learner.counts,replay_transitions=replay.transitions,
            timestamp=now(),resources=resource(out,cfg),**extra))
    try:
        if state['stage']=='bootstrap':
            todo=[(s,k) for s in training for k in range(p['bootstrap_episodes_per_scene'])]
            for i in range(state['index'],len(todo)):
                if stop.stopped():checkpoint();status();return
                s,k=todo[i];inst=new_instance(cfg,s)
                try:
                    frame=legacy.module(cfg,'room_env').ReferenceFrame(np.array(s['refs'][k%len(s['refs'])]['goal_matrix']))
                    difficulty=('small','medium','large')[(i//2+k)%3];offset=sample_start(cfg,s,frame,rng,difficulty)
                    arrays,row=episode(cfg,out,s,k%len(s['refs']),offset,learner,depth,rng,
                        steps=p['bootstrap_steps'],mode='bootstrap',collect=True,inst=inst)
                    replay.add(arrays,row,f'bootstrap_{i:04d}');json_append(out/'collection.jsonl',dict(phase='bootstrap',index=i,difficulty=difficulty,**row))
                finally:inst.close()
                state['index']=i+1;checkpoint();status()
                emit(event='bootstrap',index=i+1,total=len(todo),transitions=replay.transitions,status=row['status'])
            calibrate(learner,replay,out)
            unit_audit(out,cfg,learner,replay)
            save(out/'initial.pt',learner,state,replay,protocol)
            state.update(stage='warmup',index=0,phase_update=0);checkpoint()
        if state['stage']=='warmup':
            for i in range(state['phase_update'],p['warmup_updates']):
                if stop.stopped():checkpoint();status();return
                logs=learner.representation(replay.sample(p['batch_size']),warmup=True)
                state['phase_update']=i+1
                if (i+1)%25==0:
                    json_append(out/'training.jsonl',dict(phase='warmup',round=-1,update=i+1,counts=learner.counts.copy(),timestamp=now(),**logs))
                    emit(event='warmup',update=i+1,planned_updates=p['warmup_updates'],**logs);status()
                if (i+1)%p['checkpoint_every']==0:checkpoint()
            state.update(stage='collect',round=0,index=0,phase_update=0);checkpoint()
        while state.get('round',0)<p['rounds'] and state['stage'] not in ('trained','complete'):
            # Reserve the last two hours for complete held-out evaluation; no test result selects the model.
            budget_exhausted=(time.time()>state['wall_deadline_unix']-p['evaluation_reserve_hours']*3600)
            if budget_exhausted and (out/'best.pt').exists() and state['stage']=='collect' and state['index']==0:
                state['training_stop_reason']='wall_budget_with_evaluation_reserve';state['stage']='trained';checkpoint();break
            if state['stage']=='collect':
                todo=[(s,k) for s in training for k in range(2)]
                for i in range(state['index'],len(todo)):
                    if stop.stopped():checkpoint();status();return
                    s,k=todo[i];inst=new_instance(cfg,s)
                    try:
                        gi=(state['round']+k)%len(s['refs']);frame=legacy.module(cfg,'room_env').ReferenceFrame(np.array(s['refs'][gi]['goal_matrix']))
                        difficulty=('small','medium','large')[(state['round']+i)%3];offset=sample_start(cfg,s,frame,rng,difficulty)
                        arrays,row=episode(cfg,out,s,gi,offset,learner,depth,rng,steps=cfg['collection']['steps'],collect=True,inst=inst)
                        replay.add(arrays,row,f'round{state["round"]:02d}_{i:04d}')
                        json_append(out/'collection.jsonl',dict(phase='student',round=state['round'],index=i,difficulty=difficulty,**row))
                    finally:inst.close()
                    state['index']=i+1;checkpoint()
                    if (i+1)%6==0:status();emit(event='collection',round=state['round'],index=i+1,total=len(todo),transitions=replay.transitions)
                state.update(stage='representation',phase_update=0,index=0);checkpoint()
            for phase,budget in [('representation',p['representation_updates']),('metric',p['metric_updates'])]:
                if state['stage']!=phase:continue
                for i in range(state['phase_update'],budget):
                    if stop.stopped():checkpoint();status();return
                    batch=replay.sample(p['batch_size'])
                    logs=learner.representation(batch) if phase=='representation' else learner.metric_update(batch)
                    state['phase_update']=i+1
                    if (i+1)%25==0:
                        row=dict(phase=phase,round=state['round'],update=i+1,counts=learner.counts.copy(),timestamp=now(),**logs)
                        json_append(out/'training.jsonl',row);emit(event='update',**row);status()
                    if (i+1)%p['checkpoint_every']==0:checkpoint()
                state.update(stage='metric' if phase=='representation' else 'validation',phase_update=0);checkpoint()
            if state['stage']=='validation':
                result=evaluation(cfg,out,learner,depth,'val',f'round_{state["round"]:02d}',
                    cap=p['evaluation_validation_cap'],stop=stop)
                if not result['complete']:checkpoint();status();return
                v=result['summaries']['learned/P0'];score=[v['macro_scene_success'],v['worst_scene_success'],-v['mean_integrated_error']]
                improved=score>state['best']
                if improved:state['best']=score;save(out/'best.pt',learner,state,replay,protocol)
                json_append(out/'validation.jsonl',dict(round=state['round'],improved=improved,score=score,**v))
                emit(event='validation',round=state['round'],improved=improved,**v)
                resource(out,cfg,measure_size=True)
                state.update(stage='collect',round=state['round']+1,index=0,phase_update=0);checkpoint()
                from report_goal_fixed import report
                report(out)
        if state['stage'] not in ('complete',):
            state['stage']='trained';state['training_counts']=learner.counts.copy();checkpoint()
            json_write(out/'training_complete.json',dict(timestamp=now(),counts=learner.counts,rounds=state['round'],
                planned_rounds=p['rounds'],reason=state.get('training_stop_reason','configured_updates_complete'),
                best_validation=state['best'],runtime_seconds=time.monotonic()-start_time,resources=resource(out,cfg,True)))
            # Freeze model selection before opening the fixed test set.
            if not (out/'best.pt').exists():raise RuntimeError('No validation-selected checkpoint available for testing')
            restore(out/'best.pt',learner,replay,protocol,random_state=False)
            val=evaluation(cfg,out,learner,depth,'val','full_val',methods=['learned','identity_m'],plants=['P0'],stop=stop)
            if not val['complete']:status();return
            test=evaluation(cfg,out,learner,depth,'test','fixed_test',methods=p['evaluation_methods'],plants=p['evaluation_plants'],stop=stop)
            if not test['complete']:status();return
            for scale in p['stress_depth_scales']:
                stress=evaluation(cfg,out,learner,depth,'test',f'depth_scale_{scale}',cap=p['stress_test_cap'],
                    methods=['learned','identity_m'],plants=['P0'],stress=True,stop=stop)
                if not stress['complete']:status();return
            state['stage']='complete'
            json_write(out/'complete.json',dict(timestamp=now(),test=test,validation=val,training_counts=state['training_counts'],
                protocol_sha256=digest(out/'protocol.json'),best_sha256=digest(out/'best.pt'),resources=resource(out,cfg,True)))
            status();emit(event='complete',output=str(out))
    finally:
        from report_goal_fixed import report
        report(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',default=str(ROOT/'configs/goal_fixed_night.yaml'))
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--max-minutes',type=float)
    args=p.parse_args();cfg=load_config(args.config);cfg['evaluation']['test_starts_per_goal_difficulty']=cfg['goal_fixed']['evaluation_test_repeats']
    if args.smoke:
        cfg['scenes']['per_family']={s:1 for s in ('train','val','test')};cfg['scenes']['references']={s:1 for s in ('train','val','test')}
        cfg['scenes']['families']=['classroom'];cfg['goal_fixed'].update(bootstrap_episodes_per_scene=1,bootstrap_steps=8,
            rounds=1,warmup_updates=2,representation_updates=2,metric_updates=2,batch_size=2,evaluation_validation_cap=3,
            evaluation_test_repeats=1,evaluation_plants=['P0'],evaluation_methods=['learned','identity_m'],
            stress_depth_scales=[],evaluation_reserve_hours=0)
        cfg['collection']['steps']=8;cfg['evaluation']['max_steps']=12;cfg['evaluation']['test_starts_per_goal_difficulty']=1
    out=Path(args.output).resolve()
    with writer_lock(out):
        try:run(out,cfg,args)
        except Exception as exc:
            out.mkdir(parents=True,exist_ok=True)
            json_write(out/'error.json',dict(timestamp=now(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            emit(event='fatal',message=str(exc));raise

if __name__=='__main__':main()
