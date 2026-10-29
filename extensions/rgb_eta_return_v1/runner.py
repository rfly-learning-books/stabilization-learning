"""GPU RGB runner: fresh AE -> fixed coordinates -> eta-return critic -> M.

All environment transitions remain in the existing four-DOF P0/P1 plant. Only
this experiment uses the new learner; no other registered task is replaced.
"""
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
from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
from stabilization_gym.tasks_gpu_rgb.geometry import GPURGBGeometry
from stabilization_gym.tasks_gpu_rgb.eta_control import value, command, metric_from_eta
from stabilization_learning.rgb_eta import EtaReturnLearner, EtaConfig
from extensions.joint_certificate_v4.rollout import sample_offset
from extensions.reproducible_training_v4.deterministic_rgb import install_deterministic_blur
from .replay import EtaReplay, ImageBank

ROOT = Path(__file__).resolve().parents[2]


def atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''): h.update(data)
    return h.hexdigest()


def module_hash(module):
    h = hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Monitor:
    def __init__(self, path):
        self.path,self.rows,self.begin = Path(path),[],time.time()
        self.proc = subprocess.Popen(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-lms','1000'],
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.thread = threading.Thread(target=self.read,daemon=True); self.thread.start()
    def read(self):
        with self.path.open('w',encoding='utf-8') as f:
            for line in self.proc.stdout:
                try:
                    util,mem,power = map(float,line.split(','))
                    row = dict(unix=time.time(),utilization=util,memory_mib=mem,watts=power)
                    self.rows.append(row); f.write(json.dumps(row)+'\n'); f.flush()
                except ValueError: continue
    def close(self):
        self.proc.terminate(); self.proc.wait(timeout=10); self.thread.join(timeout=10)
        end = time.time()
        if not self.rows: return dict(samples=0,time_weighted_utilization=None)
        t = np.asarray([self.begin]+[r['unix'] for r in self.rows]+[end])
        w = np.diff(t); u = np.asarray([self.rows[0]['utilization']]+[r['utilization'] for r in self.rows])
        return dict(samples=len(self.rows),seconds=end-self.begin,time_weighted_utilization=float(w@u/w.sum()),
            peak_memory_mib=max(r['memory_mib'] for r in self.rows),scope='whole GPU, entire run, endpoint extension; no best-window selection')


class Run:
    def __init__(self, output, c):
        self.root = Path(output).resolve(); self.root.mkdir(parents=True,exist_ok=False)
        self.c = c; self.started = time.time(); self.deadline = min(self.started+60*c['maximum_minutes'],c.get('absolute_deadline_unix',float('inf')))
        self.training_deadline = self.deadline-60*c.get('evaluation_reserve_minutes',0)
        self.event_id = 0; self.selected_epoch = None; self.best = None; self.completed_epochs = 0
        self.monitor = Monitor(self.root/'GPU_SAMPLES.jsonl')
    def guard(self):
        if time.time()>=self.deadline: raise TimeoutError('Absolute experiment deadline')
        for p in (self.root/'STOP',ROOT/'STOP',ROOT/'runs/STOP'):
            if p.exists(): raise InterruptedError('STOP: '+str(p))
        if shutil.disk_usage(self.root).free < self.c['minimum_free_gib']*2**30: raise OSError('Disk reserve reached')
    def event(self,kind,**data):
        self.event_id += 1
        atomic(self.root/'events'/f'{self.event_id:06d}_{kind}.json',dict(kind=kind,unix=time.time(),elapsed=time.time()-self.started,**data))
    def updates(self, phase, n, fn):
        with (self.root/'UPDATES.jsonl').open('a',encoding='utf-8') as f:
            for i in range(n):
                self.guard(); row = dict(phase=phase,step=i+1,unix=time.time(),**fn(i))
                f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()
                if i==0 or (i+1)%50==0 or i+1==n: self.event('updates',phase=phase,completed=i+1,planned=n,metrics=row)
    def checkpoint(self, name, learner, **meta):
        self.guard(); path = self.root/name; tmp = path.with_suffix('.tmp')
        torch.save(dict(**learner.payload(),metadata=meta),tmp); os.replace(tmp,path)
        atomic(path.with_suffix('.json'),dict(sha256=sha(path),counts=learner.counts,**meta))


def cases(cfg, scenes, split, count, seed):
    rng = np.random.default_rng(seed)
    subset = sorted([s for s in scenes if s['split']==split],key=lambda s:s['id'])
    result = []
    for i in range(count):
        scene = subset[i%len(subset)]; g = (i//len(subset))%len(scene['refs'])
        difficulty = ('near','small','medium','large')[(i//len(subset)+i)%4]
        result.append(dict(case_id=f'{split}_{seed}_{i}',scene_id=scene['id'],goal_id=g,difficulty=difficulty,
            offset=sample_offset(cfg,scene,g,difficulty,rng).tolist()))
    return result


class Context:
    def __init__(self, run):
        self.run,self.c = run,run.c; self.device = self.c['device']
        self.adapter,self.cfg,self.scenes = load_runtime(ROOT)
        self.lookup = {s['id']:s for s in self.scenes}
        self.train_scenes = sorted([s for s in self.scenes if s['split']=='train'],key=lambda s:s['id'])
        if self.c.get('train_scene_limit'): self.train_scenes = self.train_scenes[:self.c['train_scene_limit']]
        self.geometry = GPURGBGeometry(self.cfg,self.device)
        # Explicit false avoids the historical AE checkpoint and bottleneck SVD.
        self.ae = self.adapter.core.make_model(self.cfg,initialize=False).to(self.device).eval()
        install_deterministic_blur(self.ae)
        self.rng = np.random.default_rng(self.c['seed']+81)
        self.refs = {}; self.feature_version = None
    def make_env(self, batch, horizon):
        cfg = deepcopy(self.cfg); cfg['collection']['required_hold_observations'] = horizon+2
        return GPUParallelRGBEnv(cfg,[self.lookup[b['scene_id']] for b in batch],plant=batch[0].get('plant','P0'),
            max_steps=horizon,device=self.device,cache_dir=ROOT/'.cache/rgb_eta_return_v1',samples_per_axis=2)
    def train_cases(self, scene, epoch):
        out = []
        for i in range(self.c['parallel_envs']):
            g = i%len(scene['refs']); difficulty = ('near','small','medium','large')[(i//len(scene['refs'])+epoch)%4]
            out.append(dict(case_id=f'train_{epoch}_{scene["id"]}_{i}',scene_id=scene['id'],goal_id=g,difficulty=difficulty,
                offset=sample_offset(self.cfg,scene,g,difficulty,self.rng).tolist()))
        return out
    @torch.no_grad()
    def reference(self, batch, public):
        for i,b in enumerate(batch):
            key = (b['scene_id'],b['goal_id'])
            if key not in self.refs:
                rgb,rot = public['goal_rgb'][i:i+1],public['rotation'][i:i+1]
                x,tangent = self.geometry.tangents(rgb,self.geometry.predict_depth(rgb),rot)
                goal = self.ae.encode(x)
                j = self.adapter.vision.encoder_jvp(self.ae,x,tangent).transpose(1,2)
                self.refs[key] = (goal,j)
        return tuple(torch.cat([self.refs[(b['scene_id'],b['goal_id'])][i] for b in batch]) for i in (0,1))
    @torch.no_grad()
    def error(self, rgb, goal):
        return (self.ae.encode(self.geometry.encode_images(rgb))-goal)/4.


def warmup(ctx):
    run,c = ctx.run,ctx.c
    bank = ImageBank(c['image_bank_capacity'],c['seed']+7)
    for room_index,scene in enumerate(ctx.train_scenes):
        batch = ctx.train_cases(scene,-1); env = ctx.make_env(batch,c['initial_collection_steps'])
        try:
            public,_ = env.reset(options=dict(offsets=[b['offset'] for b in batch],goal_ids=[b['goal_id'] for b in batch]))
            bank.add(ctx.geometry.encode_images(public['goal_rgb']))
            alive = torch.ones(len(batch),device=ctx.device,dtype=torch.bool)
            for step in range(c['initial_collection_steps']):
                run.guard()
                if not bool(alive.any()): break
                if step%4==0: bank.add(ctx.geometry.encode_images(public['rgb'][alive]))
                if step%5==0:
                    action = torch.randn(len(batch),4,device=ctx.device)*.2
                    divisor = torch.maximum(action[:,:3].norm(dim=-1),action[:,3].abs()).clamp_min(1)
                    action /= divisor[:,None]
                public,_,terminal,truncated,info = env.step(torch.where(alive[:,None],action,0.))
                if not bool(info['transition_mask'][alive].all()): raise AssertionError('AE warmup autoreset contamination')
                alive &= ~(terminal|truncated)
            run.event('ae_images',room_index=room_index+1,rooms=len(ctx.train_scenes),retained_images=bank.size,total_images=bank.seen)
        finally: env.close()
    opt = torch.optim.Adam(ctx.ae.parameters(),lr=c['ae_learning_rate'])
    def update(_):
        x = bank.batch(c['ae_batch_size'],ctx.device); _,recon = ctx.ae(x)
        rgb = (recon-x).square().mean(); dct = ctx.adapter.vision.dct_reconstruction(recon,x)
        loss = rgb+.25*dct
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ctx.ae.parameters(),1.,error_if_nonfinite=True)
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite AE reconstruction')
        opt.step()
        return dict(ae_loss=float(loss),reconstruction=float(rgb),dct=float(dct),gradient=float(norm))
    run.updates('ae_warmup',c['ae_updates'],update)
    # Calibrate coordinates once from AE-only training images, then freeze both
    # parameters and normalization for the complete eta-return experiment.
    with torch.no_grad():
        z = []
        for start in range(0,min(bank.size,2048),c['ae_batch_size']):
            run.guard(); z.append(ctx.ae.base.encode(torch.as_tensor(bank.images[start:start+c['ae_batch_size']],device=ctx.device)))
        z = torch.cat(z); ctx.ae.center.copy_(z.mean(0)); ctx.ae.scale.copy_(z.std(0,unbiased=False).clamp_min(.05))
    ctx.ae.requires_grad_(False).eval(); ctx.feature_version = module_hash(ctx.ae)
    torch.save(dict(model=ctx.ae.state_dict(),feature_version=ctx.feature_version,warmup_updates=c['ae_updates'],
        initialization='random; no historical learned AE',image_bank_seen=bank.seen),run.root/'FEATURES.pt')
    # Retain a bounded training-image evidence sample, not all historical frames.
    np.save(run.root/'AE_IMAGE_SAMPLE.npy',bank.images[:min(256,bank.size)],allow_pickle=False)
    run.event('features_frozen',feature_version=ctx.feature_version,scale=ctx.ae.scale.cpu().tolist(),images=bank.seen)
    del opt,bank,z


def rollout(ctx, learner, batch, reach, *, replay=None, variant='selected', trace_name=None):
    run,c = ctx.run,ctx.c; n = len(batch); post_steps = c['post_success_steps']; horizon = reach+post_steps
    env = ctx.make_env(batch,horizon); trace = []
    try:
        public,initial = env.reset(options=dict(offsets=[b['offset'] for b in batch],goal_ids=[b['goal_id'] for b in batch]))
        goal,j = ctx.reference(batch,public); error = ctx.error(public['rgb'],goal)
        alive = torch.ones(n,device=ctx.device,dtype=torch.bool)
        first = torch.full((n,),-1,device=ctx.device,dtype=torch.long); lengths = torch.zeros_like(first)
        post_bad = torch.zeros_like(first); failed = torch.zeros_like(alive); post_count = torch.zeros_like(first)
        integrated = torch.zeros(n,device=ctx.device); effort = integrated.clone(); rewards = integrated.clone()
        violations = torch.zeros_like(first); previous = torch.zeros(n,4,device=ctx.device)
        last_E = initial['supervision']['physical_error'].float(); reasons = ['timeout']*n
        for step in range(horizon):
            run.guard()
            if not bool(alive.any()): break
            action,diag = learner.act(error,goal,j,explore=replay is not None,variant=variant)
            action = torch.where(alive[:,None],action,0.)
            following,physical_reward,terminal,truncated,info = env.step(action)
            if not bool(info['transition_mask'][alive].all()): raise AssertionError('Autoreset transition entered eta replay')
            truth = info['supervision']; en = ctx.error(following['rgb'],goal)
            energy,next_energy = truth['physical_error_before'].float(),truth['physical_error'].float()
            physical_failure = info['failure']; failed |= physical_failure & alive
            offset = truth['next_offset']; inside = (offset[:,:3].norm(dim=-1)<.02)&(offset[:,3].abs()<math.radians(1.))
            newly = alive&(first<0)&(info['hold_observations']>=11)&~physical_failure&(step+1<=reach)
            first = torch.where(newly,step+1,first)
            post = alive&(first>=0)&(step+1>first)
            post_count += post.long(); post_bad += (post&~inside).long()
            post_end = (first>=0)&(post_count>=post_steps)
            sampling_end = post_end|((first<0)&(step+1>=reach))|truncated
            end = alive&(physical_failure|sampling_end)
            # Fixed physical reward, independent of M, eta and learned V.
            # The outside-tolerance running cost gives near-goal accuracy an
            # explicit continuing-return consequence without a history bonus.
            reward = physical_reward.float()-.1*c['outside_tolerance_cost']*(~inside).float()
            vn = value(en,diag['matrix']); delta = vn-diag['value']
            residual = delta+learner.config.decay_per_second*.1*energy
            if replay is not None:
                replay.append(dict(error=error[alive],next_error=en[alive],goal=goal[alive],interaction=j[alive],
                    eta=diag['eta'][alive],eta_valid=torch.ones(int(alive.sum()),device=ctx.device,dtype=torch.bool),
                    action=action[alive],previous=previous[alive],reward=reward[alive],failure=physical_failure[alive],
                    cutoff=sampling_end[alive]&~physical_failure[alive],energy=energy[alive],next_energy=next_energy[alive],
                    offset=truth['offset'][alive].float(),next_offset=offset[alive].float(),
                    behavior_V=diag['value'][alive],behavior_V_next=vn[alive]),ctx.feature_version)
            if trace_name:
                trace.append({k:v.detach().cpu().numpy() for k,v in dict(error=error,next_error=en,matrix=diag['matrix'],
                    eta=diag['eta'],interaction=j,action=action,V=diag['value'],V_next=vn,energy=energy,next_energy=next_energy,
                    offset=truth['offset'],next_offset=offset,reward=reward,active=alive,failure=physical_failure,
                    applied_action=(truth['actual_velocity']/env.scale).float(),previous_action=previous).items()})
            lengths += alive.long(); integrated += alive*truth['integrated_error'].float()
            effort += alive*truth['normalized_input_squared_integral'].float(); rewards += alive*reward
            violations += (alive&(residual>1e-7)).long(); last_E = torch.where(alive,next_energy,last_E)
            for lane in torch.nonzero(end).flatten().cpu().tolist():
                reasons[lane] = 'collision' if bool(info['collision'][lane]) else 'outside' if bool(info['outside'][lane]) else 'post_success_complete' if bool(post_end[lane]) else 'timeout'
            alive &= ~end; error = en; public = following; previous = action
        rows = [dict(**b,variant=variant,success=bool(first[i]>=0),
            hold_success=bool(first[i]>=0 and post_count[i]>=post_steps and post_bad[i]==0 and not failed[i]),
            hold_censored=bool(first[i]>=0 and post_count[i]<post_steps and not failed[i]),
            first_success_step=int(first[i]),steps=int(lengths[i]),post_observed=int(post_count[i]),post_outside_steps=int(post_bad[i]),
            failure=bool(failed[i]),reason=reasons[i],integrated_error=float(integrated[i]),input_squared=float(effort[i]),
            physical_final=float(last_E[i]),reward_sum=float(rewards[i]),actual_descent_violations=int(violations[i])) for i,b in enumerate(batch)]
        if trace_name and trace:
            folder = run.root/'trajectories'; folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/(trace_name+'.npz'),**{k:np.stack([r[k] for r in trace]) for k in trace[0]})
        return rows
    finally: env.close()


def summarized(rows):
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),hold_successes=sum(r['hold_success'] for r in rows),
        failures=sum(r['failure'] for r in rows),censored=sum(r['hold_censored'] for r in rows),
        mean_integrated_error=float(np.mean([r['integrated_error'] for r in rows])),mean_final_error=float(np.mean([r['physical_final'] for r in rows])),
        mean_input_squared=float(np.mean([r['input_squared'] for r in rows])),
        actual_descent_violation_fraction=sum(r['actual_descent_violations'] for r in rows)/max(1,sum(r['steps'] for r in rows)))


def geometry_preflight(ctx, learner):
    """Compare GPU preprocessing/J to the retained image geometry on train data."""
    batch = ctx.train_cases(ctx.train_scenes[0],-2)[:4]
    env = ctx.make_env(batch,2)
    try:
        public,_ = env.reset(options=dict(offsets=[b['offset'] for b in batch],goal_ids=[b['goal_id'] for b in batch]))
        depth = ctx.geometry.predict_depth(public['goal_rgb'])
        x,tangent = ctx.geometry.tangents(public['goal_rgb'],depth,public['rotation'])
        original_x,original_tangent = ctx.adapter.vision.images_and_tangents(ctx.cfg,
            public['goal_rgb'].cpu().numpy(),depth[:,0].cpu().numpy(),public['rotation'].cpu().numpy(),ctx.device)
        with torch.no_grad():
            goal = ctx.ae.encode(x); old_goal = ctx.ae.encode(original_x)
            j = ctx.adapter.vision.encoder_jvp(ctx.ae,x,tangent).transpose(1,2)
            old_j = ctx.adapter.vision.encoder_jvp(ctx.ae,original_x,original_tangent).transpose(1,2)
            image = ctx.geometry.encode_images(public['rgb'])
            old_image = ctx.adapter.vision.image_tensor(ctx.cfg,public['rgb'].cpu().numpy(),ctx.device)
            error = (ctx.ae.encode(image)-goal)/4.
            old_error = (ctx.ae.encode(old_image)-old_goal)/4.
            _,m = metric_from_eta(learner.metric(goal)); _,old_m = metric_from_eta(learner.metric(old_goal))
            u,_ = command(error,j,m); old_u,_ = ctx.adapter.core.policy(old_error,old_j,old_m)
        result = dict(image_max_abs=float((image-old_image).abs().max()),tangent_max_abs=float((tangent-original_tangent).abs().max()),
            interaction_max_abs=float((j-old_j).abs().max()),action_max_abs=float((u-old_u).abs().max()),
            scope='same frozen encoder and same public FastDepth output; GPU vs retained CPU preprocessing and target J')
        if not torch.allclose(image,old_image,atol=2e-5,rtol=1e-5): raise AssertionError('GPU preprocessing parity')
        if not torch.allclose(j,old_j,atol=1e-4,rtol=1e-4): raise AssertionError('GPU target J parity')
        if not torch.allclose(u,old_u,atol=5e-4,rtol=1e-4): raise AssertionError('GPU action parity')
        atomic(ctx.run.root/'GEOMETRY_PREFLIGHT.json',dict(passed=True,**result)); ctx.run.event('geometry_preflight',passed=True,**result)
    finally: env.close()


def evaluate(ctx, learner, items, label, variants=('selected',)):
    rows = []
    for variant in variants:
        for plant in ctx.c['evaluation_plants']:
            for room in sorted(set(b['scene_id'] for b in items)):
                subset = [dict(**b,plant=plant) for b in items if b['scene_id']==room]
                for start in range(0,len(subset),ctx.c['parallel_envs']):
                    rows += rollout(ctx,learner,subset[start:start+ctx.c['parallel_envs']],ctx.c['evaluation_steps'],
                        variant=variant,trace_name=f'{label}_{variant}_{plant}_{room}_{start}')
    groups = {f'{v}/{p}':summarized([r for r in rows if r['variant']==v and r['plant']==p]) for v in variants for p in ctx.c['evaluation_plants']}
    atomic(ctx.run.root/(label+'.json'),dict(groups=groups,episodes=rows)); ctx.run.event('evaluation_complete',label=label,groups=groups)
    return rows,groups


def collect(ctx, learner, replay, scenes, epoch, updates=False):
    before = replay.total; rows = []
    for i,scene in enumerate(scenes):
        rows += rollout(ctx,learner,ctx.train_cases(scene,epoch),ctx.c['collection_steps'],replay=replay)
        ctx.run.event('collection_room',epoch=epoch,room=scene['id'],room_index=i+1,rooms=len(scenes),new_transitions=replay.total-before,total_transitions=replay.total)
        if updates:
            ctx.run.updates(f'epoch_{epoch}_room_{i}_Q',ctx.c['q_updates_per_room'],lambda _:learner.critic_update(replay.batch(ctx.c['batch_size'])))
            ctx.run.updates(f'epoch_{epoch}_room_{i}_M',ctx.c['m_updates_per_room'],lambda k:learner.metric_update(replay.batch(ctx.c['batch_size']),gradient_audit=k==0))
    atomic(ctx.run.root/f'COLLECTION_{epoch:03d}.json',dict(new_transitions=replay.total-before,episodes=rows))


def independent_audit(root):
    root = Path(root); maximum = 0.; action_max = 0.; count = 0
    for path in (root/'trajectories').glob('*.npz'):
        with np.load(path,allow_pickle=False) as t:
            active = t['active']; m = t['matrix'].astype(np.float64)
            for key,logged in (('error','V'),('next_error','V_next')):
                e = t[key].astype(np.float64); v = .5*np.einsum('tbi,tbij,tbj->tb',e,m,e)
                maximum = max(maximum,float(np.abs(v[active]-t[logged][active]).max(initial=0)))
                if not np.allclose(v[active],t[logged][active],atol=2e-5,rtol=2e-5): raise AssertionError('Full V audit: '+str(path))
            with torch.no_grad():
                u,_ = command(torch.as_tensor(t['error']),torch.as_tensor(t['interaction']),torch.as_tensor(t['matrix']))
                delta = (u-torch.as_tensor(t['action'])).abs().numpy()[active]
                action_max = max(action_max,float(delta.max(initial=0)))
                if not np.allclose(u.numpy()[active],t['action'][active],atol=5e-4,rtol=1e-4): raise AssertionError('Recorded action audit: '+str(path))
            count += int(active.sum())
    return dict(passed=True,transitions=count,V_max_absolute_error=maximum,action_max_absolute_error=action_max,
        V_rule='both complete endpoint quadratic forms with exactly the same fixed metric and encoder')


def train(output, config):
    run = Run(output,config); status = 'failed'; failure = None; replay = learner = ctx = None
    try:
        c = config; random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        ctx = Context(run)
        manifests = {s:cases(ctx.cfg,ctx.scenes,s,c[s+'_cases'],c['evaluation_seed']+(1 if s=='val' else 2)) for s in ('val','test')}
        if c.get('engineering_smoke'):
            # Engineering checks never read development/test outcomes.
            manifests = {s:cases(ctx.cfg,ctx.train_scenes,'train',c[s+'_cases'],c['evaluation_seed']+i) for i,s in enumerate(('val','test'))}
        roots = [ROOT/'extensions/rgb_eta_return_v1',ROOT/'stabilization-learning-control/stabilization_learning/rgb_eta',
                 ROOT/'stabilization-gym/stabilization_gym/tasks_gpu_rgb',ROOT/'third_party/visual_original/runtime',
                 ROOT/'third_party/visual_original/lac_vs']
        sources = sorted(set(p for folder in roots for p in folder.rglob('*.py')))
        sources += [ROOT/'scripts/run_rgb_eta_return_v1.py',ROOT/'stabilization-gym/stabilization_gym/tasks_v2/reconstructed_rgb.py',
            ROOT/'extensions/reproducible_training_v4/deterministic_rgb.py',ROOT/'extensions/joint_certificate_v4/rollout.py']
        hashes = {str(p):sha(p) for p in sources}; snapshot = run.root/'SOURCE_SNAPSHOT'; snapshot.mkdir()
        for p in sources: shutil.copyfile(p,snapshot/(sha(p)+'_'+p.name))
        protocol = dict(schema='rgb_eta_return_v1',config=c,started_unix=run.started,deadline_unix=run.deadline,
            training_deadline_unix=run.training_deadline,source_hashes=hashes,cases=manifests,runtime_task_config=ctx.cfg,
            train_rooms=[s['id'] for s in ctx.train_scenes],scene_manifest_sha256=sha(ROOT/'migration/visual_scene_reconstruction/scenes.json'),
            depth_sha256=ctx.geometry.depth.hash,initialization='random original CAE architecture, identity-neighborhood random goal metric, fresh twin return critics and optimizers; no old learned AE or replay',
            controller='legacy target J; e and J divided by 4, goal input unscaled; exact legacy ridge, gain0.7 and joint radial saturation',
            action='eta=136 raw Cholesky entries; eta->L->M->legacy physical velocity. Actor is goal-only, deterministic at evaluation.',
            schedule='eta chosen each step; Gaussian eta exploration only in collection; deterministic next eta from target metric in Bellman target. No episode-return bandit.',
            critic='Q(error,goal,targetJ,previous_command,eta), predicts discounted continuing physical reward; twin target critics and Polyak target metric; no entropy term',
            replay='only post-warmup frozen-coordinate transitions with actual eta. Random physical-action AE warmup cannot enter Q replay.',
            reward='existing negative physical cost minus dt*outside_tolerance_cost outside2cm/1deg; no learned V, no history-dependent success bonus',
            termination='physical failure is absorbing; reach/holding/horizon are sampling cutoffs and bootstrap from the real last image',
            constraints='retain legacy M calibration, behavior replay full V descent, P0 physical/position/direction, action smoothness and matrix regularization; replay descent is not counterfactual policy descent',
            selection='validation only; lexicographic post-success50-step holds, arrivals, negative integrated physical error; initial M included as epoch0',
            tests='after freeze: selected, same-feature identity and initial M, P0/P1. P1 transfer only; fixed-coordinate P0 Q is not a P1 dynamics model.',
            limitations='single seed; existing44rooms/publicFastDepth/GPU pixels;4DOF outer loop; wallclock not in dt;input square is proxy; empirical Lyapunov penalties, no new stability proof',
            automatic_resume=False,runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
        atomic(run.root/'protocol.json',protocol); run.event('initialized',deadline_unix=run.deadline,config=c)
        warmup(ctx)
        learner = EtaReturnLearner(EtaConfig(**c.get('learner',{})),c['device'],ctx.feature_version)
        geometry_preflight(ctx,learner)
        replay = EtaReplay(c['replay_capacity'],c['device'],ctx.feature_version)
        run.checkpoint('initial.pt',learner,epoch=0)
        collect(ctx,learner,replay,ctx.train_scenes,0)
        probe_keys = sorted(ctx.refs)
        fixed_goals = torch.cat([ctx.refs[k][0] for k in probe_keys])
        np.savez_compressed(run.root/'M_PROBE_CONTEXT.npz',goal=fixed_goals.cpu().numpy(),
            interaction=torch.cat([ctx.refs[k][1] for k in probe_keys]).cpu().numpy())
        run.updates('critic_pretrain',c['critic_pretrain'],lambda _:learner.critic_update(replay.batch(c['batch_size'])))
        for epoch in range(c['epochs']+1):
            if epoch:
                if time.time()>=run.training_deadline:
                    run.event('training_budget_complete',completed_epochs=run.completed_epochs); break
                n = min(c['rooms_per_epoch'],len(ctx.train_scenes)); start = (epoch-1)*n
                rooms = [ctx.train_scenes[(start+i)%len(ctx.train_scenes)] for i in range(n)]
                collect(ctx,learner,replay,rooms,epoch,updates=True)
            if module_hash(ctx.ae)!=ctx.feature_version: raise AssertionError('Encoder changed under eta replay')
            rows,groups = evaluate(ctx,learner,manifests['val'],f'VAL_{epoch:03d}')
            score = (sum(r['hold_success'] for r in rows),sum(r['success'] for r in rows),-sum(r['integrated_error'] for r in rows))
            run.checkpoint(f'epoch_{epoch:03d}.pt',learner,epoch=epoch,validation_score=score)
            if run.best is None or score>run.best:
                run.best=score; run.selected_epoch=epoch
                run.checkpoint('selected.pt',learner,epoch=epoch,validation_score=score)
            run.completed_epochs=epoch
            with torch.no_grad():
                eta = learner.metric(fixed_goals); _,m = metric_from_eta(eta)
                atomic(run.root/f'M_PROBE_{epoch:03d}.json',dict(mean_matrix=m.mean(0).cpu().tolist(),
                    eigen_min=float(torch.linalg.eigvalsh(m).min()),eigen_max=float(torch.linalg.eigvalsh(m).max()),
                    mean_eta=eta.mean(0).cpu().tolist(),goal_keys=[list(k) for k in probe_keys],note='same frozen training goal contexts every epoch'))
            run.event('epoch_complete',epoch=epoch,planned=c['epochs'],counts=learner.counts,selected_epoch=run.selected_epoch,total_transitions=replay.total)
        run.checkpoint('final.pt',learner,epoch=run.completed_epochs)
        if any(sha(p)!=h for p,h in hashes.items()): raise AssertionError('Source changed during run')
        atomic(run.root/'FROZEN.json',dict(unix=time.time(),selected_epoch=run.selected_epoch,completed_epochs=run.completed_epochs,
            selected_sha256=sha(run.root/'selected.pt'),features_sha256=sha(run.root/'FEATURES.pt'),feature_version=ctx.feature_version))
        run.event('frozen',selected_epoch=run.selected_epoch)
        learner.restore(torch.load(run.root/'selected.pt',map_location=c['device'],weights_only=False))
        _,groups = evaluate(ctx,learner,manifests['test'],'TEST',variants=('selected','identity','initial_metric'))
        audit = independent_audit(run.root); atomic(run.root/'AUDIT.json',audit)
        meta = replay.save(run.root/'replay'); atomic(run.root/'REPLAY.json',meta)
        atomic(run.root/'RESULTS.json',dict(groups=groups,selected_epoch=run.selected_epoch,completed_epochs=run.completed_epochs,
            counts=learner.counts,new_transitions=replay.total,audit=audit))
        status='complete'
    except BaseException as ex:
        failure=repr(ex); status='stopped' if isinstance(ex,InterruptedError) else 'timeout' if isinstance(ex,TimeoutError) else 'failed'
        (run.root/'FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        run.event('failed',status=status,error=failure)
    finally:
        gpu=run.monitor.close(); atomic(run.root/'GPU_SUMMARY.json',gpu)
        atomic(run.root/'FINAL_STATE.json',dict(status=status,error=failure,completed_epochs=run.completed_epochs,
            selected_epoch=run.selected_epoch,finished_unix=time.time(),elapsed_seconds=time.time()-run.started,gpu=gpu))
    return 0 if status=='complete' else 1
