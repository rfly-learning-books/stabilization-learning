"""Physical environment adapter; fresh data and staged learner integration."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch

from extensions.rgb_longroll_v1.backend import LongRollRGBBackend
from extensions.rgb_longroll_v1.evaluation import evaluate_longroll
from extensions.unified_training_v1.backends.rgb import PROJECT, _write, _sha, _module_hash
from extensions.rgb_diagnostic_v5.legacy_transfer import LegacyPolicy
from .learner import build_learner


def make_replay(parent):
    class PhaseReplay(parent):
        """Room-balanced, then equally sampled available physical-error phases."""
        def add(self,arrays,row,identifier):
            arrays = dict(arrays)
            # Initial legacy collection generated simultaneous smooth probes.
            # Keep only probes before this lane's true termination.
            limit = max(1,(len(arrays['action'])+9)//10)
            for key in ('smooth_center','smooth_minus','smooth_plus','smooth_step'):
                arrays[key] = arrays[key][:limit]
            q = arrays['offset'][:-1]
            rho = np.maximum(np.linalg.norm(q[:,:3],axis=-1)/.02,np.abs(q[:,3])/math.radians(1.))
            arrays['sampling_phase'] = np.select((rho<=1,rho<=2.5),(0,1),2).astype(np.int8)
            super().add(arrays,row,identifier)
            self.rows[-1]['phases'] = [np.flatnonzero(arrays['sampling_phase']==i).tolist() for i in range(3)]
            # Index replacement is a writer operation, never an external probe.
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.rows),encoding='utf-8'); tmp.replace(self.path)

        def load(self,row):
            key = row['file']
            if key not in self.cache:
                with np.load(self.root/key,allow_pickle=False) as a:
                    self.cache[key] = {k:a[k] for k in a.files}
                while len(self.cache)>24: self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            return self.cache[key]

        def sample(self,size):
            scenes = sorted({r['scene_id'] for r in self.rows}); samples=[]
            if not hasattr(self,'sampled_phases'): self.sampled_phases=Counter()
            for _ in range(size):
                scene = scenes[int(self.rng.integers(len(scenes)))]
                pool = [r for r in self.rows if r['scene_id']==scene]
                phases = [i for i in range(3) if any(r['phases'][i] for r in pool)]
                phase = int(self.rng.choice(phases))
                weights = np.asarray([len(r['phases'][phase]) for r in pool],dtype=float)
                row = pool[int(self.rng.choice(len(pool),p=weights/weights.sum()))]
                a = self.load(row); t=int(self.rng.choice(row['phases'][phase]))
                tri = int(self.rng.integers(len(a['smooth_center'])))
                self.sampled_phases[phase]+=1
                item = dict(rgb=a['rgb'][t],next_rgb=a['rgb'][t+1],goal_rgb=a['goal_rgb'],
                    goal_depth=a['goal_depth'],rotation=a['rotation'],offset=a['offset'][t],next_offset=a['offset'][t+1],
                    cost=a['cost'][t],terminal=a['terminal'][t],truncated=a['truncated'][t],action=a['action'][t],
                    previous_action=a['previous_action'][t],goal_key=f"{row['scene_id']}/{row['goal']}")
                item.update({k:a[k][tri] for k in ('smooth_center','smooth_minus','smooth_plus','smooth_step')})
                samples.append(item)
            return {k:[r[k] for r in samples] if k=='goal_key' else np.stack([r[k] for r in samples]) for k in samples[0]}
    return PhaseReplay


class StableMetricRGBBackend(LongRollRGBBackend):
    def initialize(self):
        from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
        self.adapter,self.cfg,self.scenes=load_runtime(PROJECT)
        self.cfg=deepcopy(self.cfg)
        self.cfg['goal_fixed']['batch_size']=self.batch_size
        self.cfg['collection']['replay_transitions']=self.config['replay_capacity']
        self.cfg['derivative']['critic_encoder_gradient_cap']=self.config['stability']['critic_encoder_gradient_cap']
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed); torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        self.learner=build_learner(self.adapter,self.cfg,self.device,self.config['stability'])
        self.initial_state=deepcopy(self.learner.payload())
        self.initial_models=(deepcopy(self.learner.ae).requires_grad_(False),deepcopy(self.learner.metric).requires_grad_(False))
        self.initial_hash=_module_hash(self.learner.ae); self.initial_metric_hash=_module_hash(self.learner.metric)
        self.depth=self.adapter.core.FastDepth(self.cfg,self.device)
        self.replay=make_replay(self.adapter.core.Replay)(self.output,self.cfg,self.seed+1901)
        if self.replay.rows: raise FileExistsError('Fresh training needs empty replay')
        self.rng=np.random.default_rng(self.seed+90281)
        self.lookup={s['id']:s for s in self.scenes}
        self.train_scenes=sorted([s for s in self.scenes if s['split']=='train'],key=lambda s:s['id'])
        if self.config.get('engineering'):
            self.train_scenes=self.train_scenes[:self.config.get('train_scene_count',2)]
        self.cases=json.loads(Path(self.config['case_manifest']).read_text(encoding='utf-8-sig'))
        if not self.config.get('engineering'):
            if len(self.cases['val'])!=48 or len(self.cases['test'])!=72: raise ValueError('Frozen evaluation counts')
            if any(self.lookup[r['scene_id']]['split']!=split for split,rows in self.cases.items() for r in rows):
                raise ValueError('Evaluation split mismatch')
        _write(self.output/'RGB_CASES.json',self.cases)
        self.initialized=True
        self.actual_counts=Counter(); self.epoch_number=0; self.accepted_validation=None
        self.epoch_backup=None; self.acceptance=[]
        self.save(self.output/'rgb_initial.pt')
        provenance=dict(schema='rgb_stable_metric_v1',random_origin=True,old_replay_loaded=False,
            old_policy_weights_loaded=False,public_FastDepth=True,seed=self.seed,
            initial_ae_sha256=self.initial_hash,initial_metric_sha256=self.initial_metric_hash,
            critic='two component instantaneous physical derivatives; no Bellman target',
            controller='original fixed-goal J, Cholesky SPD M, trace16, floor0.1, original ridge and gain',
            sampling='training rooms only; room/available-phase balanced',
            success='position<2cm,yaw<1deg,11 consecutive observations within200 steps',
            performance_target=dict(P0=38,P1=38,denominator=48),
            checkpoint_reference_regularization=False)
        _write(self.output/'RGB_PROVENANCE.json',provenance)
        self.event('rgb_initialized',**provenance)
        return provenance

    def _policy(self,public,variant='selected',depth=None):
        if variant=='initial': ae,metric=self.initial_models
        elif variant=='shadow': ae,metric=self.learner.shadow,self.learner.metric
        elif variant in ('selected','identity'): ae,metric=self.learner.ae,self.learner.metric
        else: raise ValueError(variant)
        return LegacyPolicy(self.adapter,ae.eval(),metric.eval(),depth or self.depth,public,
            identity=variant in ('identity','shadow'))

    def warmup_representation(self,updates):
        result=super().warmup_representation(updates)
        anchors=self.learner.calibrate(self.replay,2 if self.config.get('engineering') else 32)
        _write(self.output/'FRESH_TRAINING_CALIBRATION.json',anchors)
        self.event('feature_coordinates_calibrated',anchors=anchors)
        return dict(result=result,calibration=anchors)

    def _updates(self,phase,updates):
        begin=time.perf_counter(); last=None
        for i in range(updates):
            self.guard(); batch=self.replay.sample(self.batch_size)
            if phase=='warmup': last=self.learner.representation(batch,warmup=True)
            elif phase=='representation': last=self.learner.representation(batch)
            elif phase=='critic_prefit': last=self.learner.critic_prefit(batch)
            elif phase=='metric': last=self.learner.metric_update(batch)
            else: raise ValueError(phase)
            self.actual_counts[phase]+=1
            if phase=='representation': self.actual_counts['critic_in_representation']+=1
            if phase=='metric' and last['metric_step_accepted']: self.actual_counts['metric_applied']+=1
            row=dict(phase=phase,update=self.actual_counts[phase],epoch=self.epoch_number,
                counts=self.learner.counts.copy(),actual_calls=dict(self.actual_counts),**last)
            self.history.append(row)
            with (self.output/'rgb_updates.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            self._record_probe(phase)
            if (i+1)%50==0 or i+1==updates:
                self.event('rgb_update',phase=phase,finished=i+1,planned=updates,actual_calls=dict(self.actual_counts),metrics=last)
        return dict(phase=phase,updates=updates,wall_s=time.perf_counter()-begin,last=last,
            actual_calls_including_rejected_epochs=dict(self.actual_counts),
            sampled_training_phases=dict(self.replay.sampled_phases))

    @torch.no_grad()
    def _record_probe(self,phase):
        if self.fixed_probe is None: return
        b=self.fixed_probe
        goal=self.learner.encode(b['goal_rgb']); error=(self.learner.encode(b['rgb'])-goal)/4
        matrix=self.learner.metric(goal)[1]; value=self.adapter.core.energy(error,matrix)
        physical=self.adapter.core.pose_error(self.learner.tensor(b['offset']))
        row=dict(phase=phase,epoch=self.epoch_number,actual_calls=dict(self.actual_counts),
            stage_update=sum(self.actual_counts.get(p,0) for p in ('warmup','representation','critic_prefit','metric')),
            matrix=matrix.cpu().tolist(),value=value.cpu().tolist(),physical_error=physical.cpu().tolist(),
            ratio=(value/physical.clamp_min(1e-12)).cpu().tolist(),goal_keys=b['goal_key'],fixed_training_probe=True)
        with (self.output/'rgb_step_probes.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')

    def save(self,path):
        result=super().save(path)
        _write(Path(path).with_suffix('.COUNTS.json'),dict(actual_calls_including_rejections=dict(self.actual_counts),
            retained_model_counts=self.learner.counts,validation_epoch=self.epoch_number))
        return result

    def train_epoch(self,epoch,representation_updates,evaluator_updates,certificate_updates):
        self.epoch_number=epoch
        self.epoch_backup=deepcopy(self.learner.payload())
        representation=self._updates('representation',representation_updates)
        critic=self._updates('critic_prefit',evaluator_updates)
        fixed=self.replay.sample(self.config['stability']['guard_batch'])
        self.learner.begin_metric_block(fixed)
        metric=self._updates('metric',certificate_updates)
        self.learner.guard_batch=None
        return dict(epoch=epoch,representation=representation,evaluator=critic,certificate=metric,
            actual_calls_including_rejected_epochs=dict(self.actual_counts))

    def evaluate(self,split='test',variants=('selected','initial','identity')):
        result=evaluate_longroll(self,split,variants)
        # Own completed trajectories: count out-of-tolerance low-command tails.
        for row in result['rows']:
            with np.load(row['trajectory'],allow_pickle=False) as a:
                offset=a['offset']; u=a['action']
                outside=(np.linalg.norm(offset[1:,:3],axis=-1)>=.02)|(np.abs(offset[1:,3])>=math.radians(1.))
                stalled=outside&(np.linalg.norm(u,axis=-1)<1e-3)
                row['last20_stalled_steps']=int(stalled[-20:].sum())
                row['final_position_m']=float(np.linalg.norm(offset[-1,:3]))
                row['final_yaw_deg']=float(abs(offset[-1,3])*180/math.pi)
        return result

    @staticmethod
    def metrics(result):
        groups=result['summary']['groups']
        counts=[groups[p+'/selected']['successes'] for p in ('P0','P1')]
        errors=[groups[p+'/selected']['mean_integrated_error'] for p in ('P0','P1')]
        failures=sum(r['reason'] in ('collision','outside') for r in result['rows'])
        return counts,errors,failures

    def validate(self):
        raw=self.evaluate('val',('selected',))
        counts,errors,failures=self.metrics(raw)
        accepted=True; prior=None
        if self.epoch_backup is not None:
            prior=self.metrics(self.accepted_validation)
            accepted=all(a>=b for a,b in zip(counts,prior[0])) and failures<=prior[2]
            if counts==prior[0]: accepted=accepted and all(a<=b*1.02+1e-8 for a,b in zip(errors,prior[1]))
            # Raw candidate is retained even if the sampling model is rolled back.
            self.save(self.output/'raw_candidates'/f'epoch_{self.epoch_number:04d}.pt')
            if not accepted:
                self.learner.restore(self.epoch_backup)
                self.learner.guard_batch=None
        record=dict(epoch=self.epoch_number,raw_folder=raw['folder'],raw_successes=counts,
            raw_mean_integrated_errors=errors,accepted=accepted,previous_metrics=prior,
            actual_calls=dict(self.actual_counts),scope='validation acceptance; not proof of monotone training')
        self.acceptance.append(record)
        _write(self.output/'acceptance'/f'{len(self.acceptance):04d}.json',record)
        self.event('candidate_validation',**record)
        if accepted: self.accepted_validation=raw
        outcome=deepcopy(self.accepted_validation)
        ac,ae,_=self.metrics(outcome)
        outcome.update(raw_candidate=record,selection_score=[min(ac),sum(ac),-max(ae)])
        self.epoch_backup=None
        return outcome

    def collect(self,initial=False):
        result=super().collect(initial=initial)
        boundary=self.collect_boundary()
        return dict(regular=result,boundary=boundary)

    def collect_boundary(self):
        """Fresh independent training starts at/around actual success tolerances."""
        from lac_vs.environment import valid_start
        collection=self.collection_id; cases=[]
        for scene in self.train_scenes:
            goal=(collection-1)%len(scene['refs'])
            frame=self.adapter.core.legacy.module(self.cfg,'room_env').ReferenceFrame(np.asarray(scene['refs'][goal]['goal_matrix']))
            for magnitude,yaw_deg in ((.012,.5),(.025,1.2),(.04,2.),(.08,4.)):
                for attempt in range(100):
                    direction=self.rng.normal(size=3); direction/=np.linalg.norm(direction)
                    offset=np.r_[direction*magnitude,math.radians(yaw_deg)*self.rng.choice([-1,1])]
                    if valid_start(self.cfg,scene,frame,offset): break
                else: raise RuntimeError('No valid training boundary start')
                cases.append(dict(scene_id=scene['id'],goal_id=goal,offset=offset.tolist()))
        horizon=self.config['boundary_steps']; count=0
        for start in range(0,len(cases),self.config['parallel_envs']):
            chunk=cases[start:start+self.config['parallel_envs']]
            env=self._env(chunk,max_steps=horizon)
            try:
                public,info=env.reset(options=dict(offsets=[c['offset'] for c in chunk],goal_ids=[c['goal_id'] for c in chunk]))
                goals=public['goal_rgb'].cpu().numpy(); rotations=public['rotation'].cpu().numpy()
                depths=np.stack([self.depth.predict(im) for im in goals]); tri=self._triplet(env,public)
                policy=self._policy(public)
                rows={k:[] for k in ('rgb','offset','action','previous_action','cost','terminal','truncated')}
                rows['rgb'].append(public['rgb'].cpu().numpy()); rows['offset'].append(info['supervision']['offset'].cpu().numpy())
                n=len(chunk); alive=np.ones(n,bool); lengths=np.zeros(n,int)
                for step in range(horizon):
                    self.guard()
                    if not alive.any(): break
                    action=policy(public).action
                    # Smaller exploration at the boundary; physical labels stay out of the policy.
                    noise=torch.as_tensor(self.rng.normal(0,.015,(n,4)),dtype=torch.float32,device=self.device)
                    action=self.adapter.core.saturate(action+noise)
                    public,_,term,trunc,info=env.step(action)
                    labels=info['supervision']
                    if not bool(info['transition_mask'].cpu().numpy()[alive].all()): raise RuntimeError('Boundary autoreset')
                    for key,tensor in (('rgb',public['rgb']),('offset',labels['next_offset']),('action',action),
                        ('previous_action',labels['previous_action']),('cost',labels['physical_cost']),('terminal',term),('truncated',trunc)):
                        rows[key].append(tensor.detach().cpu().numpy())
                    lengths+=alive; alive &= ~(term|trunc).cpu().numpy()
                policy.check(); arrays={k:np.stack(v) for k,v in rows.items()}
                for lane,(case,length) in enumerate(zip(chunk,lengths)):
                    item={k:v[:length+(k in ('rgb','offset')),lane] for k,v in arrays.items()}
                    item.update(goal_rgb=goals[lane],goal_depth=depths[lane],rotation=rotations[lane])
                    item.update({k:np.expand_dims(v[lane],0) for k,v in tri.items()})
                    self.replay.add(item,dict(scene_id=case['scene_id'],goal_id=case['goal_id'],split='train'),
                        f'boundary_{collection:03d}_{start+lane:04d}')
                    count+=int(length)
            finally: env.close()
        result=dict(collection=collection,episodes=len(cases),fresh_transitions=count,
            offsets_m=[.012,.025,.04,.08],training_only=True)
        _write(self.output/f'BOUNDARY_COLLECTION_{collection:03d}.json',result)
        self.event('boundary_collection',**result)
        return result
