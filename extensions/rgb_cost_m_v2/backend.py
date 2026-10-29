"""Fresh representation pretraining, frozen coordinates,20 cost-Q/M epochs."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import time
import random
import numpy as np
import torch
from stabilization_learning.rgb_stable.cost_q import FrozenFeatureReplay,visual_state,transition_cost
from extensions.rgb_stable_m_v1.backend import StableMetricRGBBackend,make_replay
from extensions.unified_training_v1.backends.rgb import PROJECT,_write,_module_hash,_sha
from .learner import build_learner
from stabilization_learning.storage import save_checkpoint
from stabilization_learning.storage.checkpoints import object_store


class CostMetricRGBBackend(StableMetricRGBBackend):
    def save(self,path):
        path=Path(path)
        if path.parent.name!='raw_candidates': return super().save(path)
        self.guard()
        if not self.learner.features_frozen: raise RuntimeError('Raw delta needs frozen visual coordinates')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        reference=self.output/'checkpoints/ready.pt'
        full=self.learner.payload()
        fixed={'ae','shadow','critics','aeopt','shadowopt','criticopt','anchors','version','features_frozen'}
        state=dict(schema='unified_rgb_legacy_v1',replay_rows=deepcopy(self.replay.rows),
            replay_rng=deepcopy(self.replay.rng.bit_generator.state),collection_rng=deepcopy(self.rng.bit_generator.state),
            collection_id=self.collection_id,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),config=self.config,fixed_probe=self.fixed_probe,
            initial_ae_sha256=self.initial_hash,initial_metric_sha256=self.initial_metric_hash)
        payload=dict(schema='rgb_frozen_candidate_delta_v2',reference=dict(path='../checkpoints/ready.pt',sha256=_sha(reference)),
            state=state,learner_changes={k:v for k,v in full.items() if k not in fixed})
        save_checkpoint(path,payload,store_root=object_store(self.config,self.output))
        _write(path.with_suffix('.COUNTS.json'),dict(actual_calls_including_rejections=dict(self.actual_counts),
            retained_model_counts=self.learner.counts,validation_epoch=self.epoch_number,
            storage='lossless delta; expand with extensions.rgb_cost_m_v2.checkpoint.load_checkpoint'))
        return dict(path=str(path),sha256=_sha(path),counts=self.learner.counts.copy())

    def initialize(self):
        from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
        self.adapter,self.cfg,self.scenes=load_runtime(PROJECT); self.cfg=deepcopy(self.cfg)
        self.cfg['goal_fixed']['batch_size']=self.batch_size
        self.cfg['collection']['replay_transitions']=self.config['replay_capacity']
        self.cfg['derivative']['critic_encoder_gradient_cap']=self.config['stability']['critic_encoder_gradient_cap']
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed); torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        self.learner=build_learner(self.adapter,self.cfg,self.device,self.config['stability'])
        self.initial_state=deepcopy(self.learner.payload())
        self.initial_models=(deepcopy(self.learner.ae).requires_grad_(False),deepcopy(self.learner.metric).requires_grad_(False))
        self.initial_hash=_module_hash(self.learner.ae); self.initial_metric_hash=_module_hash(self.learner.metric)
        self.depth=self.adapter.core.FastDepth(self.cfg,self.device)
        parent=make_replay(self.adapter.core.Replay)
        class Replay(parent):
            def add(replay,arrays,row,identifier):
                arrays=dict(arrays)
                # Every collection environment disables success termination;
                # terminal therefore means collision/outside, not arrival.
                arrays['failure']=arrays['terminal'].copy()
                super().add(arrays,row,identifier)
        self.replay=Replay(self.output,self.cfg,self.seed+1901)
        if self.replay.rows: raise FileExistsError('Replay must start empty')
        self.rng=np.random.default_rng(self.seed+90281)
        self.lookup={s['id']:s for s in self.scenes}
        self.train_scenes=sorted([s for s in self.scenes if s['split']=='train'],key=lambda s:s['id'])
        if self.config.get('engineering'): self.train_scenes=self.train_scenes[:self.config.get('train_scene_count',2)]
        self.cases=json.loads(Path(self.config['case_manifest']).read_text(encoding='utf-8-sig'))
        if not self.config.get('engineering'):
            if len(self.cases['val'])!=48 or len(self.cases['test'])!=72: raise ValueError('Preserve evaluation counts')
            if any(self.lookup[r['scene_id']]['split']!=split for split,rows in self.cases.items() for r in rows): raise ValueError('Split leak')
        _write(self.output/'RGB_CASES.json',self.cases)
        self.initialized=True; self.actual_counts=Counter(); self.epoch_number=0
        self.accepted_validation=None; self.epoch_backup=None; self.acceptance=[]; self._collecting=False
        self.feature_replay=FrozenFeatureReplay(self.config['replay_capacity'],self.device)
        self.feature_version=None
        self.save(self.output/'rgb_initial.pt')
        result=dict(schema='rgb_physical_action_cost_v2',random_origin=True,old_policy_weights_loaded=False,
            old_replay_loaded=False,seed=self.seed,public_FastDepth=True,
            critic='component derivative pretraining; double bounded costQ(x,u) plus next-step departure head',
            q_action='actual4D normalized physical velocity command, not raw Cholesky parameters',
            objective='current-policy cumulative physical cost through Bellman targets; exact P0 progress/hold constraints',
            representation='new random AE, reconstruction and derivative-guided pretraining, then frozen for20 Q/M rounds',
            controller='original target J, trace16 SPD Cholesky M, original ridge and gain',
            cost='dt*(outside+.05*min(log1p(rho_p+rho_yaw),8)+.01*||u||²)+20*physical_failure',
            terminal_semantics='failure absorbs; collection horizon/post-arrival cutoffs bootstrap',
            performance_target=dict(P0=38,P1=38,denominator=48),initial_ae_sha256=self.initial_hash,
            initial_metric_sha256=self.initial_metric_hash,automatic_resume=False)
        _write(self.output/'RGB_PROVENANCE.json',result); self.event('rgb_initialized',**result)
        return result

    def _env(self,cases,plant='P0',max_steps=None):
        if not self._collecting: return super()._env(cases,plant,max_steps)
        from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
        horizon=int(max_steps or self.config['evaluation_steps'])
        cfg=deepcopy(self.cfg); cfg['collection']['required_hold_observations']=horizon+1
        return GPUParallelRGBEnv(cfg,[self.lookup[c['scene_id']] for c in cases],plant=plant,
            max_steps=horizon,device=self.device,cache_dir=PROJECT/'.cache/rgb_gpu_warp',samples_per_axis=2)

    def collect(self,initial=False):
        self._collecting=True
        try: result=super().collect(initial=initial)
        finally: self._collecting=False
        if self.learner.features_frozen: result['feature_replay']=self.ingest_features()
        return result

    @torch.no_grad()
    def ingest_features(self):
        if not self.learner.features_frozen: raise RuntimeError('No mixed-version latent replay')
        if _module_hash(self.learner.ae)!=self.feature_version: raise RuntimeError('Frozen encoder changed')
        added=0; start=time.perf_counter()
        for row in self.replay.rows:
            if row['file'] in self.feature_replay.source_files: continue
            self.guard(); a=self.replay.load(row); n=row['length']
            b=dict(goal_key=[f"{row['scene_id']}/{row['goal']}"],goal_rgb=a['goal_rgb'][None],
                goal_depth=a['goal_depth'][None],rotation=a['rotation'][None])
            goal,j=self.learner.goal_features(b,perturb=False)
            z=torch.cat([self.learner.encode(a['rgb'][i:i+16]) for i in range(0,n+1,16)])
            goal=goal.expand(n,-1); j=j.expand(n,-1,-1)
            offset=self.learner.tensor(a['offset'][:-1]); next_offset=self.learner.tensor(a['offset'][1:])
            action=self.learner.tensor(a['action']); previous=self.learner.tensor(a['previous_action'])
            failure=torch.as_tensor(a['failure'],device=self.device,dtype=torch.bool)
            cost,inside,leaves=transition_cost(offset,next_offset,action,failure)
            batch=dict(z=z[:-1],goal=goal,j=j,e=(z[:-1]-goal)/4,en=(z[1:]-goal)/4,offset=offset,
                next_offset=next_offset,action=action,previous=previous,cost=cost,failure=failure,
                truncated=torch.as_tensor(a['truncated'],device=self.device,dtype=torch.bool),inside=inside,leaves=leaves)
            self.feature_replay.add(batch,row['scene_id'],row['file']); added+=n
        result=dict(new_transitions=added,retained=len(self.feature_replay),seen=self.feature_replay.seen,
            feature_version=self.feature_version,source_shards=len(self.feature_replay.source_files),wall_s=time.perf_counter()-start)
        self.event('frozen_feature_replay_updated',**result)
        return result

    def pretrain_evaluator(self,updates):
        derivative=self._updates('critic_prefit',updates)
        representation=self._updates('representation',self.config['representation_prefit_updates'])
        self.learner.freeze_features(); self.feature_version=_module_hash(self.learner.ae)
        _write(self.output/'FEATURES_FROZEN.json',dict(feature_version=self.feature_version,
            counts=self.learner.counts,normalization='post-AE calibration fixed; encoder and derivative critic frozen after representation prefit'))
        replay=self.ingest_features()
        batch=self.feature_replay.sample(min(4096,len(self.feature_replay)))
        self.learner.q.calibrate(self.learner.state(batch))
        self.learner.target_q.load_state_dict(self.learner.q.state_dict())
        self.learner.target_metric.load_state_dict(self.learner.metric.state_dict())
        self.event('cost_q_started',feature_version=self.feature_version)
        q=self.q_updates(self.config['q_prefit_updates'],'q_prefit')
        return dict(derivative=derivative,representation=representation,features=replay,q=q)

    def q_updates(self,updates,phase):
        begin=time.perf_counter(); last=None
        for i in range(updates):
            self.guard(); batch=self.feature_replay.sample(self.config['q_batch_size'])
            last=self.learner.q_update(batch); self.actual_counts['q']+=1
            row=dict(phase=phase,epoch=self.epoch_number,update=self.actual_counts['q'],
                counts=self.learner.counts.copy(),actual_calls=dict(self.actual_counts),**last)
            self.history.append(row)
            with (self.output/'rgb_updates.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
            if (i+1)%50==0 or i+1==updates:
                self.event('rgb_update',phase=phase,finished=i+1,planned=updates,actual_calls=dict(self.actual_counts),metrics=last)
        return dict(updates=updates,wall_s=time.perf_counter()-begin,last=last,actual_calls=dict(self.actual_counts))

    def train_epoch(self,epoch,representation_updates,evaluator_updates,certificate_updates):
        if representation_updates!=0: raise ValueError('Representation is frozen during the20 cost-Q/M rounds')
        if _module_hash(self.learner.ae)!=self.feature_version: raise RuntimeError('Frozen feature integrity')
        self.epoch_number=epoch
        q=self.q_updates(evaluator_updates,'q')
        # Q learning is retained if M is rejected; only the control model and its
        # optimizer are rolled back. Q itself is not an online control input.
        self.epoch_backup=dict(metric=deepcopy(self.learner.metric.state_dict()),mopt=deepcopy(self.learner.mopt.state_dict()),
            metric_count=self.learner.counts['metric'],metric_accepted=self.learner.metric_accepted)
        self.learner.begin_metric_block(self.feature_replay.sample(self.config['stability']['guard_batch']))
        begin=time.perf_counter(); rows=[]
        for i in range(certificate_updates):
            self.guard(); b=self.feature_replay.sample(self.config['metric_batch_size'])
            logs=self.learner.metric_update(b); self.actual_counts['metric']+=1
            if logs['metric_step_accepted']: self.actual_counts['metric_applied']+=1
            row=dict(phase='metric',epoch=epoch,update=self.actual_counts['metric'],counts=self.learner.counts.copy(),
                actual_calls=dict(self.actual_counts),**logs); rows.append(row); self.history.append(row)
            with (self.output/'rgb_updates.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
            self._record_probe('metric')
            if (i+1)%50==0 or i+1==certificate_updates:
                self.event('rgb_update',phase='metric',finished=i+1,planned=certificate_updates,metrics=logs,actual_calls=dict(self.actual_counts))
        self.learner.guard_batch=None
        return dict(epoch=epoch,representation='frozen',q=q,metric=dict(updates=certificate_updates,
            wall_s=time.perf_counter()-begin,last=rows[-1] if rows else None),actual_calls=dict(self.actual_counts))

    def validate(self):
        raw=self.evaluate('val',('selected',)); counts,errors,failures=self.metrics(raw)
        accepted=True; prior=None
        if self.epoch_backup is not None:
            prior=self.metrics(self.accepted_validation)
            accepted=all(a>=b for a,b in zip(counts,prior[0])) and failures<=prior[2]
            if counts==prior[0]: accepted=accepted and all(a<=b*1.02+1e-8 for a,b in zip(errors,prior[1]))
            self.save(self.output/'raw_candidates'/f'epoch_{self.epoch_number:04d}.pt')
            if not accepted:
                self.learner.metric.load_state_dict(self.epoch_backup['metric'])
                self.learner.mopt.load_state_dict(self.epoch_backup['mopt'])
                self.learner.counts['metric']=self.epoch_backup['metric_count']
                self.learner.metric_accepted=self.epoch_backup['metric_accepted']
        record=dict(epoch=self.epoch_number,raw_folder=raw['folder'],raw_successes=counts,
            raw_mean_integrated_errors=errors,accepted=accepted,previous_metrics=prior,
            actual_calls=dict(self.actual_counts),q_learning_retained_on_metric_rejection=True,
            scope='validation acceptance; raw training may degrade')
        self.acceptance.append(record); _write(self.output/'acceptance'/f'{len(self.acceptance):04d}.json',record)
        self.event('candidate_validation',**record)
        if accepted: self.accepted_validation=raw
        outcome=deepcopy(self.accepted_validation); ac,ae,_=self.metrics(outcome)
        outcome.update(raw_candidate=record,selection_score=[min(ac),sum(ac),-max(ae)])
        self.epoch_backup=None
        return outcome

    def close(self):
        if hasattr(self,'feature_replay'): del self.feature_replay
        return super().close()
