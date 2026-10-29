"""Injected task/policy, full successor certificate, CPU whole-update recovery."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from stabilization_gym.interfaces.certificate_v2 import VersionBinding
from stabilization_learning.objectives_v2 import descent_terms,OBJECTIVE_VERSION
from stabilization_learning.common.contracts_v2 import checkpoint,restore,identity
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash,atomic_save


def quadratic(error,M):
    return torch.einsum("...i,...ij,...j->...",error,M,error)


@dataclass(frozen=True)
class ModelMetricConfig:
    seed:int=41
    updates:int=12000
    batch_states:int=128
    model_horizon:int=16
    learning_rate:float=1e-4
    decay:float=.05
    certificate_weight:float=.1
    validation_every:int=100
    checkpoint_every:int=25
    train_seed_start:int=1000000
    validation_seeds:tuple=tuple(range(210001,210017))
    test_seeds:tuple=tuple(range(310001,310097))

    def __post_init__(self):
        for name in ("updates","batch_states","model_horizon","validation_every","checkpoint_every"):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<=0:
                raise ValueError("Positive integer budgets required")
        if self.seed!=41:
            raise ValueError("This versioned new state-conditioned scope uses seed41 only")
        if set(self.validation_seeds)&set(self.test_seeds):
            raise ValueError("Validation and test seeds overlap")
        if self.train_seed_start<=max((*self.validation_seeds,*self.test_seeds)):
            raise ValueError("Training seed domain must be separate")
        if self.learning_rate<=0 or self.decay<0 or self.certificate_weight<0:
            raise ValueError("Invalid objective or optimizer configuration")


@torch.no_grad()
def rollout_batch(task,policy,starts,*,decay=.05,certificate_weight=.1,record=False,should_stop=None):
    """Original full horizon; physical model transitions, not a Gym rollout."""
    state=torch.as_tensor(starts,dtype=torch.float64).clone()
    count=len(state)
    active=torch.ones(count,dtype=torch.bool)
    failed=~active.clone();steps=torch.zeros(count,dtype=torch.long)
    costs=torch.zeros(count,dtype=torch.float64);penalties=costs.clone()
    rises=torch.zeros(count,dtype=torch.long);violations=rises.clone()
    e=task.error(state);terminal=e.square().sum(-1)
    last_bad=torch.where(terminal<=task.success_tolerance**2,-torch.ones(count,dtype=torch.long),torch.zeros(count,dtype=torch.long))
    buffers={key:[] for key in ("states","following","M_current","M_next","requested_actions","actions","V_before","V_after","costs","penalties","active","failure","in_goal","decay_residual")} if record else None
    before_hash=module_hash(policy)
    for step in range(task.max_steps):
        if should_stop is not None and should_stop():
            raise InterruptedError("STOP requested during a full-horizon query")
        if not active.any():break
        e=task.error(state);M=policy(e)
        requested=task.requested_control(M,e,state=state)
        action=torch.maximum(task.action_low,torch.minimum(task.action_high,requested))
        # Inactive lanes are kept at a finite safe point, never extrapolated.
        safe_state=torch.where(active[:,None],state,torch.as_tensor(task.goal))
        safe_action=torch.where(active[:,None],action,task.feedforward(safe_state))
        following=task.step_model(safe_state,safe_action)
        next_error=task.error(following);next_M=policy(next_error)
        v0=quadratic(e,M);v1=quadratic(next_error,next_M)
        norm=e.square().sum(-1);next_norm=next_error.square().sum(-1)
        terms=descent_terms(v1-v0,norm,dt=task.dt,decay=decay)
        bad=task.failure(following)
        if not torch.isfinite(following).all():
            raise FloatingPointError("Nonfinite state; refusing a misleading failure tail")
        increment=task.dt*norm
        costs+=torch.where(active,increment,0);penalties+=torch.where(active,terms["penalty"],0)
        rises+=active&((v1-v0)>1e-8);violations+=active&(terms["residual"]>1e-8)
        steps+=active;terminal=torch.where(active,next_norm,terminal)
        good=next_norm<=task.success_tolerance**2
        last_bad=torch.where(active&~good,torch.full_like(last_bad,step+1),last_bad)
        if record:
            values=dict(states=state,following=following,M_current=M,M_next=next_M,
                requested_actions=requested,actions=action,V_before=v0,V_after=v1,
                costs=increment,penalties=terms["penalty"],active=active,failure=bad,
                in_goal=good,decay_residual=terms["residual"])
            for key,value in values.items():buffers[key].append(value.numpy().copy())
        failed|=active&bad
        state=torch.where(active[:,None],following,state)
        active=active&~bad
    if module_hash(policy)!=before_hash:raise RuntimeError("Policy changed during the physical query")
    start=torch.where(last_bad<0,torch.zeros_like(last_bad),last_bad+1)
    hold=torch.clamp(steps-start,min=0).to(torch.float64)*task.dt
    success=(~failed)&(steps==task.max_steps)&(hold+1e-10>=task.required_hold_seconds)
    tail=torch.where(failed,terminal.clamp_min(25.),terminal)
    fixed=costs+(task.max_steps-steps)*task.dt*tail
    rows=[]
    for i in range(count):
        rows.append(dict(success=bool(success[i]),failure=bool(failed[i]),physical_steps=int(steps[i]),
            fixed_horizon_cost=float(fixed[i]),accumulated_cost=float(costs[i]),
            certificate_penalty=float(penalties[i]),objective=float(fixed[i]+certificate_weight*penalties[i]),
            terminal_physical_error=float(terminal[i]),hold_seconds=float(hold[i]),
            settling_time=float(start[i].to(torch.float64)*task.dt if success[i] else task.dt*task.max_steps),
            arrival_time=float(start[i].to(torch.float64)*task.dt+task.required_hold_seconds if success[i] else task.dt*task.max_steps),
            V_rise_fraction=float(rises[i]/steps[i].clamp_min(1)),
            decay_violation_fraction=float(violations[i]/steps[i].clamp_min(1)),
            metric_decisions=int(steps[i]),physical_terminated=bool(failed[i] or success[i]),
            physical_truncated=bool(not failed[i] and not success[i])))
    return rows,None if not record else {key:np.stack(values) for key,values in buffers.items()}


class ModelMetricLearner:
    """No task construction, no private P0 access; policy and task are injected."""
    def __init__(self,task,policy,config,*,source_digest):
        task.require_algorithm("state_metric_model_gradient_v2")
        self.task,self.policy,self.config=task,policy,config
        self.source_digest=source_digest
        self.optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate)
        self.updates=0;self.model_transitions=0;self.metric_decisions=0;self.validation_physical_steps=0
        self.active_model_transitions=0
        self.best_score=float("inf");self.best_actor=deepcopy(policy.state_dict());self.best_update=0
        self.history=[];self.should_stop=None
        torch.manual_seed(config.seed);np.random.seed(config.seed);random.seed(config.seed)

    def protocol(self):
        return dict(schema="state_metric_model_gradient_v2",config=asdict(self.config),
            task=self.task.describe().to_dict(),source_digest=self.source_digest,
            objective_version=OBJECTIVE_VERSION,full_successor_M=True,
            model_sampling="independent original reset states, 50% log-uniform near-target scales 1e-3..1; short differentiable windows",
            learning_mode="model-based pathwise gradient, not PPO",
            scope="state-M SDC gain; learned M is not an exact online Riccati solution",
            recovery_boundary="CPU whole optimizer update; no pending physical transaction",
            actual_execution="original pure tensor dynamics; no hardware or Gym rollout claim")

    def binding(self,weights):
        return VersionBinding(task=self.task.name,model=self.source_digest,representation="physical_state_identity_v1",
            normalization=identity(dict(scale=self.policy.context_scale,goal=self.policy.goal)),
            reference="fixed_original_goal",goal=identity(self.task.goal),certificate="full_state_quadratic_v2",
            controller=self.task.controller_form,protocol=identity(self.protocol()),weights=weights,
            action_units=self.task.description.action_description,metric_scope="state_dependent")

    def update(self):
        cfg=self.config
        ids=list(range(cfg.train_seed_start+self.updates*cfg.batch_states,
                       cfg.train_seed_start+(self.updates+1)*cfg.batch_states))
        states=self.task.initial_states(ids)
        scale=torch.ones(cfg.batch_states,dtype=torch.float64)
        scale[cfg.batch_states//2:]=torch.exp(-torch.rand(cfg.batch_states-cfg.batch_states//2,dtype=torch.float64)*np.log(1000.))
        goal=torch.as_tensor(self.task.goal,dtype=torch.float64)
        states=goal+(states-goal)*scale[:,None]
        active=torch.ones(cfg.batch_states,dtype=torch.bool)
        failure_tail=torch.zeros(cfg.batch_states,dtype=torch.float64)
        losses=[];costs=[];penalties=[];failure_count=0
        for k in range(cfg.model_horizon):
            e=self.task.error(states);M=self.policy(e)
            u=self.task.control(M,e,state=states)
            following=self.task.step_model(states,u)
            next_error=self.task.error(following);next_M=self.policy(next_error)
            terms=descent_terms(quadratic(next_error,next_M)-quadratic(e,M),
                e.square().sum(-1),dt=self.task.dt,decay=cfg.decay)
            physical=self.task.dt*e.square().sum(-1)
            bad=self.task.failure(following)
            if not torch.isfinite(following).all():raise FloatingPointError("Nonfinite model-gradient trajectory")
            capped=self.task.dt*torch.maximum(next_error.square().sum(-1),next_error.new_tensor(25.))
            stage=torch.where(active,physical+cfg.certificate_weight*terms["penalty"],failure_tail)
            losses.append(stage.mean());costs.append(physical.detach().mean());penalties.append(terms["penalty"].detach().mean())
            self.active_model_transitions+=int(active.sum())
            self.metric_decisions+=int(active.sum())
            self.model_transitions+=cfg.batch_states
            failure_tail=torch.where(active&bad,capped,failure_tail)
            failure_count+=int((active&bad).sum());active=active&~bad
            states=torch.where(active[:,None],following,goal)
        loss=torch.stack(losses).mean()
        self.optimizer.zero_grad();loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.policy.parameters()):
            raise FloatingPointError("Nonfinite parameter gradient")
        grad=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.)
        self.optimizer.step();self.updates+=1
        return dict(update=self.updates,loss=float(loss.detach()),physical_model_stage_cost=float(torch.stack(costs).mean()),
            certificate_penalty=float(torch.stack(penalties).mean()),gradient_norm=float(grad),failure_lanes=failure_count,
            model_transitions=self.model_transitions,active_model_transitions=self.active_model_transitions,
            metric_decisions=self.metric_decisions)

    def validate(self):
        rows,_=rollout_batch(self.task,self.policy,self.task.initial_states(self.config.validation_seeds),
            decay=self.config.decay,certificate_weight=self.config.certificate_weight,should_stop=self.should_stop)
        self.validation_physical_steps+=sum(row["physical_steps"] for row in rows)
        objective=float(np.mean([r["objective"] for r in rows]))
        failures=sum(not r["success"] for r in rows)
        score=failures*1e6+objective
        if score<self.best_score:
            self.best_score=score;self.best_update=self.updates;self.best_actor=deepcopy(self.policy.state_dict())
        return dict(update=self.updates,score=score,non_success=failures,
            physical_cost=float(np.mean([r["fixed_horizon_cost"] for r in rows])),
            penalty=float(np.mean([r["certificate_penalty"] for r in rows])),episodes=rows)

    def save(self,path):
        counters=dict(updates=self.updates,model_transitions=self.model_transitions,active_model_transitions=self.active_model_transitions,
            metric_decisions=self.metric_decisions,
            validation_physical_steps=self.validation_physical_steps,best_score=self.best_score,best_actor=self.best_actor,
            best_update=self.best_update,history=self.history)
        payload=checkpoint(modules={"policy":self.policy},optimizers={"Adam":self.optimizer},
            binding=self.binding(module_hash(self.policy)),counters=counters)
        payload["protocol"]=self.protocol();atomic_save(payload,path)

    def restore(self,path):
        payload=torch.load(path,map_location="cpu",weights_only=False)
        if identity(payload["protocol"])!=identity(self.protocol()):raise ValueError("Training protocol/source mismatch")
        expected=self.binding(payload["binding"]["weights"])
        counters,_=restore(payload,modules={"policy":self.policy},optimizers={"Adam":self.optimizer},binding=expected)
        if module_hash(self.policy)!=expected.weights:raise ValueError("Checkpoint weights do not match their recorded identity")
        for key,value in counters.items():setattr(self,key,value)

    def train(self,output,*,emit,resume=False,max_updates=None):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if resume:self.restore(output/"last.pt")
        elif (output/"last.pt").exists():raise FileExistsError("Use explicit resume with the exact same protocol")
        if not self.history:
            self.history.append(dict(kind="validation",**self.validate()));self.save(output/"last.pt")
        stop_at=self.config.updates if max_updates is None else min(self.config.updates,self.updates+max_updates)
        while self.updates<stop_at:
            if self.should_stop and self.should_stop():
                self.save(output/"last.pt");return False
            started=time.perf_counter();row=self.update()
            row["wall_seconds"]=time.perf_counter()-started
            self.history.append(dict(kind="training",**row))
            validation=None
            if self.updates%self.config.validation_every==0 or self.updates==self.config.updates:
                validation=self.validate();self.history.append(dict(kind="validation",**validation))
            if self.updates%self.config.checkpoint_every==0 or self.updates==stop_at:
                self.save(output/"last.pt")
                emit(task=self.task.name,stage="training",update=self.updates,total_updates=self.config.updates,
                    model_transitions=self.model_transitions,best_update=self.best_update,
                    validation=None if validation is None else {k:v for k,v in validation.items() if k!="episodes"},
                    update_wall_seconds=row["wall_seconds"])
        return self.updates>=self.config.updates
