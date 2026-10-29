"""Bounded pre-training acceptance: baseline, gradients, parity and resume."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from .core import TASKS,DirectLearner,SparseValidationConfig,batch_rollout,initial_states
from . import audit
from stabilization_learning.wrappers import rollout_metric


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def run_checks(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    attempt=directory/f'attempt_{time.time_ns()}';attempt.mkdir()
    checks=[];started=time.perf_counter()
    def check(name,passed,**detail):
        checks.append(dict(name=name,passed=bool(passed),**detail))
        if not passed:
            (attempt/'FAILED.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
            raise AssertionError(name+': '+str(detail))
    for task in TASKS:
        learner=DirectLearner(task)
        p=learner.physical;inner=p.inner_env
        baseline=learner.baseline
        check(task+'_P0_recovery',np.allclose(baseline.numpy(),inner.P0,atol=1e-12,rtol=1e-12))
        gain=np.linalg.solve(inner.R0,inner.B0.T@baseline.numpy())
        check(task+'_original_K_recovery',np.allclose(gain,inner.env._K.numpy(),atol=1e-12,rtol=1e-12))
        residual=inner.A0.T@inner.P0+inner.P0@inner.A0-inner.P0@inner.B0@np.linalg.solve(inner.R0,inner.B0.T@inner.P0)+inner.Q0
        check(task+'_Riccati_residual',np.max(np.abs(residual))<1e-9,max_abs=float(np.max(np.abs(residual))))
        raw=learner.actor.center.double()+.03*torch.arange(len(learner.actor.center),dtype=torch.float64)
        raw.requires_grad_()
        state=torch.arange(1,inner.error_dim+1,dtype=torch.float64)*.003
        error=inner.tensor_error(state).requires_grad_()
        context={'state':state,'goal':torch.zeros(inner.error_dim,dtype=torch.float64)}
        action=lambda r:inner.calculate_action_from_M(learner.actor.metric(r),error,context=context)
        jac=torch.autograd.functional.jacobian(action,raw)
        position=np.unravel_index(int(jac.abs().argmax()),tuple(jac.shape))
        direction=torch.zeros_like(raw);direction[position[1]]=1.
        numerical=(action(raw+1e-6*direction)[position[0]]-action(raw-1e-6*direction)[position[0]])/2e-6
        discrepancy=float(abs(numerical-jac[position]))
        check(task+'_raw_to_actual_action_gradient',discrepancy<1e-7 and float(jac.abs().max())>1e-6,fd_error=discrepancy)
        M=learner.actor.metric(raw)
        V=error@M@error
        gradient,=torch.autograd.grad(V,error,retain_graph=True)
        from_gradient=-.5*torch.as_tensor(np.linalg.inv(inner.R0)@inner.B0.T)@gradient
        check(task+'_same_M_full_fixed_metric_gradient_control',torch.allclose(action(raw),from_gradient,atol=1e-12,rtol=1e-12))
        starts=initial_states(p,[203001,203002])
        matrices=torch.stack((baseline,M.detach()))
        rows,arrays=batch_rollout(p,matrices,starts,record=True)
        for index,seed in enumerate((203001,203002)):
            matrix=matrices[index]
            observation,_=p.reset(seed=seed)
            # Same frozen reset distribution as the independent vector executor.
            check(f'{task}_seed{seed}_reset_parity',np.array_equal(observation,starts[index]))
            scalar=rollout_metric(p,matrix,observation,certificate_weight=.1,decay=.05)
            # The old generic wrapper reports episode duration as arrival. The
            # original full-horizon task owns the corrected final-hold arrival.
            scalar['arrival_time']=p._last_info['arrival_time']
            row=rows[index];trace=p.trace;n=row['physical_steps']
            check(f'{task}_seed{seed}_scalar_metrics',all(
                np.isclose(row[key],scalar[key],atol=1e-9,rtol=1e-8) for key in
                ('fixed_horizon_cost','objective','certificate_penalty','success','failure','physical_steps','arrival_time')),
                compared={key:[row[key],scalar[key]] for key in ('objective','success','arrival_time')})
            state_difference=float(np.max(np.abs(arrays['following'][:n,index]-np.array([r['state_after'] for r in trace]))))
            action_difference=float(np.max(np.abs(arrays['actions'][:n,index]-np.array([r['action'] for r in trace]))))
            check(f'{task}_seed{seed}_scalar_full_trajectory',state_difference<1e-10 and action_difference<1e-9,
                state_max_abs=state_difference,action_max_abs=action_difference)
            z={key:value[:n,index] for key,value in arrays.items()}
            shape=learner.actor.metric.shape(learner.actor.center if index==0 else raw).detach().numpy()
            z.update(M=matrix.numpy(),S=shape,P0=inner.P0,dt=np.array(p.dt))
            measured,checked=audit.measurements(task,z)
            check(f'{task}_seed{seed}_independent_numpy_audit',checked['passed'],checks=checked['checks'])
        learner.physical.close()
    # An explicit one-step failed episode must keep the original full-horizon tail.
    probe=DirectLearner('cartpole_local')
    starts=initial_states(probe.physical,[203003,203004]);starts[0,0]=2.41
    rows,_=batch_rollout(probe.physical,probe.baseline,starts)
    check('mixed_batch_failure_stops_individually',rows[0]['failure'] and rows[0]['physical_steps']==1 and rows[1]['physical_steps']==500)
    check('failure_keeps_full_horizon_tail',rows[0]['fixed_horizon_cost']>200.)
    probe.physical.close()
    def mini(interval=5):
        return DirectLearner('point_mass',config=SparseValidationConfig(episodes=12,batch_episodes=3,
            epochs=1,validation_every_updates=interval,validation_seeds=(10001,10002)),max_steps=8)
    continuous=mini();initial=deepcopy(continuous.actor.state_dict())
    continuous.train(attempt/'continuous')
    check('real_actor_update',not same(initial,continuous.actor.state_dict()))
    check('real_optimizer_state',bool(continuous.optimizer.state_dict()['state']))
    first=mini();first.train(attempt/'resumed',max_updates=2)
    resumed=mini();resumed.train(attempt/'resumed',resume=True)
    for component in ('actor','value','optimizer','value_optimizer'):
        check('resume_exact_'+component,same(getattr(continuous,component).state_dict(),getattr(resumed,component).state_dict()))
    check('resume_exact_history',same(continuous.history,resumed.history))
    check('resume_exact_selection',same(continuous.best_actor,resumed.best_actor) and continuous.best_score==resumed.best_score)
    incremental=mini()
    for update in range(4):
        incremental.train(attempt/'incremental',resume=update>0,max_updates=1)
    check('launcher_incremental_calls_exact',same(continuous.history,incremental.history)
        and same(continuous.actor.state_dict(),incremental.actor.state_dict())
        and same(continuous.optimizer.state_dict(),incremental.optimizer.state_dict()))
    check('only_validated_rows_selected',[r['updates'] for r in continuous.history if r['validated']]==[0,4]
        and all(r['validation'] is None for r in continuous.history[1:4]))
    rejected=False
    try:mini(2).restore(attempt/'resumed/last.pt')
    except ValueError:rejected=True
    check('changed_protocol_resume_rejected',rejected)
    result=dict(passed=all(row['passed'] for row in checks),checks=checks,
        wall_seconds=time.perf_counter()-started,formal_training_started=False,
        debug_optimizer_updates=12,debug_episodes=36,
        scope='Four-task baseline recovery; differentiable same-M control; full scalar/batch physics, costs and holds; independent NumPy audit; failure tails; nonempty optimizer resume')
    (attempt/'PREFLIGHT.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result
