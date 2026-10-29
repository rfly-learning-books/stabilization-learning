"""Independent NumPy recomputation for the direct-gain four-task experiment.

No environment, optimizer, learner, or training runner is imported here.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import solve_continuous_are


def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))


def model(task):
    if task in ('coupled_linear','point_mass'):
        A = np.block([[np.zeros((2,2)),np.eye(2)],[np.zeros((2,4))]])
        B = np.vstack([np.zeros((2,2)),np.eye(2)])
        if task == 'coupled_linear':
            A[2:,:2] = -np.array([[2.,-.6],[-.6,1.5]])
            A[2:,2:] = -np.array([[.25,.03],[.03,.3]])
        return dict(A=A,B=B,Q=np.diag([4.,4.,1.,1.]),R=np.eye(2)*.5,
            dt=.04,horizon=300,initial_scale=.7,tolerance=.05,low=np.full(2,-8. if task=='coupled_linear' else -6.),
            high=np.full(2,8. if task=='coupled_linear' else 6.))
    if task == 'pendulum_local':
        return dict(A=np.array([[0.,1.],[9.81,-.15]]),B=np.array([[0.],[1.]]),
            Q=np.diag([12.,1.]),R=np.array([[.5]]),dt=.02,horizon=400,initial_scale=.15,
            tolerance=.04,low=np.array([-8.]),high=np.array([8.]))
    if task == 'cartpole_local':
        total,den = 1.1,.5*(4./3.-.1/1.1)
        A=np.zeros((4,4));A[0,1]=A[2,3]=1.
        A[1,2]=-.1*.5*9.81/(total*den);A[3,2]=9.81/den
        B=np.array([[0.],[1/total+.1*.5/(total**2*den)],[0.],[-1/(total*den)]])
        return dict(A=A,B=B,Q=np.diag([2.,.5,16.,1.]),R=np.array([[.1]]),
            dt=.02,horizon=500,initial_scale=.08,tolerance=.04,low=np.array([-10.]),high=np.array([10.]))
    raise ValueError(task)


def errors(task,states):
    e=np.array(states,dtype=float,copy=True)
    if task=='pendulum_local':e[...,0]=wrap(e[...,0])
    elif task=='cartpole_local':e[...,2]=wrap(e[...,2])
    return e


def failure_flags(task,states):
    result=~np.isfinite(states).all(-1)|(np.linalg.norm(states,axis=-1)>100.)
    if task=='pendulum_local':result|=(np.abs(wrap(states[...,0]))>np.pi/2)|(np.abs(states[...,1])>12.)
    elif task=='cartpole_local':result|=(np.abs(states[...,0])>2.4)|(np.abs(wrap(states[...,2]))>np.pi/4)
    return result


def step(task,state,action,dt):
    s=np.asarray(state,float);u=np.asarray(action,float)
    if task=='point_mass':
        return np.concatenate((s[...,:2]+dt*s[...,2:]+.5*dt**2*u,s[...,2:]+dt*u),axis=-1)
    def flow(x):
        if task=='coupled_linear':
            return np.concatenate((x[...,2:],-x[...,:2]@np.array([[2.,-.6],[-.6,1.5]]).T
                -x[...,2:]@np.array([[.25,.03],[.03,.3]]).T+u),axis=-1)
        if task=='pendulum_local':
            return np.stack((x[...,1],9.81*np.sin(x[...,0])-.15*x[...,1]+u[...,0]),axis=-1)
        sine,cosine=np.sin(x[...,2]),np.cos(x[...,2])
        temp=(u[...,0]+.05*x[...,3]**2*sine)/1.1
        angular=(9.81*sine-cosine*temp)/(.5*(4./3.-.1*cosine**2/1.1))
        return np.stack((x[...,1],temp-.05*angular*cosine/1.1,x[...,3],angular),axis=-1)
    a=flow(s);b=flow(s+.5*dt*a);c=flow(s+.5*dt*b);d=flow(s+dt*c)
    return s+dt/6*(a+2*b+2*c+d)


def measurements(task,z,*,decay=.05,certificate_weight=.1):
    cfg=model(task)
    states=np.asarray(z['states']);following=np.asarray(z['following']);actions=np.asarray(z['actions'])
    M=np.asarray(z['M']);n=len(states);dt=float(z['dt']);d=states.shape[-1]
    e=errors(task,states);next_e=errors(task,following)
    P0=solve_continuous_are(cfg['A'],cfg['B'],cfg['Q'],cfg['R'])
    L=np.linalg.cholesky(P0)
    S=np.linalg.solve(L,np.linalg.solve(L,M).T).T
    S=.5*(S+S.T)
    spectrum=np.linalg.eigvalsh(S)
    commanded=-e@M@cfg['B']@np.linalg.inv(cfg['R'])
    commanded=np.clip(commanded,cfg['low'],cfg['high'])
    stepped=step(task,states,actions,dt)
    before=np.einsum('bi,ij,bj->b',e,M,e)
    after=np.einsum('bi,ij,bj->b',next_e,M,next_e)
    norm2=np.sum(e**2,axis=-1);next_norm2=np.sum(next_e**2,axis=-1)
    all_states=np.concatenate((states[:1],following))
    all_errors=errors(task,all_states)
    good=np.sum(all_errors**2,axis=-1)<=cfg['tolerance']**2
    failed=failure_flags(task,following)
    last_bad=int(np.flatnonzero(~good)[-1]) if (~good).any() else -1
    start=last_bad+1 if last_bad>=0 else 0
    hold=max(0.,(n-start)*dt)
    failure=bool(failed[-1])
    success=bool(n==cfg['horizon'] and not failure and hold+1e-10>=1.)
    cost=dt*norm2
    accumulated=float(cost.sum());terminal=float(next_norm2[-1])
    fixed=accumulated+(cfg['horizon']-n)*dt*(max(terminal,25.) if failure else terminal)
    delta=after-before
    residual=delta+decay*dt*norm2
    scale=np.maximum(norm2,1e-6)
    penalty=float(np.sum(np.maximum(0.,delta+decay*dt*scale)/scale))
    eig=np.linalg.eigvalsh(M)
    out=dict(fixed_horizon_cost=fixed,accumulated_cost=accumulated,
        objective=fixed+certificate_weight*penalty,certificate_penalty=penalty,
        V_rise_fraction=float(np.mean(delta>1e-8)),success=success,failure=failure,
        physical_steps=n,arrival_time=float(start*dt+1. if success else cfg['horizon']*dt),
        settling_time=float(start*dt if success else cfg['horizon']*dt),hold_seconds=float(hold),
        required_hold_seconds=1.,terminal_physical_error=terminal,
        actual_decay_violation_fraction=float(np.mean(residual>1e-8)),
        actual_decay_violation_max=float(max(0.,residual.max())),
        action_total_variation=float(np.linalg.norm(np.diff(actions,axis=0),axis=-1).sum()),
        horizon_seconds=cfg['horizon']*dt,M_eigen_min=float(eig.min()),M_eigen_max=float(eig.max()),M_trace=float(np.trace(M)),
        physical_terminated=bool(failure or success),physical_truncated=bool(not failure and not success))
    checks={
        'nonempty_finite': bool(0<n<=cfg['horizon'] and all(np.isfinite(z[k]).all() for k in
            ('M','states','following','actions','errors','errors_after','costs'))),
        'time_step':math.isclose(dt,cfg['dt'],abs_tol=1e-14),
        'state_chain':np.array_equal(states[1:],following[:-1]),
        'actual_action':np.allclose(commanded,actions,atol=1e-9,rtol=1e-8),
        'independent_physics':np.allclose(stepped,following,atol=1e-10,rtol=1e-9),
        'error_before':np.allclose(e,z['errors'],atol=1e-12,rtol=1e-10),
        'error_after':np.allclose(next_e,z['errors_after'],atol=1e-12,rtol=1e-10),
        'V_before':np.allclose(before,z['V_before'],atol=1e-9,rtol=1e-8),
        'V_after':np.allclose(after,z['V_after'],atol=1e-9,rtol=1e-8),
        'physical_cost':np.allclose(cost,z['costs'],atol=1e-12,rtol=1e-10),
        'physical_error_before':np.allclose(norm2,z['physical_error_before'],atol=1e-12,rtol=1e-10),
        'physical_error_after':np.allclose(next_norm2,z['physical_error_after'],atol=1e-12,rtol=1e-10),
        'goal_before':np.array_equal(good[:-1],z['in_goal_before']),
        'goal_after':np.array_equal(good[1:],z['in_goal_after']),
        'full_horizon_or_failure':n==cfg['horizon'] or failure,
        'no_steps_after_failure':not bool(failed[:-1].any()),
        'metric_symmetric':np.allclose(M,M.T,atol=1e-10,rtol=1e-10),
        'physical_metric_positive':eig.min()>0,
        'shape_metric_spectrum':bool(spectrum.min()>=.05-2e-6 and spectrum.max()<=d-(d-1)*.05+2e-6),
        'shape_metric_trace':abs(np.trace(S)-d)<2e-5,
        'saved_shape_metric':np.allclose(S,z['S'],atol=2e-8,rtol=1e-8),
        'baseline_P0':np.allclose(P0,z['P0'],atol=1e-10,rtol=1e-10),
        'decay_residual':np.allclose(residual,z['decay_residual'],atol=1e-9,rtol=1e-8),
    }
    audit=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        max_controller_action_abs_error=float(np.max(np.abs(actions-commanded))),
        max_physical_step_abs_error=float(np.max(np.abs(stepped-following))))
    return out,audit


def local_certificate(task,M):
    cfg=model(task);A,B=cfg['A'],cfg['B'];K=np.linalg.solve(cfg['R'],B.T@M)
    C=A-B@K;dt=cfg['dt'];d=len(A)
    if task=='point_mass':
        Ad=np.eye(d)+dt*A
        Bd=np.vstack([.5*dt**2*np.eye(2),dt*np.eye(2)])
        discrete=Ad-Bd@K
    else:
        # The original RK4 holds the sampled action fixed throughout all stages.
        Ad=np.eye(d)+dt*A+dt**2/2*(A@A)+dt**3/6*np.linalg.matrix_power(A,3)+dt**4/24*np.linalg.matrix_power(A,4)
        Bd=(dt*np.eye(d)+dt**2/2*A+dt**3/6*(A@A)+dt**4/24*np.linalg.matrix_power(A,3))@B
        discrete=Ad-Bd@K
    continuous=C.T@M+M@C
    difference=discrete.T@M@discrete-M
    return dict(scope='Nominal unsaturated linear model; nonlinear tasks use local linearization only; numerical, not interval/exact proof',
        global_claim=False,continuous_max_real_eigenvalue=float(np.max(np.linalg.eigvals(C).real)),
        discrete_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(discrete)))),
        same_M_continuous_descent_max_eigenvalue=float(np.linalg.eigvalsh(.5*(continuous+continuous.T)).max()),
        same_M_discrete_descent_max_eigenvalue=float(np.linalg.eigvalsh(.5*(difference+difference.T)).max()),
        same_M_discrete_descent=bool(np.linalg.eigvalsh(.5*(difference+difference.T)).max()<-1e-10),
        A_closed_loop_discrete=discrete.tolist())
