"""Independent NumPy policy/control audit, reusing sealed physics equations."""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_continuous_are
from extensions.lqr_metric_direct_v1.audit import model,errors,step,failure_flags

ALIASES={"pendulum":"pendulum_local","cartpole":"cartpole_local"}


def coefficients(name,error):
    key=ALIASES.get(name,name);cfg=model(key)
    e=np.asarray(error);shape=e.shape[:-1];n=e.shape[-1]
    A=np.broadcast_to(cfg["A"],(*shape,n,n)).copy()
    B=np.broadcast_to(cfg["B"],(*shape,*cfg["B"].shape)).copy()
    if name=="pendulum":
        A[...,1,0]=9.81*np.sinc(e[...,0]/np.pi)
    elif name=="cartpole":
        theta,omega=e[...,2],e[...,3];s,c=np.sin(theta),np.cos(theta)
        total,moment=1.1,.05;den=.5*(4/3-.1*c*c/total)
        A[...,1,2]=-moment*c*9.81*np.sinc(theta/np.pi)/(total*den)
        A[...,1,3]=moment*omega*s/total*(1+moment*c*c/(total*den))
        A[...,3,2]=9.81*np.sinc(theta/np.pi)/den
        A[...,3,3]=-c*moment*omega*s/(total*den)
        B[...,1,0]=1/total+moment*c*c/(total*total*den)
        B[...,3,0]=-c/(total*den)
    return A,B


def numpy_policy(error,weights,spec):
    x=np.asarray(error,dtype=np.float64)
    goal=np.broadcast_to(weights["goal"],x.shape)
    z=np.concatenate((x/weights["context_scale"],goal/weights["context_scale"]),axis=-1)
    z=np.tanh(z@weights["net.0.weight"].T+weights["net.0.bias"])
    z=np.tanh(z@weights["net.2.weight"].T+weights["net.2.bias"])
    raw=z@weights["net.4.weight"].T+weights["net.4.bias"]+weights["center"]
    rows=weights["shape_metric.rows"].astype(int);cols=weights["shape_metric.cols"].astype(int)
    diagonal=weights["shape_metric.diagonal"].astype(bool)
    values=np.where(diagonal,np.logaddexp(0.,raw)+1e-8,raw)
    n=x.shape[-1]
    L=np.zeros((*x.shape[:-1],n,n));L[...,rows,cols]=values
    L=L/np.maximum(np.abs(L).max(axis=(-2,-1),keepdims=True),1e-8)
    gram=L@np.swapaxes(L,-1,-2)
    tr=np.trace(gram,axis1=-2,axis2=-1)[...,None,None]
    S=spec["floor"]*np.eye(n)+(spec["trace"]-n*spec["floor"])*gram/np.maximum(tr,1e-20)
    return weights["basis"]@S@weights["basis"].T


def audit_group(name,arrays,rows,*,role,weights,decay=.05,certificate_weight=.1):
    key=ALIASES.get(name,name);cfg=model(key)
    state=arrays["states"];following=arrays["following"];valid=arrays["active"].astype(bool)
    e=errors(key,state);en=errors(key,following)
    d=e.shape[-1]
    P0=solve_continuous_are(cfg["A"],cfg["B"],cfg["Q"],cfg["R"])
    if role=="fixed_P0":
        M=np.broadcast_to(P0,arrays["M_current"].shape)
        M1=np.broadcast_to(P0,arrays["M_next"].shape)
    elif role=="classical_SDRE":
        A,B=coefficients(name,e.reshape(-1,d))
        A1,B1=coefficients(name,en.reshape(-1,d))
        M=np.stack([solve_continuous_are(a,b,cfg["Q"],cfg["R"]) for a,b in zip(A,B)]).reshape(arrays["M_current"].shape)
        M1=np.stack([solve_continuous_are(a,b,cfg["Q"],cfg["R"]) for a,b in zip(A1,B1)]).reshape(arrays["M_next"].shape)
    else:
        spec=dict(floor=.05,trace=float(d))
        M=numpy_policy(e,weights,spec);M1=numpy_policy(en,weights,spec)
    _,B=coefficients(name,e)
    requested=-np.linalg.solve(cfg["R"],np.swapaxes(B,-1,-2)@(M@e[...,None]))[...,0]
    # All four registered default references in this protocol are zero.
    commands=np.clip(requested,cfg["low"],cfg["high"])
    stepped=step(key,state,commands,cfg["dt"])
    v0=np.einsum("...i,...ij,...j->...",e,M,e)
    v1=np.einsum("...i,...ij,...j->...",en,M1,en)
    norm=(e*e).sum(-1);next_norm=(en*en).sum(-1)
    residual=v1-v0+decay*cfg["dt"]*norm
    penalties=np.maximum(0,residual)/np.maximum(norm,1e-6)
    costs=cfg["dt"]*norm
    close=lambda a,b,atol=1e-9,rtol=1e-8:bool(np.allclose(np.asarray(a)[valid],np.asarray(b)[valid],atol=atol,rtol=rtol))
    checks=dict(policy_M=close(M,arrays["M_current"]),successor_M=close(M1,arrays["M_next"]),
        requested_action=close(requested,arrays["requested_actions"]),actual_action=close(commands,arrays["actions"]),
        physical_model=close(stepped,following,1e-10,1e-9),V_before=close(v0,arrays["V_before"]),
        V_after=close(v1,arrays["V_after"]),physical_cost=close(costs,arrays["costs"],1e-12,1e-10),
        penalty=close(penalties,arrays["penalties"]),actual_residual=close(residual,arrays["decay_residual"]))
    bad=failure_flags(key,following)
    checks["failure_flags"]=bool(np.array_equal(bad[valid],arrays["failure"][valid]))
    numerical=[]
    for lane,row in enumerate(rows):
        mask=valid[:,lane];n=int(mask.sum())
        state_i=state[mask,lane];next_i=following[mask,lane]
        if n>1 and not np.array_equal(state_i[1:],next_i[:-1]):
            checks["state_chain"]=False
        good=(errors(key,np.concatenate((state_i[:1],next_i)))**2).sum(-1)<=cfg["tolerance"]**2
        last_bad=int(np.flatnonzero(~good)[-1]) if (~good).any() else -1
        start=last_bad+1 if last_bad>=0 else 0
        hold=max(0.,(n-start)*cfg["dt"])
        failure=bool(bad[mask,lane][-1])
        success=bool(n==cfg["horizon"] and not failure and hold+1e-10>=1.)
        accumulated=float(costs[mask,lane].sum());terminal=float(next_norm[mask,lane][-1])
        fixed=accumulated+(cfg["horizon"]-n)*cfg["dt"]*(max(terminal,25.) if failure else terminal)
        p=float(penalties[mask,lane].sum())
        expected=dict(success=success,failure=failure,physical_steps=n,accumulated_cost=accumulated,
            fixed_horizon_cost=fixed,certificate_penalty=p,objective=fixed+certificate_weight*p,
            terminal_physical_error=terminal,hold_seconds=hold)
        for field,value in expected.items():
            match=(row[field]==value) if isinstance(value,(bool,int)) else np.isclose(row[field],value,atol=1e-8,rtol=1e-8)
            if not match:checks["row_"+str(lane)+"_"+field]=False
        if n<cfg["horizon"] and not failure:checks["no_unexplained_early_stop"]=False
        numerical.append(expected)
    return dict(passed=all(checks.values()),checks=checks,episodes=len(rows),
        max_action_abs_error=float(np.max(np.abs(commands[valid]-arrays["actions"][valid]))),
        max_physics_abs_error=float(np.max(np.abs(stepped[valid]-following[valid]))),
        maximum_M_abs_error=float(np.max(np.abs(M[valid]-arrays["M_current"][valid]))),
        guarantee="candidate and finite-horizon simulation only; no new global or nonlinear-region proof")
