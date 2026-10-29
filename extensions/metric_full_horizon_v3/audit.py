"""Independent equations and policy reconstruction; no learner imports."""
import numpy as np
from scipy.linalg import solve_continuous_are
from extensions.lqr_metric_direct_v1.audit import model, errors, step, failure_flags
from extensions.book_aligned_m_v2.audit import coefficients

ALIASES = {"pendulum":"pendulum_local", "cartpole":"cartpole_local"}


def metric(e, w):
    z = e / w["context_scale"]
    radius2 = (z*z).sum(-1,keepdims=True)
    x = np.concatenate((z,np.broadcast_to(w["goal"]/w["context_scale"],z.shape)),axis=-1)
    x = np.tanh(x @ w["net.0.weight"].T + w["net.0.bias"])
    x = np.tanh(x @ w["net.2.weight"].T + w["net.2.bias"])
    raw = w["center"] + radius2/(1+radius2)*w["residual_limit"]*np.tanh(x @ w["net.4.weight"].T+w["net.4.bias"])
    n=e.shape[-1]
    L=np.zeros((*e.shape[:-1],n,n))
    values=np.where(w["shape_metric.diagonal"].astype(bool),np.logaddexp(0.,raw)+1e-8,raw)
    L[...,w["shape_metric.rows"].astype(int),w["shape_metric.cols"].astype(int)]=values
    L=L/np.maximum(np.abs(L).max(axis=(-2,-1),keepdims=True),1e-8)
    gram=L@np.swapaxes(L,-1,-2)
    S=.05*np.eye(n)+(n-n*.05)*gram/np.maximum(np.trace(gram,axis1=-2,axis2=-1)[...,None,None],1e-20)
    return w["basis"]@S@w["basis"].T


def audit_group(name, arrays, rows, *, role, weights, seeds, scale, decay=.05):
    key=ALIASES.get(name,name);cfg=model(key)
    x=arrays["states"];y=arrays["following"];valid=arrays["active"].astype(bool)
    e=errors(key,x);en=errors(key,y);n=e.shape[-1]
    P0=solve_continuous_are(cfg["A"],cfg["B"],cfg["Q"],cfg["R"])
    checks={}
    expected_starts=np.stack([np.random.default_rng(int(s)).uniform(-1,1,n)*cfg["initial_scale"]*scale for s in seeds])
    checks["declared_initial_states"]=bool(np.array_equal(expected_starts,x[0]))
    if role=="fixed_P0":
        M=np.broadcast_to(P0,arrays["M_current"].shape);Mn=np.broadcast_to(P0,arrays["M_next"].shape)
    elif role=="classical_SDRE":
        # Independently certify the saved algebraic solution and stabilizing
        # branch. This avoids rerunning the same ARE solver as the executor.
        M=arrays["M_current"];Mn=arrays["M_next"]
        for label,ee,mm in (("current",e,M),("next",en,Mn)):
            A,B=coefficients(name,ee)
            gain=np.linalg.solve(cfg["R"],np.swapaxes(B,-1,-2)@mm)
            residual=np.swapaxes(A,-1,-2)@mm+mm@A-mm@B@gain+cfg["Q"]
            eig=np.linalg.eigvals(A-B@gain)
            checks["ARE_"+label]=bool(np.max(np.abs(residual[valid]))<1e-7)
            checks["stabilizing_"+label]=bool(np.max(eig[valid].real)<0)
    else:
        M=metric(e,weights);Mn=metric(en,weights)
    _,B=coefficients(name,e)
    requested=-np.linalg.solve(cfg["R"],np.swapaxes(B,-1,-2)@(M@e[...,None]))[...,0]
    u=np.clip(requested,cfg["low"],cfg["high"])
    predicted=step(key,x,u,cfg["dt"])
    v0=np.einsum("...i,...ij,...j->...",e,M,e)
    v1=np.einsum("...i,...ij,...j->...",en,Mn,en)
    norm=(e*e).sum(-1);nnorm=(en*en).sum(-1)
    residual=v1-v0+decay*cfg["dt"]*norm
    rate=residual/(cfg["dt"]*np.maximum(norm,1e-18))
    bad=failure_flags(key,y)
    def close(a,b,atol=1e-9,rtol=1e-8):
        return bool(np.allclose(np.asarray(a)[valid],np.asarray(b)[valid],atol=atol,rtol=rtol))
    checks.update(current_metric=close(M,arrays["M_current"]),successor_metric=close(Mn,arrays["M_next"]),
        requested_control=close(requested,arrays["requested_actions"]),physical_action=close(u,arrays["actions"]),
        physics=close(predicted,y,1e-10,1e-9),V_current=close(v0,arrays["V_before"]),
        V_successor=close(v1,arrays["V_after"]),residual=close(residual,arrays["decay_residual"]),
        failure=bool(np.array_equal(bad[valid],arrays["failure"][valid])),
        metric_positive=bool(np.min(np.linalg.eigvalsh(M[valid]))>0))
    for i,row in enumerate(rows):
        mask=valid[:,i];steps=int(mask.sum());xx=x[mask,i];yy=y[mask,i]
        checks[f"chain_{i}"]=bool(steps==1 or np.array_equal(xx[1:],yy[:-1]))
        good=(errors(key,np.concatenate((xx[:1],yy)))**2).sum(-1)<=cfg["tolerance"]**2
        last=int(np.flatnonzero(~good)[-1]) if (~good).any() else -1
        start=last+1;hold=max(0.,(steps-start)*cfg["dt"])
        fail=bool(bad[mask,i][-1]);success=bool(not fail and steps==cfg["horizon"] and hold+1e-10>=1.)
        accumulated=float(cfg["dt"]*norm[mask,i].sum());terminal=float(nnorm[mask,i][-1])
        fixed=accumulated+(cfg["horizon"]-steps)*cfg["dt"]*(max(25.,terminal) if fail else terminal)
        penalty=float((np.maximum(residual[mask,i],0)/np.maximum(norm[mask,i],1e-6)).sum())
        expected=dict(success=success,failure=fail,physical_steps=steps,hold_seconds=hold,
            accumulated_cost=accumulated,fixed_horizon_cost=fixed,terminal_physical_error=terminal,
            certificate_penalty=penalty,objective=fixed+.1*penalty,
            arrival_time=start*cfg["dt"]+1 if success else cfg["horizon"]*cfg["dt"],
            control_energy=float(cfg["dt"]*(u[mask,i]**2).sum()),
            max_normalized_decay_rate=float(rate[mask,i].max()),
            decay_violation_fraction=float(np.mean(rate[mask,i]>1e-6)))
        for field,value in expected.items():
            checks[f"episode_{i}_{field}"]=bool(row[field]==value if isinstance(value,(bool,int)) else
                np.isclose(row[field],value,atol=1e-8,rtol=1e-8))
    return dict(passed=all(checks.values()),checks=checks,episodes=len(rows),
        max_physics_error=float(np.max(np.abs(predicted[valid]-y[valid]))),
        max_metric_error=float(np.max(np.abs(M[valid]-arrays["M_current"][valid]))),
        scope="independent finite trajectory audit; no nonlinear-region stability proof")
