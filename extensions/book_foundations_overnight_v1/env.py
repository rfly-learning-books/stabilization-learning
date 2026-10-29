"""Physical ODE integration and exact held-input propagation."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def simulate(rhs,initial,time,guard=None):
    def wrapped(t,x):
        if guard:guard()
        return rhs(t,x)
    result=solve_ivp(wrapped,(float(time[0]),float(time[-1])),initial,t_eval=time,
                     rtol=1e-10,atol=1e-12,max_step=.05)
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(result.message)
    return result.y.T


def exact_affine(A,b,initial,time):
    A=np.asarray(A,float);b=np.asarray(b,float);n=len(b)
    augmented=np.zeros((n+1,n+1));augmented[:n,:n]=A;augmented[:n,n]=b
    start=np.r_[initial,1.]
    return np.stack([(expm(augmented*t)@start)[:n] for t in time])


def zoh(A,B,period):
    A=np.asarray(A,float);B=np.atleast_2d(B).reshape(len(A),-1);n,m=B.shape
    augmented=np.zeros((n+m,n+m));augmented[:n,:n]=A;augmented[:n,n:]=B
    discrete=expm(augmented*period)
    return discrete[:n,:n],discrete[:n,n:]


def scalar_integrator(initial,period,steps,limit=None):
    state=[float(initial)];actions=[]
    for _ in range(steps):
        requested=-2.*state[-1]
        action=requested if limit is None else np.clip(requested,-limit,limit)
        actions.append(action);state.append(state[-1]+period*action)
    return np.arange(steps+1)*period,np.array(state),np.array(actions)
