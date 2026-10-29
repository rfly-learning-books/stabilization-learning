"""Specified controllers; gains are fixed before the numerical experiments."""
import numpy as np


def pi_action(state,disturbance,route='PI',limit=None):
    x,z=state
    requested=-2*x-z if route=='PI' else -2*x-disturbance if route=='known_feedforward' else -2*x
    return requested if limit is None else float(np.clip(requested,-limit,limit)),requested


def virtual_output_action(state,reference=1.,gain=2.,route='virtual'):
    x1,x2=state
    return x1-2*x2-gain*(x1+x2+reference) if route=='virtual' else x1-gain*(-x1+x2-reference)


def resonance_controller(Q,A,B,period=1.,rho=120.):
    from .env import zoh
    n=len(Q);Ad,Bd=zoh(A,B,period)
    F0=np.zeros((n,n));F0[:2,:2]=Ad
    for j in range(n-3):F0[j+3,j+2]=1.
    G=np.zeros(n);G[:2]=Bd[:,0];G[2]=1.
    H=G@Q@G;b=G@Q@F0;K=b/(rho+H);F=F0-np.outer(G,K)
    return F0,G,H,b,K,F
