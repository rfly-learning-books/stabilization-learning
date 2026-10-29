"""Pure nominal longitudinal algebra. No environment, simulator, or rollout.

Coefficients are public Aerosonde parameter literals; the attached-flow closure,
physical ideal-input interface and task limits are this version's declared model.
"""
import json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/"PARAMETERS.json").read_text(encoding="utf-8"))["parameters"]
CONTRACT=json.loads((HERE/"DESIGN_CONTRACT.json").read_text(encoding="utf-8"))
S=np.array(CONTRACT["coordinates"]["state_scales"])
U=np.array(CONTRACT["coordinates"]["input_deviation_scales"])


def forces(v,alpha,q,delta):
    qbar=.5*P["rho"]*v*v
    qhat=P["c"]*q/(2*v)
    cl=P["C_L_0"]+P["C_L_alpha"]*alpha+P["C_L_q"]*qhat+P["C_L_delta_e"]*delta
    cd=P["C_D_p"]+(P["C_L_0"]+P["C_L_alpha"]*alpha)**2/(np.pi*P["e"]*P["AR"])+P["C_D_q"]*qhat+P["C_D_delta_e"]*delta
    cm=P["C_m_0"]+P["C_m_alpha"]*alpha+P["C_m_q"]*qhat+P["C_m_delta_e"]*delta
    return qbar*P["S_wing"]*cl,qbar*P["S_wing"]*cd,qbar*P["S_wing"]*P["c"]*cm


def rhs_numpy(state,control):
    h,v,alpha,theta,q=np.asarray(state,dtype=np.float64)
    thrust,delta=np.asarray(control,dtype=np.float64)
    if not np.isfinite(np.r_[state,control]).all() or v<=0:
        raise ValueError("RHS needs finite state/input and positive airspeed; no clipping/projection")
    lift,drag,moment=forces(v,alpha,q,delta)
    gamma=theta-alpha
    return np.array([v*np.sin(gamma),(thrust*np.cos(alpha)-drag)/P["mass"]-P["gravity"]*np.sin(gamma),
        q-(lift+thrust*np.sin(alpha))/(P["mass"]*v)+P["gravity"]*np.cos(gamma)/v,q,moment/P["Jy"]])


def rhs_torch(state,control):
    h,v,alpha,theta,q=state.unbind(-1); thrust,delta=control.unbind(-1)
    lift,drag,moment=forces(v,alpha,q,delta); gamma=theta-alpha
    return torch.stack([v*torch.sin(gamma),(thrust*torch.cos(alpha)-drag)/P["mass"]-P["gravity"]*torch.sin(gamma),
        q-(lift+thrust*torch.sin(alpha))/(P["mass"]*v)+P["gravity"]*torch.cos(gamma)/v,q,moment/P["Jy"]],dim=-1)


def riccati_numpy(M,Ad,Bd,N=100,dt=.05):
    cost_to_go=np.array(M,dtype=np.float64,copy=True); R=.1*np.eye(2)
    for _ in range(N):
        G=dt*R+Bd.T@cost_to_go@Bd
        K=np.linalg.solve(G,Bd.T@cost_to_go@Ad)
        cost_to_go=dt*M+Ad.T@cost_to_go@(Ad-Bd@K)
        cost_to_go=.5*(cost_to_go+cost_to_go.T)
    return K,cost_to_go


def riccati_torch(M,Ad,Bd,N=100,dt=.05):
    cost_to_go=M; R=.1*torch.eye(2,dtype=M.dtype,device=M.device)
    for _ in range(N):
        G=dt*R+Bd.T@cost_to_go@Bd
        K=torch.linalg.solve(G,Bd.T@cost_to_go@Ad)
        cost_to_go=dt*M+Ad.T@cost_to_go@(Ad-Bd@K)
        cost_to_go=.5*(cost_to_go+cost_to_go.T)
    return K,cost_to_go


def physical_control(M,error,Ad,Bd,trim_control):
    K,P0=riccati_numpy(M,Ad,Bd)
    requested=np.asarray(trim_control)-U*(K@error)
    bounded=np.clip(requested,[0.,-np.deg2rad(25.)],[40.,np.deg2rad(25.)])
    return bounded,{"requested":requested,"K":K,"P0_not_M":P0,"saturated":bool(np.any(bounded!=requested))}
