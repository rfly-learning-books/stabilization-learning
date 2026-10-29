"""Corrected motor inspection adapter; reuses the original engineering plant."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
from extensions.state_conditioned_m_v1.envs.dc_motor import StateDependentDCMotorEnv
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash
from stabilization_learning.common.contracts_v2 import identity


class MotorInspectionAdapter(StateDependentDCMotorEnv):
    # plan is pure; register/controller_action explicitly registers execution.
    control_call_is_pure = False
    control_call_side_effects = ("pending_metric_and_command_registration",)
    controller_call_is_pure = False

    def controller_action(self,*args,**kwargs):
        raise RuntimeError("Inspection v2 requires plan(actor), register(plan,actor), then one step")

    def clf_candidate_action(self,*args,**kwargs):
        raise RuntimeError("Legacy candidate registration is not admitted through inspection v2")

    def plan(self, actor):
        state=self.state.copy()
        e,_=self.build_error(state)
        M=actor(self.metric_context(state))
        action=self.analytic_action_from_M(M,e)
        payload=dict(state=state,step=self._elapsed_steps,weights=module_hash(actor),
            M=M.detach().cpu().numpy(),action=action.detach().cpu().numpy())
        return dict(payload=deepcopy(payload),checksum=identity(payload))

    def register(self, proposal, actor):
        p=proposal["payload"]
        if self._active_action is not None:
            raise RuntimeError("Commit or cancel the registered motor action first")
        if identity(p)!=proposal["checksum"] or p["step"]!=self._elapsed_steps or not np.array_equal(p["state"],self.state):
            raise ValueError("Stale or changed motor plan")
        if p["weights"]!=module_hash(actor):
            raise ValueError("Actor changed since planning")
        self._bound_actor=actor;self._bound_weights=p["weights"]
        self._active_metric=p["M"].copy();self._active_action=p["action"].copy()
        return self._active_action.copy()

    def cancel(self):
        self._active_metric=None;self._active_action=None

    def step(self, action):
        if self._active_action is None:
            raise RuntimeError("Register a motor plan before execution")
        submitted=np.asarray(action.detach().cpu() if torch.is_tensor(action) else action,dtype=np.float64)
        if submitted.shape!=(1,) or not np.allclose(submitted,self._active_action,atol=1e-10,rtol=1e-7):
            raise ValueError("Submitted command differs from the registered plan")
        if module_hash(self._bound_actor)!=self._bound_weights:
            raise ValueError("Actor changed during a physical transition")
        result=super().step(action)
        row=self._metric_trace[-1]
        e1=torch.as_tensor((self.state-self.goal)/self.error_scale,dtype=torch.float64)
        with torch.no_grad():
            M1=self._bound_actor(self.metric_context(self.state)).to(e1)
            v1=float(e1@M1@e1)
        row["V_after_frozen_current_M_diagnostic"]=row["V_after"]
        row["same_M_one_step_residual_diagnostic"]=row.pop("same_M_one_step_residual")
        row["M_next"]=M1.tolist();row["V_after"]=v1
        e0=(np.asarray(row["state_before"])-self.goal)/self.error_scale
        row["full_state_certificate_residual"]=v1-row["V_before"]+.05*self.dt*float(e0@e0)
        return result


def motor_one_step_loss_v2(actor,env,*,seeds,decay=.05):
    """Recompute the SAME actor version at successor states; no plan mutation."""
    from stabilization_learning.objectives_v2 import descent_terms
    states=torch.as_tensor(np.stack([env.reset(seed=int(seed))[0] for seed in seeds]),dtype=torch.float64)
    errors=env.tensor_error(states)
    M=actor(env.context_for_states(states)).to(errors)
    action=env.analytic_action_from_M(M,errors)
    following=env.torch_step(states,action)
    next_error=env.tensor_error(following)
    next_M=actor(env.context_for_states(following)).to(next_error)
    v0=torch.einsum("bi,bij,bj->b",errors,M,errors)
    v1=torch.einsum("bi,bij,bj->b",next_error,next_M,next_error)
    terms=descent_terms(v1-v0,errors.square().sum(-1),dt=env.dt,decay=decay)
    physical=env.dt*errors.square().sum(-1)
    return (env.dt*next_error.square().sum(-1)+.1*terms["penalty"]).mean(),dict(
        physical_cost=physical,penalty=terms["penalty"],delta_v=v1-v0,M_current=M,M_next=next_M,
        objective_version="full_successor_M_true_norm_v2")
