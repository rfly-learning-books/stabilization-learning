"""State-dependent-coefficient models and metric-derived physical feedback.

The learned-M controller and classical online Riccati SDRE baseline are named
separately. Original dynamics, integrators and limits are reused unchanged.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.linalg import solve_continuous_are
from ...interfaces.m_learning import (
    TaskDescription, MetricScope, GradientCapabilities, CapabilityError,
)


class SDCMetricTask:
    def __init__(self, physical, name):
        if name not in ("coupled_linear","point_mass","pendulum","cartpole"):
            raise CapabilityError("No SDC factorization is declared for this task")
        self.physical, self.name = physical, name
        self.dt, self.max_steps = float(physical.dt),int(physical.max_steps)
        self.error_dim=physical.error_dim
        self.metric_spec=physical.metric_spec
        self.goal=np.asarray(physical.goal,dtype=np.float64).copy()
        self.required_hold_seconds=1.
        self.success_tolerance=float(physical.success_tolerance)
        self.action_low=torch.as_tensor(physical.action_space.low,dtype=torch.float64)
        self.action_high=torch.as_tensor(physical.action_space.high,dtype=torch.float64)
        qs={"coupled_linear":[4.,4.,1.,1.],"point_mass":[4.,4.,1.,1.],
            "pendulum":[12.,1.],"cartpole":[2.,.5,16.,1.]}
        rs={"coupled_linear":.5,"point_mass":.5,"pendulum":.5,"cartpole":.1}
        self.Q=np.diag(qs[name]); self.R=np.eye(len(self.action_low))*rs[name]
        A,B=self.coefficients(torch.zeros(self.error_dim,dtype=torch.float64))
        self.P0=solve_continuous_are(A.numpy(),B.numpy(),self.Q,self.R)
        self.L0=np.linalg.cholesky(self.P0)
        self.controller_form="learned_state_M_SDC_gain_R_inverse_B_e_transpose_M_v2"
        self.description=TaskDescription(name=name,backend=type(self).__module__,
            support_level="versioned_state_M_candidate_control",metric_scope=MetricScope.STATE_DEPENDENT,
            conditioning=("current_physical_error","fixed_goal"),error_dim=self.error_dim,
            raw_metric_dim=self.metric_spec.raw_dim,action_description="SI physical action after original saturation",
            certificate_factor=1.,gradients=GradientCapabilities(action_wrt_M=True,action_wrt_error=True,
                full_state_certificate_derivative=True,model_wrt_action=True),
            controller_form=self.controller_form,controller_has_state=False,control_call_is_pure=True,
            cost_convention="positive_integral",supported_algorithms=("state_metric_model_gradient_v2",),
            missing_capabilities=("global nonlinear Lyapunov guarantee","hardware and GPU recovery"))

    def describe(self):
        return self.description

    def require_algorithm(self, algorithm):
        if algorithm not in self.description.supported_algorithms:
            raise CapabilityError("State-dependent M cannot use episode-held-M sampling")
        return self

    def error(self, state):
        return self.physical.tensor_error(state)

    def coefficients(self, error):
        n=self.error_dim; shape=error.shape[:-1]
        A=error.new_zeros((*shape,n,n))
        m=len(self.action_low); B=error.new_zeros((*shape,n,m))
        p=self.physical
        if self.name=="coupled_linear":
            A=A+p._A.to(error);B=B+p._B.to(error)
        elif self.name=="point_mass":
            A[...,0,2]=A[...,1,3]=1.
            B=B+p._B.to(error)
        elif self.name=="pendulum":
            theta=error[...,0]
            A[...,0,1]=1.
            A[...,1,0]=p.gravity/p.length*torch.sinc(theta/torch.pi)
            A[...,1,1]=-p.damping
            B=B+p._B.to(error)
        else:
            theta,omega=error[...,2],error[...,3]
            s,c=torch.sin(theta),torch.cos(theta)
            total=p.masscart+p.masspole;moment=p.masspole*p.length
            den=p.length*(4./3.-p.masspole*c.square()/total)
            A[...,0,1]=A[...,2,3]=1.
            A[...,1,2]=-moment*c*p.gravity*torch.sinc(theta/torch.pi)/(total*den)
            A[...,1,3]=moment*omega*s/total*(1+moment*c.square()/(total*den))
            A[...,3,2]=p.gravity*torch.sinc(theta/torch.pi)/den
            A[...,3,3]=-c*moment*omega*s/(total*den)
            B[...,1,0]=1/total+moment*c.square()/(total*total*den)
            B[...,3,0]=-c/(total*den)
        return A,B

    def feedforward(self, state):
        result=state.new_zeros((*state.shape[:-1],len(self.action_low)))
        if self.name=="coupled_linear":
            result=result+state.new_tensor(self.goal[:2])@self.physical._stiffness.to(state).T
        elif self.name=="point_mass":
            result=result+self.physical._reference_acceleration.to(state)
        return result

    def requested_control(self, M, error, *, state=None):
        if M.shape[-2:]!=(self.error_dim,self.error_dim):
            raise ValueError("Physical M coordinates differ")
        _,B=self.coefficients(error)
        weighted=(M@error.unsqueeze(-1)).squeeze(-1)
        feedback=torch.linalg.solve(error.new_tensor(self.R),
            (B.transpose(-1,-2)@weighted.unsqueeze(-1))).squeeze(-1)
        if state is None:
            state=error+error.new_tensor(self.goal)
        return self.feedforward(state)-feedback

    def control(self,M,error,*,state=None):
        requested=self.requested_control(M,error,state=state)
        return torch.maximum(self.action_low.to(error),torch.minimum(self.action_high.to(error),requested))

    def step_model(self, state, action):
        return self.physical.torch_step(state,action)

    def failure(self, state):
        bad=~torch.isfinite(state).all(-1)|(torch.linalg.vector_norm(state,dim=-1)>100)
        if self.name=="pendulum":
            e=self.error(state);bad=bad|(e[...,0].abs()>np.pi/2)|(state[...,1].abs()>12)
        elif self.name=="cartpole":
            e=self.error(state);bad=bad|(state[...,0].abs()>2.4)|(e[...,2].abs()>np.pi/4)
        return bad

    def initial_states(self, seeds, *, scale=1.):
        return torch.as_tensor(np.stack([self.goal+np.random.default_rng(int(s)).uniform(-1.,1.,self.error_dim)*
            self.physical.initial_scale*scale for s in seeds]),dtype=torch.float64)

    def classical_sdre_metric(self,error):
        """Reference baseline: solve the state-dependent ARE, not a learned M."""
        A,B=self.coefficients(error)
        if error.ndim!=1:
            return torch.stack([self.classical_sdre_metric(e) for e in error])
        P=solve_continuous_are(A.detach().cpu().numpy(),B.detach().cpu().numpy(),self.Q,self.R)
        return error.new_tensor(P)
