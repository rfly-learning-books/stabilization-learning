"""Isolated discrete-reference corrections; not the frozen benchmark models.

These variants make the exact tracking target invariant for the nominal
discrete plant. They are engineering corrections, not trained-policy results
or a convergence theorem away from the target.
"""
import math
import torch
from stabilization_gym.envs.robotics.robotics import UnicycleTrackingEnv,MultirotorTrackingEnv


def wrap(x):return torch.atan2(torch.sin(x),torch.cos(x))


def bound(value,space):
    low=torch.as_tensor(space.low,dtype=value.dtype,device=value.device)
    high=torch.as_tensor(space.high,dtype=value.dtype,device=value.device)
    return torch.minimum(torch.maximum(value,low),high)


class DiscreteConsistentUnicycle(UnicycleTrackingEnv):
    reference_model_version='discrete_midpoint_chord_v2'

    def calculate_action_from_M(self,M,e,*,context,controller_state=None,certificate=None):
        transformed=torch.cat((e[...,:2],torch.sin(e[...,2:3])),-1)
        weighted=(M@transformed.unsqueeze(-1)).squeeze(-1)
        delta=self.dt*self.reference_omega
        chord_speed=2*self.radius*math.sin(delta/2)/self.dt
        speed=chord_speed*torch.cos(e[...,2])-1.2*weighted[...,0]
        omega=self.reference_omega-self.reference_speed*(3*weighted[...,1]+3*weighted[...,2])
        return bound(torch.stack((speed,omega),-1),self.action_space)


class DiscreteConsistentCircle(MultirotorTrackingEnv):
    reference_model_version='discrete_position_velocity_pitch_rate_v2'

    def __init__(self,**kwargs):
        kwargs['trajectory']='circle'
        super().__init__(**kwargs)
        self.discrete_phase_step=self.dt*self.reference_omega
        self.discrete_speed=2*self.radius*math.sin(self.discrete_phase_step/2)/self.dt
        self.discrete_acceleration=4*self.radius*math.sin(self.discrete_phase_step/2)**2/self.dt**2
        if not (0<self.discrete_phase_step<math.pi/2 and self.discrete_acceleration<self.gravity):
            raise ValueError('Discrete reference requires small positive phase advance and acceleration below gravity')

    def _pitch(self,phase):
        a=self.discrete_acceleration
        return torch.atan2(a*torch.sin(phase),self.gravity+a*torch.cos(phase))

    def _reference(self,phase,with_alpha=False):
        d=self.discrete_phase_step
        origin=torch.as_tensor(self.goal,dtype=phase.dtype,device=phase.device)
        pitch=self._pitch(phase);previous=self._pitch(phase-d)
        omega=wrap(pitch-previous)/self.dt
        state=torch.stack((origin[0]+self.radius*torch.sin(phase),
            origin[1]+self.radius*(1-torch.cos(phase)),
            self.discrete_speed*torch.cos(phase-d/2),self.discrete_speed*torch.sin(phase-d/2),
            pitch,omega),-1)
        acceleration=torch.stack((-self.discrete_acceleration*torch.sin(phase),
                                   self.discrete_acceleration*torch.cos(phase)),-1)
        alpha=(wrap(self._pitch(phase+d)-pitch)/self.dt-omega)/self.dt if with_alpha else None
        return state,acceleration,alpha

    def reference(self,phase):
        state,acceleration,_=self._reference(phase)
        return state,acceleration

    def calculate_action_from_M(self,M,e,*,context,controller_state=None,certificate=None):
        state=context['state'];reference,feedforward,alpha=self._reference(state[...,6],True)
        weighted=(M@e.unsqueeze(-1)).squeeze(-1)
        acceleration=feedforward-2.5*weighted[...,:2]-3*weighted[...,2:4]
        fx,fz=acceleration[...,0],acceleration[...,1]+self.gravity
        desired_pitch=torch.atan2(-fx,fz)
        pitch_error=wrap(state[...,4]-desired_pitch)
        thrust=self.mass*(-fx*torch.sin(state[...,4])+fz*torch.cos(state[...,4]))
        torque=self.inertia*(alpha-25*pitch_error-8*(state[...,5]-reference[...,5])
            -2*(weighted[...,4]-e[...,4])-(weighted[...,5]-e[...,5]))
        return bound(torch.stack((thrust,torque),-1),self.action_space)
