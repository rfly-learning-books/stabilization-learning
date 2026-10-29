"""Source-pending pure tensor kernels for the seven frozen book environments.

Only an externally constructed FullHorizonTask may supply constants/methods.
No environment, RNG, optimizer, actor, trajectory or learner is constructed.
Original tensor functions execute on a private constant proxy, not live Gym
state. Success/failure masks reproduce the scalar formulas along each lane.
Engineering equivalence and speed have not been verified by this source file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


_CLASSES = {
    'coupled_linear':'CoupledLinearEnv', 'point_mass':'PointMassEnv',
    'pendulum':'PendulumEnv', 'cartpole':'CartPoleEnv',
    'two_link_arm':'TwoLinkArmEnv', 'unicycle_tracking':'UnicycleTrackingEnv',
    'multirotor_tracking':'MultirotorTrackingEnv'}
_FIELDS = {
    'coupled_linear':('_A','_B','_K','_stiffness','_damping','metric_gain','_low','_high'),
    'point_mass':('_B','_K','metric_gain','_reference_acceleration','_low','_high'),
    'pendulum':('_B','_K','metric_gain','_low','_high','gravity','length','mass','damping'),
    'cartpole':('_B','_K','metric_gain','_low','_high','gravity','masscart','masspole','length'),
    'two_link_arm':('mass1','mass2','length1','length2','gravity','kp','kd'),
    'unicycle_tracking':('radius','reference_speed','reference_omega'),
    'multirotor_tracking':('trajectory','radius','reference_omega','mass','inertia','gravity')}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {k:_json_value(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [_json_value(v) for v in value]
    return value


@dataclass(frozen=True)
class TaskTensorSpec:
    env_id: str
    setting_name: str
    state_dim: int
    error_dim: int
    action_dim: int
    metric_context_dim: int
    raw_dim: int
    dt: float
    decay: float
    max_steps: int
    required_hold_steps: int
    required_hold_seconds: float
    horizon_seconds: float
    moving_reference: bool
    reference_period: float
    goal: tuple
    action_low: tuple
    action_high: tuple
    success_tolerance: float
    metric_floor: float
    metric_trace: float
    metric_ceiling: float
    dtype: str
    device: str
    controller_form: str
    source_sha256: tuple
    signature: str
    support_level: str = 'source_pending_engineering'

    @property
    def horizon_steps(self):
        return self.max_steps

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_task(cls, task, *, dtype=torch.float64, device='cpu', decay=.05):
        spec, _, _ = _snapshot_task(task,dtype=dtype,device=device,decay=decay)
        return spec


def _snapshot_task(task, *, dtype, device, decay):
    selected_device=torch.device(device)
    if dtype != torch.float64 or selected_device.type != 'cpu':
        raise ValueError('The initial source-only kernel scope is CPU float64')
    # CPU tensors report the canonical unindexed device even when constructed
    # using cpu:0. Bind that canonical identity rather than a spelling alias.
    selected_device=torch.device('cpu')
    decay=float(decay)
    if not math.isfinite(decay) or decay < 0:
        raise ValueError('Finite nonnegative decay required')
    if getattr(task,'protocol_kind',None) != 'seven_task_full_horizon_metric_v1':
        raise TypeError('Supply an externally constructed frozen FullHorizonTask, not the bare Gym environment')
    env_id=getattr(task,'env_id',None)
    env=getattr(task,'inner_env',None)
    if env_id not in _CLASSES or type(env).__name__ != _CLASSES[env_id]:
        raise TypeError('This kernel supports only the seven declared frozen environment classes')
    expected_module=('stabilization_gym.envs.classic_control.classic' if env_id in
        ('coupled_linear','point_mass','pendulum','cartpole') else 'stabilization_gym.envs.robotics.robotics')
    if type(env).__module__ != expected_module:
        raise TypeError('A replacement/subclass environment needs an independent kernel protocol')
    dt=float(task.dt)
    if not math.isfinite(dt) or dt <= 0 or dt != float(env.dt):
        raise ValueError('Task and physical model time steps differ')
    goal=torch.as_tensor(env.goal,dtype=dtype,device=selected_device).clone().detach()
    low=torch.as_tensor(env.action_space.low,dtype=dtype,device=selected_device).clone().detach()
    high=torch.as_tensor(env.action_space.high,dtype=dtype,device=selected_device).clone().detach()
    state_dim,error_dim=int(env.state_dim),int(env.error_dim)
    if goal.shape != (state_dim,) or low.ndim != 1 or low.shape != high.shape:
        raise ValueError('Malformed frozen goal/action dimensions')
    if not bool(torch.isfinite(goal).all() & torch.isfinite(low).all() & torch.isfinite(high).all()):
        raise ValueError('Nonfinite frozen task constants')
    if not bool((low < high).all()):
        raise ValueError('Action bounds must be strictly ordered')
    proxy=SimpleNamespace(dt=float(env.dt),goal=goal,
        action_space=SimpleNamespace(low=low,high=high))
    copied={}
    for name in _FIELDS[env_id]:
        value=getattr(env,name)
        if torch.is_tensor(value):
            value=value.detach().to(dtype=dtype,device=selected_device).clone()
        elif not isinstance(value,(str,int,float,bool)):
            raise TypeError('Unexpected mutable model constant: '+name)
        setattr(proxy,name,value)
        copied[name]=_json_value(value)
    functions={name:getattr(type(env),name) for name in
               ('tensor_error','calculate_action_from_M','torch_step')}
    for helper in ('_flow','mechanics','reference'):
        if hasattr(type(env),helper):
            function=getattr(type(env),helper)
            functions[helper]=function
            setattr(proxy,helper,MethodType(function,proxy))
    files={Path(__file__).resolve()}
    for function in functions.values():
        path=inspect.getsourcefile(function)
        if path is None:
            raise ValueError('Tensor function source is unavailable')
        files.add(Path(path).resolve())
    wrapper_source=inspect.getsourcefile(type(task))
    if wrapper_source is None:
        raise ValueError('Full-horizon authority source is unavailable')
    files.add(Path(wrapper_source).resolve())
    source_hashes=tuple((str(path),_sha(path)) for path in sorted(files))
    moving=bool(task.moving_reference)
    setting=('multirotor_'+str(env.trajectory) if env_id=='multirotor_tracking'
             else 'pendulum_local' if env_id=='pendulum'
             else 'cartpole_local' if env_id=='cartpole' else env_id)
    payload=dict(env_id=env_id,setting_name=setting,state_dim=state_dim,error_dim=error_dim,
        action_dim=len(low),metric_context_dim=int(env.metric_context_dim),raw_dim=error_dim*(error_dim+1)//2,
        dt=dt,decay=decay,max_steps=int(task.max_steps),required_hold_steps=int(task.required_hold_steps),
        required_hold_seconds=float(task.required_hold_seconds),horizon_seconds=float(task.horizon_seconds),
        moving_reference=moving,reference_period=float(task.reference_period),
        goal=tuple(float(x) for x in goal),action_low=tuple(float(x) for x in low),
        action_high=tuple(float(x) for x in high),success_tolerance=float(env.success_tolerance),
        metric_floor=float(env.metric_spec.eigen_floor),metric_trace=float(env.metric_spec.trace),
        metric_ceiling=float(env.metric_spec.eigen_ceiling),dtype=str(dtype),device=str(selected_device),
        controller_form=str(task.controller_form),source_sha256=source_hashes)
    if (payload['max_steps']<1 or payload['required_hold_steps']<1 or
            payload['max_steps'] != task.max_steps or payload['required_hold_steps'] != task.required_hold_steps or
            payload['max_steps'] < payload['required_hold_steps'] or
            payload['metric_context_dim'] != state_dim or
            not math.isclose(payload['required_hold_seconds'],payload['required_hold_steps']*payload['dt'],abs_tol=1e-12)
            or not math.isclose(payload['horizon_seconds'],payload['max_steps']*payload['dt'],abs_tol=1e-12)):
        raise ValueError('Inconsistent full-horizon task metadata')
    expected_moving=(env_id=='unicycle_tracking' or
                     (env_id=='multirotor_tracking' and env.trajectory=='circle'))
    cycles=float(task.tracking_cycles)
    if moving != expected_moving or not math.isfinite(cycles) or cycles < 2.:
        raise ValueError('Task reference mode or full-period horizon authority differs')
    if moving:
        omega=float(env.reference_omega)
        if not math.isfinite(omega) or omega <= 0:
            raise ValueError('Moving reference requires a finite positive frequency')
        period=2*math.pi/omega
        expected_hold=int(math.ceil(period/dt-1e-12))
        minimum_horizon=int(math.ceil(cycles*period/dt-1e-12))
        if (not math.isclose(payload['reference_period'],period,rel_tol=1e-12,abs_tol=1e-12)
                or payload['required_hold_steps'] != expected_hold):
            raise ValueError('Moving reference must hold the original complete period')
    else:
        minimum_horizon=payload['required_hold_steps']+1
        if payload['reference_period'] != 0. or payload['required_hold_seconds'] < 1.-1e-12:
            raise ValueError('Static goal requires at least one second of final hold')
    if payload['max_steps'] != max(int(env.max_steps),minimum_horizon):
        raise ValueError('Full-horizon task budget differs from its original authority')
    signature=hashlib.sha256(json.dumps(dict(spec=payload,model_constants=copied),sort_keys=True,
                                        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return TaskTensorSpec(**payload,signature=signature),proxy,functions


class TensorTaskKernel:
    """Pure per-lane tensor arithmetic, with no trajectory or policy state.

    M is supplied explicitly. Its constancy inside a physical episode is the
    runner's responsibility. No supported PPO/learner adapter is advertised.
    """
    support_level='source_pending_engineering'
    supported_algorithms=()

    def __init__(self, task, *, dtype=torch.float64, device='cpu', decay=.05):
        self._spec,self._proxy,self._functions=_snapshot_task(task,dtype=dtype,device=device,decay=decay)
        self._dtype,self._device=dtype,torch.device(self._spec.device)
        self._low=self._proxy.action_space.low
        self._high=self._proxy.action_space.high

    @property
    def spec(self):
        return self._spec

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def decay(self):
        return self._spec.decay

    @classmethod
    def from_task(cls, task, **kwargs):
        return cls(task,**kwargs)

    def describe(self):
        return dict(support_level=self.support_level,supported_algorithms=[],
            scope='seven frozen basic classes; eight shared-goal settings; CPU float64',
            spec=self.spec.to_dict(),control_is_pure=True,owns_live_environment_state=False,
            gradients=dict(control_wrt_M='source_expected_unverified',
                model_wrt_action_state='source_expected_unverified',
                sampling_detach='runner_owned',new_representation_or_reference_gradient=False),
            engineering_equivalence='not_run',speedup='not_measured',certificate_claim='candidate_only')

    def require_algorithm(self, name):
        raise ValueError('No learner adapter is verified by this source-only kernel: '+str(name))

    def _state(self, state, *, finite=True):
        if not torch.is_tensor(state) or state.ndim!=2 or state.shape[1]!=self.spec.state_dim or state.shape[0]<1:
            raise ValueError('State must be [B,state_dim] with B>=1')
        if state.dtype!=self.dtype or state.device!=self.device:
            raise ValueError('State dtype/device differs from the kernel spec')
        if finite and not bool(torch.isfinite(state).all()):
            raise FloatingPointError('Nonfinite state is an invalid computation, not a valid cheap-failure sample')
        return state

    def _matrix(self, M, batch):
        d=self.spec.error_dim
        if not torch.is_tensor(M) or M.dtype!=self.dtype or M.device!=self.device:
            raise ValueError('M must use the physical kernel dtype/device')
        if M.shape==(d,d):
            value=M.unsqueeze(0).expand(batch,-1,-1)
        elif M.shape==(batch,d,d):
            value=M
        else:
            raise ValueError('M must be shared [d,d] or per-lane [B,d,d]')
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite certificate matrix')
        return value

    def validate_metric(self, M, batch_size):
        """One start-boundary structural check; not a stability certificate."""
        value=self._matrix(M,batch_size)
        if not bool(torch.isclose(value,value.transpose(-1,-2),atol=1e-10,rtol=1e-8).all()):
            raise ValueError('M is not symmetric within the declared structural tolerance')
        eigen=torch.linalg.eigvalsh((value+value.transpose(-1,-2))*.5)
        trace=value.diagonal(dim1=-2,dim2=-1).sum(-1)
        if not bool(((eigen[...,0]>=self.spec.metric_floor-2e-6)&
            (eigen[...,-1]<=self.spec.metric_ceiling+2e-6)&
            ((trace-self.spec.metric_trace).abs()<2e-5)).all()):
            raise ValueError('M violates the frozen spectral/trace contract')
        return value.clone()

    def context(self, batch_size=None):
        goal=self._proxy.goal.to(dtype=torch.float32)
        if batch_size is None:
            return goal.clone()
        if type(batch_size) is not int or batch_size<1:
            raise ValueError('Positive integer batch size required')
        return goal.unsqueeze(0).expand(batch_size,-1).clone()

    def error(self, state):
        state=self._state(state)
        error=self._functions['tensor_error'](self._proxy,state)
        if (error.shape!=(len(state),self.spec.error_dim) or error.dtype!=self.dtype
                or error.device!=self.device):
            raise ValueError('Original error coordinates differ from the kernel spec')
        if not bool(torch.isfinite(error).all()):
            raise FloatingPointError('Nonfinite physical error coordinates')
        return error.clone()

    def _goal_mask(self, error, physical_error):
        t=self.spec.success_tolerance
        kind=self.spec.env_id
        if kind=='two_link_arm':
            return (torch.linalg.vector_norm(error[...,:2],dim=-1)<t)&(torch.linalg.vector_norm(error[...,2:],dim=-1)<2*t)
        if kind=='unicycle_tracking':
            return (torch.linalg.vector_norm(error[...,:2],dim=-1)<t)&(error[...,2].abs()<t)
        if kind=='multirotor_tracking':
            return ((torch.linalg.vector_norm(error[...,:2],dim=-1)<t)&
                (torch.linalg.vector_norm(error[...,2:4],dim=-1)<2*t)&
                (error[...,4].abs()<t)&(error[...,5].abs()<2*t))
        return physical_error<=t*t

    def failure(self, state):
        state=self._state(state,finite=False)
        bad=~torch.isfinite(state).all(dim=-1)
        kind=self.spec.env_id
        if kind=='two_link_arm':
            return bad|(state[...,2:].abs().amax(dim=-1)>40.)
        if kind=='unicycle_tracking':
            return bad|(torch.linalg.vector_norm(state[...,:2],dim=-1)>20.)
        if kind=='multirotor_tracking':
            return bad|(state[...,1]<0.)|(torch.linalg.vector_norm(state[...,:2],dim=-1)>20.)|(state[...,4].abs()>math.pi/2.)
        bad=bad|(torch.linalg.vector_norm(state,dim=-1)>100.)
        if kind=='pendulum':
            angle=torch.atan2(torch.sin(state[...,0]),torch.cos(state[...,0]))
            bad=bad|(angle.abs()>math.pi/2.)|(state[...,1].abs()>12.)
        elif kind=='cartpole':
            angle=torch.atan2(torch.sin(state[...,2]),torch.cos(state[...,2]))
            bad=bad|(state[...,0].abs()>2.4)|(angle.abs()>math.pi/4.)
        return bad

    def measure(self, state):
        state=self._state(state)
        error=self.error(state)
        physical=error.square().sum(dim=-1)
        if not bool(torch.isfinite(physical).all()):
            raise FloatingPointError('Physical cost overflowed for a finite state')
        return dict(state=state.clone(),error=error,physical_error=physical,
            in_goal=self._goal_mask(error,physical),failure=self.failure(state),kernel_signature=self.spec.signature)

    def physical_cost(self, state):
        return self.measure(state)['physical_error']

    def in_goal(self, state):
        return self.measure(state)['in_goal']

    def _measurement(self, state, measurement):
        if measurement is None:
            return self.measure(state)
        if measurement.get('kernel_signature')!=self.spec.signature or not torch.equal(measurement['state'],state):
            raise ValueError('Cached measurement is for another kernel or state')
        error=measurement['error']
        if error.shape!=(len(state),self.spec.error_dim) or error.dtype!=self.dtype or error.device!=self.device:
            raise ValueError('Cached measurement has invalid error coordinates')
        if not bool(torch.isfinite(error).all()):
            raise FloatingPointError('Cached measurement has nonfinite error coordinates')
        return measurement

    def control(self, state, M, measurement=None):
        state=self._state(state)
        matrix=self._matrix(M,len(state))
        measured=self._measurement(state,measurement)
        action=self._functions['calculate_action_from_M'](self._proxy,matrix,measured['error'],
            context={'state':state,'goal':self._proxy.goal})
        if (action.shape!=(len(state),self.spec.action_dim) or action.dtype!=self.dtype
                or action.device!=self.device):
            raise ValueError('Original control output differs from the kernel action spec')
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError('Nonfinite controller output')
        return action.clone()

    def _applied(self, action, batch):
        if (not torch.is_tensor(action) or action.shape!=(batch,self.spec.action_dim)
                or action.dtype!=self.dtype or action.device!=self.device):
            raise ValueError('Physical action must be [B,action_dim] in the kernel dtype/device')
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError('Nonfinite physical action is rejected before model evaluation')
        return torch.minimum(torch.maximum(action,self._low),self._high)

    def step(self, state, action):
        state=self._state(state)
        applied=self._applied(action,len(state))
        following=self._functions['torch_step'](self._proxy,state,applied)
        self._state(following)
        return following.clone()

    def transition(self, state, M):
        """One pure batch model query, not a lane commit or rollout sample."""
        state=self._state(state)
        matrix=self._matrix(M,len(state))
        before=self.measure(state)
        nominal=self.control(state,matrix,measurement=before)
        applied=self._applied(nominal,len(state))
        following=self._functions['torch_step'](self._proxy,state,applied)
        self._state(following)
        after=self.measure(following)
        V0=torch.einsum('bi,bij,bj->b',before['error'],matrix,before['error'])
        V1=torch.einsum('bi,bij,bj->b',after['error'],matrix,after['error'])
        delta=V1-V0
        scale=before['physical_error'].clamp_min(1e-6)
        raw_residual=delta+self.decay*self.spec.dt*before['physical_error']
        normalized=(delta+self.decay*self.spec.dt*scale)/scale
        cost=self.spec.dt*before['physical_error']
        penalty=normalized.clamp_min(0.)
        if not all(bool(torch.isfinite(value).all()) for value in
                   (V0,V1,delta,raw_residual,normalized,cost,penalty)):
            raise FloatingPointError('Finite input produced nonfinite certificate evidence')
        return dict(state_before=state.clone(),state_after=following.clone(),
            error_before=before['error'],error_after=after['error'],action_nominal=nominal,
            applied_action=applied.clone(),V_before=V0,V_after=V1,
            physical_error_before=before['physical_error'],physical_error_after=after['physical_error'],
            cost=cost,decay_residual=raw_residual,
            normalized_decay_residual=normalized,penalty=penalty,V_rise=delta>1e-8,
            in_goal_before=before['in_goal'],in_goal_after=after['in_goal'],
            failure_before=before['failure'],failure_after=after['failure'],
            dt=self.spec.dt,kernel_signature=self.spec.signature)
