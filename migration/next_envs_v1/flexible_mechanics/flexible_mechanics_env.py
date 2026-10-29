"""Isolated nominal dual-inertia task; no registration or training changes.

N-step constant-torque prediction shapes the action but only one .01 s physical
step is executed. M is a candidate certificate, not a proved Lyapunov function.
Control queries retain their autograd graph and write only a detached pending
audit record; the executed metric is committed by the non-autograd Gym step.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral

import gymnasium as gym
import numpy as np
from scipy.linalg import expm
import torch

from stabilization_gym.envs.base import StateMetricEnv


class FlexibleMechanicsEnv(StateMetricEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}
    controller_form = "same_M_N_step_constant_torque_prediction_one_physical_step_execution"
    control_call_is_pure = False
    control_call_side_effects = ("audit_pending_metric_and_action",)

    def __init__(self, *, target_angle=None, prediction_steps=50, render_mode=None):
        if isinstance(prediction_steps, bool) or not isinstance(prediction_steps, Integral) or prediction_steps not in (1, 50):
            raise ValueError("Only predeclared N=50 or legacy N=1 is supported")
        super().__init__(state_dim=4, action_low=[-.5], action_high=[.5], error_dim=4,
                         dt=.01, max_steps=12000, goal=[0., 0., 0., 0.],
                         metric_floor=.05, render_mode=render_mode)
        self.prediction_steps = int(prediction_steps)
        self._fixed_target = None if target_angle is None else self._validate_target(target_angle)
        self._target_angle = 0. if target_angle is None else self._fixed_target
        self.goal = np.array([self.target_angle, self.target_angle, 0., 0.])
        self._goal_snapshot = self.goal.copy()
        self.metric_context_dim = 1
        self.Jm, self.Jl, self.Bm, self.Bl, self.stiffness = .02, .05, .02, .05, .5
        self.torque_limit, self.twist_limit, self.speed_limit = .5, .5, 3.
        self.required_hold_seconds, self.horizon_seconds = 2., 120.
        self.torque_regularizer = .04
        self.A = np.array([[0.,0.,1.,0.], [0.,0.,0.,1.],
                           [-25.,25.,-1.,0.], [10.,-10.,0.,-1.]])
        self.B = np.array([[0.], [0.], [50.], [0.]])
        block = np.zeros((5,5)); block[:4,:4], block[:4,4:] = self.A, self.B
        transition = expm(self.dt*block)
        self.Ad, self.Bd = transition[:4,:4].copy(), transition[:4,4:].copy()
        self.S = np.array([[0.,1.,0.,0.], [2.,-2.,0.,0.], [0.,0.,1/3,0.], [0.,0.,0.,1/3]])
        self.Sinv = np.array([[1.,.5,0.,0.], [1.,0.,0.,0.], [0.,0.,3.,0.], [0.,0.,0.,3.]])
        self.Ae, self.Be = self.S@self.Ad@self.Sinv, self.S@self.Bd
        self.AN = np.linalg.matrix_power(self.Ae, self.prediction_steps)
        self.BN = sum((np.linalg.matrix_power(self.Ae,j)@self.Be for j in range(self.prediction_steps)), np.zeros((4,1)))
        self.protocol_kind = "dual_inertia_N50_isolated_engineering_v1"
        self.controller_capabilities.update(full_horizon_evaluation=True, full_state_observation=True,
            state_coordinate_units=("rad", "rad", "rad/s", "rad/s"),
            physical_step_autograd=False, differentiable_torch_model=True,
            fixed_reference_data=True, target_context_autograd=False,
            controller_execution_device="cpu",controller_execution_dtypes=("float32","float64"),
            certificate_guarantee="candidate_only")
        self._config_snapshot = self._configuration()
        self._matrix_snapshot = {n:getattr(self,n).copy() for n in ("A","B","Ad","Bd","S","Sinv","Ae","Be","AN","BN")}
        self._pending_control = self._episode_metric = None
        self._execution_mode = self._external_origin = None
        self._in_goal_since = self._first_hold_confirmation = None
        self._metric_steps = 0
        self._trace, self._last_info = [], {}
        self._initial_state, self._seed_record = self.state.copy(), None

    def _configuration(self):
        return {n:getattr(self,n) for n in ("prediction_steps","dt","max_steps","Jm","Jl","Bm","Bl","stiffness",
            "torque_limit","twist_limit","speed_limit","required_hold_seconds","horizon_seconds","torque_regularizer")}

    def _assert_reference_fixed(self):
        if (not np.array_equal(self.goal, self._goal_snapshot) or
            not np.array_equal(self.goal, [self.target_angle,self.target_angle,0.,0.])):
            raise ValueError("Episode target is fixed data; set it through reset")
        if self._configuration() != self._config_snapshot or any(not np.array_equal(getattr(self,n),v) for n,v in self._matrix_snapshot.items()):
            raise ValueError("Physical/controller configuration changed")

    @staticmethod
    def _validate_target(value):
        if isinstance(value,(bool,np.bool_)) or np.asarray(value).shape != ():
            raise ValueError("Reference must be a scalar angle")
        value = float(value)
        if not math.isfinite(value) or not -.3 <= value <= .3:
            raise ValueError("Reference must lie in [-.3,.3] rad")
        return value

    @property
    def target_angle(self):
        return self._target_angle

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = {} if options is None else dict(options)
        if set(options)-{"target_angle","state","execution_origin"}:
            raise ValueError("Unsupported reset option")
        target = options.get("target_angle", self._fixed_target)
        self._target_angle = self._validate_target(self.np_random.uniform(-.3,.3) if target is None else target)
        self.goal = np.array([self.target_angle,self.target_angle,0.,0.])
        self._goal_snapshot = self.goal.copy()
        q = self.target_angle+self.np_random.uniform(-.2,.2)
        twist = self.np_random.uniform(-.15,.15)
        default = [q+5*twist/7,q-2*twist/7,*self.np_random.uniform(-.3,.3,2)]
        state = np.asarray(options.get("state",default),dtype=np.float64)
        if state.shape != (4,) or self.is_failure(state):
            raise ValueError("Require finite state, |twist|<=.5 rad, |speeds|<=3 rad/s; absolute angles are unbounded")
        origin = options.get("execution_origin", "external_physical_action")
        if origin not in ("external_physical_action","fixed_zero_feedforward","deliberate_fault_injection_not_M_policy_performance"):
            raise ValueError("Unsupported external execution origin")
        self.state = state.copy()
        self._initial_state, self._seed_record = state.copy(), seed
        self._elapsed_steps, self._hold_steps, self._done = 0, 0, False
        self._pending_control = self._episode_metric = None
        self._execution_mode, self._external_origin = None, origin
        self._in_goal_since = 0. if self.is_success(state) else None
        self._first_hold_confirmation = None
        self._metric_steps, self._trace = 0, []
        self._last_info = self._info()
        return state.copy(), deepcopy(self._last_info)

    def tensor_error(self,state):
        state = state if torch.is_tensor(state) else torch.as_tensor(state,dtype=torch.float64)
        if not state.is_floating_point() or state.shape[-1:] != (4,) or not torch.isfinite(state).all():
            raise ValueError("State must be finite floating [...,4]")
        return (state-state.new_tensor(self.goal))@state.new_tensor(self.S).T

    def build_error(self,observation,goal=None,representation=None,history=None):
        self._assert_reference_fixed()
        if representation is not None and not getattr(representation,"preserves_physical_coordinates",False):
            raise ValueError("Representation must preserve physical position and velocity coordinates")
        state = observation if torch.is_tensor(observation) else torch.as_tensor(observation,dtype=torch.float64)
        target = state.new_tensor(self.goal) if goal is None else torch.as_tensor(goal).to(state)
        if target.requires_grad or target.shape != (4,) or not torch.equal(target,state.new_tensor(self.goal)):
            raise ValueError("Reference is fixed episode data, not a learnable input")
        return self.tensor_error(state), {"target_angle":self.target_angle,"prediction_steps":self.prediction_steps}

    def metric_context(self,observation=None,goal=None):
        self._assert_reference_fixed()
        if goal is not None and not np.array_equal(np.asarray(goal),self.goal):
            raise ValueError("Context must use current reference")
        shape = () if observation is None else tuple(observation.shape[:-1])
        return torch.full((*shape,1), self.target_angle/.3,dtype=torch.float32)

    def physical_components(self,state):
        x = np.asarray(state,dtype=np.float64)
        return np.array([x[1]-self.target_angle,x[0]-x[1],x[2],x[3]])

    def physical_cost(self,state):
        p = self.physical_components(state)
        return float(p@p)

    def is_success(self,state):
        p = self.physical_components(state)
        return bool(np.isfinite(p).all() and np.all(abs(p)<=np.array([.01,.01,.03,.03])))

    def is_failure(self,state):
        x = np.asarray(state,dtype=np.float64)
        return bool(x.shape != (4,) or not np.isfinite(x).all() or abs(x[0]-x[1])>.5 or np.any(abs(x[2:])>3.))

    def mechanical_energy(self,state):
        tm,tl,wm,wl = np.asarray(state,dtype=np.float64)
        return float(.5*self.Jm*wm**2+.5*self.Jl*wl**2+.5*self.stiffness*(tm-tl)**2)

    def torch_step(self,state,action):
        """Differentiable exact ZOH model; Gym step below remains non-autograd."""
        if not torch.is_tensor(state) or not torch.is_tensor(action):
            raise TypeError("Torch model needs tensors")
        if not state.is_floating_point() or state.shape[-1:] != (4,) or action.shape[-1:] != (1,) or not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError("Expected finite floating state[...,4], torque[...,1]")
        return state@state.new_tensor(self.Ad).T+action.to(state).clamp(-.5,.5)@state.new_tensor(self.Bd).T

    def _validate_metric(self,M):
        if not torch.is_tensor(M) or not M.is_floating_point() or M.shape[-2:] != (4,4) or not torch.isfinite(M).all():
            raise ValueError("Expected finite floating M[...,4,4]")
        eig = torch.linalg.eigvalsh(M)
        if (not torch.allclose(M,M.transpose(-1,-2),atol=1e-7,rtol=1e-7) or
            torch.any(eig<.05-1e-6) or torch.any(eig>3.85+1e-6) or
            not torch.allclose(M.diagonal(dim1=-2,dim2=-1).sum(-1),M.new_full(M.shape[:-2],4.),atol=2e-6,rtol=1e-6)):
            raise ValueError("Metric must obey public trace4/SPD specification")

    def calculate_action_from_M(self,M,e,*,context,controller_state=None,certificate=None):
        self._assert_reference_fixed()
        if controller_state is not None or certificate is not None:
            raise ValueError("Only fixed quadratic M with no hidden controller state is supported")
        if context.get("target_angle") != self.target_angle or context.get("prediction_steps") != self.prediction_steps:
            raise ValueError("Mismatched reference/controller context")
        self._validate_metric(M)
        if (not torch.is_tensor(e) or e.dtype not in (torch.float32,torch.float64) or
            e.device.type!="cpu" or e.shape[-1:] != (4,) or not torch.isfinite(e).all()):
            raise ValueError("This isolated controller supports finite CPU float32/float64 error tensors")
        M = M.to(e)
        prediction = e@e.new_tensor(self.AN).T
        b = e.new_tensor(self.BN[:,0])
        Mb = torch.einsum("...ij,j->...i",M,b)
        raw = -(Mb*prediction).sum(-1)/((Mb*b).sum(-1)+self.torque_regularizer)
        action = raw.clamp(-.5,.5).unsqueeze(-1)
        self._pending_control = None
        if M.shape == (4,4) and e.shape == (4,):
            self._pending_control = {"mode":"metric","M":M.detach().cpu().numpy().copy(),
                "error":e.detach().cpu().numpy().copy(),"action":action.detach().cpu().numpy().copy(),
                "raw_torque":float(raw.detach()),"control_dtype":str(e.dtype)}
        return action

    def unit_metric_action(self,observation=None):
        e,ctx = self.build_error(self.state if observation is None else observation)
        return self.calculate_action_from_M(torch.eye(4,dtype=e.dtype,device=e.device),e,context=ctx)

    def _control_error_matches(self,error,state,goal,dtype_name):
        """Same audit tolerance at execution and restoration, including cancellation.

        Float64 retains the original acceptance rule. Float32 adds a standard
        gamma_5 roundoff envelope propagated from physical state/reference
        magnitudes through |S|, rather than dividing by a near-zero twist.
        This changes audit acceptance only, never the control input or action.
        """
        if dtype_name not in ("torch.float32","torch.float64"):
            return False
        recorded=np.asarray(error,dtype=np.float64)
        physical=self.S@(np.asarray(state)-np.asarray(goal))
        if recorded.shape!=(4,) or not np.isfinite(recorded).all():
            return False
        tolerance=1e-8+1e-7*np.abs(physical)
        if dtype_name=="torch.float32":
            eps=np.finfo(np.float32).eps
            gamma=5*eps/(1-5*eps)
            tolerance=np.maximum(tolerance,gamma*(np.abs(self.S)@(np.abs(state)+np.abs(goal))))
        return bool(np.all(np.abs(recorded-physical)<=tolerance))

    def _info(self):
        now = self._elapsed_steps*self.dt
        good,failure = self.is_success(self.state),self.is_failure(self.state)
        hold = 0. if self._in_goal_since is None else now-self._in_goal_since
        met = bool(good and hold+1e-11 >= self.required_hold_seconds)
        success = bool(self._elapsed_steps>=self.max_steps and met and not failure)
        return {"success":success,"failure":failure,"in_goal":good,"hold_condition_met":met,
            "hold_seconds":max(0.,hold),"required_hold_seconds":2.,
            "settling_time":self._in_goal_since if success else 120.,
            "arrival_time":self._in_goal_since+2. if success else 120.,
            "first_hold_confirmation_time":self._first_hold_confirmation,
            "steps":self._elapsed_steps,"physical_steps":self._elapsed_steps,"elapsed_seconds":now,
            "horizon_seconds":120.,"physical_error":self.physical_cost(self.state),
            "terminal_physical_error":self.physical_cost(self.state),
            "physical_components":self.physical_components(self.state).tolist(),
            "normalized_error":(self.S@(self.state-self.goal)).tolist(),"target_angle_rad":self.target_angle,
            "metric_control_steps":self._metric_steps,"execution_mode":self._execution_mode,
            "execution_origin": "M_policy" if self._execution_mode=="metric" else self._external_origin,
            "metric_fixed_in_episode":self._execution_mode=="metric","cost_convention":"positive_integral",
            "cost_quadrature":"left_physical_error_squared_dt","protocol_kind":self.protocol_kind,
            "prediction_steps":self.prediction_steps,"execution_hold_steps":1,
            "success_scope":"sampled_final_201_states_not_intersample_proof"}

    def step(self,action):
        if self._done:
            raise RuntimeError("Reset before stepping a finished episode")
        self._assert_reference_fixed()
        if self.is_failure(self.state):
            raise ValueError("Current state is already outside the physical domain")
        submitted = np.asarray(action,dtype=np.float64)
        if submitted.shape != (1,) or not np.isfinite(submitted).all():
            raise ValueError("Require one finite physical torque")
        applied = np.clip(submitted,-.5,.5)
        pending = self._pending_control
        mode = "external_physical_action" if pending is None else "metric"
        if self._execution_mode is not None and mode != self._execution_mode:
            raise ValueError("Cannot mix metric and external policies within an episode")
        if pending is not None:
            if not self._control_error_matches(pending["error"],self.state,self.goal,pending["control_dtype"]):
                raise ValueError("Pending control was calculated for another physical state")
            if not np.allclose(pending["action"],submitted,atol=1e-10,rtol=1e-7):
                raise ValueError("Submitted action differs from controller output")
            if self._episode_metric is not None and not np.array_equal(pending["M"],self._episode_metric):
                raise ValueError("Executed M must remain fixed throughout the episode")
        before = self.state.copy()
        following = self.Ad@before+self.Bd[:,0]*applied[0]
        if not np.isfinite(following).all():
            raise FloatingPointError("Nonfinite transition; no state clipping")
        good_before = self.is_success(before)
        self._execution_mode = mode
        if mode=="metric":
            if self._episode_metric is None:
                self._episode_metric = pending["M"].copy()
            self._metric_steps += 1
        self._pending_control = None
        self.state = following.copy()
        self._elapsed_steps += 1
        now = self._elapsed_steps*self.dt
        good = self.is_success(self.state)
        if not good:
            self._in_goal_since = None
        elif not good_before or self._in_goal_since is None:
            self._in_goal_since = now
        hold = 0. if self._in_goal_since is None else now-self._in_goal_since
        self._hold_steps = int(round(max(0.,hold)/self.dt))
        if good and not self.is_failure(self.state) and hold+1e-11>=2. and self._first_hold_confirmation is None:
            self._first_hold_confirmation = now
        info = self._info()
        cost = self.dt*self.physical_cost(before)
        terminated = bool(info["failure"] or info["success"])
        truncated = bool(self._elapsed_steps>=self.max_steps and not terminated)
        self._done = terminated or truncated
        info.update(cost=cost,applied_action=applied.copy(),submitted_action=submitted.copy(),
                    physical_terminated=terminated,physical_truncated=truncated)
        e0,e1 = self.S@(before-self.goal),self.S@(following-self.goal)
        M = self._episode_metric
        self._trace.append({"step":self._elapsed_steps,"time":now,"state_before":before.tolist(),
            "state_after":following.tolist(),"action":applied.tolist(),"submitted_action":submitted.tolist(),
            "raw_torque":None if pending is None else pending["raw_torque"],"cost":cost,"in_goal":good,
            "control_error":None if pending is None else pending["error"].tolist(),
            "control_dtype":None if pending is None else pending["control_dtype"],
            "success":info["success"],"failure":info["failure"],"hold_seconds":info["hold_seconds"],
            "V_before":None if M is None else float(e0@M@e0),
            "V_after":None if M is None else float(e1@M@e1),
            "mechanical_energy_before":self.mechanical_energy(before),
            "mechanical_energy_after":self.mechanical_energy(following),
            "mechanical_power_before":float(applied[0]*before[2]-self.Bm*before[2]**2-self.Bl*before[3]**2)})
        self._last_info = info
        return following.copy(),cost,terminated,truncated,deepcopy(info)

    def get_episode_audit(self):
        return {**deepcopy(self._last_info),"complete":self._done,"initial_seed":self._seed_record,
            "initial_state":self._initial_state.tolist(),"goal":self.goal.tolist(),
            "metric":None if self._episode_metric is None else self._episode_metric.tolist(),
            "trace":deepcopy(self._trace),"training_updates":0,"registered":False,
            "formal_training_authorized_here":False}

    def state_dict(self):
        """Environment-only snapshot at a completed physical-step boundary."""
        self._assert_reference_fixed()
        if self._pending_control is not None:
            raise ValueError("Finish physical step before snapshot; pending autograd control is not serializable")
        return {"schema":"dual-inertia-env-boundary-v2-precision","config":deepcopy(self._config_snapshot),
            "fixed_target":self._fixed_target,"state":self.state.copy(),"goal":self.goal.copy(),
            "initial_state":self._initial_state.copy(),"seed":self._seed_record,
            "elapsed_steps":self._elapsed_steps,"hold_steps":self._hold_steps,"done":self._done,
            "episode_metric":None if self._episode_metric is None else self._episode_metric.copy(),
            "execution_mode":self._execution_mode,"external_origin":self._external_origin,
            "in_goal_since":self._in_goal_since,"first_hold_confirmation":self._first_hold_confirmation,
            "metric_steps":self._metric_steps,"trace":deepcopy(self._trace),"last_info":deepcopy(self._last_info),
            "rng":deepcopy(self.np_random.bit_generator.state)}

    def load_state_dict(self,snapshot):
        """Validate saved completed-step history; never restore learner state."""
        if self._pending_control is not None:
            raise ValueError("Cannot overwrite pending control")
        data = deepcopy(snapshot)
        keys = set(self.state_dict())
        legacy=data.get("schema")=="dual-inertia-env-boundary-v1"
        if set(data)!=keys or data["schema"] not in ("dual-inertia-env-boundary-v1","dual-inertia-env-boundary-v2-precision") or data["config"]!=self._config_snapshot or data["fixed_target"]!=self._fixed_target:
            raise ValueError("Snapshot schema/controller configuration mismatch")
        goal = np.asarray(data["goal"],dtype=np.float64)
        if goal.shape!=(4,):
            raise ValueError("Invalid goal shape")
        target = self._validate_target(goal[0])
        if not np.array_equal(goal,[target,target,0.,0.]):
            raise ValueError("Invalid snapshot goal")
        initial,state = np.asarray(data["initial_state"],dtype=np.float64),np.asarray(data["state"],dtype=np.float64)
        steps = data["elapsed_steps"]
        if (isinstance(steps,bool) or not isinstance(steps,Integral) or not 0<=steps<=self.max_steps or
            initial.shape!=(4,) or self.is_failure(initial) or state.shape!=(4,) or not np.isfinite(state).all() or len(data["trace"])!=steps):
            raise ValueError("Invalid snapshot state/timing")
        metric = data["episode_metric"]
        if metric is not None:
            self._validate_metric(torch.as_tensor(metric,dtype=torch.float64))
            if np.asarray(metric).shape!=(4,4):
                raise ValueError("Snapshot metric cannot be batched")
        mode = data["execution_mode"]
        if mode not in (None,"metric","external_physical_action") or (mode=="metric")!=(metric is not None) or (steps==0)!=(mode is None):
            raise ValueError("Invalid execution mode/metric")
        if data["external_origin"] not in ("external_physical_action","fixed_zero_feedforward","deliberate_fault_injection_not_M_policy_performance") or data["metric_steps"] != (steps if mode=="metric" else 0):
            raise ValueError("Invalid execution origin/count")
        def good(x):
            return bool(np.all(abs(np.array([x[1]-target,x[0]-x[1],x[2],x[3]]))<=np.array([.01,.01,.03,.03])))
        since,first = (0. if good(initial) else None),None
        previous = initial
        for index,row in enumerate(data["trace"],1):
            before,after,action = np.asarray(row["state_before"]),np.asarray(row["state_after"]),np.asarray(row["action"])
            if (row["step"]!=index or row["time"]!=index*self.dt or before.shape!=(4,) or after.shape!=(4,) or action.shape!=(1,) or
                not np.isfinite(np.r_[before,after,action]).all() or abs(action[0])>.5 or not np.array_equal(previous,before) or
                not np.allclose(after,self.Ad@before+self.Bd[:,0]*action[0],atol=1e-13,rtol=1e-13)):
                raise ValueError("Snapshot trace/state chain does not match real physics")
            physical = np.array([before[1]-target,before[0]-before[1],before[2],before[3]])
            if not np.isclose(row["cost"],self.dt*float(physical@physical),atol=1e-14,rtol=1e-13):
                raise ValueError("Snapshot trace cost mismatch")
            if not good(after):
                since=None
            elif since is None:
                since=index*self.dt
            hold=0. if since is None else index*self.dt-since
            if good(after) and not self.is_failure(after) and hold+1e-11>=2. and first is None:
                first=index*self.dt
            if bool(row["in_goal"])!=good(after) or bool(row["failure"])!=self.is_failure(after) or not np.isclose(row["hold_seconds"],hold,atol=1e-12):
                raise ValueError("Snapshot hold/failure trace mismatch")
            expected_success=bool(index==self.max_steps and good(after) and hold+1e-11>=2. and not self.is_failure(after))
            submitted=np.asarray(row["submitted_action"],dtype=np.float64)
            if (bool(row["success"])!=expected_success or submitted.shape!=(1,) or not np.isfinite(submitted).all() or
                not np.array_equal(np.clip(submitted,-.5,.5),action)):
                raise ValueError("Snapshot success/submitted action mismatch")
            if metric is None:
                if row["raw_torque"] is not None or row.get("control_error") is not None or row.get("control_dtype") is not None:
                    raise ValueError("External action cannot claim metric raw torque")
                if legacy:
                    row.update(control_error=None,control_dtype=None)
            else:
                physical_error=self.S@(before-goal)
                if legacy:
                    # Version 1's validated engineering trajectories used float64.
                    row.update(control_error=physical_error.tolist(),control_dtype="torch.float64")
                dtype={"torch.float32":torch.float32,"torch.float64":torch.float64}.get(row.get("control_dtype"))
                recorded_error=np.asarray(row.get("control_error"),dtype=np.float64)
                if dtype is None or not self._control_error_matches(recorded_error,before,goal,row.get("control_dtype")):
                    raise ValueError("Snapshot control input/precision does not match physical error")
                # Replay actual control inputs in their original CPU precision.
                # Reconstructing float32 input from float64 state would change it.
                e=torch.tensor(recorded_error,dtype=dtype)
                mat=torch.tensor(metric,dtype=dtype)
                prediction=e@e.new_tensor(self.AN).T
                b=e.new_tensor(self.BN[:,0])
                mb=torch.einsum("ij,j->i",mat,b)
                raw=float(-(mb*prediction).sum()/((mb*b).sum()+self.torque_regularizer))
                if (row["raw_torque"] is None or not np.isclose(row["raw_torque"],raw,atol=1e-12,rtol=1e-12) or
                    not np.isclose(submitted[0],np.clip(raw,-.5,.5),atol=1e-12,rtol=1e-12)):
                    raise ValueError("Snapshot action did not come from saved fixed M")
            powers={"mechanical_energy_before":self.mechanical_energy(before),
                    "mechanical_energy_after":self.mechanical_energy(after),
                    "mechanical_power_before":float(action[0]*before[2]-self.Bm*before[2]**2-self.Bl*before[3]**2)}
            if any(not np.isclose(row[k],v,atol=1e-12,rtol=1e-12) for k,v in powers.items()):
                raise ValueError("Snapshot mechanical energy/power mismatch")
            for label,x in (("V_before",before),("V_after",after)):
                e=self.S@(x-goal)
                expected=None if metric is None else float(e@np.asarray(metric)@e)
                if (expected is None and row[label] is not None) or (expected is not None and (row[label] is None or not np.isclose(row[label],expected,atol=1e-12,rtol=1e-12))):
                    raise ValueError("Snapshot certificate trace mismatch")
            if self.is_failure(after) and index!=steps:
                raise ValueError("Snapshot continues after physical failure")
            previous=after
        expected_done=bool(steps>=self.max_steps or self.is_failure(state))
        hold=0. if since is None else steps*self.dt-since
        if (not np.array_equal(previous,state) or data["in_goal_since"]!=since or data["first_hold_confirmation"]!=first or
            data["hold_steps"]!=int(round(max(0.,hold)/self.dt)) or data["done"]!=expected_done):
            raise ValueError("Snapshot final state/hold/done inconsistent")
        rng=np.random.default_rng(); rng.bit_generator.state=deepcopy(data["rng"])
        # Assign only after all validation, and reconstruct the summary to reject stale audits.
        temp=type(self)(target_angle=self._fixed_target,prediction_steps=self.prediction_steps)
        temp._target_angle=target; temp.goal=goal.copy(); temp._goal_snapshot=goal.copy()
        temp.state=state.copy(); temp._initial_state=initial.copy(); temp._seed_record=data["seed"]
        temp._elapsed_steps=steps; temp._hold_steps=data["hold_steps"]; temp._done=expected_done
        temp._episode_metric=None if metric is None else np.asarray(metric,dtype=np.float64).copy()
        temp._execution_mode=mode; temp._external_origin=data["external_origin"]
        temp._in_goal_since=since; temp._first_hold_confirmation=first; temp._metric_steps=data["metric_steps"]
        for key,value in temp._info().items():
            saved=data["last_info"].get(key)
            if isinstance(value,float):
                matches=isinstance(saved,(int,float)) and math.isfinite(saved) and math.isclose(saved,value,rel_tol=1e-12,abs_tol=1e-12)
            else:
                matches=saved==value
            if not matches:
                raise ValueError("Snapshot last audit mismatch: "+key)
        if steps:
            last=data["trace"][-1]
            extras={"cost":last["cost"],"applied_action":last["action"],"submitted_action":last["submitted_action"],
                    "physical_terminated":bool(last["failure"] or last["success"]),
                    "physical_truncated":bool(steps==self.max_steps and not last["failure"] and not last["success"])}
            if set(data["last_info"])!=set(temp._info())|set(extras) or any(not np.array_equal(data["last_info"].get(k),v) for k,v in extras.items()):
                raise ValueError("Snapshot final physical-step audit mismatch")
        elif set(data["last_info"])!=set(temp._info()):
            raise ValueError("Snapshot initial audit fields mismatch")
        temp._trace=deepcopy(data["trace"]); temp._last_info=deepcopy(data["last_info"])
        temp._np_random=rng
        self.__dict__.update(temp.__dict__)
