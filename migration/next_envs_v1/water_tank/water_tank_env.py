"""Unregistered nominal water tank with explicit, known integral memory.

M defines both V=e.T@M@e and an analytic sampled controller. The strict
continuous unsaturated decrease identity requires m12<0; the unrestricted
public SPD map also permits nondecreasing V and non-target equilibria.
Actual physics uses a nonlinear constant-input scalar flow, not Euler or
the local linearization. No physical-state clipping or anti-windup is used.
"""
from __future__ import annotations

from copy import deepcopy
import math

import gymnasium as gym
import numpy as np
from scipy.optimize import brentq
import torch

from stabilization_gym.envs.base import StateMetricEnv


class WaterTankIntegralEnv(StateMetricEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    controller_form = "same_M_continuous_analytic_integral_feedback_sampled_with_ZOH"
    control_call_is_pure = False
    control_call_side_effects = ("audit_pending_metric_and_action",)

    def __init__(self, *, target_height=None, render_mode=None):
        super().__init__(state_dim=2, action_low=[0.], action_high=[1.], error_dim=2,
            dt=.05, max_steps=1200, goal=[.49, 0.], initial_scale=.4,
            success_tolerance=.01, metric_floor=.05, render_mode=render_mode)
        self._fixed_target = None if target_height is None else self._validate_target(target_height)
        self._target_height = .49 if target_height is None else self._fixed_target
        self.goal = np.array([self._target_height, 0.])
        self._goal_snapshot = self.goal.copy()
        self.area_m2, self.ku, self.ko = 1., .1, .1
        self.height_scale_m, self.integral_time_scale_s, self.k_per_s = 1., 10., .1
        self.error_scale = np.array([1., 10.])
        self.metric_context_dim = 1
        self.height_tolerance_m, self.memory_tolerance_m_s = .01, .1
        self.required_hold_seconds, self.horizon_seconds = 2., 60.
        self.protocol_kind = "water_tank_nominal_integral_preparation_v1"
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float64)
        self.controller_capabilities.update(model_gradient=False,
            differentiable_torch_model=False, physical_step_autograd=False,
            full_horizon_evaluation=True, full_state_observation=True,
            state_coordinate_units=("m", "m*s known integral memory"),
            action_error_gradient_domain="height>0, away from saturation switches",
            certificate_guarantee="candidate_only")
        self._pending_control = self._episode_metric = None
        self._execution_mode = None
        self._since = {"physical": None, "augmented": None}
        self._first_hold = {"physical": None, "augmented": None}
        self._metric_steps = 0
        self._trace, self._last_info = [], {}
        self._initial_state, self._seed_record = self.state.copy(), None

    @staticmethod
    def _validate_target(value):
        if isinstance(value, (bool, np.bool_)) or np.asarray(value).shape != ():
            raise ValueError("Target height must be a scalar in metres")
        value = float(value)
        if not math.isfinite(value) or not .25 <= value <= .64:
            raise ValueError("Target height must be in [.25,.64] m")
        return value

    @property
    def target_height(self):
        return self._target_height

    @property
    def equilibrium_opening(self):
        return self.ko * math.sqrt(self.target_height) / self.ku

    def _assert_reference_fixed(self):
        if not np.array_equal(self.goal, self._goal_snapshot) or not np.array_equal(self.goal, [self.target_height, 0.]):
            raise ValueError("Reference changed; use reset for a new target")

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = {} if options is None else dict(options)
        if set(options)-{"state", "target_height"}:
            raise ValueError("Only state and target_height reset options are supported")
        target = options.get("target_height", self._fixed_target)
        self._target_height = self._validate_target(self.np_random.uniform(.25,.64) if target is None else target)
        self.goal = np.array([self.target_height, 0.])
        self._goal_snapshot = self.goal.copy()
        default_state = [self.np_random.uniform(.1,.9), 0.]
        state = np.asarray(options.get("state", default_state), dtype=np.float64)
        if state.shape != (2,) or not np.isfinite(state).all() or not 0. <= state[0] <= 1.:
            raise ValueError("Reset requires h in [0,1] m and any finite known memory xi")
        self.state = state.copy()
        self._initial_state, self._seed_record = self.state.copy(), seed
        self._elapsed_steps, self._hold_steps, self._done = 0, 0, False
        self._pending_control = self._episode_metric = None
        self._execution_mode, self._metric_steps = None, 0
        self._since = {name: 0. if good else None for name, good in self.goal_flags(state).items()}
        self._first_hold = {"physical": None, "augmented": None}
        self._trace = []
        self._last_info = self._info()
        return self.state.copy(), dict(self._last_info)

    def tensor_error(self, state):
        state = state if torch.is_tensor(state) else torch.as_tensor(state, dtype=torch.float64)
        if not state.is_floating_point() or state.shape[-1:] != (2,) or not torch.isfinite(state).all():
            raise ValueError("State must be finite floating [...,2]")
        return (state-state.new_tensor(self.goal))/state.new_tensor(self.error_scale)

    def build_error(self, observation, goal=None, representation=None, history=None):
        self._assert_reference_fixed()
        if representation is not None and not getattr(representation, "preserves_physical_coordinates", False):
            raise ValueError("The measured level and known integral memory must be preserved")
        state = observation if torch.is_tensor(observation) else torch.as_tensor(observation, dtype=torch.float64)
        target = state.new_tensor(self.goal) if goal is None else torch.as_tensor(goal).to(state)
        if target.shape != (2,) or not torch.equal(target, state.new_tensor(self.goal)):
            raise ValueError("Only the current episode reference is supported, including batched states")
        return self.tensor_error(state), {"state": state, "goal": target, "target_height": self.target_height}

    def metric_context(self, observation=None, goal=None):
        self._assert_reference_fixed()
        if goal is not None and not np.array_equal(np.asarray(goal), self.goal):
            raise ValueError("Metric condition cannot use another reference")
        value = 2.*(self.target_height-.25)/(.64-.25)-1.
        shape = () if observation is None else tuple(observation.shape[:-1])
        return torch.full((*shape, 1), value, dtype=torch.float32)

    def goal_flags(self, state):
        state = np.asarray(state)
        physical = bool(np.isfinite(state).all() and abs(state[0]-self.target_height) <= self.height_tolerance_m)
        return {"physical": physical, "augmented": physical and bool(abs(state[1]) <= self.memory_tolerance_m_s)}

    def physical_cost(self, state):
        # Physical height error in canonical 1 m units. Memory is reported separately.
        return float((np.asarray(state)[0]-self.target_height)**2)

    def is_success(self, state):
        return self.goal_flags(state)["augmented"]

    def is_failure(self, state):
        state = np.asarray(state)
        return bool(not np.isfinite(state).all() or state[0] < 0. or state[0] > 1.)

    @staticmethod
    def _w_minus_one_minus_exp(w):
        # w-(1-exp(-w)) without cancellation for small w.
        if w < 1e-4:
            return w*w*(.5+w*(-1./6+w*(1./24+w*(-1./120+w/720))))
        return w+math.expm1(-w)

    def integrate_zoh(self, state, opening, *, dt=None):
        """Non-autograd nonlinear constant-opening flow and exact memory integral.

        Solve for logarithmic distance w to sqrt(h)'s equilibrium. The root
        expression is stable near the equilibrium and starts from h=0. At
        zero input use the exact finite-time draining solution. No step ever
        recomputes the controller, clips h, or uses Euler integration for xi.
        """
        state = np.asarray(state, dtype=np.float64)
        duration = self.dt if dt is None else float(dt)
        u = float(opening)
        if state.shape != (2,) or not np.isfinite(state).all() or state[0] < 0.:
            raise ValueError("Integrator requires nonnegative height and finite memory")
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(u) or not 0. <= u <= 1.:
            raise ValueError("Require positive finite duration and opening in [0,1]")
        y0, c = math.sqrt(state[0]), self.ko/self.area_m2
        dry_time, residual, w = 0., 0., None
        if u == 0.:
            draining = min(duration, 2*y0/c)
            y1 = y0-c*draining/2
            integral_h = 2*(y0-y1)*(y0*y0+y0*y1+y1*y1)/(3*c)
            dry_time = duration-draining
            method = "exact_zero_input_draining"
        else:
            p = self.ku*u/self.ko
            if y0 == p:
                y1, integral_h, method = y0, state[0]*duration, "exact_equilibrium"
            else:
                scaled_time = c*duration/2
                def residual_function(v):
                    return p*self._w_minus_one_minus_exp(v)+y0*(-math.expm1(-v))-scaled_time
                upper = 1.+scaled_time/p
                if not math.isfinite(upper):
                    raise FloatingPointError("Opening too small for representable positive-flow root; refusing approximation")
                w = brentq(residual_function, 0., upper, xtol=1e-14, rtol=8*np.finfo(float).eps, maxiter=200)
                decay = math.exp(-w)
                y1 = y0*decay+p*(-math.expm1(-w))
                # Long double and factored differences reduce subtraction error.
                a,b,q,t,rate = map(np.longdouble, (y0,y1,p,duration,c))
                integral_h = float(q*q*t-2/rate*(b-a)*((b*b+a*b+a*a)/3+q*(b+a)/2))
                residual = abs(residual_function(w))*2/c
                method = "nonlinear_implicit_constant_input_flow"
        height = y1*y1
        lower, upper_h = min(state[0],height)*duration, max(state[0],height)*duration
        if (not math.isfinite(height) or not math.isfinite(integral_h) or
            integral_h < lower-2e-11 or integral_h > upper_h+2e-11 or integral_h < -2e-14):
            raise FloatingPointError("Nonlinear flow or mass integral failed physical bounds; no clipping applied")
        following = np.array([height, state[1]+self.target_height*duration-integral_h])
        return following, {"method": method, "constant_opening": u, "duration_s": duration,
            "integral_height_m_s": integral_h, "dry_time_s": dry_time,
            "root_time_residual_s": residual, "log_distance_coordinate": w}

    def torch_step(self, state, action):
        raise NotImplementedError("Actual implicit nonlinear ZOH solver has no autograd contract; model_gradient=False")

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        self._assert_reference_fixed()
        if controller_state is not None or certificate is not None:
            raise ValueError("Use the fixed quadratic M; integral memory is explicitly in the state")
        if not torch.is_tensor(M) or not torch.is_tensor(e) or not e.is_floating_point():
            raise TypeError("M/error must be floating tensors retaining the caller graph")
        if M.shape[-2:] != (2,2) or e.shape[-1:] != (2,) or not torch.isfinite(M).all() or not torch.isfinite(e).all():
            raise ValueError("Invalid M/error dimensions or nonfinite values")
        if context.get("target_height") != self.target_height:
            raise ValueError("Stale or mismatched reference")
        M = M.to(e)
        eigen = torch.linalg.eigvalsh(M)
        if (not torch.allclose(M,M.transpose(-1,-2),atol=1e-7,rtol=1e-7)
            or torch.any(eigen < self.metric_spec.eigen_floor-1e-6)
            or torch.any(eigen > self.metric_spec.eigen_ceiling+1e-6)
            or not torch.allclose(M.diagonal(dim1=-2,dim2=-1).sum(-1),M.new_full(M.shape[:-2],2.),atol=2e-6,rtol=1e-6)):
            raise ValueError("M must satisfy the unrestricted public SPD/trace specification")
        h = self.target_height+self.height_scale_m*e[...,0]
        if torch.any(h < 0.):
            raise ValueError("Error implies negative height, outside the physical model")
        m11,m12,m22 = M[...,0,0],M[...,0,1],M[...,1,1]
        delta = m11*m22-m12*m12
        s = m11*e[...,0]+m12*e[...,1]
        b = self.ku/(self.area_m2*self.height_scale_m)
        correction = (-self.k_per_s*s+delta*e[...,1]/(self.integral_time_scale_s*m11)
                      +m12*e[...,0]/self.integral_time_scale_s)/(m11*b)
        raw = self.ko*torch.sqrt(h)/self.ku+correction
        action = raw.clamp(0.,1.).unsqueeze(-1)
        self._pending_control = None
        if M.shape == (2,2) and e.shape == (2,):
            self._pending_control = {"mode":"metric", "M":M.detach().cpu().numpy().copy(),
                "error":e.detach().cpu().numpy().copy(), "action":action.detach().cpu().numpy().copy(),
                "raw_opening":float(raw.detach())}
        return action

    def fixed_feedforward_action(self):
        self._assert_reference_fixed()
        action = torch.tensor([self.equilibrium_opening],dtype=torch.float64)
        self._pending_control = {"mode":"fixed_feedforward", "M":None,
            "error":(self.state-self.goal)/self.error_scale,"action":action.numpy().copy(),
            "raw_opening":self.equilibrium_opening}
        return action

    def unit_metric_action(self, observation=None):
        e,ctx = self.build_error(self.state if observation is None else observation)
        return self.calculate_action_from_M(torch.eye(2,dtype=e.dtype,device=e.device),e,context=ctx)

    def _info(self):
        now, failure = self._elapsed_steps*self.dt, self.is_failure(self.state)
        flags = self.goal_flags(self.state)
        holds = {n:0. if self._since[n] is None else max(0.,now-self._since[n]) for n in flags}
        met = {n:bool(flags[n] and holds[n]+1e-11 >= self.required_hold_seconds) for n in flags}
        final = self._elapsed_steps >= self.max_steps
        success = {n:bool(final and met[n] and not failure) for n in flags}
        settling = {n:float(self._since[n]) if success[n] else self.horizon_seconds for n in flags}
        arrival = {n:float(self._since[n]+self.required_hold_seconds) if success[n] else self.horizon_seconds for n in flags}
        return {"success":success["augmented"],"physical_only_success":success["physical"],
            "augmented_success":success["augmented"],"failure":failure,
            "in_goal":flags["augmented"],"physical_in_goal":flags["physical"],
            "hold_condition_met":met["augmented"],"physical_hold_condition_met":met["physical"],
            "hold_seconds":holds["augmented"],"physical_hold_seconds":holds["physical"],
            "settling_time":settling["augmented"],"arrival_time":arrival["augmented"],
            "physical_settling_time":settling["physical"],"physical_arrival_time":arrival["physical"],
            "first_hold_confirmation_time":dict(self._first_hold),"steps":self._elapsed_steps,
            "physical_steps":self._elapsed_steps,"elapsed_seconds":now,"horizon_seconds":60.,
            "required_hold_seconds":2.,"physical_error":self.physical_cost(self.state),
            "height_error_m":float(self.state[0]-self.target_height),"memory_error_m_s":float(self.state[1]),
            "normalized_error":((self.state-self.goal)/self.error_scale).tolist(),
            "terminal_physical_error":self.physical_cost(self.state),"target_height_m":self.target_height,
            "equilibrium_opening":self.equilibrium_opening,"metric_control_steps":self._metric_steps,
            "execution_mode":self._execution_mode,"metric_fixed_in_episode":self._execution_mode=="metric",
            "cost_convention":"positive_integral","cost_quadrature":"left_height_error_squared_dt",
            "protocol_kind":self.protocol_kind,"success_scope":"sampled_final_41_states_not_intersample_proof"}

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset before stepping a finished episode")
        self._assert_reference_fixed()
        submitted = np.asarray(action,dtype=np.float64)
        if submitted.shape != (1,) or not np.isfinite(submitted).all():
            raise ValueError("Action must be one finite physical valve opening")
        if self.is_failure(self.state):
            raise ValueError("Current state already violates the physical domain; refusing to hide that violation")
        applied = np.clip(submitted,0.,1.)
        pending = self._pending_control
        mode = "external_physical_action" if pending is None else pending["mode"]
        if self._execution_mode is not None and mode != self._execution_mode:
            raise ValueError("Cannot mix execution policy modes inside one episode")
        if pending is not None:
            if not np.allclose(pending["error"],(self.state-self.goal)/self.error_scale,atol=1e-7,rtol=1e-6):
                raise ValueError("Pending control was calculated for a different state")
            if not np.allclose(pending["action"],applied,atol=1e-10,rtol=1e-7):
                raise ValueError("Submitted action differs from recorded controller output")
            if mode=="metric" and self._episode_metric is not None and not np.array_equal(pending["M"],self._episode_metric):
                raise ValueError("M must stay fixed through the executed episode")
        before = self.state.copy()
        following,integration = self.integrate_zoh(before,applied[0])
        if not np.isfinite(following).all():
            raise FloatingPointError("Invalid flow; refusing nonfinite trajectory data")
        self._execution_mode = mode
        if mode=="metric":
            if self._episode_metric is None:
                self._episode_metric = pending["M"].copy()
            self._metric_steps += 1
        self._pending_control = None
        before_flags = self.goal_flags(before)
        self.state = following.copy()
        self._elapsed_steps += 1
        now, flags = self._elapsed_steps*self.dt,self.goal_flags(self.state)
        for name,good in flags.items():
            if not good:
                self._since[name] = None
            elif not before_flags[name] or self._since[name] is None:
                self._since[name] = now
            hold = 0. if self._since[name] is None else now-self._since[name]
            if good and hold+1e-11 >= 2. and self._first_hold[name] is None:
                self._first_hold[name] = now
        self._hold_steps = int(round((0. if self._since["augmented"] is None else now-self._since["augmented"])/self.dt))
        info = self._info()
        cost = self.dt*self.physical_cost(before)
        terminated = bool(info["failure"] or info["success"])
        truncated = bool(self._elapsed_steps >= self.max_steps and not terminated)
        self._done = terminated or truncated
        info.update(cost=cost,applied_action=applied.copy(),submitted_action=submitted.copy(),
            physical_terminated=terminated,physical_truncated=truncated)
        e0,e1 = (before-self.goal)/self.error_scale,(self.state-self.goal)/self.error_scale
        M = self._episode_metric
        v0,v1,vdot = None,None,None
        if M is not None:
            v0,v1 = float(e0@M@e0),float(e1@M@e1)
            edot = np.array([(self.ku*applied[0]-self.ko*math.sqrt(before[0]))/(self.area_m2*self.height_scale_m),
                             -e0[0]/self.integral_time_scale_s])
            vdot = float(2*e0@M@edot)
        self._trace.append({"step":self._elapsed_steps,"time":now,"state_before":before.tolist(),
            "state_after":self.state.tolist(),"action":applied.tolist(),
            "raw_opening":None if pending is None else pending["raw_opening"],"cost":cost,
            "physical_in_goal":flags["physical"],"in_goal":flags["augmented"],
            "physical_hold_seconds":info["physical_hold_seconds"],"hold_seconds":info["hold_seconds"],
            "success":info["success"],"physical_only_success":info["physical_only_success"],
            "failure":info["failure"],"V_before":v0,"V_after":v1,"actual_Vdot_at_sample":vdot,
            "integration":integration})
        self._last_info = info
        return self.state.copy(),cost,terminated,truncated,dict(info)

    def get_episode_audit(self):
        return {**self._last_info,"complete":self._done,"initial_seed":self._seed_record,
            "initial_state":self._initial_state.tolist(),"goal":self.goal.tolist(),
            "metric":None if self._episode_metric is None else self._episode_metric.tolist(),
            "trace":deepcopy(self._trace),"training_updates":0,"registered":False,
            "formal_training_authorized_here":False}
