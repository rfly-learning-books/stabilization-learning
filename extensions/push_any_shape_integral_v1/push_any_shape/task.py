"""State-dependent certificate facade and the unique accepted-time commit.

The numeric backend is the original body-response model, not contact physics.
The contact backend implements Disk only. No optimizer runs in this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import hmac
import json
import math
import secrets

import numpy as np
import torch

from stabilization_gym.interfaces.m_learning import (
    CapabilityError, CertificateResult, ControlResult, ErrorContext,
    GradientCapabilities, MetricScope, TaskDescription)

from .controller import candidates, hard_choice, tensor_candidates, torch_step
from .integral_certificate import (IntegralConfig, IntegralState,
    certificate_terms, masked_error, pose_coordinates)
from .mode_library import ModeBank


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def actor_digest(actor):
    result = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        result.update(name.encode())
        result.update(str(a.dtype).encode())
        result.update(str(a.shape).encode())
        result.update(a.tobytes())
    return result.hexdigest()


def _actor_version(actor):
    # Ordinary in-place updates/load_state_dict are versioned by Torch. Direct
    # .data writes are outside this API. Full snapshots bind actual weight bytes.
    return tuple((name, int(value._version), value.data_ptr())
                 for name, value in list(actor.named_parameters()) + list(actor.named_buffers()))


class PushAnyShapeTask:
    """One physical trajectory; tensor_candidates separately supports batches.

    query/control are pure. step accepts a bound ControlResult exactly once.
    There is no implicit actor update and no hidden fixed-M training adapter.
    """
    def __init__(self, shape='Disk', backend='numeric', *, cfg=None, bank=None,
                 device='cpu', dtype=torch.float64, max_steps=1500):
        if backend not in ('numeric', 'contact'):
            raise ValueError('backend must be numeric or contact')
        if backend == 'contact' and shape != 'Disk':
            raise CapabilityError('Contact execution is implemented for Disk only')
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('Positive integer horizon required')
        self.cfg = cfg or IntegralConfig()
        self.bank = bank or ModeBank(device=device, dtype=dtype)
        if shape not in self.bank.names:
            raise ValueError('Unknown PushAnyShape shape: ' + str(shape))
        self.shape, self.backend, self.max_steps = shape, backend, max_steps
        self.device, self.dtype = self.bank.C.device, self.bank.C.dtype
        self.index = self.bank.names.index(shape)
        self.rotation_free = bool(self.bank.rotation_free[self.index])
        self.executor = None
        if backend == 'contact':
            from .contact_executor import DiskContactExecutor
            self.executor = DiskContactExecutor(shape)
            if self.executor.config.dt != self.cfg.dt:
                raise ValueError('Controller and physical execution periods differ')
        self._ready, self._done = False, False
        self._revision = 0
        self._secret = secrets.token_bytes(32)
        self._rng = np.random.default_rng(0)
        self._last_actor = None
        self._context_signature = self._shape_context_digest()
        self._task_signature = _digest(dict(schema='push_any_shape_integral_v1',
            shape=shape, backend=backend, cfg=asdict(self.cfg), horizon=max_steps,
            response=self.bank.fingerprints[shape], dtype=str(self.dtype)))
        self.description = TaskDescription(
            name='push_any_shape_integral_v1:' + shape, backend=backend,
            support_level='stateful_execution_and_differentiable_candidates_not_optimizer',
            metric_scope=MetricScope.STATE_DEPENDENT, conditioning=('masked_D', 'masked_eta', 'C', 'valid'),
            error_dim=6, raw_metric_dim=21,
            action_description='12-mode index and nonnegative inward speed [m/s]; execution phases in contact backend',
            certificate_factor=1., gradients=GradientCapabilities(False, False, True, True,
                model_wrt_action=True, note='Hard selection and physical step are not differentiable; nominal tensor candidates are.'),
            controller_form='Push mode-speed flow/endpoint filtering with augmented signed integral certificate',
            controller_has_state=True, control_call_is_pure=True,
            cost_convention='negative position-error integral returned as reward; not a success criterion',
            supported_algorithms=(), missing_capabilities=(
                'No optimizer/rollout learner accepted through this facade yet',
                'Contact velocities and retreat gap are outside the six-dimensional certificate',
                'Disk C is a nominal ideal response, not calibrated contact response'))

    def describe(self):
        return self.description

    def require_algorithm(self, name):
        raise CapabilityError(f'{name} is not implemented through this state-dependent integral facade')

    def require_capability(self, name):
        value = self.description.control_call_is_pure if name == 'pure_control' else getattr(self.description.gradients, name, False)
        if value is not True:
            raise CapabilityError('Unsupported capability: ' + name)
        return self

    def _tensor(self, value, *, boolean=False):
        return torch.as_tensor(value, device=self.device,
            dtype=torch.bool if boolean else self.dtype)

    @property
    def pose(self):
        return self.executor.pose if self.executor else self._pose.copy()

    @property
    def state(self):
        return self.executor.state if self.executor else self._pose.copy()

    def _error(self, pose):
        q = self._tensor(pose).reshape(1, 3)
        target = self._tensor(self._canonical_target).reshape(1, 3)
        delta = (q[:, :2] - target[:, :2]) / self.cfg.length_scale
        a = target[:, 2]
        xy = torch.stack((torch.cos(a) * delta[:, 0] + torch.sin(a) * delta[:, 1],
                         -torch.sin(a) * delta[:, 0] + torch.cos(a) * delta[:, 1]), -1)
        # Disk has canonical target yaw zero; retain actual body yaw solely for
        # rotating C, while masked_error removes yaw from actor and certificate.
        yaw = q[:, 2] - a
        if self.rotation_free:
            yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        return torch.cat((xy, yaw[:, None]), -1)

    def _shape_context_digest(self):
        i = self.index
        return _digest(dict(C=self.bank.C[i],valid=self.bank.valid[i],
            mask=self.bank.task_mask[i],upper=self.bank.speed_upper[i],
            vertices=self.bank.vertices[i],vertex_mask=self.bank.vertex_mask[i]))

    def _at_position(self):
        e = self._error(self.pose)
        return bool(torch.linalg.vector_norm(e[0, :2]) * self.cfg.length_scale <= .005
                    and (self.rotation_free or abs(float(e[0, 2])) <= self.cfg.angle_tol))

    def _data(self):
        return dict(e=self._error(self.pose), eta=self.integral.eta.clone(),
            shape=torch.tensor([self.index], device=self.device),
            task_mask=self.bank.task_mask[self.index:self.index + 1].clone(),
            target=self._tensor(self._canonical_target).reshape(1, 3).clone(),
            mode=torch.tensor([self._mode], device=self.device),
            goal_hold=self._tensor([self._at_position()], boolean=True),
            blocked_axes=None,
            speed_upper=self.bank.speed_upper[self.index:self.index + 1].clone())

    def _origin(self):
        if self._context_signature != self._shape_context_digest():
            raise ValueError('Shape response/geometry context changed after construction')
        if self._goal_signature != _digest([self.target,self._canonical_target]):
            raise ValueError('Goal changed without an explicit reset/reference version')
        return _digest(dict(task=self._task_signature, episode=self._episode,
            revision=self._revision, pose=self.pose, state=self.state,
            target=self.target,canonical_target=self._canonical_target,
            integral=self.integral.snapshot(), mode=self._mode, recovery=self._recovery,
            executor=self.executor.snapshot() if self.executor else None))

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        rng = np.random.default_rng(seed)
        if self.executor:
            desired_yaw = float(options.pop('target_yaw', 0.))
            if not math.isfinite(desired_yaw):
                raise ValueError('Finite target_yaw required')
            self.executor.reset(seed=seed, options=options)
            self.target = np.r_[self.executor.goal_xy, desired_yaw]
        else:
            if set(options) - {'pose', 'target'}:
                raise ValueError('Numeric reset accepts pose and target only')
            disk = self.shape == 'Disk'
            default = [-.025, 0., 0.] if disk else [.225, .25, 0.]
            goal = [0., 0., 0.] if disk else [.25, .25, 0.]
            q = np.asarray(options.get('pose', default), dtype=float)
            g = np.asarray(options.get('target', goal), dtype=float)
            if q.shape != (3,) or g.shape != (3,) or not np.isfinite(np.r_[q,g]).all():
                raise ValueError('Finite three-dimensional pose and target required')
            if disk and (np.any(q[:2] < [-.055,-.060]) or np.any(q[:2] > [.060,.060])):
                raise ValueError('Disk numeric state outside declared workspace')
            if not disk:
                if abs(q[2]-g[2]) >= self.cfg.angle_chart_limit:
                    raise ValueError('Ordinary shapes require the declared continuous yaw chart')
                ring = self.bank.vertices[self.index][self.bank.vertex_mask[self.index]].cpu().numpy()
                for qp in (q,g):
                    rot = np.array([[math.cos(qp[2]),-math.sin(qp[2])],[math.sin(qp[2]),math.cos(qp[2])]])
                    p = qp[:2] + ring @ rot.T
                    if p.min() < 0. or p.max() > self.cfg.workspace_m:
                        raise ValueError('Numeric polygon outside declared workspace')
            self._pose, self.target = q.copy(), g.copy()
        if not np.isfinite(self.target).all():
            raise ValueError('Finite goal required')
        self._canonical_target = self.target.copy()
        if self.rotation_free:
            self._canonical_target[2] = 0.
        self._goal_signature = _digest([self.target,self._canonical_target])
        self._rng, self._seed = rng, seed
        self._episode = secrets.token_hex(16)
        self._revision += 1
        self._mode, self._recovery, self._steps = -1, False, 0
        self.integral = IntegralState.zeros(1, cfg=self.cfg, device=self.device, dtype=self.dtype)
        self._hold_start = 0. if self._physical_goal() else None
        self._ready, self._done, self._last_actor = True, False, None
        return self.state, {'shape':self.shape, 'backend':self.backend, 'target':self.target.copy()}

    def _physical_goal(self):
        if self.executor:
            cond = self.executor.physical_goal_conditions()
            return all(bool(cond[k]) for k in ('position','disk_stopped','yaw_rate_stopped','pusher_stopped','pusher_separated','safe'))
        return self._at_position()

    def error_context(self, observation=None):
        if not self._ready:
            raise RuntimeError('Reset task before querying')
        if observation is not None and not np.array_equal(np.asarray(observation), self.state):
            raise ValueError('Live facade requires current observation; offline use tensor_candidates')
        data = self._data()
        return ErrorContext(masked_error(data['e'], data['eta'], data['task_mask']),
            {'C':self.bank.C[self.index:self.index+1].clone(), 'valid':self.bank.valid[self.index:self.index+1].clone()},
            {'data':data, 'origin':self._origin()}, (self._episode,self._revision))

    def _validate_sample(self, sample):
        if not isinstance(sample, ErrorContext) or sample.representation_version != (self._episode,self._revision):
            raise ValueError('Stale episode/reference/error context')
        if sample.control_context['origin'] != self._origin():
            raise ValueError('Physical or controller state changed since query')
        expected = self._data()
        if _digest(sample.control_context['data']) != _digest(expected):
            raise ValueError('Modified control context')
        if not torch.equal(sample.error, masked_error(expected['e'],expected['eta'],expected['task_mask'])):
            raise ValueError('Modified augmented error')
        if not torch.equal(sample.metric_context['C'],self.bank.C[self.index:self.index+1]) or not torch.equal(sample.metric_context['valid'],self.bank.valid[self.index:self.index+1]):
            raise ValueError('Modified shape context')

    def certificate(self, actor, sample=None, *, training=False):
        if torch.is_tensor(actor):
            raise CapabilityError('State-dependent M requires an actor, not a detached matrix')
        if self._last_actor is not None and self._last_actor != actor_digest(actor):
            raise ValueError('Actor weights changed within the trajectory; reset explicitly')
        sample = self.error_context() if sample is None else sample
        self._validate_sample(sample)
        data = sample.control_context['data']
        c = certificate_terms(actor,sample.error,sample.metric_context['C'],sample.metric_context['valid'],
            task_mask=data['task_mask'],training=training)
        return CertificateResult(c['M'],c['V'],c['L'],c['gradient'])

    def control(self, actor, sample=None, *, training=False):
        if self._done:
            raise RuntimeError('Episode finished; reset before control')
        if torch.is_tensor(actor):
            raise CapabilityError('State-dependent M requires an actor, not a detached matrix')
        if self._last_actor is not None and self._last_actor != actor_digest(actor):
            raise ValueError('Actor weights changed within the trajectory; reset explicitly')
        sample = self.error_context() if sample is None else sample
        self._validate_sample(sample)
        data = sample.control_context['data']
        if self.shape == 'Disk':
            def physical(ep):
                p = ep[..., :2] * self.cfg.length_scale + ep.new_tensor(self._canonical_target[:2])
                return ((p >= p.new_tensor([-.055,-.060])) & (p <= p.new_tensor([.060,.060]))).all(-1)
            cand = tensor_candidates(actor,data['e'],data['eta'],sample.metric_context['C'],sample.metric_context['valid'],
                self.cfg,task_mask=data['task_mask'],current_mode=data['mode'],physical_mask=physical,
                training=training,goal_hold=data['goal_hold'],speed_upper=data['speed_upper'])
        else:
            cand = candidates(actor,data,self.bank,self.cfg,training=training)
        mode,u,recovery,fail = hard_choice(cand,data,self.cfg,
            recovery=self._tensor([self._recovery],boolean=True),at_goal=data['goal_hold'])
        mode,u = int(mode[0]),float(u[0].detach())
        source = 'pose_yaw_recovery' if bool(recovery[0]) else 'certificate_mode_speed'
        if self.executor and self.executor.busy and not bool(data['goal_hold'][0]):
            # Transit is an explicit geometric controller with latched intent.
            # It can increase V and is never claimed to satisfy the PUSH model.
            mode,u = self.executor.current_action
            source = 'latched_geometric_transit'
        elif bool(data['goal_hold'][0]):
            source = 'goal_disengage_and_hold'
        elif bool(fail[0]):
            source = 'no_descent_candidate_zero_command'
        payload = dict(origin=sample.control_context['origin'],mode=mode,u=u,
            recovery=bool(recovery[0]),no_descent_candidate=bool(fail[0]),
            goal_hold=bool(data['goal_hold'][0]),source=source,
            actor_version=_actor_version(actor))
        payload['signature'] = hmac.new(self._secret,_digest(payload).encode(),hashlib.sha256).hexdigest()
        cert = CertificateResult(cand['M'],cand['V'],cand['L'],cand['gradient'])
        low = float(self.bank.speed_lower[self.index,mode])
        high = float(self.bank.speed_upper[self.index,mode])
        return ControlResult({'mode':mode,'u':u},cert,payload,
            {'source':source,'no_descent_candidate':bool(fail[0]),
             'feasible_count':int(cand['feasible'].sum()),'claim':'candidate_only',
             'response_kind':self.bank.response_kinds[self.shape],
             'response_speed_domain_mps':[low,high],
             'outside_preserved_response_speed_domain':bool(u > 0 and (u < low or u > high))},
            cand['logits'] if training else None)

    def step(self, result, *, actor):
        if not self._ready or self._done:
            raise RuntimeError('Reset before execution')
        if not isinstance(result,ControlResult):
            raise TypeError('step requires a current bound ControlResult')
        payload = deepcopy(result.next_controller_state)
        signed = payload.pop('signature')
        if not hmac.compare_digest(signed,hmac.new(self._secret,_digest(payload).encode(),hashlib.sha256).hexdigest()):
            raise ValueError('Modified control result')
        if payload['origin'] != self._origin() or payload['actor_version'] != _actor_version(actor):
            raise ValueError('Stale/duplicate action or actor update between query and commit')
        if result.action != {'mode':payload['mode'],'u':payload['u']}:
            raise ValueError('Action differs from its bound controller payload')
        before_e = self._error(self.pose)
        before = self.state
        V_before = float(self.certificate(actor).value.detach()[0])
        if self.executor:
            execution = self.executor.step(payload['mode'],payload['u'],goal_hold=payload['goal_hold'])
            h,blocked = execution['actual_dt'],execution['blocked_axes']
        else:
            ep = torch_step(before_e,self.bank.C[self.index,payload['mode']][None,:],
                before_e.new_tensor([payload['u']]),self.cfg.dt,self.cfg.length_scale)
            a = self._canonical_target[2]
            x,y = ep[0,:2].detach().cpu().numpy() * self.cfg.length_scale
            self._pose = np.array([self._canonical_target[0]+math.cos(a)*x-math.sin(a)*y,
                self._canonical_target[1]+math.sin(a)*x+math.cos(a)*y,a+float(ep[0,2])])
            h,blocked = self.cfg.dt,None
            execution = dict(actual_dt=h,before=before,after=self.state,phase='NUMERIC_RESPONSE',
                command=[payload['mode'],payload['u']],failure=False,substep_records=[],
                diagnostics={'nominal_C_prediction_applicable':True},blocked_axes_available=False)
        after_e = self._error(self.pose)
        mask = self.bank.task_mask[self.index:self.index+1]
        self.integral = self.integral.advance(pose_coordinates(before_e)*mask,pose_coordinates(after_e)*mask,
            h,task_mask=mask,blocked=blocked,goal_hold=payload['goal_hold'],cfg=self.cfg)
        if h > 0:
            self._steps += 1
        self._mode,self._recovery = payload['mode'],payload['recovery']
        self._revision += 1
        self._last_actor = actor_digest(actor)
        t = float(self.integral.accepted_time_s[0])
        goal = self._physical_goal()
        self._hold_start = (t if self._hold_start is None else self._hold_start) if goal else None
        held = self._hold_start is not None and t-self._hold_start >= 1.-1e-10
        terminated = bool(execution['failure'])
        truncated = self._steps >= self.max_steps
        self._done = terminated or truncated
        V_after = float(self.certificate(actor).value.detach()[0])
        pos = float(torch.linalg.vector_norm(after_e[0,:2]) * self.cfg.length_scale)
        info = dict(execution=execution,eta=self.integral.eta.detach().clone(),actual_dt=h,
            V_before=V_before,V_after=V_after,delta_V=V_after-V_before,
            control_source=payload['source'],no_descent_candidate=payload['no_descent_candidate'],
            goal_conditions=self.executor.physical_goal_conditions() if self.executor else {'pose':goal},
            position_error_m=pos,goal_held=bool(held),success=bool(held and truncated and not terminated),
            accepted_time_s=t,accepted_steps=self._steps,certificate_claim='pose_integral_candidate_only')
        return self.state,-pos*pos*h,terminated,truncated,info

    def snapshot(self, actor):
        if not self._ready:
            raise RuntimeError('Reset before snapshot')
        self._origin()  # Validate fixed reference and response context as well.
        if self._last_actor is not None and self._last_actor != actor_digest(actor):
            raise ValueError('Cannot relabel a trajectory snapshot with changed actor weights')
        record = dict(schema='push_any_shape_integral_task_v1',task_signature=self._task_signature,
            shape=self.shape,backend=self.backend,pose=self.pose,target=self.target,
            canonical_target=self._canonical_target,integral=self.integral.snapshot(),
            mode=self._mode,recovery=self._recovery,steps=self._steps,done=self._done,
            hold_start=self._hold_start,rng=self._rng.bit_generator.state,seed=self._seed,
            actor_sha256=actor_digest(actor),executor=self.executor.snapshot() if self.executor else None)
        record = jsonable(record)
        record['sha256'] = _digest(record)
        return record

    def restore(self, payload, actor):
        record = deepcopy(payload)
        fields = {'schema','task_signature','shape','backend','pose','target','canonical_target',
            'integral','mode','recovery','steps','done','hold_start','rng','seed','actor_sha256','executor','sha256'}
        if not isinstance(record,dict) or set(record) != fields:
            raise ValueError('Snapshot fields differ from complete task schema')
        sha = record.pop('sha256')
        if _digest(record) != sha or record['task_signature'] != self._task_signature or record['actor_sha256'] != actor_digest(actor):
            raise ValueError('Snapshot checksum/protocol/actor mismatch')
        if record['schema'] != 'push_any_shape_integral_task_v1' or record['shape'] != self.shape or record['backend'] != self.backend:
            raise ValueError('Snapshot task mismatch')
        integral = IntegralState.from_snapshot(record['integral'],cfg=self.cfg,device=self.device,dtype=self.dtype)
        if integral.eta.shape != (1,3) or (self.rotation_free and float(integral.eta[0,2]) != 0.):
            raise ValueError('Malformed/masked integral state')
        target,canonical,pose = (np.asarray(record[k],float) for k in ('target','canonical_target','pose'))
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (target,canonical,pose)):
            raise ValueError('Invalid snapshot pose/target')
        expected = target.copy()
        if self.rotation_free:
            expected[2] = 0.
        if not np.array_equal(expected,canonical):
            raise ValueError('Invalid canonical target frame')
        steps = record['steps']
        if type(steps) is not int or not 0 <= steps <= self.max_steps or steps != int(integral.accepted_steps[0]):
            raise ValueError('Inconsistent physical/integral step counters')
        if type(record['mode']) is not int or not -1 <= record['mode'] < 12 or type(record['recovery']) is not bool or type(record['done']) is not bool:
            raise ValueError('Invalid saved controller mode/flags')
        hold = record['hold_start']
        if hold is not None and (not math.isfinite(hold) or not 0 <= hold <= float(integral.accepted_time_s[0])):
            raise ValueError('Invalid hold clock')
        rng = np.random.default_rng()
        rng.bit_generator.state = record['rng']
        elapsed = float(integral.accepted_time_s[0])
        if not 0 <= elapsed <= steps*self.cfg.dt+1e-9:
            raise ValueError('Invalid accepted clock')
        if steps and elapsed <= (steps-1)*self.cfg.dt-1e-9:
            raise ValueError('Only the final failed control step may be partial')
        failed = bool(record['executor']['failed']) if self.executor else False
        if not failed and abs(elapsed-steps*self.cfg.dt) > 1e-9:
            raise ValueError('Nonfailed trajectories must have full-period accepted clocks')
        if record['done'] != bool(steps >= self.max_steps or failed):
            raise ValueError('Inconsistent horizon/physical-failure/done flags')
        if steps and record['mode'] < 0 or self.rotation_free and record['recovery']:
            raise ValueError('Invalid saved control mode/recovery')
        if not self.rotation_free and abs(pose[2]-canonical[2]) >= self.cfg.angle_chart_limit:
            raise ValueError('Saved yaw outside the continuous control chart')
        if self.executor:
            es = record['executor']
            if not np.array_equal(es['goal_xy'],target[:2]) or es['budget']['accepted_control_steps'] != steps:
                raise ValueError('Executor clock/goal mismatch')
            if not np.array_equal(es['integration'][1:4],pose):
                raise ValueError('Executor pose differs from facade snapshot')
            if abs(es['integration'][0]-float(integral.accepted_time_s[0])) > 1e-10:
                raise ValueError('Executor and integral accepted clocks differ')
            self.executor.restore(es)
        elif record['executor'] is not None:
            raise ValueError('Numeric snapshot contains contact state')
        self._pose,self.target,self._canonical_target = pose.copy(),target.copy(),canonical.copy()
        self._goal_signature = _digest([self.target,self._canonical_target])
        self.integral,self._steps,self._mode,self._recovery = integral,steps,record['mode'],record['recovery']
        self._done,self._hold_start,self._rng,self._seed = record['done'],hold,rng,record['seed']
        self._episode,self._secret = secrets.token_hex(16),secrets.token_bytes(32)
        self._revision += 1
        self._ready,self._last_actor = True,record['actor_sha256']

    def close(self):
        if self.executor:
            self.executor.close()


def make_task(shape='Disk', backend='numeric', **kwargs):
    return PushAnyShapeTask(shape,backend,**kwargs)
