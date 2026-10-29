"""Direct gain K=R^-1 B0^T M with an LQR-relative SPD parameterization.

Physical integration and error coordinates are inherited unchanged. The batch
executor changes only execution topology and checks against the scalar path.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'stabilization-gym', ROOT/'stabilization-learning-control', ROOT/'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gymnasium as gym
import numpy as np
from scipy.linalg import solve_continuous_are
import torch
from torch import nn
from stabilization_gym import make
from stabilization_gym.interfaces.m_learning import PortableMFacade
from stabilization_learning.policies.metric import CholeskyMetric
from extensions.fast_m_learning_v2.learner import SparseValidationConfig, SparseValidationLearner
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import summarize
from m_learning_suite.task_runtime import FullHorizonTask

TASKS = ('coupled_linear', 'point_mass', 'pendulum_local', 'cartpole_local')
ENV_IDS = dict(zip(TASKS, ('coupled_linear', 'point_mass', 'pendulum', 'cartpole')))
Q_VALUES = {'coupled_linear': [4.,4.,1.,1.], 'point_mass': [4.,4.,1.,1.],
            'pendulum_local': [12.,1.], 'cartpole_local': [2.,.5,16.,1.]}
R_VALUES = {'coupled_linear': .5, 'point_mass': .5, 'pendulum_local': .5, 'cartpole_local': .1}


def linear_model(task, original):
    if task == 'coupled_linear':
        A = original._A.numpy().copy()
    elif task == 'point_mass':
        A = np.block([[np.zeros((2,2)), np.eye(2)], [np.zeros((2,4))]])
    elif task == 'pendulum_local':
        A = np.array([[0.,1.], [original.gravity/original.length,-original.damping]])
    elif task == 'cartpole_local':
        total = original.masscart+original.masspole
        den = original.length*(4./3.-original.masspole/total)
        A = np.zeros((4,4))
        A[0,1] = A[2,3] = 1.
        A[1,2] = -original.masspole*original.length*original.gravity/(total*den)
        A[3,2] = original.gravity/den
    else:
        raise ValueError(task)
    return A, original._B.numpy().copy()


class DirectMetricEnv(gym.Wrapper):
    """Use the original nominal B0; this version is not an SDRE controller."""
    @property
    def state(self):
        return self.env.state

    @state.setter
    def state(self,value):
        self.env.state=value

    def __getattr__(self,name):
        if name=='env':
            raise AttributeError(name)
        return getattr(self.env,name)

    def __init__(self, task, scale=1., max_steps=None):
        options = {} if max_steps is None else {'max_steps': max_steps}
        original = make(ENV_IDS[task], **options)
        original.initial_scale *= scale
        super().__init__(original)
        self.task_name = task
        self.controller_form = 'direct_gain_R_inverse_B0_transpose_M'
        self.A0, self.B0 = linear_model(task, original)
        self.Q0 = np.diag(Q_VALUES[task])
        self.R0 = np.eye(self.B0.shape[1])*R_VALUES[task]
        self.P0 = solve_continuous_are(self.A0, self.B0, self.Q0, self.R0)
        self.L0 = np.linalg.cholesky(self.P0)
        self.gain_left = torch.from_numpy(np.linalg.solve(self.R0, self.B0.T))
        expected = np.linalg.solve(self.R0, self.B0.T@self.P0)
        if not np.allclose(expected, original._K.numpy(), rtol=1e-12, atol=1e-12):
            raise RuntimeError('The recovered P0 does not recover the frozen LQR gain')
        self.controller_capabilities = dict(original.controller_capabilities,
            certificate_guarantee='candidate_only', physical_metric='L0_S_L0_transpose',
            normalization_applies_to='S_not_physical_M')

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        weighted = torch.matmul(M.to(e), e.unsqueeze(-1)).squeeze(-1)
        action = -(weighted @ self.gain_left.to(e).T)
        if self.task_name == 'coupled_linear':
            action = action + context['goal'].to(e)[..., :2] @ self.env._stiffness.to(e).T
        elif self.task_name == 'point_mass':
            action = action + context.get('reference_acceleration', self.env._reference_acceleration).to(e)
        low = torch.as_tensor(self.action_space.low, dtype=e.dtype, device=e.device)
        high = torch.as_tensor(self.action_space.high, dtype=e.dtype, device=e.device)
        return torch.maximum(low, torch.minimum(high, action))


class CongruenceMetric(nn.Module):
    """Network raw -> normalized S -> physical M, all computed in float64."""
    def __init__(self, specification, P0):
        super().__init__()
        self.shape_metric = CholeskyMetric(specification).double()
        self.register_buffer('P0', torch.as_tensor(P0, dtype=torch.float64).clone())
        self.register_buffer('L0', torch.linalg.cholesky(self.P0))

    def shape(self, raw):
        return self.shape_metric(raw.to(dtype=torch.float64))

    def forward(self, raw):
        S = self.shape(raw)
        M = self.L0 @ S @ self.L0.T
        return .5*(M+M.transpose(-1,-2))


def create_task(task, scale=1., max_steps=None):
    inner = DirectMetricEnv(task, scale, max_steps)
    physical = FullHorizonTask(inner, ENV_IDS[task])
    physical.controller_form = inner.controller_form
    return physical


def failures(inner, states):
    bad = ~torch.isfinite(states).all(-1) | (torch.linalg.vector_norm(states, dim=-1)>100.)
    if inner.task_name == 'pendulum_local':
        angle = torch.atan2(torch.sin(states[...,0]), torch.cos(states[...,0]))
        bad |= (angle.abs()>np.pi/2) | (states[...,1].abs()>12.)
    elif inner.task_name == 'cartpole_local':
        angle = torch.atan2(torch.sin(states[...,2]), torch.cos(states[...,2]))
        bad |= (states[...,0].abs()>2.4) | (angle.abs()>np.pi/4)
    return bad


def initial_states(physical, seeds):
    goal = np.asarray(physical.goal, dtype=np.float64)
    scale = physical.inner_env.initial_scale
    return np.stack([goal+np.random.default_rng(int(seed)).uniform(-1.,1.,len(goal))*scale for seed in seeds])


def batch_rollout(physical, matrices, starts, *, certificate_weight=.1, decay=.05, record=False,
                  should_stop=None):
    """Execute all episodes at their original dt and full horizon, vectorized."""
    inner = physical.inner_env
    state = torch.as_tensor(starts, dtype=torch.float64).clone()
    M = torch.as_tensor(matrices, dtype=torch.float64)
    if M.ndim == 2:
        M = M.expand(len(state), -1, -1)
    count = len(state)
    active = np.ones(count, dtype=bool)
    failed = np.zeros(count, dtype=bool)
    steps = np.zeros(count, dtype=int)
    costs = np.zeros(count)
    penalties = np.zeros(count)
    rises = np.zeros(count, dtype=int)
    violations = np.zeros(count, dtype=int)
    max_residual = np.zeros(count)
    variations = np.zeros(count)
    last_action = None
    e0 = inner.tensor_error(state)
    terminal_error = e0.square().sum(-1).numpy().copy()
    last_bad = np.where(terminal_error <= inner.success_tolerance**2, -1, 0)
    buffers = {key: [] for key in ('states','following','actions','errors','errors_after','V_before',
        'V_after','costs','physical_error_before','physical_error_after','in_goal_before','in_goal_after',
        'decay_residual')} if record else None
    for k in range(physical.max_steps):
        if should_stop is not None and should_stop():
            raise InterruptedError('Stop requested; resume from the last committed whole PPO update')
        if not active.any():
            break
        with torch.no_grad():
            error = inner.tensor_error(state)
            before = torch.einsum('bi,bij,bj->b', error, M, error)
            context = {'state': state, 'goal': torch.as_tensor(inner.goal, dtype=torch.float64)}
            action = inner.calculate_action_from_M(M, error, context=context)
            following = inner.torch_step(state, action)
            next_error = inner.tensor_error(following)
            after = torch.einsum('bi,bij,bj->b', next_error, M, next_error)
            bad = failures(inner, following).numpy()
        norm2 = error.square().sum(-1).numpy()
        next_norm2 = next_error.square().sum(-1).numpy()
        delta = (after-before).numpy()
        residual = delta+decay*physical.dt*norm2
        scale = np.maximum(norm2,1e-6)
        penalty = np.maximum(0.,delta+decay*physical.dt*scale)/scale
        in_goal_before = norm2 <= inner.success_tolerance**2
        in_goal_after = next_norm2 <= inner.success_tolerance**2
        increment = physical.dt*norm2
        applied = action.numpy()
        costs[active] += increment[active]
        penalties[active] += penalty[active]
        rises[active] += delta[active]>1e-8
        violations[active] += residual[active]>1e-8
        max_residual[active] = np.maximum(max_residual[active],residual[active])
        if last_action is not None:
            variations[active] += np.linalg.norm(applied-last_action,axis=-1)[active]
        steps[active] += 1
        terminal_error[active] = next_norm2[active]
        last_bad[active & ~in_goal_after] = k+1
        if record:
            values = dict(states=state.numpy().copy(), following=following.numpy().copy(),
                actions=applied.copy(), errors=error.numpy().copy(), errors_after=next_error.numpy().copy(),
                V_before=before.numpy().copy(), V_after=after.numpy().copy(), costs=increment.copy(),
                physical_error_before=norm2.copy(), physical_error_after=next_norm2.copy(),
                in_goal_before=in_goal_before.copy(), in_goal_after=in_goal_after.copy(),
                decay_residual=residual.copy())
            for key,value in values.items():
                buffers[key].append(value)
        just_failed = active & bad
        failed |= just_failed
        active &= ~bad
        state = torch.where(torch.as_tensor(active).unsqueeze(-1),following,state)
        last_action = applied
    horizon = physical.dt*physical.max_steps
    rows = []
    for i in range(count):
        n = int(steps[i])
        start = int(last_bad[i]+1) if last_bad[i]>=0 else 0
        hold = max(0.,(n-start)*physical.dt)
        success = bool(not failed[i] and n==physical.max_steps and
                       hold+1e-10>=physical.required_hold_seconds)
        fixed = costs[i]+(physical.max_steps-n)*physical.dt*(max(terminal_error[i],25.) if failed[i] else terminal_error[i])
        matrix = M[i].numpy()
        eigen = np.linalg.eigvalsh(matrix)
        row = dict(fixed_horizon_cost=float(fixed), accumulated_cost=float(costs[i]),
            objective=float(fixed+certificate_weight*penalties[i]), certificate_penalty=float(penalties[i]),
            V_rise_fraction=float(rises[i]/max(n,1)), success=success, failure=bool(failed[i]),
            physical_steps=n, arrival_time=float(start*physical.dt+physical.required_hold_seconds if success else horizon),
            settling_time=float(start*physical.dt if success else horizon), hold_seconds=float(hold),
            required_hold_seconds=physical.required_hold_seconds, terminal_physical_error=float(terminal_error[i]),
            actual_decay_violation_fraction=float(violations[i]/max(n,1)), actual_decay_violation_max=float(max_residual[i]),
            action_total_variation=float(variations[i]), horizon_seconds=horizon,
            M_eigen_min=float(eigen.min()), M_eigen_max=float(eigen.max()), M_trace=float(np.trace(matrix)),
            physical_terminated=bool(failed[i] or success), physical_truncated=bool(not success and not failed[i]))
        rows.append(row)
    arrays = None
    if record:
        arrays = {key: np.stack(value) for key,value in buffers.items()}
    return rows, arrays


class DirectLearner(SparseValidationLearner):
    def __init__(self, task, *, config=None, scale=1., max_steps=None):
        physical = create_task(task,scale,max_steps)
        self.task_name = task
        self.physical = physical
        self.should_stop = None
        super().__init__(PortableMFacade(physical), config=config or SparseValidationConfig(),
                         env_id=ENV_IDS[task], env_kwargs={'direct_metric_version':'v1','scale':scale})
        self.actor.metric = CongruenceMetric(physical.metric_spec, physical.inner_env.P0)
        self.best_actor = deepcopy(self.actor.state_dict())

    @property
    def baseline(self):
        return self.actor.metric(self.actor.center).detach()

    def _protocol(self):
        protocol = super()._protocol()
        inner = self.physical.inner_env
        protocol.update(controller_version='direct_lqr_metric_v1',task=self.task_name,
            input_matrix='frozen_nominal_B0', metric_scope='episode_fixed_goal_conditioned',
            physical_metric='M=L0*S*L0.T', parameter_metric='trace_normalized_S',
            P0=inner.P0.tolist(), L0=inner.L0.tolist(), A0=inner.A0.tolist(),
            B0=inner.B0.tolist(), Q0=inner.Q0.tolist(), R0=inner.R0.tolist(),
            executor='float64_CPU_batch_original_physics_v1')
        return protocol

    def evaluate(self,seeds=None,*,identity=False,best=False,trajectories=False):
        if trajectories:
            raise ValueError('Use the versioned recorded evaluation stage')
        seeds = list(self.config.validation_seeds if seeds is None else seeds)
        actor = self.actor
        if best:
            actor = deepcopy(self.actor)
            actor.load_state_dict(self.best_actor)
        starts = initial_states(self.physical,seeds)
        contexts = torch.stack([self._context(start) for start in starts])
        with torch.no_grad():
            matrices = (self.baseline.expand(len(seeds),-1,-1) if identity else
                        torch.stack([actor(context) for context in contexts]))
        rows,_ = batch_rollout(self.physical,matrices,starts,
            certificate_weight=self.config.certificate_weight,decay=self.config.decay,
            should_stop=self.should_stop)
        for seed,row in zip(seeds,rows):
            row['seed'] = int(seed)
        return {'summary':summarize(rows),'episodes':rows}

    def _collect(self,count):
        seeds = [self.config.seed+self.episodes+i for i in range(count)]
        if set(seeds)&set(self.config.validation_seeds):
            raise ValueError('Training and validation seeds overlap')
        starts = initial_states(self.physical,seeds)
        contexts = torch.stack([self._context(start) for start in starts])
        raws,logps,matrices = [],[],[]
        # Preserve scalar sampling order; batching only changes physics execution.
        with torch.no_grad():
            for context in contexts:
                dist = self.actor.distribution(context)
                raw = dist.sample()
                raws.append(raw)
                logps.append(dist.log_prob(raw))
                matrices.append(self.actor.metric(raw))
        rows,_ = batch_rollout(self.physical,torch.stack(matrices),starts,
            certificate_weight=self.config.certificate_weight,decay=self.config.decay,
            should_stop=self.should_stop)
        returns = torch.tensor([-row['objective']/(self.physical.dt*self.physical.max_steps) for row in rows])
        self.episodes += count
        self.physical_steps += sum(row['physical_steps'] for row in rows)
        return contexts,torch.stack(raws),torch.stack(logps),returns,rows
