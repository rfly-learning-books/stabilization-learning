"""MPPI with all lanes restored to one root; no learned rollout dynamics."""
from copy import deepcopy
import torch

FIELDS = ('task', 'amount', 'initial_xy', 'initial_yaw', 'lifted_yaw', 'last_yaw',
          'steps', 'hold', 'done', 'max_side', 'max_angle', 'backward_steps',
          'translation_error_sum')
RAW_FIELDS = ('_active_shape_id', '_active_shape_global_id', '_success_hold_count',
              '_last_control_pusher_tee_force', '_elapsed_steps')


def tree_copy(value, lane=None, n=None):
    if isinstance(value, dict):
        return {k: tree_copy(v, lane, n) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        if lane is not None and value.ndim and value.shape[0] == n:
            return value[lane:lane+1].expand_as(value).clone()
        return value.clone()
    return deepcopy(value)


def snapshot(env, lane=None):
    n = env.num_envs
    return dict(physics=tree_copy(env.raw.get_state_dict(), lane, n),
                task={k: tree_copy(getattr(env, k), lane, n) for k in FIELDS},
                raw={k: tree_copy(getattr(env.raw, k), lane, n) for k in RAW_FIELDS})


def restore(env, state):
    for k, v in state['raw'].items():
        setattr(env.raw, k, v.clone())
    env.raw.set_state_dict(tree_copy(state['physics']))
    if 'controller' in state['physics']:
        env.raw.agent.set_controller_state(tree_copy(state['physics']['controller']))
    for k, v in state['task'].items():
        setattr(env, k, v.clone())


def replay_check(original_d, original_q, replay_d, replay_q):
    d_error = abs(float(replay_d)-float(original_d))
    q_error = abs(float(replay_q)-float(original_q))
    return dict(replay_d_error=d_error, replay_q_error=q_error,
                replay_consistent=(d_error <= 1e-4+.01*abs(float(original_d)) and
                                   q_error <= 1e-3+.01*abs(float(original_q))))


@torch.no_grad()
def rollout(env, root, actions, q_limit=1., energy_weight=10., q_weight=1.):
    """Stop labels at first terminal. Padded simulator steps are never training data."""
    restore(env, root)
    initial = env.metrics()['energy'].clone()
    active = ~env.done
    q = torch.zeros_like(initial)
    energy = initial.clone()
    steps = torch.zeros_like(env.steps)
    failed = torch.zeros_like(active)
    max_q = q.clone()
    prep = env.metrics()['prep'].clone()
    for action in actions.transpose(0, 1):
        if not active.any():
            break
        env.done.zero_()  # inactive lanes continue physically, but all their labels are masked
        _, _, term, trunc, info = env.step(torch.where(active[:, None], action, 0.))
        q += torch.where(active, info['q_cost'], 0.)
        max_q = torch.maximum(max_q, torch.where(active, info['q_cost'], 0.))
        energy = torch.where(active, info['energy'], energy)
        prep = torch.where(active, info['prep'], prep)
        steps += active.long()
        failed |= active & info['fail']
        active &= ~(term | trunc)
    d = energy-initial
    rate = 1-(1-env.config.contraction*env.dt)**steps
    violation = torch.relu(d+rate*initial)
    # Finite-horizon heuristic feasibility is distinct from episode success / a certificate.
    feasible = (~failed) & (violation <= 1e-6) & (max_q <= q_limit)
    score = q_weight*q/steps.clamp_min(1)+energy_weight*energy+10*violation+prep/.1+1000*failed.float()
    if not torch.isfinite(score).all():
        raise FloatingPointError('Nonfinite PhysX MPPI score')
    return dict(d=d, q=q, energy=energy, steps=steps, feasible=feasible,
                violation=violation, score=score, failed=failed)


@torch.no_grad()
def plan(env, horizon=16, iterations=3, sigma=.5, temperature=.5, random_fraction=.25,
         initial_mean=None, energy_weight=10., q_weight=1., q_limit=1.):
    root = snapshot(env, lane=0)
    restore(env, root)
    obs = tree_copy(env.observe())
    mean = torch.zeros(horizon, 2, device=env.device) if initial_mean is None else initial_mean.clone()
    if mean.shape != (horizon, 2) or not torch.isfinite(mean).all() or (mean.abs()>1).any():
        raise ValueError('Invalid MPPI initial sequence')
    batches = []
    for _ in range(iterations):
        # Piecewise-constant noise gives temporally coherent contact exploration.
        noise = torch.randn(env.num_envs, (horizon+3)//4, 2, device=env.device)
        actions = (mean+sigma*noise.repeat_interleave(4, dim=1)[:, :horizon]).clamp(-1, 1)
        count = max(1, int(env.num_envs*random_fraction))
        actions[-count:] = (2*torch.rand(count, (horizon+3)//4, 2, device=env.device)-1).repeat_interleave(4, dim=1)[:, :horizon]
        actions[0] = mean
        if initial_mean is not None and env.num_envs > 2:
            actions[1] = 0.  # Always evaluate braking as well as the shifted trajectory.
        result = rollout(env, root, actions, energy_weight=energy_weight, q_weight=q_weight, q_limit=q_limit)
        weights = torch.softmax(-(result['score']-result['score'].min())/temperature, dim=0)
        mean = torch.sum(weights[:, None, None]*actions, dim=0)
        batches.append((actions.clone(), result))
    # Weighted average can be infeasible: re-simulate it, then compare with sampled candidates.
    actions = batches[-1][0].clone()
    actions[0] = mean
    result = rollout(env, root, actions, energy_weight=energy_weight, q_weight=q_weight, q_limit=q_limit)
    batches.append((actions.clone(), result))
    candidates = torch.cat([a for a, r in batches])
    scores = torch.cat([r['score'] for a, r in batches])
    feasible = torch.cat([r['feasible'] for a, r in batches])
    index = scores.masked_fill(~feasible, float('inf')).argmin() if feasible.any() else scores.argmin()
    selected = candidates[index]
    # Independent restored rollout checks selected trajectory. Record empty feasible sets explicitly.
    verified = rollout(env, root, selected[None].expand(env.num_envs, -1, -1), energy_weight=energy_weight,
                       q_weight=q_weight, q_limit=q_limit)
    original_d = torch.cat([r['d'] for a, r in batches])[index]
    original_q = torch.cat([r['q'] for a, r in batches])[index]
    check = replay_check(original_d, original_q, verified['d'][0], verified['q'][0])
    restore(env, root)
    return obs, selected, batches, dict(empty_feasible_set=not bool(feasible.any()),
        verified_feasible=bool(verified['feasible'][0]),
        accepted=bool(feasible[index]) and bool(verified['feasible'][0]) and check['replay_consistent'],
        **check)
