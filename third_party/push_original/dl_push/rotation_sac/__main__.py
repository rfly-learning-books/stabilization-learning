"""Train a from-scratch SAC rotation controller on GPU PhysX."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import torch

from .environment import RotationSACEnv
from .sac import Replay, SAC, SACConfig


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def train_shape(shape, args, root):
    out = root / shape
    out.mkdir(parents=True, exist_ok=False)
    curriculum_start = min(args.angle_curriculum_start, args.angle_max)
    env = RotationSACEnv(shape=shape, num_envs=args.num_envs, horizon=args.horizon,
                         contact_reset=args.contact_reset, angle_min=args.angle_min,
                         angle_max=curriculum_start if args.angle_curriculum else args.angle_max,
                         seed=args.seed, task=args.task)
    try:
        sac_config = SACConfig(batch_size=args.batch_size,
                               updates_per_vector_step=args.updates_per_vector_step,
                               learning_starts=args.learning_starts,
                               gamma=args.gamma, tau=args.tau,
                               gradient_clip=args.gradient_clip, min_alpha=args.min_alpha)
        agent = SAC(env.obs_dim, env.device, sac_config, seed=args.seed)
        replay = Replay(args.replay_size, env.obs_dim)
        obs, _ = env.reset(seed=args.seed)
        transitions = 0
        vector_steps = 0
        episodes = 0
        successes = 0
        strict_successes = 0
        reward_window = deque(maxlen=128)
        success_window = deque(maxlen=128)
        history = []
        started = time.perf_counter()
        write_json(out / 'manifest.json', dict(route='rotation_sac_physx_v1',
            from_scratch=True, planner_guided=False, demonstrations=None,
            shape=shape, total_transitions=args.steps_per_shape,
            num_envs=args.num_envs, backend='SAPIEN GPU PhysX',
            object_reset='fixed_origin: xy=(0,0), yaw=0',
            task=args.task if args.task is not None else 'CW+CCW canonicalized',
            contact_reset=args.contact_reset, action='normalized XY pusher velocity',
            reward_contract={
                'success_bonus': 100., 'position_penalty': 'sum ||p_t-p_0|| / .02',
                'd_constraint_penalty': 4., 'angle_progress': 3.,
                'time_limit_bootstrap': True},
            sac=sac_config.__dict__, environment=env.provenance(),
            angle_curriculum={'enabled':args.angle_curriculum,
                              'start':curriculum_start,
                              'warmup_transitions':args.angle_curriculum_warmup,
                              'ramp_transitions':args.angle_curriculum_ramp,
                              'target':args.angle_max}, arguments=vars(args)))
        next_save = args.checkpoint_interval
        next_log = args.log_interval
        while transitions < args.steps_per_shape:
            with torch.no_grad():
                action = (2. * torch.rand(args.num_envs, 2, device=env.device) - 1.) \
                    if transitions < args.learning_starts else agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            replay.add(obs, action, reward, next_obs, terminated, truncated, info['success'])
            obs = next_obs
            transitions += args.num_envs
            vector_steps += 1
            reward_window.extend(reward.detach().cpu().tolist())
            done = terminated | truncated
            episodes += int(done.sum())
            successes += int(info['success'].sum())
            strict_successes += int(info['strict_success'].sum())
            success_window.extend(info['success'].detach().cpu().tolist())
            update_metrics = {}
            if replay.size >= args.batch_size and transitions >= args.learning_starts:
                for _ in range(args.updates_per_vector_step):
                    update_metrics = agent.update(replay)
            if transitions >= next_log or transitions >= args.steps_per_shape:
                if args.angle_curriculum:
                    ramp = max(1, args.angle_curriculum_ramp)
                    fraction = min(1., max(0., (transitions - args.angle_curriculum_warmup) / ramp))
                    env.set_angle_max(curriculum_start + fraction * (args.angle_max - curriculum_start))
                elapsed = time.perf_counter() - started
                record = dict(shape=shape, transitions=transitions, vector_steps=vector_steps,
                              episodes=episodes, successes=successes,
                              strict_successes=strict_successes,
                              success_rate_window=float(sum(success_window) / max(1, len(success_window))),
                              mean_reward_window=float(sum(reward_window) / max(1, len(reward_window))),
                              angle_max=float(env.base.config.angle_max),
                              replay_size=replay.size, elapsed_s=elapsed,
                              transitions_per_s=transitions / max(elapsed, 1e-6),
                              torch_peak_cuda_mib=torch.cuda.max_memory_allocated(env.device) / 2**20,
                              **update_metrics)
                history.append(record)
                with (out / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
                next_log += args.log_interval
            if transitions >= next_save or transitions >= args.steps_per_shape:
                payload = agent.checkpoint(transitions, transitions, shape, vars(args), history)
                payload['environment'] = env.provenance()
                temp = out / f'checkpoint_{transitions:09d}.pt.tmp'
                torch.save(payload, temp)
                temp.replace(out / f'checkpoint_{transitions:09d}.pt')
                next_save += args.checkpoint_interval
        final = agent.checkpoint(transitions, transitions, shape, vars(args), history)
        final['environment'] = env.provenance()
        torch.save(final, out / 'last.pt')
        write_json(out / 'summary.json', dict(shape=shape, route='rotation_sac_physx_v1',
            from_scratch=True, total_transitions=transitions, episodes=episodes,
            successes=successes, strict_successes=strict_successes,
            elapsed_s=time.perf_counter() - started,
            final_success_rate=(successes / max(1, episodes)),
            final_strict_success_rate=(strict_successes / max(1, episodes)),
            checkpoint=str((out / 'last.pt').resolve()), history=history))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shapes', nargs='+', default=['T', 'Square', 'RectWide', 'Plus', 'H'])
    parser.add_argument('--task', type=int, choices=(4, 5), default=None,
                        help='Train one signed task; default trains both via CCW mirror canonicalization')
    parser.add_argument('--steps-per-shape', type=int, default=2_000_000)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=240)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--replay-size', type=int, default=500_000)
    parser.add_argument('--learning-starts', type=int, default=100_000)
    parser.add_argument('--updates-per-vector-step', type=int, default=2)
    parser.add_argument('--checkpoint-interval', type=int, default=250_000)
    parser.add_argument('--log-interval', type=int, default=10_000)
    parser.add_argument('--angle-min', type=float, default=.15)
    parser.add_argument('--angle-max', type=float, default=1.0)
    parser.add_argument('--angle-curriculum', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--angle-curriculum-start', type=float, default=.25)
    parser.add_argument('--angle-curriculum-warmup', type=int, default=250_000)
    parser.add_argument('--angle-curriculum-ramp', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--contact-reset', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--tau', type=float, default=.005)
    parser.add_argument('--gradient-clip', type=float, default=5.)
    parser.add_argument('--min-alpha', type=float, default=.05)
    args = parser.parse_args()
    if any(x < 1 for x in (args.steps_per_shape, args.num_envs, args.horizon,
                            args.batch_size, args.replay_size, args.learning_starts,
                            args.updates_per_vector_step, args.checkpoint_interval,
                            args.log_interval)):
        parser.error('Training sizes must be positive')
    if args.num_envs != 32:
        print('WARNING: requested num-envs differs from the 32-lane training protocol', flush=True)
    if not 0 < args.angle_min <= args.angle_max <= 1.57:
        parser.error('Rotation angle range must lie in (0, 1.57] rad')
    if not args.angle_min <= args.angle_curriculum_start <= args.angle_max:
        parser.error('angle-curriculum-start must lie in the target angle range')
    if args.angle_curriculum_warmup < 0 or args.angle_curriculum_ramp < 1 or args.min_alpha <= 0:
        parser.error('Invalid curriculum or entropy parameters')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'run_manifest.json', dict(route='rotation_sac_physx_v1',
        from_scratch=True, shapes=args.shapes, steps_per_shape=args.steps_per_shape,
        num_envs=args.num_envs, no_checkpoint_sharing=True, arguments=vars(args)))
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    for shape in args.shapes:
        train_shape(shape, args, args.output)


if __name__ == '__main__':
    main()
