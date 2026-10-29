"""Collect actual first-step labels; train direct one-step DQ and sequence outcome model."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from dl_push.six_primitives.environment import SixConfig, SixPrimitiveEnv
from dl_push.six_primitives.learning import DirectDQ, Replay, Actor
from dl_push.mppi_primitives.learning import SequenceDQ
from .planner import plan, snapshot, restore, tree_copy

ROUTE = 'planner_guided_physx_mppi_dq_v1'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def collect(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = SixConfig(num_envs=args.samples, fixed_shape=args.shape, fixed_task=args.task,
                       task_ids=(args.task,), reset_mode=args.reset_mode)
    env = SixPrimitiveEnv(config)
    try:
        actor = None
        if args.rollin_checkpoint:
            saved = torch.load(args.rollin_checkpoint, map_location=env.device, weights_only=False)
            if saved.get('route') != ROUTE or saved.get('from_scratch') is not False:
                raise ValueError('Policy roll-in accepts only route-B checkpoints')
            actor = Actor().to(env.device)
            actor.load_state_dict(saved['actor']); actor.eval()
        write_json(out/'manifest.json', dict(route=ROUTE, from_scratch=False, certified=False,
            arguments=vars(args), source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob('*.py')}, **env.provenance()))
        env.reset(seed=args.seed)
        for step in range(args.steps):
            obs, selected, batches, report = plan(env, args.horizon, args.iterations)
            root = snapshot(env)
            # Genuine one-step transitions for the direct actor critic: never assign D_H to a_0.
            actions = batches[-1][0][:, 0].clone()
            actions[0] = selected[0]
            nxt, _, term, trunc, info = env.step(actions)
            replay = Replay(args.samples)
            replay.add(obs, actions, nxt, info, term)
            shard = dict(route=ROUTE, replay=replay.storage, size=replay.size,
                root={k: v[:1].cpu().to(torch.uint8 if k == 'shape' else v.dtype) for k, v in obs.items()},
                sequences=torch.cat([a for a, r in batches]).cpu(),
                # D_H and Q_H explicitly condition on every applied action and terminal mask.
                labels=torch.cat([torch.stack((r['d'], r['q'], r['energy'], r['violation'], r['steps'].float()), -1) for a, r in batches]).cpu(),
                feasible=torch.cat([r['feasible'] for a, r in batches]).cpu(), report=report)
            torch.save(shard, out/f'shard_{step:06d}.pt')
            # Continue from selected first action at the real visited root, then replan.
            restore(env, root)
            with torch.no_grad():
                executed = selected[0] if actor is None else actor(obs)[0]
            _, _, term, trunc, info = env.step(executed.expand(args.samples, -1))
            report.update(step=step, success=bool(info['success'][0]), strict_success=bool(info['strict_success'][0]))
            with (out/'history.jsonl').open('a') as f:
                f.write(json.dumps(report)+'\n')
            print(json.dumps(report), flush=True)
            if bool((term | trunc)[0]):
                env.reset(seed=args.seed+step+1)
    finally:
        env.close()


def train(args):
    source = Path(args.data)
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest.get('route') != ROUTE or manifest['config']['split'] != 'train':
        raise ValueError('Only route-B training split data accepted')
    files = sorted(source.glob('shard_*.pt'))
    if len(files) < 2:
        raise ValueError('Need at least two root shards for a held-out model diagnostic')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = args.device
    learner = DirectDQ(device, behavior_weight=0., horizon_steps=1)
    model = SequenceDQ(manifest['arguments']['horizon']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Bounded memory: cache one root shard at a time. Hold out whole roots, not candidate rows.
    training, heldout = files[:-1], files[-1:]
    cached = None
    for step in range(args.updates):
        if step % args.updates_per_shard == 0:
            cached = torch.load(training[torch.randint(len(training), ()).item()], map_location='cpu', weights_only=False)
            if cached['route'] != ROUTE:
                raise ValueError('Mixed route shard')
            replay = Replay(cached['size'])
            replay.size = cached['size']; replay.storage = cached['replay']
        metrics = learner.update(replay, args.batch_size)
        ids = torch.randint(len(cached['sequences']), (args.batch_size,))
        obs = {k: v.to(device).float().expand(args.batch_size, *v.shape[1:]) for k, v in cached['root'].items()}
        target = cached['labels'][ids].to(device)
        prediction = model(obs, cached['sequences'][ids].to(device))
        # Fixed unit scaling; no held-out statistics enter training.
        scale = torch.tensor([.01, 1., .1, .01, float(model.knots)], device=device)
        loss = torch.nn.functional.smooth_l1_loss(prediction/scale, target/scale)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite sequence loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step % 100 == 0 or step == args.updates-1:
            metrics.update(step=step, sequence_loss=loss.item())
            with (out/'history.jsonl').open('a') as f:
                f.write(json.dumps(metrics)+'\n')
            print(json.dumps(metrics), flush=True)
    with torch.no_grad():
        data = torch.load(heldout[0], map_location='cpu', weights_only=False)
        total = torch.zeros(5, device=device)
        for start in range(0, len(data['sequences']), args.batch_size):
            seq = data['sequences'][start:start+args.batch_size].to(device)
            obs = {k: v.to(device).float().expand(len(seq), *v.shape[1:]) for k, v in data['root'].items()}
            total += (model(obs, seq)-data['labels'][start:start+len(seq)].to(device)).abs().sum(0)
        mae = (total/len(data['sequences'])).cpu().tolist()
    saved = learner.checkpoint()
    saved.update(route=ROUTE, from_scratch=False, method=ROUTE, config=manifest['config'],
                 args=dict(from_scratch=False), sequence_dq=model.state_dict(), sequence_optimizer=optimizer.state_dict(),
                 horizon=model.knots, data=str(source.resolve()), certified=False)
    torch.save(saved, out/'last.pt')
    write_json(out/'manifest.json', dict(route=ROUTE, from_scratch=False, certified=False,
        arguments=vars(args), source=str(source.resolve()), training_shards=[p.name for p in training],
        heldout_shards=[p.name for p in heldout], heldout_sequence_mae=mae,
        label_order=['D_H', 'Q_H', 'V_H', 'violation_H', 'valid_steps']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--output', required=True)
    c.add_argument('--samples', type=int, default=64)
    c.add_argument('--horizon', type=int, default=16)
    c.add_argument('--iterations', type=int, default=3)
    c.add_argument('--steps', type=int, default=100)
    c.add_argument('--shape', default='T')
    c.add_argument('--task', type=int, default=0)
    c.add_argument('--rollin-checkpoint', help='Route-B actor visits roots; MPPI labels those roots')
    c.add_argument('--reset-mode', default='fixed_origin', choices=['fixed_origin', 'full', 'primitive_contact'])
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--updates', type=int, default=2000)
    t.add_argument('--updates-per-shard', type=int, default=10)
    t.add_argument('--batch-size', type=int, default=128)
    t.add_argument('--device', default='cuda')
    for p in (c, t):
        p.add_argument('--seed', type=int, default=20260910)
    args = parser.parse_args()
    if args.command == 'collect' and args.samples < 2:
        parser.error('samples must be at least 2 (mean plus exploration)')
    for name in ('samples', 'horizon', 'iterations', 'steps', 'updates', 'updates_per_shard', 'batch_size'):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    torch.manual_seed(args.seed); torch.set_num_threads(2)
    (collect if args.command == 'collect' else train)(args)


if __name__ == '__main__':
    main()
