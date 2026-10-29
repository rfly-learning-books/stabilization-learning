"""Random full-M PPO with disjoint shape splits and complete successor values."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import copy
import numpy as np
import torch

from .common import ROOT, Run, paths, read, write, sha, seed_all, save_torch, tensor_hash
paths()
from dl_push.certificate import full_m_ppo as ppo
from dl_push.certificate.full_m_control import (
    FullMActor, FullMConfig, FullMControlEnv, ModeBank, candidates, hard_choice, coordinates)


class IdentityActor(FullMActor):
    def conditioned(self, D, condition):
        eye = torch.eye(3, device=D.device, dtype=D.dtype).expand(*D.shape[:-1], 3, 3)
        return eye, eye


@torch.no_grad()
def probe_metric(actor,C,valid,ell):
    """Same three normalized states per training shape, every gradient update."""
    cond=actor.encoder(C,valid,ell)
    d=C.new_tensor([[.4,-.3,.2],[-.2,.1,-.3],[.05,.05,.05]])
    d=d[None].expand(len(C),-1,-1)
    return actor.conditioned(d,cond)[1].cpu().numpy()


def probe_stats(m,initial,previous):
    flat=m.reshape(-1,3,3);mean=flat.mean(0);eig=np.linalg.eigvalsh(flat)
    return dict(M_probe_mean=mean.tolist(),M_probe_distance=float(np.linalg.norm(m-initial)/np.linalg.norm(initial)),
        M_probe_step_change=float(np.linalg.norm(m-previous)/np.linalg.norm(previous)),
        M_probe_eigen_min=float(eig.min()),M_probe_eigen_max=float(eig.max()),
        M_probe_state_variation=float(np.max(np.linalg.norm(m[:,1:]-m[:,:1],axis=(-2,-1)))))


def evaluate(actor, bank, cfg, names, seeds, out=None, guard=lambda: None):
    rows, trace_hashes = [], []
    for repeat, seed in enumerate(seeds):
        guard()
        ctrl = FullMControlEnv(bank, cfg, names, len(names))
        data = ctrl.reset(seed=seed)
        initial = ctrl.env.q.copy(); target = ctrl.env.target.copy()
        trace = []; actor.eval()
        for step in range(cfg.max_steps):
            guard()
            if ctrl.env.done.all():
                break
            with torch.no_grad():
                cand = candidates(actor, data, bank, cfg, speed_points=cfg.eval_speed_points)
                m, u, recovery, fail = hard_choice(cand, data, cfg,
                    torch.as_tensor(ctrl.recovery, device=bank.device),
                    torch.as_tensor(ctrl.at_goal(), device=bank.device))
            active = ~ctrl.env.done.copy()
            speed = np.minimum(u.cpu().numpy().astype(float), cfg.u_max)
            no_action = ~cand['feasible'].any(-1).cpu().numpy()
            before = {k: cand[k].detach().cpu().numpy() for k in ('M', 'V', 'gradient')}
            before.update(e=data['e'].cpu().numpy(), pose=ctrl.env.q.copy(), active=active,
                          action=np.c_[m.cpu().numpy(), speed], recovery=recovery.cpu().numpy())
            data, reward, done, used = ctrl.advance(m.cpu().numpy(), speed, no_action,
                                                   recovery.cpu().numpy(), fail.cpu().numpy())
            with torch.no_grad():
                cond = actor.encoder(bank.C, bank.valid, cfg.length_scale)[data['shape']]
                _, mn = actor.conditioned(coordinates(data['e']), cond)
                vn = actor.value(coordinates(data['e']), cond)
            before.update(e_next=data['e'].cpu().numpy(), M_next=mn.cpu().numpy(),
                          V_next=vn.cpu().numpy(), pose_next=ctrl.env.q.copy())
            trace.append(before)
        stacked = {k: np.stack([r[k] for r in trace]) for k in trace[0]}
        for i, name in enumerate(names):
            used = stacked['active'][:, i]
            dv = stacked['V_next'][:, i] - stacked['V'][:, i]
            e = ctrl.obs['error'][i]
            rows.append(dict(shape=name, repeat=repeat, sample_seed=seed, initial=initial[i].tolist(),
                target=target[i].tolist(), success=bool(ctrl.env.reason[i] == 'SUCCESS'),
                reason=str(ctrl.env.reason[i]), steps=int(ctrl.env.steps[i]),
                cumulative_error=float(ctrl.cost_error[i]), switches=int(ctrl.env.switches[i]),
                no_action_steps=int(ctrl.no_action_steps[i]),
                terminal_error=float(e[:2] @ e[:2] + 1 - np.cos(e[2])),
                control_energy=float(cfg.dt * np.sum(stacked['action'][used, i, 1] ** 2)),
                arrival_seconds=float(ctrl.env.steps[i] * cfg.dt) if ctrl.env.reason[i] == 'SUCCESS' else None,
                V_increase_fraction=float(np.mean(dv[used] > 1e-7)),
                M_state_variation=float(np.max(np.linalg.norm(stacked['M'][used, i] - stacked['M'][0, i], axis=(-2, -1))))))
        if out is not None:
            out = Path(out); out.mkdir(parents=True, exist_ok=True)
            f = out / f'trajectory_{repeat:04d}.npz'
            np.savez_compressed(f, **stacked)
            trace_hashes.append(dict(file=f.name, sha256=sha(f)))
        ctrl.env.close()
    result = dict(episodes=rows, success_rate=float(np.mean([x['success'] for x in rows])),
        mean_cumulative_error=float(np.mean([x['cumulative_error'] for x in rows])),
        failure_error=float(np.mean([x['terminal_error'] * (not x['success']) for x in rows])),
        mean_switches=float(np.mean([x['switches'] for x in rows])),
        mean_no_action_steps=float(np.mean([x['no_action_steps'] for x in rows])), traces=trace_hashes)
    if out is not None:
        write(out / 'RESULTS.json', result)
    return result


def train(run: Run, seed):
    c = run.config['push']; device = run.config['device']; seed_all(seed)
    cfg = FullMConfig(**c.get('control', {}))
    bank = ModeBank(ROOT / 'assets/push_geometric32', device)
    out = run.new_stage(f'push_seed{seed}', dict(seed=seed, initialization='random; no torch.load',
        pretrained_weights=False, replay_imported=False, config=c, control=asdict(cfg),
        fixed_environment_capability_fingerprints=bank.fingerprints))
    actor, critic = FullMActor(cfg).to(device), ppo.Critic(cfg).to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=c['learning_rate'])
    initial = tensor_hash(actor); best_key = None; best_update = None; history = []
    ids=[bank.names.index(n) for n in c['train_shapes']]
    probe0=probe_metric(actor,bank.C[ids],bank.valid[ids],cfg.length_scale);previous=probe0.copy()
    metric_history=[dict(update=0,**probe_stats(probe0,probe0,probe0))]
    val_seeds = [run.config['sampling_seed'] + 1000 + k for k in range(c['validation_repeats'])]
    def save(name, update):
        save_torch(out / name, dict(schema='scratch_push_v1', cfg=asdict(cfg), actor=actor.state_dict(),
            critic=critic.state_dict(), optimizer=optimizer.state_dict(), rng=ppo.rng_state(),
            next_update=update, origin='random', seed=seed, initial_actor_hash=initial,
            protocol_sha256=sha(run.root / 'protocol.json'), best_key=best_key, best_update=best_update))
    save('initial.pt', 0)
    # Validation includes update zero; training is allowed to lose to initialization.
    for update in range(c['updates'] + 1):
        run.guard()
        if update:
            # Rotate lanes so all training shapes receive equal exposure over updates.
            names = [c['train_shapes'][(i + (update-1)*c['num_envs']) % len(c['train_shapes'])]
                     for i in range(c['num_envs'])]
            ctrl = FullMControlEnv(bank, cfg, names, len(names))
            rollout, stats = ppo.collect(actor, critic, ctrl, seed * 1000000 + update)
            stats.update(ppo.update_ppo(actor, critic, optimizer, rollout, bank, cfg,
                                        epochs=c['epochs'], minibatch=c['minibatch']))
            stats.update(update=update, stage='push', seed=seed,
                         actor_change=tensor_hash(actor) != initial)
            probe=probe_metric(actor,bank.C[ids],bank.valid[ids],cfg.length_scale)
            metrics=probe_stats(probe,probe0,previous);previous=probe
            metric_history.append(dict(update=update,**metrics));stats.update(metrics)
            ctrl.env.close(); history.append(stats)
            run.event('push_update', **stats)
            save('last.pt', update)
        if update == 0 or update % c['eval_every'] == 0 or update == c['updates']:
            rng = ppo.rng_state()
            report = evaluate(actor, bank, cfg, c['validation_shapes'], val_seeds,
                              out / 'validation' / f'update_{update:05d}', run.guard)
            ppo.restore_rng(rng)
            key = ppo.selection_key(report)
            if update>0 and (best_key is None or key > best_key):
                best_key, best_update = key, update
                save('selected.pt', update)
            run.event('push_validation', seed=seed, update=update, selected_update=best_update,
                success_rate=report['success_rate'], mean_cumulative_error=report['mean_cumulative_error'])
    save('final.pt', c['updates'])
    write(out / 'TRAINING.json', history)
    write(out / 'M_PROBES.json', metric_history)
    write(out / 'FROZEN.json', dict(checkpoint='selected.pt', sha256=sha(out / 'selected.pt'),
        selected_update=best_update, updates=c['updates'], initial_actor_hash=initial,
        final_actor_hash=tensor_hash(actor), selection='validation among post-update models; random update zero is an explicit baseline'))
    run.event('push_frozen', seed=seed, selected_update=best_update, updates=c['updates'])


def test(run: Run, seed):
    run.frozen_group('push'); c = run.config['push']; paths()
    out = run.new_stage(f'push_seed{seed}/test', dict(heldout_shapes=c['test_shapes'], selection_allowed=False))
    bank = ModeBank(ROOT / 'assets/push_geometric32', run.config['device'])
    cases = [run.config['sampling_seed'] + 2000 + k for k in range(c['test_repeats'])]
    reports = {}
    for label, file in [('random_initial', 'initial.pt'), ('learned_selected', 'selected.pt'), ('final', 'final.pt'), ('identity', 'initial.pt')]:
        ck = torch.load(out.parent / file, map_location=bank.device, weights_only=True)
        cfg = FullMConfig(**ck['cfg'])
        actor = (IdentityActor(cfg) if label == 'identity' else FullMActor(cfg)).to(bank.device)
        actor.load_state_dict(ck['actor'])
        reports[label] = evaluate(actor, bank, cfg, c['test_shapes'], cases, out / label, run.guard)
        run.event('push_test_method_complete', seed=seed, method=label,
            success_rate=reports[label]['success_rate'], mean_cumulative_error=reports[label]['mean_cumulative_error'])
    write(out / 'RESULTS.json', reports)
    write(out / 'FINAL_STATE.json', dict(status='complete', unique_starts=len(cases)*len(c['test_shapes']),
        shared_across_methods_and_training_seeds=True, inference='calibrated-capability held-out shapes; no global stability proof'))
