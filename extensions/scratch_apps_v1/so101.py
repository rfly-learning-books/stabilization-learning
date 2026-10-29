"""Fresh contact calibration -> frozen scratch M -> timed PPO -> actuated simulation."""
from __future__ import annotations
from dataclasses import asdict
import copy
import numpy as np
import torch

from .common import ROOT, Run, paths, write, read, sha, seed_all, save_torch, tensor_hash
paths()
import so101_multishape as shapes
from so101_timeaware.timing import TimeConfig, TimeBank, prepare as prepare_bank
from so101_timeaware.policy import CandidateModel, TimeCritic, Selector, load_actor
from so101_timeaware.environment import TimedResponseEnv, reached, energy
from so101_timeaware.training import collect, update


def prepare(run: Run):
    c = run.config['so101']; seed_all(run.config['sampling_seed'])
    out = run.new_stage('so101_environment', dict(config=c, calibration='new actuated MuJoCo samples',
        no_policy_used_in_calibration=True, no_historical_timing_table=True,
        capability_encoding='C * [1,1,0.05]; state [x/.05,y/.05,sqrt(2)*sin(yaw/2)]',
        scope='actuated SO101 simulation, explicit inertial/PD assumptions, no hardware'))
    for i, name in enumerate(c['train_shapes'] + c['validation_shapes'] + c['test_shapes']):
        run.guard(); target = out / 'calibration' / name; target.mkdir(parents=True)
        with shapes.install(name) as (_, poly, points, normals):
            calibration = shapes.calibrate(target, points, normals, repeat=c['calibration_repeats'],
                                           seed=run.config['sampling_seed'] + 3000 + i)
        if not any(calibration['valid']):
            raise RuntimeError(f'No valid newly calibrated contact for {name}; no fallback to old data')
        run.event('so101_calibrated', shape=name, valid_modes=sum(calibration['valid']),
                  successful_trials=sum(x['success'] for x in calibration['records']))
    cfg = TimeConfig(**c.get('time_config', {}), encoder_omega_multiplier=1.)
    prepare_bank(out / 'time_bank.json', c['train_shapes'] + c['validation_shapes'] + c['test_shapes'],
                 out / 'calibration', cfg, measure_pairs=c['timing_pairs'], seed=run.config['sampling_seed']+4000)
    write(out / 'FINAL_STATE.json', dict(status='complete', sha256=sha(out/'time_bank.json')))
    run.event('so101_environment_ready', time_bank_sha256=sha(out/'time_bank.json'))


def numeric(actor, bank, names, seeds, variant, guard=lambda: None):
    rows = []
    for name in names:
        # Install shape geometry for the unchanged A persistence selector.
        with shapes.install(name):
            policy = Selector(bank, name, actor, variant)
            for seed in seeds:
                guard(); rng = np.random.default_rng(seed)
                q0 = np.r_[rng.uniform(-.012, .012, 2), rng.uniform(-.35, .35)]
                env = TimedResponseEnv(bank, [name], seed); env.reset(name, q0)
                trace = []; integral = 0.
                while not env.done:
                    guard(); q = env.q.copy(); v = float(policy.value(q)[0])
                    try:
                        if reached(env.spec, env.q):
                            index = 0
                        else:
                            mode, travel, _, recovery = policy.choose(env.q,
                                None if env.mode < 0 else env.mode, True, env.pusher)
                            k = int(np.argmin(abs(np.array(bank.config.travels)-travel)))
                            index = mode * len(bank.config.travels) + k
                        _, reward, done, info = env.step(index)
                        integral += info['integrated_error']
                        trace.append(dict(q=q.tolist(), q_next=env.q.tolist(), V=v,
                            V_next=float(policy.value(env.q)[0]), action=index, **info))
                    except RuntimeError as ex:
                        env.done = True; env.reason = str(ex)
                rows.append(dict(shape=name, seed=seed, initial=q0.tolist(), success=env.reason=='success',
                    reason=env.reason, seconds=env.elapsed, cumulative_error=integral,
                    repositions=env.repositions, actions=env.actions, trace=trace))
    success = [x['seconds'] for x in rows if x['success']]
    return dict(episodes=rows, success_rate=float(np.mean([x['success'] for x in rows])),
        mean_seconds=float(np.mean([x['seconds'] for x in rows])),
        mean_success_seconds=None if not success else float(np.mean(success)),
        mean_cumulative_error=float(np.mean([x['cumulative_error'] for x in rows])))


def train(run: Run, seed):
    run.frozen_group('push'); c = run.config['so101']; seed_all(seed)
    bank = TimeBank(run.root / 'so101_environment/time_bank.json')
    source = run.root / f'push_seed{seed}' / 'selected.pt'
    out = run.new_stage(f'so101_seed{seed}', dict(config=c, seed=seed,
        initial_checkpoint=str(source), initial_checkpoint_sha256=sha(source),
        time_bank_sha256=bank.sha256, metric_source='this run, random-origin full-M PPO',
        selection='validation success first, then elapsed time; held-out geometries never select'))
    actor, ck = load_actor(source, 'cpu')
    if ck.get('origin') != 'random' or ck['protocol_sha256'] != sha(run.root/'protocol.json'):
        raise RuntimeError('SO101 refuses a policy outside this random-origin run')
    initial = tensor_hash(actor)
    model = CandidateModel(bank, 'cpu'); critic = TimeCritic(actor.cfg.hidden)
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=c['learning_rate'])
    envs = [TimedResponseEnv(bank, c['train_shapes'], seed*10000+i) for i in range(c['num_envs'])]
    history=[]; best_key=None; best_update=None
    def save(file, n):
        save_torch(out/file, dict(schema='scratch_so101_v1', actor=actor.state_dict(), cfg=asdict(actor.cfg),
            critic=critic.state_dict(), optimizer=optimizer.state_dict(), update=n,
            environments=[e.state_dict() for e in envs], torch_rng=torch.get_rng_state(),
            initial_checkpoint_sha256=sha(source), initial_actor_hash=initial,
            time_bank_sha256=bank.sha256, protocol_sha256=sha(run.root/'protocol.json')))
    save('initial.pt', 0)
    for n in range(c['updates']+1):
        run.guard()
        if n:
            data, episodes = collect(actor, critic, model, envs, c['horizon'])
            stats = update(actor, critic, optimizer, model, data, c['epochs'], c['minibatch'])
            stats.update(update=n, seed=seed, transitions=c['num_envs']*c['horizon'],
                completed_episodes=len(episodes), success_rate=None if not episodes else
                float(np.mean([x['success'] for x in episodes])))
            history.append(stats); save('last.pt', n); run.event('so101_update', **stats)
        if n==0 or n % c['eval_every']==0 or n==c['updates']:
            rng=torch.get_rng_state()
            val=numeric(actor,bank,c['validation_shapes'],
                [run.config['sampling_seed']+5000+k for k in range(c['validation_repeats'])], 'B', run.guard)
            torch.set_rng_state(rng)
            key=(val['success_rate'], -(val['mean_success_seconds'] or bank.config.max_seconds), -val['mean_cumulative_error'])
            if n>0 and (best_key is None or key>best_key):
                best_key=key; best_update=n; save('selected.pt',n)
            write(out/'validation'/f'update_{n:05d}.json',val)
            run.event('so101_validation',seed=seed,update=n,selected_update=best_update,
                success_rate=val['success_rate'],mean_seconds=val['mean_seconds'],
                mean_success_seconds=val['mean_success_seconds'],mean_cumulative_error=val['mean_cumulative_error'])
    save('final.pt', c['updates']); write(out/'TRAINING.json',history)
    write(out/'FROZEN.json',dict(checkpoint='selected.pt',sha256=sha(out/'selected.pt'),
        selected_update=best_update, initial_actor_hash=initial, final_actor_hash=tensor_hash(actor)))
    run.event('so101_frozen',seed=seed,selected_update=best_update)


def test(run: Run, seed):
    run.frozen_group('so101'); c=run.config['so101']; bank=TimeBank(run.root/'so101_environment/time_bank.json')
    out=run.new_stage(f'so101_seed{seed}/test',dict(config=c,selection_allowed=False,
        scope='A: learned scratch M, original persistence; B: same M, time selector; C: timed learned M'))
    variants=[('A_frozen_original','initial.pt','A'),('B_frozen_time','initial.pt','B'),
              ('C_time_selected','selected.pt','C'),('C_time_final','final.pt','C')]
    reports={}; names=c['train_shapes']+c['validation_shapes']+c['test_shapes']
    for label,file,variant in variants:
        actor,_=load_actor(out.parent/file,'cpu')
        reports[label]=numeric(actor,bank,names,
            [run.config['sampling_seed']+6000+i for i in range(c['test_repeats'])],variant,run.guard)
        write(out/'numeric'/f'{label}.json',reports[label])
    arm=[]
    for label,file,variant in variants[:3]:
        actor,_=load_actor(out.parent/file,'cpu')
        for name in names:
            with shapes.install(name):
                policy=Selector(bank,name,actor,variant)
                for i in range(c['arm_repeats']):
                    run.guard(); sample_seed=run.config['sampling_seed']+7000+i
                    q=shapes.c.cases(sample_seed,1,extent=.012,angle=.35)[0]
                    folder=out/'arm'/label/name;folder.mkdir(parents=True,exist_ok=True)
                    cfg=shapes.c.Config(plant='arm',speed=bank.config.transit_speed,
                        max_time=bank.config.max_seconds,max_actions=bank.config.max_actions)
                    trace=folder/f'{i:03d}.npz'
                    # One predetermined demonstration, not the best test episode.
                    video=folder/f'{i:03d}.mp4' if (seed==c['seeds'][0] and label=='C_time_selected' and name==names[0] and i==0) else None
                    result=shapes.c.rollout(cfg,q,sample_seed,policy=policy,telemetry=trace,video=video)
                    result.update(shape=name,variant=label,heldout=name in c['test_shapes'],
                        trace_sha256=sha(trace),checkpoint_sha256=sha(out.parent/file))
                    write(folder/f'{i:03d}.json',result);arm.append(result)
                    run.event('so101_arm_episode',seed=seed,variant=label,shape=name,episode=i,
                        success=result['success'],reason=result['reason'],seconds=result['sim_time_s'])
    write(out/'RESULTS.json',dict(numeric=reports,arm=arm,
        test_shape_claim='OffsetComb3/AsymThreeBranch absent from metric training; capability calibration supplied',
        caveat='Reduced-model timing and measured MuJoCo execution are distinct; no physical hardware deployment'))
    write(out/'FINAL_STATE.json',dict(status='complete',test_selection=False))
    # The deployable bundle uses exactly the validation-selected time model.
    bundle=run.root/f'deployment_so101_seed{seed}';bundle.mkdir()
    import shutil
    shutil.copy2(out.parent/'selected.pt',bundle/'metric.pt')
    shutil.copy2(bank.path,bundle/'time_bank.json')
    write(bundle/'DEPLOYMENT.json',dict(checkpoint_sha256=sha(bundle/'metric.pt'),time_bank_sha256=bank.sha256,
        seed=seed,simulation_only=True,input='estimated planar pose, contact mode, calibrated C, pusher pose',
        source_push_checkpoint_sha256=sha(run.root/f'push_seed{seed}'/'selected.pt'),
        command=f'python scripts/run_scratch_applications_v1.py deploy --bundle {bundle} --shape T'))
