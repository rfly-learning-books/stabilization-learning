from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .common import ROOT, Run, Stopped, paths, read, write, sha, source_files


def validate(config):
    for key in ('push','so101'):
        c=config[key];splits=[c[k] for k in ('train_shapes','validation_shapes','test_shapes')]
        if any(not x for x in splits):raise ValueError('Empty shape split')
        if len(set(sum(splits,[])))!=sum(map(len,splits)):
            raise ValueError('Training/validation/test shapes overlap')
        if c['updates']<1:raise ValueError('Empty training budget')
    if config['push']['seeds']!=config['so101']['seeds']:
        raise ValueError('Every SO101 seed must use the matching newly trained Push model')


def prepare(output,config_path,smoke=False):
    output=Path(output).resolve();config=read(config_path)
    if smoke:
        config=copy.deepcopy(config);config['max_hours']=1.
        for key in ('push','so101','rgb'):config[key]['seeds']=[971]
        config['push'].update(updates=2,num_envs=4,eval_every=1,validation_repeats=1,test_repeats=1,
            control=dict(max_steps=12,train_speed_points=5,eval_speed_points=9))
        config['push']['validation_shapes']=config['push']['validation_shapes'][:1]
        config['push']['test_shapes']=config['push']['test_shapes'][:1]
        config['so101'].update(updates=2,num_envs=2,horizon=4,eval_every=1,validation_repeats=1,
            test_repeats=1,arm_repeats=1,calibration_repeats=2,timing_pairs=0,
            train_shapes=['T'],validation_shapes=['Square'],test_shapes=['OffsetComb3'],
            time_config=dict(max_seconds=12.,max_actions=3))
        config['rgb'].update(smoke=True,rounds=1,warmup_updates=2,critic_prefit_updates=2,
            representation_updates=2,metric_updates=2,bootstrap_episodes_per_scene=1,
            bootstrap_steps=8,episodes_per_scene=1,episode_steps=8,evaluation_steps=8,
            eval_every=1,validation_repeats=1,test_repeats=1,log_every=1)
    validate(config)
    output.mkdir(parents=True,exist_ok=False)
    now=time.time()
    assets={}
    for p in (ROOT/'assets/push_geometric32').rglob('*'):
        if p.is_file():assets[p.relative_to(ROOT).as_posix()]=sha(p)
    assets['migration/visual_scene_reconstruction/scenes.json']=sha(ROOT/'migration/visual_scene_reconstruction/scenes.json')
    for p in (ROOT/'third_party/so101_timeaware_v1').rglob('*.png'):
        assets[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel in ('third_party/so101_timeaware_v1/MIGRATION.json',
        'third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1/extracted_state_dict.pt',
        'third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1/models.py'):
        assets[rel]=sha(ROOT/rel)
    write(output/'protocol.json',dict(schema='scratch_application_suite_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        deadline_unix=now+config['max_hours']*3600,config=config,sources=source_files(),assets=assets,
        engineering_smoke=smoke,old_policy_weights_allowed=False,old_replay_allowed=False,
        environment_priors='Push capability libraries/room geometry are fixed environment assets; public FastDepth is fixed pretrained perception.',
        original_runs_modified=False,no_guaranteed_monotonicity=True,
        stop_policy='STOP or fixed deadline; preserve diagnostics; no automatic restart and no partial test overwrite',
        selection_policy='validation only, models of all seeds frozen before tests; SO101 starts from this run Push selection',
        report_scope='finite sampled empirical improvement and held-out calibrated geometry; not a global stability theorem'))
    Run(output).event('suite_prepared',engineering_smoke=smoke,deadline_unix=now+config['max_hours']*3600)
    return output


def jobs(config):
    yield ('so101-prepare',None)
    for stage,group in [('push-train','push'),('push-test','push'),('so101-train','so101'),('so101-test','so101'),
                        ('rgb-train','rgb'),('rgb-test','rgb')]:
        for seed in config[group]['seeds']:yield stage,seed


def supervise(output):
    run=Run(output);lock=run.root/'WRITER.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    # The lock remains as evidence. No implicit resume or stale-lock deletion.
    write(run.root/'LAUNCH.json',dict(pid=os.getpid(),python=sys.executable,unix=time.time()))
    current=None
    try:
        for stage,seed in jobs(run.config):
            run.guard();current=dict(stage=stage,seed=seed)
            exe=run.config['rgb_python'] if stage.startswith('rgb-') else run.config['python']
            command=[exe,'-u',str(ROOT/'scripts/run_scratch_applications_v1.py'),'stage','--run',str(run.root),'--stage',stage]
            if seed is not None:command+=['--seed',str(seed)]
            run.event('child_launch',**current)
            log=run.root/'supervisor_logs'/f'{stage}_{seed}.log';log.parent.mkdir(exist_ok=True)
            # Root supervisor is the sole log consumer. External monitoring uses events only.
            with log.open('w',encoding='utf-8') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONUTF8':'1'})
                write(run.root/'ACTIVE_CHILD.json',dict(pid=child.pid,command=command))
                import threading
                def consume():
                    for line in child.stdout:stream.write(line);stream.flush()
                thread=threading.Thread(target=consume,daemon=True);thread.start()
                while child.poll() is None:
                    try:run.guard()
                    except Stopped:
                        # Training guards finish an update/episode first. Only kill a stuck child.
                        try:child.wait(timeout=120)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise
                    time.sleep(1)
                thread.join(timeout=10)
            if child.returncode!=0:raise RuntimeError(f'{stage} seed {seed} exited {child.returncode}; log preserved')
            run.event('child_complete',**current)
        from .report import build
        from .audit import audit
        audit(run)
        build(run)
        write(run.root/'FINAL_STATE.json',dict(status='complete',unix=time.time()))
        run.event('suite_complete',report=str(run.root/'REPORT.html'))
    except BaseException as ex:
        write(run.root/'DIAGNOSTIC.json',dict(stage=current,error=repr(ex),traceback=traceback.format_exc(),unix=time.time()))
        write(run.root/'FINAL_STATE.json',dict(status='stopped' if isinstance(ex,Stopped) else 'failed',stage=current,error=str(ex)))
        run.event('suite_stopped' if isinstance(ex,Stopped) else 'suite_failed',stage=current,error=str(ex))
        raise


def main():
    p=argparse.ArgumentParser(description='Local random-origin Push, SO101 and RGB full training')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True);a.add_argument('--config',default=str(ROOT/'configs/scratch_applications_v1.json'));a.add_argument('--smoke',action='store_true')
    a=sub.add_parser('run');a.add_argument('--run',required=True)
    a=sub.add_parser('stage');a.add_argument('--run',required=True);a.add_argument('--stage',required=True);a.add_argument('--seed',type=int)
    a=sub.add_parser('report');a.add_argument('--run',required=True)
    a=sub.add_parser('deploy');a.add_argument('--bundle',required=True);a.add_argument('--shape',default='T');a.add_argument('--seed',type=int,default=202610999);a.add_argument('--output')
    args=p.parse_args();paths()
    if args.command=='prepare':print(prepare(args.output,args.config,args.smoke));return
    if args.command=='run':supervise(args.run);return
    if args.command=='deploy':
        import torch
        import so101_multishape as m
        from so101_timeaware.timing import TimeBank
        from so101_timeaware.policy import load_actor,Selector
        bundle=Path(args.bundle);manifest=read(bundle/'DEPLOYMENT.json')
        if sha(bundle/'metric.pt')!=manifest['checkpoint_sha256'] or sha(bundle/'time_bank.json')!=manifest['time_bank_sha256']:
            raise RuntimeError('Deployment bundle changed')
        out=Path(args.output) if args.output else bundle/f'demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        out.mkdir(parents=True,exist_ok=False);bank=TimeBank(bundle/'time_bank.json')
        actor,_=load_actor(bundle/'metric.pt');q=m.c.cases(args.seed,1,extent=.012,angle=.35)[0]
        with m.install(args.shape):
            result=m.c.rollout(m.c.Config(plant='arm',speed=bank.config.transit_speed,
                max_time=bank.config.max_seconds,max_actions=bank.config.max_actions),q,args.seed,
                policy=Selector(bank,args.shape,actor,'C'),telemetry=out/'trajectory.npz',video=out/'demo.mp4')
        write(out/'RESULTS.json',result);print(out);return
    run=Run(args.run)
    if args.command=='report':
        from .report import build
        build(run);return
    group,action=args.stage.split('-')
    if group=='push':from . import push as module
    elif group=='so101':from . import so101 as module
    elif group=='rgb':from . import rgb as module
    else:raise ValueError('Unknown stage')
    if action=='prepare':module.prepare(run)
    elif action=='train':module.train(run,args.seed)
    elif action=='test':module.test(run,args.seed)
    else:raise ValueError('Unknown action')
