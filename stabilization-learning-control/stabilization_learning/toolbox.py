"""Installed, relocatable P0 inspection/demo entry; never trains by default."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stabilization_gym import make,available_envs
from stabilization_gym.envs.classic_control.sdc_metric import SDCMetricTask
from stabilization_learning.policies.state_metric_v2 import StateMetricPolicy
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash,atomic_save
from stabilization_learning.common.contracts_v2 import identity


def main():
    parser=argparse.ArgumentParser(description="P0 task contracts and bounded CPU demos")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("list")
    demo=sub.add_parser("demo")
    demo.add_argument("--task",choices=["coupled_linear","point_mass","pendulum","cartpole"],required=True)
    demo.add_argument("--steps",type=int,default=8)
    demo.add_argument("--seed",type=int,default=71)
    demo.add_argument("--checkpoint",type=Path)
    demo.add_argument("--resume",action="store_true")
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=="list":
        print(json.dumps(dict(legacy_environments=list(available_envs()),
            state_M_SDC=["coupled_linear","point_mass","pendulum","cartpole"],
            state_backend="state_metric_model_gradient_v2",default_action="inspection only"),ensure_ascii=False,indent=2))
        return
    if args.steps<0:parser.error("--steps must be nonnegative")
    physical=make(args.task);task=SDCMetricTask(physical,args.task)
    torch.manual_seed(41)
    policy=StateMetricPolicy(task.metric_spec,task.L0,goal=task.goal,context_scale=max(float(physical.initial_scale),.1))
    source_files=[Path(__file__),Path(__import__(SDCMetricTask.__module__,fromlist=["x"]).__file__),
        Path(__import__(StateMetricPolicy.__module__,fromlist=["x"]).__file__)]
    signature=identity(dict(task=task.describe().to_dict(),goal=task.goal,dt=task.dt,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}))
    if args.resume:
        if args.checkpoint is None:parser.error("--resume requires --checkpoint")
        payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        if payload["signature"]!=signature:raise ValueError("Demo source/task signature mismatch")
        policy.load_state_dict(payload["policy"]);state=payload["state"].clone();step=payload["step"]
    else:
        state=task.initial_states([args.seed])[0];step=0
    with torch.no_grad():
        for _ in range(args.steps):
            error=task.error(state);M=policy(error);action=task.control(M,error,state=state)
            state=task.step_model(state,action);step+=1
            if task.failure(state):break
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        atomic_save(dict(signature=signature,policy=policy.state_dict(),state=state,step=step),args.checkpoint)
    print(json.dumps(dict(task=args.task,steps=step,state=state.tolist(),physical_error=float(task.error(state).square().sum()),
        M=policy(task.error(state)).detach().tolist(),policy_sha256=module_hash(policy),
        candidate_only=True,training_updates=0,source_signature=signature),ensure_ascii=False))
    physical.close()


if __name__=="__main__":main()
