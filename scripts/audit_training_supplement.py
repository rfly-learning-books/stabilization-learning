"""Quantify the two audit findings without changing existing experiments."""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from audit_training_results import ROOT, compare, digest, make, read, replay_training
from stabilization_learning.wrappers import EpisodeMetricEnv


def floor_impact(saved):
    module = importlib.import_module("stabilization_learning.algos.pytorch.metric_ppo.metric_ppo")
    original = module.rollout_metric
    counters = dict(rollouts=0, steps=0, norm_below_floor_steps=0, positive_excess_steps=0,
                    total_penalty_excess=0., max_episode_penalty_excess=0., minimum_error_norm_squared=None)
    def observed(env, M, observation, **kwargs):
        norms = []
        own_method = env.__dict__.get("build_error")
        method = env.build_error
        def capture(*args, **kw):
            e, context = method(*args, **kw)
            norms.append(float(e.detach().square().sum()))
            return e, context
        env.build_error = capture
        requested_trajectory = kwargs.get("save_trajectory", False)
        kwargs["save_trajectory"] = True
        try:
            result = original(env, M, observation, **kwargs)
        finally:
            if own_method is None:
                del env.build_error
            else:
                env.build_error = own_method
        trajectory = result["trajectory"]
        assert len(norms) == 2*len(trajectory)
        excess = 0.
        for E, row in zip(norms[::2], trajectory):
            scale = max(E, 1e-6)
            delta = row["V_after"]-row["V_before"]
            alpha = kwargs.get("decay", .05)
            actual = max(0., delta+alpha*env.dt*scale)/scale
            expected = max(0., delta+alpha*env.dt*E)/scale
            difference = actual-expected
            excess += difference
            counters["norm_below_floor_steps"] += int(E < 1e-6)
            counters["positive_excess_steps"] += int(difference > 1e-12)
        counters["rollouts"] += 1
        counters["steps"] += len(trajectory)
        counters["total_penalty_excess"] += excess
        counters["max_episode_penalty_excess"] = max(counters["max_episode_penalty_excess"], excess)
        minimum = min(norms[::2])
        old = counters["minimum_error_norm_squared"]
        counters["minimum_error_norm_squared"] = minimum if old is None else min(old, minimum)
        if not requested_trajectory:
            result.pop("trajectory")
        return result
    module.rollout_metric = observed
    try:
        replay = replay_training(saved)
    finally:
        module.rollout_metric = original
    counters["history_and_weights_still_exact"] = replay["history"]["exact"] and replay["actor"]["exact"]
    counters["scope"] = "All replayed training rollouts plus initial and per-update validation; same original objective"
    return counters


def recover_slc(output):
    from stabilization_learning.integrations.slc import train_slc, load_slc_policy
    original_folder = ROOT/"runs/slc_pointmass_smoke_seed41_v2"
    old = torch.load(original_folder/"lac_final.pt", map_location="cpu", weights_only=False)
    # Cost scales quadratically for this unsaturated zero-goal linear system.
    # This recovery is an inference, not previously recorded provenance.
    results = {}
    for scale in (.7, .3):
        env = EpisodeMetricEnv(make("point_mass", max_steps=40, initial_scale=scale), raw_bound=1.)
        rows = []
        for i, raw in enumerate(old["replay"]["act"]):
            env.reset(seed=41+i)
            actual = env.step(raw)[1]
            expected = old["records"][i]["cost"]
            rows.append(dict(original=expected, recomputed=actual, ratio=actual/expected, difference=actual-expected))
        env.close()
        results[str(scale)] = rows
    factory = lambda: EpisodeMetricEnv(make("point_mass", max_steps=40, initial_scale=.3), raw_bound=1.)
    recreated = output/"slc_exact_reconstruction"
    train_slc(factory, recreated, **old["settings"])
    new = torch.load(recreated/"lac_final.pt", map_location="cpu", weights_only=False)
    comparisons = {key:compare(new[key], old[key]) for key in
                   ("state_dict", "optimizers", "replay", "records", "diagnostics", "rng")}
    wrong = lambda:EpisodeMetricEnv(make("point_mass", max_steps=4, initial_scale=2., dt=.02, metric_gain=20.), raw_bound=1.)
    try:
        load_slc_policy(wrong, original_folder/"lac_final.pt")
        wrong_accepted = True
    except ValueError:
        wrong_accepted = False
    return {"scale_cost_replays":results,"recovered_configuration":{"env":"point_mass","max_steps":40,
            "initial_scale":.3,"raw_bound":1.,"other_values":"Current defaults"},
            "configuration_status":"Inferred from all eight recorded costs, then verified by full training-state reconstruction",
            "exact_training_reconstruction":comparisons,
            "incompatible_horizon_dt_initial_scale_metric_gain_accepted":wrong_accepted,
            "missing_checkpoint_fields":["dt","max_steps","initial_scale","metric_gain","physical goal",
                "certificate_weight","decay","physical success/failure rules"]}


def representation_replay():
    spec=importlib.util.spec_from_file_location("example",ROOT/"examples/representation_training.py")
    example=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    folder=ROOT/"runs/representation_demo_smoke"
    saved=torch.load(folder/"representation.pt",map_location="cpu",weights_only=False)
    old=read(folder/"RESULTS.json")
    torch.manual_seed(old["seed"])
    train,_=example.make_dataset(32,split="train",image_size=saved["image_size"],seed=41)
    validation,_=example.make_dataset(12,split="val",image_size=saved["image_size"],seed=1041)
    ae=example.RGBAutoEncoder(latent_dim=16,image_size=saved["image_size"])
    head=example.DerivativeHead(latent_dim=16,action_dim=4)
    learner=example.RepresentationLearner(ae,head,lr=1e-3,auxiliary_cap=.25)
    history=[]
    for step in range(12):
        ids=torch.randint(32,(16,))
        inputs=torch.cat((train["images"][ids],train["goals"][ids]))
        goals=torch.cat((train["goals"][ids],train["goals"][ids]))
        actions=torch.cat((train["velocities"][ids],train["velocities"][ids]))
        targets=torch.cat((train["targets"][ids],torch.zeros(len(ids))))
        stats=learner.update(inputs,goals,actions,targets)
        history.append(dict(**stats,update=step+1))
    return {"AE_weights":compare(ae.state_dict(),saved["autoencoder"]),
            "head_weights":compare(head.state_dict(),saved["derivative_head"]),
            "gradient_history":compare(history,old["history"]),
            "final_validation":compare(example.evaluate(ae,head,validation),old["final_validation"])}


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    output=ROOT/"runs/result_audit_20261022"
    original=read(output/"ORIGINAL_FILES.json")
    baseline=read(output/"AUDIT.json")
    data={"floor_impact":{}}
    for name in baseline["ppo"]:
        saved=torch.load(ROOT/"runs"/name/"last.pt",map_location="cpu",weights_only=False)
        data["floor_impact"][name]=floor_impact(saved)
        print(json.dumps({"run":name,**data["floor_impact"][name]}),flush=True)
    data["slc"]=recover_slc(output)
    data["representation"]=representation_replay()
    data["preservation"]={"changed_original_files":[p for p,h in original.items() if digest(ROOT/p)!=h],
        "changed_source_files":[p for p,h in baseline["source_sha256"].items() if digest(ROOT/p)!=h]}
    (output/"SUPPLEMENT.json").write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"complete":True,"supplement":str(output/"SUPPLEMENT.json")}),flush=True)


if __name__=="__main__":
    main()
