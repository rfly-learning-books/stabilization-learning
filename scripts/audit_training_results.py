"""Read-only audit of saved examples. Writes only into a new audit directory.

Deterministic optimizer replay reproduces an existing short run in memory; it
does not start a new experiment, select using test data, or replace checkpoints.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import os

import numpy as np
import torch

from stabilization_learning import make, StabilizationLearner, TrainConfig
from stabilization_learning.trainer import module_hash, summarize
from stabilization_learning.common.compat import canonical_protocol
from stabilization_learning.wrappers import rollout_metric, EpisodeMetricEnv

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    """Report exact agreement and the largest absolute numeric difference."""
    differences = []
    structure = []
    def walk(x, y, key=""):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape:
                structure.append(key)
            elif x.size:
                differences.append(float(np.max(np.abs(x.astype(float)-y.astype(float)))))
        elif isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys():
                structure.append(key)
            for k in x.keys() & y.keys():
                walk(x[k], y[k], f"{key}/{k}")
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                structure.append(key)
            for i, (xx, yy) in enumerate(zip(x, y)):
                walk(xx, yy, f"{key}/{i}")
        elif isinstance(x, (int, float, bool, np.number)) and isinstance(y, (int, float, bool, np.number)):
            differences.append(abs(float(x)-float(y)))
        elif x != y:
            structure.append(key)
    walk(a, b)
    maximum = max(differences, default=0.)
    return {"exact": not structure and maximum == 0., "max_abs_difference": maximum,
            "structure_mismatches": structure, "within_1e_7": not structure and maximum <= 1e-7}


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, float) else True


def construct(protocol, overrides=None):
    kwargs = {**protocol["env_kwargs"], **(overrides or {})}
    env = make(protocol["env_id"], **kwargs)
    learner = StabilizationLearner(env, config=TrainConfig(**protocol["config"]),
                                   env_id=protocol["env_id"], env_kwargs=kwargs)
    if not overrides:
        assert canonical_protocol(learner._protocol()) == canonical_protocol(protocol)
    return learner


def metric_details(actor, env, seed):
    obs, _ = env.reset(seed=seed)
    with torch.no_grad():
        context = env.metric_context(obs)
        raw = actor.mean(context).cpu().numpy().astype(float)
        actual = actor(context).cpu().numpy()
    n, spec = env.error_dim, env.metric_spec
    L = np.zeros((n, n))
    rows, cols = np.tril_indices(n)
    L[rows, cols] = np.where(rows == cols, np.logaddexp(0., raw)+1e-8, raw)
    gram = L @ L.T
    independent = spec.eigen_floor*np.eye(n)+(spec.trace-n*spec.eigen_floor)*gram/np.trace(gram)
    eigenvalues = np.linalg.eigvalsh(actual.astype(float))
    return {"eigen_min": float(eigenvalues.min()), "eigen_max": float(eigenvalues.max()),
            "trace": float(np.trace(actual)), "distance_from_identity": float(np.linalg.norm(actual-np.eye(n))),
            "independent_cholesky_max_error": float(np.max(np.abs(actual-independent))),
            "spectrum_ok": bool(eigenvalues.min() >= spec.eigen_floor-1e-6 and
                                eigenvalues.max() <= spec.trace-(n-1)*spec.eigen_floor+1e-6),
            "trace_ok": abs(float(np.trace(actual))-spec.trace) < 1e-5}


def replay_training(saved):
    learner = construct(saved["protocol"])
    first = learner.evaluate()["summary"]
    learner.best_score, learner.best_actor = first["objective"], deepcopy(learner.actor.state_dict())
    learner.history.append(dict(updates=0, episodes=0, physical_steps=0, validation=first))
    while learner.episodes < saved["episodes"]:
        batch = learner._collect(min(learner.config.batch_episodes, saved["episodes"]-learner.episodes))
        stats = learner._update(*batch[:4])
        validation = learner.evaluate()["summary"]
        if validation["objective"] < learner.best_score:
            learner.best_score, learner.best_actor = validation["objective"], deepcopy(learner.actor.state_dict())
        learner.history.append(dict(updates=learner.updates, episodes=learner.episodes,
            physical_steps=learner.physical_steps, train=summarize(batch[4]), validation=validation, **stats))
    result = {k: compare(getattr(learner, k).state_dict(), saved[k]) for k in
              ("actor", "value", "optimizer", "value_optimizer")}
    result.update(history=compare(learner.history, saved["history"]),
                  best_actor=compare(learner.best_actor, saved["best_actor"]),
                  physical_steps_equal=learner.physical_steps == saved["physical_steps"],
                  torch_rng_equal=torch.equal(torch.get_rng_state(), saved["torch_rng"]),
                  representation_hash_unchanged=module_hash(getattr(learner.env, "representation", None)) ==
                  saved["protocol"]["representation_hash"])
    learner.env.close()
    return result


def audit_ppo(run):
    folder = ROOT / "runs" / run
    saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    history, protocol = saved["history"], saved["protocol"]
    selected = min(history, key=lambda row: row["validation"]["objective"])
    learner = construct(protocol)
    initial_state = deepcopy(learner.actor.state_dict())
    score_errors = [abs(s["objective"]-s["fixed_horizon_cost"]-
                       learner.config.certificate_weight*s["certificate_penalty"])
                    for row in history for key, s in row.items() if key in ("train", "validation")]
    train_seeds = set(range(learner.config.seed, learner.config.seed+saved["episodes"]))
    validation_seeds = set(learner.config.validation_seeds)
    test_seeds = (20001, 20002, 20003, 20004) if run == "point_mass_demo" else (20001, 20002)
    result = {"episodes": saved["episodes"], "physical_steps": saved["physical_steps"],
              "updates": saved["updates"], "physical_horizon": protocol["physical_horizon"],
              "finite_weights_optimizers_history": finite({k:saved[k] for k in
                  ("actor", "value", "optimizer", "value_optimizer", "history")}),
              "history_matches_json": compare(history, read(folder / "progress.json")),
              "best_selected_update": selected["updates"],
              "best_matches_initial_actor": compare(best["actor"], initial_state)["exact"],
              "best_matches_saved_best_actor": compare(best["actor"], saved["best_actor"]),
              "best_score_matches_validation": best["score"] == saved["best_score"] == selected["validation"]["objective"],
              "score_arithmetic_max_error": max(score_errors),
              "seed_sets_disjoint": not (train_seeds & validation_seeds or train_seeds & set(test_seeds) or validation_seeds & set(test_seeds)),
              "actor_weight_max_change": compare(initial_state, saved["actor"])["max_abs_difference"],
              "optimizer_step_counts": {k:sorted(set(int(v["step"]) for v in saved[k]["state"].values()))
                                         for k in ("optimizer", "value_optimizer")},
              "expected_optimizer_steps": sum(row.get("epochs_run", 0) for row in history)}
    learner.actor.load_state_dict(best["actor"])
    result["best_metric"] = metric_details(learner.actor, learner.env, test_seeds[0])
    result["best_validation"] = learner.evaluate()["summary"]
    result["best_validation_match"] = compare(result["best_validation"], selected["validation"])
    result["best_test"] = learner.evaluate(test_seeds, trajectories=True)
    result["identity_test"] = learner.evaluate(test_seeds, identity=True, trajectories=True)
    learner.actor.load_state_dict(saved["actor"])
    result["final_metric"] = metric_details(learner.actor, learner.env, test_seeds[0])
    result["final_validation"] = learner.evaluate()["summary"]
    result["final_validation_match"] = compare(result["final_validation"], history[-1]["validation"])
    result["final_test_diagnostic_only"] = learner.evaluate(test_seeds, trajectories=True)
    if run.startswith("platform_smoke_v2/"):
        recorded = read(ROOT / "runs/platform_smoke_v2/RESULTS.json")[protocol["env_id"]]
        result["recorded_test_match"] = compare(result["best_test"]["summary"], recorded["evaluation"])
    elif (folder / "evaluation.json").exists():
        recorded = read(folder / "evaluation.json")
        result["recorded_test_match"] = compare(result["best_test"], recorded["learned"])
        result["recorded_identity_match"] = compare(result["identity_test"], recorded["identity"])
    learner.env.close()
    result["in_memory_training_replay"] = replay_training(saved)
    return result


def independent_point_mass(episodes, M, horizon, decay, certificate_weight):
    """Use NumPy dynamics and closed-form CARE gains; never env.step/rollout."""
    comparisons, results, rise_counts, steps = [], [], 0, 0
    kp, kd, dt = math.sqrt(8.), math.sqrt(2.+2.*math.sqrt(8.)), .04
    for recorded in episodes:
        x = np.random.default_rng(recorded["seed"]).uniform(-1., 1., 4)*.7
        total = penalty = 0.
        rises = hold = 0
        rows = []
        for t in range(horizon):
            before = float(x @ M @ x)
            a = np.clip(-kp*x[:2]-kd*x[2:]-.6*((M-np.eye(4)) @ x)[2:], -6., 6.)
            cost = dt*float(x @ x)
            new = np.r_[x[:2]+dt*x[2:]+.5*dt**2*a, x[2:]+dt*a]
            after = float(new @ M @ new)
            E = float(new @ new)
            norm = max(float(x @ x), 1e-6)
            penalty += max(0., after-before+decay*dt*norm)/norm
            rises += int(after-before > 1e-8)
            total += cost
            hold = hold+1 if E <= .05**2 else 0
            rows.append(dict(step=t+1, time=(t+1)*dt, V_before=before, V_after=after,
                             action=a.tolist(), cost=cost, physical_error=E))
            x = new
            if hold >= 5 or np.linalg.norm(x) > 100:
                break
        n = len(rows)
        success, failure = hold >= 5, np.linalg.norm(x) > 100
        fixed = total+(horizon-n)*dt*(max(E,25.) if failure else E)
        result = dict(fixed_horizon_cost=fixed, accumulated_cost=total, objective=fixed+certificate_weight*penalty,
                      certificate_penalty=penalty, V_rise_fraction=rises/n, success=bool(success), failure=bool(failure),
                      physical_steps=n, arrival_time=n*dt if success else horizon*dt, terminal_physical_error=E,
                      physical_terminated=bool(success or failure), physical_truncated=bool(n>=horizon and not (success or failure)),
                      trajectory=rows, seed=recorded["seed"])
        comparisons.append(compare(result, recorded))
        results.append(result)
        rise_counts += rises
        steps += n
    return {"episode_comparisons": comparisons, "summary": summarize(results), "V_rise_steps": rise_counts,
            "all_steps": steps, "step_weighted_V_rise_fraction": rise_counts/steps,
            "mean_episode_V_rise_fraction": float(np.mean([r["V_rise_fraction"] for r in results]))}


def audit_transfer(run, filename):
    folder = ROOT / "runs" / run
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    old = read(folder / filename)
    learner = construct(saved["protocol"], old["evaluation_env_kwargs"])
    learner.actor.load_state_dict(saved["actor"])
    learned = learner.evaluate(old["test_seeds"], trajectories=True)
    identity = learner.evaluate(old["test_seeds"], identity=True, trajectories=True)
    result = {"best": learned["summary"], "identity": identity["summary"],
              "best_all_trajectory_fields_match": compare(learned, old["learned"]),
              "identity_all_trajectory_fields_match": compare(identity, old["identity"]),
              "paired_cost_delta": [a["fixed_horizon_cost"]-b["fixed_horizon_cost"]
                  for a,b in zip(learned["episodes"],identity["episodes"])],
              "evaluation_env_kwargs":old["evaluation_env_kwargs"]}
    # Independently integrate visual body velocity in NumPy. Audit actual states,
    # signed V increments, physical cost, success hold and command constraints.
    if saved["protocol"]["env_id"] == "visual_servo":
        errors = dict(physical_step=0., physical_error=0., cost=0., V=0., action=0.)
        floor_count, excess_penalty, total_steps = 0, 0., 0
        env = learner.env
        initial_phi_hash = module_hash(env.representation)
        for episode in learned["episodes"]:
            obs, _ = env.reset(seed=episode["seed"])
            with torch.no_grad():
                M = learner.actor(env.metric_context(obs))
            state, hold = env.state.copy(), 0
            for row in episode["trajectory"]:
                e, context = env.build_error(obs)
                E = float(e.detach().square().sum())
                action = env.calculate_action_from_M(M, e, context=context).detach().numpy()
                errors["action"] = max(errors["action"],float(np.max(np.abs(action-row["action"]))))
                yaw = state[3]+env.dt*.5*action[3]
                state += env.dt*np.array([math.cos(yaw)*action[0]-math.sin(yaw)*action[1],
                                         math.sin(yaw)*action[0]+math.cos(yaw)*action[1],action[2],action[3]])
                d=state-env.goal
                yaw_error=math.atan2(math.sin(d[3]),math.cos(d[3]))
                physical=float(d[:3]@d[:3]+(yaw_error/math.radians(25))**2)
                v=float(e.detach().numpy().astype(float)@M.numpy().astype(float)@e.detach().numpy().astype(float))
                errors["V"]=max(errors["V"],abs(v-row["V_before"]))
                errors["physical_error"]=max(errors["physical_error"],abs(physical-row["physical_error"]))
                errors["cost"]=max(errors["cost"],abs(env.dt*physical-row["cost"]))
                delta=row["V_after"]-row["V_before"]
                scale=max(E,1e-6)
                excess_penalty += max(0.,delta+learner.config.decay*env.dt*scale)/scale-max(0.,delta+learner.config.decay*env.dt*E)/scale
                floor_count += E < 1e-6
                total_steps += 1
                obs, _, term, trunc, info = env.step(action)
                errors["physical_step"]=max(errors["physical_step"],float(np.max(np.abs(state-env.state))))
                good=np.linalg.norm(d[:3])<.02 and abs(yaw_error)<math.radians(1.)
                hold=hold+1 if good else 0
                assert bool(info["success"]) == (hold>=11)
                assert np.linalg.norm(action[:3]) <= .400001 and abs(action[3]) <= .350001
        result["independent_visual_replay"] = {"max_errors":errors,"steps":total_steps,
            "norm_below_floor_steps":int(floor_count),"penalty_excess_over_document_formula":excess_penalty,
            "phi_hash_unchanged":initial_phi_hash==module_hash(env.representation)}
    learner.env.close()
    return result


def audit_representation():
    spec=importlib.util.spec_from_file_location("representation_example",ROOT/"examples/representation_training.py")
    example=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    old=read(ROOT/"runs/representation_demo_smoke/RESULTS.json")
    ckpt=torch.load(ROOT/"runs/representation_demo_smoke/representation.pt",map_location="cpu",weights_only=False)
    torch.manual_seed(old["seed"])
    train,tf=example.make_dataset(old["train_samples"],split="train",image_size=ckpt["image_size"],seed=old["seed"])
    val,vf=example.make_dataset(old["validation_samples"],split="val",image_size=ckpt["image_size"],seed=old["seed"]+1000)
    ae=example.RGBAutoEncoder(latent_dim=16,image_size=ckpt["image_size"])
    head=example.DerivativeHead(latent_dim=16,action_dim=4)
    init=example.evaluate(ae,head,val)
    initial_state=deepcopy(ae.state_dict())
    ae.load_state_dict(ckpt["autoencoder"])
    head.load_state_dict(ckpt["derivative_head"])
    final=example.evaluate(ae,head,val)
    zero_rmse=float(val["targets"].square().mean().sqrt())
    aux_nonzero=all(r["encoder_aux_grad_norm"]>0 and r["effective_aux_ratio"]>0 for r in old["history"])
    return {"initial_validation":init,"final_validation":final,
            "initial_match":compare(init,old["initial_validation"]),
            "final_match":compare(final,old["final_validation"]),
            "label_finite_difference_error":max(tf,vf),"zero_predictor_rmse":zero_rmse,
            "RMSE_improvement_over_zero_percent":100*(zero_rmse-final["derivative_rmse"])/zero_rmse,
            "reported_aux_gradient_nonzero":aux_nonzero,
            "gradient_cap_satisfied":all(r["effective_aux_ratio"]<=.25+1e-7 for r in old["history"]),
            "AE_parameter_max_change":compare(initial_state,ckpt["autoencoder"])["max_abs_difference"],
            "frozen_policy_phi_hash":module_hash(ae),"finite":finite(ckpt)}


def audit_slc():
    from stabilization_learning.integrations.slc import load_slc_policy,_parameter_hash
    folder=ROOT/"runs/slc_pointmass_smoke_seed41_v2"
    c=torch.load(folder/"lac_final.pt",map_location="cpu",weights_only=False)
    old=read(folder/"RESULTS.json")
    factory=lambda:EpisodeMetricEnv(make("point_mass",max_steps=40),raw_bound=1.)
    policy=load_slc_policy(factory,folder/"lac_final.pt")
    action=policy.get_action(np.array(old["probe_observation"],dtype=np.float32),deterministic=True)
    env=factory()
    errors=[]
    for index,(raw,recorded) in enumerate(zip(c["replay"]["act"],old["records"])):
        env.reset(seed=old["settings"]["seed"]+index)
        _,cost,term,trunc,info=env.step(raw)
        errors.append(abs(cost-recorded["cost"]))
        assert info["success"]==recorded["success"] and info["failure"]==recorded["failure"]
        assert term and not trunc
    env.close()
    return {"probe_action":compare(action,np.array(old["probe_action"])),
            "actor_final_hash_match":_parameter_hash(policy.ac.pi)==old["actor_final_sha256"],
            "critic_final_hash_match":_parameter_hash(policy.ac.L)==old["critic_final_sha256"],
            "actor_changed":old["actor_final_sha256"]!=old["actor_initial_sha256"],
            "critic_changed":old["critic_final_sha256"]!=old["critic_initial_sha256"],
            "optimizer_nonempty":{k:bool(v["state"]) for k,v in c["optimizers"].items()},
            "replayed_cost_max_abs_error":max(errors),"records":len(errors),
            "finite":finite(c),"updates":old["updates"],"physical_steps":old["physical_steps"],
            "all_outer_done":bool(np.asarray(c["replay"]["done"]).all())}


def zero_error_probe():
    env=make("point_mass",max_steps=20)
    obs,_=env.reset(seed=41,options={"state":np.zeros(4)})
    result=rollout_metric(env,torch.eye(4),obs,certificate_weight=.1,decay=.05,save_trajectory=True)
    env.close()
    return {"actual":result,"document_formula_penalty":0.,
            "is_formula_mismatch":result["certificate_penalty"]!=0.}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"runs/result_audit_20261022")
    args=parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    original={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/"runs").rglob("*") if p.is_file() and output not in p.parents}
    sources={str(p.relative_to(ROOT)):digest(p) for base in ("stabilization-gym","stabilization-learning-control") for p in (ROOT/base).rglob("*.py")}
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"root":str(ROOT),
            "scope":"Saved toolkit examples; CPU audit only; no reselection or overwrite", "source_sha256":sources,"ppo":{}}
    runs=["point_mass_demo","learned_phi_metric_smoke_v1"]+[f"platform_smoke_v2/{p.name}" for p in sorted((ROOT/"runs/platform_smoke_v2").iterdir()) if p.is_dir()]
    for run in runs:
        report["ppo"][run]=audit_ppo(run)
        (output/"AUDIT.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({"audited":run,"best_update":report["ppo"][run]["best_selected_update"],
                          "replay_exact":report["ppo"][run]["in_memory_training_replay"]["history"]["exact"]}),flush=True)
    point=report["ppo"]["point_mass_demo"]
    report["independent_point_mass"]={}
    for key in ("best_test","identity_test","final_test_diagnostic_only"):
        c=torch.load(ROOT/"runs/point_mass_demo/last.pt",map_location="cpu",weights_only=False)
        learner=construct(c["protocol"])
        learner.actor.load_state_dict(c["actor"] if key=="final_test_diagnostic_only" else c["best_actor"])
        M=learner.actor(torch.zeros(4)).detach().numpy().astype(float)
        report["independent_point_mass"][key]=independent_point_mass(point[key]["episodes"],M,250,.05,.1)
        learner.env.close()
    report["transfer"]={
        "push_unseen_shape":audit_transfer("platform_smoke_v2/push_any_shape","unseen_shape_evaluation.json"),
        "visual_unseen_texture":audit_transfer("learned_phi_metric_smoke_v1","unseen_evaluation.json")}
    report["representation"]=audit_representation()
    report["slc"]=audit_slc()
    report["zero_error_probe"]=zero_error_probe()
    report["preservation"]={"original_files":len(original),
        "changed_original_files":[p for p,h in original.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h],
        "changed_source_files":[p for p,h in sources.items() if digest(ROOT/p)!=h]}
    report["complete"]=True
    (output/"AUDIT.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    (output/"ORIGINAL_FILES.json").write_text(json.dumps(original,indent=2),encoding="utf-8")
    print(json.dumps({"complete":True,"report":str(output/"AUDIT.json"),"preservation":report["preservation"]}),flush=True)


if __name__=="__main__":
    main()
