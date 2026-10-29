"""Random-origin PPO for unified pushing/time objectives, with explicit provenance.

The calibrated time bank is environment/model data, never a policy warm start.
Train uses only fixed training/validation shape splits; held-out testing requires
an external all-models-frozen barrier. No existing experimental source is edited.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.distributions import Categorical
from stabilization_gym.tasks_v2.push import JointPushCosts, JointPushEnv, vendor_modules


@dataclass(frozen=True)
class PushTrainConfig:
    seed: int = 71
    updates: int = 1000
    num_envs: int = 32
    horizon: int = 64
    epochs: int = 4
    minibatch: int = 256
    learning_rate: float = 3e-4
    eval_every: int = 20
    validation_repeats: int = 4
    device: str = "cpu"
    train_shapes: tuple = ("T", "RectWide", "L", "Plus")
    validation_shapes: tuple = ("Square", "UShape")
    test_shapes: tuple = ("OffsetComb3", "AsymThreeBranch")

    def __post_init__(self):
        for name in ("train_shapes", "validation_shapes", "test_shapes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("updates", "num_envs", "horizon", "epochs", "minibatch", "eval_every", "validation_repeats"):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"Invalid {name}")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        splits = [set(self.train_shapes), set(self.validation_shapes), set(self.test_shapes)]
        if any(not s for s in splits) or any(splits[i] & splits[j] for i in range(3) for j in range(i)):
            raise ValueError("Nonempty, disjoint train/validation/test shape splits required")


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Events:
    """Unique stage owner is established by exclusive creation of its output dir."""
    def __init__(self, root):
        self.root=Path(root);(self.root/"events").mkdir(exist_ok=False)
        self.sequence=0;self.protocol_hash=_sha(self.root/"protocol.json")

    def emit(self, kind, **data):
        self.sequence+=1
        path=self.root/"events"/f"{self.sequence:08d}_{kind}.json"
        if path.exists():raise FileExistsError("Committed events are immutable")
        _json(path,dict(sequence=self.sequence,kind=kind,unix=time.time(),protocol_sha256=self.protocol_hash,data=data))


def _tensor_hash(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_fingerprints():
    modules=["stabilization_gym.tasks_v2.push", "so101_timeaware.environment", "so101_timeaware.policy",
             "so101_timeaware.training", "so101_timeaware.timing", "so101_multishape",
             "so101_campaign", "so101_contact_demo", "dl_push.certificate.full_m_control"]
    paths=[Path(__file__).resolve()]+[Path(importlib.import_module(name).__file__).resolve() for name in modules]
    return {str(path):_sha(path) for path in paths}


def advantages(rewards, values, next_values, terminated, truncated, dt, cfg):
    """Bootstrap truncations, but never propagate GAE across episode resets."""
    discount = cfg.gamma ** (dt/cfg.discount_reference_seconds)
    # Lambda is the estimator's per-decision trace, not the physical-time objective discount.
    trace_discount = cfg.gae_lambda
    adv = torch.zeros_like(rewards); last = torch.zeros_like(rewards[0])
    for t in range(len(rewards)-1, -1, -1):
        bootstrap = (~terminated[t]).to(rewards.dtype)
        continuation = (~(terminated[t] | truncated[t])).to(rewards.dtype)
        delta = rewards[t] + discount[t]*next_values[t]*bootstrap-values[t]
        last = delta+discount[t]*trace_discount*continuation*last
        adv[t] = last
    return adv, adv+values


def collect(actor, critic, model, envs, horizon, guard=lambda: None):
    rows, episodes = [], []
    actor.eval(); critic.eval(); reached = vendor_modules()["reached"]
    for _ in range(horizon):
        guard()
        data = model.batch([e.observation() for e in envs])
        cand = model.candidates(actor, data)
        distribution = Categorical(logits=cand["logits"].detach()); indices = distribution.sample()
        with torch.no_grad():
            values = critic(data, model)
        rewards, terms, truncs, durations, policy_mask, successors = [], [], [], [], [], []
        finished = []
        for i, env in enumerate(envs):
            at_goal = reached(env.plant.spec, env.plant.q)
            physical = bool(cand["physical"][i].any())
            policy_mask.append(physical and not at_goal)
            # Goal holding is real plant time and takes precedence over push feasibility.
            result = env.step(int(indices[i])) if physical or at_goal else env.abort()
            next_obs, reward, term, trunc, info = result
            successors.append(next_obs)
            rewards.append(reward); terms.append(term); truncs.append(trunc); durations.append(info["dt"])
            if term or trunc:
                episodes.append(dict(shape=env.plant.name, success=info["success"], reason=info["reason"],
                    seconds=env.plant.elapsed, reward=env.total_reward,
                    breakdown=dict(env.total_breakdown), switches=env.plant.switches))
                finished.append(env)
        # Evaluate the actual successor BEFORE reset, including truncation successors.
        with torch.no_grad():
            next_values = critic(model.batch(successors), model)
        for env in finished:
            env.reset()
        rows.append(dict(data={k:v.detach() for k,v in data.items()}, index=indices,
            logp=distribution.log_prob(indices).detach(), value=values, next_value=next_values,
            reward=torch.as_tensor(rewards,device=model.device)*model.cfg.reward_scale,
            terminated=torch.as_tensor(terms,device=model.device),
            truncated=torch.as_tensor(truncs,device=model.device),
            dt=torch.as_tensor(durations,device=model.device),
            policy=torch.as_tensor(policy_mask,device=model.device)))
    result = {k:torch.stack([r[k] for r in rows]) for k in rows[0] if k != "data"}
    result["data"] = {k:torch.stack([r["data"][k] for r in rows]) for k in rows[0]["data"]}
    result["advantage"],result["return"] = advantages(result["reward"],result["value"],result["next_value"],
        result["terminated"],result["truncated"],result["dt"],model.cfg)
    return result, episodes


@torch.no_grad()
def probe(actor, model, names):
    ids = [model.bank.names.index(n) for n in names]
    # Three fixed physical states for every training shape, never validation/test probes.
    q = torch.tensor([[.01,-.008,.2],[-.006,.004,-.25],[.002,.002,.05]],device=model.device)
    D = model.coordinates(q)[None].expand(len(ids),-1,-1)
    condition = actor.encoder(model.C_encoded[ids],model.valid[ids],ell=model.cfg.length_scale)
    return actor.conditioned(D,condition)[1].cpu().numpy()


def evaluate(actor, model, names, seeds, costs, guard=lambda: None):
    """Deterministic greedy counterpart of the training distribution, on supplied cases."""
    rows = []; reached = vendor_modules()["reached"]
    for name in names:
        for seed in seeds:
            env = JointPushEnv(model.bank,[name],int(seed),costs)
            env.reset(shape=name)
            initial = env.plant.q.copy(); trace=[]
            while not env.plant.done:
                guard()
                data = model.batch([env.observation()]); cand = model.candidates(actor,data)
                index = int(cand["logits"][0].detach().argmax())
                physical = bool(cand["physical"][0].any())
                before = float(cand["V"][0].detach()); q = env.plant.q.copy()
                error=cand["D"][0].detach().cpu().numpy().astype(float)
                matrix=cand["M"][0].detach().cpu().numpy().astype(float)
                result = env.step(index) if physical or reached(env.plant.spec,env.plant.q) else env.abort()
                obs,reward,term,trunc,info = result
                with torch.no_grad():
                    condition = actor.encoder(model.C_encoded[[obs["shape"]]],model.valid[[obs["shape"]]],ell=model.cfg.length_scale)
                    error_next=model.coordinates(torch.as_tensor(env.plant.q[None],device=model.device,dtype=torch.float32))
                    _,matrix_next=actor.conditioned(error_next,condition)
                    following = actor.value(error_next,condition)
                en=error_next[0].cpu().numpy().astype(float);mn=matrix_next[0].cpu().numpy().astype(float)
                numpy_v=float(error@matrix@error);numpy_vn=float(en@mn@en)
                arithmetic=max(abs(numpy_v-before),abs(numpy_vn-float(following[0])))
                if not np.isclose(numpy_v,before,atol=1e-6,rtol=2e-5) or not np.isclose(numpy_vn,float(following[0]),atol=1e-6,rtol=2e-5):
                    raise RuntimeError("Complete successor V arithmetic mismatch")
                trace.append(dict(q=q.tolist(),q_next=env.plant.q.tolist(),V=before,V_next=float(following[0]),
                                  error=error.tolist(),error_next=en.tolist(),M=matrix.tolist(),M_next=mn.tolist(),
                                  numpy_V_max_difference=arithmetic,action=index,reward=reward,**info))
            rows.append(dict(shape=name,seed=int(seed),initial=initial.tolist(),success=info["success"],
                reason=info["reason"],seconds=env.plant.elapsed,reward=env.total_reward,
                error_integral=sum(t["integrated_error"] for t in trace),switches=env.plant.switches,
                input_square_integral=sum(t["input_square_integral"] for t in trace),trace=trace))
    return dict(episodes=rows,success_rate=float(np.mean([r["success"] for r in rows])),
                mean_combined_cost=-float(np.mean([r["reward"] for r in rows])),
                mean_seconds=float(np.mean([r["seconds"] for r in rows])),
                mean_error=float(np.mean([r["error_integral"] for r in rows])),
                numpy_V_max_difference=max(t["numpy_V_max_difference"] for r in rows for t in r["trace"]))


def train(time_bank, output, config=None, costs=None, *, guard=lambda: None, runtime_limits=None):
    cfg, costs = config or PushTrainConfig(), costs or JointPushCosts()
    out=Path(output)
    if out.exists():
        raise FileExistsError("Use a fresh directory; completed/partial runs are never overwritten")
    vendor=vendor_modules(); bank=vendor["Bank"](time_bank)
    if set(cfg.train_shapes+cfg.validation_shapes+cfg.test_shapes)-set(bank.names):
        raise ValueError("Declared split has shapes absent from the supplied calibrated bank")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.set_num_threads(1)
    if str(cfg.device).startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
    actor=vendor["Actor"](vendor["ActorConfig"]()).to(cfg.device)
    critic=vendor["Critic"](actor.cfg.hidden).to(cfg.device)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=cfg.learning_rate)
    assert not optimizer.state
    model=vendor["Model"](bank,cfg.device)
    envs=[JointPushEnv(bank,cfg.train_shapes,cfg.seed*10000+i,costs) for i in range(cfg.num_envs)]
    out.mkdir(parents=True)
    protocol=dict(schema="joint_push_v2",config=asdict(cfg),costs=asdict(costs),time_bank=str(Path(time_bank).resolve()),
        time_bank_sha256=bank.sha256,initialization="random FullMActor and critic; empty optimizer; no policy checkpoint reads",
        scope=JointPushEnv.scope,normalization="task time divided by bank time_reference; physical error uses bank length_scale",
        fixed_speed_input_note="push_speed^2 times push duration; collinear with duration, not electrical energy",
        budget_note="updates times num_envs times horizon transitions; not the former full-episode PPO budget",
        discount=dict(gamma=bank.config.gamma,reference_seconds=bank.config.discount_reference_seconds,
                      rule="gamma ** (actual_dt/reference_seconds); gamma=1 gives undiscounted finite task cost",
                      gae_lambda=bank.config.gae_lambda,gae_trace="per-decision estimator trace"),
        deadlines="task max_seconds and max_actions terminate as failures; external cutoffs truncate and bootstrap",
        selection="validation success first, then complete combined cost; updated checkpoints only",
        source_sha256=source_fingerprints(),runtime_limits=runtime_limits)
    _json(out/"protocol.json",protocol)
    events=_Events(out)
    committed_protocol_hash=events.protocol_hash
    outer_guard=guard
    def guard():
        outer_guard()
        if _sha(out/"protocol.json")!=committed_protocol_hash:
            raise RuntimeError("Frozen training protocol changed")
        if (out/"STOP").exists():raise RuntimeError("STOP present")
    initial_hash=_tensor_hash(actor); initial_probe=probe(actor,model,cfg.train_shapes); previous=initial_probe
    probes=[]; history=[]; best_key=None; best_step=None
    def save(name, update):
        state=dict(schema="joint_push_v2",actor=actor.state_dict(),cfg=asdict(actor.cfg),critic=critic.state_dict(),
            optimizer=optimizer.state_dict(),update=update,environments=[e.state_dict() for e in envs],
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(cfg.device).startswith("cuda") else [],
            initial_actor_hash=initial_hash,protocol_sha256=_sha(out/"protocol.json"),time_bank_sha256=bank.sha256,
            best_step=best_step,best_key=best_key,origin="random")
        temporary=out/(name+".tmp");torch.save(state,temporary);temporary.replace(out/name)
    from so101_timeaware.training import update as ppo_update
    save("initial.pt",0)
    events.emit("initialized",seed=cfg.seed,initial_actor_hash=initial_hash,optimizer_state_empty=True)
    val_seeds=[cfg.seed*100000+8000+i for i in range(cfg.validation_repeats)]
    for step in range(cfg.updates+1):
        guard()
        if step:
            rollout,episodes=collect(actor,critic,model,envs,cfg.horizon,guard)
            stats=ppo_update(actor,critic,optimizer,model,rollout,cfg.epochs,cfg.minibatch)
            stats.update(update=step,simulated_seconds=float(rollout["dt"].sum()),
                         transitions=cfg.num_envs*cfg.horizon,completed_episodes=len(episodes),episodes=episodes)
            history.append(stats)
        value=probe(actor,model,cfg.train_shapes)
        probes.append(dict(update=step,M=value.tolist(),distance_from_initial=float(np.linalg.norm(value-initial_probe)),
                           step_change=float(np.linalg.norm(value-previous))))
        previous=value
        if step==0 or step%cfg.eval_every==0 or step==cfg.updates:
            rng=torch.get_rng_state(); report=evaluate(actor,model,cfg.validation_shapes,val_seeds,costs,guard)
            torch.set_rng_state(rng)
            _json(out/"validation"/f"update_{step:05d}.json",report)
            key=(report["success_rate"],-report["mean_combined_cost"])
            if step>0 and (best_key is None or key>best_key):
                best_key,best_step=key,step;save("selected.pt",step)
            events.emit("validation_complete",update=step,selected_update=best_step,
                success_rate=report["success_rate"],mean_combined_cost=report["mean_combined_cost"],
                mean_seconds=report["mean_seconds"],numpy_V_max_difference=report["numpy_V_max_difference"])
        if step:
            save("last.pt",step)
            _json(out/"TRAINING.json",history);_json(out/"M_PROBES.json",probes)
            events.emit("update_complete",**{k:v for k,v in stats.items() if k!="episodes"},
                M_distance_from_initial=probes[-1]["distance_from_initial"],M_step_change=probes[-1]["step_change"])
    save("final.pt",cfg.updates)
    for path,digest in protocol["source_sha256"].items():
        if _sha(path)!=digest:raise RuntimeError("Scientific source changed during training")
    if _sha(bank.path)!=bank.sha256:raise RuntimeError("Calibrated time bank changed during training")
    changed=_tensor_hash(actor)!=initial_hash
    if not changed:
        raise RuntimeError("No actor update occurred; refusing to freeze as trained")
    frozen=dict(schema="joint_push_v2",checkpoint="selected.pt",sha256=_sha(out/"selected.pt"),
                selected_update=best_step,updates=cfg.updates,initial_actor_hash=initial_hash,
                final_actor_hash=_tensor_hash(actor),time_bank_sha256=bank.sha256,
                protocol_sha256=_sha(out/"protocol.json"),test_started=False,
                checkpoint_sha256={name:_sha(out/name) for name in ("initial.pt","selected.pt","final.pt")})
    _json(out/"FROZEN.json",frozen)
    events.emit("selection_frozen",**frozen)
    _json(out/"FINAL_STATE.json",dict(status="complete",updates=cfg.updates,selected_update=best_step,testing_started=False))
    return frozen


def freeze_group(run_dirs, output, expected_seeds):
    """Commit an all-seed selection barrier, before any held-out test runs."""
    target=Path(output)
    if target.exists():
        raise FileExistsError("Freeze barriers are immutable; use a new path")
    rows=[];seen=[];common=None
    for directory in run_dirs:
        root=Path(directory).resolve()
        frozen=json.loads((root/"FROZEN.json").read_text(encoding="utf-8"))
        protocol=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
        if frozen.get("schema")!="joint_push_v2" or frozen.get("selected_update",0)<1:
            raise ValueError("Every run must have an updated, completed selection")
        if _sha(root/"protocol.json") != frozen["protocol_sha256"]:
            raise ValueError("Protocol changed since training froze")
        if frozen["updates"] != protocol["config"]["updates"]:
            raise ValueError("Incomplete training budget")
        for name,digest in frozen["checkpoint_sha256"].items():
            if _sha(root/name)!=digest:raise ValueError("Frozen checkpoint changed")
        comparison=copy.deepcopy(protocol)
        comparison["config"].pop("seed")
        comparison.pop("runtime_limits",None)  # Execution timestamps are not scientific hyperparameters.
        if common is None:common=comparison
        elif common!=comparison:raise ValueError("Seed group has incompatible protocols or assets")
        seen.append(protocol["config"]["seed"])
        rows.append(dict(path=str(root),frozen_sha256=_sha(root/"FROZEN.json"),
                         selected_sha256=frozen["sha256"],protocol_sha256=frozen["protocol_sha256"]))
    if not rows or len(set(seen))!=len(seen) or sorted(seen)!=sorted(expected_seeds):
        raise ValueError("Freeze must contain exactly the declared unique training seeds")
    result=dict(schema="joint_push_group_frozen_v2",expected_seeds=list(expected_seeds),runs=rows)
    _json(target,result)
    return result


def test_frozen(run_dir, group_manifest, output=None, repeats=4, *, guard=lambda:None):
    """Held-out tests only after all declared seeds freeze; never selects a model."""
    root=Path(run_dir).resolve();barrier_path=Path(group_manifest).resolve()
    barrier=json.loads(barrier_path.read_text(encoding="utf-8"))
    if barrier.get("schema")!="joint_push_group_frozen_v2":raise ValueError("Missing all-seed freeze barrier")
    if not isinstance(repeats,int) or isinstance(repeats,bool) or repeats<1:raise ValueError("Positive repeats required")
    matched=False;seeds=[]
    for record in barrier["runs"]:
        member=Path(record["path"]).resolve()
        if _sha(member/"FROZEN.json")!=record["frozen_sha256"] or _sha(member/"selected.pt")!=record["selected_sha256"]:
            raise ValueError("Group member changed after freeze")
        if _sha(member/"protocol.json")!=record["protocol_sha256"]:raise ValueError("Group protocol changed")
        seeds.append(json.loads((member/"protocol.json").read_text(encoding="utf-8"))["config"]["seed"])
        matched |= member==root
    if not matched or len(set(seeds))!=len(seeds) or sorted(seeds)!=sorted(barrier["expected_seeds"]):
        raise ValueError("Run absent from complete seed barrier")
    protocol=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    frozen=json.loads((root/"FROZEN.json").read_text(encoding="utf-8"))
    for path,digest in protocol["source_sha256"].items():
        if _sha(path)!=digest:raise ValueError("Frozen scientific source changed")
    bank_path=Path(protocol["time_bank"])
    if _sha(bank_path)!=protocol["time_bank_sha256"]:raise ValueError("Frozen time bank changed")
    target=Path(output) if output else root/"test"
    if target.exists():raise FileExistsError("Existing/partial tests are preserved")
    target.mkdir(parents=True)
    _json(target/"protocol.json",dict(group_manifest=str(barrier_path),group_sha256=_sha(barrier_path),
        heldout_shapes=protocol["config"]["test_shapes"],repeats=repeats,selection_allowed=False,
        cases="same starts across methods and training seeds; repeated seeds are not independent test starts"))
    events=_Events(target)
    vendor=vendor_modules();bank=vendor["Bank"](bank_path)
    model=vendor["Model"](bank,protocol["config"]["device"]);costs=JointPushCosts(**protocol["costs"])
    reports={};test_seeds=[990000+i for i in range(repeats)]
    for method,name in (("random_initial","initial.pt"),("selected","selected.pt"),("final","final.pt"),("identity","initial.pt")):
        guard()
        if _sha(root/name)!=frozen["checkpoint_sha256"][name]:raise ValueError("Frozen local checkpoint changed")
        # Only this newly trained run's hash-verified snapshots are accepted.
        ck=torch.load(root/name,map_location=model.device,weights_only=False)
        actor=vendor["Actor"](vendor["ActorConfig"](**ck["cfg"])).to(model.device)
        actor.load_state_dict(ck["actor"])
        if method=="identity":
            def identity(D,condition):
                eye=torch.eye(3,device=D.device,dtype=D.dtype).expand(*D.shape[:-1],3,3)
                return eye,eye
            actor.conditioned=identity
        report=evaluate(actor,model,protocol["config"]["test_shapes"],test_seeds,costs,guard)
        _json(target/(method+".json"),report)
        reports[method]={k:v for k,v in report.items() if k!="episodes"}
        events.emit("method_complete",method=method,**reports[method])
    _json(target/"SUMMARY.json",reports)
    _json(target/"FINAL_STATE.json",dict(status="complete",group_sha256=_sha(barrier_path),selection_allowed=False))
    events.emit("test_complete",methods=list(reports),selection_allowed=False)
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-bank");parser.add_argument("--output",required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare",action="store_true",help="Fresh actuated calibration into output (explicit opt-in)")
    modes.add_argument("--freeze-group",nargs="+",metavar="RUN",help="Completed seed run directories")
    modes.add_argument("--test",metavar="RUN",help="Test this frozen run after the all-seed barrier")
    parser.add_argument("--group-manifest");parser.add_argument("--expected-seeds",type=int,nargs="+")
    parser.add_argument("--repeats",type=int,default=4);parser.add_argument("--timing-pairs",type=int,default=4)
    parser.add_argument("--max-hours",type=float,default=14.,help="One original process wall-clock cap; no budget extension")
    parser.add_argument("--config",help="JSON fields of PushTrainConfig; includes fixed shape splits")
    parser.add_argument("--costs",help="JSON fields of JointPushCosts")
    args=parser.parse_args()
    if not np.isfinite(args.max_hours) or args.max_hours<=0:parser.error("max-hours must be positive and finite")
    started=time.time();deadline=time.monotonic()+args.max_hours*3600
    stop_paths=[Path(args.output)/"STOP"]
    if args.test:stop_paths.append(Path(args.test)/"STOP")
    def guard():
        if time.monotonic()>=deadline:raise RuntimeError("Original wall-clock budget reached")
        if any(path.exists() for path in stop_paths):raise RuntimeError("STOP present")
    if args.prepare:
        from stabilization_gym.tasks_v2.push import prepare_time_bank
        print(prepare_time_bank(args.output,repeats=args.repeats,timing_pairs=args.timing_pairs,guard=guard));return
    if args.freeze_group:
        if not args.expected_seeds:parser.error("freeze-group requires expected-seeds")
        guard();print(json.dumps(freeze_group(args.freeze_group,args.output,args.expected_seeds),indent=2));return
    if args.test:
        if not args.group_manifest:parser.error("test requires group-manifest")
        print(json.dumps(test_frozen(args.test,args.group_manifest,args.output,args.repeats,guard=guard),indent=2));return
    if not args.time_bank:parser.error("Training requires an explicitly supplied time-bank")
    fields=json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    for key in ("train_shapes","validation_shapes","test_shapes"):
        if key in fields:fields[key]=tuple(fields[key])
    costs=JointPushCosts(**json.loads(Path(args.costs).read_text(encoding="utf-8"))) if args.costs else JointPushCosts()
    print(json.dumps(train(args.time_bank,args.output,PushTrainConfig(**fields),costs,guard=guard,
        runtime_limits=dict(started_unix=started,deadline_unix=started+args.max_hours*3600,max_hours=args.max_hours)),indent=2))


if __name__=="__main__":
    main()
