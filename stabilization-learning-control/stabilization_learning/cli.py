from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from . import make, available_envs
from .trainer import StabilizationLearner, TrainConfig
from .common.compat import canonical_protocol


def main():
    parser = argparse.ArgumentParser(description="Unified certificate-metric learning")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    train = sub.add_parser("train")
    train.add_argument("--env", choices=available_envs())
    train.add_argument("--config", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--episodes", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-updates", type=int)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--seeds", type=int, nargs="+", default=[20001, 20002, 20003, 20004])
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--evaluation-config", type=Path,
                          help="JSON with env_kwargs overrides for held-out scenes/shapes or plants")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "list":
        print("\n".join(available_envs()))
        return
    if args.command == "train":
        cfg = {} if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
        env_id = args.env or cfg.get("env", "point_mass")
        kwargs, training = cfg.get("env_kwargs", {}), cfg.get("training", {})
        if args.episodes is not None:
            training["episodes"] = args.episodes
        if "validation_seeds" in training:
            training["validation_seeds"] = tuple(training["validation_seeds"])
        config = TrainConfig(**training)
        env = make(env_id, **kwargs)
        learner = StabilizationLearner(env, config=config, env_id=env_id, env_kwargs=kwargs)
        learner.train(args.output, resume=args.resume, max_updates=args.max_updates)
        print(json.dumps({"output": str(args.output.resolve()), "episodes": learner.episodes,
                          "physical_steps": learner.physical_steps, "validation_objective": learner.best_score}))
        env.close()
    else:
        checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
        protocol = checkpoint["protocol"]
        env = make(protocol["env_id"], **protocol["env_kwargs"])
        learner = StabilizationLearner(env, config=TrainConfig(**protocol["config"]),
                                       env_id=protocol["env_id"], env_kwargs=protocol["env_kwargs"])
        if canonical_protocol(learner._protocol()) != canonical_protocol(protocol):
            raise ValueError("Evaluation assets or representation differ from training")
        training_seeds = set(range(learner.config.seed, learner.config.seed + learner.config.episodes))
        if training_seeds.intersection(args.seeds) or set(learner.config.validation_seeds).intersection(args.seeds):
            raise ValueError("Test seeds overlap train/validation; choose disjoint test seeds")
        evaluation_kwargs = protocol["env_kwargs"].copy()
        if args.evaluation_config is not None:
            evaluation_kwargs.update(json.loads(args.evaluation_config.read_text(encoding="utf-8"))["env_kwargs"])
            env.close()
            env = make(protocol["env_id"], **evaluation_kwargs)
            candidate = StabilizationLearner(env, config=learner.config,
                env_id=protocol["env_id"], env_kwargs=evaluation_kwargs)
            if (candidate.env.metric_spec != learner.env.metric_spec or
                candidate.env.metric_context_dim != learner.env.metric_context_dim or
                candidate.representation_hash != learner.representation_hash):
                raise ValueError("Evaluation must preserve phi and the metric/context coordinates")
            learner = candidate
        learner.actor.load_state_dict(checkpoint["actor"])
        learned = learner.evaluate(args.seeds, trajectories=True)
        identity = learner.evaluate(args.seeds, identity=True, trajectories=True)
        results = {"test_seeds": args.seeds, "evaluation_env_kwargs": evaluation_kwargs,
                   "learned": learned, "identity": identity,
                   "paired_cost_delta": [a["fixed_horizon_cost"] - b["fixed_horizon_cost"]
                       for a, b in zip(learned["episodes"], identity["episodes"])],
                   "scope": "explicit evaluation configuration; same starts paired between learned M and identity; no model reselection"}
        path = args.output or args.run / "evaluation.json"
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(path.resolve()), "learned": learned["summary"], "identity": identity["summary"]}))
        env.close()


if __name__ == "__main__":
    main()
