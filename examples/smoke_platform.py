"""Short integration training, not a performance or generalization experiment."""
import argparse
import json
from pathlib import Path
import torch
from stabilization_learning import make, available_envs, StabilizationLearner, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/platform_smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    results = {}
    for env_id in available_envs():
        kwargs = {"max_steps": 12}
        env = make(env_id, **kwargs)
        config = TrainConfig(episodes=4, batch_episodes=4, epochs=2, validation_seeds=(10001, 10002))
        learner = StabilizationLearner(env, config=config, env_id=env_id, env_kwargs=kwargs)
        initial = torch.cat([p.detach().flatten().clone() for p in learner.actor.parameters()])
        learner.train(args.output / env_id)
        final = torch.cat([p.detach().flatten() for p in learner.actor.parameters()])
        evaluation = learner.evaluate((20001, 20002), best=True)
        results[env_id] = {"physical_steps": learner.physical_steps, "updates": learner.updates,
                           "actor_changed": bool(torch.any(initial != final)),
                           "evaluation": evaluation["summary"],
                           "integration_only": True}
        if not results[env_id]["actor_changed"]:
            raise AssertionError(f"Policy parameters did not update in {env_id}")
        env.close()
        print(env_id, "OK", flush=True)
    (args.output / "RESULTS.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
