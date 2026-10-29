"""Small AE + physical-derivative representation-training example.

Run from the repository after ``pip install -e .``::

    python examples/representation_training.py --updates 40

This uses the new portable downward-camera texture renderer, not the historical
indoor benchmark. Privileged poses construct labels only; the derivative head
receives RGB features and commanded physical velocity. The encoder is trained
offline, between episodes. A short example is not evidence of improved servo
success or a Lyapunov proof.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from stabilization_learning.envs.visual_servo import VisualServoEnv
from stabilization_learning.representations import (
    DerivativeHead, RGBAutoEncoder, RepresentationLearner,
)


def derivative_label(state, goal, velocity):
    """Exact instantaneous P0 dE/dt, in metres/seconds and radians/seconds."""
    delta = state-goal
    yaw_error = math.atan2(math.sin(delta[3]), math.cos(delta[3]))
    cosine, sine = math.cos(state[3]), math.sin(state[3])
    coefficient = np.array([
        2*(delta[0]*cosine+delta[1]*sine),
        2*(-delta[0]*sine+delta[1]*cosine),
        2*delta[2], 2*yaw_error/math.radians(25.)**2,
    ])
    return float(coefficient @ velocity)


def physical_error(state, goal):
    delta = state-goal
    yaw_error = math.atan2(math.sin(delta[3]), math.cos(delta[3]))
    return float(delta[:3]@delta[:3]+(yaw_error/math.radians(25.))**2)


def make_dataset(count, *, split, image_size, seed):
    rng = np.random.default_rng(seed)
    images, goals, velocities, targets = [], [], [], []
    finite_difference_errors = []
    for sample in range(count):
        # Distinct split seeds produce genuinely different synthetic textures.
        env = VisualServoEnv(image_size=image_size, split=split, scene_id=sample % 4)
        goal = np.r_[rng.uniform(-.1, .1, 2), rng.uniform(1.8, 2.2), rng.uniform(-.2, .2)]
        state = goal + np.r_[rng.uniform(-.25, .25, 3), rng.uniform(-.25, .25)]
        observation, _ = env.reset(seed=seed+sample, options={"state": state, "goal": goal})
        velocity = np.r_[rng.uniform(-.2, .2, 3), rng.uniform(-.25, .25)]
        target = derivative_label(state, goal, velocity)
        # Independent central finite difference through the actual public P0
        # transition with a proportionally tiny commanded displacement.
        epsilon = 1e-5
        plus = env.predict_next(torch.from_numpy(state),
                                torch.from_numpy(velocity)*(epsilon/env.dt)).numpy()
        minus = env.predict_next(torch.from_numpy(state),
                                 torch.from_numpy(velocity)*(-epsilon/env.dt)).numpy()
        finite_difference = (physical_error(plus, goal)-physical_error(minus, goal))/(2*epsilon)
        finite_difference_errors.append(abs(finite_difference-target))
        images.append(observation["rgb"].transpose(2, 0, 1).copy())
        goals.append(observation["goal_rgb"].transpose(2, 0, 1).copy())
        velocities.append(velocity)
        targets.append(target)
        env.close()
    data = {
        "images": torch.from_numpy(np.stack(images)).float()/255.,
        "goals": torch.from_numpy(np.stack(goals)).float()/255.,
        "velocities": torch.tensor(np.stack(velocities), dtype=torch.float32),
        "targets": torch.tensor(targets, dtype=torch.float32),
    }
    return data, float(max(finite_difference_errors))


@torch.no_grad()
def evaluate(autoencoder, head, data):
    z, zg = autoencoder(data["images"]), autoencoder(data["goals"])
    prediction = head(z, zg, data["velocities"])
    target = data["targets"]
    valid = target.abs() > 1e-5
    sign_accuracy = ((prediction[valid]*target[valid]) > 0).float().mean()
    return {
        "reconstruction_mse": float((autoencoder.decoder(z)-data["images"]).square().mean()),
        "derivative_rmse": float((prediction-target).square().mean().sqrt()),
        "derivative_sign_accuracy": float(sign_accuracy) if valid.any() else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path, default=Path("runs/representation_demo"))
    args = parser.parse_args(argv)
    if min(args.updates, args.train_samples, args.val_samples, args.batch_size) < 1:
        parser.error("Updates, sample counts and batch size must be positive")
    if args.image_size < 16 or args.image_size % 8:
        parser.error("image-size must be at least 16 and divisible by 8")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train, train_fd = make_dataset(args.train_samples, split="train", image_size=args.image_size, seed=args.seed)
    validation, val_fd = make_dataset(args.val_samples, split="val", image_size=args.image_size, seed=args.seed+1000)
    if max(train_fd, val_fd) > 1e-5:
        raise RuntimeError("Analytic derivative labels failed the independent physical finite-difference check")
    autoencoder = RGBAutoEncoder(latent_dim=16, image_size=args.image_size)
    head = DerivativeHead(latent_dim=16, action_dim=4)
    learner = RepresentationLearner(autoencoder, head, lr=1e-3, auxiliary_cap=.25)
    initial = evaluate(autoencoder, head, validation)
    history = []
    for update in range(args.updates):
        indices = torch.randint(args.train_samples, (min(args.batch_size, args.train_samples),))
        # Reconstruct current and goal RGB with the same shared encoder/decoder.
        images = torch.cat((train["images"][indices], train["goals"][indices]))
        goals = torch.cat((train["goals"][indices], train["goals"][indices]))
        velocities = torch.cat((train["velocities"][indices], train["velocities"][indices]))
        targets = torch.cat((train["targets"][indices], torch.zeros(indices.shape[0])))
        stats = learner.update(images, goals, velocities, targets)
        stats["update"] = update+1
        history.append(stats)
    final = evaluate(autoencoder, head, validation)
    result = {
        "scope": "Small offline representation example on a new synthetic texture renderer; no servo-success claim",
        "label": "P0 analytic d/dt [position_error_m_squared + (yaw_error_rad / radians(25))^2]",
        "head_inputs": ["phi(current RGB)", "phi(goal RGB)", "physical body velocity"],
        "privileged_pose_usage": "Labels and independent checks only; never encoder/head inputs",
        "seed": args.seed, "updates": args.updates,
        "train_samples": args.train_samples, "validation_samples": args.val_samples,
        "validation_split": "Different synthetic texture seeds from the same renderer family",
        "derivative_label_max_finite_difference_error": max(train_fd, val_fd),
        "initial_validation": initial, "final_validation": final,
        "maximum_effective_auxiliary_encoder_gradient_ratio": max(row["effective_aux_ratio"] for row in history),
        "gradient_cap": .25,
        "limitation": "The cap controls parameter-gradient norm before Adam, not the optimizer step norm",
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "RESULTS.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"autoencoder": autoencoder.state_dict(), "derivative_head": head.state_dict(),
                "image_size": args.image_size, "latent_dim": 16,
                "scope": result["scope"]}, args.output / "representation.pt")
    printable = {key: value for key, value in result.items() if key != "history"}
    printable["report"] = str(destination.resolve())
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
