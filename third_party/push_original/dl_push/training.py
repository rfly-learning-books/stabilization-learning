"""Supervised sequence replay through a stabilizing observer's internal updates."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import os
import random
import tempfile

import numpy as np
import torch

from .config import ControlConfig, ObserverConfig
from .control import task_energy, task_energy_torch
from .data import split_episodes
from .observer import Memory, ResidualObserver
from .runtime import code_hashes, sha256, write_json
from .state import RESIDUAL_SCALE, STATE_NAMES


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Small observer networks are faster without spawning a large BLAS thread pool.
    torch.set_num_threads(1)


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, config, optimizer=None, step=0, history=None, data_paths=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format="dl_push_observer_v1", observer=asdict(model.settings),
        model=model.state_dict(), model_hash=model_hash(model), step=step,
        config=config.to_dict(), normalization=RESIDUAL_SCALE.tolist(),
        optimizer=None if optimizer is None else optimizer.state_dict(),
        history=history or [], data_paths=[str(p) for p in data_paths or []],
        code_hashes=code_hashes(),
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        torch.save(payload, stream)
        temp = Path(stream.name)
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "dl_push_observer_v1":
        raise ValueError("Not a DL-Push observer checkpoint")
    if payload["normalization"] != RESIDUAL_SCALE.tolist():
        raise ValueError("Residual normalization differs from the trained model")
    model = ResidualObserver(ObserverConfig(**payload["observer"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    if model_hash(model) != payload["model_hash"]:
        raise ValueError("Corrupt observer checkpoint")
    return model, payload


def upgrade_checkpoint_model(path, observer_config, device):
    """Add a zero-initialized static residual head while preserving old behavior."""
    old, payload = load_checkpoint(path, device)
    old_settings, new_settings = asdict(old.settings), asdict(observer_config)
    permitted = {"static_residual_head", "static_residual_limit"}
    if any(old_settings[key] != new_settings[key]
           for key in old_settings if key not in permitted):
        raise ValueError("Observer upgrade may only add the bounded static residual head")
    if old.settings.static_residual_head or not observer_config.static_residual_head:
        raise ValueError("Requested observer change is not a supported one-way upgrade")
    model = ResidualObserver(observer_config).to(device)
    state = model.state_dict()
    for key, value in old.state_dict().items():
        if key in state and state[key].shape == value.shape:
            state[key] = value
    model.load_state_dict(state, strict=True)
    payload = dict(payload)
    payload["optimizer"] = None
    payload["upgraded_from_model_hash"] = payload["model_hash"]
    return model, payload


def convert_checkpoint_to_constant_gain(path, observer_config, device):
    """Freeze the analytic adaptation gain while preserving learned representations."""
    old, payload = load_checkpoint(path, device)
    old_settings, new_settings = asdict(old.settings), asdict(observer_config)
    permitted = {"gain", "eta"}
    if any(old_settings[key] != new_settings[key]
           for key in old_settings if key not in permitted):
        raise ValueError("Gain conversion may only change gain mode and constant eta")
    if old.settings.gain != "learned" or observer_config.gain != "constant":
        raise ValueError("Only learned-to-constant gain conversion is supported")
    model = ResidualObserver(observer_config).to(device)
    state = model.state_dict()
    for key, value in old.state_dict().items():
        if key in state and state[key].shape == value.shape:
            state[key] = value
    model.load_state_dict(state, strict=True)
    payload = dict(payload)
    payload["optimizer"] = None
    payload["converted_from_model_hash"] = payload["model_hash"]
    return model, payload


def replay_episode(model, episode, settings, training=False):
    """Replay from W0; score before adapting. No stored hidden states or stale innovations."""
    device = next(model.parameters()).device
    arrays = episode.arrays
    ctx = torch.as_tensor(arrays["contexts"], device=device)
    actions = torch.as_tensor(arrays["actions"], device=device)
    nominal = torch.as_tensor(arrays["nominal"], device=device)
    real_next = torch.as_tensor(arrays["states"][1:], device=device)
    target = (real_next - nominal) / model.residual_scale
    grid = torch.as_tensor(episode.geometry.grid[None], device=device)
    dims = torch.as_tensor(episode.geometry.dimensions[None], device=device)
    embedding = model.encode_geometry(grid, dims)
    memory = model.initial()
    errors, normalized_losses, etas = [], [], []
    for t in range(episode.length):
        if training and t and t % settings.tbptt_steps == 0:
            # State is reconstructed with current psi, then only gradients are truncated.
            memory = memory.detach()
        prediction, phi = model.predict(ctx[t:t + 1], embedding, memory)
        error = prediction - target[t:t + 1]
        errors.append(error[0] * model.residual_scale)
        normalized_losses.append(error.square().mean())
        memory, eta = model.feedback(memory, phi, error, target[t:t + 1], actions[t:t + 1])
        etas.append(eta[0, 0])
    losses = torch.stack(normalized_losses)
    contact = torch.as_tensor(arrays["contacts"] > 0, device=device)
    weights = torch.where(contact, losses.new_tensor(settings.contact_weight), losses.new_tensor(1.))
    main_loss = (weights * losses).sum() / weights.sum()
    regularizer = losses.new_zeros(())
    m = settings.lk_window
    if settings.lk_weight and len(losses) >= 2 * m:
        averages = losses.unfold(0, m, 1).mean(-1)
        # Disjoint windows, reference detached; empirical only, not a certificate.
        violation = (averages[m:] - settings.lk_contraction * averages[:-m].detach()
                     - settings.lk_tolerance).relu()
        regularizer = violation.square().mean()
    loss = main_loss + settings.lk_weight * regularizer
    predicted_residual = torch.stack([
        error / model.residual_scale + target[index]
        for index, error in enumerate(errors)
    ])
    details = dict(error=torch.stack(errors), nominal_error=nominal - real_next,
                   normalized_loss=losses, contact=contact, eta=torch.stack(etas),
                   window_regularizer=regularizer,
                   predicted_residual=predicted_residual, target=target)
    return loss, details


def replay_batch(model, episodes, settings, control=None, return_components=False):
    """Batch independent trajectories; preserve causal adaptation along each time axis."""
    device = next(model.parameters()).device
    batch, time_steps = len(episodes), max(ep.length for ep in episodes)
    contexts = np.zeros((batch, time_steps, episodes[0].arrays["contexts"].shape[-1]), np.float32)
    targets = np.zeros((batch, time_steps, 12), np.float32)
    actions = np.zeros((batch, time_steps, 2), np.float32)
    states = np.zeros((batch, time_steps + 1, 12), np.float32)
    nominal = np.zeros((batch, time_steps, 12), np.float32)
    valid = np.zeros((batch, time_steps), bool)
    contacts = np.zeros((batch, time_steps), bool)
    for row, ep in enumerate(episodes):
        n = ep.length
        contexts[row, :n] = ep.arrays["contexts"]
        targets[row, :n] = (ep.arrays["states"][1:] - ep.arrays["nominal"]) / RESIDUAL_SCALE
        actions[row, :n] = ep.arrays["actions"]
        states[row, :n + 1] = ep.arrays["states"]
        nominal[row, :n] = ep.arrays["nominal"]
        valid[row, :n] = True
        contacts[row, :n] = ep.arrays["contacts"] > 0
    contexts, targets, actions, states, nominal = [
        torch.as_tensor(a, device=device)
        for a in (contexts, targets, actions, states, nominal)
    ]
    mask = torch.as_tensor(valid, device=device)
    contact = torch.as_tensor(contacts, device=device)
    grid = torch.as_tensor(np.stack([ep.geometry.grid for ep in episodes]), device=device)
    dims = torch.as_tensor(np.stack([ep.geometry.dimensions for ep in episodes]), device=device)
    embedding = model.encode_geometry(grid, dims)
    expanded_embedding = embedding[:, None].expand(-1, time_steps, -1)
    phi = model.features(contexts, expanded_embedding)
    baseline = model.base_residual(contexts, expanded_embedding)
    memory = model.initial(batch)
    loss_rows, predictions, dissipation_rows = [], [], []
    diagnostic_truth = None
    if settings.dissipation_weight:
        count = mask.sum(-1).clamp_min(1).to(phi.dtype)
        weighted_phi = phi * mask[..., None]
        gram = torch.einsum(
            "btr,bts->brs", weighted_phi, phi
        ) / count[:, None, None]
        cross = torch.einsum(
            "btr,btn->brn", weighted_phi, targets - baseline
        ) / count[:, None, None]
        identity = torch.eye(
            model.settings.feature_dim, dtype=phi.dtype, device=device
        )[None]
        solution = torch.linalg.solve(
            gram + settings.realizability_ridge * identity, cross
        )
        diagnostic_truth = solution.transpose(-2, -1)
        truth_norm = torch.linalg.vector_norm(
            diagnostic_truth, dim=(-2, -1), keepdim=True
        ).clamp_min(1e-12)
        diagnostic_truth = (
            diagnostic_truth
            * (model.settings.radius / truth_norm).clamp(max=1.)
        ).detach()
    for t in range(time_steps):
        if t and t % settings.tbptt_steps == 0:
            memory = memory.detach()
        prediction = baseline[:, t] + torch.einsum(
            "bnr,br->bn", memory.weights, phi[:, t]
        )
        predictions.append(prediction)
        error = prediction - targets[:, t]
        loss_rows.append(error.square().mean(-1))
        updated, _ = model.feedback(memory, phi[:, t], error, targets[:, t], actions[:, t])
        active = mask[:, t]
        if diagnostic_truth is not None:
            before = (memory.weights - diagnostic_truth).square().sum((-2, -1))
            after = (updated.weights - diagnostic_truth).square().sum((-2, -1))
            supply = (
                after - before
                + model.dissipation_coefficient * error.square().sum(-1)
            )
            # Relative parameter-energy supply prevents a few large-U episodes
            # from dominating while keeping the desired sign explicit.
            dissipation_rows.append(
                (supply / (1 + before.detach())).relu().square()
            )
        memory = Memory(
            torch.where(active[:, None, None], updated.weights, memory.weights),
            torch.where(active[:, None], updated.hidden, memory.hidden),
        )
    losses = torch.stack(loss_rows, dim=1)
    predicted_residual = torch.stack(predictions, dim=1)
    weights = torch.where(contact, losses.new_tensor(settings.contact_weight), losses.new_tensor(1.)) * mask
    state_per_episode = (weights * losses).sum(-1) / weights.sum(-1)
    per_episode = state_per_episode
    if settings.lk_weight:
        terms = []
        m = settings.lk_window
        for row, ep in enumerate(episodes):
            if ep.length >= 2 * m:
                avg = losses[row, :ep.length].unfold(0, m, 1).mean(-1)
                terms.append((avg[m:] - settings.lk_contraction * avg[:-m].detach()
                              - settings.lk_tolerance).relu().square().mean())
            else:
                terms.append(losses.new_zeros(()))
        per_episode = per_episode + settings.lk_weight * torch.stack(terms)
    state_loss = state_per_episode.mean()
    lyapunov_loss = state_loss.new_zeros(())
    clf_boundary_loss = state_loss.new_zeros(())
    realizability_loss = state_loss.new_zeros(())
    realizability_norm_penalty = state_loss.new_zeros(())
    dissipation_loss = state_loss.new_zeros(())
    if settings.lyapunov_difference_weight or settings.clf_boundary_weight:
        if control is None:
            raise ValueError("Joint observer loss requires the task-energy control configuration")
        predicted_state = nominal + predicted_residual * model.residual_scale
        goals = torch.as_tensor(
            np.stack([ep.metadata["goal"] for ep in episodes]), device=device
        )
        periods = torch.as_tensor(
            [ep.metadata["orientation_period"] for ep in episodes], device=device
        )
        predicted_energy = task_energy_torch(predicted_state, goals, periods, control)
        real_energy = task_energy_torch(states[:, 1:], goals, periods, control)
        current_energy = task_energy_torch(states[:, :-1], goals, periods, control)
        indices = torch.arange(time_steps, device=device) - control.window + 1
        oldest = task_energy_torch(
            states[:, indices.clamp_min(0)], goals, periods, control
        )
        history_mask = mask & (indices >= 0)[None]
        history_weights = torch.where(
            contact, losses.new_tensor(settings.contact_weight), losses.new_tensor(1.)
        ) * history_mask
        true_difference = real_energy - oldest
        predicted_difference = predicted_energy - oldest
        difference_error = (
            (predicted_difference - true_difference)
            / (settings.lyapunov_difference_scale + true_difference.abs())
        ).square()
        true_constraint = (
            (1 + control.delta) * real_energy - oldest
            + control.alpha * current_energy
            + control.beta * actions.square().sum(-1)
        )
        predicted_constraint = (
            (1 + control.delta) * predicted_energy - oldest
            + control.alpha * current_energy
            + control.beta * actions.square().sum(-1)
        )
        boundary_weight = (
            settings.clf_boundary_epsilon
            / (true_constraint.abs() + settings.clf_boundary_epsilon)
        )
        boundary_error = boundary_weight * (
            (predicted_constraint - true_constraint) / settings.clf_error_scale
        ).square()
        denominator = history_weights.sum(-1).clamp_min(1.)
        lyapunov_loss = (
            (history_weights * difference_error).sum(-1) / denominator
        ).mean()
        clf_boundary_loss = (
            (history_weights * boundary_error).sum(-1) / denominator
        ).mean()
        per_episode = (
            per_episode
            + settings.lyapunov_difference_weight
            * (history_weights * difference_error).sum(-1) / denominator
            + settings.clf_boundary_weight
            * (history_weights * boundary_error).sum(-1) / denominator
        )
    if settings.realizability_weight:
        adaptation_target = targets - baseline
        identity = torch.eye(
            model.settings.feature_dim, dtype=phi.dtype, device=device
        )[None]
        fit_terms, norm_terms = [], []
        time_index = torch.arange(time_steps, device=device)
        for parity in (0, 1):
            support = mask & ((time_index % 2) == parity)[None]
            query = mask & ~support
            count = support.sum(-1).clamp_min(1).to(phi.dtype)
            weighted_phi = phi * support[..., None]
            gram = torch.einsum(
                "btr,bts->brs", weighted_phi, phi
            ) / count[:, None, None]
            cross = torch.einsum(
                "btr,btn->brn", weighted_phi, adaptation_target
            ) / count[:, None, None]
            solution = torch.linalg.solve(
                gram + settings.realizability_ridge * identity, cross
            )
            query_prediction = torch.einsum("btr,brn->btn", phi, solution)
            query_error = (query_prediction - adaptation_target).square().mean(-1)
            query_count = query.sum(-1).clamp_min(1)
            fit_terms.append(
                (query_error * query).sum(-1) / query_count
            )
            norm = torch.linalg.vector_norm(solution, dim=(-2, -1))
            norm_terms.append(
                ((norm / model.settings.radius) - 1).relu().square()
            )
        fit_per_episode = torch.stack(fit_terms).mean(0)
        norm_per_episode = torch.stack(norm_terms).mean(0)
        realizability_loss = fit_per_episode.mean()
        realizability_norm_penalty = norm_per_episode.mean()
        per_episode = (
            per_episode
            + settings.realizability_weight * fit_per_episode
            + settings.realizability_weight
            * settings.realizability_norm_weight * norm_per_episode
        )
    if settings.dissipation_weight:
        supply_loss = torch.stack(dissipation_rows, dim=1)
        denominator = weights.sum(-1).clamp_min(1.)
        dissipation_per_episode = (
            (weights * supply_loss).sum(-1) / denominator
        )
        dissipation_loss = dissipation_per_episode.mean()
        per_episode = (
            per_episode
            + settings.dissipation_weight * dissipation_per_episode
        )
    total = per_episode.mean()
    components = {
        "total": total,
        "state": state_loss,
        "lyapunov_difference": lyapunov_loss,
        "clf_boundary": clf_boundary_loss,
        "realizability": realizability_loss,
        "realizability_norm": realizability_norm_penalty,
        "dissipation": dissipation_loss,
    }
    return (total, components) if return_components else total


@torch.no_grad()
def evaluate_prediction(model, episodes, settings):
    model.eval()
    errors, nominal, contacts, losses = [], [], [], []
    by_shape = defaultdict(list)
    nominal_by_shape = defaultdict(list)
    for episode in episodes:
        loss, info = replay_episode(model, episode, settings)
        errors.append(info["error"].cpu().numpy())
        nominal.append(info["nominal_error"].cpu().numpy())
        contacts.append(info["contact"].cpu().numpy())
        losses.append(float(loss))
        by_shape[episode.metadata["shape"]].append(float(loss))
        normalized_nominal = (info["nominal_error"] / model.residual_scale).square().mean(-1)
        weight = torch.where(info["contact"], normalized_nominal.new_tensor(settings.contact_weight),
                             normalized_nominal.new_tensor(1.))
        nominal_by_shape[episode.metadata["shape"]].append(
            float((weight * normalized_nominal).sum() / weight.sum()))
    e, e0, mask = np.concatenate(errors), np.concatenate(nominal), np.concatenate(contacts)
    def rmse(a):
        return {name: float(v) for name, v in zip(STATE_NAMES, np.sqrt(np.mean(a ** 2, axis=0)))}
    shape_losses = {name: float(np.mean(values)) for name, values in by_shape.items()}
    return dict(
        episodes=len(episodes), transitions=len(e), contact_transitions=int(mask.sum()),
        normalized_loss=float(np.mean(list(shape_losses.values()))),
        nominal_normalized_prediction_loss=float(np.mean([np.mean(v) for v in nominal_by_shape.values()])),
        per_shape_loss=shape_losses, physical_rmse=rmse(e), nominal_physical_rmse=rmse(e0),
        contact_physical_rmse=rmse(e[mask]) if mask.any() else None,
        contact_nominal_physical_rmse=rmse(e0[mask]) if mask.any() else None,
        prediction_p95_abs={name: float(v) for name, v in zip(STATE_NAMES, np.quantile(np.abs(e), .95, axis=0))},
    )


@torch.no_grad()
def evaluate_stability_prediction(model, episodes, settings, control):
    """Evaluate executed-action D/CLF prediction; this is not candidate coverage."""
    model.eval()
    difference_errors, true_differences, predicted_differences = [], [], []
    predicted_constraints, true_constraints = [], []
    energy_relative, sample_weights = [], []
    for episode in episodes:
        _, info = replay_episode(model, episode, settings)
        predicted_state = (
            episode.arrays["nominal"]
            + info["predicted_residual"].cpu().numpy() * RESIDUAL_SCALE
        )
        states = episode.arrays["states"]
        actions = episode.arrays["actions"]
        goal = np.asarray(episode.metadata["goal"])
        period = episode.metadata["orientation_period"]
        predicted_energy = task_energy(predicted_state, goal, period, control)
        real_energy = task_energy(states[1:], goal, period, control)
        current_energy = task_energy(states[:-1], goal, period, control)
        for t in range(control.window - 1, episode.length):
            oldest = float(task_energy(
                states[t - control.window + 1], goal, period, control
            ))
            true_difference = float(real_energy[t] - oldest)
            predicted_difference = float(predicted_energy[t] - oldest)
            true_constraint = (
                (1 + control.delta) * float(real_energy[t]) - oldest
                + control.alpha * float(current_energy[t])
                + control.beta * float(actions[t] @ actions[t])
            )
            predicted_constraint = (
                (1 + control.delta) * float(predicted_energy[t]) - oldest
                + control.alpha * float(current_energy[t])
                + control.beta * float(actions[t] @ actions[t])
            )
            difference_errors.append(predicted_difference - true_difference)
            true_differences.append(true_difference)
            predicted_differences.append(predicted_difference)
            predicted_constraints.append(predicted_constraint)
            true_constraints.append(true_constraint)
            energy_relative.append(
                abs(float(predicted_energy[t] - real_energy[t]))
                / (abs(float(real_energy[t])) + 1e-8)
            )
            sample_weights.append(
                settings.contact_weight if episode.arrays["contacts"][t] > 0 else 1.
            )
    error = np.asarray(difference_errors)
    difference = np.asarray(true_differences)
    predicted_difference = np.asarray(predicted_differences)
    predicted_c = np.asarray(predicted_constraints)
    true_c = np.asarray(true_constraints)
    weights = np.asarray(sample_weights)
    difference_loss = np.average(
        (error / (settings.lyapunov_difference_scale + np.abs(difference))) ** 2,
        weights=weights,
    )
    boundary = settings.clf_boundary_epsilon / (
        np.abs(true_c) + settings.clf_boundary_epsilon
    )
    boundary_loss = np.average(
        boundary * ((predicted_c - true_c) / settings.clf_error_scale) ** 2,
        weights=weights,
    )
    actual_feasible = true_c <= 0
    predicted_feasible = predicted_c <= 0
    false_reject = actual_feasible & ~predicted_feasible
    false_accept = ~actual_feasible & predicted_feasible
    relative = np.asarray(energy_relative)
    return {
        "samples": len(error),
        "lyapunov_difference_loss": float(difference_loss),
        "clf_boundary_loss": float(boundary_loss),
        "lyapunov_difference_rmse": float(np.sqrt(np.mean(error ** 2))),
        "energy_relative_median_percent": float(100 * np.median(relative)),
        "energy_relative_p95_percent": float(100 * np.quantile(relative, .95)),
        "difference_sign_error_rate": float(np.mean(
            np.sign(predicted_difference) != np.sign(difference)
        )),
        "clf_classification_error_rate": float(np.mean(
            predicted_feasible != actual_feasible
        )),
        "actual_feasible": int(actual_feasible.sum()),
        "predicted_feasible": int(predicted_feasible.sum()),
        "false_rejections": int(false_reject.sum()),
        "false_acceptances": int(false_accept.sum()),
        "false_rejection_rate": float(
            false_reject.sum() / max(1, actual_feasible.sum())
        ),
        "false_acceptance_rate": float(
            false_accept.sum() / max(1, (~actual_feasible).sum())
        ),
    }


@torch.no_grad()
def evaluate_training_objective(model, episodes, settings, control):
    model.eval()
    _, components = replay_batch(
        model, episodes, settings, control, return_components=True
    )
    return {name: float(value) for name, value in components.items()}


def fit(config, data_roots, output, initial_checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    training, validation = split_episodes(data_roots, config)
    if initial_checkpoint:
        model, payload = load_checkpoint(initial_checkpoint, config.device)
        if asdict(model.settings) != asdict(config.observer):
            if config.training.allow_observer_upgrade:
                model, payload = upgrade_checkpoint_model(
                    initial_checkpoint, config.observer, config.device
                )
            else:
                raise ValueError("Checkpoint observer structure differs from config")
        step = int(payload["step"])
    else:
        model, payload, step = ResidualObserver(config.observer).to(config.device), {}, 0
    if config.training.trainable_modules != ["all"]:
        model.requires_grad_(False)
        for name in config.training.trainable_modules:
            module = getattr(model, name, None)
            if module is None:
                raise ValueError(f"Observer does not contain trainable module {name}")
            module.requires_grad_(True)
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No observer parameters selected for training")
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=config.training.learning_rate
    )
    if payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config.training.learning_rate
    groups = defaultdict(list)
    for ep in training:
        groups[ep.metadata["shape"]].append(ep)
    names = sorted(groups)
    rng = np.random.default_rng(config.seed + step)
    data_paths = [ep.path for ep in training + validation]
    write_json(output / "config.json", config.to_dict())
    write_json(output / "data_manifest.json", {
        "training": [str(ep.path) for ep in training],
        "validation": [str(ep.path) for ep in validation],
        "metadata_hashes": {str(p): sha256(p) for p in data_paths},
        "test_episodes_used": False,
    })
    initial = evaluate_prediction(model, validation, config.training)
    initial["stability_prediction"] = evaluate_stability_prediction(
        model, validation, config.training, config.control
    )
    initial["training_objective"] = evaluate_training_objective(
        model, validation, config.training, config.control
    )
    initial["joint_selection_score"] = (
        initial["training_objective"]["total"]
    )
    best_loss = (
        initial["joint_selection_score"]
        if config.training.selection_metric == "joint"
        else initial["normalized_loss"]
    )
    history = []
    # Keep the untouched initial model as a legitimate baseline/best candidate.
    save_checkpoint(output / "best.pt", model, config, optimizer, step, history, data_paths)
    write_json(output / "initial_validation.json", initial)
    before = model_hash(model)
    for update in range(config.training.updates_per_iteration):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scalar_loss = 0.
        sampled = []
        for _ in range(config.training.batch_episodes):
            shape = names[int(rng.integers(len(names)))]
            ep = groups[shape][int(rng.integers(len(groups[shape])))]
            sampled.append(ep)
        if config.training.vectorized_replay:
            loss, components = replay_batch(
                model, sampled, config.training, config.control,
                return_components=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite sequence loss; retain data for diagnosis")
            loss.backward()
            scalar_loss = float(loss.detach())
        else:
            for ep in sampled:
                loss, _ = replay_episode(model, ep, config.training, training=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite sequence loss")
                (loss / config.training.batch_episodes).backward()
                scalar_loss += float(loss.detach()) / config.training.batch_episodes
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            trainable_parameters, config.training.gradient_clip
        ))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("Non-finite observer gradient")
        optimizer.step()
        step += 1
        row = dict(
            step=step, loss=scalar_loss, gradient_norm=grad_norm,
            loss_components={
                name: float(value.detach())
                for name, value in components.items()
            } if config.training.vectorized_replay else {"total": scalar_loss},
        )
        if (update + 1) % max(1, config.training.updates_per_iteration // 4) == 0 or update == 0:
            result = evaluate_prediction(model, validation, config.training)
            result["stability_prediction"] = evaluate_stability_prediction(
                model, validation, config.training, config.control
            )
            result["training_objective"] = evaluate_training_objective(
                model, validation, config.training, config.control
            )
            result["joint_selection_score"] = (
                result["training_objective"]["total"]
            )
            row["validation"] = result
            candidate_loss = (
                result["joint_selection_score"]
                if config.training.selection_metric == "joint"
                else result["normalized_loss"]
            )
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                save_checkpoint(output / "best.pt", model, config, optimizer, step, history + [row], data_paths)
            stability = result["stability_prediction"]
            print(
                f"update={step} loss={scalar_loss:.6g} "
                f"val_x={result['normalized_loss']:.6g} "
                f"val_D={stability['lyapunov_difference_loss']:.6g} "
                f"val_C={stability['clf_boundary_loss']:.6g} "
                f"grad={grad_norm:.5g}",
                flush=True,
            )
        history.append(row)
        write_json(output / "history.json", history)
    save_checkpoint(output / "last.pt", model, config, optimizer, step, history, data_paths)
    best, _ = load_checkpoint(output / "best.pt", config.device)
    final = evaluate_prediction(best, validation, config.training)
    final["stability_prediction"] = evaluate_stability_prediction(
        best, validation, config.training, config.control
    )
    final["training_objective"] = evaluate_training_objective(
        best, validation, config.training, config.control
    )
    final["joint_selection_score"] = (
        final["training_objective"]["total"]
    )
    report = dict(initial_validation=initial, best_validation=final,
                  parameters_changed=before != model_hash(model), optimizer_steps=len(history),
                  learned_parameter_count=sum(p.numel() for p in model.parameters()),
                  trained_parameter_count=sum(p.numel() for p in trainable_parameters),
                  trainable_modules=config.training.trainable_modules,
                  best_checkpoint=str(output / "best.pt"), last_checkpoint=str(output / "last.pt"),
                  selection_metric=config.training.selection_metric,
                  best_selection_value=best_loss,
                  theoretical_assumptions_verified_on_PhysX=False,
                  note="Validation prediction metrics, not a closed-loop convergence result")
    write_json(output / "report.json", report)
    return report
