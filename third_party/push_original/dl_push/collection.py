"""Real trajectories for observer learning and sampled-CLF evaluation."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math
import os
import time
import traceback

import numpy as np
import torch

from .control import FiniteMemoryCLF, metrics, sample_candidates
from .data import save_episode, save_npz
from .environment import PushWorld
from .evaluation import convergence_report
from .geometry import prepare_shape, shape_specs
from .observer import ResidualObserver
from .phases import CEMPlanner, PhaseController
from .runtime import activate_reference, code_hashes, write_json
from .state import CONTEXT_DIM, DT, corrected_state, model_context, residual_target
from .training import load_checkpoint, model_hash, save_checkpoint, seed_everything


def _make_trace(state, obs):
    return dict(states=[state.copy()], obs17=[obs.copy()], actions=[], nominal=[],
                contexts=[], contacts=[], predictions=[], phases=[], eta=[],
                candidates=[], candidate_predictions=[], candidate_constraints=[],
                chosen=[], admitted=[], multistep_admitted=[],
                recovery_step=[],
                multistep_constraint=[], multistep_horizon=[],
                planning_seconds=[], slack=[], slack_used=[],
                slack_remaining=[], rho=[], storage=[],
                terminated=[], truncated=[], loop_seconds=[], dl_hold_count=0,
                dl_success=False, environment_success=False, outcome="running", last_attempt=None)


def _arrays(trace, count):
    n = len(trace["actions"])
    fields = {
        "states": (n + 1, 12), "obs17": (n + 1, 17), "actions": (n, 2),
        "nominal": (n, 12), "contexts": (n, CONTEXT_DIM), "contacts": (n,),
        "predictions": (n, 12), "eta": (n,), "candidates": (n, count, 2),
        "candidate_predictions": (n, count, 12), "candidate_constraints": (n, count),
        "chosen": (n,), "admitted": (n,), "multistep_admitted": (n,),
        "recovery_step": (n,),
        "multistep_constraint": (n,), "multistep_horizon": (n,),
        "planning_seconds": (n,), "slack": (n,), "rho": (n,),
        "slack_used": (n,), "slack_remaining": (n,),
        "storage": (n,), "terminated": (n,), "truncated": (n,), "loop_seconds": (n,),
    }
    result = {key: np.asarray(trace[key], dtype=np.float32).reshape(shape) for key, shape in fields.items()}
    result["phases"] = np.asarray(trace["phases"], dtype="U12")
    return result


@torch.no_grad()
def collect(config, output, split="train", checkpoint=None, behavior="prior", correction=True, shape_subset=None):
    """Collect independent per-lane episodes. Completed lanes never supply replay labels."""
    if split not in {"train", "validation", "test"} or behavior not in {"prior", "clf", "phase"}:
        raise ValueError("Unknown collection split or behavior")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    activate_reference(config.mjx_backend)
    from starter.priors import load_priors, verify_checkpoints
    from starter.mjx_one_step import OneStepMJX
    from dl_push.contact import ContactOneStepMJX

    specs = shape_specs(config.path(config.shape_catalog) if config.shape_catalog else None)
    names = getattr(config, f"{split}_shapes")
    if not names:
        raise ValueError(f"No {split} shapes declared; configure an independent pool explicitly")
    missing = set(names) - specs.keys()
    if missing:
        raise ValueError(f"Unknown shapes: {missing}")
    selected_shapes = set(names if shape_subset is None else shape_subset)
    if not selected_shapes or selected_shapes - set(names):
        raise ValueError("Requested shape subset is outside the declared split")
    if checkpoint:
        model, _ = load_checkpoint(checkpoint, config.device)
        if asdict(model.settings) != asdict(config.observer):
            raise ValueError("Collector config must match checkpoint observer architecture")
    else:
        model = ResidualObserver(config.observer).to(config.device)
    model.eval()
    save_checkpoint(output / "collector_model.pt", model, config)
    initial_model_hash = model_hash(model)
    reference_hashes = verify_checkpoints()
    planned = [dict(shape=name, episode=i, status="pending") for name in names if name in selected_shapes
               for i in range(config.training.episodes_per_shape)]
    manifest = dict(schema=1, config=config.to_dict(), config_hash=config.fingerprint,
                    split=split, behavior=behavior, correction=correction,
                    shared_model_hash=initial_model_hash, prior_hashes=reference_hashes,
                    code_hashes=code_hashes(), planned=planned, results=[],
                    test_generalization_claim=False)
    write_json(output / "collection.json", manifest)
    lookup = {(row["shape"], row["episode"]): row for row in planned}
    bank = load_priors(config.device)
    for shape_number, name in enumerate(names):
        if name not in selected_shapes:
            continue
        geometry = prepare_shape(specs[name], config.path(config.asset_root))
        predictor = (
            ContactOneStepMJX(geometry.xml, config.control.candidate_count)
            if behavior == "phase"
            else OneStepMJX(geometry.xml, config.control.candidate_count)
        )
        sequence_predictor = None
        if config.control.multistep_horizon > 1:
            from .contact import ContactPairwiseOneStepMJX
            from .planning import PairwiseOneStepMJX
            planner_batch = (
                config.control.multistep_random_sequences
                if config.control.multistep_proposal == "random"
                else len(bank.actors)
                * (1 + config.control.multistep_samples_per_prior)
            )
            sequence_class = (
                ContactPairwiseOneStepMJX
                if behavior == "phase"
                else PairwiseOneStepMJX
            )
            sequence_predictor = sequence_class(
                geometry.xml, planner_batch,
            )
        expected_backend = "gpu" if config.mjx_backend == "cuda" else "cpu"
        if predictor.backend != expected_backend:
            raise RuntimeError("MJX is not on the explicitly requested backend")
        if sequence_predictor is not None and sequence_predictor.backend != expected_backend:
            raise RuntimeError("Finite-step MJX is not on the explicitly requested backend")
        embedding = model.encode_geometry(
            torch.as_tensor(geometry.grid[None], device=config.device),
            torch.as_tensor(geometry.dimensions[None], device=config.device),
        )
        groups = math.ceil(config.training.episodes_per_shape / config.environment.num_envs)
        for group in range(groups):
            ids = list(range(group * config.environment.num_envs,
                             min((group + 1) * config.environment.num_envs, config.training.episodes_per_shape)))
            seed = config.seed + shape_number * 100003 + group * 1009
            domain_rng = np.random.default_rng(seed)
            mass = float(domain_rng.choice(config.environment.masses))
            friction = float(domain_rng.choice(config.environment.frictions))
            world, traces = None, None
            for i in ids:
                lookup[(name, i)]["status"] = "running"
            write_json(output / "collection.json", manifest)
            try:
                world = PushWorld(specs[name], config.environment, mass, friction)
                obs, states, _ = world.reset(seed)
                traces = [_make_trace(states[lane], obs[lane].cpu().numpy()) for lane in range(len(ids))]
                memories = [model.initial() for _ in ids]
                filters = [FiniteMemoryCLF(config.control, world.goal, geometry.period, geometry,
                                           device=config.device) for _ in ids]
                phase_controllers = None
                if behavior == "phase":
                    phase_controllers = [
                        PhaseController(config.control, world.goal,
                                        geometry.period, geometry, config.device)
                        for _ in ids
                    ]
                generators = [np.random.default_rng(seed + lane * 97) for lane in range(len(ids))]
                for lane in range(len(ids)):
                    filters[lane].observe(states[lane])
                    if phase_controllers is not None:
                        phase_controllers[lane].observe(states[lane], 0.0)
                clf_started = [False for _ in ids]
                recovery_used = [0 for _ in ids]
                recovery_active = [False for _ in ids]
                recovery_last_end = [-config.control.stagnation_cooldown_steps for _ in ids]
                recovery_target_local = [None for _ in ids]
                for t in range(config.environment.episode_steps):
                    started = time.perf_counter()
                    executed = np.zeros((config.environment.num_envs, 2), np.float32)
                    pending = {}
                    for lane, trace in enumerate(traces):
                        if trace["outcome"] != "running":
                            continue
                        state, rng, clf = states[lane], generators[lane], filters[lane]
                        actions = sample_candidates(bank, obs[lane].to(config.device), rng, config.control,
                                                    geometry, state, world.goal)
                        recorded_phase = None
                        if behavior == "prior":
                            choices = 6 if config.training.zero_action_blocks else 5
                            expert = ((t // config.training.behavior_switch_steps) + ids[lane]) % choices
                            selected = expert if expert < 5 else len(actions) - 1
                            if (expert < 5 and config.control.samples_per_prior
                                    and rng.random() < config.training.prior_sample_probability):
                                selected = 5 + expert * config.control.samples_per_prior + int(rng.integers(config.control.samples_per_prior))
                            phase = "prior"
                        else:
                            if behavior == "phase":
                                phase_controller = phase_controllers[lane]
                                current_phase = phase_controller.current_phase
                                if current_phase == "approach":
                                    selected = len(actions) - 2
                                    actions[selected] = geometry.approach_action(state, world.goal)
                                    phase = "startup"
                                elif not clf.ready:
                                    selected = len(actions) - 2
                                    actions[selected] = geometry.approach_action(
                                        state, world.goal
                                    )
                                    phase = "startup"
                                else:
                                    clf_started[lane] = True
                                    phase = "clf"
                                    selected = None
                            else:
                                recorded_phase = None
                                if (not config.control.start_when_feasible and clf.ready
                                        and geometry.contact_gap(state) <= config.control.contact_gap):
                                    clf_started[lane] = True
                                if not clf_started[lane]:
                                    if t >= config.control.warmup_steps:
                                        trace["outcome"] = "approach_timeout"
                                        continue
                                    selected = len(actions) - 2
                                    actions[selected] = geometry.approach_action(
                                        state, world.goal
                                    )
                                    phase = "startup"
                                else:
                                    phase, selected = "clf", None
                        if behavior == "phase":
                            recorded_phase = phase_controller.current_phase
                        else:
                            recorded_phase = phase
                        if behavior == "phase":
                            nominal, _, candidate_contact = predictor.predict_contact(
                                state, actions
                            )
                        else:
                            nominal, _ = predictor.predict(state, actions)
                        ctx = model_context(state, actions, nominal, geometry.local(state))
                        residual, phi = model.predict(
                            torch.as_tensor(ctx, device=config.device),
                            embedding.expand(len(actions), -1), memories[lane].expand(len(actions)))
                        corrected = corrected_state(nominal, residual.cpu().numpy())
                        predictions = corrected if correction else nominal
                        admission = clf.select(actions, predictions) if clf.ready else None
                        predicted_phases = None
                        if behavior == "phase":
                            predicted_phases = phase_controller.candidate_phases(
                                predictions, candidate_contact
                            )
                        if (phase == "startup" and config.control.start_when_feasible and clf.ready
                                and geometry.contact_gap(state) <= config.control.contact_gap
                                and admission["index"] is not None):
                            clf_started[lane] = True
                            phase = "clf"
                        if phase == "clf":
                            for refinement in range(config.control.refinement_rounds):
                                # Search multiple starts without advancing physics or observer memory.
                                rank = np.where(admission["domain"], admission["residual"], np.inf)
                                if not np.isfinite(rank).any():
                                    break
                                elites = actions[np.argsort(rank)[:min(8, len(actions) - 1)]]
                                width = config.control.refinement_sigma * (.3 ** refinement)
                                offsets = rng.normal(0., width, (len(actions) - len(elites) - 1, 2))
                                centers = elites[np.arange(len(offsets)) % len(elites)]
                                actions = np.concatenate([elites, np.clip(centers + offsets, -.30, .30),
                                                          np.zeros((1, 2))]).astype(np.float32)
                                nominal, _ = predictor.predict(state, actions)
                                ctx = model_context(state, actions, nominal, geometry.local(state))
                                residual, phi = model.predict(
                                    torch.as_tensor(ctx, device=config.device),
                                    embedding.expand(len(actions), -1), memories[lane].expand(len(actions)))
                                corrected = corrected_state(nominal, residual.cpu().numpy())
                                predictions = corrected if correction else nominal
                                admission = clf.select(actions, predictions)
                        multistep_admitted = False
                        recovery_step = False
                        multistep_constraint = np.nan
                        multistep_horizon = 0
                        multistep_required_slack = 0.
                        planning_seconds = 0.
                        plan = None
                        prefix_audit = None
                        block_admission = None
                        if phase == "clf":
                            dynamic_state = (
                                np.linalg.norm(state[4:7])
                                > config.control.settle_object_speed
                                or abs(state[7])
                                > config.control.settle_object_omega
                                or (not recovery_active[lane]
                                    and np.linalg.norm(state[10:12])
                                    > config.control.settle_pusher_speed)
                            )
                            if recovery_active[lane]:
                                cosine, sine = np.cos(state[3]), np.sin(state[3])
                                rotation = np.array(
                                    [[cosine, -sine], [sine, cosine]]
                                )
                                fixed_target = (
                                    state[:2]
                                    + rotation @ recovery_target_local[lane]
                                )
                                target_error = np.linalg.norm(
                                    fixed_target - state[8:10]
                                )
                            else:
                                target_error = np.inf
                            if (recovery_active[lane]
                                    and not dynamic_state
                                    and target_error
                                    <= config.control.contact_recovery_tolerance):
                                recovery_active[lane] = False
                                recovery_last_end[lane] = t
                                recovery_target_local[lane] = None
                            force_recovery = recovery_active[lane]
                            selected = None if force_recovery else admission["index"]
                            if (selected is not None
                                    and config.control.stagnation_recovery
                                    and recovery_used[lane]
                                    < config.control.contact_recovery_steps
                                    and clf.accepted_steps
                                    >= config.control.stagnation_min_admitted_steps
                                    and clf.task_history[-1]
                                    <= config.control.stagnation_activation_energy
                                    and t - recovery_last_end[lane]
                                    >= config.control.stagnation_cooldown_steps):
                                current_metrics = metrics(
                                    state, world.goal, geometry.period
                                )
                                outside_strict_pose = (
                                    current_metrics["position_m"]
                                    > config.evaluation.position_tolerance
                                    or current_metrics["angle_rad"]
                                    > config.evaluation.angle_tolerance
                                )
                                real_window_drop = (
                                    clf.task_history[0] - clf.task_history[-1]
                                )
                                if (outside_strict_pose
                                        and real_window_drop
                                        <= config.control.stagnation_min_drop):
                                    selected = None
                                    force_recovery = True
                                    recovery_active[lane] = True
                                    target_world = geometry.approach_target(
                                        state, world.goal,
                                        config.control.contact_recovery_turn_gain,
                                    )
                                    cosine, sine = np.cos(state[3]), np.sin(state[3])
                                    rotation = np.array(
                                        [[cosine, -sine], [sine, cosine]]
                                    )
                                    recovery_target_local[lane] = (
                                        rotation.T @ (target_world - state[:2])
                                    )
                            cem_accepted = False
                            if (selected is None and not force_recovery
                                    and sequence_predictor is not None):
                                from .planning import (
                                    rollout_action_sequences,
                                    rollout_prior_means,
                                    rollout_prior_samples,
                                    sample_piecewise_sequences,
                                    select_shortest_feasible_prefix,
                                )
                                use_cem = (
                                    behavior == "phase"
                                    and config.control.phase_control_enabled
                                    and phase_controller.current_phase == "push"
                                    and hasattr(sequence_predictor, "predict_contact")
                                )
                                if use_cem:
                                    cem = CEMPlanner(
                                        sequence_predictor, model, geometry,
                                        embedding, memories[lane], world.goal,
                                        config.control, clf, rng, config.device,
                                    )
                                    cem_result = cem.plan(
                                        state,
                                        seed_actions=actions[:len(bank.actors)],
                                    )
                                    best = cem_result["best"]
                                    if best is not None:
                                        plan = dict(
                                            actions=best["actions"][None],
                                            predictions=best["predictions"][None],
                                            contact_forces=best["contact_forces"][None],
                                            contact_normals=best["contact_normals"][None],
                                            contact_points=best["contact_points"][None],
                                            contact_active=best["contact_active"][None],
                                        )
                                        planning_seconds = (
                                            best["rollout_seconds"]
                                            * int(config.control.cem_iterations)
                                        )
                                        prefix_audit = select_shortest_feasible_prefix(
                                            clf, plan,
                                            [int(config.control.cem_horizon)],
                                        )
                                        block_admission = prefix_audit["admission"]
                                        if block_admission is not None:
                                            selected = len(actions) - 2
                                            actions[selected] = plan["actions"][0, 0]
                                            nominal, _ = predictor.predict(
                                                state, actions
                                            )
                                            ctx = model_context(
                                                state, actions, nominal,
                                                geometry.local(state),
                                            )
                                            residual, phi = model.predict(
                                                torch.as_tensor(
                                                    ctx, device=config.device
                                                ),
                                                embedding.expand(
                                                    len(actions), -1
                                                ),
                                                memories[lane].expand(
                                                    len(actions)
                                                ),
                                            )
                                            corrected = corrected_state(
                                                nominal, residual.cpu().numpy()
                                            )
                                            predictions = (
                                                corrected if correction else nominal
                                            )
                                            admission = clf.select(
                                                actions, predictions
                                            )
                                            phase = "multistep"
                                            multistep_admitted = True
                                            multistep_horizon = int(
                                                config.control.cem_horizon
                                            )
                                            multistep_constraint = float(
                                                block_admission[
                                                    "robust_residual"
                                                ][0]
                                            )
                                            multistep_required_slack = float(
                                                block_admission[
                                                    "required_slack"
                                                ][0]
                                            )
                                            cem_accepted = True
                                if not cem_accepted:
                                    if config.control.multistep_proposal == "random":
                                        seeds = list(actions[:len(bank.actors)])
                                        seeds.append(
                                            geometry.approach_action(
                                                state, world.goal
                                            )
                                        )
                                        action_sequences = sample_piecewise_sequences(
                                            rng,
                                            config.control.multistep_random_sequences,
                                            config.control.multistep_horizon,
                                            config.control.multistep_random_blocks,
                                            seeds,
                                        )
                                        plan = rollout_action_sequences(
                                            sequence_predictor, model, geometry,
                                            embedding, memories[lane], state,
                                            action_sequences, config.device,
                                            config.control.multistep_correction_mode,
                                        )
                                    elif config.control.multistep_samples_per_prior:
                                        plan = rollout_prior_samples(
                                            sequence_predictor, bank, model, geometry,
                                            embedding, memories[lane], state, world.goal,
                                            config.control.multistep_horizon,
                                            config.device,
                                            config.control.multistep_samples_per_prior,
                                            rng,
                                            config.control.prior_std_scales or [1.],
                                            config.control.multistep_correction_mode,
                                        )
                                    else:
                                        plan = rollout_prior_means(
                                            sequence_predictor, bank, model, geometry,
                                            embedding, memories[lane], state, world.goal,
                                            config.control.multistep_horizon,
                                            config.device,
                                            config.control.multistep_correction_mode,
                                        )
                                    planning_seconds = plan["rollout_seconds"]
                                    if config.control.multistep_proposal == "prior":
                                        mean_indices = (
                                            np.arange(len(bank.actors))
                                            * (
                                                1
                                                + config.control.multistep_samples_per_prior
                                            )
                                        )
                                        first_difference = np.max(np.abs(
                                            plan["actions"][mean_indices, 0]
                                            - actions[:len(bank.actors)]
                                        ))
                                        if first_difference > 2e-5:
                                            raise RuntimeError(
                                                "Reconstructed prior state changed "
                                                "the first planned action"
                                            )
                                    prefix_audit = select_shortest_feasible_prefix(
                                        clf, plan, config.control.multistep_horizons
                                    )
                                    block_admission = prefix_audit["admission"]
                                    selected = (
                                        block_admission["index"]
                                        if block_admission is not None else None
                                    )
                                    if block_admission is not None:
                                        plan_index = selected
                                        phase = "multistep"
                                        multistep_admitted = True
                                        multistep_horizon = block_admission["horizon"]
                                        multistep_constraint = float(
                                            block_admission[
                                                "robust_residual"
                                            ][plan_index]
                                        )
                                        multistep_required_slack = float(
                                            block_admission[
                                                "required_slack"
                                            ][plan_index]
                                        )
                                        if (
                                            config.control.multistep_proposal
                                            == "random"
                                            or config.control.multistep_samples_per_prior
                                        ):
                                            selected = len(actions) - 2
                                            actions[selected] = (
                                                plan["actions"][plan_index, 0]
                                            )
                                            nominal, _ = predictor.predict(
                                                state, actions
                                            )
                                            ctx = model_context(
                                                state, actions, nominal,
                                                geometry.local(state),
                                            )
                                            residual, phi = model.predict(
                                                torch.as_tensor(
                                                    ctx, device=config.device
                                                ),
                                                embedding.expand(
                                                    len(actions), -1
                                                ),
                                                memories[lane].expand(
                                                    len(actions)
                                                ),
                                            )
                                            corrected = corrected_state(
                                                nominal, residual.cpu().numpy()
                                            )
                                            predictions = (
                                                corrected if correction else nominal
                                            )
                                            admission = clf.select(
                                                actions, predictions
                                            )
                            if (selected is None
                                    and recovery_used[lane]
                                    < config.control.contact_recovery_steps):
                                recovery_active[lane] = True
                                if recovery_target_local[lane] is None:
                                    target_world = geometry.approach_target(
                                        state, world.goal,
                                        config.control.contact_recovery_turn_gain,
                                    )
                                    cosine, sine = np.cos(state[3]), np.sin(state[3])
                                    rotation = np.array(
                                        [[cosine, -sine], [sine, cosine]]
                                    )
                                    recovery_target_local[lane] = (
                                        rotation.T @ (target_world - state[:2])
                                    )
                                if dynamic_state:
                                    selected = len(actions) - 1
                                    phase = "settle"
                                else:
                                    selected = len(actions) - 2
                                    actions[selected] = geometry.contact_recovery_action(
                                        state, world.goal,
                                        config.control.contact_recovery_clearance,
                                        config.control.contact_recovery_speed,
                                        config.control.contact_recovery_turn_gain,
                                        recovery_target_local[lane],
                                    )
                                    nominal, _ = predictor.predict(state, actions)
                                    ctx = model_context(
                                        state, actions, nominal, geometry.local(state)
                                    )
                                    residual, phi = model.predict(
                                        torch.as_tensor(ctx, device=config.device),
                                        embedding.expand(len(actions), -1),
                                        memories[lane].expand(len(actions)),
                                    )
                                    corrected = corrected_state(
                                        nominal, residual.cpu().numpy()
                                    )
                                    predictions = corrected if correction else nominal
                                    admission = clf.select(actions, predictions)
                                    phase = "recontact"
                                recovery_used[lane] += 1
                                recovery_step = True
                            if selected is None:
                                trace["outcome"] = "clf_infeasible"
                                attempt = dict(
                                    state=state.copy(), candidates=actions.copy(),
                                    nominal=nominal.copy(), predictions=predictions.copy(),
                                    constraints=admission["residual"].copy(),
                                    oldest_energy=np.array(clf.history[0]),
                                    current_task=np.array(clf.task_history[-1]),
                                    slack=np.array(admission["slack"]))
                                if plan is not None:
                                    attempt.update(
                                        multistep_actions=plan["actions"],
                                        multistep_predictions=plan["predictions"],
                                        multistep_horizons=np.asarray(
                                            prefix_audit["horizons"]
                                        ),
                                        multistep_constraints=np.stack([
                                            item["residual"]
                                            for item in prefix_audit["audits"]
                                        ]),
                                        multistep_robust_constraints=np.stack([
                                            item["robust_residual"]
                                            for item in prefix_audit["audits"]
                                        ]),
                                    )
                                trace["last_attempt"] = attempt
                                continue
                        admitted = phase == "clf"
                        constraints = admission["residual"] if admission else np.full(len(actions), np.nan)
                        slack = (
                            admission["slack"] if admitted
                            else block_admission["slack"] if multistep_admitted else 0.
                        )
                        slack_used = (
                            float(admission["required_slack"][selected])
                            if admitted else
                            multistep_required_slack
                            if multistep_admitted else 0.
                        )
                        slack_remaining = max(0., clf.remaining_slack - slack_used)
                        executed[lane] = actions[selected]
                        pending[lane] = (selected, actions, nominal, ctx, phi, residual,
                                         predictions, phase, constraints, slack, admitted,
                                         multistep_admitted, recovery_step,
                                         multistep_constraint,
                                         multistep_horizon, planning_seconds,
                                         slack_used, slack_remaining)
                    if not pending:
                        break
                    next_obs, next_states, reward, terminated, truncated, info = world.step(executed)
                    force = info["pusher_tee_contact_force"].detach().cpu().numpy()
                    if not torch.isfinite(reward).all():
                        raise FloatingPointError("PhysX returned a non-finite reward")
                    done_array = np.broadcast_to(torch.as_tensor(terminated).cpu().numpy(), (config.environment.num_envs,))
                    cut_array = np.broadcast_to(torch.as_tensor(truncated).cpu().numpy(), (config.environment.num_envs,))
                    for lane, record in pending.items():
                        (selected, actions, nominal, ctx, phi, residual, predictions,
                         phase, constraints, slack, admitted, multistep_admitted,
                         recovery_step,
                         multistep_constraint, multistep_horizon,
                         planning_seconds, slack_used, slack_remaining) = record
                        trace, clf = traces[lane], filters[lane]
                        if phase_controllers is not None:
                            phase_controllers[lane].observe(
                                next_states[lane], force[lane]
                            )
                        target = residual_target(next_states[lane], nominal[selected])
                        error = residual[selected:selected + 1] - torch.as_tensor(target[None], device=config.device)
                        memories[lane], eta = model.feedback(
                            memories[lane], phi[selected:selected + 1], error,
                            torch.as_tensor(target[None], device=config.device),
                            torch.as_tensor(actions[selected:selected + 1], device=config.device))
                        trace["states"].append(next_states[lane].copy())
                        trace["obs17"].append(next_obs[lane].cpu().numpy().copy())
                        values = dict(actions=actions[selected].copy(), nominal=nominal[selected].copy(),
                                      contexts=ctx[selected].copy(),
                                      contacts=float(force[lane] > config.environment.contact_threshold_n),
                                      predictions=predictions[selected].copy(),
                                      phases=recorded_phase, eta=float(eta.item()),
                                      candidates=actions.copy(), candidate_predictions=predictions.copy(),
                                      candidate_constraints=constraints.copy(), chosen=selected, admitted=admitted,
                                      multistep_admitted=multistep_admitted,
                                      recovery_step=recovery_step,
                                      multistep_constraint=multistep_constraint,
                                      multistep_horizon=multistep_horizon,
                                      planning_seconds=planning_seconds,
                                      slack=slack, slack_used=slack_used,
                                      slack_remaining=slack_remaining,
                                      rho=clf.history[-1], storage=clf.storage,
                                      terminated=bool(done_array[lane]), truncated=bool(cut_array[lane]))
                        for key, value in values.items():
                            trace[key].append(value)
                        clf.executed(
                            actions[selected], admitted or multistep_admitted,
                            slack_used,
                        )
                        clf.observe(next_states[lane])
                        goal_metrics = metrics(next_states[lane], world.goal, geometry.period)
                        quiet = (goal_metrics["position_m"] <= .02 and goal_metrics["angle_rad"] <= .10
                                 and goal_metrics["object_speed_mps"] <= .02
                                 and goal_metrics["object_omega_radps"] <= .20
                                 and goal_metrics["pusher_speed_mps"] <= .02)
                        trace["dl_hold_count"] = trace["dl_hold_count"] + 1 if quiet else 0
                        trace["dl_success"] |= trace["dl_hold_count"] >= 10
                        trace["environment_success"] |= bool(info["success"][lane].item())
                        if done_array[lane]:
                            trace["outcome"] = "environment_success" if trace["environment_success"] else "terminated"
                        elif cut_array[lane]:
                            trace["outcome"] = "time_limit"
                    elapsed = time.perf_counter() - started
                    for lane in pending:
                        traces[lane]["loop_seconds"].append(elapsed)
                    obs, states = next_obs, next_states
                for lane, episode_id in enumerate(ids):
                    trace = traces[lane]
                    if trace["outcome"] == "running":
                        trace["outcome"] = "step_budget"
                    f = config.training.validation_fraction
                    held_out = math.floor((episode_id + 1) * f) > math.floor(episode_id * f)
                    role = "validation" if split == "validation" or (split == "train" and held_out) else split
                    directory = output / "episodes" / f"{name}_{episode_id:04d}"
                    arrays = _arrays(trace, config.control.candidate_count)
                    n = len(arrays["actions"])
                    convergence = convergence_report(arrays["states"], arrays["actions"], world.goal,
                                                     geometry.period, config.evaluation)
                    meta = dict(shape=name, episode=episode_id, seed=seed, lane=lane, role=role,
                                collection_split=split, behavior=behavior, correction=correction,
                                outcome=trace["outcome"], transitions=n, mass_kg=mass, friction=friction,
                                reset_mode=config.environment.reset_mode, control_dt=DT,
                                asset_relative_path=os.path.relpath(geometry.path, directory),
                                asset_hash=geometry.fingerprint, model_hash=initial_model_hash,
                                goal=world.goal.tolist(), orientation_period=geometry.period,
                                contact_transitions=int(arrays["contacts"].sum()),
                                admitted_steps=int(arrays["admitted"].sum()),
                                multistep_admitted_steps=int(arrays["multistep_admitted"].sum()),
                                recovery_steps=int(arrays["recovery_step"].sum()),
                                startup_steps=trace["phases"].count("startup"),
                                approach_steps=(trace["phases"].count("startup")
                                                + trace["phases"].count("approach")),
                                push_steps=(trace["phases"].count("push")
                                            + trace["phases"].count("multistep")),
                                terminal_steps=(trace["phases"].count("terminal")
                                                + trace["phases"].count("clf")),
                                summed_slack=float(arrays["slack_used"].sum()),
                                summed_slack_allowance=float(arrays["slack"].sum()),
                                remaining_slack=float(filters[lane].remaining_slack),
                                final_metrics=metrics(trace["states"][-1], world.goal, geometry.period),
                                environment_success=trace["environment_success"],
                                dl_pose_velocity_hold_success=trace["dl_success"],
                                input_tail_rms_mps=float(np.sqrt(np.mean(arrays["actions"][-20:] ** 2))) if n else None,
                                max_vector_loop_seconds=float(arrays["loop_seconds"].max()) if n else None,
                                warm_vector_loop_mean_seconds=float(arrays["loop_seconds"][1:].mean()) if n > 1 else None,
                                over_50ms_steps=int((arrays["loop_seconds"] > DT).sum()),
                                config_hash=config.fingerprint, candidate_count=config.control.candidate_count)
                    meta["mjx_backend"] = predictor.backend
                    meta["observer_device"] = str(next(model.parameters()).device)
                    meta["clf_device"] = str(config.device)
                    meta["convergence"] = convergence
                    meta["reference_success_terminates"] = config.environment.terminate_on_success
                    save_episode(directory, arrays, meta)
                    if trace["last_attempt"] is not None:
                        save_npz(directory / "rejected_attempt.npz", **trace["last_attempt"])
                    lookup[(name, episode_id)]["status"] = trace["outcome"]
                    manifest["results"].append(meta)
                    print(f"{name} episode={episode_id} steps={n} contact={meta['contact_transitions']} "
                          f"clf={meta['admitted_steps']} outcome={trace['outcome']}", flush=True)
            except Exception as exc:
                details = dict(error=repr(exc), traceback=traceback.format_exc(),
                               partial_steps=[len(t["actions"]) for t in traces] if traces else [])
                write_json(output / f"error_{name}_{group}.json", details)
                for i in ids:
                    if lookup[(name, i)]["status"] == "running":
                        lookup[(name, i)]["status"] = "error"
                print(f"{name} group={group} ERROR {exc!r}", flush=True)
            finally:
                if world is not None:
                    world.close()
                write_json(output / "collection.json", manifest)
    if reference_hashes != verify_checkpoints() or initial_model_hash != model_hash(model):
        raise RuntimeError("Frozen assets changed during collection")
    results = manifest["results"]
    summary = dict(planned_episodes=len(planned), recorded_episodes=len(results),
                   errors=sum(p["status"] == "error" for p in planned),
                   transitions=sum(r["transitions"] for r in results),
                   contact_transitions=sum(r["contact_transitions"] for r in results),
                   approach_steps=sum(r.get("approach_steps", 0) for r in results),
                   push_steps=sum(r.get("push_steps", 0) for r in results),
                   terminal_steps=sum(r.get("terminal_steps", 0) for r in results),
                   admitted_steps=sum(r["admitted_steps"] for r in results),
                   multistep_admitted_steps=sum(r["multistep_admitted_steps"] for r in results),
                   recovery_steps=sum(r["recovery_steps"] for r in results),
                   environment_successes=sum(r["environment_success"] for r in results),
                   dl_pose_velocity_hold_successes=sum(r["dl_pose_velocity_hold_success"] for r in results),
                   clf_infeasible_episodes=sum(r["outcome"] == "clf_infeasible" for r in results),
                   strict_final_holds=sum(r["convergence"]["strict_final_hold"] for r in results),
                   frozen_prior_hashes_unchanged=True, shared_observer_unchanged=True,
                   note="Real PhysX episodes; finite tests do not establish global CLF feasibility or convergence")
    write_json(output / "summary.json", summary)
    if summary["errors"]:
        raise RuntimeError(f"{summary['errors']} planned episodes failed; see {output}/error_*.json")
    return summary
