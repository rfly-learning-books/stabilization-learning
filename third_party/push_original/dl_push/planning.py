"""Frozen-prior model rollouts for experimental finite-step CLF recovery."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .state import corrected_state, model_context, state12_to_obs17


class PairwiseOneStepMJX:
    """Evaluate one 50 ms MJX step for paired states and actions on one backend."""

    def __init__(self, xml, batch):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        native = mujoco.MjModel.from_xml_path(str(xml))
        expected = ("obj_x", "obj_y", "obj_z", "obj_yaw", "pusher_x", "pusher_y")
        if (native.nq, native.nv, native.nu) != (6, 6, 2):
            raise ValueError("Expected the six-joint H=1 starter model")
        if tuple(native.joint(i).name for i in range(6)) != expected:
            raise ValueError("State ordering does not match the starter model")
        if not np.isclose(native.opt.timestep, .005, rtol=0, atol=1e-12):
            raise ValueError("Pairwise planner requires a 5 ms integrator timestep")
        if int(batch) < 1:
            raise ValueError("Pairwise batch must be positive")

        self.batch = int(batch)
        self.z0 = float(native.body("object").pos[2])
        self.backend = jax.default_backend()
        self.compile_seconds = None
        model, base = mjx.put_model(native), mjx.make_data(native)

        def one(state, action):
            data = base.replace(
                qpos=jnp.array([
                    state[0], state[1], state[2] - self.z0,
                    state[3], state[8], state[9],
                ]),
                qvel=state[jnp.array([4, 5, 6, 7, 10, 11])],
                ctrl=action,
            )
            result = jax.lax.fori_loop(
                0, 10, lambda _, value: mjx.step(model, value), data
            )
            return jnp.array([
                result.qpos[0], result.qpos[1], result.qpos[2] + self.z0,
                result.qpos[3], result.qvel[0], result.qvel[1],
                result.qvel[2], result.qvel[3], result.qpos[4],
                result.qpos[5], result.qvel[4], result.qvel[5],
            ])

        self._predict = jax.jit(jax.vmap(one))

    def predict(self, states, actions):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.shape != actions.shape[:-1] + (12,) or actions.shape[-1] != 2:
            raise ValueError("Expected paired state (N,12) and action (N,2)")
        if not 1 <= len(states) <= self.batch:
            raise ValueError("Pair count exceeds compiled planner batch")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Planner states and actions must be finite")
        if np.any(np.abs(actions) > .300001):
            raise ValueError("Planner action exceeds physical velocity limits")
        padded_states = np.repeat(states[-1:], self.batch, axis=0)
        padded_actions = np.zeros((self.batch, 2), np.float32)
        padded_states[:len(states)] = states
        padded_actions[:len(actions)] = actions
        started = time.perf_counter()
        result = np.asarray(self._predict(padded_states, padded_actions))[:len(states)]
        elapsed = time.perf_counter() - started
        if self.compile_seconds is None:
            self.compile_seconds = elapsed
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite pairwise MJX rollout")
        return result.copy(), elapsed


@torch.no_grad()
def rollout_prior_means(
    predictor, bank, observer, geometry, embedding, memory,
    state, goal, horizon, device, correction_mode="full",
):
    """Roll out one deterministic sequence per frozen prior without fake feedback."""
    count = len(bank.actors)
    if count != predictor.batch or horizon < 1:
        raise ValueError("Planner batch/horizon mismatch")
    if correction_mode not in {"full", "first", "none"}:
        raise ValueError("Unknown finite-step correction mode")
    states = np.repeat(np.asarray(state, np.float32)[None], count, axis=0)
    predicted_states, actions = [], []
    total_seconds = 0.
    diagonal = np.arange(count)
    for step in range(horizon):
        observation = torch.as_tensor(
            state12_to_obs17(
                states, goal, geometry.period,
                geometry.physics_center_xy, geometry.model_origin_xy,
            ),
            device=device,
        )
        all_prior_actions = bank.actor_actions(observation)
        action = (
            all_prior_actions[diagonal, diagonal].cpu().numpy() * .30
        ).astype(np.float32)
        nominal, elapsed = predictor.predict(states, action)
        total_seconds += elapsed
        use_correction = (
            correction_mode == "full"
            or (correction_mode == "first" and step == 0)
        )
        if use_correction:
            contexts = np.stack([
                model_context(
                    states[index], action[index:index + 1],
                    nominal[index:index + 1], geometry.local(states[index]),
                )[0]
                for index in range(count)
            ])
            residual, _ = observer.predict(
                torch.as_tensor(contexts, device=device),
                embedding.expand(count, -1), memory.expand(count),
            )
            states = corrected_state(
                nominal, residual.cpu().numpy()
            ).astype(np.float32)
        else:
            states = nominal.astype(np.float32)
        actions.append(action)
        predicted_states.append(states.copy())
    return {
        "actions": np.stack(actions, axis=1),
        "predictions": np.stack(predicted_states, axis=1),
        "rollout_seconds": total_seconds,
        "actor_indices": np.arange(count, dtype=np.int64),
    }


@torch.no_grad()
def rollout_prior_samples(
    predictor, bank, observer, geometry, embedding, memory,
    state, goal, horizon, device, samples_per_prior, rng,
    std_scales=(1.,), correction_mode="first",
):
    """Roll out deterministic and stochastic trajectories from every prior."""
    if samples_per_prior < 1 or not std_scales or any(x <= 0 for x in std_scales):
        raise ValueError("Sampled rollouts need positive counts and scales")
    prior_count = len(bank.actors)
    actor_indices = np.repeat(
        np.arange(prior_count, dtype=np.int64), samples_per_prior + 1
    )
    count = len(actor_indices)
    if predictor.batch != count or horizon < 1:
        raise ValueError("Sampled planner batch/horizon mismatch")
    if correction_mode not in {"full", "first", "none"}:
        raise ValueError("Unknown finite-step correction mode")

    states = np.repeat(np.asarray(state, np.float32)[None], count, axis=0)
    predicted_states, action_rows = [], []
    total_seconds = 0.
    for step in range(horizon):
        observation = torch.as_tensor(
            state12_to_obs17(
                states, goal, geometry.period,
                geometry.physics_center_xy, geometry.model_origin_xy,
            ),
            device=device,
        )
        action = np.zeros((count, 2), np.float32)
        for actor_index, actor in enumerate(bank.actors):
            indices = np.flatnonzero(actor_indices == actor_index)
            mean, log_std = actor(observation[indices])
            latent = mean.clone()
            noise = torch.as_tensor(
                rng.standard_normal((samples_per_prior, 2)),
                dtype=mean.dtype, device=mean.device,
            )
            multipliers = torch.as_tensor(
                [
                    std_scales[index % len(std_scales)]
                    for index in range(samples_per_prior)
                ],
                dtype=mean.dtype, device=mean.device,
            )[:, None]
            latent[1:] = (
                mean[1:] + log_std[1:].exp() * noise * multipliers
            )
            physical = (
                torch.tanh(latent) * actor.action_scale + actor.action_bias
            ) * .30
            action[indices] = physical.cpu().numpy()
        nominal, elapsed = predictor.predict(states, action)
        total_seconds += elapsed
        use_correction = (
            correction_mode == "full"
            or (correction_mode == "first" and step == 0)
        )
        if use_correction:
            contexts = np.stack([
                model_context(
                    states[index], action[index:index + 1],
                    nominal[index:index + 1], geometry.local(states[index]),
                )[0]
                for index in range(count)
            ])
            residual, _ = observer.predict(
                torch.as_tensor(contexts, device=device),
                embedding.expand(count, -1), memory.expand(count),
            )
            states = corrected_state(
                nominal, residual.cpu().numpy()
            ).astype(np.float32)
        else:
            states = nominal.astype(np.float32)
        action_rows.append(action)
        predicted_states.append(states.copy())
    return {
        "actions": np.stack(action_rows, axis=1),
        "predictions": np.stack(predicted_states, axis=1),
        "rollout_seconds": total_seconds,
        "actor_indices": actor_indices,
    }


def sample_piecewise_sequences(
    rng, count, horizon, blocks, seed_actions=(),
):
    """Generate smooth broad open-loop velocity sequences in physical units."""
    if min(count, horizon, blocks) < 1:
        raise ValueError("Random-shooting dimensions must be positive")
    blocks = min(blocks, horizon)
    scales = np.asarray([.015, .04, .10, .20, .30], np.float32)
    scale = scales[np.arange(count) % len(scales), None, None]
    values = rng.standard_normal((count, blocks, 2)).astype(np.float32) * scale
    for index in range(1, blocks):
        values[:, index] = .65 * values[:, index - 1] + .35 * values[:, index]
    values = np.clip(values, -.30, .30)
    edges = np.linspace(0, horizon, blocks + 1, dtype=int)
    actions = np.empty((count, horizon, 2), np.float32)
    for block in range(blocks):
        actions[:, edges[block]:edges[block + 1]] = values[:, block:block + 1]
    actions[0] = 0.
    for index, action in enumerate(seed_actions, start=1):
        if index >= count:
            break
        actions[index] = np.asarray(action, np.float32)
    return actions


@torch.no_grad()
def rollout_action_sequences(
    predictor, observer, geometry, embedding, memory,
    state, action_sequences, device, correction_mode="first",
):
    """Roll out prescribed actions; no synthetic feedback updates the observer."""
    actions = np.asarray(action_sequences, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError("Expected prescribed actions with shape (K,H,2)")
    count, horizon = actions.shape[:2]
    if predictor.batch != count or horizon < 1:
        raise ValueError("Prescribed planner batch/horizon mismatch")
    if correction_mode not in {"full", "first", "none"}:
        raise ValueError("Unknown finite-step correction mode")
    states = np.repeat(np.asarray(state, np.float32)[None], count, axis=0)
    predicted_states, total_seconds = [], 0.
    for step in range(horizon):
        nominal, elapsed = predictor.predict(states, actions[:, step])
        total_seconds += elapsed
        use_correction = (
            correction_mode == "full"
            or (correction_mode == "first" and step == 0)
        )
        if use_correction:
            contexts = np.stack([
                model_context(
                    states[index], actions[index, step:step + 1],
                    nominal[index:index + 1], geometry.local(states[index]),
                )[0]
                for index in range(count)
            ])
            residual, _ = observer.predict(
                torch.as_tensor(contexts, device=device),
                embedding.expand(count, -1), memory.expand(count),
            )
            states = corrected_state(
                nominal, residual.cpu().numpy()
            ).astype(np.float32)
        else:
            states = nominal.astype(np.float32)
        predicted_states.append(states.copy())
    return {
        "actions": actions,
        "predictions": np.stack(predicted_states, axis=1),
        "rollout_seconds": total_seconds,
        "actor_indices": np.full(count, -1, dtype=np.int64),
    }


@torch.no_grad()
def rollout_action_sequences_contact(
    predictor, observer, geometry, embedding, memory,
    state, action_sequences, device, correction_mode="first",
):
    """Roll out prescribed actions and return contact summaries at each step.

    The predictor must be a :class:`ContactPairwiseOneStepMJX`.  This is the
    contact-aware counterpart of :func:`rollout_action_sequences`; it does not
    alter the observer's causal update.
    """
    actions = np.asarray(action_sequences, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError("Expected prescribed actions with shape (K,H,2)")
    count, horizon = actions.shape[:2]
    if predictor.batch != count or horizon < 1:
        raise ValueError("Prescribed contact planner batch/horizon mismatch")
    if correction_mode not in {"full", "first", "none"}:
        raise ValueError("Unknown finite-step correction mode")
    states = np.repeat(np.asarray(state, np.float32)[None], count, axis=0)
    predicted_states, contact_force, contact_normal = [], [], []
    contact_point, contact_active, total_seconds = [], [], 0.
    for step in range(horizon):
        nominal, elapsed, contact = predictor.predict_contact(
            states, actions[:, step]
        )
        total_seconds += elapsed
        use_correction = (
            correction_mode == "full"
            or (correction_mode == "first" and step == 0)
        )
        if use_correction:
            contexts = np.stack([
                model_context(
                    states[index], actions[index, step:step + 1],
                    nominal[index:index + 1], geometry.local(states[index]),
                )[0]
                for index in range(count)
            ])
            residual, _ = observer.predict(
                torch.as_tensor(contexts, device=device),
                embedding.expand(count, -1), memory.expand(count),
            )
            states = corrected_state(
                nominal, residual.cpu().numpy()
            ).astype(np.float32)
        else:
            states = nominal.astype(np.float32)
        predicted_states.append(states.copy())
        contact_force.append(contact.force.copy())
        contact_normal.append(contact.normal.copy())
        contact_point.append(contact.point.copy())
        contact_active.append(contact.active.copy())
    return {
        "actions": actions,
        "predictions": np.stack(predicted_states, axis=1),
        "contact_forces": np.stack(contact_force, axis=1),
        "contact_normals": np.stack(contact_normal, axis=1),
        "contact_points": np.stack(contact_point, axis=1),
        "contact_active": np.stack(contact_active, axis=1),
        "rollout_seconds": total_seconds,
        "actor_indices": np.full(count, -1, dtype=np.int64),
    }


def select_shortest_feasible_prefix(clf, plan, horizons):
    """Return the first horizon having a feasible sequence and all diagnostics."""
    maximum = plan["actions"].shape[1]
    values = list(horizons) if horizons else [maximum]
    if not values or values != sorted(set(values)) or values[-1] != maximum:
        raise ValueError("Prefix horizons must be sorted, unique and end at rollout length")
    audits = []
    selected = None
    for horizon in values:
        admission = clf.select_sequences(
            plan["actions"][:, :horizon],
            plan["predictions"][:, :horizon],
        )
        audits.append(admission)
        if selected is None and admission["index"] is not None:
            selected = admission
    return {
        "admission": selected,
        "audits": audits,
        "horizons": values,
    }
