"""Contact-aware MJX one-step and pairwise interfaces.

The original ``starter.mjx_one_step.OneStepMJX`` intentionally returns only the
next physical state.  The new H=1 interface keeps that contract for backward
compatibility and additionally exposes the pusher-object contact summary through
``predict_contact``.  This lets the phase controller decide whether a candidate
is approaching, pushing, or already in the terminal region without asking the
real PhysX environment to evaluate every candidate.

The contact force is extracted with MuJoCo MJX's ``support.contact_force``.  This
is an internal helper of the pinned MuJoCo 3.5.0 package, so the class verifies
the expected contact fields and records the model version in metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import numpy as np


@dataclass
class ContactSummary:
    """Aggregated pusher-object contact data for one or many candidates.

    ``force`` is the world-frame normal and tangential force at the pusher-object
    contact, with shape (N, 3).  ``normal`` and ``point`` are the contact frame
    normal and contact position.  ``active`` is a boolean per candidate.
    """

    force: np.ndarray
    normal: np.ndarray
    point: np.ndarray
    active: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.force, axis=-1)

    def __len__(self) -> int:
        return len(self.force)


def _check_h1_model(native, expected_steps):
    expected = ("obj_x", "obj_y", "obj_z", "obj_yaw", "pusher_x", "pusher_y")
    if (native.nq, native.nv, native.nu) != (6, 6, 2):
        raise ValueError("Expected the six-joint H=1 starter model")
    if tuple(native.joint(i).name for i in range(6)) != expected:
        raise ValueError("State ordering does not match the starter model")
    if not np.isclose(native.opt.timestep, .005, rtol=0, atol=1e-12):
        raise ValueError("Contact planner requires a 5 ms integrator timestep")
    if expected_steps < 1:
        raise ValueError("Contact planner needs at least one substep")


def _pusher_geom_index(native):
    names = [native.geom(i).name for i in range(native.ngeom)]
    for candidate in ("pusher_geom", "pusher"):
        if candidate in names:
            return int(names.index(candidate))
    raise ValueError("H=1 MJX model does not contain a named pusher geom")


def _sum_contact(state, model, data, pusher_geom):
    """JAX-traceable aggregate of pusher-object contact data.

    MuJoCo stores a fixed-size contact array.  Inactive entries have
    ``efc_address < 0`` and are ignored.  ``support.contact_force`` requires a
    Python integer contact id, so the loop is unrolled at trace time.
    """
    import jax.numpy as jnp
    from mujoco.mjx._src import support

    contact = data._impl.contact
    count = int(contact.geom1.shape[0])
    force = jnp.zeros(3)
    normal = jnp.zeros(3)
    point = jnp.zeros(3)
    active = jnp.zeros((), dtype=jnp.int32)
    for index in range(count):
        geom1 = contact.geom1[index]
        geom2 = contact.geom2[index]
        is_push = (geom1 == pusher_geom) | (geom2 == pusher_geom)
        is_active = contact.efc_address[index] >= 0
        mask = is_push & is_active
        value = support.contact_force(model, data, index, to_world_frame=True)
        force = force + jnp.where(mask, value[:3], jnp.zeros(3))
        normal = normal + jnp.where(mask, contact.frame[index][0], jnp.zeros(3))
        point = point + jnp.where(mask, contact.pos[index], jnp.zeros(3))
        active = active + jnp.where(mask, jnp.int32(1), jnp.int32(0))
    return force, normal, point, active


class ContactOneStepMJX:
    """Like ``starter.mjx_one_step.OneStepMJX``, plus contact diagnostics."""

    def __init__(self, xml, candidates=32, substeps=10):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        native = mujoco.MjModel.from_xml_path(str(Path(xml)))
        _check_h1_model(native, substeps)
        if int(candidates) < 1:
            raise ValueError("candidates must be positive")
        self.candidates = int(candidates)
        self.z0 = float(native.body("object").pos[2])
        self.substeps = int(substeps)
        self.pusher_geom = _pusher_geom_index(native)
        self.backend = jax.default_backend()
        self.compile_seconds = None
        self.native_version = mujoco.__version__
        self.model, self.base = mjx.put_model(native), mjx.make_data(native)

        def predict(state, actions):
            data = self.base.replace(
                qpos=jnp.array([
                    state[0], state[1], state[2] - self.z0,
                    state[3], state[8], state[9],
                ]),
                qvel=state[jnp.array([4, 5, 6, 7, 10, 11])],
            )

            @jax.vmap
            def one(action):
                result = jax.lax.fori_loop(
                    0, self.substeps,
                    lambda _, value: mjx.step(self.model, value),
                    data.replace(ctrl=action),
                )
                next_state = jnp.array([
                    result.qpos[0], result.qpos[1], result.qpos[2] + self.z0,
                    result.qpos[3], result.qvel[0], result.qvel[1],
                    result.qvel[2], result.qvel[3], result.qpos[4],
                    result.qpos[5], result.qvel[4], result.qvel[5],
                ])
                summary = _sum_contact(next_state, self.model, result, self.pusher_geom)
                return (next_state, *summary)

            return one(actions)

        self._predict = jax.jit(predict)

    def _padding(self, state, actions):
        state = np.asarray(state, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if state.shape != (12,) or actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("Expected state (12,) and physical actions (N,2)")
        if not 1 <= len(actions) <= self.candidates:
            raise ValueError("Candidate count exceeds this compiled predictor")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("State/actions must be finite")
        if np.any(np.abs(actions) > .300001):
            raise ValueError("Physical action exceeds +/-0.30 m/s")
        padded = np.zeros((self.candidates, 2), np.float32)
        padded[:len(actions)] = actions
        return state, actions, padded

    def predict_contact(self, state, actions_mps):
        """Return ``(states, elapsed, ContactSummary)`` for candidate actions."""
        state, actions, padded = self._padding(state, actions_mps)
        started = time.perf_counter()
        raw = self._predict(state, padded)
        output = tuple(np.asarray(value) for value in raw)
        elapsed = time.perf_counter() - started
        if self.compile_seconds is None:
            self.compile_seconds = elapsed
        states = output[0][:len(actions)].copy()
        force = output[1][:len(actions)].copy()
        normal = output[2][:len(actions)].copy()
        point = output[3][:len(actions)].copy()
        active = output[4][:len(actions)].copy()
        if not np.isfinite(states).all() or not np.isfinite(force).all():
            raise FloatingPointError("Non-finite MJX prediction")
        return (
            states,
            elapsed,
            ContactSummary(
                force=force,
                normal=normal,
                point=point,
                active=(active > 0.5),
            ),
        )

    def predict(self, state, actions_mps):
        """Backward-compatible H=1 interface (states, elapsed)."""
        states, elapsed, _ = self.predict_contact(state, actions_mps)
        return states, elapsed


class ContactPairwiseOneStepMJX:
    """Paired-state multi-candidate MJX rollout with contact diagnostics."""

    def __init__(self, xml, batch, substeps=10):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        native = mujoco.MjModel.from_xml_path(str(Path(xml)))
        _check_h1_model(native, substeps)
        if int(batch) < 1:
            raise ValueError("Pairwise batch must be positive")
        self.batch = int(batch)
        self.z0 = float(native.body("object").pos[2])
        self.substeps = int(substeps)
        self.pusher_geom = _pusher_geom_index(native)
        self.backend = jax.default_backend()
        self.compile_seconds = None
        self.native_version = mujoco.__version__
        self.model, self.base = mjx.put_model(native), mjx.make_data(native)

        def one(state, action):
            data = self.base.replace(
                qpos=jnp.array([
                    state[0], state[1], state[2] - self.z0,
                    state[3], state[8], state[9],
                ]),
                qvel=state[jnp.array([4, 5, 6, 7, 10, 11])],
                ctrl=action,
            )
            result = jax.lax.fori_loop(
                0, self.substeps,
                lambda _, value: mjx.step(self.model, value),
                data,
            )
            next_state = jnp.array([
                result.qpos[0], result.qpos[1], result.qpos[2] + self.z0,
                result.qpos[3], result.qvel[0], result.qvel[1],
                result.qvel[2], result.qvel[3], result.qpos[4],
                result.qpos[5], result.qvel[4], result.qvel[5],
            ])
            summary = _sum_contact(next_state, self.model, result, self.pusher_geom)
            return (next_state, *summary)

        self._predict = jax.jit(jax.vmap(one))

    def predict_contact(self, states, actions):
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
        raw = self._predict(padded_states, padded_actions)
        output = tuple(np.asarray(value) for value in raw)
        elapsed = time.perf_counter() - started
        if self.compile_seconds is None:
            self.compile_seconds = elapsed
        states_out = output[0][:len(states)].copy()
        force = output[1][:len(states)].copy()
        normal = output[2][:len(states)].copy()
        point = output[3][:len(states)].copy()
        active = output[4][:len(states)].copy()
        if not np.isfinite(states_out).all() or not np.isfinite(force).all():
            raise FloatingPointError("Non-finite pairwise MJX rollout")
        return (
            states_out,
            elapsed,
            ContactSummary(
                force=force,
                normal=normal,
                point=point,
                active=(active > 0.5),
            ),
        )

    def predict(self, states, actions):
        states, elapsed, _ = self.predict_contact(states, actions)
        return states, elapsed
