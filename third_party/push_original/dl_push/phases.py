"""Phase classification and CEM/MPPI-style planning for the new controller.

The controller is deliberately separated from the observer training.  The
phase controller changes only the action-selection policy; the residual
observer still learns the correction ``F0 + b_psi + W * phi`` from real
one-step feedback.  Contact force and phase are control-side observable
features, not extra learnable parameters.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .control import metrics, task_energy


PHASE_APPROACH = "approach"
PHASE_PUSH = "push"
PHASE_TERMINAL = "terminal"


def class_label(state, position_m, angle_rad, energy, contact_force, geometry, config):
    """Return the raw phase for a state without hysteresis.

    The terminal region is entered from pose/angle/energy, so the object can
    hold at the goal even after the pusher detaches.  The push phase is only
    entered when the MJX/PhysX contact force exceeds the declared threshold and
    the object is not yet in the terminal region.
    """
    active = float(contact_force) >= float(config.phase_contact_force_threshold)
    near = (
        float(energy) <= float(config.phase_terminal_energy)
        or (
            float(position_m) <= float(config.phase_terminal_pose_m)
            and float(angle_rad) <= float(config.phase_terminal_angle_rad)
        )
    )
    if near:
        return PHASE_TERMINAL
    if (
        not active
        and float(geometry.contact_gap(state))
        > float(config.phase_approach_gap)
    ):
        return PHASE_APPROACH
    return PHASE_PUSH


class PhaseController:
    """Stateful phase machine with small hysteresis.

    ``observe`` is called after each real feedback state and records the real
    pusher-object force.  ``candidate_phases`` is a pure function used for
    candidate ranking: it classifies each predicted state with the predicted
    contact summary, without mutating the controller.
    """

    def __init__(self, config, goal, period, geometry, device="cpu"):
        self.config = config
        self.goal = np.asarray(goal, dtype=np.float32)
        self.period = float(period)
        self.geometry = geometry
        self.device = device
        self.phase = PHASE_APPROACH
        self.last_force = 0.0
        self._pending = 0
        self.force_history = deque(maxlen=max(1, int(config.phase_contact_lookback)))

    @property
    def current_phase(self):
        return self.phase

    def classify(self, state, contact_force=0.0):
        m = metrics(state, self.goal, self.period)
        energy = task_energy(state, self.goal, self.period, self.config)
        return class_label(
            state, m["position_m"], m["angle_rad"], energy,
            contact_force, self.geometry, self.config,
        )

    def observe(self, state, contact_force=0.0):
        force = np.asarray(contact_force, dtype=np.float32)
        magnitude = float(np.linalg.norm(force) if np.ndim(force) else force)
        self.last_force = magnitude
        self.force_history.append(magnitude)
        raw = self.classify(state, magnitude)
        if self.phase is None:
            self.phase = raw
            self._pending = 0
            return raw
        if raw == self.phase:
            self._pending = 0
            return raw
        self._pending += 1
        if self._pending >= int(self.config.phase_transition_hold):
            self.phase = raw
            self._pending = 0
        return self.phase

    def reset(self, state, contact_force=0.0):
        self.phase = PHASE_APPROACH
        self.last_force = 0.0
        self._pending = 0
        self.force_history.clear()
        self.observe(state, contact_force)

    def candidate_phases(self, predictions, contacts):
        predictions = np.asarray(predictions, dtype=np.float32)
        if contacts is None:
            forces = np.zeros(len(predictions), dtype=np.float32)
        elif hasattr(contacts, "magnitude"):
            forces = np.asarray(contacts.magnitude, dtype=np.float32)
        else:
            forces = np.linalg.norm(np.asarray(contacts, dtype=np.float32), axis=-1)
        result = []
        for state, force in zip(predictions, forces):
            result.append(self.classify(state, force))
        return np.asarray(result, dtype="U12")


class CEMPlanner:
    """Cross-entropy-method planning for the push phase.

    The planner samples action sequences around a running mean, rolls them out
    with the contact-aware pairwise MJX predictor plus observer correction,
    scores them by terminal energy, input/smoothness cost, terminal contact
    placement and CLF residual, then updates the mean/covariance from the elite
    set.  The calling controller still applies the block CLF as the admission
    gate before execution.
    """

    def __init__(self, predictor, observer, geometry, embedding, memory,
                 goal, config, clf, rng, device="cpu"):
        self.predictor = predictor
        self.observer = observer
        self.geometry = geometry
        self.embedding = embedding
        self.memory = memory
        self.goal = np.asarray(goal, dtype=np.float32)
        self.period = float(geometry.period)
        self.config = config
        self.clf = clf
        self.rng = rng
        self.device = device

    def _sample(self, mean, sigma, count):
        horizon = int(self.config.cem_horizon)
        noise = self.rng.standard_normal((count, horizon, 2))
        actions = np.clip(
            np.asarray(mean, dtype=np.float32)[None] + sigma * noise,
            -.30, .30,
        ).astype(np.float32)
        return actions

    def _cost(self, actions, plan):
        terminal = task_energy(
            plan["predictions"][:, -1], self.goal, self.period, self.config
        )
        input_cost = np.mean(np.sum(np.square(actions), axis=-1), axis=-1)
        smooth_cost = np.mean(
            np.sum(np.square(np.diff(actions, axis=1)), axis=-1), axis=-1
        )
        total = (
            terminal
            + float(self.config.cem_input_weight) * input_cost
            + float(self.config.cem_smooth_weight) * smooth_cost
        )
        if self.config.cem_contact_weight > 0 and self.geometry is not None:
            terminal_states = plan["predictions"][:, -1]
            targets = self.geometry.contact_targets(
                terminal_states, self.goal,
                self.config.contact_target_turn_gain,
            )
            contact_cost = np.sum(
                np.square(
                    (
                        terminal_states[:, 8:10]
                        - np.asarray(targets, dtype=np.float32)
                    ) / max(float(self.config.contact_target_scale), 1e-8)
                ),
                axis=-1,
            )
            total = total + float(self.config.cem_contact_weight) * contact_cost
        if self.clf is not None and self.clf.ready:
            admission = self.clf.select_sequences(
                actions, plan["predictions"]
            )
            residual = np.asarray(admission["robust_residual"], dtype=np.float32)
            feasible = np.asarray(admission["feasible"], dtype=bool)
            penalty = np.where(
                feasible,
                np.maximum(residual, 0.0),
                float(self.config.cem_clf_weight)
                * (1.0 + np.maximum(residual, 0.0)),
            )
            total = total + float(self.config.cem_clf_weight) * penalty
        return total

    def plan(self, state, seed_actions=(), count=None):
        count = int(count or self.config.cem_sequences)
        horizon = int(self.config.cem_horizon)
        if count < 2 or horizon < 1:
            raise ValueError("CEM needs at least two sequences and one horizon")
        candidates = (
            np.asarray(seed_actions, dtype=np.float32)
            if seed_actions is not None and len(seed_actions)
            else None
        )
        if candidates is None:
            mean = np.zeros((horizon, 2), dtype=np.float32)
            sigma = float(self.config.cem_sigma)
        else:
            if candidates.ndim == 2 and candidates.shape[1] == 2:
                candidates = np.broadcast_to(
                    candidates[:, None, :], (len(candidates), horizon, 2)
                )
            mean = np.mean(candidates[:, :horizon], axis=0)
            sigma = max(float(self.config.cem_min_sigma),
                        float(np.std(candidates[:, :horizon], axis=0).mean()))
        best = None
        elite_count = max(2, int(np.ceil(count * float(self.config.cem_elite_fraction))))
        for iteration in range(int(self.config.cem_iterations)):
            actions = self._sample(mean, sigma, count)
            plan = self._rollout(state, actions)
            costs = self._cost(actions, plan)
            order = np.argsort(costs, kind="stable")[:elite_count]
            elite = actions[order]
            mean = np.mean(elite, axis=0).astype(np.float32)
            sigma = max(
                float(self.config.cem_min_sigma),
                float(np.std(elite, axis=0).mean()),
            )
            index = int(np.argmin(costs))
            candidate = dict(
                index=index, cost=float(costs[index]), mean=mean.copy(),
                sigma=sigma, iteration=iteration,
                actions=actions[index].copy(),
                predictions=plan["predictions"][index].copy(),
                contact_forces=plan["contact_forces"][index].copy(),
                contact_normals=plan["contact_normals"][index].copy(),
                contact_points=plan["contact_points"][index].copy(),
                contact_active=plan["contact_active"][index].copy(),
                rollout_seconds=float(plan["rollout_seconds"]),
            )
            if best is None or candidate["cost"] < best["cost"]:
                best = candidate
        return {
            "best": best,
            "mean": mean,
            "sigma": sigma,
            "elite_count": elite_count,
            "iterations": int(self.config.cem_iterations),
            "horizon": horizon,
        }

    def _rollout(self, state, actions):
        from .planning import rollout_action_sequences_contact

        plan = rollout_action_sequences_contact(
            self.predictor, self.observer, self.geometry,
            self.embedding, self.memory, state, actions, self.device,
            correction_mode="first",
        )
        return plan
