"""Task geometry energy and auditable sampled finite-memory D/CLF constraints."""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import ControlConfig
from .state import DT, wrap


def task_energy(states, goal, period, config: ControlConfig):
    x = np.asarray(states)
    order = 2 * np.pi / period
    position = np.sum(((x[..., :2] - goal[:2]) / config.position_scale) ** 2, axis=-1)
    # Squared unit-vector difference gives 1-cos; do not square that scalar again.
    direction = (1 - np.cos(order * (x[..., 3] - goal[2]))) / order ** 2
    linear = np.sum((x[..., 4:7] / .10) ** 2, axis=-1)
    angular = x[..., 7] ** 2
    pusher = np.sum((x[..., 10:12] / .10) ** 2, axis=-1)
    return (position + config.orientation_weight * direction
            + config.linear_velocity_weight * linear
            + config.angular_velocity_weight * angular
            + config.pusher_velocity_weight * pusher)


def task_energy_torch(states, goal, period, config: ControlConfig):
    """Differentiable equivalent of task_energy; excludes optional reach shaping."""
    import torch
    x = states
    goal = torch.as_tensor(goal, dtype=x.dtype, device=x.device)
    period = torch.as_tensor(period, dtype=x.dtype, device=x.device)
    while goal.ndim < x.ndim:
        goal = goal.unsqueeze(-2)
    while period.ndim < x.ndim - 1:
        period = period.unsqueeze(-1)
    order = 2 * torch.pi / period
    position = ((x[..., :2] - goal[..., :2]) / config.position_scale).square().sum(-1)
    direction = (1 - torch.cos(order * (x[..., 3] - goal[..., 2]))) / order.square()
    linear = (x[..., 4:7] / .10).square().sum(-1)
    angular = x[..., 7].square()
    pusher = (x[..., 10:12] / .10).square().sum(-1)
    return (position + config.orientation_weight * direction
            + config.linear_velocity_weight * linear
            + config.angular_velocity_weight * angular
            + config.pusher_velocity_weight * pusher)


def metrics(state, goal, period):
    order = 2 * np.pi / period
    return dict(position_m=float(np.linalg.norm(state[:2] - goal[:2])),
                angle_rad=float(abs(wrap(order * (state[3] - goal[2]))) / order),
                object_speed_mps=float(np.linalg.norm(state[4:7])),
                object_omega_radps=float(abs(state[7])),
                pusher_speed_mps=float(np.linalg.norm(state[10:12])))


class FiniteMemoryCLF:
    def __init__(self, config, goal, period, geometry=None, device="cpu"):
        self.config, self.goal, self.period = config, np.asarray(goal), period
        self.geometry = geometry
        self.device = device
        self._cuda_sdf = None
        self.history = deque(maxlen=config.window)
        self.task_history = deque(maxlen=config.window)
        self.previous_action = np.zeros(2)
        self.accepted_steps = 0
        self.remaining_slack = float(config.slack_budget)

    def observe(self, real_state):
        value = float(self.energy(real_state))
        if not np.isfinite(value):
            raise ValueError("Non-finite real task energy")
        self.history.append(value)  # Never fabricate M copies of the first state.
        self.task_history.append(float(task_energy(real_state, self.goal, self.period, self.config)))
        return value

    def energy(self, states):
        states = np.asarray(states)
        task = task_energy(states, self.goal, self.period, self.config)
        if not (
            self.config.reach_weight or self.config.contact_target_weight
        ):
            return task
        if self.geometry is None:
            raise ValueError("Contact shaping requires declared geometry")
        value = task
        if self.config.reach_weight:
            gaps = np.array([
                max(0., self.geometry.contact_gap(s))
                for s in np.atleast_2d(states)
            ])
            if states.ndim == 1:
                gaps = gaps[0]
            value = (
                value + self.config.reach_weight
                * task / (task + .05) * (gaps / .10) ** 2
            )
        if self.config.contact_target_weight:
            targets = self.geometry.contact_targets(
                states, self.goal, self.config.contact_target_turn_gain
            )
            distance = np.linalg.norm(states[..., 8:10] - targets, axis=-1)
            active = task <= self.config.contact_target_activation_energy
            value = (
                value + self.config.contact_target_weight * active
                * task / (task + .001)
                * (distance / self.config.contact_target_scale) ** 2
            )
        return float(np.asarray(value).item()) if states.ndim == 1 else value

    @property
    def ready(self):
        return len(self.history) == self.config.window

    @property
    def storage(self):
        return DT * sum(self.history)

    def available_slack(self, horizon=1, sequence=False):
        c = self.config
        if c.slack_budget == 0:
            return 0.
        if c.slack_strategy == "ledger":
            if sequence or self.task_history[-1] > c.slack_activation_energy:
                return 0.
            return min(self.remaining_slack, c.slack_per_step_max)
        return sum(
            c.slack_budget
            / ((self.accepted_steps + step + 1)
               * (self.accepted_steps + step + 2))
            for step in range(horizon)
        )

    def select(self, actions, predictions):
        if not self.ready:
            raise RuntimeError("CLF needs M genuine feedback states")
        if str(self.device).startswith("cuda"):
            return self._select_cuda(actions, predictions)
        actions, predictions = np.asarray(actions), np.asarray(predictions)
        c = self.config
        value = self.energy(predictions)
        terminal_mode = (
            c.terminal_instantaneous_energy > 0
            and self.task_history[-1] <= c.terminal_instantaneous_energy
        )
        reference = self.history[-1] if terminal_mode else self.history[0]
        residual = ((1 + c.delta) * value - reference
                    + c.alpha * self.task_history[-1] + c.beta * np.sum(actions ** 2, axis=-1))
        # Both schedules have a predeclared finite total budget.
        slack = self.available_slack()
        domain = (np.isfinite(predictions).all(axis=-1)
                  & (np.abs(predictions[:, :2]) <= .35).all(axis=-1)
                  & (predictions[:, 2] >= 0) & (predictions[:, 2] <= .10)
                  & (np.abs(predictions[:, 8:10]) <= .232001).all(axis=-1)
                  & (np.abs(actions) <= .300001).all(axis=-1))
        feasible = domain & np.isfinite(residual) & (residual <= slack)
        score = (np.sum(actions ** 2, axis=-1)
                 + c.smooth_weight * np.sum((actions - self.previous_action) ** 2, axis=-1))
        if c.selection == "energy":
            score = value + 1e-4 * score
        if (c.contact_target_score_weight
                and self.task_history[-1]
                <= c.contact_target_score_activation_energy):
            if self.geometry is None:
                raise ValueError("Contact-target scoring requires declared geometry")
            targets = self.geometry.contact_targets(
                predictions, self.goal, c.contact_target_turn_gain
            )
            contact_score = np.square(
                (predictions[:, 8:10] - targets) / c.contact_target_scale
            ).sum(-1)
            gate = (
                self.task_history[-1]
                / c.contact_target_score_activation_energy
            )
            score = score + c.contact_target_score_weight * gate * contact_score
        index = int(np.argmin(np.where(feasible, score, np.inf))) if feasible.any() else None
        result = dict(index=index, feasible=feasible, residual=residual, slack=slack,
                      required_slack=np.maximum(residual, 0.),
                      predicted_energy=value, domain=domain,
                      reference_energy=reference, terminal_mode=terminal_mode)
        return result

    def select_sequences(self, actions, predictions):
        """Select an H-step predicted sequence by exact moving-window bookkeeping.

        The caller may execute only the first action and replan after real
        feedback.  Therefore this is an experimental finite-step recovery
        filter, not by itself a recursive-feasibility proof.
        """
        if not self.ready:
            raise RuntimeError("Finite-step CLF needs M genuine feedback states")
        actions = np.asarray(actions)
        predictions = np.asarray(predictions)
        if (actions.ndim != 3 or actions.shape[-1] != 2
                or predictions.shape != actions.shape[:-1] + (12,)):
            raise ValueError("Expected action (K,H,2) and prediction (K,H,12)")
        count, horizon = actions.shape[:2]
        if horizon < 1 or count < 1:
            raise ValueError("Finite-step candidate set must be nonempty")
        c = self.config
        energies = self.energy(predictions.reshape(-1, 12)).reshape(count, horizon)
        task = task_energy(
            predictions.reshape(-1, 12), self.goal, self.period, c
        ).reshape(count, horizon)
        history = np.asarray(self.history, dtype=float)
        terminal_mode = (
            c.terminal_instantaneous_energy > 0
            and self.task_history[-1] <= c.terminal_instantaneous_energy
        )
        if terminal_mode:
            delta_storage = energies[:, -1] - history[-1]
        else:
            if horizon >= c.window:
                final_history = energies[:, -c.window:]
            else:
                prefix = np.broadcast_to(
                    history[horizon:], (count, c.window - horizon)
                )
                final_history = np.concatenate((prefix, energies), axis=1)
            delta_storage = final_history.sum(-1) - history.sum()
        prediction_reserve = c.delta * energies.sum(-1)
        task_cost = c.alpha * (
            self.task_history[-1]
            + (task[:, :-1].sum(-1) if horizon > 1 else 0.)
        )
        input_cost = c.beta * np.square(actions).sum(axis=(1, 2))
        residual = delta_storage + prediction_reserve + task_cost + input_cost
        robust_residual = residual + c.multistep_robust_margin
        slack = self.available_slack(horizon, sequence=True)
        flat_state = predictions.reshape(-1, 12)
        flat_action = actions.reshape(-1, 2)
        domain = (
            np.isfinite(predictions).all(axis=(1, 2))
            & np.isfinite(actions).all(axis=(1, 2))
            & (np.abs(flat_state[:, :2]) <= .35).all(axis=-1).reshape(count, horizon).all(-1)
            & ((flat_state[:, 2] >= 0) & (flat_state[:, 2] <= .10)).reshape(count, horizon).all(-1)
            & (np.abs(flat_state[:, 8:10]) <= .232001).all(axis=-1).reshape(count, horizon).all(-1)
            & (np.abs(flat_action) <= .300001).all(axis=-1).reshape(count, horizon).all(-1)
        )
        feasible = domain & np.isfinite(robust_residual) & (robust_residual <= slack)
        score = (
            energies[:, -1]
            + 1e-4 * np.square(actions).sum(axis=(1, 2))
            + 1e-4 * c.smooth_weight
            * np.square(actions[:, 0] - self.previous_action).sum(-1)
        )
        if (c.contact_target_score_weight
                and self.task_history[-1]
                <= c.contact_target_score_activation_energy):
            if self.geometry is None:
                raise ValueError("Contact-target scoring requires declared geometry")
            terminal = predictions[:, -1]
            targets = self.geometry.contact_targets(
                terminal, self.goal, c.contact_target_turn_gain
            )
            contact_score = np.square(
                (terminal[:, 8:10] - targets) / c.contact_target_scale
            ).sum(-1)
            gate = (
                self.task_history[-1]
                / c.contact_target_score_activation_energy
            )
            score = score + c.contact_target_score_weight * gate * contact_score
        index = int(np.argmin(np.where(feasible, score, np.inf))) if feasible.any() else None
        return {
            "index": index,
            "feasible": feasible,
            "residual": residual,
            "robust_residual": robust_residual,
            "slack": slack,
            "required_slack": np.maximum(robust_residual, 0.),
            "predicted_energy": energies,
            "domain": domain,
            "delta_storage": delta_storage,
            "prediction_reserve": prediction_reserve,
            "task_cost": task_cost,
            "input_cost": input_cost,
            "horizon": horizon,
            "terminal_mode": terminal_mode,
        }

    def _select_cuda(self, actions, predictions):
        """Evaluate the complete candidate constraint vector in parallel on CUDA."""
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            x = torch.as_tensor(predictions, dtype=torch.float32, device=self.device)
            u = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            goal = torch.as_tensor(self.goal, dtype=x.dtype, device=self.device)
            c = self.config
            order = 2 * np.pi / self.period
            value = (((x[:, :2] - goal[:2]) / c.position_scale).square().sum(-1)
                     + c.orientation_weight * (1 - torch.cos(order * (x[:, 3] - goal[2]))) / order**2
                     + c.linear_velocity_weight * (x[:, 4:7] / .10).square().sum(-1)
                     + c.angular_velocity_weight * x[:, 7].square()
                     + c.pusher_velocity_weight * (x[:, 10:12] / .10).square().sum(-1))
            task_value = value
            if c.reach_weight:
                if self.geometry is None:
                    raise ValueError("Reach shaping requires declared geometry")
                if self._cuda_sdf is None:
                    self._cuda_sdf = torch.as_tensor(self.geometry.sdf, device=self.device)[None, None]
                relative = x[:, 8:10] - x[:, :2]
                cos, sin = torch.cos(x[:, 3]), torch.sin(x[:, 3])
                local_x = cos * relative[:, 0] + sin * relative[:, 1]
                local_y = -sin * relative[:, 0] + cos * relative[:, 1]
                height, width = self.geometry.sdf.shape
                pixel = torch.stack([local_x / self.geometry.mpp + width / 2 - .5,
                                     height / 2 - local_y / self.geometry.mpp - .5], -1)
                limit = pixel.new_tensor([width - 1, height - 1])
                clipped = torch.minimum(pixel.clamp_min(0), limit)
                grid = (2 * clipped / limit - 1).reshape(1, -1, 1, 2)
                sdf = F.grid_sample(self._cuda_sdf, grid, align_corners=True,
                                    padding_mode="border").reshape(-1)
                sdf += torch.linalg.vector_norm(pixel - clipped, dim=-1) * self.geometry.mpp
                gap = (sdf - .018).clamp_min(0)
                value = value + c.reach_weight * value / (value + .05) * (gap / .10).square()
            if c.contact_target_weight:
                if self.geometry is None:
                    raise ValueError("Contact-target shaping requires declared geometry")
                targets = torch.as_tensor(
                    self.geometry.contact_targets(
                        predictions, self.goal, c.contact_target_turn_gain
                    ),
                    dtype=x.dtype, device=self.device,
                )
                distance = torch.linalg.vector_norm(
                    x[:, 8:10] - targets, dim=-1
                )
                active = task_value <= c.contact_target_activation_energy
                value = (
                    value + c.contact_target_weight * active
                    * task_value / (task_value + .001)
                    * (distance / c.contact_target_scale).square()
                )
            terminal_mode = (
                c.terminal_instantaneous_energy > 0
                and self.task_history[-1] <= c.terminal_instantaneous_energy
            )
            reference = self.history[-1] if terminal_mode else self.history[0]
            residual = ((1 + c.delta) * value - reference
                        + c.alpha * self.task_history[-1] + c.beta * u.square().sum(-1))
            slack = self.available_slack()
            domain = (torch.isfinite(x).all(-1) & (x[:, :2].abs() <= .35).all(-1)
                      & (x[:, 2] >= 0) & (x[:, 2] <= .10)
                      & (x[:, 8:10].abs() <= .232001).all(-1)
                      & (u.abs() <= .300001).all(-1))
            feasible = domain & torch.isfinite(residual) & (residual <= slack)
            previous = u.new_tensor(self.previous_action)
            score = u.square().sum(-1) + c.smooth_weight * (u - previous).square().sum(-1)
            if c.selection == "energy":
                score = value + 1e-4 * score
            if (c.contact_target_score_weight
                    and self.task_history[-1]
                    <= c.contact_target_score_activation_energy):
                if self.geometry is None:
                    raise ValueError("Contact-target scoring requires declared geometry")
                targets = torch.as_tensor(
                    self.geometry.contact_targets(
                        predictions, self.goal, c.contact_target_turn_gain
                    ),
                    dtype=x.dtype, device=self.device,
                )
                contact_score = (
                    (x[:, 8:10] - targets) / c.contact_target_scale
                ).square().sum(-1)
                gate = (
                    self.task_history[-1]
                    / c.contact_target_score_activation_energy
                )
                score = (
                    score + c.contact_target_score_weight
                    * gate * contact_score
                )
            index = int(torch.where(feasible, score, torch.inf).argmin().item()) if feasible.any().item() else None
            residual_numpy = residual.cpu().numpy()
            return dict(index=index, feasible=feasible.cpu().numpy(), residual=residual_numpy,
                        slack=slack, required_slack=np.maximum(residual_numpy, 0.),
                        predicted_energy=value.cpu().numpy(), domain=domain.cpu().numpy(),
                        reference_energy=reference, terminal_mode=terminal_mode)

    def executed(self, action, admitted, used_slack=0.):
        self.previous_action = np.asarray(action).copy()
        if admitted:
            self.accepted_steps += 1
        used_slack = float(used_slack)
        if used_slack < -1e-12 or used_slack > self.remaining_slack + 1e-9:
            raise ValueError("Executed action exceeds the remaining slack ledger")
        if self.config.slack_strategy == "ledger":
            self.remaining_slack = max(0., self.remaining_slack - max(0., used_slack))


def sample_candidates(bank, obs, rng, config, geometry=None, state=None, goal=None):
    import torch
    means, samples = [], []
    with torch.no_grad():
        for actor in bank.actors:
            mean, log_std = actor(obs.reshape(1, 17))
            mean, std = mean[0].cpu().numpy(), log_std[0].exp().cpu().numpy()
            scale, bias = actor.action_scale.cpu().numpy(), actor.action_bias.cpu().numpy()
            means.append((np.tanh(mean) * scale + bias) * .30)
            for j in range(config.samples_per_prior):
                multiplier = config.prior_std_scales[j % len(config.prior_std_scales)] if config.prior_std_scales else 1.
                proposal_std = np.maximum(config.prior_std_scale * multiplier * std, config.prior_std_floor)
                samples.append((np.tanh(mean + proposal_std * rng.standard_normal(2)) * scale + bias) * .30)
    angles = np.arange(6) * np.pi / 3
    anchors = config.anchor_speed * np.c_[np.cos(angles), np.sin(angles)]
    extra = []
    for speed in config.radial_speeds:
        theta = np.arange(config.radial_directions) * 2 * np.pi / config.radial_directions
        extra.extend(speed * np.c_[np.cos(theta), np.sin(theta)])
    if config.geometric_candidates:
        if geometry is None or state is None or goal is None:
            raise ValueError("Geometric candidates need state, geometry and target")
        approach = geometry.approach_action(state, goal)
        turning = geometry.approach_action(state, goal, turn_gain=.06)
        extra.extend([approach, approach * .5, turning, turning * .5])
    actions = np.asarray(means + samples + list(anchors) + extra + [np.zeros(2)], np.float32)
    if len(actions) != config.candidate_count or not np.isfinite(actions).all():
        raise ValueError("Invalid candidate generation")
    return actions
