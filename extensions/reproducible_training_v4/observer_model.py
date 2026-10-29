"""Scenario-aware observer model and explicit full-history failure semantics."""
import math

import torch

from stabilization_gym.observation.models import PoseModel
from stabilization_learning.observation import reference_metric
from extensions.observer_learning_v1.tensor_model import TensorModel, rotation, skew


class ScenarioTensorModel(TensorModel):
    """Known geometry belongs to each episode, independently of its metric."""
    def __init__(self, model, scenarios):
        super().__init__(model, device="cpu", dtype=torch.float64)
        self.scenarios = tuple(scenarios)
        if self.spec.name == "cooperative_pose":
            self.points = torch.stack([self.tensor(PoseModel(degenerate=s == "degenerate").points)
                                       for s in self.scenarios])

    def camera_points(self, state):
        return state[..., None, :3] + torch.einsum("...ij,...pj->...pi", rotation(state[..., 3:7]), self.points)

    def measure_and_jacobian(self, state):
        if self.spec.name != "cooperative_pose":
            return super().measure_and_jacobian(state)
        matrix = rotation(state[..., 3:7])
        points = self.camera_points(state)
        x, y, z = points.unbind(-1)
        # Keep finite differentiable proposals beyond the physical domain;
        # exact_domain_ok separately and irreversibly marks those failures.
        z = z.clamp(min=.01)
        measurement = torch.stack((600*x/z+320, 600*y/z+240), -1).flatten(-2)
        zero = torch.zeros_like(z)
        projection = torch.stack((600/z, zero, -600*x/z.square(), zero, 600/z, -600*y/z.square()), -1).reshape(*points.shape[:-1], 2, 3)
        oriented = -torch.einsum("...ij,...pjk->...pik", matrix, skew(self.points))
        h = torch.cat((projection, projection @ oriented,
            torch.zeros((*projection.shape[:-1], 6), dtype=state.dtype, device=state.device)), -1)
        return measurement, h.flatten(-3, -2)

    def exact_domain_ok(self, state):
        """Same thresholds and coordinate conventions as physical domain_ok."""
        finite = torch.isfinite(state).all(-1)
        if self.spec.name == "motor_speed":
            return finite & (state.abs().amax(-1) <= 6.)
        if self.spec.name == "cartpole_output":
            return finite & (state[..., 0].abs() <= 2.4) & (state[..., 2].abs() <= math.pi/4) & (state[..., [1, 3]].abs().amax(-1) <= 20.)
        return finite & ((torch.linalg.vector_norm(state[..., 3:7], dim=-1)-1.).abs() < 1e-6) & (
            self.camera_points(state)[..., 2].amin(-1) >= .2) & (
            torch.linalg.vector_norm(state[..., :3], dim=-1) <= 20.) & (state[..., 7:].abs().amax(-1) <= 10.)

    def domain_penalty(self, state):
        """Smooth companion of exact boundaries, never used as the success gate."""
        if self.spec.name == "motor_speed":
            return super().domain_penalty(state)
        if self.spec.name == "cartpole_output":
            return (torch.relu(state[..., 0].abs()-2.4).square() + torch.relu(state[..., 2].abs()-math.pi/4).square()
                    + torch.relu(state[..., [1, 3]].abs()-20.).square().sum(-1))
        return (torch.relu(.2-self.camera_points(state)[..., 2]).square().sum(-1)
                + torch.relu(torch.linalg.vector_norm(state[..., :3], dim=-1)-20.).square()
                + torch.relu(state[..., 7:].abs()-10.).square().sum(-1)
                + torch.relu((torch.linalg.vector_norm(state[..., 3:7], dim=-1)-1.).abs()-1e-6).square())


def corrected_rollout(network, tensor_model, batch, *, keep_trace=False):
    """Full differentiable surrogate plus deployment-comparable failure scoring.

    Preserve the explicit training normalization: d(V/reference_scale) +
    0.1*dt*||e||², with 0.01*dt slack for every non-nominal scenario. This is
    a training surrogate, not the raw-V evaluation residual or a noise bound.
    """
    spec = tensor_model.spec
    estimate = batch["initial"]
    horizon_steps = batch["truth"].shape[0]
    errors, values, domains, energies, failures = [], [], [], [], []
    estimates, matrices = [], []
    for k in range(horizon_steps):
        predicted_y, h = tensor_model.measure_and_jacobian(estimate)
        residual = (batch["measurement"][k]-predicted_y)*batch["mask"][k]
        matrix = network(estimate)
        correction = spec.dt*torch.linalg.solve(matrix,
            (h*tensor_model.scale).transpose(-1, -2) @ (residual*tensor_model.weights).unsqueeze(-1)).squeeze(-1)
        estimate = tensor_model.retract(estimate, correction*tensor_model.scale)
        error = tensor_model.error(batch["truth"][k], estimate)/tensor_model.scale
        following_matrix = network(estimate)
        values.append(torch.einsum("bi,bij,bj->b", error, following_matrix, error))
        errors.append(error)
        domains.append(tensor_model.domain_penalty(estimate))
        energies.append(correction.square().sum(-1))
        failed = ~tensor_model.exact_domain_ok(estimate) | ~tensor_model.exact_domain_ok(batch["truth"][k])
        if spec.name == "cooperative_pose":
            failed = failed | (torch.linalg.vector_norm((error*tensor_model.scale)[..., 3:6], dim=-1) >= math.pi-.05)
        failures.append(failed)
        if keep_trace:
            estimates.append(estimate)
            matrices.append(following_matrix)
        if k+1 < horizon_steps:
            estimate = tensor_model.predict(estimate, batch["control"][k])
    error, value = torch.stack(errors), torch.stack(values)
    energy, failed = error.square().sum(-1), torch.stack(failures)
    reference_scale = torch.trace(tensor_model.tensor(reference_metric(tensor_model.model).base))/spec.error_dim
    delta = (value[1:]-value[:-1])/reference_scale+.1*spec.dt*energy[:-1]
    certificate_loss = torch.relu(delta-batch["noise"]*.01*spec.dt).square().mean()
    full_cost = energy[:-1].sum(0)*spec.dt
    terminal = energy[-max(2, round(spec.hold_seconds/spec.dt)):].mean(0)
    domain_loss = torch.stack(domains).mean()
    loss = full_cost.mean()+.5*terminal.mean()+2.*certificate_loss+100.*domain_loss
    groups = tensor_model.groups(error*tensor_model.scale)
    within = (groups <= tensor_model.tensor(spec.group_tolerances)).all(-1)
    hold_count = round(spec.hold_seconds/spec.dt)+1
    ever_failed = failed.any(0)
    success = within[-hold_count:].all(0) & ~ever_failed
    # Deployment stops on its first failed posterior, and scores only the
    # recorded prefix plus the declared unobserved-tail penalty (25/sec floor).
    index = torch.arange(horizon_steps, device=energy.device)[:, None]
    first_failure = torch.where(failed, index, horizon_steps).amin(0)
    prefix = (index[:-1] < first_failure[None]).to(energy)
    failure_energy = energy.gather(0, first_failure.clamp_max(horizon_steps-1)[None]).squeeze(0)
    tail_seconds = (spec.horizon-first_failure.to(energy)*spec.dt).clamp_min(0.)
    physical_cost = (energy[:-1]*prefix).sum(0)*spec.dt + torch.where(ever_failed,
        tail_seconds*failure_energy.clamp_min(25.), torch.zeros_like(full_cost))
    result = dict(loss=loss, cost=physical_cost, training_full_sequence_cost=full_cost,
        terminal_mse=terminal, certificate_loss=certificate_loss, success=success,
        failed=ever_failed, failure_by_step=failed, first_failure_index=first_failure,
        domain_loss=domain_loss, correction_energy=torch.stack(energies).mean(), errors=error, values=value,
        reference_scale=reference_scale)
    if keep_trace:
        result.update(estimates=torch.stack(estimates), matrices=torch.stack(matrices))
    return result
