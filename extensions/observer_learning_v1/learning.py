"""Actual M-network learning; deployment features contain no truth labels."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from stabilization_gym.observation import make_observation_task
from stabilization_learning.observation.certificate import reference_metric
from .tensor_model import TensorModel


class MetricNetwork(nn.Module):
    def __init__(self, model, hidden=32, radius=3., dtype=torch.float64):
        super().__init__()
        n = model.spec.error_dim
        base = reference_metric(model).base
        self.net = nn.Sequential(nn.Linear(model.spec.state_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n*(n+1)//2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("base_cholesky", torch.tensor(np.linalg.cholesky(base), dtype=dtype))
        self.register_buffer("feature_scale", torch.ones(model.spec.state_dim, dtype=dtype))
        indices = torch.tril_indices(n, n); self.register_buffer("row", indices[0]); self.register_buffer("column", indices[1])
        self.register_buffer("diagonal", indices[0] == indices[1])
        self.n, self.radius, self.model = n, float(radius), model
        self.to(dtype=dtype)
        # Off-diagonal Frobenius norm <= .25, diagonal entries in [exp(-r),exp(r)].
        eig = np.linalg.eigvalsh(base)
        self.bounds = (float(eig.min()*np.exp(-2*self.radius)*.75**2),
                       float(eig.max()*np.exp(2*self.radius)*1.25**2))

    def forward(self, estimate):
        raw = self.net(torch.tanh(estimate/self.feature_scale))
        # T = D*(I+E); E strictly lower triangular with ||E||_2 <= .25.
        off = torch.where(self.diagonal, torch.zeros_like(raw), torch.tanh(raw))
        off = .25*off/(1+torch.linalg.vector_norm(off, dim=-1, keepdim=True))
        factor = torch.diag_embed(torch.ones((*estimate.shape[:-1], self.n), dtype=estimate.dtype, device=estimate.device))
        factor[..., self.row, self.column] += off
        logdiag = self.radius*torch.tanh(raw[..., self.diagonal]/self.radius)
        factor = logdiag.exp().unsqueeze(-1)*factor
        composed = self.base_cholesky@factor
        return composed@composed.transpose(-1, -2)


class FrozenNetworkMetric:
    def __init__(self, network):
        self.network = network.eval()
        self.bounds = network.bounds
        self.amplitude = 1.  # MetricObserver labels are overwritten by caller.

    @property
    def version(self):
        digest = hashlib.sha256()
        for key, parameter in self.network.state_dict().items():
            digest.update(key.encode()); digest.update(parameter.detach().cpu().numpy().tobytes())
        digest.update(repr(self.network.radius).encode())
        return digest.hexdigest()

    def matrix(self, estimate):
        parameter = next(self.network.parameters())
        with torch.no_grad():
            return self.network(torch.as_tensor(estimate, dtype=parameter.dtype, device=parameter.device)).cpu().numpy()


def episode_data(task, split, index, scenario):
    """Privileged dataset construction is separate from the network call path."""
    env = make_observation_task(task)
    obs, _ = env.reset(options={"split": split, "index": int(index), "scenario": scenario})
    initial = obs["estimate"].copy()
    supervised = env.supervised_trajectory()
    return dict(**supervised, initial=initial, split=split, index=index, scenario=scenario,
                replay_sha256=env.supervision()["episode"]["replay_sha256"])


def batch_data(model, episodes, device="cpu", dtype=torch.float64):
    tensor_model = TensorModel(model, device, dtype)
    batch = {key: tensor_model.tensor(np.stack([e[key] for e in episodes], axis=1))
             for key in ("truth", "measurement", "mask", "control")}
    batch["initial"] = tensor_model.tensor(np.stack([e["initial"] for e in episodes]))
    batch["noise"] = tensor_model.tensor([e["scenario"] != "nominal" for e in episodes])
    return batch


def rollout(network, tensor_model, batch, *, keep_trace=False):
    spec = tensor_model.spec; estimate = batch["initial"]
    horizon_steps = batch["truth"].shape[0]
    errors, matrices, values, domain, correction_energy = [], [], [], [], []
    estimates = []
    for k in range(horizon_steps):
        predicted_y, h = tensor_model.measure_and_jacobian(estimate)
        residual = (batch["measurement"][k]-predicted_y)*batch["mask"][k]
        weighted = residual*tensor_model.weights
        matrix = network(estimate)
        normalized_correction = spec.dt*torch.linalg.solve(matrix, (h*tensor_model.scale).transpose(-1, -2)@weighted.unsqueeze(-1)).squeeze(-1)
        estimate = tensor_model.retract(estimate, normalized_correction*tensor_model.scale)
        error = tensor_model.error(batch["truth"][k], estimate)/tensor_model.scale
        following_matrix = network(estimate)
        value = torch.einsum("bi,bij,bj->b", error, following_matrix, error)
        errors.append(error); values.append(value); domain.append(tensor_model.domain_penalty(estimate))
        correction_energy.append(normalized_correction.square().sum(-1))
        if keep_trace:
            estimates.append(estimate); matrices.append(following_matrix)
        if k+1 < horizon_steps:
            estimate = tensor_model.predict(estimate, batch["control"][k])
    error = torch.stack(errors); energy = error.square().sum(-1)
    value = torch.stack(values)
    reference_scale = torch.trace(tensor_model.tensor(reference_metric(tensor_model.model).base))/spec.error_dim
    delta = (value[1:]-value[:-1])/reference_scale+.1*spec.dt*energy[:-1]
    # Fixed training noise slack; no claim of a certified disturbance bound.
    allowance = batch["noise"]*.01*spec.dt
    certificate_loss = torch.relu(delta-allowance).square().mean()
    cost = energy[:-1].sum(0)*spec.dt
    terminal = energy[-max(2, round(spec.hold_seconds/spec.dt)):].mean(0)
    domain_loss = torch.stack(domain).mean()
    loss = cost.mean()+.5*terminal.mean()+2.*certificate_loss+100.*domain_loss
    groups = tensor_model.groups(error*tensor_model.scale)
    within = (groups <= tensor_model.tensor(spec.group_tolerances)).all(-1)
    hold_count = round(spec.hold_seconds/spec.dt)+1
    success = within[-hold_count:].all(0) & (torch.stack(domain).max(0).values == 0)
    result = {"loss": loss, "cost": cost, "terminal_mse": terminal,
              "certificate_loss": certificate_loss, "success": success,
              "domain_loss": domain_loss, "correction_energy": torch.stack(correction_energy).mean(),
              "errors": error, "values": value}
    if keep_trace:
        result.update(estimates=torch.stack(estimates), matrices=torch.stack(matrices))
    return result
