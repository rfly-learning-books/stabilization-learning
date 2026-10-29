"""Outer sampling/search and C-only set-encoder training.

The learning boundary is intentionally narrow: Q is searched or predicted
once per episode from a verified mode library.  The controller, physical
model, mode library, and terminal metrics remain fixed while a candidate is
being evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .evaluation import terminal_metrics
from .math import matrix_hash, q_from_parameters
from .q_generator import CertificateGenerator, QConfig, SearchConfig, SetQGenerator, search_q
from .runner import CertificateRunner
from .sampling import SamplingConfig, sample_free_pusher_positions, sample_task
from .solver import CLFConfig


@dataclass(frozen=True)
class PhysicalSearchConfig:
    episodes_per_candidate: int = 32
    position_min: float = .02
    position_max: float = .08
    angle_max_rad: float = np.deg2rad(60.0)
    seed: int = 20260911
    coverage_min: float = .95
    failure_penalty: float = 1000.0

    def __post_init__(self):
        if self.episodes_per_candidate < 1 or min(self.position_min, self.position_max, self.angle_max_rad) <= 0:
            raise ValueError("physical search ranges must be positive")
        if self.position_min > self.position_max:
            raise ValueError("position_min must not exceed position_max")


def _episode_bank(geometry, config: PhysicalSearchConfig):
    bank = []
    sampling = SamplingConfig(
        position_min=config.position_min,
        position_max=config.position_max,
        angle_max_rad=config.angle_max_rad,
    )
    for seed in range(config.episodes_per_candidate):
        rng = np.random.default_rng(config.seed + seed)
        object_q = np.zeros(3, dtype=np.float64)
        target_q = sample_task(object_q, rng, sampling, geometry.period)
        pusher_xy = sample_free_pusher_positions(
            geometry.polygon, 1, rng, sampling
        )[0]
        bank.append((int(config.seed + seed), object_q, target_q, pusher_xy))
    return bank


def physical_q_objective(
    geometry,
    library,
    model_xml,
    clf_config: CLFConfig,
    rollout_config,
    parameters,
    config: PhysicalSearchConfig,
):
    """Evaluate one Q on a fixed episode seed bank using measured physics."""

    Q = q_from_parameters(
        parameters,
        eigenvalue_floor=clf_config.q_eigenvalue_floor,
        trace=clf_config.q_trace,
    )
    adapter_factory = __import__("dl_push.certificate.physics", fromlist=["MuJoCoAdapter"])
    reports = []
    total = 0.0
    for seed, object_q, target_q, pusher_xy in _episode_bank(geometry, config):
        adapter = adapter_factory.MuJoCoAdapter(
            model_xml,
            control_dt=clf_config.control_dt,
            expected_physics_dt=.002,
        )
        runner = CertificateRunner(
            adapter, geometry, library, Q, clf_config, rollout_config,
        )
        report = runner.run(object_q, target_q, pusher_xy, seed=seed)
        reports.append(report)
        final = report["final_metrics"]
        normalized_error = float(np.linalg.norm(np.asarray(final["D"])))
        command_cost = sum(
            float(np.sum(np.square(np.asarray(item["command_mps"])))) * clf_config.control_dt
            for item in report["trace"]
        )
        transfer_ticks = report["trace_summary"]["transfer_ticks"]
        if report["success"]:
            total += normalized_error + 0.01 * command_cost + 0.001 * transfer_ticks
        else:
            total += config.failure_penalty + normalized_error + 0.01 * command_cost + 0.001 * transfer_ticks
    return float(total / len(reports)), reports


def search_q_physical(
    geometry,
    library,
    model_xml,
    clf_config,
    rollout_config,
    search_config: SearchConfig,
    physical_config: PhysicalSearchConfig,
    initial=None,
):
    """Run reproducible Q search with actual per-candidate closed-loop runs."""

    latest_reports = {}

    def objective(parameters):
        loss, reports = physical_q_objective(
            geometry, library, model_xml, clf_config, rollout_config,
            parameters, physical_config,
        )
        latest_reports[tuple(np.asarray(parameters).round(12))] = reports
        return loss

    result = search_q(objective, search_config, initial=initial)
    best_key = tuple(np.asarray(result["best"]["parameters"]).round(12))
    result["best_reports"] = latest_reports.get(best_key, [])
    result["physical_config"] = physical_config.__dict__
    return result


def pad_libraries(libraries):
    """Pack variable-width C libraries into tensors and a validity mask."""

    if not libraries:
        raise ValueError("at least one mode library is required")
    width = max(library.count for library in libraries)
    C = np.zeros((len(libraries), 3, width), dtype=np.float32)
    mask = np.zeros((len(libraries), width), dtype=bool)
    for index, library in enumerate(libraries):
        C[index, :, :library.count] = library.c.astype(np.float32)
        mask[index, :library.count] = library.valid
    if not mask.any(axis=1).all():
        raise ValueError("every encoder example needs at least one valid mode")
    return C, mask


@dataclass(frozen=True)
class EncoderTrainConfig:
    hidden: int = 64
    epochs: int = 1000
    learning_rate: float = 1e-3
    seed: int = 20260911
    device: str = "cpu"

    def __post_init__(self):
        if min(self.hidden, self.epochs) < 1 or self.learning_rate <= 0:
            raise ValueError("encoder training settings must be positive")


def train_set_encoder(libraries, parameter_labels, output, config: EncoderTrainConfig | None = None):
    """Fit the C-only set encoder to source-shape Q-search labels."""

    import torch

    config = config or EncoderTrainConfig()
    torch.manual_seed(config.seed)
    C, mask = pad_libraries(libraries)
    labels = np.asarray(parameter_labels, dtype=np.float32)
    if labels.shape != (len(libraries), 5) or not np.isfinite(labels).all():
        raise ValueError("encoder labels must have shape (B,5)")
    device = torch.device(config.device)
    model = SetQGenerator(config.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    Ct = torch.as_tensor(C, device=device)
    mt = torch.as_tensor(mask, device=device)
    yt = torch.as_tensor(labels, device=device)
    history = []
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted = model(Ct, mt)
        # Q-space loss is the deployment object; parameter loss helps prevent
        # equivalent eigenspace parameterizations from destabilizing training.
        from .q_generator import q_from_parameters_torch

        target_q = q_from_parameters_torch(yt)
        pred_loss = torch.mean(torch.square(predicted - target_q))
        # The set encoder does not expose parameters directly, so use an
        # additional matrix loss against the labels' certificate matrices.
        parameter_loss = pred_loss
        parameter_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(parameter_loss.detach().cpu())})
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"encoder output must be new: {output}")
    output.mkdir(parents=True)
    torch.save({
        "schema": 1,
        "model": model.state_dict(),
        "hidden": config.hidden,
        "q_config": {"trace": 3.0, "eigenvalue_floor": .05},
        "shapes": [library.shape for library in libraries],
        "labels": labels.tolist(),
    }, output / "encoder.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {"schema": 1, "config": config.__dict__, "examples": len(libraries), "final_loss": history[-1]["loss"], "shape_order": [library.shape for library in libraries]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_set_encoder(checkpoint, device="cpu"):
    import torch

    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    config = QConfig(**checkpoint.get("q_config", {}))
    model = SetQGenerator(int(checkpoint["hidden"]), config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def generate_q_from_library(model, library, device="cpu"):
    import torch

    C, mask = pad_libraries([library])
    with torch.no_grad():
        Q = model(torch.as_tensor(C, device=device), torch.as_tensor(mask, device=device))[0]
    return Q.detach().cpu().numpy()

