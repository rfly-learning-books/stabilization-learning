"""Runnable shared-RGB/goal-metric learning core, isolated from frozen trainers.

This module learns a structured controller, not a free action network. A short
feature block and a short metric block optimize the same physical objective.
Only the nominal P0 label model is differentiated; real renderers and measured
success/arrival times remain rollout evaluation. No long training is launched on
import. ``python -m stabilization_learning.joint_v2.visual --smoke`` exercises
arithmetic fixtures; it is NOT a visual-servo performance experiment.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stabilization_gym.tasks_v2.visual import (
    VisualControlConfig, VisualObservation, VisualTaskAdapter, quadratic_value,
)


class TinyRGBBackbone(nn.Module):
    """Smooth, randomly initialized RGB autoencoder for engineering/small tests.

    Production backbones can be supplied instead: implement encode(image) and
    decode(raw_latent). This compact model makes no performance claim about the
    existing 256-pixel room task and loads no pretrained weights.
    """

    def __init__(self, image_shape: tuple[int, int, int] = (3, 16, 16),
                 latent_dim: int = 16, hidden: int = 32):
        super().__init__()
        if image_shape[0] != 3 or min(image_shape[1:]) < 4 or latent_dim < 4 or hidden < 4:
            raise ValueError("Expected RGB >=4x4, >=4 latent coordinates and hidden width >=4")
        self.image_shape, self.latent_dim = tuple(image_shape), latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.SiLU(),
            nn.AvgPool2d(2), nn.Conv2d(8, 8, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(32, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, math.prod(image_shape)), nn.Sigmoid())

    def encode(self, image: Tensor) -> Tensor:
        if tuple(image.shape[1:]) != self.image_shape:
            raise ValueError(f"Backbone expects B,{self.image_shape}")
        return self.encoder(image)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent).reshape(-1, *self.image_shape)


class NormalizedRGBFeatures(nn.Module):
    """Registered coordinate buffers shared by encoding, JVPs and checkpoints.

    Reconstructing the raw latent intentionally cannot constrain normalized
    control-feature magnitude. Calibration and geometry losses do that job.
    """

    def __init__(self, backbone: nn.Module, latent_dim: int):
        super().__init__()
        self.backbone, self.latent_dim = backbone, latent_dim
        self.register_buffer("center", torch.zeros(latent_dim))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("coordinate_version", torch.zeros((), dtype=torch.long))

    def encode(self, image: Tensor) -> Tensor:
        z = self.backbone.encode(image)
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError("Backbone encode must return B,latent_dim")
        return (z - self.center) / self.scale

    def reconstruct(self, image: Tensor) -> Tensor:
        return self.backbone.decode(self.backbone.encode(image))

    @torch.no_grad()
    def calibrate(self, image: Tensor, goal: Tensor, physical_error: Tensor,
                  *, minimum_scale: float = 1e-6) -> None:
        if bool(self.calibrated):
            raise RuntimeError("Coordinates are already calibrated; create an explicit new protocol boundary")
        if minimum_scale <= 0 or not math.isfinite(minimum_scale):
            raise ValueError("minimum_scale must be positive")
        e_phys = physical_error.detach()
        if (not torch.isfinite(e_phys).all() or (e_phys < 0).any()
                or e_phys.mean() <= 1e-10):
            raise ValueError("Nondegenerate, nonnegative training physical errors are required")
        raw, target = self.backbone.encode(image), self.backbone.encode(goal)
        mean_energy = 0.5 * (raw - target).square().sum(-1).mean()
        if not torch.isfinite(mean_energy) or mean_energy <= 1e-16:
            raise ValueError("Feature calibration failed: indistinguishable training image pairs")
        self.center.copy_(torch.cat((raw, target)).mean(0))
        self.scale.copy_((mean_energy / e_phys.mean()).sqrt().clamp_min(minimum_scale))
        self.calibrated.fill_(True)
        self.coordinate_version.add_(1)


class GoalConditionedMetric(nn.Module):
    """Goal-only SPD metric with trace d and eigenvalues >= floor.

    A common coordinate scale is fixed separately. The upper eigenvalue bound is
    d-(d-1)*floor, implied by the fixed trace and lower bound.
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 32, floor: float = 0.1):
        super().__init__()
        if latent_dim < 4 or not 0 < floor < 1:
            raise ValueError("At least four features and a floor strictly between 0 and 1 are required")
        self.latent_dim, self.floor = latent_dim, float(floor)
        rows, cols = torch.tril_indices(latent_dim, latent_dim)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.register_buffer("diagonal", rows == cols)
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, len(rows)))
        nn.init.normal_(self.net[-1].weight, std=0.03)
        nn.init.normal_(self.net[-1].bias, std=0.01)
        with torch.no_grad():
            self.net[-1].bias[self.diagonal] += math.log(math.expm1(1.0))

    def forward(self, goal: Tensor, context: Tensor | None = None) -> Tensor:
        # Optional two-argument route implements NeuralCertificate's matrix
        # protocol. The error is deliberately ignored: the metric is goal-only.
        if context is not None:
            goal = context
        raw = self.net(goal)
        entries = torch.where(self.diagonal, F.softplus(raw) + 1e-4, raw)
        lower = entries.new_zeros(*entries.shape[:-1], self.latent_dim, self.latent_dim)
        lower[..., self.rows, self.cols] = entries
        gram = lower @ lower.transpose(-1, -2)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        eye = torch.eye(self.latent_dim, dtype=goal.dtype, device=goal.device)
        return self.floor * eye + self.latent_dim * (1 - self.floor) * gram / trace


@dataclass(frozen=True)
class VisualBatch:
    """Public observations and detached training truth have separate fields.

    ``applied_action`` is the realized normalized motion over the interval for
    local-response supervision, not an unexecuted request under actuator lag.
    ``controller_transition`` marks controller data eligible for a same-version
    certificate-fit loss. Random exploration must have this mask false. Stored
    behavior_version is diagnostic; the loss does not claim a counterfactual
    successor for the newly updated policy.
    """

    observation: VisualObservation
    next_image: Tensor
    offset: Tensor
    next_offset: Tensor
    applied_action: Tensor
    dt: Tensor
    controller_transition: Tensor
    previous_action: Tensor | None = None
    behavior_version: int | None = None
    split: str = "train"

    def validate(self) -> None:
        self.observation.validate()
        b = len(self.observation.image)
        if self.split != "train":
            raise ValueError("Only training-split transitions can enter a learner update")
        if self.next_image.shape != self.observation.image.shape:
            raise ValueError("Next RGB has an incompatible shape")
        for label in (self.offset, self.next_offset, self.applied_action):
            if label.shape != (b, 4) or not torch.isfinite(label).all():
                raise ValueError("Training offsets and applied action must be finite B,4")
        if self.previous_action is not None and self.previous_action.shape != (b, 4):
            raise ValueError("previous_action must have shape B,4")
        if self.dt.shape != (b,) or (self.dt <= 0).any() or not torch.isfinite(self.dt).all():
            raise ValueError("dt must contain B positive finite durations")
        if self.controller_transition.shape != (b,) or self.controller_transition.dtype != torch.bool:
            raise ValueError("controller_transition must be an explicit B boolean mask")
        if self.controller_transition.any() and self.behavior_version is None:
            raise ValueError("Controller transitions must declare their behavior version")
        if not torch.isfinite(self.next_image).all():
            raise ValueError("Next RGB must be finite")
        reference = self.observation.image
        for value in (self.next_image, self.offset, self.next_offset, self.applied_action,
                      self.dt, self.controller_transition, self.previous_action):
            if value is not None and value.device != reference.device:
                raise ValueError("Batch tensors must share a device")
            if value is not None and value.is_floating_point() and value.dtype != reference.dtype:
                raise ValueError("Batch floating tensors must share a dtype")


@dataclass(frozen=True)
class JointVisualConfig:
    feature_lr: float = 2e-4
    metric_lr: float = 3e-4
    physical_weight: float = 1.0
    calibration_weight: float = 0.2
    interaction_weight: float = 0.02
    decrease_weight: float = 0.1
    reconstruction_weight: float = 0.1
    geometry_weight: float = 0.01
    value_lower: float = 0.1
    value_upper: float = 4.0
    physical_floor: float = 0.01
    motion_floor: float = 0.01
    decrease_per_second: float = 0.1
    input_weight: float = 0.01
    action_change_weight: float = 0.01
    feature_std_lower: float = 0.05
    feature_std_upper: float = 3.0
    gradient_clip: float = 10.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("All learning configuration values must be finite and nonnegative")
        if min(self.feature_lr, self.metric_lr, self.physical_floor,
               self.motion_floor, self.gradient_clip, self.value_lower) <= 0:
            raise ValueError("Learning rates, denominators, clip and lower physical bound must be positive")
        if self.value_upper <= self.value_lower or self.feature_std_upper <= self.feature_std_lower:
            raise ValueError("Upper calibration/geometry bounds must exceed lower bounds")


@contextmanager
def frozen_parameters(module: nn.Module):
    """Freeze parameter optimization while retaining gradients through inputs."""
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def gradient_norm(module: nn.Module) -> float:
    terms = [p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


class JointVisualLearner:
    """Short-block co-optimization of the RGB encoder and goal-conditioned M.

    External sampling remains the Runner's responsibility. This class does not
    select checkpoints, touch test data, restart a run, or change an environment.
    Use ``warmup`` then ``calibrate`` before joint updates. The coordinate buffers
    stay fixed thereafter; both parameter blocks enforce physical calibration.
    """

    checkpoint_format = "joint_visual_v2_1"

    def __init__(self, features: NormalizedRGBFeatures, metric: GoalConditionedMetric,
                 adapter: VisualTaskAdapter | None = None,
                 config: JointVisualConfig | None = None):
        if features.latent_dim != metric.latent_dim:
            raise ValueError("Feature and metric dimensions differ")
        self.features, self.metric = features, metric
        self.adapter, self.config = adapter or VisualTaskAdapter(), config or JointVisualConfig()
        self.feature_optimizer = torch.optim.Adam(features.parameters(), lr=self.config.feature_lr)
        self.metric_optimizer = torch.optim.Adam(metric.parameters(), lr=self.config.metric_lr)
        self.counts = {"warmup": 0, "feature": 0, "metric": 0}
        self.version = 0
        # Deterministic feature coordinates/J: dropout and mutable normalization
        # statistics must not change during a frozen policy snapshot.
        self.features.eval()
        self.metric.eval()

    def policy(self, observation: VisualObservation, *, create_graph: bool = True):
        return self.adapter.policy(self.features, self.metric, observation, create_graph=create_graph)

    def _zero(self) -> None:
        self.feature_optimizer.zero_grad(set_to_none=True)
        self.metric_optimizer.zero_grad(set_to_none=True)

    def warmup(self, batch: VisualBatch) -> dict[str, float | int]:
        batch.validate()
        if bool(self.features.calibrated):
            raise RuntimeError("Standalone AE warmup is only allowed before coordinate calibration")
        self._zero()
        images = torch.cat((batch.observation.image, batch.observation.goal_image))
        loss = F.mse_loss(self.features.reconstruct(images), images.detach())
        loss.backward()
        norm = self._step(self.features, self.feature_optimizer)
        self.counts["warmup"] += 1
        self.version += 1
        return {"loss": float(loss.detach()), "feature_gradient_norm": norm, "version": self.version}

    def calibrate(self, batch: VisualBatch) -> None:
        batch.validate()
        if not self.counts["warmup"] or self.counts["feature"] or self.counts["metric"]:
            raise RuntimeError("Calibrate after within-run AE warmup and before joint updates")
        self.features.calibrate(batch.observation.image, batch.observation.goal_image,
                                self.adapter.physical_error(batch.offset.detach()))
        self.version += 1

    def same_version_difference(self, batch: VisualBatch) -> Tensor:
        """Recompute both endpoints with current phi/M/normalization, no cache."""
        goal = self.features.encode(batch.observation.goal_image)
        error = self.features.encode(batch.observation.image) - goal
        following = self.features.encode(batch.next_image) - goal
        metric = self.metric(goal)
        return quadratic_value(following, metric) - quadratic_value(error, metric)

    def losses(self, batch: VisualBatch, *, feature_graph: bool = True) -> dict[str, Tensor]:
        batch.validate()
        if not bool(self.features.calibrated):
            raise RuntimeError("Joint optimization requires explicitly calibrated coordinates")
        cfg = self.config
        out = self.policy(batch.observation, create_graph=feature_graph)
        offset, next_offset = batch.offset.detach(), batch.next_offset.detach()
        physical = self.adapter.physical_error(offset)
        physical_next = self.adapter.physical_error(next_offset)
        dt = batch.dt.detach()
        counterfactual = self.adapter.nominal_step(offset, out.action, dt)
        predicted_error = self.adapter.physical_error(counterfactual)
        previous = (torch.zeros_like(out.action) if batch.previous_action is None
                    else batch.previous_action.detach())
        # This is a one-step nominal-model loss, NOT an observed outcome for a
        # new action. Observed physical outcomes are returned as diagnostics.
        task_loss = (predicted_error / (physical + cfg.physical_floor)
                     + cfg.input_weight * out.action.square().sum(-1) * dt
                     + cfg.action_change_weight * (out.action - previous).square().sum(-1)).mean()
        next_error = self.features.encode(batch.next_image.detach()) - out.goal
        next_value = quadratic_value(next_error, out.metric)

        def cal(value: Tensor, truth: Tensor) -> Tensor:
            denominator = truth + cfg.physical_floor
            return ((F.relu(cfg.value_lower * truth - value) / denominator).square()
                    + (F.relu(value - cfg.value_upper * truth) / denominator).square()).mean()

        calibration = 0.5 * (cal(out.value, physical) + cal(next_value, physical_next))
        observed_feature_change = next_error - out.error
        predicted_feature_change = (out.interaction @ batch.applied_action.detach().unsqueeze(-1)).squeeze(-1) * dt[:, None]
        movement_scale = batch.applied_action.detach().norm(dim=-1) * dt + cfg.motion_floor
        interaction = ((observed_feature_change - predicted_feature_change)
                       / movement_scale[:, None]).square().mean()
        difference = next_value - out.value
        residual = difference + cfg.decrease_per_second * dt * physical
        mask = batch.controller_transition
        decrease = ((F.relu(residual[mask]) / (physical[mask] + cfg.physical_floor)).square().mean()
                    if mask.any() else out.value.sum() * 0.0)
        images = torch.cat((batch.observation.image, batch.observation.goal_image))
        reconstruction = F.mse_loss(self.features.reconstruct(images), images.detach())
        # Bounds are applied to within-goal differences, not just across-room
        # latent variance. Calibration supplies a physically anchored lower bound.
        spread = out.error.square().mean(0).add(1e-12).sqrt()
        geometry = (F.relu(cfg.feature_std_lower - spread).square()
                    + F.relu(spread - cfg.feature_std_upper).square()).mean()
        total = (cfg.physical_weight * task_loss + cfg.calibration_weight * calibration
                 + cfg.interaction_weight * interaction + cfg.decrease_weight * decrease
                 + cfg.reconstruction_weight * reconstruction + cfg.geometry_weight * geometry)
        feasible = 0.5 * self.metric.floor * out.error.square().sum(-1) <= cfg.value_upper * physical
        return {"total": total, "physical": task_loss, "calibration": calibration,
                "interaction": interaction, "decrease": decrease, "reconstruction": reconstruction,
                "geometry": geometry, "actual_V_difference_mean": difference.detach().mean(),
                "actual_physical_error_mean": physical_next.mean(),
                "predicted_physical_error_mean": predicted_error.detach().mean(),
                "calibration_necessary_feasible_fraction": feasible.float().mean(),
                "controller_transition_fraction": mask.float().mean()}

    def _step(self, module: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        norm = gradient_norm(module)
        if not math.isfinite(norm):
            self._zero()
            raise FloatingPointError("Nonfinite parameter gradient; optimizer was not stepped")
        torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        return norm

    def _update(self, batch: VisualBatch, *, feature: bool) -> dict[str, float | int | bool]:
        self._zero()
        frozen = self.metric if feature else self.features
        trained = self.features if feature else self.metric
        optimizer = self.feature_optimizer if feature else self.metric_optimizer
        with frozen_parameters(frozen):
            terms = self.losses(batch, feature_graph=feature)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError("Nonfinite joint loss; optimizer was not stepped")
            terms["total"].backward()
            if any(p.grad is not None for p in frozen.parameters()):
                raise RuntimeError("Unexpected gradients in frozen parameter block")
            norm = self._step(trained, optimizer)
        report = {key: float(value.detach()) for key, value in terms.items()}
        report.update(gradient_norm=norm, feature_gradient_norm=norm if feature else 0.0,
                      metric_gradient_norm=0.0 if feature else norm,
                      behavior_matches_preupdate_version=batch.behavior_version == self.version)
        self.counts["feature" if feature else "metric"] += 1
        self.version += 1
        report["version"] = self.version
        return report

    def update_features(self, batch: VisualBatch) -> dict[str, Any]:
        return self._update(batch, feature=True)

    def update_metric(self, batch: VisualBatch) -> dict[str, Any]:
        return self._update(batch, feature=False)

    def train_blocks(self, batch: VisualBatch, *, feature_steps: int = 1,
                     metric_steps: int = 1) -> list[dict[str, Any]]:
        if feature_steps < 0 or metric_steps < 0 or feature_steps + metric_steps == 0:
            raise ValueError("Choose nonnegative block counts with at least one update")
        return ([dict(block="feature", **self.update_features(batch)) for _ in range(feature_steps)]
                + [dict(block="metric", **self.update_metric(batch)) for _ in range(metric_steps)])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": self.checkpoint_format, "features": self.features.state_dict(),
                   "metric": self.metric.state_dict(), "feature_optimizer": self.feature_optimizer.state_dict(),
                   "metric_optimizer": self.metric_optimizer.state_dict(), "counts": self.counts.copy(),
                   "version": self.version, "config": asdict(self.config),
                   "control_config": asdict(self.adapter.config), "metric_floor": self.metric.floor,
                   "torch_rng": torch.random.get_rng_state(), "numpy_rng": np.random.get_state(),
                   "python_rng": random.getstate(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load(self, path: str | Path, *, restore_rng: bool = True) -> None:
        """Restore a trusted local checkpoint into the same declared architecture."""
        payload = torch.load(path, map_location=next(self.features.parameters()).device, weights_only=False)
        if (payload.get("format") != self.checkpoint_format or payload["config"] != asdict(self.config)
                or payload["control_config"] != asdict(self.adapter.config)
                or payload["metric_floor"] != self.metric.floor):
            raise ValueError("Checkpoint protocol/config mismatch")
        self.features.load_state_dict(payload["features"], strict=True)
        self.metric.load_state_dict(payload["metric"], strict=True)
        self.feature_optimizer.load_state_dict(payload["feature_optimizer"])
        self.metric_optimizer.load_state_dict(payload["metric_optimizer"])
        self.counts, self.version = payload["counts"].copy(), int(payload["version"])
        if restore_rng:
            torch.random.set_rng_state(payload["torch_rng"].cpu())
            np.random.set_state(payload["numpy_rng"])
            random.setstate(payload["python_rng"])
            if payload["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng"]])


def engineering_fixture(seed: int = 7, *, batch_size: int = 3,
                        image_shape: tuple[int, int, int] = (3, 8, 8),
                        dtype: torch.dtype = torch.float64) -> tuple[JointVisualLearner, VisualBatch]:
    """Finite arithmetic fixture, deliberately NOT a rendered physical task.

    Images and local image directions are generated to exercise derivatives.
    Only offsets follow P0. No feature/performance correspondence is claimed.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        features = NormalizedRGBFeatures(TinyRGBBackbone(image_shape, 4, 12), 4).to(dtype=dtype)
        metric = GoalConditionedMetric(4, 12).to(dtype=dtype)
        learner = JointVisualLearner(features, metric)
        image = torch.rand(batch_size, *image_shape, dtype=dtype)
        goal = torch.rand_like(image)
        tangents = torch.randn(batch_size, 4, *image_shape, dtype=dtype) * 0.02
        action = torch.randn(batch_size, 4, dtype=dtype) * 0.15
        dt = torch.full((batch_size,), 0.1, dtype=dtype)
        following = (image + torch.einsum("bachw,ba->bchw", tangents, action) * dt[:, None, None, None]).detach()
        offset = torch.randn(batch_size, 4, dtype=dtype) * 0.1
        next_offset = learner.adapter.nominal_step(offset, action, dt).detach()
        batch = VisualBatch(VisualObservation(image, goal, tangents), following, offset, next_offset,
                            action, dt, torch.zeros(batch_size, dtype=torch.bool))
    return learner, batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a short arithmetic/gradient check, not training performance")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Only the explicit --smoke entry point is provided; sampling belongs to a Runner")
    torch.set_num_threads(1)
    learner, batch = engineering_fixture()
    learner.warmup(batch)
    learner.calibrate(batch)
    report = learner.train_blocks(batch)
    print(json.dumps({"scope": "synthetic arithmetic fixture; no renderer or performance claim",
                      "counts": learner.counts, "updates": report}, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["TinyRGBBackbone", "NormalizedRGBFeatures", "GoalConditionedMetric", "VisualBatch",
           "JointVisualConfig", "JointVisualLearner", "frozen_parameters", "engineering_fixture"]
