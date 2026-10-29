"""Fixed-scale C-only certificate generators and black-box Q search."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .math import matrix_hash, q_from_parameters, validate_q


@dataclass(frozen=True)
class QConfig:
    trace: float = 3.0
    eigenvalue_floor: float = 0.05

    def __post_init__(self):
        if not np.isfinite(self.trace) or self.trace <= 0:
            raise ValueError("Q trace must be positive")
        if not np.isfinite(self.eigenvalue_floor) or not 0 < self.eigenvalue_floor < 1:
            raise ValueError("Q eigenvalue floor must lie in (0,1)")


class CertificateGenerator:
    """Stateless five-parameter certificate generator."""

    def __init__(self, config: QConfig | None = None):
        self.config = config or QConfig()

    def __call__(self, parameters):
        return q_from_parameters(
            parameters,
            eigenvalue_floor=self.config.eigenvalue_floor,
            trace=self.config.trace,
        )

    def identity(self):
        return np.eye(3, dtype=np.float64) * (self.config.trace / 3.0)

    def record(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        Q = self(parameters)
        return {
            "schema": 1,
            "parameters": parameters.tolist(),
            "Q": Q.tolist(),
            "Q_hash": matrix_hash(Q),
            "trace": float(np.trace(Q)),
            "eigenvalues": np.linalg.eigvalsh(Q).tolist(),
            "config": {
                "trace": self.config.trace,
                "eigenvalue_floor": self.config.eigenvalue_floor,
            },
        }

    def save(self, path, parameters):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record(parameters), indent=2) + "\n")
        return path


class SetQGenerator(nn.Module):
    """Permutation-invariant PyTorch C-only network.

    The network consumes only raw mode responses and a validity mask.  It has
    no shape name, task ID, or trajectory feedback input.  Each deployment
    episode calls it exactly once and freezes the resulting Q.
    """

    def __init__(self, hidden=64, config: QConfig | None = None):
        super().__init__()
        self.config = config or QConfig()
        self.mode_net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden * 2 + 1, 5)

    def forward(self, C, valid):
        if C.ndim != 3 or C.shape[1] != 3:
            raise ValueError("C must have shape (B,3,N)")
        if valid.ndim != 2 or valid.shape != C.shape[:1] + C.shape[2:]:
            raise ValueError("valid must have shape (B,N)")
        values = C.transpose(1, 2)
        mask = valid.to(dtype=values.dtype).unsqueeze(-1)
        # Include one validity feature per mode; no shape identity feature.
        values = torch.cat([values, mask], dim=-1)
        embedded = self.mode_net(values)
        masked = embedded.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        maximum = masked.amax(dim=1)
        total = torch.clamp(mask.sum(dim=1), min=1.0)
        mean = (embedded * mask).sum(dim=1) / total
        summary = torch.cat([mean, maximum, total / values.shape[1]], dim=-1)
        parameters = self.head(summary)
        return q_from_parameters_torch(
            parameters,
            eigenvalue_floor=self.config.eigenvalue_floor,
            trace=self.config.trace,
        )


def q_from_parameters_torch(parameters, eigenvalue_floor=0.05, trace=3.0):
    """Torch-differentiable counterpart of :func:`q_from_parameters`."""

    import torch

    if parameters.shape[-1] != 5:
        raise ValueError("parameters must end in dimension 5")
    a, b, c, d, e = parameters.unbind(dim=-1)
    S = torch.stack(
        [a, b, c, b, d, e, c, e, -a - d], dim=-1
    ).reshape(*parameters.shape[:-1], 3, 3)
    values, vectors = torch.linalg.eigh(S)
    values = values - values.amax(dim=-1, keepdim=True)
    exponent = torch.exp(values)
    normalized = (vectors * exponent.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    normalized = normalized / torch.diagonal(normalized, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(3, dtype=parameters.dtype, device=parameters.device)
    return float(eigenvalue_floor) * eye + float(trace) * (1.0 - float(eigenvalue_floor)) * normalized


@dataclass(frozen=True)
class SearchConfig:
    population: int = 16
    generations: int = 40
    sigma: float = 1.0
    seed: int = 20260911
    restarts: int = 3
    elite_fraction: float = 0.5

    def __post_init__(self):
        if min(self.population, self.generations, self.restarts) < 1 or self.sigma <= 0:
            raise ValueError("Q search dimensions must be positive")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0,1]")


class DiagonalCMAES:
    """Small dependency-free diagonal CMA-ES for five certificate parameters."""

    def __init__(self, mean, config: SearchConfig, rng=None):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        if self.mean.size != 5 or not np.isfinite(self.mean).all():
            raise ValueError("CMA-ES mean must be a finite five-vector")
        self.config = config
        self.rng = rng or np.random.default_rng(config.seed)
        self.sigma = float(config.sigma)
        self.variance = np.ones(5, dtype=np.float64)
        self.generation = 0

    def ask(self):
        noise = self.rng.standard_normal((self.config.population, 5))
        return self.mean[None] + self.sigma * noise * np.sqrt(self.variance)[None]

    def tell(self, candidates, losses):
        candidates = np.asarray(candidates, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64).reshape(-1)
        if candidates.shape != (self.config.population, 5) or losses.shape != (self.config.population,):
            raise ValueError("CMA-ES candidates/losses have invalid shapes")
        if not np.isfinite(candidates).all() or not np.isfinite(losses).all():
            raise ValueError("CMA-ES inputs must be finite")
        count = max(1, int(np.ceil(self.config.population * self.config.elite_fraction)))
        elite = candidates[np.argsort(losses, kind="stable")[:count]]
        previous = self.mean.copy()
        self.mean = elite.mean(axis=0)
        centered = elite - previous[None]
        self.variance = 0.8 * self.variance + 0.2 * np.mean(centered * centered, axis=0) / max(self.sigma * self.sigma, 1e-12)
        self.variance = np.clip(self.variance, 1e-4, 1e4)
        spread = float(np.sqrt(np.mean(np.square(elite - self.mean[None]))))
        self.sigma = float(np.clip(0.85 * self.sigma + 0.15 * spread, 1e-3, 10.0))
        self.generation += 1


def search_q(objective, config: SearchConfig | None = None, initial=None):
    """Search Q parameters with an objective returning a finite scalar loss.

    ``objective`` is called once per candidate and should run the same sampled
    episodes and inner controller budget for every candidate.  The function
    returns every ask/tell record for reproducibility, not just the winner.
    """

    config = config or SearchConfig()
    initial = np.zeros(5, dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64)
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError("initial Q parameters must be a finite five-vector")
    best = None
    history = []
    for restart in range(config.restarts):
        rng = np.random.default_rng(config.seed + restart)
        mean = initial.copy() if restart == 0 else rng.normal(0.0, 0.5, size=5)
        optimizer = DiagonalCMAES(mean, config, rng=rng)
        for generation in range(config.generations):
            candidates = optimizer.ask()
            losses = np.asarray([float(objective(candidate)) for candidate in candidates])
            if not np.isfinite(losses).all():
                raise FloatingPointError("Q objective returned a non-finite value")
            optimizer.tell(candidates, losses)
            for index, (candidate, loss) in enumerate(zip(candidates, losses)):
                record = dict(restart=restart, generation=generation, index=index,
                              parameters=candidate.tolist(), loss=float(loss),
                              mean=optimizer.mean.tolist(), sigma=optimizer.sigma)
                history.append(record)
                if best is None or loss < best["loss"]:
                    best = record
    return {"best": best, "history": history, "config": config.__dict__}
