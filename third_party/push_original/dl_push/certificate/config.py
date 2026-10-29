"""Versioned configuration for the new sampling-first controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

from .calibration import CalibrationConfig
from .q_generator import QConfig, SearchConfig
from .runner import RolloutConfig
from .solver import CLFConfig


@dataclass
class CertificateExperimentConfig:
    schema: int = 1
    asset_root: str = "assets/native_v1"
    source_shapes: list[str] = field(default_factory=lambda: ["T", "Square", "RectWide", "Plus", "H"])
    validation_shapes: list[str] = field(default_factory=lambda: ["C_wide", "C_narrow"])
    sealed_test_shapes: list[str] = field(default_factory=lambda: ["L", "Z"])
    boundary_candidates: int = 64
    selected_modes: int = 8
    q: QConfig = field(default_factory=QConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    clf: CLFConfig = field(default_factory=CLFConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self):
        if self.schema != 1:
            raise ValueError("unsupported certificate config schema")
        if self.boundary_candidates < 1 or self.selected_modes < 1:
            raise ValueError("mode counts must be positive")
        all_names = self.source_shapes + self.validation_shapes + self.sealed_test_shapes
        if len(set(all_names)) != len(all_names):
            raise ValueError("shape splits must be disjoint")

    def to_dict(self):
        value = asdict(self)
        return value

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")
        return path


def _nested(cls, data):
    return cls(**(data or {}))


def load_config(path):
    path = Path(path)
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("certificate config must be a JSON object")
    return CertificateExperimentConfig(
        schema=values.get("schema", 1),
        asset_root=values.get("asset_root", "assets/native_v1"),
        source_shapes=list(values.get("source_shapes", ["T", "Square", "RectWide", "Plus", "H"])),
        validation_shapes=list(values.get("validation_shapes", ["C_wide", "C_narrow"])),
        sealed_test_shapes=list(values.get("sealed_test_shapes", ["L", "Z"])),
        boundary_candidates=int(values.get("boundary_candidates", 64)),
        selected_modes=int(values.get("selected_modes", 8)),
        q=_nested(QConfig, values.get("q")),
        calibration=_nested(CalibrationConfig, values.get("calibration")),
        clf=_nested(CLFConfig, values.get("clf")),
        rollout=_nested(RolloutConfig, values.get("rollout")),
        search=_nested(SearchConfig, values.get("search")),
    )

