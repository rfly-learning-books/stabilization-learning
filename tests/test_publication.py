"""Regression checks for the public source distribution."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_public_configs_do_not_bind_author_machine():
    for path in (ROOT / "configs/public").glob("*.json"):
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        assert "runtime_lock" not in config
        assert "D:/" not in path.read_text(encoding="utf-8-sig")

def test_compound_object_geometries_all_registered(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("shapely")
    from extensions.reproducible_training_v4 import so101
    def initialize(self, *args, **kwargs):
        self.model = SimpleNamespace(geom_bodyid=np.array([0, 7, 7, 4, 7, 7]))
        self.env = SimpleNamespace(_block_body=7)
        self.block_geoms = {1, 2}
    monkeypatch.setattr(so101.campaign.Plant, "__init__", initialize)
    instance = so101.ContinuousHoldPlant(None, 0)
    assert instance.block_geoms == {1, 2, 4, 5}
