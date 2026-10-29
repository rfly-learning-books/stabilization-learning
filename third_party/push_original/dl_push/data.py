"""Complete episodes, explicit split roles and hash-checked replay."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import numpy as np

from .geometry import Geometry
from .runtime import read_json, sha256, write_json
from .state import CONTEXT_DIM


def save_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite generated data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        np.savez_compressed(stream, **arrays)
        temp = Path(stream.name)
    os.replace(temp, path)


def save_episode(directory, arrays, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_npz(directory / "trajectory.npz", **arrays)
    metadata = dict(metadata, schema=1, trajectory_sha256=sha256(directory / "trajectory.npz"))
    write_json(directory / "episode.json", metadata)
    return directory / "episode.json"


@dataclass
class Episode:
    path: Path
    metadata: dict
    arrays: dict
    geometry: Geometry

    @property
    def length(self):
        return len(self.arrays["actions"])


def load_episode(metadata_path, geometries=None):
    path = Path(metadata_path)
    meta = read_json(path)
    if meta.get("schema") != 1:
        raise ValueError(f"Unknown trajectory schema: {path}")
    trajectory = path.parent / "trajectory.npz"
    if sha256(trajectory) != meta["trajectory_sha256"]:
        raise ValueError(f"Trajectory checksum mismatch: {trajectory}")
    keys = ("states", "actions", "nominal", "contexts", "contacts")
    with np.load(trajectory, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in keys}
        # Optional control-side diagnostics.  They are stored for auditing and
        # future phase-aware weighting, but are not required by the residual
        # observer training objective in this version.
        for key in (
            "phases",
            "contact_forces",
            "contact_normals",
            "contact_points",
            "predicted_contact_forces",
        ):
            if key in data:
                arrays[key] = data[key].copy()
    n = len(arrays["actions"])
    expected = {"states": (n + 1, 12), "actions": (n, 2),
                "nominal": (n, 12), "contexts": (n, CONTEXT_DIM), "contacts": (n,)}
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid episode field {key}: {path}")
    if np.any(np.abs(arrays["actions"]) > .300001):
        raise ValueError("Replay actions must be physical m/s")
    cache = geometries if geometries is not None else {}
    asset_path = (path.parent / meta["asset_relative_path"]).resolve()
    if str(asset_path) not in cache:
        cache[str(asset_path)] = Geometry(asset_path)
    geometry = cache[str(asset_path)]
    if geometry.fingerprint != meta["asset_hash"]:
        raise ValueError("Episode references a different geometry version")
    return Episode(path, meta, arrays, geometry)


def discover_episodes(roots):
    paths = set()
    for root in roots:
        path = Path(root)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            paths.add(path.resolve())
        else:
            paths.update(p.resolve() for p in path.rglob("episode.json"))
    return sorted(paths)


def split_episodes(roots, config):
    cache, training, validation = {}, [], []
    for path in discover_episodes(roots):
        ep = load_episode(path, cache)
        if ep.length < 2:
            continue  # Still reported in the dataset manifest, unusable for adaptation training.
        role, shape = ep.metadata["role"], ep.metadata["shape"]
        if role == "train":
            if shape not in config.train_shapes:
                raise ValueError(f"Training leakage: {shape} outside configured train pool")
            training.append(ep)
        elif role == "validation":
            if shape not in set(config.train_shapes + config.validation_shapes):
                raise ValueError("Validation geometry outside declared development pools")
            validation.append(ep)
        # Test trajectories never participate in parameter fitting or model selection.
    if not training or not validation:
        raise ValueError("Need train and validation episodes of >=2 steps; collect enough complete episodes")
    return training, validation
