from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" in cfg:
        cfg = merge(load_config(path.parent / cfg.pop("extends")), cfg)
    if cfg["encoder"]["input_channels"] != 3 or cfg["encoder"]["latent_dim"] not in (16, 32):
        raise ValueError("This RGB AEVS protocol supports 16 or 32 latent dimensions")
    if cfg["encoder"]["input_size"] != 128:
        raise ValueError("The inherited RGB AEVS processes 128x128 images")
    if cfg["plant"]["translation_cap_m_s"] != 0.4 or cfg["plant"]["yaw_cap_rad_s"] != 0.35:
        raise ValueError("Inherited plant and image tangents fix action scales at 0.4 m/s and 0.35 rad/s")
    if cfg["training"]["encoder_updates_from_actor"]:
        raise ValueError("The experiment requires actor -> encoder gradient isolation")
    if cfg["plant"]["dt"] != 0.1:
        raise ValueError("The inherited simulator and interaction model use dt=0.1 seconds")
    if cfg["plant"]["training_plant"] != "P0":
        raise ValueError("P1 has hidden actuator state; v1 trains P0 and reports P1 only as empirical stress testing")
    if cfg["ablation"] not in ("lac_ae", "ae_only", "critic_ae_no_drift", "identity_m"):
        raise ValueError("Unknown ablation")
    return cfg


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def json_append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def source_hashes():
    files = [ROOT / "run.py", *sorted((ROOT / "lac_vs").glob("*.py"))]
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
