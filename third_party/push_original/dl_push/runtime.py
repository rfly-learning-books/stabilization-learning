"""Resolve the bundled, unchanged reference implementation explicitly."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"


def activate_reference(mjx_backend=None) -> Path:
    if not (REFERENCE / "starter/mjx_one_step.py").is_file():
        raise FileNotFoundError(f"Missing reference bundle at {REFERENCE}")
    requested = mjx_backend or os.environ.get("JAX_PLATFORMS", "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Choose a single explicit MJX backend: cpu or cuda")
    if "jax" in sys.modules:
        actual = sys.modules["jax"].default_backend()
        expected = "gpu" if requested == "cuda" else "cpu"
        if actual != expected:
            raise RuntimeError("JAX was initialized on another backend; start a new process")
    os.environ["JAX_PLATFORMS"] = requested
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for name in ("starter", "ca_glpr_unseen_shape", "mani_skill"):
        for key, module in list(sys.modules.items()):
            if key == name or key.startswith(name + "."):
                origin = getattr(module, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(REFERENCE):
                    raise RuntimeError(f"Conflicting import {key}: {origin}. Start a new process.")
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    for name in ("starter", "ca_glpr_unseen_shape"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(REFERENCE):
            raise RuntimeError(f"Wrong import source: {name}")
    return REFERENCE


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Atomically publish generated metadata, without partial JSON on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temp = Path(stream.name)
    os.replace(temp, path)


def code_hashes():
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted((ROOT / "dl_push").glob("*.py"))
    }
