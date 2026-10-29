"""Single-writer evidence and complete-update checkpoints for new v2 runs.

No process launching, automatic recovery, checkpoint selection or budget extension
is hidden in this module. A caller must own the lease and supply its full state.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Mapping
import uuid

import numpy as np
import torch


class RunStopped(RuntimeError):
    pass


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    """One owner; a reader sees either the complete old or complete new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG checkpoint requires the original CUDA execution backend")
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.protocol = json.loads((self.root / "protocol.json").read_text(encoding="utf-8"))
        self.protocol_hash = file_sha256(self.root / "protocol.json")
        self.token = None
        self.sequence = 0

    @classmethod
    def create(cls, root, config, *, max_hours, sources=(), assets=(), minimum_free_gib=20., deadline_unix=None):
        if not math.isfinite(max_hours) or max_hours <= 0:
            raise ValueError("A positive finite wall-clock cap is required")
        if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
            raise ValueError("Invalid disk reserve")
        if deadline_unix is not None and (not math.isfinite(deadline_unix) or deadline_unix <= time.time()):
            raise RunStopped("Original shared deadline has already passed or is invalid")
        source_hashes = {str(Path(p).resolve()): file_sha256(p) for p in sources}
        asset_hashes = {str(Path(p).resolve()): file_sha256(p) for p in assets}
        root = Path(root).resolve()
        # Validate JSON before claiming a new output directory.
        json.dumps(config, allow_nan=False)
        root.mkdir(parents=True, exist_ok=False)
        started = time.time()
        deadline = min(started + max_hours * 3600, deadline_unix) if deadline_unix is not None else started + max_hours * 3600
        atomic_json(root / "protocol.json", dict(schema="joint_certificate_v2", config=config,
                    created_utc=datetime.now(timezone.utc).isoformat(), started_unix=started,
                    deadline_unix=deadline, max_hours=max_hours,
                    original_shared_deadline_unix=deadline_unix,
                    minimum_free_gib=minimum_free_gib, sources=source_hashes, assets=asset_hashes))
        (root / "events").mkdir()
        return cls(root)

    def acquire(self):
        if self.token is not None:
            raise RuntimeError("This writer already owns a lease")
        token = uuid.uuid4().hex
        lease = dict(pid=os.getpid(), token=token, started_unix=time.time())
        # A stale lease is evidence of an interrupted owner; never silently steal it.
        with (self.root / "WRITER.lock").open("x", encoding="utf-8") as stream:
            json.dump(lease, stream)
            stream.flush()
            os.fsync(stream.fileno())
        self.token = token
        try:
            existing = list((self.root / "events").glob("*.json"))
            self.sequence = max((int(p.name.split("_")[0]) for p in existing), default=0)
            self.verify_inputs()
            self.guard()
        except BaseException:
            self.close()
            raise
        return self

    def _owned(self):
        if not self.token:
            raise RuntimeError("Acquire the exclusive writer lease first")
        lease = json.loads((self.root / "WRITER.lock").read_text(encoding="utf-8"))
        if lease.get("token") != self.token or lease.get("pid") != os.getpid():
            raise RuntimeError("Writer lease was changed")

    def guard(self):
        self._owned()
        if file_sha256(self.root / "protocol.json") != self.protocol_hash:
            raise RuntimeError("Frozen protocol changed")
        if (self.root / "STOP").exists():
            raise RunStopped("STOP present")
        if time.time() >= self.protocol["deadline_unix"]:
            raise RunStopped("Original wall-clock deadline reached")
        if shutil.disk_usage(self.root).free < self.protocol["minimum_free_gib"] * 1024**3:
            raise RunStopped("Disk reserve reached")

    def verify_inputs(self):
        if file_sha256(self.root / "protocol.json") != self.protocol_hash:
            raise RuntimeError("Frozen protocol changed")
        for group in ("sources", "assets"):
            for path, expected in self.protocol.get(group, {}).items():
                if file_sha256(path) != expected:
                    raise RuntimeError("Frozen input changed: " + path)

    def event(self, kind, **data):
        # Final diagnostic events are allowed after STOP/deadline, under the lease.
        self._owned()
        if not kind.replace("_", "").isalnum():
            raise ValueError("Event kind must be a simple name")
        self.sequence += 1
        path = self.root / "events" / f"{self.sequence:08d}_{kind}.json"
        if path.exists():
            raise FileExistsError("Committed events cannot be overwritten")
        record = dict(sequence=self.sequence, kind=kind, unix=time.time(),
                      protocol_sha256=self.protocol_hash, data=data)
        atomic_json(path, record)
        return path

    def checkpoint(self, name, payload, *, completed_updates):
        self.guard()
        if Path(name).name != name or not name.endswith(".pt") or completed_updates < 0:
            raise ValueError("Checkpoint name/update invalid")
        # Caller supplies models, optimizers, replay/env and private-generator state.
        required = {"models", "optimizers", "runtime_state"}
        if not required.issubset(payload):
            raise ValueError("Checkpoint requires models, optimizers and runtime_state")
        path = self.root / name
        # Immutable versioned checkpoints: never replace a partial/previous result.
        if path.exists() or path.with_suffix(".COMMITTED.json").exists():
            raise FileExistsError(path)
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            raise FileExistsError("Preserve interrupted checkpoint temporary file")
        torch.save(dict(schema="joint_checkpoint_v2", protocol_sha256=self.protocol_hash,
                        completed_updates=int(completed_updates), boundary="complete_update",
                        rng=capture_rng(), payload=payload), temporary)
        os.replace(temporary, path)
        marker = dict(file=path.name, sha256=file_sha256(path), completed_updates=int(completed_updates),
                      boundary="complete_update", protocol_sha256=self.protocol_hash)
        atomic_json(path.with_suffix(".COMMITTED.json"), marker)
        self.event("checkpoint_committed", **marker)
        return path

    def load_checkpoint(self, name, *, map_location="cpu"):
        self.guard()
        self.verify_inputs()
        if Path(name).name != name:
            raise ValueError("Checkpoint must belong to this run")
        path = self.root / name
        marker = json.loads(path.with_suffix(".COMMITTED.json").read_text(encoding="utf-8"))
        if (marker["protocol_sha256"] != self.protocol_hash or marker["boundary"] != "complete_update"
                or file_sha256(path) != marker["sha256"]):
            raise RuntimeError("Checkpoint protocol/boundary/hash mismatch")
        # Only load the locally produced trusted checkpoint after checking its commit hash.
        ck = torch.load(path, map_location=map_location, weights_only=False)
        if (ck["protocol_sha256"] != self.protocol_hash or ck["boundary"] != "complete_update"
                or ck["completed_updates"] != marker["completed_updates"]):
            raise RuntimeError("Checkpoint metadata mismatch")
        return ck

    def freeze(self, name, *, selected_update, selection):
        self.guard()
        ck = self.load_checkpoint(name)
        if selected_update <= 0 or ck["completed_updates"] != selected_update:
            raise ValueError("Select an actually updated checkpoint; retain update zero as baseline")
        if (self.root / "FROZEN.json").exists():
            raise FileExistsError("Model selection is already frozen")
        atomic_json(self.root / "FROZEN.json", dict(checkpoint=name, sha256=file_sha256(self.root / name),
                    selected_update=selected_update, selection=selection,
                    protocol_sha256=self.protocol_hash))
        self.event("selection_frozen", checkpoint=name, selected_update=selected_update)

    @staticmethod
    def require_frozen_group(roots, *, expected_seeds=None):
        roots = list(roots)
        if not roots:
            raise ValueError("Declare the entire model group before testing")
        markers = []
        seeds = []
        for root in roots:
            root = Path(root).resolve()
            marker = json.loads((root / "FROZEN.json").read_text(encoding="utf-8"))
            path = (root / marker["checkpoint"]).resolve()
            if path.parent != root or file_sha256(path) != marker["sha256"]:
                raise RuntimeError("Frozen selection mismatch")
            if file_sha256(root / "protocol.json") != marker["protocol_sha256"]:
                raise RuntimeError("Frozen protocol mismatch")
            protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
            seeds.append(protocol.get("config", {}).get("seed"))
            markers.append(marker)
        if expected_seeds is not None:
            expected = list(expected_seeds)
            if (not expected or len(set(expected)) != len(expected) or len(set(seeds)) != len(seeds)
                    or set(seeds) != set(expected)):
                raise RuntimeError("Entire predeclared seed group must be frozen before testing")
        return markers

    def close(self):
        if self.token:
            self._owned()
            (self.root / "WRITER.lock").unlink()
            self.token = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.close()

