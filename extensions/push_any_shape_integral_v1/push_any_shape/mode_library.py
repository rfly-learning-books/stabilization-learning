"""Read-only preserved measured C maps and an explicit nominal Disk C map.

C[:, i] maps nonnegative inward pusher speed [m/s] to body-frame object twist
[m/s, m/s, rad/s]. No column normalization or synthetic measurement is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .shapes import DEFAULT_LIBRARY_ROOT, DISK_NAME, DISK_RADIUS_M, file_sha256


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _immutable(value, dtype=float):
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ModeSet:
    shape: str
    c: np.ndarray
    contact_points: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    valid: np.ndarray
    metadata: dict
    fingerprint: str
    measured: bool

    @property
    def C(self):
        return self.c

    @property
    def count(self):
        return self.c.shape[1]

    @property
    def response_kind(self):
        return "preserved_measured_response" if self.measured else "nominal_ideal_no_slip_not_calibrated"


def disk_modes():
    angles = np.arange(12, dtype=float) * (2. * np.pi / 12.)
    normals = np.stack((np.cos(angles), np.sin(angles)), -1)
    tangents = np.stack((-normals[:, 1], normals[:, 0]), -1)
    c = np.vstack((-normals.T, np.zeros(12)))
    metadata = {
        "protocol": "push_any_shape_integral_v1_disk_nominal",
        "response_source": "analytic ideal no-slip radial translation assumption",
        "is_measured": False, "is_calibrated": False,
        "physical_validation": "pending; actual contact response may differ",
        "input": "absolute inward normal pusher command u [m/s]",
        "response": "body COM twist = C[:, mode] * u; yaw component is nominally zero",
        "normal_convention": "outward body normal; positive input pushes inward",
        "disk_radius_m": DISK_RADIUS_M, "pusher_radius_m": .004,
        "speed_domain_kind": "declared execution domain, not calibrated bounds",
        "speed_domain_mps": [0., .06],
        "yaw_target": "free; observe yaw and yaw rate for geometry/safety",
    }
    payload = {"shape": DISK_NAME, "c": c.tolist(), "normals": normals.tolist(), "metadata": metadata}
    return ModeSet(DISK_NAME, _immutable(c), _immutable(DISK_RADIUS_M * normals),
        _immutable(normals), _immutable(tangents), _immutable(np.zeros(12)),
        _immutable(np.full(12, .06)), _immutable(np.ones(12), bool), metadata, _digest(payload), False)


def load_mode_library(shape_name, library_root=DEFAULT_LIBRARY_ROOT):
    if shape_name == DISK_NAME:
        return disk_modes()
    folder = Path(library_root).resolve() / shape_name / "library"
    record_path, arrays_path = folder / "library.json", folder / "library.npz"
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    with np.load(arrays_path, allow_pickle=False) as values:
        arrays = {key: values[key].copy() for key in values.files}
    if record["shape"] != shape_name:
        raise ValueError("Mode library name mismatch")
    # This is the original ModeLibrary fingerprint payload, including its
    # default JSON ASCII convention. It checks provenance without importing it.
    payload = {"shape": record["shape"], "c": arrays["c"].tolist(),
        "contact_points": arrays["contact_points"].tolist(), "normals": arrays["normals"].tolist(),
        "u_min": arrays["u_min"].tolist(), "u_max": arrays["u_max"].tolist(),
        "valid": arrays["valid"].astype(int).tolist(),
        "face_normals": arrays["face_normals"].tolist(), "face_mask": arrays["face_mask"].astype(int).tolist(),
        "contact_offset": arrays["contact_offset"].tolist(), "mode_type": arrays["mode_type"].tolist(),
        "metadata": record["metadata"]}
    if _digest(payload) != record["fingerprint"]:
        raise ValueError("Preserved mode library fingerprint mismatch")
    c = arrays["c"]
    if c.shape != (3, 12) or not np.isfinite(c).all():
        raise ValueError("Expected finite unnormalized original C[3,12]")
    metadata = dict(record["metadata"])
    metadata["preserved_source"] = {"json": str(record_path), "json_sha256": file_sha256(record_path),
        "npz": str(arrays_path), "npz_sha256": file_sha256(arrays_path)}
    return ModeSet(shape_name, _immutable(c), _immutable(arrays["contact_points"]),
        _immutable(arrays["normals"]), _immutable(arrays["tangents"]),
        _immutable(arrays["u_min"]), _immutable(arrays["u_max"]), _immutable(arrays["valid"], bool),
        metadata, record["fingerprint"], True)


load_library = load_mode_library


class ModeBank:
    """Original geometry-mask layout, dynamically sized to all 33 entries."""
    def __init__(self, root=DEFAULT_LIBRARY_ROOT, device="cpu", dtype=None):
        import torch
        from .shapes import catalog
        self.root = Path(root).resolve()
        self.specs = catalog(self.root)
        self.names = list(self.specs)
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        self.libraries = [load_mode_library(name, self.root) for name in self.names]
        self.C = torch.as_tensor(np.stack([lib.c.T for lib in self.libraries]), device=self.device, dtype=self.dtype)
        self.valid = torch.as_tensor(np.stack([lib.valid for lib in self.libraries]), device=self.device)
        self.speed_lower = torch.as_tensor(np.stack([lib.u_min for lib in self.libraries]), device=self.device, dtype=self.dtype)
        self.speed_upper = torch.as_tensor(np.stack([lib.u_max for lib in self.libraries]), device=self.device, dtype=self.dtype)
        self.task_mask = torch.as_tensor([self.specs[name].task_mask for name in self.names], device=self.device, dtype=self.dtype)
        self.rotation_free = torch.as_tensor([self.specs[name].yaw_free for name in self.names], device=self.device)
        self.fingerprints = {lib.shape: lib.fingerprint for lib in self.libraries}
        self.response_kinds = {lib.shape: lib.response_kind for lib in self.libraries}
        rings = [np.asarray(self.specs[name].polygon.exterior.coords)[:-1] for name in self.names]
        width = max(map(len, rings))
        vertices = np.zeros((len(self.names), width, 2))
        mask = np.zeros((len(self.names), width), bool)
        for index, ring in enumerate(rings):
            vertices[index, :len(ring)] = ring
            mask[index, :len(ring)] = True
        self.vertices = torch.as_tensor(vertices, device=self.device, dtype=self.dtype)
        self.vertex_mask = torch.as_tensor(mask, device=self.device)
