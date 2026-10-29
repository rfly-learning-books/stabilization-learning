"""Bounded raw RGB replay; geometry is cached, learned features never are.

The v2 archive and scene-balanced sampling protocol are deliberately retained.
Providers may implement ``observe_batch(publics)`` and ``encode_images(images)``.
A mutable geometry provider must advance ``cache_revision`` whenever its depth,
camera geometry or preprocessing changes. No phi, feature Jacobian or M is stored.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from extensions.joint_certificate_v2.rgb import RawRGBReplay
from stabilization_gym.tasks_v2.visual import VisualObservation
from stabilization_learning.joint_v2.visual import VisualBatch
from stabilization_learning.joint_v2.runtime import file_sha256


def _nonnegative_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _stat(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino


def _json_default(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"shape": array.shape, "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    raise TypeError(f"Geometry cache revision must be serializable: {type(value).__name__}")


class FastRawRGBReplay(RawRGBReplay):
    """Immutable shards with bounded LRU caches and batched device transfers.

    Counts and byte limits both apply. ``geometry_cache_device=None`` retains
    geometry on its computation device; ``'cpu'`` optionally uses pinned memory.
    Byte limits cover retained cache payload, not transient outputs/current batch.
    Shard SHA is checked on first read and again after any filesystem-stat change.
    The archive and active-row population retain the original v2 semantics.
    """
    def __init__(self, root, rng, *, max_episodes=96, cache_episodes=32,
                 geometry_cache=512, replay_cache_bytes=512 << 20,
                 geometry_cache_bytes=256 << 20, geometry_cache_device=None,
                 pin_memory=False):
        for name, value in (("max_episodes", max_episodes), ("cache_episodes", cache_episodes),
                            ("geometry_cache", geometry_cache), ("replay_cache_bytes", replay_cache_bytes),
                            ("geometry_cache_bytes", geometry_cache_bytes)):
            _nonnegative_integer(name, value)
        if max_episodes == 0:
            raise ValueError("max_episodes must be positive")
        super().__init__(root, rng, max_episodes=max_episodes,
                         cache_episodes=cache_episodes, geometry_cache=geometry_cache)
        self.replay_cache_bytes = replay_cache_bytes
        self.geometry_cache_bytes = geometry_cache_bytes
        self.geometry_cache_device = (None if geometry_cache_device is None
                                      else torch.device(geometry_cache_device))
        self.pin_memory = bool(pin_memory)
        self.cache_bytes = self.geometry_bytes = 0
        self._cache_signatures = {}
        self._verified = OrderedDict()
        self._verification_limit = max(256, 4 * max_episodes, cache_episodes)
        self._geometry_fingerprint = None

    def _forget_shard(self, name):
        if name in self.cache:
            self.cache_bytes -= sum(value.nbytes for value in self.cache.pop(name).values())
        self._cache_signatures.pop(name, None)
        self._verified.pop(name, None)
        for key in list(self.geometry_cache):
            if key[1] == name:
                values = self.geometry_cache.pop(key)
                self.geometry_bytes -= sum(t.numel() * t.element_size() for t in values)

    def _checked_signature(self, row):
        name = row["file"]
        if Path(name).name != name or Path(name).suffix != ".npz":
            raise ValueError("Raw replay shard must be a local npz filename")
        path = self.root / name
        stamp = _stat(path)
        signature = row["sha256"], stamp
        if self._verified.get(name) != signature:
            actual = file_sha256(path)
            if _stat(path) != stamp:
                self._forget_shard(name)
                raise RuntimeError("Raw replay shard changed during hash verification")
            if actual != row["sha256"]:
                self._forget_shard(name)
                raise RuntimeError("Raw replay shard hash mismatch")
            self._verified[name] = signature
            while len(self._verified) > self._verification_limit:
                self._verified.popitem(last=False)
        self._verified.move_to_end(name)
        return signature

    def load(self, row):
        name = row["file"]
        signature = self._checked_signature(row)
        if name in self.cache and self._cache_signatures[name] == signature:
            self.cache.move_to_end(name)
            return self.cache[name]
        if name in self.cache:
            self.cache_bytes -= sum(value.nbytes for value in self.cache.pop(name).values())
            self._cache_signatures.pop(name, None)
        path = self.root / name
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        if _stat(path) != signature[1]:
            self._forget_shard(name)
            raise RuntimeError("Raw replay shard changed during decoding")
        for value in data.values():
            value.setflags(write=False)
        size = sum(value.nbytes for value in data.values())
        if self.cache_limit and size <= self.replay_cache_bytes:
            self.cache[name] = data
            self._cache_signatures[name] = signature
            self.cache_bytes += size
            while len(self.cache) > self.cache_limit or self.cache_bytes > self.replay_cache_bytes:
                removed, arrays = self.cache.popitem(last=False)
                self.cache_bytes -= sum(value.nbytes for value in arrays.values())
                self._cache_signatures.pop(removed, None)
        return data

    def _fingerprint(self, geometry):
        revision = getattr(geometry, "cache_revision", None)
        if callable(revision):
            revision = revision()
        provenance = getattr(geometry, "provenance", {})
        if provenance.get("trainable", False) and revision is None:
            raise ValueError("Mutable geometry requires an explicit cache_revision")
        payload = {"provider": f"{type(geometry).__module__}.{type(geometry).__qualname__}",
                   "instance": id(geometry), "revision": revision,
                   "device": str(geometry.device), "cfg": getattr(geometry, "cfg", {}),
                   "provenance": provenance}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, default=_json_default,
                                                separators=(",", ":")).encode()).hexdigest()
        if fingerprint != self._geometry_fingerprint:
            self.geometry_cache.clear()
            self.geometry_bytes = 0
            self._geometry_fingerprint = fingerprint
        return fingerprint

    def _store_geometry(self, key, values):
        size = sum(value.numel() * value.element_size() for value in values)
        if not self.geometry_limit or size > self.geometry_cache_bytes:
            return
        device = self.geometry_cache_device or values[0].device
        # Independent storage avoids retaining a whole source batch through a view.
        stored = tuple(value.detach().to(device=device, copy=True).contiguous() for value in values)
        if self.pin_memory and device.type == "cpu" and torch.cuda.is_available():
            stored = tuple(value.pin_memory() for value in stored)
        self.geometry_cache[key] = stored
        self.geometry_bytes += size
        while len(self.geometry_cache) > self.geometry_limit or self.geometry_bytes > self.geometry_cache_bytes:
            _, removed = self.geometry_cache.popitem(last=False)
            self.geometry_bytes -= sum(value.numel() * value.element_size() for value in removed)

    @staticmethod
    def _images(geometry, images):
        encode = getattr(geometry, "encode_images", None)
        if callable(encode):
            return encode(images)
        return geometry.vision.image_tensor(geometry.cfg, images, geometry.device)

    @staticmethod
    def _check(guard):
        if guard is not None:
            guard()

    def _combine_transfer(self, parts, device):
        combined = torch.cat(parts)
        target = torch.device(device)
        if (self.pin_memory and combined.device.type == "cpu" and target.type == "cuda"
                and not combined.is_pinned()):
            # cat does not preserve the pinned allocation of cached source rows.
            combined = combined.pin_memory()
        return combined.to(target, non_blocking=self.pin_memory)

    def batch(self, keys, geometry, *, with_geometry=True, guard=None):
        keys = list(keys)
        if not keys:
            raise ValueError("Cannot create an empty replay batch")
        self._check(guard)
        fingerprint = self._fingerprint(geometry) if with_geometry else None
        records, publics, geometry_keys = [], [], []
        # Loading each distinct shard once per batch also covers disabled/small caches.
        loaded = {}
        for row, index in keys:
            self._check(guard)
            if row.get("split") != "train" or not row.get("complete", False):
                raise ValueError("Only complete training episodes can form a training batch")
            if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
                    or not 0 <= index < row["transitions"]):
                raise IndexError("Transition index is outside the recorded episode")
            shard = row["file"], row["sha256"]
            if shard not in loaded:
                data = self.load(row)
                loaded[shard] = data, self._verified[row["file"]][1]
            data, stamp = loaded[shard]
            if len(data["rgb"]) != row["transitions"] + 1 or len(data["offset"]) != row["transitions"] + 1:
                raise ValueError("Raw replay must preserve every actual successor")
            public = dict(rgb=data["rgb"][index], goal_rgb=data["goal_rgb"],
                          rotation=data["rotation"], previous_action=data["previous_action"][index])
            publics.append(public)
            records.append((row, int(index), data))
            geometry_keys.append((fingerprint, row["file"], row["sha256"],
                                  stamp, int(index)))

        with torch.no_grad():
            if with_geometry:
                resolved, missing = {}, OrderedDict()
                for key, public in zip(geometry_keys, publics):
                    if key in self.geometry_cache:
                        self.geometry_cache.move_to_end(key)
                        resolved[key] = self.geometry_cache[key]
                    elif key not in missing:
                        missing[key] = public
                observe_batch = getattr(geometry, "observe_batch", None)
                if missing and callable(observe_batch):
                    self._check(guard)
                    values = observe_batch(list(missing.values()))
                    if len(values) != 3 or any(value.shape[0] != len(missing) for value in values):
                        raise ValueError("observe_batch must return three tensors with one row per observation")
                    for index, key in enumerate(missing):
                        self._check(guard)
                        resolved[key] = tuple(value[index:index + 1].detach() for value in values)
                        self._store_geometry(key, resolved[key])
                else:
                    for key, public in missing.items():
                        self._check(guard)
                        resolved[key] = tuple(value.detach() for value in geometry.observe(public))
                        self._store_geometry(key, resolved[key])
                # Group by source device, then transfer whole groups rather than samples.
                combined = []
                for axis in range(3):
                    parts = [resolved[key][axis] for key in geometry_keys]
                    devices = {part.device for part in parts}
                    if len(devices) == 1:
                        value = self._combine_transfer(parts, geometry.device)
                    else:
                        groups = {}
                        for index, part in enumerate(parts):
                            groups.setdefault(part.device, []).append((index, part))
                        ordered = [None] * len(parts)
                        for group in groups.values():
                            transfer = self._combine_transfer([part for _, part in group], geometry.device)
                            for index, (position, _) in enumerate(group):
                                ordered[position] = transfer[index:index + 1]
                        value = torch.cat(ordered)
                    combined.append(value)
                images, goals, tangents = combined
            else:
                images = self._images(geometry, [public["rgb"] for public in publics])
                goals = self._images(geometry, [public["goal_rgb"] for public in publics])
                tangents = images.new_zeros(len(keys), 4, *images.shape[1:])
            self._check(guard)
            following = self._images(geometry, [data["rgb"][index + 1] for _, index, data in records])

        tensor = lambda value: torch.as_tensor(np.asarray(value), device=geometry.device, dtype=images.dtype)
        versions = [row["behavior_version"] for row, _, _ in records]
        version = versions[0] if len(set(versions)) == 1 else -1
        self._check(guard)
        return VisualBatch(
            VisualObservation(images, goals, tangents), following,
            tensor([data["offset"][i] for _, i, data in records]),
            tensor([data["offset"][i + 1] for _, i, data in records]),
            tensor([data["actual_normalized_action"][i] for _, i, data in records]),
            tensor([data["dt"][i] for _, i, data in records]),
            torch.as_tensor([row["policy"] == "current_controller" for row, _, _ in records],
                            device=geometry.device, dtype=torch.bool),
            previous_action=tensor([data["previous_action"][i] for _, i, data in records]),
            behavior_version=version, split="train")


__all__ = ["FastRawRGBReplay"]
