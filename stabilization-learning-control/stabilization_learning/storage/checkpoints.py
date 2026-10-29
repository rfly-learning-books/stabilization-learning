"""Deduplicate PyTorch ZIP records without changing tensors or optimizer state.

Each job owns a .checkpoint_objects directory. Checkpoints contain relative,
hash-checked references, so a complete job can be moved to another directory.
Old standalone torch.save files remain readable. Only trusted checkpoints should
be loaded, as with torch.load(..., weights_only=False).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile
import zlib

import torch

MAGIC = b"SLCHK1\n"
CHUNK_BYTES = 4 * 1024 * 1024


def _new_file(path, write):
    """Publish once, with no interval in which a partial target is visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Hard links work on NTFS, ext4 and APFS, and cannot replace another writer.
    os.link(temp, path)
    temp.unlink()


def _manifest(path):
    with Path(path).open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            return None
        result = json.load(stream)
    if result.get("schema") != "sl_checkpoint_objects_v1":
        raise ValueError("Unknown compact checkpoint schema")
    if not re.fullmatch(r"(?:\.\./){0,2}\.checkpoint_objects", result.get("store", "")):
        raise ValueError("Invalid checkpoint object store path")
    seen = set()
    for record in result["records"]:
        name = record["name"]
        if (name in seen or PurePosixPath(name).is_absolute() or
                ".." in PurePosixPath(name).parts or "\\" in name):
            raise ValueError("Invalid archive record")
        seen.add(name)
        if sum(c["bytes"] for c in record["chunks"]) != record["bytes"]:
            raise ValueError("Archive record size mismatch")
        for chunk in record["chunks"]:
            if (not re.fullmatch(r"[0-9a-f]{64}", chunk["sha256"]) or
                    type(chunk["bytes"]) is not int or not 0 < chunk["bytes"] <= CHUNK_BYTES):
                raise ValueError("Invalid checkpoint chunk")
    return result


def _object_path(root, digest):
    path = root / digest[:2] / (digest + ".z")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Object path escapes checkpoint store")
    return path


def _read_chunk(root, chunk):
    path = _object_path(root, chunk["sha256"])
    decoder = zlib.decompressobj()
    try:
        data = decoder.decompress(path.read_bytes(), chunk["bytes"] + 1)
    except zlib.error as error:
        raise ValueError("Corrupt checkpoint object: " + str(path)) from error
    if (not decoder.eof or decoder.unused_data or len(data) != chunk["bytes"] or
            hashlib.sha256(data).hexdigest() != chunk["sha256"]):
        raise ValueError("Checkpoint object hash/size mismatch: " + str(path))
    return data


def pack_checkpoint(source, path, *, store_root):
    """Write a new compact copy of a standard PyTorch ZIP; leave source intact."""
    path, root = Path(path), Path(store_root).resolve()
    relative = os.path.relpath(root, path.parent.resolve()).replace("\\", "/")
    if not re.fullmatch(r"(?:\.\./){0,2}\.checkpoint_objects", relative):
        raise ValueError("Use a job-local .checkpoint_objects directory")
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    records, new_bytes, logical_bytes = [], 0, 0
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            chunks = []
            with archive.open(info) as stream:
                while data := stream.read(CHUNK_BYTES):
                    digest = hashlib.sha256(data).hexdigest()
                    chunk = dict(sha256=digest, bytes=len(data))
                    target = _object_path(root, digest)
                    if target.exists():
                        # Existing bytes must actually match; do not trust a name.
                        _read_chunk(root, chunk)
                    else:
                        encoded = zlib.compress(data, level=1)
                        _new_file(target, lambda out, encoded=encoded: out.write(encoded))
                        new_bytes += len(encoded)
                    logical_bytes += len(data)
                    chunks.append(chunk)
            records.append(dict(name=info.filename, bytes=info.file_size, chunks=chunks))
    manifest = dict(schema="sl_checkpoint_objects_v1", store=relative, records=records)
    encoded = MAGIC + json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _new_file(path, lambda out: out.write(encoded))
    return dict(format="lossless_objects_v1", logical_bytes=logical_bytes,
                new_object_bytes=new_bytes, manifest_bytes=len(encoded))


def save_checkpoint(path, payload, *, store_root=None):
    """Keep legacy serialization unless an object store is explicitly requested."""
    if store_root is None:
        _new_file(path, lambda stream: torch.save(payload, stream))
        return dict(format="torch", bytes=Path(path).stat().st_size)
    # Temporary space is bounded by one standard checkpoint, not all epochs.
    # TemporaryFile also implements seekable() on Python 3.10; the 3.10 spooled
    # wrapper does not, which prevents zipfile from reading its records.
    with tempfile.TemporaryFile() as archive:
        torch.save(payload, archive)
        archive.seek(0)
        return pack_checkpoint(archive, path, store_root=store_root)


def _expand(path, manifest, stream):
    root = (Path(path).parent / manifest["store"]).resolve()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for record in manifest["records"]:
            info = zipfile.ZipInfo(record["name"])
            with archive.open(info, "w", force_zip64=True) as output:
                for chunk in record["chunks"]:
                    output.write(_read_chunk(root, chunk))


def load_checkpoint(path, map_location="cpu", *, weights_only=False):
    manifest = _manifest(path)
    if manifest is None:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    with tempfile.TemporaryFile() as archive:
        _expand(path, manifest, archive)
        archive.seek(0)
        return torch.load(archive, map_location=map_location, weights_only=weights_only)


def verify_checkpoint(path, *, verified=None):
    """Verify every referenced byte, optionally sharing a cache during an audit."""
    manifest = _manifest(path)
    if manifest is None:
        return dict(format="torch", objects=0)
    verified = set() if verified is None else verified
    root = (Path(path).parent / manifest["store"]).resolve()
    objects = 0
    for record in manifest["records"]:
        for chunk in record["chunks"]:
            key = (str(root), chunk["sha256"], chunk["bytes"])
            if key not in verified:
                _read_chunk(root, chunk)
                verified.add(key)
            objects += 1
    return dict(format="lossless_objects_v1", objects=objects)


def export_checkpoint(path, output):
    """Create a standalone torch.load-compatible file without modifying a run."""
    manifest = _manifest(path)
    if manifest is None:
        import shutil
        def copy(stream):
            with Path(path).open("rb") as source:
                shutil.copyfileobj(source, stream, CHUNK_BYTES)
        _new_file(output, copy)
    else:
        _new_file(output, lambda stream: _expand(path, manifest, stream))


def object_store(config, output):
    mode = config.get("storage", {}).get("checkpoints", "torch")
    if mode not in ("torch", "lossless_objects_v1"):
        raise ValueError("Unknown checkpoint storage mode: " + str(mode))
    return Path(output) / ".checkpoint_objects" if mode == "lossless_objects_v1" else None
