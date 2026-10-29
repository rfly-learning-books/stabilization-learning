"""Bound disk replay only for shards created by this fresh writer.

Active rows and every full-checkpoint pin survive. Imported bootstrap shards and
unknown files are never removed. This does not operate on historical run folders.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def bounded_replay_class(parent, mode, cache_limit):
    """Preserve the parent's index and sampling, pruning only after its commit."""
    class BoundedReplay(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.retention = ReplayRetention(self.root, mode)

        def add(self, arrays, row, identifier):
            super().add(arrays, row, identifier)
            self.retention.register(self.rows[-1]["file"])
            self.retention.prune(self.rows, self.cache)

        def load(self, row):
            if row["file"] not in self.cache:
                with np.load(self.root / row["file"], allow_pickle=False) as archive:
                    self.cache[row["file"]] = {key: archive[key] for key in archive.files}
                while len(self.cache) > cache_limit:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(row["file"])
            return self.cache[row["file"]]
    return BoundedReplay


class ReplayRetention:
    def __init__(self, root, mode="all"):
        if mode not in ("all", "active_and_restart"):
            raise ValueError("Unknown replay retention: " + str(mode))
        self.root = Path(root).resolve()
        self.mode, self.owned, self.pinned = mode, set(), set()
        if mode == "all":
            return
        if any(self.root.iterdir()):
            raise FileExistsError("Bounded replay requires a fresh empty replay directory")
        self.marker = self.root / "STORAGE.json"
        self.marker.write_text(json.dumps(dict(schema="fresh_replay_retention_v1", mode=mode,
            scope="new writer-owned inactive shards only; full checkpoint rows pinned")), encoding="utf-8")

    def _path(self, name):
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("Replay shard must be a local .npz filename")
        path = self.root / name
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError("Replay shard escapes its owned directory")
        return path

    def register(self, name):
        if self.mode == "all":
            return
        if not self._path(name).is_file():
            raise FileNotFoundError(name)
        self.owned.add(name)

    def pin(self, rows, checkpoint):
        if self.mode == "all":
            return
        names = {row["file"] for row in rows}
        for name in names:
            if not self._path(name).is_file():
                raise FileNotFoundError("Cannot pin absent replay shard: " + name)
        self.pinned.update(names)
        with (self.root / "retention.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event="pin", checkpoint=str(checkpoint), files=sorted(names))) + "\n")

    def prune(self, rows, cache):
        if self.mode == "all":
            return 0
        # The writer cannot adopt an existing directory or remove foreign files.
        marker = json.loads(self.marker.read_text(encoding="utf-8"))
        if marker.get("schema") != "fresh_replay_retention_v1" or marker.get("mode") != self.mode:
            raise RuntimeError("Replay ownership marker changed")
        active = {row["file"] for row in rows}
        removed = 0
        for name in sorted(self.owned - active - self.pinned):
            path = self._path(name)
            data = path.read_bytes()
            with (self.root / "retention.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(event="evict_inactive", file=name, bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest())) + "\n")
                stream.flush()
            path.unlink()
            cache.pop(name, None)
            self.owned.remove(name)
            removed += len(data)
        return removed
