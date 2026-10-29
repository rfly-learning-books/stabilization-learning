from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .config import json_write

REQUIRED = ("rgb", "depth", "goal_rgb", "action", "error2", "stage_cost", "terminal", "truncated",
            "previous_action", "camera_from_body", "offset", "L")


class Replay:
    """Disk-backed raw-frame episodes, scene-balanced transitions, train split only."""
    def __init__(self, root, max_transitions=20000, cache_episodes=2, seed=41):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.entries = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else []
        self.max_transitions, self.cache_episodes = max_transitions, cache_episodes
        self.cache = OrderedDict()
        self.rng = np.random.default_rng(seed)
        if any(row["split"] != "train" for row in self.entries):
            raise ValueError("Validation/test episode found in training replay")

    @property
    def transitions(self):
        return sum(row["length"] for row in self.entries)

    def add(self, arrays, metadata):
        if metadata["split"] != "train":
            raise ValueError("Only training scenes may enter replay")
        for key in REQUIRED:
            if key not in arrays:
                raise ValueError(f"Missing replay field {key}")
        length = len(arrays["action"])
        if length < 1 or len(arrays["rgb"]) != length + 1 or len(arrays["depth"]) != length + 1:
            raise ValueError("Episode must have T actions and T+1 observations")
        if arrays["rgb"].dtype != np.uint8 or arrays["goal_rgb"].dtype != np.uint8:
            raise ValueError("Replay stores raw uint8 RGB, never stale latent features")
        if np.any(arrays["terminal"] & arrays["truncated"]):
            raise ValueError("Termination and time-limit truncation are different events")
        identifier = f"{metadata['scene_id']}_{uuid.uuid4().hex[:16]}"
        path = self.root / f"{identifier}.npz"
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
        temp.replace(path)
        self.entries.append({"file": path.name, "length": length, **metadata})
        # Old shards stay on disk as evidence; only the active sampling index rolls.
        while len(self.entries) > 1 and self.transitions > self.max_transitions:
            self.entries.pop(0)
        json_write(self.index_path, self.entries)
        return path

    def _load(self, row):
        name = row["file"]
        if name not in self.cache:
            path = (self.root / name).resolve()
            path.relative_to(self.root)
            with np.load(path, allow_pickle=False) as data:
                self.cache[name] = {k: data[k] for k in REQUIRED}
            while len(self.cache) > self.cache_episodes:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name]

    def sample(self, size):
        if not self.entries:
            raise RuntimeError("Replay is empty: collect bootstrap episodes first")
        scenes = sorted({row["scene_id"] for row in self.entries})
        records = []
        for _ in range(size):
            scene = scenes[int(self.rng.integers(len(scenes)))]
            pool = [row for row in self.entries if row["scene_id"] == scene]
            lengths = np.asarray([row["length"] for row in pool], float)
            row = pool[int(self.rng.choice(len(pool), p=lengths / lengths.sum()))]
            ep = self._load(row)
            t = int(self.rng.integers(row["length"]))
            records.append({"rgb": ep["rgb"][t], "next_rgb": ep["rgb"][t + 1], "goal_rgb": ep["goal_rgb"],
                            "depth": ep["depth"][t], "next_depth": ep["depth"][t + 1],
                            "action": ep["action"][t], "cost": ep["stage_cost"][t],
                            "terminal": ep["terminal"][t], "truncated": ep["truncated"][t],
                            "error2": ep["error2"][t], "next_error2": ep["error2"][t + 1],
                            "previous_action": ep["previous_action"][t],
                            "camera_from_body": ep["camera_from_body"], "scene_id": scene})
        result = {key: np.stack([record[key] for record in records]) for key in records[0] if key != "scene_id"}
        result["scene_id"] = [record["scene_id"] for record in records]
        return result
