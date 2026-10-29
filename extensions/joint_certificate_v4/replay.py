"""Append-only full RGB archive and stratified, bounded-memory replay.

All episodes remain eligible. A cache eviction closes a mapping, never removes
data from the sampling population. Learned features/J/M are never cached.
"""
from collections import OrderedDict, defaultdict
from pathlib import Path
import json
import os
import numpy as np
import torch
from stabilization_gym.tasks_v2.visual import VisualObservation
from stabilization_learning.joint_v2.visual import VisualBatch
from stabilization_learning.joint_v2.runtime import atomic_json, file_sha256


def write_array(path, value):
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("xb") as stream:
        # Time-major vector rollouts produce strided per-lane views. NumPy's
        # real-file tofile path is much slower on those views; one bounded copy
        # preserves identical .npy bytes and enables a contiguous bulk write.
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


class StratifiedRGBReplay:
    def __init__(self, root, rng, *, mmap_episodes=64):
        self.root, self.rng = Path(root), rng
        self.root.mkdir(parents=True, exist_ok=False)
        self.rows, self.by_stratum, self.cache = [], defaultdict(list), OrderedDict()
        self.mmap_episodes = mmap_episodes
        self.samples_drawn = 0
        self.unique_samples = set()
        self.draws_per_stratum = defaultdict(int)

    def add(self, arrays, metadata):
        n = len(arrays["dt"])
        if metadata["split"] != "train" or n < 1:
            raise ValueError("Only nonempty training episodes enter replay")
        if arrays["rgb"].dtype != np.uint8 or len(arrays["rgb"]) != n+1:
            raise ValueError("Need every true RGB successor, including terminal")
        if len(arrays["offset"]) != n+1 or len(arrays["rotation"]) != n+1:
            raise ValueError("Incomplete state/rotation sequence")
        if any(len(arrays[k]) != n for k in ("action", "applied_action", "previous_action", "controller", "terminated", "truncated")):
            raise ValueError("Transition array lengths differ")
        if not np.isfinite(arrays["dt"]).all() or (arrays["dt"] <= 0).any():
            raise ValueError("Reset calls cannot enter replay")
        if np.any(arrays["terminated"][:-1]) or np.any(arrays["truncated"][:-1]):
            raise ValueError("A replay episode cannot cross its first boundary")
        episode = len(self.rows)
        folder = self.root / f"episode_{episode:06d}"
        folder.mkdir(exist_ok=False)
        # Keep the large uint8 frame sequence mappable. Pack all small arrays
        # into one uncompressed ZIP instead of fsync'ing 10+ files per episode.
        # No arrays, bits, sampling weights or episode boundaries are dropped.
        rgb_path, labels_path = folder/"rgb.npy", folder/"labels.npz"
        write_array(rgb_path, arrays["rgb"])
        temporary = folder/"labels.npz.tmp"
        with temporary.open("xb") as stream:
            np.savez(stream, **{k: np.asarray(v) for k, v in arrays.items() if k != "rgb"})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, labels_path)
        files = {"rgb": dict(path=str(rgb_path), sha256=file_sha256(rgb_path)),
                 "labels": dict(path=str(labels_path), sha256=file_sha256(labels_path))}
        row = dict(episode=episode, transitions=n, **metadata, files=files)
        atomic_json(folder / "COMMITTED.json", row)
        self.rows.append(row)
        self.by_stratum[row["stratum"]].extend((episode, i) for i in range(n))
        return row

    def _load(self, episode):
        if episode not in self.cache:
            row = self.rows[episode]
            # The in-process writer committed and hashed these immutable arrays.
            # Final audit verifies every hash again; rereading 25GB per sample is
            # not an integrity check with a meaningful training cost.
            with np.load(row["files"]["labels"]["path"], allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays["rgb"] = np.load(row["files"]["rgb"]["path"], mmap_mode="r", allow_pickle=False)
            self.cache[episode] = arrays
        self.cache.move_to_end(episode)
        while len(self.cache) > self.mmap_episodes:
            _, expired = self.cache.popitem(last=False)
            for array in expired.values():
                if isinstance(array, np.memmap):
                    array._mmap.close()
        return self.cache[episode]

    def sample_keys(self, count):
        strata = sorted(self.by_stratum)
        if not strata:
            raise RuntimeError("Empty replay")
        # Random permutation in blocks guarantees balanced large calibration
        # batches and uniform individual draws even when counts differ.
        order = []
        while len(order) < count:
            order.extend(self.rng.permutation(len(strata)).tolist())
        result = []
        for index in order[:count]:
            population = self.by_stratum[strata[index]]
            result.append(population[int(self.rng.integers(len(population)))])
            self.draws_per_stratum[strata[index]] += 1
        self.samples_drawn += count
        self.unique_samples.update(result)
        return result

    def batch(self, keys, geometry, *, with_geometry=True):
        names = ("rgb", "next_rgb", "goal_rgb", "rotation", "offset", "next_offset",
                 "applied_action", "previous_action", "dt", "controller")
        data = {k: [] for k in names}
        versions = []
        for episode, i in keys:
            arrays = self._load(episode)
            # Copy immediately: a later key can evict this episode's mmap.
            for key, name, index in (("rgb", "rgb", i), ("next_rgb", "rgb", i+1),
                    ("rotation", "rotation", i), ("offset", "offset", i), ("next_offset", "offset", i+1),
                    ("applied_action", "applied_action", i), ("previous_action", "previous_action", i),
                    ("dt", "dt", i), ("controller", "controller", i)):
                data[key].append(np.array(arrays[name][index], copy=True))
            data["goal_rgb"].append(np.array(arrays["goal_rgb"], copy=True))
            versions.append(self.rows[episode]["behavior_version"])
        tensors = {}
        for name, items in data.items():
            host = torch.from_numpy(np.stack(items))
            if host.is_floating_point():
                host = host.float()
            tensors[name] = host.pin_memory().to(geometry.device, non_blocking=True)
        public = {k: tensors[k] for k in ("rgb", "goal_rgb", "rotation", "previous_action")}
        if with_geometry:
            image, goal, tangents = geometry.observe(public)
        else:
            image, goal = geometry.encode_images(tensors["rgb"]), geometry.encode_images(tensors["goal_rgb"])
            tangents = image.new_zeros((len(keys), 4, *image.shape[1:]))
        batch = VisualBatch(VisualObservation(image, goal, tangents), geometry.encode_images(tensors["next_rgb"]),
                            tensors["offset"], tensors["next_offset"], tensors["applied_action"], tensors["dt"],
                            tensors["controller"].bool(), previous_action=tensors["previous_action"],
                            behavior_version=versions[0] if len(set(versions)) == 1 else -1)
        batch.validate()
        return batch

    def snapshot(self):
        return dict(schema="full_stratified_raw_rgb_v4", rows=self.rows, samples_drawn=self.samples_drawn,
                    sampling="uniform strata then uniform transition from complete archive",
                    transitions=sum(r["transitions"] for r in self.rows),
                    strata={k: len(v) for k, v in sorted(self.by_stratum.items())},
                    unique_transition_keys_drawn=sorted(self.unique_samples),
                    draws_per_stratum=dict(self.draws_per_stratum),
                    draw_count_note="includes fixed probe and calibration draws, not only gradient batches",
                    mmap_episodes=self.mmap_episodes, discarded_episodes=0)

    def close(self):
        for arrays in self.cache.values():
            for value in arrays.values():
                if isinstance(value, np.memmap):
                    value._mmap.close()
        self.cache.clear()
