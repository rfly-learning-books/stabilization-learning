"""Batch public geometry; the physical plant and its units remain unchanged."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from stabilization_gym.tasks_v2.reconstructed_rgb import (
    RGBGeometry as ReferenceRGBGeometry, ReconstructedRGBEnv, _public,
    load_runtime, random_raw_autoencoder,
)


class BatchedRGBGeometry(ReferenceRGBGeometry):
    """Frozen FastDepth and identical preprocessing, in bounded image batches.

    No encoder or M is owned here. Geometry caches contain only public image
    transformations and fixed-depth estimates, never learned features or J.
    """
    def __init__(self, cfg, device="cpu", *, depth_batch_size=4):
        if isinstance(depth_batch_size, bool) or not isinstance(depth_batch_size, int) or depth_batch_size < 1:
            raise ValueError("depth_batch_size must be a positive integer")
        super().__init__(cfg, device)
        self.depth_batch_size = depth_batch_size
        from lac_vs import legacy
        sources = [Path(__file__), Path(inspect.getfile(ReferenceRGBGeometry)),
                   Path(inspect.getfile(self.vision)),
                   Path(inspect.getfile(legacy.module(cfg, "vision"))),
                   Path(inspect.getfile(legacy.module(cfg, "common")))]
        identity = dict(config=deepcopy(cfg), depth=self.provenance,
                        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                        depth_batch_size=depth_batch_size)
        self.cache_revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @torch.no_grad()
    def _predict_batch(self, images):
        prepared = np.stack([cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
                             .astype(np.float32) / 255. for rgb in images])
        tensor = torch.from_numpy(np.ascontiguousarray(prepared.transpose(0, 3, 1, 2))).to(self.device)
        result = self.depth.model(tensor)[:, 0].cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite batched FastDepth output")
        return [np.clip(cv2.resize(z, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR),
                        self.cfg["goal_fixed"]["min_depth"], self.cfg["goal_fixed"]["max_depth"])
                .astype(np.float32) for z, rgb in zip(result, images)]

    def encode_images(self, images):
        if not images:
            raise ValueError("Cannot preprocess an empty image batch")
        return self.vision.image_tensor(self.cfg, images, self.device)

    @torch.no_grad()
    def observe_batch(self, publics):
        if not publics:
            raise ValueError("Cannot observe an empty batch")
        validated = [_public(public) for public in publics]
        images, goals, rotations = map(list, zip(*validated))
        depth = []
        for start in range(0, len(images), self.depth_batch_size):
            depth.extend(self._predict_batch(images[start:start + self.depth_batch_size]))
        x, tangent = self.vision.images_and_tangents(self.cfg, images, depth, rotations, self.device)
        if not torch.isfinite(tangent).all():
            raise FloatingPointError("Estimated-depth image tangents are nonfinite")
        return x, self.encode_images(goals), tangent

    def observe(self, public):
        return self.observe_batch([public])
