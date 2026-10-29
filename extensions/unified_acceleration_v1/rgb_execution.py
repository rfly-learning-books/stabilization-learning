"""Instance-local preparation for the preserved RGB derivative learner.

Images remain the exact CPU/OpenCV preprocessing; the returned tensors and
goal image tangents may be reused on GPU. Encoder-dependent JVP is never cached
across parameter updates. No globals or historical files are modified.
"""
from __future__ import annotations
import hashlib
from types import FunctionType, MethodType
import numpy as np
import torch


class PreparedRGBExecutor:
    """Optional finite-window input cache, with unchanged original update code.

    ``with PreparedRGBExecutor(learner) as execution: execution.prepare(batches)``
    then call learner.representation/metric_update normally. Preparation cost
    and memory are part of the end-to-end cost, not an uncounted speedup.
    """
    def __init__(self, learner):
        self.learner = learner
        self.images = {}
        self.saved = {}
        self.hits = self.misses = 0

    @staticmethod
    def key(images):
        array = np.ascontiguousarray(images)
        return (array.shape, array.dtype.str, hashlib.sha256(array).digest())

    def image_tensor(self, cfg, images, device):
        key = self.key(images)
        if key not in self.images:
            self.misses += 1
            self.images[key] = self.original_image_tensor(cfg, images, device)
        else:
            self.hits += 1
        return self.images[key]

    def __enter__(self):
        for name in ("encode", "model_update"):
            method = getattr(self.learner, name)
            function = method.__func__
            self.original_image_tensor = function.__globals__["image_tensor"]
            namespace = dict(function.__globals__, image_tensor=self.image_tensor)
            clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
            clone.__kwdefaults__ = function.__kwdefaults__
            self.saved[name] = self.learner.__dict__.get(name)
            setattr(self.learner, name, MethodType(clone, self.learner))
        return self

    def prepare(self, batches, *, goal_geometry=True):
        from lac_vs.vision import images_and_tangents
        for batch in batches:
            for images in (np.concatenate([batch["rgb"], batch["goal_rgb"]]), batch["rgb"], batch["next_rgb"],
                           np.concatenate([batch["smooth_minus"][:2], batch["smooth_center"][:2], batch["smooth_plus"][:2]])):
                self.image_tensor(self.learner.cfg, images, self.learner.device)
            if goal_geometry:
                for index, key in enumerate(batch["goal_key"]):
                    if key not in self.learner.inputs:
                        x, q = images_and_tangents(self.learner.cfg, [batch["goal_rgb"][index]],
                            [batch["goal_depth"][index].astype(np.float32)], [batch["rotation"][index]], "cpu")
                        self.learner.inputs[key] = (x.to(self.learner.device), q.to(self.learner.device))
        torch.cuda.synchronize()
        return self.statistics()

    def statistics(self):
        return dict(image_entries=len(self.images), hits=self.hits, misses=self.misses,
                    image_tensor_bytes=sum(t.numel()*t.element_size() for t in self.images.values()),
                    goal_geometry_entries=len(self.learner.inputs), encoder_JVP_recomputed=True,
                    CPU_OpenCV_preprocessing_preserved=True)

    def __exit__(self, *args):
        for name, prior in self.saved.items():
            if prior is None:
                delattr(self.learner, name)
            else:
                setattr(self.learner, name, prior)
        self.images.clear()
