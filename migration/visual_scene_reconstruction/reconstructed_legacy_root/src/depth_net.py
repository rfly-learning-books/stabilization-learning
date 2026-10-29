"""Frozen Depth Anything V2 Metric Indoor Small wrapper plus an fp16 disk cache.

Control always uses fresh FP32 inference; the cache serves training tensors only and
its effect compared with FP32 is audited separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .common import CACHE_DIR, DEPTH_MODEL_FILE, DEPTH_MODEL_SHA256, OLD_DEPTH_MODEL_DIR, OLD_DEPTH_RUNTIME, sha_bytes


def _load_transformers():
    import importlib

    sys.path.insert(0, str(OLD_DEPTH_RUNTIME))
    try:
        import transformers
        if not transformers.__version__.startswith("4.5"):
            raise ImportError(f"wrong transformers {transformers.__version__}")
    except Exception:
        for name in list(sys.modules):
            if name.split(".")[0] in ("transformers", "tokenizers", "huggingface_hub", "safetensors"):
                del sys.modules[name]
        importlib.invalidate_caches()
        import transformers  # noqa: F401
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    return AutoImageProcessor, AutoModelForDepthEstimation


class MetricDepth:
    def __init__(self, device="cuda", config_hash=None):
        AutoImageProcessor, AutoModelForDepthEstimation = _load_transformers()
        self.processor = AutoImageProcessor.from_pretrained(OLD_DEPTH_MODEL_DIR, local_files_only=True, use_fast=False)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(OLD_DEPTH_MODEL_DIR, local_files_only=True)
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        self.device = device
        self.config_hash = config_hash or sha_bytes(DEPTH_MODEL_FILE.read_bytes())[:16]

    def predict_fresh(self, images):
        """FP32 control-path inference. images: list of uint8 (256,256,3). -> (N,256,256) float32."""
        batch = self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode():
            out = self.model(pixel_values=batch).predicted_depth
            out = torch.nn.functional.interpolate(out[:, None], size=(256, 256), mode="bilinear",
                                                  align_corners=False)[:, 0]
        z = out.float().cpu().numpy().astype(np.float32)
        if not np.isfinite(z).all():
            raise RuntimeError("Non-finite estimated depth")
        return np.clip(z, 0.10, 20.0)


class DepthCache:
    """fp16 disk cache keyed by (RGB bytes, depth weights, preprocessing config)."""

    def __init__(self, namespace="default"):
        self.root = CACHE_DIR / "depth" / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def key(self, rgb):
        return sha_bytes(np.ascontiguousarray(rgb).tobytes())[:24]

    def get(self, rgb):
        path = self.root / f"{self.key(rgb)}.npy"
        if path.exists():
            self.hits += 1
            return np.load(path).astype(np.float32)
        self.misses += 1
        return None

    def put(self, rgb, depth_fp32):
        path = self.root / f"{self.key(rgb)}.npy"
        if not path.exists():
            np.save(path, depth_fp32.astype(np.float16))


def depth_cache_config_hash():
    return {
        "weights_sha256": DEPTH_MODEL_SHA256,
        "preprocessing": "official 518; metric; bilinear 256; clip [0.1,20]",
        "storage_dtype": "float16",
    }
