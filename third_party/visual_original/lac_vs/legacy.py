"""Read-only, namespaced access to the audited simulator / RGB AEVS backbone.

No old trainer, replay writer, protocol generator or report is executed here.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from .config import digest

NAME = "lac_readonly_rgb_adapter"


def module(cfg, name):
    root = Path(cfg["paths"]["legacy_root"]).resolve()
    if NAME not in sys.modules:
        package = types.ModuleType(NAME)
        package.__path__ = [str(root / "src")]
        sys.modules[NAME] = package
    elif Path(sys.modules[NAME].__path__[0]).resolve() != root / "src":
        raise ValueError("Cannot mix legacy roots in one process")
    return importlib.import_module(f"{NAME}.{name}")


def dependency_hashes(cfg):
    root = Path(cfg["paths"]["legacy_root"])
    out = {f"legacy/{name}": digest(root / "src" / name) for name in
           ("common.py", "models.py", "vision.py", "room_env.py", "scene_builders.py", "depth_net.py")}
    common = module(cfg, "common")
    out["scene_assets_builder"] = digest(common.OLD_SCENES_DIR / "scenes.py")
    out["depth_weights"] = digest(common.DEPTH_MODEL_FILE)
    for name in ("config.json", "preprocessor_config.json"):
        path = common.OLD_DEPTH_MODEL_DIR / name
        if path.exists():
            out[f"depth/{name}"] = digest(path)
    initial = cfg["paths"].get("initial_autoencoder")
    out["initial_autoencoder"] = digest(initial) if initial else "random"
    return out


def autoencoder(cfg, device, initialize=True):
    import torch

    models = module(cfg, "models")
    model = models.AutoEncoder(channels=3)
    initial = cfg["paths"].get("initial_autoencoder")
    if initialize and initial:
        payload = torch.load(initial, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
    # Removing WeightNorm parametrizations before creating optimizers avoids
    # the known torch 2.3 parametrization / snapshot problem in the old run.
    models.bake_weights(model)
    resize_bottleneck(model, cfg["encoder"]["latent_dim"], pretrained=bool(initialize and initial))
    return model.to(device)


def resize_bottleneck(model, latent, pretrained=False):
    """Set real trainable encoder/decoder dimensions, never slice saved latents.

    Existing 32-D weights use a weight-SVD orthogonal subspace for a 16-D warm
    start. This is not data PCA and does not preserve reconstruction exactly.
    """
    import torch
    from torch import nn

    old_encoder, old_expand = model.encoder[-1], model.expand[0]
    original = old_encoder.out_features
    if latent == original:
        return
    encoder = nn.Linear(old_encoder.in_features, latent).to(old_encoder.weight)
    expand = nn.Linear(latent, old_expand.out_features).to(old_expand.weight)
    if pretrained:
        if latent > original:
            raise ValueError("Pretrained bottleneck expansion is not defined")
        with torch.no_grad():
            basis = torch.linalg.svd(old_encoder.weight, full_matrices=False).U[:, :latent].T
            pivots = basis.abs().argmax(-1)
            signs = basis[torch.arange(latent), pivots].sign()
            basis = basis * signs[:, None]
            encoder.weight.copy_(basis @ old_encoder.weight)
            encoder.bias.copy_(basis @ old_encoder.bias)
            expand.weight.copy_(old_expand.weight @ basis.T)
            expand.bias.copy_(old_expand.bias)
    model.encoder[-1], model.expand[0] = encoder, expand
