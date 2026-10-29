"""RGB autoencoder (ResNet18 + spatial collapse + mirrored decoder) and the exact
grayscale-to-RGB weight conversion required by experiment_plan.md stage A."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm
from torchvision.models import resnet18

from .common import LATENT

GRAY_MIX = (0.299, 0.587, 0.114)
GRAY_MIX_SQ = sum(w * w for w in GRAY_MIX)


def wnconv(cin, cout, kernel=3, stride=1, padding=1, groups=1):
    return weight_norm(nn.Conv2d(cin, cout, kernel, stride, padding, groups=groups, bias=True))


class ReverseBlock(nn.Module):
    def __init__(self, cin, cout, up=False):
        super().__init__()
        self.up = up
        self.c1 = wnconv(cin, cout)
        self.c2 = wnconv(cout, cout)
        self.skip = wnconv(cin, cout, 1, 1, 0) if cin != cout else nn.Identity()

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        residual = self.skip(x)
        return F.relu(self.c2(F.relu(self.c1(x))) + residual)


class AutoEncoder(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        r = resnet18(weights=None)
        r.conv1 = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)

        def replace(m):
            for name, child in list(m.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(m, name, nn.Identity())
                elif isinstance(child, (nn.Conv2d, nn.Linear)):
                    setattr(m, name, weight_norm(child))
                else:
                    replace(child)

        replace(r)
        self.encoder = nn.Sequential(
            r.conv1, nn.ReLU(), r.maxpool,
            r.layer1, r.layer2, r.layer3, r.layer4,
            wnconv(512, 512, 4, 1, 0, groups=512), nn.Flatten(),
            weight_norm(nn.Linear(512, LATENT)),
        )
        self.expand = nn.Sequential(weight_norm(nn.Linear(LATENT, 512)), nn.ReLU())
        self.expand_spatial = weight_norm(nn.ConvTranspose2d(512, 512, 4, groups=512))
        self.decoder = nn.Sequential(
            ReverseBlock(512, 512), ReverseBlock(512, 256, True),
            ReverseBlock(256, 256), ReverseBlock(256, 128, True),
            ReverseBlock(128, 128), ReverseBlock(128, 64, True),
            ReverseBlock(64, 64), ReverseBlock(64, 64),
            nn.Upsample(scale_factor=2, mode="nearest"), wnconv(64, 64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), wnconv(64, channels, 7, 1, 3), nn.Sigmoid(),
        )
        with torch.no_grad():
            for child in self.modules():
                if isinstance(child, nn.Conv2d) and child.bias is not None:
                    child.bias.zero_()
            for child in self.encoder.modules():
                if hasattr(child, "conv2"):
                    child.conv2.parametrizations.weight.original0.mul_(0.1)
            for child in self.decoder.modules():
                if isinstance(child, ReverseBlock):
                    child.c2.parametrizations.weight.original0.mul_(0.1)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        y = self.expand_spatial(self.expand(z)[..., None, None])
        return z, self.decoder(y)


def convert_gray_state_dict(sd_gray):
    """Map the single-channel checkpoint to the 3-channel model.

    Conv1 input channels are expanded with w_c * W_gray (w = 0.299/0.587/0.114) and
    WeightNorm magnitude g rescaled by sqrt(sum w^2); the decoder output layer is
    replicated over the three output channels with g rescaled by sqrt(3). With a
    neutral grayscale input replicated to three channels this reproduces the old
    encoder output up to numerical precision. All other tensors are copied.
    """
    sd = {}
    w = torch.tensor(GRAY_MIX, dtype=torch.float32)
    for key, value in sd_gray.items():
        if key == "encoder.0.parametrizations.weight.original1":
            v = value.clone()
            sd[key] = torch.stack([w[c] * v[:, 0] for c in range(3)], dim=1)
        elif key == "encoder.0.parametrizations.weight.original0":
            sd[key] = value.clone() * (GRAY_MIX_SQ ** 0.5)
        elif key == "decoder.12.parametrizations.weight.original1":
            v = value.clone()
            sd[key] = v.expand(3, *v.shape[1:]).clone()
        elif key == "decoder.12.parametrizations.weight.original0":
            sd[key] = value.expand(3, *value.shape[1:]).clone()
        elif key == "decoder.12.bias":
            sd[key] = value.expand(3).clone()
        else:
            sd[key] = value.clone()
    return sd


def new_rgb_model_from_gray(checkpoint_path, device="cuda"):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AutoEncoder(channels=3)
    model.load_state_dict(convert_gray_state_dict(saved["state_dict"]))
    model = model.to(device)
    model.eval()
    return model


def new_gray_model(checkpoint_path, device="cuda"):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AutoEncoder(channels=1)
    model.load_state_dict(saved["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def save_checkpoint(path, model, optimizer=None, extra=None):
    payload = {"state_dict": model.state_dict(), "channels": model.channels}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    from pathlib import Path

    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, device="cuda", build=True):
    saved = torch.load(path, map_location=device, weights_only=False)
    if not build:
        return saved
    model = AutoEncoder(channels=saved.get("channels", 3))
    model.load_state_dict(saved["state_dict"])
    model = model.to(device)
    return model, saved


def bake_weights(model):
    for child in model.modules():
        if parametrize.is_parametrized(child, "weight"):
            parametrize.remove_parametrizations(child, "weight", leave_parametrized=True)
    return model


def _state_clone(model):
    clone = AutoEncoder(channels=model.channels)
    clone.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return clone.to(next(model.parameters()).device)


def frozen_encoder(model):
    """Fresh state-dict clone (deepcopy corrupts parametrized modules on torch 2.3)."""
    clone = _state_clone(model)
    bake_weights(clone)
    clone.eval()
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone


def trainable_encoder(model):
    clone = _state_clone(model)
    bake_weights(clone)
    clone.train()
    for p in clone.parameters():
        p.requires_grad_(True)
    return clone


def encoder_fingerprint(model):
    import hashlib

    digest = hashlib.sha256()
    for key, value in model.encoder.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
