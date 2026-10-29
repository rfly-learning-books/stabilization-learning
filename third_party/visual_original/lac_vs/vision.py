from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from . import legacy
from .models import frozen_parameters


def calibrated_frame(cfg, rotation):
    """Only known camera/body extrinsics; no current or target position input."""
    common = legacy.module(cfg, "common")
    rotation = np.asarray(rotation, np.float32)
    a = np.zeros((common.SIZE, common.SIZE, 2, 3), np.float32)
    a[..., 0, 0], a[..., 0, 2] = -1.0, common.X
    a[..., 1, 1], a[..., 1, 2] = -1.0, common.Y
    kx, ky, kz = rotation[:, 2]
    rot = np.stack([common.X * common.Y * kx - (1 + common.X**2) * ky + common.Y * kz,
                    (1 + common.Y**2) * kx - common.X * common.Y * ky - common.X * kz], -1)
    return SimpleNamespace(trans=a @ rotation, rot=rot.astype(np.float32))


def image_tensor(cfg, images, device):
    prepare = legacy.module(cfg, "vision").preprocess_rgb
    return torch.from_numpy(np.stack([prepare(image) for image in images])).to(device)


def images_and_tangents(cfg, images, depths, rotations, device):
    observe = legacy.module(cfg, "vision").observe_rgb
    im, tangent = [], []
    for rgb, depth, rotation in zip(images, depths, rotations):
        x, li = observe(calibrated_frame(cfg, rotation), rgb, depth.astype(np.float32))
        im.append(x)
        tangent.append(li.transpose(2, 3, 0, 1))
    return (torch.from_numpy(np.ascontiguousarray(np.stack(im))).to(device),
            torch.from_numpy(np.ascontiguousarray(np.stack(tangent))).to(device))


def frozen_features(cfg, autoencoder, images, goals, depths, rotations, device):
    """No encoder parameter gradient, but image JVP remains available."""
    x, tangent = images_and_tangents(cfg, images, depths, rotations, device)
    goal = image_tensor(cfg, goals, device)
    with frozen_parameters(autoencoder), torch.no_grad():
        z = autoencoder.encode(x)
        zg = autoencoder.encode(goal)
        j = encoder_jvp(autoencoder, x, tangent).transpose(1, 2)
    return z.detach(), zg.detach(), j.detach()


def encoder_jvp(model, x, tangents):
    """Four forward derivatives, normalized by the actual bottleneck dimension."""
    batch = x.shape[0]
    expanded = x.repeat_interleave(4, dim=0)
    directions = tangents.reshape(batch * 4, *x.shape[1:])

    def one(image, direction):
        return torch.func.jvp(lambda value: model.encode(value[None])[0], (image,), (direction,))[1]

    derivatives = torch.vmap(one)(expanded, directions)
    latent = derivatives.shape[-1]
    return derivatives.reshape(batch, 4, latent) / math.sqrt(latent)


def dct_reconstruction(prediction, target):
    size = prediction.shape[-1]
    n = torch.arange(size, device=prediction.device, dtype=prediction.dtype)
    transform = torch.cos(torch.pi / size * (n[None, :] + 0.5) * n[:, None])
    transform = transform * (2.0 / size)**0.5
    transform[0] *= 0.5**0.5
    weight = 1.0 / (n[:, None] + n[None, :] + 1)
    error = transform @ (prediction - target) @ transform.T
    return (error.abs() * weight).mean()
