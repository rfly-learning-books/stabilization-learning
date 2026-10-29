"""Vision pipeline: RGB preprocessing, pixel-level interaction matrices (analytic),
latent Jacobians via forward-mode AD, and the fixed LM control layer.

All certificate math uses the single scaling convention e = (h(I) - h(I*))/sqrt(32),
V = 0.5 ||e||^2. Pixel-space channel derivatives are built on the native 256 grid
per color channel and then receive exactly the same resampling and blur as the image.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal

from .common import ACTION_SCALE, DT, GAIN, LATENT, MU, NET_SIZE, SIZE, S_YAW, VF, WMAX, VMAX
from .common import FOCAL, X, Y

GRAY_W = (0.299, 0.587, 0.114)


def preprocess_rgb(rgb):
    raw = np.asarray(rgb, np.float32) / 255.0
    out = np.empty((3, NET_SIZE, NET_SIZE), np.float32)
    for c in range(3):
        channel = cv2.resize(raw[..., c], (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA)
        out[c] = cv2.GaussianBlur(channel, (5, 5), 0.8, borderType=cv2.BORDER_REFLECT_101)
    return out


def preprocess_gray(rgb):
    g = cv2.cvtColor(np.asarray(rgb, np.float32) / 255.0, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (5, 5), 0.8, borderType=cv2.BORDER_REFLECT_101)


def _down(channel):
    return cv2.GaussianBlur(
        cv2.resize(channel, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA),
        (5, 5), 0.8, borderType=cv2.BORDER_REFLECT_101,
    )


def observe_rgb(frame, rgb, depth):
    """Return (im (3,H,W) float32, li (H,W,4,3) float32) for one RGB frame."""
    raw = np.asarray(rgb, np.float32) / 255.0
    li_native = np.empty((SIZE, SIZE, 4, 3), np.float32)
    for c in range(3):
        gx = cv2.Sobel(raw[..., c], cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8,
                       borderType=cv2.BORDER_REFLECT_101)
        gy = cv2.Sobel(raw[..., c], cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8,
                       borderType=cv2.BORDER_REFLECT_101)
        li_native[..., :3, c] = -FOCAL * (gx[..., None] * frame.trans[..., 0, :]
                                          + gy[..., None] * frame.trans[..., 1, :]) / depth[..., None]
        li_native[..., 3, c] = -FOCAL * (gx * frame.rot[..., 0] + gy * frame.rot[..., 1])
    li = np.empty((NET_SIZE, NET_SIZE, 4, 3), np.float32)
    for j in range(4):
        for c in range(3):
            li[..., j, c] = _down(li_native[..., j, c])
    li *= ACTION_SCALE[None, None, :, None]
    return preprocess_rgb(rgb), li


def observe_gray(frame, rgb, depth):
    raw = cv2.cvtColor(np.asarray(rgb, np.float32) / 255.0, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(raw, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(raw, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8, borderType=cv2.BORDER_REFLECT_101)
    li_native = np.empty((SIZE, SIZE, 4), np.float32)
    li_native[..., :3] = -FOCAL * (gx[..., None] * frame.trans[..., 0, :]
                                   + gy[..., None] * frame.trans[..., 1, :]) / depth[..., None]
    li_native[..., 3] = -FOCAL * (gx * frame.rot[..., 0] + gy * frame.rot[..., 1])
    li = np.stack([_down(li_native[..., j]) for j in range(4)], axis=-1)
    return preprocess_gray(rgb), li * ACTION_SCALE[None, None, :]


def moment_basis():
    j = np.arange(1, NET_SIZE, dtype=np.float64)
    off = 0.5 * j * np.sqrt((NET_SIZE * NET_SIZE - j * j) / (4 * j * j - 1))
    nodes, T = eigh_tridiagonal(np.zeros(NET_SIZE), off, lapack_driver="stev")
    T *= np.sign(T[0])[None, :]
    return T


def dom_pairs(n):
    pairs = []
    for s in range(NET_SIZE):
        diagonal = [(p, s - p) for p in range(s + 1)]
        pairs.extend(diagonal if s % 2 else diagonal[::-1])
        if len(pairs) >= n:
            break
    return np.array(pairs[:n])


class DOMFeature:
    def __init__(self, reference_gray, order=120):
        self.T = moment_basis()
        self.p, self.q = dom_pairs(order).T
        self.Tn = self.T[:max(self.p.max(), self.q.max()) + 1]
        self.reference = reference_gray
        self.ref = self.calc(reference_gray)

    def calc(self, im):
        out = self.Tn @ im.astype(np.float64) @ self.Tn.T
        return out[self.p, self.q]

    def error_jacobian(self, im, li):
        s = self.calc(im)
        J = np.column_stack([self.calc(li[..., j]) for j in range(4)])
        return s - self.ref, J


class PVSFeature:
    def __init__(self, reference_gray):
        self.reference = reference_gray.ravel().astype(np.float64)

    def error_jacobian(self, im, li):
        return im.ravel().astype(np.float64) - self.reference, li.reshape(-1, 4).astype(np.float64)


def encoder_jvp(model, x, tangents):
    """Batched JVP of the latent map. x (B,3,H,W); tangents (B,4,3,H,W).

    Returns d h / d a (B,4,LATENT) / sqrt(LATENT), differentiable w.r.t. parameters.
    """
    B = x.shape[0]
    xa = x.repeat_interleave(4, dim=0)
    ta = tangents.reshape(B * 4, *x.shape[1:])

    def one(xi, ti):
        return torch.func.jvp(lambda v: model.encode(v[None])[0], (xi,), (ti,))[1]

    dz = torch.vmap(one)(xa, ta)
    return dz.reshape(B, 4, LATENT) / math.sqrt(LATENT)


class RGBFeature:
    """Frozen-model latent features with the shared certificate scaling."""

    def __init__(self, model, reference_rgb):
        self.model = model
        self.reference = reference_rgb
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(reference_rgb))[None].to(next(model.parameters()).device)
            self.zref = model.encode(x)[0].detach().cpu().numpy().astype(np.float64)

    def error_jacobian(self, im, li):
        device = next(self.model.parameters()).device
        x = torch.from_numpy(np.ascontiguousarray(im))[None].to(device)
        tangents = torch.from_numpy(np.ascontiguousarray(li.transpose(2, 3, 0, 1))).to(device)
        with torch.no_grad():
            z = self.model.encode(x)
            dz = encoder_jvp(self.model, x, tangents[None])[0]
        e = (z[0].cpu().numpy().astype(np.float64) - self.zref) / math.sqrt(LATENT)
        J = dz.cpu().numpy().astype(np.float64).T
        return e, J


def certificate(e):
    return 0.5 * float(np.dot(e, e))


def control_numpy(e, J):
    H = J.T @ J
    g = J.T @ e
    diag = np.diag(H).copy()
    eps = max(1e-8 * float(np.trace(H)) / 4.0, 1e-10)
    a = -GAIN * np.linalg.solve(H + MU * np.diag(diag) + eps * np.eye(4), g)
    scale = min(1.0, 1.0 / max(float(np.linalg.norm(a[:3])), 1e-12),
                1.0 / max(abs(float(a[3])), 1e-12))
    cmd = scale * a
    sv = np.linalg.svd(J, compute_uv=False)
    stat = {
        "feature_error_norm": float(np.linalg.norm(e)),
        "sigma_min": float(sv[-1]),
        "sigma_max": float(sv[0]),
        "condition": float(sv[0] / max(sv[-1], 1e-12)),
        "rank": int(np.count_nonzero(sv > max(sv[0] * 1e-6, 1e-12))),
        "predicted_Vdot": float(e @ (J @ cmd)),
        "common_scale": float(scale),
    }
    return cmd, stat


def control_torch(e, J):
    """Differentiable LM action with the common positive scaling (same law as numpy)."""
    H = J.transpose(-1, -2) @ J
    g = (J.transpose(-1, -2) @ e.unsqueeze(-1)).squeeze(-1)
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    trace = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)
    eps = torch.clamp(1e-8 * trace / 4.0, min=1e-10)
    eye = torch.eye(4, device=H.device, dtype=H.dtype)
    H_reg = H + MU * torch.diag_embed(diag) + eps[:, None, None] * eye
    araw = -GAIN * torch.linalg.solve(H_reg, g.unsqueeze(-1)).squeeze(-1)
    n_trans = torch.linalg.norm(araw[..., :3], dim=-1)
    yaw = araw[..., 3].abs()
    scale = torch.clamp(torch.minimum(torch.ones_like(n_trans), 1.0 / torch.clamp(n_trans, min=1e-12)), max=1.0)
    scale = torch.minimum(scale, torch.clamp(1.0 / torch.clamp(yaw, min=1e-12), max=1.0))
    return araw * scale.unsqueeze(-1)


def descent_target(V, H=1):
    eta = 1 - (1 - 0.03) ** H
    delta = 1e-6 if H == 1 else 5e-6
    W = eta * V / (1.0 + V / 1.0)
    return W - delta


def scatter_tensor(tensor, index, dim_size):
    out = torch.zeros(dim_size, dtype=tensor.dtype, device=tensor.device)
    out.index_add_(0, index, tensor)
    return out
