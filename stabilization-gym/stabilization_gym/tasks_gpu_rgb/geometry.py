"""GPU RGB/FastDepth/image geometry with the reference preprocessing formulas.

Only public RGB and camera calibration are accepted. The learned encoder and M
are external; their current interaction matrix must still be recomputed.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F

from stabilization_gym.tasks_v2.reconstructed_rgb import RGBGeometry, PUBLIC_KEYS


def area_weights(source, target, device):
    lo = torch.arange(target, device=device, dtype=torch.float64) * (source/target)
    hi = lo + source/target
    left = torch.arange(source, device=device, dtype=torch.float64)
    return (torch.minimum(hi[:, None], left[None]+1) - torch.maximum(lo[:, None], left[None])).clamp_min(0).float()/(source/target)


class GPURGBGeometry(RGBGeometry):
    def __init__(self, cfg, device="cuda:0"):
        super().__init__(cfg, device)
        from lac_vs import legacy
        common = legacy.module(cfg, "common")
        self.size, self.net_size, self.focal = common.SIZE, common.NET_SIZE, common.FOCAL
        self.action_scale = torch.as_tensor(common.ACTION_SCALE, device=self.device)
        self.x = torch.as_tensor(common.X, device=self.device)
        self.y = torch.as_tensor(common.Y, device=self.device)
        grid = torch.arange(-2, 3, device=self.device, dtype=torch.float64)
        g = torch.exp(-grid.square() / (2*.8**2)); g /= g.sum()
        self.gaussian = (g[:, None]*g[None, :]).float()[None, None]
        k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=self.device)/8
        self.sobel = torch.stack((k, k.T))
        self.depth_area = area_weights(self.size, 224, self.device)
        self.cache_revision = "gpu_rgb_geometry_v1/reference_area_sobel_gaussian/" + self.depth.hash

    def _raw(self, images):
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1:] != (self.size, self.size, 3):
            raise ValueError("Expected GPU B,H,W,3 uint8 RGB at the original camera resolution")
        if images.device.type != "cuda":
            raise ValueError("The GPU geometry path does not silently upload CPU images")
        return images.permute(0, 3, 1, 2).float()/255.

    def _down(self, raw):
        if self.size % self.net_size:
            raise ValueError("This exact preprocessing bridge requires integral image downsampling")
        factor = self.size//self.net_size
        down = F.avg_pool2d(raw, factor, factor)
        channels = down.shape[1]
        return F.conv2d(F.pad(down, (2, 2, 2, 2), mode="reflect"),
                        self.gaussian.expand(channels, -1, -1, -1), groups=channels)

    def encode_images(self, images):
        return self._down(self._raw(images))

    @torch.no_grad()
    def predict_depth(self, images):
        # OpenCV resizes uint8 before conversion to float. Preserve that
        # quantization and true pixel-area weights (adaptive pooling differs).
        raw = images.permute(0, 3, 1, 2).float()
        resized = torch.matmul(torch.matmul(self.depth_area, raw), self.depth_area.T)
        resized = resized.round().clamp(0, 255)/255.
        z = self.depth.model(resized)[:, :1]
        z = F.interpolate(z, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return z.clamp(self.cfg["goal_fixed"]["min_depth"], self.cfg["goal_fixed"]["max_depth"])

    def tangents(self, images, depth, rotations):
        raw = self._raw(images)
        gx = F.conv2d(F.pad(raw, (1, 1, 1, 1), mode="reflect"),
                      self.sobel[0:1, None].expand(3, -1, -1, -1), groups=3)
        gy = F.conv2d(F.pad(raw, (1, 1, 1, 1), mode="reflect"),
                      self.sobel[1:2, None].expand(3, -1, -1, -1), groups=3)
        # Native pixel interaction, expressed per normalized four-DOF command.
        r = rotations.float()
        tx = -r[:, 0, :, None, None] + self.x*r[:, 2, :, None, None]
        ty = -r[:, 1, :, None, None] + self.y*r[:, 2, :, None, None]
        kx, ky, kz = (r[:, i, 2, None, None] for i in range(3))
        rx = self.x*self.y*kx - (1+self.x.square())*ky + self.y*kz
        ry = (1+self.y.square())*kx - self.x*self.y*ky - self.x*kz
        translation = -self.focal*(gx[:, None]*tx[:, :, None] + gy[:, None]*ty[:, :, None])/depth[:, None]
        yaw = -self.focal*(gx*rx[:, None] + gy*ry[:, None])
        native = torch.cat((translation, yaw[:, None]), 1)
        b = len(raw)
        small = self._down(native.reshape(b, 12, self.size, self.size)).reshape(b, 4, 3, self.net_size, self.net_size)
        return self._down(raw), small*self.action_scale[None, :, None, None, None]

    @torch.no_grad()
    def observe(self, public):
        if set(public) != PUBLIC_KEYS:
            raise ValueError("Only RGB, target RGB, camera rotation and previous action are public")
        images, goals, rotations = public["rgb"], public["goal_rgb"], public["rotation"]
        raw = self._raw(images)
        if rotations.shape != (len(images), 3, 3) or rotations.device != images.device:
            raise ValueError("One public camera/body rotation per image is required")
        depth = self.predict_depth(images)
        x, tangent = self.tangents(images, depth, rotations)
        if not torch.isfinite(tangent).all():
            raise FloatingPointError("Nonfinite GPU image geometry")
        return x, self.encode_images(goals), tangent
