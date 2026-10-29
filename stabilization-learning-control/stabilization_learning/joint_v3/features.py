"""Feature-owned image directional derivatives with unchanged coordinates.

The task supplies image-motion directions. This provider owns the encoder,
registered normalization and differentiation backend. No model weights or
optimizer behavior are changed by choosing a derivative implementation.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..joint_v2.visual import NormalizedRGBFeatures


class AcceleratedRGBFeatures(NormalizedRGBFeatures):
    """Optional ``interaction_jvp`` capability for normalized RGB features.

    ``interaction_jvp`` returns B,A,D directional feature derivatives; the task
    assembles the B,D,A interaction matrix. ``forward`` is the default forward-AD
    loop. ``legacy`` is the preserved reverse-over-reverse implementation and
    ``vmap`` batches forward-AD directions explicitly. Unsupported operations
    raise from the selected backend; there is no silent backend/gradient switch.

    State-dict keys remain compatible with NormalizedRGBFeatures. A runner must
    record and validate ``execution_spec()`` alongside checkpoint state, because
    the differentiation backend is execution metadata, not a learned tensor.
    """

    _BACKENDS = frozenset({"forward", "legacy", "vmap"})

    def __init__(self, backbone: nn.Module, latent_dim: int, *, backend: str = "forward"):
        if backend not in self._BACKENDS:
            raise ValueError("JVP backend must be 'forward', 'legacy', or 'vmap'")
        super().__init__(backbone, latent_dim)
        self._jvp_backend = backend

    @property
    def backend(self) -> str:
        return self._jvp_backend

    def execution_spec(self) -> dict[str, str]:
        return {
            "schema": "normalized_rgb_interaction_jvp_v1",
            "backend": self.backend,
            "output_layout": "batch,action,feature",
            "coordinates": "encode_registered_center_and_scale",
            "create_graph_false": "detached",
            "inference_mode": "rejected",
        }

    def interaction_jvp(self, image: Tensor, image_tangents: Tensor, *,
                        create_graph: bool = True) -> Tensor:
        """Differentiate this provider's actual normalized ``encode`` function.

        ``torch.no_grad`` is supported for inference. ``inference_mode`` disables
        forward AD and is rejected even for the compatibility backend, so an
        inference optimization cannot silently turn the controller's J into 0.
        ``create_graph=False`` detaches the result, also when outer grad is on.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError("interaction_jvp requires forward AD; use torch.no_grad(), not inference_mode")
        if not isinstance(create_graph, bool):
            raise TypeError("create_graph must be a boolean")
        if image.ndim != 4 or image.shape[0] == 0:
            raise ValueError("JVP images must have nonempty B,C,H,W shape")
        if (image_tangents.ndim != 5 or image_tangents.shape[0] != image.shape[0]
                or image_tangents.shape[1] == 0 or image_tangents.shape[2:] != image.shape[1:]):
            raise ValueError("Image directions must have matching nonempty B,A,C,H,W shape")
        if (not image.is_floating_point() or not image_tangents.is_floating_point()
                or image_tangents.dtype != image.dtype or image_tangents.device != image.device):
            raise ValueError("Images and directions must share a floating dtype and device")

        # Explicit create_graph takes the same precedence as the legacy JVP API.
        # no_grad does not disable torch.func's forward-mode directional tangent.
        with torch.set_grad_enabled(create_graph):
            if self.backend == "legacy":
                directions = [torch.autograd.functional.jvp(
                    self.encode, image, image_tangents[:, axis],
                    create_graph=create_graph, strict=False,
                )[1] for axis in range(image_tangents.shape[1])]
                result = torch.stack(directions, dim=1)
            elif self.backend == "forward":
                result = torch.stack([
                    torch.func.jvp(self.encode, (image,), (image_tangents[:, axis],))[1]
                    for axis in range(image_tangents.shape[1])
                ], dim=1)
            else:
                def directional(direction: Tensor) -> Tensor:
                    return torch.func.jvp(self.encode, (image,), (direction,))[1]

                result = torch.vmap(directional)(image_tangents.movedim(1, 0)).movedim(0, 1)
        expected = (len(image), image_tangents.shape[1], self.latent_dim)
        if tuple(result.shape) != expected:
            raise ValueError(f"Feature JVP returned {tuple(result.shape)}, expected {expected}")
        return result if create_graph else result.detach()


__all__ = ["AcceleratedRGBFeatures"]
