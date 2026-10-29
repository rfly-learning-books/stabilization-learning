"""Task-side J assembly while differentiation stays with the feature provider.

The physical geometry, error definition, control solve and TaskAdapter contract
are inherited unchanged. This package never imports a learning implementation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from ..tasks_v2.visual import VisualControlConfig, VisualTaskAdapter


@runtime_checkable
class FeatureProvider(Protocol):
    def encode(self, image: Tensor) -> Tensor: ...


@runtime_checkable
class InteractionJVPProvider(FeatureProvider, Protocol):
    """Optional capability returning B,A,D derivatives in encode coordinates."""

    def interaction_jvp(self, image: Tensor, image_tangents: Tensor, *,
                        create_graph: bool = True) -> Tensor: ...


class AcceleratedVisualTaskAdapter(VisualTaskAdapter):
    """Use provider JVPs; retain the declared legacy path for encode-only users.

    Set ``encode_only_backend='error'`` to require the optional capability.
    Otherwise an encode-only provider uses v2's documented legacy derivative
    adapter. A provider capability that raises is never silently bypassed.
    """

    def __init__(self, config: VisualControlConfig | None = None, *,
                 encode_only_backend: str = "legacy"):
        if encode_only_backend not in {"legacy", "error"}:
            raise ValueError("encode_only_backend must be 'legacy' or 'error'")
        super().__init__(config)
        self.encode_only_backend = encode_only_backend

    def interaction(self, features: FeatureProvider, image: Tensor, tangents: Tensor, *,
                    create_graph: bool = True) -> Tensor:
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Visual interaction derivatives require AD; use torch.no_grad(), not inference_mode")
        if not isinstance(create_graph, bool):
            raise TypeError("create_graph must be a boolean")
        derivative = getattr(features, "interaction_jvp", None)
        if derivative is None:
            if self.encode_only_backend == "error":
                raise TypeError("Feature provider needs interaction_jvp; configure the explicit legacy fallback for encode-only providers")
            return super().interaction(features, image, tangents, create_graph=create_graph)
        if not callable(derivative):
            raise TypeError("Feature provider interaction_jvp capability must be callable")
        directions = derivative(image, tangents, create_graph=create_graph)
        if (not isinstance(directions, Tensor) or directions.ndim != 3
                or directions.shape[:2] != tangents.shape[:2] or directions.shape[2] == 0):
            raise ValueError("Feature provider interaction_jvp must return B,A,D derivatives")
        if directions.dtype != image.dtype or directions.device != image.device:
            raise ValueError("Feature derivatives must share the input dtype and device")
        # Geometry owns action-axis order; the feature provider owns D and scale.
        matrix = directions.transpose(1, 2)
        return matrix if create_graph else matrix.detach()


__all__ = ["FeatureProvider", "InteractionJVPProvider", "AcceleratedVisualTaskAdapter"]
