"""Load project geometry assets into the certificate reference frame."""

from __future__ import annotations

from pathlib import Path

from .model import build_certificate_model


def load_geometry(asset_root, shape):
    from ..geometry import Geometry

    geometry = Geometry(Path(asset_root) / str(shape))
    if geometry.name != str(shape):
        raise ValueError("geometry directory and metadata shape disagree")
    return geometry


def model_origin_to_com_offset(geometry):
    """Return the body-frame vector from model origin to physical COM."""

    reconstruction = geometry.metadata.get("reconstruction", {})
    value = reconstruction.get("image_centroid_world_m")
    if value is None:
        value = geometry.metadata.get("model_origin_xy_in_body", [0.0, 0.0])
    import numpy as np

    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("asset lacks a finite origin-to-COM calibration")
    return value


def certificate_polygon(geometry):
    """Return the evaluation polygon expressed around the physical COM."""

    from shapely.affinity import translate

    offset = model_origin_to_com_offset(geometry)
    return translate(geometry.polygon, xoff=-offset[0], yoff=-offset[1])


def make_certificate_model(geometry, output, **kwargs):
    """Build a free-body, normal-only-pusher model from an existing asset."""

    settings = dict(
        physics_dt=kwargs.pop("physics_dt", .002),
        object_friction=kwargs.pop("object_friction", 3.0),
        floor_friction=kwargs.pop("floor_friction", .5),
        pusher_friction=kwargs.pop("pusher_friction", 0.0),
        actuator_force_limit=kwargs.pop("actuator_force_limit", 30.0),
        actuator_kv=kwargs.pop("actuator_kv", 100.0),
        free_body=kwargs.pop("free_body", True),
    )
    if kwargs:
        raise TypeError(f"unknown certificate model settings: {sorted(kwargs)}")
    return build_certificate_model(geometry.xml, output, **settings)
