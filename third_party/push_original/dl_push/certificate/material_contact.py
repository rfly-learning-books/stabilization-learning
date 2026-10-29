"""Absolute-normal-speed primitive for a fixed material contact on one face."""
import numpy as np
from .math import rot2


def material_contact_velocity(position, yaw, velocity, omega, pusher_position,
                              point_body, normal_body, radius, speed, gain=80., limit=.25):
    """Return a raw world velocity; reject saturation instead of changing u.

    All points are XY about the object's physical COM. Normal speed is an
    absolute pusher velocity component, not relative penetration speed.
    Tangential rigid-body transport follows the prescribed material point.
    Finite contact compliance must be measured separately by the caller.
    """
    point_body = np.asarray(point_body, dtype=float)
    normal_body = np.asarray(normal_body, dtype=float)
    if not np.isclose(np.linalg.norm(normal_body), 1., atol=1e-10):
        raise ValueError('unit normal required')
    if speed < 0 or not np.isfinite(speed):
        raise ValueError('nonnegative finite speed required')
    R=rot2(yaw)
    anchor=point_body+radius*normal_body
    tangent=np.array([-normal_body[1],normal_body[0]])
    local=R.T@(np.asarray(pusher_position)-position)
    transport=R.T@velocity+omega*np.array([-anchor[1],anchor[0]])
    command_body=-speed*normal_body+tangent*(tangent@transport-gain*(tangent@(local-anchor)))
    if not np.isfinite(command_body).all() or np.linalg.norm(command_body)>limit:
        raise ValueError('material-contact command exceeds speed limit')
    return R@command_body
