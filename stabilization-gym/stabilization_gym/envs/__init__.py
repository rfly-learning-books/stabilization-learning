"""Task constructors with no dependency on the learning package."""
from .base import StateMetricEnv
from .classic_control import CoupledLinearEnv, PointMassEnv, PendulumEnv, CartPoleEnv
from .robotics import TwoLinkArmEnv, UnicycleTrackingEnv, MultirotorTrackingEnv

ENVIRONMENTS = {
    "coupled_linear": CoupledLinearEnv,
    "point_mass": PointMassEnv,
    "pendulum": PendulumEnv,
    "cartpole": CartPoleEnv,
    "two_link_arm": TwoLinkArmEnv,
    "unicycle_tracking": UnicycleTrackingEnv,
    "multirotor_tracking": MultirotorTrackingEnv,
}


def available_envs():
    return tuple(ENVIRONMENTS) + ("push_any_shape", "visual_servo")


def make(env_id, **kwargs):
    if env_id == "push_any_shape":
        from .tasks.push_any_shape import PushAnyShapeEnv
        return PushAnyShapeEnv(**kwargs)
    if env_id == "visual_servo":
        from .tasks.visual_servo import VisualServoEnv
        return VisualServoEnv(**kwargs)
    if env_id not in ENVIRONMENTS:
        raise KeyError(f"Unknown env {env_id!r}. Available: {available_envs()}")
    return ENVIRONMENTS[env_id](**kwargs)


def __getattr__(name):
    if name == "PushAnyShapeEnv":
        from .tasks.push_any_shape import PushAnyShapeEnv
        return PushAnyShapeEnv
    if name == "VisualServoEnv":
        from .tasks.visual_servo import VisualServoEnv
        return VisualServoEnv
    raise AttributeError(name)
