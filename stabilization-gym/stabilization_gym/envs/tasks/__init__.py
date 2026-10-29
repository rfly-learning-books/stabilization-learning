"""Unified task adapters. Assets load only when their task is requested."""


def __getattr__(name):
    if name == "PushAnyShapeEnv":
        from .push_any_shape import PushAnyShapeEnv
        return PushAnyShapeEnv
    if name in ("VisualServoEnv", "LegacyVisualServoEnv"):
        from . import visual_servo
        return getattr(visual_servo, name)
    raise AttributeError(name)
