"""Explicit independent-task routes for reproducible new experiments."""
import math
from numbers import Integral
import numpy as np
import torch

from scripts.m_learning_suite.task_runtime import FullHorizonTask, make_task as legacy_task
from extensions.fast_m_learning_v2.discrete_tracking import DiscreteConsistentUnicycle, DiscreteConsistentCircle
from stabilization_gym.envs.tasks.visual_servo import VisualServoEnv


class SafeVisualServoEnv(VisualServoEnv):
    def __init__(self, **kwargs):
        for key in ("image_size", "error_dim", "max_steps"):
            if key in kwargs and (isinstance(kwargs[key], bool) or not isinstance(kwargs[key], Integral) or kwargs[key] < 1):
                raise ValueError(f"{key} must be a positive integer")
        for key in ("estimated_depth", "depth_scale"):
            if key in kwargs and (not math.isfinite(kwargs[key]) or kwargs[key] <= 0):
                raise ValueError(f"{key} must be finite and positive")
        super().__init__(**kwargs)
        p1 = self.plant_name == "P1"
        self.reference_model_version = "portable_visual_validated_goal_v4"
        self.controller_form = "weighted_fixed_reference_feedback"
        self.controller_capabilities = dict(self.controller_capabilities,
            model_gradient=not p1, differentiable_model=not p1,
            augmented_model_gradient=True, plant_has_memory=p1,
            prediction_input="applied_velocity; commanded prediction needs P1 velocity memory",
            observation_scope="RGB only; P1 velocity is internal plant memory")
        self.asset_manifest = dict(self.asset_manifest, protocol_version=self.reference_model_version,
            plant_has_memory=p1, predict_next_uses="applied_velocity")

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        goal = np.asarray(options.get("goal", [0., 0., 2., 0.]), dtype=float)
        if (goal.shape != (4,) or not np.isfinite(goal).all() or
                np.linalg.norm(goal[:2]) > 3 or not .35 <= goal[2] <= 4):
            raise ValueError("Goal must be finite [x,y,height,yaw] inside the declared workspace")
        return super().reset(seed=seed, options=options)

    def predict_next(self, state, applied_action):
        """Pose kinematics from ACTUAL velocity, not a P1 commanded-input model."""
        return super().predict_next(state, applied_action)

    def _info(self, failure=False):
        info = super()._info(failure)
        if failure:
            info["success"] = info["is_success"] = False
        return info

    def predict_commanded(self, state, command, *, velocity=None):
        """Pure complete-state model: return next pose and actual velocity."""
        state = torch.as_tensor(state)
        command = torch.as_tensor(command, dtype=state.dtype, device=state.device)
        if self.plant_name == "P1":
            if velocity is None:
                raise ValueError("P1 commanded prediction requires the velocity memory")
            velocity = torch.as_tensor(velocity, dtype=state.dtype, device=state.device)
            applied = velocity + .45*(command-velocity)
        else:
            applied = command
        if not torch.isfinite(applied).all() or not torch.isfinite(state).all():
            raise FloatingPointError("Nonfinite visual plant state or action")
        return self.predict_next(state, applied), applied


def make_task(task_id, env_kwargs=None, *, hold_seconds=1., tracking_cycles=2.):
    """Fresh independent tasks; moving references are discrete-consistent by default."""
    kwargs = dict(env_kwargs or {})
    if task_id == "visual_servo":
        return SafeVisualServoEnv(**kwargs)
    if task_id == "push_any_shape":
        from .push import SafePushAnyShapeEnv
        return SafePushAnyShapeEnv(**kwargs)
    if task_id in ("multirotor_hover", "multirotor_circle"):
        trajectory = "hover" if task_id.endswith("hover") else "circle"
        if "trajectory" in kwargs and kwargs["trajectory"] != trajectory:
            raise ValueError("Task id and trajectory disagree")
        kwargs["trajectory"] = trajectory
        task_id = "multirotor_tracking"
    for key in ("dt", "initial_scale", "radius", "reference_speed", "reference_omega", "success_tolerance"):
        if key in kwargs and (isinstance(kwargs[key], bool) or not math.isfinite(kwargs[key]) or kwargs[key] <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if "max_steps" in kwargs and (isinstance(kwargs["max_steps"], bool) or not isinstance(kwargs["max_steps"], Integral) or kwargs["max_steps"] < 1):
        raise ValueError("max_steps must be a positive integer")
    if task_id == "unicycle_tracking":
        inner = DiscreteConsistentUnicycle(**kwargs)
        chord_speed = 2*inner.radius*math.sin(inner.dt*inner.reference_omega/2)/inner.dt
        if not 0 < inner.dt*inner.reference_omega < math.pi/2 or abs(chord_speed) > 1 or inner.reference_omega > 2.5:
            raise ValueError("Declared unicycle reference exceeds the discrete input envelope")
    elif task_id == "multirotor_tracking" and kwargs.get("trajectory", "hover") == "circle":
        inner = DiscreteConsistentCircle(**kwargs)
    else:
        task = legacy_task(task_id, kwargs, hold_seconds=hold_seconds, tracking_cycles=tracking_cycles)
        task.reference_model_version = "preserved_static_physics_full_horizon_v4"
        return task
    task = FullHorizonTask(inner, task_id, hold_seconds=hold_seconds, tracking_cycles=tracking_cycles)
    task.reference_model_version = inner.reference_model_version
    return task
