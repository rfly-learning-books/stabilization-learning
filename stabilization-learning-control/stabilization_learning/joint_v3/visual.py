"""Execution upgrade of joint RGB learning; retain v2 objectives and updates."""
from stabilization_gym.tasks_v3.visual import AcceleratedVisualTaskAdapter
from stabilization_learning.joint_v2.visual import (
    JointVisualLearner as ReferenceJointVisualLearner,
    JointVisualConfig, GoalConditionedMetric, VisualBatch, TinyRGBBackbone,
)
from .features import AcceleratedRGBFeatures


class AcceleratedVisualLearner(ReferenceJointVisualLearner):
    """Same losses, same phi/M block ordering, optional feature derivative API.

    No inter-update feature/J cache is used. The M block always recomputes J
    through the just-updated encoder. Reconstruction remains in both diagnostic
    totals so they keep the same meaning as the reference implementation.
    """
    def __init__(self, features, metric, adapter=None, config=None):
        if not isinstance(features, AcceleratedRGBFeatures):
            raise TypeError("Use the explicit accelerated feature provider")
        super().__init__(features, metric, adapter or AcceleratedVisualTaskAdapter(), config)
        # Standalone checkpoint loading cannot silently cross execution backends.
        self.checkpoint_format = "joint_visual_v3_1/" + features.backend

    @property
    def execution_spec(self):
        return dict(implementation="joint_rgb_v3", interaction_backend=self.features.backend,
                    feature_provider="normalized_shared_rgb_with_directional_derivatives",
                    objective="joint_visual_v2_unchanged", physical_interface="tasks_v2.contracts")
