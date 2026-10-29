"""Frozen-policy microbatch service; owns no physical state or optimizer."""
from collections import OrderedDict
import hashlib
import torch
from stabilization_gym.tasks_v2.visual import VisualPolicyOutput, quadratic_value
from stabilization_gym.tasks_v2.contracts import TaskContext


class FrozenRGBPolicy:
    """Cache only goal features/M, reject parameter changes during a session.

    Current-image features and L are computed every call. Per-image derivative
    semantics do not depend on the number of independent environments in a batch.
    """
    def __init__(self, learner, *, identity=False, max_goals=32):
        if max_goals < 1:
            raise ValueError("max_goals must be positive")
        self.learner, self.identity, self.max_goals = learner, identity, max_goals
        self._stamp = self._version()
        self._goals = OrderedDict()

    def _version(self):
        modules = (self.learner.features, self.learner.metric)
        return (self.learner.version, getattr(self.learner.features, "backend", None),
                tuple((id(t), t._version, t.device, t.dtype) for m in modules
                      for t in (*m.parameters(), *m.buffers())))

    def _check(self):
        if self._version() != self._stamp:
            raise RuntimeError("Frozen RGB policy parameter/coordinate version changed; create a new session")
        if self.learner.features.training or self.learner.metric.training:
            raise RuntimeError("Frozen RGB policy requires deterministic eval modules")

    @staticmethod
    def goal_key(raw_goal):
        # Hash raw public bytes before any GPU transfer, once per case.
        return (raw_goal.shape, raw_goal.dtype.str, hashlib.sha256(raw_goal.tobytes()).hexdigest())

    @torch.no_grad()
    def __call__(self, observation, *, goal_keys=None):
        self._check()
        observation.validate()
        if goal_keys is None:
            # Explicit no-cache path when a caller has not supplied image identities.
            goals = self.learner.features.encode(observation.goal_image)
            metrics = self._metric(goals)
        else:
            if len(goal_keys) != len(observation.image):
                raise ValueError("One goal identity is required per image")
            missing = {}
            for i, key in enumerate(goal_keys):
                if key not in self._goals:
                    missing.setdefault(key, i)
            if missing:
                indices = list(missing.values())
                encoded = self.learner.features.encode(observation.goal_image[indices])
                matrix = self._metric(encoded)
                for i, key in enumerate(missing):
                    self._goals[key] = (encoded[i:i+1].clone(), matrix[i:i+1].clone())
            goals = torch.cat([self._goals[k][0] for k in goal_keys])
            metrics = torch.cat([self._goals[k][1] for k in goal_keys])
            for key in goal_keys:
                self._goals.move_to_end(key)
            while len(self._goals) > self.max_goals:
                self._goals.popitem(last=False)
        error = self.learner.features.encode(observation.image) - goals
        interaction = self.learner.adapter.interaction(self.learner.features, observation.image,
                                                       observation.image_tangents, create_graph=False)
        context = TaskContext(error=error, metric_context=goals, model_context={"interaction": interaction})
        plan = self.learner.adapter.construct_action(context, lambda goal: metrics)
        self._check()
        return VisualPolicyOutput(plan.action, plan.diagnostics["raw_action"], error, goals, interaction,
                                  metrics, plan.diagnostics["V"], plan.diagnostics["hessian"])

    def _metric(self, goal):
        if self.identity:
            return torch.eye(goal.shape[-1], device=goal.device, dtype=goal.dtype).expand(len(goal), -1, -1)
        return self.learner.metric(goal)
