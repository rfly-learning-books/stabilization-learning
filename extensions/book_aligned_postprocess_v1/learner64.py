"""Explicit future learner aggregation revision; never loads v2 as a resume."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from stabilization_learning.algos.pytorch.state_metric_model.learner import ModelMetricLearner
from .rollout import rollout_batch64


class ModelMetricLearner64(ModelMetricLearner):
    def protocol(self):
        value=super().protocol()
        value["schema"]="state_metric_model_gradient_tail64_v3"
        value["aggregation_version"]="float64_remaining_time_v3"
        value["aggregation_sources"]={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("learner64.py","costs.py","rollout.py")}
        return value

    def validate(self):
        rows,_=rollout_batch64(self.task,self.policy,self.task.initial_states(self.config.validation_seeds),
            decay=self.config.decay,certificate_weight=self.config.certificate_weight,should_stop=self.should_stop)
        self.validation_physical_steps+=sum(row["physical_steps"] for row in rows)
        objective=float(np.mean([row["objective"] for row in rows]))
        failures=sum(not row["success"] for row in rows)
        score=failures*1e6+objective
        if score<self.best_score:
            self.best_score=score;self.best_update=self.updates;self.best_actor=deepcopy(self.policy.state_dict())
        return dict(update=self.updates,score=score,non_success=failures,
            physical_cost=float(np.mean([r["fixed_horizon_cost"] for r in rows])),
            penalty=float(np.mean([r["certificate_penalty"] for r in rows])),episodes=rows)
