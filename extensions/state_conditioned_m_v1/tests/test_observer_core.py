from __future__ import annotations
import sys
from pathlib import Path
import torch
PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from extensions.state_conditioned_m_v1.core.metric import StateDependentMetricActor
from extensions.state_conditioned_m_v1.observers.core import MetricObserver
from extensions.state_conditioned_m_v1.observers.linear_toy import LinearObserverToy
from stabilization_gym.common.types import MetricSpec


def test_observer_context_has_no_truth_field_and_update_runs():
    torch.set_num_threads(1)
    toy = LinearObserverToy(torch.tensor([[1., .05], [0., .95]]),
                            torch.eye(2), torch.eye(2) * .01, torch.eye(2) * .001)
    obs = toy.reset(seed=3)
    assert not hasattr(obs, "true_state")
    fields = set(obs.__dataclass_fields__)
    assert "true_state" not in fields
    # Feature dimension is inferred from the toy contract: estimate 2,
    # innovation 2, Jacobian 4, quality 2, control 1, prior metric 4.
    actor = StateDependentMetricActor(15, MetricSpec(2, .05, 2.), hidden=16)
    observer = MetricObserver(actor, torch.eye(2) * .01)
    next_context, update = toy.step(observer)
    assert update.gain.shape == (2, 2)
    assert update.correction.shape == (2,)
    assert update.metric.shape == (2, 2)
    assert torch.isfinite(update.posterior_estimate).all()
    assert torch.isfinite(next_context.innovation).all()
