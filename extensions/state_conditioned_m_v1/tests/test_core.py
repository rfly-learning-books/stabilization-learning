from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from extensions.state_conditioned_m_v1.core.metric import StateDependentMetricActor
from extensions.state_conditioned_m_v1.core.gain import (
    information_gain, contraction_gram, contraction_margin, trace_eigen_bounds,
)
from extensions.state_conditioned_m_v1.envs.dc_motor import StateDependentDCMotorEnv


def test_spd_actor_and_bounds():
    torch.set_num_threads(1)
    from stabilization_gym.common.types import MetricSpec
    spec = MetricSpec(error_dim=2, eigen_floor=.05, trace=2.)
    actor = StateDependentMetricActor(context_dim=4, spec=spec, hidden=16)
    context = torch.randn(7, 4)
    M = actor(context)
    assert M.shape == (7, 2, 2)
    sym = .5 * (M + M.transpose(-1, -2))
    assert torch.allclose(M, sym, atol=1e-6)
    trace, lo, hi = trace_eigen_bounds(M)
    assert torch.allclose(trace, torch.full_like(trace, 2.), atol=2e-5)
    assert bool((lo >= .05 - 2e-5).all())
    assert bool((hi <= 1.95 + 2e-5).all())


def test_gain_shape_and_contraction_diagnostics():
    torch.set_num_threads(1)
    M = torch.eye(2, dtype=torch.float64).reshape(1, 2, 2)
    H = torch.tensor([[1., 0.]], dtype=torch.float64)
    R = torch.tensor([[.1]], dtype=torch.float64)
    L = information_gain(M, H, R)
    assert L.shape == (1, 2, 1)
    A = torch.tensor([[0., 1.], [-.2, -.3]], dtype=torch.float64)
    gram = contraction_gram(A, H, M, L, .01)
    assert gram.shape == (1, 2, 2)
    margin = contraction_margin(A, H, M, L, .01)
    assert margin.shape == (1,)


def test_dc_motor_state_dependent_step():
    torch.set_num_threads(1)
    from stabilization_gym.common.types import MetricSpec
    env = StateDependentDCMotorEnv(target_speed=.5)
    obs, _ = env.reset(seed=123, options={"state": np.array([1.0, .5]), "target_speed": .5})
    spec = MetricSpec(error_dim=2, eigen_floor=.05, trace=2.)
    actor = StateDependentMetricActor(context_dim=4, spec=spec, hidden=16)
    context = env.metric_context(obs)
    M = actor(context)
    e, control_context = env.build_error(obs)
    action = env.controller_action(M, e, control_context)
    following, cost, terminated, truncated, info = env.step(action)
    assert following.shape == (2,)
    assert np.isfinite(cost) and cost >= 0
    trace = env.get_episode_audit()["metric_trace"]
    assert len(trace) == 1
    assert trace[0]["M"] is not None
    assert trace[0]["V_before"] is not None
    assert trace[0]["V_after"] is not None
