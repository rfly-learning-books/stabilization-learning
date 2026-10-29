from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extensions.fast_m_learning_v2.discrete_tracking import DiscreteConsistentUnicycle,DiscreteConsistentCircle
from stabilization_gym.envs.robotics.robotics import UnicycleTrackingEnv,MultirotorTrackingEnv


def target(env,phase):
    phase=torch.as_tensor(phase,dtype=torch.float64)
    reference=env.reference(phase)
    if isinstance(reference,tuple):reference=reference[0]
    return torch.cat((reference,phase.reshape(1)))


@pytest.mark.parametrize('factory',[DiscreteConsistentUnicycle,DiscreteConsistentCircle])
def test_exact_reference_manifold_one_step_invariance(factory):
    env=factory()
    for phase in np.linspace(-np.pi,np.pi,17,endpoint=False):
        state=target(env,phase);error,context=env.build_error(state)
        assert torch.linalg.vector_norm(error)<1e-14
        for diagonal in (torch.ones(env.error_dim,dtype=torch.float64),
                         torch.linspace(.5,1.5,env.error_dim,dtype=torch.float64)):
            metric=torch.diag(diagonal)
            action=env.calculate_action_from_M(metric,error,context=context)
            successor=env.torch_step(state,action)
            assert torch.linalg.vector_norm(env.tensor_error(successor))<3e-12


@pytest.mark.parametrize('factory',[UnicycleTrackingEnv,lambda:MultirotorTrackingEnv(trajectory='circle')])
def test_old_exact_target_defect_is_independent_of_M(factory):
    env=factory();state=target(env,np.pi/2);error,context=env.build_error(state)
    actions=[]
    for diagonal in (torch.ones(env.error_dim,dtype=torch.float64),
                     torch.linspace(.5,1.5,env.error_dim,dtype=torch.float64)):
        action=env.calculate_action_from_M(torch.diag(diagonal),error,context=context);actions.append(action)
        successor=env.torch_step(state,action)
        assert torch.linalg.vector_norm(env.tensor_error(successor))>1e-8
    assert torch.equal(actions[0],actions[1])


@pytest.mark.parametrize('factory',[DiscreteConsistentUnicycle,DiscreteConsistentCircle])
def test_corrected_controller_still_depends_on_M(factory):
    env=factory();state=target(env,.4);state[0]+=.01
    error,context=env.build_error(state)
    metric=torch.eye(env.error_dim,dtype=torch.float64,requires_grad=True)
    action=env.calculate_action_from_M(metric,error,context=context)
    gradient=torch.autograd.grad(action.square().sum(),metric)[0]
    assert torch.isfinite(gradient).all() and torch.linalg.vector_norm(gradient)>0


@pytest.mark.parametrize('factory',[DiscreteConsistentUnicycle,DiscreteConsistentCircle])
def test_corrected_nominal_target_remains_on_reference_for200steps(factory):
    env=factory();state=target(env,.7);metric=torch.eye(env.error_dim,dtype=torch.float64)
    with torch.no_grad():
        for _ in range(200):
            error,context=env.build_error(state)
            state=env.torch_step(state,env.calculate_action_from_M(metric,error,context=context))
            assert torch.linalg.vector_norm(env.tensor_error(state))<1e-10
