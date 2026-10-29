"""Library paths retain the existing full-horizon mathematical objectives."""
from copy import deepcopy

import pytest
import torch

from stabilization_learning.library.model_gradient import ModelGradientLearner
from stabilization_gym.observation import make_observation_task
from extensions.observer_learning_v1.learning import MetricNetwork,episode_data,rollout
from extensions.observer_learning_v1.tensor_model import TensorModel


def observer_fixture():
    torch.set_num_threads(1);torch.manual_seed(829)
    task=make_observation_task("motor_speed")
    network=MetricNetwork(task.model,hidden=8)
    learner=ModelGradientLearner.from_observer_task(task,network,device="cpu")
    batch=learner.prepare_batch([episode_data("motor_speed","train",60000,"nominal"),
                                 episode_data("motor_speed","train",60001,"noise")])
    return learner,batch


def test_observer_loss_gradient_and_adam_update_match_original():
    learner,batch=observer_fixture()
    original=deepcopy(learner.network)
    optimizer=torch.optim.Adam(original.parameters(),lr=.003)
    direct=rollout(original,TensorModel(learner.model),batch)
    wrapped=learner.loss(batch)
    torch.testing.assert_close(wrapped["loss"],direct["loss"],rtol=0,atol=0)
    grad1=torch.autograd.grad(wrapped["loss"],tuple(learner.network.parameters()))
    grad2=torch.autograd.grad(direct["loss"],tuple(original.parameters()))
    for a,b in zip(grad1,grad2):torch.testing.assert_close(a,b,rtol=0,atol=0)
    optimizer.zero_grad(set_to_none=True)
    reference=rollout(original,TensorModel(learner.model),batch)
    reference["loss"].backward();torch.nn.utils.clip_grad_norm_(original.parameters(),2.);optimizer.step()
    result=learner.update(batch)
    for a,b in zip(learner.network.parameters(),original.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert result["training_transitions"]==400
    assert result["horizon_steps"]==200
    assert learner.capabilities["supports_cuda"]


def test_observer_refuses_short_horizon_and_cpu_only_state_is_explicit():
    learner,batch=observer_fixture()
    batch["truth"]=batch["truth"][:-1]
    with pytest.raises(ValueError,match="complete original"):
        learner.loss(batch)
    with pytest.raises(ValueError,match="CPU float64 only"):
        ModelGradientLearner.from_state_task(None,None,device="cuda")


def test_state_delegates_original_full_horizon_and_preserves_initialization():
    from stabilization_gym import make
    from stabilization_gym.envs.classic_control.sdc_full_horizon import FullHorizonSDCTask
    from stabilization_learning.policies.state_metric_residual_v3 import AnchoredStateMetricPolicy
    from stabilization_learning.algos.pytorch.state_metric_full_horizon import FullHorizonConfig,FullHorizonMetricLearner
    from stabilization_learning.algos.pytorch.state_metric_full_horizon.objectives import full_horizon_components
    torch.set_num_threads(1);torch.manual_seed(829)
    task=FullHorizonSDCTask(make("point_mass"),"point_mass")
    network=AnchoredStateMetricPolicy(task.metric_spec,task.L0,goal=task.goal,
        context_scale=max(task.physical.initial_scale,.1),hidden=8)
    initial=deepcopy(network.state_dict())
    wrapped=ModelGradientLearner.from_state_task(task,network,FullHorizonConfig(),device="auto")
    assert isinstance(wrapped.learner,FullHorizonMetricLearner)
    assert wrapped.capabilities["device"]=="cpu"
    for key,value in network.state_dict().items():torch.testing.assert_close(value,initial[key],rtol=0,atol=0)
    starts=task.initial_states([910001,910002])
    parts=full_horizon_components(task,wrapped.network,starts,checkpoint_stride=0)
    assert parts.shape==(2,5) and torch.isfinite(parts).all()
    assert task.max_steps==wrapped.capabilities["full_horizon_steps"]
    grads=torch.autograd.grad(parts[:,0].mean(),tuple(network.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
