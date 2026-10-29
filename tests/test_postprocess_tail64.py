from pathlib import Path
import sys
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.book_aligned_postprocess_v1.costs import fixed_horizon_cost64


@pytest.mark.parametrize("dt",[.02,.04,.001])
def test_failure_tail_preserves_declared_float64_clock(dt):
    steps=torch.tensor([0,1,141,342,359,499,500],dtype=torch.int64)
    terminal=torch.tensor([.1,1.,30.,2.,3.,4.,5.],dtype=torch.float64)
    failed=torch.tensor([False,True,True,True,True,False,False])
    accumulated=torch.arange(7,dtype=torch.float64)
    actual=fixed_horizon_cost64(accumulated,steps,terminal,failed,horizon=500,dt=dt)
    expected=accumulated.numpy()+(500-steps.numpy())*dt*np.where(failed.numpy(),np.maximum(terminal.numpy(),25.),terminal.numpy())
    assert actual.dtype==torch.float64
    np.testing.assert_allclose(actual.numpy(),expected,rtol=0,atol=1e-12)


def test_original_float32_clock_defect_is_observable_without_relaxing_tolerance():
    n=torch.tensor([342],dtype=torch.int64)
    old=(500-n)*.02*torch.tensor([25.],dtype=torch.float64)
    new=fixed_horizon_cost64(torch.zeros(1),n,torch.ones(1),torch.ones(1,dtype=torch.bool),horizon=500,dt=.02)
    assert abs(float(old-new))>1e-7
    assert float(new)==79.


@pytest.mark.parametrize("name",["pendulum","cartpole"])
def test_new_aggregation_adapter_on_actual_early_failed_model_rollout(name):
    from stabilization_gym import make
    from stabilization_gym.envs.classic_control.sdc_metric import SDCMetricTask
    from stabilization_learning.policies.state_metric_v2 import StateMetricPolicy
    from extensions.book_aligned_postprocess_v1.rollout import rollout_batch64
    torch.set_num_threads(1)
    task=SDCMetricTask(make(name),name)
    policy=StateMetricPolicy(task.metric_spec,task.L0,goal=task.goal,context_scale=.5)
    start=torch.zeros((1,task.error_dim),dtype=torch.float64)
    start[0,0 if name=="pendulum" else 2]=1.8 if name=="pendulum" else .9
    rows,trace=rollout_batch64(task,policy,start,record=True)
    row=rows[0]
    assert row["failure"] and row["physical_steps"]<task.max_steps
    expected=row["accumulated_cost"]+(task.max_steps-row["physical_steps"])*task.dt*max(row["terminal_physical_error"],25.)
    assert abs(row["fixed_horizon_cost"]-expected)<1e-12
    assert abs(row["legacy_v2_summary"]["fixed_horizon_cost"]-expected)>1e-7
    assert trace["active"].sum()==row["physical_steps"]


def test_new_learner_uses_tail64_and_rejects_frozen_v2_resume(tmp_path):
    from stabilization_gym import make
    from stabilization_gym.envs.classic_control.sdc_metric import SDCMetricTask
    from stabilization_learning.policies.state_metric_v2 import StateMetricPolicy
    from stabilization_learning.algos.pytorch.state_metric_model.learner import ModelMetricConfig,ModelMetricLearner
    from extensions.book_aligned_postprocess_v1.learner64 import ModelMetricLearner64
    torch.set_num_threads(1)
    task=SDCMetricTask(make("pendulum"),"pendulum")
    def policy():return StateMetricPolicy(task.metric_spec,task.L0,goal=task.goal,context_scale=.5)
    cfg=ModelMetricConfig(updates=2,batch_states=2,model_horizon=2,validation_seeds=(210001,),test_seeds=(310001,))
    old=ModelMetricLearner(task,policy(),cfg,source_digest="fixture")
    old.validate();old.save(tmp_path/"old.pt")
    new=ModelMetricLearner64(task,policy(),cfg,source_digest="fixture")
    checked=new.validate()
    assert checked["episodes"][0]["aggregation_version"]=="float64_remaining_time_v3"
    assert new.protocol()["schema"]!=old.protocol()["schema"]
    with pytest.raises(ValueError):new.restore(tmp_path/"old.pt")
    new.save(tmp_path/"new.pt")
    replica=ModelMetricLearner64(task,policy(),cfg,source_digest="fixture")
    replica.restore(tmp_path/"new.pt")
    assert replica.best_score==new.best_score
