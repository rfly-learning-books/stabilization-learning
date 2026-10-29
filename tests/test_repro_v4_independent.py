"""CPU-only engineering regressions, not new scientific performance runs."""
from copy import deepcopy
import math
import numpy as np
import pytest
import torch

from extensions.reproducible_training_v4.independent import make_task, SafeVisualServoEnv
from extensions.reproducible_training_v4.rollout import rollout_metric
from extensions.reproducible_training_v4.ppo import StableEpisodeLearner, TrainConfig
from stabilization_learning.trainer import module_hash


TASKS = [
    ("two_link_arm",dict(max_steps=60,initial_scale=.01)),
    ("unicycle_tracking",dict(radius=.1,reference_speed=.2,dt=.1,max_steps=8,initial_scale=.01)),
    ("multirotor_circle",dict(radius=.03,reference_omega=1.,dt=.1,max_steps=8,initial_scale=.01)),
]


def _learner(task="two_link_arm", kwargs=None):
    torch.set_num_threads(1)
    kwargs = kwargs or dict(max_steps=60,initial_scale=.01)
    return StableEpisodeLearner(make_task(task,kwargs), config=TrainConfig(
        episodes=2,batch_episodes=1,epochs=1,seed=97231,validation_seeds=(97331,)),
        env_id=task,env_kwargs=kwargs)


@pytest.mark.parametrize("task,kwargs", TASKS)
def test_three_real_tasks_same_seed_actual_ppo_updates_are_exact(task,kwargs):
    records = []
    for _ in range(2):
        learner = _learner(task,kwargs)
        initial = module_hash(learner.actor)
        batch = learner._collect(1)
        losses = learner._update(*batch[:4])
        evaluation = learner.evaluate()
        records.append((module_hash(learner.actor),module_hash(learner.value),losses,evaluation,batch[4]))
        assert learner.updates==1 and learner.episodes==1
        assert module_hash(learner.actor)!=initial
        learner.close()
    assert records[0]==records[1]


def test_save_restore_replays_next_real_batch_exactly(tmp_path):
    first = _learner()
    first._update(*first._collect(1)[:4])
    path = tmp_path / "whole_batch.pt"
    first.save(path)
    batch = first._collect(1)
    expected = first._update(*batch[:4]),module_hash(first.actor),module_hash(first.value),batch[4]
    first.close()
    restored = _learner()
    restored.restore(path)
    batch = restored._collect(1)
    actual = restored._update(*batch[:4]),module_hash(restored.actor),module_hash(restored.value),batch[4]
    assert actual==expected
    restored.close()


@pytest.mark.parametrize("task", ["unicycle_tracking","multirotor_circle"])
def test_new_default_reference_has_zero_error_invariance(task):
    task = make_task(task)
    env = task.inner_env
    for phase in np.linspace(-math.pi,math.pi,9):
        phase = torch.tensor(phase,dtype=torch.float64)
        reference = env.reference(phase)
        if isinstance(reference,tuple): reference=reference[0]
        state = torch.cat((reference,phase.reshape(1)))
        error,ctx=env.build_error(state)
        action=env.calculate_action_from_M(torch.eye(env.error_dim,dtype=torch.float64),error,context=ctx)
        assert env.tensor_error(env.torch_step(state,action)).norm()<1e-12
    assert task.horizon_seconds>=2*task.reference_period
    task.close()


def test_zero_goal_penalty_and_arrival_keep_physical_meaning():
    env=make_task("two_link_arm",dict(max_steps=60))
    obs,_=env.reset(options={"state":env.goal.copy()},seed=97431)
    result=rollout_metric(env,torch.eye(4,dtype=torch.float64),obs,certificate_weight=.1)
    assert result["success"]
    assert result["objective"]==0. and result["certificate_penalty"]==0.
    assert result["arrival_time"]==1. and result["physical_steps"]==60
    env.close()


@pytest.mark.parametrize("goal", [[float("nan"),0,2,0],[0,0,.1,0],[4,0,2,0]])
def test_visual_invalid_goal_rejected_before_reset(goal):
    env=SafeVisualServoEnv(image_size=16)
    with pytest.raises(ValueError,match="Goal"):
        env.reset(options={"state":[0,0,2,0],"goal":goal})
    assert not env._ready
    env.close()


def test_visual_p1_complete_model_and_action_trace():
    env=SafeVisualServoEnv(image_size=16,max_steps=1,plant="P1")
    obs,_=env.reset(seed=97531,options={"state":[.1,0,2,0]})
    state=torch.as_tensor(env.state.copy())
    with pytest.raises(ValueError,match="memory"):
        env.predict_commanded(state,torch.tensor([.2,0,0,0]))
    predicted,velocity=env.predict_commanded(state,torch.tensor([.2,0,0,0]),velocity=torch.zeros(4))
    env.calculate_action_from_M=lambda *_args,**_kwargs:torch.tensor([.2,0,0,0])
    result=rollout_metric(env,torch.eye(16),obs,save_trajectory=True)
    row=result["trajectory"][0]
    np.testing.assert_allclose(env.state,predicted.numpy(),rtol=0,atol=1e-12)
    np.testing.assert_allclose(row["applied_action"],velocity.numpy(),rtol=0,atol=1e-12)
    assert row["action"]==row["applied_action"]
    assert row["commanded_action"]!=row["applied_action"]
    assert not env.controller_capabilities["model_gradient"]
    env.close()


def test_visual_hold_resets_on_interruption():
    env=SafeVisualServoEnv(image_size=16)
    env.reset(seed=97531,options={"state":[0,0,2,0]})
    for _ in range(10):env.step(np.zeros(4))
    assert env._hold==10 and not env._done
    env.step([.3,0,0,0]);assert env._hold==0
    env.step([-.3,0,0,0]);assert env._hold==1
    for _ in range(10):env.step(np.zeros(4))
    assert env._hold==11 and env._done
    env.close()


def test_visual_physical_failure_dominates_hold_success():
    env=SafeVisualServoEnv(image_size=16)
    env.reset(seed=97531,options={"goal":[0,0,.35,0],"state":[0,0,.3605,0]})
    for _ in range(11):
        _,_,done,_,info=env.step(np.array([0.,0.,-.01,0.]))
    assert env._hold==11 and done and info["failure"]
    assert not info["success"] and not info["is_success"]
    env.close()


def test_finite_return_overflow_is_atomic_and_blocks_future_save(tmp_path):
    learner=_learner()
    obs,_=learner.env.reset(seed=97531)
    context=learner._context(obs).unsqueeze(0)
    with torch.no_grad():
        dist=learner.actor.distribution(context);raw=dist.sample();logp=dist.log_prob(raw)
    before=(module_hash(learner.actor),module_hash(learner.value))
    with pytest.raises(FloatingPointError,match="losses"):
        learner._update(context,raw,logp,torch.tensor([1e20]))
    assert before==(module_hash(learner.actor),module_hash(learner.value))
    assert learner.updates==0
    with pytest.raises(FloatingPointError,match="stopped"):
        learner.save(tmp_path/"bad.pt")
    assert not (tmp_path/"bad.pt").exists()
    learner.close()


def test_restoring_bad_checkpoint_does_not_mutate_parameters(tmp_path):
    learner=_learner();learner._update(*learner._collect(1)[:4])
    path=tmp_path/"healthy.pt";learner.save(path)
    payload=torch.load(path,weights_only=False)
    next(iter(payload["actor"].values())).fill_(float("nan"))
    bad=tmp_path/"invalid.pt";torch.save(payload,bad)
    before=module_hash(learner.actor)
    with pytest.raises(FloatingPointError):learner.restore(bad)
    assert before==module_hash(learner.actor)
    learner.close()
