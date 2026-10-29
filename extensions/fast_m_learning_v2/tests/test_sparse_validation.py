from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from extensions.fast_m_learning_v2 import SparseValidationConfig, SparseValidationLearner
from stabilization_gym.envs.classic_control.classic import PointMassEnv
from stabilization_learning.trainer import StabilizationLearner, TrainConfig


def learner(interval=5):
    return SparseValidationLearner(PointMassEnv(max_steps=8, initial_scale=.2),
        config=SparseValidationConfig(episodes=12, batch_episodes=2, epochs=1,
            validation_every_updates=interval, validation_seeds=(10001,10002)))


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else:
        assert a==b


def test_initial_every_five_and_nonmultiple_final_validation(tmp_path):
    model=learner()
    calls=[]
    evaluate=model.evaluate
    def record(*args,**kwargs):
        calls.append(model.updates)
        return evaluate(*args,**kwargs)
    model.evaluate=record
    model.train(tmp_path/'run')
    assert calls==[0,5,6]
    assert [r['updates'] for r in model.history if r['validated']]==[0,5,6]
    assert all(r['validation'] is None for r in model.history if not r['validated'])
    saved=torch.load(tmp_path/'run/best.pt',map_location='cpu',weights_only=False)
    assert saved['selected_update'] in calls
    assert saved['score']==min(r['validation']['objective'] for r in model.history if r['validated'])


def test_resume_before_validation_reproduces_optimizer_rng_and_selection(tmp_path):
    continuous=learner();continuous.train(tmp_path/'continuous')
    first=learner();first.train(tmp_path/'resumed',max_updates=3)
    assert first.updates==3 and first.history[-1]['validation'] is None
    assert first.optimizer.state_dict()['state']
    resumed=learner();resumed.train(tmp_path/'resumed',resume=True)
    for name in ('actor','value','optimizer','value_optimizer'):
        equal(getattr(continuous,name).state_dict(),getattr(resumed,name).state_dict())
    equal(continuous.history,resumed.history)
    equal(continuous.best_actor,resumed.best_actor)
    assert continuous.best_score==resumed.best_score
    assert continuous.physical_steps==resumed.physical_steps


def test_changed_validation_schedule_rejects_old_checkpoint(tmp_path):
    model=learner();model.train(tmp_path/'run',max_updates=1)
    with pytest.raises(ValueError,match='protocol'):
        learner(2).train(tmp_path/'run',resume=True)


def test_dense_and_sparse_validation_keep_final_training_parameters_identical(tmp_path):
    dense=StabilizationLearner(PointMassEnv(max_steps=8,initial_scale=.2),
        config=TrainConfig(episodes=12,batch_episodes=2,epochs=1,validation_seeds=(10001,10002)))
    dense.train(tmp_path/'dense')
    sparse=learner();sparse.train(tmp_path/'sparse')
    for name in ('actor','value','optimizer','value_optimizer'):
        equal(getattr(dense,name).state_dict(),getattr(sparse,name).state_dict())
    assert dense.physical_steps==sparse.physical_steps
    # Best-model candidates differ by declared design; final training stays equal
    # for this deterministic-state diagnostic and unchanged sampling schedule.


@pytest.mark.parametrize('interval',[0,-1,True,1.5])
def test_invalid_interval_rejected(interval):
    with pytest.raises(ValueError,match='validation_every_updates'):
        learner(interval)
