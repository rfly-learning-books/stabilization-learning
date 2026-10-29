"""Scientific boundary tests for the new independent training chain."""
import sys
from pathlib import Path
import copy
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.scratch_apps_v1.common import paths,seed_all,tensor_hash,read,ROOT
paths()
from extensions.scratch_apps_v1.suite import validate
from extensions.scratch_apps_v1.push import FullMActor,FullMConfig,ModeBank,FullMControlEnv,ppo,coordinates


def test_disjoint_geometry_and_matching_parent_seeds():
    c=read(ROOT/'configs/scratch_applications_v1.json');validate(c)
    c['push']['test_shapes'].append('T')
    with pytest.raises(ValueError):validate(c)


def test_random_actor_gradients_and_whole_episode_update(monkeypatch):
    def blocked(*a,**kw):raise AssertionError('Historical weight read attempted')
    monkeypatch.setattr(torch,'load',blocked)
    cfg=FullMConfig(max_steps=5,train_speed_points=5,eval_speed_points=5)
    seed_all(71);actor=FullMActor(cfg);first=tensor_hash(actor)
    seed_all(72);assert tensor_hash(FullMActor(cfg))!=first
    bank=ModeBank(ROOT/'assets/push_geometric32','cpu')
    ctrl=FullMControlEnv(bank,cfg,['T','L'],2);critic=ppo.Critic(cfg)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=2e-4)
    rollout,stats=ppo.collect(actor,critic,ctrl,987123)
    assert rollout['done'][-1].all() and stats['transitions']>0
    ppo.update_ppo(actor,critic,optimizer,rollout,bank,cfg,epochs=1,minibatch=8)
    assert tensor_hash(actor)!=first
    d=torch.tensor([[.3,-.2,.1]],requires_grad=True)
    cond=actor.encoder(bank.C[:1],bank.valid[:1],cfg.length_scale)
    _,m=actor.conditioned(d,cond);v=actor.value(d,cond)
    grad,=torch.autograd.grad(v.sum(),d)
    eps=1e-3;delta=torch.tensor([[eps,0.,0.]])
    fd=(actor.value(d+delta,cond)-actor.value(d-delta,cond))/(2*eps)
    assert torch.allclose(fd,grad[:,0],atol=2e-4,rtol=2e-3)
    assert torch.linalg.eigvalsh(m).min()>=cfg.eigen_floor
    assert torch.allclose(m.diagonal(dim1=-2,dim2=-1).sum(-1),torch.tensor([3.]),atol=1e-5)
    ctrl.env.close()


def test_time_capability_scaling_matches_physical_similarity():
    seed_all(71);actor=FullMActor()
    c=torch.randn(2,12,3);valid=torch.ones(2,12,dtype=torch.bool)
    # Halving the physical object's size doubles radians per metre of push.
    scaled=c.clone();scaled[:,:,2]*=2
    assert torch.allclose(actor.encoder(c,valid,ell=.1),actor.encoder(scaled,valid,ell=.05))


def test_time_discount_terminal_and_horizon_bootstrap():
    from so101_timeaware.training import advantages
    from so101_timeaware.timing import TimeConfig
    cfg=TimeConfig(gamma=.9,gae_lambda=1.)
    reward=torch.ones(1,2);value=torch.zeros(1,2);tail=torch.ones(1,2)*10
    done=torch.tensor([[True,False]]);dt=torch.tensor([[2.,2.]])
    a,_=advantages(reward,value,tail,done,dt,cfg)
    assert torch.allclose(a,torch.tensor([[1.,9.1]]))
