"""Numerical and learning-contract checks for the fresh certificate backend."""
import numpy as np
import pytest
import torch
from stabilization_gym.tasks_gpu_rgb.certificate_control import candidates, choose, value, CertificateControlConfig
from stabilization_learning.certificate_rgb.learner import RGBCertificateLearner, LearnerConfig, GoalMetric
from extensions.rgb_certificate_replay_v1.replay import Replay


def fixture(device='cpu'):
    torch.manual_seed(41)
    learner=RGBCertificateLearner(LearnerConfig(hidden=32),device)
    n,d=12,16
    error=torch.randn(n,d,device=device)*.1
    b=dict(error=error,next_error=error+torch.randn_like(error)*.01,goal=torch.randn(n,d,device=device)*.1,
           previous=torch.zeros(n,4,device=device),remaining=torch.ones(n,device=device),next_remaining=torch.full((n,),.9,device=device),
           action=torch.randn(n,4,device=device)*.1,reward=torch.ones(n,device=device)*.2,
           terminal=torch.ones(n,device=device,dtype=torch.bool),energy=error.square().sum(-1)*.5)
    return learner,b


def test_spd_and_fixed_scale():
    metric=GoalMetric();m=metric(torch.randn(40,16))
    assert torch.allclose(m,m.transpose(-1,-2),atol=1e-6)
    assert torch.linalg.eigvalsh(m).min()>=.1-1e-6
    assert torch.allclose(m.diagonal(dim1=-2,dim2=-1).sum(-1),torch.full((40,),16.),atol=1e-5)


def test_complete_discrete_delta_and_hard_gate():
    torch.manual_seed(3)
    e=torch.randn(8,4,dtype=torch.float64)
    j=torch.eye(4,dtype=torch.float64).expand(8,-1,-1)
    m=torch.diag(torch.tensor([1.,2.,3.,4.],dtype=torch.float64)).expand(8,-1,-1)
    r=candidates(e,m,j)
    a,info=choose(r)
    successor=e+.1*torch.einsum('bij,bj->bi',j,a)
    exact=value(successor,m)-value(e,m)
    assert torch.allclose(exact,info['model_delta'],atol=1e-12)
    assert info['model_feasible'].all()
    assert (info['model_residual']<=0).all()
    assert (a[:,:3].norm(dim=-1)<=1+1e-7).all()
    assert (a[:,3].abs()<=1+1e-7).all()
    for _ in range(12):
        _,sample=choose(r,True)
        assert sample['model_feasible'].all()


def test_infeasible_recovery_is_never_certified():
    e=torch.ones(3,4);m=torch.eye(4).expand(3,-1,-1);j=torch.zeros(3,4,4)
    r=candidates(e,m,j);_,info=choose(r)
    assert info['recovery'].all()
    assert not info['model_feasible'].any()
    assert torch.allclose(r['probability'].sum(-1),torch.ones(3))


def test_terminal_target_and_optimizer_isolation():
    learner,b=fixture()
    metric={k:v.clone() for k,v in learner.metric.state_dict().items()}
    r=learner.critic_update(b)
    assert r['target_mean']==pytest.approx(.2,abs=1e-6)
    assert all(torch.equal(v,learner.metric.state_dict()[k]) for k,v in metric.items())
    q={k:v.clone() for k,v in learner.q1.state_dict().items()}
    response={k:v.clone() for k,v in learner.response.state_dict().items()}
    r=learner.metric_update(b)
    assert np.isfinite(r['m_loss']) and r['gradient']>0
    assert any(not torch.equal(v,learner.metric.state_dict()[k]) for k,v in metric.items())
    assert all(torch.equal(v,learner.q1.state_dict()[k]) for k,v in q.items())
    assert all(torch.equal(v,learner.response.state_dict()[k]) for k,v in response.items())


def test_performance_gradient_reaches_metric_without_auxiliary_losses():
    torch.manual_seed(4)
    learner,b=fixture()
    learner.config=LearnerConfig(hidden=32,calibration_weight=0.,feasibility_weight=0.)
    result=learner.metric_update(b)
    assert result['gradient']>1e-8


def test_response_learns_real_finite_motion():
    learner,b=fixture()
    torch.manual_seed(5)
    b['action']=torch.randn(12,4)*.3
    b['next_error']=b['error']+.1*torch.cat((b['action'],b['action']*.2,b['action']*-.3,b['action']*.5),-1)
    first=learner.response_update(b)['response_loss']
    for _ in range(350):last=learner.response_update(b)['response_loss']
    assert last<first*.2


def test_zero_error_has_zero_feasible_command():
    e=torch.zeros(4,16);m=torch.eye(16).expand(4,-1,-1);j=torch.randn(4,16,4)
    a,info=choose(candidates(e,m,j))
    assert torch.equal(a,torch.zeros_like(a))
    assert info['model_feasible'].all()


def test_replay_is_bounded_and_features_use_actual_successors():
    learner,b=fixture()
    replay=Replay(8,41,'cpu')
    goals=torch.randint(0,255,(2,3,64,64),dtype=torch.uint8)
    ids=replay.register_goals(['a','b'],goals)
    for _ in range(6):
        x=torch.randint(0,255,(2,3,64,64),dtype=torch.uint8)
        replay.append(dict(image=x,next_image=x.clone(),goal_id=ids,offset=torch.zeros(2,4),energy=torch.ones(2),
            action=torch.zeros(2,4),previous=torch.zeros(2,4),remaining=torch.ones(2),next_remaining=torch.zeros(2),
            reward=torch.zeros(2),terminal=torch.ones(2,dtype=torch.bool),bootstrap=torch.zeros(2,dtype=torch.bool)))
    assert replay.size==8 and replay.total==12
    replay.bind_features(learner.features)
    sample=replay.batch(16)
    assert torch.equal(sample['error'],sample['next_error'])
    assert sample['terminal'].all()
