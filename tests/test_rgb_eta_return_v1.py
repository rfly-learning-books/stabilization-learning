"""Metric-action semantics, unchanged control, Bellman targets and gradients."""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import pytest
import torch
from stabilization_gym.tasks_gpu_rgb.eta_control import metric_from_eta, command, value
from stabilization_learning.rgb_eta.learner import EtaReturnLearner, EtaConfig, td_target
from extensions.rgb_eta_return_v1.replay import EtaReplay


def batch(learner, n=20):
    d = learner.config.dim; dev = learner.device
    goal = torch.randn(n,d,device=dev)*.2
    error = torch.randn_like(goal)*.1
    eta = learner.metric(goal).detach()+torch.randn(n,d*(d+1)//2,device=dev)*.08
    interaction = torch.randn(n,d,4,device=dev)*.1
    u,_ = command(error,interaction,metric_from_eta(eta,d)[1])
    q = torch.randn(n,4,device=dev)*.1
    E = (q/q.new_tensor([1,1,1,math.radians(25)])).square().sum(-1)
    return dict(error=error,next_error=error*.9,goal=goal,interaction=interaction,
        eta=eta,eta_valid=torch.ones(n,dtype=torch.bool,device=dev),action=u,
        previous=torch.zeros_like(u),offset=q,energy=E,next_energy=E*.9,
        reward=-E*.1,failure=torch.zeros(n,dtype=torch.bool,device=dev),
        cutoff=torch.zeros(n,dtype=torch.bool,device=dev),feature_version=learner.feature_version)


@pytest.fixture(scope='module')
def runtime():
    from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
    return load_runtime(Path('D:/stabilization-learning'))[0]


def test_exact_original_metric_and_controller(runtime):
    torch.manual_seed(41)
    learner = EtaReturnLearner(EtaConfig(hidden=32),'cpu','test')
    original = runtime.core.GoalMetric()
    original.net.load_state_dict(learner.metric.net.state_dict())
    goal = torch.randn(24,16)
    lo,m = original(goal); lo2,m2 = metric_from_eta(learner.metric(goal))
    assert torch.equal(lo,lo2) and torch.equal(m,m2)
    e,j = torch.randn(24,16),torch.randn(24,16,4)
    expected,h = runtime.core.policy(e,j,m)
    actual,h2 = command(e,j,m2)
    assert torch.equal(h,h2)
    assert torch.equal(expected,actual)
    assert torch.linalg.eigvalsh(m2).min()>=.1-1e-6
    assert torch.allclose(m2.diagonal(dim1=-2,dim2=-1).sum(-1),torch.full((24,),16.),atol=1e-5)


def test_q_rejects_velocity_and_accepts_eta():
    learner = EtaReturnLearner(EtaConfig(hidden=32),'cpu','test'); b = batch(learner)
    with pytest.raises(ValueError,match='eta'):
        learner.q1(b['error'],b['goal'],b['interaction'],b['previous'],b['action'])
    assert learner.q1(b['error'],b['goal'],b['interaction'],b['previous'],b['eta']).shape == (20,)


def test_failure_stops_bootstrap_sampling_cutoff_does_not():
    reward = torch.tensor([1.,1.,1.]); qn = torch.tensor([4.,4.,4.])
    failure = torch.tensor([True,False,False])
    assert torch.equal(td_target(reward,failure,qn,.5),torch.tensor([1.,3.,3.]))


def test_target_uses_next_policy_eta_not_recorded_velocity():
    l = EtaReturnLearner(EtaConfig(hidden=32),'cpu','test'); b = batch(l)
    recorded = []
    handle = l.target1.register_forward_pre_hook(lambda module,args:recorded.append(tuple(v.detach().clone() for v in args)))
    expected = l.target_metric(b['goal']).detach().clone()
    l.critic_update(b); handle.remove()
    assert torch.equal(recorded[0][-1],expected)
    assert torch.equal(recorded[0][3],b['action'])  # previous-command context only
    assert recorded[0][-1].shape[-1] == 136


def test_optimizer_isolation_and_return_gradient_to_eta_actor():
    torch.manual_seed(12)
    l = EtaReturnLearner(EtaConfig(hidden=32),'cpu','test'); b = batch(l)
    actor_before = deepcopy(l.metric.state_dict()); q_before = deepcopy(l.q1.state_dict())
    l.critic_update(b)
    assert all(torch.equal(v,actor_before[k]) for k,v in l.metric.state_dict().items())
    assert any(not torch.equal(v,q_before[k]) for k,v in l.q1.state_dict().items())
    q_before = deepcopy(l.q1.state_dict())
    row = l.metric_update(b,gradient_audit=True)
    assert row['gradient_performance']>0
    assert any(not torch.equal(v,actor_before[k]) for k,v in l.metric.state_dict().items())
    assert all(torch.equal(v,q_before[k]) for k,v in l.q1.state_dict().items())
    assert all(p.grad is None for p in l.q1.parameters())


def test_supervised_return_fit_depends_on_eta():
    torch.manual_seed(3); torch.set_num_threads(2)
    l = EtaReturnLearner(EtaConfig(hidden=64,q_lr=.003),'cpu','test'); b = batch(l,128)
    # An absorbing one-step task: its cumulative return is exactly r(eta).
    b['failure'].fill_(True); b['reward'] = b['eta'][:,1]*4
    with torch.no_grad():
        before = ((l.q1(b['error'],b['goal'],b['interaction'],b['previous'],b['eta'])-b['reward'])**2).mean()
    for _ in range(180): l.critic_update(b)
    with torch.no_grad():
        after = ((l.q1(b['error'],b['goal'],b['interaction'],b['previous'],b['eta'])-b['reward'])**2).mean()
    assert after<.03*before


def test_actual_successor_value_same_metric_full_quadratic():
    torch.manual_seed(5)
    eta = torch.randn(9,136,dtype=torch.float64); _,m = metric_from_eta(eta)
    e,en = torch.randn(9,16,dtype=torch.float64),torch.randn(9,16,dtype=torch.float64)
    delta = value(en,m)-value(e,m)
    independent = .5*(np.einsum('bi,bij,bj->b',en.numpy(),m.numpy(),en.numpy())-
                     np.einsum('bi,bij,bj->b',e.numpy(),m.numpy(),e.numpy()))
    assert np.allclose(delta.numpy(),independent,rtol=1e-12,atol=1e-12)


def test_replay_refuses_coordinate_change_and_invalid_eta():
    l = EtaReturnLearner(EtaConfig(hidden=32),'cpu','version1'); b = batch(l,8)
    r = EtaReplay(10,'cpu','version1'); data = {k:v for k,v in b.items() if torch.is_tensor(v)}
    with pytest.raises(ValueError): r.append(data,'version2')
    r.append(data,'version1'); r.append(data,'version1')
    assert r.size==10 and r.total==16
    sampled = r.batch(12); l.critic_update(sampled)
    sampled['feature_version']='version2'
    with pytest.raises(ValueError): l.critic_update(sampled)
    data['eta_valid'][0]=False
    with pytest.raises(ValueError): r.append(data,'version1')


def test_report_process_check_rejects_unknown_stale_and_writer(tmp_path):
    import json
    import time
    from extensions.rgb_eta_return_v1.report import writer_check
    path=tmp_path/'snapshot.json'
    s=dict(schema='unified_training_process_snapshot',query_success=True,created_unix=time.time(),processes=[])
    path.write_text(json.dumps(s)); writer_check(path,tmp_path/'run')
    s['processes']=[dict(ProcessId=1,ParentProcessId=0,CommandLine=None)]
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError): writer_check(path,tmp_path/'run')
    s['processes'][0]['CommandLine']='python run_rgb_eta_return_v1.py train --output '+str(tmp_path/'run')
    path.write_text(json.dumps(s))
    with pytest.raises(RuntimeError): writer_check(path,tmp_path/'run')
    s['processes']=[]; s['created_unix']-=61
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError): writer_check(path,tmp_path/'run')
