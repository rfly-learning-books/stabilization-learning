from copy import deepcopy
import json
import math
from pathlib import Path
import pytest
import torch
from stabilization_learning.rgb_stable.cost_q import (
    transition_cost,bellman_cost,visual_state,PhysicalActionCostQ,FrozenFeatureReplay)


def test_cost_prefers_arrival_and_marks_departure_separately():
    outside=torch.tensor([[.025,0.,0.,0.]])
    inside=torch.tensor([[.015,0.,0.,0.]])
    zero=torch.zeros(1,4); no=torch.zeros(1,dtype=torch.bool)
    stall=transition_cost(outside,outside,zero,no)[0]
    arrival=transition_cost(outside,inside,zero,no)[0]
    departure=transition_cost(inside,outside,zero,no)
    assert arrival.item()<stall.item()
    assert departure[1].item() and departure[2].item()
    assert not transition_cost(outside,inside,zero,no)[2].item()


def test_only_physical_failure_stops_bootstrap():
    cost=torch.tensor([.1,.1,20.1]); failure=torch.tensor([False,False,True])
    # The second sample may be truncated; the identical nonfailure mask must
    # keep its cost-to-go. The third sample is an actual collision/outside.
    target=bellman_cost(cost,failure,torch.full((3,),10.),.99)
    torch.testing.assert_close(target,torch.tensor([10.,10.,20.1]))


def test_bellman_bound_includes_maximum_action_and_failure_cost():
    state=torch.tensor([[1e10,0.,0.,1e10],[1e10,0.,0.,1e10]])
    action=torch.tensor([[1.,0.,0.,1.],[1.,0.,0.,1.]])
    failure=torch.tensor([False,True])
    cost,_,_=transition_cost(state,state,action,failure)
    target=bellman_cost(cost,failure,torch.full((2,),32.))
    assert bool((target<32).all())
    assert cost[1]>20


def test_visual_state_is_observable_and_has_action_history():
    e,g,j,u=torch.randn(3,16),torch.randn(3,16),torch.randn(3,16,4),torch.randn(3,4)
    state=visual_state(e,g,j,u)
    assert state.shape==(3,100)
    torch.testing.assert_close(state[:,-4:],u)


def test_frozen_replay_room_phase_sampling_and_duplicate_protection():
    replay=FrozenFeatureReplay(120,'cpu')
    for room,n in [('a',100),('b',10)]:
        offsets=torch.zeros(n,4); offsets[:,0]=.01 if room=='a' else .04
        replay.add(dict(e=torch.zeros(n,16),offset=offsets),room,room+'.npz')
    torch.manual_seed(41); sampled=replay.sample(2000)
    fraction=(sampled['offset'][:,0]>.02).float().mean().item()
    assert .43<fraction<.57
    with pytest.raises(ValueError,match='twice'):
        replay.add(dict(e=torch.zeros(1,16),offset=torch.zeros(1,4)),'a','a.npz')


def test_gpu_replay_sampling_respects_strict_determinism():
    before=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        replay=FrozenFeatureReplay(16,'cuda:0')
        replay.add(dict(e=torch.arange(16.)[:,None].expand(-1,16),offset=torch.ones(16,4)*.01),'a','a')
        torch.manual_seed(41); first=replay.sample(32)
        torch.manual_seed(41); second=replay.sample(32)
        assert torch.equal(first['e'],second['e'])
        assert first['e'].is_cuda
    finally: torch.use_deterministic_algorithms(before)


@pytest.fixture(scope='module')
def learner():
    from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
    from extensions.rgb_cost_m_v2.learner import build_learner
    root=Path('D:/stabilization-learning')
    adapter,cfg,_=load_runtime(root)
    settings=json.loads((root/'configs/rgb_cost_m_v2/engineering.json').read_text())['jobs'][0]['config']['stability']
    torch.manual_seed(41)
    return build_learner(adapter,deepcopy(cfg),'cuda:0',settings)


def sample(learner,n=8):
    device=learner.device
    offset=torch.randn(n,4,device=device)*.02
    goal=torch.randn(n,16,device=device)*.1
    e=torch.randn(n,16,device=device)*.01
    action=torch.randn(n,4,device=device)*.05
    nxt=offset*.98
    failure=torch.zeros(n,device=device,dtype=torch.bool)
    cost,inside,leaves=transition_cost(offset,nxt,action,failure)
    return dict(z=goal+4*e,goal=goal,j=torch.randn(n,16,4,device=device)*.1,e=e,en=e*.99,
        offset=offset,next_offset=nxt,previous=action*0,action=action,cost=cost,
        inside=inside,leaves=leaves,failure=failure,truncated=torch.ones_like(failure))


def test_q_updates_do_not_change_encoder_or_M_and_freeze_is_enforced(learner):
    learner.freeze_features()
    before={name:deepcopy(getattr(learner,name).state_dict()) for name in ('ae','metric','q')}
    logs=learner.q_update(sample(learner))
    assert logs['truncated_bootstrap_samples']==8
    for name in ('ae','metric'):
        assert all(torch.equal(v,getattr(learner,name).state_dict()[k]) for k,v in before[name].items())
    assert any(not torch.equal(v,learner.q.state_dict()[k]) for k,v in before['q'].items())
    with pytest.raises(RuntimeError,match='frozen visual'):
        learner.representation({})


def test_cost_gradient_reaches_M_through_command_without_updating_Q(learner):
    from lac_vs.models import frozen_parameters
    b=sample(learner)
    m=learner.metric(b['goal'])[1]
    # Same controller used by the real metric objective.
    from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
    adapter,_,_=load_runtime(Path('D:/stabilization-learning'))
    u=adapter.core.policy(b['e'],b['j'],m)[0]
    with frozen_parameters(learner.q):
        values,_=learner.q(learner.state(b),u)
    grads=torch.autograd.grad(values.amax(-1).mean(),list(learner.metric.parameters()))
    assert sum(float(g.square().sum()) for g in grads)>0
    learner.zero_all()
    before=deepcopy(learner.q.state_dict())
    logs=learner.metric_update(b)
    assert all(torch.equal(v,learner.q.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in learner.q.parameters())
    assert all(p.grad is None for p in learner.ae.parameters())
    assert math.isfinite(logs['primary'])


def test_rejected_M_restores_control_only_and_retains_Q_learning(tmp_path,learner):
    from extensions.rgb_cost_m_v2.backend import CostMetricRGBBackend
    backend=CostMetricRGBBackend.__new__(CostMetricRGBBackend)
    backend.learner=learner; backend.output=tmp_path; backend.epoch_number=1
    backend.actual_counts={'q':9,'metric':8}; backend.acceptance=[]
    def result(p0,p1):
        return dict(folder='engineering',summary={'groups':{
            'P0/selected':dict(successes=p0,mean_integrated_error=1.),
            'P1/selected':dict(successes=p1,mean_integrated_error=1.)}},rows=[])
    backend.accepted_validation=result(2,2)
    backend.evaluate=lambda *args:result(1,2)
    backend.save=lambda *args:None
    backend.event=lambda *args,**kwargs:None
    backend.epoch_backup=dict(metric=deepcopy(learner.metric.state_dict()),mopt=deepcopy(learner.mopt.state_dict()),
        metric_count=learner.counts['metric'],metric_accepted=learner.metric_accepted)
    expected=deepcopy(learner.metric.state_dict()); q_before=deepcopy(learner.q.state_dict())
    with torch.no_grad(): next(learner.metric.parameters()).add_(.01)
    learner.q_update(sample(learner))
    outcome=backend.validate()
    assert not outcome['raw_candidate']['accepted']
    assert all(torch.equal(v,learner.metric.state_dict()[k]) for k,v in expected.items())
    assert any(not torch.equal(v,learner.q.state_dict()[k]) for k,v in q_before.items())
