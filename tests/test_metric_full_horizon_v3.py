from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "stabilization-gym", ROOT / "stabilization-learning-control"):
    sys.path.insert(0, str(folder))
from stabilization_gym import make
from stabilization_gym.envs.classic_control.sdc_full_horizon import FullHorizonSDCTask
from stabilization_learning.policies.state_metric_residual_v3 import AnchoredStateMetricPolicy
from stabilization_learning.algos.pytorch.state_metric_full_horizon import FullHorizonConfig, FullHorizonMetricLearner
from stabilization_learning.algos.pytorch.state_metric_full_horizon.objectives import full_horizon_components, certificate_transition, quadratic, falsify
from stabilization_learning.algos.pytorch.state_metric_full_horizon.evaluation import evaluate, local_check, selection_gate

torch.set_num_threads(1)


def construct(name="point_mass", **kwargs):
    torch.manual_seed(41)
    task = FullHorizonSDCTask(make(name, **kwargs), name)
    policy = AnchoredStateMetricPolicy(task.metric_spec, task.L0, goal=task.goal,
        context_scale=max(task.physical.initial_scale,.1), hidden=8)
    return task, policy


@pytest.mark.parametrize("name", ["coupled_linear","point_mass","pendulum","cartpole"])
def test_anchor_preserves_nominal_local_dynamics_even_after_parameter_changes(name):
    t,p = construct(name)
    original = local_check(t,p)
    with torch.no_grad():
        for v in p.parameters():
            v.add_(torch.randn_like(v)*.8)
    e = torch.zeros(t.error_dim,dtype=torch.float64,requires_grad=True)
    np.testing.assert_allclose(p(e).detach(),t.P0,rtol=0,atol=1e-12)
    derivative = torch.autograd.functional.jacobian(p,e)
    assert derivative.abs().max()==0
    checked = local_check(t,p)
    assert original["passed"] and checked["passed"]
    np.testing.assert_allclose(checked["F"],original["F"],rtol=0,atol=1e-12)
    x=t.initial_states([100,101],scale=2)
    assert (p(t.error(x))[0]-p(t.error(x))[1]).abs().max()>1e-7


def test_segment_checkpoint_matches_full_graph_and_finite_difference():
    t,p = construct()
    with torch.no_grad():
        p.net[-1].bias.add_(torch.linspace(-.07,.06,p.net[-1].bias.numel()))
    x=t.initial_states([99,100])
    def loss(stride):
        c=full_horizon_components(t,p,x,checkpoint_stride=stride,horizon=40)
        return c[:,0].mean()+.01*c[:,3].mean()
    a=loss(0); a.backward()
    ga=[z.grad.clone() for z in p.parameters()]
    p.zero_grad(set_to_none=True)
    b=loss(7); b.backward()
    torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-13)
    for expected,actual in zip(ga,p.parameters()):
        torch.testing.assert_close(expected,actual.grad,rtol=1e-10,atol=1e-11)
    parameter=p.net[-1].bias
    analytic=float(parameter.grad[3]);epsilon=1e-5
    with torch.no_grad():
        parameter[3]+=epsilon; plus=float(loss(0))
        parameter[3]-=2*epsilon; minus=float(loss(0))
        parameter[3]+=epsilon
    assert abs(analytic-(plus-minus)/(2*epsilon))<1e-7
    # Changing events after step 16 affects the gradient; this is not TBPTT.
    p.zero_grad(set_to_none=True)
    full_horizon_components(t,p,x,checkpoint_stride=7,horizon=16)[:,0].mean().backward()
    assert abs(float(parameter.grad[3])-analytic)>1e-5


def test_full_successor_metric_is_used_and_zero_target_has_no_artificial_penalty():
    t,p=construct()
    with torch.no_grad():p.net[-1].bias.add_(.3)
    x=t.initial_states([105],scale=1.7)
    y,_,norm,_,residual,_=certificate_transition(t,p,x)
    e=t.error(x);en=t.error(y)
    full=quadratic(en,p(en))-quadratic(e,p(e))+.05*t.dt*norm
    frozen=quadratic(en,p(e))-quadratic(e,p(e))+.05*t.dt*norm
    torch.testing.assert_close(residual,full)
    assert abs(float(full-frozen))>1e-7
    zero=torch.zeros_like(x)
    assert certificate_transition(t,p,zero)[-1].item()==0


@pytest.mark.parametrize("name",["pendulum","cartpole"])
def test_failed_physical_rollout_has_exact_float64_tail(name):
    t,p=construct(name)
    x=torch.zeros((1,t.error_dim),dtype=torch.float64)
    x[0,0 if name=="pendulum" else 2]=1.8 if name=="pendulum" else .9
    rows,_=evaluate(t,p,x)
    r=rows[0]
    assert r["failure"] and not r["success"]
    expected=r["accumulated_cost"]+(t.max_steps-r["physical_steps"])*t.dt*max(25.,r["terminal_physical_error"])
    assert abs(expected-r["fixed_horizon_cost"])<1e-12
    with torch.no_grad():components=full_horizon_components(t,p,x,checkpoint_stride=0)
    assert abs(float(components[0,0])-expected)<1e-12
    assert components[0,2]>0


def test_selection_rejects_cheaper_unsafe_or_slow_or_high_effort_policy():
    row=dict(success=True,failure=False,fixed_horizon_cost=1.,arrival_time=3.,
             control_energy=2.,max_normalized_decay_rate=-.1)
    baseline={1.:[row]}
    good={1.:[dict(row,fixed_horizon_cost=.8)]}
    assert selection_gate(good,baseline,local={"passed":True},probe_rate=-.1)["qualified"]
    for change in (dict(success=False),dict(max_normalized_decay_rate=.01),dict(arrival_time=3.2),
                   dict(control_energy=2.3),dict(fixed_horizon_cost=1.01)):
        result=selection_gate({1.:[dict(good[1.][0],**change)]},baseline,local={"passed":True},probe_rate=-.1)
        assert not result["qualified"]


def test_empirical_falsification_keeps_or_improves_each_start_and_stays_in_domain():
    t,p=construct()
    with torch.no_grad():p.net[-1].bias.add_(.5)
    x=t.initial_states(range(20,40))
    low=x.new_full((4,),-1.4);high=-low
    old=certificate_transition(t,p,x)[-1].detach()
    hard,rate=falsify(t,p,x,lower=low,upper=high)
    assert ((hard>=low)&(hard<=high)).all()
    assert (rate>=old-1e-12).all()
    assert all(v.grad is None for v in p.parameters())


def test_whole_update_restore_replays_optimizer_rng_dual_pool_and_rejects_mismatch(tmp_path):
    t,p=construct(max_steps=12)
    cfg=FullHorizonConfig(updates=3,batch_states=2,checkpoint_stride=4,probe_batch=4,
        attack_steps=1,validation_every=1,scales=(1.,),validation_seeds=(5100001,5100002),test_seeds=(6100001,))
    a=FullHorizonMetricLearner(t,p,cfg,source_digest="test_source")
    a.initialize();a.update();a.validate();a.save(tmp_path/"last.pt")
    expected=a.update();expected_weights=deepcopy(a.policy.state_dict())
    tt,pp=construct(max_steps=12)
    b=FullHorizonMetricLearner(tt,pp,cfg,source_digest="test_source")
    b.restore(tmp_path/"last.pt");actual=b.update()
    assert expected==actual
    for key,value in expected_weights.items():
        assert torch.equal(value,b.policy.state_dict()[key])
    assert torch.equal(a.replay,b.replay)
    with pytest.raises(ValueError):
        FullHorizonMetricLearner(tt,pp,replace(cfg,decay=.06),source_digest="test_source").restore(tmp_path/"last.pt")
    with pytest.raises(ValueError):
        FullHorizonMetricLearner(tt,pp,cfg,source_digest="other_source").restore(tmp_path/"last.pt")
    b.in_update=True
    with pytest.raises(RuntimeError):b.save(tmp_path/"unsafe.pt")


def test_seed_leakage_and_old_algorithm_are_rejected():
    with pytest.raises(ValueError):FullHorizonConfig(validation_seeds=(6100001,))
    with pytest.raises(ValueError):FullHorizonConfig(validation_seeds=(4100002,))
    t,_=construct()
    with pytest.raises(ValueError):t.require_algorithm("state_metric_model_gradient_v2")


def test_independent_numpy_audit_detects_corrupted_action_and_wrong_initial_seed():
    from extensions.metric_full_horizon_v3.audit import audit_group
    t,p=construct("pendulum")
    with torch.no_grad():p.net[-1].bias.add_(.1)
    seeds=(9100001,9100002)
    rows,trace=evaluate(t,p,t.initial_states(seeds),record=True)
    weights={k:v.detach().numpy() for k,v in p.state_dict().items()}
    kwargs=dict(role="learned",weights=weights,seeds=seeds,scale=1.)
    assert audit_group("pendulum",trace,rows,**kwargs)["passed"]
    damaged={k:v.copy() for k,v in trace.items()};damaged["actions"][3,0,0]+=.01
    assert not audit_group("pendulum",damaged,rows,**kwargs)["passed"]
    kwargs["seeds"]=(9100002,9100001)
    assert not audit_group("pendulum",trace,rows,**kwargs)["passed"]


def test_completion_event_accepts_saved_timestamp_and_publishes_only_complete_json(tmp_path):
    import json
    from extensions.metric_full_horizon_v3.pipeline import emit
    emit(tmp_path,"pipeline_complete",at="fixture-time",status="complete")
    files=list((tmp_path/"events").glob("*.json"))
    assert len(files)==1
    assert json.loads(files[0].read_text())["at"]=="fixture-time"
    assert not list((tmp_path/"events").glob("*.tmp"))


def test_stop_during_uncommitted_gradient_keeps_checkpoint_and_replays_exactly(tmp_path):
    import hashlib
    t,p=construct(max_steps=12)
    cfg=FullHorizonConfig(updates=2,batch_states=2,checkpoint_stride=4,probe_batch=4,
        attack_steps=1,scales=(1.,),validation_seeds=(5100001,),test_seeds=(6100001,))
    learner=FullHorizonMetricLearner(t,p,cfg,source_digest="stop_fixture")
    learner.initialize();path=tmp_path/"last.pt";learner.save(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    calls=0
    def stop():
        nonlocal calls
        calls+=1
        return calls>=3
    learner.should_stop=stop
    with pytest.raises(InterruptedError):learner.update()
    assert learner.in_update and learner.updates==0
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    with pytest.raises(RuntimeError):learner.save(path)
    learner.restore(path);learner.should_stop=None
    expected=learner.update()
    tt,pp=construct(max_steps=12)
    replica=FullHorizonMetricLearner(tt,pp,cfg,source_digest="stop_fixture")
    replica.restore(path)
    assert replica.update()==expected
    for key,value in learner.policy.state_dict().items():
        assert torch.equal(value,replica.policy.state_dict()[key])
