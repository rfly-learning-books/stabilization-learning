"""Current P0 risks: loss, complete M, SDC, provenance and recovery."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

from stabilization_gym import make
from stabilization_gym.envs.classic_control.sdc_metric import SDCMetricTask
from stabilization_gym.interfaces.certificate_v2 import VersionBinding,CertificateEvaluator
from stabilization_gym.interfaces.m_learning import CapabilityError,PortableMFacade
from stabilization_learning.objectives_v2 import descent_terms
from stabilization_learning.policies.state_metric_v2 import StateMetricPolicy
from stabilization_learning.common.contracts_v2 import checkpoint,restore,SealedRecord,identity
from extensions.toolbox_p0_v1.corrected_kernel import CorrectedTensorKernel
from extensions.fast_m_learning_v1.fast_m_learning.batch_rollout import FullHorizonRunner
from m_learning_suite.task_runtime import FullHorizonTask


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def binding(**kwargs):
    values=dict(task="test",model="m1",representation="identity",normalization="SI",
        reference="r1",goal="g1",certificate="full_quadratic_v2",controller="state_M",
        protocol="p1",weights="w1",action_units="N",metric_scope="state_dependent")
    return VersionBinding(**dict(values,**kwargs))


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("dt",[.001,.02,.1])
def test_descent_target_floor_separation(dtype,dt):
    delta=torch.tensor([0.,0.,1e-10,-1e-10,.2,-.2],dtype=dtype)
    norm=torch.tensor([0.,1e-12,1e-12,1e-12,2.,2.],dtype=dtype)
    out=descent_terms(delta,norm,dt=dt)
    assert out["penalty"][0]==0
    expected=(delta+.05*dt*norm).clamp_min(0)/norm.clamp_min(1e-6)
    torch.testing.assert_close(out["penalty"],expected,rtol=0,atol=0)
    assert out["penalty"][1]>0 and out["penalty"][3]==0
    if dtype==torch.float64:
        x=torch.zeros(400,dtype=dtype)
        assert float(descent_terms(x,x,dt=.02)["penalty"].sum())==0.


def test_complete_state_derivative_and_successor_recomputation():
    def metric(e,context):
        values=torch.stack((1+e[...,0].square(),1+2*e[...,1].square()),-1)
        return torch.diag_embed(values)
    oracle=CertificateEvaluator(metric,binding())
    e=torch.tensor([.3,-.4],dtype=torch.float64)
    result=oracle.evaluate(e,derivative=True)
    expected=torch.tensor([2*.3+4*.3**3,2*(-.4)+8*(-.4)**3],dtype=torch.float64)
    torch.testing.assert_close(result.full_error_gradient,expected,rtol=1e-13,atol=1e-13)
    frozen=2*result.M@e
    assert not torch.allclose(frozen,expected,atol=1e-6)
    flow=torch.tensor([.2,.1],dtype=torch.float64)
    h=1e-6
    finite=(oracle.evaluate(e+h*flow).value-oracle.evaluate(e-h*flow).value)/(2*h)
    torch.testing.assert_close(oracle.directional_derivative(e,flow),finite,rtol=1e-8,atol=1e-10)
    successor=e*.8
    correct=oracle.difference(e,successor)
    frozen_difference=successor@result.M@successor-result.value
    assert abs(float(correct-frozen_difference))>1e-5
    zero=oracle.evaluate(torch.zeros(2,dtype=torch.float64),derivative=True)
    assert zero.value==0 and torch.equal(zero.full_error_gradient,torch.zeros(2,dtype=torch.float64))
    assert result.value >= torch.linalg.eigvalsh(result.M).min()*e.square().sum()-1e-14
    assert result.claim=="candidate_only"
    with pytest.raises(ValueError):oracle.evaluate(e,binding=binding(weights="new"))


def test_independent_context_derivative_and_version_cache_rejection():
    live=[binding()]
    oracle=CertificateEvaluator(lambda e,c:torch.diag_embed(1+c.square()),live[0],live_version=lambda:live[0])
    e=torch.tensor([.2,.3],dtype=torch.float64)
    c=torch.tensor([.4,.5],dtype=torch.float64)
    dc=torch.tensor([.1,.2],dtype=torch.float64)
    expected=(2*c*e.square()*dc).sum()
    torch.testing.assert_close(oracle.directional_derivative(e,torch.zeros_like(e),c,dc),expected)
    live[0]=binding(reference="updated")
    with pytest.raises(ValueError):oracle.evaluate(e,c)


@pytest.mark.parametrize("name",["coupled_linear","point_mass","pendulum","cartpole"])
def test_SDC_exact_model_same_M_baseline_gradients_and_state_dependence(name):
    env=make(name);task=SDCMetricTask(env,name)
    e=torch.arange(1,env.error_dim+1,dtype=torch.float64)*.013
    state=e+e.new_tensor(task.goal);u=e.new_zeros(len(task.action_low))
    A,B=task.coefficients(e)
    if name=="point_mass":
        actual=torch.cat((state[2:],u+task.feedforward(state)))
    elif hasattr(env,"_flow"):
        actual=env._flow(state,u+task.feedforward(state))
    else:
        actual=env._A@state+env._B@(u+task.feedforward(state))
    torch.testing.assert_close(A@e+B@u,actual,atol=1e-12,rtol=1e-12)
    policy=StateMetricPolicy(env.metric_spec,task.L0,goal=task.goal,context_scale=.5)
    M=policy(e)
    torch.testing.assert_close(M,e.new_tensor(task.P0),atol=1e-12,rtol=1e-12)
    zero=e*0
    torch.testing.assert_close(task.classical_sdre_metric(zero),e.new_tensor(task.P0),atol=1e-12,rtol=1e-12)
    baseline=task.control(M,e,state=state)
    _,B0=task.coefficients(zero)
    expected=-torch.linalg.solve(e.new_tensor(task.R),B0.T@M@e)+task.feedforward(state)
    if name!="cartpole":
        torch.testing.assert_close(baseline,expected,atol=1e-12,rtol=1e-12)
    else:
        assert (B-B0).abs().max()>1e-7
    matrix=M.detach().requires_grad_()
    action=task.control(matrix,e,state=state)
    gradient=torch.autograd.grad(action.sum(),matrix)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    following=task.step_model(state,baseline)
    assert torch.isfinite(following).all()
    # Nonzero final weights make M depend on state; derivative includes dM.
    with torch.no_grad():policy.net[-1].weight.fill_(.02)
    assert not torch.allclose(policy(e),policy(e*.2),atol=1e-10)
    oracle=CertificateEvaluator(lambda error,context:policy(error),binding())
    res=oracle.evaluate(e,derivative=True)
    assert not torch.allclose(res.full_error_gradient,2*(res.M@e),atol=1e-9)
    before=env.state.copy()
    task.control(policy(e),e,state=state);task.control(policy(e),e,state=state)
    np.testing.assert_array_equal(before,env.state)
    with pytest.raises(CapabilityError):task.require_algorithm("episode_metric_ppo")
    env.close()


def test_sealed_physical_and_high_level_version_contract():
    stamp=binding()
    r=SealedRecord.physical(binding=stamp,observation=[1.,2.],goal=[0.,0.],
        requested_action=[12.],final_command=[10.],actual_execution=None,following_observation=[.9,1.8],
        time=0.,dt=.02,physical_cost=.1,certificate_penalty=0.,delta_v=-.1,
        terminated=False,truncated=False,reason="running",decision_id="episode:1/metric:1")
    assert r.validate(stamp)["kind"]=="physical_transition"
    with pytest.raises(ValueError):r.validate(binding(weights="new_weights_recompute_D"))
    r.payload["physical_cost"]=0.
    with pytest.raises(ValueError):r.validate(stamp)


def test_whole_update_CPU_recovery_modules_optimizer_rng_and_rejection():
    torch.manual_seed(41);np.random.seed(41);random.seed(41)
    actor=torch.nn.Linear(2,2).double();opt=torch.optim.Adam(actor.parameters(),lr=.01)
    x=torch.ones(3,2,dtype=torch.float64)
    def update():
        loss=(actor(x)+torch.randn(3,2,dtype=torch.float64)).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    update()
    saved=checkpoint(modules={"actor":actor},optimizers={"Adam":opt},binding=binding(),
        counters={"updates":1,"physical_steps":3,"M_decisions":3},collector={"boundary":"whole_update","replay":x})
    update();expected=deepcopy(actor.state_dict())
    sequence=(torch.rand(3),np.random.rand(3),random.random())
    other=torch.nn.Linear(2,2).double();otheropt=torch.optim.Adam(other.parameters(),lr=.01)
    counters,collector=restore(saved,modules={"actor":other},optimizers={"Adam":otheropt},binding=binding())
    actor,opt=other,otheropt;update()
    for k,v in actor.state_dict().items():assert torch.equal(v,expected[k])
    assert torch.equal(torch.rand(3),sequence[0]);np.testing.assert_array_equal(np.random.rand(3),sequence[1])
    assert random.random()==sequence[2] and counters["M_decisions"]==3
    with pytest.raises(ValueError):restore(saved,modules={"actor":actor},optimizers={"Adam":opt},binding=binding(action_units="m/s"))
    with pytest.raises(RuntimeError):checkpoint(modules={},optimizers={},binding=binding(),counters={},pending=True)


def runner_for(name,*,dense=False):
    task=FullHorizonTask(make(name),name)
    kernel=CorrectedTensorKernel(task)
    import extensions.fast_m_learning_v1.fast_m_learning.batch_rollout as mod
    manifest=dict(kernel.spec.source_sha256)
    manifest[str(Path(mod.__file__).resolve())]=__import__("hashlib").sha256(Path(mod.__file__).read_bytes()).hexdigest()
    runner=FullHorizonRunner(kernel,source_manifest=manifest,config_sha256="1"*64,protocol_sha256="2"*64,dense_audit=dense)
    return task,kernel,runner


def start_runner(task,kernel,runner,states):
    from stabilization_learning.policies.metric import CholeskyMetric
    n=len(states)
    raw=CholeskyMetric(task.metric_spec).identity_raw().expand(n,-1).clone()
    M=CholeskyMetric(task.metric_spec)(raw.double())
    return runner.start(states,M,episode_ids=[str(i) for i in range(n)],seeds=list(range(n)),
        initial_scales=torch.ones(n,dtype=torch.float64),actor_versions=["frozen"]*n,phi_versions=["identity"]*n,
        raw=raw,raw_logp=torch.zeros(n),context=kernel.context(n),rng_protocol="external",
        rng_state_by_episode={str(i):{"seed":i} for i in range(n)})


@pytest.mark.parametrize("name",["coupled_linear","point_mass","pendulum","cartpole"])
def test_runner_full_horizon_queries_once_commit_recovery_and_failure_tail(name):
    task,kernel,runner=runner_for(name,dense=True)
    obs,_=task.reset(seed=715)
    states=torch.as_tensor(np.stack((obs,obs)),dtype=torch.float64)
    # Force a failure lane, retaining its real first transition and capped tail.
    if name=="cartpole":states[1,0]=3.
    elif name=="pendulum":states[1,0]=2.
    else:states[1,0]=101.
    start_runner(task,kernel,runner,states)
    before=runner.state.state.clone()
    plan=runner.propose();again=runner.propose()
    assert plan.checksum==again.checksum and torch.equal(before,runner.state.state)
    with pytest.raises(RuntimeError):runner.snapshot()
    corrupt=plan.clone();corrupt.transition["applied_action"].add_(1.)
    with pytest.raises(RuntimeError):runner.commit(corrupt)
    runner.commit(plan)
    with pytest.raises(RuntimeError):runner.commit(plan)
    snapshot=runner.snapshot()
    _,_,resumed=runner_for(name,dense=True);resumed.restore(snapshot)
    runner.advance();resumed.advance()
    assert torch.equal(runner.state.state,resumed.state.state)
    while not runner.complete:runner.advance()
    while not resumed.complete:resumed.advance()
    a,b=runner.summary(),resumed.summary()
    for key in ("fixed_horizon_cost","success","failure","physical_steps","certificate_penalty"):
        assert torch.equal(a[key],b[key])
    assert a["physical_steps"][0]==task.max_steps and a["failure"][1]
    assert a["fixed_horizon_cost"][1]>=a["accumulated_cost"][1]
    # Independent scalar execution of the same original model/hold protocol.
    scalar=FullHorizonTask(make(name),name)
    scalar.reset(seed=715,options={"state":obs.copy()})
    M=torch.eye(task.error_dim,dtype=torch.float64);cost=0.
    for k in range(scalar.max_steps):
        e,ctx=scalar.build_error(scalar.state)
        u=scalar.calculate_action_from_M(M,e,context=ctx)
        _,c,terminated,truncated,info=scalar.step(u.detach().numpy());cost+=c
        if terminated or truncated:break
    assert abs(float(a["accumulated_cost"][0])-cost)<=1e-9
    assert bool(a["success"][0])==info["success"]


def test_motor_plan_registration_and_full_successor_certificate():
    from extensions.toolbox_p0_v1.motor_adapter import MotorInspectionAdapter
    from extensions.state_conditioned_m_v1.core.metric import StateDependentMetricActor
    env=MotorInspectionAdapter(target_speed=.5);env.reset(seed=321)
    actor=StateDependentMetricActor(4,env.metric_spec,hidden=16)
    with torch.no_grad():actor.net[-1].weight.fill_(.05)
    original=env.state.copy();plan=env.plan(actor);again=env.plan(actor)
    assert plan["checksum"]==again["checksum"] and env._active_action is None
    np.testing.assert_array_equal(original,env.state)
    action=env.register(plan,actor)
    with pytest.raises(RuntimeError):env.register(plan,actor)
    with pytest.raises(ValueError):env.step(action+1)
    env.step(action)
    with pytest.raises(RuntimeError):env.step(action)
    with pytest.raises(ValueError):env.register(plan,actor)
    row=env._metric_trace[-1]
    assert "M_next" in row and "full_state_certificate_residual" in row
    assert env.control_call_is_pure is False


@pytest.mark.parametrize("name",["coupled_linear","point_mass","pendulum","cartpole"])
def test_real_state_M_optimizer_and_exact_whole_update_resume(name,tmp_path):
    from stabilization_learning.algos.pytorch.state_metric_model.learner import ModelMetricConfig,ModelMetricLearner
    torch.manual_seed(41)
    task=SDCMetricTask(make(name),name)
    cfg=ModelMetricConfig(updates=4,batch_states=8,model_horizon=3,validation_every=2,
        validation_seeds=(210001,210002),test_seeds=(310001,310002))
    def learner():
        policy=StateMetricPolicy(task.metric_spec,task.L0,goal=task.goal,context_scale=.5,hidden=16)
        return ModelMetricLearner(task,policy,cfg,source_digest="test_version")
    a=learner();before=deepcopy(a.policy.state_dict())
    metrics=a.update()
    assert metrics["model_transitions"]==24 and metrics["metric_decisions"]==24
    assert any(not torch.equal(before[k],v) for k,v in a.policy.state_dict().items())
    a.save(tmp_path/"last.pt");expected_metrics=a.update();expected=deepcopy(a.policy.state_dict())
    b=learner();b.restore(tmp_path/"last.pt");actual_metrics=b.update()
    assert actual_metrics==expected_metrics
    for key,value in b.policy.state_dict().items():assert torch.equal(value,expected[key])
    e=torch.ones(task.error_dim,dtype=torch.float64)*.07
    assert not torch.allclose(b.policy(e),b.policy(e*.1),rtol=0,atol=1e-10)
    changed=replace(cfg,decay=.1)
    bad=ModelMetricLearner(task,b.policy,changed,source_digest="test_version")
    with pytest.raises(ValueError):bad.restore(tmp_path/"last.pt")


def test_motor_new_loss_recomputes_state_M_and_does_not_register_control():
    from extensions.toolbox_p0_v1.motor_adapter import MotorInspectionAdapter,motor_one_step_loss_v2
    from extensions.state_conditioned_m_v1.core.metric import StateDependentMetricActor
    env=MotorInspectionAdapter(target_speed=.5)
    actor=StateDependentMetricActor(4,env.metric_spec,hidden=16)
    with torch.no_grad():actor.net[-1].weight.fill_(.03)
    loss,data=motor_one_step_loss_v2(actor,env,seeds=[71,72,73])
    assert env._active_action is None and not torch.equal(data["M_current"],data["M_next"])
    loss.backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in actor.parameters())


def test_new_full_horizon_hold_clock_keeps_float64_precision():
    from stabilization_learning.algos.pytorch.state_metric_model.learner import rollout_batch
    task=SDCMetricTask(make("coupled_linear"),"coupled_linear")
    policy=StateMetricPolicy(task.metric_spec,task.L0,goal=task.goal,context_scale=.7)
    rows,arrays=rollout_batch(task,policy,task.initial_states([310001,310002]),record=True)
    for lane,row in enumerate(rows):
        physical=np.concatenate((arrays["states"][:1,lane],arrays["following"][:,lane]))
        good=np.sum(physical**2,axis=-1)<=task.success_tolerance**2
        last_bad=np.flatnonzero(~good)
        start=int(last_bad[-1]+1) if len(last_bad) else 0
        expected=(task.max_steps-start)*task.dt
        assert abs(row["hold_seconds"]-expected)<1e-12


@pytest.mark.parametrize("name",["coupled_linear","point_mass"])
def test_tensor_sampler_real_PPO_adaptation_and_batch_rejection(name,tmp_path):
    from extensions.toolbox_p0_v1.tensor_learner import TensorEpisodeLearner
    from extensions.fast_m_learning_v2 import SparseValidationConfig
    task=FullHorizonTask(make(name),name)
    cfg=SparseValidationConfig(episodes=4,batch_episodes=2,epochs=1,validation_seeds=(10001,10002))
    learner=TensorEpisodeLearner(task,config=cfg)
    batch=learner._collect(2)
    assert learner.episodes==2 and learner.physical_steps==2*task.max_steps
    assert torch.isfinite(batch[2]).all()
    with pytest.raises(RuntimeError):learner.save(tmp_path/"invalid_pending.pt")
    altered=batch[1].clone();altered[0,0]+=.1
    with pytest.raises(ValueError):learner._update(batch[0],altered,*batch[2:4])
    old=deepcopy(learner.actor.state_dict());learner._update(*batch[:4])
    assert any(not torch.equal(v,old[k]) for k,v in learner.actor.state_dict().items())
    learner.save(tmp_path/"valid.pt")
    expected=learner._collect(2)
    other=TensorEpisodeLearner(FullHorizonTask(make(name),name),config=cfg)
    other.restore(tmp_path/"valid.pt");actual=other._collect(2)
    for a,b in zip(expected[:4],actual[:4]):torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_final_description_identity_and_collapsed_feature_detection():
    from extensions.toolbox_p0_v1.description_binding import refresh_description_binding,separation_diagnostic
    facade=PortableMFacade(make("point_mass"))
    old=facade.source_signature_sha256
    facade.description=replace(facade.description,control_call_is_pure=False,
        control_call_side_effects=("explicit_test_pending",),supported_algorithms=())
    refresh_description_binding(facade,sources={"original":"test_source"})
    assert facade.source_signature_sha256!=old
    assert facade.source_signature_sha256==identity(facade.final_description_binding)
    zero=torch.zeros(2,dtype=torch.float64)
    diagnostic=separation_diagnostic(zero,torch.eye(2,dtype=torch.float64),torch.tensor(1.))
    assert diagnostic["feature_collapse"] and diagnostic["candidate_only"]
    assert diagnostic["value"]==0


def test_real_contact_final_description_source_binding_without_a_physical_step():
    pytest.importorskip("mujoco")
    from migration.next_envs_v1.contact_single_disk.contact_single_disk_env import ContactSingleDiskEnv
    from extensions.toolbox_p0_v1.description_binding import source_bound_contact_facade
    env=ContactSingleDiskEnv()
    facade=source_bound_contact_facade(env)
    assert facade.source_signature_sha256==identity(facade.final_description_binding)
    assert facade.describe().control_call_is_pure is False
    assert facade.describe().supported_algorithms==()
    with pytest.raises(CapabilityError):facade.require_algorithm("episode_metric_ppo")
    assert env._elapsed_steps==0
    env.close()
