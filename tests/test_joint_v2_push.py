"""Targeted arithmetic, termination and real optimizer checks for new joint Push."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from stabilization_gym.tasks_v2.push import JointPushCosts, JointPushEnv, vendor_modules
from stabilization_learning.joint_v2.push import PushTrainConfig, advantages, collect, train, freeze_group, test_frozen as frozen_test


def toy_bank(tmp_path, **overrides):
    """Synthetic deterministic fixture; never represented as robot calibration."""
    from so101_timeaware.timing import TimeConfig
    cfg=TimeConfig(max_actions=3,max_seconds=20.,encoder_omega_multiplier=1.,**overrides)
    angles=np.arange(12)*2*np.pi/12
    response=np.c_[np.cos(angles),np.sin(angles),np.where(np.arange(12)%2,5.,-5.)]
    shape=dict(polygon=[[-.02,-.02],[.02,-.02],[.02,.02],[-.02,.02],[-.02,-.02]],
        points=np.c_[.02*np.cos(angles),.02*np.sin(angles)].tolist(),
        normals=np.c_[np.cos(angles),np.sin(angles)].tolist(),C=response.tolist(),
        valid=[True]*12,transfer_seconds=np.full((13,12),.2).tolist(),
        reachable=np.ones((13,12),bool).tolist(),source="synthetic engineering test fixture")
    path=tmp_path/"time_bank.json"
    path.write_text(json.dumps(dict(schema="so101_timebank_v1",config=asdict(cfg),
        shapes={name:shape for name in ("T","Square","L")})),encoding="utf-8")
    return vendor_modules()["Bank"](path)


@pytest.fixture(scope="module",autouse=True)
def vendor_imports():
    vendor_modules()


def test_joint_cost_clock_switch_and_input_square(tmp_path):
    bank=toy_bank(tmp_path);costs=JointPushCosts(time=2,error=3,switch=4,input_square=5)
    env=JointPushEnv(bank,["T"],1,costs);env.reset(shape="T",pose=[.01,.01,.25])
    _,reward,term,trunc,info=env.step(0)
    assert sum(info["stages"].values()) == pytest.approx(info["dt"])
    assert info["input_square_integral"] == pytest.approx(bank.config.push_speed**2*info["stages"]["push"])
    assert reward == pytest.approx(sum(info["reward_breakdown"].values()))
    assert info["reward_breakdown"]["physical_error"] == pytest.approx(-3*info["integrated_error"])
    assert info["reward_breakdown"]["time"] == pytest.approx(-2*info["dt"]/bank.config.time_reference)
    assert info["switches"]==0
    _,_,_,_,following=env.step(len(bank.config.travels))
    assert following["switches"]==1
    assert following["reward_breakdown"]["switching"] == -4


def test_deadline_failure_and_external_truncation(tmp_path):
    bank=toy_bank(tmp_path);env=JointPushEnv(bank,["T"],1)
    env.reset(shape="T",pose=[.01,.01,.25]);env.plant.elapsed=bank.config.max_seconds-.01
    _,reward,term,trunc,info=env.step(0)
    assert term and not trunc and info["reason"]=="task_deadline"
    assert info["dt"]==pytest.approx(.01) and reward <= -env.costs.failure_penalty
    env.reset(shape="T",pose=[.01,.01,.25])
    _,reward,term,trunc,info=env.abort()
    assert term and not trunc and info["dt"]==0 and reward==-env.costs.failure_penalty
    env.reset(shape="T",pose=[.01,.01,.25])
    _,reward,term,trunc,info=env.truncate()
    assert not term and trunc and reward==0 and info["reason"]=="external_sampling_cutoff"


def test_gym_reset_seed_and_split_boundary(tmp_path):
    env=JointPushEnv(toy_bank(tmp_path),["T"],1)
    first,_=env.reset(seed=901,options={"shape":"T"})
    second,_=env.reset(seed=901,options={"shape":"T"})
    assert np.array_equal(first["q"],second["q"])
    with pytest.raises(ValueError):env.reset(options={"shape":"L"})


def test_semimarkov_bootstrap_masks_are_distinct():
    cfg=SimpleNamespace(gamma=.9,discount_reference_seconds=1.,gae_lambda=.95)
    rewards=torch.tensor([[1.,1.],[10.,10.]])
    values=torch.zeros_like(rewards);nxt=torch.tensor([[2.,2.],[0.,0.]])
    terms=torch.tensor([[True,False],[True,True]])
    truncs=torch.tensor([[False,True],[False,False]])
    adv,_=advantages(rewards,values,nxt,terms,truncs,torch.ones_like(rewards)*2,cfg)
    assert adv[0,0]==pytest.approx(1.)
    assert adv[0,1]==pytest.approx(1.+.9**2*2.)
    # The second episode's reward must not leak across the truncation boundary.
    assert adv[1].tolist()==[10.,10.]


def test_collect_uses_pre_reset_truncation_successor(monkeypatch):
    import stabilization_learning.joint_v2.push as module
    monkeypatch.setattr(module,"vendor_modules",lambda:{"reached":lambda spec,q:False})
    class Model:
        device=torch.device("cpu")
        cfg=SimpleNamespace(reward_scale=1.,gamma=1.,discount_reference_seconds=1.,gae_lambda=.95)
        def batch(self,observations):return {"q":torch.tensor(np.asarray([x["q"] for x in observations]),dtype=torch.float32)}
        def candidates(self,actor,data):return {"logits":torch.zeros(len(data["q"]),2),"physical":torch.ones(len(data["q"]),2,dtype=torch.bool)}
    class Critic(torch.nn.Module):
        def forward(self,data,model):return data["q"][:,0]
    class Env:
        def __init__(self):
            self.plant=SimpleNamespace(q=np.array([1.]),spec={},name="toy",elapsed=0.,switches=0)
            self.total_reward=0.;self.total_breakdown={}
        def observation(self):return {"q":self.plant.q.copy()}
        def step(self,index):
            self.plant.q=np.array([2.])
            return self.observation(),0.,False,True,{"dt":1.,"success":False,"reason":"external"}
        def reset(self):self.plant.q=np.array([100.])
    rollout,_=collect(torch.nn.Identity(),Critic(),Model(),[Env()],1)
    assert rollout["next_value"][0,0]==2.
    assert rollout["advantage"][0,0]==1.


def test_random_joint_training_updates_without_checkpoint_reads(tmp_path,monkeypatch):
    bank=toy_bank(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError("Policy checkpoint loading is forbidden for random initialization")
    monkeypatch.setattr(torch,"load",forbidden)
    cfg=PushTrainConfig(updates=2,num_envs=2,horizon=4,epochs=1,minibatch=8,eval_every=1,
        validation_repeats=1,train_shapes=("T",),validation_shapes=("Square",),test_shapes=("L",))
    output=tmp_path/"new_run"
    frozen=train(bank.path,output,cfg)
    assert frozen["initial_actor_hash"] != frozen["final_actor_hash"]
    assert frozen["selected_update"] in (1,2)
    probes=json.loads((output/"M_PROBES.json").read_text())
    assert [p["update"] for p in probes]==[0,1,2]
    assert probes[-1]["distance_from_initial"]>0
    assert not (output/"test").exists()
    assert all((output/f).exists() for f in ("initial.pt","final.pt","selected.pt","FROZEN.json"))
    events=[json.loads(p.read_text()) for p in sorted((output/"events").glob("*.json"))]
    assert [r["data"]["update"] for r in events if r["kind"]=="update_complete"]==[1,2]
    assert events[-1]["kind"]=="selection_frozen"
    with pytest.raises(FileExistsError):train(bank.path,output,cfg)


def test_shape_splits_reject_overlap():
    with pytest.raises(ValueError):PushTrainConfig(train_shapes=("T",),validation_shapes=("T",),test_shapes=("L",))


def test_freeze_barrier_checks_completeness_and_test_comparisons(tmp_path):
    bank=toy_bank(tmp_path)
    cfg=PushTrainConfig(updates=1,num_envs=2,horizon=4,epochs=1,minibatch=8,eval_every=1,
        validation_repeats=1,train_shapes=("T",),validation_shapes=("Square",),test_shapes=("L",))
    run=tmp_path/"frozen_run";train(bank.path,run,cfg)
    barrier=tmp_path/"all_frozen.json"
    with pytest.raises(ValueError):freeze_group([run],barrier,[71,72])
    freeze_group([run],barrier,[71])
    results=frozen_test(run,barrier,repeats=1)
    assert set(results)=={"random_initial","selected","final","identity"}
    starts=[]
    for method in results:
        content=json.loads((run/"test"/(method+".json")).read_text())
        assert content["episodes"][0]["shape"]=="L"
        starts.append(content["episodes"][0]["initial"])
    assert all(start==starts[0] for start in starts)
    with pytest.raises(FileExistsError):frozen_test(run,barrier,repeats=1)
