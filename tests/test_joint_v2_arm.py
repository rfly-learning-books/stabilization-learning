"""Frozen actuated bridge contracts; long physics suites are not unit tests."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from stabilization_gym.tasks_v2.push import JointPushSelector, vendor_modules
from stabilization_learning.joint_v2.push import PushTrainConfig, train, freeze_group
from extensions.joint_certificate_v2.arm import (actuated_config,audit_telemetry,
    create_deployment_bundle,paired_summary,validate_frozen_inputs,command_telemetry,evaluate_frozen_arm)


def bank_fixture(tmp_path):
    vendor=vendor_modules()
    from so101_timeaware.timing import TimeConfig
    cfg=TimeConfig(max_seconds=240,max_actions=3,encoder_omega_multiplier=1.)
    angle=np.arange(12)*2*np.pi/12
    record=dict(polygon=[[-.02,-.02],[.02,-.02],[.02,.02],[-.02,.02],[-.02,-.02]],
        points=np.c_[.02*np.cos(angle),.02*np.sin(angle)].tolist(),normals=np.c_[np.cos(angle),np.sin(angle)].tolist(),
        C=np.c_[np.cos(angle),np.sin(angle),np.where(np.arange(12)%2,5.,-5.)].tolist(),valid=[True]*12,
        transfer_seconds=np.full((13,12),.2).tolist(),reachable=np.ones((13,12),bool).tolist())
    path=tmp_path/"time_bank.json";path.write_text(json.dumps(dict(schema="so101_timebank_v1",config=asdict(cfg),
        shapes={name:record for name in ("T","Square","L")})),encoding="utf-8")
    return vendor["Bank"](path)


def test_actuated_selector_uses_same_greedy_candidates(tmp_path,monkeypatch):
    vendor=vendor_modules();bank=bank_fixture(tmp_path);actor=vendor["Actor"]()
    monkeypatch.setattr(torch,"load",lambda *a,**k:(_ for _ in ()).throw(AssertionError("Old actor read")))
    selector=JointPushSelector(bank,"T",actor)
    q=np.array([.01,-.006,.2]);pusher=np.array([-.05,-.055])
    mode,travel,V,_=selector.choose(q,None,True,pusher)
    tau,reachable,reposition=bank.row("T",-1,False)
    data=selector.model.batch([dict(shape=0,q=q,pusher=pusher,mode=-1,contact=False,
        remaining=bank.config.max_seconds,tau=tau,reachable=reachable,reposition=reposition)])
    expected=int(selector.model.candidates(actor,data)["logits"][0].detach().argmax())
    assert mode==expected//len(bank.config.travels)
    assert travel==bank.config.travels[expected%len(bank.config.travels)]
    assert V==pytest.approx(selector.value(q)[0])
    cfg=actuated_config(bank)
    assert cfg.plant=="arm" and cfg.max_time==240 and cfg.max_actions==bank.config.max_actions


def test_complete_successor_audit_recomputes_metric(tmp_path):
    vendor=vendor_modules();bank=bank_fixture(tmp_path);actor=vendor["Actor"]()
    poses=np.array([[.01,-.006,.2],[.008,-.004,.18],[.005,-.003,.1]])
    selector=JointPushSelector(bank,"T",actor);times=np.array([0.,.02,.04])
    error=np.sum((poses[:,:2]/.05)**2,axis=1)+2*np.sin(poses[:,2]/2)**2
    trace=tmp_path/"telemetry.npz"
    np.savez_compressed(trace,time=times,pose=poses,V=selector.value(poses),error=error)
    audit=audit_telemetry(trace,actor,bank,"T")
    assert audit["passed"] and audit["transitions"]==2
    assert audit["max_V_absolute_difference"]<1e-6
    arrays=np.load(tmp_path/audit["certificate_file"])
    assert arrays["M"].shape==(3,3,3) and arrays["delta_V"].shape==(2,)
    assert audit["physical_error_integral"]==pytest.approx(np.sum(.5*(error[:-1]+error[1:])*np.diff(times)))


def test_command_instrumentation_preserves_time_and_failure_cleanup():
    from types import SimpleNamespace
    class Plant:
        def __init__(self):self.data=SimpleNamespace(time=0.);self.phase="PUSH";self.closed=False
        def step(self,command):self.data.time+=.02
        def close(self):self.closed=True
    shapes=SimpleNamespace(c=SimpleNamespace(Plant=Plant))
    with command_telemetry(shapes) as commands:
        instance=shapes.c.Plant();instance.step(np.array([.006,0.]));instance.phase="HOLD";instance.step(np.zeros(2))
    assert shapes.c.Plant is Plant and instance.closed
    assert sum(r["dt"] for r in commands)==pytest.approx(.04)
    assert [r["phase"] for r in commands]==["PUSH","HOLD"]


def test_common_success_pairing_keeps_failures():
    rows=[dict(method="random_initial",shape="T",case_index=i,success=s,sim_time_s=t,heldout=True)
          for i,s,t in ((0,True,20.),(1,True,30.),(2,False,240.))]
    rows += [dict(method="selected",shape="T",case_index=i,success=s,sim_time_s=t,heldout=True)
             for i,s,t in ((0,True,10.),(1,False,240.),(2,True,15.))]
    report=paired_summary(rows)
    assert report["common_success"]==1 and report["paired_seconds_change"]==-10.
    assert report["lost_success"]==1 and report["gained_success"]==1


def test_bundle_is_hash_bound_and_full_group_checked(tmp_path):
    bank=bank_fixture(tmp_path)
    cfg=PushTrainConfig(seed=81,updates=1,num_envs=2,horizon=4,epochs=1,minibatch=8,eval_every=1,
        validation_repeats=1,train_shapes=("T",),validation_shapes=("Square",),test_shapes=("L",))
    run=tmp_path/"run";train(bank.path,run,cfg)
    barrier=tmp_path/"group.json";freeze_group([run],barrier,[81])
    protocol,frozen,_=validate_frozen_inputs(run,barrier)
    bundle=tmp_path/"bundle";manifest=create_deployment_bundle(run,protocol,frozen,bundle,barrier)
    assert manifest["simulation_only"] and not manifest["rgb_localization"]
    assert (bundle/"metric.pt").read_bytes()==(run/"selected.pt").read_bytes()
    with pytest.raises(FileExistsError):create_deployment_bundle(run,protocol,frozen,bundle,barrier)
    data=json.loads(barrier.read_text());data["expected_seeds"]=[81,82];barrier.write_text(json.dumps(data))
    with pytest.raises(ValueError):validate_frozen_inputs(run,barrier)


def test_full_arm_driver_predeclares_cases_and_video(tmp_path,monkeypatch):
    import extensions.joint_certificate_v2.arm as module
    bank=bank_fixture(tmp_path)
    cfg=PushTrainConfig(seed=81,updates=1,num_envs=2,horizon=4,epochs=1,minibatch=8,eval_every=1,
        validation_repeats=1,train_shapes=("T",),validation_shapes=("Square",),test_shapes=("L",))
    run=tmp_path/"run";train(bank.path,run,cfg);barrier=tmp_path/"group.json";freeze_group([run],barrier,[81])
    called=[]
    def fake_episode(actor,bank,shape,initial,case_seed,folder,*,method,video):
        # Driver-only fixture; actual simulator path is separately smoke checked.
        folder.mkdir(parents=True,exist_ok=False)
        called.append((method,shape,list(initial),video))
        return dict(shape=shape,method=method,success=method!="random_initial",reason="success" if method!="random_initial" else "time_limit",
            sim_time_s=10.,integrated_error=.2,mode_changes=1,input_square_push=.001,
            audit=dict(passed=True,transitions=10,max_V_absolute_difference=0.))
    monkeypatch.setattr(module,"run_actuated_episode",fake_episode)
    output=tmp_path/"arm"
    summary=evaluate_frozen_arm(run,barrier,output,repeats=1)
    assert len(called)==12 and [x[:2] for x in called if x[3]]==[("selected","T")]
    assert all(x[2]==called[0][2] for x in called)
    assert summary["groups"]["selected/heldout"]["episodes"]==1
    assert summary["paired"][0]["gained_success"]==3
    assert (Path(summary["deployment"])/"metric.pt").exists()
    assert json.loads((output/"FINAL_STATE.json").read_text())["status"]=="complete"
    events=[json.loads(p.read_text()) for p in sorted((output/"events").glob("*.json"))]
    assert events[-1]["kind"]=="arm_complete"
    with pytest.raises(FileExistsError):evaluate_frozen_arm(run,barrier,output,repeats=1)
