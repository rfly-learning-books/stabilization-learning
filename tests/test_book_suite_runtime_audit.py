"""Small CPU engineering checks; no production gate is bypassed by the CLI."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("book_suite_review",ROOT/"scripts/train_book_m_suite.py")
suite=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)
torch.set_num_threads(1)


def tiny_config():
    cfg=suite.read(ROOT/"configs/book_m_night_suite.json")
    cfg["seeds"]=[41]
    cfg["training"].update(episodes=4,batch_episodes=2,epochs=1,validation_seeds=[10001])
    cfg["test_seed_count"]=1
    cfg["test_initial_scale_factors"]=[1.0]
    cfg["tasks"]=[{"id":"engineering_point_mass","env":"point_mass",
                   "env_kwargs":{"dt":.1,"max_steps":12,"initial_scale":.1}}]
    return cfg


def learner(cfg=None):
    cfg=tiny_config() if cfg is None else cfg
    model,physical=suite.make_learner(cfg["tasks"][0],cfg,41)
    model._write_report=lambda output:None
    return model,physical


def train_to_end(folder,cfg=None):
    model,physical=learner(cfg)
    folder.mkdir(parents=True)
    suite.atomic_save(model.actor.state_dict(),folder/"initial_policy.pt")
    model.train(folder)
    return model,physical


def test_batch_interrupt_replays_exact_optimizer_rng_and_counts(tmp_path):
    baseline,_=train_to_end(tmp_path/"baseline")
    interrupted,_=learner()
    interrupted.train(tmp_path/"interrupted",max_updates=1)
    committed=suite.sha(tmp_path/"interrupted/last.pt")
    calls=[0]
    def stop_inside_episode():
        calls[0]+=1
        return calls[0]>=6
    suite.guard_physical_boundaries(interrupted,stop_inside_episode)
    with pytest.raises(suite.SafeStop):
        interrupted.train(tmp_path/"interrupted",resume=True,max_updates=1)
    assert suite.sha(tmp_path/"interrupted/last.pt")==committed
    resumed,_=learner()
    resumed.train(tmp_path/"interrupted",resume=True)
    left=torch.load(tmp_path/"baseline/last.pt",weights_only=False)
    right=torch.load(tmp_path/"interrupted/last.pt",weights_only=False)
    assert suite.same_tree(left,right)
    assert right["updates"]==2 and right["episodes"]==4
    assert len(right["optimizer"]["state"])>0 and len(right["value_optimizer"]["state"])>0
    baseline.env.close();interrupted.env.close();resumed.env.close()


def test_last_committed_before_best_recovers_and_final_is_immutable(tmp_path):
    folder=tmp_path/"commit_window"
    model,_=learner()
    folder.mkdir()
    suite.atomic_save(model.actor.state_dict(),folder/"initial_policy.pt")
    model.train(folder,max_updates=1)
    stale=torch.load(folder/"best.pt",weights_only=False)
    model.train(folder,resume=True,max_updates=1)
    # Recreate the crash window: committed final last.pt, obsolete best sidecar.
    stale["selected_update"]=-999
    suite.atomic_save(stale,folder/"best.pt")
    checkpoint,frozen=suite.freeze_training_artifacts(folder)
    assert suite.same_tree(torch.load(folder/"best.pt",weights_only=False),suite.committed_best_payload(checkpoint))
    assert list(folder.glob("best.pre_recovery.*.pt"))
    assert suite.sha(folder/"final_training.pt")==suite.sha(folder/"last.pt")
    hashes=deepcopy(frozen["immutable_sha256"])
    _,again=suite.freeze_training_artifacts(folder)
    assert hashes==again["immutable_sha256"]
    assert all(suite.sha(folder/k)==v for k,v in hashes.items())
    broken=torch.load(folder/"final_policy.pt",weights_only=False)
    broken["updates"]=-1
    suite.atomic_save(broken,folder/"final_policy.pt")
    damaged=suite.sha(folder/"final_policy.pt")
    with pytest.raises(RuntimeError,match="Immutable"):
        suite.freeze_training_artifacts(folder)
    assert suite.sha(folder/"final_policy.pt")==damaged
    model.env.close()


def test_best_initial_is_reported_and_probe_distinguishes_weight_changes(tmp_path):
    model,physical=learner()
    initial=deepcopy(model.actor.state_dict())
    original=model.evaluate
    def constant_validation(*args,**kwargs):
        result=original(*args,**kwargs)
        result["summary"]["objective"]=100.
        return result
    model.evaluate=constant_validation
    folder=tmp_path/"initial_best";folder.mkdir()
    suite.atomic_save(initial,folder/"initial_policy.pt")
    model.train(folder)
    checkpoint,frozen=suite.freeze_training_artifacts(folder)
    assert frozen["best_is_initial"] is True
    assert suite.same_tree(checkpoint["best_actor"],initial)
    effects=suite.probe_policy_effects(model,physical,initial,checkpoint["best_actor"],checkpoint["actor"])
    assert effects["effects"]["learned"]=={"max_M_delta":0.,"max_action_delta":0.}
    assert any(not torch.equal(initial[k],checkpoint["actor"][k]) for k in initial)
    assert effects["effects"]["final"]["max_M_delta"]>0
    assert effects["effects"]["final"]["max_action_delta"]>0
    model.env.close()


def test_episode_recomputes_actions_physics_V_time_and_rejects_cache_damage(tmp_path):
    model,physical=learner()
    before=suite.module_hash(model.actor)
    row=suite.evaluate_one(model,physical,"identity",20001,tmp_path)
    assert row["independent_episode_audit"]["passed"]
    assert row["physical_steps"]==physical.max_steps
    assert suite.module_hash(model.actor)==before
    assert suite.check_cached_episode(model,physical,"identity",20001,tmp_path)==row
    data=dict(np.load(tmp_path/row["trajectory"],allow_pickle=False))
    assert np.array_equal(data["states"][1:],data["following"][:-1])
    original=data["actions"].copy();data["actions"][0,0]+=.01
    with pytest.raises(RuntimeError,match="episode checks failed"):
        suite.recompute_episode(physical,data,model.config.decay,model.config.certificate_weight)
    data["actions"]=original
    suite.write(tmp_path/"identity_20001.json",dict(row,arrival_time=row["arrival_time"]+.2))
    with pytest.raises(RuntimeError,match="recomputation"):
        suite.check_cached_episode(model,physical,"identity",20001,tmp_path)
    model.env.close()


def test_config_and_preparation_source_gate_reject_drift(tmp_path):
    cfg=tiny_config();suite.validate_config(cfg)
    bad=deepcopy(cfg);bad["test_seed_start"]=10001
    with pytest.raises(ValueError,match="overlap"):
        suite.validate_config(bad)
    config=tmp_path/"config.json"
    audit_path=tmp_path/"preparation.json"
    cfg["preparation_audit"]=str(audit_path.relative_to(ROOT))
    suite.write(config,cfg)
    suite.write(audit_path,{"passed":True,"scope":"engineering_only_no_formal_performance_claim",
                            "source_sha256":suite.source_manifest(config)})
    assert suite.preparation_gate(cfg,config)["sha256"]==suite.sha(audit_path)
    changed=deepcopy(cfg);changed["training"]["std"]=.16
    suite.write(config,changed)
    with pytest.raises(RuntimeError,match="hashes"):
        suite.preparation_gate(cfg,config)


def test_resources_and_single_writer(tmp_path,monkeypatch):
    cfg=tiny_config()
    monkeypatch.setattr(suite.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout="750, 1000\n"))
    monkeypatch.setattr(suite.shutil,"disk_usage",lambda p:SimpleNamespace(free=100*1024**3))
    assert suite.resources(tmp_path,cfg)[0]=="shared_GPU_memory_budget"
    monkeypatch.setattr(suite.shutil,"disk_usage",lambda p:SimpleNamespace(free=50*1024**3))
    assert suite.resources(tmp_path,cfg)[0]=="resource_margin"
    with suite.WriterLock(tmp_path/"writer.lock"):
        with pytest.raises(RuntimeError,match="Another suite writer"):
            with suite.WriterLock(tmp_path/"writer.lock"):
                pass


def test_tiny_orchestration_resume_pairing_and_user_STOP(tmp_path,monkeypatch):
    cfg=tiny_config()
    fake_gate=tmp_path/"engineering_visual_gate.json"
    suite.write(fake_gate,{"passed":True,"scope":"test_fixture_not_formal_visual_results"})
    cfg["depends_on"]=str(fake_gate.relative_to(ROOT))
    config=tmp_path/"tiny_config.json";suite.write(config,cfg)
    # In-process fixtures exercise orchestration; no CLI option can skip gates.
    monkeypatch.setattr(suite,"prerequisites",lambda value:None)
    monkeypatch.setattr(suite,"preparation_gate",lambda value,path:{"test_fixture":True})
    monkeypatch.setattr(suite,"resources",lambda *args:(None,{"engineering_fixture":True}))
    monkeypatch.setattr(suite.StabilizationLearner,"_write_report",lambda *args:None)
    out=tmp_path/"tiny_suite"
    args=SimpleNamespace(config=config,output=out,max_minutes=5.,preflight=False)
    suite.run(args)
    folder=out/"engineering_point_mass_seed41"
    complete=suite.read(folder/"complete.json")
    assert complete["independent_final_audit"].startswith("pending")
    assert "actor_parameters_changed" in complete and "M_changed" not in complete
    assert len(complete["evaluations"]["1.0"]["summaries"])==4
    before={k:suite.sha(folder/k) for k in ["last.pt","best.pt","final_training.pt","final_policy.pt"]}
    suite.run(args)
    assert before=={k:suite.sha(folder/k) for k in before}
    suite.verify_completed(folder,cfg["tasks"][0],41)
    stopped=tmp_path/"user_stopped";stopped.mkdir();(stopped/"STOP").write_text("user stop")
    suite.run(SimpleNamespace(config=config,output=stopped,max_minutes=5.,preflight=False))
    assert not (stopped/"protocol.json").exists() and (stopped/"STOP").read_text()=="user stop"
