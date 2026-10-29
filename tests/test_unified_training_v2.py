"""Exercise v2 source barrier and compact Push evaluation snapshots."""
import json
import time
from pathlib import Path
import pytest
import torch
import numpy as np

from extensions.unified_training_v1.backends.push import PushBackend, _module_hash
from extensions.unified_training_v2.push_curves import RecordedPushBackend
from stabilization_learning.unified_v1.runner import sha256, write_new


def test_push_compact_snapshot_keeps_all_three_evaluation_variants(tmp_path):
    cfg = dict(seed=1731, num_envs=2, initial_steps=4, collection_steps=2,
               batch_size=8, validation_repeats=1, test_repeats=1)
    outputs = []
    for cls, name in ((PushBackend, "original"), (RecordedPushBackend, "compact")):
        root = tmp_path/name; folder = root/"jobs"/"push"; folder.mkdir(parents=True)
        backend = cls(cfg, folder, deadline=time.time()+300)
        backend.initialize(); backend.collect(initial=True)
        backend.pretrain_evaluator(2)
        backend.train_epoch(epoch=1, representation_updates=0, evaluator_updates=2, certificate_updates=2)
        saved_hash = _module_hash(backend.actor)
        ckpt = folder/"checkpoints"/"selected.pt"; backend.save(ckpt)
        marker = dict(checkpoint="checkpoints/selected.pt", sha256=sha256(ckpt))
        write_new(folder/"FROZEN.json", marker)
        write_new(root/"GLOBAL_FROZEN.json", dict(schema="unified_frozen_group_v1", jobs={"push": sha256(folder/"FROZEN.json")}))
        replay_id, replay_size = id(backend.replay), len(backend.replay)
        backend.load(ckpt)
        assert _module_hash(backend.actor) == saved_hash
        if cls is RecordedPushBackend:
            assert id(backend.replay) == replay_id and len(backend.replay) == replay_size
            with pytest.raises(RuntimeError, match="evaluation-only"):
                backend._actor_update()
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            assert not payload["resume_training_supported"] and "replay" not in payload
            rows = [json.loads(line) for line in (folder/"push_all_updates.jsonl").read_text(encoding="utf-8").splitlines()]
            assert [r["kind"] for r in rows] == ["critic", "critic", "critic", "actor", "critic", "actor"]
            assert [r["stage"] for r in rows[:2]] == ["pretrain_evaluator"]*2
        result = backend.evaluate("test", ("selected", "initial", "identity"))
        outputs.append(result)
        close = getattr(backend, "close", None)
        if close: close()
    for variant in ("selected", "initial", "identity"):
        a, b = (x["variants"][variant] for x in outputs)
        assert a["success_rate"] == b["success_rate"]
        assert a["mean_combined_cost"] == pytest.approx(b["mean_combined_cost"], abs=1e-8)
        assert a["numpy_V_max_difference"] == pytest.approx(b["numpy_V_max_difference"], abs=1e-8)


@pytest.mark.parametrize("change", ["historical", "scene"])
def test_added_source_change_blocks_group_freeze(tmp_path, monkeypatch, change):
    from extensions.unified_training_v2 import runner
    from stabilization_learning.unified_v1 import runner as v1
    project = tmp_path/"project"
    for relative in ("extensions/unified_training_v2/historical_rgb.py", "extensions/unified_training_v2/push_curves.py",
                     "extensions/unified_training_v2/reporting.py", "scripts/run_unified_training_v2.py"):
        p=project/relative; p.parent.mkdir(parents=True,exist_ok=True); p.write_text("frozen", encoding="utf-8")
    scene=project/"scene_source.py"; scene.write_text("audited scene",encoding="utf-8")
    asset=project/"asset.png"; asset.write_bytes(b"audited asset")
    depth=project/"third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1/extracted_state_dict.pt"
    depth.parent.mkdir(parents=True); depth.write_bytes(b"depth weights")
    source_protocol=project/"assets/original_visual_initialization/protocol.json"; source_protocol.parent.mkdir(parents=True)
    write_new(source_protocol,dict(dependencies=dict(fastdepth={r"runtime\dependencies\fastdepth_mobilenet_l1\extracted_state_dict.pt":sha256(depth)})))
    audit=project/"migration/visual_scene_reconstruction/SCENE_AUDIT.json"; audit.parent.mkdir(parents=True)
    write_new(audit,dict(passed=True,files=[dict(path=str(scene),sha256=sha256(scene))],
                         asset_files=[dict(path=str(asset),sha256=sha256(asset))]))
    local=project/"assets/original_visual_initialization/best.pt"; local.parent.mkdir(parents=True,exist_ok=True)
    local.write_bytes(b"historical")
    old=tmp_path/"original.pt"; old.write_bytes(b"historical")
    config_path=project/"resolved.json"; config_path.write_text("{}",encoding="utf-8")
    variants=["selected", "historical_trained_reference", "historical_trained_identity"]
    job=lambda name: dict(id=name,backend="rgb_legacy",config=dict(include_historical_reference=True),
                          schedule=dict(epochs=1,certificate_updates=1),test_variants=variants)
    config=dict(schema="unified_training_v1",maximum_minutes=1,absolute_deadline_unix=time.time()+300,
                minimum_free_gib=0,jobs=[job("first"),job("second")],
                historical_reference=dict(checkpoint=str(local),original_checkpoint=str(old),
                    checkpoint_sha256=sha256(local),source_paths=[str(local),str(old)]))
    monkeypatch.setattr(v1,"source_inventory",lambda project: {})
    class Fake:
        def __init__(self, output): self.output=output
        def initialize(self): return {}
        def validate(self): return {"selection_score":[1]}
        def save(self,path): path.write_text("snapshot")
        def collect(self,initial=False): return {}
        def warmup_representation(self,updates): return {}
        def pretrain_evaluator(self,updates): return {}
        def train_epoch(self,**kwargs):
            (old if change == "historical" else scene).write_bytes(b"changed")
            return {}
        def evaluate(self,**kwargs): raise AssertionError("must not test")
    with pytest.raises(RuntimeError,match="Frozen scientific source changed"):
        runner.run_suite(config,tmp_path/"run",project=project,config_path=config_path,telemetry=False,
                         factory=lambda name,cfg,output,**kw:Fake(output))
    assert not (tmp_path/"run/GLOBAL_FROZEN.json").exists()
    assert v1.read(tmp_path/"run/FINAL_STATE.json")["status"] == "failed"


def test_scene_inventory_rejects_mismatch_before_training(tmp_path):
    from extensions.unified_training_v2.runner import visual_source_inventory
    scene=tmp_path/"scene.py"; scene.write_text("scene",encoding="utf-8")
    asset=tmp_path/"asset.png"; asset.write_bytes(b"image")
    depth=tmp_path/"third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1/extracted_state_dict.pt"
    depth.parent.mkdir(parents=True); depth.write_bytes(b"depth")
    source_protocol=tmp_path/"assets/original_visual_initialization/protocol.json"; source_protocol.parent.mkdir(parents=True)
    write_new(source_protocol,dict(dependencies=dict(fastdepth={r"runtime\dependencies\fastdepth_mobilenet_l1\extracted_state_dict.pt":sha256(depth)})))
    audit=tmp_path/"migration/visual_scene_reconstruction/SCENE_AUDIT.json"; audit.parent.mkdir(parents=True)
    write_new(audit,dict(passed=True,files=[dict(path=str(scene),sha256=sha256(scene))],
                         asset_files=[dict(path=str(asset),sha256=sha256(asset))]))
    inventory=visual_source_inventory(tmp_path)
    assert all(str(p.resolve()) in inventory for p in (scene,asset,depth,audit))
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError,match="differs from audited hash"):
        visual_source_inventory(tmp_path)


def test_rgb_case_comparison_reports_match_mismatch_and_partial(tmp_path):
    from extensions.unified_training_v2.reporting import compare_rgb_cases
    jobs=[dict(id=name,backend="rgb_legacy",config=dict(validation_cases=1,test_cases=1)) for name in ("first","second")]
    protocol=dict(config=dict(jobs=jobs))
    first=tmp_path/"jobs/first/RGB_CASES.json"; second=tmp_path/"jobs/second/RGB_CASES.json"
    cases=dict(val=[dict(offset=[1,2])],test=[dict(offset=[3,4])])
    write_new(first,cases)
    assert compare_rgb_cases(tmp_path,protocol)["status"] == "not_checkable"
    write_new(second,cases)
    assert compare_rgb_cases(tmp_path,protocol)["status"] == "matched"
    second.write_text(json.dumps(dict(val=cases["val"],test=[dict(offset=[3,5])])),encoding="utf-8")
    result=compare_rgb_cases(tmp_path,protocol)
    assert result["status"] == "mismatch" and not result["splits"]["test"]["equal"]


def test_report_includes_all_push_updates_and_checks_historical_hash(tmp_path, monkeypatch):
    from extensions.unified_training_v2 import reporting
    from stabilization_learning.unified_v1 import reporting as v1
    root=tmp_path/"run"; folder=root/"jobs/rgb"; directory=folder/"RGB_EVAL_001_test"
    directory.mkdir(parents=True)
    checkpoint_hash="a"*64
    write_new(root/"protocol.json",dict(config=dict(historical_reference=dict(checkpoint_sha256=checkpoint_hash))))
    write_new(directory/"FROZEN.json",dict(historical_reference=dict(checkpoint_sha256=checkpoint_hash,ae_sha256="b",metric_sha256="c")))
    rows=[]; groups={}
    for plant in ("P0", "P1"):
        for variant in ("selected", "historical_trained_reference", "historical_trained_identity"):
            path=directory/f"{plant}_{variant}.npz"
            np.savez(path,error=np.array([[1.,0.],[.5,0.]]), M=np.eye(2), V=np.array([.5,.125]))
            rows.append(dict(plant=plant,variant=variant,case_id="1",success=True,elapsed_s=1.,
                             trajectory=str(path),trajectory_sha256=sha256(path),audit=dict(passed=True),
                             V_over_E_median=1.))
            groups[f"{plant}/{variant}"]=dict(episodes=1,successes=1)
    job=dict(backend="rgb_legacy",config=dict(include_historical_reference=True,test_cases=1))
    test=dict(folder=str(directory),summary=dict(groups=groups),rows=rows)
    # Intercept v1 report at the point where the wrapper installs its parsers.
    def report(root_arg,output,snapshot):
        rgb=v1._evaluate(folder,job,test)
        assert rgb["historical_reference_audit"]["passed"]
        assert rgb["postprocess_numeric_audit"]["passed"]
        assert rgb["paired_time"]["P0/historical_trained_identity"]["common_successes"] == 1
        push=root/"jobs/push"; push.mkdir()
        (push/"push_all_updates.jsonl").write_text(json.dumps(dict(kind="actor",stage="train_epoch",epoch=1,
            counts=dict(certificate_updates=0),result=dict(actor_skipped=True,actor_loss=0.)))+"\n",encoding="utf-8")
        curves=v1._curves(push,"push_discrete_sac",[])
        assert curves["all_optimizer_calls"][0]["result"]["actor_skipped"]
        assert "actor_loss" not in curves["training"][0]
        write_new(directory/"replace.json",{})
        return {"passed":True}
    monkeypatch.setattr(v1,"report_suite",report)
    assert reporting.report_suite(root,tmp_path/"out",tmp_path/"process.json")["passed"]
