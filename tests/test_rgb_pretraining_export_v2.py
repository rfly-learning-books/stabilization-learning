from pathlib import Path
import importlib.util
import pytest
import torch

SPEC=importlib.util.spec_from_file_location('rgb_export',Path(__file__).resolve().parents[1]/'scripts/export_rgb_pretraining_v2.py')
MODULE=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def payload():
    model={key:{'weight':torch.tensor([float(i)])} for i,key in enumerate(
        ('ae','shadow','critics','q','target_q','qopt','metric','target_metric','mopt'))}
    model.update(anchors={'scale':2.},version=7000,features_frozen=True,
        counts=dict(warmup=4000,representation=3000,critic_prefit=1000,q=2000,metric=0))
    return dict(schema='unified_rgb_legacy_v1',learner=model)


def test_visual_package_includes_coordinates_but_excludes_learned_cost_Q():
    original=payload()
    visual,cost=MODULE.packages(original,dict(feature_version='fixed'))
    assert all(k in visual for k in ('ae','shadow','critics','anchors','version'))
    assert not any(k in visual for k in ('q','qopt','metric'))
    assert all(k in cost for k in ('q','target_q','qopt','metric','target_metric','mopt'))
    assert cost['requires_identical_feature_version']=='fixed'
    visual['ae']['weight'].add_(10)
    assert original['learner']['ae']['weight'].item()==0


def test_rejects_checkpoint_after_M_training():
    original=payload(); original['learner']['counts']['metric']=1
    with pytest.raises(ValueError,match='pretraining-only'):
        MODULE.packages(original,dict(feature_version='fixed'))


def test_rejects_unfrozen_visual_coordinates():
    original=payload(); original['learner']['features_frozen']=False
    with pytest.raises(ValueError,match='frozen visual'):
        MODULE.packages(original,dict(feature_version='fixed'))


def test_export_refuses_active_writer_before_touching_training_files(tmp_path):
    import json,time
    snapshot=tmp_path/'snapshot.json'
    snapshot.write_text(json.dumps(dict(schema='unified_training_process_snapshot',query_success=True,
        created_unix=time.time(),processes=[dict(ProcessId=1,ParentProcessId=0,Name='python.exe',
        CommandLine='python run_rgb_cost_m_v2.py worker --output active_run')])))
    with pytest.raises(RuntimeError,match='Active writer'):
        MODULE.export(tmp_path/'absent_run',tmp_path/'output',snapshot)
    assert not (tmp_path/'output').exists()
