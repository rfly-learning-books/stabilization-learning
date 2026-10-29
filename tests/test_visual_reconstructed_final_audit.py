"""CPU checks against committed telemetry, with corruptions only in tiny fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('independent_visual_final',ROOT/'scripts/audit_visual_reconstructed_final.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)
RUN=ROOT/'runs/visual_reconstructed_seed41_20261023_v1'
CFG=audit.read(RUN/'protocol.json')['config']
SCENES={s['id']:s for s in audit.read(RUN/'scenes.json')['scenes']}
STARTS=audit.read(RUN/'evaluation_starts.json')['test']


def identity(row):
    return ('fixed_test',row['method'],row['plant'],row['index'],'test',1.)


def clone_fixture(tmp_path,source):
    row=audit.read(source)
    start=STARTS[row['index']]
    target=tmp_path/'evaluation/fixed_test'/source.name
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(row),encoding='utf-8')
    shutil.copyfile(source.with_suffix('.npz'),target.with_suffix('.npz'))
    reference=f'references/{row["scene_id"]}_{row["goal_id"]:02d}.npz'
    (tmp_path/'references').mkdir()
    shutil.copyfile(RUN/reference,tmp_path/reference)
    return target,row,start


def load_arrays(path):
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:return {k:saved[k] for k in saved.files}


def evaluate(path,row,start):
    return audit.episode_audit(path,SCENES[row['scene_id']],start,identity(row),CFG)


def test_stratified_committed_real_trajectories_are_consistent(tmp_path):
    picked={}
    for path in sorted((RUN/'evaluation/fixed_test').glob('*.json')):
        if path.name=='summary.json':continue
        row=audit.read(path);key=(row['method'],row['plant'],row['status'])
        picked.setdefault(key,path)
    assert any(k[1]=='P1' for k in picked) and any(k[2]=='success' for k in picked)
    assert any(k[2] in ('collision','outside','timeout') for k in picked)
    results=[]
    for path in picked.values():
        row=audit.read(path);result=evaluate(path,row,STARTS[row['index']])
        assert result['passed']
        results.append({'file':str(path.relative_to(RUN)),'status':row['status'],'plant':row['plant'],
                        'sha256':audit.sha(path),'max_errors':result['max_errors']})
    audit.write(tmp_path/'stratified_positive_evidence.json',{'passed':True,'episodes':len(results),'records':results})


@pytest.mark.parametrize('kind',['action','V','hold','P1_hidden_state','P1_derivative_scope'])
def test_corrupted_telemetry_is_rejected(tmp_path,kind):
    source=RUN/'evaluation/fixed_test/learned_P1_0000.json'
    if kind=='hold':
        source=next(p for p in sorted((RUN/'evaluation/fixed_test').glob('learned_P1_*.json')) if audit.read(p)['status']=='timeout')
    path,row,start=clone_fixture(tmp_path,source)
    arrays=load_arrays(path)
    expected=''
    if kind=='action':
        arrays['action'][0,0]+=.2;arrays['previous_action'][1]=arrays['action'][0];expected='commands'
    elif kind=='V':arrays['V'][0]+=.1;expected='Certificate values'
    elif kind=='hold':
        row['success']=True;row['status']='success';arrays['terminal'][-1]=True;arrays['truncated'][-1]=False
        expected='Success/failure/status'
    elif kind=='P1_hidden_state':
        p0,_=audit.physical_rollout(start['offset'],arrays['action'].astype(float),'P0',arrays['offset'].astype(float))
        arrays['offset'][1]=p0[0];expected='Physical step replay'
    else:row['derivative_critic']={'rmse':0.};expected='P0 derivative diagnostic scope'
    audit.write(path,row);np.savez_compressed(path.with_suffix('.npz'),**arrays)
    with pytest.raises(ValueError,match=expected):evaluate(path,row,start)


def test_reference_change_invalidates_cache_and_is_rejected(tmp_path):
    path,row,start=clone_fixture(tmp_path,RUN/'evaluation/fixed_test/learned_P1_0000.json')
    relative=path.relative_to(tmp_path).as_posix()
    original,_=audit.episode_identity(tmp_path,relative,start)
    reference=tmp_path/'references'/f'{row["scene_id"]}_{row["goal_id"]:02d}.npz'
    with np.load(reference,allow_pickle=False) as saved:values={k:saved[k] for k in saved.files}
    values['depth']=values['depth']+1.
    np.savez_compressed(reference,**values)
    changed,dependencies=audit.episode_identity(tmp_path,relative,start)
    assert changed!=original and reference.relative_to(tmp_path).as_posix() in dependencies
    with pytest.raises(ValueError,match='frozen goal reference'):evaluate(path,row,start)


def test_bounded_development_and_incomplete_audits_never_pass():
    full={'coverage_complete':True,'audited_episodes':3360,'errors':[],'pending':[],'reports':{'passed':True}}
    assert audit.final_passable(full,0)
    assert not audit.final_passable(full,12)
    assert not audit.final_passable(full,4000)
    for key,value in [('coverage_complete',False),('audited_episodes',3359),('errors',['bad action']),('pending',['video missing'])]:
        corrupted=deepcopy(full);corrupted[key]=value
        assert not audit.final_passable(corrupted,0)
