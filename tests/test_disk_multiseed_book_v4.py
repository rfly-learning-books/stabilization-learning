from pathlib import Path
import importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diskv4',ROOT/'scripts/book_checks/run_disk_multiseed_book_v4.py')
disk=importlib.util.module_from_spec(spec);spec.loader.exec_module(disk)


def test_book_equations_fit_and_independent_audit():
    data=disk.training_data(41);model=disk.fit(data)
    validation,checked=disk.validate(41,model['selected_K'])
    trace=disk.trajectories(41,model['selected_K'],count=4,steps=20)
    assert model['selected_K']==1. and model['feature_rank']==2
    assert checked['q_upper_family_99']<1
    assert disk.audit(data,validation,trace,model)['passed']
    bad=data.copy();bad[0,3]+=.001
    assert not disk.audit(bad,validation,trace,model)['passed']
    bad_trace={k:v.copy() for k,v in trace.items()};bad_trace['actions'][2,0]+=.005
    assert not disk.audit(data,validation,bad_trace,model)['passed']


def test_declared_random_streams_are_reproducible_and_separate():
    np.testing.assert_array_equal(disk.training_data(42),disk.training_data(42))
    assert not np.array_equal(disk.training_data(41),disk.training_data(42))
    assert len({tuple(disk.generator(41,k).integers(2**31,size=4)) for k in (1,2,3)})==3


def test_known_deterministic_model_recovered_from_transfer_triples():
    data=disk.training_data(43);x,a,_,_=data.T
    following=x+.1*a
    known=np.column_stack((x,a,following,.5*(following**2-x**2)))
    np.testing.assert_allclose(disk.fit(known)['beta'],[.1,.005],rtol=0,atol=1e-13)
