"""Scientific isolation and selection boundaries, rather than runtime smoke only."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT,ROOT/"stabilization-gym",ROOT/"stabilization-learning-control"):
    sys.path.insert(0,str(folder))
from extensions.book_multiseed_v4.pipeline import make_jobs,commit_all,require_commitment,write,read,sha


def test_all_seeds_have_distinct_training_but_identical_paired_test_streams():
    settings=read(ROOT/"configs/book_multiseed_v4.json")
    jobs=make_jobs(settings,{"example.py":"digest"})
    assert len(jobs)==20
    assert len({j["protocol"]["training"]["train_seed_start"] for j in jobs})==20
    assert len({j["protocol"]["training"]["train_probe_seed"] for j in jobs})==20
    for task in settings["tasks"]:
        candidates=[j["protocol"] for j in jobs if j["task"]==task]
        assert len({tuple(p["training"]["test_seeds"]) for p in candidates})==1
        assert len({tuple(p["training"]["validation_seeds"]) for p in candidates})==1
        for p in candidates:
            c=p["training"]
            assert not set(c["test_seeds"])&set(c["validation_seeds"])
            assert c["patience"]>c["updates"]//c["validation_every"]
            assert len({c["train_probe_seed"],c["validation_probe_seed"],p["test_probe_seed"]})==3


def fixture_selection(tmp_path):
    protocol={"jobs":[dict(id="a",path="jobs/a",task="point_mass"),dict(id="b",path="jobs/b",task="point_mass")]}
    write(tmp_path/"protocol.json",protocol)
    choices={}
    for job in protocol["jobs"]:
        dest=tmp_path/job["path"]
        (dest/"point_mass").mkdir(parents=True)
        (dest/"point_mass/last.pt").write_bytes(job["id"].encode())
        choices[job["id"]]={"checkpoint_sha256":sha(dest/"point_mass/last.pt"),"best_update":16}
        write(dest/"SELECTION_COMMITTED.json",{"tasks":{"point_mass":choices[job["id"]]}})
    return protocol,choices


def test_test_access_requires_all_seed_commitment_and_immutable_choices(tmp_path):
    protocol,choices=fixture_selection(tmp_path)
    with pytest.raises(FileNotFoundError):
        require_commitment(tmp_path,protocol)
    with pytest.raises(ValueError,match="Every"):
        commit_all(tmp_path,protocol,{"a":choices["a"]})
    commit_all(tmp_path,protocol,choices)
    require_commitment(tmp_path,protocol)
    changed=deepcopy(choices);changed["a"]["best_update"]=32
    with pytest.raises(ValueError,match="changed"):
        commit_all(tmp_path,protocol,changed)
    (tmp_path/"jobs/b/point_mass/last.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="checkpoint changed"):
        require_commitment(tmp_path,protocol)


def test_duplicate_or_overlapping_training_streams_rejected():
    settings=read(ROOT/"configs/book_multiseed_v4.json")
    settings["seeds"]=[41,41]
    with pytest.raises(ValueError,match="Unique"):
        make_jobs(settings,{})
    settings["seeds"]=[41,42]
    settings["training"]["updates"]=10000
    with pytest.raises(ValueError,match="overlap"):
        make_jobs(settings,{})


def test_smoke_never_exposes_formal_holdouts_or_probe_streams():
    settings=read(ROOT/"configs/book_multiseed_v4.json")
    formal=make_jobs(settings,{})
    settings.update(development_only=True,stream_offset=500000000)
    smoke=make_jobs(settings,{})
    for a,b in zip(formal,smoke):
        x,y=a["protocol"],b["protocol"]
        for key in ("validation_seeds","test_seeds"):
            assert not set(x["training"][key])&set(y["training"][key])
        for key in ("train_seed_start","train_probe_seed","validation_probe_seed"):
            assert x["training"][key]!=y["training"][key]
        assert x["test_probe_seed"]!=y["test_probe_seed"]
