"""Runner persistence checks from saved evidence: no new physics or planning."""
from copy import deepcopy
import json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[2]
sys.dont_write_bytecode=True
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
for p in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control"): sys.path.insert(0,str(p))
from contact_single_disk_env import ContactSingleDiskEnv,PROTOCOL,_digest
from run_full_fixtures import light_snapshot,commit,read_commit,restore_prefix,verify_sources,write_json,sha,resource_status
OUTPUT=HERE/"checks/runner_boundary_v2"
OUTPUT.mkdir(exist_ok=False)
checks=[]
def add(name,passed):
    checks.append({"name":name,"passed":bool(passed)})
    if not passed: raise AssertionError(name)
sources,frozen=verify_sources()
saved=json.loads((HERE/"checks/preflight_v1/BOUNDARY_SNAPSHOT.json").read_text(encoding="utf-8"))
env=ContactSingleDiskEnv(); env.load_state_dict(saved)
light=light_snapshot(env); complete=deepcopy(light)
complete["trace"]=deepcopy(env._trace); complete["plan_log"]=deepcopy(env._plan_log)
add("light_plus_separate_trace_equals_full_state_dict",_digest(complete)==_digest(env.state_dict()))
add("saved_fixture_one_step",env._elapsed_steps==1 and len(env._trace)==1 and len(env._plan_log)==1)
fixture=PROTOCOL["five_full_fixtures"][0]
record={"fixture":fixture,"sources":sources,"step":1,"previous_commit_sha256":None,
        "row":env._trace[0],"plan_log":env._plan_log[0],"boundary":light}
commit_path=OUTPUT/"step_000001.pkl.gz"
digest=commit(commit_path,record)
add("commit_roundtrip_exact",_digest(record)==_digest(read_commit(commit_path)))
try: commit(commit_path,record); blocked=False
except RuntimeError: blocked=True
add("commit_overwrite_rejected",blocked)
clone=ContactSingleDiskEnv(); clone.reset(seed=41,options={k:v for k,v in fixture.items() if k!="name"})
returned=restore_prefix(clone,OUTPUT,fixture,sources)
add("prefix_chain_and_full_state_restore_exact",returned==digest and _digest(clone.state_dict())==_digest(env.state_dict()))
wrong={**sources,"run_full_fixtures.py":"wrong"}
try: restore_prefix(clone,OUTPUT,fixture,wrong); blocked=False
except RuntimeError: blocked=True
add("changed_runner_source_rejected",blocked)
add("shared_frozen_sources_unchanged",verify_sources()[1]==frozen)
resource=resource_status(); add("resources_safe_now",resource["safe"])
write_json(OUTPUT/"CHECKS.json",{"passed":all(c["passed"] for c in checks),"checks":checks,
    "sources":sources,"runner_check_source_sha256":sha(__file__),"resource":resource,
    "new_physics_substeps":0,"planning_queries":0,"full_episodes":0,"optimizers":0,
    "models_constructed":4,"note":"Restored saved preflight boundary; no model stepping."})
print(json.dumps({"passed":True,"checks":len(checks),"new_physics_substeps":0}))
