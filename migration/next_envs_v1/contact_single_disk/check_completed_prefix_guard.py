"""Adversarial missing-tail completed-prefix check; no physics/planning."""
import json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[2]
sys.dont_write_bytecode=True
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
for p in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control"): sys.path.insert(0,str(p))
import run_full_fixtures as runner
from contact_single_disk_env import ContactSingleDiskEnv,PROTOCOL
OUTPUT=HERE/"checks/completed_prefix_guard_v1"; OUTPUT.mkdir(exist_ok=False)
folder=OUTPUT/"01_correct_side_separated"; folder.mkdir()
record=runner.read_commit(HERE/"checks/runner_boundary_v2/step_000001.pkl.gz")
digest=runner.commit(folder/"step_000001.pkl.gz",record)
runner.write_json(folder/"COMPLETE.json",{"fixture":PROTOCOL["five_full_fixtures"][0],
    "sources":record["sources"],"last_commit_sha256":digest,"deliberate_corruption_test":True})
calls={"plan":0,"step":0}
def forbidden_plan(*a,**kw): calls["plan"]+=1; raise AssertionError("Forbidden planner call")
def forbidden_step(*a,**kw): calls["step"]+=1; raise AssertionError("Forbidden physics call")
ContactSingleDiskEnv.plan=forbidden_plan; ContactSingleDiskEnv.step=forbidden_step
runner.OUTPUT=OUTPUT
sys.argv=[str(HERE/"run_full_fixtures.py"),"--max-minutes","75"]
rejection=None
try: runner.main()
except RuntimeError as exc: rejection=str(exc)
passed=rejection=="Completed fixture has missing/changed transactions; refusing any re-execution" and calls=={"plan":0,"step":0}
runner.write_json(OUTPUT/"CHECKS.json",{"passed":passed,"expected_rejection":rejection,"calls":calls,
    "new_physics_substeps":0,"new_planner_queries":0,"training_updates":0,"optimizers":0,
    "source_sha256":runner.sha(__file__),"runner_sha256":runner.sha(HERE/"run_full_fixtures.py"),
    "note":"The ERROR file in this test-only folder is the deliberately expected rejection, not a fixture runtime failure."})
if not passed: raise AssertionError(rejection)
print(json.dumps({"passed":passed,"new_physics_steps":0,"new_planning_queries":0}))
