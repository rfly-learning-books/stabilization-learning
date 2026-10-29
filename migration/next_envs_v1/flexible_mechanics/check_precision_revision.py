"""CPU boundary-only verification of precision-aware environment snapshots."""
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
OUTPUT=HERE/"checks/precision_r2_attempt2"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def default(value):
    if hasattr(value,"tolist"): return value.tolist()
    raise TypeError(type(value).__name__)
def write(name,data):
    (OUTPUT/name).write_text(json.dumps(data,indent=2,default=default,allow_nan=False)+"\n",encoding="utf-8")
frozen=json.loads((PROJECT/"runs/book_m_night_suite_20261023_v1/protocol.json").read_text(encoding="utf-8"))["sources"]
assert {n:sha(PROJECT/n) for n in frozen}==frozen
OUTPUT.mkdir(parents=True,exist_ok=False)
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[name]="1"
os.environ.update(CUDA_VISIBLE_DEVICES="-1",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1")
for path in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control",PROJECT/"third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages"):
    sys.path.insert(0,str(path))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from flexible_mechanics_env import FlexibleMechanicsEnv
torch.set_num_threads(1); torch.set_num_interop_threads(1)
checks=[]; records=[]; physical_steps=0
def add(name,value,detail=None): checks.append({"name":name,"passed":bool(value),"detail":detail})
def rejected(fn):
    try: fn(); return False
    except ValueError: return True
def step(env,action):
    global physical_steps
    physical_steps+=1
    return env.step(action.detach().numpy())
def old_fingerprint(path,method,targets):
    node=next(n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n,ast.FunctionDef) and n.name==method)
    return [ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in targets for t in n.targets)]
with threadpool_limits(limits=1):
    old=HERE/"revisions/r1_float64_engineering/flexible_mechanics_env.py"
    new=HERE/"flexible_mechanics_env.py"
    add("prediction_and_action_expressions_unchanged",old_fingerprint(old,"calculate_action_from_M",{"M","prediction","b","Mb","raw","action"})==old_fingerprint(new,"calculate_action_from_M",{"M","prediction","b","Mb","raw","action"}))
    add("physical_execution_expressions_unchanged",old_fingerprint(old,"step",{"before","following","applied","cost","terminated","truncated"})==old_fingerprint(new,"step",{"before","following","applied","cost","terminated","truncated"}))
    for dtype in (torch.float32,torch.float64):
        env=FlexibleMechanicsEnv(target_angle=.1)
        observation,_=env.reset(seed=41,options={"state":[.2+5*.05/7,.2-2*.05/7,.1,-.1]})
        state=torch.tensor(observation,dtype=dtype,requires_grad=True)
        e,context=env.build_error(state)
        M=torch.eye(4,dtype=dtype,requires_grad=True)
        action=env.calculate_action_from_M(M,e,context=context)
        gradients=torch.autograd.grad(action.sum(),(M,state))
        add(str(dtype)+"_control_graph_preserved",all(torch.isfinite(g).all() and float(g.abs().max())>0. for g in gradients))
        step(env,action)
        saved=env.state_dict()
        other=FlexibleMechanicsEnv(target_angle=.1); other.load_state_dict(saved)
        add(str(dtype)+"_snapshot_self_load",other.state_dict()["schema"]=="dual-inertia-env-boundary-v2-precision")
        add(str(dtype)+"_saved_input_and_dtype",saved["trace"][0]["control_dtype"]==str(dtype) and np.array_equal(saved["trace"][0]["control_error"],e.detach().numpy()))
        actions=[]
        for instance in (env,other):
            ee,cc=instance.build_error(torch.tensor(instance.state,dtype=dtype))
            actions.append(instance.calculate_action_from_M(M.detach(),ee,context=cc))
        left,right=step(env,actions[0]),step(other,actions[1])
        add(str(dtype)+"_next_action_exact",torch.equal(actions[0],actions[1]))
        add(str(dtype)+"_next_state_exact",np.array_equal(left[0],right[0]) and left[1:4]==right[1:4])
        add(str(dtype)+"_full_audit_exact",json.dumps(env.get_episode_audit(),sort_keys=True,default=default)==json.dumps(other.get_episode_audit(),sort_keys=True,default=default))
        add(str(dtype)+"_next_RNG_reset_exact",np.array_equal(env.reset()[0],other.reset()[0]))
        for field in ("control_error","control_dtype","raw_torque"):
            bad=deepcopy(saved)
            if field=="control_error": bad["trace"][0][field][0]+=.001
            elif field=="control_dtype": bad["trace"][0][field]="unrecognized"
            else: bad["trace"][0][field]+=.001
            add(str(dtype)+"_rejects_bad_"+field,rejected(lambda b=bad:FlexibleMechanicsEnv(target_angle=.1).load_state_dict(b)))
        # Corruption larger than coordinate roundoff must still be rejected at execution.
        stale=FlexibleMechanicsEnv(target_angle=.1); stale.reset(seed=41,options={"state":observation})
        bad_e,bad_ctx=stale.build_error(torch.tensor(observation+np.array([.001,0.,0.,0.]),dtype=dtype))
        bad_action=stale.calculate_action_from_M(M.detach(),bad_e,context=bad_ctx)
        add(str(dtype)+"_stale_physical_state_still_rejected",rejected(lambda:stale.step(bad_action.detach().numpy())))
        add(str(dtype)+"_float16_explicitly_rejected",rejected(lambda:stale.calculate_action_from_M(M.detach(),bad_e.half(),context=bad_ctx)))
        records.append({"dtype":str(dtype),"snapshot":saved,"initial_action":action.detach().numpy(),"next_action":actions[0].detach().numpy()})
    legacy=json.loads((HERE/"checks/static_v1/BOUNDARY_SNAPSHOT.json").read_text(encoding="utf-8"))
    converted=FlexibleMechanicsEnv(target_angle=.1); converted.load_state_dict(legacy)
    version2=converted.state_dict(); again=FlexibleMechanicsEnv(target_angle=.1); again.load_state_dict(version2)
    add("legacy_float64_snapshot_preserved_and_upgraded",np.array_equal(legacy["state"],again.state) and len(version2["trace"])==4 and all(row["control_dtype"]=="torch.float64" for row in version2["trace"]))
    add("frozen_51_sources_unchanged",{n:sha(PROJECT/n) for n in frozen}==frozen)
    add("no_new_full_rollouts",physical_steps==6)
    summary={"passed":all(c["passed"] for c in checks),"checks_passed":sum(c["passed"] for c in checks),"checks_total":len(checks),
        "physical_boundary_steps":physical_steps,"full_rollouts":0,"training_updates":0,"optimizers":0,
        "revision_scope":"precision metadata, shared boundary error validation and snapshot replay only; physical/control formulas unchanged",
        "source_sha256":sha(new),"old_source_sha256":sha(old),"runner_sha256":sha(Path(__file__)),
        "scope_limit":"CPU float32 and float64 snapshots verified; GPU snapshot bitwise reproducibility not established"}
    write("CHECKS.json",checks); write("BOUNDARY_RECORDS.json",records); write("SUMMARY.json",summary)
    write("MANIFEST.json",{p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()})
    print(json.dumps(summary))
    if not summary["passed"]: raise SystemExit(1)
