"""Proposed five-case engineering runner; explicit root gate before execution."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def atomic(path,value,new=False):
    if new and path.exists(): raise RuntimeError("Refuse evidence overwrite: "+str(path))
    temp=path.with_name(path.name+".tmp")
    with temp.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)
def resources(out):
    result={"utc":now(),"D_free_GiB":shutil.disk_usage(PROJECT).free/2**30,"output_GiB":sum(p.stat().st_size for p in out.rglob("*") if p.is_file())/2**30,"GPU_memory":[]}
    try:
        raw=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
        for line in raw.stdout.strip().splitlines():
            used,total=map(float,line.split(","))
            if not 0<=used<=total or total<=0: raise ValueError("GPU memory counters invalid")
            result["GPU_memory"].append({"used_MiB":used,"total_MiB":total,"fraction":used/total})
        if not result["GPU_memory"]: raise ValueError("No GPU counters")
    except (OSError,ValueError,subprocess.SubprocessError) as exc: result["measurement_error"]=str(exc)
    result["accepted"]=result["D_free_GiB"]>=50 and result["output_GiB"]<=35 and "measurement_error" not in result and all(g["fraction"]<=.65 for g in result["GPU_memory"])
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--execute-reviewed-full",action="store_true")
    parser.add_argument("--resume",action="store_true"); parser.add_argument("--max-wall-seconds",type=float,default=600)
    args=parser.parse_args()
    if not args.execute_reviewed_full: raise SystemExit("Root full-engineering gate required")
    if not 0<args.max_wall_seconds<=600: raise ValueError("Engineering slice at most600s")
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[name]="1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    for p in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control",PROJECT/"third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages",HERE): sys.path.insert(0,str(p))
    gate=json.loads((HERE/"checks/short_v1/SUMMARY.json").read_text(encoding="utf-8"))
    if not gate["passed"]: raise RuntimeError("Short engineering gate failed")
    sources={p:sha(HERE/p) for p in ("DESIGN_PROTOCOL.json","fixed_wing_env.py","run_full_fixtures.py")}
    if any(gate["sources"][p]!=sources[p] for p in ("DESIGN_PROTOCOL.json","fixed_wing_env.py")): raise RuntimeError("Short-tested sources changed")
    frozen=json.loads((PROJECT/"runs/book_m_night_suite_20261023_v1/protocol.json").read_text(encoding="utf-8"))["sources"]
    if {p:sha(PROJECT/p) for p in frozen}!=frozen: raise RuntimeError("Protected51 source mismatch")
    out=HERE/"checks/full_v1"
    if args.resume:
        if not out.exists(): raise RuntimeError("No full run to resume")
    else: out.mkdir(parents=True,exist_ok=False)
    import msvcrt
    lock=(out/"runner.lock").open("a+b"); lock.seek(0); lock.write(b"0"); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits,threadpool_info
    import fixed_wing_env as mod
    from fixed_wing_env import FixedWingLongitudinalEnv,FixedWingMFacade,PROTOCOL,plain,digest
    from stabilization_learning.policies.metric import CholeskyMetric
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    manifest={"sources":sources,"frozen":frozen,"dependencies":mod.source_hashes(),"scope":"five_full_unit_M_nonlinear_engineering_cases_no_training",
        "fixtures":PROTOCOL["five_full_fixtures"],"caps":PROTOCOL["full_plan"],"transaction":"unique pre-attempt marker then atomic committed record; uncommitted attempts are diagnosed, never silently replayed","utc":now()}
    if args.resume:
        old=json.loads((out/"START_MANIFEST.json").read_text(encoding="utf-8"))
        if any(old[k]!=manifest[k] for k in ("sources","frozen","dependencies","fixtures","caps")): raise RuntimeError("Resume manifest mismatch")
    else: atomic(out/"START_MANIFEST.json",manifest,new=True)
    status={"state":"starting","utc":now(),"pid":os.getpid(),"sources":sources}; atomic(out/"STATUS.json",status)
    if (out/"SUMMARY.json").exists():
        prior=json.loads((out/"SUMMARY.json").read_text(encoding="utf-8"))
        if prior["sources"]!=sources: raise RuntimeError("Existing summary source mismatch")
        status.update(state="complete",note="Existing completed run; no new physical/synthesis/RHS calls",counts=prior["counts"])
        atomic(out/"STATUS.json",status); print(json.dumps(status)); return 0
    activity={"Riccati_synthesis_call_attempts":0,"Riccati_backward_stages_max":0,"numpy_RHS_call_attempts":0,"new_control_attempts":0}
    original_riccati,original_rhs=mod.riccati,mod.rhs_numpy
    def counted_riccati(*a,**kw): activity["Riccati_synthesis_call_attempts"]+=1; activity["Riccati_backward_stages_max"]+=100; return original_riccati(*a,**kw)
    def counted_rhs(*a,**kw): activity["numpy_RHS_call_attempts"]+=1; return original_rhs(*a,**kw)
    mod.riccati=counted_riccati; mod.rhs_numpy=counted_rhs
    started=time.monotonic(); last_resource=-float("inf"); results=[]
    def metadata(env):
        return plain({"sources":env._sources,"initial_state":env._initial,"seed":env._seed,"state":env.state,"context":env._context,
            "steps":env._steps,"physical_substeps":env._physical_substeps,"physical_time":env._physical_time,
            "episode_M":env._episode_M,"cache":env._cache,"pending":env._pending,"queries":env._queries,
            "done":env._done,"failed":env._failed,"failure":env._failure,"hold_samples":env._hold_samples,"first_hold":env._first_hold,
            "cost":env._cost,"failure_tail":env._failure_tail,"budget":env._budget,"last_info":env._last_info,"diagnostics":env._diagnostics,
            "rng":env.np_random.bit_generator.state})
    def stop(reason,env,case_dir,extra=None):
        atomic(case_dir/"boundary_snapshot.json",env.state_dict())
        status.update(state="stopped",reason=reason,utc=now(),fixture=case_dir.name,fixture_attempts=env._steps,activity=activity)
        if extra: status.update(extra)
        atomic(out/"STATUS.json",status)
        atomic(out/("invocation_"+str(os.getpid())+".json"),{"utc":now(),"reason":reason,"activity":activity},new=True)
    try:
      with threadpool_limits(limits=1):
        for fixture in PROTOCOL["five_full_fixtures"]:
            name=fixture["name"]; case_dir=out/name; case_dir.mkdir(exist_ok=True)
            for folder in ("attempts","steps"): (case_dir/folder).mkdir(exist_ok=True)
            env=FixedWingLongitudinalEnv(); facade=FixedWingMFacade(env); env.reset(seed=41,options={"delta_state_SI":fixture["delta_state_SI"]})
            mapping=CholeskyMetric(env.metric_spec).double(); raw=mapping.identity_raw(dtype=torch.float64); M=mapping(raw)
            initial=env.state_dict(); initial_path=case_dir/"initial_snapshot.json"
            if initial_path.exists():
                if json.loads(initial_path.read_text(encoding="utf-8"))!=initial: raise RuntimeError("Initial state mismatch")
            else: atomic(initial_path,initial,new=True)
            cursor={k:v for k,v in initial.items() if k!="snapshot_sha256"}; last_sha=initial["snapshot_sha256"]
            files=sorted((case_dir/"steps").glob("*.json")); attempts=sorted((case_dir/"attempts").glob("*.json"))
            if [p.name for p in files]!=[p.name for p in attempts]: raise RuntimeError("Uncommitted attempt; diagnose without replay")
            for index,path in enumerate(files,1):
                tr=json.loads(path.read_text(encoding="utf-8")); token=tr.pop("transaction_sha256")
                attempt=json.loads((case_dir/"attempts"/path.name).read_text(encoding="utf-8"))
                if path.name!=f"{index:04d}.json" or digest(tr)!=token or tr["previous_transaction_sha256"]!=last_sha or tr["sources"]!=sources or tr["fixture"]!=name or attempt["sources"]!=sources or attempt["step"]!=index or not np.array_equal(attempt["state"],tr["trace_row"]["state_before"]): raise RuntimeError("Transaction/attempt chain mismatch")
                cursor["trace"].append(tr["trace_row"]); cursor["plans"].append(tr["plan_log"]); cursor.update(tr["metadata"]); last_sha=token
            cursor["snapshot_sha256"]=digest(cursor)
            if files: env.load_state_dict(cursor)
            result_path=case_dir/"RESULT.json"
            if result_path.exists():
                prior=json.loads(result_path.read_text(encoding="utf-8"))
                if not env._done or prior["snapshot_sha256"]!=env.state_dict()["snapshot_sha256"] or prior["last_transaction_sha256"]!=last_sha or prior["final_snapshot_file_sha256"]!=sha(case_dir/"final_snapshot.json"): raise RuntimeError("Completed result binding mismatch")
                results.append(prior); continue
            while not env._done:
                if (HERE/"STOP").exists() or (out/"STOP").exists(): stop("user_STOP",env,case_dir); return 0
                if time.monotonic()-started>=args.max_wall_seconds: stop("safety_time_slice",env,case_dir); return 0
                if time.monotonic()-last_resource>=30:
                    resource=resources(out); last_resource=time.monotonic(); atomic(out/"LAST_RESOURCE_CHECK.json",resource)
                    if not resource["accepted"]: stop("resource_gate",env,case_dir,{"resource":resource}); return 0
                committed=sum(r["control_attempts"] for r in results)+env._steps
                if committed>=3000: raise RuntimeError("Control attempt cap reached")
                atomic(case_dir/"attempts"/f"{env._steps+1:04d}.json",{"utc":now(),"step":env._steps+1,"state":env.state.tolist(),"sources":sources},new=True)
                e,c=env.build_error(env.state); action=facade.calculate_action_from_M(M,e,context=c); env.step(action); activity["new_control_attempts"]+=1
                tr=plain({"fixture":name,"sources":sources,"step":env._steps,"previous_transaction_sha256":last_sha,
                    "trace_row":env._trace[-1],"plan_log":env._plans[-1],"metadata":metadata(env)})
                token=digest(tr); tr["transaction_sha256"]=token; atomic(case_dir/"steps"/f"{env._steps:04d}.json",tr,new=True); last_sha=token
                status.update(state="running",utc=now(),fixture=name,fixture_attempts=env._steps,actual_time=env._physical_time,activity=activity)
                atomic(out/"STATUS.json",status)
            final=env.state_dict(); final_path=case_dir/"final_snapshot.json"
            if final_path.exists():
                if json.loads(final_path.read_text(encoding="utf-8"))!=final: raise RuntimeError("Final immutable snapshot differs")
            else: atomic(final_path,final,new=True)
            rows=env._trace; vals=np.array([rows[0]["V_before"]]+[r["V_after"] for r in rows]); actions=np.array([r["action"] for r in rows])
            complete_prefix=sum(r["integration"]["duration"]==.05 and not r["failure"] for r in rows)
            comparable_values=vals[:complete_prefix+1]; changes={}
            for horizon in (1,2,5,10,20,40,100):
                delta=comparable_values[horizon:]-comparable_values[:-horizon]
                changes[str(horizon)]={"full_control_steps_per_window":horizon,"window_seconds":horizon*.05,"comparable_windows":len(delta),
                    "excluded_partial_or_failed_attempts":len(rows)-complete_prefix,"rise_count_above_1e_minus_12":int(np.sum(delta>1e-12)),
                    "max_delta":float(delta.max()) if len(delta) else None,"min_delta":float(delta.min()) if len(delta) else None}
            result=plain({"fixture":name,"control_attempts":env._steps,"actual_physical_time":env._physical_time,
                "success":env._last_info["success"],"failure":env._failed,"failure_location":env._failure,"info":env._last_info,
                "fixed_horizon_cost":env._cost+env._failure_tail,"candidate_V_initial":vals[0],"candidate_V_final":vals[-1],"actual_same_M_changes":changes,
                "input_total_variation_components":np.abs(np.diff(actions,axis=0)).sum(axis=0) if len(actions)>1 else np.zeros(2),
                "input_variation_definition":"sum abs(u_j-u_(j-1)) over consecutive executed control attempts; no assumed previous input before first action",
                "clipping_count_components":np.array([r["plan"]["saturation_mask"] for r in rows]).sum(axis=0),
                "budget":env._budget,"M":M,"raw":raw,"cache":env._cache,"snapshot_sha256":final["snapshot_sha256"],
                "final_snapshot_file_sha256":sha(final_path),"last_transaction_sha256":last_sha,
                "description":facade.describe().to_dict(),"source_signature":facade.source_signature_sha256})
            atomic(result_path,result,new=True); results.append(result)
        counts={"control_attempts":sum(r["control_attempts"] for r in results),"RK4_substeps_started":sum(r["budget"]["RK4_substeps_started"] for r in results),
            "RK4_substeps_completed":sum(r["budget"]["RK4_substeps_completed"] for r in results),"RHS_evaluations":sum(r["budget"]["RHS_evaluations"] for r in results),
            "completed_full_30s_cases":sum(r["control_attempts"]==600 and not r["failure"] for r in results),"optimizer_instances":0,"training_updates":0}
        if counts["control_attempts"]>3000 or counts["RK4_substeps_started"]>30000 or counts["RHS_evaluations"]>120000: raise RuntimeError("Declared full physical budget exceeded")
        if {p:sha(PROJECT/p) for p in frozen}!=frozen or {p:sha(HERE/p) for p in sources}!=sources: raise RuntimeError("Source changed")
        atomic(out/("invocation_"+str(os.getpid())+".json"),{"utc":now(),"reason":"complete","activity":activity},new=True)
        activities=[json.loads(p.read_text(encoding="utf-8"))["activity"] for p in out.glob("invocation_*.json")]
        total_activity={key:sum(a[key] for a in activities) for key in activity}
        summary={"utc":now(),"state":"complete","scope":"five_unit_M_engineering_cases_no_learning","sources":sources,"counts":counts,
            "all_invocation_actual_activity":total_activity,"results":results,"threadpools":threadpool_info(),"wall_seconds_this_invocation":time.monotonic()-started,
            "independent_full_review":"pending","certificate_claim":"candidate only; actual same-M rises retained; P0 never substituted for M"}
        atomic(out/"SUMMARY.json",summary,new=True); status.update(state="complete",utc=now(),counts=counts,activity=activity); atomic(out/"STATUS.json",status)
        print(json.dumps({"state":"complete","counts":counts,"results":[{k:r[k] for k in ("fixture","success","failure","fixed_horizon_cost")} for r in results]}))
    except Exception as exc:
        import traceback
        status.update(state="error",utc=now(),activity=activity,error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        atomic(out/"ERROR.json",status); atomic(out/"STATUS.json",status); raise
    return 0


if __name__=="__main__": raise SystemExit(main())
