"""Three predeclared full unit-M engineering cases; no learner/optimizer."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; PROJECT = HERE.parents[2]
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[name] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
for p in (PROJECT/"stabilization-gym", PROJECT/"stabilization-learning-control",
          PROJECT/"third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages", HERE):
    sys.path.insert(0, str(p))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def atomic(path, obj, *, new=False):
    if new and path.exists(): raise RuntimeError("Refusing to overwrite committed evidence: "+str(path))
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def resources(out):
    free = shutil.disk_usage(PROJECT).free/(1024**3)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())/(1024**3)
    state = {"utc":now(),"D_free_GiB":free,"output_GiB":size,"GPU_memory":[]}
    try:
        check = subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],
            capture_output=True,text=True,check=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
        for line in check.stdout.strip().splitlines():
            used,total=map(float,line.split(","))
            if not 0 <= used <= total or total <= 0: raise ValueError("Invalid GPU memory counters")
            state["GPU_memory"].append({"used_MiB":used,"total_MiB":total,"fraction":used/total})
        if not state["GPU_memory"]: raise ValueError("No GPU memory counters returned")
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        state["measurement_error"]=str(exc)
    state["accepted"]=bool(free>=50 and size<=35 and "measurement_error" not in state and
                           all(gpu["fraction"]<=.65 for gpu in state["GPU_memory"]))
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-wall-seconds", type=float, default=600.)
    args = ap.parse_args()
    if not 0 < args.max_wall_seconds <= 600: raise ValueError("Engineering safety slice must be in (0,600] seconds")
    out = HERE/"checks/full_v1"
    if args.resume:
        if not out.is_dir(): raise RuntimeError("No full run exists to resume")
    else: out.mkdir(parents=True, exist_ok=False)
    # OS lock is released on crash; evidence lock file is never deleted.
    import msvcrt
    lock = (out/"runner.lock").open("a+b"); lock.seek(0); lock.write(b"0"); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    frozen = json.loads((PROJECT/"runs/book_m_night_suite_20261023_v1/protocol.json").read_text(encoding="utf-8"))["sources"]
    if {p: sha(PROJECT/p) for p in frozen} != frozen: raise RuntimeError("Frozen formal sources changed")
    sources = {p: sha(HERE/p) for p in ("DESIGN_PROTOCOL.json", "swarm_env.py", "run_full_fixtures.py")}
    gate = json.loads((HERE/"checks/short_v1/SUMMARY.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["counts"]["physical_steps"] != 25 or any(gate["sources"][p] != sources[p] for p in ("DESIGN_PROTOCOL.json", "swarm_env.py")):
        raise RuntimeError("Short gate not passed for these sources")
    if args.resume:
        manifest = json.loads((out/"START_MANIFEST.json").read_text(encoding="utf-8"))
        if manifest["sources"] != sources or manifest["frozen"] != frozen: raise RuntimeError("Resume sources differ")
    else:
        manifest = {"utc": now(), "sources": sources, "frozen": frozen, "scope": "three_full_unit_M_engineering_only",
            "fixtures": ["ordinary_identity", "exact_target", "known_non_target_deadlock"],
            "steps_per_fixture": 600, "max_total_physical_steps": 1800, "max_total_QP_calls": 1800,
            "optimizers": 0, "training_updates": 0, "CPU_BLAS_threads": 1,
            "transaction_semantics": "Atomic unique per-step records and pre-step attempt markers. Restart reconstructs complete snapshot from committed prefix then validates without solver/physical calls. An attempt without a committed step is ambiguous and requires diagnosis; it is never silently replayed. No completed fixture reruns.",
            "source_readiness": "short_v1 passed; root authorizes full execution separately"}
        atomic(out/"START_MANIFEST.json", manifest, new=True)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from swarm_env import SwarmAssignedStatic3Env, SwarmMFacade, PROTOCOL, plain, digest
    from stabilization_learning.policies.metric import CholeskyMetric
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    started = time.monotonic(); new_steps = 0; results = []; last_resource_time=-float("inf")
    current = {"state": "starting", "utc": now(), "pid": os.getpid(), "resume": args.resume,
               "sources": sources, "new_physical_steps_this_invocation": 0, "formal_training_updates": 0}
    atomic(out/"STATUS.json", current)
    if (out/"SUMMARY.json").exists():
        previous = json.loads((out/"SUMMARY.json").read_text(encoding="utf-8"))
        if previous["sources"] != sources or previous["state"] != "complete": raise RuntimeError("Existing summary invalid")
        current.update(state="complete",utc=now(),counts=previous["counts"],note="Existing completed run; no physical or solver calls")
        atomic(out/"STATUS.json",current); print(json.dumps(current)); return 0
    try:
      with threadpool_limits(limits=1):
        for fixture in PROTOCOL["engineering_fixtures"][:3]:
            name = fixture["name"]; case_dir = out/name
            case_dir.mkdir(exist_ok=True); steps_dir = case_dir/"steps"; steps_dir.mkdir(exist_ok=True)
            attempts_dir = case_dir/"attempts"; attempts_dir.mkdir(exist_ok=True)
            env = SwarmAssignedStatic3Env(); facade = SwarmMFacade(env)
            obs, _ = env.reset(seed=41, options={k: fixture[k] for k in ("p", "q", "assignment")})
            mapping = CholeskyMetric(env.metric_spec).double(); raw = mapping.identity_raw(dtype=torch.float64); M = mapping(raw)
            initial_path = case_dir/"initial_snapshot.json"
            if initial_path.exists():
                initial = json.loads(initial_path.read_text(encoding="utf-8"))
                if initial["snapshot_sha256"] != env.state_dict()["snapshot_sha256"]: raise RuntimeError("Initial reset differs")
            else: initial = env.state_dict(); atomic(initial_path, initial, new=True)
            cursor = {k: v for k, v in initial.items() if k != "snapshot_sha256"}
            previous_sha = initial["snapshot_sha256"]
            files = sorted(steps_dir.glob("*.json"))
            if len(files) > 600: raise RuntimeError("Physical step budget exceeded")
            attempts = sorted(attempts_dir.glob("*.json"))
            if [p.name for p in attempts] != [p.name for p in files]:
                raise RuntimeError("Uncommitted attempt requires diagnosis; no automatic physical replay")
            for k, path in enumerate(files, 1):
                tr = json.loads(path.read_text(encoding="utf-8")); token = tr.pop("transaction_sha256")
                attempt = json.loads((attempts_dir/path.name).read_text(encoding="utf-8"))
                if attempt["fixture"] != name or attempt["step"] != k or attempt["sources"] != sources or not np.array_equal(attempt["pre_step_state"],tr["trace_row"]["state_before"]): raise RuntimeError("Attempt identity/state mismatch")
                if path.name != f"{k:04d}.json" or digest(tr) != token or tr["previous_transaction_sha256"] != previous_sha or tr["step"] != k or tr["fixture"] != name or tr["sources"] != sources:
                    raise RuntimeError("Broken transaction prefix")
                cursor["trace"].append(tr["trace_row"]); cursor["plans"].append(tr["plan_log"])
                cursor.update(tr["metadata"]); previous_sha = token
            cursor["snapshot_sha256"] = digest(cursor)
            if files: env.load_state_dict(cursor)
            if (case_dir/"RESULT.json").exists():
                result = json.loads((case_dir/"RESULT.json").read_text(encoding="utf-8"))
                if not env._done or result["snapshot_sha256"] != env.state_dict()["snapshot_sha256"] or result["last_transaction_sha256"] != previous_sha or result["final_snapshot_file_sha256"] != sha(case_dir/"final_snapshot.json"):
                    raise RuntimeError("Completed fixture inconsistent with state, transaction chain or immutable artifact")
                results.append(result); continue
            while not env._done:
                elapsed = time.monotonic()-started
                if elapsed >= args.max_wall_seconds or (out/"STOP").exists() or (HERE/"STOP").exists():
                    reason = "safety_time_slice" if elapsed >= args.max_wall_seconds else "user_STOP"
                    atomic(case_dir/"boundary_snapshot.json", env.state_dict())
                    current.update(state="stopped",reason=reason,fixture=name,fixture_steps=env._steps,utc=now(),new_physical_steps_this_invocation=new_steps)
                    atomic(out/"STATUS.json",current); print(json.dumps(current)); return 0
                if time.monotonic()-last_resource_time>=30:
                    resource=resources(out); last_resource_time=time.monotonic()
                    atomic(out/"LAST_RESOURCE_CHECK.json",resource)
                    if not resource["accepted"]:
                        atomic(case_dir/"boundary_snapshot.json",env.state_dict())
                        current.update(state="stopped",reason="resource_gate",resource=resource,fixture=name,fixture_steps=env._steps,
                                       utc=now(),new_physical_steps_this_invocation=new_steps)
                        atomic(out/"STATUS.json",current); print(json.dumps(current)); return 0
                atomic(attempts_dir/f"{env._steps+1:04d}.json", {"fixture":name,"step":env._steps+1,"sources":sources,
                    "utc":now(),"pid":os.getpid(),"pre_step_state":env.state.tolist()},new=True)
                sample = facade.error_context(env._observation()); action = facade.control(M, sample).action
                _, cost, terminated, truncated, info = env.step(action)
                metadata = plain({"state":env.state,"steps":env._steps,"queries":env._queries,
                    "episode_M":env._episode_M,"pending":None,"done":env._done,"failed":env._failed,
                    "hold_intervals":env._hold_intervals,"first_hold":env._first_hold,
                    "last_info":env._last_info,"rng":env.np_random.bit_generator.state})
                tr = plain({"fixture":name,"step":env._steps,"sources":sources,
                    "previous_transaction_sha256":previous_sha,"trace_row":env._trace[-1],"plan_log":env._plans[-1],"metadata":metadata})
                token = digest(tr); tr["transaction_sha256"] = token
                atomic(steps_dir/f"{env._steps:04d}.json",tr,new=True); previous_sha=token; new_steps += 1
                current.update(state="running",fixture=name,fixture_steps=env._steps,utc=now(),new_physical_steps_this_invocation=new_steps,
                               committed_steps_so_far=sum(r["physical_steps"] for r in results)+env._steps)
                atomic(out/"STATUS.json",current)
            final = env.state_dict(); final_path = case_dir/"final_snapshot.json"
            if final_path.exists():
                if json.loads(final_path.read_text(encoding="utf-8")) != final: raise RuntimeError("Existing immutable final snapshot differs")
            else: atomic(final_path,final,new=True)
            trace=env._trace
            result={"fixture":name,"utc":now(),"physical_steps":env._steps,"QP_calls":env._queries,"raw":raw.tolist(),"M":M.tolist(),
                "success":env._last_info["success"],"failure":env._failed,"final_info":plain(env._last_info),
                "integrated_cost":sum(r["cost"]+r["failure_tail"] for r in trace),"first_hold_confirmation":env._first_hold,
                "max_delta_V":max(r["delta_V"] for r in trace),"V_rise_count":sum(r["delta_V"]>1e-12 for r in trace),
                "same_M_bound_violation_count":sum(r["delta_V"]>r["same_M_bound"]+1e-11 for r in trace),
                "minimum_safe_margin_m":min(r["minimum_safe_margin_m"] for r in trace),
                "raw_safe_distance_violation_count":sum(r["raw_safe_distance_violation"] for r in trace),
                "tolerance_safe_distance_violation_count":sum(r["safe_distance_violation"] for r in trace),
                "physical_contact_count":sum(r["physical_contact"] for r in trace),"solver_failure_count":sum(r["plan"]["solver_failure"] for r in trace),
                "initial_V":trace[0]["V_before"],"final_V":trace[-1]["V_after"],"snapshot_sha256":final["snapshot_sha256"],
                "final_snapshot_file_sha256":sha(case_dir/"final_snapshot.json"),"last_transaction_sha256":previous_sha,
                "facade_description":facade.describe().to_dict(),"facade_source_signature":facade.source_signature_sha256}
            atomic(case_dir/"RESULT.json",plain(result),new=True); results.append(result)
        if {p:sha(PROJECT/p) for p in frozen} != frozen or {p:sha(HERE/p) for p in sources} != sources: raise RuntimeError("Sources changed during run")
        summary={"utc":now(),"state":"complete","scope":"three_unit_M_full_engineering_cases_no_learning",
            "all_episodes_complete":all(r["physical_steps"]==600 for r in results),"counts":{"physical_steps":sum(r["physical_steps"] for r in results),
            "QP_calls":sum(r["QP_calls"] for r in results),"complete_episodes":sum(r["physical_steps"]==600 for r in results),"optimizer_instances":0,"training_updates":0},
            "sources":sources,"results":results,"wall_seconds_this_invocation":time.monotonic()-started,"threadpools":threadpool_info(),
            "limitations":["Unit M only, no learned performance or statistical success rate.","Known safe non-target deadlock retained and expected failure of attraction.","Finite precision tolerance is reported separately from exact safety proof.","Independent saved-evidence audit pending."]}
        atomic(out/"SUMMARY.json",plain(summary),new=True)
        current.update(state="complete",utc=now(),new_physical_steps_this_invocation=new_steps,counts=summary["counts"])
        atomic(out/"STATUS.json",current); print(json.dumps({"state":"complete","counts":summary["counts"],"results":[{k:r[k] for k in ("fixture","success","integrated_cost","final_V")} for r in results]}))
    except Exception as exc:
        import traceback
        current.update(state="error",utc=now(),exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc(),new_physical_steps_this_invocation=new_steps)
        atomic(out/"ERROR.json",current); atomic(out/"STATUS.json",current); raise
    return 0


if __name__ == "__main__": raise SystemExit(main())
