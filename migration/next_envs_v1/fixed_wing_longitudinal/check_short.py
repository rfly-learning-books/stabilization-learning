"""Proposed finite engineering check. Do not run until root reviews sources."""
from copy import deepcopy
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[2]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--execute-reviewed-short",action="store_true"); args=parser.parse_args()
    if not args.execute_reviewed_short: raise SystemExit("Static proposal only; explicit reviewed-short invocation required")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    for p in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control",PROJECT/"third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages",HERE): sys.path.insert(0,str(p))
    frozen=json.loads((PROJECT/"runs/book_m_night_suite_20261023_v1/protocol.json").read_text(encoding="utf-8"))["sources"]
    if {p:sha(PROJECT/p) for p in frozen}!=frozen: raise RuntimeError("Formal frozen source mismatch")
    resource={"D_free_GiB":shutil.disk_usage(PROJECT).free/2**30,"directory_GiB":sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())/2**30,"GPU_memory":[]}
    gpu=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    for line in gpu.stdout.strip().splitlines():
        used,total=map(float,line.split(","))
        if not 0<=used<=total or total<=0: raise RuntimeError("Invalid GPU counters")
        resource["GPU_memory"].append({"used_MiB":used,"total_MiB":total,"fraction":used/total})
    if resource["D_free_GiB"]<50 or resource["directory_GiB"]>35 or not resource["GPU_memory"] or any(g["fraction"]>.65 for g in resource["GPU_memory"]): raise RuntimeError("Resource gate; no physical execution")
    out=HERE/"checks/short_v1"; out.mkdir(parents=True,exist_ok=False)
    sources={p:sha(HERE/p) for p in ("DESIGN_PROTOCOL.json","fixed_wing_env.py","check_short.py")}
    (out/"START_MANIFEST.json").write_text(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),"sources":sources,"frozen":frozen,
        "max_environment_control_attempts":21,"max_total_integrated_control_spans":23,"complete_episodes":0,"optimizers":0,"resources":resource},indent=2)+"\n",encoding="utf-8")
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits,threadpool_info
    import fixed_wing_env as mod
    from fixed_wing_env import FixedWingLongitudinalEnv,FixedWingMFacade,PROTOCOL,plain,digest
    from stabilization_learning.policies.metric import CholeskyMetric
    from stabilization_gym.interfaces.m_learning import CapabilityError
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    checks=[]; evidence={}; error=None
    counts={"environment_constructions":0,"environment_control_attempts":0,"numerically_integrated_control_spans":0,
        "RK4_substeps_started":0,"RK4_substeps_completed":0,"numpy_RHS_call_attempts":0,
        "Riccati_synthesis_call_attempts":0,"Riccati_backward_stages_max":0,"pure_tensor_RHS_calls":0,
        "full_episodes":0,"optimizer_instances":0,"training_updates":0}
    init=FixedWingLongitudinalEnv.__init__; step=FixedWingLongitudinalEnv.step
    integration=mod.integrate_held; rhs=mod.rhs_numpy; riccati=mod.riccati
    def count_init(self): counts["environment_constructions"]+=1; init(self)
    def count_step(self,u):
        if counts["environment_control_attempts"]>=21: raise RuntimeError("Environment control attempt cap")
        result=step(self,u); counts["environment_control_attempts"]+=1; return result
    def count_rhs(*a,**kw):
        if counts["numpy_RHS_call_attempts"]>=1000: raise RuntimeError("RHS cap")
        counts["numpy_RHS_call_attempts"]+=1; return rhs(*a,**kw)
    def count_integration(*a,**kw):
        if counts["numerically_integrated_control_spans"]>=23: raise RuntimeError("Integrated span cap")
        if counts["RK4_substeps_started"]+kw.get("substeps",10)>240: raise RuntimeError("RK started cap")
        result=integration(*a,**kw); counts["numerically_integrated_control_spans"]+=1
        counts["RK4_substeps_started"]+=result["substeps_started"]; counts["RK4_substeps_completed"]+=result["substeps_completed"]
        return result
    def count_riccati(*a,**kw):
        if counts["Riccati_synthesis_call_attempts"]>=100: raise RuntimeError("Synthesis cap")
        counts["Riccati_synthesis_call_attempts"]+=1; counts["Riccati_backward_stages_max"]+=100
        return riccati(*a,**kw)
    FixedWingLongitudinalEnv.__init__=count_init; FixedWingLongitudinalEnv.step=count_step
    mod.rhs_numpy=count_rhs; mod.integrate_held=count_integration; mod.riccati=count_riccati
    def add(name,value,detail=None): checks.append({"name":name,"passed":bool(value),"detail":plain(detail)})
    def rejects(fn,kind=ValueError):
        try: fn(); return False
        except kind: return True
    def make(options=None):
        env=FixedWingLongitudinalEnv(); env.reset(seed=41,options=options); return env,FixedWingMFacade(env)
    def advance(env,facade,M):
        e,c=env.build_error(env.state); return env.step(facade.calculate_action_from_M(M,e,context=c))
    def save(name,obj): (out/name).write_text(json.dumps(plain(obj),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    start=time.perf_counter()
    try:
      with threadpool_limits(limits=1):
        env,facade=make({"delta_state_SI":PROTOCOL["short_plan"]["coupling_state_delta_SI"]})
        mapping=CholeskyMetric(env.metric_spec).double(); rawI=mapping.identity_raw(dtype=torch.float64); I=mapping(rawI)
        desired=.06*torch.eye(5,dtype=torch.float64); desired[0,0]+=4.7
        L=torch.linalg.cholesky(desired-.05*torch.eye(5,dtype=torch.float64)); rawA=L[mapping.rows,mapping.cols].clone()
        rawA[mapping.diagonal]=torch.log(torch.expm1(rawA[mapping.diagonal]-1e-8)); anis=mapping(rawA)
        evidence["raw_metrics"]={"identity_raw":rawI,"identity_M":I,"axis_raw":rawA,"axis_M":anis}
        add("public_M_identity",torch.equal(I,torch.eye(5,dtype=torch.float64)))
        add("public_axis_finite_reconstruction",torch.isfinite(rawA).all() and torch.max(torch.abs(anis-desired))<1e-12)
        desc=facade.describe().to_dict()
        add("capability_truth",desc["gradients"]["action_wrt_M"] and not desc["gradients"]["model_wrt_action"] and not desc["control_call_is_pure"] and not desc["supported_algorithms"])
        add("final_description_signature",facade.source_signature_sha256==digest({"description":desc,"sources":env._sources}))
        add("formal_PPO_rejected",rejects(lambda:facade.require_algorithm("episode_metric_ppo"),CapabilityError))
        e,c=env.build_error(env.state); before=env.state_dict(); t=time.perf_counter(); p=env.plan(I,e,context=c); duration=time.perf_counter()-t
        add("pure_query_no_state_RNG_cache_change",digest(before)==digest(env.state_dict()),{"seconds":duration})
        add("M_distinct_from_Riccati_P0",not np.allclose(p.record["P0_not_M"],I.numpy()))
        evidence["first_plan"]=p.record
        grad_raw=rawI.clone().requires_grad_(True); grad_plan=env.plan(mapping(grad_raw),e,context=c)
        objective=grad_plan.action[0]/20+grad_plan.action[1]/.2
        gradient=torch.autograd.grad(objective,grad_raw)[0].detach().numpy(); finite_differences={}
        for index in PROTOCOL["short_plan"]["raw_FD_indices"]:
            a=rawI.clone(); b=rawI.clone(); a[index]+=1e-5; b[index]-=1e-5
            ua=env.plan(mapping(a),e,context=c).action.detach().numpy(); ub=env.plan(mapping(b),e,context=c).action.detach().numpy()
            fd=((ua[0]-ub[0])/20+(ua[1]-ub[1])/.2)/(2e-5); finite_differences[str(index)]={"autograd":gradient[index],"FD":fd}
        add("CPU64_raw_M_action_gradient_FD",all(abs(v["autograd"]-v["FD"])<=2e-6+2e-4*abs(v["FD"]) for v in finite_differences.values()),finite_differences)
        add("gradient_probe_away_from_clip",not any(grad_plan.record["saturation_mask"]))
        bad=I.clone(); bad[0,1]=1e-9
        add("asymmetric_M_rejected",rejects(lambda:env.plan(bad,e,context=c)))
        for name,options in (("negative_speed",{"state":[100,0,.1,.1,0]}),("outside_height",{"state":[181,20,.1,.1,0]}),("target_override",{"target_v":22})):
            add("invalid_reset_"+name,rejects(lambda o=options:env.reset(seed=41,options=o)))
        # Exactly predeclared five fixtures, up to 3 control attempts each.
        evidence["fixtures"]={}; retained_env=None
        for fixture in PROTOCOL["five_full_fixtures"]:
            one,onefac=make({"delta_state_SI":fixture["delta_state_SI"]})
            for _ in range(3):
                advance(one,onefac,I)
                if one._done: break
            evidence["fixtures"][fixture["name"]]=one.state_dict()
            add(fixture["name"]+"_three_valid_short_steps",one._steps==3 and not one._failed)
            add(fixture["name"]+"_no_full_success_claim",not one._last_info["success"])
            add(fixture["name"]+"_gain_cached_after_first",one._budget["submitted_gain_syntheses"]==1)
            if fixture["name"]=="exact_trim_zero_error": add("trim_invariance",np.max(np.abs(one.state-mod.XTRIM))<1e-12)
            if fixture["name"]=="height_below_5m": retained_env=(one,onefac)
        # Two physical M interventions at same predeclared small state.
        evidence["coupling"]={}
        coupling_env=None
        for label,M in (("identity",I),("height_axis",anis)):
            one,onefac=make({"delta_state_SI":PROTOCOL["short_plan"]["coupling_state_delta_SI"]})
            advance(one,onefac,M); evidence["coupling"][label]=one.state_dict()
            if label=="identity": coupling_env=(one,onefac)
        ai=np.array(evidence["coupling"]["identity"]["trace"][0]["action"]); aa=np.array(evidence["coupling"]["height_axis"]["trace"][0]["action"])
        add("M_changes_actual_clipped_physical_action",np.linalg.norm(ai-aa)>1e-6,{"identity":ai,"axis":aa})
        # Existing cached numeric K may not sever M gradients.
        one,onefac=coupling_env; e,c=one.build_error(one.state); differentiated=rawI.clone().requires_grad_(True)
        cache_before=one.state_dict(); cached_graph=one.plan(mapping(differentiated),e,context=c)
        actual_grad=torch.autograd.grad(cached_graph.action.sum(),differentiated)[0]
        add("cached_rollout_does_not_cut_M_graph",not cached_graph.record["cache_used"] and cached_graph.record["gain_syntheses"]==1 and torch.isfinite(actual_grad).all() and torch.linalg.norm(actual_grad)>1e-10)
        add("gradient_query_keeps_cache_state_pure",digest(cache_before)==digest(one.state_dict()))
        error_grad=e.detach().clone().requires_grad_(True)
        cached_error_plan=one.plan(I,error_grad,context=c)
        jacobian=torch.stack([torch.autograd.grad(cached_error_plan.action[j],error_grad,retain_graph=True)[0] for j in range(2)])
        expected_jacobian=-mod.US[:,None]*np.asarray(cached_error_plan.record["K"])
        expected_jacobian[np.array(cached_error_plan.record["saturation_mask"])]=0.
        add("cached_numeric_K_preserves_error_gradient",cached_error_plan.record["cache_used"] and cached_error_plan.record["gain_syntheses"]==0 and np.allclose(jacobian.numpy(),expected_jacobian,rtol=1e-12,atol=1e-12))
        # Complete nonempty/pending/cache/RNG restore; original + 2 clones = 3 attempts.
        one,onefac=retained_env; e,c=one.build_error(one.state); action=onefac.calculate_action_from_M(I,e,context=c)
        pending=one.state_dict(); clone1,_=make(); clone2,_=make()
        restore1=clone1.load_state_dict(pending); restore2=clone2.load_state_dict(json.loads(json.dumps(pending)))
        add("complete_pending_restore_identical",digest(pending)==digest(clone1.state_dict())==digest(clone2.state_dict()))
        add("restore_synthesis_disclosed",restore1["validation_gain_syntheses"]==1 and restore2["validation_gain_syntheses"]==1)
        add("pending_action_mutation_rejected",rejects(lambda:one.step(action.detach().numpy()+1e-6)))
        for obj in (one,clone1,clone2): obj.step(action)
        add("restored_next_step_exact",digest(one.state_dict())==digest(clone1.state_dict())==digest(clone2.state_dict()))
        evidence["recovery"]={"pending":pending,"original_after":one.state_dict(),"clone1_after":clone1.state_dict(),"clone2_after":clone2.state_dict(),"restore_audit":[restore1,restore2]}
        # One invalid-stage probe: retain last accepted state, do not fake .05s.
        near=mod.XTRIM.copy(); near[0]=179.9999; near[3]+=.05
        one,onefac=make({"state":near}); result=advance(one,onefac,I)
        evidence["failure"]=one.state_dict()
        add("invalid_RK_stage_failure_recorded",result[2] and one._failure["kind"]=="invalid_RK_stage")
        add("partial_time_not_full_control_interval",one._physical_time<.05 and one._physical_time==one._budget["RK4_substeps_completed"]*.005)
        expected_tail=max(float((((one.state-mod.XTRIM)/mod.S)**2).sum()),25.)*(30-one._physical_time)
        add("failure_padded_actual_remaining_time",np.isclose(result[1],one._cost+expected_tail,rtol=0,atol=1e-10))
        failed_clone,_=make(); restore_failure=failed_clone.load_state_dict(one.state_dict())
        add("failed_partial_snapshot_restore",digest(failed_clone.state_dict())==digest(one.state_dict()))
        evidence["failure_restore_activity"]=restore_failure
        # Two explicitly budgeted held-input numerical integrations, not static checks.
        state=mod.XTRIM+np.array(PROTOCOL["short_plan"]["coupling_state_delta_SI"])
        coarse=mod.integrate_held(state,mod.UTRIM,dt=.005,substeps=10)
        fine=mod.integrate_held(state,mod.UTRIM,dt=.0025,substeps=20)
        evidence["step_halving"]={"coarse":coarse,"fine":fine}
        add("step_halving_no_domain_failure",coarse["failure"] is None and fine["failure"] is None)
        add("step_halving_scaled_difference",np.max(np.abs((coarse["state"]-fine["state"])/mod.S))<1e-7,{"scaled_max":np.max(np.abs((coarse["state"]-fine["state"])/mod.S))})
        add("exact_proposed_attempt_and_span_counts",counts["environment_control_attempts"]==21 and counts["numerically_integrated_control_spans"]==23,counts)
        add("no_complete_episodes_or_learning",counts["full_episodes"]==0 and counts["optimizer_instances"]==0 and counts["training_updates"]==0)
        add("protected51_unchanged",{p:sha(PROJECT/p) for p in frozen}==frozen)
        add("local_frozen_source_unchanged",{p:sha(HERE/p) for p in sources}==sources)
        evidence["threadpools"]=threadpool_info()
    except Exception as exc:
        error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}; add("execution_exception",False,error)
    finally:
        save("SAVED_EVIDENCE.json",evidence)
        summary={"utc":datetime.now(timezone.utc).isoformat(),"scope":"bounded_short_engineering_no_training",
            "passed":bool(checks and all(x["passed"] for x in checks)),"checks_passed":sum(x["passed"] for x in checks),"checks_total":len(checks),
            "sources":sources,"counts":counts,"checks":checks,"error":error,"elapsed_seconds":time.perf_counter()-start,
            "limitations":["No complete 30s fixture or final21-state hold acceptance.","Same-M actual V may rise; no certificate acceptance.","Additional pure query/FD/cache-validation Riccati calls included in global counts.","Two step-halving spans are actual numerical integrations and included in23 spans.","No learner, optimizer, split generation or formal training."]}
        save("SUMMARY.json",summary); print(json.dumps({k:summary[k] for k in ("passed","checks_passed","checks_total","counts","error")},ensure_ascii=False))
    return 1 if error else 0 if all(x["passed"] for x in checks) else 2


if __name__=="__main__": raise SystemExit(main())
