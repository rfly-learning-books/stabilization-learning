"""CPU1 static/boundary checks and five predeclared episodes, no optimization."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
PROTOCOL=PROJECT/"runs/book_m_night_suite_20261023_v1/protocol.json"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_default(value):
    if hasattr(value,"tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("static","episodes"),default="static")
    parser.add_argument("--static-evidence",type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(HERE):
        raise ValueError("Fresh output directory must be below isolated flexible_mechanics folder")
    frozen=json.loads(PROTOCOL.read_text(encoding="utf-8"))["sources"]
    if {p:sha(PROJECT/p) for p in frozen}!=frozen:
        raise RuntimeError("Active suite frozen sources differ before imports")
    local={p.name:sha(p) for p in (HERE/"flexible_mechanics_env.py",Path(__file__),HERE/"ENGINEERING_DECLARATION.json")}
    protocol_sha=sha(PROTOCOL)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    for p in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control",
              PROJECT/"third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages"):
        sys.path.insert(0,str(p))
    import numpy as np
    import torch
    from scipy.integrate import solve_ivp
    from threadpoolctl import threadpool_limits,threadpool_info
    from stabilization_gym.interfaces.m_learning import PortableMFacade,CapabilityError
    from stabilization_learning.policies.metric import CholeskyMetric
    from stabilization_learning.wrappers import rollout_metric
    from flexible_mechanics_env import FlexibleMechanicsEnv
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    checks=[]; evidence={}; summaries=[]; executed={"boundary_steps":0,"full_episode_steps":0,"full_episodes":0}
    current_full=False
    original_step=FlexibleMechanicsEnv.step
    def counted_step(env,action):
        result=original_step(env,action)
        executed["full_episode_steps" if current_full else "boundary_steps"]+=1
        return result
    FlexibleMechanicsEnv.step=counted_step
    def add(name,passed,detail=None):
        checks.append({"name":name,"passed":bool(passed),"detail":detail})
    def write(name,data):
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=json_default,allow_nan=False)+"\n",encoding="utf-8")
    def rejects(fn,kind=ValueError):
        try:
            fn(); return False
        except kind:
            return True
    def env_at(state=None,N=50):
        env=FlexibleMechanicsEnv(target_angle=.1,prediction_steps=N)
        env.reset(seed=41,options={} if state is None else {"state":state})
        return env
    I=torch.eye(4,dtype=torch.float64)
    initial=np.array([.2+5*.05/7,.2-2*.05/7,.1,-.1])
    failure=None
    try:
        with threadpool_limits(limits=1):
            write("START_MANIFEST.json",{"utc":datetime.now(timezone.utc).isoformat(),"local_sources":local,
                "frozen_sources":frozen,"protocol_sha256":protocol_sha,"phase":args.phase,
                "cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"threads":threadpool_info(),
                "optimizers":0,"training_episodes":0,"engineering_only":True})
            if args.phase=="static":
                env=env_at(initial); facade=PortableMFacade(env,name="isolated_dual_inertia")
                desc=facade.describe().to_dict(); metric_map=CholeskyMetric(env.metric_spec).double()
                raw=metric_map.identity_raw(dtype=torch.float64)
                add("public_metric_raw_dimension",env.metric_spec.raw_dim==10 and desc["raw_metric_dim"]==10)
                add("public_metric_identity",torch.allclose(metric_map(raw),I,atol=1e-14,rtol=0))
                add("public_facade_pending_audit_declared",not desc["control_call_is_pure"] and "audit_pending_metric_and_action" in desc["control_call_side_effects"])
                add("capabilities_match_physics",desc["gradients"]["action_wrt_M"] and desc["gradients"]["model_wrt_action"] and not desc["gradients"]["physical_step_autograd"])
                add("fixed_target_context_only",np.array_equal(env.metric_context(initial).numpy(),np.array([1/3],dtype=np.float32)))
                add("target_context_has_no_autograd",not env.metric_context(initial).requires_grad)
                add("pure_control_capability_rejected",rejects(lambda:facade.require_capability("pure_control"),CapabilityError))
                add("N_other_than_1_50_rejected",rejects(lambda:FlexibleMechanicsEnv(prediction_steps=25)))
                add("target_out_of_domain_rejected",rejects(lambda:FlexibleMechanicsEnv(target_angle=.31)))
                add("coordinate_inverse",np.max(abs(env.S@env.Sinv-np.eye(4)))<1e-14)
                # Independent integration solves original physical ODE for each augmented basis.
                lifted=np.zeros((5,5)); lifted[:4,:4],lifted[:4,4:]=env.A,env.B
                independent=solve_ivp(lambda t,y:(lifted@y.reshape(5,5)).ravel(),(0,.01),np.eye(5).ravel(),
                    method="DOP853",rtol=2.5e-14,atol=1e-15).y[:,-1].reshape(5,5)
                add("independent_ZOH_match",np.max(abs(independent[:4,:4]-env.Ad))<2e-13 and np.max(abs(independent[:4,4:]-env.Bd))<2e-13)
                add("continuous_modes",np.allclose(np.sort_complex(np.linalg.eigvals(env.A)),np.sort_complex(np.array([0.,-1.,-.5+5.894913061275798j,-.5-5.894913061275798j])),atol=1e-10))
                add("target_is_equilibrium",np.allclose(env.Ad@env.goal,env.goal,atol=1e-15,rtol=0))
                prediction=env.S@(initial-env.goal)
                for _ in range(50): prediction=env.Ae@prediction+env.Be[:,0]*.07
                add("N50_constant_torque_prediction",np.max(abs(prediction-(env.AN@(env.S@(initial-env.goal))+env.BN[:,0]*.07)))<1e-14)
                add("N50_does_not_change_actual_dt",env.dt==.01 and env.max_steps==12000)
                e,ctx=env.build_error(initial)
                # M directions preserve symmetry and trace, and stay strictly inside constraints.
                m=I.clone().requires_grad_(True); ee=e.clone().requires_grad_(True)
                action=env.calculate_action_from_M(m,ee,context=ctx)
                gm,ge=torch.autograd.grad(action.sum(),(m,ee))
                directions=[]
                for j in range(1,4):
                    d=torch.zeros_like(I); d[0,j]=d[j,0]=1.; directions.append(d)
                d=torch.diag(torch.tensor([1.,-1.,0.,0.],dtype=torch.float64)); directions.append(d)
                eps=1e-6; m_errors=[]
                for d in directions:
                    fd=(env.calculate_action_from_M(I+eps*d,e,context=ctx)-env.calculate_action_from_M(I-eps*d,e,context=ctx))/(2*eps)
                    m_errors.append(abs(float(fd)-float((gm*d).sum())))
                add("M_symmetric_trace_preserving_finite_difference",max(m_errors)<2e-10,max(m_errors))
                e_errors=[]
                for j in range(4):
                    d=torch.eye(4,dtype=torch.float64)[j]
                    fd=(env.calculate_action_from_M(I,e+eps*d,context=ctx)-env.calculate_action_from_M(I,e-eps*d,context=ctx))/(2*eps)
                    e_errors.append(abs(float(fd)-float(ge[j])))
                add("error_gradient_finite_difference",max(e_errors)<2e-10,max(e_errors))
                rr=(raw+torch.linspace(-.04,.05,10,dtype=torch.float64)).requires_grad_(True)
                ar=env.calculate_action_from_M(metric_map(rr),e,context=ctx)
                gr,=torch.autograd.grad(ar.sum(),rr); raw_errors=[]
                for j in range(10):
                    d=torch.eye(10,dtype=torch.float64)[j]
                    fd=(env.calculate_action_from_M(metric_map(rr.detach()+eps*d),e,context=ctx)-env.calculate_action_from_M(metric_map(rr.detach()-eps*d),e,context=ctx))/(2*eps)
                    raw_errors.append(abs(float(fd)-float(gr[j])))
                add("public_raw_to_M_to_action_gradient",max(raw_errors)<2e-10 and float(gr.abs().max())>1e-5,max(raw_errors))
                add("query_does_not_advance_physics",env._elapsed_steps==0 and np.array_equal(env.state,initial) and env._episode_metric is None)
                x=torch.tensor(initial,dtype=torch.float64,requires_grad=True); u=torch.tensor([.07],dtype=torch.float64,requires_grad=True)
                jx,ju=torch.autograd.functional.jacobian(lambda xx,uu:env.torch_step(xx,uu),(x,u))
                add("Torch_state_jacobian_is_Ad",np.max(abs(jx.numpy()-env.Ad))<1e-14)
                add("Torch_action_jacobian_is_Bd",np.max(abs(ju.numpy()-env.Bd))<1e-14)
                real=env_at(initial); obs,_,_,_,_=real.step([.07])
                add("Gym_step_matches_differentiable_ZOH",np.max(abs(obs-env.torch_step(x,u).detach().numpy()))<1e-14)
                add("Gym_step_executes_one_not_50",real._elapsed_steps==1 and np.max(abs(obs-(env.goal+env.Sinv@prediction)))>1e-3)
                add("Gym_observation_is_non_autograd",isinstance(obs,np.ndarray))
                saturated=env_at(initial); so,_,_,_,si=saturated.step([2.])
                add("physical_torque_saturation",si["applied_action"][0]==.5 and np.allclose(so,env.Ad@initial+env.Bd[:,0]*.5,atol=1e-14))
                sat_e=torch.tensor([100.,0.,0.,0.],dtype=torch.float64,requires_grad=True)
                sa=env.calculate_action_from_M(I,sat_e,context=ctx)
                add("controller_saturation_is_finite_piecewise_gradient",float(sa)==-.5 and torch.equal(torch.autograd.grad(sa.sum(),sat_e)[0],torch.zeros(4,dtype=torch.float64)))
                add("finite_action_validation",rejects(lambda:env_at().step([float("nan")])))
                add("finite_error_validation",rejects(lambda:env.calculate_action_from_M(I,torch.full((4,),float("nan")),context=ctx)))
                add("metric_bounds_validation",rejects(lambda:env.calculate_action_from_M(torch.diag(torch.tensor([.01,1.,1.,1.99],dtype=torch.float64)),e,context=ctx)))
                add("no_artificial_absolute_angle_limit",not env.is_failure([100.,100.,0.,0.]))
                add("no_angle_or_twist_wrapping",env.is_failure([2*np.pi,0.,0.,0.]))
                edge=env_at([.5,0.,2.9,-2.9]); eo,_,et,ex,ei=edge.step([.5])
                add("physical_failure_unclipped",et and not ex and ei["failure"] and abs(eo[0]-eo[1])>.5)
                add("cannot_continue_failed_episode",rejects(lambda:edge.step([0.]),RuntimeError))
                changed=env_at(initial); changed.goal[0]+=.01
                add("changed_reference_rejected",rejects(lambda:changed.build_error(initial)))
                changed=env_at(initial); changed.prediction_steps=1
                add("changed_N_rejected",rejects(lambda:changed.build_error(initial)))
                bad_goal=torch.tensor(env.goal,requires_grad=True)
                add("learnable_goal_rejected",rejects(lambda:env.build_error(initial,goal=bad_goal)))
                # Pending control rejection and executed-M locking, no graph detachment.
                p=env_at(initial); pa=p.unit_metric_action(); before=p.state.copy()
                add("pending_snapshot_rejected",rejects(p.state_dict))
                add("tampered_controller_action_rejected",rejects(lambda:p.step(pa.detach().numpy()+.001)) and np.array_equal(before,p.state))
                p.step(pa.detach().numpy()); pe,pc=p.build_error(p.state)
                altered=I.clone(); altered[0,2]=altered[2,0]=-.5
                ua=p.calculate_action_from_M(altered,pe,context=pc)
                add("episode_M_change_rejected",rejects(lambda:p.step(ua.detach().numpy())))
                p._pending_control=None
                add("mixed_external_policy_rejected",rejects(lambda:p.step([0.])))
                p=env_at(initial); se,sc=p.build_error(p.state+np.array([.001,0.,0.,0.]))
                stale=p.calculate_action_from_M(I,se,context=sc)
                add("stale_state_control_rejected",rejects(lambda:p.step(stale.detach().numpy())))
                # Exact environment boundary restoration and deterministic subsequent RNG reset.
                p=env_at(initial)
                for _ in range(4): p.step(p.unit_metric_action().detach().numpy())
                saved=p.state_dict(); restored=FlexibleMechanicsEnv(target_angle=.1)
                restored.load_state_dict(saved)
                a=p.unit_metric_action(); b=restored.unit_metric_action()
                left=p.step(a.detach().numpy()); right=restored.step(b.detach().numpy())
                add("boundary_restore_next_action_exact",torch.equal(a,b))
                add("boundary_restore_next_state_exact",np.array_equal(left[0],right[0]) and left[1:4]==right[1:4])
                add("boundary_restore_audit_exact",json.dumps(p.get_episode_audit(),default=json_default,sort_keys=True)==json.dumps(restored.get_episode_audit(),default=json_default,sort_keys=True))
                add("boundary_restore_rng_next_reset_exact",np.array_equal(p.reset()[0],restored.reset()[0]))
                for field in ("state","episode_metric","config","trace_cost","trace_raw","trace_action","trace_energy","trace_success","last_info","pending_key"):
                    corrupted=deepcopy(saved)
                    if field=="state": corrupted["state"][0]+=.001
                    elif field=="episode_metric": corrupted["episode_metric"][0,2]=corrupted["episode_metric"][2,0]=-.5
                    elif field=="config": corrupted["config"]["prediction_steps"]=1
                    elif field=="trace_cost": corrupted["trace"][0]["cost"]+=.01
                    elif field=="trace_raw": corrupted["trace"][0]["raw_torque"]+=.01
                    elif field=="trace_action": corrupted["trace"][0]["submitted_action"][0]+=.01
                    elif field=="trace_energy": corrupted["trace"][0]["mechanical_energy_after"]+=.01
                    elif field=="trace_success": corrupted["trace"][0]["success"]=True
                    elif field=="last_info": corrupted["last_info"]["hold_seconds"]+=.01
                    elif field=="pending_key": corrupted["pending_control"]={}
                    add("snapshot_rejects_corruption_"+field,rejects(lambda c=corrupted:FlexibleMechanicsEnv(target_angle=.1).load_state_dict(c)))
                seed_a=FlexibleMechanicsEnv(); seed_b=FlexibleMechanicsEnv()
                add("reset_seed_reproducible",np.array_equal(seed_a.reset(seed=41)[0],seed_b.reset(seed=41)[0]) and np.array_equal(seed_a.goal,seed_b.goal))
                # Candidate V increase witness retained from predeclared static design.
                b=env.BN[:,0]; K=(b@env.AN)/(b@b+.04); F=env.Ae-env.Be@K[None,:]
                w,v=np.linalg.eigh(F.T@F-np.eye(4)); ew=.01*v[:,-1]
                witness=env_at(env.goal+env.Sinv@ew)
                wobs=witness.state.copy(); wa=witness.unit_metric_action(); wn,_,_,_,_=witness.step(wa.detach().numpy())
                prediction=env.AN@ew+b*float(wa)
                actual=env.S@(wn-env.goal)
                evidence["V_counterexample"]={"initial_V":float(ew@ew),"predicted_terminal_V":float(prediction@prediction),
                    "actual_next_V":float(actual@actual),"state":wobs,"action":float(wa),"max_eigenvalue_actual_delta":float(w[-1])}
                add("same_M_actual_V_rise_retained",actual@actual>ew@ew and prediction@prediction<ew@ew)
                add("N50_identity_actual_step_Schur",max(abs(np.linalg.eigvals(F)))<1.)
                neg=I.numpy().copy(); neg[0,2]=neg[2,0]=-.5
                nk=(b@neg@env.AN)/(b@neg@b+.04)
                add("negative_cross_M_local_instability_retained",max(abs(np.linalg.eigvals(env.Ae-env.Be@nk[None,:])))>1.)
                # Failure-tail check: one physical boundary step, not another full episode.
                tailenv=env_at([.5,0.,2.9,-2.9]); tobs=tailenv.state.copy()
                tail_result=rollout_metric(PortableMFacade(tailenv),I,tobs)
                ta=tailenv.get_episode_audit(); expected=ta["trace"][0]["cost"]+(120.-.01)*max(ta["physical_error"],25.)
                add("public_boundary_failure_tail",tail_result["failure"] and tail_result["physical_steps"]==1 and abs(tail_result["fixed_horizon_cost"]-expected)<1e-10)
                evidence.update(description=desc,Ad=env.Ad,Bd=env.Bd,K50=K,
                    restore_scope="Environment only at completed physical-step boundaries, no learner/replay/optimizer",
                    boundary_tail=tail_result)
                write("BOUNDARY_SNAPSHOT.json",saved)
            else:
                if args.static_evidence is None:
                    raise ValueError("Episodes require completed static evidence")
                static=json.loads((args.static_evidence/"SUMMARY.json").read_text(encoding="utf-8"))
                if not static["passed"] or static["source_hashes"]!=local or static["phase"]!="static":
                    raise RuntimeError("Static engineering evidence missing/failed or source changed")
                ledger=HERE/"FULL_EPISODE_EXECUTION_LOCK.json"
                with ledger.open("x",encoding="utf-8") as handle:
                    json.dump({"output":str(output),"predeclared_maximum":5,"utc":datetime.now(timezone.utc).isoformat()},handle,indent=2)
                cases=[("N50_unit_M",50,"metric"),("N1_legacy_unit_M",1,"metric"),
                       ("zero_feedforward",50,"zero"),("N50_negative_cross_M",50,"negative"),
                       ("late_torque_after_early_hold",50,"late")]
                for name,N,kind in cases:
                    current_full=True
                    env=FlexibleMechanicsEnv(target_angle=.1,prediction_steps=N)
                    origin="fixed_zero_feedforward" if kind=="zero" else "deliberate_fault_injection_not_M_policy_performance" if kind=="late" else "external_physical_action"
                    obs,_=env.reset(seed=41,options={"state":[.1,.1,0.,0.] if kind=="late" else initial,"execution_origin":origin})
                    if kind in ("metric","negative"):
                        M=I.clone()
                        if kind=="negative": M[0,2]=M[2,0]=-.5
                        public=rollout_metric(PortableMFacade(env,name="isolated_dual_inertia"),M,obs)
                    else:
                        public=None
                        for i in range(12000):
                            _,_,term,trunc,_=env.step([.5 if kind=="late" and i>=11800 else 0.])
                            if term or trunc: break
                    current_full=False; executed["full_episodes"]+=1
                    audit=env.get_episode_audit()
                    audit.update(engineering_case=name,public_rollout_result=public,
                        not_a_training_result=True,not_a_statistical_performance_benchmark=True)
                    write(name+".json",audit)
                    traces=audit["trace"]
                    states=np.array([audit["initial_state"]]+[row["state_after"] for row in traces])
                    actions=np.array([row["action"] for row in traces])
                    goal=np.array(audit["goal"])
                    physical=np.column_stack((states[:,1]-.1,states[:,0]-states[:,1],states[:,2],states[:,3]))
                    good=np.all(abs(physical)<=np.array([.01,.01,.03,.03]),axis=1)
                    n=len(traces); independent_failure=env.is_failure(states[-1])
                    independent_success=bool(n==12000 and np.all(good[-201:]) and not independent_failure)
                    raw_cost=float(.01*np.square(physical[:-1]).sum())
                    terminal=float(physical[-1]@physical[-1])
                    fixed=raw_cost+(120.-n*.01)*(max(terminal,25.) if independent_failure else terminal)
                    add(name+"_real_ZOH_chain",np.max(abs(states[1:]-(states[:-1]@env.Ad.T+actions@env.Bd.T)))<1e-13)
                    add(name+"_action_bounds",np.max(abs(actions))<=.5)
                    add(name+"_final_201_hold",audit["success"]==independent_success)
                    add(name+"_left_physical_cost",abs(sum(row["cost"] for row in traces)-raw_cost)<1e-10)
                    add(name+"_no_early_success",all(not row["success"] for row in traces[:-1]))
                    add(name+"_mechanical_power",max(abs(row["mechanical_power_before"]-(row["action"][0]*row["state_before"][2]-.02*row["state_before"][2]**2-.05*row["state_before"][3]**2)) for row in traces)<1e-14)
                    rises=positive_count=0; max_delta=None; actual_vs=None
                    if public is not None:
                        mat=np.array(audit["metric"]); errors=(states-goal)@env.S.T
                        vs=np.einsum("ni,ij,nj->n",errors,mat,errors); delta=np.diff(vs)
                        rises=int((delta>1e-8).sum()); positive_count=int((delta>0.).sum()); max_delta=float(delta.max())
                        v_saved=np.array([[r["V_before"],r["V_after"]] for r in traces])
                        add(name+"_same_M_V",np.max(abs(v_saved-np.column_stack((vs[:-1],vs[1:]))))<1e-11)
                        b=env.BN[:,0]; k=(b@mat@env.AN)/(b@mat@b+.04)
                        reconstructed=np.clip(-(errors[:-1]@k),-.5,.5)
                        add(name+"_M_drives_executed_action",np.max(abs(reconstructed-actions[:,0]))<1e-12)
                        add(name+"_public_fixed_cost_tail",abs(public["fixed_horizon_cost"]-fixed)<1e-9)
                        add(name+"_public_M_audit",audit["metric_control_steps"]==n and audit["execution_origin"]=="M_policy")
                        add(name+"_public_V_rise_fraction",abs(public["V_rise_fraction"]-rises/n)<1e-14)
                        if independent_success:
                            add(name+"_public_arrival_limit_explicit",public["arrival_time"]==120. and audit["arrival_time"]<=120.)
                    else:
                        add(name+"_V_not_applicable",audit["metric"] is None and all(row["V_before"] is None and row["V_after"] is None for row in traces))
                    summary={"case":name,"steps":n,"success":independent_success,"failure":independent_failure,
                        "raw_integral":raw_cost,"fixed_horizon_cost":fixed,"terminal_physical_error":terminal,
                        "terminal_physical_components":physical[-1],"settling_time":audit["settling_time"],
                        "held_arrival_time":audit["arrival_time"],"first_hold_confirmation":audit["first_hold_confirmation_time"],
                        "V_rises_above_1e-8":None if public is None else rises,
                        "V_strictly_positive_deltas":None if public is None else positive_count,
                        "V_max_delta":max_delta,"V_rise_fraction":None if public is None else rises/n,
                        "execution_origin":audit["execution_origin"],"snapshot_role":"engineering trajectory, not training checkpoint"}
                    summaries.append(summary)
                    if kind=="late":
                        add("late_fault_early_hold_does_not_imply_final_success",audit["first_hold_confirmation_time"]==2. and not independent_success and audit["execution_origin"]=="deliberate_fault_injection_not_M_policy_performance")
                    print(json.dumps(summary,default=json_default),flush=True)
                add("exactly_five_predeclared_full_episodes",executed["full_episodes"]==5 and executed["full_episode_steps"]<=60000)
            add("frozen_sources_unchanged",{p:sha(PROJECT/p) for p in frozen}==frozen and sha(PROTOCOL)==protocol_sha)
            add("local_sources_unchanged",{n:sha(HERE/n) for n in local}==local)
            add("CPU_single_thread",torch.get_num_threads()==1 and torch.get_num_interop_threads()==1 and all(item["num_threads"]==1 for item in threadpool_info()))
    except Exception as exc:
        failure={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        write("ERROR.json",failure)
    finally:
        summary={"phase":args.phase,"passed":failure is None and all(c["passed"] for c in checks),
            "checks_passed":sum(c["passed"] for c in checks),"checks_total":len(checks),
            "source_hashes":local,"frozen_source_count":len(frozen),"executed":executed,
            "episodes":summaries,"error":failure,"optimizer_count":0,"formal_training":False,
            "same_M_Lyapunov_proof":False,"N_or_threshold_tuning_after_episodes":False}
        write("CHECKS.json",checks); write("EVIDENCE.json",evidence); write("SUMMARY.json",summary)
        write("FINAL_MANIFEST.json",{"sources":local,"frozen_sources_unchanged":{p:sha(PROJECT/p) for p in frozen}==frozen,
            "files":{p.name:sha(p) for p in output.iterdir() if p.is_file()},"utc":datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary,default=json_default),flush=True)
    return 0 if summary["passed"] else 1

if __name__=="__main__":
    raise SystemExit(main())
