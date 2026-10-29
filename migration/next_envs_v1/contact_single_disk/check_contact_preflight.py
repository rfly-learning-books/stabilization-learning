"""Frozen-design CPU short checks and runtime gate; never a full fixture."""
from copy import deepcopy,copy
from datetime import datetime,timezone
import hashlib,json,os,sys,time,traceback
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[2]
OUTPUT=HERE/"checks/preflight_v1"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ready(value):
    if hasattr(value,"tolist"): return value.tolist()
    raise TypeError(type(value).__name__)
def write(name,data):
    (OUTPUT/name).write_text(json.dumps(data,indent=2,default=ready,allow_nan=False)+"\n",encoding="utf-8")
frozen=json.loads((PROJECT/"runs/book_m_night_suite_20261023_v1/protocol.json").read_text(encoding="utf-8"))["sources"]
if {p:sha(PROJECT/p) for p in frozen}!=frozen: raise RuntimeError("Frozen sources differ")
sources={p.name:sha(p) for p in (HERE/"DESIGN_PROTOCOL.json",HERE/"contact_single_disk_env.py",Path(__file__))}
OUTPUT.mkdir(parents=True,exist_ok=False)
write("START_MANIFEST.json",{"utc":datetime.now(timezone.utc).isoformat(),"sources":sources,"frozen":frozen,
    "full_episodes_authorized_here":0,"training_updates":0,"optimizers":0})
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
os.environ.update(CUDA_VISIBLE_DEVICES="-1",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1")
for p in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control",PROJECT/"third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages"):
    sys.path.insert(0,str(p))
import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info
import mujoco
from contact_single_disk_env import ContactSingleDiskEnv,ContactMFacade,PROTOCOL,_digest
from stabilization_gym.interfaces.m_learning import CapabilityError
from stabilization_learning.policies.metric import CholeskyMetric
torch.set_num_threads(1); torch.set_num_interop_threads(1)
checks=[]; records=[]; coupling=[]; error=None
counts={"model_constructions":0,"queries":0,"prediction_substeps":0,"direct_clone_substeps":0,"real_control_steps":0,"real_substeps":0,"full_episodes":0}
real_init=ContactSingleDiskEnv.__init__; real_plan=ContactSingleDiskEnv.plan; real_step=ContactSingleDiskEnv.step
def counted_init(self,*a,**kw):
    result=real_init(self,*a,**kw); counts["model_constructions"]+=1; return result
def counted_plan(self,*a,**kw):
    if counts["queries"]>=12: raise RuntimeError("Predeclared query budget exhausted")
    result=real_plan(self,*a,**kw); counts["queries"]+=1
    counts["prediction_substeps"]+=result["budget"]["prediction_substeps"]
    if counts["prediction_substeps"]>108000: raise RuntimeError("Prediction step budget exceeded")
    return result
def counted_step(self,*a,**kw):
    if counts["real_control_steps"]>=50: raise RuntimeError("Predeclared real-step budget exhausted")
    result=real_step(self,*a,**kw); counts["real_control_steps"]+=1; counts["real_substeps"]+=4
    return result
ContactSingleDiskEnv.__init__=counted_init; ContactSingleDiskEnv.plan=counted_plan; ContactSingleDiskEnv.step=counted_step
def add(name,value,detail=None): checks.append({"name":name,"passed":bool(value),"detail":detail})
def rejects(fn,kind=ValueError):
    try: fn(); return False
    except kind: return True
def new(fixture):
    env=ContactSingleDiskEnv(); env.reset(seed=41,options={k:v for k,v in fixture.items() if k!="name"}); return env
I=torch.eye(8,dtype=torch.float64)
fixtures=PROTOCOL["five_full_fixtures"]
try:
  with threadpool_limits(limits=1):
    env=new(fixtures[0]); facade=ContactMFacade(env); desc=facade.describe().to_dict()
    add("engine_pinned",mujoco.__version__=="3.2.3")
    add("physical_dimensions",env.model.nq==5 and env.model.nv==5 and env.model.nu==2 and env.model.na==0)
    add("metric_dimensions",env.error_dim==8 and env.metric_spec.raw_dim==36 and env.metric_context_dim==10)
    add("table_floor_no_collision",np.all(env.model.geom_contype[:3]==0) and np.all(env.model.geom_conaffinity[:3]==0))
    add("margins_zero",np.all(env.model.geom_margin==0) and np.all(env.model.jnt_margin==0))
    add("disk_pusher_masses",np.isclose(env.model.body_mass[2],.025) and np.isclose(env.model.body_mass[3],.0301),env.model.body_mass)
    add("facade_gradient_truth",not desc["gradients"]["action_wrt_M"] and not desc["gradients"]["model_wrt_action"] and desc["gradients"]["candidate_scores_wrt_certificate"])
    add("facade_audit_side_effect_explicit",not desc["control_call_is_pure"] and desc["control_call_side_effects"]==("audit_pending_plan",))
    add("pure_environment_query_declared",env.control_call_is_pure)
    sample=facade.error_context(env.state)
    add("hard_action_gradient_rejected",rejects(lambda:facade.control(I,sample,training=True),CapabilityError))
    add("float32_error_rejected",rejects(lambda:env.build_error(torch.tensor(env.state,dtype=torch.float32))))
    mapping=CholeskyMetric(env.metric_spec).double(); raw=mapping.identity_raw(dtype=torch.float64)
    add("public_cholesky_identity",torch.allclose(mapping(raw),I,atol=1e-14,rtol=0))
    for name,options in (("overlap",{"disk_xy":[0,0],"pusher_xy":[.01,0]}),
                         ("boundary",{"disk_xy":[-.055,0]}),("bad_goal",{"goal_xy":[.05,0]}),
                         ("nonfinite",{"disk_xy":[float("nan"),0]})):
        add("invalid_reset_"+name,rejects(lambda o=options:env.reset(seed=41,options=o)))
    # Reset returns to the declaration after invalid-input probes.
    env=new(fixtures[0]); e,ctx=env.build_error(env.state)
    before=env.state_dict(); one=env.plan(I,e,context=ctx); two=env.plan(I,e,context=ctx)
    after=env.state_dict()
    add("pure_plan_full_environment_unchanged",_digest(before)==_digest(after))
    stable1=deepcopy(one);stable2=deepcopy(two)
    for p in (stable1,stable2): p.pop("plan_sha256");p["budget"].pop("planning_seconds")
    add("same_query_same_candidates_and_action",_digest(stable1)==_digest(stable2))
    write("PURE_QUERY_PAIR.json",{"first":one,"second":two})
    plans=[]; first_env=None
    for fixture in fixtures:
        current=new(fixture); public=ContactMFacade(current)
        observation=current.state.copy(); ee,cc=public.build_error(observation)
        action=public.calculate_action_from_M(I.float(),ee,context=cc)
        plan=deepcopy(current._pending_plan)
        plans.append(plan)
        add(fixture["name"]+"_actor_float32_M_to_float64",np.array_equal(plan["M"],np.eye(8)))
        add(fixture["name"]+"_query_did_not_lock_M",current._episode_metric is None and current._elapsed_steps==0)
        add(fixture["name"]+"_pending_snapshot_rejected",rejects(current.state_dict))
        next_state,_,terminated,truncated,info=public.step(action.numpy())
        if plan["selected_index"] is not None and not plan["target_zero_shortcut"]:
            predicted=np.array(plan["candidates"][plan["selected_index"]]["first_state"])
            add(fixture["name"]+"_clone_real_first_step",np.array_equal(predicted,next_state),float(np.max(abs(predicted-next_state))))
        add(fixture["name"]+"_exactly_four_physical_substeps",len(current._trace[0]["substeps"])==4 and abs(current.data.time-.02)<1e-14)
        add(fixture["name"]+"_no_early_success",not info["success"] and not truncated)
        add(fixture["name"]+"_M_committed_only_at_step",np.array_equal(current._episode_metric,np.eye(8)))
        records.append(current.get_episode_audit())
        write(fixture["name"]+"_FIRST_STEP.json",records[-1])
        if first_env is None: first_env=current
    # Predeclared M list, evaluated only on the already fixed candidates.
    vectors=list(np.eye(8)); rng=np.random.default_rng(41)
    for _ in range(8):
        v=rng.normal(size=8);vectors.append(v/np.linalg.norm(v))
    probes=[mapping(raw)]; raw_records=[{"raw":raw,"M":probes[0],"reconstruction_error":float((probes[0]-I).abs().max())}]
    for v in vectors:
        desired=.06*I+7.52*torch.tensor(np.outer(v,v))
        lower=torch.linalg.cholesky(desired-.05*I)
        inverse=lower[mapping.rows,mapping.cols].clone()
        inverse[mapping.diagonal]=torch.log(torch.expm1(inverse[mapping.diagonal]-1e-8))
        actual=mapping(inverse)
        probes.append(actual)
        raw_records.append({"raw":inverse,"M":actual,"reconstruction_error":float((actual-desired).abs().max())})
    add("intervention_probes_are_finite_public_raw_outputs",all(torch.isfinite(r["raw"]).all() and r["reconstruction_error"]<1e-12 for r in raw_records))
    write("M_PROBE_RAW.json",raw_records)
    intervention=None
    score_env=new(fixtures[0])
    for fi,plan in enumerate(plans):
        baseline=np.array(plan["selected_command"])
        for mi,M in enumerate(probes):
            scores=score_env.candidate_scores(M,plan).detach().numpy()
            idx=int(np.argmin(scores)); command=np.array(plan["candidates"][idx]["first_command"])
            changed=not np.array_equal(command,baseline)
            coupling.append({"fixture":fixtures[fi]["name"],"probe_index":mi,"candidate_index":idx,"command":command,"changed_from_identity":changed})
            if changed and intervention is None: intervention=(fi,mi,M)
    add("M_changes_actual_first_command_in_predeclared_set",intervention is not None)
    if intervention is not None:
        fi,mi,M=intervention; changed=new(fixtures[fi]); public=ContactMFacade(changed)
        ee,cc=public.build_error(changed.state); action=public.calculate_action_from_M(M,ee,context=cc)
        changed.step(action.numpy())
        add("M_intervention_first_command_executed",not np.array_equal(action.numpy(),plans[fi]["selected_command"]))
        write("M_INTERVENTION_EXECUTED.json",{"fixture_index":fi,"probe_index":mi,"M":M,"audit":changed.get_episode_audit()})
    # Candidate-score gradient is terminal outer product, not action gradient.
    p=plans[0]; valid=[i for i,c in enumerate(p["candidates"]) if c["valid"]]
    if valid:
        k=valid[-1]; m=I.clone().requires_grad_(True)
        value=score_env.candidate_scores(m,p)[k]; grad,=torch.autograd.grad(value,m)
        end=np.array(p["candidates"][k]["terminal_error"])
        add("score_M_gradient_outer_product",np.max(abs(grad.numpy()-np.outer(end,end)))<1e-12)
        d=torch.zeros_like(I);d[0,1]=d[1,0]=1.;eps=1e-5
        fd=float((score_env.candidate_scores(I+eps*d,p)[k]-score_env.candidate_scores(I-eps*d,p)[k])/(2*eps))
        add("score_M_gradient_FD",abs(fd-float((grad*d).sum()))<1e-8)
        rr=raw.clone().requires_grad_(True); rawscore=score_env.candidate_scores(mapping(rr),p)[k]
        gr,=torch.autograd.grad(rawscore,rr)
        add("raw_to_score_graph",torch.isfinite(gr).all() and float(gr.abs().max())>0.)
    # Full simulator state and RNG boundary recovery; planning clocks are extrinsic diagnostics.
    saved=first_env.state_dict(); restored=ContactSingleDiskEnv();restored.load_state_dict(saved)
    add("integration_snapshot_roundtrip_exact",np.array_equal(first_env._integration(),restored._integration()))
    add("RNG_state_exact",_digest(first_env.np_random.bit_generator.state)==_digest(restored.np_random.bit_generator.state))
    next_actions=[]
    for current in (first_env,restored):
        ee,cc=current.build_error(current.state); facade=ContactMFacade(current)
        a=facade.calculate_action_from_M(I,ee,context=cc);next_actions.append(a)
        current.step(a.numpy())
    add("restored_next_action_exact",torch.equal(*next_actions))
    add("restored_next_integration_exact",np.array_equal(first_env._integration(),restored._integration()))
    write("BOUNDARY_SNAPSHOT.json",saved)
    # Exact target, deliberately stale warmstart, and strict-interior boundary target.
    for name,pusher,warm in (("target_warmstart",[-.03,0.],True),("target_near_joint_boundary",[.07-5e-6,0.],False)):
        target=new({**fixtures[-1],"pusher_xy":pusher})
        if warm: target.data.qacc_warmstart[:]=np.linspace(-3.,3.,5)
        target.data.ctrl[:]=0.; state=target.state.copy()
        add(name+"_error_exact_zero",torch.count_nonzero(target.tensor_error(state))==0)
        filtered=target._filter_command(np.zeros(2),target.data)
        add(name+"_zero_filter",np.array_equal(filtered,np.zeros(2)))
        for _ in range(3): target.step(np.zeros(2))
        add(name+"_zero_drive_stationary",np.array_equal(target.state,state))
        add(name+"_no_contact_force",target.data.ncon==0 and np.max(abs(target.data.qfrc_constraint))<1e-12 and np.max(abs(target.data.actuator_force))<1e-12)
        write(name+".json",target.get_episode_audit())
    contact=new({**fixtures[-1],"pusher_xy":[-.019,0.]})
    touching=False
    for _ in range(3):
        _,_,done,_,_=contact.step([.03,0.])
        touching=any(c["force_contact_frame"][0]>0. for row in contact._trace for sub in row["substeps"] for c in sub["contacts"])
        if touching or done: break
    add("actual_side_contact_positive_normal_force",touching)
    add("contact_probe_nonempty_solver_warmstart",np.linalg.norm(contact.data.qacc_warmstart)>0.)
    if not contact._done:
        contact_saved=contact.state_dict(); recovered=ContactSingleDiskEnv();recovered.load_state_dict(contact_saved)
        clone=copy(contact.data)
        clone_next,_,_,_=contact._advance(clone,[.03,0.]);counts["direct_clone_substeps"]+=4
        left=contact.step([.03,0.]);right=recovered.step([.03,0.])
        add("contact_clone_next_state_exact",np.array_equal(clone_next,left[0]))
        add("contact_restore_next_complete_integration_exact",np.array_equal(contact._integration(),recovered._integration()))
        write("NONEMPTY_CONTACT_SNAPSHOT.json",contact_saved)
        write("NONEMPTY_CONTACT_CONTINUATION.json",{"original":contact.get_episode_audit(),"restored":recovered.get_episode_audit()})
    else:
        add("contact_snapshot_boundary_safe",False,"Predeclared contact probe failed; no hidden retry")
        write("FAILED_CONTACT_PROBE.json",contact.get_episode_audit())
    add("CPU1",torch.get_num_threads()==1 and all(p["num_threads"]==1 for p in threadpool_info()))
    add("no_full_episodes",counts["full_episodes"]==0)
    add("source_hashes_unchanged",{name:sha(HERE/name) for name in sources}==sources)
    add("frozen_51_unchanged",{p:sha(PROJECT/p) for p in frozen}==frozen)
except Exception as exc:
    error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    write("ERROR.json",error)
finally:
    write("CHECKS.json",checks);write("M_COUPLING.json",coupling)
    durations=[r["plan"]["budget"]["planning_seconds"] for audit in records for r in audit["trace"] if r["plan"] and not r["plan"]["target_zero_shortcut"]]
    estimate=None if not durations else {"seconds_per_nontrivial_query":durations,
        "mean_seconds":float(np.mean(durations)),"max_seconds":max(durations),
        "four_nontrivial_full_fixture_minutes_mean":4*1500*float(np.mean(durations))/60,
        "five_fixture_worst_query_minutes":5*1500*max(durations)/60,
        "max_prediction_substeps_five_fixtures":67500000,
        "note":"Upper budget; invalid candidates may stop early and exact target guard skips prediction. Timing estimate is not training throughput acceptance."}
    result={"passed":error is None and all(c["passed"] for c in checks),"checks_passed":sum(c["passed"] for c in checks),"checks_total":len(checks),
        "counts":counts,"sources":sources,"error":error,"runtime_estimate":estimate,
        "full_fixture_execution_gate":"Await root runtime/resource decision; no full trajectory executed",
        "training_updates":0,"optimizers":0}
    write("SUMMARY.json",result)
    write("FINAL_MANIFEST.json",{p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()})
    print(json.dumps(result,default=ready),flush=True)
if not result["passed"]: raise SystemExit(1)
