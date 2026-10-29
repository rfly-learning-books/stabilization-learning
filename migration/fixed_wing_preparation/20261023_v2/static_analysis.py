"""Predeclared root/linear-algebra/gradient checks. No physical integration."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,sys,time
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[2]
sys.dont_write_bytecode=True
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
for p in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control",PROJECT/"third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages"):
    sys.path.insert(0,str(p))
import numpy as np
import torch
from scipy.optimize import root
from scipy.linalg import expm
from threadpoolctl import threadpool_limits,threadpool_info
from stabilization_gym.common.types import MetricSpec
from stabilization_learning.policies.metric import CholeskyMetric
from longitudinal_model import P,S,U,CONTRACT,forces,rhs_numpy,rhs_torch,riccati_numpy,riccati_torch,physical_control
torch.set_num_threads(1); torch.set_num_interop_threads(1)
OUTPUT=HERE/"analysis"; OUTPUT.mkdir(exist_ok=False)
counts={"environment_instances":0,"integrated_physics_steps":0,"closed_loop_episodes":0,"training_updates":0,
        "optimizers":0,"trim_root_solves":0,"trim_residual_evaluations":0,"linearization_rhs_evaluations":0,
        "rhs_benchmark_evaluations":0,"riccati_syntheses":0,"installs":0}
checks=[]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(x):
    if hasattr(x,"tolist"): return x.tolist()
    raise TypeError(type(x).__name__)
def write(name,x): (OUTPUT/name).write_text(json.dumps(x,indent=2,default=ready,allow_nan=False)+"\n",encoding="utf-8")
def add(name,passed,detail=None): checks.append({"name":name,"passed":bool(passed),"detail":detail})
sources={str(p.relative_to(HERE)):sha(p) for p in [HERE/"DESIGN_CONTRACT.json",HERE/"PARAMETERS.json",HERE/"SOURCE_MANIFEST.json",HERE/"longitudinal_model.py",Path(__file__)]}
frozen=json.loads((PROJECT/"runs/book_m_night_suite_20261023_v1/protocol.json").read_text())["sources"]
add("frozen_51_before",{p:sha(PROJECT/p) for p in frozen}==frozen)
write("START_MANIFEST.json",{"utc":datetime.now(timezone.utc).isoformat(),"sources":sources,"protected_sources":frozen,"activity_limits":counts})

with threadpool_limits(limits=1):
    def residual(unknown):
        counts["trim_residual_evaluations"]+=1
        alpha,thrust,delta=unknown
        lift,drag,moment=forces(20.,alpha,0.,delta)
        return [(thrust*np.cos(alpha)-drag)/(P["mass"]*P["gravity"]),
                (lift+thrust*np.sin(alpha)-P["mass"]*P["gravity"])/(P["mass"]*P["gravity"]),moment/(P["mass"]*P["gravity"]*P["c"])]
    counts["trim_root_solves"]+=1
    solution=root(residual,[.1,8.,-.25],tol=1e-11)
    alpha,thrust,delta=solution.x
    trim=np.array([100.,20.,alpha,alpha,0.]); control=np.array([thrust,delta])
    ftrim=rhs_numpy(trim,control); counts["linearization_rhs_evaluations"]+=1
    lift,drag,moment=forces(20.,alpha,0.,delta)
    balance=np.array([thrust*np.cos(alpha)-drag,lift+thrust*np.sin(alpha)-P["mass"]*P["gravity"],moment])
    add("root_converged",solution.success,solution.message)
    add("trim_rhs_residual",np.max(np.abs(ftrim))<1e-10,ftrim)
    add("independent_force_moment_balance",np.max(np.abs(balance))<1e-9,balance)
    add("trim_inside_declared_envelope",0<thrust<40 and abs(delta)<np.deg2rad(25) and -.15<alpha<.2 and -.35<alpha<.35)
    x=torch.tensor(trim,dtype=torch.float64,requires_grad=True); u=torch.tensor(control,dtype=torch.float64,requires_grad=True)
    A= torch.autograd.functional.jacobian(lambda xx:rhs_torch(xx,u),x).detach().numpy()
    B= torch.autograd.functional.jacobian(lambda uu:rhs_torch(x,uu),u).detach().numpy()
    counts["linearization_rhs_evaluations"]+=2
    finiteA=np.empty((5,5)); finiteB=np.empty((5,2))
    for k in range(5):
        change=np.zeros(5); change[k]=S[k]*1e-5
        finiteA[:,k]=(rhs_numpy(trim+change,control)-rhs_numpy(trim-change,control))/(2*change[k])
    for k in range(2):
        change=np.zeros(2); change[k]=U[k]*1e-5
        finiteB[:,k]=(rhs_numpy(trim,control+change)-rhs_numpy(trim,control-change))/(2*change[k])
    counts["linearization_rhs_evaluations"]+=14
    add("A_autograd_vs_independent_centered_FD",np.max(np.abs(A-finiteA))<1e-7,float(np.max(np.abs(A-finiteA))))
    add("B_autograd_vs_independent_centered_FD",np.max(np.abs(B-finiteB))<1e-7,float(np.max(np.abs(B-finiteB))))
    As=A*S[None,:]/S[:,None]; Bs=B*U[None,:]/S[:,None]
    augment=np.zeros((7,7)); augment[:5,:5]=As; augment[:5,5:]=Bs
    discrete=expm(.05*augment); Ad=discrete[:5,:5]; Bd=discrete[:5,5:]
    controllability=np.concatenate([np.linalg.matrix_power(As,k)@Bs for k in range(5)],axis=1)
    control_discrete=np.concatenate([np.linalg.matrix_power(Ad,k)@Bd for k in range(5)],axis=1)
    sv=np.linalg.svd(controllability,compute_uv=False); svd=np.linalg.svd(control_discrete,compute_uv=False)
    add("continuous_controllability_rank5",np.linalg.matrix_rank(controllability)==5,sv)
    add("discrete_controllability_rank5",np.linalg.matrix_rank(control_discrete)==5,svd)
    add("height_continuous_input_row_zero",np.array_equal(B[0],np.zeros(2)),B[0])
    add("pitch_damping_negative",A[4,4]<0,float(A[4,4]))
    write("TRIM_LINEARIZATION.json",{"trim_state":trim,"trim_input":control,"alpha_deg":float(np.rad2deg(alpha)),
        "elevator_deg":float(np.rad2deg(delta)),"rhs_residual":ftrim,"force_moment_balance":balance,
        "lift_N":float(lift),"drag_N":float(drag),"thrust_horizontal_N":float(thrust*np.cos(alpha)),
        "thrust_normal_N":float(thrust*np.sin(alpha)),"A_SI":A,"B_SI":B,"A_scaled":As,"B_scaled":Bs,
        "Ad_ZOH":Ad,"Bd_ZOH":Bd,"state_scales":S,"input_scales":U,
        "continuous_open_loop_eigenvalues":[[float(v.real),float(v.imag)] for v in np.linalg.eigvals(As)],
        "controllability_singular_values_scaled":sv,"discrete_controllability_singular_values_scaled":svd,
        "A_FD_max_abs":float(np.max(np.abs(A-finiteA))),"B_FD_max_abs":float(np.max(np.abs(B-finiteB)))})
    metric=CholeskyMetric(MetricSpec(5,.05,5.)).double()
    raw_list=[metric.identity_raw(dtype=torch.float64).numpy()]
    rng=np.random.default_rng(41); normals=rng.normal(size=(5,5)); normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    for direction in np.vstack((np.eye(5),normals)):
        desired=.06*np.eye(5)+4.7*np.outer(direction,direction)
        lower=np.linalg.cholesky(desired-.05*np.eye(5))
        raw=lower[metric.rows.numpy(),metric.cols.numpy()]
        diagonal=metric.diagonal.numpy(); raw[diagonal]=np.log(np.expm1(raw[diagonal]-1e-8))
        raw_list.append(raw)
    metrics=[metric(torch.tensor(raw,dtype=torch.float64)).detach().numpy() for raw in raw_list]
    probes=np.vstack((.01*np.eye(5),-.01*np.eye(5),[.02,-.01,.015,-.01,.005]))
    records=[]; baselines=None
    for mi,M in enumerate(metrics):
        K,P0=riccati_numpy(M,Ad,Bd); counts["riccati_syntheses"]+=1
        commands=np.array([np.clip(control-U*(K@e),[0,-np.deg2rad(25)],[40,np.deg2rad(25)]) for e in probes])
        if mi==0: baselines=commands
        closed=Ad-Bd@K
        decrement=closed.T@M@closed-M
        eigen=np.linalg.eigvalsh(M)
        record={"probe_index":mi,"raw":raw_list[mi],"M":M,"M_eigenvalues":eigen,"K":K,"P0_not_M":P0,
            "commands":commands,"commands_max_change_vs_identity":float(np.max(np.abs(commands-baselines))),
            "linear_closed_spectral_radius":float(np.max(np.abs(np.linalg.eigvals(closed)))),
            "same_M_one_step_decrement_eigenvalues":np.linalg.eigvalsh(decrement),
            "same_M_strict_discrete_decrease_all_directions":bool(np.max(np.linalg.eigvalsh(decrement))<0),
            "P0_equals_M":bool(np.allclose(P0,M)),"saturation_count":int(sum(np.any(commands[i]!=(control-U*(K@e))) for i,e in enumerate(probes)))}
        records.append(record)
        add(f"M_probe_{mi}_public_map_bound_trace",eigen[0]>=.05-1e-12 and eigen[-1]<=4.8+1e-12 and abs(np.trace(M)-5)<1e-11)
    add("some_actual_M_changes_saturated_physical_action",any(r["commands_max_change_vs_identity"]>1e-6 for r in records[1:]))
    raw=metric.identity_raw(dtype=torch.float64).requires_grad_()
    At=torch.tensor(Ad,dtype=torch.float64); Bt=torch.tensor(Bd,dtype=torch.float64)
    Kt,Pt=riccati_torch(metric(raw),At,Bt); counts["riccati_syntheses"]+=1
    e=torch.tensor(probes[-1],dtype=torch.float64)
    normalized_first_action=-(Kt@e)
    analytic=torch.autograd.grad(normalized_first_action[1],raw)[0].detach().numpy()
    finite=np.zeros(15)
    for k in range(15):
        plus=raw.detach().numpy().copy(); minus=plus.copy(); plus[k]+=1e-5; minus[k]-=1e-5
        kp,_=riccati_numpy(metric(torch.tensor(plus)).detach().numpy(),Ad,Bd)
        km,_=riccati_numpy(metric(torch.tensor(minus)).detach().numpy(),Ad,Bd)
        counts["riccati_syntheses"]+=2
        finite[k]=(-(kp@probes[-1])[1]+(km@probes[-1])[1])/2e-5
    add("M_to_Riccati_action_gradient_nonzero",np.linalg.norm(analytic)>1e-8,float(np.linalg.norm(analytic)))
    add("M_to_Riccati_action_FD",np.max(np.abs(analytic-finite))<1e-7,float(np.max(np.abs(analytic-finite))))
    add("torch_numpy_gain_agree",np.max(np.abs(Kt.detach().numpy()-records[0]["K"]))<1e-10)
    write("M_COUPLING.json",{"probes":records,"error_probes":probes,"autograd_raw":analytic,"finite_difference_raw":finite,
        "observed_same_M_decrease_pass_count":sum(r["same_M_strict_discrete_decrease_all_directions"] for r in records),
        "note":"Candidate V is the same M used to synthesize K; P0 is separate. Local poles do not establish same-M decrease or nonlinear stability."})
    delays=[]
    for n in CONTRACT["predeclared_static_probes"]["delay_steps"]:
        G=sum((np.linalg.matrix_power(Ad,j)@Bd for j in range(n)),start=np.zeros_like(Bd))
        Kshort,_=riccati_numpy(metrics[0],Ad,Bd,N=n); counts["riccati_syntheses"]+=1
        delays.append({"steps":n,"seconds":n*.05,"height_normalized_constant_input_sensitivity":G[0],
                       "SI_height_per_N_and_rad":S[0]*G[0]/U,"pure_altitude_feedback_normalized":-Kshort[:,0]})
    write("HEIGHT_DELAY.json",{"continuous_B_height_row":B[0],"constant_input_sensitivity":delays,
        "note":"Matrix powers/expm only; no integration or open/closed-loop trajectory was executed. Sign changes reveal inverse response and invalidate an instantaneous altitude-only control argument."})
    times=[]
    for _ in range(20):
        start=time.perf_counter(); riccati_numpy(metrics[0],Ad,Bd); times.append(time.perf_counter()-start); counts["riccati_syntheses"]+=1
    start=time.perf_counter()
    for _ in range(1000): rhs_numpy(trim,control)
    rhs_seconds=(time.perf_counter()-start)/1000; counts["rhs_benchmark_evaluations"]+=1000
    full_evals=5*600*10*4
    write("COMPUTE_BUDGET.json",{"Riccati_100_recursions_seconds":times,"Riccati_median_seconds":float(np.median(times)),
        "single_rhs_mean_seconds":rhs_seconds,"proposed_five_fixtures_control_steps":3000,
        "proposed_RK4_substeps":30000,"proposed_rhs_evaluations":full_evals,
        "raw_rhs_time_lower_bound_seconds":rhs_seconds*full_evals,
        "warning":"Microbenchmark is a lower bound excluding environment/audit/I/O and Python overhead. Engineering launcher must first measure bounded stepping, then obtain runtime gate before full fixtures.",
        "N_recursions_per_episode":100,"per_step_feedback":"cached 2x5 matrix-vector multiply plus clipping",
        "threadpools":threadpool_info(),"no_prediction_rollouts":True})
add("frozen_51_after",{p:sha(PROJECT/p) for p in frozen}==frozen)
add("local_sources_unchanged",{p:sha(HERE/p) for p in sources}==sources)
write("CHECKS.json",{"passed":all(c["passed"] for c in checks),"passed_count":sum(c["passed"] for c in checks),
    "total_count":len(checks),"checks":checks,"activity_counts":counts,"sources":sources,
    "activity_note":"RHS algebra, root solving, finite differences, matrix exponentials and Riccati recursion only. No state propagation, environment construction, trajectory, learner or optimizer."})
print(json.dumps({"checks_passed":sum(c["passed"] for c in checks),"checks_total":len(checks),"trim":trim.tolist(),"input":control.tolist(),
    "same_M_decrease_count":sum(r["same_M_strict_discrete_decrease_all_directions"] for r in records),
    "Riccati_median_s":float(np.median(times)),"rhs_mean_s":rhs_seconds,"activity_counts":counts}))
