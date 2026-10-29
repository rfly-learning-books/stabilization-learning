"""Predeclared bounded CPU engineering, never formal training/full episodes."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; PROJECT = HERE.parents[2]
OUT = HERE / "checks/short_v1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
for p in (PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control",
          PROJECT / "third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages"):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(HERE))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
frozen = json.loads((PROJECT / "runs/book_m_night_suite_20261023_v1/protocol.json").read_text(encoding="utf-8"))["sources"]
if {p: sha(PROJECT/p) for p in frozen} != frozen: raise RuntimeError("Frozen sources differ before engineering")
OUT.mkdir(parents=True, exist_ok=False)
sources = {p: sha(HERE/p) for p in ("DESIGN_PROTOCOL.json", "swarm_env.py", "check_short.py")}
(OUT / "START_MANIFEST.json").write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
    "sources": sources, "frozen": frozen, "physical_step_budget": 30, "planned_steps": 25,
    "full_episodes": 0, "optimizer_instances": 0}, indent=2)+"\n", encoding="utf-8")

import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_info, threadpool_limits
import swarm_env as mod
from swarm_env import SwarmAssignedStatic3Env, SwarmMFacade, PROTOCOL, plain, digest
from stabilization_learning.policies.metric import CholeskyMetric
from stabilization_gym.interfaces.m_learning import CapabilityError
torch.set_num_threads(1); torch.set_num_interop_threads(1)

checks, snapshots, static_qps, retained = [], {}, [], {}
counts = {"environment_constructions": 0, "QP_solver_calls": 0, "physical_steps": 0,
          "full_episodes": 0, "optimizer_instances": 0, "training_updates": 0}
native_init, native_solve, native_step = SwarmAssignedStatic3Env.__init__, mod.solve_qp, SwarmAssignedStatic3Env.step
def counted_init(self):
    if counts["environment_constructions"] >= 40: raise RuntimeError("construction budget")
    native_init(self); counts["environment_constructions"] += 1
def counted_solve(*args, **kw):
    if counts["QP_solver_calls"] >= 200: raise RuntimeError("solver budget")
    counts["QP_solver_calls"] += 1
    return native_solve(*args, **kw)
def counted_step(self, *args):
    if counts["physical_steps"] >= 30: raise RuntimeError("physical budget")
    result = native_step(self, *args); counts["physical_steps"] += 1
    if self._steps >= 600: counts["full_episodes"] += 1
    return result
SwarmAssignedStatic3Env.__init__ = counted_init
SwarmAssignedStatic3Env.step = counted_step
mod.solve_qp = counted_solve

def add(name, passed, detail=None): checks.append({"name": name, "passed": bool(passed), "detail": plain(detail)})
def rejects(fn, kind=ValueError):
    try: fn(); return False
    except kind: return True
def write(name, data):
    (OUT/name).write_text(json.dumps(plain(data), indent=2, allow_nan=False)+"\n", encoding="utf-8")
def create(options=None):
    env = SwarmAssignedStatic3Env(); obs, _ = env.reset(seed=41, options=options)
    return env, SwarmMFacade(env), obs
def perform(env, facade, M, *, dtype=torch.float64):
    e, ctx = env.build_error(torch.tensor(env.state, dtype=dtype))
    action = facade.calculate_action_from_M(M, e, context=ctx)
    return env.step(action.detach().numpy())


def independently_reconstruct(p, M, g):
    p = np.asarray(p); A = np.zeros((15, 6)); b = np.zeros(15)
    for j in range(6):
        A[j, j] = 1; b[j] = min(.4/np.sqrt(2), (1.9-p[j])/.05)
        A[j+6, j] = -1; b[j+6] = min(.4/np.sqrt(2), (p[j]+1.9)/.05)
    for index, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 12):
        for axis in range(2):
            delta = p[2*i+axis]-p[2*j+axis]
            A[index, 2*i+axis] = -2*delta; A[index, 2*j+axis] = 2*delta
        delta = p[2*i:2*i+2]-p[2*j:2*j+2]; b[index] = delta@delta-.09
    c = .5*np.asarray(M)@(p-np.asarray(g))
    return A, b, c


def independent_record(row, goal, M):
    p, pn, u = (np.asarray(row[k]) for k in ("state_before", "state_after", "action"))
    A, b, c = independently_reconstruct(p, M, goal)
    plan = row["plan"]; solver = plan["solver"]
    checks_local = [np.array_equal(pn, p+.05*u), np.allclose(A, plan["A"], rtol=0, atol=1e-15), np.allclose(b, plan["b"], rtol=0, atol=1e-15), np.array_equal(c, plan["c"])]
    if solver["ok"]:
        lam = np.asarray(solver["dual"])
        checks_local += [np.max(A@u-b) <= 1e-10, np.max(np.abs(u+c+A.T@lam)) <= 1e-9,
                         np.min(lam) >= -1e-10, np.max(np.abs(lam*(b-A@u))) <= 1e-9]
    else: checks_local += [np.array_equal(u, np.zeros(6)), row["failure"]]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        r, d = p[2*i:2*i+2]-p[2*j:2*j+2], u[2*i:2*i+2]-u[2*j:2*j+2]
        t = 0. if d@d == 0 else max(0., min(.05, float(-(r@d)/(d@d))))
        checks_local.append(np.linalg.norm(r+t*d) >= .3-1e-12)
    e, en = p-goal, pn-goal
    checks_local += [np.isclose(row["V_before"], e@M@e, rtol=1e-12, atol=1e-12),
                     np.isclose(row["V_after"], en@M@en, rtol=1e-12, atol=1e-12),
                     np.isclose(row["cost"], .05*(e@e), rtol=1e-12, atol=1e-12),
                     row["delta_V"] <= row["same_M_bound"]+1e-11]
    return bool(all(checks_local))


started = time.perf_counter(); error = None
try:
  with threadpool_limits(limits=1):
    env, facade, obs = create()
    desc = facade.describe().to_dict()
    add("facade_capability_truth", not desc["gradients"]["action_wrt_M"] and not desc["gradients"]["candidate_scores_wrt_certificate"] and desc["gradients"]["model_wrt_action"] and not desc["control_call_is_pure"] and not desc["supported_algorithms"])
    add("final_description_bound_by_signature", facade.source_signature_sha256 == digest({"description": desc, "sources": env._sources}))
    add("full_observation_space", env.observation_space.contains(obs))
    add("no_formal_training_claim", rejects(lambda: facade.require_algorithm("episode_metric_ppo"), CapabilityError))
    add("no_hard_QP_action_gradient", rejects(lambda: facade.control(torch.eye(6), facade.error_context(obs), training=True), CapabilityError))
    mapping = CholeskyMetric(env.metric_spec).double()
    rawI = mapping.identity_raw(dtype=torch.float64); I = mapping(rawI)
    target = torch.diag(torch.tensor([2., .8, .8, .8, .8, .8], dtype=torch.float64))
    L = torch.linalg.cholesky(target-.05*torch.eye(6, dtype=torch.float64))
    raw = L[mapping.rows, mapping.cols].clone(); diagonal = mapping.diagonal
    raw[diagonal] = torch.log(torch.expm1(raw[diagonal]-1e-8))
    anis = mapping(raw)
    add("public_identity_raw", torch.equal(I, torch.eye(6, dtype=torch.float64)))
    add("finite_public_anisotropic_raw", torch.isfinite(raw).all() and torch.max(torch.abs(anis-target)) < 1e-12)
    retained["public_metric_inputs"] = {"identity_raw": rawI.numpy(), "identity_M": I.numpy(), "anisotropic_raw": raw.numpy(), "anisotropic_M": anis.numpy()}
    sample = facade.error_context(obs)
    before = env.state_dict(); plan = env.plan(I, sample.error, context=sample.control_context)
    add("pure_plan_no_rng_or_state_mutation", digest(before) == digest(env.state_dict()))
    add("static_identity_expected_action", np.allclose(plan["action"], [-.1, 0, -.05, 0, .1, 0], atol=1e-14))
    bad = torch.eye(6, dtype=torch.float64); bad[0, 1] = 1e-9
    add("asymmetric_M_rejected", rejects(lambda: env.plan(bad, sample.error, context=sample.control_context)))
    add("invalid_M_rejected", rejects(lambda: env.plan(2*I, sample.error, context=sample.control_context)))
    add("nonfinite_M_rejected", rejects(lambda: env.plan(I*float("nan"), sample.error, context=sample.control_context)))
    old_q = env.q.copy(); env.q[0] += .001
    add("external_goal_mutation_rejected", rejects(lambda: env.build_error(obs))); env.q = old_q
    old_a = env.assignment.copy(); env.assignment[:] = [1, 0, 2]
    add("external_assignment_mutation_rejected", rejects(lambda: env.metric_context())); env.assignment = old_a
    add("step_without_audit_rejected", rejects(lambda: env.step(np.zeros(6)), RuntimeError))
    x = torch.tensor(env.state, dtype=torch.float64, requires_grad=True); u = torch.ones(6, dtype=torch.float64, requires_grad=True)
    dx, du = torch.autograd.grad(env.torch_step(x, u).sum(), (x, u))
    add("analytic_model_derivatives", torch.equal(dx, torch.ones(6, dtype=torch.float64)) and torch.equal(du, torch.full((6,), .05, dtype=torch.float64)))
    for label, options in (("overlap", {"p": [0,0,.1,0,0,1]}), ("outside", {"p": [-2,0,0,1,1,0]}), ("goal_boundary", {"q": [-1.9,0,0,1,1,0]}), ("goal_overlap", {"q": [0,0,.4,0,0,1]}), ("assignment_duplicate", {"assignment": [0,0,2]}), ("assignment_float", {"assignment": [0.,1.,2.]})):
        add("invalid_reset_"+label, rejects(lambda o=options: env.reset(seed=41, options=o)))
    # Check QP numerics on exactly 24 fixed-seed safe configurations, no physical step.
    rng = np.random.default_rng(731); statics_ok = []; reference_errors = []
    for index in range(24):
        for attempt in range(10000):
            p = rng.uniform(-1.5, 1.5, 6); g = rng.uniform(-1.4, 1.4, 6)
            if mod.geometry(p) and min(np.linalg.norm(p.reshape(3,2)[i]-p.reshape(3,2)[j]) for i,j in mod.PAIRS) >= .45 and mod.geometry(g, goal=True): break
        else: raise RuntimeError("static sampler exhausted")
        mr = rawI+torch.tensor(rng.normal(0., .6, 21), dtype=torch.float64); m = mapping(mr).numpy()
        A, b, c = independently_reconstruct(p, m, g)
        result = mod.solve_qp(A, b, c)
        reference = minimize(lambda z: .5*z@z+c@z, np.zeros(6), jac=lambda z:z+c,
            constraints={"type":"ineq", "fun":lambda z:b-A@z, "jac":lambda z:-A},
            method="SLSQP", options={"ftol":1e-12,"maxiter":200})
        ok = result["ok"]
        if ok:
            z, lam = result["action"], result["dual"]
            residuals = {"primal": float(max(0., np.max(A@z-b))), "stationarity": float(np.max(np.abs(z+c+A.T@lam))),
                         "dual_min": float(np.min(lam)), "complementarity": float(np.max(np.abs(lam*(b-A@z))))}
            objective_error = abs(.5*z@z+c@z-reference.fun)
            ok = ok and reference.success and residuals["primal"] <= 1e-10 and residuals["stationarity"] <= 1e-9 and residuals["dual_min"] >= -1e-10 and residuals["complementarity"] <= 1e-9 and objective_error <= 1e-8
        else: residuals = {}; objective_error = None
        statics_ok.append(ok); reference_errors.append(objective_error)
        static_qps.append({"index":index,"p":p,"g":g,"raw":mr.numpy(),"M":m,"A":A,"b":b,"c":c,"solution":result,"independent_residuals":residuals,"SLSQP_success":bool(reference.success),"SLSQP_action":reference.x,"SLSQP_objective":float(reference.fun),"objective_difference":objective_error})
    add("24_static_QP_KKT_and_independent_SLSQP", all(statics_ok), {"pass_count":sum(statics_ok),"objective_differences":reference_errors})
    # The declared physical fixtures: total 15 steps, no full episode.
    for fixture in PROTOCOL["engineering_fixtures"]:
        options = {k:fixture[k] for k in ("p","q","assignment")}; env, facade, _ = create(options)
        M = anis if fixture["name"] == "anisotropic_action_intervention" else I
        for _ in range(fixture["short_steps"]): perform(env, facade, M)
        snapshots[fixture["name"]] = env.state_dict()
        add(fixture["name"]+"_actual_control_state_KKT_V", all(independent_record(r, env.goal, M.numpy()) for r in env._trace))
        add(fixture["name"]+"_no_short_success_claim", not env._last_info["success"] and not env._done)
        if fixture["name"] == "ordinary_identity":
            ordinary_env = env; ordinary_facade = facade
            add("ordinary_unit_M_error_decreases", np.linalg.norm(env.state-env.goal) < np.linalg.norm(env._initial-env.goal))
            add("within_episode_M_change_rejected", rejects(lambda: facade.calculate_action_from_M(anis, *(), **{"e":facade.error_context(env._observation()).error,"context":facade.error_context(env._observation()).control_context})))
            add("stale_error_context_rejected", rejects(lambda: env.plan(I, sample.error, context=sample.control_context)))
        if fixture["name"] == "exact_target": add("exact_target_zero_and_invariant", all(np.array_equal(r["action"], np.zeros(6)) for r in env._trace) and np.array_equal(env.state, env._initial))
        if fixture["name"] == "known_non_target_deadlock":
            add("deadlock_preserved_non_target", np.max(np.abs(env.state-env._initial)) < 1e-14 and not env._last_info["in_goal"] and all(abs(r["V_after"]-1.125)<1e-14 and np.max(np.abs(r["action"]))<1e-14 for r in env._trace))
            add("deadlock_KKT_multiplier", abs(env._trace[0]["plan"]["solver"]["dual"][12]-.625)<1e-12)
        if fixture["name"] == "boundary_inward": add("boundary_no_position_projection", all(np.array_equal(r["state_after"],np.asarray(r["state_before"])+.05*np.asarray(r["action"])) for r in env._trace))
    i_action = np.asarray(snapshots["ordinary_identity"]["trace"][0]["action"])
    a_action = np.asarray(snapshots["anisotropic_action_intervention"]["trace"][0]["action"])
    add("public_M_changes_executed_action", np.linalg.norm(a_action-i_action)>.05,{"unit_action":i_action,"anisotropic_action":a_action})
    # Six simultaneous coordinate relabelings, one physical step apiece.
    ordinary = PROTOCOL["engineering_fixtures"][0]; p0=np.asarray(ordinary["p"]); q0=np.asarray(ordinary["q"])
    permutation_records=[]
    for permutation in itertools.permutations(range(3)):
        ix=np.array([[2*i,2*i+1] for i in permutation]).reshape(6)
        env,facade,_=create({"p":p0[ix],"q":q0,"assignment":list(permutation)})
        M=torch.tensor(anis.numpy()[np.ix_(ix,ix)],dtype=torch.float64)
        perform(env,facade,M)
        permutation_records.append({"permutation":list(permutation),"snapshot":env.state_dict()})
        add("permutation_"+"".join(map(str,permutation)),np.allclose(env._trace[0]["action"],a_action[ix],rtol=0,atol=2e-14) and independent_record(env._trace[0],env.goal,M.numpy()))
    snapshots["six_permutations"]=permutation_records
    # Two dtypes are checked without extra physical steps; the float32 goal/state
    # query is canonicalized to exact stored coordinates, never used as a model state.
    env,facade,_=create(); e64,c64=env.build_error(torch.tensor(env.state,dtype=torch.float64)); e32,c32=env.build_error(torch.tensor(env.state,dtype=torch.float32))
    m32=CholeskyMetric(env.metric_spec)(torch.tensor(raw.numpy(),dtype=torch.float32))
    p64=env.plan(m32.double(),e64,context=c64); p32=env.plan(m32,e32,context=c32)
    add("float32_promoted_query_identical", np.array_equal(p64["action"],p32["action"]))
    # Nonempty previous trajectory plus a pending exact plan, then 3 continuations.
    env=ordinary_env; facade=ordinary_facade
    e,ctx=env.build_error(env._observation()); action=facade.calculate_action_from_M(I,e,context=ctx)
    state=env.state_dict(); clone1,facade1,_=create(); clone2,facade2,_=create()
    clone1.load_state_dict(state); clone2.load_state_dict(json.loads(json.dumps(state)))
    add("nonempty_pending_snapshot_roundtrip", digest(state)==digest(clone1.state_dict())==digest(clone2.state_dict()))
    add("pending_blocks_duplicate_query", rejects(lambda: facade.calculate_action_from_M(I,e,context=ctx),RuntimeError))
    add("pending_action_mutation_rejected", rejects(lambda: env.step(action.numpy()+1e-6)))
    outcomes=[one.step(action.numpy()) for one in (env,clone1,clone2)]
    add("restored_next_step_exact", digest(env.state_dict())==digest(clone1.state_dict())==digest(clone2.state_dict()))
    add("restored_RNG_exact", digest(env.np_random.bit_generator.state)==digest(clone1.np_random.bit_generator.state)==digest(clone2.np_random.bit_generator.state))
    snapshots["recovery"]= {"pending_before":state,"original_after":env.state_dict(),"clone1_after":clone1.state_dict(),"clone2_after":clone2.state_dict()}
    damaged=deepcopy(state); damaged["state"][0]+=.01
    add("tampered_snapshot_rejected", rejects(lambda: clone1.load_state_dict(damaged)))
    # Exactly one deliberately failed solver call and one zero fallback physical step.
    env,facade,_=create(); original_solver=mod.solve_qp
    def forced_failure(A,b,c): return original_solver(A,b,c,max_iterations=0)
    mod.solve_qp=forced_failure
    try: result=perform(env,facade,I)
    finally: mod.solve_qp=original_solver
    add("solver_failure_zero_tail_and_termination", result[2] and not result[3] and result[4]["solver_failure"] and result[4]["failure"] and not result[4]["success"] and np.array_equal(env._trace[0]["action"],np.zeros(6)) and np.isclose(result[4]["failure_tail"],25*29.95,rtol=0,atol=1e-12))
    snapshots["solver_failure"]=env.state_dict()
    # Numeric exception handling is a query-only fault probe, no extra step.
    env,facade,_=create(); e,ctx=env.build_error(env._observation())
    def numeric_failure(A,b,c): raise np.linalg.LinAlgError("predeclared numeric exception probe")
    mod.solve_qp=numeric_failure
    try: failed_plan=env.plan(I,e,context=ctx)
    finally: mod.solve_qp=original_solver
    add("numeric_solver_exception_recorded", failed_plan["solver_failure"] and failed_plan["solver"]["reason"]=="numeric_solver_exception" and np.array_equal(failed_plan["action"],np.zeros(6)))
    retained["numeric_exception_plan"]=failed_plan
    add("budget_exactly_25_physical_steps",counts["physical_steps"]==25,counts)
    add("zero_full_episodes_and_training",counts["full_episodes"]==0 and counts["training_updates"]==0 and counts["optimizer_instances"]==0)
    add("frozen51_unchanged", {p:sha(PROJECT/p) for p in frozen}==frozen)
    add("local_sources_unchanged", {p:sha(HERE/p) for p in sources}==sources)
    retained["threadpools"]=threadpool_info()
except Exception as exc:
    error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    add("execution_exception",False,error)
finally:
    write("SAVED_EVIDENCE.json", {"snapshots":snapshots,"static_QPs":static_qps,"retained":retained})
    summary={"utc":datetime.now(timezone.utc).isoformat(),"scope":"short_CPU_engineering_only",
        "passed":bool(checks and all(c["passed"] for c in checks)),"checks_passed":sum(c["passed"] for c in checks),
        "checks_total":len(checks),"counts":counts,"sources":sources,"elapsed_seconds":time.perf_counter()-started,
        "error":error,"checks":checks,"limitations":["No full 600-step trajectory, success rate, training improvement, or sampled ROA established.","Known safe non-target deadlock intentionally retained.","QP finite-tolerance acceptance is not an exact-arithmetic safety proof.","Moving goals, online assignment, 3D dynamics and virtual pipe remain unimplemented.","No formal learner/optimizer/recovery integration acceptance; adapter supported_algorithms is empty."]}
    write("SUMMARY.json",summary)
    print(json.dumps({k:summary[k] for k in ("passed","checks_passed","checks_total","counts","elapsed_seconds","error")},ensure_ascii=False))
if error: raise SystemExit(1)
if not all(c["passed"] for c in checks): raise SystemExit(2)
