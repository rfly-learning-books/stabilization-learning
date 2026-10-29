"""CPU-only isolated water-tank engineering, no learner/optimizer creation.

Four predeclared deterministic full episodes plus static/one-step probes.
Fresh evidence only; do not overwrite evidence or write outside this folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PROTOCOL = PROJECT/"runs/book_m_night_suite_20261023_v1/protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(value):
    if hasattr(value,"tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    output = parser.parse_args().output.resolve()
    if output.exists() or not output.is_relative_to(HERE):
        raise ValueError("Use a fresh output directory below migration/next_envs_v1/water_tank")
    frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))["sources"]
    protocol_sha = sha(PROTOCOL)
    if {p:sha(PROJECT/p) for p in frozen} != frozen:
        raise RuntimeError("Frozen source hashes mismatch; stop before importing packages")
    preparation = json.loads((PROJECT/"migration/water_tank_preparation/20261023_v1/MODEL_ANALYSIS.json").read_text(encoding="utf-8"))
    for item in preparation["sources"]:
        if sha(item["path"]) != item["sha256"]:
            raise RuntimeError("Book/shared metric source no longer matches preparation")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    for p in (PROJECT/"stabilization-gym",PROJECT/"stabilization-learning-control",
              PROJECT/"third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages"):
        sys.path.insert(0,str(p))
    import numpy as np
    import scipy
    from scipy.integrate import solve_ivp, quad
    import torch
    from threadpoolctl import threadpool_info,threadpool_limits
    from gymnasium.utils.env_checker import check_env
    from gymnasium.envs.registration import registry
    from stabilization_gym.interfaces.m_learning import PortableMFacade,CapabilityError
    from stabilization_learning.policies.metric import CholeskyMetric
    from stabilization_learning.wrappers import rollout_metric
    from water_tank_env import WaterTankIntegralEnv
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    checks,evidence,episodes = [],{},[]
    def add(name,passed,detail=None):
        checks.append({"name":name,"passed":bool(passed),"detail":detail})
    def rejects(fn,kind=ValueError):
        try:
            fn()
            return False
        except kind:
            return True
    def write(name,data):
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=ready,allow_nan=False)+"\n",encoding="utf-8")
    def execute_one(M,state,target=.49):
        env = WaterTankIntegralEnv(target_height=target)
        observation,_ = env.reset(seed=41,options={"state":state})
        e,ctx = env.build_error(observation)
        action = env.calculate_action_from_M(M,e,context=ctx)
        env.step(action.detach().numpy())
        return env.get_episode_audit()
    def run_episode(name,kind):
        env = WaterTankIntegralEnv(target_height=.49)
        state = [.49,0.] if kind=="late_pulse" else [.2,0.]
        obs,_ = env.reset(seed=41,options={"state":state})
        if kind=="unit":
            facade = PortableMFacade(env,name="isolated_water_tank")
            public = rollout_metric(facade,torch.eye(2,dtype=torch.float64),obs)
            result = env.get_episode_audit()
            result["public_rollout_result"] = public
        else:
            for i in range(env.max_steps):
                if kind=="feedforward":
                    action = env.fixed_feedforward_action().numpy()
                elif kind=="negative":
                    e,ctx = env.build_error(obs)
                    action = env.calculate_action_from_M(torch.tensor([[1.,-.5],[-.5,1.]],dtype=torch.float64),e,context=ctx).detach().numpy()
                else:
                    action = [.7 if i<1160 else 0.]
                obs,_,terminated,truncated,info = env.step(action)
                if i<1199 and not info["failure"]:
                    assert not terminated and not truncated and not info["success"]
                if terminated or truncated:
                    break
            result = env.get_episode_audit()
        result["engineering_case"] = name
        result["not_a_training_or_performance_benchmark"] = True
        episodes.append(result)
        write(f"episode_{len(episodes):02d}.json",result)
        return result

    with threadpool_limits(limits=1):
        pools = threadpool_info()
        add("cpu_single_thread",torch.get_num_threads()==1 and all(p["num_threads"]==1 for p in pools),pools)
        env = WaterTankIntegralEnv(target_height=.64)
        obs,initial = env.reset(seed=41,options={"state":[.5,.2]})
        e,ctx = env.build_error(obs)
        unit = torch.eye(2,dtype=torch.float64)
        negative = torch.tensor([[1.,-.5],[-.5,1.]],dtype=torch.float64)
        alternate = torch.tensor([[1.3,-.3],[-.3,.7]],dtype=torch.float64)
        add("new_nominal_protocol_exact_values",env.dt==.05 and env.max_steps==1200 and env.required_hold_seconds==2.
            and env.area_m2==1. and env.ku==env.ko==env.k_per_s==.1 and env.integral_time_scale_s==10.
            and env.height_tolerance_m==.01 and env.memory_tolerance_m_s==.1 and env.metric_spec.raw_dim==3)
        add("public_metric_bounds_unchanged",env.metric_spec.eigen_floor==.05 and env.metric_spec.trace==2. and env.metric_spec.eigen_ceiling==1.95)
        add("state_error_and_target_condition",np.allclose(e.numpy(),[-.14,.02]) and env.metric_context().tolist()==[1.])
        action_negative = env.calculate_action_from_M(negative,e,context=ctx)
        action_alternate = env.calculate_action_from_M(alternate,e,context=ctx)
        action_unit = env.calculate_action_from_M(unit,e,context=ctx)
        candidate_actions = [float(action_negative),float(action_alternate),float(action_unit)]
        add("prepared_same_state_actions_reproduced",np.max(abs(np.array(candidate_actions)-[.9421067811865477,.8937340001214589,.8671067811865476]))<1e-13,candidate_actions)
        add("M_changes_actual_action",min(abs(candidate_actions[0]-candidate_actions[1]),abs(candidate_actions[0]-candidate_actions[2]))>1e-3)
        add("control_queries_preserve_state_and_time",np.array_equal(obs,env.state) and env._elapsed_steps==0)

        m = alternate.clone().requires_grad_()
        error = e.clone().requires_grad_()
        a = env.calculate_action_from_M(m,error,context=ctx)
        gm,ge = torch.autograd.grad(a.sum(),(m,error))
        eps=1e-6
        matrix_dirs = [torch.diag(torch.tensor([1.,-1.],dtype=torch.float64)),torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)]
        m_errors=[]
        for direction in matrix_dirs:
            plus = env.calculate_action_from_M(m.detach()+eps*direction,error.detach(),context=ctx)
            minus = env.calculate_action_from_M(m.detach()-eps*direction,error.detach(),context=ctx)
            m_errors.append(abs(float((plus-minus)/(2*eps))-float((gm*direction).sum())))
        e_errors=[]
        for direction in torch.eye(2,dtype=torch.float64):
            plus=env.calculate_action_from_M(m.detach(),error.detach()+eps*direction,context=ctx)
            minus=env.calculate_action_from_M(m.detach(),error.detach()-eps*direction,context=ctx)
            e_errors.append(abs(float((plus-minus)/(2*eps))-float((ge*direction).sum())))
        add("symmetric_trace_preserving_M_gradient",max(m_errors)<1e-7,m_errors)
        add("error_gradient_includes_current_height_feedforward",max(e_errors)<1e-7,e_errors)
        construct = CholeskyMetric(env.metric_spec).double()
        raw=(construct.identity_raw(dtype=torch.float64)+torch.tensor([.1,-.2,-.1],dtype=torch.float64)).requires_grad_()
        raw_action=env.calculate_action_from_M(construct(raw),e,context=ctx)
        grad_raw,=torch.autograd.grad(raw_action.sum(),raw)
        raw_errors=[]
        for direction in torch.eye(3,dtype=torch.float64):
            plus=env.calculate_action_from_M(construct(raw.detach()+eps*direction),e,context=ctx)
            minus=env.calculate_action_from_M(construct(raw.detach()-eps*direction),e,context=ctx)
            raw_errors.append(abs(float((plus-minus)/(2*eps))-float((grad_raw*direction).sum())))
        add("public_Cholesky_raw_gradient_nonzero",float(grad_raw.abs().max())>1e-7 and max(raw_errors)<1e-7,
            {"gradient":grad_raw.tolist(),"finite_difference_error":raw_errors})
        state_grad=torch.tensor([.5,.2],dtype=torch.float64,requires_grad=True)
        err,_=env.build_error(state_grad)
        j=torch.autograd.functional.jacobian(lambda x:env.build_error(x)[0],state_grad)
        add("state_to_error_gradient",torch.allclose(j,torch.diag(torch.tensor([1.,.1],dtype=torch.float64))))
        batched=torch.tensor([[.5,.2],[.6,-.2]],dtype=torch.float64)
        eb,cb=env.build_error(batched)
        ab=env.calculate_action_from_M(torch.stack([negative,alternate]),eb,context=cb)
        separate=[]
        for row,metric in zip(batched,[negative,alternate]):
            er,cr=env.build_error(row)
            separate.append(env.calculate_action_from_M(metric,er,context=cr))
        add("batched_control_uses_same_reference",torch.allclose(ab,torch.stack(separate)) and env.metric_context(batched).shape==(2,1))
        add("mismatched_reference_rejected",rejects(lambda:env.build_error(obs,goal=[.49,0.])) and rejects(lambda:env.metric_context(goal=[.49,0.])))
        negative_context=dict(ctx,target_height=.49)
        add("stale_control_context_rejected",rejects(lambda:env.calculate_action_from_M(negative,e,context=negative_context)))
        add("no_autograd_model_claim",not env.controller_capabilities["model_gradient"] and rejects(lambda:env.torch_step(torch.tensor(obs),torch.tensor([.7])),NotImplementedError))

        physical_probes=[]
        for h0,u in [(0.,.7),(.0001,.9),(.01,.2),(.4,.6),(.49,.7),(.8,.2),(1.,1.),(.9,.0001)]:
            state=np.array([h0,.3])
            got,diag=env.integrate_zoh(state,u)
            negative_stages=[]
            def ode(t,s):
                if s[0]<0.:
                    negative_stages.append(float(s[0]))
                return [.1*u-.1*math_sqrt(max(s[0],0.)),env.target_height-s[0]]
            math_sqrt=np.sqrt
            independent=solve_ivp(ode,[0.,.05],state,method="DOP853",rtol=2e-12,atol=2e-14,max_step=.001)
            assert independent.success
            discrepancy=float(np.max(abs(got-independent.y[:,-1])))
            physical_probes.append({"h0":h0,"u":u,"exact_flow":got.tolist(),"independent_ode":independent.y[:,-1].tolist(),
                "max_difference":discrepancy,"internal_negative_ode_stages":negative_stages,"integration":diag})
        add("nonlinear_flow_and_memory_match_independent_ODE",max(x["max_difference"] for x in physical_probes)<2e-10,physical_probes)
        dry=[]
        for h0 in [0.,1e-6,1e-4,.4]:
            state=[h0,.2]
            got,diag=env.integrate_zoh(state,0.)
            integral=quad(lambda t:max(np.sqrt(h0)-.05*t,0.)**2,0.,.05,epsabs=1e-14,epsrel=1e-12,
                          points=[min(.05,2*np.sqrt(h0)/.1)])[0]
            expected_h=max(np.sqrt(h0)-.05*.05,0.)**2
            dry.append({"h0":h0,"after":got.tolist(),"height_error":abs(got[0]-expected_h),
                        "memory_error":abs(got[1]-(.2+.64*.05-integral)),"integration":diag})
        add("zero_input_exact_drying_and_memory",max(max(d["height_error"],d["memory_error"]) for d in dry)<1e-13,dry)
        upper=[]
        for h0 in [0.,1.]:
            for u in [0.,1.]:
                got,diag=env.integrate_zoh([h0,1000.],u)
                upper.append(got.tolist())
        add("physical_height_boundaries_invariant_no_memory_clip",all(0.<=s[0]<=1. for s in upper) and all(abs(s[1])>999. for s in upper),upper)
        equilibrium=[]
        for target in [.25,.49,.64]:
            qenv=WaterTankIntegralEnv(target_height=target)
            o,_=qenv.reset(seed=41,options={"state":[target,0.]})
            er,cr=qenv.build_error(o)
            ac=qenv.calculate_action_from_M(negative,er,context=cr)
            after=qenv.step(ac.detach().numpy())[0]
            equilibrium.append(float(np.max(abs(after-o))))
        add("zero_error_equilibrium_for_all_goal_extremes",max(equilibrium)<1e-13,equilibrium)
        seen=[]
        for seed in [41,42,43]:
            aenv,benv=WaterTankIntegralEnv(),WaterTankIntegralEnv()
            oa,_=aenv.reset(seed=seed); ob,_=benv.reset(seed=seed)
            sa=aenv.step([.7])[0]; sb=benv.step([.7])[0]
            seen.append({"seed":seed,"same":np.array_equal(oa,ob) and np.array_equal(sa,sb) and aenv.target_height==benv.target_height,
                "initial_state":oa.tolist(),"target":aenv.target_height})
        add("seed_and_default_zero_memory",all(v["same"] and v["initial_state"][1]==0. and .1<=v["initial_state"][0]<=.9 for v in seen),seen)
        robust=WaterTankIntegralEnv(target_height=.49)
        _,rinfo=robust.reset(options={"state":[.49,-30.]})
        add("nonzero_memory_not_physical_failure",not rinfo["failure"] and rinfo["physical_in_goal"] and not rinfo["in_goal"])
        add("bad_physical_inputs_rejected",rejects(lambda:robust.reset(options={"state":[-.1,0.]}))
            and rejects(lambda:robust.reset(options={"state":[1.1,0.]}))
            and rejects(lambda:robust.step([float("nan")]))
            and rejects(lambda:WaterTankIntegralEnv(target_height=.8)))
        obs,_=env.reset(options={"state":[.5,.2]}); e,ctx=env.build_error(obs)
        add("invalid_M_rejected",rejects(lambda:env.calculate_action_from_M(torch.zeros((2,2),dtype=torch.float64),e,context=ctx))
            and rejects(lambda:env.calculate_action_from_M(torch.tensor([[1.,2.],[2.,1.]],dtype=torch.float64),e,context=ctx)))

        # Analytic identity, signed-domain limits and actual nonlinear one-step differences.
        en=np.array([-.14,.02]); mn=negative.numpy(); s=float(mn[0]@en); det=float(np.linalg.det(mn))
        nominal_dot=-2*.1*s*s/mn[0,0]+2*det*mn[0,1]*en[1]**2/(10*mn[0,0]**2)
        u=float(env.calculate_action_from_M(negative,e,context=ctx))
        field=np.array([.1*u-.1*np.sqrt(.5),-.14/-10])
        direct_dot=float(2*en@mn@field)
        add("continuous_negative_cross_identity",abs(nominal_dot-direct_dot)<1e-14 and nominal_dot<0.,{"formula":nominal_dot,"direct":direct_dot})
        actual_negative=execute_one(negative,[.5,.2],.64)
        delta=actual_negative["trace"][0]["V_after"]-actual_negative["trace"][0]["V_before"]
        add("same_M_V_actual_step_recomputed",abs(actual_negative["trace"][0]["actual_Vdot_at_sample"]-nominal_dot)<1e-13 and delta<0.,actual_negative["trace"][0])
        positive=torch.tensor([[1.,.5],[.5,1.]],dtype=torch.float64)
        positive_case=execute_one(positive,[.44,1.])
        pstep=positive_case["trace"][0]
        add("positive_cross_V_increase_counterexample_retained",pstep["actual_Vdot_at_sample"]>0 and pstep["V_after"]>pstep["V_before"],pstep)
        saturated_case=execute_one(negative,[.95,-30.],.25)
        sat=saturated_case["trace"][0]
        add("negative_cross_saturation_V_increase_retained",sat["action"]==[0.] and sat["actual_Vdot_at_sample"]>0 and sat["V_after"]>sat["V_before"],sat)
        senv=WaterTankIntegralEnv(target_height=.25); so,_=senv.reset(options={"state":[.95,-30.]})
        se,sc=senv.build_error(so); se=se.detach().requires_grad_()
        sm=negative.clone().requires_grad_(); sa=senv.calculate_action_from_M(sm,se,context=sc)
        sg=torch.autograd.grad(sa.sum(),(sm,se))
        add("saturation_gradients_zero",all(torch.count_nonzero(g)==0 for g in sg))
        c=(np.sqrt(5.)-1)/2
        degenerate=execute_one(torch.tensor([[1.,c],[c,1.]],dtype=torch.float64),[.49,1.])
        ds=degenerate["trace"][0]
        add("allowed_SPD_non_target_equilibrium_counterexample",np.max(abs(np.array(ds["state_after"])-[.49,1.]))<1e-13 and ds["physical_in_goal"] and not ds["in_goal"],ds)
        evidence["counterexample_probes"]={"positive_cross":positive_case,"saturation":saturated_case,"degenerate":degenerate}
        write("COUNTEREXAMPLE_STEPS.json",evidence["counterexample_probes"])

        freezing=WaterTankIntegralEnv(target_height=.49)
        fo,_=freezing.reset(options={"state":[.4,0.]}); fe,fc=freezing.build_error(fo)
        fa=freezing.calculate_action_from_M(unit,fe,context=fc)
        freezing.step(fa.numpy()); fe,fc=freezing.build_error(freezing.state)
        changed=freezing.calculate_action_from_M(negative,fe,context=fc)
        add("executed_M_change_rejected",rejects(lambda:freezing.step(changed.numpy())))
        stale=WaterTankIntegralEnv(target_height=.49); stale.reset(options={"state":[.4,0.]})
        st_e,st_c=stale.build_error([.45,0.]); st_a=stale.calculate_action_from_M(unit,st_e,context=st_c)
        add("pending_wrong_state_rejected",rejects(lambda:stale.step(st_a.numpy())))
        tamper=WaterTankIntegralEnv(target_height=.49); to,_=tamper.reset(options={"state":[.4,0.]})
        te,tc=tamper.build_error(to); ta=tamper.calculate_action_from_M(unit,te,context=tc)
        add("modified_pending_action_rejected",rejects(lambda:tamper.step(ta.numpy()+.01)))
        external=WaterTankIntegralEnv(target_height=.49); external.reset(options={"state":[.4,0.]})
        result=external.step([5.])
        add("physical_action_saturation_only",result[-1]["applied_action"].tolist()==[1.] and result[-1]["submitted_action"].tolist()==[5.])
        facade=PortableMFacade(WaterTankIntegralEnv(target_height=.49),name="isolated_water_tank")
        add("actual_public_facade",facade.describe().gradients.action_wrt_M and not facade.describe().gradients.model_wrt_action
            and not facade.describe().control_call_is_pure and facade.describe().raw_metric_dim==3)
        add("facade_refuses_unavailable_model_gradient",rejects(lambda:facade.require_capability("model_wrt_action"),CapabilityError))
        registry_before=set(registry)
        check_env(WaterTankIntegralEnv(),skip_render_check=True)
        add("gymnasium_env_checker",True)
        add("new_environment_unregistered",set(registry)==registry_before and not any("water_tank" in str(v.entry_point).lower() for v in registry.values()))

        # Four fixed cases were chosen before any trajectory was observed.
        run_episode("unit_M_public_rollout", "unit")
        run_episode("fixed_negative_cross_M", "negative")
        run_episode("fixed_equilibrium_feedforward", "feedforward")
        run_episode("external_late_drain_after_early_hold", "late_pulse")
        add("exactly_four_full_engineering_episodes_no_optimizer",len(episodes)==4 and all(len(r["trace"])==1200 for r in episodes),
            {"episodes":len(episodes),"steps":sum(len(r["trace"]) for r in episodes)})
        hold_rechecks=[]
        for record in episodes:
            states=np.array([record["initial_state"]]+[t["state_after"] for t in record["trace"]])
            flags_h=abs(states[:,0]-record["goal"][0])<=.01
            flags_a=flags_h & (abs(states[:,1])<=.1)
            final_h,final_a=bool(flags_h[1160:1201].all()),bool(flags_a[1160:1201].all())
            chain=np.max(abs(states[:-1]-np.array([t["state_before"] for t in record["trace"]])))
            cost_expected=.05*np.sum((states[:-1,0]-record["goal"][0])**2)
            actual_cost=sum(t["cost"] for t in record["trace"])
            memory_error=max(abs((t["state_after"][1]-t["state_before"][1])-(.05*record["goal"][0]-t["integration"]["integral_height_m_s"])) for t in record["trace"])
            rise_count=None
            if record["metric"] is not None:
                mm=np.array(record["metric"]); ee=(states-np.array(record["goal"]))/[1.,10.]
                vv=np.einsum("ni,ij,nj->n",ee,mm,ee)
                assert np.max(abs(vv[:-1]-[t["V_before"] for t in record["trace"]]))<1e-12
                assert np.max(abs(vv[1:]-[t["V_after"] for t in record["trace"]]))<1e-12
                rise_count=int((np.diff(vv)>1e-8).sum())
            else:
                assert all(t["V_before"] is None and t["V_after"] is None for t in record["trace"])
            hold_rechecks.append({"case":record["engineering_case"],"final_41_state_physical":final_h,"final_41_state_augmented":final_a,
                "agrees":final_h==record["physical_only_success"] and final_a==record["success"],
                "state_chain_error":float(chain),"left_integral_cost_error":float(abs(cost_expected-actual_cost)),
                "memory_integral_residual":memory_error,"sampled_V_rises":rise_count,
                "height_final_m":float(states[-1,0]),"memory_final_m_s":float(states[-1,1]),
                "height_only_cost":float(actual_cost),"augmented_arrival":record["arrival_time"],"physical_arrival":record["physical_arrival_time"]})
        add("independent_final_41_state_hold_and_record_chain",all(x["agrees"] and x["state_chain_error"]<1e-13 for x in hold_rechecks),hold_rechecks)
        add("independent_cost_memory_and_candidate_V_reconstruction",all(x["left_integral_cost_error"]<1e-11 and x["memory_integral_residual"]<1e-13 for x in hold_rechecks))
        late=episodes[3]
        add("early_hold_not_early_termination",late["first_hold_confirmation_time"]["augmented"]==2. and len(late["trace"])==1200 and not any(t["success"] for t in late["trace"][:-1]))
        add("late_pulse_clears_success_and_truncates",not late["success"] and not late["physical_only_success"] and late["hold_seconds"]==0. and late["physical_truncated"])
        ff=episodes[2]
        add("feedforward_physical_vs_augmented_separate",ff["physical_only_success"] and not ff["success"] and ff["metric"] is None)
        public=episodes[0]["public_rollout_result"]
        evidence["generic_rollout_arrival_limitation"]={"generic_arrival_time":public["arrival_time"],"environment_augmented_arrival":episodes[0]["arrival_time"],
            "environment_physical_arrival":episodes[0]["physical_arrival_time"],
            "explanation":"Generic rollout always uses full episode duration for success, so it cannot report final-settling onset. Environment audit is authoritative. Failed augmented case is horizon-capped even when physical-only holds."}
        add("generic_full_episode_rollout_compatible_not_learner_acceptance",public["physical_steps"]==1200 and public["success"]==episodes[0]["success"])
        add("all_51_frozen_sources_unchanged",{p:sha(PROJECT/p) for p in frozen}==frozen)
        add("active_protocol_unchanged",sha(PROTOCOL)==protocol_sha)
        add("book_and_original_preparation_sources_unchanged",all(sha(x["path"])==x["sha256"] for x in preparation["sources"]))
        evidence["episode_summaries"] = hold_rechecks
        evidence["module_origins"]={"StateMetricEnv":str(PROJECT/"stabilization-gym/stabilization_gym/envs/base.py"),
            "PortableMFacade":str(PROJECT/"stabilization-gym/stabilization_gym/interfaces/m_learning.py"),
            "CholeskyMetric":str(PROJECT/"stabilization-learning-control/stabilization_learning/policies/metric.py")}
    audit={"schema":"water-tank-isolated-engineering-v1","completed_utc":datetime.now(timezone.utc).isoformat(),
        "passed":all(c["passed"] for c in checks),"check_count":len(checks),"checks":checks,"evidence":evidence,
        "scope":"Engineering only; no learning, benchmark, model selection or global stability acceptance.",
        "formal_training_started":False,"registered":False,"optimizer_instances":0,"optimization_updates":0,
        "deterministic_full_engineering_episodes":4,"full_episode_steps":4800,
        "sources":{"frozen_suite_protocol":{"path":str(PROTOCOL),"sha256":protocol_sha},"frozen_51":frozen,
            "original_preparation_sources":preparation["sources"],
            "local":{n:sha(HERE/n) for n in ["water_tank_env.py","check_water_tank_engineering.py","STATIC_COUNTEREXAMPLES.json"]}},
        "dependencies":{"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__},
        "negative_findings":[
            "Arbitrary public SPD M can make augmented target non-attractive, including an exact line of non-target equilibria.",
            "Strict continuous V decrease identity only covers fixed negative-cross M, nominal parameters, current continuous feedback and no saturation.",
            "Positive-cross and saturation cases have actual one-step V increases. These probes use explicit states and are not claims about default-zero-memory reachability.",
            "Unit M has semidefinite ideal V derivative, not a generic instability conclusion.",
            "Fixed equilibrium feedforward can regulate physical height while memory does not converge; the two success labels must remain distinct.",
            "The actuator is held for a full .05 second; continuous certificate algebra is not a discrete/intersample proof.",
            "No anti-windup, arbitrary memory truncation or unknown disturbances are included.",
            "The actual nonlinear implicit ZOH physics has no autograd model contract. M/e control gradients are local, with sqrt singularity at dry h=0 and piecewise saturation.",
            "Generic rollout detaches M and overwrites successful arrival time; this check does not accept learner optimization/recovery.",
            "Four deterministic same-start engineering cases are not evidence of training benefit or generalization."]}
    write("ENGINEERING_AUDIT.json",audit)
    manifest=[{"path":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.glob("*.json"))]
    write("ARTIFACT_MANIFEST.json",manifest)
    print(json.dumps({"output":str(output),"passed":audit["passed"],"checks":len(checks),
        "failed":[x["name"] for x in checks if not x["passed"]],"episodes":4,"steps":4800,"optimization_updates":0},ensure_ascii=False))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
