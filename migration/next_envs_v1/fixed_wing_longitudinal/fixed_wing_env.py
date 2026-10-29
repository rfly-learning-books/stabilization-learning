# SPDX-License-Identifier: GPL-3.0-only
"""New nominal longitudinal model using attributed public parameter literals.

The upstream coefficient source and GPL-3.0 license are retained in provenance/.
This is not the upstream flight simulator and not a certified same-M controller.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np
import torch
import gymnasium as gym
from stabilization_gym.common.types import MetricSpec
from stabilization_gym.interfaces.m_learning import PortableMFacade, MetricScope

HERE=Path(__file__).resolve().parent
PROTOCOL=json.loads((HERE/"DESIGN_PROTOCOL.json").read_text(encoding="utf-8"))
PARAM=MappingProxyType(json.loads((HERE/"provenance/PARAMETERS.json").read_text(encoding="utf-8"))["parameters"])
TRIM=json.loads((HERE/"provenance/TRIM_LINEARIZATION.json").read_text(encoding="utf-8"))
XTRIM=np.array(TRIM["trim_state"]); UTRIM=np.array(TRIM["trim_input"])
S=np.array([10.,5.,.1,.1,.2]); US=np.array([20.,.2])
AD=np.array(TRIM["Ad_ZOH"]); BD=np.array(TRIM["Bd_ZOH"])
LOW=np.array([20.,15.,-.15,-.35,-1.]); HIGH=np.array([180.,30.,.2,.35,1.])
ULOW=np.array([0.,-np.deg2rad(25.)]); UHIGH=np.array([40.,np.deg2rad(25.)])
TOL=np.array([.5,.25,.01,.01,.02]); DT=.05; H=.005
for array in (XTRIM,UTRIM,S,US,AD,BD,LOW,HIGH,ULOW,UHIGH,TOL): array.setflags(write=False)


def plain(x):
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x


def digest(x):
    return hashlib.sha256(json.dumps(plain(x),sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()


def source_hashes():
    paths=[HERE/"DESIGN_PROTOCOL.json",HERE/"fixed_wing_env.py",HERE/"LICENSE",*sorted((HERE/"provenance").glob("*"))]
    values={str(p.relative_to(HERE)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    project=HERE.parents[2]
    for relative in ("stabilization-gym/stabilization_gym/common/types.py","stabilization-gym/stabilization_gym/interfaces/m_learning.py",
                     "stabilization-learning-control/stabilization_learning/policies/metric.py"):
        values["@project/"+relative]=hashlib.sha256((project/relative).read_bytes()).hexdigest()
    return values


MODULE_SOURCE_HASHES=source_hashes()


def vector(x,n,name):
    if torch.is_tensor(x):
        if x.device.type!="cpu" or x.dtype not in (torch.float32,torch.float64): raise ValueError(name+" requires CPU float32/64")
        x=x.detach().numpy()
    a=np.asarray(x,dtype=np.float64)
    if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError(name+" must be finite shape "+str((n,)))
    return a.copy()


def domain(state):
    a=np.asarray(state,dtype=np.float64)
    finite=a.shape==(5,) and bool(np.isfinite(a).all())
    return {"valid":bool(finite and np.all(a>=LOW) and np.all(a<=HIGH)),"finite":finite,
            "signed_lower_margin":a-LOW if finite else None,"signed_upper_margin":HIGH-a if finite else None}


def rhs_numpy(state,control):
    """Pure SI algebra; domain checks belong to the caller before each stage."""
    h,v,alpha,theta,q=vector(state,5,"state"); thrust,delta=vector(control,2,"input")
    if v<=0: raise ValueError("No algebra at nonpositive airspeed")
    p=PARAM; qbar=.5*p["rho"]*v*v; qhat=p["c"]*q/(2*v)
    cl=p["C_L_0"]+p["C_L_alpha"]*alpha+p["C_L_q"]*qhat+p["C_L_delta_e"]*delta
    cd=p["C_D_p"]+(p["C_L_0"]+p["C_L_alpha"]*alpha)**2/(np.pi*p["e"]*p["AR"])+p["C_D_q"]*qhat+p["C_D_delta_e"]*delta
    cm=p["C_m_0"]+p["C_m_alpha"]*alpha+p["C_m_q"]*qhat+p["C_m_delta_e"]*delta
    lift,drag,moment=qbar*p["S_wing"]*cl,qbar*p["S_wing"]*cd,qbar*p["S_wing"]*p["c"]*cm
    gamma=theta-alpha
    return np.array([v*np.sin(gamma),(thrust*np.cos(alpha)-drag)/p["mass"]-p["gravity"]*np.sin(gamma),
        q-(lift+thrust*np.sin(alpha))/(p["mass"]*v)+p["gravity"]*np.cos(gamma)/v,q,moment/p["Jy"]])


def rhs_torch(state,control):
    """Pure differentiable RHS on the declared domain; not a step operation."""
    if state.dtype!=torch.float64 or control.dtype!=torch.float64 or state.device.type!="cpu" or control.device.type!="cpu" or state.shape!=(5,) or control.shape!=(2,):
        raise ValueError("Tensor RHS requires CPU64 scalar state/input")
    if not domain(state.detach().numpy())["valid"] or not torch.isfinite(control).all(): raise ValueError("RHS outside declared domain")
    h,v,alpha,theta,q=state.unbind(); thrust,delta=control.unbind(); p=PARAM
    qbar=.5*p["rho"]*v*v; qhat=p["c"]*q/(2*v)
    cl=p["C_L_0"]+p["C_L_alpha"]*alpha+p["C_L_q"]*qhat+p["C_L_delta_e"]*delta
    cd=p["C_D_p"]+(p["C_L_0"]+p["C_L_alpha"]*alpha)**2/(np.pi*p["e"]*p["AR"])+p["C_D_q"]*qhat+p["C_D_delta_e"]*delta
    cm=p["C_m_0"]+p["C_m_alpha"]*alpha+p["C_m_q"]*qhat+p["C_m_delta_e"]*delta
    lift,drag,moment=qbar*p["S_wing"]*cl,qbar*p["S_wing"]*cd,qbar*p["S_wing"]*p["c"]*cm
    gamma=theta-alpha
    return torch.stack([v*torch.sin(gamma),(thrust*torch.cos(alpha)-drag)/p["mass"]-p["gravity"]*torch.sin(gamma),
        q-(lift+thrust*torch.sin(alpha))/(p["mass"]*v)+p["gravity"]*torch.cos(gamma)/v,q,moment/p["Jy"]])


def riccati(M):
    """100 backward stages, M in running/terminal cost; differentiable in M."""
    A=M.new_tensor(AD); B=M.new_tensor(BD); R=.1*torch.eye(2,dtype=M.dtype,device=M.device); P=M
    for _ in range(100):
        G=DT*R+B.T@P@B
        K=torch.linalg.solve(G,B.T@P@A)
        P=DT*M+A.T@P@(A-B@K)
        P=.5*(P+P.T)
    return K,P


def integrate_held(state,control,*,dt=H,substeps=10,start_time=0.):
    """Pure but real numerical integration; every invocation must be budgeted.

    An invalid trial stage is saved as a trial, never accepted or evaluated.
    This function does not modify an environment, but is not 'static algebra'.
    """
    x=vector(state,5,"state"); u=vector(control,2,"input")
    if not domain(x)["valid"] or np.any(u<ULOW) or np.any(u>UHIGH): raise ValueError("Invalid integration start/input")
    if not isinstance(substeps,int) or not 1<=substeps<=20 or dt not in (H,H/2): raise ValueError("Only predeclared integration resolutions")
    records=[]; rhs_calls=0; completed=0; failure=None
    for k in range(substeps):
        start=x.copy(); base_time=start_time+k*dt
        record={"index":k,"start":start.copy(),"start_time":base_time,"dt":dt,"stages":[],"accepted":False}
        rates=[]
        for j in range(4):
            trial=start if j==0 else start+(dt/2 if j in (1,2) else dt)*rates[-1]
            info=domain(trial); stage={"stage":j+1,"trial_state":trial.copy(),"trial_time":base_time+(0 if j==0 else dt/2 if j in (1,2) else dt),"domain":info,"rhs":None,"rhs_attempted":False}
            if not info["valid"]:
                failure={"kind":"invalid_RK_stage","substep":k,"stage":j+1,"trial_time":stage["trial_time"],"trial_state":trial.copy()}
                record["stages"].append(stage); break
            rhs_calls+=1
            stage["rhs_attempted"]=True
            try:
                with np.errstate(all="raise"): rate=rhs_numpy(trial,u)
                if not np.isfinite(rate).all(): raise FloatingPointError("Nonfinite RHS")
            except FloatingPointError as exc:
                failure={"kind":"nonfinite_RHS","substep":k,"stage":j+1,"trial_time":stage["trial_time"],"message":str(exc)}
                record["stages"].append(stage); break
            stage["rhs"]=rate.copy(); record["stages"].append(stage); rates.append(rate)
        if failure is None:
            endpoint=start+dt*(rates[0]+2*rates[1]+2*rates[2]+rates[3])/6
            record["trial_endpoint"]=endpoint.copy(); record["endpoint_domain"]=domain(endpoint)
            if not record["endpoint_domain"]["valid"]:
                failure={"kind":"invalid_RK_endpoint","substep":k,"stage":"endpoint","trial_time":base_time+dt,"trial_state":endpoint.copy()}
            else:
                x=endpoint; completed+=1; record["accepted"]=True; record["accepted_endpoint"]=x.copy()
        records.append(record)
        if failure is not None: break
    return {"state":x.copy(),"input":u,"records":records,"failure":failure,"duration":completed*dt,
        "substeps_started":len(records),"substeps_completed":completed,"rhs_evaluations":rhs_calls}


@dataclass
class Plan:
    action: torch.Tensor
    record: dict
    cache: dict | None


class FixedWingLongitudinalEnv(gym.Env):
    metadata={"render_modes":[]}
    controller_form="same_M_running_terminal_Riccati_K_then_physical_saturation"
    control_call_is_pure=True
    control_call_side_effects=()

    def __init__(self):
        self.state_dim=self.error_dim=5; self.metric_context_dim=3
        self.dt=DT; self.max_steps=600; self.metric_spec=MetricSpec(5,.05,5.)
        self.action_space=gym.spaces.Box(ULOW,UHIGH,dtype=np.float64)
        self.observation_space=gym.spaces.Box(LOW,HIGH,dtype=np.float64)
        self.controller_capabilities={"action_gradient":True,"model_gradient":False,"episode_fixed_metric":True,
            "state_dependent_certificate":False,"cost_convention":"positive_integral","full_horizon_evaluation":True,
            "pure_tensor_RHS_gradient":True,"physical_step_autograd":False,"differentiable_candidate_scores":False,
            "certificate_guarantee":"candidate_only"}
        if source_hashes()!=MODULE_SOURCE_HASHES: raise ValueError("Sources changed after module load; reload a new revision")
        self._sources=deepcopy(MODULE_SOURCE_HASHES); self.state=XTRIM.copy(); self._done=True

    @property
    def goal(self): return XTRIM.copy()

    def _assert_fixed(self):
        if source_hashes()!=self._sources: raise ValueError("Frozen loaded source/dependency changed")
        if not np.array_equal(self.state,self._expected_state): raise ValueError("State changed outside reset/step")
        if not np.array_equal(self._context,[1.,1.,1.]): raise ValueError("Only fixed nominal context supported")

    def reset(self,*,seed=None,options=None):
        options={} if options is None else dict(options)
        if set(options)-{"state","delta_state_SI"}: raise ValueError("Only nominal state/delta reset options")
        if "state" in options and "delta_state_SI" in options: raise ValueError("Specify state or delta, not both")
        x=vector(options["state"],5,"state") if "state" in options else XTRIM+vector(options.get("delta_state_SI",[0]*5),5,"delta")
        if not domain(x)["valid"]: raise ValueError("Reset state outside declared attached-flow engineering envelope")
        super().reset(seed=seed); self.state=x.copy(); self._expected_state=x.copy(); self._initial=x.copy()
        self._context=np.ones(3); self._seed=seed; self._steps=0; self._physical_substeps=0; self._physical_time=0.
        self._done=self._failed=False; self._failure=None
        self._episode_M=self._cache=self._pending=None
        self._trace=[]; self._plans=[]; self._queries=0; self._cost=0.; self._failure_tail=0.; self._diagnostics={}
        self._hold_samples=int(self.in_goal(x)); self._first_hold=None
        self._budget={"control_attempts":0,"RK4_substeps_started":0,"RK4_substeps_completed":0,"RHS_evaluations":0,
            "submitted_gain_syntheses":0,"submitted_backward_stages":0}
        self._last_info=self._info()
        return x.copy(),deepcopy(self._last_info)

    @staticmethod
    def in_goal(x): return bool(np.all(np.abs(np.asarray(x)-XTRIM)<=TOL))

    def _info(self):
        success=bool(self._steps==600 and not self._failed and self._hold_samples>=21)
        return {"steps":self._steps,"actual_physical_time":self._physical_time,"remaining_seconds":max(0.,30-self._physical_time),
            "success":success,"failure":self._failed,"failure_location":deepcopy(self._failure),"in_goal":self.in_goal(self.state),
            "hold_samples":self._hold_samples,"hold_seconds":max(0,self._hold_samples-1)*DT,
            "first_hold_confirmation":self._first_hold,"final_sustained_arrival_start":None if self._hold_samples<21 else self._physical_time-(self._hold_samples-1)*DT,
            "integrated_cost_so_far":self._cost,"failure_tail":self._failure_tail,"cost_convention":"positive_integral"}

    def build_error(self,observation,goal=None,representation=None,history=None):
        self._assert_fixed()
        if representation is not None or history is not None: raise ValueError("Physical full-state coordinates only")
        if goal is not None and not np.array_equal(vector(goal,5,"goal"),XTRIM): raise ValueError("Reference override requires new protocol")
        x=observation if torch.is_tensor(observation) else torch.tensor(observation,dtype=torch.float64)
        value=vector(x,5,"observation")
        tolerance=1e-5 if x.dtype==torch.float32 else 0.
        if not np.allclose(value,self.state,rtol=0,atol=tolerance): raise ValueError("Stale physical observation")
        # Controller contract is CPU64; conversion preserves a float32 input graph.
        x=x.double(); e=(x-x.new_tensor(XTRIM))/x.new_tensor(S)
        return e,{"step":self._steps,"time":self._physical_time,"state":self.state.copy(),"goal":XTRIM.copy(),"metric_context":self._context.copy()}

    def metric_context(self,observation=None,goal=None):
        self._assert_fixed()
        if goal is not None and not np.array_equal(vector(goal,5,"goal"),XTRIM): raise ValueError("Reference override forbidden")
        return torch.tensor(self._context,dtype=torch.float64)

    def _metric(self,M):
        if not torch.is_tensor(M) or M.device.type!="cpu" or M.dtype not in (torch.float32,torch.float64): raise ValueError("M requires CPU float32/64")
        a=M.detach().double().numpy()
        if a.shape!=(5,5) or not np.isfinite(a).all() or not np.array_equal(a,a.T): raise ValueError("Finite exactly symmetric M required")
        eig=np.linalg.eigvalsh(a)
        if abs(np.trace(a)-5)>1e-5 or eig[0]<.05-2e-6 or eig[-1]>4.8+2e-6: raise ValueError("M spectral/trace bounds")
        if self._episode_M is not None and not np.array_equal(a,self._episode_M): raise ValueError("Episode M changed")
        return M.double()

    def _cache_key(self,M): return digest({"M":M,"sources":self._sources,"context":self._context})

    def plan(self,M,e,*,context,force_recompute=False):
        self._assert_fixed()
        if self._done: raise RuntimeError("Reset required")
        m=self._metric(M); error=vector(e,5,"error")
        if not torch.is_tensor(e) or e.device.type!="cpu" or e.dtype!=torch.float64: raise ValueError("CPU64 control error required")
        if context["step"]!=self._steps or context["time"]!=self._physical_time or not np.array_equal(context["state"],self.state) or not np.array_equal(context["goal"],XTRIM) or not np.array_equal(context["metric_context"],self._context): raise ValueError("Stale reference/control context")
        if not np.allclose(error,(self.state-XTRIM)/S,rtol=0,atol=2e-6): raise ValueError("Error differs from physical coordinates")
        key=self._cache_key(m); synth=0; failure=None; cache=None
        use_cache=self._cache is not None and self._cache["key"]==key and not m.requires_grad and not force_recompute
        try:
            if use_cache:
                K=m.new_tensor(self._cache["K"]); P0=m.new_tensor(self._cache["P0_not_M"]); cache=deepcopy(self._cache)
            else:
                synth=1; K,P0=riccati(m)
                if not torch.isfinite(K).all() or not torch.isfinite(P0).all(): raise FloatingPointError("Nonfinite Riccati result")
                F=AD-BD@K.detach().numpy(); diff=F.T@m.detach().numpy()@F-m.detach().numpy()
                cache=plain({"key":key,"M":m,"K":K,"P0_not_M":P0,"sources":self._sources,
                    "local_spectral_radius":np.max(np.abs(np.linalg.eigvals(F))),"same_M_linear_delta_eigenvalues":np.linalg.eigvalsh(diff)})
            requested=e.new_tensor(UTRIM)-e.new_tensor(US)*(K@e)
            action=torch.maximum(torch.minimum(requested,e.new_tensor(UHIGH)),e.new_tensor(ULOW))
        except (torch.linalg.LinAlgError,FloatingPointError) as exc:
            failure={"kind":"Riccati_numeric_failure","exception":type(exc).__name__,"message":str(exc)}
            requested=e.new_tensor(UTRIM); action=requested; K=P0=None; cache=None
        record=plain({"sources":self._sources,"step":self._steps,"time":self._physical_time,"state":self.state,
            "goal":XTRIM,"context":self._context,"M":m,"error_used":e,"requested":requested,"action":action,
            "saturation_mask":requested.detach().numpy()!=action.detach().numpy(),"cache_key":key,"cache_used":use_cache,
            "gain_syntheses":synth,"backward_stage_budget":100*synth,"failure":failure,
            "K":K,"P0_not_M":P0,"metric_eigenvalues":np.linalg.eigvalsh(m.detach().numpy()),"metric_trace":np.trace(m.detach().numpy())})
        record["plan_sha256"]=digest(record)
        return Plan(action,record,cache)

    def calculate_action_from_M(self,M,e,*,context,controller_state=None,certificate=None):
        if controller_state is not None or certificate is not None: raise ValueError("Episode-fixed nominal controller only")
        return self.plan(M,e,context=context).action

    def submit_plan(self,plan):
        self._assert_fixed()
        if self._done or self._pending is not None: raise RuntimeError("Finished episode or pending action")
        r=deepcopy(plan.record); signature=r.pop("plan_sha256")
        if signature!=digest(r) or r["sources"]!=self._sources or r["step"]!=self._steps or r["time"]!=self._physical_time or not np.array_equal(r["state"],self.state) or not np.array_equal(r["goal"],XTRIM) or not np.array_equal(r["context"],self._context): raise ValueError("Plan binding mismatch")
        m=self._metric(torch.tensor(r["M"],dtype=torch.float64)).numpy()
        if r["cache_key"]!=self._cache_key(m) or not np.array_equal(plan.action.detach().numpy(),r["action"]): raise ValueError("Plan action/cache key mismatch")
        if r["failure"] is None:
            if plan.cache is None or plan.cache["key"]!=r["cache_key"] or not np.array_equal(plan.cache["K"],r["K"]) or not np.array_equal(plan.cache["P0_not_M"],r["P0_not_M"]): raise ValueError("Missing valid gain cache")
            req=UTRIM-US*(np.array(r["K"])@np.array(r["error_used"]))
            if not np.allclose(req,r["requested"],atol=1e-12,rtol=1e-12) or not np.array_equal(np.clip(r["requested"],ULOW,UHIGH),r["action"]): raise ValueError("Saved saturated command differs")
        elif not np.array_equal(r["action"],UTRIM): raise ValueError("Controller failure fallback differs from trim")
        self._episode_M=m.copy(); self._cache=deepcopy(plan.cache); self._pending=deepcopy(plan.record)
        self._queries+=1; self._budget["submitted_gain_syntheses"]+=r["gain_syntheses"]; self._budget["submitted_backward_stages"]+=r["backward_stage_budget"]
        self._plans.append({"plan_sha256":signature,"step":self._steps,"executed":False})
        self._diagnostics={"cache_used":r["cache_used"],"gain_syntheses":r["gain_syntheses"],"saturation_mask":r["saturation_mask"]}

    def step(self,action):
        self._assert_fixed()
        if self._done or self._pending is None: raise RuntimeError("Reset and facade audit plan required")
        u=vector(action,2,"physical action"); plan=deepcopy(self._pending)
        if not np.array_equal(u,plan["action"]): raise ValueError("Action differs from pending command; no hidden step clipping")
        before=self.state.copy(); time_before=self._physical_time; e=(before-XTRIM)/S; m=self._episode_M
        if plan["failure"] is None:
            result=integrate_held(before,u,start_time=time_before)
        else:
            result={"state":before,"input":u,"records":[],"failure":plan["failure"],"duration":0.,"substeps_started":0,"substeps_completed":0,"rhs_evaluations":0}
        self.state=np.asarray(result["state"]).copy(); self._expected_state=self.state.copy(); self._steps+=1
        self._physical_substeps+=result["substeps_completed"]; self._physical_time=self._physical_substeps*H
        self._budget["control_attempts"]+=1
        for key in ("substeps_started","substeps_completed"): self._budget["RK4_"+key]+=result[key]
        self._budget["RHS_evaluations"]+=result["rhs_evaluations"]
        self._failure=deepcopy(result["failure"]); self._failed=self._failure is not None
        en=(self.state-XTRIM)/S; cost=result["duration"]*float(e@e)
        tail=max(float(en@en),25.)*max(0.,30-self._physical_time) if self._failed else 0.
        self._cost+=cost; self._failure_tail=tail
        self._hold_samples=self._hold_samples+1 if not self._failed and self.in_goal(self.state) else 0
        if self._hold_samples>=21 and self._first_hold is None: self._first_hold=self._physical_time
        self._done=bool(self._failed or self._steps>=600)
        row=plain({"attempt":self._steps,"state_before":before,"state_after":self.state,"time_before":time_before,
            "time_after":self._physical_time,"action":u,"plan":plan,"integration":result,"cost":cost,"failure_tail":tail,
            "V_before":e@m@e,"V_after":en@m@en,"delta_V":en@m@en-e@m@e,
            "failure":self._failed,"hold_samples":self._hold_samples,"budget_cumulative":self._budget})
        self._trace.append(row); self._plans[-1]["executed"]=True; self._pending=None
        self._last_info=self._info(); self._last_info.update(applied_action=u.copy(),cost=cost,saturation_mask=plan["saturation_mask"])
        return self.state.copy(),cost+tail,self._failed,self._steps>=600 and not self._failed,deepcopy(self._last_info)

    def state_dict(self):
        self._assert_fixed()
        state=plain({"sources":self._sources,"initial_state":self._initial,"seed":self._seed,"state":self.state,"context":self._context,
            "steps":self._steps,"physical_substeps":self._physical_substeps,"physical_time":self._physical_time,
            "episode_M":self._episode_M,"cache":self._cache,"pending":self._pending,"queries":self._queries,
            "trace":self._trace,"plans":self._plans,"done":self._done,"failed":self._failed,"failure":self._failure,
            "hold_samples":self._hold_samples,"first_hold":self._first_hold,"cost":self._cost,"failure_tail":self._failure_tail,
            "budget":self._budget,"last_info":self._last_info,"diagnostics":self._diagnostics,"rng":self.np_random.bit_generator.state})
        state["snapshot_sha256"]=digest(state); return deepcopy(state)

    def load_state_dict(self,state):
        s=deepcopy(state); token=s.pop("snapshot_sha256",None)
        if token!=digest(s) or s["sources"]!=source_hashes(): raise ValueError("Snapshot hash/source/protocol mismatch")
        if type(s["steps"]) is not int or not 0<=s["steps"]<=600 or type(s["physical_substeps"]) is not int or not 0<=s["physical_substeps"]<=6000 or len(s["trace"])!=s["steps"]: raise ValueError("Snapshot counter ranges")
        temp=type(self)(); temp.reset(seed=s["seed"],options={"state":s["initial_state"]})
        if s["context"]!=[1.,1.,1.]: raise ValueError("Nominal context changed")
        metric=None if s["episode_M"] is None else temp._metric(torch.tensor(s["episode_M"],dtype=torch.float64)).numpy()
        previous=temp.state.copy(); completed=0; cost=0.; hold=int(temp.in_goal(previous)); first=None; failed=False
        started=rhs_count=0
        for index,row in enumerate(s["trace"],1):
            if failed or row["attempt"]!=index or not np.array_equal(previous,row["state_before"]) or row["time_before"]!=completed*H: raise ValueError("Broken physical trace")
            plan=deepcopy(row["plan"]); psig=plan.pop("plan_sha256")
            if digest(plan)!=psig or plan["sources"]!=self._sources or plan["step"]!=index-1 or not np.array_equal(plan["state"],previous) or not np.array_equal(plan["M"],metric) or not np.array_equal(plan["goal"],XTRIM) or plan["context"]!=[1.,1.,1.] or not np.array_equal(plan["action"],row["action"]): raise ValueError("Saved plan binding")
            integ=row["integration"]; accepted=0; called=0; stage_state=previous.copy()
            for sub in integ["records"]:
                if not np.array_equal(sub["start"],stage_state): raise ValueError("RK start mismatch")
                rates=[]
                for j,stage in enumerate(sub["stages"]):
                    expected=stage_state if j==0 else stage_state+(H/2 if j in (1,2) else H)*rates[-1]
                    if not np.array_equal(expected,stage["trial_state"]) or digest(domain(expected))!=digest(stage["domain"]): raise ValueError("RK trial/domain mismatch")
                    called+=int(stage["rhs_attempted"])
                    if stage["rhs"] is not None:
                        if not stage["domain"]["valid"]: raise ValueError("RHS evaluated outside domain")
                        rates.append(vector(stage["rhs"],5,"saved RHS"))
                if sub["accepted"]:
                    if len(rates)!=4: raise ValueError("Accepted RK stage count")
                    endpoint=stage_state+H*(rates[0]+2*rates[1]+2*rates[2]+rates[3])/6
                    if not np.array_equal(endpoint,sub["accepted_endpoint"]) or not domain(endpoint)["valid"]: raise ValueError("Accepted RK endpoint mismatch")
                    stage_state=endpoint; accepted+=1
            if accepted!=integ["substeps_completed"] or len(integ["records"])!=integ["substeps_started"] or called!=integ["rhs_evaluations"] or not np.array_equal(stage_state,row["state_after"]) or integ["duration"]!=accepted*H: raise ValueError("Integrated count/state mismatch")
            completed+=accepted; started+=len(integ["records"]); rhs_count+=called; failed=integ["failure"] is not None
            e=(previous-XTRIM)/S; en=(stage_state-XTRIM)/S
            expected_cost=accepted*H*float(e@e); expected_tail=max(float(en@en),25.)*(30-completed*H) if failed else 0.
            if row["time_after"]!=completed*H or row["failure"]!=failed or not np.isclose(row["cost"],expected_cost,rtol=1e-12,atol=1e-12) or not np.isclose(row["failure_tail"],expected_tail,rtol=1e-12,atol=1e-12) or not np.isclose(row["V_before"],e@metric@e,rtol=1e-12,atol=1e-12) or not np.isclose(row["V_after"],en@metric@en,rtol=1e-12,atol=1e-12): raise ValueError("Trace cost/V/time mismatch")
            cost+=expected_cost; hold=hold+1 if not failed and temp.in_goal(stage_state) else 0
            if hold>=21 and first is None: first=completed*H
            if row["hold_samples"]!=hold: raise ValueError("Hold count mismatch")
            if s["plans"][index-1]!={"plan_sha256":psig,"step":index-1,"executed":True}: raise ValueError("Plan log mismatch")
            previous=stage_state
        if not np.array_equal(previous,s["state"]) or completed!=s["physical_substeps"] or s["physical_time"]!=completed*H or s["failed"]!=failed or s["done"]!=(failed or s["steps"]>=600) or s["hold_samples"]!=hold or s["first_hold"]!=first or not np.isclose(cost,s["cost"],atol=1e-12,rtol=1e-12): raise ValueError("Snapshot aggregate mismatch")
        if s["budget"]["control_attempts"]!=s["steps"] or s["budget"]["RK4_substeps_started"]!=started or s["budget"]["RK4_substeps_completed"]!=completed or s["budget"]["RHS_evaluations"]!=rhs_count: raise ValueError("Snapshot numerical budget mismatch")
        if type(s["queries"]) is not int or s["queries"]!=len(s["plans"]) or s["queries"]!=s["steps"]+int(s["pending"] is not None): raise ValueError("Snapshot query accounting")
        cache=s["cache"]
        if cache is not None:
            if metric is None or cache["key"]!=temp._cache_key(metric) or cache["sources"]!=self._sources: raise ValueError("Cache source key mismatch")
            K,P0=riccati(torch.tensor(metric,dtype=torch.float64))
            if not np.allclose(K.numpy(),cache["K"],atol=1e-12,rtol=1e-12) or not np.allclose(P0.numpy(),cache["P0_not_M"],atol=1e-12,rtol=1e-12): raise ValueError("Gain cache recomputation mismatch")
        for attr,key in (("state","state"),("_initial","initial_state"),("_context","context"),("_episode_M","episode_M")):
            setattr(temp,attr,None if s[key] is None else np.array(s[key],dtype=np.float64))
        temp._expected_state=temp.state.copy()
        for attr,key in (("_steps","steps"),("_physical_substeps","physical_substeps"),("_physical_time","physical_time"),
            ("_cache","cache"),("_queries","queries"),("_trace","trace"),("_plans","plans"),("_done","done"),("_failed","failed"),("_failure","failure"),
            ("_hold_samples","hold_samples"),("_first_hold","first_hold"),("_cost","cost"),("_failure_tail","failure_tail"),("_budget","budget"),("_last_info","last_info"),("_diagnostics","diagnostics")):
            setattr(temp,attr,deepcopy(s[key]))
        expected_info=temp._info()
        if s["trace"]:
            last=s["trace"][-1]
            if digest(temp._failure)!=digest(last["integration"]["failure"]) or temp._failure_tail!=last["failure_tail"]: raise ValueError("Final failure location/tail mismatch")
            expected_info.update(applied_action=np.array(last["action"]),cost=last["cost"],saturation_mask=last["plan"]["saturation_mask"])
        if digest(expected_info)!=digest(s["last_info"]): raise ValueError("Final info differs from restored physical state")
        temp._last_info=deepcopy(expected_info)
        last_plan=s["pending"] if s["pending"] is not None else s["trace"][-1]["plan"] if s["trace"] else None
        expected_diag={} if last_plan is None else {"cache_used":last_plan["cache_used"],"gain_syntheses":last_plan["gain_syntheses"],"saturation_mask":last_plan["saturation_mask"]}
        if digest(expected_diag)!=digest(s["diagnostics"]): raise ValueError("Public diagnostics differ from saved plan")
        temp.np_random.bit_generator.state=deepcopy(s["rng"])
        if s["pending"] is not None:
            if s["plans"][-1]!={"plan_sha256":s["pending"]["plan_sha256"],"step":s["steps"],"executed":False}: raise ValueError("Pending log mismatch")
            pending=Plan(torch.tensor(s["pending"]["action"],dtype=torch.float64),s["pending"],cache)
            old_budget=deepcopy(temp._budget); temp.submit_plan(pending)
            temp._budget=old_budget; temp._queries=s["queries"]; temp._plans=deepcopy(s["plans"])
        self.__dict__.update(temp.__dict__)
        return {"validation_gain_syntheses":int(cache is not None),"validation_backward_stages":100*int(cache is not None),"physical_steps":0,"RHS_evaluations":0}


class FixedWingMFacade(PortableMFacade):
    def __init__(self,env):
        super().__init__(env,name="isolated_fixed_wing_longitudinal_v1")
        self.description=replace(self.description,support_level="engineering_implementation_pending_acceptance",
            metric_scope=MetricScope.GOAL_CONDITIONED,conditioning=("nominal_v_density_mass_constants",),
            control_call_is_pure=False,control_call_side_effects=("pending_audit_plan","explicit_gain_cache_commit","query_and_synthesis_counters"),
            supported_algorithms=(),missing_capabilities=("nonlinear engineering acceptance","formal training/replay integration","varying trim/context","same-M certificate proof"),
            gradients=replace(self.description.gradients,model_wrt_action=False,note="CPU64 M/error-to-action Riccati graph with piecewise clip derivative; numeric rollout cache bypassed when M requires_grad. Pure tensor RHS exists, full physical-step/model gradient not promised."))
        self.source_signature_sha256=digest({"description":self.description.to_dict(),"sources":env._sources})

    def calculate_action_from_M(self,M,e,*,context,controller_state=None,certificate=None):
        if controller_state is not None or certificate is not None: raise ValueError("Nominal episode-fixed controller only")
        plan=self.env.plan(M,e,context=context); self.env.submit_plan(plan)
        return plan.action
