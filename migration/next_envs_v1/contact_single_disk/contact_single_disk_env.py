"""Isolated CPU MuJoCo3.2.3 state-feedback contact task, d8 gap version.

plan() is a pure cloned-simulation query. ContactMFacade explicitly records its
returned plan for execution/audit and is therefore not a pure control facade.
No gradient through contact dynamics or hard argmin is claimed.
"""
from __future__ import annotations
from copy import copy,deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import mujoco
import numpy as np
import torch

from stabilization_gym.envs.base import StateMetricEnv
from stabilization_gym.interfaces.m_learning import PortableMFacade

HERE=Path(__file__).resolve().parent
PROTOCOL=json.loads((HERE/"DESIGN_PROTOCOL.json").read_text(encoding="utf-8"))
MODEL_XML="""<mujoco model="isolated_single_disk_gap_v2">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" cone="pyramidal" jacobian="auto" iterations="100" tolerance="1e-8"/>
  <size njmax="1000" nconmax="200"/>
  <default>
    <joint margin="0"/>
    <geom friction="0.85 0.01 0.001" solimp="0.90 0.95 0.001" solref="0.012 1" margin="0"/>
    <velocity ctrllimited="true" ctrlrange="-0.1 0.1" kv="4"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 -0.045" size="2 2 .1" contype="0" conaffinity="0"/>
    <geom name="table" type="box" pos="-.10 0 -.02" size=".48 .28 .02" contype="0" conaffinity="0"/>
    <body name="goal" mocap="true" pos="0 0 .0008">
      <geom name="goal_marker" type="cylinder" size=".015 .0003" contype="0" conaffinity="0"/>
    </body>
    <body name="disk" pos="0 0 .004">
      <joint name="disk_x" type="slide" axis="1 0 0" limited="true" range="-.055 .060" damping="1"/>
      <joint name="disk_y" type="slide" axis="0 1 0" limited="true" range="-.060 .060" damping="1"/>
      <joint name="disk_yaw" type="hinge" axis="0 0 1" limited="false" damping=".08"/>
      <geom name="disk_geom" type="cylinder" size=".015 .004" mass=".025"/>
    </body>
    <body name="pusher" pos="0 0 .0045">
      <joint name="pusher_x" type="slide" axis="1 0 0" limited="true" range="-.065 .070" damping=".06"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" limited="true" range="-.070 .070" damping=".06"/>
      <geom name="pusher_geom" type="cylinder" size=".004 .0045" mass=".030"/>
      <geom name="pusher_marker" type="sphere" pos="0 0 .0045" size=".0055" mass=".0001" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator><velocity name="x_velocity" joint="pusher_x"/><velocity name="y_velocity" joint="pusher_y"/></actuator>
</mujoco>"""

def _ready(value):
    if hasattr(value,"tolist"): return value.tolist()
    raise TypeError(type(value).__name__)

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=_ready,allow_nan=False,separators=(",",":")).encode()).hexdigest()


class ContactSingleDiskEnv(StateMetricEnv):
    controller_form="MuJoCo_cloned_candidate_prediction_M_terminal_weight_hard_argmin"
    control_call_is_pure=True
    control_call_side_effects=()
    metadata={"render_modes":["rgb_array"],"render_fps":50}

    def __init__(self,*,render_mode=None):
        if mujoco.__version__!="3.2.3": raise RuntimeError("This new isolated protocol pins installed MuJoCo3.2.3")
        super().__init__(state_dim=10,error_dim=8,action_low=[-.1,-.1],action_high=[.1,.1],
            dt=.02,max_steps=1500,goal=np.zeros(10),metric_floor=.05,render_mode=render_mode)
        self.metric_context_dim=10
        self.model=mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data=mujoco.MjData(self.model)
        self.state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.integration_state_size=mujoco.mj_stateSize(self.model,self.state_spec)
        self.xml_sha256=hashlib.sha256(MODEL_XML.encode()).hexdigest()
        self.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.protocol_sha256=hashlib.sha256((HERE/"DESIGN_PROTOCOL.json").read_bytes()).hexdigest()
        self.disk_bounds=np.array([[-.055,.060],[-.060,.060]])
        self.pusher_bounds=np.array([[-.065,.070],[-.070,.070]])
        self.goal_bounds=np.array([[-.032,.037],[-.037,.037]])
        self.goal_xy=np.zeros(2)
        self._goal_snapshot=self.goal_xy.copy()
        self.controller_capabilities.update(action_gradient=False,model_gradient=False,
            differentiable_candidate_scores=True,differentiable_representation=False,
            physical_step_autograd=False,full_horizon_evaluation=True,
            controller_precision="CPU float64 errors; float32/float64 M accepted and promoted",
            full_state_observation=True,certificate_guarantee="candidate_only")
        self._pending_plan=self._episode_metric=None
        self._execution_mode=None
        self._origin="external_physical_command"
        self._in_goal_since=self._first_hold_confirmation=None
        self._trace=[]; self._last_info={}; self._plan_log=[]
        self._context=np.zeros(10,dtype=np.float32)
        self._initial_state=np.zeros(10); self._seed_record=None
        self._planning_failure=False
        self._budget={"plan_queries":0,"prediction_control_steps":0,"prediction_substeps":0,
                      "real_control_steps":0,"real_substeps":0,"failed_candidates":0,"planning_seconds":0.}
        self._model_signature=self._physical_model_signature()

    def _physical_model_signature(self):
        return _digest({"dt":self.model.opt.timestep,"solver":int(self.model.opt.solver),"iterations":self.model.opt.iterations,
            "tolerance":self.model.opt.tolerance,"integrator":int(self.model.opt.integrator),
            "gravity":self.model.opt.gravity,"body_mass":self.model.body_mass,"body_inertia":self.model.body_inertia,
            "body_pos":self.model.body_pos,"damping":self.model.dof_damping,"ranges":self.model.jnt_range,
            "joint_margin":self.model.jnt_margin,"geom_size":self.model.geom_size,"geom_pos":self.model.geom_pos,
            "geom_friction":self.model.geom_friction,"geom_solref":self.model.geom_solref,"geom_solimp":self.model.geom_solimp,
            "geom_margin":self.model.geom_margin,"geom_contype":self.model.geom_contype,"geom_conaffinity":self.model.geom_conaffinity,
            "actuator_gain":self.model.actuator_gainprm,"actuator_bias":self.model.actuator_biasprm})

    def _assert_fixed(self):
        if not np.array_equal(self.goal_xy,self._goal_snapshot) or self._physical_model_signature()!=self._model_signature:
            raise ValueError("Goal/model changed outside reset")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError("External forces are not part of this protocol")

    def _integration(self,data=None):
        values=np.empty(self.integration_state_size)
        mujoco.mj_getState(self.model,self.data if data is None else data,values,self.state_spec)
        return values

    @staticmethod
    def _observation(data): return np.concatenate((data.qpos.copy(),data.qvel.copy()))

    @staticmethod
    def surface_gap(state):
        x=np.asarray(state)
        return float(np.linalg.norm(x[3:5]-x[:2])-.019)

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        options={} if options is None else dict(options)
        if set(options)-{"disk_xy","pusher_xy","goal_xy","qvel","yaw","execution_origin"}: raise ValueError("Unsupported reset field")
        goal=np.asarray(options.get("goal_xy",[0.,0.]),dtype=np.float64)
        disk=np.asarray(options.get("disk_xy",[-.025,0.]),dtype=np.float64)
        pusher=np.asarray(options.get("pusher_xy",[-.05,0.]),dtype=np.float64)
        velocities=np.asarray(options.get("qvel",np.zeros(5)),dtype=np.float64)
        yaw=float(options.get("yaw",0.))
        if goal.shape!=(2,) or disk.shape!=(2,) or pusher.shape!=(2,) or velocities.shape!=(5,) or not np.isfinite(np.r_[goal,disk,pusher,velocities,yaw]).all():
            raise ValueError("Reset requires finite physical shapes")
        if np.any(goal<self.goal_bounds[:,0]) or np.any(goal>self.goal_bounds[:,1]): raise ValueError("Goal must permit the 21mm radial target projection plus2mm workspace buffer")
        for xy,bounds in ((disk,self.disk_bounds),(pusher,self.pusher_bounds)):
            if np.any(xy<=bounds[:,0]+1e-6) or np.any(xy>=bounds[:,1]-1e-6): raise ValueError("Initial centers must be strictly within joint limits")
        x=np.r_[disk,yaw,pusher,velocities]
        if self.surface_gap(x)<0. or self._failure_reason(x) is not None: raise ValueError("Invalid overlapping or unsafe reset; no projection/fallback")
        origin=options.get("execution_origin","external_physical_command")
        if origin not in ("external_physical_command","deliberate_fault_injection_not_M_policy_performance"): raise ValueError("Invalid execution origin")
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:]=x[:5]; self.data.qvel[:]=x[5:]
        self.data.mocap_pos[0,:]=[goal[0],goal[1],.0008]
        self.data.mocap_quat[0,:]=[1,0,0,0]
        mujoco.mj_forward(self.model,self.data)
        self.goal_xy=goal.copy(); self._goal_snapshot=goal.copy()
        self.goal=np.r_[goal,0.,np.zeros(7)]
        self.state=self._observation(self.data)
        self._context=np.r_[goal/.05,(disk-goal)/.05,(pusher-disk)/.05,.015/.015,.004/.015,1.,1.].astype(np.float32)
        self._initial_state=self.state.copy(); self._seed_record=seed
        self._elapsed_steps=self._hold_steps=0; self._done=False
        self._pending_plan=self._episode_metric=None; self._execution_mode=None; self._origin=origin
        self._planning_failure=False; self._trace=[]; self._plan_log=[]
        self._in_goal_since=0. if self.is_success(self.state) else None
        self._first_hold_confirmation=None
        self._budget={"plan_queries":0,"prediction_control_steps":0,"prediction_substeps":0,
                      "real_control_steps":0,"real_substeps":0,"failed_candidates":0,"planning_seconds":0.}
        self._last_info=self._info()
        return self.state.copy(),deepcopy(self._last_info)

    def tensor_error(self,state):
        x=state if torch.is_tensor(state) else torch.as_tensor(state,dtype=torch.float64)
        if x.dtype!=torch.float64 or x.device.type!="cpu" or x.shape[-1:]!=(10,) or not torch.isfinite(x).all():
            raise ValueError("This first contact version requires CPU float64 physical observations/errors")
        gap=torch.linalg.vector_norm(x[...,3:5]-x[...,:2],dim=-1)-.019
        return torch.cat(((x[...,:2]-x.new_tensor(self.goal_xy))/.05,x[...,5:7]/.1,x[...,7:8]/3,
                          x[...,8:10]/.1,((.002-gap).clamp_min(0.)/.02).unsqueeze(-1)),dim=-1)

    def build_error(self,observation,goal=None,representation=None,history=None):
        self._assert_fixed()
        if representation is not None and not getattr(representation,"preserves_physical_coordinates",False):
            raise ValueError("Identity physical representation only; this is not visual estimation")
        if goal is not None and not np.array_equal(np.asarray(goal),self.goal): raise ValueError("Use episode reference")
        return self.tensor_error(observation),{"goal_xy":self.goal_xy.copy(),"model_sha256":self.xml_sha256,
            "integration_sha256":hashlib.sha256(self._integration().tobytes()).hexdigest()}

    def metric_context(self,observation=None,goal=None):
        self._assert_fixed()
        if goal is not None and not np.array_equal(np.asarray(goal),self.goal): raise ValueError("Use episode reference")
        return torch.as_tensor(self._context.copy())

    def physical_components(self,state):
        x=np.asarray(state,dtype=np.float64)
        return np.r_[x[:2]-self.goal_xy,x[5:7],x[7],x[8:10],max(0.,.002-self.surface_gap(x))]

    def physical_cost(self,state):
        p=self.physical_components(state); return float(p@p)

    def is_success(self,state):
        x=np.asarray(state)
        return bool(np.isfinite(x).all() and np.linalg.norm(x[:2]-self.goal_xy)<=.008 and np.linalg.norm(x[5:7])<=.003 and
            abs(x[7])<=.05 and np.linalg.norm(x[8:10])<=.003 and self.surface_gap(x)>=.002 and self._failure_reason(x) is None)

    def _failure_reason(self,state):
        x=np.asarray(state)
        if x.shape!=(10,) or not np.isfinite(x).all(): return "nonfinite_or_invalid_state"
        if np.any(x[:2]<=self.disk_bounds[:,0]) or np.any(x[:2]>=self.disk_bounds[:,1]): return "disk_joint_boundary"
        if np.any(x[3:5]<=self.pusher_bounds[:,0]) or np.any(x[3:5]>=self.pusher_bounds[:,1]): return "pusher_joint_boundary"
        if self.surface_gap(x)<-.001: return "excessive_soft_penetration"
        if np.linalg.norm(x[5:7])>2. or np.linalg.norm(x[8:10])>2. or abs(x[7])>20.: return "velocity_safety_limit"
        return None

    def is_failure(self,state): return self._failure_reason(state) is not None

    def _filter_command(self,command,data):
        requested=np.asarray(command,dtype=np.float64)
        if requested.shape!=(2,) or not np.isfinite(requested).all(): raise ValueError("Require finite physical velocity[m/s] command")
        command=np.clip(requested,-.1,.1)
        low=np.minimum(0.,(self.pusher_bounds[:,0]+1e-5-data.qpos[3:5])/.02)
        high=np.maximum(0.,(self.pusher_bounds[:,1]-1e-5-data.qpos[3:5])/.02)
        # Domain violations are handled as failures; this never clips qpos.
        return np.clip(np.minimum(np.maximum(command,low),high),-.1,.1)

    def _contact_records(self,data):
        records=[]
        for i in range(data.ncon):
            contact=data.contact[i]; force=np.zeros(6)
            mujoco.mj_contactForce(self.model,data,i,force)
            records.append({"geom1":int(contact.geom1),"geom2":int(contact.geom2),"distance":float(contact.dist),
                "normal":contact.frame[:3].copy().tolist(),"force_contact_frame":force.tolist()})
        return records

    def _advance(self,data,command,*,record_substeps=False):
        applied=self._filter_command(command,data)
        data.ctrl[:]=applied
        rows=[]; reason=None
        for index in range(4):
            start_state=self._observation(data); start_time=float(data.time)
            mujoco.mj_step(self.model,data)
            x=self._observation(data); failure=self._failure_reason(x)
            if abs(float(data.time)-start_time-.005)>1e-12:
                failure="engine_time_reset_or_inconsistent_step"
            if reason is None and failure is not None: reason=failure
            if record_substeps:
                rows.append({"substep":index+1,"time":float(data.time),"state":x.tolist(),
                    "substep_start_time":start_time,"substep_start_state":start_state.tolist(),
                    "contact_force_evaluated_at":"substep_start_state_not_integrated_endpoint",
                    "surface_gap":self.surface_gap(x),"failure_reason":failure,"contacts":self._contact_records(data),
                    "qacc_at_start":data.qacc.copy().tolist(),"qacc_warmstart_after":data.qacc_warmstart.copy().tolist(),
                    "stored_velocity_command":data.ctrl.copy().tolist(),"actuator_force_at_start":data.actuator_force.copy().tolist(),
                    "qfrc_constraint_at_start":data.qfrc_constraint.copy().tolist(),"qfrc_passive_at_start":data.qfrc_passive.copy().tolist()})
        return self._observation(data),applied,reason,rows

    def torch_step(self,state,action):
        raise NotImplementedError("Hard MuJoCo contact model has no claimed autograd contract")

    def _validate_metric(self,M):
        if not torch.is_tensor(M) or M.dtype not in (torch.float32,torch.float64) or M.device.type!="cpu" or M.shape!=(8,8) or not torch.isfinite(M).all():
            raise ValueError("Expected one CPU float32/float64 M[8,8]")
        m=M.to(torch.float64); eig=torch.linalg.eigvalsh(m)
        if not torch.allclose(m,m.T,atol=1e-7,rtol=1e-7) or float(eig.min())<.05-1e-6 or float(eig.max())>7.65+1e-6 or abs(float(torch.trace(m))-8.)>1e-5:
            raise ValueError("M violates shared SPD/trace specification")
        return m

    @staticmethod
    def _segment_clear(a,b,radius):
        direction=b-a
        t=0. if float(direction@direction)==0. else float(np.clip(-a@direction/(direction@direction),0.,1.))
        return float(np.linalg.norm(a+t*direction))>=radius-1e-12

    def _retreat_brake(self,x):
        relative=x[3:5]-x[:2]; distance=np.linalg.norm(relative)
        if distance<.022 and distance>1e-12:
            return relative/distance*min(.03,(.022-distance)/.02)
        return np.clip(-x[8:10],-.1,.1)

    def _candidate_command(self,x,candidate,j):
        if candidate["kind"]=="zero": return np.zeros(2)
        if candidate["kind"]=="braking": return np.clip(-x[8:10],-.1,.1)
        if j>=105 or np.linalg.norm(x[:2]-self.goal_xy)<=.003: return self._retreat_brake(x)
        n=np.asarray(candidate["direction"]); speed=candidate["speed"]
        relative=x[3:5]-x[:2]; distance=np.linalg.norm(relative); desired=-n*.022
        aligned=distance>1e-12 and float((-n)@relative/distance)>=.98
        if aligned and distance<=.023: return speed*n
        if distance<.021 and distance>1e-12:
            target=relative/distance*.023
        elif self._segment_clear(relative,desired,.021):
            target=desired
        else:
            angle=math.atan2(relative[1],relative[0]); final=math.atan2(desired[1],desired[0])
            delta=(final-angle+math.pi)%(2*math.pi)-math.pi
            step=np.clip(delta,-math.pi/8,math.pi/8)
            target=.023*np.array([math.cos(angle+step),math.sin(angle+step)])
        displacement=target-relative; length=np.linalg.norm(displacement)
        return np.zeros(2) if length<1e-12 else displacement/length*min(speed,length/.02)

    def _candidate_definitions(self):
        delta=self.goal_xy-self.data.qpos[:2]; base=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-12 else 0.
        candidates=[{"kind":"zero","index":0},{"kind":"braking","index":1}]
        for angle in range(8):
            direction=[math.cos(base+angle*math.pi/4),math.sin(base+angle*math.pi/4)]
            for speed in (.03,.07):
                candidates.append({"kind":"push","index":len(candidates),"direction":direction,"speed":speed})
        return candidates

    def candidate_scores(self,M,plan):
        m=self._validate_metric(M)
        terminal=torch.as_tensor([c["terminal_error"] for c in plan["candidates"]],dtype=torch.float64)
        fixed=torch.as_tensor([c["running_cost"]+c["command_energy"]+c["approach_cost"] for c in plan["candidates"]],dtype=torch.float64)
        valid=torch.as_tensor([c["valid"] for c in plan["candidates"]],dtype=torch.bool)
        scores=fixed+torch.einsum("ni,ij,nj->n",terminal,m,terminal)
        return torch.where(valid,scores,torch.full_like(scores,float("inf")))

    def plan(self,M,e,*,context):
        """Pure query: all candidates run on independent full-MjData copies."""
        self._assert_fixed(); m=self._validate_metric(M)
        if e.dtype!=torch.float64 or e.device.type!="cpu" or e.shape!=(8,) or not torch.isfinite(e).all(): raise ValueError("Planning error must be CPU float64[8]")
        current=self._observation(self.data); expected=self.tensor_error(current)
        fingerprint=hashlib.sha256(self._integration().tobytes()).hexdigest()
        if (context.get("integration_sha256")!=fingerprint or context.get("model_sha256")!=self.xml_sha256 or
            not np.array_equal(context.get("goal_xy"),self.goal_xy) or not torch.allclose(e,expected,atol=1e-12,rtol=1e-12)):
            raise ValueError("Planning query must match current physical/solver state and target")
        start=time.perf_counter()
        result={"schema":"contact-pure-plan-v1","initial_integration_sha256":fingerprint,
            "model_sha256":self.xml_sha256,"source_sha256":self.source_sha256,"goal_xy":self.goal_xy.tolist(),
            "M":m.detach().numpy().tolist(),"error":e.detach().numpy().tolist(),"candidates":[],
            "budget":{"prediction_control_steps":0,"prediction_substeps":0,"failed_candidates":0},
            "target_zero_shortcut":bool(torch.count_nonzero(expected)==0)}
        if result["target_zero_shortcut"]:
            result["candidates"]=[{"index":0,"kind":"exact_target_zero","valid":True,"terminal_error":[0.]*8,
                "running_cost":0.,"command_energy":0.,"approach_cost":0.,"first_command":[0.,0.],
                "first_state":None,"failure_reason":None,"simulated_control_steps":0}]
        else:
            for definition in self._candidate_definitions():
                clone=copy(self.data)
                running=energy=0.; failure=None; first_command=None; first_state=None
                for j in range(125):
                    x=self._observation(clone)
                    ee=self.tensor_error(x).numpy()
                    requested=self._candidate_command(x,definition,j)
                    following,command,failed,_=self._advance(clone,requested)
                    running+=.02*float(ee@ee); energy+=.01*float(command@command)
                    result["budget"]["prediction_control_steps"]+=1
                    result["budget"]["prediction_substeps"]+=4
                    if first_command is None:
                        first_command=command.tolist(); first_state=following.tolist()
                    if failed is not None:
                        failure=failed; result["budget"]["failed_candidates"]+=1; break
                terminal=self._observation(clone); final_error=self.tensor_error(terminal).numpy()
                if definition["kind"]=="push":
                    stage=terminal[:2]-.022*np.asarray(definition["direction"])
                    gate=min(1.,float(np.square((terminal[:2]-self.goal_xy)/.05).sum()))
                    approach=.1*gate*float(np.linalg.norm(terminal[3:5]-stage))/.05
                else: approach=0.
                result["candidates"].append({**definition,"valid":failure is None,"terminal_error":final_error.tolist(),
                    "terminal_state":terminal.tolist(),"running_cost":running,"command_energy":energy,"approach_cost":approach,
                    "first_command":first_command,"first_state":first_state,"failure_reason":failure,"simulated_control_steps":j+1})
        scores=self.candidate_scores(m,result).detach().numpy()
        feasible=np.isfinite(scores)
        selected=int(np.argmin(scores)) if feasible.any() else None
        result.update(selected_index=selected,selected_command=[0.,0.] if selected is None else result["candidates"][selected]["first_command"],
            scores=[float(v) if math.isfinite(v) else None for v in scores],planning_failure=selected is None)
        result["budget"]["planning_seconds"]=time.perf_counter()-start
        result["plan_sha256"]=_digest(result)
        return result

    def calculate_action_from_M(self,M,e,*,context,controller_state=None,certificate=None):
        if controller_state is not None or certificate is not None: raise ValueError("Episode-fixed M only")
        # Plain environment call is pure; execution needs the explicit audit facade.
        return torch.tensor(self.plan(M,e,context=context)["selected_command"],dtype=torch.float64)

    def submit_plan_for_execution(self,plan):
        """Explicit facade side effect; counts even plans later not executed."""
        plan=deepcopy(plan); checksum=plan.pop("plan_sha256",None)
        if checksum!=_digest(plan): raise ValueError("Plan was modified")
        plan["plan_sha256"]=checksum
        if plan["initial_integration_sha256"]!=hashlib.sha256(self._integration().tobytes()).hexdigest() or plan["source_sha256"]!=self.source_sha256 or plan["model_sha256"]!=self.xml_sha256 or not np.array_equal(plan["goal_xy"],self.goal_xy):
            raise ValueError("Stale plan or wrong protocol")
        if self._execution_mode not in (None,"metric"): raise ValueError("Cannot mix metric/external control in an episode")
        if self._episode_metric is not None and not np.array_equal(plan["M"],self._episode_metric): raise ValueError("Cannot change executed M mid-episode")
        self._budget["plan_queries"]+=1
        for key in ("prediction_control_steps","prediction_substeps","failed_candidates","planning_seconds"):
            self._budget[key]+=plan["budget"][key]
        self._plan_log.append({"plan_sha256":checksum,"at_step":self._elapsed_steps,"budget":deepcopy(plan["budget"]),"executed":False})
        plan["audit_index"]=len(self._plan_log)-1
        self._pending_plan=plan

    def _info(self):
        now=self._elapsed_steps*.02; x=self._observation(self.data)
        physical_failure=self._failure_reason(x)
        failure=bool(physical_failure or self._planning_failure)
        good=self.is_success(x); hold=0. if self._in_goal_since is None else max(0.,now-self._in_goal_since)
        success=bool(self._elapsed_steps>=1500 and good and hold+1e-12>=1. and not failure)
        return {"success":success,"failure":failure,"failure_reason":physical_failure or ("no_feasible_prediction_candidate" if self._planning_failure else None),
            "legacy_position_success":bool(np.linalg.norm(x[:2]-self.goal_xy)<=.008),"in_goal":good,
            "hold_seconds":hold,"hold_condition_met":bool(good and hold+1e-12>=1.),
            "settling_time":self._in_goal_since if success else 30.,"arrival_time":self._in_goal_since+1. if success else 30.,
            "first_hold_confirmation_time":self._first_hold_confirmation,"physical_error":self.physical_cost(x),
            "terminal_physical_error":self.physical_cost(x),"surface_gap":self.surface_gap(x),
            "steps":self._elapsed_steps,"physical_steps":self._elapsed_steps,"elapsed_seconds":now,"horizon_seconds":30.,
            "execution_mode":self._execution_mode,"execution_origin":"M_candidate_policy" if self._execution_mode=="metric" else self._origin,
            "metric_fixed_in_episode":self._execution_mode=="metric","cost_convention":"positive_integral",
            "budget":deepcopy(self._budget),"state_projection_applied":False,"engine_version":mujoco.__version__}

    def step(self,action):
        if self._done: raise RuntimeError("Reset before step")
        self._assert_fixed()
        if self._failure_reason(self._observation(self.data)): raise ValueError("Current state already outside domain")
        requested=np.asarray(action,dtype=np.float64)
        if requested.shape!=(2,) or not np.isfinite(requested).all(): raise ValueError("Physical m/s command must be finite[2]")
        pending=self._pending_plan; mode="external" if pending is None else "metric"
        if self._execution_mode not in (None,mode): raise ValueError("Cannot mix executed policies in an episode")
        if pending is not None:
            if pending["initial_integration_sha256"]!=hashlib.sha256(self._integration().tobytes()).hexdigest(): raise ValueError("Pending plan state/solver memory is stale")
            if not np.array_equal(requested,pending["selected_command"]): raise ValueError("Command differs from audited plan")
            if self._episode_metric is not None and not np.array_equal(self._episode_metric,pending["M"]): raise ValueError("M must stay fixed")
        before=self._observation(self.data); integration_before=self._integration()
        good_before=self.is_success(before)
        following,applied,substep_failure,substeps=self._advance(self.data,requested,record_substeps=True)
        self.state=following.copy(); self._elapsed_steps+=1
        self._budget["real_control_steps"]+=1; self._budget["real_substeps"]+=4
        self._execution_mode=mode
        if pending is not None:
            if self._episode_metric is None: self._episode_metric=np.asarray(pending["M"],dtype=np.float64).copy()
            self._plan_log[pending["audit_index"]]["executed"]=True
            self._planning_failure=pending["planning_failure"]
        self._pending_plan=None
        now=self._elapsed_steps*.02; good=self.is_success(following)
        if not good: self._in_goal_since=None
        elif not good_before or self._in_goal_since is None: self._in_goal_since=now
        hold=0. if self._in_goal_since is None else now-self._in_goal_since
        self._hold_steps=int(round(max(0.,hold)/.02))
        if good and not substep_failure and hold+1e-12>=1. and self._first_hold_confirmation is None: self._first_hold_confirmation=now
        info=self._info()
        if substep_failure is not None:
            info.update(failure=True,failure_reason=substep_failure,success=False)
        cost=.02*self.physical_cost(before)
        terminated=bool(info["failure"] or info["success"]); truncated=bool(self._elapsed_steps>=1500 and not terminated)
        self._done=terminated or truncated
        e0=self.tensor_error(before).numpy(); e1=self.tensor_error(following).numpy(); M=self._episode_metric
        row={"step":self._elapsed_steps,"time":now,"state_before":before.tolist(),"state_after":following.tolist(),
            "requested_command":requested.tolist(),"applied_command":applied.tolist(),"cost":cost,
            "V_before":None if M is None else float(e0@M@e0),"V_after":None if M is None else float(e1@M@e1),
            "success":info["success"],"failure":info["failure"],"failure_reason":info["failure_reason"],
            "in_goal":good,"hold_seconds":info["hold_seconds"],"substeps":substeps,
            "integration_before":integration_before.tolist(),"integration_after":self._integration().tolist(),
            "plan":None if pending is None else deepcopy(pending)}
        self._trace.append(row)
        info.update(applied_action=applied.copy(),requested_action=requested.copy(),cost=cost,
            physical_terminated=terminated,physical_truncated=truncated)
        self._last_info=info
        return following.copy(),cost,terminated,truncated,deepcopy(info)

    def get_episode_audit(self):
        return {**deepcopy(self._last_info),"initial_state":self._initial_state.tolist(),"initial_seed":self._seed_record,
            "goal_xy":self.goal_xy.tolist(),"metric":None if self._episode_metric is None else self._episode_metric.tolist(),
            "trace":deepcopy(self._trace),"plan_log":deepcopy(self._plan_log),"budget":deepcopy(self._budget),
            "complete":self._done,"xml_sha256":self.xml_sha256,"source_sha256":self.source_sha256,
            "protocol_sha256":self.protocol_sha256,"training_updates":0,"registered":False,
            "certificate_claim":"candidate_only","state_feedback_not_vision":True}

    def state_dict(self):
        self._assert_fixed()
        if self._pending_plan is not None: raise ValueError("Complete/cancel pending audit before boundary snapshot")
        return {"schema":"contact-environment-integration-boundary-v1","engine_version":mujoco.__version__,
            "xml_sha256":self.xml_sha256,"source_sha256":self.source_sha256,"protocol_sha256":self.protocol_sha256,
            "integration_spec":int(self.state_spec),"integration":self._integration().copy(),
            "goal_xy":self.goal_xy.copy(),"context":self._context.copy(),"initial_state":self._initial_state.copy(),
            "initial_seed":self._seed_record,"rng":deepcopy(self.np_random.bit_generator.state),
            "elapsed_steps":self._elapsed_steps,"hold_steps":self._hold_steps,"done":self._done,
            "execution_mode":self._execution_mode,"origin":self._origin,"episode_metric":deepcopy(self._episode_metric),
            "in_goal_since":self._in_goal_since,"first_hold_confirmation":self._first_hold_confirmation,
            "planning_failure":self._planning_failure,"budget":deepcopy(self._budget),"plan_log":deepcopy(self._plan_log),
            "trace":deepcopy(self._trace),"last_info":deepcopy(self._last_info)}

    def load_state_dict(self,snapshot):
        if self._pending_plan is not None: raise ValueError("Cannot restore over pending execution")
        s=deepcopy(snapshot)
        if (s.get("schema")!="contact-environment-integration-boundary-v1" or s.get("engine_version")!=mujoco.__version__ or
            s.get("xml_sha256")!=self.xml_sha256 or s.get("source_sha256")!=self.source_sha256 or
            s.get("protocol_sha256")!=self.protocol_sha256 or s.get("integration_spec")!=int(self.state_spec)):
            raise ValueError("Snapshot protocol/version mismatch")
        integration=np.asarray(s["integration"],dtype=np.float64)
        if integration.shape!=(self.integration_state_size,) or not np.isfinite(integration).all(): raise ValueError("Bad complete integration state")
        steps=s["elapsed_steps"]
        if isinstance(steps,bool) or not isinstance(steps,int) or not 0<=steps<=1500 or len(s["trace"])!=steps: raise ValueError("Bad timing/trace length")
        temp=type(self)()
        initial=np.asarray(s["initial_state"],dtype=np.float64)
        if initial.shape!=(10,): raise ValueError("Bad initial state")
        temp.reset(seed=s["initial_seed"],options={"disk_xy":initial[:2],"pusher_xy":initial[3:5],
            "yaw":initial[2],"qvel":initial[5:],"goal_xy":s["goal_xy"],"execution_origin":s["origin"]})
        if not np.array_equal(s["context"],temp._context): raise ValueError("Initial fixed context mismatch")
        mujoco.mj_setState(temp.model,temp.data,integration,temp.state_spec)
        # Rebuild derived fields then restore the complete input/warmstart vector.
        mujoco.mj_forward(temp.model,temp.data)
        mujoco.mj_setState(temp.model,temp.data,integration,temp.state_spec)
        if not np.array_equal(temp._integration(),integration): raise ValueError("Full simulator state did not restore exactly")
        temp._assert_fixed()
        if abs(temp.data.time-steps*.02)>1e-10 or not np.array_equal(temp.data.mocap_pos[0],[*temp.goal_xy,.0008]): raise ValueError("Time/goal mocap inconsistent")
        metric=s["episode_metric"]
        if metric is not None: temp._validate_metric(torch.as_tensor(metric,dtype=torch.float64))
        mode=s["execution_mode"]
        if mode not in (None,"metric","external") or (mode=="metric")!=(metric is not None) or (steps==0)!=(mode is None): raise ValueError("Mode/M mismatch")
        previous=initial; since=0. if temp.is_success(initial) else None; first=None
        for index,row in enumerate(s["trace"],1):
            before=np.asarray(row["state_before"]); after=np.asarray(row["state_after"])
            if row["step"]!=index or not np.array_equal(before,previous) or len(row["substeps"])!=4 or after.shape!=(10,) or not np.isfinite(after).all(): raise ValueError("Broken physical trace chain")
            if not np.array_equal(after,row["substeps"][-1]["state"]) or not math.isclose(row["cost"],.02*temp.physical_cost(before),rel_tol=1e-12,abs_tol=1e-14): raise ValueError("Trace endpoint/cost mismatch")
            if not temp.is_success(after): since=None
            elif since is None: since=index*.02
            hold=0. if since is None else index*.02-since
            if temp.is_success(after) and not row["failure"] and hold+1e-12>=1. and first is None: first=index*.02
            if abs(row["hold_seconds"]-hold)>1e-12: raise ValueError("Hold trace mismatch")
            for key,x in (("V_before",before),("V_after",after)):
                e=temp.tensor_error(x).numpy(); expected=None if metric is None else float(e@np.asarray(metric)@e)
                if expected is None and row[key] is not None or expected is not None and not math.isclose(row[key],expected,rel_tol=1e-12,abs_tol=1e-12): raise ValueError("Certificate mismatch")
            if row["plan"] is not None:
                plan=deepcopy(row["plan"]); plan.pop("audit_index",None); checksum=plan.pop("plan_sha256",None)
                if checksum!=_digest(plan) or not np.array_equal(plan["M"],metric) or not np.array_equal(plan["selected_command"],row["requested_command"]): raise ValueError("Plan provenance/action mismatch")
            elif mode=="metric": raise ValueError("Metric step missing plan")
            if row["failure"] and index!=steps: raise ValueError("History continues after failure")
            previous=after
        if not np.array_equal(previous,temp._observation(temp.data)) or s["in_goal_since"]!=since or s["first_hold_confirmation"]!=first: raise ValueError("Final state/hold mismatch")
        if steps and not np.array_equal(integration,s["trace"][-1]["integration_after"]): raise ValueError("Final solver state mismatch")
        budget=s["budget"]; logs=s["plan_log"]
        if budget["plan_queries"]!=len(logs) or budget["real_control_steps"]!=steps or budget["real_substeps"]!=4*steps: raise ValueError("Budget counters mismatch")
        for key in ("prediction_control_steps","prediction_substeps","failed_candidates","planning_seconds"):
            if not math.isclose(budget[key],sum(p["budget"][key] for p in logs),rel_tol=1e-12,abs_tol=1e-12): raise ValueError("Prediction budget mismatch")
        if sum(bool(p["executed"]) for p in logs)!=(steps if mode=="metric" else 0): raise ValueError("Executed plan count mismatch")
        rng=np.random.default_rng(); rng.bit_generator.state=deepcopy(s["rng"])
        temp._np_random=rng; temp.state=temp._observation(temp.data)
        for attr,key in (("_elapsed_steps","elapsed_steps"),("_hold_steps","hold_steps"),("_done","done"),
            ("_execution_mode","execution_mode"),("_origin","origin"),("_episode_metric","episode_metric"),
            ("_in_goal_since","in_goal_since"),("_first_hold_confirmation","first_hold_confirmation"),
            ("_planning_failure","planning_failure"),("_budget","budget"),("_plan_log","plan_log"),("_trace","trace"),("_last_info","last_info")):
            setattr(temp,attr,deepcopy(s[key]))
        expected_done=bool(steps>=1500 or (steps and s["trace"][-1]["failure"]))
        if temp._done!=expected_done or temp._hold_steps!=int(round((0. if since is None else steps*.02-since)/.02)): raise ValueError("Done/hold counters mismatch")
        self.__dict__.update(temp.__dict__)


class ContactMFacade(PortableMFacade):
    """Only this explicit audit adapter mutates pending plan/budget records."""
    def __init__(self,env):
        super().__init__(env,name="isolated_contact_single_disk_gap_v2")
        self.description=replace(self.description,control_call_is_pure=False,
            control_call_side_effects=("audit_pending_plan",),
            gradients=replace(self.description.gradients,candidate_scores_wrt_certificate=True),
            supported_algorithms=(),support_level="engineering_only_control_and_step",
            missing_capabilities=("formal PPO/checkpoint acceptance not run","visual representation","differentiable hard contact dynamics"))

    def calculate_action_from_M(self,M,e,*,context,controller_state=None,certificate=None):
        if controller_state is not None or certificate is not None: raise ValueError("Episode-fixed M only")
        plan=self.env.plan(M,e,context=context)
        self.env.submit_plan_for_execution(plan)
        return torch.tensor(plan["selected_command"],dtype=torch.float64)

    def candidate_scores(self,M,plan): return self.env.candidate_scores(M,plan)
