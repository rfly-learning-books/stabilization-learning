"""Frozen joint-Push policies in actuated SO101 MuJoCo, plus deployment bundles.

The simulator receives pose measurements, not RGB localization. No policy update
or checkpoint selection occurs here. The preserved rollout is not interruptible
by the Python guard: its caller/supervisor must enforce the original deadline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from stabilization_gym.tasks_v2.push import JointPushSelector, vendor_modules
from stabilization_learning.joint_v2.push import _Events, _json, _sha

CASE_SEED = 991000
METHODS = (("random_initial","initial.pt"),("selected","selected.pt"),
           ("final","final.pt"),("identity","initial.pt"))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_frozen_inputs(run_dir, group_manifest):
    """Check the complete declared seed group, all snapshots, sources and banks."""
    root=Path(run_dir).resolve();barrier_path=Path(group_manifest).resolve();barrier=_read(barrier_path)
    if barrier.get("schema")!="joint_push_group_frozen_v2":raise ValueError("All-seed freeze barrier required")
    seen=[];matches=[]
    for record in barrier["runs"]:
        member=Path(record["path"]).resolve()
        for filename,key in (("FROZEN.json","frozen_sha256"),("selected.pt","selected_sha256"),("protocol.json","protocol_sha256")):
            if _sha(member/filename)!=record[key]:raise ValueError("Frozen group changed: "+str(member/filename))
        protocol=_read(member/"protocol.json");frozen=_read(member/"FROZEN.json")
        if frozen.get("schema")!="joint_push_v2" or frozen.get("selected_update",0)<1:
            raise ValueError("Every selected model must follow a real update")
        if frozen["updates"]!=protocol["config"]["updates"] or frozen["protocol_sha256"]!=record["protocol_sha256"]:
            raise ValueError("Incomplete or incompatible frozen run")
        for filename,digest in frozen["checkpoint_sha256"].items():
            if Path(filename).name!=filename or _sha(member/filename)!=digest:raise ValueError("Frozen snapshot changed")
        for source,digest in protocol["source_sha256"].items():
            if _sha(source)!=digest:raise ValueError("Frozen scientific source changed: "+source)
        if _sha(protocol["time_bank"])!=protocol["time_bank_sha256"]:raise ValueError("Frozen bank changed")
        seen.append(protocol["config"]["seed"])
        if member==root:matches.append((protocol,frozen))
    if len(matches)!=1 or len(set(seen))!=len(seen) or sorted(seen)!=sorted(barrier["expected_seeds"]):
        raise ValueError("Run is not in the complete declared seed group")
    return matches[0][0],matches[0][1],barrier


def load_actor(run_dir, frozen, method):
    filename=dict(METHODS)[method];path=Path(run_dir)/filename
    if _sha(path)!=frozen["checkpoint_sha256"][filename]:raise ValueError("Frozen checkpoint changed")
    vendor=vendor_modules();ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("origin")!="random":raise ValueError("Actuated evaluation requires this random-origin run")
    actor=vendor["Actor"](vendor["ActorConfig"](**ck["cfg"]));actor.load_state_dict(ck["actor"]);actor.eval()
    if method=="identity":
        def identity(D,condition):
            matrix=torch.eye(3,device=D.device,dtype=D.dtype).expand(*D.shape[:-1],3,3)
            return matrix,matrix
        actor.conditioned=identity
    return actor


def actuated_config(bank, method="selected"):
    vendor_modules()
    import so101_multishape as shapes
    from so101_timeaware.timing import validate_arm_timing
    validate_arm_timing(bank.config)
    if not np.isclose(bank.config.length_scale,.05,rtol=0,atol=1e-12):
        raise ValueError("Preserved arm physical-error telemetry uses the declared 0.05-m scale")
    return shapes.c.Config(plant="arm",metric=method,speed=bank.config.transit_speed,
                           max_time=bank.config.max_seconds,max_actions=bank.config.max_actions)


@contextmanager
def command_telemetry(shapes):
    """Log actual elapsed intervals around the unchanged step; no dynamics edits."""
    original=shapes.c.Plant;records=[];instances=[]
    class RecordedPlant(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self._joint_v2_closed=False;instances.append(self)
        def step(self,command):
            before=float(self.data.time);phase=self.phase;value=np.asarray(command,float).copy()
            try:return super().step(command)
            finally:
                dt=float(self.data.time)-before
                if dt>0:records.append(dict(time=before+dt,dt=dt,phase=phase,command=value.tolist()))
        def close(self):
            if not self._joint_v2_closed:
                self._joint_v2_closed=True;return super().close()
    shapes.c.Plant=RecordedPlant
    try:yield records
    finally:
        shapes.c.Plant=original
        for plant in instances:plant.close()


def audit_telemetry(path, actor, bank, shape):
    """Recompute current AND successor M/V at every saved 20-ms state in NumPy.

The trace also contains the final safety-stop state. It is not a certificate
check at every 2.5-ms MuJoCo integration substep, nor a domain-wide proof.
"""
    with np.load(path,allow_pickle=False) as saved:
        times=np.asarray(saved["time"],float);poses=np.asarray(saved["pose"],float)
        recorded_v=np.asarray(saved["V"],float);recorded_error=np.asarray(saved["error"],float)
    if len(times)!=len(poses) or len(times)!=len(recorded_v) or not np.isfinite(poses).all() or not np.isfinite(times).all():
        raise RuntimeError("Invalid complete telemetry")
    if times[0]!=0 or np.any(np.diff(times)<=0):raise RuntimeError("Telemetry clock is not strictly increasing from zero")
    model=vendor_modules()["Model"](bank,"cpu");index=bank.names.index(shape)
    e_parts=[];m_parts=[]
    with torch.no_grad():
        condition=actor.encoder(model.C_encoded[[index]],model.valid[[index]],ell=bank.config.length_scale)
        for segment in np.array_split(poses,max(1,(len(poses)+1023)//1024)):
            error=model.coordinates(torch.as_tensor(segment,dtype=torch.float32))
            _,matrix=actor.conditioned(error,condition.expand(len(error),-1))
            e_parts.append(error.numpy().astype(float));m_parts.append(matrix.numpy().astype(float))
    error=np.concatenate(e_parts);matrices=np.concatenate(m_parts)
    values=np.einsum("ti,tij,tj->t",error,matrices,error)
    physical=np.sum((poses[:,:2]/bank.config.length_scale)**2,axis=1)+2*np.sin(poses[:,2]/2)**2
    if not np.allclose(values,recorded_v,atol=1e-6,rtol=2e-5):raise RuntimeError("Independent full V recomputation failed")
    if not np.allclose(physical,recorded_error,atol=1e-10,rtol=1e-10):raise RuntimeError("Physical error recomputation failed")
    integral=float(np.sum(.5*(physical[:-1]+physical[1:])*np.diff(times)))
    result=dict(passed=True,samples=len(times),transitions=max(0,len(times)-1),
        max_V_absolute_difference=float(np.max(abs(values-recorded_v))),
        max_V_relative_difference=float(np.max(abs(values-recorded_v)/np.maximum(abs(recorded_v),1e-12))),
        physical_error_integral=integral,
        actual_V_increase_fraction=float(np.mean(np.diff(values)>1e-7)) if len(values)>1 else 0.,
        scope="all recorded 20-ms endpoints plus final stop, not every 2.5-ms physics substep or a stability proof")
    audit_path=Path(path).with_name(Path(path).stem+"_certificate.npz")
    if audit_path.exists():raise FileExistsError("Preserve prior certificate audit")
    np.savez_compressed(audit_path,time=times,error_coordinates=error,M=matrices,V_numpy=values,
                        V_recorded=recorded_v,delta_V=np.diff(values),physical_error=physical)
    result["certificate_file"]=audit_path.name;result["certificate_sha256"]=_sha(audit_path)
    return result


def run_actuated_episode(actor,bank,shape,initial,case_seed,folder,*,method="selected",video=False):
    """One original-horizon rollout; caller guards before/after this noninterruptible call."""
    vendor_modules();import so101_multishape as shapes
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    cfg=actuated_config(bank,method);trace=folder/"trajectory.npz"
    with shapes.install(shape):
        policy=JointPushSelector(bank,shape,actor)
        with command_telemetry(shapes) as commands:
            result=shapes.c.rollout(cfg,np.asarray(initial,float),int(case_seed),policy=policy,telemetry=trace,
                                    video=folder/"episode.mp4" if video else None)
    phases={}
    for item in commands:phases[item["phase"]]=phases.get(item["phase"],0.)+item["dt"]
    if not np.isclose(sum(phases.values()),result["sim_time_s"],atol=1e-7,rtol=1e-9):
        raise RuntimeError("Command telemetry does not cover the full physical clock")
    push_input=sum(float(np.dot(r["command"],r["command"]))*r["dt"] for r in commands if r["phase"]=="PUSH")
    all_input=sum(float(np.dot(r["command"],r["command"]))*r["dt"] for r in commands)
    _json(folder/"COMMANDS.json",dict(records=commands,units="commanded planar pusher speed m/s; not electrical energy"))
    audit=audit_telemetry(trace,actor,bank,shape)
    if not np.isclose(audit["physical_error_integral"],result["integrated_error"],atol=1e-10,rtol=1e-10):
        raise RuntimeError("Recorded error integral disagrees with independent arithmetic")
    result.update(shape=shape,method=method,stage_seconds=phases,input_square_push=push_input,
        input_square_all_commands=all_input,input_units="m^2/s commanded-speed proxy; not motor energy",
        trace_sha256=_sha(trace),audit=audit,simulation_only=True,pose_input_not_rgb_localization=True)
    _json(folder/"RESULT.json",result)
    return result


def create_deployment_bundle(run_dir,protocol,frozen,output,group_manifest):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);root=Path(run_dir)
    if _sha(root/"selected.pt")!=frozen["sha256"]:raise ValueError("Selected policy changed")
    if _sha(protocol["time_bank"])!=protocol["time_bank_sha256"]:raise ValueError("Bank changed")
    shutil.copy2(root/"selected.pt",out/"metric.pt");shutil.copy2(protocol["time_bank"],out/"time_bank.json")
    manifest=dict(schema="joint_so101_deployment_v2",seed=protocol["config"]["seed"],selected_update=frozen["selected_update"],
        checkpoint_sha256=_sha(out/"metric.pt"),time_bank_sha256=_sha(out/"time_bank.json"),
        group_manifest_sha256=_sha(group_manifest),training_protocol_sha256=frozen["protocol_sha256"],
        source_sha256=protocol["source_sha256"],simulation_only=True,hardware_verified=False,rgb_localization=False,
        input="planar pose measurement/estimate, pusher pose, contact memory, calibrated shape responses",
        policy="stabilization_gym.tasks_v2.push.JointPushSelector; same greedy logits as training validation",
        calibrated_heldout_shapes=protocol["config"]["test_shapes"],
        scope="held-out shapes have capability calibration; pose measurement is supplied by the simulator",
        command="python -m extensions.joint_certificate_v2.arm --deploy BUNDLE --shape T --output NEW_EPISODE")
    _json(out/"DEPLOYMENT.json",manifest)
    return manifest


def paired_summary(rows,baseline="random_initial",method="selected",*,heldout=None):
    base={(r["shape"],r["case_index"]):r for r in rows if r["method"]==baseline and (heldout is None or r["heldout"]==heldout)}
    comparison={(r["shape"],r["case_index"]):r for r in rows if r["method"]==method and (heldout is None or r["heldout"]==heldout)}
    shared=sorted(set(base)&set(comparison));successful=[k for k in shared if base[k]["success"] and comparison[k]["success"]]
    bt=[base[k]["sim_time_s"] for k in successful];mt=[comparison[k]["sim_time_s"] for k in successful]
    return dict(baseline=baseline,method=method,heldout=heldout,paired_cases=len(shared),common_success=len(successful),
        baseline_common_success_seconds=None if not bt else float(np.mean(bt)),
        method_common_success_seconds=None if not mt else float(np.mean(mt)),
        paired_seconds_change=None if not bt else float(np.mean(np.asarray(mt)-bt)),
        lost_success=sum(base[k]["success"] and not comparison[k]["success"] for k in shared),
        gained_success=sum(not base[k]["success"] and comparison[k]["success"] for k in shared))


def evaluate_frozen_arm(run_dir,group_manifest,output,*,repeats=4,guard=lambda:None):
    guard()
    if not isinstance(repeats,int) or isinstance(repeats,bool) or repeats<1:raise ValueError("Positive repeats required")
    protocol,frozen,barrier=validate_frozen_inputs(run_dir,group_manifest)
    root=Path(run_dir).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError("Preserve completed and partial arm tests")
    vendor=vendor_modules();bank=vendor["Bank"](protocol["time_bank"])
    cfg=actuated_config(bank);seed=protocol["config"]["seed"]
    names=protocol["config"]["train_shapes"]+protocol["config"]["validation_shapes"]+protocol["config"]["test_shapes"]
    import so101_multishape as shapes
    cases=[dict(index=i,seed=CASE_SEED+i,initial=shapes.c.cases(CASE_SEED+i,1,extent=.012,angle=.35)[0].tolist()) for i in range(repeats)]
    out.mkdir(parents=True)
    _json(out/"protocol.json",dict(schema="joint_so101_arm_test_v2",training_seed=seed,group_manifest=str(Path(group_manifest).resolve()),
        group_sha256=_sha(group_manifest),training_protocol_sha256=frozen["protocol_sha256"],
        methods=[x[0] for x in METHODS],shapes=names,cases=cases,config=asdict(cfg),
        evaluator_source_sha256={str(Path(__file__).resolve()):_sha(__file__)},selection_allowed=False,
        video=dict(training_seed=81,shape="T",case_index=0,method="selected",chosen_before_results=True),
        deadline_note="Guard runs between rollouts; external supervisor must interrupt a rollout at the original wall-clock deadline",
        simulation_only=True,rgb_localization=False,
        audit_scope="complete recorded 20-ms successor V; not all 2.5-ms substeps; no nonlinear-domain stability proof"))
    events=_Events(out)
    deployment=out/f"deployment_so101_seed{seed}"
    create_deployment_bundle(root,protocol,frozen,deployment,group_manifest)
    events.emit("deployment_packaged",seed=seed,path=str(deployment))
    rows=[]
    try:
        for method,filename in METHODS:
            actor=load_actor(root,frozen,method)
            for shape in names:
                for case in cases:
                    guard()
                    if (out/"STOP").exists():raise RuntimeError("STOP present")
                    video=seed==81 and shape=="T" and case["index"]==0 and method=="selected"
                    folder=out/"episodes"/method/shape/f"{case['index']:03d}"
                    result=run_actuated_episode(actor,bank,shape,case["initial"],case["seed"],folder,method=method,video=video)
                    result.update(case_index=case["index"],heldout=shape in protocol["config"]["test_shapes"],
                                  checkpoint_sha256=frozen["checkpoint_sha256"][filename])
                    # Success/failure/time/error/input each remain visible; no test-time tuning.
                    w=protocol["costs"]
                    result["joint_cost"]=w["time"]*result["sim_time_s"]/bank.config.time_reference+w["error"]*result["integrated_error"]+w["switch"]*result["mode_changes"]+w["input_square"]*result["input_square_push"]-w["success_bonus"]*result["success"]+w["failure_penalty"]*(not result["success"])
                    _json(folder/"RESULT.json",result);rows.append(result)
                    events.emit("arm_episode_complete",method=method,shape=shape,case_index=case["index"],
                        success=result["success"],reason=result["reason"],seconds=result["sim_time_s"],
                        integrated_error=result["integrated_error"],audit_max_V_difference=result["audit"]["max_V_absolute_difference"])
                    guard()
        groups={}
        for method,_ in METHODS:
            for heldout in (None,True,False):
                group=[r for r in rows if r["method"]==method and (heldout is None or r["heldout"]==heldout)]
                key=method+("/all" if heldout is None else "/heldout" if heldout else "/seen")
                groups[key]=dict(episodes=len(group),successes=sum(r["success"] for r in group),success_rate=float(np.mean([r["success"] for r in group])),
                    mean_seconds=float(np.mean([r["sim_time_s"] for r in group])),mean_error=float(np.mean([r["integrated_error"] for r in group])),
                    mean_input_square_push=float(np.mean([r["input_square_push"] for r in group])),mean_joint_cost=float(np.mean([r["joint_cost"] for r in group])),
                    failure_reasons={reason:sum(r["reason"]==reason for r in group if not r["success"]) for reason in sorted({r["reason"] for r in group if not r["success"]})})
        summary=dict(schema="joint_so101_arm_results_v2",training_seed=seed,groups=groups,
            paired=[paired_summary(rows,baseline,method,heldout=heldout) for baseline in ("random_initial","identity") for method in ("selected","final") for heldout in (None,True)],
            deployment=str(deployment),audit=dict(passed=True,transitions=sum(r["audit"]["transitions"] for r in rows),
                max_V_absolute_difference=max(r["audit"]["max_V_absolute_difference"] for r in rows)),
            video=str(out/"episodes/selected/T/000/episode.mp4") if seed==81 and "T" in names else None,
            simulation_only=True,unique_initial_cases_per_training_seed=repeats*len(names),
            note="Starts shared across methods/training seeds. Across-seed SD is not a confidence interval. Calibration on held-out shapes is disclosed.")
        validate_frozen_inputs(root,group_manifest)
        _json(out/"RESULTS.json",dict(summary=summary,episodes=rows));_json(out/"SUMMARY.json",summary)
        _json(out/"FINAL_STATE.json",dict(status="complete",selection_allowed=False,episodes=len(rows)))
        events.emit("arm_complete",episodes=len(rows),audit=summary["audit"])
        return summary
    except BaseException as exc:
        _json(out/"FINAL_STATE.json",dict(status="stopped_or_failed",error=repr(exc),completed_episodes=len(rows),partial_results_preserved=True))
        events.emit("arm_stopped_or_failed",error=repr(exc),completed_episodes=len(rows))
        raise


def deploy_episode(bundle,shape,output,*,case_index=0,video=True,guard=lambda:None):
    guard();bundle=Path(bundle);manifest=_read(bundle/"DEPLOYMENT.json")
    if manifest.get("schema")!="joint_so101_deployment_v2":raise ValueError("Unknown deployment bundle")
    if _sha(bundle/"metric.pt")!=manifest["checkpoint_sha256"] or _sha(bundle/"time_bank.json")!=manifest["time_bank_sha256"]:
        raise ValueError("Deployment bundle changed")
    for source,digest in manifest["source_sha256"].items():
        if _sha(source)!=digest:raise ValueError("Deployment source changed")
    vendor=vendor_modules();bank=vendor["Bank"](bundle/"time_bank.json")
    if shape not in bank.names:raise ValueError("Shape absent from calibrated deployment bank")
    ck=torch.load(bundle/"metric.pt",map_location="cpu",weights_only=False)
    actor=vendor["Actor"](vendor["ActorConfig"](**ck["cfg"]));actor.load_state_dict(ck["actor"])
    import so101_multishape as shapes
    seed=CASE_SEED+case_index;q=shapes.c.cases(seed,1,extent=.012,angle=.35)[0]
    result=run_actuated_episode(actor,bank,shape,q,seed,output,method="selected",video=video)
    guard();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy",required=True);parser.add_argument("--shape",default="T")
    parser.add_argument("--output",required=True);parser.add_argument("--case-index",type=int,default=0)
    parser.add_argument("--no-video",action="store_true");parser.add_argument("--max-hours",type=float,default=1.)
    args=parser.parse_args()
    if not np.isfinite(args.max_hours) or args.max_hours<=0:parser.error("Positive finite max-hours required")
    if args.case_index<0:parser.error("Nonnegative case-index required")
    deadline=time.monotonic()+args.max_hours*3600
    def guard():
        if time.monotonic()>=deadline or (Path(args.output)/"STOP").exists():raise RuntimeError("Deadline or STOP")
    result=deploy_episode(args.deploy,args.shape,args.output,case_index=args.case_index,video=not args.no_video,guard=guard)
    print(json.dumps({k:v for k,v in result.items() if k not in ("transfers","push_delta_V")},indent=2))


if __name__=="__main__":main()
