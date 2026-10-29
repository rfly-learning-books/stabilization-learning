"""Offline audit of the independently migrated original visual controller.

No environment is constructed and no historical RGB frame is invented.
Saved telemetry is replayed numerically; this is not new closed-loop testing.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stabilization-gym"))
from stabilization_gym.envs.tasks.original_visual import OriginalVisualAdapter

# Declared before observing migration results. Saved float32 evaluation had
# independently measured maxima 3.01e-5 action, 5.78e-6 V, 8.72e-7 P0 step.
LIMITS = dict(normalized_action_abs=5e-5, certificate_abs=1e-5,
              physical_step_abs=1e-6, goal_depth_abs=1e-5,
              reference_M_abs=2e-6, reference_J_abs=2e-6,
              reference_H_abs=2e-6, gradient_relative=5e-5,
              gradient_absolute=1e-8)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maximum(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def resources(output, device):
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < 50:
        raise RuntimeError(f"Storage boundary: {free:.3f} GiB free")
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) / 2**30
    if total > 35:
        raise RuntimeError("Output exceeds 35 GiB")
    row = dict(free_gib=free, output_gib=total)
    if device.startswith("cuda"):
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        used, capacity, util = map(float, raw.split(","))
        row.update(gpu_used_mib=used, gpu_total_mib=capacity, gpu_utilization=util,
                   torch_allocated_mib=torch.cuda.memory_allocated()/2**20,
                   torch_peak_mib=torch.cuda.max_memory_allocated()/2**20)
        if used / capacity > .65:
            raise RuntimeError(f"GPU boundary: {used}/{capacity} MiB")
    return row


def preserved_plants(adapter):
    """Load only exact preserved kinematics AST, avoiding unavailable scenes."""
    path = adapter.legacy_root / "src/room_env.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ("PlantP0", "PlantP1")]
    if len(nodes) != 2:
        raise RuntimeError("Missing exact P0/P1 class definitions")
    common = adapter.core.legacy.module(adapter.cfg, "common")
    ns = dict(np=np, math=math, DT=common.DT, TAU=common.TAU,
              ACTION_SCALE=common.ACTION_SCALE, rot_z=common.rot_z)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return {"P0": ns["PlantP0"], "P1": ns["PlantP1"]}, sha(path)


def audit_gradients(adapter, matrix, error, jacobian):
    gen = torch.Generator().manual_seed(20261022)
    M = torch.tensor(matrix, dtype=torch.float64)[None]
    J = torch.tensor(jacobian, dtype=torch.float64)[None]
    e = torch.tensor(error, dtype=torch.float64)[None]
    records = []
    for scale in (.01, 20.):
        m = M.clone().requires_grad_(True)
        x = (e * scale).clone().requires_grad_(True)
        weights = torch.tensor([[.23, -.41, .37, .19]], dtype=torch.float64)
        value = (adapter.calculate_action_from_M(m, x, J) * weights).sum()
        gm, ge = torch.autograd.grad(value, (m, x))
        actual = adapter.calculate_action_from_M(m, x, J).detach()
        saturated = bool(actual[0, :3].norm() > .39999 or actual[0, 3].abs() > .34999)
        for variable in ("M", "e"):
            for index in range(8):
                direction = torch.randn(m.shape if variable == "M" else x.shape,
                                        dtype=torch.float64, generator=gen)
                if variable == "M":
                    direction = (direction + direction.transpose(-1, -2)) / 2
                direction /= direction.norm()
                analytic = float(((gm if variable == "M" else ge) * direction).sum())
                estimates = []
                for eps in (1e-4, 5e-5):
                    plus_m = m.detach() + eps * direction if variable == "M" else m.detach()
                    minus_m = m.detach() - eps * direction if variable == "M" else m.detach()
                    plus_e = x.detach() + eps * direction if variable == "e" else x.detach()
                    minus_e = x.detach() - eps * direction if variable == "e" else x.detach()
                    a = (adapter.calculate_action_from_M(plus_m, plus_e, J) * weights).sum()
                    b = (adapter.calculate_action_from_M(minus_m, minus_e, J) * weights).sum()
                    estimates.append(float((a-b)/(2*eps)))
                absolute = abs(analytic-estimates[-1])
                relative = absolute / max(abs(analytic), abs(estimates[-1]), 1e-8)
                records.append(dict(variable=variable, scale=scale, saturated=saturated,
                                    direction=index, analytic=analytic, central_differences=estimates,
                                    absolute_error=absolute, relative_error=relative,
                                    passed=absolute <= LIMITS["gradient_absolute"] or relative <= LIMITS["gradient_relative"]))
    return dict(passed=all(r["passed"] for r in records), directions=len(records),
                M_gradient_nonzero=any(abs(r["analytic"])>1e-9 for r in records if r["variable"]=="M"),
                e_gradient_nonzero=any(abs(r["analytic"])>1e-9 for r in records if r["variable"]=="e"),
                max_absolute_error=max(r["absolute_error"] for r in records),
                max_relative_error=max(r["relative_error"] for r in records), records=records)


def telemetry(adapter, plants, path, starts):
    row = read(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
        a = {key: saved[key] for key in saved.files}
    e = torch.from_numpy(a["features"].astype(np.float64))
    m = torch.from_numpy(a["M"].astype(np.float64))
    j = torch.from_numpy(a["J_reference"].astype(np.float64))
    physical = adapter.calculate_action_from_M(m, e[:-1], j).detach().numpy()
    normalized = physical / np.asarray(adapter.physical_scales)
    values = adapter.certificate(m, e).detach().numpy()
    action_error = maximum(normalized, a["action"])
    value_error = maximum(values, a["V"])
    _, h = adapter.core.policy(e[:1], j, m)
    h_error = maximum(h.numpy(), a["H"])
    plant = plants[row["plant"]]
    first = starts[row["index"]]["offset"]
    state = dict(position=np.asarray(first[:3], float), yaw=float(first[3]), body0=np.eye(3), **plant.initial_extra())
    allstep = []
    localstep = []
    for index, command in enumerate(a["action"]):
        # Preserve P1 hidden actuator state, but reset observed pose to recorded
        # values for the independent local one-step residual.
        local = {**state, "position": a["offset"][index, :3].astype(float),
                 "yaw": float(a["offset"][index, 3])}
        one = plant.step_state(local, command)
        state = plant.step_state(state, command)
        for item, target in ((one, localstep), (state, allstep)):
            residual = np.r_[item["position"], item["yaw"]] - a["offset"][index+1]
            residual[3] = (residual[3]+math.pi)%(2*math.pi)-math.pi
            target.append(float(np.max(np.abs(residual))))
    E = (a["offset"][:, :3].astype(float)**2).sum(1) + (a["offset"][:, 3].astype(float)/math.radians(25))**2
    expected_cost = .1*(E[1:]+.02*(a["action"].astype(float)**2).sum(1))
    if row["status"] in ("collision", "outside"):
        expected_cost[-1] += 20
    good = (np.linalg.norm(a["offset"][1:, :3], axis=1)<.02)&(np.abs(a["offset"][1:, 3])<math.radians(1))
    checks = dict(action=action_error<=LIMITS["normalized_action_abs"],
                  certificate=value_error<=LIMITS["certificate_abs"],
                  local_physics=max(localstep)<=LIMITS["physical_step_abs"],
                  accumulated_physics=max(allstep)<=LIMITS["physical_step_abs"],
                  hold=(not row["success"] or bool(good[-11:].all())),
                  frozen_start=maximum(a["offset"][0], first)<1e-6,
                  physical_cost=bool(np.allclose(expected_cost,a["cost"],atol=2e-5,rtol=2e-5)),
                  command_limits=bool(np.linalg.norm(a["action"][:, :3],axis=1).max()<=1.00001 and np.abs(a["action"][:, 3]).max()<=1.00001))
    return dict(file=path.name, method=row["method"], plant=row["plant"], index=row["index"],
                scene_id=row["scene_id"], goal_id=row["goal_id"], steps=len(a["action"]),
                historical_status=row["status"], normalized_action_max_abs=action_error,
                physical_action_max_abs=maximum(physical,a["action"]*np.asarray(adapter.physical_scales)),
                V_max_abs=value_error, H_max_abs=h_error, physics_local_max_abs=max(localstep),
                physics_accumulated_max_abs=max(allstep), cost_max_abs=maximum(expected_cost,a["cost"]),
                checks=checks, passed=all(checks.values())), a


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(schema=1, started_utc=datetime.now(timezone.utc).isoformat(),
                  scope="offline numerical migration; no renderer, new rollout, new success rate, or online training",
                  limits=LIMITS, device=args.device, passed=False, errors=[],
                  script_sha256=sha(__file__))
    try:
        report["resources_start"] = resources(output,args.device)
        if args.device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(.50)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        adapter = OriginalVisualAdapter(ROOT)
        adapter_path = ROOT / "stabilization-gym/stabilization_gym/envs/tasks/original_visual.py"
        report["adapter_sha256"] = sha(adapter_path)
        source_manifest = read(ROOT/"migration/visual_source_bundle.json")
        for row in source_manifest["source_records"]:
            if sha(row["destination"]) != row["sha256"]:
                raise RuntimeError("Bundled source/dependency changed: "+row["relative"])
        init_manifest = read(ROOT/"migration/visual_initialization_bundle.json")
        for row in init_manifest["entries"]:
            path=adapter.bundle/row["relative"]
            if sha(path)!=row["sha256"]:
                raise RuntimeError("Bundled model/evidence changed: "+row["relative"])
        report["verified_bundle_files"]=len(source_manifest["source_records"])+len(init_manifest["entries"])
        ae, metric = adapter.load_models("best.pt", args.device)
        ae.requires_grad_(False); metric.requires_grad_(False)
        depth = adapter.core.FastDepth(adapter.cfg,args.device)
        plants, plant_sha = preserved_plants(adapter)
        report["preserved_physics_source_sha256"] = plant_sha
        starts=read(adapter.bundle/"evaluation_starts.json")["test"]
        paths=sorted((adapter.bundle/"regression/fixed_test").glob("*.json"))
        needed={(read(path)["scene_id"],read(path)["goal_id"]) for path in paths}
        full_paths=[]
        if args.full_telemetry:
            full_paths=[p for p in sorted(args.full_telemetry.glob("*.json"))
                        if p.name.startswith(("learned_","identity_m_"))]
            needed.update((read(p)["scene_id"],read(p)["goal_id"]) for p in full_paths)
        needed.update((family+"_test_00",0) for family in ("classroom","office","meeting","library"))
        refs={}; depth_rows=[]
        for index,path in enumerate(sorted((adapter.bundle/"references").glob("*.npz"))):
            with np.load(path,allow_pickle=False) as data:
                rgb=data["rgb"]; expected=data["depth"]; rotation=data["camera_from_body"]
            fresh=depth.predict(rgb)
            delta=maximum(fresh,expected)
            depth_rows.append(dict(file=path.name,max_abs=delta,passed=delta<=LIMITS["goal_depth_abs"]))
            key=path.stem.rsplit("_",1); key=(key[0],int(key[1]))
            if key in needed:
                for identity in (False,True):
                    refs[key+(identity,)]=adapter.reference(ae,metric,rgb,rotation,depth=fresh,
                                                          device=args.device,identity=identity)
            if (index+1)%24==0:
                print(json.dumps(dict(event="depth_recomputed",references=index+1)),flush=True)
                resources(output,args.device)
        report["goal_depth"]=dict(references=len(depth_rows),max_abs=max(r["max_abs"] for r in depth_rows),
                                  passed=all(r["passed"] for r in depth_rows),records=depth_rows)
        records=[]; reference_rows=[]
        gradient_data=None
        for path in paths:
            record,a=telemetry(adapter,plants,path,starts)
            key=(record["scene_id"],record["goal_id"],record["method"]=="identity_m")
            ref=refs[key]
            vals=dict(M=maximum(ref.m[0].detach().cpu().numpy(),a["M"]),
                      J=maximum(ref.j[0].detach().cpu().numpy(),a["J_reference"]),
                      H=maximum(ref.h[0].detach().cpu().numpy(),a["H"]))
            # Rebuilt reference changes only the audited goal path; saved e is
            # retained because test telemetry deliberately omits current RGB.
            m=ref.m.detach().cpu().double(); j=ref.j.detach().cpu().double()
            e=torch.from_numpy(a["features"][:-1].astype(float))
            action=adapter.calculate_action_from_M(m,e,j).detach().numpy()/np.asarray(adapter.physical_scales)
            record["rebuilt_reference_action_max_abs"]=maximum(action,a["action"])
            vals["passed"]=all(vals[k]<=LIMITS["reference_"+k+"_abs"] for k in ("M","J","H"))
            reference_rows.append(dict(file=path.name,**vals))
            records.append(record)
            if record["method"]=="learned" and gradient_data is None:
                gradient_data=(a["M"],a["features"][0],a["J_reference"])
        report["compact_telemetry"]=dict(episodes=len(records),steps=sum(r["steps"] for r in records),
                                        passed=all(r["passed"] for r in records),records=records)
        report["rebuilt_references"]=dict(unique_goals=len(needed),passed=all(r["passed"] for r in reference_rows),records=reference_rows)
        report["gradients"]=audit_gradients(adapter,*gradient_data)
        # Exercise current RGB encoding on saved training RGB. Reference.observe
        # and the public encode_error API must share exactly the same backbone.
        bootstrap=next((adapter.bundle/"replay").glob("*.npz"))
        with np.load(bootstrap,allow_pickle=False) as data:
            current=data["rgb"][0]; goal=data["goal_rgb"]; rotation=data["rotation"]; d=data["goal_depth"]
        reference=adapter.reference(ae,metric,goal,rotation,depth=d,device=args.device)
        with torch.no_grad():
            encoded,_=adapter.encode_error(ae,current,goal,args.device)
            u,e,V=reference.observe(current)
            physical=adapter.calculate_action_from_M(reference.m,encoded,reference).cpu().numpy()[0]
        report["current_rgb_interface"]=dict(source=str(bootstrap.relative_to(adapter.bundle)),
            e_max_abs=maximum(encoded.cpu().numpy()[0],e),
            physical_action_max_abs=maximum(physical,u*np.asarray(adapter.physical_scales)),
            passed=maximum(encoded.cpu().numpy()[0],e)<2e-6 and maximum(physical,u*np.asarray(adapter.physical_scales))<5e-5,
            limitation="Interface equivalence on retained training RGB, not reconstructed historical test RGB.")
        if args.full_telemetry:
            full=[]
            for index,path in enumerate(full_paths):
                record,a=telemetry(adapter,plants,path,starts)
                key=(record["scene_id"],record["goal_id"],record["method"]=="identity_m")
                ref=refs[key]
                deltas=dict(M=maximum(ref.m[0].detach().cpu().numpy(),a["M"]),
                            J=maximum(ref.j[0].detach().cpu().numpy(),a["J_reference"]),
                            H=maximum(ref.h[0].detach().cpu().numpy(),a["H"]))
                record["rebuilt_reference"]=deltas
                record["checks"]["rebuilt_reference"]=all(deltas[k]<=LIMITS["reference_"+k+"_abs"] for k in ("M","J","H"))
                record["passed"]=all(record["checks"].values())
                full.append(record)
                if (index+1)%120==0:
                    print(json.dumps(dict(event="historical_telemetry_recomputed",episodes=index+1)),flush=True)
                    resources(output,args.device)
            report["additional_original_telemetry"]=dict(episodes=len(full),steps=sum(r["steps"] for r in full),
                passed=all(r["passed"] for r in full),records=full,
                limitation="Read-only original evidence outside independent bundle; no RGB or rollout regeneration.")
        report["limitations"]=[
            "Original scene_builders.py remains incomplete; no room was rendered and no new closed-loop trajectory was collected.",
            "Historical test NPZ intentionally excludes rgb/goal_rgb. Reference RGB is preserved separately; current test RGB cannot be exactly regenerated from lossy videos.",
            "Historical success/failure labels describe selected old trajectories and are not a new test success-rate estimate.",
            "P1 hidden velocity and previous command are replayed with its own exact preserved dynamics, not P0 derivative labels.",
        ]
        checks=[report[k]["passed"] for k in ("goal_depth","compact_telemetry","rebuilt_references","gradients","current_rgb_interface")]
        if "additional_original_telemetry" in report:
            checks.append(report["additional_original_telemetry"]["passed"])
        report["passed"]=all(checks)
        if sha(adapter_path)!=report["adapter_sha256"]:
            report["errors"].append("Adapter changed during audit")
            report["passed"]=False
        report["resources_final"]=resources(output,args.device)
    except Exception:
        report["errors"].append(traceback.format_exc())
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        report["finished_utc"]=datetime.now(timezone.utc).isoformat()
        target=output/"CONTROL_AUDIT.json"
        if target.exists():
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            shutil.copy2(target,output/("CONTROL_AUDIT.previous_"+stamp+".json"))
        temp=target.with_suffix(".tmp")
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        temp.replace(target)
        print(json.dumps(dict(passed=report["passed"],output=str(target),elapsed_seconds=report["elapsed_seconds"],errors=report["errors"]),ensure_ascii=False),flush=True)
    return 0 if report["passed"] else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/visual_original_offline_migration_20261022_v1")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--full-telemetry",type=Path)
    raise SystemExit(run(parser.parse_args()))
