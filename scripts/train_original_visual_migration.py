"""Run the preserved derivative-Critic training loop on an audited new renderer.

The experiment keeps the original numerical configuration, split and starts,
but explicitly declares reconstructed appearance and freshly rendered replay.
Vendored source, old checkpoints, and old experiment outputs are immutable.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "assets/original_visual_initialization"
VENDOR = ROOT / "third_party/visual_original"
ONLINE_ADAPTER = ROOT / "stabilization-gym/stabilization_gym/envs/tasks/original_visual_online.py"
OFFLINE_ADAPTER = ONLINE_ADAPTER.with_name("original_visual.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def within(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise RuntimeError("Asset path escapes its declared bundle: " + str(relative))
    return target


def checked_entries(root, entries):
    values = {}
    for row in entries:
        relative = row["relative"]
        path = within(root, relative)
        if sha(path) != row["sha256"]:
            raise RuntimeError("Bundle hash mismatch: " + str(path))
        if "bytes" in row and path.stat().st_size != row["bytes"]:
            raise RuntimeError("Bundle size mismatch: " + str(path))
        values[Path(relative).as_posix()] = row["sha256"]
    return values


def preflight(args):
    """Read-only, standard-library-only checks; no Torch/CUDA imports."""
    data = args.data_bundle.resolve()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "runs"):
        raise RuntimeError("New training output must be inside this project's runs directory")
    if data == ORIGINAL or data.is_relative_to(ORIGINAL):
        raise RuntimeError("Old RGB bootstrap/references cannot initialize the reconstructed renderer")
    scene_path = args.scene_audit.resolve()
    scene = read(scene_path)
    if scene.get("passed") is not True:
        raise RuntimeError("SCENE_AUDIT has not passed; no training may start")
    if scene.get("protocol_kind") != "reconstructed_environment" or scene.get("original_rgb_equivalent") is not False:
        raise RuntimeError("SCENE_AUDIT must explicitly declare reconstructed, non-equivalent RGB")
    if Path(scene["legacy_root"]).resolve() != args.legacy_root.resolve():
        raise RuntimeError("Legacy root differs from SCENE_AUDIT")
    renderer_manifest = Path(scene["scene_manifest"])
    if not renderer_manifest.is_absolute():
        renderer_manifest = scene_path.parent / renderer_manifest
    renderer_manifest = renderer_manifest.resolve()
    if renderer_manifest != (args.legacy_root.resolve().parent / "scenes.json").resolve():
        raise RuntimeError("Scene audit manifest is not the renderer's bound scenes.json")
    if sha(renderer_manifest) != scene["scene_manifest_sha256"]:
        raise RuntimeError("Renderer scene manifest changed after audit: " + str(renderer_manifest))
    for key in ("structured_geometry_preserved", "collision_semantics_preserved", "p0_p1_dynamics_preserved"):
        if scene.get("checks", {}).get(key) is not True:
            raise RuntimeError("Required scene semantics check missing/failed: " + key)
    for row in scene["source_files"]:
        path = Path(row["path"])
        if not path.is_absolute():
            path = scene_path.parent / path
        if sha(path) != row["sha256"]:
            raise RuntimeError("Scene source changed after audit: " + str(path))
    if not scene["asset_files"]:
        raise RuntimeError("Scene audit must declare renderer assets")
    for row in scene["asset_files"]:
        path = Path(row["path"])
        if not path.is_absolute():
            path = scene_path.parent / path
        if sha(path) != row["sha256"]:
            raise RuntimeError("Scene asset changed after audit: " + str(path))
    manifest_path = data / "DATA_MANIFEST.json"
    manifest = read(manifest_path)
    if manifest.get("protocol_kind") != "reconstructed_environment" or manifest.get("original_rgb_equivalent") is not False:
        raise RuntimeError("New RGB data provenance is missing")
    if manifest.get("scene_audit_sha256") != sha(scene_path):
        raise RuntimeError("Data were generated under a different scene audit")
    if manifest.get("source_initial_checkpoint_sha256") != sha(ORIGINAL / "initial.pt"):
        raise RuntimeError("Data refer to a different initial checkpoint")
    entries = checked_entries(data, manifest["entries"])
    for name in ("scenes.json", "evaluation_starts.json", "training_anchors.json", "replay/index.json"):
        if name not in entries:
            raise RuntimeError("Required file absent from data manifest: " + name)
    for name in ("scenes.json", "evaluation_starts.json", "training_anchors.json"):
        if sha(data / name) != sha(ORIGINAL / name):
            raise RuntimeError("This protocol preserves original scenes/starts/anchors: " + name)
    if scene["scene_manifest_sha256"] != sha(data / "scenes.json"):
        raise RuntimeError("Renderer and data scene manifests differ")
    rows = read(data / "replay/index.json")
    if len(rows) != 48 or sum(row["length"] for row in rows) != 2822:
        raise RuntimeError("Expected exactly 48 shards and 2822 bootstrap transitions")
    if manifest.get("bootstrap", {}).get("shards") != 48 or manifest.get("bootstrap", {}).get("transitions") != 2822:
        raise RuntimeError("Data manifest bootstrap declaration differs from actual replay")
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("Held-out data entered training replay")
    scenes = {row["id"]: row for row in read(data / "scenes.json")["scenes"]}
    for row in rows:
        if scenes[row["scene_id"]]["split"] != "train" or "replay/"+row["file"] not in entries:
            raise RuntimeError("Invalid or unhashed training shard")
    for name in entries:
        if name.startswith("references/"):
            sid = Path(name).stem.rsplit("_", 1)[0]
            if sid not in scenes or scenes[sid]["split"] != "train":
                raise RuntimeError("Bootstrap package references must be train-only")
    for name in ("CONTROL_AUDIT.json", "TRAINING_AUDIT.json"):
        path = ROOT / "runs/visual_original_offline_migration_20261022_v1" / name
        if read(path).get("passed") is not True:
            raise RuntimeError("Prior numerical migration audit is not passed: " + name)
    source_bundle = read(ROOT / "migration/visual_source_bundle.json")
    for row in source_bundle["source_records"]:
        if sha(row["destination"]) != row["sha256"]:
            raise RuntimeError("Vendored source/dependency changed: " + row["relative"])
    if (out / "STOP").exists():
        raise RuntimeError("User STOP exists; it will not be removed")
    if args.resume and not (out / "last.pt").is_file():
        raise RuntimeError("--resume requires an existing full checkpoint")
    if not args.resume and out.exists() and any(out.iterdir()):
        raise RuntimeError("A new run requires an empty output directory")
    if shutil.disk_usage(ROOT).free / 2**30 < 51:
        raise RuntimeError("At least 51 GiB free is required, including checkpoint reserve")
    return dict(data_manifest=manifest, data_hashes=entries, scene_audit=scene,
                scene_audit_sha256=sha(scene_path), data_manifest_sha256=sha(manifest_path),
                source_bundle=source_bundle, bootstrap_rows=rows)


def gpu_snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10)
    used, total, utilization = map(float, raw.strip().splitlines()[0].split(","))
    return dict(gpu_used_mib=used, gpu_total_mib=total, gpu_utilization=utilization)


def other_trainers():
    if os.name != "nt":
        return []
    script = "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match '(train_.*\\.py|audit_visual_control_transfer\\.py)' }; @($p | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
    raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", script], text=True, timeout=20).strip()
    rows = json.loads(raw) if raw else []
    if isinstance(rows, dict):
        rows = [rows]
    return [row for row in rows if int(row["ProcessId"]) != os.getpid()]


class BoundaryMonitor:
    """Request vendor safe-boundary checkpointing before hard resource limits."""
    def __init__(self, output, torch_module):
        self.output, self.torch = output, torch_module
        self.last = 0.; self.reason = None; self.snapshot = {}

    def poll(self, force=False):
        if not force and time.monotonic()-self.last < 5:
            return self.snapshot
        self.last = time.monotonic()
        snap = gpu_snapshot()
        snap["free_gib"] = shutil.disk_usage(ROOT).free / 2**30
        snap["run_gib"] = sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file()) / 2**30 if self.output.exists() else 0.
        snap["cuda_allocated_mib"] = self.torch.cuda.memory_allocated()/2**20
        snap["cuda_peak_mib"] = self.torch.cuda.max_memory_allocated()/2**20
        reasons = []
        if snap["free_gib"] < 51:
            reasons.append("disk reserve including one GiB checkpoint margin")
        if snap["run_gib"] >= 34:
            reasons.append("run size including one GiB checkpoint margin")
        if snap["gpu_used_mib"] / snap["gpu_total_mib"] >= .64:
            reasons.append("GPU total use reached 64 percent soft boundary")
        if reasons:
            self.reason = "; ".join(reasons)
        snap["migration_safe_stop_reason"] = self.reason
        self.snapshot = snap
        return snap


def execute(args, evidence):
    sys.path.insert(0, str(ROOT / "stabilization-gym"))
    from stabilization_gym.envs.tasks.original_visual_online import OriginalVisualOnlineAdapter
    import torch
    import importlib
    import yaml
    adapter = OriginalVisualOnlineAdapter(ROOT, args.legacy_root, args.scene_audit)
    vendor = importlib.import_module("train_derivative_critic")
    cfg = adapter.training_config(args.data_bundle)
    if args.smoke:
        cfg["engineering_smoke"] = True
        cfg["goal_fixed"].update(rounds=1, warmup_updates=2, representation_updates=2,
                                 metric_updates=2, batch_size=2, evaluation_validation_cap=3,
                                 evaluation_plants=["P0"], stress_depth_scales=[])
        cfg["derivative"]["critic_prefit_updates"] = 2
        cfg["collection"].update(steps=8, episodes_per_scene_per_round=1)
        cfg["evaluation"]["max_steps"] = 12
    output = args.output.resolve(); data = args.data_bundle.resolve()
    monitor = BoundaryMonitor(output, torch)
    initial = adapter.checkpoint("initial.pt")
    if any(initial["learner"]["counts"].values()):
        raise RuntimeError("Training initial checkpoint has nonzero updates")
    if initial["learner"]["anchors"] != read(data/"training_anchors.json"):
        raise RuntimeError("Original anchors must remain exactly unchanged")
    row_fields = ("file", "length", "scene_id", "split", "goal")
    original_rows = [tuple(row[key] for key in row_fields) for row in initial["replay_rows"]]
    new_rows = [tuple(row[key] for key in row_fields) for row in evidence["bootstrap_rows"]]
    if original_rows != new_rows:
        raise RuntimeError("Re-rendered replay must preserve original shard order and metadata")
    original_learner = vendor.DerivativeLearner
    original_stop = vendor.joint.Stop
    original_audit = vendor.audit

    class RestoredLearner(original_learner):
        def __init__(self, local_cfg, source, device="cuda"):
            snapshot = monitor.poll(force=True)
            available = .65-snapshot["gpu_used_mib"]/snapshot["gpu_total_mib"]-.03
            if available <= .10:
                raise RuntimeError("Insufficient GPU headroom")
            torch.cuda.set_per_process_memory_fraction(min(.50, available))
            super().__init__(local_cfg, source, device)
            self.restore(source["learner"])
            self.initial.load_state_dict(source["learner"]["ae"], strict=True)
            adapter.config_module.restore_rng(source["rng"])

    class SafeBoundaryStop(original_stop):
        def stopped(self):
            monitor.poll()
            return bool(monitor.reason) or super().stopped()

    def payload_hook(_cfg):
        return ORIGINAL, initial

    source_hashes = {row["relative"]: row["sha256"] for row in evidence["source_bundle"]["source_records"]}
    migration_sources = {str(path.relative_to(ROOT)): sha(path) for path in (Path(__file__), ONLINE_ADAPTER, OFFLINE_ADAPTER)}
    expected = dict(config=cfg, sources=source_hashes,
        dependencies=dict(scene_audit_sha256=evidence["scene_audit_sha256"],
                          scene_sources=evidence["scene_audit"]["source_files"],
                          data_manifest_sha256=evidence["data_manifest_sha256"],
                          data_files=evidence["data_hashes"],
                          migration_sources=migration_sources),
        initialization={name:sha(ORIGINAL/name) for name in ("protocol.json","initial.pt","training_anchors.json")},
        migration=dict(protocol_kind="reconstructed_environment", original_rgb_equivalent=False,
            mathematical_algorithm="unchanged vendored derivative-Critic loop and update formulas",
            appearance="new reconstructed room rendering; never described as historical pixel reproduction",
            bootstrap="2822 retained P0 off-policy state/action transitions re-rendered under the new renderer; not newly collected PVS behavior",
            smooth_triplets="re-rendered under separately declared new data seed; see DATA_MANIFEST.json",
            initialization="complete derivative initial AE/shadow/M/Critic/four optimizers/counts/version/anchors and RNG",
            anchors="original rec/curvature/latent_std/center/scale preserved; no appearance recalibration",
            goal_references="only newly rendered train references imported; remaining train/val/test references generated by reconstructed renderer",
            safe_boundaries="disk 50 GiB and run 35 GiB hard budgets with one GiB checkpoint margins; total GPU below 65 percent",
            smoke=bool(args.smoke)))

    def prepare_hook(out, local_cfg):
        marker = out/"protocol.json"
        if marker.exists():
            saved = read(marker)
            for key, value in expected.items():
                if saved.get(key) != value:
                    raise RuntimeError("Migration protocol changed: "+key+"; create a new output")
            return saved
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("Nonempty output without migration protocol")
        out.mkdir(parents=True,exist_ok=True)
        original = adapter.original_protocol
        metadata = {key:value for key,value in original.items()
                    if key not in {"schema","created","config","sources","dependencies","initialization"}}
        protocol = dict(schema=2,created=vendor.joint.now(),**metadata,**expected)
        for name in ("scenes.json","evaluation_starts.json","training_anchors.json"):
            shutil.copy2(data/name,out/name)
        for relative in source_hashes:
            if Path(relative).suffix in (".py",".yaml"):
                target=out/"provenance/vendor"/relative; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(VENDOR/relative,target)
        for relative in migration_sources:
            target=out/"provenance/migration"/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,target)
        shutil.copy2(args.scene_audit,out/"SCENE_AUDIT.json")
        shutil.copy2(data/"DATA_MANIFEST.json",out/"DATA_MANIFEST.json")
        vendor.json_write(marker,protocol)
        (out/"resolved_config.yaml").write_text(yaml.safe_dump(local_cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
        return protocol

    def import_hook(out, _cfg, replay, source):
        rows=copy.deepcopy(evidence["bootstrap_rows"])
        for row in rows:
            src=data/"replay"/row["file"]; dst=replay.root/row["file"]
            if dst.exists() and sha(dst)!=sha(src):
                raise RuntimeError("Existing bootstrap shard differs")
            if not dst.exists():
                shutil.copy2(src,dst)
            if sha(dst)!=evidence["data_hashes"]["replay/"+row["file"]]:
                raise RuntimeError("Bootstrap copy verification failed")
        replay.rows=rows
        replay.rng.bit_generator.state=copy.deepcopy(source["replay_rng"])
        replay.cache.clear()
        vendor.json_write(replay.path,rows)
        shutil.copytree(data/"references",out/"references",dirs_exist_ok=True)
        adapter.config_module.restore_rng(source["rng"])
        vendor.json_write(out/"import_provenance.json",dict(
            source_checkpoint=str(ORIGINAL/"initial.pt"), source_checkpoint_sha256=sha(ORIGINAL/"initial.pt"),
            source_data_bundle=str(data), data_manifest_sha256=evidence["data_manifest_sha256"],
            transitions=replay.transitions,shards=len(rows),train_only=True,
            critic_reinitialized=False,full_initial_learner_restored=True,original_anchors_preserved=True,
            original_rgb_imported=False,bootstrap_behavior="old P0 states/actions with newly rendered RGB and FastDepth",
            replay_rng_restored=True,global_rng_restored=True))

    def audit_hook(out, local_cfg, learner, replay):
        original_audit(out,local_cfg,learner,replay)
        import numpy as np
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
            if isinstance(a, np.ndarray):
                return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
            if isinstance(a, dict):
                return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
            if isinstance(a, (tuple, list)):
                return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b
        checks = dict(full_initial_learner=equal(learner.payload(), initial["learner"]),
                      initial_baseline=equal(learner.initial.state_dict(), initial["learner"]["ae"]),
                      global_rng=equal(adapter.config_module.rng_state(), initial["rng"]),
                      replay_rng=equal(replay.rng.bit_generator.state, initial["replay_rng"]),
                      original_anchors=learner.anchors==initial["learner"]["anchors"],
                      all_bootstrap=replay.transitions==2822 and len(replay.rows)==48)
        # The original audit restores model/RNG exactly. Expose the declared
        # initialization policy separately instead of changing that evidence.
        vendor.json_write(out/"migration_initialization_audit.json",dict(
            passed=all(checks.values()), checks=checks,
            restored_updates=learner.counts,transitions=replay.transitions,
            original_anchors_preserved=True,initial_baseline_restored=True,
            independent_offline_update_audit="TRAINING_AUDIT.json previously passed"))
        if not all(checks.values()):
            raise RuntimeError("Complete initialization did not restore exactly: "+str(checks))

    vendor.prepare=prepare_hook
    vendor.source_payload=payload_hook
    vendor.import_training_data=import_hook
    vendor.DerivativeLearner=RestoredLearner
    vendor.audit=audit_hook
    vendor.joint.Stop=SafeBoundaryStop
    vendor.resource=lambda out,local_cfg,measure_size=False: monitor.poll(force=measure_size)
    with vendor.writer_lock(ROOT/"runs/visual_training_gpu_slot"):
        conflicts=other_trainers()
        if conflicts:
            raise RuntimeError("Other training/audit processes must finish first: "+json.dumps(conflicts))
        if monitor.poll(force=True)["gpu_used_mib"]/monitor.snapshot["gpu_total_mib"]>=.64:
            raise RuntimeError("GPU already exceeds safe starting boundary")
        with vendor.writer_lock(output):
            # Reject cross-protocol episode progress. A progress file tied to an
            # older full checkpoint is retained and ignored by the vendor loop.
            if args.resume:
                saved_protocol=read(output/"protocol.json")
                identity=hashlib.sha256(json.dumps(saved_protocol,sort_keys=True).encode()).hexdigest()
                for name in ("last.pt","episode_progress.pt"):
                    if (output/name).exists():
                        saved=torch.load(output/name,map_location="cpu",weights_only=False)
                        if saved["protocol_hash"]!=identity:
                            raise RuntimeError("Resume protocol mismatch: "+name)
            try:
                vendor.run(output,cfg,SimpleNamespace(resume=args.resume,max_minutes=args.max_minutes))
                if monitor.reason:
                    vendor.json_write(output/"safe_stop.json",dict(timestamp=vendor.joint.now(),
                        reason=monitor.reason,resources=monitor.snapshot,resume_allowed=not(output/"STOP").exists()))
            except Exception as exc:
                output.mkdir(parents=True,exist_ok=True)
                vendor.json_write(output/"error.json",dict(timestamp=vendor.joint.now(),type=type(exc).__name__,
                    message=str(exc),traceback=traceback.format_exc()))
                raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--legacy-root",type=Path,required=True)
    parser.add_argument("--scene-audit",type=Path,required=True)
    parser.add_argument("--data-bundle",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--preflight",action="store_true",help="Read-only standard-library checks; do not initialize CUDA")
    parser.add_argument("--max-minutes",type=float,default=120.)
    args=parser.parse_args()
    if args.max_minutes<=0:
        parser.error("--max-minutes must be positive")
    try:
        evidence=preflight(args)
    except Exception as exc:
        print(json.dumps(dict(ready=False,stage="preflight",reason=str(exc),
                              training_started=False),ensure_ascii=False),flush=True)
        return 2
    if args.preflight:
        print(json.dumps(dict(ready=True,training_started=False,protocol_kind="reconstructed_environment",
            original_rgb_equivalent=False,bootstrap_transitions=2822,original_anchors_preserved=True,
            output=str(args.output.resolve())),ensure_ascii=False),flush=True)
        return 0
    execute(args,evidence)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
