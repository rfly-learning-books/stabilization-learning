"""Stage a byte-identical visual research source/dependency bundle.

This copies passive source/assets only. It never imports the research modules,
loads a checkpoint, starts a renderer, or invokes a GPU. Existing differing
destination files cause a failure rather than being overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
OLD = Path("D:/书稿论文")
SOURCE = OLD / "lac_ae_certificate_vs"
DEST = PROJECT / "third_party/visual_original"
ORIGINAL_RUN = SOURCE / "runs/ae_derivative_critic_seed41_20261019_v1"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def selection():
    rows = {}

    def add(source, relative, role):
        source, relative = Path(source), Path(relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        key = relative.as_posix()
        if key in rows and rows[key][0] != source:
            raise RuntimeError(f"Destination collision: {key}")
        rows[key] = (source, role)

    for source in sorted((SOURCE / "lac_vs").glob("*.py")):
        add(source, source.relative_to(SOURCE), "original_python_package")
    for relative in ("run.py", "requirements.txt"):
        add(SOURCE / relative, relative, "original_entrypoint_or_requirements")
    for name in (
        "train_derivative_critic.py", "derivative_critic_core.py",
        "train_goal_fixed.py", "goal_fixed_core.py", "cae_vs_common.py",
        "report_goal_fixed.py", "audit_derivative_critic_run.py",
        "verify_derivative_critic_gradients.py",
    ):
        relative = Path("runtime") / name
        add(SOURCE / relative, relative, "original_runtime_or_independent_audit")
    for name in ("full.yaml", "goal_fixed_night.yaml", "derivative_critic.yaml",
                 "goal_fixed_night_repair_v2.yaml"):
        relative = Path("configs") / name
        add(SOURCE / relative, relative, "original_configuration_unmodified")
    depth = Path("runtime/dependencies/fastdepth_mobilenet_l1")
    for name in ("models.py", "load_pretrained.py", "nyudepthv2.py",
                 "provenance.json", "README.md", "extracted_state_dict.pt"):
        add(SOURCE / depth / name, depth / name, "fastdepth_dependency")

    # Preserve this sibling relationship: scenes.py dynamically imports
    # ../dense_flow_block_depth_servo_20261013/experiment.py. Only its class
    # definitions and rendering constants are needed by the room scene code;
    # its old optical-flow experiment entrypoint is never invoked here.
    scene_rel = Path("assets/scene_dependencies/bright_natural_dense_servo_20261013")
    scene_src = OLD / "output/bright_natural_dense_servo_20261013"
    for name in ("scenes.py", "asset_provenance.json"):
        add(scene_src / name, scene_rel / name, "scene_builder_or_provenance")
    base_rel = Path("dense_flow_block_depth_servo_20261013/experiment.py")
    add(OLD / "output" / base_rel, Path("assets/scene_dependencies") / base_rel,
        "passive_scene_base_module_no_optical_flow_execution")
    asset_src = scene_src / "assets"
    for source in sorted(asset_src.glob("*/diffuse.png")):
        add(source, scene_rel / "assets" / source.relative_to(asset_src), "runtime_texture")
    for source in sorted((asset_src / "classroom_graphics").glob("*.png")):
        add(source, scene_rel / "assets" / source.relative_to(asset_src), "runtime_graphic")
    for source in sorted(asset_src.glob("*/converted.json")):
        add(source, scene_rel / "assets" / source.relative_to(asset_src), "runtime_prop_manifest")
        for part in json.loads(source.read_text(encoding="utf-8")):
            # Include all declared parts, even ones currently skipped for alpha.
            for field in ("mesh", "texture"):
                if part.get(field):
                    asset = source.parent / part[field]
                    add(asset, scene_rel / "assets" / asset.relative_to(asset_src),
                        "runtime_prop_" + field)
    for source in sorted(asset_src.glob("*/polyhaven_files.json")):
        add(source, scene_rel / "assets" / source.relative_to(asset_src), "asset_source_metadata")

    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        add(OLD / "output/li_depth_vs_fd_20261014/model" / name,
            Path("assets/old_depth_audit/model") / name,
            "legacy_depth_hash_audit_only_not_controller_depth")
    for name in ("protocol.json", "resolved_config.yaml"):
        add(ORIGINAL_RUN / name, Path("assets/historical_protocol") / name,
            "immutable_historical_protocol_evidence")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    selected = selection()
    total = sum(path.stat().st_size for path, _ in selected.values())
    plan = dict(destination=str(DEST), files=len(selected), bytes=total,
                mib=total / 2**20, free_gib=shutil.disk_usage(PROJECT).free / 2**30)
    print(json.dumps(plan, ensure_ascii=False))
    if not args.copy:
        return
    if shutil.disk_usage(PROJECT).free - total < 50 * 2**30:
        raise RuntimeError("Copy would violate the 50 GiB free-space boundary")

    protocol = json.loads((ORIGINAL_RUN / "protocol.json").read_text(encoding="utf-8"))
    expected = {Path(key).as_posix(): value for key, value in protocol["sources"].items()}
    expected.update({Path(key).as_posix(): value
                     for key, value in protocol["dependencies"]["fastdepth"].items()})
    readonly = protocol["dependencies"]["read_only"]
    expected["assets/scene_dependencies/bright_natural_dense_servo_20261013/scenes.py"] = readonly["scene_assets_builder"]
    for name, key in (("model.safetensors", "depth_weights"), ("config.json", "depth/config.json"),
                      ("preprocessor_config.json", "depth/preprocessor_config.json")):
        expected["assets/old_depth_audit/model/" + name] = readonly[key]

    # Complete validation before the first file copy.
    records = []
    for relative, (source, role) in sorted(selected.items()):
        value = sha(source)
        if relative in expected and value != expected[relative]:
            raise RuntimeError(f"Historical hash mismatch: {source}")
        target = DEST / relative
        if target.exists() and sha(target) != value:
            raise RuntimeError(f"Refusing to overwrite differing destination: {target}")
        records.append(dict(source=str(source), destination=str(target), relative=relative,
                            role=role, bytes=source.stat().st_size, sha256=value,
                            historical_hash_verified=relative in expected))
    for row in records:
        target = Path(row["destination"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(row["source"], target)
        if sha(target) != row["sha256"]:
            raise RuntimeError(f"Copy hash mismatch: {target}")

    result = dict(
        schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="byte-identical original visual source and passive dependency bundle",
        source_project=str(SOURCE), destination=str(DEST),
        historical_protocol=str(ORIGINAL_RUN / "protocol.json"),
        historical_protocol_sha256=sha(ORIGINAL_RUN / "protocol.json"),
        files=len(records), bytes=total, all_copies_byte_identical=True,
        gpu_or_renderer_started=False, research_source_modified=False,
        source_records=records,
        paths={
            "runtime": str(DEST / "runtime"), "package": str(DEST / "lac_vs"),
            "original_config": str(DEST / "configs/derivative_critic.yaml"),
            "fastdepth": str(DEST / "runtime/dependencies/fastdepth_mobilenet_l1"),
            "scene_assets_builder_dir": str(DEST / "assets/scene_dependencies/bright_natural_dense_servo_20261013"),
            "scene_base_dir": str(DEST / "assets/scene_dependencies/dense_flow_block_depth_servo_20261013"),
            "old_depth_audit_model_dir": str(DEST / "assets/old_depth_audit/model"),
        },
        exclusions=["__pycache__", "training/evaluation outputs and checkpoints", "raw FastDepth remote pickle",
                    "original asset downloads/gltf/bin/unused normal/roughness maps", "legacy reconstruction owned by another agent"],
        limitations=[
            "A source bundle is not a closed-loop or training parity claim.",
            "The six legacy modules and earliest AE are not included by this operation.",
            "Original config paths remain byte-identical; a separately recorded path adapter is required.",
            "The imported historical base module contains unused optical-flow code; no optical-flow capability is run.",
            "Historical source hashes did not cover the full transitive package, scene base or texture files; new manifest covers them.",
        ],
        path_adaptation_suggestions=[
            "Keep original source bytes unchanged. Record every relocated dependency in a new migration protocol.",
            "Supply a new resolved config with recovered legacy_root and explicit initial_autoencoder provenance.",
            "Set derivative.source_run to the independently staged original v2 initialization package; keep its original protocol/checkpoint hashes.",
            "Keep goal_fixed.depth_path relative to this bundle, where the original FastDepth hierarchy is preserved.",
            "Map recovered common.OLD_SCENES_DIR to paths.scene_assets_builder_dir and OLD_DEPTH_MODEL_DIR/DEPTH_MODEL_FILE to old_depth_audit_model_dir.",
            "Record any loader/global path rebinding in adapter source and migration manifest before parity checks.",
            "Use the original normalized-command policy and convert only its final command to physical velocity.",
        ],
    )
    target = PROJECT / "migration/visual_source_bundle.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)
    print(json.dumps(dict(manifest=str(target), copied_files=len(records), copied_mib=total / 2**20,
                          historical_hashes_verified=sum(r["historical_hash_verified"] for r in records)),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
