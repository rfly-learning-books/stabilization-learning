"""New-run bridge: source freeze, historical RGB reference, full Push updates."""
from pathlib import Path
import time

from stabilization_learning.unified_v1 import runner as v1
from stabilization_learning.unified_v1.registry import make_backend


def visual_source_inventory(project):
    """Freeze the actual depth weights and every source and asset in the scene audit."""
    project = Path(project).resolve()
    audit_path = project / "migration/visual_scene_reconstruction/SCENE_AUDIT.json"
    audit = v1.read(audit_path)
    if audit.get("passed") is not True:
        raise ValueError("Scene provenance audit did not pass")
    entries = audit.get("files", []) + audit.get("asset_files", [])
    if not audit.get("files") or not audit.get("asset_files"):
        raise ValueError("Scene audit must declare source files and visual assets")
    paths = {}
    for entry in entries:
        path = Path(entry["path"]).resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError("Missing or non-project scene source: " + str(path))
        actual = v1.sha256(path)
        if actual != entry["sha256"]:
            raise ValueError("Scene source differs from audited hash: " + str(path))
        paths[str(path)] = actual
    depth = project / "third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1/extracted_state_dict.pt"
    if not depth.is_file():
        raise FileNotFoundError(depth)
    source_protocol = project / "assets/original_visual_initialization/protocol.json"
    expected = v1.read(source_protocol)["dependencies"]["fastdepth"][r"runtime\dependencies\fastdepth_mobilenet_l1\extracted_state_dict.pt"]
    actual = v1.sha256(depth)
    if actual != expected:
        raise ValueError("FastDepth weights differ from historical protocol hash")
    paths[str(depth.resolve())] = actual
    paths[str(source_protocol.resolve())] = v1.sha256(source_protocol)
    paths[str(audit_path.resolve())] = v1.sha256(audit_path)
    return paths


def validate_extra(config, project):
    v1.validate_config(config)
    deadline = config.get("absolute_deadline_unix")
    if not isinstance(deadline, (float, int)) or isinstance(deadline, bool):
        raise ValueError("Declare the original absolute wall-clock deadline")
    if time.time() + 60 * config["maximum_minutes"] > deadline:
        raise ValueError("The configured run budget exceeds the original absolute deadline")
    jobs = config["jobs"]
    required = {"historical_trained_reference", "historical_trained_identity"}
    historical = [j for j in jobs if required & set(j.get("test_variants", []))]
    enabled = [j for j in jobs if j.get("config", {}).get("include_historical_reference")]
    rgb = [j for j in jobs if j["backend"] == "rgb_legacy"]
    if len(rgb) != 2 or historical != enabled or historical != rgb or any(not required <= set(j.get("test_variants", [])) for j in rgb):
        raise ValueError("Both RGB arms must declare both trained reference and its same-feature identity")
    spec = config.get("historical_reference", {})
    source_paths = spec.get("source_paths", [])
    if not source_paths or not all(Path(p).is_absolute() and Path(p).is_file() for p in source_paths):
        raise ValueError("Declare all historical reference source files as existing absolute paths")
    local = Path(spec.get("checkpoint", "")); original = Path(spec.get("original_checkpoint", ""))
    expected = spec.get("checkpoint_sha256", "")
    declared = {Path(p).resolve() for p in source_paths}
    if not local.is_absolute() or not original.is_absolute() or local.resolve() not in declared or original.resolve() not in declared:
        raise ValueError("Declare both local and original historical checkpoint as sources")
    if local.resolve() != (Path(project) / "assets/original_visual_initialization/best.pt").resolve():
        raise ValueError("Historical checkpoint must be the asset actually loaded by the evaluator")
    if v1.sha256(local) != expected or v1.sha256(original) != expected:
        raise ValueError("Historical checkpoint source or copy does not match the frozen hash")


def run_suite(config, root, *, project, config_path, telemetry=True, factory=None):
    project = Path(project).resolve()
    validate_extra(config, project)
    extra = [Path(__file__), project / "extensions/unified_training_v2/historical_rgb.py",
             project / "extensions/unified_training_v2/push_curves.py",
             project / "extensions/unified_training_v2/reporting.py",
             project / "scripts/run_unified_training_v2.py", Path(config_path).resolve()]
    extra += [Path(p) for p in config["historical_reference"]["source_paths"]]
    original_inventory = v1.source_inventory
    def inventory(root_project):
        sources = original_inventory(root_project)
        sources.update(visual_source_inventory(root_project))
        for path in extra:
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            sources[str(path)] = v1.sha256(path)
        return sources
    def default_factory(name, cfg, output, *, event, deadline):
        if name == "rgb_legacy" and cfg.get("include_historical_reference"):
            from extensions.unified_training_v2.historical_rgb import HistoricalRGBBackend
            return HistoricalRGBBackend(cfg, output, event=event, deadline=deadline)
        if name == "push_discrete_sac":
            from extensions.unified_training_v2.push_curves import RecordedPushBackend
            return RecordedPushBackend(cfg, output, event=event, deadline=deadline)
        return make_backend(name, cfg, output, event=event, deadline=deadline)
    v1.source_inventory = inventory
    try:
        # Inventory hashing has a cost; recheck the hard deadline immediately
        # after hashing as well, so it cannot silently add to the budget.
        def checked_inventory(root_project):
            result = inventory(root_project)
            if time.time() + 60 * config["maximum_minutes"] + 5 > config["absolute_deadline_unix"]:
                raise ValueError("Source inventory consumed the remaining launch margin")
            return result
        v1.source_inventory = checked_inventory
        return v1.run_suite(config, root, project=project, factory=factory or default_factory, telemetry=telemetry)
    finally:
        v1.source_inventory = original_inventory
