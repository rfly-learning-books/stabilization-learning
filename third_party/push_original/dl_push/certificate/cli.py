"""Command-line workflow for the sampling-first certificate controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .assets import certificate_polygon, load_geometry, make_certificate_model
from .calibration import CalibrationConfig, ModeCalibrator
from .config import load_config
from .evaluation import json_safe
from .library import ModeLibrary, boundary_mode_library, fixed_mode_library, support_mode_library
from .math import matrix_hash
from .physics import MuJoCoAdapter
from .q_generator import CertificateGenerator, QConfig, SearchConfig, search_q
from .runner import CertificateRunner
from .sampling import SamplingConfig, make_seed_bank, sample_free_pusher_positions, sample_task
from .selection import attach_transfer_graph, select_modes
from .solver import CLFConfig
from .transfer import build_transfer_graph


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n")
    return path


def _new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"output must be a new directory: {path}")
    path.mkdir(parents=True)
    return path


def _config_and_geometry(args):
    config = load_config(args.config) if getattr(args, "config", None) else None
    asset_root = getattr(args, "asset_root", None) or (config.asset_root if config else "assets/native_v1")
    shape = getattr(args, "shape", None)
    if not shape:
        raise ValueError("--shape is required")
    geometry = load_geometry(asset_root, shape)
    return config, geometry


def cmd_build_model(args):
    _, geometry = _config_and_geometry(args)
    output = _new_output(args.output)
    model_dir = output / "model"
    metadata = make_certificate_model(
        geometry,
        model_dir,
        physics_dt=args.physics_dt,
        object_friction=args.object_friction,
        floor_friction=args.floor_friction,
        pusher_friction=args.pusher_friction,
    )
    _write_json(output / "build_report.json", metadata)
    return metadata


def _base_library(args, geometry):
    polygon = certificate_polygon(geometry)
    count = int(args.count)
    metadata = {
        "geometry_fingerprint": geometry.fingerprint,
        "geometry_asset": str(geometry.path),
        "reference_point": "physical_com",
        "orientation_period": geometry.period,
    }
    if getattr(args, "mode_scheme", "fixed") == "fixed":
        return fixed_mode_library(
            polygon, geometry.name, mode_count=count,
            pusher_radius=args.pusher_radius, clearance=args.clearance,
            u_max=args.u_max, metadata=metadata,
        )
    if getattr(args, "mode_scheme", "fixed") == "support":
        return support_mode_library(
            polygon, geometry.name, directions=count,
            pusher_radius=args.pusher_radius, clearance=args.clearance,
            u_max=args.u_max, metadata=metadata,
        )
    return boundary_mode_library(
        polygon, geometry.name, count=count,
        pusher_radius=args.pusher_radius, clearance=args.clearance,
        u_max=args.u_max, metadata=metadata,
    )


def cmd_calibrate(args):
    _, geometry = _config_and_geometry(args)
    output = _new_output(args.output)
    model_dir = output / "model"
    model_meta = make_certificate_model(
        geometry,
        model_dir,
        physics_dt=args.physics_dt,
        object_friction=args.object_friction,
        floor_friction=args.floor_friction,
        pusher_friction=args.pusher_friction,
    )
    library = _base_library(args, geometry)
    library.metadata["model_xml"] = str((model_dir / "model.xml").resolve())
    if args.quick:
        calibration = CalibrationConfig(
            speeds_mps=(.03, .06),
            yaws_rad=(0.0, np.deg2rad(45.0)),
            repeats=1,
            hold_ticks=2,
            pusher_radius=args.pusher_radius,
            contact_clearance=args.clearance,
            max_command_mps=min(args.max_command_mps, .20),
        )
    else:
        calibration = CalibrationConfig(
            speeds_mps=tuple(args.speeds),
            yaws_rad=tuple(np.deg2rad(args.yaws_deg)),
            repeats=args.repeats,
            hold_ticks=args.hold_ticks,
            pusher_radius=args.pusher_radius,
            contact_clearance=args.clearance,
            max_command_mps=args.max_command_mps,
        )
    adapter = MuJoCoAdapter(model_dir / "model.xml", control_dt=args.control_dt, expected_physics_dt=args.physics_dt)
    result = ModeCalibrator(adapter, calibration).calibrate(library, output / "samples")
    result.save(output / "library")
    _write_json(output / "calibration_report.json", {"model": model_meta, "library": result.metadata, "fingerprint": result.fingerprint})
    return {"output": str(output), "library": str(output / "library"), "valid_modes": int(result.valid.sum()), "mode_count": result.count}


def cmd_select_modes(args):
    library = ModeLibrary.load(args.library)
    selected, report = select_modes(library, args.count, args.feature_length, args.samples)
    output = _new_output(args.output)
    selected.save(output / "library")
    _write_json(output / "selection_report.json", report.to_dict())
    return {"output": str(output), "library": str(output / "library"), **report.to_dict()}


def cmd_build_transfer(args):
    library = ModeLibrary.load(args.library)
    _, geometry = _config_and_geometry(args)
    polygon = certificate_polygon(geometry)
    graph = build_transfer_graph(
        polygon,
        library.contact_points,
        library.normals,
        pusher_radius=args.pusher_radius,
        clearance=args.clearance,
        workspace_limit=args.workspace_limit,
        free_speed=args.free_speed,
        engage_speed=args.engage_speed,
    )
    output = _new_output(args.output)
    result = attach_transfer_graph(library, graph)
    result.save(output / "library")
    _write_json(output / "transfer_report.json", {"valid_edges": int(graph.valid.sum()), "mode_count": result.count, "metadata": graph.metadata})
    return {"output": str(output), "library": str(output / "library"), "valid_edges": int(graph.valid.sum())}


def _load_q(args):
    generator = CertificateGenerator()
    if args.q:
        data = json.loads(Path(args.q).read_text())
        if "Q" in data:
            return np.asarray(data["Q"], dtype=float), data.get("parameters", np.zeros(5).tolist())
        raise ValueError("Q record must contain Q")
    if args.parameters is None:
        return generator.identity(), np.zeros(5).tolist()
    parameters = np.asarray(args.parameters, dtype=float)
    return generator(parameters), parameters.tolist()


def _analytic_objective(library, config, samples, parameters):
    from .q_generator import CertificateGenerator

    Q = CertificateGenerator(QConfigProxy(config)).__call__(parameters)
    solver = __import__("dl_push.certificate.solver", fromlist=["CLFSolver"]).CLFSolver(library, Q, config)
    losses = []
    for error in samples:
        decision = solver.select_stage(error)
        if decision.feasible:
            chosen = library.c[:, decision.mode]
            losses.append(float(decision.speed ** 2 + .001 / max(np.linalg.norm(chosen[:2]), 1e-8)))
        else:
            losses.append(10.0 + float(np.linalg.norm(error)))
    return float(np.mean(losses))


class QConfigProxy:
    """Adapt CLFConfig scale values for the stateless Q generator."""

    def __init__(self, config):
        self.eigenvalue_floor = config.q_eigenvalue_floor
        self.trace = config.q_trace


def cmd_search_q(args):
    library = ModeLibrary.load(args.library)
    config = CLFConfig(
        feature_length=args.feature_length,
        control_dt=args.control_dt,
        input_upper_mps=args.input_upper_mps,
        endpoint_check=not args.no_endpoint_check,
    )
    rng = np.random.default_rng(args.seed)
    errors = np.column_stack((
        rng.uniform(-args.position_error, args.position_error, args.samples),
        rng.uniform(-args.position_error, args.position_error, args.samples),
        rng.uniform(-args.angle_error, args.angle_error, args.samples),
    ))
    search_config = SearchConfig(
        population=args.population,
        generations=args.generations,
        sigma=args.sigma,
        seed=args.seed,
        restarts=args.restarts,
    )
    if args.physical:
        from .learning import PhysicalSearchConfig, search_q_physical

        model = args.model or library.metadata.get("model_xml")
        if not model:
            raise ValueError("--model is required for physical Q search")
        geometry = type("GeometryView", (), {
            "polygon": __import__("dl_push.certificate.assets", fromlist=["certificate_polygon"]).certificate_polygon(
                load_geometry(args.asset_root, library.shape)
            ),
            "period": load_geometry(args.asset_root, library.shape).period,
        })()
        result = search_q_physical(
            geometry,
            library,
            model,
            config,
            __import__("dl_push.certificate.runner", fromlist=["RolloutConfig"]).RolloutConfig(
                max_control_ticks=args.max_ticks,
                hold_ticks=args.hold_ticks,
            ),
            search_config,
            PhysicalSearchConfig(
                episodes_per_candidate=args.episodes_per_candidate,
                position_min=args.position_min,
                position_max=args.position_max,
                angle_max_rad=args.angle_error,
                seed=args.seed,
                failure_penalty=args.failure_penalty,
            ),
        )
    else:
        result = search_q(lambda parameters: _analytic_objective(library, config, errors, parameters), search_config)
    output = _new_output(args.output)
    generator = CertificateGenerator(QConfig(trace=config.q_trace, eigenvalue_floor=config.q_eigenvalue_floor))
    best_parameters = np.asarray(result["best"]["parameters"])
    generator.save(output / "q.json", best_parameters)
    _write_json(output / "search_report.json", result)
    return {"output": str(output), "q": str(output / "q.json"), "best_loss": result["best"]["loss"], "Q_hash": matrix_hash(generator(best_parameters))}


def cmd_train_encoder(args):
    from .learning import EncoderTrainConfig, train_set_encoder

    root = Path(args.data).resolve()
    libraries, labels = [], []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not (entry / "library").is_dir() or not (entry / "q.json").is_file():
            continue
        libraries.append(ModeLibrary.load(entry / "library"))
        record = json.loads((entry / "q.json").read_text())
        if "parameters" not in record:
            raise ValueError(f"Q record lacks parameters: {entry / 'q.json'}")
        labels.append(record["parameters"])
    if not libraries:
        raise ValueError("data root must contain shape subdirectories with library/ and q.json")
    return train_set_encoder(
        libraries,
        labels,
        args.output,
        EncoderTrainConfig(
            hidden=args.hidden,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        ),
    )


def _evaluate_fixed_q(args, geometry, library, q, output, episodes, seed):
    from .runner import RolloutConfig

    model = args.model or library.metadata.get("model_xml")
    if not model:
        raise ValueError(f"no model recorded for shape {geometry.name}")
    adapter = MuJoCoAdapter(model, control_dt=args.control_dt, expected_physics_dt=args.physics_dt)
    polygon = certificate_polygon(geometry)
    sampling = SamplingConfig(
        position_min=args.position_min,
        position_max=args.position_max,
        angle_max_rad=args.angle_error,
        workspace_limit=args.workspace_limit,
    )
    rng = np.random.default_rng(seed)
    positions = sample_free_pusher_positions(polygon, episodes, rng, sampling)
    view = type("GeometryView", (), {"polygon": polygon, "period": geometry.period})()
    runner = CertificateRunner(
        adapter, view, library, q,
        CLFConfig(feature_length=args.feature_length, control_dt=args.control_dt, input_upper_mps=args.input_upper_mps),
        RolloutConfig(max_control_ticks=args.max_ticks, hold_ticks=args.hold_ticks),
    )
    reports = []
    for index in range(episodes):
        object_q = np.zeros(3, dtype=float)
        target = sample_task(object_q, rng, sampling, geometry.period)
        report = runner.run(object_q, target, positions[index], seed=seed + index)
        runner.save_report(report, output / f"episode_{index:04d}.json")
        reports.append(report)
    return {
        "shape": geometry.name,
        "episodes": episodes,
        "successes": sum(bool(report["success"]) for report in reports),
        "failures": [report["failure_code"] for report in reports if not report["success"]],
    }


def cmd_sealed_test(args):
    from .learning import generate_q_from_library, load_set_encoder

    output = _new_output(args.output)
    model = load_set_encoder(args.encoder, device=args.device)
    summaries = []
    for shape in args.shapes:
        geometry = load_geometry(args.asset_root, shape)
        library = ModeLibrary.load(Path(args.library_root) / shape / "library")
        q = generate_q_from_library(model, library, device=args.device)
        shape_output = output / shape
        shape_output.mkdir(parents=True)
        summary = _evaluate_fixed_q(args, geometry, library, q, shape_output, args.episodes, args.seed)
        summary["q_hash"] = matrix_hash(q)
        summaries.append(summary)
        _write_json(shape_output / "report.json", summary)
    result = {"schema": 1, "split": "sealed_test", "summaries": summaries,
              "successes": sum(item["successes"] for item in summaries),
              "episodes": sum(item["episodes"] for item in summaries)}
    _write_json(output / "report.json", result)
    return result


def cmd_evaluate(args):
    _, geometry = _config_and_geometry(args)
    library = ModeLibrary.load(args.library)
    output = _new_output(args.output)
    q, parameters = _load_q(args)
    model = args.model or library.metadata.get("model_xml")
    if not model:
        raise ValueError("--model is required unless the library records model_xml")
    adapter = MuJoCoAdapter(model, control_dt=args.control_dt, expected_physics_dt=args.physics_dt)
    polygon = certificate_polygon(geometry)
    sampling = SamplingConfig(
        position_min=args.position_min,
        position_max=args.position_max,
        angle_min_rad=0.0,
        angle_max_rad=args.angle_error,
        workspace_limit=args.workspace_limit,
    )
    rng = np.random.default_rng(args.seed)
    positions = sample_free_pusher_positions(polygon, args.episodes, rng, sampling)
    runner = CertificateRunner(
        adapter,
        type("GeometryView", (), {"polygon": polygon, "period": geometry.period})(),
        library,
        q,
        CLFConfig(feature_length=args.feature_length, control_dt=args.control_dt, input_upper_mps=args.input_upper_mps),
    )
    reports = []
    for index in range(args.episodes):
        object_q = np.zeros(3, dtype=float)
        target = sample_task(object_q, rng, sampling, geometry.period)
        reports.append(runner.run(object_q, target, positions[index], seed=int(args.seed) + index))
        runner.save_report(reports[-1], output / f"episode_{index:04d}.json")
    summary = {
        "schema": 1,
        "shape": geometry.name,
        "parameters": parameters,
        "episodes": args.episodes,
        "successes": sum(bool(report["success"]) for report in reports),
        "failures": [report["failure_code"] for report in reports if not report["success"]],
        "q_hash": matrix_hash(q),
    }
    _write_json(output / "report.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Sampling-first cross-shape CLF controller")
    sub = parser.add_subparsers(dest="command", required=True)

    model = sub.add_parser("build-model")
    model.add_argument("--shape", required=True)
    model.add_argument("--asset-root", default="assets/native_v1")
    model.add_argument("--output", required=True)
    model.add_argument("--physics-dt", type=float, default=.002)
    model.add_argument("--object-friction", type=float, default=3.)
    model.add_argument("--floor-friction", type=float, default=.5)
    model.add_argument("--pusher-friction", type=float, default=0.)
    model.set_defaults(function=cmd_build_model)

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--shape", required=True)
    calibrate.add_argument("--asset-root", default="assets/native_v1")
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--count", type=int, default=8)
    calibrate.add_argument("--mode-scheme", choices=["fixed", "support", "boundary"], default="fixed")
    calibrate.add_argument("--pusher-radius", type=float, default=.018)
    calibrate.add_argument("--clearance", type=float, default=.002)
    calibrate.add_argument("--u-max", type=float, default=.06)
    calibrate.add_argument("--physics-dt", type=float, default=.002)
    calibrate.add_argument("--control-dt", type=float, default=.2)
    calibrate.add_argument("--object-friction", type=float, default=3.)
    calibrate.add_argument("--floor-friction", type=float, default=.5)
    calibrate.add_argument("--pusher-friction", type=float, default=0.)
    calibrate.add_argument("--max-command-mps", type=float, default=.20)
    calibrate.add_argument("--speeds", type=float, nargs="+", default=[.005,.01,.02,.04,.06])
    calibrate.add_argument("--yaws-deg", type=float, nargs="+", default=[0,45,90,135,180,225,270,315])
    calibrate.add_argument("--repeats", type=int, default=3)
    calibrate.add_argument("--hold-ticks", type=int, default=3)
    calibrate.add_argument("--quick", action="store_true")
    calibrate.set_defaults(function=cmd_calibrate)

    select = sub.add_parser("select-modes")
    select.add_argument("--library", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--count", type=int, default=8)
    select.add_argument("--feature-length", type=float, default=.10)
    select.add_argument("--samples", type=int, default=2048)
    select.set_defaults(function=cmd_select_modes)

    transfer = sub.add_parser("build-transfer")
    transfer.add_argument("--library", required=True)
    transfer.add_argument("--shape", required=True)
    transfer.add_argument("--asset-root", default="assets/native_v1")
    transfer.add_argument("--output", required=True)
    transfer.add_argument("--pusher-radius", type=float, default=.018)
    transfer.add_argument("--clearance", type=float, default=.002)
    transfer.add_argument("--workspace-limit", type=float, default=.232)
    transfer.add_argument("--free-speed", type=float, default=.20)
    transfer.add_argument("--engage-speed", type=float, default=.01)
    transfer.set_defaults(function=cmd_build_transfer)

    search = sub.add_parser("search-q")
    search.add_argument("--library", required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--feature-length", type=float, default=.10)
    search.add_argument("--control-dt", type=float, default=.2)
    search.add_argument("--input-upper-mps", type=float, default=.06)
    search.add_argument("--position-error", type=float, default=.8)
    search.add_argument("--angle-error", type=float, default=np.deg2rad(60))
    search.add_argument("--samples", type=int, default=256)
    search.add_argument("--population", type=int, default=16)
    search.add_argument("--generations", type=int, default=40)
    search.add_argument("--restarts", type=int, default=3)
    search.add_argument("--sigma", type=float, default=1.)
    search.add_argument("--seed", type=int, default=20260911)
    search.add_argument("--no-endpoint-check", action="store_true")
    search.add_argument("--physical", action="store_true")
    search.add_argument("--model")
    search.add_argument("--asset-root", default="assets/native_v1")
    search.add_argument("--episodes-per-candidate", type=int, default=32)
    search.add_argument("--failure-penalty", type=float, default=1000.)
    search.add_argument("--position-min", type=float, default=.02)
    search.add_argument("--position-max", type=float, default=.08)
    search.add_argument("--max-ticks", type=int, default=100)
    search.add_argument("--hold-ticks", type=int, default=5)
    search.set_defaults(function=cmd_search_q)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--shape", required=True)
    evaluate.add_argument("--asset-root", default="assets/native_v1")
    evaluate.add_argument("--library", required=True)
    evaluate.add_argument("--model")
    evaluate.add_argument("--q")
    evaluate.add_argument("--parameters", type=float, nargs=5)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--seed", type=int, default=20260911)
    evaluate.add_argument("--physics-dt", type=float, default=.002)
    evaluate.add_argument("--control-dt", type=float, default=.2)
    evaluate.add_argument("--feature-length", type=float, default=.10)
    evaluate.add_argument("--input-upper-mps", type=float, default=.06)
    evaluate.add_argument("--position-min", type=float, default=.02)
    evaluate.add_argument("--position-max", type=float, default=.08)
    evaluate.add_argument("--angle-error", type=float, default=np.deg2rad(60))
    evaluate.add_argument("--workspace-limit", type=float, default=.232)
    evaluate.set_defaults(function=cmd_evaluate)

    encoder = sub.add_parser("train-encoder")
    encoder.add_argument("--data", required=True)
    encoder.add_argument("--output", required=True)
    encoder.add_argument("--hidden", type=int, default=64)
    encoder.add_argument("--epochs", type=int, default=1000)
    encoder.add_argument("--learning-rate", type=float, default=1e-3)
    encoder.add_argument("--seed", type=int, default=20260911)
    encoder.add_argument("--device", default="cpu")
    encoder.set_defaults(function=cmd_train_encoder)

    sealed = sub.add_parser("sealed-test")
    sealed.add_argument("--encoder", required=True)
    sealed.add_argument("--library-root", required=True)
    sealed.add_argument("--asset-root", default="assets/native_v1")
    sealed.add_argument("--shapes", nargs="+", required=True)
    sealed.add_argument("--output", required=True)
    sealed.add_argument("--model")
    sealed.add_argument("--episodes", type=int, default=200)
    sealed.add_argument("--seed", type=int, default=20260911)
    sealed.add_argument("--device", default="cpu")
    sealed.add_argument("--physics-dt", type=float, default=.002)
    sealed.add_argument("--control-dt", type=float, default=.2)
    sealed.add_argument("--feature-length", type=float, default=.10)
    sealed.add_argument("--input-upper-mps", type=float, default=.06)
    sealed.add_argument("--position-min", type=float, default=.02)
    sealed.add_argument("--position-max", type=float, default=.08)
    sealed.add_argument("--angle-error", type=float, default=np.deg2rad(60))
    sealed.add_argument("--workspace-limit", type=float, default=.232)
    sealed.add_argument("--max-ticks", type=int, default=100)
    sealed.add_argument("--hold-ticks", type=int, default=5)
    sealed.set_defaults(function=cmd_sealed_test)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = args.function(args)
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
    return 0
