"""Sealed external-shape inference for a frozen Full-M actor.

The original 32-shape fingerprints are validated against the checkpoint.  A
separate one-shape mode bank supplies only the measured C, validity mask, and
geometry needed for numerical inference.  This module has no training path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from shapely.geometry import shape as shapely_shape

from .full_m_control import (
    CODEC,
    FullMActor,
    FullMConfig,
    FullMControlEnv,
    ModeBank,
    candidates,
    hard_choice,
)
from .full_m_ppo import source_hashes
from .library import ModeLibrary
from .numeric_env import DEFAULT_LIBRARY


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def actor_parameter_sha256(actor: FullMActor) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def external_source_hashes() -> dict:
    paths = [Path(__file__)]
    cli = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_full_m_external.py"
    if cli.is_file():
        paths.append(cli)
    return {path.name: file_sha256(path) for path in paths}


class ExternalModeBank:
    """A one-row, shape-local bank whose only legal shape index is zero."""

    def __init__(
        self,
        root: str | Path,
        shape_name: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.root = Path(root).resolve()
        self.names = [str(shape_name)]
        reports = sorted(path.parent.name for path in self.root.glob("*/report.json"))
        if reports != self.names:
            raise ValueError(
                "external library root must contain exactly the requested shape report"
            )
        library = ModeLibrary.load(self.root / shape_name / "library")
        if library.shape != shape_name or library.count != 12 or not library.valid.all():
            raise ValueError("external shape requires exactly 12 valid measured modes")
        report = json.loads((self.root / shape_name / "report.json").read_text())
        authority = report.get("authority", {})
        if (
            report.get("status") != "COMPLETE"
            or not authority.get("four_direction_translation")
            or not authority.get("rotation_positive")
            or not authority.get("rotation_negative")
        ):
            raise ValueError("external library did not pass the frozen authority audit")

        self.device = torch.device(device)
        self.dtype = dtype
        self.fingerprints = {shape_name: library.fingerprint}
        self.C = torch.as_tensor(
            library.c.T[None], device=self.device, dtype=self.dtype
        )
        self.valid = torch.as_tensor(library.valid[None], device=self.device)
        polygon_record = json.loads(
            (self.root / shape_name / "polygon.json").read_text()
        )
        polygon = shapely_shape(polygon_record)
        vertices = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)[None]
        self.vertices = torch.as_tensor(
            vertices, device=self.device, dtype=self.dtype
        )
        self.vertex_mask = torch.ones(
            self.vertices.shape[:2], device=self.device, dtype=torch.bool
        )

    def batch(self, obs, recovery=None, hold=None, hold_steps=5):
        count = len(obs["pose"])
        local_shape = np.asarray(obs["shape_id"])
        if not np.array_equal(local_shape, np.zeros(count, dtype=np.int64)):
            raise ValueError("external environment emitted a nonlocal shape index")
        return {
            "e": torch.as_tensor(
                obs["error"], device=self.device, dtype=self.dtype
            ),
            "q": torch.as_tensor(obs["pose"], device=self.device, dtype=self.dtype),
            "target": torch.as_tensor(
                obs["target"], device=self.device, dtype=self.dtype
            ),
            "shape": torch.zeros(count, device=self.device, dtype=torch.long),
            "mode": torch.as_tensor(
                obs["current_mode"] - 1, device=self.device, dtype=torch.long
            ),
            "remaining": torch.as_tensor(
                obs["remaining_steps"], device=self.device, dtype=self.dtype
            ),
            "recovery": torch.as_tensor(
                np.zeros(count) if recovery is None else recovery,
                device=self.device,
                dtype=self.dtype,
            ),
            "hold": torch.as_tensor(
                np.zeros(count) if hold is None else hold,
                device=self.device,
                dtype=self.dtype,
            )
            / hold_steps,
        }


def load_frozen_actor(
    checkpoint: str | Path,
    training_library_root: str | Path = DEFAULT_LIBRARY,
    device: str | torch.device = "cpu",
):
    """Validate provenance and return an actor with all parameters frozen."""
    checkpoint = Path(checkpoint).resolve()
    record = torch.load(checkpoint, map_location=device, weights_only=True)
    if record.get("codec") != CODEC:
        raise ValueError("checkpoint uses an incompatible error codec")
    training_bank = ModeBank(training_library_root, device=device)
    if record.get("fingerprints") != training_bank.fingerprints:
        raise ValueError("original 32-shape fingerprint mismatch")
    if record.get("source_sha256") != source_hashes():
        raise ValueError("checkpoint core source hashes do not match current code")
    config = FullMConfig(**record["cfg"])
    actor = FullMActor(config).to(device)
    actor.load_state_dict(record["actor"])
    actor.eval()
    actor.requires_grad_(False)
    return actor, config, record, training_bank


def generate_test_cases(
    bank: ExternalModeBank,
    config: FullMConfig,
    shape_name: str,
    count: int,
    seed: int,
):
    if count < 100:
        raise ValueError("external final test requires at least 100 initial states")
    control = FullMControlEnv(bank, config, [shape_name] * count, count)
    data = control.reset(seed=seed)
    if not torch.equal(data["shape"], torch.zeros_like(data["shape"])):
        raise RuntimeError("external test cases are not using local shape index zero")
    pose = control.env.q.copy()
    target = control.env.target.copy()
    control.env.close()
    return pose, target


def create_seal(
    checkpoint: str | Path,
    output: str | Path,
    external_library_root: str | Path,
    shape_name: str,
    training_library_root: str | Path = DEFAULT_LIBRARY,
    count: int = 128,
    seed: int = 20260913033,
):
    """Freeze checkpoint, libraries, and initial states before any test result."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    actor, config, checkpoint_record, training_bank = load_frozen_actor(
        checkpoint, training_library_root, device="cpu"
    )
    external_bank = ExternalModeBank(
        external_library_root, shape_name, device="cpu"
    )
    pose, target = generate_test_cases(
        external_bank, config, shape_name, count, seed
    )
    cases_path = output / "test_cases.npz"
    np.savez_compressed(cases_path, pose=pose, target=target)
    checkpoint = Path(checkpoint).resolve()
    external_root = Path(external_library_root).resolve()
    external_shape_root = external_root / shape_name
    artifact_paths = [
        external_shape_root / "polygon.json",
        external_shape_root / "report.json",
        external_shape_root / "library" / "library.json",
        external_shape_root / "library" / "library.npz",
    ]
    artifact_paths += [
        path
        for path in [
            external_root / "shape_design.json",
            external_root / "calibration_cost.json",
            external_shape_root / "model.xml",
            external_shape_root / "modes.mp4",
            *sorted(external_shape_root.glob("mode_*.mp4")),
        ]
        if path.is_file()
    ]
    artifacts = {
        str(path.relative_to(external_root)): file_sha256(path)
        for path in sorted(set(artifact_paths))
    }
    seal = {
        "schema": 1,
        "status": "SEALED_BEFORE_EXTERNAL_CLOSED_LOOP_EVALUATION",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_next_update": int(checkpoint_record["next_update"]),
        "checkpoint_best_key": checkpoint_record.get("best_key"),
        "actor_parameter_sha256": actor_parameter_sha256(actor),
        "codec": config.codec,
        "config": asdict(config),
        "training_library_root": str(Path(training_library_root).resolve()),
        "training_fingerprints": training_bank.fingerprints,
        "training_fingerprint_set_sha256": canonical_sha256(
            training_bank.fingerprints
        ),
        "training_core_source_sha256": checkpoint_record["source_sha256"],
        "external_inference_source_sha256": external_source_hashes(),
        "external_library_root": str(Path(external_library_root).resolve()),
        "external_shape": shape_name,
        "external_fingerprint": external_bank.fingerprints[shape_name],
        "external_artifact_sha256": artifacts,
        "test_cases": str(cases_path),
        "test_cases_sha256": file_sha256(cases_path),
        "test_case_count": int(count),
        "test_seed": int(seed),
        "representative_indices_fixed_before_results": list(range(min(4, count))),
        "contract": {
            "scope": "frozen-M pure numerical integration with measured external C",
            "max_steps_including_hold": config.max_steps,
            "hold_steps": config.hold_steps,
            "position_tolerance_m": config.position_tol,
            "yaw_tolerance_rad": config.angle_tol,
            "coverage_tolerance": config.coverage_tol,
            "speed_domain_mps": [0.0, config.u_max],
            "positive_speed_grid_points": config.eval_speed_points,
            "mode_rule": "hold current normal/recovery mode until infeasible",
            "actor_parameter_updates": 0,
        },
    }
    write_json(output / "seal.json", seal)
    return seal


def verify_seal(seal_path: str | Path, device: str | torch.device):
    seal_path = Path(seal_path).resolve()
    seal = json.loads(seal_path.read_text())
    if seal.get("status") != "SEALED_BEFORE_EXTERNAL_CLOSED_LOOP_EVALUATION":
        raise ValueError("invalid external evaluation seal")
    checkpoint = Path(seal["checkpoint"])
    cases_path = Path(seal["test_cases"])
    if file_sha256(checkpoint) != seal["checkpoint_sha256"]:
        raise ValueError("sealed checkpoint bytes changed")
    if file_sha256(cases_path) != seal["test_cases_sha256"]:
        raise ValueError("sealed initial-state file changed")
    actor, config, checkpoint_record, training_bank = load_frozen_actor(
        checkpoint, seal["training_library_root"], device=device
    )
    if training_bank.fingerprints != seal["training_fingerprints"]:
        raise ValueError("sealed training library fingerprints changed")
    if checkpoint_record["source_sha256"] != seal["training_core_source_sha256"]:
        raise ValueError("sealed core source record changed")
    if external_source_hashes() != seal["external_inference_source_sha256"]:
        raise ValueError("sealed external inference source changed")
    if actor_parameter_sha256(actor) != seal["actor_parameter_sha256"]:
        raise ValueError("sealed actor parameter hash mismatch")
    external_bank = ExternalModeBank(
        seal["external_library_root"], seal["external_shape"], device=device
    )
    if external_bank.fingerprints[seal["external_shape"]] != seal[
        "external_fingerprint"
    ]:
        raise ValueError("sealed external library fingerprint changed")
    external_root = Path(seal["external_library_root"])
    for relative, expected in seal["external_artifact_sha256"].items():
        if file_sha256(external_root / relative) != expected:
            raise ValueError(f"sealed external artifact changed: {relative}")
    cases = np.load(cases_path)
    pose = np.asarray(cases["pose"])
    target = np.asarray(cases["target"])
    if pose.shape != (seal["test_case_count"], 3) or target.shape != pose.shape:
        raise ValueError("sealed test-case dimensions changed")
    return seal, actor, config, external_bank, pose, target


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def render_representatives(
    output: Path,
    trace: list[dict],
    report: dict,
    seal: dict,
    polygon_record: dict,
) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    fixed = list(seal["representative_indices_fixed_before_results"])
    failures = [episode["index"] for episode in report["episodes"] if not episode["success"]]
    selected = fixed + [index for index in failures[:4] if index not in fixed]
    size = 256
    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)
    rings = [np.asarray(ring, dtype=np.float64) for ring in polygon_record["coordinates"]]

    def world(q, points):
        cosine, sine = np.cos(q[2]), np.sin(q[2])
        return points @ np.array([[cosine, sine], [-sine, cosine]]) + q[:2]

    def pixels(points):
        return [
            (float(x / 0.5 * size), float((1 - y / 0.5) * size))
            for x, y in points
        ]

    frames = []
    for step, row in enumerate(trace):
        canvas = Image.new("RGB", (cols * size, rows * size), "white")
        for tile_index, episode_index in enumerate(selected):
            image = Image.new("RGB", (size, size), "white")
            draw = ImageDraw.Draw(image)
            target = row["target"][episode_index]
            pose = row["q"][episode_index]
            draw.polygon(pixels(world(target, rings[0])), fill="#d7efdc", outline="#21866a", width=2)
            draw.polygon(pixels(world(pose, rings[0])), fill="#bed8ed", outline="#2369b0", width=2)
            action = row["action"][episode_index]
            label = (
                f"case {episode_index} step {step + 1}\n"
                f"m={int(action[0])} u={action[1]:.3f} "
                f"R={int(row['recovery_active'][episode_index])}"
            )
            draw.text((5, 5), label, fill="black")
            canvas.paste(image, ((tile_index % cols) * size, (tile_index // cols) * size))
        frames.append(np.asarray(canvas))
    video = output / "representative_closed_loop.mp4"
    with imageio.get_writer(video, fps=5, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)
    write_json(
        output / "representative_cases.json",
        {
            "fixed_before_results": fixed,
            "first_failures_in_full_denominator": failures[:4],
            "rendered_indices": selected,
        },
    )


def evaluate_sealed(
    seal_path: str | Path,
    output: str | Path,
    device: str | torch.device = "cuda",
    video: bool = True,
):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seal, actor, config, bank, pose, target = verify_seal(seal_path, device)
    parameter_hash_before = actor_parameter_sha256(actor)
    count = len(pose)
    shape_name = seal["external_shape"]
    control = FullMControlEnv(bank, config, [shape_name] * count, count)
    data = control.reset(options={"pose": pose, "target": target})
    trace = []
    started = time.perf_counter()
    try:
        for _ in range(config.max_steps):
            if control.env.done.all():
                break
            with torch.no_grad():
                candidate = candidates(
                    actor,
                    data,
                    bank,
                    config,
                    speed_points=config.eval_speed_points,
                )
                at_goal = torch.as_tensor(control.at_goal(), device=bank.device)
                mode, speed, recovery, failed = hard_choice(
                    candidate,
                    data,
                    config,
                    torch.as_tensor(control.recovery, device=bank.device),
                    at_goal,
                )
            grid_no_action = ~candidate["feasible"].any(-1).cpu().numpy()
            row = {key: value.detach().cpu().numpy() for key, value in data.items()}
            row.update(
                {
                    key: candidate[key].detach().cpu().numpy()
                    for key in [
                        "M",
                        "L",
                        "V",
                        "gradient",
                        "beta",
                        "flow",
                        "endpoint",
                        "both",
                        "physical",
                        "feasible",
                    ]
                }
            )
            row.update(
                action=np.c_[mode.cpu().numpy(), speed.cpu().numpy()],
                active=~control.env.done.copy(),
                recovery_active=recovery.cpu().numpy(),
                grid_no_action=grid_no_action,
            )
            canonical_speed = np.minimum(
                speed.cpu().numpy().astype(float), config.u_max
            )
            data, reward, _, _ = control.advance(
                mode.cpu().numpy(),
                canonical_speed,
                grid_no_action,
                recovery.cpu().numpy(),
                failed.cpu().numpy(),
            )
            row.update(
                reward=reward,
                pose_next=control.env.q.copy(),
                coverage=control.info["coverage"].copy(),
            )
            trace.append(row)

        success = control.env.reason == "SUCCESS"
        error = control.obs["error"]
        position_error = np.linalg.norm(error[:, :2], axis=1) * config.length_scale
        yaw_error_deg = np.abs(np.rad2deg(error[:, 2]))
        coverage = control.info["coverage"]
        parameter_hash_after = actor_parameter_sha256(actor)
        parameter_gradients_created = any(
            parameter.grad is not None for parameter in actor.parameters()
        )
        episodes = [
            {
                "index": index,
                "success": bool(success[index]),
                "reason": str(control.env.reason[index]),
                "steps": int(control.env.steps[index]),
                "position_error_m": float(position_error[index]),
                "yaw_error_deg_abs": float(yaw_error_deg[index]),
                "coverage": float(coverage[index]),
                "switches": int(control.env.switches[index]),
                "no_action_steps": int(control.no_action_steps[index]),
                "recovery_steps": int(control.recovery_steps[index]),
                "cumulative_error": float(control.cost_error[index]),
                "max_position_error_m": float(control.max_position[index]),
            }
            for index in range(count)
        ]
        report = {
            "schema": 1,
            "scope": "sealed external-shape hard numeric evaluation; no contact simulation",
            "shape": shape_name,
            "codec": config.codec,
            "denominator": count,
            "successes": int(success.sum()),
            "failures": int((~success).sum()),
            "success_rate": float(success.mean()),
            "failure_reasons": dict(Counter(control.env.reason[~success].astype(str))),
            "terminal_position_error_m": distribution(position_error),
            "terminal_yaw_error_deg_abs": distribution(yaw_error_deg),
            "terminal_coverage": distribution(coverage),
            "switches": distribution(control.env.switches),
            "no_action_steps": distribution(control.no_action_steps),
            "recovery_steps": distribution(control.recovery_steps),
            "cumulative_error": distribution(control.cost_error),
            "steps": distribution(control.env.steps),
            "max_position_error_m": distribution(control.max_position),
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_sha256": seal["checkpoint_sha256"],
            "training_fingerprint_set_sha256": seal[
                "training_fingerprint_set_sha256"
            ],
            "external_fingerprint": seal["external_fingerprint"],
            "test_cases_sha256": seal["test_cases_sha256"],
            "actor_parameter_sha256_before": parameter_hash_before,
            "actor_parameter_sha256_after": parameter_hash_after,
            "actor_parameters_changed": parameter_hash_before != parameter_hash_after,
            "actor_parameter_gradients_created": parameter_gradients_created,
            "optimizer_created": False,
            "loss_backward_called": False,
            "state_gradient_method": "torch.autograd.grad with respect to D for full dM/dD",
            "episodes": episodes,
        }
        if report["actor_parameters_changed"] or parameter_gradients_created:
            raise RuntimeError("external inference modified actor training state")
        write_json(output / "report.json", report)
        arrays = {
            key: np.stack([row[key] for row in trace]) for key in trace[0]
        }
        np.savez_compressed(output / "trajectory_and_candidates.npz", **arrays)
        examples = {}
        example_indices = list(seal["representative_indices_fixed_before_results"])
        example_indices += [
            episode["index"]
            for episode in episodes
            if not episode["success"] and episode["index"] not in example_indices
        ][:4]
        for episode_index in example_indices:
            active_steps = [
                step for step, row in enumerate(trace) if row["active"][episode_index]
            ]
            chosen_steps = sorted(
                set(
                    [
                        active_steps[0],
                        active_steps[len(active_steps) // 2],
                        active_steps[-1],
                    ]
                )
            )
            examples[str(episode_index)] = [
                {
                    "step": step + 1,
                    "M": trace[step]["M"][episode_index].tolist(),
                    "L": trace[step]["L"][episode_index].tolist(),
                    "V": float(trace[step]["V"][episode_index]),
                    "gradient": trace[step]["gradient"][episode_index].tolist(),
                    "action": trace[step]["action"][episode_index].tolist(),
                }
                for step in chosen_steps
            ]
        write_json(output / "matrix_examples.json", examples)
        if video:
            polygon_record = json.loads(
                (
                    Path(seal["external_library_root"])
                    / shape_name
                    / "polygon.json"
                ).read_text()
            )
            render_representatives(output, trace, report, seal, polygon_record)
        return report
    finally:
        control.env.close()
