"""Frozen RGB controller contrasts on declared development/validation starts.

This optional diagnostic module changes no original training implementation.
Finite-difference J is an oracle diagnostic: it can render unobserved nearby
poses and is not a deployable RGB policy. The P1 actuator delay remains intact.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import math
import time

import numpy as np
import torch

from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
from stabilization_gym.tasks_v2.visual import VisualPolicyOutput, quadratic_value, radial_limit
from stabilization_learning.joint_v3.inference import FrozenRGBPolicy
from stabilization_learning.joint_v2.runtime import atomic_json, file_sha256
from extensions.joint_certificate_v4.rollout import diagnostics
from extensions.joint_certificate_v2.rgb import _npz


VARIANTS = ("baseline", "target_J", "relative_damping", "identity_M", "FD_current_J")


@torch.no_grad()
def finite_difference_interaction(env, features, geometry, tau):
    """Feature response per normalized command per second, no physical step.

    A translation column uses the current body frame, not the goal frame.
    ``tau`` is a command-time product. It corresponds to 0.4*tau metres on
    translation axes and 0.35*tau radians on yaw in the frozen task.
    Rendering mutates renderer camera buffers only; all plant state is retained.
    """
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Finite-difference tau must be finite and positive")
    from .derivative_geometry import fd_interaction
    return fd_interaction(env, features, geometry, step=tau)


class FrozenAblationPolicy:
    """Task-side controller construction with immutable external phi and M."""

    def __init__(self, learner, geometry, initial_public, variant, *, fd_tau=None,
                 fd_callback=finite_difference_interaction):
        if variant not in VARIANTS:
            raise ValueError(f"Undeclared controller variant: {variant}")
        if variant == "FD_current_J" and fd_tau is None:
            raise ValueError("FD requires a tau fixed by training geometry diagnostics")
        self.learner, self.geometry, self.variant = learner, geometry, variant
        self.fd_tau, self.fd_callback = fd_tau, fd_callback
        self.frozen = FrozenRGBPolicy(learner, identity=variant == "identity_M")
        with torch.no_grad():
            goal_images = geometry.encode_images(initial_public["goal_rgb"])
            self.goal = learner.features.encode(goal_images)
            self.metric = self.frozen._metric(self.goal)
            self.target_interaction = None
            if variant == "target_J":
                depth = geometry.predict_depth(initial_public["goal_rgb"])
                image, tangents = geometry.tangents(initial_public["goal_rgb"], depth,
                                                    initial_public["rotation"])
                self.target_interaction = learner.adapter.interaction(
                    learner.features, image, tangents, create_graph=False)

    @torch.no_grad()
    def __call__(self, public, env=None):
        self.frozen._check()
        if self.variant in {"target_J", "FD_current_J"}:
            image = self.geometry.encode_images(public["rgb"])
            interaction = (self.target_interaction if self.variant == "target_J" else
                           self.fd_callback(env, self.learner.features, self.geometry, self.fd_tau))
        else:
            image, _, tangents = self.geometry.observe(public)
            interaction = self.learner.adapter.interaction(
                self.learner.features, image, tangents, create_graph=False)
        error = self.learner.features.encode(image) - self.goal
        if self.variant == "relative_damping":
            jt = interaction.transpose(-1, -2)
            gram = jt @ self.metric @ interaction
            scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1)
            eye = torch.eye(4, dtype=gram.dtype, device=gram.device)
            config = self.learner.adapter.config
            # Predeclared dimensionless ratio; the floor is numerical only.
            regularizer = (config.damping * scale + 1e-8)[..., None, None] * eye
            hessian = gram + config.diagonal_damping * torch.diag_embed(
                gram.diagonal(dim1=-2, dim2=-1)) + regularizer
            gradient = (jt @ self.metric @ error.unsqueeze(-1)).squeeze(-1)
            raw = -config.gain * torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
            action = radial_limit(raw)
        else:
            action, raw, hessian = self.learner.adapter.solve(error, interaction, self.metric)
        self.frozen._check()
        return VisualPolicyOutput(action, raw, error, self.goal, interaction, self.metric,
                                  quadratic_value(error, self.metric), hessian)


def _summary(rows):
    values = np.asarray([r["integrated_error"] for r in rows], dtype=float)
    return dict(episodes=len(rows), successes=sum(r["success"] for r in rows),
                success_rate=float(np.mean([r["success"] for r in rows])),
                success_cases=[r["case_id"] for r in rows if r["success"]],
                reasons=dict(Counter(r["reason"] for r in rows)),
                mean_integrated_error=float(values.mean()), median_integrated_error=float(np.median(values)),
                p90_integrated_error=float(np.quantile(values, .9)), max_integrated_error=float(values.max()),
                mean_final_physical_error=float(np.mean([r["final_physical_error"] for r in rows])),
                mean_input_squared_integral=float(np.mean([r["input_squared_integral"] for r in rows])),
                mean_policy_wall_s=float(np.mean([r["mean_policy_wall_s"] for r in rows])),
                aggregation="episode-level means; no confidence interval or independent-time-sample claim")


def paired_comparison(baseline, candidate):
    base = {r["case_id"]: r for r in baseline}
    changed = {r["case_id"]: r for r in candidate}
    if set(base) != set(changed):
        raise ValueError("Paired comparison requires the same complete case population")
    keys = sorted(base)
    error = np.array([changed[k]["integrated_error"] - base[k]["integrated_error"] for k in keys])
    common = [k for k in keys if base[k]["success"] and changed[k]["success"]]
    arrival = [changed[k]["elapsed_s"] - base[k]["elapsed_s"] for k in common]
    return dict(episodes=len(keys), candidate_only_success=[k for k in keys if changed[k]["success"] and not base[k]["success"]],
                baseline_only_success=[k for k in keys if base[k]["success"] and not changed[k]["success"]],
                lower_error_cases=int(np.sum(error < 0)), higher_error_cases=int(np.sum(error > 0)),
                mean_error_difference=float(error.mean()), median_error_difference=float(np.median(error)),
                common_success_cases=common, common_success_count=len(common),
                mean_common_success_time_difference_s=float(np.mean(arrival)) if arrival else None,
                paired_time_differences_s=arrival)


def evaluate_variant(*, cfg, scenes_by_id, cases, learner, geometry, output, project,
                     variant, plant, guard=lambda: None, event=lambda *a, **k: None,
                     parallel_envs=16, max_steps=200, samples_per_axis=2,
                     fd_tau=None, fd_callback=finite_difference_interaction):
    """Full frozen evaluation, preserving first terminal and true successors."""
    if any(scenes_by_id[c["scene_id"]]["split"] not in {"train", "val"} for c in cases):
        raise ValueError("Diagnostic control evaluation rejects held-out test scenes")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Case identifiers must be unique")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    atomic_json(output / "PROTOCOL.json", dict(variant=variant, plant=plant, cases=cases,
        steps=max_steps, control=asdict(learner.adapter.config), fd_tau=fd_tau,
        source=str(source), source_sha256=file_sha256(source),
        relative_damping="H=G+0.01diag(G)+(0.02tr(G)/4+1e-8)I",
        oracle=variant == "FD_current_J", inference_wall_time_injected=False,
        criterion="unchanged task position/yaw/hold/collision/workspace criteria"))
    rows = []
    begun = time.perf_counter()
    event("control_variant_started", variant=variant, plant=plant, cases=len(cases), fd_tau=fd_tau)
    for start in range(0, len(cases), parallel_envs):
        guard()
        group = cases[start:start + parallel_envs]
        env = GPUParallelRGBEnv(cfg, [scenes_by_id[c["scene_id"]] for c in group], plant=plant,
            max_steps=max_steps, device=str(geometry.device),
            cache_dir=Path(project) / ".cache/rgb_gpu_warp", samples_per_axis=samples_per_axis)
        keys = ("error", "V", "J", "action", "applied_action", "offset", "physical_error", "dt",
                "physical_cost", "input_cost", "integrated_error", "previous_action", "actual_velocity",
                "hold_observations", "success", "terminated", "truncated")
        trace = {k: [] for k in keys}
        wall = []
        try:
            public, info = env.reset(options=dict(offsets=[c["offset"] for c in group], goal_ids=[c["goal_id"] for c in group]))
            session = FrozenAblationPolicy(learner, geometry, public, variant, fd_tau=fd_tau, fd_callback=fd_callback)
            alive = torch.ones(len(group), dtype=torch.bool, device=geometry.device)
            lengths = torch.zeros(len(group), dtype=torch.int64, device=geometry.device)
            last = {k: torch.zeros_like(alive) for k in ("success", "collision", "outside")}
            trace["offset"].append(info["supervision"]["offset"])
            trace["physical_error"].append(info["supervision"]["physical_error"])
            for step in range(max_steps):
                guard()
                if not bool(alive.any()):
                    break
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = session(public, env)
                torch.cuda.synchronize()
                wall.append(time.perf_counter() - t)
                if step == 0:
                    matrix = out.metric.clone()
                    trace["error"].append(out.error)
                    trace["V"].append(out.value)
                elif not torch.equal(out.metric, matrix):
                    raise RuntimeError("Frozen M changed during a control episode")
                following, _, terminal, truncated, info = env.step(out.action)
                if not bool(info["transition_mask"][alive].all()):
                    raise RuntimeError("Reset transition contaminated control evaluation")
                truth = info["supervision"]
                with torch.no_grad():
                    next_error = learner.features.encode(geometry.encode_images(following["rgb"])) - out.goal
                    next_value = quadratic_value(next_error, out.metric)
                for name, value in (("error", next_error), ("V", next_value), ("J", out.interaction),
                    ("action", out.action), ("applied_action", (truth["actual_velocity"] / env.scale).float()),
                    ("offset", truth["next_offset"]), ("physical_error", truth["physical_error"]),
                    ("dt", info["dt"]), ("physical_cost", truth["physical_cost"]),
                    ("input_cost", truth["normalized_input_squared_integral"]),
                    ("integrated_error", truth["integrated_error"]), ("previous_action", truth["previous_action"]),
                    ("actual_velocity", truth["actual_velocity"]), ("hold_observations", info["hold_observations"]),
                    ("success", info["success"]), ("terminated", terminal), ("truncated", truncated)):
                    trace[name].append(value)
                lengths += alive.long()
                ending = alive & (terminal | truncated)
                for name in last:
                    last[name] |= ending & info[name]
                alive &= ~(terminal | truncated)
                public = following
            session.frozen._check()
            host = {k: torch.stack(v).cpu().numpy() for k, v in trace.items()}
            matrices = matrix.cpu().numpy()
            last = {k: v.cpu().tolist() for k, v in last.items()}
            for i, (case, n) in enumerate(zip(group, lengths.cpu().tolist())):
                arrays = {k: v[:n + (k in {"error", "V", "offset", "physical_error"}), i] for k, v in host.items()}
                arrays["M"] = matrices[i]
                path = output / f"case_{start+i:04d}.npz"
                _npz(path, arrays)
                row = dict(**case, plant=plant, method=variant, steps=n, version=learner.version,
                    success=last["success"][i], reason="success" if last["success"][i] else
                    "collision" if last["collision"][i] else "outside" if last["outside"][i] else "timeout",
                    elapsed_s=float(arrays["dt"].sum()), integrated_error=float(arrays["integrated_error"].sum()),
                    input_squared_integral=float(arrays["input_cost"].sum()), physical_cost=float(arrays["physical_cost"].sum()),
                    final_physical_error=float(arrays["physical_error"][-1]),
                    action_variation=float(np.square(arrays["action"] - arrays["previous_action"]).sum()),
                    mean_policy_wall_s=float(np.mean(wall[:n])), trajectory=str(path), trajectory_sha256=file_sha256(path),
                    oracle=variant == "FD_current_J", **diagnostics(arrays, learner.config))
                if row["success"] and arrays["hold_observations"][-1] < cfg["collection"]["required_hold_observations"]:
                    raise RuntimeError("Success lacks required hold observations")
                rows.append(row)
                event("control_episode_complete", plant=plant, variant=variant, case_id=case["case_id"],
                      success=row["success"], integrated_error=row["integrated_error"], reason=row["reason"])
            event("control_batch_complete", plant=plant, variant=variant, completed=len(rows), total=len(cases))
        except BaseException as error:
            if any(trace.values()):
                _npz(output / f"batch_{start:04d}_PARTIAL.npz", {k: torch.stack(v).cpu().numpy() for k, v in trace.items() if v})
            atomic_json(output / f"batch_{start:04d}_FAILURE.json", dict(error=repr(error), eligible_result=False))
            raise
        finally:
            env.close()
    result = dict(rows=rows, summary=_summary(rows), wall_s=time.perf_counter() - begun)
    atomic_json(output / "RESULTS.json", result)
    event("control_variant_complete", variant=variant, plant=plant, **result["summary"], wall_s=result["wall_s"])
    return result


def aggregate_directory(output):
    """Combine complete variants only; preserve every failed/partial directory."""
    output = Path(output)
    complete = {}
    for path in sorted(output.glob("*/*/RESULTS.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        complete[f"{path.parent.parent.name}/{path.parent.name}"] = result
    summaries = {key: value["summary"] for key, value in complete.items()}
    pairs = {}
    for plant in ("P0", "P1"):
        baseline = complete.get(f"{plant}/baseline")
        if baseline:
            for variant in VARIANTS[1:]:
                result = complete.get(f"{plant}/{variant}")
                if result:
                    pairs[f"{plant}/{variant}"] = paired_comparison(baseline["rows"], result["rows"])
    return dict(summaries=summaries, paired=pairs)


def run_ablation(ctx, output=None, *, learner=None, methods=VARIANTS,
                 plants=("P0", "P1"), cases=None, fd_tau=None,
                 parallel_envs=16, max_steps=200, verify_original_baseline=False):
    """Use a supplied learner so later training arms retain their own weights.

    Each destination must be new. This does not resume partial variants or
    overwrite results. The caller freezes methods, cases and FD selection.
    """
    learner = ctx.learner if learner is None else learner
    cases = ctx.validation_cases if cases is None else cases
    output = Path(output or (Path(ctx.root) / "control"))
    output.mkdir(parents=True, exist_ok=False)
    scenes_by_id = {s["id"]: s for s in ctx.scenes}
    complete = {}
    for plant in plants:
        for variant in methods:
            ctx.guard()
            complete[f"{plant}/{variant}"] = evaluate_variant(
                cfg=ctx.cfg, scenes_by_id=scenes_by_id, cases=cases, learner=learner,
                geometry=ctx.geometry, output=output / plant / variant, project=ctx.project,
                variant=variant, plant=plant, guard=ctx.guard,
                event=lambda kind, **kwargs: ctx.event("control", kind=kind, **kwargs),
                parallel_envs=parallel_envs, max_steps=max_steps, fd_tau=fd_tau)
            if verify_original_baseline and plant == "P0" and variant == "baseline":
                reference_path = Path(ctx.old_root) / "validation/pair_006500/RESULTS.json"
                reference = json.loads(reference_path.read_text(encoding="utf-8"))
                expected = {row["case_id"]: row for row in reference}
                actual = {row["case_id"]: row for row in complete[f"{plant}/{variant}"]["rows"]}
                if set(actual) != set(expected):
                    raise RuntimeError("Baseline validation starts differ from original full population")
                mismatched_success = [key for key in actual if actual[key]["success"] != expected[key]["success"]]
                absolute = max(abs(actual[key]["integrated_error"] - expected[key]["integrated_error"]) for key in actual)
                close = all(np.isclose(actual[key]["integrated_error"], expected[key]["integrated_error"],
                                       rtol=1e-3, atol=1e-5) for key in actual)
                check = dict(reference=str(reference_path), reference_sha256=file_sha256(reference_path),
                             episodes=len(actual), success_mismatches=mismatched_success,
                             maximum_integrated_error_difference=absolute,
                             error_tolerance=dict(rtol=1e-3, atol=1e-5),
                             passed=not mismatched_success and close)
                atomic_json(output / "BASELINE_REPRODUCTION.json", check)
                ctx.event("control_baseline_reproduction", **check)
                if not check["passed"]:
                    raise RuntimeError("Baseline reproduction needs investigation before attributing ablation effects")
    combined = aggregate_directory(output)
    atomic_json(output / "RESULTS.json", combined)
    return combined


def main():
    import argparse
    from .common import bootstrap, freeze_stage, gpu_lease, read_json, event
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--fd-tau", type=float)
    parser.add_argument("--methods", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--plants", nargs="+", choices=("P0", "P1"), default=["P0", "P1"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.root)
    protocol = read_json(root / "PROTOCOL.json")
    if "FD_current_J" in args.methods and args.fd_tau is None:
        raise ValueError("FD tau must be declared from training-state geometry diagnostics")
    output = args.output or str(root / "control")
    freeze_stage(root, "control_ablation", dict(methods=args.methods, plants=args.plants,
        validation_cases_sha256=protocol["validation_cases_sha256"], fd_tau=args.fd_tau,
        max_steps=200, parallel_envs=16, output=str(output),
        relative_damping="0.02*trace(G)/4+1e-8; diagonal damping and gain unchanged",
        fd_scope="shared derivative_geometry.fd_interaction; oracle instant geometry only; P1 delay retained",
        fd_selection="training-state numerical consistency, not validation success"),
        [Path(__file__), Path(__file__).with_name("derivative_geometry.py")])
    try:
        with gpu_lease(root, "control_ablation"):
            ctx = bootstrap(args.root, device=args.device)
            result = run_ablation(ctx, output, methods=args.methods, plants=args.plants, fd_tau=args.fd_tau,
                                  verify_original_baseline=True)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        event(root, "control_ablation_failure", error=repr(error))
        raise


if __name__ == "__main__":
    main()
