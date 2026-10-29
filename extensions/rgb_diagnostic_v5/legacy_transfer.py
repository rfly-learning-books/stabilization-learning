"""Frozen historical RGB controllers transferred to the current GPU rooms.

This does not recreate the historical renderer or its 300-step evaluation. All
methods use the same declared 48 development cases, 200 steps, and unchanged
success tolerances. Retained historical weights are explicit diagnostic inputs.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .common import bootstrap, freeze_stage, gpu_lease, read_json, sha256, write_json, event
from .control_ablation import FrozenAblationPolicy, _summary, paired_comparison
from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime
from stabilization_gym.tasks_v2.visual import quadratic_value, radial_limit
from extensions.joint_certificate_v2.rgb import _npz
from extensions.joint_certificate_v4.rollout import diagnostics


METHODS = ("legacy_initial_identity", "legacy_best_identity", "legacy_best_M",
           "v4_current_J_legacy_ridge", "v4_target_J_legacy_ridge")


def legacy_solve(error, interaction, metric):
    """Exact retained normalized-command ridge and radial saturation."""
    jt = interaction.transpose(-1, -2)
    gram = jt @ metric @ interaction
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    ridge = torch.maximum(1.01*trace/(1e4-1), trace.new_full(trace.shape, 1e-8))
    eye = torch.eye(4, dtype=gram.dtype, device=gram.device)
    hessian = gram + .01*torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1)) + ridge[..., None, None]*eye
    gradient = (jt @ metric @ error.unsqueeze(-1)).squeeze(-1)
    raw = -.7*torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
    return radial_limit(raw), raw, hessian


def model_stamp(*modules):
    return tuple((id(t), t._version, str(t.device), str(t.dtype))
                 for module in modules for t in (*module.parameters(), *module.buffers()))


class LegacyPolicy:
    def __init__(self, adapter, ae, metric, depth, public, *, identity=False):
        self.adapter, self.ae, self.metric_net = adapter, ae, metric
        self.device = next(ae.parameters()).device
        self.stamp = model_stamp(ae, metric)
        rgb = public["goal_rgb"].detach().cpu().numpy()
        rotations = public["rotation"].detach().cpu().numpy()
        # Direct original Reference preserves CPU image preprocessing, J/4,
        # historical CAE center/scale, goal metric input, ridge, and gain.
        refs, cache = [], {}
        for image, rotation in zip(rgb, rotations):
            key = (image.tobytes(), rotation.tobytes())
            if key not in cache:
                estimated_depth = depth.predict(image)
                cache[key] = adapter.reference(ae, metric, image, rotation, depth=estimated_depth,
                                               device=str(self.device), identity=identity)
            refs.append(cache[key])
        self.references = refs
        self.goal = torch.cat([r.goal for r in refs])/4.
        self.interaction = torch.cat([r.j for r in refs])
        self.metric = torch.cat([r.m for r in refs])
        self.gain = torch.cat([r.k for r in refs])
        self.hessian = torch.cat([r.h for r in refs])
        self.check()

    def check(self):
        if self.stamp != model_stamp(self.ae, self.metric_net):
            raise RuntimeError("Historical model/coordinate parameters changed")
        if self.ae.training or self.metric_net.training:
            raise RuntimeError("Historical inference modules must remain in eval mode")

    @torch.no_grad()
    def error(self, rgb):
        images = rgb.detach().cpu().numpy()
        prepared = self.adapter.vision.image_tensor(self.adapter.cfg, images, str(self.device))
        return self.ae.encode(prepared)/4. - self.goal

    @torch.no_grad()
    def __call__(self, public):
        self.check()
        error = self.error(public["rgb"])
        raw = -(self.gain @ error.unsqueeze(-1)).squeeze(-1)
        action = self.adapter.core.saturate(raw)
        self.check()
        return SimpleNamespace(action=action, raw_action=raw, error=error, goal=self.goal,
            interaction=self.interaction, metric=self.metric, hessian=self.hessian,
            value=quadratic_value(error, self.metric))


class V4LegacyRidgePolicy:
    def __init__(self, ctx, public, *, target=False):
        self.base = FrozenAblationPolicy(ctx.learner, ctx.geometry, public, "target_J" if target else "baseline")
        self.ctx = ctx

    @torch.no_grad()
    def error(self, rgb):
        return self.ctx.learner.features.encode(self.ctx.geometry.encode_images(rgb)) - self.base.goal

    @torch.no_grad()
    def __call__(self, public):
        result = self.base(public)
        result.action, result.raw_action, result.hessian = legacy_solve(result.error, result.interaction, result.metric)
        return result

    def check(self):
        self.base.frozen._check()


def load_historical(ctx):
    # The online subclass inherits the audited OriginalVisualAdapter loader;
    # it legitimately binds shared numerical modules to reconstructed rooms.
    adapter, _, _ = load_runtime(ctx.project)
    models = {}
    for name in ("initial.pt", "best.pt"):
        ae, metric = adapter.load_models(name, device=ctx.device)
        ae.eval().requires_grad_(False)
        metric.eval().requires_grad_(False)
        models[name] = (ae, metric)
    depth = adapter.core.FastDepth(adapter.cfg, ctx.device)
    return adapter, models, depth


def make_policy(ctx, historical, public, method):
    if method.startswith("legacy_"):
        adapter, models, depth = historical
        name = "initial.pt" if method == "legacy_initial_identity" else "best.pt"
        ae, metric = models[name]
        return LegacyPolicy(adapter, ae, metric, depth, public, identity=method != "legacy_best_M")
    return V4LegacyRidgePolicy(ctx, public, target=method == "v4_target_J_legacy_ridge")


def preflight(ctx, historical, output):
    cases = ctx.train_cases[:4]
    env = ctx.make_env(cases, plant="P0", max_steps=200)
    results = {}
    try:
        public, _ = env.reset(options=dict(offsets=[c["offset"] for c in cases], goal_ids=[c["goal_id"] for c in cases]))
        raw = public["rgb"].cpu().numpy()
        adapter = historical[0]
        for method in METHODS:
            ctx.guard()
            policy = make_policy(ctx, historical, public, method)
            out = policy(public)
            expected, _ = adapter.core.policy(out.error, out.interaction, out.metric)
            formula_error = float((out.action-expected).abs().max())
            assert torch.allclose(out.action, expected, atol=5e-4, rtol=1e-4)
            record = dict(formula_max_action_difference=formula_error,
                          finite=bool(torch.isfinite(out.action).all()),
                          translation_bound=bool((out.action[:, :3].norm(dim=1) <= 1+1e-6).all()),
                          yaw_bound=bool((out.action[:, 3].abs() <= 1+1e-6).all()))
            if isinstance(policy, LegacyPolicy):
                max_action = max_error = max_value = max_physical_action = 0.
                for i, (image, reference) in enumerate(zip(raw, policy.references)):
                    command, error, value = reference.observe(image)
                    max_action = max(max_action, float(np.max(np.abs(command-out.action[i].cpu().numpy()))))
                    max_error = max(max_error, float(np.max(np.abs(error-out.error[i].cpu().numpy()))))
                    max_value = max(max_value, abs(value-float(out.value[i])))
                    physical = adapter.calculate_action_from_M(out.metric[i:i+1], out.error[i:i+1], reference)
                    normalized = physical / physical.new_tensor(adapter.physical_scales)
                    max_physical_action = max(max_physical_action, float((normalized-out.action[i:i+1]).abs().max()))
                    assert np.allclose(command, out.action[i].cpu().numpy(), atol=5e-4, rtol=1e-4)
                    assert np.allclose(error, out.error[i].cpu().numpy(), atol=1e-4, rtol=1e-5)
                original = adapter.vision.image_tensor(adapter.cfg, raw, ctx.device)
                gpu = ctx.geometry.encode_images(public["rgb"])
                record.update(reference_observe_max_action_difference=max_action,
                    reference_observe_max_error_difference=max_error,
                    reference_observe_max_V_difference=max_value,
                    adapter_physical_action_normalized_max_difference=max_physical_action,
                    GPU_vs_original_preprocessing_max_difference=float((gpu-original).abs().max()),
                    inference_preprocessing="exact original CPU image_tensor regardless of GPU comparison",
                    J_scaling="direct original Reference encoder_jvp divides by sqrt(16)=4",
                    model_center=policy.ae.center.detach().cpu().tolist(),
                    model_scale=policy.ae.scale.detach().cpu().tolist())
            results[method] = record
            policy.check()
        write_json(output/"PREFLIGHT.json", dict(passed=True, cases=cases, checks=results,
            note="Engineering check on four fixed training states; no validation outcome used to choose settings"))
        ctx.event("legacy_transfer_preflight", passed=True, checks=results)
    finally:
        env.close()


def evaluate(ctx, historical, output, method, plant):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.perf_counter()
    ctx.event("legacy_transfer_started", method=method, plant=plant, cases=48)
    for start in range(0, len(ctx.validation_cases), 16):
        ctx.guard()
        cases = ctx.validation_cases[start:start+16]
        env = ctx.make_env(cases, plant=plant, max_steps=200)
        trace = {k: [] for k in ("error", "V", "J", "action", "applied_action", "offset", "physical_error",
            "dt", "physical_cost", "input_cost", "integrated_error", "previous_action", "actual_velocity",
            "hold_observations", "success", "terminated", "truncated")}
        wall = []
        try:
            public, info = env.reset(options=dict(offsets=[c["offset"] for c in cases], goal_ids=[c["goal_id"] for c in cases]))
            policy = make_policy(ctx, historical, public, method)
            alive = torch.ones(len(cases), dtype=torch.bool, device=ctx.device)
            lengths = torch.zeros(len(cases), dtype=torch.int64, device=ctx.device)
            last = {k: torch.zeros_like(alive) for k in ("success", "collision", "outside")}
            trace["offset"].append(info["supervision"]["offset"])
            trace["physical_error"].append(info["supervision"]["physical_error"])
            for step in range(200):
                ctx.guard()
                if not bool(alive.any()):
                    break
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = policy(public)
                torch.cuda.synchronize()
                wall.append(time.perf_counter()-t)
                if step == 0:
                    matrix = out.metric.clone()
                    trace["error"].append(out.error)
                    trace["V"].append(out.value)
                elif not torch.equal(matrix, out.metric):
                    raise RuntimeError("Goal-only historical M changed during an episode")
                following, _, terminal, truncated, info = env.step(out.action)
                if not bool(info["transition_mask"][alive].all()):
                    raise RuntimeError("Autoreset contaminated transfer evaluation")
                truth = info["supervision"]
                with torch.no_grad():
                    successor_error = policy.error(following["rgb"])
                    successor_value = quadratic_value(successor_error, out.metric)
                for key, value in (("error", successor_error), ("V", successor_value), ("J", out.interaction),
                    ("action", out.action), ("applied_action", (truth["actual_velocity"]/env.scale).float()),
                    ("offset", truth["next_offset"]), ("physical_error", truth["physical_error"]), ("dt", info["dt"]),
                    ("physical_cost", truth["physical_cost"]), ("input_cost", truth["normalized_input_squared_integral"]),
                    ("integrated_error", truth["integrated_error"]), ("previous_action", truth["previous_action"]),
                    ("actual_velocity", truth["actual_velocity"]), ("hold_observations", info["hold_observations"]),
                    ("success", info["success"]), ("terminated", terminal), ("truncated", truncated)):
                    trace[key].append(value)
                lengths += alive.long()
                ending = alive & (terminal | truncated)
                for key in last:
                    last[key] |= ending & info[key]
                alive &= ~(terminal | truncated)
                public = following
            policy.check()
            host = {k: torch.stack(v).cpu().numpy() for k, v in trace.items()}
            matrices = matrix.cpu().numpy()
            last = {k: v.cpu().tolist() for k, v in last.items()}
            for i, (case, n) in enumerate(zip(cases, lengths.cpu().tolist())):
                arrays = {k: v[:n+(k in {"error", "V", "offset", "physical_error"}), i] for k, v in host.items()}
                arrays["M"] = matrices[i]
                path = output/f"case_{start+i:04d}.npz"
                _npz(path, arrays)
                row = dict(**case, plant=plant, method=method, steps=n,
                    success=last["success"][i], reason="success" if last["success"][i] else
                    "collision" if last["collision"][i] else "outside" if last["outside"][i] else "timeout",
                    elapsed_s=float(arrays["dt"].sum()), integrated_error=float(arrays["integrated_error"].sum()),
                    input_squared_integral=float(arrays["input_cost"].sum()), physical_cost=float(arrays["physical_cost"].sum()),
                    final_physical_error=float(arrays["physical_error"][-1]),
                    mean_policy_wall_s=float(np.mean(wall[:n])), trajectory=str(path), trajectory_sha256=sha256(path),
                    **diagnostics(arrays, ctx.learner.config))
                rows.append(row)
                ctx.event("legacy_transfer_episode", method=method, plant=plant, case_id=case["case_id"],
                          success=row["success"], integrated_error=row["integrated_error"], reason=row["reason"])
        except BaseException as error:
            if any(trace.values()):
                _npz(output/f"batch_{start:04d}_PARTIAL.npz", {k: torch.stack(v).cpu().numpy() for k, v in trace.items() if v})
            write_json(output/f"batch_{start:04d}_FAILURE.json", dict(error=repr(error), eligible_result=False))
            raise
        finally:
            env.close()
    result = dict(rows=rows, summary=_summary(rows), wall_s=time.perf_counter()-started)
    write_json(output/"RESULTS.json", result)
    ctx.event("legacy_transfer_complete", method=method, plant=plant, **result["summary"], wall_s=result["wall_s"])
    return result


def audit(output, ctx):
    maxima = dict(action=0., value=0., physical_error=0., integrated_error=0.)
    count = transitions = 0
    for path in output.glob("*/*/RESULTS.json"):
        for row in read_json(path)["rows"]:
            assert sha256(row["trajectory"]) == row["trajectory_sha256"]
            with np.load(row["trajectory"], allow_pickle=False) as archive:
                a = {k: archive[k] for k in archive.files}
            e, J, M = (a[k].astype(float) for k in ("error", "J", "M"))
            G = J.transpose(0, 2, 1) @ M @ J
            eta = np.maximum(1.01*np.trace(G, axis1=1, axis2=2)/(1e4-1), 1e-8)
            H = G.copy()
            index = np.arange(4)
            H[:, index, index] += .01*G[:, index, index] + eta[:, None]
            rhs = (J.transpose(0, 2, 1) @ M @ e[:-1, :, None])[..., 0]
            raw = -.7*np.linalg.solve(H, rhs[..., None])[..., 0]
            radius = np.maximum(1, np.maximum(np.linalg.norm(raw[:, :3], axis=1), np.abs(raw[:, 3])))
            u = raw/radius[:, None]
            maxima["action"] = max(maxima["action"], float(np.max(np.abs(u-a["action"]))))
            assert np.allclose(u, a["action"], atol=5e-4, rtol=1e-4)
            V = .5*np.einsum("ti,ij,tj->t", e, M, e)
            maxima["value"] = max(maxima["value"], float(np.max(np.abs(V-a["V"]))))
            assert np.allclose(V, a["V"], atol=1e-5, rtol=1e-5)
            q = a["offset"].astype(float)
            E = np.sum(q[:, :3]**2, axis=1)+(q[:, 3]/math.radians(25))**2
            maxima["physical_error"] = max(maxima["physical_error"], float(np.max(np.abs(E-a["physical_error"]))))
            integral = float((E[1:]*a["dt"]).sum())
            maxima["integrated_error"] = max(maxima["integrated_error"], abs(integral-row["integrated_error"]))
            assert np.isclose(integral, row["integrated_error"], atol=1e-5, rtol=1e-6)
            count_hold, counts = 0, []
            criteria = ctx.cfg["collection"]
            good = (np.linalg.norm(q[1:, :3], axis=1) < criteria["position_tolerance_m"]) & (np.abs(q[1:, 3]) < math.radians(criteria["yaw_tolerance_deg"]))
            for hit in good:
                count_hold = count_hold+1 if hit else 0
                counts.append(count_hold)
            assert np.array_equal(counts, a["hold_observations"])
            if row["success"]:
                assert counts[-1] >= criteria["required_hold_observations"] and a["success"][-1]
            assert not a["terminated"][:-1].any() and not a["truncated"][:-1].any()
            count += 1
            transitions += row["steps"]
    assert count == 480
    result = dict(passed=True, episodes=count, transitions=transitions, maximum_absolute_differences=maxima,
                  complete_successors=True, actual_hold_recomputed=True, trajectory_hashes_match=True,
                  note="Independent NumPy audit; original historical ridge used for every method")
    write_json(output/"INDEPENDENT_AUDIT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root/"legacy_transfer"
    project = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), Path(__file__).with_name("control_ablation.py"),
                    project/"stabilization-gym/stabilization_gym/envs/tasks/original_visual.py",
                    project/"stabilization-gym/stabilization_gym/envs/tasks/original_visual_online.py",
                    project/"third_party/visual_original/runtime/goal_fixed_core.py",
                    project/"third_party/visual_original/lac_vs/vision.py"]
    assets = {str(project/"assets/original_visual_initialization"/name):
              sha256(project/"assets/original_visual_initialization"/name) for name in ("initial.pt", "best.pt", "protocol.json")}
    freeze_stage(root, "legacy_transfer", dict(methods=METHODS, plants=["P0", "P1"], cases=48, steps=200,
        cases_sha256=sha256(root/"VALIDATION_CASES.json"), historical_assets=assets,
        ridge="max(1.01*trace(G)/(1e4-1),1e-8)", diagonal_damping=.01, gain=.7,
        historical_coordinates="checkpoint CAE center/scale preserved; e and J divide by4; metric sees raw goal encode",
        historical_preprocessing="exact original CPU image_tensor/Reference; online-subclass path binding only",
        v4_scope="only exact legacy ridge replaced; current and target J both declared",
        interpretation="old model/controller combination under new GPU rendering and current validation starts; not historical-renderer reproduction",
        evaluation="all methods frozen before execution; no result-driven selection or threshold change"), source_files)
    with gpu_lease(root, "legacy_transfer"):
        try:
            ctx = bootstrap(root)
            historical = load_historical(ctx)
            output.mkdir(parents=True, exist_ok=False)
            preflight(ctx, historical, output)
            results = {}
            for plant in ("P0", "P1"):
                for method in METHODS:
                    results[f"{plant}/{method}"] = evaluate(ctx, historical, output/plant/method, method, plant)
            baseline = {plant: read_json(root/"control"/plant/"baseline/RESULTS.json")["rows"] for plant in ("P0", "P1")}
            pairs = {key: paired_comparison(baseline[key.split('/')[0]], result["rows"]) for key, result in results.items()}
            audit_result = audit(output, ctx)
            for path, digest in assets.items():
                assert sha256(path) == digest
            summary = dict(summaries={key: value["summary"] for key, value in results.items()},
                           paired_vs_v4_baseline=pairs, audit=audit_result,
                           caveat="development validation and changed renderer; no untouched-test or from-scratch claim")
            write_json(output/"RESULTS.json", summary)
            ctx.event("legacy_transfer_finished", summaries=summary["summaries"], audit=audit_result)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        except BaseException as error:
            event(root, "legacy_transfer_failure", error=repr(error))
            raise


if __name__ == "__main__":
    main()
