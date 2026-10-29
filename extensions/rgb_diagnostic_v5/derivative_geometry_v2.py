"""Frozen-model RGB derivative and physical-geometry diagnostics.

No model parameter updates and no held-out test starts.  Simulator state/depth
are privileged diagnostic instruments, never inputs to the RGB controller.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
for folder in (PROJECT, PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from stabilization_gym.tasks_gpu_rgb.environment import rotation_z


@torch.no_grad()
def render_command_perturbation(env, axis: int, step: float):
    """Render +/- one normalized-command flow for ``step`` seconds.

    All axes have units of feature units / second / normalized command after
    dividing a central difference by 2*step.  These are exact single-axis P0
    flows.  The environment's position/yaw/velocity/queues are not advanced.
    Only the renderer's reusable camera buffers are overwritten.
    """
    if not 0 <= axis < 4 or step <= 0:
        raise ValueError("A positive time and a declared four-DOF axis are required")
    position, yaw = env.position, env.yaw
    if axis < 3:
        body = rotation_z(yaw) @ env.body0
        displacement = body[:, :, axis] * env.scale[axis] * step
        plus, minus = env._render(position + displacement, yaw), env._render(position - displacement, yaw)
    else:
        angle = env.scale[3] * step
        plus, minus = env._render(position, yaw + angle), env._render(position, yaw - angle)
    return plus, minus


@torch.no_grad()
def fd_interaction(env, features, geometry, step: float):
    """Privileged finite-difference J for diagnostics, not deployable sensing."""
    columns = []
    for axis in range(4):
        plus, minus = render_command_perturbation(env, axis, step)
        encoded = features.encode(geometry.encode_images(torch.cat((plus, minus))))
        p, m = encoded.chunk(2)
        columns.append((p - m) / (2 * step))
    return torch.stack(columns, -1)


class DiagnosticDepth:
    """Separate RGB-plus-depth MJWarp contexts using the RGB renderer's geometry.

    MJWarp raw depth_data is metric optical-axis depth, not Euclidean ray
    distance. get_depth() is deliberately not used because it scales/clips.
    """

    def __init__(self, env):
        self.items = []
        for indices, renderer in env._groups:
            wp, mjw = renderer.wp, renderer.mjw
            stream = wp.stream_from_torch(torch.cuda.current_stream(env.device))
            with wp.ScopedDevice(str(env.device)), wp.ScopedStream(stream):
                context = mjw.create_render_context(
                    renderer.cpu_model, nworld=len(indices),
                    cam_res=(env.common.SIZE, env.common.SIZE),
                    cam_active=[renderer.camera_id], render_rgb=True,
                    render_depth=True, render_seg=False, use_textures=True,
                    use_shadows=True, render_skybox=True,
                    samples_per_pixel=2, use_fast_math=False, deterministic=True,
                )
                mjw.refit_bvh(renderer.model, renderer.data, context)
            self.items.append((indices, renderer, context))

    @torch.no_grad()
    def render(self, env):
        matrix = rotation_z(env.yaw) @ env.goal_matrix @ env.optical_to_gl
        size = env.common.SIZE
        result = torch.empty((env.num_envs, 1, size, size), device=env.device)
        for indices, renderer, context in self.items:
            wp, mjw = renderer.wp, renderer.mjw
            stream = wp.stream_from_torch(torch.cuda.current_stream(env.device))
            with wp.ScopedStream(stream):
                renderer.camera_position[:, renderer.camera_id].copy_(env.position[indices].float())
                renderer.camera_matrix[:, renderer.camera_id].copy_(matrix[indices].float())
                mjw.render(renderer.model, renderer.data, context)
                raw = wp.to_torch(context.depth_data)
                address = int(context.depth_adr.numpy()[0])
                result[indices, 0] = raw[:, address:address + size * size].reshape(-1, size, size)
        return result

    def ray_checks(self, env, depth):
        """Check planar-depth convention against independent CPU MuJoCo rays."""
        import mujoco
        size = env.common.SIZE
        matrix = (rotation_z(env.yaw) @ env.goal_matrix @ env.optical_to_gl).cpu().numpy()
        positions, reference = env.position.cpu().numpy(), depth[:, 0].cpu().numpy()
        pixels = [(size // 2, size // 2), (size // 4, size // 4),
                  (size // 4, 3 * size // 4), (3 * size // 4, size // 4),
                  (3 * size // 4, 3 * size // 4)]
        rows = []
        for indices, renderer, context in self.items:
            data = mujoco.MjData(renderer.cpu_model)
            mujoco.mj_forward(renderer.cpu_model, data)
            rays = renderer.wp.to_torch(context.ray).cpu().numpy()
            for lane in indices.cpu().tolist():
                for y, x in pixels:
                    local = rays[y * size + x].astype(np.float64)
                    direction = matrix[lane] @ local
                    geomid = np.full(1, -1, dtype=np.int32)
                    distance = mujoco.mj_ray(renderer.cpu_model, data, positions[lane],
                                             direction, None, 1, -1, geomid)
                    metric = float(distance * -local[2]) if distance >= 0 else 0.
                    actual = float(reference[lane, y, x])
                    rows.append(dict(lane=lane, y=y, x=x, cpu_planar_depth_m=metric,
                                     gpu_planar_depth_m=actual,
                                     absolute_difference_m=abs(metric - actual),
                                     both_hit=bool(metric > 0 and actual > 0)))
        return rows


def vector_metrics(pred, actual):
    """Flatten non-batch dimensions; no cosine for vanishing vectors."""
    a, b = np.asarray(pred, float), np.asarray(actual, float)
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid = (an > 1e-8) & (bn > 1e-8)
    cosine = np.full(len(a), np.nan)
    cosine[valid] = (a[valid] * b[valid]).sum(1) / (an[valid] * bn[valid])
    return dict(pred_norm=an, actual_norm=bn, cosine=cosine,
                relative_error=np.linalg.norm(a - b, axis=1) / np.maximum(bn, 1e-8),
                norm_ratio=an / np.maximum(bn, 1e-8), valid=valid)


def describe(value):
    x = np.asarray(value, float)
    x = x[np.isfinite(x)]
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                q10_q90=np.quantile(x, [.1, .9]).tolist() if len(x) else None)


def as_array(tensor):
    return tensor.detach().cpu().numpy()


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(ctx, output: Path, *, steps=(.0025, .01, .04), image_amplitudes=(1e-4, 1e-3, 1e-2)):
    from extensions.rgb_diagnostic_v5.common import write_json
    output.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    features, metric, geometry = ctx.learner.features, ctx.learner.metric, ctx.geometry
    features.eval(); metric.eval()
    for module in (features, metric):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    by_scene = defaultdict(list)
    for case in ctx.train_cases:
        by_scene[case['scene_id']].append(case)
    arrays = defaultdict(list)
    rows, ray_rows, depth_rows, unit_rows, delay_rows = [], [], [], [], []
    case_order = []
    for scene_index, (scene_id, cases) in enumerate(by_scene.items()):
        ctx.guard()
        env = ctx.make_env(cases, plant='P0')
        try:
            options = dict(offsets=np.asarray([c['offset'] for c in cases]),
                           goal_ids=[c['goal_id'] for c in cases])
            public, _ = env.reset(seed=91081, options=options)
            image, goal, fast_tangent = geometry.observe(public)
            depth_renderer = DiagnosticDepth(env)
            truth_depth = depth_renderer.render(env)
            valid_depth = torch.isfinite(truth_depth) & (truth_depth > 1e-6)
            safe_truth = torch.where(valid_depth, truth_depth, 1e8)
            _, true_tangent = geometry.tangents(public['rgb'], safe_truth, public['rotation'])
            clipped = safe_truth.clamp(ctx.cfg['goal_fixed']['min_depth'], ctx.cfg['goal_fixed']['max_depth'])
            _, clipped_tangent = geometry.tangents(public['rgb'], clipped, public['rotation'])
            with torch.no_grad():
                z, zg = features.encode(image), features.encode(goal)
                matrices = metric(zg)
                j_fast = features.interaction_jvp(image, fast_tangent, create_graph=False).transpose(1, 2)
                j_true = features.interaction_jvp(image, true_tangent, create_graph=False).transpose(1, 2)
                j_clipped = features.interaction_jvp(image, clipped_tangent, create_graph=False).transpose(1, 2)
            estimated_depth = geometry.predict_depth(public['rgb'])
            for lane, case in enumerate(cases):
                valid = valid_depth[lane]
                ratio = as_array(estimated_depth[lane][valid] / truth_depth[lane][valid])
                depth_rows.append(dict(case_id=case['case_id'], valid_fraction=float(valid.float().mean()),
                    true_depth_m=describe(as_array(truth_depth[lane][valid])),
                    fast_to_true_depth_ratio=describe(ratio),
                    fast_vs_true_log_rmse=float(np.sqrt(np.mean(np.log(np.maximum(ratio, 1e-8)) ** 2)))))
            checks = depth_renderer.ray_checks(env, truth_depth)
            for check in checks:
                check['case_id'] = cases[check.pop('lane')]['case_id']
            ray_rows.extend(checks)

            # The mathematical directional derivative does not require clipping
            # perturbed image values into the sensor range.
            jvp_fd = []
            tangent_max = fast_tangent.abs().flatten(2).amax(-1).clamp_min(1e-7)
            for amplitude in image_amplitudes:
                columns = []
                for axis in range(4):
                    direction = fast_tangent[:, axis] / tangent_max[:, axis, None, None, None]
                    with torch.no_grad():
                        both = features.encode(torch.cat((image + amplitude * direction, image - amplitude * direction)))
                    plus, minus = both.chunk(2)
                    derivative = (plus - minus) / (2 * amplitude)
                    columns.append(derivative * tangent_max[:, axis, None])
                jvp_fd.append(torch.stack(columns, -1))
            arrays['jvp_numerical'].append(as_array(torch.stack(jvp_fd, 1)))
            arrays['j_fast'].append(as_array(j_fast)); arrays['j_true_depth'].append(as_array(j_true))
            arrays['j_true_clipped'].append(as_array(j_clipped))
            arrays['error'].append(as_array(z - zg)); arrays['M'].append(as_array(matrices))
            arrays['offset'].append(as_array(env._offset()))

            j_real, j_tangent_fd, b_metrics, pixel_changed = [], [], [], []
            for step in steps:
                columns, image_directions, changed = [], [], []
                for axis in range(4):
                    plus_rgb, minus_rgb = render_command_perturbation(env, axis, step)
                    plus_img, minus_img = geometry.encode_images(plus_rgb), geometry.encode_images(minus_rgb)
                    with torch.no_grad():
                        ep, em = features.encode(torch.cat((plus_img, minus_img))).chunk(2)
                    columns.append((ep - em) / (2 * step))
                    image_directions.append((plus_img - minus_img) / (2 * step))
                    changed.append((plus_rgb != minus_rgb).float().flatten(1).mean(1))
                fd_j = torch.stack(columns, -1)
                fd_b = torch.stack(image_directions, 1)
                linear_fd = features.interaction_jvp(image, fd_b, create_graph=False).transpose(1, 2)
                j_real.append(fd_j); j_tangent_fd.append(linear_fd)
                pixel_changed.append(torch.stack(changed, -1))
                for source, pred_b in [('fast', fast_tangent), ('true', true_tangent), ('true_clipped', clipped_tangent)]:
                    for axis in range(4):
                        stats = vector_metrics(as_array(pred_b[:, axis]), as_array(fd_b[:, axis]))
                        for lane, case in enumerate(cases):
                            rows.append(dict(kind='image_motion', source=source, step=step, axis=axis,
                                case_id=case['case_id'], difficulty=case['difficulty'],
                                **{k: (bool(v[lane]) if k == 'valid' else float(v[lane])) for k, v in stats.items()}))
            arrays['j_render_fd'].append(as_array(torch.stack(j_real, 1)))
            arrays['j_render_tangent_jvp'].append(as_array(torch.stack(j_tangent_fd, 1)))
            arrays['pixel_changed_fraction'].append(as_array(torch.stack(pixel_changed, 1)))

            # Independently verify the single-axis spatial perturbation against
            # PhysicalEnv.step.  step <= dt keeps the requested action bounded.
            for axis in range(4):
                env.plant = 'P0'
                env.reset(seed=91081, options=options)
                plus, _ = render_command_perturbation(env, axis, steps[1])
                action = torch.zeros((len(cases), 4), device=env.device)
                action[:, axis] = steps[1] / env.dt
                following, _, _, _, _ = env.step(action)
                for lane, case in enumerate(cases):
                    unit_rows.append(dict(case_id=case['case_id'], axis=axis,
                        rgb_max_uint8_difference=int((plus[lane].int() - following['rgb'][lane].int()).abs().max())))
            # P1 command-response semantics: first command has no motion; next
            # interval executes the delayed, filtered command.  Never compare a
            # current-command J to the first delayed successor as a geometry test.
            env.plant = 'P1'
            public0, _ = env.reset(seed=91081, options=options)
            command = torch.zeros((len(cases), 4), device=env.device)
            command[:, 0] = .1
            first, _, _, _, info1 = env.step(command)
            second, _, _, _, info2 = env.step(torch.zeros_like(command))
            expected = env.mix * env.scale * command.double()
            delay_rows.append(dict(scene_id=scene_id,
                first_actual_velocity_max_abs=float(info1['supervision']['actual_velocity'].abs().max()),
                first_rgb_max_uint8_difference=int((first['rgb'].int() - public0['rgb'].int()).abs().max()),
                second_velocity_expected_max_abs_difference=float((info2['supervision']['actual_velocity'] - expected).abs().max())))
            case_order.extend(cases)
            ctx.event('derivative_geometry_scene_complete', scene_id=scene_id, completed_states=len(case_order),
                      total_states=len(ctx.train_cases), seconds=time.perf_counter() - begun)
            print(json.dumps(dict(scene_id=scene_id, completed_states=len(case_order), seconds=time.perf_counter() - begun)), flush=True)
        finally:
            env.close()
    merged = {k: np.concatenate(v) for k, v in arrays.items()}
    for i, case in enumerate(case_order):
        for axis in range(4):
            for ai, amplitude in enumerate(image_amplitudes):
                stats = vector_metrics(merged['j_fast'][i:i+1, :, axis], merged['jvp_numerical'][i:i+1, ai, :, axis])
                rows.append(dict(kind='jvp_arithmetic', source='fast', amplitude=amplitude, axis=axis,
                    case_id=case['case_id'], difficulty=case['difficulty'],
                    **{k: (bool(v[0]) if k == 'valid' else float(v[0])) for k, v in stats.items()}))
            for si, step in enumerate(steps):
                for source, key in [('fast', 'j_fast'), ('true', 'j_true_depth'), ('true_clipped', 'j_true_clipped')]:
                    stats = vector_metrics(merged[key][i:i+1, :, axis], merged['j_render_fd'][i:i+1, si, :, axis])
                    rows.append(dict(kind='feature_motion', source=source, step=step, axis=axis,
                        case_id=case['case_id'], difficulty=case['difficulty'],
                        **{k: (bool(v[0]) if k == 'valid' else float(v[0])) for k, v in stats.items()}))
                stats = vector_metrics(merged['j_render_tangent_jvp'][i:i+1, si, :, axis],
                                       merged['j_render_fd'][i:i+1, si, :, axis])
                rows.append(dict(kind='encoder_finite_motion', source='render_tangent_jvp', step=step, axis=axis,
                    case_id=case['case_id'], difficulty=case['difficulty'],
                    **{k: (bool(v[0]) if k == 'valid' else float(v[0])) for k, v in stats.items()}))
    summary = summarize(rows)
    sensitivity = []
    physical_scale = np.array([1., 1., 1., math.radians(25)])
    command_scale = np.asarray([.4, .4, .4, .35])
    for si, step in enumerate(steps):
        jac = merged['j_render_fd'][:, si] / command_scale * physical_scale
        singular = np.linalg.svd(jac, compute_uv=False)
        gram = .5 * np.einsum('ndi,nde,nej->nij', jac, merged['M'], jac)
        eigen = np.linalg.eigvalsh(gram)
        sensitivity.append(dict(step=step, minimum_singular=describe(singular[:, -1]),
            maximum_singular=describe(singular[:, 0]),
            condition=describe(singular[:, 0] / np.maximum(singular[:, -1], 1e-12)),
            metric_physical_min_eigenvalue=describe(eigen[:, 0]),
            metric_physical_max_eigenvalue=describe(eigen[:, -1]),
            local_linear_lower_bound_fail_fraction=float((eigen[:, 0] < .1).mean()),
            meaning='Finite-difference local feature metric in physically normalized 4DOF coordinates; not a domain proof'))
    stability = []
    for si in range(len(steps) - 1):
        stats = vector_metrics(merged['j_render_fd'][:, si].transpose(0, 2, 1).reshape(-1, 16),
                               merged['j_render_fd'][:, si + 1].transpose(0, 2, 1).reshape(-1, 16))
        stability.append(dict(steps=[steps[si], steps[si + 1]], **{k: describe(v) for k, v in stats.items()}))
    # Predeclared diagnostic reference: middle step is used unless stability
    # shows the finite differences are too inconsistent to use as a reference.
    middle_stable = stability[1]['cosine']['median'] is not None and stability[1]['cosine']['median'] >= .9
    result = dict(schema='rgb_v5_derivative_geometry', seconds=time.perf_counter() - begun,
        cases=len(case_order), split='train', parameter_updates=0, steps=list(steps), image_amplitudes=list(image_amplitudes),
        scope='Frozen v4 encoder and metric; privileged simulator perturbations/depth only for diagnostics',
        depth_semantics='raw metric optical-axis depth, independently checked with CPU MuJoCo rays; no-hit is inverse-depth zero',
        pixel_semantics='Original uint8 RGB path retained; tiny-step differences are quantization affected',
        j_units='normalized feature per second per normalized command',
        summary=summary, sensitivity=sensitivity, finite_difference_stability=stability,
        fd_control_reference=dict(step=steps[1], reason='predeclared middle physical step, not selected by validation control success',
                                  middle_vs_largest_median_cosine_at_least_0p9=middle_stable),
        unit_checks=dict(n=len(unit_rows), max_rgb_uint8_difference=max(r['rgb_max_uint8_difference'] for r in unit_rows)),
        depth_ray_check=dict(n=len(ray_rows), both_hit=sum(r['both_hit'] for r in ray_rows),
                             absolute_difference_m=describe([r['absolute_difference_m'] for r in ray_rows if r['both_hit']])),
        p1_delay_checks=delay_rows,
        caveats=['States are 96 designed training diagnostics, not iid confidence intervals',
                 'True depth and finite-difference J are not RGB-only deployment methods',
                 'No geometry model or trainable parameter was changed',
                 'The optical brightness model omits occlusion and lighting changes'])
    write_json(output / 'SUMMARY.json', result)
    write_json(output / 'ROWS.json', json_ready(rows))
    write_json(output / 'CASES.json', case_order)
    write_json(output / 'DEPTH_CHECKS.json', json_ready(dict(depth=depth_rows, rays=ray_rows, units=unit_rows)))
    with (output / 'ARRAYS.npz').open('xb') as handle:
        np.savez_compressed(handle, **merged)
    ctx.event('derivative_geometry_complete', output=str(output), cases=len(case_order), seconds=result['seconds'])
    return result


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        size = row.get('step', row.get('amplitude'))
        for stratum in ('all', 'axis_' + str(row['axis']), row['difficulty']):
            grouped[(row['kind'], row['source'], size, stratum)].append(row)
    result = []
    for (kind, source, size, stratum), group in grouped.items():
        valid = [r for r in group if r['valid']]
        result.append(dict(kind=kind, source=source, size=size, stratum=stratum, n=len(group), valid=len(valid),
            cosine=describe([r['cosine'] for r in valid]),
            opposite_fraction=float(np.mean([r['cosine'] < 0 for r in valid])) if valid else None,
            relative_error=describe([r['relative_error'] for r in group]),
            norm_ratio=describe([r['norm_ratio'] for r in group])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from extensions.rgb_diagnostic_v5.common import bootstrap, gpu_lease, freeze_stage, write_json
    root = args.root.resolve()
    settings = dict(states=96, axes=4, steps=[.0025, .01, .04], image_amplitudes=[1e-4, 1e-3, 1e-2],
                    fd_control_reference_predeclared=.01, privileged_depth='raw_metric_optical_axis',
                    reference_jvp_dtype='float32', updates=0, test_split_used=False)
    freeze_stage(root, 'derivative_geometry_v2', settings, [__file__])
    with gpu_lease(root, 'derivative_geometry_v2'):
        ctx = bootstrap(root)
        try:
            result = run(ctx, args.output or root / 'derivative_geometry_v2')
        except BaseException as error:
            write_json(root / 'derivative_geometry_v2' / 'FAILED.json', dict(error=repr(error), automatic_resume=False))
            raise
    print(json.dumps({k: result[k] for k in ('cases', 'seconds', 'unit_checks', 'depth_ray_check', 'fd_control_reference')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
