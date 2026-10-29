"""Book episode evaluation through the reconstructed physical environment.

The frozen original episode owns the physical stepping and rendering. Models
remain external arguments. This module adds immutable telemetry and independent
audits without changing the episode's controller or success criterion.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch

from extensions.unified_training_v1.backends.rgb import _write, _sha, _module_hash
from stabilization_learning.unified_v1.runner import check_frozen


def verify_frozen(backend):
    check_frozen(backend.output.parent.parent, [backend.output.name])
    marker = json.loads((backend.output / "FROZEN.json").read_text(encoding="utf-8"))
    saved = torch.load(backend.output / marker["checkpoint"], map_location="cpu", weights_only=False)
    for name in ("ae", "metric"):
        for key, tensor in getattr(backend.learner, name).state_dict().items():
            if not torch.equal(tensor.detach().cpu(), saved["learner"][name][key].cpu()):
                raise RuntimeError("Evaluation model differs from frozen selection")


def paired(rows, target, baseline, plant):
    def index(variant):
        selected = [r for r in rows if r["variant"] == variant and r["plant"] == plant]
        result = {r["case_id"]: r for r in selected}
        if len(result) != len(selected):
            raise ValueError("Duplicate evaluation case_id in a variant/plant group")
        return result
    a, b = index(target), index(baseline)
    if not a or a.keys() != b.keys():
        return None
    keys = list(a)
    common = [k for k in keys if a[k]["success"] and b[k]["success"]]
    return dict(target=target, baseline=baseline, plant=plant, episodes=len(keys),
        gained=sum(a[k]["success"] and not b[k]["success"] for k in keys),
        lost=sum(b[k]["success"] and not a[k]["success"] for k in keys),
        common_successes=len(common),
        mean_arrival_time_difference_seconds=float(np.mean([a[k]["elapsed_s"]-b[k]["elapsed_s"] for k in common])) if common else None)


def summarize(rows):
    groups = {}
    for plant, variant in sorted({(r["plant"], r["variant"]) for r in rows}):
        rs = [r for r in rows if r["plant"] == plant and r["variant"] == variant]
        group = dict(episodes=len(rs), successes=sum(r["success"] for r in rs),
            success_rate=float(np.mean([r["success"] for r in rs])),
            mean_integrated_error=float(np.mean([r["integrated_error"] for r in rs])),
            mean_input_squared=float(np.mean([r["input_squared_integral"] for r in rs])),
            failures={k: sum(r["reason"] == k for r in rs) for k in ("timeout", "collision", "outside")})
        for field in ("difficulty", "scene_id"):
            group[field] = {name: dict(episodes=len(sub), successes=sum(r["success"] for r in sub))
                for name in sorted({r[field] for r in rs}) if (sub := [r for r in rs if r[field] == name])}
        rates = [v["successes"]/v["episodes"] for v in group["scene_id"].values()]
        group["macro_scene_success"] = float(np.mean(rates))
        group["worst_scene_success"] = float(min(rates))
        groups[f"{plant}/{variant}"] = group
    selected = [r for r in rows if r["variant"] == "selected"]
    return dict(episodes=len(rows), groups=groups,
        success_rate=float(np.mean([r["success"] for r in selected])) if selected else None,
        mean_integrated_error=float(np.mean([r["integrated_error"] for r in selected])) if selected else None)



def native_trace(arrays, dt):
    """Retain book telemetry and add the common report's coordinate aliases."""
    result = {key: value for key, value in arrays.items()
              if key not in ('rgb', 'goal_rgb', 'smooth_center', 'smooth_minus', 'smooth_plus')}
    n = len(arrays['action'])
    if n < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('A trajectory requires actions and a positive finite physical dt')
    if any(len(arrays[key]) != n+1 for key in ('features', 'offset', 'V')):
        raise ValueError('Book trajectory lacks complete successor endpoints')
    if any(len(arrays[key]) != n for key in ('terminal', 'truncated', 'cost')):
        raise ValueError('Book trajectory transition metadata length mismatch')
    offset = np.asarray(arrays['offset'], dtype=np.float64)
    physical = np.sum(offset[:, :3]**2, axis=1)+(offset[:, 3]/math.radians(25))**2
    good = (np.linalg.norm(offset[1:, :3], axis=1)<.02) & (np.abs(offset[1:, 3])<math.radians(1))
    count, holds = 0, []
    for inside in good:
        count = count+1 if inside else 0
        holds.append(count)
    result.update(error=arrays['features'], J=arrays['J_reference'],
        terminated=arrays['terminal'], truncated=arrays['truncated'],
        dt=np.full(n, dt, np.float64), integrated_error=physical[1:]*dt,
        input_squared=np.sum(np.asarray(arrays['action'], np.float64)**2, axis=1)*dt,
        hold=np.asarray(holds, np.int64))
    return result


def _variant_models(backend, variant):
    learner = backend.learner
    if variant in ('selected', 'learned'):
        return learner.ae, learner.metric, False
    if variant in ('identity', 'identity_m'):
        return learner.ae, learner.metric, True
    if variant == 'shadow':
        return learner.shadow, learner.metric, True
    if variant in ('initial_identity', 'initial_ae_identity'):
        return learner.initial, learner.metric, True
    if variant == 'initial':
        if not hasattr(backend, 'initial_models'):
            ae, metric = deepcopy(learner.ae), deepcopy(learner.metric)
            ae.load_state_dict(backend.initial_state['ae'], strict=True)
            metric.load_state_dict(backend.initial_state['metric'], strict=True)
            ae.eval().requires_grad_(False); metric.eval().requires_grad_(False)
            backend.initial_models = ae, metric
        return *backend.initial_models, False
    identity = variant.endswith('_identity')
    name = variant[:-9] if identity else variant
    if name in backend.reference_models:
        return *backend.reference_models[name], identity
    raise ValueError('Unknown frozen RGB model variant: '+str(variant))


class _GuardedInstance:
    """Bound STOP/deadline checks to the original episode's render boundary."""
    def __init__(self, instance, guard):
        self.instance, self.guard = instance, guard

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def render_rgb(self, *args, **kwargs):
        self.guard()
        return self.instance.render_rgb(*args, **kwargs)


def evaluate_cases(self, cases, tag, variants, plants, steps, depth_scale=1.):
    self.guard()
    if (not variants or len(set(variants)) != len(variants) or
            not plants or len(set(plants)) != len(plants) or
            not cases or type(steps) is not int or steps <= 0 or
            not math.isfinite(depth_scale) or depth_scale <= 0):
        raise ValueError('Distinct variants/plants, nonempty cases and finite evaluation required')
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate evaluation case_id')
    if set(plants)-{'P0', 'P1'}:
        raise ValueError('Unsupported physical evaluation plant')
    models = {name: _variant_models(self, name) for name in variants}
    hashes = {name: dict(ae=_module_hash(ae), metric=_module_hash(metric), identity=identity)
              for name, (ae, metric, identity) in models.items()}
    folder = self.output / f'RGB_EVAL_{self.evaluation_id:03d}_{tag}'
    folder.mkdir(exist_ok=False); self.evaluation_id += 1
    dt = float(self.cfg['plant']['dt'])
    _write(folder / 'FROZEN.json', dict(split=tag, variants=list(variants), plants=list(plants),
        cases=cases, max_steps=steps, dt=dt, depth_scale=depth_scale, model_hashes=hashes,
        ae_sha256=_module_hash(self.learner.ae), metric_sha256=_module_hash(self.learner.metric),
        counts=self.learner.counts.copy(), created_unix=time.time(),
        episode_implementation='frozen train_goal_fixed.episode',
        rendering='audited reconstructed native physical renderer',
        success='position < .02 m and abs(yaw) < 1 degree for 11 consecutive successor observations',
        elapsed_s='physical time: number of transitions times dt',
        wall_s='original episode measured wall time', rgb_frames_saved=False))
    joint = self.adapter.joint
    rng = np.random.default_rng(self.seed+77001)
    rows, begin, instance, scene_id = [], time.perf_counter(), None, None
    try:
        # Keep book ordering: method, then plant, then fixed start list.
        for variant in variants:
            ae, metric, identity = models[variant]
            view = SimpleNamespace(ae=ae, metric=metric, version=self.learner.version,
                                   device=self.learner.device)
            mode = 'identity_m' if identity else 'learned'
            for plant in plants:
                for index, case in enumerate(cases):
                    self.guard()
                    scene = self.lookup[case['scene_id']]
                    if scene_id != scene['id']:
                        if instance is not None:
                            instance.close()
                        instance = None
                        instance = _GuardedInstance(joint.new_instance(self.cfg, scene), self.guard)
                        scene_id = scene['id']
                    arrays, original = joint.episode(self.cfg, self.output, scene, case['goal_id'],
                        case['offset'], view, self.depth, rng, steps=steps, mode=mode,
                        plant_name=plant, collect=False, inst=instance, depth_scale=depth_scale)
                    telemetry = native_trace(arrays, dt)
                    audit = self._audit(telemetry, original['status'])
                    length = len(telemetry['action'])
                    if original['steps'] != length:
                        raise ValueError('Book evaluator step count differs from saved trajectory')
                    expected_error = float(telemetry['integrated_error'].sum())
                    if not math.isclose(original['integrated_error'], expected_error, rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError('Book evaluator integrated error differs from physical trace')
                    trajectory = folder / f'{plant}_{variant}_{index:04d}.npz'
                    with trajectory.open('xb') as stream:
                        np.savez_compressed(stream, **telemetry)
                    offset = telemetry['offset']
                    physical = np.sum(offset[:, :3]**2, axis=1)+(offset[:, 3]/math.radians(25))**2
                    result_row = dict(original)
                    result_row.update(case)
                    result_row.update(plant=plant, variant=variant, method=variant,
                        book_episode_mode=mode, reason=original['status'],
                        success=bool(original['success']), steps=length,
                        elapsed_s=float(length*dt), wall_s=float(original['elapsed_s']),
                        integrated_error=expected_error,
                        input_squared_integral=float(telemetry['input_squared'].sum()),
                        final_error=float(physical[-1]),
                        V_rise_fraction=float(original['actual_V_increase_fraction']),
                        V_lower_violation=float(np.mean(telemetry['V'] < self.cfg['goal_fixed']['value_lower']*physical-1e-7)),
                        V_over_E_median=float(original['V_over_E_quantiles'][1]),
                        policy_wall_s=float(original['latency_ms']/1000),
                        trajectory=str(trajectory), trajectory_sha256=_sha(trajectory), audit=audit)
                    _write(trajectory.with_suffix('.json'), result_row)
                    rows.append(result_row)
                    self.event('rgb_evaluation_episode', split=tag, plant=plant, variant=variant,
                        case_id=case['case_id'], success=result_row['success'], reason=result_row['reason'])
        after = {name: dict(ae=_module_hash(ae), metric=_module_hash(metric), identity=identity)
                 for name, (ae, metric, identity) in models.items()}
        if after != hashes:
            raise RuntimeError('Evaluation mutated a frozen encoder or metric')
    except BaseException as error:
        _write(folder / 'PARTIAL.json', dict(complete=False, error=repr(error), completed_episodes=len(rows), rows=rows))
        raise
    finally:
        if instance is not None:
            instance.close()
    summary = summarize(rows)
    summary.update(unique_cases=len(cases), wall_s=time.perf_counter()-begin,
        independent_successor_audit=True, max_steps=steps, depth_scale=depth_scale,
        episode_implementation='frozen train_goal_fixed.episode')
    comparisons = []
    for target in ('selected', 'cross'):
        if target not in variants:
            continue
        for baseline in variants:
            if baseline == target:
                continue
            for plant in plants:
                result = paired(rows, target, baseline, plant)
                if result is not None:
                    comparisons.append(result)
    result = dict(split=tag, complete=True, summary=summary, rows=rows, paired=comparisons, folder=str(folder))
    _write(folder / 'RESULTS.json', result)
    self.event('evaluation_suite_completed', suite=tag, summary=summary, paired=comparisons)
    return result
