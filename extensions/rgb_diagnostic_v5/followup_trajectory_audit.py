"""Independent CPU audit of the completed repair and historical-transfer groups.

This reads archived transitions only after a fresh external process snapshot.
It does not load models, render images, choose settings, or rerun evaluations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from .common import read_json, sha256, write_json


def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                q10=float(np.quantile(x, .1)) if len(x) else None,
                q90=float(np.quantile(x, .9)) if len(x) else None)


def compare(base_rows, changed_rows):
    base = {r['case_id']: r for r in base_rows}
    changed = {r['case_id']: r for r in changed_rows}
    assert set(base) == set(changed)
    keys = sorted(base)
    common = [k for k in keys if base[k]['success'] and changed[k]['success']]
    errors = [changed[k]['integrated_error'] - base[k]['integrated_error'] for k in keys]
    return dict(episodes=len(keys), candidate_only_success=[k for k in keys if changed[k]['success'] and not base[k]['success']],
        base_only_success=[k for k in keys if base[k]['success'] and not changed[k]['success']],
        paired_integrated_error_difference=describe(errors),
        common_success_cases=common, common_success_count=len(common),
        common_success_time_difference_s=describe([changed[k]['elapsed_s'] - base[k]['elapsed_s'] for k in common]),
        mean_input_difference=float(np.mean([changed[k]['input_squared_integral'] - base[k]['input_squared_integral'] for k in keys])))


def snapshot_check(path):
    snap = read_json(path)
    if not snap.get('query_success') or not snap.get('all_diagnostic_writers_absent'):
        raise RuntimeError('A successful complete external process snapshot is required')
    if not 0 <= time.time() - snap['created_unix'] < 60:
        raise RuntimeError('External process snapshot must be less than 60 seconds old')
    for process in snap['processes']:
        command = process.get('CommandLine')
        if not command:
            raise RuntimeError('Python process command line unavailable')
        if 'meetings_mcp_entrypoint.py' not in command:
            raise RuntimeError('Uncleared Python process: ' + command)
    return snap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--process-snapshot', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    snapshot_check(args.process_snapshot)
    protocol = read_json(root/'PROTOCOL.json')
    criteria = protocol['control']['success']
    expected_cases = {r['case_id'] for r in read_json(root/'VALIDATION_CASES.json')}
    suite = read_json(root/'TRAINING_SUITE_COMPLETE.json')
    assert suite['status'] == 'complete' and suite['completed_evaluation_episodes'] == 768
    historical = read_json(root/'legacy_transfer/RESULTS.json')
    assert len(historical['summaries']) == 10
    frozen = {}
    for name in ('conditional_training_suite', 'legacy_transfer'):
        stage = read_json(root/'stage_protocols'/f'{name}.json')
        assert sha256(root/'PROTOCOL.json') == stage['protocol_sha256']
        for source, expected in stage['sources'].items():
            assert sha256(source) == expected, source
        frozen[name] = stage
    for row in suite['arms']:
        assert sha256(row['checkpoint']) == row['sha256']
        assert sha256(row['marker']) == row['marker_sha256']
    for source, expected in frozen['legacy_transfer']['settings']['historical_assets'].items():
        assert sha256(source) == expected, source
    assert sha256(Path(protocol['old_root'])/protocol['checkpoint']) == protocol['checkpoint_sha256']
    events = [read_json(p) for p in (root/'events').glob('*.json')]
    group_frozen = next(e['unix'] for e in events if e.get('stage') == 'training_group_frozen')
    first_repair = min(e['unix'] for e in events if e.get('stage') == 'repair_validation_started')
    first_transfer = min(e['unix'] for e in events if e.get('stage') == 'legacy_transfer_started')
    assert frozen['conditional_training_suite']['created_unix'] < group_frozen < first_repair
    assert frozen['legacy_transfer']['created_unix'] < first_transfer
    maxima = dict(action=0., value=0., physical_error=0., integrated_error=0., input_integral=0., initial_pose=0.)
    values, dynamics = defaultdict(list), defaultdict(list)
    initial, rows_by_group, counts = {}, {}, defaultdict(int)
    paths = [('training', p) for p in sorted((root/'training_evaluation').glob('*/*/*/RESULTS.json'))]
    paths += [('legacy', p) for p in sorted((root/'legacy_transfer').glob('*/*/RESULTS.json'))]
    assert len(paths) == 26
    for family, path in paths:
        group = family + '/' + '/'.join(path.parent.relative_to(root/('training_evaluation' if family == 'training' else 'legacy_transfer')).parts)
        rows = read_json(path)['rows']
        assert len(rows) == 48 and {r['case_id'] for r in rows} == expected_cases
        rows_by_group[group] = rows
        for row in rows:
            trajectory = Path(row['trajectory'])
            assert sha256(trajectory) == row['trajectory_sha256']
            with np.load(trajectory, allow_pickle=False) as archive:
                a = {k: np.asarray(archive[k]) for k in archive.files}
            n = row['steps']
            e, J, M = (a[k].astype(float) for k in ('error', 'J', 'M'))
            assert e.shape == (n+1, 16) and J.shape == (n, 16, 4) and M.shape == (16, 16)
            assert len(a['dt']) == n and np.isfinite(e).all() and np.isfinite(J).all() and np.isfinite(M).all()
            assert np.allclose(a['dt'], .1) and (a['dt'] > 0).all()
            eig = np.linalg.eigvalsh(M)
            assert eig[0] >= .1-1e-5 and abs(np.trace(M)-16) < 1e-4
            if row['method'] == 'identity_M' or row['method'].endswith('_identity'):
                assert np.allclose(M, np.eye(16), atol=1e-7, rtol=0)
            G = J.transpose(0, 2, 1) @ M @ J
            H, index = G.copy(), np.arange(4)
            eta = (np.maximum(1.01*np.trace(G, axis1=1, axis2=2)/(1e4-1), 1e-8)
                   if family == 'legacy' else np.full(n, .02))
            H[:, index, index] += .01*G[:, index, index]+eta[:, None]
            rhs = (J.transpose(0, 2, 1) @ M @ e[:-1, :, None])[..., 0]
            raw = -.7*np.linalg.solve(H, rhs[..., None])[..., 0]
            radius = np.maximum(1., np.maximum(np.linalg.norm(raw[:, :3], axis=1), np.abs(raw[:, 3])))
            u = raw/radius[:, None]
            maxima['action'] = max(maxima['action'], float(np.max(np.abs(u-a['action']))))
            assert np.allclose(u, a['action'], atol=5e-4 if family == 'legacy' else 5e-5, rtol=1e-4 if family == 'legacy' else 5e-5)
            assert (np.linalg.norm(a['action'][:, :3], axis=1) <= 1+1e-6).all()
            assert (np.abs(a['action'][:, 3]) <= 1+1e-6).all()
            V = .5*np.einsum('ti,ij,tj->t', e, M, e)
            maxima['value'] = max(maxima['value'], float(np.max(np.abs(V-a['V']))))
            assert np.allclose(V, a['V'], atol=1e-5, rtol=1e-5)
            q = a['offset'].astype(float)
            E = np.sum(q[:, :3]**2, axis=1)+(q[:, 3]/math.radians(25))**2
            maxima['physical_error'] = max(maxima['physical_error'], float(np.max(np.abs(E-a['physical_error']))))
            assert np.allclose(E, a['physical_error'], atol=1e-8, rtol=1e-6)
            integral = float(E[1:]@a['dt'].astype(float))
            effort = float(np.sum(a['action'].astype(float)**2, axis=1)@a['dt'].astype(float))
            maxima['integrated_error'] = max(maxima['integrated_error'], abs(integral-row['integrated_error']))
            maxima['input_integral'] = max(maxima['input_integral'], abs(effort-row['input_squared_integral']))
            assert np.isclose(integral, row['integrated_error'], atol=1e-7, rtol=1e-6)
            assert np.isclose(effort, row['input_squared_integral'], atol=1e-7, rtol=1e-6)
            assert not a['terminated'][:-1].any() and not a['truncated'][:-1].any() and not a['success'][:-1].any()
            assert bool(a['terminated'][-1] or a['truncated'][-1])
            tolerance = ((np.linalg.norm(q[:, :3], axis=1) < criteria['position_tolerance_m']) &
                         (np.abs(q[:, 3]) < math.radians(criteria['yaw_tolerance_deg'])))
            hold, hold_counts = 0, []
            for good in tolerance[1:]:
                hold = hold+1 if good else 0
                hold_counts.append(hold)
            assert np.array_equal(hold_counts, a['hold_observations'])
            assert bool(row['success']) == bool(a['success'][-1])
            if row['success']:
                assert hold_counts[-1] >= criteria['required_hold_observations'] and a['terminated'][-1] and row['reason'] == 'success'
            else:
                assert hold_counts[-1] < criteria['required_hold_observations']
            initial[(group, row['case_id'])] = dict(error=e[0], metric=M, offset=q[0], J=J[0], version=row.get('version'))
            valid = E > 1e-12
            near = ((np.linalg.norm(q[:, :3], axis=1) < .1) & (np.abs(q[:, 3]) < math.radians(5)) & ~tolerance)
            outcome = 'success' if row['success'] else 'failure'
            feat2 = np.sum(e**2, axis=1)
            eps = 1e-7+1e-4*E
            columns = np.column_stack((V/np.maximum(E, 1e-30), V < .1*E-eps, V > 4*E+eps,
                                       7.25*feat2 < .1*E-eps, .05*feat2 > 4*E+eps))
            masks = dict(initial=np.arange(n+1) == 0, all=np.ones(n+1, dtype=bool), near=near)
            masks['states_'+outcome] = masks['all']
            for name, mask in masks.items():
                values[group+'/'+name].append(columns[mask & valid])
            predicted = np.einsum('tij,tj->ti', J, a['applied_action'])*a['dt'][:, None]
            actual = np.diff(e, axis=0)
            norms = np.linalg.norm(predicted, axis=1)*np.linalg.norm(actual, axis=1)
            valid_J = (np.linalg.norm(predicted, axis=1) > 1e-7) & (np.linalg.norm(actual, axis=1) > 1e-7)
            cosine = (predicted*actual).sum(axis=1)/np.maximum(norms, 1e-30)
            pred_e = e[:-1]+predicted
            predicted_delta = .5*np.einsum('ti,ij,tj->t', pred_e, M, pred_e)-V[:-1]
            actual_delta = np.diff(V)
            v_tol = 1e-7+1e-4*np.abs(V[:-1])
            dyn = np.column_stack((cosine, cosine < 0, actual_delta > v_tol,
                                   (predicted_delta < -v_tol) & (actual_delta > v_tol),
                                   np.linalg.norm(a['action'], axis=1)))
            dynamics[group+'/all'].append(dyn[valid_J])
            dynamics[group+'/near'].append(dyn[valid_J & near[:-1]])
            counts[family+'_episodes'] += 1
            counts[family+'_transitions'] += n
    assert counts['training_episodes'] == 768 and counts['legacy_episodes'] == 480
    initial_feature_max, initial_metric_max, initial_J_max = 0., 0., 0.
    for arm in [r['arm'] for r in suite['arms']]:
        for plant in ('P0', 'P1'):
            for case in expected_cases:
                base = initial[(f'training/{arm}/{plant}/baseline', case)]
                identity = initial[(f'training/{arm}/{plant}/identity_M', case)]
                initial_feature_max = max(initial_feature_max, float(np.max(np.abs(base['error']-identity['error']))))
                initial_J_max = max(initial_J_max, float(np.max(np.abs(base['J']-identity['J']))))
                assert np.array_equal(base['error'], identity['error']) and np.array_equal(base['J'], identity['J'])
                assert base['version'] == identity['version']
                if plant == 'P1':
                    p0 = initial[(f'training/{arm}/P0/baseline', case)]
                    initial_metric_max = max(initial_metric_max, float(np.max(np.abs(base['metric']-p0['metric']))))
                    assert np.array_equal(base['error'], p0['error']) and np.array_equal(base['metric'], p0['metric'])
    for plant in ('P0', 'P1'):
        for case in expected_cases:
            best = initial[(f'legacy/{plant}/legacy_best_M', case)]
            identity = initial[(f'legacy/{plant}/legacy_best_identity', case)]
            assert np.array_equal(best['error'], identity['error']) and np.array_equal(best['J'], identity['J'])
            current = initial[(f'legacy/{plant}/v4_current_J_legacy_ridge', case)]
            target = initial[(f'legacy/{plant}/v4_target_J_legacy_ridge', case)]
            assert np.array_equal(current['error'], target['error']) and np.array_equal(current['metric'], target['metric'])
    for case in expected_cases:
        records = [value for (key, identifier), value in initial.items() if identifier == case]
        for value in records:
            diff = float(np.max(np.abs(value['offset']-records[0]['offset'])))
            maxima['initial_pose'] = max(maxima['initial_pose'], diff)
            assert np.array_equal(value['offset'], records[0]['offset'])
    strata = {}
    for key, blocks in values.items():
        data = np.concatenate(blocks, axis=0)
        strata[key] = dict(states=len(data), V_over_E=describe(data[:, 0]),
            lower_violation=float(data[:, 1].mean()) if len(data) else None,
            upper_violation=float(data[:, 2].mean()) if len(data) else None,
            lower_impossible_even_with_best_allowed_M=float(data[:, 3].mean()) if len(data) else None,
            upper_impossible=float(data[:, 4].mean()) if len(data) else None)
    motion = {}
    for key, blocks in dynamics.items():
        data = np.concatenate(blocks, axis=0)
        motion[key] = dict(effective_transitions=len(data), cosine=describe(data[:, 0]),
            opposite_fraction=float(data[:, 1].mean()) if len(data) else None,
            actual_V_increase_fraction=float(data[:, 2].mean()) if len(data) else None,
            predicted_descent_actual_increase_fraction=float(data[:, 3].mean()) if len(data) else None,
            normalized_action_norm=describe(data[:, 4]))
    comparisons = {}
    for plant in ('P0', 'P1'):
        v4 = read_json(root/'control'/plant/'baseline/RESULTS.json')['rows']
        for group, rows in rows_by_group.items():
            if f'/{plant}/' in group:
                comparisons['v4_baseline_to/'+group] = compare(v4, rows)
        for arm in [r['arm'] for r in suite['arms']]:
            base = rows_by_group[f'training/{arm}/{plant}/identity_M']
            learned = rows_by_group[f'training/{arm}/{plant}/baseline']
            comparisons[f'identity_to_M/{arm}/{plant}'] = compare(base, learned)
        comparisons[f'legacy_identity_to_M/{plant}'] = compare(rows_by_group[f'legacy/{plant}/legacy_best_identity'], rows_by_group[f'legacy/{plant}/legacy_best_M'])
        for loss in ('original', 'cal_precision'):
            for method in ('baseline', 'identity_M'):
                comparisons[f'split_to_alternating/{loss}/{plant}/{method}'] = compare(
                    rows_by_group[f'training/{loss}_split/{plant}/{method}'],
                    rows_by_group[f'training/{loss}_alternating/{plant}/{method}'])
    result = dict(schema='rgb_v5_followup_trajectory_audit', passed=True, counts=dict(counts), maximum_absolute_differences=maxima,
        process_snapshot=str(args.process_snapshot.resolve()), process_snapshot_sha256=sha256(args.process_snapshot),
        audit_source=str(Path(__file__).resolve()), audit_source_sha256=sha256(__file__),
        source_order=dict(training_stage_frozen=frozen['conditional_training_suite']['created_unix'],
                          all_four_models_frozen=group_frozen, first_repair_validation=first_repair,
                          legacy_stage_frozen=frozen['legacy_transfer']['created_unix'], first_legacy_evaluation=first_transfer),
        same_initial_coordinates=dict(training_learned_vs_identity_feature_max_difference=initial_feature_max,
                                      training_learned_vs_identity_J_max_difference=initial_J_max,
                                      training_P0_vs_P1_metric_max_difference=initial_metric_max),
        groups={key:dict(episodes=len(rows),successes=sum(r['success'] for r in rows)) for key,rows in rows_by_group.items()},
        strata=strata, motion=motion, comparisons=comparisons,
        definitions=dict(V='0.5 e.T M e, same fixed parameters and scale at both endpoints',
            E='squared position metres plus squared yaw/radians(25)',
            near='position<0.1m, yaw<5deg, excluding original success tolerance; descriptive only',
            feasible_envelope='0.05||e||^2<=V<=7.25||e||^2; necessary, different optimum M allowed per state',
            motion='only nonzero predicted and actual feature changes; P1 uses archived applied velocity',
            all='time-correlated states, includes terminal; unequal duration weights; no confidence interval',
            paired_time='only the identical starts where both methods strictly succeeded'),
        limitations=['validation development only, not untouched testing',
                     'historical models retained pretrained weights; changed GPU renderer and 200-step horizon',
                     'trajectory e/J/M arithmetic independently audited; images not reencoded by this CPU audit',
                     'physical consistency of J is not proved by accurate numerical action recomputation',
                     'single selected source seed; no new nonlinear/global stability result'])
    write_json(root/'FOLLOWUP_TRAJECTORY_AUDIT.json', result)
    print(json.dumps(dict(passed=True, counts=dict(counts), maxima=maxima), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
