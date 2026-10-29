"""Bounded, explicitly gated execution checks for PushAnyShape integral v1.

Writing/importing this file does not construct an environment. The run function
requires a separate root-issued gate binding the entire current source set.
The intended path executes fourteen control steps (eight numeric and six
contact), including two snapshot replay branches. Rejected API commits are
counted separately; every attempted execution has a durable preattempt marker.
No optimizer, old checkpoint, full episode, or automatic retry is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXTENSION_ROOT.parents[1]
SCHEMA = 'push_any_shape_integral_contract_short_v1'
HARD_MAX_CONTROL_STEPS = 40
HARD_MAX_CONTACT_SUBSTEPS = 160
SOURCE_RELATIVE = (
    'push_any_shape/__init__.py', 'push_any_shape/shapes.py', 'push_any_shape/mode_library.py',
    'push_any_shape/integral_certificate.py', 'push_any_shape/controller.py',
    'push_any_shape/contact_executor.py', 'push_any_shape/task.py',
    'scripts/check_task_contract.py', 'configs/disk_engineering_v1.json',
)
SHARED_RELATIVE = (
    'stabilization-learning-control/stabilization_learning/policies/metric.py',
    'stabilization-gym/stabilization_gym/common/types.py',
    'stabilization-gym/stabilization_gym/interfaces/m_learning.py',
)


def utc8_now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    # Include __init__ and future additions: a new package module after the
    # gate is a source-inventory mismatch, even if existing files are intact.
    owned = {EXTENSION_ROOT / rel for rel in SOURCE_RELATIVE}
    owned.update((EXTENSION_ROOT / 'push_any_shape').rglob('*.py'))
    return {str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'): sha(path)
            for path in sorted(owned)} | {rel: sha(PROJECT_ROOT / rel) for rel in SHARED_RELATIVE}


def jsonable(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    """Never replace prior execution evidence or a possibly partial attempt."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(jsonable(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


class Ledger:
    def __init__(self, output, max_seconds):
        self.output = Path(output)
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.attempts = 0
        self.committed = {'numeric': 0, 'contact': 0}
        self.rejected = 0
        self.contact_accepted_substeps = 0
        self.contact_attempted_substeps = 0
        self.checks = {}
        self.environments = {'numeric': 0, 'contact': 0}
        self.replay_commits = 0
        self.planned_calls = 0
        self.snapshot_restores = 0
        self.query_rejections = 0
        self.version_probes = 0

    def check(self, name, condition, details=None):
        self.checks[name] = {'passed': bool(condition), 'details': jsonable(details)}
        if not condition:
            raise AssertionError(name)

    def check_boundary(self):
        if time.monotonic() - self.started > self.max_seconds:
            raise RuntimeError('Predeclared short wall-time boundary reached; do not retry automatically')
        for path in (PROJECT_ROOT / 'STOP', EXTENSION_ROOT / 'STOP', self.output / 'STOP'):
            if path.exists():
                raise RuntimeError('User STOP exists: ' + str(path))

    def control(self, task, actor, sample=None, *, training=False):
        self.check_boundary()
        self.planned_calls += 1
        return task.control(actor, sample, training=training)

    def step(self, task, actor, control, label, *, expect_rejection=False,
             replay=False, perturb=None, cleanup=None):
        self.check_boundary()
        total = sum(self.committed.values())
        if not expect_rejection and total >= HARD_MAX_CONTROL_STEPS:
            raise RuntimeError('Hard total control-step budget exhausted')
        if task.backend == 'contact' and not expect_rejection:
            # Reserve worst-case four physical substeps before entering physics.
            if self.contact_attempted_substeps + 4 > HARD_MAX_CONTACT_SUBSTEPS:
                raise RuntimeError('Hard contact substep budget exhausted')
        self.attempts += 1
        sequence = self.attempts
        prefix = self.output / f'{sequence:03d}_{label}'
        before = task.snapshot(actor)
        write_new(str(prefix) + '.ATTEMPT.json', dict(sequence=sequence, label=label,
            backend=task.backend, created_at=utc8_now(), expect_rejection=expect_rejection,
            replay_branch=replay, action=control.action, before=before))
        if perturb is not None:
            if not expect_rejection:
                raise ValueError('Perturbations are limited to declared zero-physics negative tests')
            perturb()
        try:
            result = task.step(control, actor=actor)
        except (ValueError, TypeError, RuntimeError) as exc:
            if cleanup is not None:
                cleanup()
            if not expect_rejection:
                # Physics may have started before an exception. Retain an explicit
                # uncertainty marker; never silently retry a partially executed call.
                write_new(str(prefix) + '.UNCOMMITTED_ERROR.json', dict(
                    exception=type(exc).__name__, message=str(exc),
                    possible_uncommitted_execution=True, no_replay_authorized=True))
                raise
            after = task.snapshot(actor)
            self.check(f'{label}_rejected_before_state_change', before == after)
            self.rejected += 1
            write_new(str(prefix) + '.REJECTED.json', dict(exception=type(exc).__name__,
                message=str(exc), before_sha256=before['sha256'], after_sha256=after['sha256']))
            return None
        else:
            if cleanup is not None:
                cleanup()
        self.committed[task.backend] += 1
        self.replay_commits += int(replay)
        observation, reward, terminated, truncated, info = result
        after = task.snapshot(actor)
        h = float(info['actual_dt'])
        previous_t = float(before['integral']['accepted_time_s'][0])
        current_t = float(after['integral']['accepted_time_s'][0])
        actual_substeps = attempted = 0
        if task.backend == 'contact':
            ex = info['execution']
            actual_substeps = len(ex['substep_records'])
            old_budget = before['executor']['budget']
            new_budget = after['executor']['budget']
            attempted = new_budget['attempted_substeps'] - old_budget['attempted_substeps']
            self.contact_accepted_substeps += actual_substeps
            self.contact_attempted_substeps += attempted
        suffix = '.UNEXPECTED_EXECUTION.json' if expect_rejection else '.COMMITTED.json'
        write_new(str(prefix) + suffix, dict(sequence=sequence, label=label,
            backend=task.backend, action=control.action, observation=observation,
            reward=reward, terminated=terminated, truncated=truncated,
            info=info, after=after, replay_branch=replay,
            expected_rejection=expect_rejection,
            current_global_execution_counters=dict(committed_control_steps=self.committed,
                contact_accepted_substeps=self.contact_accepted_substeps,
                contact_attempted_substeps=self.contact_attempted_substeps)))
        if expect_rejection:
            raise AssertionError('A rejected-result probe was unexpectedly executed; actual evidence and counts saved')
        self.check(f'{label}_actual_integral_clock', abs(current_t - previous_t - h) < 1e-12)
        self.check(f'{label}_accepted_step_count',
            after['integral']['accepted_steps'][0] - before['integral']['accepted_steps'][0] == int(h > 0))
        self.check(f'{label}_finite_candidate_value',
            all(__import__('math').isfinite(float(info[k])) for k in ('V_before', 'V_after', 'delta_V')))
        if task.backend == 'contact':
            self.check(f'{label}_physical_elapsed_matches_integral', abs(h - .005 * actual_substeps) < 1e-12)
            self.check(f'{label}_attempted_substeps_bounded', actual_substeps <= attempted <= 4)
        self.check(f'{label}_no_early_termination', not terminated and not truncated,
                   {'terminated':terminated, 'truncated':truncated})
        return result, after

    def reject_query(self, task, actor, sample, label):
        before = task.snapshot(actor)
        try:
            self.control(task, actor, sample)
        except ValueError as exc:
            self.query_rejections += 1
            self.check(label + '_rejected', True)
            self.check(label + '_query_preserved_state', before == task.snapshot(actor))
            write_new(self.output / (label + '.QUERY_REJECTED.json'), dict(
                created_at=utc8_now(), exception=type(exc).__name__, message=str(exc),
                before_sha256=before['sha256'], after_sha256=task.snapshot(actor)['sha256']))
        else:
            self.check(label + '_rejected', False)

    def summary(self):
        complete = bool(self.checks.get('planned_task_step_api_attempts', {}).get('passed'))
        return dict(schema=SCHEMA, created_at=utc8_now(),
            completed_expected_scope=complete,
            passed=complete and all(x['passed'] for x in self.checks.values()), checks=self.checks,
            counters=dict(environment_instances=self.environments,
                task_step_api_attempts=self.attempts,
                committed_control_steps=self.committed,
                total_committed_control_steps=sum(self.committed.values()),
                rejected_commit_attempts=self.rejected,
                rejected_query_attempts=self.query_rejections,
                contact_accepted_physical_substeps=self.contact_accepted_substeps,
                contact_attempted_physical_substeps=self.contact_attempted_substeps,
                snapshot_replay_extra_control_steps=self.replay_commits,
                snapshot_restores=self.snapshot_restores, control_queries=self.planned_calls,
                version_only_zero_add_probes=self.version_probes,
                optimizer_instances=0, optimizer_updates=0, full_episodes=0),
            wall_seconds=time.monotonic() - self.started,
            limits=dict(control_steps=HARD_MAX_CONTROL_STEPS,
                        contact_physical_substeps=HARD_MAX_CONTACT_SUBSTEPS),
            predeclared_expected=dict(committed_control_steps=14, task_step_api_attempts=19,
                rejected_commit_attempts=5, rejected_query_attempts=1),
            scope='bounded interface and execution engineering; not learned task performance or stability proof',
            untested=['Short contact steps do not establish PUSH-phase model accuracy.',
                'Initially separated goal case tests SETTLED/zero drive, not physical DISENGAGE.',
                'Full 30s held completion and cross-shape learning remain untested by this short runner.'])


def validate_gate(gate_path):
    gate = json.loads(Path(gate_path).read_text(encoding='utf-8-sig'))
    if gate.get('schema') != 'push_integral_execution_gate_v1' or gate.get('scope') != SCHEMA:
        raise ValueError('A root-issued gate for this exact short scope is required')
    if gate.get('approved') is not True or gate.get('max_control_steps') != 40:
        raise ValueError('Gate must explicitly approve at most forty control steps')
    actual = source_hashes()
    if gate.get('source_sha256') != actual:
        raise ValueError('Source/config/shared dependency hashes differ from the reviewed gate')
    return gate, actual


def run(args):
    gate, hashes = validate_gate(args.gate)
    if not 1 <= args.max_wall_seconds <= 240:
        raise ValueError('Short engineering wall boundary must be 1..240 seconds')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output; this runner never replays prior attempts')
    output.mkdir(parents=True)
    write_new(output / 'WRITER.json', dict(created_at=utc8_now(), gate=str(Path(args.gate).resolve()),
        gate_sha256=sha(args.gate), source_sha256=hashes, no_resume=True))
    ledger = Ledger(output, args.max_wall_seconds)
    tasks = []
    try:
        # Environment imports/construction happen only after explicit source gate.
        sys.path[:0] = [str(EXTENSION_ROOT), str(PROJECT_ROOT / 'stabilization-gym'),
                        str(PROJECT_ROOT / 'stabilization-learning-control')]
        import numpy as np
        import torch
        from copy import copy, deepcopy
        from dataclasses import replace
        from push_any_shape.integral_certificate import IntegralConfig, IntegralMActor, integrate_eta
        from push_any_shape.task import make_task, actor_digest

        torch.set_num_threads(1)
        torch.manual_seed(41)
        protocol = json.loads((EXTENSION_ROOT / 'configs/disk_engineering_v1.json').read_text(encoding='utf-8'))
        cfg = IntegralConfig(**protocol['integral_config'])
        actor = IntegralMActor(cfg).double().eval()
        actor_hash = actor_digest(actor)

        def new_task(shape, backend):
            ledger.check_boundary()
            task = make_task(shape=shape, backend=backend, cfg=cfg)
            tasks.append(task)
            ledger.environments[backend] += 1
            return task

        def assert_pure(task, label):
            before = task.snapshot(actor)
            sample = task.error_context()
            cert = task.certificate(actor, sample)
            c1 = ledger.control(task, actor, sample)
            c2 = ledger.control(task, actor, sample)
            after = task.snapshot(actor)
            ledger.check(label + '_query_purity', before == after)
            ledger.check(label + '_deterministic_action', c1.action == c2.action)
            ledger.check(label + '_identity_metric', torch.equal(cert.M,
                torch.eye(6, dtype=torch.float64)[None]))
            ledger.check(label + '_candidate_claim', cert.claim == 'candidate_only')
            desc = task.describe()
            ledger.check(label + '_pure_declared', desc.control_call_is_pure)
            try:
                task.require_algorithm('episode_metric_ppo')
            except ValueError:
                rejected = True
            else:
                rejected = False
            ledger.check(label + '_incompatible_learner_rejected', rejected)
            return sample, c1

        def exercise_branch(task, label):
            sample, first = assert_pure(task, label)
            ledger.step(task, actor, first, label + '_step1')
            ledger.step(task, actor, first, label + '_duplicate', expect_rejection=True)
            unchanged = task.snapshot(actor)
            try:
                task.control(actor, sample)
            except ValueError:
                stale_rejected = True
            else:
                stale_rejected = False
            ledger.check(label + '_stale_sample_rejected', stale_rejected)
            ledger.check(label + '_stale_query_pure', unchanged == task.snapshot(actor))
            ledger.step(task, actor, ledger.control(task, actor), label + '_step2')
            checkpoint = task.snapshot(actor)
            proposal_a = ledger.control(task, actor)
            result_a, next_a = ledger.step(task, actor, proposal_a, label + '_branchA')
            task.restore(json.loads(json.dumps(checkpoint)), actor)
            ledger.snapshot_restores += 1
            ledger.check(label + '_full_restore', checkpoint == task.snapshot(actor))
            proposal_b = ledger.control(task, actor)
            ledger.check(label + '_restored_next_action', proposal_a.action == proposal_b.action)
            result_b, next_b = ledger.step(task, actor, proposal_b, label + '_branchB', replay=True)
            ledger.check(label + '_restored_next_snapshot_exact', next_a == next_b)
            ledger.check(label + '_restored_observation_exact', np.array_equal(result_a[0], result_b[0]))
            ledger.check(label + '_real_signed_integral_changed',
                any(abs(x) > 0 for x in next_b['integral']['eta'][0][:2]))

        numeric = new_task('Disk', 'numeric')
        # Four fixed negative tests precede real execution. The one in-place
        # operation adds exactly zero: weight bytes remain unchanged, while
        # Torch's parameter version changes and must invalidate an old result.
        numeric.reset(seed=40, options={'pose':[-.025,0.,.4], 'target':[0.,0.,0.]})
        valid_control = ledger.control(numeric, actor)
        bad_payload = deepcopy(valid_control.next_controller_state)
        bad_payload['u'] += .001
        bad_control = replace(valid_control, next_controller_state=bad_payload)
        ledger.step(numeric, actor, bad_control, 'tampered_hmac_payload', expect_rejection=True)
        parameter = next(actor.parameters())
        def bump_parameter_version_only():
            with torch.no_grad():
                parameter.add_(0.)
            ledger.version_probes += 1
        ledger.step(numeric, actor, valid_control, 'actor_version_changed',
                    expect_rejection=True, perturb=bump_parameter_version_only)
        numeric.reset(seed=40, options={'pose':[-.025,0.,.4], 'target':[0.,0.,0.]})
        original_sample = numeric.error_context()
        modified_sample = copy(original_sample)
        modified_sample.metric_context = dict(original_sample.metric_context)
        original_C = original_sample.metric_context['C'].clone()
        modified_sample.metric_context['C'] = original_C.clone()
        modified_sample.metric_context['C'][0,0,0] += .001
        ledger.reject_query(numeric, actor, modified_sample, 'modified_sample_C')
        ledger.check('modified_sample_did_not_pollute_bank', torch.equal(
            numeric.error_context().metric_context['C'], original_C))
        origin_control = ledger.control(numeric, actor)
        original_target = numeric.target.copy()
        def perturb_target():
            numeric.target[0] += .001
        def restore_target():
            numeric.target[:] = original_target
        ledger.step(numeric, actor, origin_control, 'direct_target_mutation',
                    expect_rejection=True, perturb=perturb_target, cleanup=restore_target)
        numeric.reset(seed=41, options={'pose':[-.025,0.,.4], 'target':[0.,0.,0.]})
        # Actor gradients use a fresh training query but no backward optimizer.
        before_grad = numeric.snapshot(actor)
        differentiable = ledger.control(numeric, actor, training=True)
        grads = torch.autograd.grad(differentiable.policy_logits.sum(), tuple(actor.parameters()),
                                    allow_unused=True)
        ledger.check('candidate_logits_parameter_gradient', any(g is not None and
            bool(torch.isfinite(g).all()) and bool((g.abs() > 1e-12).any()) for g in grads))
        ledger.check('gradient_query_did_not_commit', before_grad == numeric.snapshot(actor))
        exercise_branch(numeric, 'numeric_disk')

        # Free target yaw is a gauge: compare reset-query outputs, no execution.
        numeric.reset(seed=42, options={'pose':[-.025,.005,.4], 'target':[0.,0.,0.]})
        gauge_e0 = numeric.error_context().error.clone()
        gauge_action0 = ledger.control(numeric, actor).action
        numeric.reset(seed=42, options={'pose':[-.025,.005,.4], 'target':[0.,0.,1.7]})
        gauge_e1 = numeric.error_context().error.clone()
        gauge_action1 = ledger.control(numeric, actor).action
        ledger.check('disk_target_yaw_gauge_error', torch.equal(gauge_e0, gauge_e1))
        ledger.check('disk_target_yaw_gauge_action', gauge_action0 == gauge_action1)

        original = new_task('Ellipse', 'numeric')
        original.reset(seed=43, options={'pose':[.225,.25,.02], 'target':[.25,.25,0.]})
        assert_pure(original, 'preserved_ellipse')
        for i in range(2):
            ledger.step(original, actor, ledger.control(original, actor), f'preserved_ellipse_{i}')

        numeric.reset(seed=44, options={'pose':[0.,0.,.4], 'target':[0.,0.,2.1]})
        for i in range(2):
            control = ledger.control(numeric, actor)
            ledger.check(f'numeric_goal_{i}_zero_push', control.action['u'] == 0.)
            _, snap = ledger.step(numeric, actor, control, f'numeric_goal_{i}')
            ledger.check(f'numeric_goal_{i}_masked_integral_zero', snap['integral']['eta'] == [[0.,0.,0.]])

        # Nonzero eta hold leakage is checked as arithmetic, not a fake reset.
        eta = torch.tensor([[.2,-.1,0.]], dtype=torch.float64)
        D = torch.tensor([[.01,-.01,0.]], dtype=torch.float64)
        held = integrate_eta(eta,D,D,.02,task_mask=[1.,1.,0.],goal_hold=True,cfg=cfg)
        ledger.check('nonzero_eta_hold_leak_arithmetic_only',
            torch.allclose(held, eta * torch.exp(torch.tensor(-.004,dtype=torch.float64)),atol=1e-14,rtol=0))

        contact = new_task('Disk', 'contact')
        contact.reset(seed=45, options={'disk_xy':[-.025,0.], 'pusher_xy':[-.05,0.],
            'goal_xy':[0.,0.], 'yaw':.4, 'qvel':[0.,0.,0.,0.,0.]})
        exercise_branch(contact, 'contact_disk')
        contact.reset(seed=46, options={'disk_xy':[0.,0.], 'pusher_xy':[-.03,0.],
            'goal_xy':[0.,0.], 'yaw':.4, 'target_yaw':2.1, 'qvel':[0.,0.,0.,0.,0.]})
        for i in range(2):
            control = ledger.control(contact, actor)
            ledger.check(f'contact_goal_{i}_zero_push', control.action['u'] == 0.)
            result, snap = ledger.step(contact, actor, control, f'contact_goal_{i}')
            ledger.check(f'contact_goal_{i}_physical_target_invariant',
                all(result[4]['goal_conditions'][k] for k in
                    ('position','disk_stopped','yaw_rate_stopped','pusher_stopped','pusher_separated','safe')))
            ledger.check(f'contact_goal_{i}_masked_integral_zero', snap['integral']['eta'] == [[0.,0.,0.]])

        ledger.check('actor_weights_unchanged', actor_digest(actor) == actor_hash)
        ledger.check('planned_control_budget', sum(ledger.committed.values()) == 14,
                     ledger.committed)
        ledger.check('planned_contact_control_budget', ledger.committed['contact'] == 6)
        ledger.check('planned_zero_physics_step_rejections', ledger.rejected == 5)
        ledger.check('planned_zero_physics_query_rejection', ledger.query_rejections == 1)
        ledger.check('planned_task_step_api_attempts', ledger.attempts == 19)
        ledger.check('contact_substep_global_budget', ledger.contact_attempted_substeps <= 24)
        ledger.check('source_hashes_unchanged', hashes == source_hashes())
        report = ledger.summary()
        report.update(source_sha256=hashes, gate_sha256=sha(args.gate), actor_sha256=actor_hash,
                      fixture_config_sha256=sha(EXTENSION_ROOT / 'configs/disk_engineering_v1.json'))
        write_new(output / 'SUMMARY.json', report)
        print(json.dumps({'passed':report['passed'],'summary':str(output/'SUMMARY.json'),
                          'counters':report['counters']},ensure_ascii=False))
        return 0
    except Exception as exc:
        write_new(output / 'ERROR.json', dict(created_at=utc8_now(),exception=type(exc).__name__,
            message=str(exc),summary=ledger.summary(),no_automatic_retry=True,
            possible_uncommitted_execution='Inspect every ATTEMPT marker without a COMMITTED/REJECTED marker'))
        raise
    finally:
        for task in tasks:
            task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-reviewed-short', action='store_true')
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wall-seconds', type=float, default=120.)
    args = parser.parse_args()
    if not args.execute_reviewed_short:
        parser.error('Execution requires --execute-reviewed-short and an independent root gate')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
