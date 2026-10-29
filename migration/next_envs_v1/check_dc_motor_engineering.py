"""CPU engineering checks only; zero optimizer construction/updates.

Writes only a fresh subdirectory beside this script. Four deterministic full
episodes check protocol and interfaces, not learned performance or test scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
FROZEN_PROTOCOL = PROJECT / "runs/book_m_night_suite_20261023_v1/protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_ready(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(HERE):
        raise ValueError("Use a fresh output subdirectory of migration/next_envs_v1")
    protocol_hash = sha(FROZEN_PROTOCOL)
    frozen = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))["sources"]
    source_before = {name: sha(PROJECT / name) for name in frozen}
    if source_before != frozen:
        raise RuntimeError("Frozen suite sources differ; stop before importing/using packages")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    for directory in (PROJECT / "stabilization-gym", PROJECT / "stabilization-learning-control",
                      PROJECT / "third_party/python/book_frequency_threadpoolctl_3_6_0/site-packages"):
        sys.path.insert(0, str(directory))
    import numpy as np
    import scipy
    from scipy.linalg import expm
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    from gymnasium.utils.env_checker import check_env
    from gymnasium.envs.registration import registry
    from stabilization_gym.interfaces.m_learning import PortableMFacade
    from stabilization_learning.policies.metric import CholeskyMetric
    from stabilization_learning.wrappers import rollout_metric
    from dc_motor_env import DCMotorExactZOHEnv
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checks, evidence, episodes = [], {}, []

    def add(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def rejects(function, exception=ValueError):
        try:
            function()
            return False
        except exception:
            return True

    def complete_episode(label, policy):
        env = DCMotorExactZOHEnv(target_speed=1.)
        obs, _ = env.reset(seed=41, options={"state": [0., 0.]})
        for index in range(env.max_steps):
            if policy == "feedforward":
                action = env.fixed_feedforward_action()
            else:
                e, ctx = env.build_error(obs)
                M = torch.eye(2, dtype=torch.float64) if policy == "unit" else torch.tensor([[1.3,.25],[.25,.7]], dtype=torch.float64)
                action = env.calculate_action_from_M(M, e, context=ctx)
            obs, _, terminated, truncated, info = env.step(action.detach().numpy())
            if index < env.max_steps-1 and not info["failure"]:
                assert not terminated and not truncated and not info["success"]
            if terminated or truncated:
                break
        record = env.get_episode_audit(); record["engineering_case"] = label
        episodes.append(record)
        return record

    with threadpool_limits(limits=1):
        pools = threadpool_info()
        add("cpu_single_thread", torch.get_num_threads() == 1 and all(p.get("num_threads") == 1 for p in pools), pools)
        env = DCMotorExactZOHEnv(target_speed=1.)
        obs, initial = env.reset(seed=17, options={"state": [.95, 1.02]})
        sample_error, context = env.build_error(obs)
        block = np.array([[-100., -10., 100.], [10., -10., 0.], [0., 0., 0.]])
        expected = expm(.01*block)
        zoh_error = max(np.max(abs(env.Ad-expected[:2,:2])), np.max(abs(env.Bd-expected[:2,2:])))
        add("continuous_parameters_exact_zoh", zoh_error < 1e-14, float(zoh_error))
        original = json.loads((PROJECT/"migration/dc_motor_preparation/20261023_v1/MODEL_ANALYSIS.json").read_text(encoding="utf-8"))
        high_ad = np.array(original["independent_closed_form_decimal"]["Ad"], float)
        high_bd = np.array(original["independent_closed_form_decimal"]["Bd"], float).reshape(2,1)
        add("independent_60_digit_model_match", max(np.max(abs(env.Ad-high_ad)),np.max(abs(env.Bd-high_bd))) < 1e-14)
        equilibrium = env.Ad@env.goal+env.Bd[:,0]*env.equilibrium_voltage
        add("equilibrium_current_speed_and_voltage", np.max(abs(equilibrium-env.goal)) < 1e-14 and env.equilibrium_voltage == 1.1)
        add("new_protocol_exact_values", env.dt == .01 and env.max_steps == 200 and env.required_hold_seconds == 1.
            and env.voltage_deviation_weight == .01/25 and env.metric_spec.raw_dim == 3 and env.metric_spec.trace == 2.)
        add("context_only_target_speed", torch.equal(env.metric_context(), torch.tensor([.5])) and env.metric_context_dim == 1)
        add("physical_cost_separate_from_certificate_scale", abs(env.physical_cost(obs)-float(sample_error.square().sum())*36) < 1e-13)
        identity = torch.eye(2,dtype=torch.float64)
        metric = torch.tensor([[1.3,.25],[.25,.7]],dtype=torch.float64,requires_grad=True)
        error = sample_error.detach().requires_grad_()
        before_state, before_steps = env.state.copy(), env._elapsed_steps
        action = env.calculate_action_from_M(metric,error,context=context)
        identity_action = env.calculate_action_from_M(identity,error,context=context)
        add("nonunit_M_changes_action", float(abs(action-identity_action)) > 1e-6,
            {"unit_voltage":float(identity_action.detach()),"nonunit_voltage":float(action.detach())})
        add("control_query_does_not_advance_state_time", np.array_equal(env.state,before_state) and env._elapsed_steps == before_steps)
        grad_m, grad_e = torch.autograd.grad(action.sum(),(metric,error))
        eps=1e-6
        directions=[torch.diag(torch.tensor([1.,-1.],dtype=torch.float64)),torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)]
        metric_fd=[]
        for direction in directions:
            plus=env.calculate_action_from_M(metric.detach()+eps*direction,error.detach(),context=context)
            minus=env.calculate_action_from_M(metric.detach()-eps*direction,error.detach(),context=context)
            metric_fd.append(abs(float((plus-minus)/(2*eps))-float((grad_m*direction).sum())))
        error_fd=[]
        for direction in torch.eye(2,dtype=torch.float64):
            plus=env.calculate_action_from_M(metric.detach(),error.detach()+eps*direction,context=context)
            minus=env.calculate_action_from_M(metric.detach(),error.detach()-eps*direction,context=context)
            error_fd.append(abs(float((plus-minus)/(2*eps))-float((grad_e*direction).sum())))
        add("M_symmetric_trace_preserving_finite_difference", max(metric_fd)<1e-7,metric_fd)
        add("error_finite_difference", max(error_fd)<1e-7,error_fd)
        construct=CholeskyMetric(env.metric_spec).double()
        raw=(construct.identity_raw(dtype=torch.float64)+torch.tensor([.1,.2,-.1],dtype=torch.float64)).requires_grad_()
        raw_action=env.calculate_action_from_M(construct(raw),error.detach(),context=context)
        raw_grad,=torch.autograd.grad(raw_action.sum(),raw)
        raw_fd=[]
        for direction in torch.eye(3,dtype=torch.float64):
            plus=env.calculate_action_from_M(construct(raw.detach()+eps*direction),error.detach(),context=context)
            minus=env.calculate_action_from_M(construct(raw.detach()-eps*direction),error.detach(),context=context)
            raw_fd.append(abs(float((plus-minus)/(2*eps))-float((raw_grad*direction).sum())))
        add("shared_cholesky_raw_gradient_nonzero_and_correct", float(raw_grad.abs().max())>1e-8 and max(raw_fd)<1e-7,
            {"gradient":raw_grad.tolist(),"finite_difference_error":raw_fd})
        large=error.detach().new_tensor([10.,10.]).requires_grad_()
        saturated=env.calculate_action_from_M(metric,large,context=context)
        saturation_grad,=torch.autograd.grad(saturated.sum(),large)
        add("saturation_declares_zero_gradient", abs(float(saturated.detach()))==5. and torch.count_nonzero(saturation_grad)==0)
        # Independent scalar constrained objective: clipping must solve the box QP.
        prediction=env.Ae@error.detach().numpy(); b=env.Be; m=metric.detach().numpy()
        unconstrained=-float(b@m@prediction)/float(b@m@b+env.voltage_deviation_weight)
        optimum=np.clip(unconstrained,-5-env.equilibrium_voltage,5-env.equilibrium_voltage)
        def objective(nu):
            candidate=prediction+b*nu
            return float(candidate@m@candidate+env.voltage_deviation_weight*nu*nu)
        add("scalar_box_optimizer_matches_independent_solution", abs(float(action.detach())-env.equilibrium_voltage-optimum)<1e-12
            and objective(optimum)<=min(objective(-5-env.equilibrium_voltage),objective(5-env.equilibrium_voltage))+1e-12)
        state_t=torch.tensor([.3,-.2],dtype=torch.float64,requires_grad=True)
        input_t=torch.tensor([.7],dtype=torch.float64,requires_grad=True)
        jac_x=torch.autograd.functional.jacobian(lambda x:env.torch_step(x,input_t),state_t)
        jac_u=torch.autograd.functional.jacobian(lambda u:env.torch_step(state_t,u),input_t)
        add("exact_torch_model_state_action_gradients", np.max(abs(jac_x.detach().numpy()-env.Ad))<1e-14 and np.max(abs(jac_u.detach().numpy()-env.Bd))<1e-14)
        batch=env.torch_step(torch.stack([state_t,state_t]),torch.stack([input_t,input_t]))
        add("torch_model_batch", batch.shape==(2,2) and torch.equal(batch[0],batch[1]))
        env.reset(seed=5,options={"state":[.3,-.2]})
        expected_step=env.Ad@np.array([.3,-.2])+env.Bd[:,0]*.7
        next_obs,cost,_,_,_=env.step([.7])
        add("Gym_numpy_torch_one_step_same", np.max(abs(next_obs-expected_step))<1e-14 and abs(cost-.01*((.3-1)**2+(-.2-1)**2))<1e-14)
        shared=DCMotorExactZOHEnv(); other=DCMotorExactZOHEnv()
        seed_equal=True; targets=[]; states=[]
        for seed in (41,42,43,57):
            first,_=shared.reset(seed=seed); second,_=other.reset(seed=seed)
            seed_equal &= np.array_equal(first,second) and shared.target_speed==other.target_speed
            targets.append(shared.target_speed); states.append(first.tolist())
            seed_equal &= np.array_equal(shared.step([.2])[0],other.step([.2])[0])
        add("seed_pairs_identical_reset_and_step",seed_equal,{"targets":targets,"initial_states":states})
        add("default_reset_ranges",all(-2<=r<=2 for r in targets) and np.max(abs(np.array(states)))<=3)
        env.reset(seed=1,options={"state":[6.,-6.]})
        applied=env.step([50.])
        add("physical_input_saturates_without_clipping_state", applied[-1]["applied_action"][0]==5.
            and np.max(abs(applied[0]-(env.Ad@np.array([6.,-6.])+env.Bd[:,0]*5)))<1e-13)
        boundary_max=0.
        for current in (-6.,6.):
            for speed in (-6.,6.):
                for voltage in (-5.,5.):
                    successor=env.torch_step(torch.tensor([current,speed]),torch.tensor([voltage]))
                    boundary_max=max(boundary_max,float(successor.abs().max()))
        add("domain_boundary_one_step_no_escape",boundary_max<=6.+1e-6,boundary_max)
        add("reject_invalid_reset_action_and_metric", rejects(lambda:env.reset(options={"state":[6.1,0.]}))
            and rejects(lambda:env.reset(options={"target_speed":2.1}))
            and rejects(lambda:env.step([float('nan')]))
            and rejects(lambda:env.step([0.,0.])))
        env.reset(seed=2,options={"state":[.95,1.02]})
        err,ctx=env.build_error(env.state)
        add("reject_non_SPD_or_wrong_trace",rejects(lambda:env.calculate_action_from_M(-identity,err,context=ctx))
            and rejects(lambda:env.calculate_action_from_M(identity*2,err,context=ctx)))
        first=env.unit_metric_action(); env.step(first.numpy())
        err,ctx=env.build_error(env.state)
        changed=env.calculate_action_from_M(metric.detach(),err,context=ctx)
        add("reject_executed_M_change",rejects(lambda:env.step(changed.detach().numpy())))
        env.reset(seed=2,options={"state":[.95,1.02]})
        stale=env.unit_metric_action()
        add("reject_modified_executed_action",rejects(lambda:env.step(stale.numpy()+.1)))
        env.reset(seed=2,options={"state":[1.,1.]})
        add("instantaneous_goal_is_not_episode_success",env.is_success(env.state) and not env._last_info['success'] and not env._done)
        # Explicit fault injection verifies failure priority and no state clipping.
        env.torch_step=lambda state,action:state.new_tensor([6.2,0.])
        failed=env.step([1.1]); failure_info=failed[-1]
        add("physical_limit_violation_terminates_no_state_clip",failed[0][0]==6.2 and failed[2] and not failed[3]
            and failure_info['failure'] and not failure_info['success'])
        add("finished_episode_rejects_step",rejects(lambda:env.step([1.1]),RuntimeError))
        # Real facade and real rollout path, but no learner/optimizer is constructed.
        portable_env=DCMotorExactZOHEnv(target_speed=1.)
        facade=PortableMFacade(portable_env,name="dc_motor_exact_zoh_unregistered")
        obs,_=facade.reset(seed=41,options={"state":[0.,0.]})
        sample=facade.error_context(obs)
        control=facade.control(identity,sample)
        add("real_PortableMFacade_accepts_contract",facade.describe().gradients.action_wrt_M
            and not facade.describe().control_call_is_pure and control.certificate.claim=='candidate_only')
        generic=rollout_metric(facade,identity,obs,save_trajectory=True)
        record=portable_env.get_episode_audit(); record['engineering_case']='unit_M_via_original_rollout_metric'
        episodes.append(record)
        add("real_rollout_metric_completes_no_optimizer",generic['physical_steps']==200 and record['metric_control_steps']==200)
        evidence['generic_rollout_arrival_time_limitation']={"generic_arrival_time":generic['arrival_time'],
            "environment_arrival_time":record['arrival_time'],"environment_settling_time":record['settling_time'],
            "explanation":"Generic rollout overwrites arrival with steps*dt. Use environment final-segment audit; no frozen wrapper modified."}
        for label,policy in [('fixed_equilibrium_feedforward','feedforward'),('fixed_nonunit_M','nonunit')]:
            complete_episode(label,policy)
        # Fourth full episode: late physical pulse destroys a previously met hold.
        pulse=DCMotorExactZOHEnv(target_speed=1.)
        pulse.reset(seed=41,options={"state":[1.,1.]})
        early_hold=False
        for index in range(200):
            _,_,terminated,truncated,info=pulse.step([1.1 if index<190 else 5.])
            if index==149:
                early_hold=info['hold_condition_met'] and not info['success'] and not terminated
        record=pulse.get_episode_audit(); record['engineering_case']='late_pulse_breaks_final_hold'
        episodes.append(record)
        add("early_hold_not_early_termination",early_hold)
        add("late_goal_exit_resets_hold_and_times_out",not record['success'] and record['physical_truncated']
            and not record['failure'] and record['hold_seconds']==0. and record['arrival_time']==2.)
        # Independent audit of closed [1,2] sampled interval: 101 state samples.
        holds_ok=True
        for record in episodes:
            trace=record['trace']; state_rows=[np.array(record['initial_state'])]+[np.array(r['state_after']) for r in trace]
            final_samples=state_rows[100:201]
            recomputed=len(trace)==200 and len(final_samples)==101 and all(np.all(abs(x-np.array(record['goal']))<=.02) for x in final_samples)
            holds_ok &= record['success']==recomputed and not any(r['success'] for r in trace[:-1])
        add("independent_101_sample_final_second_hold",holds_ok)
        add("four_engineering_full_episodes_only",len(episodes)==4)
        # Standard Gym one-step API checks; no render or learning runs.
        checker=DCMotorExactZOHEnv()
        check_env(checker,skip_render_check=True)
        add("gymnasium_env_checker",True)
        add("new_env_not_registered",checker.spec is None and not any('DCMotor' in key for key in registry))
        sys.path.insert(0,str(PROJECT/'scripts'))
        from m_learning_suite.task_runtime import FullHorizonTask
        add("existing_seven_task_wrapper_rejects_new_task",rejects(lambda:FullHorizonTask(DCMotorExactZOHEnv(),'dc_motor')))
        evidence['module_origins']={'StateMetricEnv':sys.modules['stabilization_gym.envs.base'].__file__,
            'PortableMFacade':sys.modules['stabilization_gym.interfaces.m_learning'].__file__,
            'CholeskyMetric':sys.modules['stabilization_learning.policies.metric'].__file__,
            'isolated_motor':sys.modules['dc_motor_env'].__file__}
    source_after={name:sha(PROJECT/name) for name in frozen}
    add("all_51_frozen_sources_unchanged",len(frozen)==51 and source_before==source_after==frozen)
    add("active_suite_protocol_unchanged",sha(FROZEN_PROTOCOL)==protocol_hash)
    book=Path('E:/rfly-books/Rfly-Stable-Learning-Book/ch2/chapter2.tex')
    add("book_source_matches_preparation",sha(book)==original['source']['sha256'])
    output.mkdir(parents=True)
    for index,record in enumerate(episodes):
        (output/f'episode_{index+1:02d}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=json_ready,allow_nan=False)+'\n',encoding='utf-8')
    report={'schema':'dc-motor-isolated-engineering-v1','completed_utc':datetime.now(timezone.utc).isoformat(),
        'passed':all(r['passed'] for r in checks),'check_count':len(checks),'checks':checks,'evidence':evidence,
        'scope':'Engineering interface/physics/gradient/termination checks, not performance benchmark or learning acceptance.',
        'registered':False,'formal_training_started':False,'optimizer_instances':0,'optimization_updates':0,
        'deterministic_full_engineering_episodes':len(episodes),'total_full_episode_steps':sum(r['physical_steps'] for r in episodes),
        'protocol_parameters':{'dt_s':.01,'horizon_s':2.,'final_hold_s':1.,'current_tolerance_A':.02,'speed_tolerance_rad_s':.02,
            'R_physical_voltage_deviation':.01/25,'M_eigen_floor':.05,'M_trace':2.,'observation':'full_state_new_assumption'},
        'sources':{'book':{'path':str(book),'sha256':sha(book)},'frozen_suite_protocol':{'path':str(FROZEN_PROTOCOL),'sha256':protocol_hash},
            'local':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))},'frozen_51':source_after},
        'dependencies':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
        'negative_findings':['Arbitrary SPD M is not a stability guarantee.','Nominal motor is already Hurwitz; retain fixed equilibrium feedforward baseline.',
            'Saturation has zero/piecewise gradients.','Success is sampled final-window hold, not intersample certification.',
            'Existing FullHorizonTask rejects this task; no registration or wrapper bypass.',
            'Generic rollout_metric overwrites arrival time and detaches M; PPO does not backpropagate through the physical rollout.']}
    (output/'ENGINEERING_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=json_ready,allow_nan=False)+'\n',encoding='utf-8')
    manifest=[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]
    (output/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),'failed':[r for r in checks if not r['passed']],
        'output':str(output),'full_engineering_episodes':len(episodes),'optimization_updates':0},default=json_ready))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
