"""Existing physical applications with new, explicitly frozen overnight protocols."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import random
import time

from .queue import PROJECT, read, sha, write


def observers(config, output, parent_stop=None):
    """Use the repaired v4 observer backend; no legacy-success semantics."""
    from extensions.reproducible_training_v4.runtime import configure
    from extensions.reproducible_training_v4.runner import run_suite
    configure()
    output = Path(output).resolve()
    if config.get("schema") != "unified_training_v4" or any(j["backend"] != "observer_model_gradient" for j in config["jobs"]):
        raise ValueError("Observer step requires a v4 observer-only suite")
    if parent_stop and Path(parent_stop).exists():
        raise InterruptedError("Queue STOP")
    cfg_file = output.with_name(output.name + ".config.json")
    write(cfg_file, config)
    final = run_suite(config, output, project=PROJECT, config_path=cfg_file, telemetry=False)
    results = read(output / "RESULTS.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "overnight_figures"; figures.mkdir()
    for jid, job in results["jobs"].items():
        rows = [stage for stage in job.get("stages", []) if stage["stage"].startswith("validate_epoch_")]
        figure, axis = plt.subplots(figsize=(7, 3.5))
        points = [(int(r["stage"].rsplit("_", 1)[-1]), r["result"]["selection_score"][0]) for r in rows]
        if points:
            axis.plot(*zip(*points), "o-")
        axis.set(xlabel="Epoch", ylabel="Validation primary score", title=jid)
        axis.grid(alpha=.25); figure.tight_layout()
        figure.savefig(figures/(jid+".png"), dpi=300); figure.savefig(figures/(jid+".pdf")); plt.close(figure)
    record = dict(schema="book_overnight_observer_application_v1", status=final["status"],
        actual_training_seeds=sorted({j["config"]["seed"] for j in config["jobs"]}),
        mapped_task="E22", E23="unknown-scene self-motion not implemented by the cooperative-target observer",
        E24="known-input observer tests do not implement observer-controller coupled feedback",
        environment="stabilization_gym.observation: physical truth, sensors and data splits",
        algorithm="v4 CorrectedObserverBackend: estimated-state M, full-sequence gradient learning",
        runner="v4 source/runtime-locked collection, training, validation, global freeze and independent tests",
        raw_results=str(output / "RESULTS.json"), figures=[str(p) for p in figures.glob("*.png")],
        task_acceptance="synthetic known-target keypoint application; detector and hardware tests remain distinct")
    write(output / "OVERNIGHT_APPLICATION.json", record)
    write(output / "FIGURES_COMPLETE.json", dict(figures={str(p):sha(p) for p in figures.glob("*.png")},
                                                  scientific_results=sha(output / "RESULTS.json")))
    return record


def _timed_training_module():
    """Load the preserved PPO mathematics against the explicit repaired v2 APIs."""
    path = PROJECT / "third_party/so101_timeaware_v1/so101_timeaware/training.py"
    name = "third_party.so101_timeaware_v2.so101_timeaware.overnight_preserved_training"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _metric_evidence(policy, poses):
    import numpy as np
    import torch
    q = torch.tensor(np.asarray(poses), dtype=torch.float32)
    with torch.no_grad():
        error = policy.model.coordinates(q)
        condition = policy.cond.expand(len(q), -1)
        _, matrices = policy.actor.conditioned(error, condition)
        reported = policy.actor.value(error, condition)
    e, m, v = error.numpy(), matrices.numpy(), reported.numpy()
    actual = np.einsum("ni,nij,nj->n", e.astype(float), m.astype(float), e.astype(float))
    if not np.allclose(actual, v, atol=2e-5, rtol=1e-5):
        raise AssertionError("Independent full state-dependent M quadratic form differs")
    return e, m, v, float(np.max(abs(actual-v)))


def e20(config, output):
    """Book A/B/C contact metrics: fresh C training, then paired new starts."""
    import numpy as np
    import torch
    from third_party.so101_timeaware_v2 import so101_multishape as shapes
    from third_party.so101_timeaware_v2.so101_timeaware.policy import load_actor, CandidateModel, TimeCritic, Selector
    from third_party.so101_timeaware_v2.so101_timeaware.environment import TimedResponseEnv, reached
    from third_party.so101_timeaware_v2.so101_timeaware.timing import TimeBank, validate_arm_timing
    from extensions.reproducible_training_v4.so101 import rollout, Config
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if config.get("schema") != "book_e20_overnight_v1":
        raise ValueError("Expected book_e20_overnight_v1")
    bank = TimeBank(config["time_bank"])
    validate_arm_timing(bank.config)
    if bank.config.encoder_omega_multiplier != 2.:
        raise ValueError("This book A/B/C protocol preserves the original omega scaling")
    training, training_source = _timed_training_module()
    output.mkdir(parents=True)
    root_deadline = time.time()+config.get("maximum_seconds", 7200)
    if config.get("deadline_unix") is not None:
        root_deadline = min(root_deadline, config["deadline_unix"])
    source_files = list((PROJECT / "extensions/book_overnight_v1").glob("*.py"))
    source_files += list((PROJECT / "third_party/so101_timeaware_v2").rglob("*.py"))
    source_files += list((PROJECT / "third_party/push_original/dl_push").rglob("*.py"))
    source_files += [p for p in (PROJECT / "third_party/so101_timeaware_v1/so101_vendor").rglob("*")
                     if p.is_file() and p.suffix in (".py", ".xml", ".png")]
    source_files += [training_source, Path(config["time_bank"]), Path(config["initial_checkpoint"]),
                     PROJECT / "extensions/reproducible_training_v4/so101.py"]
    sources = {str(p): sha(p) for p in source_files}
    protocol = dict(schema="book_e20_overnight_protocol_v1", config=config, sources=sources,
        started_unix=time.time(), deadline_unix=root_deadline,
        environment="v2 timed reduced response model for training; actuated MuJoCo with v4 physical-substep hold for deployment",
        algorithm="unchanged book PPO loss/GAE mathematics; Full-M-induced candidate control; fresh critic/optimizer and seed",
        runner="initialize, fixed 300 updates, freeze last checkpoint, evaluate independent paired starts",
        selection="fixed final update budget; validation or test results never select C",
        sampling_seed=config["seed"], test_seed=config["test_seed"],
        source_scope="book calibration and original A/B weights; new C training; simulation only")
    write(output / "protocol.json", protocol)
    (output / "trajectories").mkdir()
    (output / "episodes").mkdir()

    def guard():
        if (output / "STOP").exists() or time.time() >= root_deadline:
            raise InterruptedError("E20 STOP/deadline")

    torch.set_num_threads(1)
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    actor, original = load_actor(config["initial_checkpoint"], "cpu")
    baseline = deepcopy(actor)
    model = CandidateModel(bank, "cpu")
    critic = TimeCritic(actor.cfg.hidden)
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=config.get("learning_rate", 1e-4))
    envs = [TimedResponseEnv(bank, config["train_shapes"], config["seed"]+i) for i in range(config.get("num_envs", 32))]
    history = []
    start_hash = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    final = dict(status="running")
    rows = []
    try:
        for step in range(config.get("updates", 300)):
            guard()
            samples, episodes = training.collect(actor, critic, model, envs, config.get("horizon", 64))
            statistics = training.update(actor, critic, optimizer, model, samples, config.get("epochs", 2), config.get("minibatch", 128))
            statistics.update(update=step+1, transitions=len(envs)*config.get("horizon", 64),
                elapsed_seconds=float(samples["dt"].sum()), completed_episodes=len(episodes),
                success_rate=float(np.mean([r["success"] for r in episodes])) if episodes else None)
            history.append(statistics)
            with (output / "TRAINING.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(statistics, allow_nan=False)+"\n")
            with (output / "TRAINING_EPISODES.jsonl").open("a", encoding="utf-8") as stream:
                for episode in episodes:
                    stream.write(json.dumps(dict(update=step+1, **episode), allow_nan=False)+"\n")
            if (step+1) % 10 == 0:
                print(json.dumps(dict(event="e20_training", **statistics)), flush=True)
        checkpoint = output / "C_final.pt"
        torch.save(dict(schema="so101_timeaware_ppo_v1", actor=actor.state_dict(), cfg=asdict(actor.cfg),
            critic=critic.state_dict(), optimizer=optimizer.state_dict(), update=len(history),
            time_bank_sha256=bank.sha256, seed=config["seed"]), checkpoint)
        write(output / "FROZEN.json", dict(checkpoint=str(checkpoint), sha256=sha(checkpoint),
            original_checkpoint_sha256=sha(config["initial_checkpoint"]), time_bank_sha256=bank.sha256,
            updates=len(history), selection="fixed final update; no test access",
            maximum_actor_parameter_change=max(float((v-start_hash[k]).abs().max()) for k,v in actor.state_dict().items())))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        training_figures = output / "figures"; training_figures.mkdir()
        figure, axis = plt.subplots(figsize=(8, 3.5))
        for key in ("loss", "ppo", "critic"):
            axis.plot([r["update"] for r in history], [r[key] for r in history], label=key)
        axis.set(xlabel="C training update", title="All recorded C updates"); axis.grid(alpha=.25); axis.legend()
        figure.tight_layout(); figure.savefig(training_figures / "C_training_before_test.png", dpi=300)
        figure.savefig(training_figures / "C_training_before_test.pdf"); plt.close(figure)

        class IdentityActor(torch.nn.Module):
            def __init__(self, reference):
                super().__init__(); self.encoder = deepcopy(reference.encoder); self.cfg = reference.cfg
            def conditioned(self, error, condition):
                identity = torch.eye(error.shape[-1], dtype=error.dtype, device=error.device).expand(*error.shape[:-1], error.shape[-1], error.shape[-1])
                return identity, identity
            def value(self, error, condition):
                return error.square().sum(-1)

        variants = [("fixed_M_time", IdentityActor(baseline), "B"), ("A_original", baseline, "A"),
                    ("B_frozen_time", baseline, "B"), ("C_new_time", actor, "C")]
        for _, network, _ in variants:
            network.eval()
            for parameter in network.parameters():
                parameter.requires_grad_(False)
        rng = np.random.default_rng(config["test_seed"])
        starts = [np.r_[rng.uniform(-.012, .012, 2), rng.uniform(-.35, .35)] for _ in range(config.get("numeric_repeats", 16))]
        write(output / "TEST_STARTS.json", dict(seed=config["test_seed"], starts=[q.tolist() for q in starts],
            arm_uses_prefix=config.get("arm_repeats", 2), train_seed=config["seed"]))
        for shape in config["test_shapes"]:
            with shapes.install(shape):
                for label, network, variant in variants:
                    policy = Selector(bank, shape, network, variant)
                    for index, initial in enumerate(starts):
                        guard()
                        env = TimedResponseEnv(bank, [shape], config["test_seed"]+index); env.reset(shape, initial)
                        poses, transitions = [env.q.copy()], []
                        while not env.done:
                            guard()
                            try:
                                if reached(env.spec, env.q):
                                    action = 0
                                else:
                                    mode, travel, _, recovery = policy.choose(env.q, None if env.mode < 0 else env.mode, True, env.pusher)
                                    action = mode*len(bank.config.travels)+int(np.argmin(abs(np.asarray(bank.config.travels)-travel)))
                                _, reward, _, info = env.step(action)
                                transitions.append(dict(action=action, reward=reward, **info)); poses.append(env.q.copy())
                            except RuntimeError as error:
                                env.done = True; env.reason = str(error)
                        e, matrices, values, residual = _metric_evidence(policy, poses)
                        path = output / "trajectories" / f"numeric_{shape}_{label}_{index:03d}.npz"
                        np.savez_compressed(path, pose=np.asarray(poses), error=e, M=matrices, V=values,
                            dt=np.asarray([r["dt"] for r in transitions]))
                        item = dict(backend="numeric", shape=shape, variant=label, test_index=index,
                            initial=initial.tolist(), success=env.reason=="success", reason=env.reason,
                            seconds=env.elapsed, actions=env.actions, repositions=env.repositions,
                            transfer_seconds=env.total_transfer, trajectory=str(path), trajectory_sha256=sha(path),
                            V_recompute_max_error=residual, transitions=transitions)
                        write(output / "episodes" / (path.stem+".json"), item); rows.append(item)
                    for index, initial in enumerate(starts[:config.get("arm_repeats", 2)]):
                        guard()
                        path = output / "trajectories" / f"arm_{shape}_{label}_{index:03d}.npz"
                        cfg = Config(plant="arm", speed=bank.config.transit_speed,
                            max_time=bank.config.max_seconds, max_actions=bank.config.max_actions)
                        item = rollout(cfg, initial, config["test_seed"]+index, policy=policy, telemetry=path)
                        # Telemetry contains every controller observation and full M-dependent V.
                        with np.load(path, allow_pickle=False) as archive:
                            _, _, actual_v, residual = _metric_evidence(policy, archive["pose"])
                            if not np.allclose(actual_v, archive["V"], atol=2e-5, rtol=1e-5):
                                raise AssertionError("MuJoCo telemetry V differs from frozen policy")
                        item.update(backend="arm", shape=shape, variant=label, test_index=index,
                            seconds=item["sim_time_s"], trajectory=str(path), trajectory_sha256=sha(path), V_recompute_max_error=residual)
                        write(output / "episodes" / (path.stem+".json"), item); rows.append(item)
                        print(json.dumps(dict(event="e20_arm", shape=shape, variant=label, index=index,
                                              success=item["success"], reason=item["reason"])), flush=True)
        for path, expected in sources.items():
            if sha(path) != expected:
                raise RuntimeError("Frozen E20 source changed: " + path)
        summary = _e20_report(output, rows, history)
        final.update(status="complete", updates=len(history), evaluated_episodes=len(rows), summary=summary)
    finally:
        final["finished_unix"] = time.time()
        if final["status"] == "running":
            final["status"] = "incomplete"
        write(output / "FINAL_STATE.json", final)
    return final


def _e20_report(output, rows, history):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {}
    for backend, variant in sorted({(r["backend"], r["variant"]) for r in rows}):
        group = [r for r in rows if (r["backend"], r["variant"]) == (backend, variant)]
        groups[backend+"/"+variant] = dict(episodes=len(group), successes=sum(r["success"] for r in group),
            success_rate=float(np.mean([r["success"] for r in group])), mean_seconds=float(np.mean([r["seconds"] for r in group])),
            failures={reason:sum(r["reason"]==reason for r in group) for reason in sorted({r["reason"] for r in group if not r["success"]})})
    paired = []
    for backend in ("numeric", "arm"):
        selected = {(r["shape"], r["test_index"]):r for r in rows if r["backend"]==backend and r["variant"]=="C_new_time"}
        for label in ("fixed_M_time", "A_original", "B_frozen_time"):
            baseline = {(r["shape"], r["test_index"]):r for r in rows if r["backend"]==backend and r["variant"]==label}
            common = [k for k in selected if selected[k]["success"] and baseline[k]["success"]]
            paired.append(dict(backend=backend, baseline=label, gained=sum(selected[k]["success"] and not baseline[k]["success"] for k in selected),
                lost=sum(baseline[k]["success"] and not selected[k]["success"] for k in selected), common_successes=len(common),
                C_minus_baseline_seconds=float(np.mean([selected[k]["seconds"]-baseline[k]["seconds"] for k in common])) if common else None))
    result = dict(schema="book_e20_results_v1", groups=groups, paired=paired, episodes=rows,
        physical_success="continuous physical-substep hold, collision-free; separately from numerical response model",
        scope="new single-seed C training and independent starts on previously calibrated shapes; no real robot")
    write(output / "RESULTS.json", result)
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("loss", "ppo", "critic"):
        axes[0].plot([r["update"] for r in history], [r[key] for r in history], label=key)
    axes[0].set(xlabel="C training update", title="Fresh-seed training objectives"); axes[0].legend()
    keys = list(groups); axes[1].bar(np.arange(len(keys)), [groups[k]["success_rate"] for k in keys])
    axes[1].set(xticks=np.arange(len(keys)), xticklabels=keys, ylabel="Success rate", ylim=(0, 1), title="Every completed frozen comparison")
    axes[1].tick_params(axis="x", labelrotation=60)
    figure.tight_layout(); figure.savefig(figures / "training_and_success.png", dpi=300)
    figure.savefig(figures / "training_and_success.pdf"); plt.close(figure)
    first = [r for r in rows if r["shape"] == rows[0]["shape"] and r["test_index"] == 0 and r["backend"]=="arm"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row in first:
        with np.load(row["trajectory"], allow_pickle=False) as trace:
            axes[0].plot(trace["time"], trace["error"], label=row["variant"])
            axes[1].plot(trace["time"], trace["V"], label=row["variant"])
    for axis in axes:
        axis.set_xlabel("Full execution seconds"); axis.grid(alpha=.25); axis.legend(fontsize=7)
    axes[0].set_ylabel("Physical error"); axes[1].set_ylabel("Full V with current-state M")
    figure.tight_layout(); figure.savefig(figures / "first_declared_arm_trajectory.png", dpi=300)
    figure.savefig(figures / "first_declared_arm_trajectory.pdf"); plt.close(figure)
    lines = ["# E20 Push-T书稿机制：新种子训练与配对评价", "", "A/B保留同一原始Full-M；C从相同初始化重新训练。测试只在固定训练预算结束后执行。", "",
        "|物理层/方法|成功/总数|平均完整回合秒数|", "|---|---:|---:|"]
    lines += [f"|{key}|{g['successes']}/{g['episodes']}|{g['mean_seconds']:.3f}|" for key,g in groups.items()]
    lines += ["", "数值响应与MuJoCo关节级执行分别统计。MuJoCo保持在物理子步检查；既有形状标定成本和历史权重来源见protocol.json。新增结果属于仿真验证。", "",
              "![训练与成功率](figures/training_and_success.png)", "", "![首个预声明物理起点](figures/first_declared_arm_trajectory.png)"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write(output / "FIGURES_COMPLETE.json", dict(figures={str(p):sha(p) for p in figures.glob("*.png")},
                                                  scientific_results=sha(output / "RESULTS.json")))
    return groups


def e09(output, seed=41):
    """Independent Monte Carlo noise/sampling extension of the scalar disk model."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    levels, periods = [0., .0001, .0005, .001], [.05, .1, .2]
    count, horizon, gain = 2048, 240, 1.
    protocol = dict(schema="book_e09_noise_extension_v1", seed=seed, noise_std=levels, sampling_periods=periods,
        trajectory_count=count, horizon=horizon, feedback_gain=gain, gain_distribution="uniform[.8,1.2]",
        additive_distribution="uniform[-sqrt(3)*sigma,sqrt(3)*sigma]",
        scope="unconstrained scalar error dynamics x+=(1-dt*kappa*K)x+w; one-sided disk domain exits separately counted",
        interpretation="Persistent additive noise has a nonzero mean-square floor; trajectories leaving the one-sided domain are not certified disk trajectories",
        source_file_sha256=sha(__file__), analytical_factor="1-2*dt*K+(76/75)*dt^2*K^2",
        moments="E[x_k^2]=q^k*x0^2+sigma^2*(1-q^k)/(1-q)")
    write(output / "protocol.json", protocol)
    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for pi, period in enumerate(periods):
        for ni, sigma in enumerate(levels):
            rng = np.random.default_rng(np.random.SeedSequence([seed, pi, ni, 9029]))
            states = np.empty((horizon+1, count)); states[0] = -.05
            kappas = rng.uniform(.8, 1.2, (horizon, count))
            noise = rng.uniform(-np.sqrt(3)*sigma, np.sqrt(3)*sigma, (horizon, count))
            for step in range(horizon):
                states[step+1] = (1-period*kappas[step]*gain)*states[step]+noise[step]
            actual = states**2
            mean = actual.mean(axis=1)
            q = 1-2*period*gain+(76/75)*(period*gain)**2
            theory = q**np.arange(horizon+1)*.05**2+sigma**2*(1-q**np.arange(horizon+1))/(1-q)
            path = output / f"dt{period:g}_sigma{sigma:g}.npz"
            np.savez_compressed(path, states=states, gains=kappas, noise=noise, mean_square=mean, theoretical_mean_square=theory)
            stderr = actual.std(axis=1, ddof=1)/np.sqrt(count)
            row = dict(dt=period, additive_std=sigma, q=q, theoretical_floor=sigma**2/(1-q),
                terminal_MC_mean_square=float(mean[-1]), terminal_standard_error=float(stderr[-1]),
                one_sided_domain_exit_fraction=float(np.mean(np.any((states<-.05)|(states>0), axis=0))),
                formula_max_difference=float(abs(mean-theory).max()),
                recurrence_max_residual=float(abs(states[1:]-(1-period*kappas*gain)*states[:-1]-noise).max()),
                trajectory=str(path), sha256=sha(path))
            rows.append(row)
            axes[pi].semilogy(np.arange(horizon+1)*period, np.maximum(mean, 1e-18), label=f"sigma={sigma:g} MC")
            axes[pi].semilogy(np.arange(horizon+1)*period, np.maximum(theory, 1e-18), "--", linewidth=.8)
        axes[pi].set(xlabel="Seconds", ylabel="Mean-square error", title=f"dt={period:g}"); axes[pi].grid(alpha=.25); axes[pi].legend(fontsize=6)
    figure.tight_layout(); figure.savefig(output / "noise_and_sampling_MC.png", dpi=300)
    figure.savefig(output / "noise_and_sampling_MC.pdf"); plt.close(figure)
    result = dict(status="complete", rows=rows, trajectories=count*len(levels)*len(periods),
        all_recurrences_passed=all(r["recurrence_max_residual"]<1e-12 for r in rows))
    write(output / "RESULTS.json", result)
    lines = ["# E09 乘性与持续加性噪声独立轨迹", "", "每个条件使用2048条真实采样递推轨迹。虚线为解析均方值，实线为蒙特卡洛均值。", "",
        "持续加性噪声的结论是非零均方平台；另列越出一维单侧圆盘运行域的比例。自由线性递推不能自动作为接触受限圆盘的可行性证据。", "",
        "![噪声与采样周期](noise_and_sampling_MC.png)"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write(output / "FIGURES_COMPLETE.json", dict(figures={str(output / "noise_and_sampling_MC.png"):sha(output / "noise_and_sampling_MC.png")}))
    return result
