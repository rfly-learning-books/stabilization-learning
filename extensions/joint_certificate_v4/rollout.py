"""GPU collection/evaluation. Physical truth never enters the online policy."""
from collections import Counter
from pathlib import Path
import math
import time
import numpy as np
import torch
from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
from stabilization_gym.tasks_v2.visual import VisualObservation, quadratic_value
from stabilization_learning.joint_v3.inference import FrozenRGBPolicy
from stabilization_learning.joint_v2.runtime import atomic_json, file_sha256
from extensions.joint_certificate_v2.rgb import _npz

DIFFICULTIES = ("near", "small", "medium", "large")


def stratum(scene_id, goal_id, difficulty):
    return f"{scene_id}/g{goal_id}/{difficulty}"


def sample_offset(cfg, scene, goal, difficulty, rng):
    from lac_vs import legacy
    from lac_vs.environment import sample_start, valid_start
    frame = legacy.module(cfg, "room_env").ReferenceFrame(np.asarray(scene["refs"][goal]["goal_matrix"]))
    if difficulty != "near":
        return sample_start(cfg, scene, frame, rng, difficulty)
    # Both sides of the unchanged success region; labels are sampling-only.
    for _ in range(2000):
        direction = rng.normal(size=3); direction[2] *= .3
        direction /= np.linalg.norm(direction)
        radius = cfg["collection"]["position_tolerance_m"] * rng.uniform(.5, 2.)
        yaw = math.radians(cfg["collection"]["yaw_tolerance_deg"]) * rng.uniform(.5, 2.) * rng.choice([-1, 1])
        offset = np.r_[direction*radius, yaw]
        if valid_start(cfg, scene, frame, offset):
            return offset
    raise RuntimeError("Near-goal start rejection exhausted: " + scene["id"])


def observation(geometry, public):
    return VisualObservation(*geometry.observe(public))


def random_actions(rng, count, device):
    xyz = rng.normal(size=(count, 3))
    xyz *= rng.uniform(0., 1., (count, 1))**(1/3) / np.linalg.norm(xyz, axis=1, keepdims=True)
    return torch.as_tensor(np.c_[xyz, rng.uniform(-1., 1., count)], device=device, dtype=torch.float32)


def make_env(cfg, scenes, config, project, plant="P0", max_steps=None):
    return GPUParallelRGBEnv(cfg, scenes, plant=plant, max_steps=max_steps or config.evaluation_max_steps,
                             device=config.device, cache_dir=project/".cache/rgb_gpu_warp",
                             samples_per_axis=config.samples_per_axis)


def collect(cfg, scenes, learner, geometry, replay, rng, config, project, store, *, round_id, quota):
    begun = time.perf_counter()
    original_count = sum(r["transitions"] for r in replay.rows)
    coverage = {}
    component_seconds = dict(environment_setup=0., rollout=0., archive=0.)
    store.event("collection_started", round_id=round_id, quota_per_stratum=quota, scenes=len(scenes))
    for scene in scenes:
        store.guard()
        keys = [(g, d) for g in range(len(scene["refs"])) for d in DIFFICULTIES]
        counts = {key: 0 for key in keys}
        setup_start = time.perf_counter()
        env = make_env(cfg, [scene]*config.parallel_envs, config, project, max_steps=config.collection_horizon)
        torch.cuda.synchronize()
        component_seconds["environment_setup"] += time.perf_counter()-setup_start
        try:
            attempts = 0
            while min(counts.values()) < quota:
                store.guard(); attempts += 1
                if attempts > 2000:
                    raise RuntimeError("Collection quota not reached; preserve failed episodes without shrinking scope")
                remaining = [key for key in keys if counts[key] < quota]
                rng.shuffle(remaining)
                lanes = [remaining[i % len(remaining)] for i in range(config.parallel_envs)]
                lane_counts = Counter(lanes)
                horizon = min(config.collection_horizon, max(1, min(math.ceil((quota-counts[k])/v) for k, v in lane_counts.items())))
                env.max_steps = horizon
                rollout_start = time.perf_counter()
                offsets = np.stack([sample_offset(cfg, scene, g, d, rng) for g, d in lanes])
                goals = [g for g, _ in lanes]
                public, info = env.reset(seed=int(rng.integers(2**31-1)), options=dict(offsets=offsets, goal_ids=goals))
                exploratory = np.ones(len(lanes), dtype=bool) if round_id == 0 else rng.random(len(lanes)) < config.random_episode_fraction_after_bootstrap
                random_mask = torch.as_tensor(exploratory, device=config.device)
                session = FrozenRGBPolicy(learner)
                goal_keys = [(scene["id"], g) for g in goals]
                trace = {k: [] for k in ("rgb", "rotation", "offset", "action", "applied_action", "previous_action", "dt", "controller", "terminated", "truncated")}
                trace["rgb"].append(public["rgb"])
                trace["rotation"].append(public["rotation"])
                trace["offset"].append(info["supervision"]["offset"])
                alive = torch.ones(len(lanes), device=config.device, dtype=torch.bool)
                lengths = torch.zeros(len(lanes), device=config.device, dtype=torch.int64)
                last = {k: torch.zeros_like(alive) for k in ("success", "collision", "outside")}
                try:
                    for step in range(horizon):
                        store.guard()
                        if not bool(alive.any()):
                            break
                        if step % config.random_action_hold_steps == 0:
                            exploratory_action = random_actions(rng, len(lanes), config.device)
                        if exploratory.all():
                            action = exploratory_action
                        else:
                            out = session(observation(geometry, public), goal_keys=goal_keys)
                            action = torch.where(random_mask[:, None], exploratory_action, out.action)
                        following, _, terminal, truncated, info = env.step(action)
                        if not bool(info["transition_mask"][alive].all()):
                            raise RuntimeError("Reset call presented as an active training transition")
                        truth = info["supervision"]
                        trace["action"].append(action.detach())
                        trace["applied_action"].append((truth["actual_velocity"]/env.scale).float())
                        trace["previous_action"].append(truth["previous_action"].float())
                        trace["dt"].append(info["dt"].float())
                        trace["controller"].append(~random_mask)
                        trace["terminated"].append(terminal); trace["truncated"].append(truncated)
                        trace["rgb"].append(following["rgb"])
                        trace["rotation"].append(following["rotation"])
                        trace["offset"].append(truth["next_offset"])
                        lengths += alive.long()
                        ending = alive & (terminal | truncated)
                        for name in last:
                            last[name] |= ending & info[name]
                        alive &= ~(terminal | truncated)
                        public = following
                except BaseException as error:
                    partial = store.root/"partial_collection"/f"round_{round_id}_{scene['id']}_{attempts}.npz"
                    _npz(partial, {k: torch.stack(v).cpu().numpy() for k, v in trace.items() if v})
                    atomic_json(partial.with_suffix(".json"), dict(error=repr(error), eligible_replay=False, restart_allowed=False))
                    raise
                torch.cuda.synchronize()
                component_seconds["rollout"] += time.perf_counter()-rollout_start
                archive_start = time.perf_counter()
                host = {k: torch.stack(v).cpu().numpy() for k, v in trace.items()}
                goal_images = public["goal_rgb"].cpu().numpy()
                lengths = lengths.cpu().tolist()
                last = {k: v.cpu().tolist() for k, v in last.items()}
                for i, ((g, d), n) in enumerate(zip(lanes, lengths)):
                    arrays = {k: v[:n+(k in {"rgb", "rotation", "offset"}), i] for k, v in host.items()}
                    arrays["goal_rgb"] = goal_images[i]
                    reason = "success" if last["success"][i] else "collision" if last["collision"][i] else "outside" if last["outside"][i] else "collection_horizon"
                    row = replay.add(arrays, dict(split="train", scene_id=scene["id"], goal_id=g, difficulty=d,
                                     stratum=stratum(scene["id"], g, d), round_id=round_id,
                                     initial_offset=offsets[i].tolist(), behavior="random_bounded" if exploratory[i] else "current_controller",
                                     behavior_version=learner.version, reason=reason, complete=True))
                    counts[(g, d)] += n
                    store.event("replay_episode_committed", episode=row["episode"], transitions=n,
                                stratum=row["stratum"], round_id=round_id, reason=reason, behavior=row["behavior"])
                del host, trace
                component_seconds["archive"] += time.perf_counter()-archive_start
            coverage.update({stratum(scene["id"], g, d): n for (g, d), n in counts.items()})
        finally:
            env.close()
    added = sum(r["transitions"] for r in replay.rows)-original_count
    result = dict(round_id=round_id, quota_per_stratum=quota, coverage=coverage, added_transitions=added,
                  elapsed_s=time.perf_counter()-begun, component_seconds=component_seconds,
                  all_quotas_met=all(n >= quota for n in coverage.values()))
    atomic_json(store.root/f"COLLECTION_{round_id:02d}.json", result)
    store.event("collection_complete", **result)
    return result


def diagnostics(arrays, config):
    v, physical = arrays["V"].astype(float), arrays["physical_error"].astype(float)
    tol = 1e-7 + 1e-4*np.abs(v[:-1])
    cal_tol = 1e-7 + 1e-4*physical
    predicted = np.einsum("tij,tj->ti", arrays["J"], arrays["applied_action"]) * arrays["dt"][:, None]
    actual = np.diff(arrays["error"], axis=0)
    norm_p, norm_a = np.linalg.norm(predicted, axis=1), np.linalg.norm(actual, axis=1)
    valid = (norm_p > 1e-7) & (norm_a > 1e-7)
    cos = (predicted*actual).sum(-1)[valid] / (norm_p*norm_a)[valid]
    return dict(actual_V_increase_fraction=float(np.mean(np.diff(v) > tol)),
                actual_decay_violation_fraction=float(np.mean(np.diff(v)+config.decrease_per_second*arrays["dt"]*physical[:-1] > tol)),
                certificate_comparable_fraction=float(np.mean((v >= config.value_lower*physical-cal_tol)&(v <= config.value_upper*physical+cal_tol))),
                value_lower_violation_fraction=float(np.mean(v < config.value_lower*physical-cal_tol)),
                value_upper_violation_fraction=float(np.mean(v > config.value_upper*physical+cal_tol)),
                J_effective_steps=int(valid.sum()), J_cosine_mean=float(cos.mean()) if len(cos) else None,
                J_opposite_fraction=float(np.mean(cos < 0)) if len(cos) else None,
                final_error_ratio=float(physical[-1]/max(physical[0], 1e-12)))


def evaluate(cfg, scenes_by_id, cases, learner, geometry, config, project, store, *, plant="P0", method="selected", output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = []
    for start in range(0, len(cases), config.parallel_envs):
        store.guard()
        group = cases[start:start+config.parallel_envs]
        env = make_env(cfg, [scenes_by_id[c["scene_id"]] for c in group], config, project, plant)
        begun = time.perf_counter()
        trace = {k: [] for k in ("error", "V", "J", "action", "applied_action", "offset", "physical_error", "dt", "physical_cost", "input_cost", "integrated_error", "previous_action", "actual_velocity")}
        try:
            public, info = env.reset(options=dict(offsets=[c["offset"] for c in group], goal_ids=[c["goal_id"] for c in group]))
            session = FrozenRGBPolicy(learner, identity=method == "identity", max_goals=128)
            goal_keys = [(c["scene_id"], c["goal_id"]) for c in group]
            alive = torch.ones(len(group), device=config.device, dtype=torch.bool)
            lengths = torch.zeros(len(group), device=config.device, dtype=torch.int64)
            last = {k: torch.zeros_like(alive) for k in ("success", "collision", "outside")}
            trace["offset"].append(info["supervision"]["offset"])
            trace["physical_error"].append(info["supervision"]["physical_error"])
            wall = []
            for step in range(config.evaluation_max_steps):
                store.guard()
                if not bool(alive.any()):
                    break
                t = time.perf_counter()
                out = session(observation(geometry, public), goal_keys=goal_keys)
                torch.cuda.synchronize()
                wall.append(time.perf_counter()-t)
                if step == 0:
                    matrix = out.metric.clone()
                    trace["error"].append(out.error)
                    trace["V"].append(out.value)
                elif not torch.equal(out.metric, matrix):
                    raise RuntimeError("Goal-only M changed inside a frozen evaluation")
                following, _, terminal, truncated, info = env.step(out.action)
                if not bool(info["transition_mask"][alive].all()):
                    raise RuntimeError("Autorestart contaminated evaluation")
                truth = info["supervision"]
                with torch.no_grad():
                    successor_error = learner.features.encode(geometry.encode_images(following["rgb"])) - out.goal
                    successor_value = quadratic_value(successor_error, out.metric)
                trace["error"].append(successor_error)
                trace["V"].append(successor_value)
                trace["J"].append(out.interaction)
                trace["action"].append(out.action)
                trace["applied_action"].append((truth["actual_velocity"]/env.scale).float())
                trace["offset"].append(truth["next_offset"])
                trace["physical_error"].append(truth["physical_error"])
                trace["dt"].append(info["dt"])
                for key, source in (("physical_cost", "physical_cost"), ("input_cost", "normalized_input_squared_integral"),
                                    ("integrated_error", "integrated_error"), ("previous_action", "previous_action"), ("actual_velocity", "actual_velocity")):
                    trace[key].append(truth[source])
                lengths += alive.long()
                ending = alive & (terminal | truncated)
                for name in last:
                    last[name] |= ending & info[name]
                alive &= ~(terminal | truncated)
                public = following
            session._check()
            host = {k: torch.stack(v).cpu().numpy() for k, v in trace.items()}
            matrices = matrix.cpu().numpy()
            last = {k: v.cpu().tolist() for k, v in last.items()}
            for i, (case, n) in enumerate(zip(group, lengths.cpu().tolist())):
                arrays = {k: v[:n+(k in {"error", "V", "offset", "physical_error"}), i] for k, v in host.items()}
                arrays["M"] = matrices[i]
                path = output / f"case_{start+i:04d}.npz"
                _npz(path, arrays)
                row = dict(**case, plant=plant, method=method, steps=n, version=learner.version,
                           success=last["success"][i], reason="success" if last["success"][i] else "collision" if last["collision"][i] else "outside" if last["outside"][i] else "timeout",
                           elapsed_s=float(arrays["dt"].sum()), integrated_error=float(arrays["integrated_error"].sum()),
                           input_squared_integral=float(arrays["input_cost"].sum()), physical_cost=float(arrays["physical_cost"].sum()),
                           final_physical_error=float(arrays["physical_error"][-1]),
                           action_variation=float(np.square(arrays["action"]-arrays["previous_action"]).sum()),
                           mean_policy_wall_s=float(np.mean(wall[:n])), trajectory=str(path), trajectory_sha256=file_sha256(path),
                           policy_wall_semantics="batch latency, not added to physical dt; no realtime claim",
                           **diagnostics(arrays, learner.config))
                rows.append(row)
                store.event("evaluation_episode", **row)
            store.event("evaluation_batch_complete", plant=plant, method=method, episodes=len(group),
                        transitions=sum(r["steps"] for r in rows[-len(group):]), wall_s=time.perf_counter()-begun,
                        output=str(output))
        except BaseException as error:
            if any(trace.values()):
                _npz(output/f"batch_{start:04d}_partial.npz", {k: torch.stack(v).cpu().numpy() for k, v in trace.items() if v})
            atomic_json(output/f"batch_{start:04d}_failure.json", dict(error=repr(error), eligible_result=False, restart_allowed=False))
            raise
        finally:
            env.close()
    atomic_json(output/"RESULTS.json", rows)
    return rows
