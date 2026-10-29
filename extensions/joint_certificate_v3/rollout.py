"""Synchronous RGB microbatches over independent standard physical envs.

All task timing/cost/termination remains per environment. GPU inference is
batched; CPU rendering/physical steps are sequential, not claimed parallel.
"""
from pathlib import Path
import sys
import time
import numpy as np
import torch

from stabilization_gym.tasks_v2.reconstructed_rgb import ReconstructedRGBEnv
from stabilization_gym.tasks_v2.visual import VisualObservation, quadratic_value
from stabilization_learning.joint_v3.inference import FrozenRGBPolicy
from stabilization_learning.joint_v2.runtime import file_sha256, atomic_json
from extensions.joint_certificate_v2.rgb import _npz


def _close_all(entries):
    errors = []
    for entry in entries:
        try:
            entry["env"].close()
        except Exception as error:
            errors.append(error)
    if errors and sys.exc_info()[0] is None:
        raise RuntimeError("Physical environment close failed after attempting all closures") from errors[0]


def _infer(session, geometry, entries):
    observation = VisualObservation(*geometry.observe_batch([e["public"] for e in entries]))
    out = session(observation, goal_keys=[e["goal_key"] for e in entries])
    return out, out.action.detach().cpu().numpy()


def collect_batch(cfg, scenes, learner, geometry, replay, rng, episode_id, max_steps,
                  guard, *, exploratory, store):
    if not scenes or max_steps < 1:
        raise ValueError("Declare nonempty scenes and positive rollout length")
    if any(scene["split"] != "train" for scene in scenes):
        raise ValueError("Only training scenes enter collection")
    session = FrozenRGBPolicy(learner)
    entries, rows = [], []

    def save(entry, complete):
        if entry["saved"] or not entry["records"]["action"]:
            return
        arrays = {k: np.asarray(v) for k, v in entry["records"].items()}
        arrays.update(goal_rgb=entry["initial"]["goal_rgb"], rotation=entry["initial"]["rotation"])
        reason = entry["info"]["reason"] if complete else "interrupted"
        row = replay.add(arrays, dict(entry["metadata"], reason=reason), complete=complete)
        entry["saved"] = True
        store.event("replay_episode" if complete else "partial_replay_preserved", **row)
        if complete:
            rows.append(row)

    try:
        for i, scene in enumerate(scenes):
            guard()
            goal = int(rng.integers(len(scene["refs"])))
            difficulty = ("small", "medium", "large")[int(rng.integers(3))]
            reset_seed, action_seed = map(int, rng.integers(2**31-1, size=2))
            env = ReconstructedRGBEnv(cfg, scene, plant="P0", max_steps=max_steps)
            entry = dict(env=env, saved=False, records={k: [] for k in
                         ("rgb", "offset", "action", "actual_normalized_action", "previous_action", "dt")})
            entries.append(entry)
            public, info = env.reset(seed=reset_seed, options=dict(goal_id=goal, difficulty=difficulty))
            entry.update(initial=public, public=public, info=info, done=False,
                         goal_key=FrozenRGBPolicy.goal_key(public["goal_rgb"]),
                         rng=np.random.default_rng(action_seed),
                         metadata=dict(episode_id=episode_id+i, scene_id=scene["id"], split="train",
                                       goal_id=goal, difficulty=difficulty, reset_seed=reset_seed,
                                       action_seed=action_seed, behavior_version=learner.version,
                                       policy="random_bounded" if exploratory else "current_controller"))
            entry["records"]["rgb"].append(public["rgb"])
            entry["records"]["offset"].append(info["supervision"]["offset"])
        for _ in range(max_steps):
            guard()
            active = [e for e in entries if not e["done"]]
            if not active:
                break
            session._check()
            if exploratory:
                actions = []
                for e in active:
                    a = e["rng"].uniform(-1, 1, 4)
                    actions.append(a/max(1., np.linalg.norm(a[:3]), abs(a[3])))
            else:
                _, actions = _infer(session, geometry, active)
            for e, action in zip(active, actions):
                guard()
                public = e["public"]
                following, _, terminated, truncated, info = e["env"].step(action)
                truth, records = info["supervision"], e["records"]
                records["action"].append(action.copy())
                records["actual_normalized_action"].append(truth["actual_velocity"]/np.array([.4,.4,.4,.35]))
                records["previous_action"].append(public["previous_action"])
                records["dt"].append(info["dt"])
                records["rgb"].append(following["rgb"])
                records["offset"].append(truth["next_offset"])
                e.update(public=following, info=info, done=bool(terminated or truncated))
        session._check()
        for e in entries:  # deterministic archive ordering, independent of finish order
            save(e, True)
        return rows
    except BaseException:
        for e in entries:
            if "initial" in e:
                save(e, e.get("done", False))
        raise
    finally:
        _close_all(entries)


def evaluate_batch(cfg, scenes_by_id, cases, learner, geometry, guard, *, plant="P0",
                   method="learned", max_steps=200, output=None, store=None, batch_size=4):
    if batch_size < 1 or max_steps < 1:
        raise ValueError("Positive batch size and horizon required")
    if method not in {"learned", "selected", "final", "identity", "random_initial"}:
        raise ValueError("Unknown frozen comparison method")
    rows = []
    for start in range(0, len(cases), batch_size):
        session = FrozenRGBPolicy(learner, identity=method == "identity")
        entries = []
        try:
            for i, case in enumerate(cases[start:start+batch_size]):
                guard()
                env = ReconstructedRGBEnv(cfg, scenes_by_id[case["scene_id"]], plant=plant, max_steps=max_steps)
                e = dict(env=env, case=case, number=start+i)
                entries.append(e)
                public, info = env.reset(seed=0, options={"goal_id":case["goal_id"], "offset":case["offset"]})
                trace = {k: [] for k in ("error", "V", "J", "action", "offset", "dt", "actual_velocity", "physical_error")}
                trace["offset"].append(info["supervision"]["offset"])
                trace["physical_error"].append(info["supervision"]["physical_error"])
                e.update(public=public, info=info, done=False, trace=trace, elapsed=0., integrated=0.,
                         effort=0., physical_cost=0., variation=0., previous=np.zeros(4), wall=[], amortized=[],
                         metric=None, goal_key=FrozenRGBPolicy.goal_key(public["goal_rgb"]))
            for _ in range(max_steps):
                guard()
                active = [e for e in entries if not e["done"]]
                if not active:
                    break
                begun = time.perf_counter()
                out, actions = _infer(session, geometry, active)
                latency = time.perf_counter() - begun
                for index, (e, action) in enumerate(zip(active, actions)):
                    guard()
                    trace = e["trace"]
                    e["wall"].append(latency)
                    e["amortized"].append(latency/len(active))
                    trace["error"].append(out.error[index].cpu().numpy())
                    trace["V"].append(float(out.value[index]))
                    trace["J"].append(out.interaction[index].cpu().numpy())
                    matrix = out.metric[index].cpu().numpy()
                    if e["metric"] is None:
                        e["metric"] = matrix.copy()
                        e["goal_feature"] = out.goal[index:index+1].clone()
                    elif not np.array_equal(matrix, e["metric"]):
                        raise RuntimeError("Goal metric changed during a frozen episode")
                    public, _, terminated, truncated, info = e["env"].step(action)
                    truth = info["supervision"]
                    e["elapsed"] += info["dt"]
                    e["integrated"] += truth["integrated_error"]
                    e["effort"] += truth["normalized_input_squared_integral"]
                    e["physical_cost"] += truth["physical_cost"]
                    e["variation"] += float(np.square(action-e["previous"]).sum())
                    e["previous"] = action.copy()
                    trace["action"].append(action.copy()); trace["dt"].append(info["dt"])
                    trace["actual_velocity"].append(truth["actual_velocity"])
                    trace["offset"].append(truth["next_offset"])
                    trace["physical_error"].append(truth["physical_error"])
                    e.update(public=public, info=info, done=bool(terminated or truncated))
            session._check()
            with torch.no_grad():
                following = learner.features.encode(geometry.encode_images([e["public"]["rgb"] for e in entries]))
                for i, e in enumerate(entries):
                    error = following[i:i+1] - e["goal_feature"]
                    matrix = torch.as_tensor(e["metric"], device=following.device)[None]
                    e["trace"]["error"].append(error[0].cpu().numpy())
                    e["trace"]["V"].append(float(quadratic_value(error, matrix)[0]))
            for e in entries:
                trace, info = e["trace"], e["info"]
                values, physical = np.asarray(trace["V"]), np.asarray(trace["physical_error"])
                tolerance = 1e-7 + 1e-4*np.abs(values[:-1])
                residual = np.diff(values) + learner.config.decrease_per_second*np.asarray(trace["dt"])*physical[:-1]
                physical_tolerance = 1e-7 + 1e-4*physical
                comparable = ((values >= learner.config.value_lower*physical-physical_tolerance)
                              & (values <= learner.config.value_upper*physical+physical_tolerance))
                row = dict(**e["case"], plant=plant, method=method, success=bool(info["success"]), reason=info["reason"],
                           steps=len(trace["action"]), elapsed_s=e["elapsed"], integrated_error=e["integrated"],
                           physical_cost=e["physical_cost"], input_squared_integral=e["effort"], action_variation=e["variation"],
                           mean_policy_wall_s=float(np.mean(e["wall"])),
                           mean_amortized_policy_wall_s=float(np.mean(e["amortized"])),
                           policy_wall_semantics="batch latency; amortized field divides by active envs; excludes env step",
                           final_physical_error=float(physical[-1]),
                           actual_V_increase_fraction=float(np.mean(np.diff(values)>tolerance)),
                           actual_decay_violation_fraction=float(np.mean(residual>tolerance)),
                           certificate_comparable_fraction=float(np.mean(comparable)), version=learner.version)
                if output is not None:
                    path = Path(output)/f"case_{e['number']:04d}.npz"
                    _npz(path, dict(**{k:np.asarray(v) for k,v in trace.items()}, M=e["metric"]))
                    row.update(trajectory=str(path), trajectory_sha256=file_sha256(path),
                               trajectory_scope="features,J,M,physical truth,actions; raw RGB replay is training-only")
                rows.append(row)
                if store is not None:
                    store.event("evaluation_episode", **row)
        except BaseException as error:
            # Interrupted evaluation is evidence, never an eligible result and
            # never a request to retry/overwrite a partly completed test group.
            if output is not None:
                for e in entries:
                    if "trace" not in e or not e["trace"]["action"]:
                        continue
                    path = Path(output)/f"case_{e['number']:04d}.npz"
                    if path.exists():
                        continue
                    partial = path.with_name(path.stem + "_partial.npz")
                    _npz(partial, dict(**{k:np.asarray(v) for k,v in e["trace"].items()}, M=e["metric"]))
                    atomic_json(partial.with_suffix(".json"), dict(case=e["case"], error=repr(error),
                                physical_complete=e["done"], eligible_result=False,
                                field_lengths={k:len(v) for k,v in e["trace"].items()},
                                raw_trace_sha256=file_sha256(partial), overwrite_or_resume_allowed=False))
            raise
        finally:
            _close_all(entries)
    return rows
