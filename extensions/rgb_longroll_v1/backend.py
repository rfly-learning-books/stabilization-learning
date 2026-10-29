"""Long rollouts with matched per-room transition budgets and strict evidence."""
from collections import Counter
from copy import deepcopy
import json
import math
import time

import numpy as np
import torch

from extensions.unified_training_v1.backends.rgb import PROJECT, _sha, _write
from extensions.unified_training_v2.historical_rgb import HistoricalRGBBackend
from .evaluation import evaluate_longroll, previous_short_policy
from .initialization import initialize_longroll


def phase_ids(offset, position_tolerance, yaw_tolerance):
    """Physical supervision for counting only; never a policy input."""
    offset = np.asarray(offset)
    distance = np.maximum(np.linalg.norm(offset[..., :3], axis=-1)/position_tolerance,
                          np.abs(offset[..., 3])/yaw_tolerance)
    return np.select((distance <= 1, distance <= 5, distance <= 15), (0, 1, 2), 3)


class LongRollRGBBackend(HistoricalRGBBackend):
    def initialize(self):
        result = initialize_longroll(self)
        expected = json.loads(open(self.config["previous_cases"], encoding="utf-8").read())
        if self.cases != expected:
            raise AssertionError("Long/short evaluation starts must match exactly")
        policy = dict(schema="rgb_longroll_collection_v1", training_seed=self.seed, evaluation_seed=int(self.config.get("evaluation_seed", self.seed)), regular_max_steps=int(self.config["collection_steps"]),
            after_first_strict_success_steps=int(self.config["post_success_steps"]),
            strict_position_tolerance_m=self.cfg["collection"]["position_tolerance_m"],
            strict_yaw_tolerance_deg=self.cfg["collection"]["yaw_tolerance_deg"],
            strict_hold_observations=self.cfg["collection"]["required_hold_observations"],
            quotas=self.config["regular_collection_quotas"], initial_collection="unchanged v1, fresh data",
            phase_bins="max(position/2cm, abs(yaw)/1deg): inside<=1, near<=5, mid<=15, far>15",
            sampling="retained scene-balanced replay, no new phase reweighting",
            intervention="long trajectories + post-success collection + alive-only smooth triplets",
            causal_scope="combined collection intervention, not an isolated horizon-only ablation")
        _write(self.output / "LONGROLL_PROTOCOL.json", policy)
        self.event("rgb_longroll_initialized", **policy)
        return dict(**result, longroll=policy)

    def _policy(self, public, variant="selected", depth=None):
        if variant in ("previous_short_selected", "previous_short_identity"):
            return previous_short_policy(self, public, variant, depth)
        return super()._policy(public, variant, depth)

    def evaluate(self, split="test", variants=("selected", "initial", "identity")):
        return evaluate_longroll(self, split, variants)

    def collect(self, initial=False):
        if initial:
            return super().collect(initial=True)
        from extensions.joint_certificate_v4.rollout import sample_offset
        from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
        from train_goal_fixed import augment_depth

        self.guard()
        started = time.perf_counter()
        collection = self.collection_id
        quotas = self.config["regular_collection_quotas"][collection-1]
        scene_ids = [scene["id"] for scene in self.train_scenes]
        if set(quotas) != set(scene_ids) or any(type(v) is not int or v <= 0 for v in quotas.values()):
            raise ValueError("Every training room needs a positive frozen transition quota")
        remaining = dict(quotas)
        attempts = Counter()
        horizon = int(self.config["collection_steps"])
        lanes = int(self.config["parallel_envs"])
        post_steps = int(self.config["post_success_steps"])
        strict_hold = int(self.cfg["collection"]["required_hold_observations"])
        ptol = float(self.cfg["collection"]["position_tolerance_m"])
        ytol = math.radians(self.cfg["collection"]["yaw_tolerance_deg"])
        if not 1 <= horizon <= 120 or post_steps < 0 or lanes < 1:
            raise ValueError("Invalid bounded collection settings")
        before = self.replay.transitions
        coverage = []
        phase_counts = Counter()
        passes = 0
        while any(remaining.values()):
            passes += 1
            pending = [sid for sid in scene_ids if remaining[sid] > 0]
            # Every pending room gets a turn before another trajectory is assigned.
            for start in range(0, len(pending), lanes):
                self.guard()
                chunk = pending[start:start+lanes]
                cases = []
                caps = []
                for sid in chunk:
                    attempts[sid] += 1
                    if attempts[sid] > int(self.config.get("maximum_episodes_per_room", 128)):
                        raise RuntimeError("Collection attempt cap reached without filling frozen quota: "+sid)
                    scene = self.lookup[sid]
                    goal = (collection-1+attempts[sid]-1) % len(scene["refs"])
                    difficulty = ("small", "medium", "large")[(collection+attempts[sid]+scene_ids.index(sid)) % 3]
                    offset = sample_offset(self.cfg, scene, goal, difficulty, self.rng)
                    cases.append(dict(scene_id=sid, goal_id=goal, difficulty=difficulty,
                                      offset=offset.tolist(), episode=int(attempts[sid])))
                    caps.append(min(horizon, remaining[sid]))
                caps = np.asarray(caps, dtype=int)
                cfg = deepcopy(self.cfg)
                # Only collection continues after original success. Evaluation is unchanged.
                cfg["collection"]["required_hold_observations"] = horizon+1
                env = GPUParallelRGBEnv(cfg, [self.lookup[c["scene_id"]] for c in cases], plant="P0",
                    max_steps=horizon, device=self.device, cache_dir=PROJECT/".cache/rgb_gpu_warp", samples_per_axis=2)
                try:
                    public, info = env.reset(options=dict(offsets=[c["offset"] for c in cases],
                                                         goal_ids=[c["goal_id"] for c in cases]))
                    goals = public["goal_rgb"].cpu().numpy()
                    rotations = public["rotation"].cpu().numpy()
                    base_depth = np.stack([self.depth.predict(image) for image in goals])
                    used_depth = np.stack([augment_depth(d.copy(), self.rng, self.cfg)[0] for d in base_depth])
                    depth_map = {image.tobytes(): d for image, d in zip(goals, used_depth)}
                    class DepthMap:
                        def predict(_, image):
                            return depth_map[image.tobytes()]
                    policy = self._policy(public, depth=DepthMap())
                    n = len(chunk)
                    alive = np.ones(n, dtype=bool)
                    lengths = np.zeros(n, dtype=int)
                    first_success = np.full(n, -1, dtype=int)
                    max_hold = np.zeros(n, dtype=int)
                    reasons = np.full(n, "horizon", dtype=object)
                    ar = {key: [] for key in ("rgb", "offset", "action", "cost", "terminal", "truncated", "previous_action", "phase", "strict_success", "post_success")}
                    ar["rgb"].append(public["rgb"].cpu().numpy())
                    ar["offset"].append(info["supervision"]["offset"].cpu().numpy())
                    triplets = []
                    for step in range(horizon):
                        self.guard()
                        if not alive.any():
                            break
                        if step % 10 == 0 and len(triplets) < self.cfg["goal_fixed"]["smooth_triplet_limit_per_episode"]:
                            triplets.append((step, alive.copy(), self._triplet(env, public)))
                        action = policy(public).action
                        noise = torch.as_tensor(self.rng.normal(0, .06, (n, 4)), dtype=torch.float32, device=self.device)
                        action = self.adapter.core.saturate(action+noise)
                        action = torch.where(torch.as_tensor(alive, device=self.device)[:, None], action, torch.zeros_like(action))
                        following, _, terminal, truncated, info = env.step(action)
                        if not info["transition_mask"].cpu().numpy()[alive].all():
                            raise RuntimeError("Autoreset entered active long trajectory")
                        labels = info["supervision"]
                        bad = (info["collision"] | info["outside"]).cpu().numpy()
                        hold = info["hold_observations"].cpu().numpy()
                        max_hold = np.maximum(max_hold, np.where(alive, hold, 0))
                        was_post = first_success >= 0
                        strict = alive & (hold >= strict_hold) & ~bad
                        first_success[(first_success < 0) & strict] = step+1
                        end_post = alive & (first_success >= 0) & ((step+1-first_success) >= post_steps)
                        end_quota = alive & (step+1 >= caps)
                        custom_end = end_post | end_quota
                        trunc_host = truncated.cpu().numpy() | (custom_end & ~terminal.cpu().numpy())
                        trunc_tensor = torch.as_tensor(trunc_host, device=self.device)
                        phases = phase_ids(labels["offset"].cpu().numpy(), ptol, ytol)
                        for key, tensor in (("rgb", following["rgb"]), ("offset", labels["next_offset"]),
                            ("action", action), ("cost", labels["physical_cost"]), ("terminal", terminal),
                            ("truncated", trunc_tensor), ("previous_action", labels["previous_action"])):
                            ar[key].append(tensor.detach().cpu().numpy())
                        ar["phase"].append(phases)
                        ar["strict_success"].append(strict)
                        ar["post_success"].append(was_post & alive)
                        ended = alive & (terminal.cpu().numpy() | trunc_host)
                        reasons[ended & end_quota] = "quota" if (caps < horizon).any() else "horizon"
                        reasons[ended & end_quota & (caps >= horizon)] = "horizon"
                        reasons[ended & end_post] = "post_success_window"
                        reasons[ended & info["collision"].cpu().numpy()] = "collision"
                        reasons[ended & info["outside"].cpu().numpy()] = "outside"
                        lengths += alive
                        alive &= ~ended
                        public = following
                    policy.check()
                    arrays = {key: np.stack(values) for key, values in ar.items()}
                    for lane, (case, length) in enumerate(zip(cases, lengths)):
                        if length <= 0:
                            raise AssertionError("Empty long trajectory")
                        item = {key: values[:length+(key in ("rgb", "offset")), lane] for key, values in arrays.items()}
                        item.update(goal_rgb=goals[lane], goal_depth=base_depth[lane],
                                    collection_depth=used_depth[lane], rotation=rotations[lane])
                        valid_triplets = [tri for t, mask, tri in triplets if t < length and mask[lane]]
                        if not valid_triplets:
                            raise AssertionError("Every trajectory must include its own initial triplet")
                        for key in valid_triplets[0]:
                            item[key] = np.stack([tri[key][lane] for tri in valid_triplets])
                        row = dict(scene_id=case["scene_id"], split="train", goal_id=case["goal_id"])
                        self.replay.add(item, row, f"long_{collection:03d}_{case['scene_id']}_{case['episode']:03d}")
                        sid = case["scene_id"]
                        remaining[sid] -= int(length)
                        if remaining[sid] < 0:
                            raise AssertionError("Room transition quota exceeded")
                        bins = Counter({name:int((item["phase"] == code).sum()) for code, name in enumerate(("inside", "near", "mid", "far"))})
                        phase_counts.update(bins)
                        reached = int(first_success[lane])
                        post = item["post_success"]
                        next_inside = phase_ids(item["offset"][1:], ptol, ytol) == 0
                        coverage.append(dict(**case, transitions=int(length), reason=str(reasons[lane]),
                            phase_counts=dict(bins), first_strict_success_step=(reached if reached >= 0 else None),
                            post_success_transitions=int(post.sum()), post_success_outside_tolerance=int((post & ~next_inside).sum()),
                            post_success_window_complete=bool(reached >= 0 and length-reached >= post_steps),
                            maximum_consecutive_inside_observations=int(max_hold[lane]),
                            smooth_triplets=len(valid_triplets), remaining_room_quota=remaining[sid]))
                    self.event("rgb_longroll_batch", collection=collection, passes=passes,
                        fresh_transitions=sum(int(v) for v in lengths), remaining_transitions=sum(remaining.values()))
                finally:
                    env.close()
        self.collection_id += 1
        fresh = sum(row["transitions"] for row in coverage)
        if fresh != sum(quotas.values()) or sum(phase_counts.values()) != fresh:
            raise AssertionError("Exact transition/phase accounting failed")
        result = dict(stage="collection", schema="rgb_longroll_collection_v1", fresh_transitions=fresh,
            replay_transitions=self.replay.transitions, prior_replay_transitions=before, old_replay_loaded=False,
            scene_count=len(scene_ids), wall_s=time.perf_counter()-started, coverage=coverage,
            phase_counts=dict(phase_counts), maximum_episode_steps=horizon, post_success_steps=post_steps,
            matched_previous_room_quotas=quotas, room_attempts=dict(attempts))
        _write(self.output / f"RGB_COLLECTION_{collection:03d}.json", result)
        return result

    def close(self):
        if hasattr(self, "previous_short_models"):
            del self.previous_short_models
        return super().close()
