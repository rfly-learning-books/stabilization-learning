"""Frozen, same-start long/short evaluation and post-arrival retention evidence."""
from copy import deepcopy
from pathlib import Path
import json
import math
import time
import numpy as np
import torch

from extensions.unified_training_v1.backends.rgb import PROJECT, _sha, _module_hash, _write
from extensions.rgb_diagnostic_v5.legacy_transfer import LegacyPolicy


def load_previous_short(self):
    if hasattr(self, "previous_short_models"):
        return
    source = Path(self.config["previous_short_checkpoint"])
    if _sha(source) != self.config["previous_short_checkpoint_sha256"]:
        raise RuntimeError("Previous short selected checkpoint changed")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    ae, metric = deepcopy(self.learner.ae), deepcopy(self.learner.metric)
    ae.load_state_dict(saved["learner"]["ae"])
    metric.load_state_dict(saved["learner"]["metric"])
    ae.eval().requires_grad_(False); metric.eval().requires_grad_(False)
    self.previous_short_models = (ae, metric)
    self.previous_short_provenance = dict(checkpoint=str(source), checkpoint_sha256=_sha(source),
        ae_sha256=_module_hash(ae), metric_sha256=_module_hash(metric),
        trained_reference=True, selection_candidate=False, training_initialization=False)
    _write(self.output / "PREVIOUS_SHORT_REFERENCE.json", self.previous_short_provenance)


def previous_short_policy(self, public, variant, depth=None):
    if not hasattr(self, "previous_short_models"):
        raise RuntimeError("Short model is only loaded after frozen test barrier")
    ae, metric = self.previous_short_models
    return LegacyPolicy(self.adapter, ae, metric, depth or self.depth, public,
                        identity=variant == "previous_short_identity")


def retention_diagnostic(self, variants):
    """Original arrival by 200 steps, then 50 further steps at unchanged tolerance."""
    from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv
    if not (self.output.parent.parent / "GLOBAL_FROZEN.json").is_file():
        raise RuntimeError("Retention requires the same frozen test barrier")
    folder = self.output / "RGB_RETENTION_TEST"
    folder.mkdir(exist_ok=False)
    declared = tuple(self.config["retention_variants"])
    if not set(declared) <= set(variants):
        raise ValueError("Retention variants must be declared main-test variants")
    cases = self.cases["test"]
    reach_limit = int(self.config["evaluation_steps"])
    post_limit = int(self.config["post_success_steps"])
    horizon = reach_limit+post_limit
    cfg = deepcopy(self.cfg)
    strict_hold = cfg["collection"]["required_hold_observations"]
    cfg["collection"]["required_hold_observations"] = horizon+1
    _write(folder / "FROZEN.json", dict(cases_sha256=_sha(self.output/"RGB_CASES.json"),
        variants=declared, reach_limit=reach_limit, post_success_steps=post_limit,
        strict_hold_observations=strict_hold, selection_candidate=False,
        ae_sha256=_module_hash(self.learner.ae), metric_sha256=_module_hash(self.learner.metric)))
    rows = []
    for plant in self.config.get("evaluation_plants", ("P0", "P1")):
        for variant in declared:
            for start in range(0, len(cases), int(self.config["evaluation_batch"])):
                chunk = cases[start:start+int(self.config["evaluation_batch"])]
                env = GPUParallelRGBEnv(cfg, [self.lookup[c["scene_id"]] for c in chunk], plant=plant,
                    max_steps=horizon, device=self.device, cache_dir=PROJECT/".cache/rgb_gpu_warp", samples_per_axis=2)
                try:
                    public, info = env.reset(options=dict(offsets=[c["offset"] for c in chunk], goal_ids=[c["goal_id"] for c in chunk]))
                    policy = self._policy(public, variant)
                    first = policy(public)
                    n = len(chunk)
                    alive = np.ones(n, bool)
                    lengths = np.zeros(n, int)
                    arrival = np.full(n, -1, int)
                    reason = np.full(n, "not_reached", dtype=object)
                    trace = {k: [] for k in ("error", "V", "offset", "action", "applied_action", "dt", "integrated_error", "input_squared", "hold", "terminated", "truncated")}
                    for key, tensor in (("error", first.error), ("V", first.value), ("offset", info["supervision"]["offset"])):
                        trace[key].append(tensor.detach().cpu().numpy())
                    for step in range(horizon):
                        self.guard()
                        if not alive.any():
                            break
                        out = policy(public)
                        following, _, terminal, truncated, info = env.step(out.action)
                        if not info["transition_mask"].cpu().numpy()[alive].all():
                            raise RuntimeError("Retention autoreset contamination")
                        labels = info["supervision"]
                        bad = (info["collision"] | info["outside"]).cpu().numpy()
                        strict = alive & (info["hold_observations"].cpu().numpy() >= strict_hold) & ~bad
                        newly = strict & (arrival < 0) & (step+1 <= reach_limit)
                        arrival[newly] = step+1
                        end_window = alive & (arrival >= 0) & (step+1 >= arrival+post_limit)
                        end_no_arrival = alive & (arrival < 0) & (step+1 >= reach_limit)
                        cut = truncated.cpu().numpy() | ((end_window | end_no_arrival) & ~terminal.cpu().numpy())
                        with torch.no_grad():
                            successor = policy.error(following["rgb"])
                            value = self.adapter.core.energy(successor, out.metric)
                        for key, tensor in (("error", successor), ("V", value), ("offset", labels["next_offset"]),
                            ("action", out.action), ("applied_action", labels["actual_velocity"]/env.scale), ("dt", info["dt"]),
                            ("integrated_error", labels["integrated_error"]), ("input_squared", labels["normalized_input_squared_integral"]),
                            ("hold", info["hold_observations"]), ("terminated", terminal),
                            ("truncated", torch.as_tensor(cut, device=self.device))):
                            trace[key].append(tensor.detach().cpu().numpy())
                        ended = alive & (terminal.cpu().numpy() | cut)
                        reason[ended & end_window] = "window_complete"
                        reason[ended & info["collision"].cpu().numpy()] = "collision"
                        reason[ended & info["outside"].cpu().numpy()] = "outside"
                        lengths += alive
                        alive &= ~ended
                        public = following
                    policy.check()
                    host = {key: np.stack(values) for key, values in trace.items()}
                    for lane, (case, length) in enumerate(zip(chunk, lengths)):
                        arrays = {key: values[:length+(key in ("error", "V", "offset")), lane] for key, values in host.items()}
                        arrays.update(M=first.metric[lane].cpu().numpy(), J=first.interaction[lane].cpu().numpy())
                        audit = self._audit(arrays, "retention")
                        path = folder/f"{plant}_{variant}_{start+lane:04d}.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, **arrays)
                        q = arrays["offset"]
                        good = (np.linalg.norm(q[:, :3], axis=1) < self.cfg["collection"]["position_tolerance_m"]) & (abs(q[:, 3]) < math.radians(self.cfg["collection"]["yaw_tolerance_deg"]))
                        at = int(arrival[lane])
                        post = good[at+1:] if at >= 0 else np.empty(0, bool)
                        complete = at >= 0 and length-at >= post_limit
                        row = dict(**case, plant=plant, variant=variant, first_strict_success_step=at if at >= 0 else None,
                            reached_by_original_limit=at >= 0, post_steps=len(post), full_window=bool(complete),
                            retained_all_5s=bool(complete and post.all()),
                            post_outside_tolerance=int((~post).sum()), reason=str(reason[lane]),
                            final_position_error_m=float(np.linalg.norm(q[-1, :3])), final_yaw_error_deg=float(abs(q[-1, 3])*180/math.pi),
                            trajectory=str(path), trajectory_sha256=_sha(path), audit=audit)
                        rows.append(row)
                        self.event("rgb_retention_episode", plant=plant, variant=variant, case_id=case["case_id"],
                                   reached=row["reached_by_original_limit"], retained=row["retained_all_5s"])
                finally:
                    env.close()
    groups = {}
    for plant in self.config.get("evaluation_plants", ("P0", "P1")):
        for variant in declared:
            group = [r for r in rows if r["plant"] == plant and r["variant"] == variant]
            reached = [r for r in group if r["reached_by_original_limit"]]
            groups[f"{plant}/{variant}"] = dict(episodes=len(group), reached=len(reached),
                full_windows=sum(r["full_window"] for r in reached),
                retained_all_5s=sum(r["retained_all_5s"] for r in reached),
                fraction_retained_given_reached=(sum(r["retained_all_5s"] for r in reached)/len(reached) if reached else None))
    result = dict(schema="rgb_frozen_retention_v1", groups=groups, rows=rows,
                  scope="additional frozen diagnostic; never model selection or revised main success")
    _write(folder/"RESULTS.json", result)
    return result


def evaluate_longroll(self, split="test", variants=("selected", "initial", "identity")):
    self.guard(); split = "val" if split == "validation" else split
    if split not in self.cases: raise ValueError("Unknown evaluation split")
    if split == "test":
        frozen_path = self.output / "FROZEN.json"
        global_path = self.output.parent.parent / "GLOBAL_FROZEN.json"
        if not frozen_path.is_file() or not global_path.is_file():
            raise RuntimeError("Test requires job and global freeze barriers")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        global_frozen = json.loads(global_path.read_text(encoding="utf-8"))
        if global_frozen.get("schema") != "unified_frozen_group_v1":
            raise RuntimeError("Unknown global freeze schema")
        if global_frozen["jobs"].get(self.output.name) != _sha(frozen_path):
            raise RuntimeError("RGB job freeze is absent from or changed after group freeze")
        job_protocol_path = self.output / "protocol.json"
        if _sha(job_protocol_path) != frozen["protocol_sha256"]:
            raise RuntimeError("RGB job protocol changed after freezing")
        job_protocol = json.loads(job_protocol_path.read_text(encoding="utf-8"))
        if global_frozen["protocol_sha256"] != job_protocol["suite_protocol_sha256"]:
            raise RuntimeError("RGB job does not belong to the frozen group protocol")
        if _sha(self.output / frozen["checkpoint"]) != frozen["sha256"]:
            raise RuntimeError("Frozen RGB checkpoint changed")
        saved = torch.load(self.output / frozen["checkpoint"], map_location="cpu", weights_only=False)
        for name in ("ae", "metric"):
            for key, value in getattr(self.learner, name).state_dict().items():
                if not torch.equal(value.detach().cpu(), saved["learner"][name][key].cpu()):
                    raise RuntimeError("Active RGB model is not the frozen selected model")
    variants = tuple(variants)
    if len(set(variants)) != len(variants) or set(variants)-{"selected", "initial", "identity", "shadow", "historical_trained_reference", "historical_trained_identity", "previous_short_selected", "previous_short_identity"}:
        raise ValueError("Unknown or repeated evaluation variant")
    if {"historical_trained_reference", "historical_trained_identity"} & set(variants):
        if split != "test":
            raise ValueError("Historical reference is reserved for the frozen test, never selection")
        self._load_historical_reference()
    if {"previous_short_selected", "previous_short_identity"} & set(variants):
        if split != "test":
            raise ValueError("Previous short models are frozen-test references only")
        load_previous_short(self)
    cases = self.cases[split]; folder = self.output / f"RGB_EVAL_{self.evaluation_id:03d}_{split}"
    folder.mkdir(exist_ok=False); self.evaluation_id += 1
    _write(folder / "FROZEN.json", dict(created_unix=time.time(), variants=variants, split=split,
        ae_sha256=_module_hash(self.learner.ae), metric_sha256=_module_hash(self.learner.metric),
        cases_sha256=_sha(self.output / "RGB_CASES.json"), counts=self.learner.counts.copy(),
        historical_reference=(self.historical_provenance if {"historical_trained_reference", "historical_trained_identity"} & set(variants) else None)))
    rows = []; started = time.perf_counter()
    for plant in self.config.get("evaluation_plants", ("P0", "P1")):
        for variant in variants:
            for start in range(0, len(cases), int(self.config.get("evaluation_batch", 4))):
                chunk = cases[start:start+int(self.config.get("evaluation_batch", 4))]
                env = self._env(chunk, plant)
                try:
                    public, info = env.reset(options=dict(offsets=[c["offset"] for c in chunk], goal_ids=[c["goal_id"] for c in chunk]))
                    policy = self._policy(public, variant)
                    n = len(chunk); alive = np.ones(n, bool); lengths = np.zeros(n, int)
                    reason = np.array(["timeout"]*n, dtype=object)
                    trace = {k: [] for k in ("error", "V", "offset", "action", "applied_action", "dt", "integrated_error", "input_squared", "hold", "terminated", "truncated")}
                    first = policy(public); matrix = first.metric.detach().cpu().numpy()
                    trace["error"].append(first.error.cpu().numpy()); trace["V"].append(first.value.cpu().numpy())
                    trace["offset"].append(info["supervision"]["offset"].cpu().numpy())
                    wall = []
                    for step in range(env.max_steps):
                        self.guard()
                        if not alive.any(): break
                        torch.cuda.synchronize(); t = time.perf_counter(); out = policy(public)
                        torch.cuda.synchronize(); wall.append(time.perf_counter()-t)
                        following, _, terminal, truncated, info = env.step(out.action)
                        labels = info["supervision"]
                        if not info["transition_mask"].cpu().numpy()[alive].all(): raise RuntimeError("Evaluation autoreset contamination")
                        with torch.no_grad():
                            successor = policy.error(following["rgb"])
                            value = self.adapter.core.energy(successor, out.metric)
                        for key, tensor in (("error", successor), ("V", value), ("offset", labels["next_offset"]),
                            ("action", out.action), ("applied_action", labels["actual_velocity"]/env.scale), ("dt", info["dt"]),
                            ("integrated_error", labels["integrated_error"]), ("input_squared", labels["normalized_input_squared_integral"]),
                            ("hold", info["hold_observations"]), ("terminated", terminal), ("truncated", truncated)):
                            trace[key].append(tensor.detach().cpu().numpy())
                        ended = alive & (terminal | truncated).cpu().numpy()
                        for label in ("success", "collision", "outside"):
                            reason[ended & info[label].cpu().numpy()] = label
                        lengths += alive; alive &= ~(terminal | truncated).cpu().numpy(); public = following
                    policy.check()
                    host = {key: np.stack(values) for key, values in trace.items()}
                    for lane, (case, length) in enumerate(zip(chunk, lengths)):
                        arrays = {key: values[:length+(key in ("error", "V", "offset")), lane] for key, values in host.items()}
                        arrays.update(M=matrix[lane], J=first.interaction[lane].cpu().numpy())
                        audit = self._audit(arrays, reason[lane])
                        path = folder / f"{plant}_{variant}_{start+lane:04d}.npz"
                        with path.open("xb") as stream: np.savez_compressed(stream, **arrays)
                        q = arrays["offset"]; E = (q[:, :3]**2).sum(1)+(q[:, 3]/math.radians(25))**2
                        dv = np.diff(arrays["V"]); tol = 1e-7+1e-4*arrays["V"][:-1]
                        row = dict(**case, plant=plant, variant=variant, success=reason[lane]=="success", reason=str(reason[lane]), steps=int(length),
                            elapsed_s=float(arrays["dt"].sum()), integrated_error=float(arrays["integrated_error"].sum()),
                            input_squared_integral=float(arrays["input_squared"].sum()), final_error=float(E[-1]),
                            V_rise_fraction=float((dv>tol).mean()), V_lower_violation=float((arrays["V"]<.25*E-1e-7).mean()),
                            V_over_E_median=float(np.median(arrays["V"]/(E+1e-12))),
                            policy_wall_s=float(np.mean(wall[:length])), trajectory=str(path), trajectory_sha256=_sha(path), audit=audit)
                        rows.append(row)
                        self.event("rgb_evaluation_episode", split=split, plant=plant, variant=variant,
                                   case_id=case["case_id"], success=row["success"], reason=row["reason"])
                finally:
                    env.close()
    groups = {}
    for plant in self.config.get("evaluation_plants", ("P0", "P1")):
        for variant in variants:
            group = [r for r in rows if r["plant"]==plant and r["variant"]==variant]
            groups[f"{plant}/{variant}"] = dict(episodes=len(group), successes=sum(r["success"] for r in group),
                success_rate=float(np.mean([r["success"] for r in group])),
                mean_integrated_error=float(np.mean([r["integrated_error"] for r in group])),
                mean_input_squared=float(np.mean([r["input_squared_integral"] for r in group])),
                failures={reason:sum(r["reason"]==reason for r in group) for reason in ("timeout", "collision", "outside")})
    selected = [r for r in rows if r["variant"]=="selected"]
    summary = dict(episodes=len(rows), unique_cases=len(cases), groups=groups,
        success_rate=float(np.mean([r["success"] for r in selected])) if selected else None,
        mean_integrated_error=float(np.mean([r["integrated_error"] for r in selected])) if selected else None,
        wall_s=time.perf_counter()-started, independent_successor_audit=True)
    result = dict(split=split, summary=summary, rows=rows, folder=str(folder))
    _write(folder / "RESULTS.json", result)
    if split == "test" and self.config.get("retention_diagnostic", False):
        result["retention"] = retention_diagnostic(self, variants)
    return result

