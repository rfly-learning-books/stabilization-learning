"""Read-only historical RGB reference in the unified frozen test evaluator.

Training and validation remain the unmodified v1 implementation. The historical
trained AE and M are loaded only after the group test barrier, never as a training
initialization or a selection candidate. Pixels are the new GPU-rendered pixels.
"""
from pathlib import Path
import json
import math
import time
import numpy as np
import torch

from extensions.unified_training_v1.backends.rgb import Backend as V1Backend, _module_hash, _sha, _write
from extensions.rgb_diagnostic_v5.legacy_transfer import LegacyPolicy

EXPECTED_BEST_SHA256 = "3edf4756000daa3373da2e2988c1ad5090201fdd45215aed2092bf1edaad6bca"

class HistoricalRGBBackend(V1Backend):
    def _load_historical_reference(self):
        if hasattr(self, "historical_models"):
            return
        source = self.adapter.bundle / "best.pt"
        if _sha(source).lower() != EXPECTED_BEST_SHA256:
            raise RuntimeError("Historical best checkpoint hash mismatch")
        ae, metric = self.adapter.load_models("best.pt", self.device)
        ae.eval().requires_grad_(False)
        metric.eval().requires_grad_(False)
        self.historical_models = (ae, metric)
        self.historical_provenance = dict(
            checkpoint=str(source), checkpoint_sha256=EXPECTED_BEST_SHA256,
            ae_sha256=_module_hash(ae), metric_sha256=_module_hash(metric),
            trained_reference=True, training_initialization=False, selection_candidate=False,
            controller="LegacyPolicy: goal J, e/J divided by 4, raw goal M, original ridge, gain 0.7",
            pixel_equivalent_to_original=False)

    def _policy(self, public, variant="selected", depth=None):
        if variant not in ("historical_trained_reference", "historical_trained_identity"):
            return super()._policy(public, variant, depth)
        if not hasattr(self, "historical_models"):
            raise RuntimeError("Historical model requires frozen test barrier")
        ae, metric = self.historical_models
        return LegacyPolicy(self.adapter, ae, metric, depth or self.depth, public, identity=(variant == "historical_trained_identity"))

    def close(self):
        if hasattr(self, "historical_models"):
            del self.historical_models
        return super().close()

    def evaluate(self, split="test", variants=("selected", "initial", "identity")):
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
            from stabilization_learning.storage import load_checkpoint
            saved = load_checkpoint(self.output / frozen["checkpoint"], map_location="cpu")
            for name in ("ae", "metric"):
                for key, value in getattr(self.learner, name).state_dict().items():
                    if not torch.equal(value.detach().cpu(), saved["learner"][name][key].cpu()):
                        raise RuntimeError("Active RGB model is not the frozen selected model")
        variants = tuple(variants)
        if len(set(variants)) != len(variants) or set(variants)-{"selected", "initial", "identity", "shadow", "historical_trained_reference", "historical_trained_identity"}:
            raise ValueError("Unknown or repeated evaluation variant")
        if {"historical_trained_reference", "historical_trained_identity"} & set(variants):
            if split != "test":
                raise ValueError("Historical reference is reserved for the frozen test, never selection")
            self._load_historical_reference()
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
        return result

