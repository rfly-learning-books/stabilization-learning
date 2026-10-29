"""Recorded SAC backend using the fixed, independently namespaced Push env."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import random
import types

import numpy as np
import torch
from torch import nn

from extensions.unified_training_v1.backends.push import CandidateQ, _module_hash, _sha
from extensions.unified_training_v2.push_curves import RecordedPushBackend
from .environment import JointPushCosts, JointPushEnv, vendor_modules
from .evaluation import evaluate as evaluate_push
from .numerical import CheckedReplay, FiniteAdam, require_finite


class FixedPushBackend(RecordedPushBackend):
    """Keep v2 update/checkpoint semantics, replacing unsafe initialization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("learning_rate", "alpha_learning_rate", "initial_alpha",
                     "entropy_fraction", "target_tau"):
            value = getattr(self.config, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("Finite numerical SAC setting required: " + name)
        self._numerical_failure = None
        self.replay = CheckedReplay(failure=self._fail_numerically)

    def _fail_numerically(self, error):
        # A failed step may already have touched parameters. Never silently
        # continue, select, evaluate, or publish it as a healthy checkpoint.
        if self._numerical_failure is None:
            self._numerical_failure = str(error)

    def _guard(self):
        if self._numerical_failure is not None:
            raise FloatingPointError("Push backend stopped after numerical failure: " + self._numerical_failure)
        return super()._guard()

    def _check_temperature(self):
        try:
            alpha = self.log_alpha.exp()
            require_finite(alpha, "SAC temperature")
            if not bool(alpha > 0):
                raise FloatingPointError("SAC temperature underflowed to zero")
        except FloatingPointError as error:
            self._fail_numerically(error)
            raise

    def _distribution(self, data, *, actor=None, training=False):
        self._guard()
        try:
            require_finite(data, "candidate input")
            cand = self.model.candidates(self.actor if actor is None else actor, data, training=training)
            require_finite(cand, "candidate output")
            physical = cand["physical"]
            valid = physical.any(-1)
            mask = physical.clone()
            mask[~valid, 0] = True  # Execution still aborts; this only defines a categorical.
            logits = cand["logits"].masked_fill(~mask, -torch.inf)
            logits[~valid, 0] = 0.
            # A finite sentinel can dominate legitimate very negative logits,
            # leaving zero probability mass after the invalid slots are removed.
            logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.)
            probability = logp.exp().masked_fill(~mask, 0.)
            require_finite(logp, "candidate log probabilities")
            require_finite(probability, "candidate probabilities")
            return probability, logp, valid, cand
        except FloatingPointError as error:
            self._fail_numerically(error)
            raise

    def _batch(self):
        try:
            batch = super()._batch()
            require_finite(batch, "sampled replay batch")
            if bool((batch["dt"] < 0).any()):
                raise FloatingPointError("Negative sampled replay duration")
            if bool(((batch["action"] < 0) | (batch["action"] >= self.action_count)).any()):
                raise FloatingPointError("Sampled replay action outside candidate range")
            return batch
        except FloatingPointError as error:
            self._fail_numerically(error)
            raise

    def _critic_update(self):
        self._guard()
        self._check_temperature()
        try:
            return super()._critic_update()
        except FloatingPointError as error:
            self._fail_numerically(error)
            raise

    def _actor_update(self):
        self._guard()
        self._check_temperature()
        try:
            return super()._actor_update()
        except FloatingPointError as error:
            self._fail_numerically(error)
            raise

    def _record_update(self, kind, result):
        try:
            self._check_temperature()
            require_finite(result, kind + " update metrics")
        except FloatingPointError as error:
            self._fail_numerically(error)
            raise
        return super()._record_update(kind, result)

    def _probe(self, stage):
        try:
            with torch.no_grad():
                ids = [self.bank.names.index(name) for name in self.config.train_shapes]
                q = torch.tensor([[.01, -.008, .2], [-.006, .004, -.25], [.002, .002, .05]], device=self.config.device)
                D = self.model.coordinates(q)[None].expand(len(ids), -1, -1)
                condition = self.actor.encoder(self.model.C_encoded[ids], self.model.valid[ids], ell=self.model.cfg.length_scale)
                _, matrix = self.actor.conditioned(D, condition)
                require_finite(matrix, "fixed metric probe")
            record = dict(stage=stage, step=self.counts["certificate_updates"], counts=dict(self.counts),
                shapes=list(self.config.train_shapes), physical_q=q.cpu().tolist(), M=matrix.cpu().tolist())
            self.probe_records.append(record)
            with (self.output / "push_fixed_step_probes.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        except FloatingPointError as error:
            self._fail_numerically(error)
            raise

    def save(self, path):
        self._guard()
        try:
            self._check_temperature()
            for name in ("actor", "q1", "q2", "target1", "target2"):
                require_finite(getattr(self, name).state_dict(), name + " checkpoint")
            for optimizer in (self.actor_optimizer, self.q_optimizer, self.alpha_optimizer):
                optimizer.check()
        except FloatingPointError as error:
            self._fail_numerically(error)
            raise
        return super().save(path)

    def source_files(self):
        modules = (
            "extensions.unified_training_v3.push", "extensions.unified_training_v3.environment",
            "extensions.unified_training_v3.numerical",
            "extensions.unified_training_v3.evaluation", "extensions.unified_training_v2.push_curves",
            "extensions.unified_training_v1.backends.push", "stabilization_gym.tasks_v2.push",
            "third_party.so101_timeaware_v2.so101_timeaware.environment",
            "third_party.so101_timeaware_v2.so101_timeaware.policy",
            "third_party.so101_timeaware_v2.so101_timeaware.timing",
            "third_party.so101_timeaware_v2.so101_multishape",
            "third_party.so101_timeaware_v2.so101_campaign",
            "third_party.so101_timeaware_v2.so101_contact_demo",
            "dl_push.certificate.full_m_control",
        )
        return [str(Path(importlib.import_module(name).__file__).resolve()) for name in modules]

    def initialize(self):
        if self.initialized:
            raise RuntimeError("Backend already initialized")
        self._guard()
        cfg = self.config
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        if cfg.device.startswith("cuda"):
            torch.cuda.manual_seed_all(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed + 802810)
        self.vendor = vendor_modules()
        # Legacy SO101 imports set a process-wide thread count. Reassert the
        # backend's intended value even when other plugins were imported first.
        torch.set_num_threads(1)
        bank_path = Path(cfg.time_bank).expanduser()
        if not bank_path.is_absolute():
            bank_path = Path(__file__).resolve().parents[2] / bank_path
        self.bank = self.vendor["Bank"](bank_path.resolve())
        if set(cfg.train_shapes + cfg.validation_shapes + cfg.test_shapes) - set(self.bank.names):
            raise ValueError("Shape absent from declared calibration bank")
        self.costs = JointPushCosts()
        self.model = self.vendor["Model"](self.bank, cfg.device)
        self.actor = self.vendor["Actor"](self.vendor["ActorConfig"]()).to(cfg.device)
        self.initial_actor = deepcopy(self.actor.state_dict())
        self.action_count = 12 * len(self.bank.config.travels)
        self.q1 = CandidateQ(self.action_count).to(cfg.device)
        self.q2 = CandidateQ(self.action_count).to(cfg.device)
        self.target1, self.target2 = deepcopy(self.q1), deepcopy(self.q2)
        for target in (self.target1, self.target2):
            target.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.initial_alpha), device=cfg.device))
        self._check_temperature()
        self.actor_optimizer = FiniteAdam(self.actor.parameters(), lr=cfg.learning_rate,
            failure=self._fail_numerically, label="actor")
        self.q_optimizer = FiniteAdam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.learning_rate,
            failure=self._fail_numerically, label="twin Q")
        self.alpha_optimizer = FiniteAdam([self.log_alpha], lr=cfg.alpha_learning_rate,
            failure=self._fail_numerically, label="temperature")
        self.envs = [JointPushEnv(self.bank, list(cfg.train_shapes), cfg.seed + 1000 + i, self.costs)
                     for i in range(cfg.num_envs)]
        self.initialized = True
        self.initial_actor_hash = _module_hash(self.actor)
        self.initial_q_hashes = [_module_hash(self.q1), _module_hash(self.q2)]
        self.output.mkdir(parents=True, exist_ok=True)
        self._probe("initialize")
        result = dict(algorithm=self.algorithm, config=asdict(cfg), capabilities=self.capabilities,
            initialization="random actor/M/shape encoders and twin Q; empty optimizers; no learned checkpoint import",
            bank_sha256=self.bank.sha256, sources={p: _sha(p) for p in self.source_files()},
            reward_scale=self.bank.config.reward_scale, discount_gamma=self.bank.config.gamma,
            discount_reference_seconds=self.bank.config.discount_reference_seconds,
            objective="soft Q return over combined task rewards; entropy per decision; physical-time discount",
            alpha_target="entropy_fraction * log(number of physically available candidates)",
            scope="reduced calibrated response/time model; not SO101 joint MuJoCo or hardware",
            counts=dict(self.counts), initial_actor_sha256=self.initial_actor_hash,
            initial_q_sha256=self.initial_q_hashes, optimizer_state_empty=all(not o.state for o in
                (self.actor_optimizer, self.q_optimizer, self.alpha_optimizer)))
        self.event("push_initialized", **result)
        return result

    def evaluate(self, split, variants=("selected", "initial", "identity")):
        if split not in ("validation", "test", "train"):
            raise ValueError("Unknown evaluation split")
        if split == "test":
            marker, group = self.output / "FROZEN.json", self.output.parents[1] / "GLOBAL_FROZEN.json"
            if not marker.is_file() or not group.is_file():
                raise RuntimeError("Held-out evaluation requires local and global freeze barriers")
            frozen = json.loads(marker.read_text(encoding="utf-8"))
            frozen_group = json.loads(group.read_text(encoding="utf-8"))
            if frozen_group.get("schema") != "unified_frozen_group_v1" or frozen_group.get("jobs", {}).get(self.output.name) != _sha(marker):
                raise ValueError("Job absent from intact all-backends freeze barrier")
            selected_path = (self.output / frozen["checkpoint"]).resolve()
            if not selected_path.is_relative_to(self.output.resolve()) or _sha(selected_path) != frozen["sha256"]:
                raise ValueError("Frozen selected checkpoint changed or escaped job directory")
            selected = torch.load(selected_path, map_location="cpu", weights_only=False)
            current = deepcopy(self.actor).cpu()
            current.load_state_dict(selected["actor"])
            if _module_hash(current) != _module_hash(self.actor):
                raise ValueError("Load the frozen selected checkpoint before held-out evaluation")
        names = getattr(self.config, {"validation": "validation_shapes", "test": "test_shapes", "train": "train_shapes"}[split])
        repeats = self.config.test_repeats if split == "test" else self.config.validation_repeats
        start = self.config.test_seed_start if split == "test" else self.config.validation_seed_start
        if split == "train":
            start = self.config.seed + 2900000
        before_hash, reports = _module_hash(self.actor), {}
        for variant in variants:
            self._guard()
            actor = deepcopy(self.actor).eval()
            if variant in ("initial", "random_initial"):
                actor.load_state_dict(self.initial_actor)
            elif variant in ("identity", "identity_m"):
                def identity(this, D, condition):
                    eye = torch.eye(3, dtype=D.dtype, device=D.device).expand(*D.shape[:-1], 3, 3)
                    return eye, eye
                actor.conditioned = types.MethodType(identity, actor)
            elif variant not in ("selected", "learned", "final"):
                raise ValueError("Unknown evaluation variant " + str(variant))
            report = evaluate_push(actor, self.model, names, range(start, start + repeats), self.costs, self._guard)
            rows = report["episodes"]
            report.update(mean_input_square=float(np.mean([r["input_square_integral"] for r in rows])),
                mean_input_cost=float(np.mean([r["input_square_integral"] for r in rows])),
                mean_cost=report["mean_combined_cost"], mean_integrated_error=report["mean_error"],
                num_episodes=len(rows), successes=sum(r["success"] for r in rows),
                mean_switches=float(np.mean([r["switches"] for r in rows])), failures=sum(not r["success"] for r in rows),
                failure_reasons={reason:sum(r["reason"] == reason for r in rows) for reason in sorted({r["reason"] for r in rows if not r["success"]})},
                V_increase_fraction=float(np.mean([t["V_next"] > t["V"] + 1e-8 for r in rows for t in r["trace"]])),
                model_sha256=_module_hash(actor), feature_sha256=_module_hash(actor.encoder),
                metric_override="identity" if variant in ("identity", "identity_m") else "learned",
                split=split, cases=len(rows))
            reports[variant] = report
        if _module_hash(self.actor) != before_hash:
            raise RuntimeError("Evaluation mutated actor")
        paired = {}
        if "selected" in reports:
            selected = {(r["shape"], r["seed"]): r for r in reports["selected"]["episodes"]}
            for name, report in reports.items():
                if name == "selected":
                    continue
                pairs = [(selected[(r["shape"], r["seed"])], r) for r in report["episodes"]
                         if r["success"] and selected[(r["shape"], r["seed"])]["success"]]
                paired[name] = dict(common_successes=len(pairs), selected_minus_comparator_seconds=None if not pairs
                    else float(np.mean([a["seconds"] - b["seconds"] for a, b in pairs])))
        return dict(family=self.family, algorithm=self.algorithm, split=split, variants=reports,
                    paired_time_on_common_success=paired, selection_allowed=split == "validation",
                    scope="single-seed integration pilot; calibrated response/time bank; no robot simulation claim")
