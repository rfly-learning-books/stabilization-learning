"""M-induced discrete SAC for the existing time-aware, reduced Push plant.

This is a new learner, not a relabelled PPO experiment.  No policy checkpoint is
imported.  Geometry/response/time banks remain explicitly external model data.
The runner owns selection and the all-backends-frozen barrier before testing.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[3]
for _package in (ROOT / "stabilization-gym", ROOT / "stabilization-learning-control"):
    if str(_package) not in sys.path:
        sys.path.insert(0, str(_package))

from stabilization_gym.tasks_v2.push import JointPushCosts, JointPushEnv, vendor_modules


@dataclass(frozen=True)
class PushConfig:
    seed: int = 81
    device: str = "cpu"
    time_bank: str = str(ROOT / "runs/scratch_apps_so101_full_v1_20261026_184356/so101_environment/time_bank.json")
    num_envs: int = 8
    initial_steps: int = 32
    collection_steps: int = 32
    batch_size: int = 128
    replay_capacity: int = 30000
    learning_rate: float = 3e-4
    alpha_learning_rate: float = 1e-4
    initial_alpha: float = .05
    entropy_fraction: float = .5
    target_tau: float = .01
    validation_repeats: int = 4
    test_repeats: int = 8
    validation_seed_start: int = 2810000
    test_seed_start: int = 2820000
    train_shapes: tuple = ("T", "RectWide", "L", "Plus")
    validation_shapes: tuple = ("Square", "UShape")
    test_shapes: tuple = ("OffsetComb3", "AsymThreeBranch")

    def __post_init__(self):
        for name in ("train_shapes", "validation_shapes", "test_shapes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        groups = [set(self.train_shapes), set(self.validation_shapes), set(self.test_shapes)]
        if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Disjoint nonempty shape splits are required")
        for name in ("num_envs", "initial_steps", "collection_steps", "batch_size", "replay_capacity",
                     "validation_repeats", "test_repeats"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("Positive integer required: " + name)
        if not 0 < self.target_tau <= 1 or not 0 <= self.entropy_fraction <= 1:
            raise ValueError("Invalid SAC target/entropy setting")
        if min(self.learning_rate, self.alpha_learning_rate, self.initial_alpha) <= 0:
            raise ValueError("Positive SAC optimization settings required")
        v = set(range(self.validation_seed_start, self.validation_seed_start+self.validation_repeats))
        t = set(range(self.test_seed_start, self.test_seed_start+self.test_repeats))
        if v & t:
            raise ValueError("Validation and test starts overlap")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _frozen(module):
    previous = [parameter.requires_grad for parameter in module.parameters()]
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), previous):
            parameter.requires_grad_(flag)


class CandidateQ(nn.Module):
    """One action-value vector; its shape encoder is NOT the actor's encoder."""
    def __init__(self, action_count, hidden=64):
        super().__init__()
        from dl_push.certificate.full_m_control import ModeEncoder
        self.encoder = ModeEncoder(hidden)
        self.net = nn.Sequential(nn.Linear(2*hidden+44, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, action_count))

    def forward(self, data, model):
        cfg = model.cfg
        ids = data["shape"]
        condition = self.encoder(model.C_encoded[ids], model.valid[ids], ell=cfg.length_scale)
        state = torch.cat([model.coordinates(data["q"]), data["pusher"]/cfg.length_scale,
            torch.nn.functional.one_hot(data["mode"]+1, 13).to(data["q"]),
            data["tau"].clamp(0, cfg.max_seconds)/cfg.max_seconds, data["reachable"].to(data["q"]),
            data["remaining"][:, None]/cfg.max_seconds, data["contact"][:, None].to(data["q"])], -1)
        return self.net(torch.cat([condition, state], -1))


def semi_markov_target(reward, next_soft_value, terminated, truncated, dt, gamma, seconds):
    """Only true terminal stops bootstrap; truncation keeps the pre-reset successor.

    ``truncated`` is intentionally explicit even though it does not enter the
    mask.  Time is actual consumed time, including transfer and goal hold.
    """
    if terminated.shape != truncated.shape or reward.shape != dt.shape:
        raise ValueError("Transition tensors must have matching shapes")
    return reward + torch.pow(torch.as_tensor(gamma, dtype=dt.dtype, device=dt.device), dt/seconds) * (
        ~terminated).to(reward) * next_soft_value


class PushBackend:
    family = "push"
    algorithm = "M_induced_discrete_SAC"
    capabilities = {"representation_warmup": False, "evaluator_pretraining": True,
                    "representation_updates": "actor encoder updated inside certificate step",
                    "physical_action": "discrete_mode_and_travel", "physical_model": "calibrated_reduced_response"}

    def __init__(self, config=None, output=None, event=None, deadline=None):
        self.config = config if isinstance(config, PushConfig) else PushConfig(**(config or {}))
        self.output = Path(output or ROOT / "artifacts/unified_push_unspecified")
        self.event = event or (lambda kind, **data: None)
        self.deadline = deadline
        self.initialized = False
        self.counts = dict(transitions=0, evaluator_updates=0, certificate_updates=0,
                           representation_updates=0, alpha_updates=0, target_updates=0)
        self.replay, self.episodes, self.history, self.probe_records = [], [], [], []

    def _guard(self):
        if self.deadline is not None and time.time() >= float(self.deadline):
            raise TimeoutError("Push backend deadline reached")
        if any((parent/"STOP").exists() for parent in [self.output, *list(self.output.parents)[:2]]):
            raise InterruptedError("Push STOP requested")

    def source_files(self):
        paths = [Path(__file__), ROOT/"stabilization-gym/stabilization_gym/tasks_v2/push.py",
                 ROOT/"stabilization-learning-control/stabilization_learning/joint_v2/push.py"]
        if self.initialized:
            for name in ("so101_timeaware.environment", "so101_timeaware.policy", "so101_timeaware.timing",
                         "so101_multishape", "dl_push.certificate.full_m_control"):
                paths.append(Path(importlib.import_module(name).__file__))
        return [str(path.resolve()) for path in paths]

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
        self.rng = np.random.default_rng(cfg.seed+802810)
        self.vendor = vendor_modules()
        self.bank = self.vendor["Bank"](cfg.time_bank)
        if set(cfg.train_shapes+cfg.validation_shapes+cfg.test_shapes)-set(self.bank.names):
            raise ValueError("Shape absent from declared calibration bank")
        self.costs = JointPushCosts()
        self.model = self.vendor["Model"](self.bank, cfg.device)
        self.actor = self.vendor["Actor"](self.vendor["ActorConfig"]()).to(cfg.device)
        self.initial_actor = deepcopy(self.actor.state_dict())
        self.action_count = 12*len(self.bank.config.travels)
        self.q1 = CandidateQ(self.action_count).to(cfg.device)
        self.q2 = CandidateQ(self.action_count).to(cfg.device)
        self.target1, self.target2 = deepcopy(self.q1), deepcopy(self.q2)
        for target in (self.target1, self.target2):
            target.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.initial_alpha), device=cfg.device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=cfg.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_learning_rate)
        self.envs = [JointPushEnv(self.bank, list(cfg.train_shapes), cfg.seed+1000+i, self.costs)
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

    def _distribution(self, data, *, actor=None, training=False):
        cand = self.model.candidates(self.actor if actor is None else actor, data, training=training)
        physical = cand["physical"]
        valid = physical.any(-1)
        mask = physical.clone()
        mask[~valid, 0] = True  # Dummy categorical only; execution must abort if invalid.
        logits = cand["logits"].masked_fill(~mask, -1e9)
        logits[~valid, 0] = 0.
        logp = torch.log_softmax(logits, -1)
        probability = logp.exp().masked_fill(~mask, 0.)
        return probability, logp, valid, cand

    def collect(self, initial=False):
        self._guard()
        started = time.perf_counter()
        cfg = self.config
        steps = cfg.initial_steps if initial else cfg.collection_steps
        completed = []
        before = self.counts["transitions"]
        for _ in range(steps):
            self._guard()
            observations = [env.observation() for env in self.envs]
            data = self.model.batch(observations)
            probability, _, valid, cand = self._distribution(data)
            indices = Categorical(probs=probability.detach()).sample()
            for i, env in enumerate(self.envs):
                at_goal = self.vendor["reached"](env.plant.spec, env.plant.q)
                policy = bool(valid[i]) and not at_goal
                successor, reward, term, trunc, info = env.step(int(indices[i])) if bool(valid[i]) or at_goal else env.abort()
                self.replay.append(dict(observation=deepcopy(observations[i]), successor=deepcopy(successor),
                    action=int(indices[i]), reward=reward*self.bank.config.reward_scale,
                    terminated=bool(term), truncated=bool(trunc), dt=float(info["dt"]), policy=policy,
                    behavior_version=self.counts["certificate_updates"]))
                self.counts["transitions"] += 1
                if term or trunc:
                    row = dict(shape=env.plant.name, success=bool(info["success"]), reason=info["reason"],
                               seconds=float(env.plant.elapsed), reward=float(env.total_reward),
                               breakdown=dict(env.total_breakdown), switches=env.plant.switches)
                    completed.append(row)
                    self.episodes.append(row)
                    env.reset()
            if len(self.replay) > cfg.replay_capacity:
                del self.replay[:-cfg.replay_capacity]
        result = dict(new_transitions=self.counts["transitions"]-before, replay_size=len(self.replay),
                      completed_episodes=len(completed), training_success_rate=None if not completed else
                      float(np.mean([r["success"] for r in completed])), wall_seconds=time.perf_counter()-started,
                      initial=bool(initial), counts=dict(self.counts))
        self.event("push_collection", **result)
        return result

    def warmup_representation(self, updates):
        if int(updates) != 0:
            raise ValueError("Push has no independent representation pretraining loss; set updates=0")
        return dict(skipped=True, reason="No standalone shape reconstruction target; actor encoder learns with SAC", updates=0)

    def _batch(self):
        if not self.replay:
            raise RuntimeError("Collect real transitions before learning")
        ids = self.rng.integers(len(self.replay), size=self.config.batch_size)
        rows = [self.replay[int(i)] for i in ids]
        batch = {"data": self.model.batch([r["observation"] for r in rows]),
                 "next": self.model.batch([r["successor"] for r in rows])}
        for key in ("action", "reward", "terminated", "truncated", "dt", "policy"):
            dtype = torch.long if key == "action" else torch.bool if key in ("terminated", "truncated", "policy") else torch.float32
            batch[key] = torch.as_tensor([r[key] for r in rows], dtype=dtype, device=self.config.device)
        return batch

    def _critic_update(self):
        self._guard()
        batch = self._batch()
        # CandidateModel needs local D derivatives even in inference; detach the
        # computed policy before the target so target gradients never leak.
        probability, logp, valid, _ = self._distribution(batch["next"])
        with torch.no_grad():
            q_next = torch.minimum(self.target1(batch["next"], self.model), self.target2(batch["next"], self.model))
            soft_value = (probability.detach()*(q_next-self.log_alpha.exp().detach()*logp.detach())).sum(-1)
            # An invalid next state has a deterministic abort with the unchanged
            # task failure cost, not a fictitious sampled invalid push.
            soft_value = torch.where(valid, soft_value,
                soft_value.new_full(soft_value.shape, -self.costs.failure_penalty*self.bank.config.reward_scale))
            target = semi_markov_target(batch["reward"], soft_value, batch["terminated"], batch["truncated"],
                batch["dt"], self.bank.config.gamma, self.bank.config.discount_reference_seconds)
        q1 = self.q1(batch["data"], self.model).gather(1, batch["action"][:, None]).squeeze(1)
        q2 = self.q2(batch["data"], self.model).gather(1, batch["action"][:, None]).squeeze(1)
        loss = (q1-target).square().mean()+(q2-target).square().mean()
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite twin Q update")
        self.q_optimizer.step()
        with torch.no_grad():
            for source, destination in ((self.q1, self.target1), (self.q2, self.target2)):
                for parameter, target_parameter in zip(source.parameters(), destination.parameters()):
                    target_parameter.lerp_(parameter, self.config.target_tau)
        self.counts["evaluator_updates"] += 1
        self.counts["target_updates"] += 1
        return dict(q_loss=float(loss.detach()), q_gradient_norm=float(norm), target_mean=float(target.mean()))

    def _actor_update(self):
        self._guard()
        batch = self._batch()
        self.actor_optimizer.zero_grad(set_to_none=True)
        probability, logp, valid, cand = self._distribution(batch["data"], training=True)
        used = batch["policy"] & valid
        if not used.any():
            return dict(actor_skipped=True, actor_loss=0., entropy=0.)
        with torch.no_grad():
            q = torch.minimum(self.q1(batch["data"], self.model), self.q2(batch["data"], self.model))
        loss = (probability*(self.log_alpha.exp().detach()*logp-q)).sum(-1)[used].mean()
        loss.backward()
        encoder_norm = torch.sqrt(sum(p.grad.detach().square().sum() for p in self.actor.encoder.parameters() if p.grad is not None))
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite M-induced SAC actor update")
        self.actor_optimizer.step()
        entropy = -(probability.detach()*logp.detach()).sum(-1)
        target_entropy = self.config.entropy_fraction*cand["physical"].sum(-1).clamp_min(1).to(entropy).log()
        alpha_loss = (self.log_alpha.exp()*(entropy-target_entropy).detach())[used].mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.counts["certificate_updates"] += 1
        self.counts["representation_updates"] += 1
        self.counts["alpha_updates"] += 1
        self._probe("certificate_update")
        return dict(actor_loss=float(loss.detach()), actor_gradient_norm=float(norm),
                    shape_encoder_gradient_norm=float(encoder_norm), entropy=float(entropy[used].mean()),
                    alpha=float(self.log_alpha.exp().detach()), alpha_loss=float(alpha_loss.detach()), actor_skipped=False)

    def pretrain_evaluator(self, updates):
        before = _module_hash(self.actor)
        started = time.perf_counter()
        stats = [self._critic_update() for _ in range(int(updates))]
        if before != _module_hash(self.actor):
            raise RuntimeError("Critic pretraining changed actor or its shared representation")
        result = dict(updates=len(stats), actor_unchanged=True, wall_seconds=time.perf_counter()-started,
                      counts=dict(self.counts), statistics=_means(stats))
        self.event("push_evaluator_pretrained", **result)
        return result

    def train_epoch(self, epoch, representation_updates=0, evaluator_updates=64, certificate_updates=64):
        if int(representation_updates) != 0:
            raise ValueError("Actor shape representation updates are inside the certificate/SAC step")
        started = time.perf_counter()
        q_stats, a_stats = [], []
        # This method learns only. The common Runner calls collect once first.
        for index in range(max(int(evaluator_updates), int(certificate_updates))):
            if index < int(evaluator_updates):
                q_stats.append(self._critic_update())
            if index < int(certificate_updates):
                a_stats.append(self._actor_update())
        result = dict(epoch=int(epoch), counts=dict(self.counts), q_statistics=_means(q_stats),
                      actor_statistics=_means(a_stats), wall_seconds=time.perf_counter()-started,
                      actor_parameter_changed=_module_hash(self.actor) != self.initial_actor_hash)
        self.history.append(result)
        self.event("push_epoch_trained", **result)
        return result

    def _probe(self, stage):
        with torch.no_grad():
            ids = [self.bank.names.index(name) for name in self.config.train_shapes]
            q = torch.tensor([[.01, -.008, .2], [-.006, .004, -.25], [.002, .002, .05]], device=self.config.device)
            D = self.model.coordinates(q)[None].expand(len(ids), -1, -1)
            condition = self.actor.encoder(self.model.C_encoded[ids], self.model.valid[ids], ell=self.model.cfg.length_scale)
            _, matrix = self.actor.conditioned(D, condition)
        record = dict(stage=stage, step=self.counts["certificate_updates"], counts=dict(self.counts),
                      shapes=list(self.config.train_shapes), physical_q=q.cpu().tolist(), M=matrix.cpu().tolist())
        self.probe_records.append(record)
        with (self.output/"push_fixed_step_probes.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")

    def validate(self):
        report = self.evaluate("validation", ("selected",))
        result = report["variants"]["selected"]
        return dict(selection_score=[result["success_rate"], -result["mean_combined_cost"]], **result)

    def evaluate(self, split, variants=("selected", "initial", "identity")):
        if split not in ("validation", "test", "train"):
            raise ValueError("Unknown evaluation split")
        if split == "test":
            marker = self.output/"FROZEN.json"
            group = self.output.parents[1]/"GLOBAL_FROZEN.json"
            if not marker.is_file() or not group.is_file():
                raise RuntimeError("Held-out evaluation requires local and global freeze barriers")
            frozen = json.loads(marker.read_text(encoding="utf-8"))
            frozen_group = json.loads(group.read_text(encoding="utf-8"))
            if frozen_group.get("schema") != "unified_frozen_group_v1" or frozen_group.get("jobs", {}).get(self.output.name) != _sha(marker):
                raise ValueError("Job absent from intact all-backends freeze barrier")
            selected_path = (self.output/frozen["checkpoint"]).resolve()
            if not selected_path.is_relative_to(self.output.resolve()) or _sha(selected_path) != frozen["sha256"]:
                raise ValueError("Frozen selected checkpoint changed or escaped job directory")
            selected = torch.load(selected_path, map_location="cpu", weights_only=False)
            current = deepcopy(self.actor).cpu()
            current.load_state_dict(selected["actor"])
            if _module_hash(current) != _module_hash(self.actor):
                raise ValueError("Load the frozen selected checkpoint before held-out evaluation")
        from stabilization_learning.joint_v2.push import evaluate as original_evaluate
        names = getattr(self.config, {"validation": "validation_shapes", "test": "test_shapes", "train": "train_shapes"}[split])
        repeats = self.config.test_repeats if split == "test" else self.config.validation_repeats
        start = self.config.test_seed_start if split == "test" else self.config.validation_seed_start
        if split == "train":
            start = self.config.seed+2900000
        before_hash = _module_hash(self.actor)
        reports = {}
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
            report = original_evaluate(actor, self.model, names, range(start, start+repeats), self.costs, self._guard)
            rows = report["episodes"]
            report.update(mean_input_square=float(np.mean([r["input_square_integral"] for r in rows])),
                mean_input_cost=float(np.mean([r["input_square_integral"] for r in rows])),
                mean_cost=report["mean_combined_cost"], mean_integrated_error=report["mean_error"],
                num_episodes=len(rows), successes=sum(r["success"] for r in rows),
                mean_switches=float(np.mean([r["switches"] for r in rows])), failures=sum(not r["success"] for r in rows),
                failure_reasons={reason:sum(r["reason"] == reason for r in rows) for reason in sorted({r["reason"] for r in rows if not r["success"]})},
                V_increase_fraction=float(np.mean([t["V_next"] > t["V"]+1e-8 for r in rows for t in r["trace"]])),
                model_sha256=_module_hash(actor), feature_sha256=_module_hash(actor.encoder),
                metric_override="identity" if variant in ("identity", "identity_m") else "learned",
                split=split, cases=len(rows))
            reports[variant] = report
        if _module_hash(self.actor) != before_hash:
            raise RuntimeError("Evaluation mutated actor")
        paired = {}
        if "selected" in reports:
            selected = {(r["shape"], r["seed"]):r for r in reports["selected"]["episodes"]}
            for name, report in reports.items():
                if name == "selected":
                    continue
                pairs = [(selected[(r["shape"], r["seed"])], r) for r in report["episodes"]
                         if r["success"] and selected[(r["shape"], r["seed"])]["success"]]
                paired[name] = dict(common_successes=len(pairs),
                    selected_minus_comparator_seconds=None if not pairs else float(np.mean([a["seconds"]-b["seconds"] for a,b in pairs])))
        return dict(family=self.family, algorithm=self.algorithm, split=split, variants=reports,
                    paired_time_on_common_success=paired, selection_allowed=split == "validation",
                    scope="single-seed integration pilot; calibrated response/time bank; no robot simulation claim")

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError("Checkpoint snapshots are immutable")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema="unified_push_discrete_sac_v1", config=asdict(self.config), actor=self.actor.state_dict(),
            initial_actor=self.initial_actor, q1=self.q1.state_dict(), q2=self.q2.state_dict(),
            target1=self.target1.state_dict(), target2=self.target2.state_dict(), log_alpha=self.log_alpha.detach(),
            actor_optimizer=self.actor_optimizer.state_dict(), q_optimizer=self.q_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict(), counts=dict(self.counts), replay=self.replay,
            environments=[env.state_dict() for env in self.envs], episodes=self.episodes, history=self.history,
            probe_records=self.probe_records, rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), bank_sha256=self.bank.sha256,
            initial_actor_sha256=self.initial_actor_hash, initial_q_sha256=self.initial_q_hashes,
            cuda_rng=torch.cuda.get_rng_state_all() if self.config.device.startswith("cuda") else [])
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return dict(path=str(path.resolve()), sha256=_sha(path), counts=dict(self.counts))

    def load(self, path):
        payload = torch.load(path, map_location=self.config.device, weights_only=False)
        if payload["schema"] != "unified_push_discrete_sac_v1" or payload["bank_sha256"] != self.bank.sha256:
            raise ValueError("Checkpoint schema or bank mismatch")
        if payload["config"] != asdict(self.config):
            raise ValueError("Checkpoint configuration mismatch")
        for name in ("actor", "q1", "q2", "target1", "target2", "actor_optimizer", "q_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"])
        self.counts, self.replay = dict(payload["counts"]), payload["replay"]
        self.initial_actor = payload["initial_actor"]
        self.initial_actor_hash = payload["initial_actor_sha256"]
        self.initial_q_hashes = payload["initial_q_sha256"]
        self.episodes, self.history, self.probe_records = payload["episodes"], payload["history"], payload["probe_records"]
        for env, state in zip(self.envs, payload["environments"]):
            env.plant.load_state_dict(state["plant"])
            env.total_reward, env.total_breakdown = state["total_reward"], state["total_breakdown"]
        self.rng.bit_generator.state = payload["rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return dict(path=str(Path(path).resolve()), counts=dict(self.counts), actor_sha256=_module_hash(self.actor))


def _means(rows):
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}
