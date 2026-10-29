"""Reproducible observer semantics for newly declared v4 experiments only."""
import os
from pathlib import Path
import time

import torch

from extensions.unified_training_v1.backends.observer import _digest
from extensions.unified_training_v3.observer import StableObserverBackend, _integer, validate_config
from extensions.observer_learning_v1.learning import episode_data, batch_data
from .observer_model import ScenarioTensorModel, corrected_rollout


SEMANTICS = dict(schema="corrected_observer_semantics_v4", geometry="declared per-episode nominal or degenerate geometry",
    success="final hold AND no exact posterior/truth domain or pose chart failure in complete history",
    training_certificate="delta(V / mean_reference_eigenvalue) + 0.1*dt*normalized_error_squared",
    training_noise_slack="0 for nominal; fixed 0.01*dt for all other declared scenarios; not a disturbance proof",
    evaluation_certificate="raw full-successor V difference + 0.1*dt*normalized_error_squared",
    training_cost="full-sequence differentiable error integral with explicit domain surrogate",
    selection_cost="deployment prefix integral plus first-failure unobserved-tail penalty; floor=25/sec")


class CorrectedObserverBackend(StableObserverBackend):
    def source_files(self):
        return super().source_files() + [str(Path(__file__).resolve()),
            str(Path(__file__).with_name("observer_model.py").resolve())]

    def _make_batch(self, episodes):
        batch = batch_data(self.model, episodes)
        batch["scenarios"] = tuple(episode["scenario"] for episode in episodes)
        return batch

    def initialize(self):
        result = super().initialize()
        episodes = [episode_data(self.name, "validation", index, scenario)
                    for index, scenario in self.validation_descriptors]
        self._episodes(episodes, split="validation")
        self.validation_batch = self._make_batch(episodes)
        self.numerics.check(self.validation_batch, "scenario-aware validation batch")
        return dict(result, observer_semantics=SEMANTICS)

    def _rollout(self, batch, label):
        if "scenarios" not in batch or len(batch["scenarios"]) != len(batch["initial"]):
            raise ValueError("Observer batch must bind one declared geometry/scenario per episode")
        try:
            model = ScenarioTensorModel(self.model, batch["scenarios"])
            result = corrected_rollout(self.network, model, batch)
        except torch.linalg.LinAlgError as error:
            self.numerics.fail(error)
            raise
        self.numerics.check(result, label)
        return result

    def train_epoch(self, epoch, representation_updates=0, evaluator_updates=0, certificate_updates=4):
        _integer(epoch, "epoch")
        _integer(certificate_updates, "certificate_updates")
        if representation_updates != 0 or evaluator_updates != 0:
            raise ValueError("Observer has no independent representation or evaluator update")
        rows = []
        for _ in range(certificate_updates):
            self.guard()
            started = time.perf_counter()
            count = self.config.get("batch_size", 2)
            if self.pending:
                if len(self.pending) < count:
                    raise ValueError("Observer pending data cannot provide a full declared batch")
                episodes = self.pending[:count]
                self._episodes(episodes, split="train")
                self.pending = self.pending[count:]
            else:
                episodes = self._fresh()
            batch = self._make_batch(episodes)
            self.numerics.check(batch, "scenario-aware training batch")
            before = _digest(self.network)
            self.optimizer.zero_grad(set_to_none=True)
            result = self._rollout(batch, "full training rollout")
            result["loss"].backward()
            for parameter in self.network.parameters():
                if parameter.grad is None:
                    error = FloatingPointError("Observer parameter has no training gradient")
                    self.numerics.fail(error)
                    raise error
                self.numerics.check(parameter.grad, "training gradient")
            try:
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                    float(self.config.get("gradient_clip", 2.)), error_if_nonfinite=True)
            except RuntimeError as error:
                if "non-finite" not in str(error):
                    raise
                failure = FloatingPointError("Nonfinite observer total gradient norm")
                self.numerics.fail(failure)
                raise failure from error
            self.numerics.check(norm, "total gradient norm")
            self.guard()
            self.optimizer.step()
            with torch.no_grad():
                probe = self.network(torch.as_tensor(self.model.initial_estimate(), dtype=torch.float64))
            self.numerics.check(probe, "fixed public estimate metric")
            self.updates += 1
            row = dict(epoch=epoch, update=self.updates, loss=float(result["loss"].detach()),
                mean_cost=float(result["cost"].mean().detach()),
                mean_training_full_sequence_cost=float(result["training_full_sequence_cost"].mean().detach()),
                exact_domain_failures=int(result["failed"].sum()),
                certificate_loss=float(result["certificate_loss"].detach()), gradient_norm=float(norm),
                parameters_changed=before != _digest(self.network), collected_transitions=self.collected_transitions,
                wall_seconds=time.perf_counter()-started, fixed_public_estimate_M=probe.tolist(),
                training_episodes=[dict(index=e["index"], scenario=e["scenario"], replay_sha256=e["replay_sha256"]) for e in episodes])
            self.numerics.check(row, "training update record")
            rows.append(row)
            self.history.append(row)
            self.event("certificate_update", task=self.name, **row)
        return dict(epoch=epoch, actual_updates=self.updates, certificate_updates=len(rows),
                    representation_updates=0, evaluator_updates=0, rows=rows)

    def validate(self):
        result = super().validate()
        result["observer_semantics"] = SEMANTICS
        return result

    def save(self, path):
        self.guard()
        self.numerics.check_all()
        self.numerics.check((self.initial_state, self.pending, self.history), "checkpoint data")
        path = Path(path)
        temporary = path.with_suffix(path.suffix+".tmp")
        if path.exists() or temporary.exists():
            raise FileExistsError("Never replace an observer checkpoint or interrupted temporary: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(config=self.config, network=self.network.state_dict(), optimizer=self.optimizer.state_dict(),
            initial_state=self.initial_state, updates=self.updates, train_index=self.train_index,
            collected_transitions=self.collected_transitions, pending=self.pending, history=self.history,
            torch_rng=torch.get_rng_state(), parameter_hash=_digest(self.network), observer_semantics=SEMANTICS)
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes only if the destination is still absent (no overwrite race).
        os.link(temporary, path)
        temporary.unlink()
        return dict(path=str(path), parameter_hash=payload["parameter_hash"], actual_updates=self.updates)

    def load(self, path):
        self.guard()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("observer_semantics") != SEMANTICS:
            raise ValueError("Observer checkpoint uses a different geometry/domain/loss protocol")
        return super().load(path)

    def evaluate(self, split, variants=("selected", "initial", "identity")):
        return dict(super().evaluate(split, variants), observer_semantics=SEMANTICS)
