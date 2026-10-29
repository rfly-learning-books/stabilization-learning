"""Keep each real Push update without changing the preserved SAC mathematics."""
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch

from extensions.unified_training_v1.backends.push import PushBackend, _sha, _module_hash


class RecordedPushBackend(PushBackend):
    def _record_update(self, kind, result):
        if not hasattr(self, "_update_stream"):
            self._update_stream = (self.output / "push_all_updates.jsonl").open("x", encoding="utf-8", buffering=1)
            self._update_sequence = 0
        self._update_sequence += 1
        record = dict(sequence=self._update_sequence, unix=time.time(), kind=kind,
                      stage=getattr(self, "_record_stage", "unknown"),
                      epoch=getattr(self, "_record_epoch", None),
                      counts=dict(self.counts), result=result)
        self._update_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._update_stream.flush()

    def _critic_update(self):
        if getattr(self, "_evaluation_only", False):
            raise RuntimeError("Compact snapshot is evaluation-only; no resumed training")
        result = super()._critic_update()
        self._record_update("critic", result)
        return result

    def _actor_update(self):
        if getattr(self, "_evaluation_only", False):
            raise RuntimeError("Compact snapshot is evaluation-only; no resumed training")
        result = super()._actor_update()
        self._record_update("actor", result)
        return result

    def pretrain_evaluator(self, updates):
        self._record_stage, self._record_epoch = "pretrain_evaluator", None
        return super().pretrain_evaluator(updates)

    def train_epoch(self, epoch, **kwargs):
        self._record_stage, self._record_epoch = "train_epoch", int(epoch)
        return super().train_epoch(epoch, **kwargs)

    def save(self, path):
        """Save immutable candidate weights without replay or physical state.

        The live backend remains resident throughout the suite. These snapshots
        are only for selecting and evaluating a frozen candidate, never resume.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError("Checkpoint snapshots are immutable")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema="unified_push_evaluation_only_v2", config=asdict(self.config),
            actor=self.actor.state_dict(), initial_actor=self.initial_actor,
            q1=self.q1.state_dict(), q2=self.q2.state_dict(),
            target1=self.target1.state_dict(), target2=self.target2.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), q_optimizer=self.q_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict(), log_alpha=self.log_alpha.detach(),
            counts=dict(self.counts), bank_sha256=self.bank.sha256,
            initial_actor_sha256=self.initial_actor_hash, initial_q_sha256=self.initial_q_hashes,
            resume_training_supported=False)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        torch.save(payload, temporary)
        temporary.replace(path)
        return dict(path=str(path.resolve()), sha256=_sha(path), counts=dict(self.counts))

    def load(self, path):
        """Only restore the selected model after the complete group is frozen."""
        path = Path(path).resolve()
        marker, group = self.output / "FROZEN.json", self.output.parents[1] / "GLOBAL_FROZEN.json"
        if not marker.is_file() or not group.is_file():
            raise RuntimeError("Evaluation-only restore requires the complete group freeze")
        frozen = json.loads(marker.read_text(encoding="utf-8"))
        global_record = json.loads(group.read_text(encoding="utf-8"))
        if global_record.get("schema") != "unified_frozen_group_v1" or global_record.get("jobs", {}).get(self.output.name) != _sha(marker):
            raise RuntimeError("Push checkpoint is absent from the frozen group")
        if path != (self.output / frozen["checkpoint"]).resolve() or _sha(path) != frozen["sha256"]:
            raise RuntimeError("Only the frozen selected checkpoint can be restored")
        payload = torch.load(path, map_location=self.config.device, weights_only=False)
        if payload["schema"] != "unified_push_evaluation_only_v2" or payload["bank_sha256"] != self.bank.sha256 or payload["config"] != asdict(self.config):
            raise ValueError("Evaluation snapshot schema, bank or config mismatch")
        for name in ("actor", "q1", "q2", "target1", "target2", "actor_optimizer", "q_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"])
        self.counts = dict(payload["counts"])
        self.initial_actor = payload["initial_actor"]
        self.initial_actor_hash = payload["initial_actor_sha256"]
        self.initial_q_hashes = payload["initial_q_sha256"]
        self._evaluation_only = True
        return dict(path=str(path), counts=dict(self.counts), actor_sha256=_module_hash(self.actor), evaluation_only=True)

    def close(self):
        if hasattr(self, "_update_stream"):
            self._update_stream.close()
        parent = getattr(super(), "close", None)
        if parent:
            parent()
