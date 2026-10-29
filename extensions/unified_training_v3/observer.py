"""New-run observer training boundaries; preserve full-sequence learning math."""
from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch

from stabilization_gym.observation.environment import SCENARIOS
from extensions.unified_training_v1.backends import observer as legacy
from extensions.observer_learning_v1.learning import episode_data, batch_data, rollout
from .guards import NumericalGuard


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid integer observer setting: {name}")
    return value


def validate_config(config):
    """Reject invalid declarations before creating output or generating episodes."""
    cfg = dict(config)
    name = cfg.get("task", "motor_speed")
    if name not in legacy.TASKS:
        raise ValueError("Unknown observer task")
    if cfg.get("device", "cpu") != "cpu":
        raise ValueError("Observer backend currently requires device='cpu'")
    for key, default in (("batch_size", 2), ("hidden", 32), ("cpu_threads", 1)):
        _integer(cfg.get(key, default), key, 1)
    _integer(cfg.get("seed", 81), "seed", 0, 2**64 - 1)
    for key, default in (("learning_rate", .003), ("gradient_clip", 2.), ("metric_radius", 3.)):
        value = cfg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Positive finite observer setting required: {key}")
    radius = cfg.get("metric_radius", 3.)
    if 2 * radius >= math.log(np.finfo(np.float64).max):
        raise ValueError("Observer metric_radius has nonfinite declared bounds")
    for split in ("training", "validation", "test"):
        start = _integer(cfg.get(split + "_index_start", 60000), split + "_index_start", 0, 99999)
        count = cfg.get("batch_size", 2) if split == "training" else cfg.get(split + "_count", 4 if split == "validation" else 8)
        _integer(count, split + "_count", 1)
        if start + count > 100000:
            raise ValueError(f"Observer {split} episode indices exceed split namespace")
        scenarios = cfg.get("scenarios", ("nominal", "noise", "occlusion", "mismatch"))
        if split != "training":
            scenarios = cfg.get(split + "_scenarios", scenarios)
        if not isinstance(scenarios, (tuple, list)) or not scenarios:
            raise ValueError("Observer scenarios must be a nonempty sequence")
        for scenario in scenarios:
            if scenario not in SCENARIOS or (scenario in {"degenerate", "wrong_match"} and name != "cooperative_pose"):
                raise ValueError(f"Unsupported observer task/scenario combination: {name}/{scenario}")
    return cfg


class StableObserverBackend(legacy.Backend):
    """Fail permanently on corrupt batches, optimizer state, or metric output."""
    def __init__(self, config, output, event=None, deadline=None):
        super().__init__(validate_config(config), output, event=event, deadline=deadline)
        self.numerics = NumericalGuard("observer " + self.name)

    def guard(self):
        self.numerics.guard()
        return super().guard()

    def source_files(self):
        return super().source_files() + [str(Path(__file__).resolve()),
            str(Path(__file__).with_name("guards.py").resolve()),
            str(Path(__file__).with_name("numerical.py").resolve())]

    def initialize(self):
        result = super().initialize()
        self.numerics.watch_module(self.network, "metric network")
        self.numerics.watch_optimizer(self.optimizer, "Adam")
        self.numerics.check(self.validation_batch, "validation batch")
        self.numerics.check(self.network.bounds, "metric spectral bounds")
        if not 0 < self.network.bounds[0] <= self.network.bounds[1]:
            raise ValueError("Observer metric has invalid declared spectral bounds")
        self.numerics.check_all()
        return result

    def _episodes(self, episodes, *, split):
        if not episodes:
            raise ValueError("Observer update requires a nonempty full batch")
        indices = []
        spec = self.model.spec
        length = round(spec.horizon / spec.dt) + 1
        shapes = {"truth": (length, spec.state_dim), "measurement": (length, spec.measurement_dim),
                  "mask": (length, spec.measurement_dim), "control": (length, spec.input_dim),
                  "initial": (spec.state_dim,)}
        for episode in episodes:
            self.numerics.check(episode, "episode data")
            if episode.get("split") != split:
                raise ValueError("Observer episode belongs to a different data split")
            index = _integer(episode.get("index"), "episode index", 0, 99999)
            indices.append(index)
            if episode.get("scenario") not in self.scenarios and split == "train":
                raise ValueError("Observer training episode has undeclared scenario")
            for key, expected in shapes.items():
                if np.shape(episode.get(key)) != expected:
                    raise ValueError(f"Observer episode {key} does not cover the declared full horizon")
            mask = np.asarray(episode["mask"])
            if not np.isin(mask, (0, 1)).all():
                raise ValueError("Observer measurement mask must be binary")
        if len(indices) != len(set(indices)):
            raise ValueError("Observer batch contains repeated trajectory indices")

    def _fresh(self):
        self.guard()
        count = self.config.get("batch_size", 2)
        # Reserve the whole batch before generating anything. Exhaustion must
        # not advance one index and then fail midway through the next episode.
        if not 0 <= self.train_index <= 100000 - count:
            raise RuntimeError("Observer training split exhausted; no index recycling or budget extension")
        episodes = []
        for index in range(self.train_index, self.train_index + count):
            self.guard()
            scenario = self.scenarios[index % len(self.scenarios)]
            episodes.append(episode_data(self.name, "train", index, scenario))
        self._episodes(episodes, split="train")
        self.train_index += count
        self.collected_transitions += sum(len(episode["truth"]) - 1 for episode in episodes)
        return episodes

    def _rollout(self, batch, label):
        try:
            result = rollout(self.network, self.tensor_model, batch)
        except torch.linalg.LinAlgError as error:
            # A singular/nonfinite metric solve is a failed training trajectory;
            # its current parameters must not subsequently be published as healthy.
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
            batch = batch_data(self.model, episodes)
            self.numerics.check(batch, "training batch")
            before = legacy._digest(self.network)
            self.optimizer.zero_grad(set_to_none=True)
            result = self._rollout(batch, "full training rollout")
            result["loss"].backward()
            for parameter in self.network.parameters():
                if parameter.grad is None:
                    error = FloatingPointError("Observer parameter has no training gradient")
                    self.numerics.fail(error)
                    raise error
                self.numerics.check(parameter.grad, "training gradient")
            # Finite gradient elements can still overflow the accumulated norm.
            # Reject that case before clipping silently turns the update to zero.
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
                mean_cost=float(result["cost"].mean().detach()), certificate_loss=float(result["certificate_loss"].detach()),
                gradient_norm=float(norm), parameters_changed=before != legacy._digest(self.network),
                collected_transitions=self.collected_transitions, wall_seconds=time.perf_counter() - started,
                fixed_public_estimate_M=probe.tolist(),
                training_episodes=[dict(index=e["index"], scenario=e["scenario"], replay_sha256=e["replay_sha256"]) for e in episodes])
            self.numerics.check(row, "training update record")
            rows.append(row)
            self.history.append(row)
            self.event("certificate_update", task=self.name, **row)
        return dict(epoch=epoch, actual_updates=self.updates, certificate_updates=len(rows),
                    representation_updates=0, evaluator_updates=0, rows=rows)

    def validate(self):
        self.guard()
        self.numerics.check(self.validation_batch, "validation batch")
        with torch.no_grad():
            result = self._rollout(self.validation_batch, "full validation rollout")
        rows = [dict(index=i, scenario=s, success=bool(result["success"][k]),
            cost=float(result["cost"][k]), terminal_mse=float(result["terminal_mse"][k]))
            for k, (i, s) in enumerate(self.validation_descriptors)]
        summary = dict(episodes=len(rows), successes=sum(r["success"] for r in rows),
            success_rate=float(np.mean([r["success"] for r in rows])), mean_cost=float(result["cost"].mean()),
            mean_terminal_error=float(result["terminal_mse"].mean()), certificate_loss=float(result["certificate_loss"]))
        return dict(summary=summary, rows=rows, selection_score=[summary["successes"], -summary["mean_cost"]],
                    parameter_hash=legacy._digest(self.network))

    def save(self, path):
        self.guard()
        self.numerics.check_all()
        self.numerics.check((self.initial_state, self.pending, self.history), "checkpoint data")
        temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError("Preserve interrupted checkpoint temporary file: " + str(temporary))
        return super().save(path)

    def load(self, path):
        self.guard()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"config", "network", "optimizer", "initial_state", "updates", "train_index",
                    "collected_transitions", "pending", "history", "torch_rng", "parameter_hash"}
        if not isinstance(payload, dict) or not required.issubset(payload) or payload["config"] != self.config:
            raise ValueError("Observer checkpoint keys or configuration mismatch")
        self.numerics.check(payload, "checkpoint payload")
        staged = deepcopy(self.network)
        staged.load_state_dict(payload["network"])
        if legacy._digest(staged) != payload["parameter_hash"]:
            raise ValueError("Observer checkpoint parameter digest mismatch")
        staged.load_state_dict(payload["initial_state"])
        if set(payload["initial_state"]) != set(self.initial_state) or any(
                not torch.equal(value, self.initial_state[key]) for key, value in payload["initial_state"].items()):
            raise ValueError("Observer checkpoint initial reference differs from declared initialization")
        staged_optimizer = torch.optim.Adam(staged.parameters(), lr=self.config.get("learning_rate", .003))
        staged_optimizer.load_state_dict(payload["optimizer"])
        expected_groups = [{key: value for key, value in group.items() if key != "params"}
                           for group in self.optimizer.param_groups]
        restored_groups = [{key: value for key, value in group.items() if key != "params"}
                           for group in staged_optimizer.param_groups]
        if restored_groups != expected_groups:
            raise ValueError("Observer checkpoint changes declared optimizer settings")
        for group in staged_optimizer.param_groups:
            for parameter in group["params"]:
                state = staged_optimizer.state.get(parameter, {})
                if state:
                    if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                        raise ValueError("Observer checkpoint has incomplete Adam state")
                    if state["step"].numel() != 1 or bool(state["step"] < 0):
                        raise ValueError("Observer checkpoint has invalid Adam step")
                    for key in ("exp_avg", "exp_avg_sq"):
                        if state[key].shape != parameter.shape:
                            raise ValueError("Observer checkpoint has incompatible Adam moment shape")
        for key in ("updates", "collected_transitions"):
            _integer(payload[key], key)
        start = self.config.get("training_index_start", 60000)
        _integer(payload["train_index"], "train_index", start, 100000)
        if not isinstance(payload["pending"], list) or not isinstance(payload["history"], list):
            raise ValueError("Observer checkpoint pending/history must be lists")
        if len(payload["history"]) != payload["updates"]:
            raise ValueError("Observer checkpoint update count disagrees with history")
        if payload["pending"]:
            self._episodes(payload["pending"], split="train")
            if any(not start <= e["index"] < payload["train_index"] for e in payload["pending"]):
                raise ValueError("Observer checkpoint pending episode index is outside collected history")
        expected_transitions = (payload["train_index"] - start) * round(self.model.spec.horizon / self.model.spec.dt)
        if payload["collected_transitions"] != expected_transitions:
            raise ValueError("Observer checkpoint collected transition count is inconsistent")
        # Validate RNG shape/type before changing the process-wide generator.
        torch.Generator(device="cpu").set_state(payload["torch_rng"])
        try:
            self.network.load_state_dict(payload["network"])
            self.optimizer.load_state_dict(payload["optimizer"])
            for key in ("initial_state", "updates", "train_index", "collected_transitions", "pending", "history"):
                setattr(self, key, payload[key])
            torch.set_rng_state(payload["torch_rng"])
            self.numerics.check_all()
        except BaseException as error:
            self.numerics.fail(error)
            raise
        return dict(parameter_hash=legacy._digest(self.network), actual_updates=self.updates)

    def evaluate(self, split, variants=("selected", "initial", "identity")):
        self.guard()
        self.numerics.check_all()
        result = super().evaluate(split, variants)
        self.numerics.check(result, "evaluation result")
        return result

    def close(self):
        self.numerics.close()
