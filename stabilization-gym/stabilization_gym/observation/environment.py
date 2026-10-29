"""Replayable estimation environments: the action corrects the INTERNAL estimate."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import numpy as np
import gymnasium as gym
from .models import MODEL_TYPES, PoseModel, vector


SPLIT_BASES = {"engineering": 910000, "train": 11000000, "validation": 21000000, "test": 31000000}
SCENARIOS = ("nominal", "noise", "occlusion", "mismatch", "degenerate", "bounded_noise", "initial_offset", "wrong_match")


def episode_seed(split, index):
    if split not in SPLIT_BASES or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 100000:
        raise ValueError("Use a declared split and integer trajectory index in [0,100000)")
    return SPLIT_BASES[split] + index


class ObservationEnv(gym.Env):
    """At t_k: observe predicted estimate and y_k; action is tangent correction.

    step applies that correction, scores the posterior, then advances object and
    estimate to t_{k+1}. The known input belongs to [t_k,t_{k+1}). Only explicit
    supervision()/evaluation_record() expose truth, never observation or info.
    """
    metadata = {"render_modes": []}

    def __init__(self, task):
        if task not in MODEL_TYPES:
            raise ValueError(f"Unknown observation task: {task}")
        self.task = task; self.model = MODEL_TYPES[task]()
        spec = self.model.spec
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (spec.error_dim,), dtype=np.float64)
        self.observation_space = gym.spaces.Dict({
            "measurement": gym.spaces.Box(-np.inf, np.inf, (spec.measurement_dim,), dtype=np.float64),
            "measurement_mask": gym.spaces.MultiBinary(spec.measurement_dim),
            "confidence": gym.spaces.Box(0., 1., (spec.measurement_dim,), dtype=np.float64),
            "estimate": gym.spaces.Box(-np.inf, np.inf, (spec.state_dim,), dtype=np.float64),
            "known_input": gym.spaces.Box(-np.inf, np.inf, (spec.input_dim,), dtype=np.float64),
            "time": gym.spaces.Box(0., np.inf, (1,), dtype=np.float64),
            "dt": gym.spaces.Box(0., np.inf, (1,), dtype=np.float64),
        })
        self._done = True; self._last_record = None

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)
        if set(options) - {"scenario", "split", "index"}:
            raise ValueError("Only scenario/split/index reset options are accepted")
        scenario, split, index = options.get("scenario", "nominal"), options.get("split", "engineering"), options.get("index", 0)
        expected_seed = episode_seed(split, index)
        if seed is not None and seed != expected_seed:
            raise ValueError("Seed must match the declared whole-trajectory split/index")
        if scenario not in SCENARIOS or (scenario in {"degenerate", "wrong_match"} and self.task != "cooperative_pose"):
            raise ValueError("Unsupported task/scenario combination")
        super().reset(seed=expected_seed)
        self.model = PoseModel(degenerate=scenario == "degenerate") if self.task == "cooperative_pose" else MODEL_TYPES[self.task]()
        spec = self.model.spec
        task_index = list(MODEL_TYPES).index(self.task)
        motion_rng, sensor_rng = [np.random.default_rng(s) for s in np.random.SeedSequence([expected_seed, task_index]).spawn(2)]
        count = round(spec.horizon/spec.dt)+1
        self._times = np.arange(count)*spec.dt
        phase = motion_rng.uniform(0., 2*np.pi)
        self._inputs = np.array([self.model.known_input(t, phase) for t in self._times])
        truth = [self.model.initial_truth(motion_rng)]
        for k, time in enumerate(self._times[:-1]):
            truth.append(self.model.truth_step(truth[-1], self._inputs[k], time, scenario))
        self._truth = np.asarray(truth)
        self._measurements, self._masks = [], []
        for k, time in enumerate(self._times):
            values = self.model.measure(self._truth[k]); mask = self.model.visible(self._truth[k])
            if scenario == "occlusion":
                if .4*spec.horizon <= time < .6*spec.horizon:
                    mask[:] = False
                elif self.task == "cooperative_pose":
                    hidden = (k//5) % 8; mask[2*hidden:2*hidden+2] = False
            if scenario == "bounded_noise":
                values += sensor_rng.uniform(-1., 1., spec.measurement_dim)*spec.sensor_std
            elif scenario not in {"nominal", "initial_offset"}:
                values += sensor_rng.normal(size=spec.measurement_dim) * spec.sensor_std
            if scenario == "wrong_match" and .3*spec.horizon <= time < .7*spec.horizon:
                values[:4] = values[[2, 3, 0, 1]]
            if scenario == "mismatch" and self.task == "cooperative_pose":
                values[0::2] = 320. + 1.015*(values[0::2]-320.)
            values[~mask] = 0.
            self._measurements.append(values); self._masks.append(mask)
        self._measurements = np.asarray(self._measurements); self._masks = np.asarray(self._masks)
        self._estimate = self.model.initial_estimate()  # Public prior, independent of hidden truth.
        if scenario == "initial_offset":
            offsets = {"motor_speed": [1.5, -1.5], "cartpole_output": [.15, .1, .12, -.1],
                       "cooperative_pose": [.6, -.4, .5, .4, -.2, .3, .1, -.1, .1, .1, -.1, .1]}
            self._estimate = self.model.retract(self._estimate, np.array(offsets[self.task]))
        self._index, self._done, self._last_record = 0, False, None
        self._meta = dict(task=spec.name, scenario=scenario, split=split, index=index, seed=expected_seed)
        self._meta["replay_sha256"] = hashlib.sha256(b"".join(a.tobytes() for a in
            (self._truth, self._inputs, self._measurements, self._masks))).hexdigest()
        return self._observation(), self._info()

    def _observation(self):
        k = self._index
        return {"measurement": self._measurements[k].copy(), "measurement_mask": self._masks[k].astype(np.int8),
                "confidence": self._masks[k].astype(np.float64), "estimate": self._estimate.copy(),
                "known_input": self._inputs[k].copy(), "time": np.array([self._times[k]]),
                "dt": np.array([self.model.spec.dt])}

    def _info(self):
        return dict(task=self.task, time=float(self._times[self._index]), phase="prior" if not self._done else "posterior")

    def supervision(self):
        """Privileged labels: training/evaluation only, never passed to observers."""
        return {"truth": self._truth[self._index].copy(), "episode": deepcopy(self._meta)}

    def evaluation_record(self):
        if self._last_record is None:
            raise RuntimeError("No correction has been executed")
        return deepcopy(self._last_record)

    def supervised_trajectory(self):
        """Privileged full trajectory export for OFFLINE training/evaluation.

        Never pass this mapping to a deployment policy; the public observation
        remains unchanged. Future measurements are labels, not policy features.
        """
        return {"truth": self._truth.copy(), "measurement": self._measurements.copy(),
                "mask": self._masks.copy(), "control": self._inputs.copy()}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode")
        delta = vector(action, self.model.spec.error_dim)
        observation = self._observation()
        corrected = self.model.retract(self._estimate, delta)
        truth = self._truth[self._index]
        error = self.model.error(truth, corrected)
        failed = not self.model.domain_ok(corrected) or not self.model.domain_ok(truth)
        if self.task == "cooperative_pose" and np.linalg.norm(error[3:6]) >= np.pi-.05:
            failed = True
        final = self._index == len(self._times)-1
        self._last_record = dict(time=float(self._times[self._index]), observation=observation,
            truth=truth.copy(), estimate=corrected.copy(), correction=delta.copy(), error=error,
            failure=bool(failed), failure_reason="declared_domain_exit" if failed else None,
            complete=bool(final and not failed))
        self._done = bool(failed or final)
        reward = -float(np.sum((error/np.asarray(self.model.spec.error_scale))**2)) * (0. if final else self.model.spec.dt)
        if self._done:
            self._estimate = corrected
        else:
            self._estimate = self.model.predict(corrected, self._inputs[self._index])
            self._index += 1
        return self._observation(), reward, bool(failed), bool(final and not failed), self._info()


def make_observation_task(name):
    return ObservationEnv(name)
