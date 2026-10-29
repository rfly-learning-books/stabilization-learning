"""Numerical boundaries for fresh RGB runs, retaining the legacy update math.

The legacy classes and optimizer types stay unchanged. Instance-local checks
stop a run on corrupt data or an invalid update; they never repair or resume it.
"""
from functools import wraps

import numpy as np

from extensions.unified_training_v1.backends.rgb import RGBBackend
from extensions.unified_training_v2.historical_rgb import HistoricalRGBBackend
from .guards import NumericalGuard


class _StableRGB:
    _module_names = ("ae", "shadow", "initial", "metric", "critics")
    _optimizer_names = ("aeopt", "shadowopt", "criticopt", "mopt")
    _learner_calls = ("supervised_loss", "regularizer", "model_update",
                      "representation", "critic_prefit", "metric_update")

    def __init__(self, *args, **kwargs):
        self._numerics = NumericalGuard("RGB")
        self._watched_modules, self._watched_optimizers = set(), set()
        self._wrapped_learner = self._wrapped_replay = None
        super().__init__(*args, **kwargs)

    def guard(self):
        self._numerics.guard()
        return super().guard()

    def _checked_call(self, function, label):
        @wraps(function)
        def checked(*args, **kwargs):
            self.guard()
            self._numerics.check((args, kwargs), label + " inputs")
            try:
                result = function(*args, **kwargs)
                self._numerics.check(result, label + " outputs")
                return result
            except FloatingPointError as error:
                self._numerics.fail(error)
                raise
        return checked

    def _check_episode(self, arrays):
        self._numerics.check(arrays, "replay episode")
        # These lengths are used directly by Replay.sample for t and t + 1.
        try:
            length = len(arrays["action"])
            if length < 1:
                raise ValueError("RGB replay episode has no transitions")
            for key in ("rgb", "offset"):
                if len(arrays[key]) != length + 1:
                    raise ValueError("RGB replay successor length mismatch: " + key)
            for key in ("cost", "terminal", "truncated", "previous_action"):
                if len(arrays[key]) != length:
                    raise ValueError("RGB replay transition length mismatch: " + key)
            for key in ("terminal", "truncated"):
                terminal = np.asarray(arrays[key])
                if terminal.dtype != np.bool_ or terminal.shape != (length,):
                    raise ValueError("RGB replay terminal flags must be boolean vectors")
                if terminal[:-1].any():
                    raise ValueError("RGB replay contains transitions after termination")
            count = len(arrays["smooth_center"])
            if count < 1 or any(len(arrays[key]) != count for key in
                               ("smooth_minus", "smooth_plus", "smooth_step")):
                raise ValueError("RGB replay smoothing triplets are empty or mismatched")
        except (KeyError, TypeError, ValueError) as error:
            # Broken transition structure is as unusable as nonfinite data.
            failure = FloatingPointError(str(error))
            self._numerics.fail(failure)
            raise failure from error

    def _attach_guards(self):
        learner = getattr(self, "learner", None)
        if learner is not None:
            for name in self._module_names:
                module = getattr(learner, name, None)
                if module is not None and id(module) not in self._watched_modules:
                    self._numerics.watch_module(module, "RGB learner " + name)
                    self._watched_modules.add(id(module))
            for name in self._optimizer_names:
                optimizer = getattr(learner, name, None)
                if optimizer is not None and id(optimizer) not in self._watched_optimizers:
                    self._numerics.watch_optimizer(optimizer, "RGB " + name)
                    self._watched_optimizers.add(id(optimizer))
            if learner is not self._wrapped_learner:
                for name in self._learner_calls:
                    function = getattr(learner, name, None)
                    if function is not None:
                        setattr(learner, name, self._checked_call(function, "RGB " + name))
                self._wrapped_learner = learner
        replay = getattr(self, "replay", None)
        if replay is not None and replay is not self._wrapped_replay:
            original_add, original_load, original_sample = replay.add, replay.load, replay.sample

            @wraps(original_add)
            def add(arrays, row, identifier):
                self.guard()
                self._check_episode(arrays)
                return original_add(arrays, row, identifier)

            @wraps(original_load)
            def load(row):
                self.guard()
                arrays = original_load(row)
                self._check_episode(arrays)
                return arrays

            replay.add, replay.load = add, load
            replay.sample = self._checked_call(original_sample, "RGB replay sample")
            self._wrapped_replay = replay

    def _check_all(self):
        self.guard()
        self._attach_guards()
        self._numerics.check_all()
        learner = getattr(self, "learner", None)
        if learner is not None:
            self._numerics.check(getattr(learner, "anchors", {}), "RGB normalization anchors")
        for name in ("initial_state", "fixed_probe"):
            self._numerics.check(getattr(self, name, None), "RGB " + name)

    def initialize(self):
        self.guard()
        try:
            result = super().initialize()
            self._check_all()
            self._numerics.check(result, "RGB initialization")
            return result
        except FloatingPointError as error:
            self._numerics.fail(error)
            raise

    def collect(self, initial=False):
        self._check_all()
        try:
            result = super().collect(initial=initial)
            self._numerics.check(result, "RGB collection")
            self._check_all()
            return result
        except FloatingPointError as error:
            self._numerics.fail(error)
            raise

    def _record_probe(self, phase):
        self.guard()
        try:
            return super()._record_probe(phase)
        except ValueError as error:
            # The retained probe already uses JSON allow_nan=False. Normalize
            # its numerical rejection so a caller cannot catch it and save.
            if "Out of range float values" not in str(error):
                raise
            failure = FloatingPointError("Nonfinite RGB training probe")
            self._numerics.fail(failure)
            raise failure from error

    def _updates(self, phase, updates):
        self._check_all()
        try:
            result = super()._updates(phase, updates)
            self._check_all()
            self._numerics.check(result, "RGB update result")
            return result
        except FloatingPointError as error:
            self._numerics.fail(error)
            raise

    def save(self, path):
        # The original initialize() saves before returning, so attach here too.
        self._check_all()
        return super().save(path)

    def load(self, path):
        # Loading a selected candidate must not silently conceal corruption in
        # the active training state that has not yet reached another boundary.
        self._check_all()
        try:
            result = super().load(path)
            self._check_all()
            self._numerics.check(result, "RGB restored checkpoint")
            return result
        except FloatingPointError as error:
            self._numerics.fail(error)
            raise

    def evaluate(self, *args, **kwargs):
        self._check_all()
        try:
            result = super().evaluate(*args, **kwargs)
            self._check_all()
            self._numerics.check(result, "RGB evaluation")
            return result
        except FloatingPointError as error:
            self._numerics.fail(error)
            raise

    def close(self):
        # Drop guard references before the retained backend frees GPU tensors.
        self._wrapped_learner = self._wrapped_replay = None
        self._numerics.close()
        return super().close()


class StableRGBBackend(_StableRGB, RGBBackend):
    """The retained RGB learner with numerical checks for new runs."""


class StableHistoricalRGBBackend(_StableRGB, HistoricalRGBBackend):
    """Same safeguards, retaining the historical frozen-test reference."""
