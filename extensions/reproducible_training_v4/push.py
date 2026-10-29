"""Explicit, guarded Push entries; generic fixed-M and SO101 time protocols differ."""
from functools import wraps
import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from stabilization_gym.envs.tasks.push_any_shape import PushAnyShapeEnv, DEFAULT_LIBRARY
from extensions.unified_training_v3.numerical import require_finite

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/push_original"
PROTOCOL = "generic_episode_fixed_metric_push_v1"


def source_files_for_config(config=None):
    """Return actual consumed calibration/geometry files without constructing envs.

    ModeBank reads all 32 libraries even for a single selected task shape, so
    freeze every report, polygon, library JSON and numeric archive.
    """
    config = dict(config or {})
    if "jobs" in config:
        files = set()
        for job in config["jobs"]:
            inner = job.get("config", {})
            if inner.get("task_id") == "push_any_shape" or inner.get("task") == "push_any_shape":
                files.update(source_files_for_config(inner))
        return sorted(files)
    kwargs = dict(config.get("env_kwargs", config))
    library = Path(kwargs.get("library_root") or DEFAULT_LIBRARY).expanduser().resolve()
    source = Path(kwargs.get("source_root") or SOURCE).expanduser().resolve()
    reports = sorted(library.glob("*/report.json"))
    if len(reports) != 32:
        raise ValueError(f"Expected 32 calibrated shape reports, got {len(reports)}: {library}")
    paths = set(source.joinpath("dl_push").rglob("*.py"))
    if not source.joinpath("dl_push/certificate/numeric_env.py").is_file():
        raise FileNotFoundError(source / "dl_push/certificate/numeric_env.py")
    for report in reports:
        for relative in ("report.json", "polygon.json", "library/library.json", "library/library.npz"):
            path = report.parent / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.add(path)
    return sorted(str(path.resolve()) for path in paths)


def _guarded(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        if self._numerical_failure is not None:
            raise FloatingPointError("Push environment stopped after numerical failure: " + self._numerical_failure)
        try:
            return method(self, *args, **kwargs)
        except FloatingPointError as error:
            self._numerical_failure = str(error)
            self._done = True
            raise
    return call


class SafePushAnyShapeEnv(PushAnyShapeEnv):
    """Finite boundary checks around the preserved episode-fixed-M integrator.

    Costs are dt times pose error. There is no SO101 reposition, engagement,
    actuator or settling time; these results cannot substitute for timed Push.
    """
    protocol = PROTOCOL

    def __init__(self, shape_name="T", *, source_root=None, library_root=None,
                 max_steps=100, speed_points=65, render_mode="rgb_array", **kwargs):
        self._numerical_failure = None
        for key, value, minimum in (("max_steps", max_steps, 1), ("speed_points", speed_points, 2)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        source_root = Path(source_root or SOURCE).resolve()
        library_root = Path(library_root or DEFAULT_LIBRARY).resolve()
        files = source_files_for_config(dict(source_root=source_root, library_root=library_root))
        super().__init__(shape_name, source_root=source_root, library_root=library_root,
                         max_steps=max_steps, speed_points=speed_points, render_mode=render_mode, **kwargs)
        require_finite((self.bank.C, self.bank.vertices, self.plant.vector.C,
                        self.plant.vector.cal_bounds), "Push calibration arrays")
        self.reference_model_version = "finite_episode_fixed_M_push_v4"
        self.asset_manifest = dict(self.asset_manifest, protocol=PROTOCOL,
            time_protocol="fixed dt numeric flow; no reposition/contact/settling time",
            sources={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files})

    @_guarded
    def reset(self, *, seed=None, options=None):
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
            raise ValueError("Nonnegative integer seed required")
        require_finite(options, "Push reset options")
        result = super().reset(seed=seed, options=options)
        require_finite(result, "Push reset output")
        return result

    @_guarded
    def metric_context(self, observation=None, goal=None):
        require_finite((observation, goal), "Push metric context input")
        value = super().metric_context(observation, goal)
        require_finite(value, "Push metric context output")
        return value

    @_guarded
    def build_error(self, observation, goal=None, *, representation=None, history=None):
        require_finite((observation, goal), "Push error input")
        result = super().build_error(observation, goal, representation=representation, history=history)
        require_finite(result, "Push error output")
        return result

    @_guarded
    def candidate_scores(self, M, e, *, context):
        M = torch.as_tensor(M)
        require_finite((M, e, context), "Push candidates input")
        if M.shape != (3, 3) or not M.is_floating_point():
            raise ValueError("Push metric must be a floating 3x3 matrix")
        if not torch.allclose(M, M.T, atol=1e-6, rtol=1e-6):
            raise ValueError("Push metric must be symmetric")
        if not bool((torch.linalg.eigvalsh(M.detach()) > 0).all()):
            raise ValueError("Push metric must be positive definite")
        value = super().candidate_scores(M, e, context=context)
        require_finite(value, "Push candidate output")
        return value

    @_guarded
    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        require_finite((M, e, context), "Push controller input")
        value = super().calculate_action_from_M(M, e, context=context,
                    controller_state=controller_state, certificate=certificate)
        require_finite(value, "Push controller output")
        return value

    @_guarded
    def step(self, physical_action):
        require_finite(physical_action, "Push physical action")
        require_finite(self._obs, "Push current observation")
        result = super().step(physical_action)
        require_finite(result, "Push transition")
        return result

    def _info(self, info):
        return dict(super()._info(info), protocol=PROTOCOL,
                    time_protocol="fixed_dt_numeric_flow_no_SO101_transfer_time")


def make_timeaware_task(*, shapes, time_bank=None, bank=None, seed=81, **kwargs):
    """Explicit repaired v2 vendor and v3 task route for calibrated timed Push."""
    from extensions.unified_training_v3.environment import make_task
    return make_task("push", "cross_shape_calibrated", shapes=shapes,
                     time_bank=time_bank, bank=bank, seed=seed, **kwargs)
