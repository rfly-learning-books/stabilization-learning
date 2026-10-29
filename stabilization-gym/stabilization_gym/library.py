"""Public environment factories; no learner, checkpoint or optimizer ownership.

The physical device and observation device are deliberately distinct. A CPU
environment returning tensors is still CPU simulation. RGB rendering/stepping
uses the explicit CUDA vector backend. Optional vendor assets must be supplied
by the caller; this module never imports ``stabilization_learning`` or training
extensions, discovers an experiment, or loads a learned controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


STATE_TASKS = ("coupled_linear", "point_mass", "pendulum", "cartpole")
OBSERVER_TASKS = ("motor_speed", "cartpole_output", "cooperative_pose")


@dataclass(frozen=True)
class DeviceCapabilities:
    physical_device: str
    observation_device: str
    adapter_device: str
    tensor_observations: bool
    num_envs: int
    physical_action_kind: str
    adapter_action_kind: str
    physical_step_gradient: bool = False
    autoreset: str = "disabled"
    transfers: str = "none"


@dataclass(frozen=True)
class TaskSpec:
    family: str
    name: str
    scope: str
    required_resources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _tensor_tree(value):
    if isinstance(value, Mapping):
        return {key: _tensor_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_tree(item) for item in value)
    if isinstance(value, np.ndarray):
        # Own the buffer: an environment may reuse its NumPy arrays next step.
        return torch.from_numpy(value.copy())
    if isinstance(value, (float, int, bool, np.number)):
        return torch.as_tensor(value)
    return value


def _native_tree(value):
    if isinstance(value, Mapping):
        return {key: _native_tree(item) for key, item in value.items()}
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
        return array.item() if array.ndim == 0 else array.copy()
    return value


class _CPUTensorObservations:
    """Optional owned CPU tensors; never advertises GPU physical simulation."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return _tensor_tree(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(_native_tree(action))
        return _tensor_tree(observation), reward, terminated, truncated, info

    def observation(self):
        return _tensor_tree(self.env.observation())

    def abort(self, *args, **kwargs):
        observation, reward, terminated, truncated, info = self.env.abort(*args, **kwargs)
        return _tensor_tree(observation), reward, terminated, truncated, info

    def truncate(self, *args, **kwargs):
        observation, reward, terminated, truncated, info = self.env.truncate(*args, **kwargs)
        return _tensor_tree(observation), reward, terminated, truncated, info


class _NativeObservationAdapter:
    """Restore CPU public observations for legacy geometry/forcing contracts."""
    def __init__(self, adapter):
        self.adapter = adapter

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def make_context(self, observation, reference=None, **kwargs):
        return self.adapter.make_context(_native_tree(observation), reference, **kwargs)

    def construct_action(self, context, certificate):
        return self.adapter.construct_action(context, certificate)


@dataclass
class TaskBundle:
    env: Any
    adapter: Any
    spec: TaskSpec
    capabilities: DeviceCapabilities
    resources: dict[str, Any] = field(default_factory=dict)

    def describe(self):
        return dict(spec=asdict(self.spec), capabilities=asdict(self.capabilities))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, physical_action):
        from .tasks_v2.contracts import ActionPlan
        if isinstance(physical_action, ActionPlan):
            raise TypeError("step accepts a physical action; use advance(plan, observation) for an ActionPlan")
        return self.env.step(physical_action)

    def advance(self, plan, observation=None):
        """Execute a plan without feeding an estimator correction to physics.

        Observer callers keep ``plan.memory`` themselves. The physical replay
        advances with its public known forcing; correction remains algo-side.
        """
        from .tasks_v2.contracts import ActionPlan
        if not isinstance(plan, ActionPlan):
            raise TypeError("advance requires an ActionPlan")
        if self.capabilities.adapter_action_kind == "estimator_update":
            if observation is None or "known_input" not in observation:
                raise ValueError("Observer advance requires the public known-input observation")
            return self.env.step(observation["known_input"])
        if plan.action is None:
            abort = getattr(self.env, "abort", None)
            if abort is None:
                raise ValueError("No executable physical action in plan")
            return abort(plan.diagnostics.get("failure", "no_physical_candidate"))
        return self.env.step(plan.action)

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


def _require_cpu(device, family):
    if torch.device(device).type != "cpu":
        raise ValueError(f"{family} physical backend is CPU; a CUDA learner does not change its physics device")


def make_state(name="point_mass", *, device="cpu", tensor_observations=False, **physical):
    """Four preserved state plants, with separate differentiable task model."""
    _require_cpu(device, "state")
    from .tasks_v2.state_observer import make_state_control
    env, adapter = make_state_control(name, **physical)
    if tensor_observations:
        env = _CPUTensorObservations(env)
    return TaskBundle(env, adapter, TaskSpec("state", name,
        "original NumPy plant; differentiable prediction is a separate model",
        notes=("Gym reward is negative physical cost; cost remains in info.",)),
        DeviceCapabilities("cpu", "cpu", "input_tensor_device", bool(tensor_observations),
            1, "physical_control", "physical"))


def make_observer(name="motor_speed", *, device="cpu", tensor_observations=False):
    """Public sensor replay and separate estimator-memory adapter."""
    _require_cpu(device, "observer")
    if name not in OBSERVER_TASKS:
        raise ValueError(f"Unknown observer task {name}")
    from .tasks_v2.state_observer import ObservationReplayEnv, ObserverTaskAdapter
    env = ObservationReplayEnv(name)
    adapter = ObserverTaskAdapter(env.model)
    if tensor_observations:
        env, adapter = _CPUTensorObservations(env), _NativeObservationAdapter(adapter)
    return TaskBundle(env, adapter, TaskSpec("observer", name,
        "declared forcing/sensor replay; correction is not a physical actuator command",
        notes=("True state is available only through env.supervision().",
               "Reset uses options={split, index, scenario}; seed, if supplied, must match episode_seed(split, index).",
               "Carry adapter memory explicitly; advance(plan, obs) uses obs['known_input'].")),
        DeviceCapabilities("cpu", "cpu", "cpu_geometry", bool(tensor_observations),
            1, "declared_known_forcing", "estimator_update"))


def make_push(*, shapes, time_bank=None, bank=None, seed=81, costs=None,
              device="cpu", adapter_device="cpu", tensor_observations=False):
    """CPU calibrated plant; candidate construction may use a CUDA device.

    Supply exactly one explicit bank object or time-bank JSON path. The existing
    vendor geometry modules must be installed/available; no policy is restored.
    """
    _require_cpu(device, "push")
    if (time_bank is None) == (bank is None):
        raise ValueError("Supply exactly one explicit time_bank path or bank object")
    if not shapes or len(set(shapes)) != len(shapes):
        raise ValueError("Declare a nonempty, unique shape set")
    from .tasks_v2.push import JointPushEnv, PushTaskAdapter, vendor_modules
    if bank is None:
        path = Path(time_bank).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        bank = vendor_modules()["Bank"](path)
    if set(shapes)-set(bank.names):
        raise ValueError("Shape absent from supplied calibration bank")
    env, adapter = JointPushEnv(bank, list(shapes), seed, costs), PushTaskAdapter(bank, adapter_device)
    if tensor_observations:
        env, adapter = _CPUTensorObservations(env), _NativeObservationAdapter(adapter)
    target = str(torch.device(adapter_device))
    return TaskBundle(env, adapter, TaskSpec("push", "cross_shape_calibrated",
        "CPU/Shapely reduced response/time model; not joint-driven MuJoCo or hardware",
        ("explicit response/time bank", "vendor geometry modules"),
        ("Bank geometry for a shape does not authorize using it for learning or selection.",)),
        DeviceCapabilities("cpu", "cpu", target, bool(tensor_observations),
            1, "discrete_mode_and_travel", "push_mode_and_travel",
            transfers="CPU observation to candidate-model device" if target != "cpu" else "none"),
        resources={"bank": bank})


def make_rgb(*, cfg, scenes, adapter, device="cuda:0", plant="P0", max_steps=200,
             cache_dir=None, samples_per_axis=2):
    """Explicit GPU RGB vector environment, with caller-owned task adapter.

    ``adapter`` is mandatory: selecting a current-J controller by default would
    silently change the successful legacy goal-J mechanism. Inject the desired
    task adapter explicitly; its feature/depth/model providers remain caller
    owned. No AE, M, depth weights, policy or optimizer is loaded here.
    """
    if torch.device(device).type != "cuda":
        raise ValueError("GPUParallelRGBEnv requires CUDA; no implicit CPU fallback")
    if adapter is None or not callable(getattr(adapter, "make_context", None)) or not callable(getattr(adapter, "construct_action", None)):
        raise TypeError("Supply an explicit TaskAdapter with make_context and construct_action")
    from .tasks_gpu_rgb.environment import GPUParallelRGBEnv
    env = GPUParallelRGBEnv(cfg, scenes, plant=plant, max_steps=max_steps,
        device=device, cache_dir=cache_dir, samples_per_axis=samples_per_axis)
    return TaskBundle(env, adapter, TaskSpec("rgb", f"reconstructed_{plant}",
        "GPU RGB rendering and 4DOF velocity outer-loop; no end-to-end renderer gradient",
        ("cfg", "scene descriptions/assets", "explicit task adapter", "MuJoCo Warp/CUDA"),
        ("Public GPU observation and privileged info['supervision'] remain separate.",
         "Next-step autoreset transitions must honor info['transition_mask'].",
         "Adapter device/capability claims are owned by the injected adapter.")),
        DeviceCapabilities(str(env.device), str(env.device), "caller_supplied", True,
            env.num_envs, "normalized_body_velocity_and_yaw_rate",
            getattr(getattr(adapter, "capabilities", None), "action_kind", "caller_declared"),
            autoreset="next_step", transfers="CPU asset/reset setup; GPU public RGB and physical stepping"))


def make_task(family, name=None, **kwargs):
    """Factory dispatcher. Required task resources stay explicit in kwargs."""
    if family == "rgb":
        if name not in (None, "reconstructed"):
            raise ValueError("RGB factory supports reconstructed rooms only")
        return make_rgb(**kwargs)
    if family == "push":
        if name not in (None, "cross_shape_calibrated"):
            raise ValueError("Push factory supports the calibrated cross-shape plant")
        return make_push(**kwargs)
    if family == "state":
        return make_state(name or "point_mass", **kwargs)
    if family == "observer":
        return make_observer(name or "motor_speed", **kwargs)
    raise ValueError("family must be rgb, push, state or observer")


__all__ = ["TaskBundle", "TaskSpec", "DeviceCapabilities", "STATE_TASKS", "OBSERVER_TASKS",
           "make_task", "make_rgb", "make_push", "make_state", "make_observer"]
