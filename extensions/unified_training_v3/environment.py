"""Encoding-safe calibrated Push environment and public factory for new runs.

This version preserves the task equations and the immutable v1 vendor sources.
It does not load a policy or perform calibration when constructing an env.
"""
from pathlib import Path

import torch

from stabilization_gym.library import (
    DeviceCapabilities, TaskBundle, TaskSpec, _CPUTensorObservations,
    _NativeObservationAdapter, _require_cpu, make_task as _legacy_make_task,
)
from stabilization_gym.tasks_v2.contracts import Capabilities
from stabilization_gym.tasks_v2.push import (
    JointPushCosts, JointPushEnv as _OriginalEnv,
    JointPushSelector as _OriginalSelector, PushTaskAdapter as _OriginalAdapter,
)


def vendor_modules():
    # Namespace the fixed vendor independently of any already-imported v1.
    from third_party.so101_timeaware_v2.so101_timeaware.environment import TimedResponseEnv, reached
    from third_party.so101_timeaware_v2.so101_timeaware.policy import CandidateModel, TimeCritic
    from third_party.so101_timeaware_v2.so101_timeaware.timing import TimeBank
    from dl_push.certificate.full_m_control import FullMActor, FullMConfig
    from dl_push.certificate import full_m_control
    from dl_push_contact.envs import so101_push_t_env
    project = Path(__file__).resolve().parents[2]
    expected = (
        (full_m_control, project / "third_party/push_original/dl_push/certificate/full_m_control.py"),
        (so101_push_t_env, project / "third_party/so101_timeaware_v1/so101_vendor/dl_push_contact/envs/so101_push_t_env.py"),
    )
    for module, path in expected:
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError(f"Unexpected previously imported vendor module: {module.__name__}")
    return dict(Env=TimedResponseEnv, reached=reached, Model=CandidateModel,
                Critic=TimeCritic, Bank=TimeBank, Actor=FullMActor, ActorConfig=FullMConfig)


def _deployment_logits(candidates):
    """An unavailable action must remain impossible at every logit scale."""
    physical = candidates["physical"]
    logits = candidates["logits"]
    if not torch.isfinite(logits[physical]).all():
        raise FloatingPointError("Non-finite logits for physical Push candidates")
    for name in ("V", "V_next"):
        if not torch.isfinite(candidates[name]).all():
            raise FloatingPointError(f"Non-finite Push candidate {name}")
    return logits.masked_fill(~physical, -torch.inf)


class JointPushEnv(_OriginalEnv):
    def __init__(self, bank, shapes, seed=0, costs=None):
        self.bank, self.costs = bank, costs or JointPushCosts()
        self.plant = vendor_modules()["Env"](bank, list(shapes), seed)
        self.total_reward = 0.0
        self.total_breakdown = {}

    def step(self, action):
        from third_party.so101_timeaware_v2.so101_timeaware.environment import action_index
        return super().step(action_index(action))


class PushTaskAdapter(_OriginalAdapter):
    def __init__(self, bank, device="cpu"):
        self.capabilities = Capabilities(
            candidate_probabilities=True, action_kind="push_mode_and_travel",
            note="M-induced discrete distribution; calibrated reduced response, not a differentiable physical plant")
        self.model = vendor_modules()["Model"](bank, device)

    def construct_action(self, context, certificate):
        from stabilization_gym.tasks_v2.contracts import ActionPlan
        actor = getattr(certificate, "actor", certificate)
        candidates = self.model.candidates(actor, context.model_context["batch"])
        if not bool(candidates["physical"][0].any()):
            return ActionPlan(action=None, memory=context.memory,
                              diagnostics={"failure": "no_physical_candidate"})
        logits = _deployment_logits(candidates)
        index = int(logits[0].detach().argmax())
        distribution = torch.distributions.Categorical(logits=logits)
        return ActionPlan(action=index, memory=context.memory, distribution=distribution,
            diagnostics={"predicted_V": float(candidates["V"][0].detach()),
                         "predicted_V_next": float(candidates["V_next"][0,index].detach()),
                         "scope": JointPushEnv.scope})


class JointPushSelector(_OriginalSelector):
    def __init__(self, bank, shape, actor):
        self.bank, self.name, self.actor = bank, shape, actor.eval()
        self.model = vendor_modules()["Model"](bank, next(actor.parameters()).device)
        self.metric = "joint_random_origin"

    def choose(self, q, previous, hold, pusher):
        previous = -1 if previous is None else previous
        contact = self.bank.contact_usable(self.name, q, previous, pusher)
        tau, reachable, reposition = self.bank.row(self.name, previous, contact)
        observation = dict(shape=self.bank.names.index(self.name), q=q, pusher=pusher, mode=previous,
            contact=contact, remaining=self.bank.config.max_seconds, tau=tau,
            reachable=reachable, reposition=reposition)
        candidates = self.model.candidates(self.actor, self.model.batch([observation]))
        if not bool(candidates["physical"].any()):
            raise RuntimeError("no_physical_candidate")
        index = int(_deployment_logits(candidates)[0].detach().argmax())
        mode, k = divmod(index, len(self.bank.config.travels))
        return (mode, float(self.bank.config.travels[k]), float(candidates["V"][0].detach()),
                not bool(candidates["feasible"][0,index]))


def make_push(*, shapes, time_bank=None, bank=None, seed=81, costs=None,
              device="cpu", adapter_device="cpu", tensor_observations=False):
    """Construct a CPU plant with an explicit, UTF-8 calibrated bank."""
    _require_cpu(device, "push")
    if (time_bank is None) == (bank is None):
        raise ValueError("Supply exactly one explicit time_bank path or bank object")
    if not shapes or len(set(shapes)) != len(shapes):
        raise ValueError("Declare a nonempty, unique shape set")
    if bank is None:
        path = Path(time_bank).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        bank = vendor_modules()["Bank"](path)
    if set(shapes) - set(bank.names):
        raise ValueError("Shape absent from supplied calibration bank")
    env = JointPushEnv(bank, list(shapes), seed, costs)
    adapter = PushTaskAdapter(bank, adapter_device)
    if tensor_observations:
        env, adapter = _CPUTensorObservations(env), _NativeObservationAdapter(adapter)
    target = str(torch.device(adapter_device))
    return TaskBundle(env, adapter, TaskSpec(
        "push", "cross_shape_calibrated",
        "CPU/Shapely reduced response/time model; not joint-driven MuJoCo or hardware",
        ("explicit response/time bank", "vendor geometry modules"),
        ("Bank geometry for a shape does not authorize using it for learning or selection.",)),
        DeviceCapabilities("cpu", "cpu", target, bool(tensor_observations), 1,
            "discrete_mode_and_travel", "push_mode_and_travel",
            transfers="CPU observation to candidate-model device" if target != "cpu" else "none"),
        resources={"bank": bank})


def make_task(family, name=None, **kwargs):
    if family == "push":
        if name not in (None, "cross_shape_calibrated"):
            raise ValueError("Push factory supports the calibrated cross-shape plant")
        return make_push(**kwargs)
    return _legacy_make_task(family, name, **kwargs)
