"""One pushing task with execution-time costs present from its first sample.

The backend is a calibrated, reduced response model.  Its time clock includes
transfer/engage/push/settle/hold; this module never pretends to run the robot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np


def vendor_modules():
    """Load preserved implementations without loading any learned policy."""
    root = Path(__file__).resolve().parents[3]
    for path in (root / "third_party/push_original", root / "third_party/so101_timeaware_v1"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from so101_timeaware.environment import TimedResponseEnv, reached
    from so101_timeaware.policy import CandidateModel, TimeCritic
    from so101_timeaware.timing import TimeBank
    from dl_push.certificate.full_m_control import FullMActor, FullMConfig
    return dict(Env=TimedResponseEnv, reached=reached, Model=CandidateModel,
                Critic=TimeCritic, Bank=TimeBank, Actor=FullMActor, ActorConfig=FullMConfig)


def prepare_time_bank(output, *, shapes=("T","RectWide","L","Plus","Square","UShape","OffsetComb3","AsymThreeBranch"),
                      repeats=4, timing_pairs=4, seed=71027001, guard=lambda:None):
    """Optional fresh actuated calibration, never policy training or old-weight loading.

This can be expensive and is never triggered implicitly by ``train``. Existing
audited banks are the explicit default input. Held-out shapes may be calibrated
but must remain absent from policy optimization and checkpoint selection.
"""
    if repeats<2 or timing_pairs<0 or len(set(shapes))!=len(shapes):
        raise ValueError("At least two response trials, nonnegative timing pairs, unique shapes required")
    out=Path(output)
    if out.exists():raise FileExistsError("Calibration output must be a new directory")
    vendor_modules()
    import torch
    import so101_multishape as geometry
    from so101_timeaware.timing import TimeConfig, prepare, write_json
    if not shapes or any(name not in geometry.PARTS for name in shapes):raise ValueError("Unknown shape")
    cfg=TimeConfig(encoder_omega_multiplier=1.,gamma=1.)
    out.mkdir(parents=True)
    write_json(out/"protocol.json",dict(schema="joint_push_calibration_v2",shapes=list(shapes),repeats=repeats,
        timing_pairs=timing_pairs,seed=seed,config=asdict(cfg),policy_used=False,
        scope="new actuated SO101 MuJoCo response/time calibration; not hardware"))
    (out/"events").mkdir()
    event_sequence=0
    protocol_hash=hashlib.sha256((out/"protocol.json").read_bytes()).hexdigest()
    def event(kind,**data):
        nonlocal event_sequence
        event_sequence+=1
        path=out/"events"/f"{event_sequence:08d}_{kind}.json"
        if path.exists():raise FileExistsError("Calibration events are immutable")
        write_json(path,dict(sequence=event_sequence,kind=kind,unix=time.time(),protocol_sha256=protocol_hash,data=data))
    previous_load=torch.load
    def forbidden(*args,**kwargs):raise RuntimeError("Calibration must not load a learned policy")
    torch.load=forbidden
    try:
        for index,name in enumerate(shapes):
            guard();target=out/"calibration"/name;target.mkdir(parents=True)
            with geometry.install(name) as (_,polygon,points,normals):
                result=geometry.calibrate(target,points,normals,repeat=repeats,seed=seed+index)
            if not any(result["valid"]):raise RuntimeError("No valid new response for "+name)
            event("response_calibrated",shape=name,valid_modes=int(sum(result["valid"])))
        guard()
        prepare(out/"time_bank.json",list(shapes),out/"calibration",cfg,measure_pairs=timing_pairs,seed=seed+1000)
    finally:
        torch.load=previous_load
    digest=hashlib.sha256((out/"time_bank.json").read_bytes()).hexdigest()
    write_json(out/"FINAL_STATE.json",dict(status="complete",time_bank_sha256=digest,policy_used=False))
    event("calibration_complete",time_bank_sha256=digest,policy_used=False)
    return out/"time_bank.json"


@dataclass(frozen=True)
class JointPushCosts:
    time: float = 1.0
    error: float = 1.0
    switch: float = 0.25
    input_square: float = 0.02
    success_bonus: float = 200.0
    failure_penalty: float = 600.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Costs must be finite and nonnegative")
        if self.failure_penalty <= 0:
            raise ValueError("Failure needs an explicit positive penalty")


class JointPushEnv:
    """Task-cost wrapper; the preserved reduced plant owns actual time and pose.

max_seconds/max_actions are task deadlines, hence terminal failures. A caller's
sampling cutoff is a truncation and must bootstrap the *pre-reset* observation.
Input cost is the push-speed-squared integral, not electrical/mechanical energy.
At fixed speed it is proportional to push duration. No transit input is inferred.
"""
    scope = "calibrated reduced response model; not MuJoCo training or hardware"

    def __init__(self, bank, shapes, seed=0, costs=None):
        self.bank, self.costs = bank, costs or JointPushCosts()
        self.plant = vendor_modules()["Env"](bank, list(shapes), seed)
        self.total_reward = 0.0
        self.total_breakdown = {}

    def reset(self, *, seed=None, options=None, shape=None, pose=None):
        options={} if options is None else dict(options)
        if set(options)-{"shape","pose"}:raise ValueError("Unknown reset options")
        if shape is not None and "shape" in options or pose is not None and "pose" in options:
            raise ValueError("Supply native shape/pose or options, not both")
        shape=options.get("shape",shape);pose=options.get("pose",pose)
        if shape is not None and shape not in self.plant.names:
            raise ValueError("Reset cannot access a shape outside the declared split")
        if seed is not None:self.plant.rng=np.random.default_rng(seed)
        observation = self.plant.reset(shape=shape, pose=pose)
        self.total_reward, self.total_breakdown = 0.0, {}
        return observation, {"scope": self.scope}

    def observation(self):
        return self.plant.observation()

    def _score(self, info, switches, terminated, truncated):
        w, cfg = self.costs, self.bank.config
        push_seconds = float(info.get("stages", {}).get("push", 0.0))
        input_square = cfg.push_speed ** 2 * push_seconds
        success = info["reason"] == "success"
        failed = terminated and not success
        breakdown = dict(time=-w.time * info["dt"] / cfg.time_reference,
                         physical_error=-w.error * info["integrated_error"],
                         switching=-w.switch * switches,
                         input_square=-w.input_square * input_square,
                         success=w.success_bonus if success else 0.0,
                         failure=-w.failure_penalty if failed else 0.0)
        reward = float(sum(breakdown.values()))
        self.total_reward += reward
        for key, value in breakdown.items():
            self.total_breakdown[key] = self.total_breakdown.get(key, 0.0) + value
        info = dict(info, reward_breakdown=breakdown, switches=switches,
                    input_square_integral=input_square, input_square_units="m^2/s (pusher speed proxy)",
                    terminated=terminated, truncated=truncated, scope=self.scope,
                    episode_reward=self.total_reward)
        return self.observation(), reward, terminated, truncated, info

    def step(self, action):
        before_switches, before_time = self.plant.switches, self.plant.elapsed
        _, _, done, info = self.plant.step(int(action))
        info = dict(info)
        if info["reason"] == "time_limit":
            info["reason"] = "task_deadline"
        elif info["reason"] == "action_limit":
            info["reason"] = "task_action_budget"
        # Keep the plant reason as-is for its own checkpoint compatibility.
        if not np.isclose(sum(info["stages"].values()), self.plant.elapsed - before_time):
            raise RuntimeError("Stage durations disagree with the plant clock")
        return self._score(info, self.plant.switches-before_switches, bool(done), False)

    def abort(self, reason="no_physical_candidate"):
        """A rejected plan incurs failure cost, without inventing physical time."""
        if self.plant.done:
            raise RuntimeError("Cannot abort an already finished episode")
        self.plant.done, self.plant.reason = True, reason
        return self._score(dict(dt=0., stages={}, integrated_error=0., success=False,
                                reason=reason, elapsed_seconds=self.plant.elapsed), 0, True, False)

    def truncate(self, reason="external_sampling_cutoff"):
        if self.plant.done:
            raise RuntimeError("Cannot truncate an already finished episode")
        self.plant.done, self.plant.reason = True, reason
        return self._score(dict(dt=0., stages={}, integrated_error=0., success=False,
                                reason=reason, elapsed_seconds=self.plant.elapsed), 0, False, True)

    def state_dict(self):
        return dict(plant=self.plant.state_dict(), total_reward=self.total_reward,
                    total_breakdown=dict(self.total_breakdown))


class PushTaskAdapter:
    """Task-owned action construction using a supplied full-M actor/certificate.

``certificate`` can expose ``actor`` or be the preserved FullMActor itself.
No optimizer lives here. The training backend uses the identical candidate map.
"""
    def __init__(self, bank, device="cpu"):
        from .contracts import Capabilities
        self.capabilities = Capabilities(candidate_probabilities=True, action_kind="push_mode_and_travel",
            note="M-induced discrete distribution; calibrated reduced response, not a differentiable physical plant")
        self.model = vendor_modules()["Model"](bank, device)

    def make_context(self, observation, reference=None, *, features=None, model=None,
                     memory=None, binding=None):
        from .contracts import TaskContext
        batch = self.model.batch([observation])
        return TaskContext(error=self.model.coordinates(batch["q"]), metric_context=batch["shape"],
                           model_context={"batch": batch}, memory=memory or {}, binding=binding)

    def construct_action(self, context, certificate):
        from .contracts import ActionPlan
        import torch
        actor = getattr(certificate, "actor", certificate)
        candidates = self.model.candidates(actor, context.model_context["batch"])
        physical = candidates["physical"][0]
        if not bool(physical.any()):
            return ActionPlan(action=None, memory=context.memory,
                              diagnostics={"failure": "no_physical_candidate"})
        # Frozen deployment is the greedy version of the trained distribution.
        index = int(candidates["logits"][0].detach().argmax())
        distribution = torch.distributions.Categorical(logits=candidates["logits"])
        return ActionPlan(action=index, memory=context.memory, distribution=distribution,
                          diagnostics={"predicted_V": float(candidates["V"][0].detach()),
                                       "predicted_V_next": float(candidates["V_next"][0,index].detach()),
                                       "scope": JointPushEnv.scope})


class JointPushSelector:
    """Deployment bridge for the preserved actuated simulator's rollout API.

It uses the same greedy logits as this version's validation, not the old A/B
heuristic. Construction never calls the old Policy initializer/checkpoint loader.
"""
    def __init__(self, bank, shape, actor):
        self.bank,self.name,self.actor=bank,shape,actor.eval()
        self.model=vendor_modules()["Model"](bank,next(actor.parameters()).device)
        self.metric="joint_random_origin"

    def value(self, q):
        import torch
        q=torch.as_tensor(np.atleast_2d(q),dtype=torch.float32,device=self.model.device)
        index=self.bank.names.index(self.name)
        with torch.no_grad():
            condition=self.actor.encoder(self.model.C_encoded[[index]],self.model.valid[[index]],ell=self.bank.config.length_scale)
            return self.actor.value(self.model.coordinates(q),condition.expand(len(q),-1)).cpu().numpy()

    def choose(self,q,previous,hold,pusher):
        previous=-1 if previous is None else previous
        contact=self.bank.contact_usable(self.name,q,previous,pusher)
        tau,reachable,reposition=self.bank.row(self.name,previous,contact)
        observation=dict(shape=self.bank.names.index(self.name),q=q,pusher=pusher,mode=previous,
            contact=contact,remaining=self.bank.config.max_seconds,tau=tau,reachable=reachable,reposition=reposition)
        candidates=self.model.candidates(self.actor,self.model.batch([observation]))
        if not bool(candidates["physical"].any()):raise RuntimeError("no_physical_candidate")
        index=int(candidates["logits"][0].detach().argmax())
        mode,k=divmod(index,len(self.bank.config.travels))
        return mode,float(self.bank.config.travels[k]),float(candidates["V"][0].detach()),not bool(candidates["feasible"][0,index])
