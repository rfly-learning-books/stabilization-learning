"""Sealed physical records, decision identity, and whole-update recovery."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import numpy as np
import torch
from stabilization_gym.interfaces.certificate_v2 import VersionBinding


def _json(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.detach().cpu().tolist() if torch.is_tensor(value) else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, default=_json).encode()).hexdigest()


@dataclass(frozen=True)
class SealedRecord:
    payload: dict
    binding: VersionBinding
    checksum: str

    @classmethod
    def physical(cls, *, binding, observation, goal, requested_action, final_command,
                 actual_execution, following_observation, time, dt, physical_cost,
                 certificate_penalty, delta_v, terminated, truncated, reason,
                 decision_id, supervision_truth=None):
        payload = dict(kind="physical_transition", observation=observation, goal=goal,
            requested_action=requested_action, final_command=final_command, actual_execution=actual_execution,
            following_observation=following_observation,time=float(time),dt=float(dt),
            physical_cost=physical_cost,certificate_penalty=certificate_penalty,delta_v=delta_v,
            terminated=terminated,truncated=truncated,reason=reason,decision_id=decision_id,
            supervision_truth=supervision_truth)
        if payload["dt"] <= 0 or not decision_id:
            raise ValueError("Positive physical dt and separate high-level decision identity required")
        payload=deepcopy(payload)
        return cls(payload,binding,identity(dict(payload=payload,binding=asdict(binding))))

    def validate(self, expected):
        self.binding.require(expected)
        if self.checksum != identity(dict(payload=self.payload,binding=asdict(self.binding))):
            raise ValueError("Record changed after sealing")
        return deepcopy(self.payload)


def checkpoint(*, modules, optimizers, binding, counters, collector=None, pending=False):
    if pending:
        raise RuntimeError("Commit or cancel the pending control query before snapshot")
    return deepcopy(dict(schema="whole_update_cpu_v2",binding=asdict(binding),
        modules={k:v.state_dict() for k,v in modules.items()},
        optimizers={k:v.state_dict() for k,v in optimizers.items()},counters=counters,
        collector=collector,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),boundary="whole_update_cpu_only"))


def restore(payload, *, modules, optimizers, binding):
    binding.require(VersionBinding(**payload["binding"]))
    if payload.get("schema") != "whole_update_cpu_v2":
        raise ValueError("Unsupported recovery schema")
    if set(modules) != set(payload["modules"]) or set(optimizers) != set(payload["optimizers"]):
        raise ValueError("All modules and optimizers must be present")
    for name, module in modules.items():
        current=module.state_dict(); saved=payload["modules"][name]
        if set(current) != set(saved) or any(current[k].shape!=saved[k].shape or current[k].dtype!=saved[k].dtype for k in current):
            raise ValueError("Module coordinates/dtype differ")
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"]); random.setstate(payload["python_rng"])
    return deepcopy(payload["counters"]),deepcopy(payload["collector"])
