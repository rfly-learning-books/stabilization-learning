from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from extensions.reproducible_training_v4.so101 import (
    ContinuousHoldPlant, Config, rollout, verify_terminal_hold,
)


class FakePlant:
    supports_substep_observer = True
    substep_observer = None
    path = "return"
    instances = []

    def __init__(self, cfg=None, seed=0, video=None):
        self.data = SimpleNamespace(time=0.)
        self.q = np.zeros(3)
        self.count = 0
        self.distance = self.transit_contacts = self.arm_contacts = 0
        self.transfers = []
        self.trace = []
        self.unsafe_pairs = set()
        self.min_clearance = np.inf
        self.max_speed = self.max_accel = self.max_torque = self.saturated = self.max_track = 0
        self.closed = False
        self.instances.append(self)

    def reset(self, q):
        self.q = np.array(q)

    def pose(self):
        return self.q.copy()

    def settle(self):
        self.data.time += .5

    def step(self, command):
        self.count += 1
        self.data.time += .02
        self.q[0] = (0.01 if self.count < 50 else 0.) if self.path == "return" else (0.01 if self.path == "outside" else 0.)
        if self.substep_observer:
            self.substep_observer(self.pose(), self.data.time)

    def close(self):
        self.closed = True


def goal(q):
    return np.linalg.norm(q[:2]) < .002


@pytest.mark.parametrize("path,success,violations", [("return", False, 49), ("inside", True, 0), ("outside", False, 50)])
def test_continuous_hold_rejects_return_at_final_step(path, success, violations):
    plant = FakePlant()
    plant.path = path
    result = verify_terminal_hold(plant, goal)
    assert result["continuous_success"] is success
    assert result["violated_substeps"] == violations
    assert result["elapsed_seconds"] == pytest.approx(1.)
    assert plant.substep_observer is None


@pytest.mark.parametrize("path,success", [("return", False), ("inside", True), ("outside", False)])
def test_rollout_reports_hold_failure_and_closes(path, success):
    class Plant(FakePlant):
        pass
    Plant.path = path
    result = rollout(Config(max_actions=1), np.zeros(3), 9, policy=object(), plant_factory=Plant,
                     base_module=SimpleNamespace(reached=goal, coverage=lambda q: 1.))
    assert result["success"] is success
    assert result["reason"] == ("success" if success else "terminal_hold_failed")
    assert Plant.instances[-1].closed


def test_endpoint_only_plant_is_rejected():
    plant = FakePlant()
    plant.supports_substep_observer = False
    with pytest.raises(TypeError, match="substep"):
        verify_terminal_hold(plant, goal)


def test_callback_omission_fails_closed():
    class Omit(FakePlant):
        def step(self, command):
            self.data.time += .02
    with pytest.raises(RuntimeError, match="omitted"):
        verify_terminal_hold(Omit(), goal)


def test_real_step_implementation_observes_mid_command_excursions():
    # Exercise the actual ContinuousHoldPlant.step implementation; only physics
    # integration is a deterministic CPU fixture. Each 20 ms endpoint is inside.
    plant = object.__new__(ContinuousHoldPlant)
    plant.cfg = Config(plant="planar")
    plant.data = SimpleNamespace(qpos=np.zeros(11), qvel=np.zeros(11), ctrl=np.zeros(2), time=0., contact=[])
    plant.model = SimpleNamespace(opt=SimpleNamespace(timestep=.002))
    plant.queue = deque()
    plant.writer = None
    plant.distance = plant.tick = plant.transit_contacts = 0
    plant.qadr = np.arange(5, 11)
    plant.rod = 1
    plant.block_geoms = {2}
    plant.trace = []
    plant.mode = -1
    count = [0]
    def step(model, data):
        count[0] += 1
        data.qpos[0] = .01 if count[0] % 10 else 0.
        data.time += model.opt.timestep
    plant.physics = SimpleNamespace(mj_step=step, mj_forward=lambda model, data: None)
    result = verify_terminal_hold(plant, goal)
    assert goal(plant.pose())
    assert result["checked_substeps"] == 500
    assert result["violated_substeps"] == 450
    assert not result["continuous_success"]


def test_policy_loading_must_be_explicit():
    with pytest.raises(ValueError, match="Explicit policy"):
        rollout(Config(), np.zeros(3), 9)
