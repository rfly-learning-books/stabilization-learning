"""Explicit SO101 arm evaluation with continuous physical-substep goal hold.

The historical campaign is retained as an immutable dynamics dependency.  This
module never replaces its globals, launches training, or loads a checkpoint.
Scratch callers can explicitly call ``rollout(..., policy=their_policy)``.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from third_party.so101_timeaware_v2 import so101_campaign as campaign
from extensions.unified_training_v3.numerical import require_finite

Config = campaign.Config


class ContinuousHoldPlant(campaign.Plant):
    """Same controller, observing every integration substep during HOLD.

    ``physics`` is an instance dependency for deterministic CPU fixtures.  The
    default is real MuJoCo; no library function or module global is patched.
    """
    supports_substep_observer = True
    physics = campaign.mujoco
    substep_observer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Compound shapes may have more than the original two T geometries.
        # Use the body membership for both contact detection and classification.
        self.block_geoms = set(np.flatnonzero(
            self.model.geom_bodyid == self.env._block_body).tolist())

    def step(self, v):
        require_finite(v, "SO101 command")
        require_finite((self.data.qpos, self.data.qvel), "SO101 physical state")
        if np.asarray(v).shape != (2,):
            raise ValueError("SO101 planar command must have two components")
        if self.data.time > self.cfg.max_time:
            raise RuntimeError("time_limit")
        v = np.asarray(v)
        self.queue.append(v.copy())
        cmd = self.queue.popleft()
        before = self.pusher()
        substeps = int(round(.02 / self.model.opt.timestep))
        if substeps <= 0 or not np.isclose(substeps * self.model.opt.timestep, .02):
            raise ValueError("SO101 timestep must divide the 20 ms command period")
        if self.cfg.plant == "arm":
            self.target_xy += cmd * .02
            offset = self.target_xy - before
            if np.linalg.norm(offset) > .004:
                self.target_xy = before + offset / np.linalg.norm(offset) * .004
            desired = self.solve(self.target_xy)
            err = desired - self.qref
            vdes = np.sign(err) * np.minimum(self.cfg.joint_speed,
                np.minimum(np.sqrt(2 * self.cfg.joint_accel * abs(err)), abs(err) / .02))
            self.vref += np.clip(vdes - self.vref, -self.cfg.joint_accel * .02, self.cfg.joint_accel * .02)
            self.qref += self.vref * .02
        for _ in range(substeps):
            if self.cfg.plant == "arm":
                torque = 40 * (self.qref - self.data.qpos[self.qadr]) - .35 * self.data.qvel[self.dadr] + self.data.qfrc_bias[self.dadr]
                require_finite(torque, "SO101 torque")
                self.saturated += int(np.any(abs(torque) > self.cfg.torque_limit))
                torque = np.clip(torque, -self.cfg.torque_limit, self.cfg.torque_limit)
                self.data.ctrl[:2] = 0
                self.data.ctrl[2:] = torque
                self.max_torque = max(self.max_torque, float(max(abs(torque))))
            else:
                self.data.ctrl[:] = np.clip(cmd, -.04, .04)
            self.physics.mj_step(self.model, self.data)
            require_finite((self.data.qpos, self.data.qvel, self.data.time), "SO101 integrated state")
            if self.phase == "TRANSIT" and self.contact():
                self.transit_contacts += 1
            if self.cfg.plant == "arm":
                velocity = self.data.qvel[self.dadr].copy()
                self.max_speed = max(self.max_speed, float(max(abs(velocity))))
                self.max_accel = max(self.max_accel, float(max(abs(velocity - self.last_joint_velocity))) / self.model.opt.timestep)
                self.last_joint_velocity = velocity
                for contact in self.data.contact:
                    pair = {int(contact.geom1), int(contact.geom2)}
                    intentional = self.rod in pair and bool(pair & self.block_geoms)
                    if pair & self.arm_geoms and not intentional:
                        self.arm_contacts += 1
                        self.unsafe_pairs.add(tuple(sorted(self.physics.mj_id2name(
                            self.model, self.physics.mjtObj.mjOBJ_GEOM, g) for g in pair)))
            if self.substep_observer is not None:
                # Arm pose uses derived xpos/xmat. Refresh them after mj_step;
                # otherwise a callback could inspect the previous substep pose.
                self.physics.mj_forward(self.model, self.data)
                self.substep_observer(self.pose(), float(self.data.time))
        self.physics.mj_forward(self.model, self.data)
        if self.cfg.plant == "planar" and self.writer:
            self.env._track_pusher_with_arm()
        self.distance += float(np.linalg.norm(self.pusher() - before))
        self.tick += 1
        if self.cfg.plant == "arm":
            self.max_track = max(self.max_track, float(np.linalg.norm(self.pusher() - self.target_xy)))
        if self.phase == "TRANSIT":
            clearance = campaign.base.shape(self.pose()).distance(Point(self.pusher())) - .004
            self.min_clearance = min(self.min_clearance, clearance)
            if self.transit_contacts:
                raise RuntimeError("transit_collision")
        if self.cfg.plant == "arm" and self.arm_contacts:
            raise RuntimeError("arm_collision")
        if self.cfg.plant == "arm":
            if self.max_speed > self.cfg.joint_speed * 1.05:
                raise RuntimeError("joint_speed_limit")
            if self.max_accel > self.cfg.actual_accel_stop:
                raise RuntimeError("joint_acceleration_limit")
            limits = self.model.jnt_range[self.env._arm_joint_ids]
            if np.any(self.data.qpos[self.qadr] < limits[:, 0]) or np.any(self.data.qpos[self.qadr] > limits[:, 1]):
                raise RuntimeError("joint_position_limit")
            if self.data.xpos[self.env._block_body, 2] > .015:
                raise RuntimeError("object_lifted")
            if self.data.xmat[self.env._block_body].reshape(3, 3)[2, 2] < np.cos(np.deg2rad(15)):
                raise RuntimeError("object_tipped")
        q = self.pose()
        self.trace.append([float(self.data.time), *q, *self.pusher(), self.mode, self.phase,
                          int(self.contact()), *self.data.qpos[self.qadr]])
        if self.writer and self.tick % 2 == 0:
            self.frame()


def verify_terminal_hold(plant, reached, *, steps=50):
    """Require the initial pose AND every physics substep to satisfy the goal.

    An excursion permanently fails this hold even if the endpoint recovers.
    Callback-capable plants are required; endpoint-only plants fail closed.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("Positive integer hold steps required")
    if not getattr(plant, "supports_substep_observer", False):
        raise TypeError("Continuous hold requires a physics-substep observer")
    if getattr(plant, "substep_observer", None) is not None:
        raise RuntimeError("A substep observer is already installed")
    start = float(plant.data.time)
    initial = np.asarray(plant.pose())
    require_finite((initial, start), "SO101 hold initial state")
    record = dict(protocol="continuous_physics_substep_hold_v1", command_steps=steps,
                  checked_substeps=0, violated_substeps=0, initial_goal=bool(reached(initial)),
                  first_violation_time_s=None, continuous_success=False)

    def observe(q, time_s):
        require_finite((q, time_s), "SO101 hold state")
        record["checked_substeps"] += 1
        if not reached(q):
            record["violated_substeps"] += 1
            if record["first_violation_time_s"] is None:
                record["first_violation_time_s"] = time_s

    plant.phase = "HOLD"
    plant.substep_observer = observe
    try:
        for _ in range(steps):
            count = record["checked_substeps"]
            plant.step(np.zeros(2))
            if count == record["checked_substeps"]:
                raise RuntimeError("Plant omitted hold physics-substep observations")
        require_finite(plant.pose(), "SO101 final hold state")
        record["elapsed_seconds"] = float(plant.data.time) - start
        if not np.isclose(record["elapsed_seconds"], steps * .02, rtol=0, atol=1e-6):
            raise RuntimeError("Hold duration differs from the declared command periods")
        record["final_goal"] = bool(reached(plant.pose()))
        record["continuous_success"] = (record["initial_goal"] and record["final_goal"]
                                         and record["violated_substeps"] == 0)
        return record
    finally:
        plant.substep_observer = None


def rollout(cfg, q, seed, policy=None, video=None, telemetry=None, *, plant_factory=None,
            base_module=None, transfer=None, push=None):
    """Scratch-compatible explicit policy evaluation; defaults use v2 dynamics.

    An alternative geometry/plant must be passed explicitly and must provide
    physical substep observations. The function never installs global geometry.
    """
    if policy is None:
        raise ValueError("Explicit policy required; no implicit historical checkpoint loading")
    require_finite(asdict(cfg), "SO101 configuration")
    initial = np.asarray(q, dtype=float).copy()
    require_finite(initial, "SO101 initial pose")
    if initial.shape != (3,) or type(seed) is not int or type(cfg.max_actions) is not int or cfg.max_actions <= 0:
        raise ValueError("A three-component pose, integer seed and positive max_actions are required")
    base = base_module or campaign.base
    transfer = transfer or campaign.transfer
    push = push or campaign.push
    plant = (plant_factory or ContinuousHoldPlant)(cfg, seed, video)
    previous, reason, retained, changed, actions, descent = None, "action_limit", 0, 0, 0, []
    hold = None
    try:
        try:
            plant.reset(initial)
            for i in range(cfg.max_actions):
                require_finite(plant.pose(), "SO101 pose")
                if base.reached(plant.pose()):
                    reason = "success"
                    break
                obs = plant.observe()
                mode, travel, value, recovery = policy.choose(obs, previous, cfg.persistence, plant.pusher())
                require_finite((mode, travel, value), "SO101 policy decision")
                if isinstance(mode, (bool, np.bool_)) or not float(mode).is_integer() or not 0 <= mode < len(base.POINTS) or travel <= 0:
                    raise ValueError("Invalid SO101 mode or travel")
                mode = int(mode)
                surface_gap = base.shape(plant.pose()).distance(Point(plant.pusher())) - .004
                same = mode == previous and (plant.contact() or -.0003 <= surface_gap <= .0006)
                if same:
                    local = base.rot(-plant.pose()[2]) @ (plant.pusher() - plant.pose()[:2])
                    tangent = np.array([-base.NORMALS[mode, 1], base.NORMALS[mode, 0]])
                    same = abs(float((local - base.POINTS[mode]) @ tangent)) < .0015
                if cfg.persistence and same:
                    retained += 1
                else:
                    changed += int(previous is not None and previous != mode)
                    transfer(plant, mode, previous, policy)
                before = float(policy.value(plant.pose())[0])
                push(plant, mode, travel)
                after = float(policy.value(plant.pose())[0])
                require_finite((before, after), "SO101 certificate values")
                descent.append(after - before)
                previous, actions = mode, i + 1
            plant.settle()
            if base.reached(plant.pose()):
                reason = "success"
            hold = verify_terminal_hold(plant, base.reached)
            if reason == "success" and not hold["continuous_success"]:
                reason = "terminal_hold_failed"
        except RuntimeError as error:
            reason = str(error)
        final = plant.pose()
        require_finite(final, "SO101 final pose")
        report = dict(config=asdict(cfg), seed=seed, initial=initial.tolist(), final=final.tolist(),
            success=(reason == "success" and hold is not None and hold["continuous_success"]
                     and plant.transit_contacts == 0 and plant.arm_contacts == 0),
            reason=reason, actions=actions, sim_time_s=float(plant.data.time), distance_m=plant.distance,
            mode_changes=changed, repositions=len(plant.transfers), retained_actions=retained,
            position_mm=float(np.linalg.norm(final[:2]) * 1000), angle_deg=float(np.rad2deg(final[2])),
            coverage=base.coverage(final), transit_contact_substeps=plant.transit_contacts,
            arm_contact_substeps=plant.arm_contacts, unsafe_pairs=sorted(plant.unsafe_pairs),
            min_transit_clearance_mm=None if not np.isfinite(plant.min_clearance) else plant.min_clearance * 1000,
            max_joint_speed_rad_s=plant.max_speed, max_joint_acceleration_rad_s2=plant.max_accel,
            max_torque_nm=plant.max_torque, torque_saturation_substeps=plant.saturated,
            max_tracking_error_mm=plant.max_track * 1000, transfers=plant.transfers,
            push_delta_V=descent, terminal_hold=hold, protocol="so101_continuous_hold_v4",
            scope="SO101 simulation evaluation; no hardware or training-stability claim")
        if telemetry is not None:
            time = np.r_[0., [row[0] for row in plant.trace]]
            poses = np.vstack([initial, np.array([row[1:4] for row in plant.trace]).reshape(-1, 3)])
            if plant.data.time > time[-1] + 1e-10:
                time = np.r_[time, plant.data.time]
                poses = np.vstack([poses, final])
            values = np.concatenate([policy.value(part) for part in np.array_split(poses, max(1, len(poses) // 1024 + 1))])
            require_finite(values, "SO101 telemetry certificate")
            err = np.sum((poses[:, :2] / .05) ** 2, axis=1) + 2 * np.sin(poses[:, 2] / 2) ** 2
            report["integrated_error"] = float(np.sum(.5 * (err[:-1] + err[1:]) * np.diff(time)))
            report["complete_V_increase_fraction"] = float(np.mean(np.diff(values) > 1e-7)) if len(values) > 1 else 0.
            with Path(telemetry).open("xb") as file:
                np.savez_compressed(file, time=time, pose=poses, V=values, error=err)
        if video:
            plant.phase = "SUCCESS" if report["success"] else "FAILED"
            plant.frame()
            with Path(video).with_suffix(".json").open("x", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            with Path(video).with_suffix(".csv").open("x", encoding="utf-8") as file:
                file.write("time,x,y,yaw,pusher_x,pusher_y,mode,phase,contact,j1,j2,j3,j4,j5,j6\n")
                for row in plant.trace:
                    file.write(",".join(map(str, row)) + "\n")
        require_finite(report, "SO101 report")
        return report
    finally:
        plant.close()
