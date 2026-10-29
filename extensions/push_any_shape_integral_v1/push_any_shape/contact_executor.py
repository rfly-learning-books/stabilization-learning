"""Disk-only MuJoCo execution of PushAnyShape radial mode/speed commands.

The phase machine performs real approach, geometric bypass, engagement, push,
disengagement and settled hold. It does not rank predicted trajectories. Plans
are pure queries; step() alone commits physics and phase state. The caller owns
the certificate/integral state and must use returned accepted_dt for updates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .mode_library import disk_modes
from .shapes import DISK_NAME, DISK_RADIUS_M


# Exact XML literal copied from the retained single-disk engineering model.
# Its planar viscous joints provide resistance; the table has no collision.
# Neither the old task's d8 certificate nor its 18-candidate planner is imported.
MODEL_XML = """<mujoco model="isolated_single_disk_gap_v2">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" cone="pyramidal" jacobian="auto" iterations="100" tolerance="1e-8"/>
  <size njmax="1000" nconmax="200"/>
  <default>
    <joint margin="0"/>
    <geom friction="0.85 0.01 0.001" solimp="0.90 0.95 0.001" solref="0.012 1" margin="0"/>
    <velocity ctrllimited="true" ctrlrange="-0.1 0.1" kv="4"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 -0.045" size="2 2 .1" contype="0" conaffinity="0"/>
    <geom name="table" type="box" pos="-.10 0 -.02" size=".48 .28 .02" contype="0" conaffinity="0"/>
    <body name="goal" mocap="true" pos="0 0 .0008">
      <geom name="goal_marker" type="cylinder" size=".015 .0003" contype="0" conaffinity="0"/>
    </body>
    <body name="disk" pos="0 0 .004">
      <joint name="disk_x" type="slide" axis="1 0 0" limited="true" range="-.055 .060" damping="1"/>
      <joint name="disk_y" type="slide" axis="0 1 0" limited="true" range="-.060 .060" damping="1"/>
      <joint name="disk_yaw" type="hinge" axis="0 0 1" limited="false" damping=".08"/>
      <geom name="disk_geom" type="cylinder" size=".015 .004" mass=".025"/>
    </body>
    <body name="pusher" pos="0 0 .0045">
      <joint name="pusher_x" type="slide" axis="1 0 0" limited="true" range="-.065 .070" damping=".06"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" limited="true" range="-.070 .070" damping=".06"/>
      <geom name="pusher_geom" type="cylinder" size=".004 .0045" mass=".030"/>
      <geom name="pusher_marker" type="sphere" pos="0 0 .0045" size=".0055" mass=".0001" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator><velocity name="x_velocity" joint="pusher_x"/><velocity name="y_velocity" joint="pusher_y"/></actuator>
</mujoco>"""


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_jsonable,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _norm_clip(value, limit):
    value = np.asarray(value, dtype=float)
    return value * min(1., limit / max(float(np.linalg.norm(value)), 1e-15))


def _wrap(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


@dataclass(frozen=True)
class ExecutorConfig:
    dt: float = .02
    physics_dt: float = .005
    route_gap_m: float = .006
    terminal_retreat_gap_m: float = .004
    required_gap_m: float = .002
    waypoint_tolerance_m: float = .00025
    contact_gap_tolerance_m: float = .0003
    angle_tolerance_rad: float = .025
    bypass_increment_rad: float = .20
    transfer_speed_mps: float = .03
    retreat_speed_mps: float = .015
    engage_speed_mps: float = .005
    maximum_push_speed_mps: float = .06
    tangent_gain_per_s: float = 4.
    tangent_limit_mps: float = .015
    position_tolerance_m: float = .005
    velocity_tolerance_mps: float = .003
    angular_velocity_tolerance_radps: float = .05

    def __post_init__(self):
        numbers = list(asdict(self).values())
        if not np.isfinite(numbers).all() or min(numbers) <= 0:
            raise ValueError("Executor configuration must be finite and positive")
        if abs(self.dt / self.physics_dt - round(self.dt / self.physics_dt)) > 1e-12:
            raise ValueError("Control dt must contain whole physical substeps")
        if self.dt != .02 or self.physics_dt != .005:
            raise ValueError("Version v1 pins .02s/.005s timing")
        if self.terminal_retreat_gap_m <= self.required_gap_m or self.route_gap_m <= self.terminal_retreat_gap_m:
            raise ValueError("Routing and terminal gaps must exceed task separation")


@dataclass(frozen=True)
class ExecutionProposal:
    origin_digest: str
    requested_mode: int
    requested_speed_mps: float
    terminal_requested: bool
    command_world_mps: tuple
    phase: str
    phase_state: dict
    diagnostics: dict

    @property
    def digest(self):
        return _digest(asdict(self))


class DiskContactExecutor:
    """Actual disk physics; unsupported shape names fail before model creation.

    Observation order is [disk_x,disk_y,yaw,pusher_x,pusher_y,
    disk_vx,disk_vy,yaw_rate,pusher_vx,pusher_vy]. Every qpos write occurs only
    in reset or a named full-state restore. Normal execution writes ctrl only.
    """
    control_call_is_pure = True
    physical_step_autograd = False
    physical_supported_shapes = (DISK_NAME,)

    def __init__(self, shape_name=DISK_NAME, *, config=None):
        if shape_name != DISK_NAME:
            raise NotImplementedError("This contact backend implements Disk only; original 32 shapes remain numeric")
        import mujoco
        if mujoco.__version__ != "3.2.3":
            raise RuntimeError("The version v1 contact protocol pins MuJoCo 3.2.3")
        self.mj = mujoco
        self.config = config or ExecutorConfig()
        self.library = disk_modes()
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = mujoco.mj_stateSize(self.model, self.state_spec)
        self.xml_sha256 = hashlib.sha256(MODEL_XML.encode()).hexdigest()
        self.disk_bounds = np.array([[-.055, .060], [-.060, .060]])
        self.pusher_bounds = np.array([[-.065, .070], [-.070, .070]])
        self._model_digest = self._model_signature()
        self._ready = False
        self._failed = False
        self.goal_xy = np.zeros(2)
        self._phase_state = self._initial_phase()
        self._budget = self._initial_budget()

    @staticmethod
    def _initial_phase():
        return {"phase": "IDLE", "current_mode": -1, "requested_mode": -1,
                "requested_speed_mps": 0., "bypass_sign": 0, "phase_elapsed": 0.}

    @staticmethod
    def _initial_budget():
        return {"control_attempts": 0, "accepted_control_steps": 0,
                "attempted_substeps": 0, "accepted_substeps": 0}

    def _model_signature(self):
        model = self.model
        return _digest({"xml": self.xml_sha256, "config": asdict(self.config),
            "timestep": model.opt.timestep, "gravity": model.opt.gravity,
            "solver": int(model.opt.solver), "iterations": model.opt.iterations,
            "tolerance": model.opt.tolerance, "integrator": int(model.opt.integrator),
            **{name: getattr(model, name) for name in (
                "body_mass", "body_inertia", "body_pos", "jnt_range", "jnt_axis",
                "dof_damping", "geom_size", "geom_pos", "geom_friction", "geom_margin",
                "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity",
                "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange")}})

    def _assert_fixed(self):
        if self._model_signature() != self._model_digest:
            raise ValueError("Physical model/config mutation outside this versioned protocol")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError("External forces are outside the contact protocol")
        if self._ready and not np.array_equal(self.goal_xy, self._goal_snapshot):
            raise ValueError("Goal changed outside reset/restore")

    def _integration(self):
        values = np.empty(self.state_size)
        self.mj.mj_getState(self.model, self.data, values, self.state_spec)
        return values

    def _restore_integration(self, values):
        self.mj.mj_setState(self.model, self.data, np.asarray(values, float), self.state_spec)
        self.mj.mj_forward(self.model, self.data)
        # mj_forward computes derived contact state; reapply solver warmstart,
        # ctrl and all integration fields from the saved snapshot exactly.
        self.mj.mj_setState(self.model, self.data, np.asarray(values, float), self.state_spec)

    def observation(self):
        return np.r_[self.data.qpos.copy(), self.data.qvel.copy()]

    @staticmethod
    def surface_gap(state):
        state = np.asarray(state, float)
        return float(np.linalg.norm(state[3:5] - state[:2]) - (DISK_RADIUS_M + .004))

    def failure_reason(self, state):
        state = np.asarray(state, float)
        if state.shape != (10,) or not np.isfinite(state).all():
            return "nonfinite_or_invalid_state"
        if np.any(state[:2] <= self.disk_bounds[:, 0]) or np.any(state[:2] >= self.disk_bounds[:, 1]):
            return "disk_joint_boundary"
        if np.any(state[3:5] <= self.pusher_bounds[:, 0]) or np.any(state[3:5] >= self.pusher_bounds[:, 1]):
            return "pusher_joint_boundary"
        if self.surface_gap(state) < -.001:
            return "excessive_soft_penetration"
        if np.linalg.norm(state[5:7]) > 2. or np.linalg.norm(state[8:10]) > 2. or abs(state[7]) > 20.:
            return "velocity_safety_limit"
        return None

    def physical_goal_conditions(self, state=None):
        state = self.observation() if state is None else np.asarray(state, float)
        cfg = self.config
        return {"position": bool(np.linalg.norm(state[:2] - self.goal_xy) <= cfg.position_tolerance_m),
            "disk_stopped": bool(np.linalg.norm(state[5:7]) <= cfg.velocity_tolerance_mps),
            "yaw_rate_stopped": bool(abs(state[7]) <= cfg.angular_velocity_tolerance_radps),
            "pusher_stopped": bool(np.linalg.norm(state[8:10]) <= cfg.velocity_tolerance_mps),
            "pusher_separated": bool(self.surface_gap(state) >= cfg.required_gap_m),
            "safe": self.failure_reason(state) is None,
            "yaw_free": True}

    def reset(self, *, disk_xy=(-.025, 0.), pusher_xy=(-.05, 0.), goal_xy=(0., 0.),
              yaw=0., qvel=None, seed=None, options=None):
        self._assert_fixed()
        if options is not None:
            options = dict(options)
            if set(options) - {"disk_xy", "pusher_xy", "goal_xy", "yaw", "qvel"}:
                raise ValueError("Unknown disk executor reset option")
            disk_xy, pusher_xy = options.get("disk_xy", disk_xy), options.get("pusher_xy", pusher_xy)
            goal_xy, yaw, qvel = options.get("goal_xy", goal_xy), options.get("yaw", yaw), options.get("qvel", qvel)
        disk, pusher, goal = (np.asarray(v, float) for v in (disk_xy, pusher_xy, goal_xy))
        velocity = np.zeros(5) if qvel is None else np.asarray(qvel, float)
        if any(v.shape != (2,) for v in (disk, pusher, goal)) or velocity.shape != (5,) or not np.isfinite(np.r_[disk, pusher, goal, yaw, velocity]).all():
            raise ValueError("Reset requires finite disk/pusher/goal xy, yaw and qvel[5]")
        if np.any(goal < [-.032, -.037]) or np.any(goal > [.037, .037]):
            raise ValueError("Goal must admit retreat and workspace clearance")
        initial = np.r_[disk, float(yaw), pusher, velocity]
        if self.failure_reason(initial) is not None or self.surface_gap(initial) < 0.:
            raise ValueError("Unsafe or overlapping reset; no automatic projection")
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = initial[:5]
        self.data.qvel[:] = initial[5:]
        self.data.mocap_pos[0] = [goal[0], goal[1], .0008]
        self.mj.mj_forward(self.model, self.data)
        self.goal_xy = goal.copy()
        self._goal_snapshot = goal.copy()
        self._seed = seed
        self._ready, self._failed = True, False
        self._phase_state, self._budget = self._initial_phase(), self._initial_budget()
        return self.observation()

    def _origin_digest(self):
        return _digest({"integration": self._integration(), "phase_state": self._phase_state,
            "goal": self.goal_xy, "failed": self._failed, "budget": self._budget})

    def _contacts(self):
        records = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            force = np.zeros(6)
            self.mj.mj_contactForce(self.model, self.data, index, force)
            records.append({"geom1": int(contact.geom1), "geom2": int(contact.geom2),
                "distance_m": float(contact.dist), "force_contact_frame": force.tolist()})
        return records

    def plan(self, mode, speed_mps, *, terminal_requested=False):
        """Pure query returning a mode execution proposal, never an integral update."""
        if not self._ready or self._failed:
            raise RuntimeError("Reset before planning or after a physical failure")
        self._assert_fixed()
        if isinstance(mode, (bool, np.bool_)) or not np.isscalar(mode) or not float(mode).is_integer() or not 0 <= int(mode) < 12:
            raise ValueError("Require a radial mode integer in [0,11]")
        speed = float(speed_mps)
        if not np.isfinite(speed) or not 0. <= speed <= self.config.maximum_push_speed_mps:
            raise ValueError("Positive inward speed must lie in [0,.06] m/s")
        mode, cfg = int(mode), self.config
        state = self.observation()
        disk, pusher = state[:2], state[3:5]
        relative = pusher - disk
        radius = float(np.linalg.norm(relative))
        if radius <= 1e-12:
            raise ValueError("Undefined pusher radial direction")
        radial = relative / radius
        rotation = np.array([[np.cos(state[2]), -np.sin(state[2])], [np.sin(state[2]), np.cos(state[2])]])
        phase_state = deepcopy(self._phase_state)
        position_reached = np.linalg.norm(disk - self.goal_xy) <= cfg.position_tolerance_m
        # Task arrival always disengages. A changing certificate cannot command
        # another push while the object is already inside position tolerance.
        retiring = bool(terminal_requested or position_reached)
        desired = phase_state["requested_mode"]
        if phase_state["phase"] in ("IDLE", "PUSH", "SETTLED", "DISENGAGE") or desired < 0:
            desired = mode
            executed_speed = speed
        else:
            executed_speed = float(phase_state["requested_speed_mps"])
        normal = rotation @ self.library.normals[desired]
        tangent = np.array([-normal[1], normal[0]])
        angle = _wrap(math.atan2(normal[1], normal[0]) - math.atan2(radial[1], radial[0]))
        route_radius = DISK_RADIUS_M + .004 + cfg.route_gap_m
        gap = radius - (DISK_RADIUS_M + .004)
        command = np.zeros(2)
        bypass_sign = int(phase_state["bypass_sign"])
        engaged = bool(self.data.ncon > 0 and gap <= cfg.contact_gap_tolerance_m)
        if retiring:
            desired = mode
            if gap < cfg.terminal_retreat_gap_m - cfg.waypoint_tolerance_m:
                phase = "DISENGAGE"
                command = radial * min(cfg.retreat_speed_mps, (cfg.terminal_retreat_gap_m - gap) / cfg.dt)
            else:
                phase = "SETTLED"
        elif speed == 0.:
            # A caller may safely cancel a latched transfer. A zero macro
            # action outside the goal never starts approach or engagement.
            # At-goal zero remains the explicit disengagement branch above.
            phase, desired, executed_speed, bypass_sign = "IDLE", mode, 0., 0
        elif abs(angle) > cfg.angle_tolerance_rad:
            if radius < route_radius - cfg.waypoint_tolerance_m:
                phase = "RETRACT_FOR_BYPASS"
                command = radial * min(cfg.retreat_speed_mps, (route_radius - radius) / cfg.dt)
            else:
                phase = "BYPASS"
                if bypass_sign == 0:
                    bypass_sign = 1 if angle >= 0. else -1
                # Release the chosen side when less than one waypoint remains.
                if abs(angle) <= cfg.bypass_increment_rad:
                    delta = angle
                else:
                    delta = bypass_sign * cfg.bypass_increment_rad
                theta = math.atan2(radial[1], radial[0]) + delta
                waypoint = disk + route_radius * np.array([np.cos(theta), np.sin(theta)])
                command = _norm_clip((waypoint - pusher) / cfg.dt, cfg.transfer_speed_mps)
        elif not engaged:
            phase = "APPROACH" if gap > .001 else "ENGAGE"
            # A measured contact is required. Do not stop at a positive shell.
            inward = min(cfg.engage_speed_mps, max(cfg.engage_speed_mps / 2., gap / cfg.dt))
            lateral = -cfg.tangent_gain_per_s * float(tangent @ relative)
            command = -normal * inward + tangent * np.clip(lateral, -cfg.tangent_limit_mps, cfg.tangent_limit_mps)
            bypass_sign = 0
        else:
            phase = "PUSH"
            # Absolute normal pusher velocity matches the original C input
            # convention. Tangential feedback tracks the material radial ray.
            lateral = float(tangent @ state[5:7]) - cfg.tangent_gain_per_s * float(tangent @ relative)
            command = -normal * executed_speed + tangent * np.clip(lateral, -cfg.tangent_limit_mps, cfg.tangent_limit_mps)
            phase_state["current_mode"] = desired
            bypass_sign = 0
        requested_command = command.copy()
        # Actuator/workspace guards constrain commands; they never clip state.
        low = np.minimum(0., (self.pusher_bounds[:, 0] + 1e-5 - pusher) / cfg.dt)
        high = np.maximum(0., (self.pusher_bounds[:, 1] - 1e-5 - pusher) / cfg.dt)
        command = np.clip(np.minimum(np.maximum(command, low), high), -.1, .1)
        previous_phase = phase_state["phase"]
        phase_state.update(phase=phase, requested_mode=desired, requested_speed_mps=executed_speed, bypass_sign=bypass_sign,
            phase_elapsed=phase_state["phase_elapsed"] if phase == previous_phase else 0.)
        diagnostics = {"high_level_mode": mode, "latched_mode": desired,
            "mode_speed_mps": executed_speed, "phase": phase, "engaged": engaged,
            "nominal_C_prediction_applicable": phase == "PUSH",
            "response_source": self.library.response_kind,
            "actual_contact_response_calibrated": False,
            "position_reached": bool(position_reached), "terminal_disengagement": retiring,
            "surface_gap_m": gap, "body_mode_angle_error_rad": angle,
            "command_saturated": bool(not np.array_equal(command, requested_command)),
            "requested_command_world_mps": requested_command.tolist()}
        return ExecutionProposal(self._origin_digest(), mode, speed, bool(terminal_requested),
            tuple(command.tolist()), phase, phase_state, diagnostics)

    query = plan

    def step(self, proposal, speed_mps=None, *, goal_hold=False):
        if not isinstance(proposal, ExecutionProposal):
            if speed_mps is None:
                raise TypeError("step requires ExecutionProposal or explicit mode and speed")
            proposal = self.plan(proposal, speed_mps, terminal_requested=goal_hold)
        elif speed_mps is not None or goal_hold:
            raise ValueError("ExecutionProposal already binds its terminal/speed request")
        expected = self.plan(proposal.requested_mode, proposal.requested_speed_mps,
            terminal_requested=proposal.terminal_requested)
        if proposal.digest != expected.digest:
            raise ValueError("Stale or modified execution proposal")
        before = self.observation()
        start_time = float(self.data.time)
        accepted = self._integration()
        substeps, failure = [], None
        self._budget["control_attempts"] += 1
        for _ in range(round(self.config.dt / self.config.physics_dt)):
            self._budget["attempted_substeps"] += 1
            self.data.ctrl[:] = proposal.command_world_mps
            try:
                self.mj.mj_step(self.model, self.data)
                current = self.observation()
                failure = self.failure_reason(current)
            except Exception as exc:
                failure = "physics_exception:" + type(exc).__name__ + ":" + str(exc)
            if failure is not None:
                rejected = {"state": self.observation().tolist(), "time": float(self.data.time), "reason": failure}
                self._restore_integration(accepted)
                self._failed = True
                break
            accepted = self._integration()
            self._budget["accepted_substeps"] += 1
            substeps.append({"time": float(self.data.time), "state": current.tolist(),
                "integration": accepted.tolist(), "contacts": self._contacts()})
        actual_dt = float(self.data.time - start_time)
        if actual_dt > 0.:
            self._phase_state = deepcopy(proposal.phase_state)
            self._phase_state["phase_elapsed"] += actual_dt
            self._budget["accepted_control_steps"] += 1
        after = self.observation()
        return {"before": before, "after": after, "accepted_dt": actual_dt, "actual_dt": actual_dt,
            "requested_dt": self.config.dt, "accepted_substeps": len(substeps),
            "substep_records": substeps, "failure": failure is not None,
            "failure_reason": failure, "rejected_substep": rejected if failure else None,
            "phase": proposal.phase, "phase_state": deepcopy(self._phase_state),
            "applied_command_world_mps": np.asarray(proposal.command_world_mps),
            "command": np.asarray(proposal.command_world_mps), "blocked_axes": None,
            "blocked_axes_available": False,
            "antiwindup_semantics": "Executor saturation does not establish which object-error axes are blocked; caller integral uses declared hard bounds/leak only",
            "diagnostics": deepcopy(proposal.diagnostics), "contacts": self._contacts(),
            "goal_conditions": self.physical_goal_conditions(after), "budget": deepcopy(self._budget),
            "time": float(self.data.time)}

    def snapshot(self):
        self._assert_fixed()
        record = {"schema": "disk_contact_executor_v1", "xml_sha256": self.xml_sha256,
            "model_digest": self._model_digest, "config": asdict(self.config),
            "integration": self._integration().copy(), "goal_xy": self.goal_xy.copy(),
            "phase_state": deepcopy(self._phase_state), "budget": deepcopy(self._budget),
            "ready": self._ready, "failed": self._failed, "seed": getattr(self, "_seed", None)}
        record["snapshot_digest"] = _digest(record)
        return record

    state_dict = snapshot

    def restore(self, snapshot):
        self._assert_fixed()
        required = {"schema", "xml_sha256", "model_digest", "config", "integration", "goal_xy",
                    "phase_state", "budget", "ready", "failed", "seed", "snapshot_digest"}
        if not isinstance(snapshot, dict) or set(snapshot) != required:
            raise ValueError("Executor snapshot fields differ from the complete schema")
        if _digest({key: value for key, value in snapshot.items() if key != "snapshot_digest"}) != snapshot["snapshot_digest"]:
            raise ValueError("Executor snapshot payload checksum mismatch")
        if snapshot["schema"] != "disk_contact_executor_v1" or snapshot["xml_sha256"] != self.xml_sha256 or snapshot["model_digest"] != self._model_digest or snapshot["config"] != asdict(self.config):
            raise ValueError("Executor snapshot model/protocol mismatch")
        integration = np.asarray(snapshot["integration"], float)
        if integration.shape != (self.state_size,) or not np.isfinite(integration).all():
            raise ValueError("Invalid full solver/integration snapshot")
        goal = np.asarray(snapshot["goal_xy"], float)
        if goal.shape != (2,) or not np.isfinite(goal).all() or np.any(goal < [-.032, -.037]) or np.any(goal > [.037, .037]):
            raise ValueError("Invalid snapshot goal")
        phase_state, budget = snapshot["phase_state"], snapshot["budget"]
        if not isinstance(phase_state, dict) or set(phase_state) != set(self._initial_phase()):
            raise ValueError("Invalid phase-state structure")
        if phase_state["phase"] not in ("IDLE", "PUSH", "SETTLED", "DISENGAGE", "RETRACT_FOR_BYPASS", "BYPASS", "APPROACH", "ENGAGE"):
            raise ValueError("Invalid execution phase")
        for key in ("current_mode", "requested_mode"):
            if type(phase_state[key]) is not int or not -1 <= phase_state[key] < 12:
                raise ValueError("Invalid saved mode index")
        if type(phase_state["bypass_sign"]) is not int or phase_state["bypass_sign"] not in (-1, 0, 1):
            raise ValueError("Invalid bypass direction")
        elapsed, speed = phase_state["phase_elapsed"], phase_state["requested_speed_mps"]
        if not np.isfinite([elapsed, speed]).all() or elapsed < 0. or not 0. <= speed <= self.config.maximum_push_speed_mps:
            raise ValueError("Invalid saved phase duration/speed")
        if not isinstance(budget, dict) or set(budget) != set(self._initial_budget()) or any(type(value) is not int or value < 0 for value in budget.values()):
            raise ValueError("Invalid saved execution counters")
        if budget["accepted_substeps"] > budget["attempted_substeps"] or budget["accepted_control_steps"] > budget["control_attempts"] or budget["attempted_substeps"] > 4 * budget["control_attempts"]:
            raise ValueError("Inconsistent execution counters")
        if type(snapshot["ready"]) is not bool or type(snapshot["failed"]) is not bool:
            raise ValueError("Invalid snapshot ready/failure flags")
        # mjSTATE_INTEGRATION starts with TIME,QPOS,QVEL in MuJoCo 3.2.3.
        # Validate the saved physical domain before mutating the live data.
        state = np.r_[integration[1:1 + self.model.nq], integration[1 + self.model.nq:1 + self.model.nq + self.model.nv]]
        if snapshot["ready"] and (integration[0] < 0. or self.failure_reason(state) is not None):
            raise ValueError("Invalid physical state in ready snapshot")
        self._restore_integration(integration)
        self.goal_xy = goal.copy()
        self._goal_snapshot = self.goal_xy.copy()
        self._phase_state = deepcopy(snapshot["phase_state"])
        self._budget = deepcopy(snapshot["budget"])
        self._ready, self._failed = bool(snapshot["ready"]), bool(snapshot["failed"])
        self._seed = snapshot["seed"]

    load_state_dict = restore

    @property
    def phase_state(self):
        return deepcopy(self._phase_state)

    @property
    def state(self):
        return self.observation()

    @property
    def pose(self):
        return self.data.qpos[:3].copy()

    @property
    def phase(self):
        return self._phase_state["phase"]

    @property
    def current_mode(self):
        return self._phase_state["current_mode"]

    @property
    def busy(self):
        return self.phase in ("APPROACH", "ENGAGE", "RETRACT_FOR_BYPASS", "BYPASS")

    @property
    def current_action(self):
        return (self._phase_state["requested_mode"], self._phase_state["requested_speed_mps"])

    @property
    def elapsed_steps(self):
        return self._budget["accepted_control_steps"]

    @property
    def budget(self):
        return deepcopy(self._budget)

    def close(self):
        pass
