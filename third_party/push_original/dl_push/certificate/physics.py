"""Named-index MuJoCo adapter for real certificate-controller rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy

import numpy as np

from .math import rot2


@dataclass
class ContactObservation:
    active: bool
    force_world: np.ndarray
    point_world: np.ndarray
    normal_world: np.ndarray
    pair_count: int
    pair_forces_world: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    pair_points_world: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    pair_normals_world: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))

    @property
    def force_norm(self):
        return float(np.linalg.norm(self.force_world))

    def __post_init__(self):
        self.force_world = np.asarray(self.force_world, dtype=np.float64)
        self.point_world = np.asarray(self.point_world, dtype=np.float64)
        self.normal_world = np.asarray(self.normal_world, dtype=np.float64)
        self.pair_forces_world = np.asarray(self.pair_forces_world, dtype=np.float64)
        self.pair_points_world = np.asarray(self.pair_points_world, dtype=np.float64)
        self.pair_normals_world = np.asarray(self.pair_normals_world, dtype=np.float64)
        if self.force_world.shape != (3,) or self.point_world.shape != (3,) or self.normal_world.shape != (3,):
            raise ValueError("aggregate contact fields must have shape (3,)")
        if self.pair_forces_world.ndim != 2 or self.pair_forces_world.shape[1] != 3:
            raise ValueError("pair_forces_world must have shape (K,3)")
        if self.pair_points_world.shape != self.pair_forces_world.shape or self.pair_normals_world.shape != self.pair_forces_world.shape:
            raise ValueError("pair contact arrays must have matching shapes")


@dataclass
class PhysicsState:
    """Consistent state snapshot used by calibration, execution, and logging."""

    object_q: np.ndarray
    object_position: np.ndarray
    object_velocity: np.ndarray
    object_angular_velocity: np.ndarray
    pusher_position: np.ndarray
    pusher_velocity: np.ndarray
    roll: float
    pitch: float
    contact: ContactObservation

    def __post_init__(self):
        for name, shape in (
            ("object_q", (3,)),
            ("object_position", (3,)),
            ("object_velocity", (3,)),
            ("object_angular_velocity", (3,)),
            ("pusher_position", (3,)),
            ("pusher_velocity", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            setattr(self, name, value)
        if not np.isfinite([self.roll, self.pitch]).all():
            raise ValueError("roll and pitch must be finite")

    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        return {
            "object_q": self.object_q.tolist(),
            "object_position": self.object_position.tolist(),
            "object_velocity": self.object_velocity.tolist(),
            "object_angular_velocity": self.object_angular_velocity.tolist(),
            "pusher_position": self.pusher_position.tolist(),
            "pusher_velocity": self.pusher_velocity.tolist(),
            "roll": float(self.roll),
            "pitch": float(self.pitch),
            "contact": {
                "active": bool(self.contact.active),
                "force_world": self.contact.force_world.tolist(),
                "point_world": self.contact.point_world.tolist(),
                "normal_world": self.contact.normal_world.tolist(),
                "pair_count": int(self.contact.pair_count),
                "pair_forces_world": self.contact.pair_forces_world.tolist(),
                "pair_points_world": self.contact.pair_points_world.tolist(),
                "pair_normals_world": self.contact.pair_normals_world.tolist(),
            },
        }


@dataclass
class PhysicsSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    time: float


class MuJoCoAdapter:
    """Single-scene CPU MuJoCo adapter with no hard-coded qpos slices."""

    def __init__(self, xml, control_dt=0.20, expected_physics_dt=0.002):
        import mujoco

        self.mujoco = mujoco
        self.xml = str(xml)
        self.model = mujoco.MjModel.from_xml_path(self.xml)
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.physics_dt = float(self.model.opt.timestep)
        if not np.isclose(self.physics_dt, expected_physics_dt, atol=1e-12, rtol=0):
            raise ValueError(
                f"unexpected physics dt {self.physics_dt}; expected {expected_physics_dt}"
            )
        self.substeps = int(round(self.control_dt / self.physics_dt))
        if self.substeps < 1 or not np.isclose(self.substeps * self.physics_dt, self.control_dt, atol=1e-10):
            raise ValueError("control_dt must be an integer number of physics steps")
        self.object_body = self._body("object")
        self.pusher_body = self._body("pusher")
        self.object_joint = self._joint("object_free")
        if int(self.model.jnt_type[self.object_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("object_free must be a free joint")
        self.object_qpos = int(self.model.jnt_qposadr[self.object_joint])
        self.object_qvel = int(self.model.jnt_dofadr[self.object_joint])
        self.pusher_joints = (self._joint("pusher_x"), self._joint("pusher_y"))
        self.pusher_qpos = tuple(int(self.model.jnt_qposadr[j]) for j in self.pusher_joints)
        self.pusher_qvel = tuple(int(self.model.jnt_dofadr[j]) for j in self.pusher_joints)
        pusher_geoms = self._geoms_for_body(self.pusher_body)
        if not pusher_geoms:
            raise ValueError("pusher body has no collision geom")
        self.pusher_geoms = set(pusher_geoms)
        self.object_geoms = set(self._geoms_for_body(self.object_body))
        self._last_state = None

    def _body(self, name):
        index = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name))
        if index < 0:
            raise ValueError(f"missing body {name!r}")
        return index

    def _joint(self, name):
        index = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name))
        if index < 0:
            raise ValueError(f"missing joint {name!r}")
        return index

    def _geoms_for_body(self, body):
        return [int(index) for index, owner in enumerate(self.model.geom_bodyid) if int(owner) == int(body)]

    @staticmethod
    def _yaw_from_rotation(rotation):
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    @staticmethod
    def _roll_pitch(rotation):
        # ZYX decomposition, sufficient to detect the planar validity domain.
        pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        return roll, pitch

    def _contact_observation(self):
        force = np.zeros(3, dtype=np.float64)
        point = np.zeros(3, dtype=np.float64)
        normal = np.zeros(3, dtype=np.float64)
        pair_forces, pair_points, pair_normals = [], [], []
        count = 0
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & self.pusher_geoms and pair & self.object_geoms):
                continue
            values = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(self.model, self.data, index, values)
            # MuJoCo returns the contact-frame force.  The first row of frame
            # is the contact normal in world coordinates.
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            local_force = values[:3]
            world_force = frame.T @ local_force
            force += world_force
            contact_point = np.asarray(contact.pos, dtype=np.float64)
            contact_normal = frame[0].copy()
            point += contact_point
            normal += contact_normal
            pair_forces.append(world_force)
            pair_points.append(contact_point)
            pair_normals.append(contact_normal)
            count += 1
        if count:
            point /= count
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                normal /= normal_norm
        return ContactObservation(
            count > 0,
            force,
            point,
            normal,
            count,
            np.asarray(pair_forces, dtype=np.float64).reshape(-1, 3),
            np.asarray(pair_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(pair_normals, dtype=np.float64).reshape(-1, 3),
        )

    def observe(self):
        rotation = np.asarray(self.data.xmat[self.object_body], dtype=np.float64).reshape(3, 3)
        position = np.asarray(self.data.xipos[self.object_body], dtype=np.float64).copy()
        yaw = self._yaw_from_rotation(rotation)
        roll, pitch = self._roll_pitch(rotation)
        qvel = np.asarray(self.data.qvel[self.object_qvel:self.object_qvel + 6], dtype=np.float64)
        # Free-joint translational qvel is the body reference velocity; use the
        # COM velocity correction so the logged q is the declared xipos point.
        inertial_offset_world = rotation @ np.asarray(self.model.body_ipos[self.object_body], dtype=np.float64)
        angular = qvel[3:6].copy()
        velocity = qvel[:3] + np.cross(angular, inertial_offset_world)
        pusher_position = np.asarray(self.data.xpos[self.pusher_body], dtype=np.float64).copy()
        pusher_velocity = np.zeros(3, dtype=np.float64)
        pusher_velocity[:2] = [self.data.qvel[index] for index in self.pusher_qvel]
        state = PhysicsState(
            object_q=np.array([position[0], position[1], yaw]),
            object_position=position,
            object_velocity=velocity,
            object_angular_velocity=angular,
            pusher_position=pusher_position,
            pusher_velocity=pusher_velocity,
            roll=roll,
            pitch=pitch,
            contact=self._contact_observation(),
        )
        self._last_state = state.copy()
        return state

    def reset(self, object_q=(0.0, 0.0, 0.0), pusher_xy=(-0.18, 0.0)):
        object_q = np.asarray(object_q, dtype=np.float64)
        pusher_xy = np.asarray(pusher_xy, dtype=np.float64)
        if object_q.shape != (3,) or pusher_xy.shape != (2,) or not np.isfinite(object_q).all() or not np.isfinite(pusher_xy).all():
            raise ValueError("reset pose must be finite object_q=(x,y,yaw), pusher_xy=(x,y)")
        self.mujoco.mj_resetData(self.model, self.data)
        rotation = np.array([
            [np.cos(object_q[2]), -np.sin(object_q[2]), 0.0],
            [np.sin(object_q[2]), np.cos(object_q[2]), 0.0],
            [0.0, 0.0, 1.0],
        ])
        com_offset = rotation @ np.asarray(self.model.body_ipos[self.object_body], dtype=np.float64)
        self.data.qpos[self.object_qpos:self.object_qpos + 3] = [
            object_q[0] - com_offset[0],
            object_q[1] - com_offset[1],
            self.model.body_pos[self.object_body][2],
        ]
        self.data.qpos[self.object_qpos + 3:self.object_qpos + 7] = [
            np.cos(object_q[2] / 2.0), 0.0, 0.0, np.sin(object_q[2] / 2.0)
        ]
        self.data.qpos[list(self.pusher_qpos)] = pusher_xy
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self._last_state = None
        return self.observe()

    def snapshot(self):
        return PhysicsSnapshot(self.data.qpos.copy(), self.data.qvel.copy(), float(self.data.time))

    def restore(self, snapshot):
        if not isinstance(snapshot, PhysicsSnapshot):
            raise TypeError("expected PhysicsSnapshot")
        if snapshot.qpos.shape != self.data.qpos.shape or snapshot.qvel.shape != self.data.qvel.shape:
            raise ValueError("snapshot model layout differs")
        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        self.data.time = float(snapshot.time)
        self.mujoco.mj_forward(self.model, self.data)
        self._last_state = None
        return self.observe()

    def step(self, pusher_velocity, substeps=None):
        command = np.asarray(pusher_velocity, dtype=np.float64)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError("pusher_velocity must be finite shape (2,)")
        if np.any(np.abs(command) > 0.300001):
            raise ValueError("pusher velocity exceeds +/-0.30 m/s")
        count = self.substeps if substeps is None else int(substeps)
        if count < 1:
            raise ValueError("substeps must be positive")
        for _ in range(count):
            self.step_physics(command)
        return self.observe()

    def step_physics(self, pusher_velocity):
        """Advance exactly one physics step with a bounded XY command."""

        command = np.asarray(pusher_velocity, dtype=np.float64)
        if command.shape != (2,) or not np.isfinite(command).all() or np.any(np.abs(command) > .300001):
            raise ValueError("pusher_velocity must be finite shape (2,) and bounded")
        self.data.ctrl[:] = command
        self.mujoco.mj_step(self.model, self.data)


class PlanarMuJoCoAdapter(MuJoCoAdapter):
    """MuJoCo adapter for the explicit planar calibration reference scene.

    The certificate specification's free-body scene remains the default
    ``MuJoCoAdapter``.  This adapter is an explicitly named G3 diagnostic for
    isolating planar contact-response calibration before adding tilt dynamics;
    it still preserves floor friction and normal-only pusher contacts.
    """

    def __init__(self, xml, control_dt=0.20, expected_physics_dt=0.002):
        import mujoco

        self.mujoco = mujoco
        self.xml = str(xml)
        self.model = mujoco.MjModel.from_xml_path(self.xml)
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.physics_dt = float(self.model.opt.timestep)
        if not np.isclose(self.physics_dt, expected_physics_dt, atol=1e-12, rtol=0):
            raise ValueError("unexpected physics timestep")
        self.substeps = int(round(self.control_dt / self.physics_dt))
        if self.substeps < 1 or not np.isclose(self.substeps * self.physics_dt, self.control_dt, atol=1e-10):
            raise ValueError("control_dt must be an integer number of physics steps")
        self.object_body = self._body("object")
        self.pusher_body = self._body("pusher")
        self.object_joint_names = ("obj_x", "obj_y", "obj_z", "obj_yaw")
        self.object_joints = tuple(self._joint(name) for name in self.object_joint_names)
        self.object_qpos = tuple(int(self.model.jnt_qposadr[j]) for j in self.object_joints)
        self.object_qvel = tuple(int(self.model.jnt_dofadr[j]) for j in self.object_joints)
        self.object_joint = self.object_joints[0]
        self.pusher_joints = (self._joint("pusher_x"), self._joint("pusher_y"))
        self.pusher_qpos = tuple(int(self.model.jnt_qposadr[j]) for j in self.pusher_joints)
        self.pusher_qvel = tuple(int(self.model.jnt_dofadr[j]) for j in self.pusher_joints)
        pusher_geoms = self._geoms_for_body(self.pusher_body)
        if not pusher_geoms:
            raise ValueError("pusher body has no collision geom")
        self.pusher_geoms = set(pusher_geoms)
        self.object_geoms = set(self._geoms_for_body(self.object_body))
        self._last_state = None

    def observe(self):
        rotation = np.asarray(self.data.xmat[self.object_body], dtype=np.float64).reshape(3, 3)
        position = np.asarray(self.data.xipos[self.object_body], dtype=np.float64).copy()
        yaw = self._yaw_from_rotation(rotation)
        qvel = np.zeros(4, dtype=np.float64)
        for index, address in enumerate(self.object_qvel):
            qvel[index] = self.data.qvel[address]
        angular = np.array([0.0, 0.0, qvel[3]], dtype=np.float64)
        offset_world = rotation @ np.asarray(self.model.body_ipos[self.object_body], dtype=np.float64)
        velocity = np.array([qvel[0], qvel[1], qvel[2]], dtype=np.float64) + np.cross(angular, offset_world)
        pusher_position = np.asarray(self.data.xpos[self.pusher_body], dtype=np.float64).copy()
        pusher_velocity = np.zeros(3, dtype=np.float64)
        pusher_velocity[:2] = [self.data.qvel[index] for index in self.pusher_qvel]
        state = PhysicsState(
            object_q=np.array([position[0], position[1], yaw]),
            object_position=position,
            object_velocity=velocity,
            object_angular_velocity=angular,
            pusher_position=pusher_position,
            pusher_velocity=pusher_velocity,
            roll=0.0,
            pitch=0.0,
            contact=self._contact_observation(),
        )
        self._last_state = state.copy()
        return state

    def reset(self, object_q=(0., 0., 0.), pusher_xy=(-.18, 0.)):
        object_q = np.asarray(object_q, dtype=np.float64)
        pusher_xy = np.asarray(pusher_xy, dtype=np.float64)
        if object_q.shape != (3,) or pusher_xy.shape != (2,) or not np.isfinite(object_q).all() or not np.isfinite(pusher_xy).all():
            raise ValueError("reset pose must be finite")
        self.mujoco.mj_resetData(self.model, self.data)
        rotation = rot2(float(object_q[2]))
        offset = rotation @ np.asarray(self.model.body_ipos[self.object_body], dtype=np.float64)[:2]
        self.data.qpos[self.object_qpos[0]] = object_q[0] - offset[0]
        self.data.qpos[self.object_qpos[1]] = object_q[1] - offset[1]
        self.data.qpos[self.object_qpos[2]] = self.model.body_pos[self.object_body][2]
        self.data.qpos[self.object_qpos[3]] = object_q[2]
        self.data.qpos[list(self.pusher_qpos)] = pusher_xy
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self._last_state = None
        return self.observe()


class MJXBatchAdapter:
    """GPU/CPU batched one-tick predictor for one fixed certificate model.

    Static model topology is compiled once per shape.  The public state uses
    the same COM/yaw/velocity convention as :class:`MuJoCoAdapter`; changing a
    shape therefore creates a new adapter rather than padding different MJCF
    topologies into one ``vmap``.
    """

    def __init__(self, xml, batch, control_dt=0.20, expected_physics_dt=0.002):
        import mujoco
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        self.xml = str(xml)
        self.model_native = mujoco.MjModel.from_xml_path(self.xml)
        self.batch = int(batch)
        if self.batch < 1:
            raise ValueError("MJX batch must be positive")
        self.physics_dt = float(self.model_native.opt.timestep)
        self.control_dt = float(control_dt)
        if not np.isclose(self.physics_dt, expected_physics_dt, atol=1e-12, rtol=0):
            raise ValueError("unexpected MJX physics timestep")
        self.substeps = int(round(self.control_dt / self.physics_dt))
        if self.substeps < 1 or not np.isclose(self.substeps * self.physics_dt, self.control_dt, atol=1e-10):
            raise ValueError("control_dt must be an integer number of MJX substeps")
        self.object_body = int(mujoco.mj_name2id(self.model_native, mujoco.mjtObj.mjOBJ_BODY, "object"))
        self.object_joint = int(mujoco.mj_name2id(self.model_native, mujoco.mjtObj.mjOBJ_JOINT, "object_free"))
        self.pusher_joints = tuple(int(mujoco.mj_name2id(self.model_native, mujoco.mjtObj.mjOBJ_JOINT, name)) for name in ("pusher_x", "pusher_y"))
        if min(self.object_body, self.object_joint, *self.pusher_joints) < 0:
            raise ValueError("MJX certificate model is missing named joints/bodies")
        self.object_qpos = int(self.model_native.jnt_qposadr[self.object_joint])
        self.object_qvel = int(self.model_native.jnt_dofadr[self.object_joint])
        self.pusher_qpos = tuple(int(self.model_native.jnt_qposadr[j]) for j in self.pusher_joints)
        self.pusher_qvel = tuple(int(self.model_native.jnt_dofadr[j]) for j in self.pusher_joints)
        self.body_pos_z = float(self.model_native.body_pos[self.object_body][2])
        self.body_ipos = np.asarray(self.model_native.body_ipos[self.object_body], dtype=np.float64)
        self.backend = jax.default_backend()
        self.model = mjx.put_model(self.model_native)
        self.base = mjx.make_data(self.model_native)
        qpos_count = int(self.model_native.nq)
        qvel_count = int(self.model_native.nv)
        object_qpos = self.object_qpos
        object_qvel = self.object_qvel
        object_body = self.object_body
        pusher_qpos = self.pusher_qpos
        pusher_qvel = self.pusher_qvel
        body_ipos = jnp.asarray(self.body_ipos)
        body_z = self.body_pos_z
        substeps = self.substeps
        base = self.base
        model = self.model

        def one(state, action):
            yaw = state[3]
            c, s = jnp.cos(yaw), jnp.sin(yaw)
            offset_x = c * body_ipos[0] - s * body_ipos[1]
            offset_y = s * body_ipos[0] + c * body_ipos[1]
            qpos = jnp.zeros((qpos_count,), dtype=state.dtype)
            qpos = qpos.at[object_qpos:object_qpos + 3].set(
                jnp.array([state[0] - offset_x, state[1] - offset_y, body_z], dtype=state.dtype)
            )
            qpos = qpos.at[object_qpos + 3:object_qpos + 7].set(
                jnp.array([jnp.cos(yaw / 2.0), 0.0, 0.0, jnp.sin(yaw / 2.0)], dtype=state.dtype)
            )
            qpos = qpos.at[pusher_qpos[0]].set(state[8])
            qpos = qpos.at[pusher_qpos[1]].set(state[9])
            omega = state[7]
            root_vx = state[4] + omega * offset_y
            root_vy = state[5] - omega * offset_x
            qvel = jnp.zeros((qvel_count,), dtype=state.dtype)
            qvel = qvel.at[object_qvel:object_qvel + 6].set(
                jnp.array([root_vx, root_vy, state[6], 0.0, 0.0, omega], dtype=state.dtype)
            )
            qvel = qvel.at[pusher_qvel[0]].set(state[10])
            qvel = qvel.at[pusher_qvel[1]].set(state[11])
            data = base.replace(qpos=qpos, qvel=qvel, ctrl=action)
            data = __import__("jax").lax.fori_loop(
                0, substeps,
                lambda _, value: mjx.step(model, value),
                data,
            )
            rotation = data.xmat[object_body].reshape(3, 3)
            cyaw = jnp.arctan2(rotation[1, 0], rotation[0, 0])
            position = data.xipos[object_body]
            root_velocity = data.qvel[object_qvel:object_qvel + 3]
            angular = data.qvel[object_qvel + 3:object_qvel + 6]
            world_offset = rotation @ body_ipos
            velocity = root_velocity + jnp.cross(angular, world_offset)
            return jnp.array([
                position[0], position[1], position[2], cyaw,
                velocity[0], velocity[1], velocity[2], angular[2],
                data.qpos[pusher_qpos[0]], data.qpos[pusher_qpos[1]],
                data.qvel[pusher_qvel[0]], data.qvel[pusher_qvel[1]],
            ])

        self._predict = jax.jit(jax.vmap(one))
        self._compiled = False

    def predict(self, states, actions):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 12 or actions.shape != (len(states), 2):
            raise ValueError("MJX states/actions must have shapes (N,12)/(N,2)")
        if not 1 <= len(states) <= self.batch:
            raise ValueError("MJX request exceeds compiled batch")
        if not np.isfinite(states).all() or not np.isfinite(actions).all() or np.any(np.abs(actions) > .300001):
            raise ValueError("MJX states/actions must be finite and bounded")
        padded_states = np.repeat(states[-1:], self.batch, axis=0)
        padded_actions = np.zeros((self.batch, 2), dtype=np.float32)
        padded_states[:len(states)] = states
        padded_actions[:len(actions)] = actions
        import time
        started = time.perf_counter()
        result = np.asarray(self._predict(padded_states, padded_actions))[:len(states)]
        result = np.asarray(result, dtype=np.float32)
        result.block_until_ready() if hasattr(result, "block_until_ready") else None
        elapsed = time.perf_counter() - started
        self._compiled = True
        if not np.isfinite(result).all():
            raise FloatingPointError("MJX returned a non-finite state")
        return result.copy(), elapsed
