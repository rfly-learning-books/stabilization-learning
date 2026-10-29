"""Export reset adaptation for bundled ManiSkill partial-lane setter semantics.

Episode geometry/distributions copied from the specified export. Only actor
pose/velocity writes use selected-lane tensors instead of full-batch tensors.
"""
import numpy as np
import torch
from mani_skill.utils.structs import Pose
from ca_glpr_unseen_shape.env.mixed_shapes import ManiSkillPushMixedShapesEnv


class PartialResetScene(ManiSkillPushMixedShapesEnv):
    _FIXED_ORIGIN_RESET = False
    _RESET_PUSHER_CLEARANCE = .002

    def _sample_free_pusher(self, shape_ids):
        """Uniform rejection sampling with exact circle-vs-box clearance."""
        radius = float(self.agent.pusher_radius)
        clearance = self._RESET_PUSHER_CLEARANCE
        limit = min(self._PUSHER_JOINT_LIMIT - .001,
                    self._POLICY_WORLD_SIZE / 2 - radius - clearance)
        positions = torch.empty(len(shape_ids), 2, device=self.device)
        pending = torch.ones(len(shape_ids), dtype=torch.bool, device=self.device)
        for _ in range(512):
            ids = pending.nonzero().flatten()
            if not len(ids):
                return positions
            proposal = (2 * torch.rand(len(ids), 2, device=self.device) - 1) * limit
            free = torch.ones(len(ids), dtype=torch.bool, device=self.device)
            for shape_id, spec in enumerate(self._shape_specs):
                selected = shape_ids[ids] == shape_id
                for center, half in spec.boxes:
                    offset = (proposal - torch.tensor(center, device=self.device)).abs()
                    distance = (offset - torch.tensor(half, device=self.device)).clamp_min(0).norm(dim=-1)
                    free &= ~selected | (distance >= radius + clearance)
            positions[ids[free]] = proposal[free]
            pending[ids[free]] = False
        raise RuntimeError("Could not sample a non-overlapping pusher reset")

    @staticmethod
    def _scatter_pose(actor, env_idx, xyz, q):
        actor.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        b = len(env_idx)
        device = self.device
        if self.fixed_shape is not None:
            fixed_id = self._SHAPE_NAMES.index(self.fixed_shape)
            shape_ids = torch.full((b,), fixed_id, dtype=torch.long, device=device)
        else:
            shape_ids = torch.randint(0, len(self._shape_specs), (b,), device=device)
        self._active_shape_id[env_idx] = shape_ids
        self._active_shape_global_id[env_idx] = self._shape_global_ids[shape_ids]
        self._success_hold_count[env_idx] = 0

        radii = torch.tensor(
            [spec.radius() for spec in self._shape_specs],
            dtype=torch.float32,
            device=device,
        )[shape_ids]
        inner_limit = self._POLICY_WORLD_SIZE / 2.0 - self._RESET_VIEW_CLEARANCE - radii
        nominal_low = torch.full((b, 2), -0.10, device=device)
        nominal_high = torch.full((b, 2), 0.10, device=device)
        feasible_low = torch.maximum(nominal_low, -inner_limit[:, None])
        feasible_high = torch.minimum(nominal_high, inner_limit[:, None])
        tee_xy = feasible_low + torch.rand((b, 2), device=device) * (
            feasible_high - feasible_low
        )
        if self._FIXED_ORIGIN_RESET:
            tee_xy = torch.zeros((b, 2), device=device)
            tee_yaw = torch.zeros(b, device=device)
        elif self.reset_mode == "full":
            tee_yaw = torch.rand((b,), device=device) * (2.0 * np.pi) - np.pi
        elif self.reset_mode == "contact_translation":
            tee_yaw = torch.full(
                (b,), self._GOAL_YAW, dtype=torch.float32, device=device
            )
        else:
            tee_yaw = (
                torch.full((b,), self._GOAL_YAW, dtype=torch.float32, device=device)
                + (torch.rand((b,), device=device) - 0.5) * np.pi
            )
        tee_xyz = torch.cat(
            (
                tee_xy,
                torch.full((b, 1), self._BLOCK_Z, dtype=torch.float32, device=device),
            ),
            dim=-1,
        )
        tee_q = self._yaw_quaternion(tee_yaw)

        goal_xyz = torch.tensor(
            [*self._GOAL_XY, self._GOAL_Z], dtype=torch.float32, device=device
        ).repeat(b, 1)
        goal_yaw = torch.full((b,), self._GOAL_YAW, dtype=torch.float32, device=device)
        goal_q = self._yaw_quaternion(goal_yaw)

        identity_q = torch.zeros((b, 4), dtype=torch.float32, device=device)
        identity_q[:, 0] = 1.0

        for index, spec in enumerate(self._shape_specs):
            active = shape_ids == index
            far_x, far_y = self._far_xy(index)
            tee_pose = torch.tensor(
                [far_x, far_y, self._BLOCK_Z], dtype=torch.float32, device=device
            ).repeat(b, 1)
            goal_pose = torch.tensor(
                [far_x, far_y, self._GOAL_Z], dtype=torch.float32, device=device
            ).repeat(b, 1)
            tee_pose[active] = tee_xyz[active]
            tee_q_pose = identity_q.clone()
            tee_q_pose[active] = tee_q[active]
            goal_pose[active] = goal_xyz[active]
            goal_q_pose = identity_q.clone()
            goal_q_pose[active] = goal_q[active]

            self._scatter_pose(self.tees[index], env_idx, tee_pose, tee_q_pose)
            self._scatter_pose(self.goals[index], env_idx, goal_pose, goal_q_pose)

            linear = self.tees[index].get_linear_velocity()
            angular = self.tees[index].get_angular_velocity()
            linear[env_idx] = 0.0
            angular[env_idx] = 0.0
            self.tees[index].set_linear_velocity(linear[env_idx])
            self.tees[index].set_angular_velocity(angular[env_idx])

        if self._FIXED_ORIGIN_RESET:
            pusher_xy = self._sample_free_pusher(shape_ids)
        elif self.reset_mode == "full":
            pusher_xy = self._balanced_edge_reset(tee_xy)
        else:
            # Contact curricula keep the object pose distribution broad but
            # place the pusher just outside the *actual compound boundary*.
            # A bounding-circle approximation is too conservative for thin
            # arms (and made the first curriculum run mostly non-contact).
            radial_angle = torch.rand((b,), device=device) * (2.0 * np.pi)
            outward_world = torch.stack(
                (torch.cos(radial_angle), torch.sin(radial_angle)), dim=-1
            )
            outward_local = self._rotate(outward_world, -tee_yaw)
            support_distance = torch.zeros((b,), dtype=torch.float32, device=device)
            for index, spec in enumerate(self._shape_specs):
                selected = shape_ids == index
                if not selected.any():
                    continue
                vertices = torch.tensor(
                    [
                        [cx + sx * hx, cy + sy * hy]
                        for (cx, cy), (hx, hy) in spec.boxes
                        for sx in (-1.0, 1.0)
                        for sy in (-1.0, 1.0)
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                support_distance[selected] = torch.amax(
                    torch.sum(
                        outward_local[selected, None, :] * vertices[None, :, :],
                        dim=-1,
                    ),
                    dim=-1,
                )
            pusher_radius = float(getattr(self.agent, "pusher_radius", 0.018))
            gap = 0.003 + torch.rand((b,), device=device) * 0.004
            pusher_xy = (
                tee_xy
                + outward_world * (support_distance + pusher_radius + gap)[:, None]
            )
        pusher_xy = torch.clamp(
            pusher_xy, -self._PUSHER_JOINT_LIMIT, self._PUSHER_JOINT_LIMIT
        )
        self.agent.reset(pusher_xy)
