#!/usr/bin/env python3
"""MuJoCo PushT with planar task physics and a visual-only SO-101 arm.

The task remains planar: a hidden vertical cylinder on XY slide joints is the
only physical pusher.  A visual-only SO-101 holds a long, slender round rod.
The gripper stays high above the workpiece while the rod tip follows the planar
pusher and remains vertical, so left-to-right and right-to-left pushes share the
same circular contact geometry.  Recording writes separate 2D and 3D videos.
"""

import os

from typing import Iterable, Optional

import mujoco
import numpy as np
from ._assets import texture_path

# -----------------------------------------------------------------------------
# Task scale: conservative high-TCP patch of the official SO-101 kinematics.
# The base is 0.34 m left of the task center.  The full rectangle remains well
# inside joint limits with the tool vertical and the gripper about 105 mm high.
# -----------------------------------------------------------------------------
TABLE_TOP_Z = 0.0
TABLE_CENTER_X = -0.10
TABLE_HALF_SIZE = np.array([0.48, 0.28, 0.02], dtype=np.float64)

ARM_BASE_POS = np.array([-0.34, 0.0, TABLE_TOP_Z], dtype=np.float64)
ARM_SAFE_RADIUS = np.array([0.240, 0.430], dtype=np.float64)

PUSHER_X_RANGE = np.array([-0.065, 0.070], dtype=np.float64)
PUSHER_Y_RANGE = np.array([-0.070, 0.070], dtype=np.float64)
OBJECT_X_RANGE = np.array([-0.055, 0.060], dtype=np.float64)
OBJECT_Y_RANGE = np.array([-0.060, 0.060], dtype=np.float64)

# The real 2-D contact proxy is a small vertical cylinder.  The red marker is
# slightly wider so it remains easy to see from the top camera.  The hidden
# physical cylinder exactly matches the visual rod radius.
PUSHER_RADIUS = 0.0040
PUSHER_MARKER_RADIUS = 0.0055
PUSHER_COLLISION_HALF_HEIGHT = 0.0045
PUSHER_BODY_Z = PUSHER_COLLISION_HALF_HEIGHT
PUSHER_MARKER_Z = 0.0045
PUSHER_MAX_SPEED = 0.10
HOME_PUSHER_XY = np.array([-0.050, -0.055], dtype=np.float64)

# The gripper clamps the rod near z=105 mm; only the rod reaches the table.
# At the task plane the physical proxy and visual rod are concentric circles.
TOOL_TCP_Z = 0.1050
TOOL_ROD_BOTTOM_Z = 0.0010
TOOL_ROD_TOP_Z = 0.1200
TOOL_ROD_RADIUS = 0.0040
TOOL_ROD_LENGTH = TOOL_ROD_TOP_Z - TOOL_ROD_BOTTOM_Z
TOOL_GRIP_SLEEVE_RADIUS = 0.0060
TOOL_GRIP_SLEEVE_HALF_LENGTH = 0.0090
TOOL_GRIP_SLEEVE_CENTER_OFFSET = 0.0050

FRICTION = 0.85
RENDER_SIZE = 512
ORTHO_VIEW_HEIGHT = 0.16
PERSPECTIVE_FOVY = 45.0
RECORD_FPS = 15
PHYSICS_TIMESTEP = 0.005
FRAME_SKIP = 4

# Small tabletop T: 28 mm crossbar width, 38 mm overall length, 8 mm high.
T_HALF_W = 0.014
T_HALF_H = 0.004
T_STEM_HALF_LENGTH = 0.015
T_HALF_HEIGHT = 0.004
BLOCK_MASS = 0.025
INTERSECTION_THRESH = 0.90
GOAL_OFFSET = np.array([0.0, 0.0], dtype=np.float64)
GOAL_Z_ROT = 0.0
SPAWN_CENTER = np.array([-0.025, 0.0], dtype=np.float64)
SPAWN_HALF_SIZE = np.array([0.006, 0.012], dtype=np.float64)


# -----------------------------------------------------------------------------
# Official SO-101 new-calibration kinematics.
# Source frame offsets and limits match TheRobotStudio's SO-101 MJCF.
# -----------------------------------------------------------------------------
ARM_JOINT_NAMES = tuple(f"r_j{i}" for i in range(1, 7))
ARM_IK_JOINT_NAMES = ARM_JOINT_NAMES[:5]
ARM_JOINT_RANGES = np.array(
    [
        [-1.9198621771937616, 1.9198621771937634],
        [-1.7453292519943224, 1.7453292519943366],
        [-1.69, 1.69],
        [-1.6580628494556928, 1.6580627293335335],
        [-2.7438472969992493, 2.841206309382605],
        [-0.17453297762778586, 1.7453291995659765],
    ],
    dtype=np.float64,
)

# Tool-up seed for the high-gripper long-rod configuration (TCP z=105 mm).
ARM_INIT_SEED = np.array([0.0, 0.30, 0.80, -1.10, 0.04868], dtype=np.float64)
ARM_GRIPPER_Q = 0.25
WORLD_TOOL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

IK_INIT_POS_TOL = 3e-5
IK_INIT_TILT_TOL_DEG = 0.20
IK_TRACK_POS_TOL = 2e-5
IK_TRACK_TILT_TOL_DEG = 0.05
IK_ORIENTATION_LENGTH = 0.10
IK_MAX_ITERS = 30
IK_MAX_JOINT_STEP = 0.12
IK_JOINT_MARGIN = 0.012
IK_DAMPING_MIN = 1e-3
IK_DAMPING_EXTRA = 1.2e-2
IK_SINGULAR_VALUE = 2.5e-2
IK_POSTURE_GAIN = 1e-8

SO101_REQUIRED_MESHES = (
    "base_motor_holder_so101_v1.stl",
    "base_so101_v2.stl",
    "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl",
    "moving_jaw_so101_v1.stl",
    "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl",
    "sts3215_03a_v1.stl",
    "under_arm_so101_v1.stl",
    "upper_arm_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl",
    "wrist_roll_follower_so101_v1.stl",
    "wrist_roll_pitch_so101_v2.stl",
)


def _xml_path(path: str) -> str:
    return (
        os.path.abspath(path)
        .replace(os.sep, "/")
        .replace("&", "&amp;")
        .replace('"', "&quot;")
    )


def _contains_so101_meshes(path: str) -> bool:
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, name)) for name in SO101_REQUIRED_MESHES
    )


def _resolve_so101_mesh_dir() -> Optional[str]:
    """Use self-contained procedural visuals or an explicit optional mesh directory."""
    from ._assets import resolve_so101_mesh_dir
    path = resolve_so101_mesh_dir(SO101_REQUIRED_MESHES)
    return _xml_path(path) if path is not None else None


def _arm_geom(name: str, attrs: str, material: str = "r_shell", group: int = 2) -> str:
    return (
        f'<geom name="{name}" {attrs} material="{material}" group="{group}" '
        'contype="0" conaffinity="0" mass="0.001"/>'
    )


def _so101_asset_xml(mesh_mode: bool) -> str:
    materials = """
    <material name="r_shell" rgba="0.92 0.93 0.95 1"/>
    <material name="r_motor" rgba="0.07 0.08 0.10 1"/>
    <material name="r_joint" rgba="0.30 0.33 0.38 1"/>
    <material name="r_tool" rgba="0.30 0.32 0.36 1"/>
    <material name="table_mat" rgba="0.90 0.90 0.88 1"/>
    <material name="floor_mat" rgba="0.18 0.20 0.24 1"/>
    """
    if not mesh_mode:
        return materials
    return materials + """
    <mesh name="r_base_motor_holder" file="base_motor_holder_so101_v1.stl"/>
    <mesh name="r_base" file="base_so101_v2.stl"/>
    <mesh name="r_motor_holder_base" file="motor_holder_so101_base_v1.stl"/>
    <mesh name="r_motor_holder_wrist" file="motor_holder_so101_wrist_v1.stl"/>
    <mesh name="r_moving_jaw" file="moving_jaw_so101_v1.stl"/>
    <mesh name="r_rotation_pitch" file="rotation_pitch_so101_v1.stl"/>
    <mesh name="r_servo_no_horn" file="sts3215_03a_no_horn_v1.stl"/>
    <mesh name="r_servo" file="sts3215_03a_v1.stl"/>
    <mesh name="r_lower_arm" file="under_arm_so101_v1.stl"/>
    <mesh name="r_upper_arm" file="upper_arm_so101_v1.stl"/>
    <mesh name="r_mounting_plate" file="waveshare_mounting_plate_so101_v2.stl"/>
    <mesh name="r_wrist_roll" file="wrist_roll_follower_so101_v1.stl"/>
    <mesh name="r_wrist_pitch" file="wrist_roll_pitch_so101_v2.stl"/>
    """


def _so101_visual_chunks(mesh_mode: bool):
    if mesh_mode:
        return {
            "base": "\n".join(
                [
                    _arm_geom("r_base_holder_v", 'type="mesh" mesh="r_base_motor_holder" pos="-0.00636471 -9.94414e-05 -0.0024" quat="0.5 0.5 0.5 0.5"'),
                    _arm_geom("r_base_v", 'type="mesh" mesh="r_base" pos="-0.00636471 -8.97657e-09 -0.0024" quat="0.5 0.5 0.5 0.5"'),
                    _arm_geom("r_base_servo_v", 'type="mesh" mesh="r_servo" pos="0.0263353 -8.97657e-09 0.0437" quat="1 0 0 0"', "r_motor"),
                    _arm_geom("r_mount_v", 'type="mesh" mesh="r_mounting_plate" pos="-0.0309827 -0.000199441 0.0474" quat="0.5 0.5 0.5 0.5"'),
                ]
            ),
            "shoulder": "\n".join(
                [
                    _arm_geom("r_shoulder_servo_v", 'type="mesh" mesh="r_servo" pos="-0.0303992 0.000422241 -0.0417" quat="0.5 0.5 0.5 -0.5"', "r_motor"),
                    _arm_geom("r_shoulder_holder_v", 'type="mesh" mesh="r_motor_holder_base" pos="-0.0675992 -0.000177759 0.0158499" quat="0.5 0.5 -0.5 0.5"'),
                    _arm_geom("r_rotation_pitch_v", 'type="mesh" mesh="r_rotation_pitch" pos="0.0122008 2.22413e-05 0.0464" quat="0.707107 -0.707107 0 0"'),
                ]
            ),
            "upper": "\n".join(
                [
                    _arm_geom("r_upper_servo_v", 'type="mesh" mesh="r_servo" pos="-0.11257 -0.0155 0.0187" quat="0 -0.707107 0.707107 0"', "r_motor"),
                    _arm_geom("r_upper_v", 'type="mesh" mesh="r_upper_arm" pos="-0.065085 0.012 0.0182" quat="0 1 0 0"'),
                ]
            ),
            "lower": "\n".join(
                [
                    _arm_geom("r_lower_v", 'type="mesh" mesh="r_lower_arm" pos="-0.0648499 -0.032 0.0182" quat="0 1 0 0"'),
                    _arm_geom("r_wrist_holder_v", 'type="mesh" mesh="r_motor_holder_wrist" pos="-0.0648499 -0.032 0.018" quat="0 -1 0 0"'),
                    _arm_geom("r_lower_servo_v", 'type="mesh" mesh="r_servo" pos="-0.1224 0.0052 0.0187" quat="0 0 1 0"', "r_motor"),
                ]
            ),
            "wrist": "\n".join(
                [
                    _arm_geom("r_wrist_servo_v", 'type="mesh" mesh="r_servo_no_horn" pos="0 -0.0424 0.0306" quat="0.5 0.5 0.5 -0.5"', "r_motor"),
                    _arm_geom("r_wrist_pitch_v", 'type="mesh" mesh="r_wrist_pitch" pos="0 -0.028 0.0181" quat="0.5 -0.5 -0.5 -0.5"'),
                ]
            ),
            "gripper": "\n".join(
                [
                    _arm_geom("r_gripper_servo_v", 'type="mesh" mesh="r_servo" pos="0.0077 0.0001 -0.0234" quat="0.707107 -0.707107 0 0"', "r_motor"),
                    _arm_geom("r_fixed_jaw_v", 'type="mesh" mesh="r_wrist_roll" pos="0 -0.000218214 0.000949706" quat="0 1 0 0"'),
                ]
            ),
            "jaw": _arm_geom("r_moving_jaw_v", 'type="mesh" mesh="r_moving_jaw" pos="0 0 0.0189" quat="1 0 0 0"'),
        }

    return {
        "base": "\n".join(
            [
                _arm_geom("r_base_v", 'type="cylinder" size="0.044 0.023" pos="0 0 0.023"'),
                _arm_geom("r_base_servo_v", 'type="box" size="0.020 0.016 0.020" pos="0.026 0 0.050"', "r_motor"),
                _arm_geom("r_mount_v", 'type="box" size="0.047 0.037 0.0025" pos="-0.020 0 0.0025"', "r_joint"),
            ]
        ),
        "shoulder": "\n".join(
            [
                _arm_geom("r_shoulder_joint_v", 'type="cylinder" size="0.019 0.016" quat="0.707107 0.707107 0 0"', "r_joint"),
                _arm_geom("r_shoulder_link_v", 'type="capsule" fromto="0 0 0 -0.0303992 -0.0182778 -0.0542" size="0.0115"'),
                _arm_geom("r_shoulder_servo_v", 'type="box" size="0.019 0.015 0.018" pos="-0.030 0 -0.041"', "r_motor"),
            ]
        ),
        "upper": "\n".join(
            [
                _arm_geom("r_upper_joint_v", 'type="cylinder" size="0.019 0.016" quat="0.707107 0.707107 0 0"', "r_joint"),
                _arm_geom("r_upper_link_v", 'type="capsule" fromto="0 0 0 -0.11257 -0.028 0" size="0.0120"'),
                _arm_geom("r_upper_servo_v", 'type="box" size="0.019 0.015 0.018" pos="-0.106 -0.026 0"', "r_motor"),
            ]
        ),
        "lower": "\n".join(
            [
                _arm_geom("r_elbow_joint_v", 'type="cylinder" size="0.0185 0.0155" quat="0.707107 0.707107 0 0"', "r_joint"),
                _arm_geom("r_lower_link_v", 'type="capsule" fromto="0 0 0 -0.1349 0.0052 0" size="0.0115"'),
                _arm_geom("r_lower_servo_v", 'type="box" size="0.019 0.015 0.018" pos="-0.126 0.005 0"', "r_motor"),
            ]
        ),
        "wrist": "\n".join(
            [
                _arm_geom("r_wrist_joint_v", 'type="cylinder" size="0.0175 0.0150" quat="0.707107 0.707107 0 0"', "r_joint"),
                _arm_geom("r_wrist_link_v", 'type="capsule" fromto="0 0 0 0 -0.0611 0.0181" size="0.0105"'),
                _arm_geom("r_wrist_servo_v", 'type="box" size="0.018 0.015 0.019" pos="0 -0.043 0.025"', "r_motor"),
            ]
        ),
        "gripper": "\n".join(
            [
                _arm_geom("r_gripper_body_v", 'type="box" size="0.018 0.016 0.024" pos="0 0 -0.025"'),
                _arm_geom("r_fixed_finger_v", 'type="capsule" fromto="-0.010 -0.018 -0.040 -0.008 -0.018 -0.098" size="0.0055"'),
                _arm_geom("r_gripper_servo_v", 'type="box" size="0.018 0.015 0.019" pos="0.008 0 -0.023"', "r_motor"),
            ]
        ),
        "jaw": _arm_geom("r_moving_jaw_v", 'type="capsule" fromto="0 0 0.0189 0 -0.052 0.0189" size="0.0050"'),
    }


def _so101_body_xml(mesh_mode: bool) -> str:
    v = _so101_visual_chunks(mesh_mode)
    rod_half = TOOL_ROD_LENGTH / 2.0
    # The site's local +Z is the tool axis.  At the desired pose it is world +Z.
    # Offset the cylinder so it spans TOOL_ROD_BOTTOM_Z..TOOL_ROD_TOP_Z.
    rod_center_offset = (
        0.5 * (TOOL_ROD_BOTTOM_Z + TOOL_ROD_TOP_Z) - TOOL_TCP_Z
    )
    rod_center_x = -0.0079 + rod_center_offset
    sleeve_center_x = -0.0079 + TOOL_GRIP_SLEEVE_CENTER_OFFSET
    return f"""
    <!-- Exact SO-101 new-calibration kinematic frames; visual-only geoms. -->
    <body name="r_base" pos="{ARM_BASE_POS[0]} {ARM_BASE_POS[1]} {ARM_BASE_POS[2]}">
      {v['base']}
      <body name="r_shoulder" pos="0.0388353 -8.97657e-09 0.0624"
            quat="3.56167e-16 1.22818e-15 -1 -4.14635e-16" gravcomp="1">
        <joint name="r_j1" type="hinge" axis="0 0 1"
               range="{ARM_JOINT_RANGES[0,0]} {ARM_JOINT_RANGES[0,1]}"
               damping="0.6" frictionloss="0.052" armature="0.028"/>
        {v['shoulder']}
        <body name="r_upper_arm" pos="-0.0303992 -0.0182778 -0.0542"
              quat="0.5 -0.5 -0.5 -0.5" gravcomp="1">
          <joint name="r_j2" type="hinge" axis="0 0 1"
                 range="{ARM_JOINT_RANGES[1,0]} {ARM_JOINT_RANGES[1,1]}"
                 damping="0.6" frictionloss="0.052" armature="0.028"/>
          {v['upper']}
          <body name="r_lower_arm" pos="-0.11257 -0.028 1.73763e-16"
                quat="0.707107 -5.98613e-17 -2.58051e-17 0.707107" gravcomp="1">
            <joint name="r_j3" type="hinge" axis="0 0 1"
                   range="{ARM_JOINT_RANGES[2,0]} {ARM_JOINT_RANGES[2,1]}"
                   damping="0.6" frictionloss="0.052" armature="0.028"/>
            {v['lower']}
            <body name="r_wrist" pos="-0.1349 0.0052 3.62355e-17"
                  quat="0.707107 9.58722e-16 -7.51313e-16 -0.707107" gravcomp="1">
              <joint name="r_j4" type="hinge" axis="0 0 1"
                     range="{ARM_JOINT_RANGES[3,0]} {ARM_JOINT_RANGES[3,1]}"
                     damping="0.6" frictionloss="0.052" armature="0.028"/>
              {v['wrist']}
              <body name="r_gripper" pos="5.55112e-17 -0.0611 0.0181"
                    quat="0.0172091 -0.0172091 0.706897 0.706897" gravcomp="1">
                <joint name="r_j5" type="hinge" axis="0 0 1"
                       range="{ARM_JOINT_RANGES[4,0]} {ARM_JOINT_RANGES[4,1]}"
                       damping="0.6" frictionloss="0.052" armature="0.028"/>
                {v['gripper']}
                <site name="arm_tcp" pos="-0.0079 -0.000218121 -0.0981274"
                      quat="0.707107 0 0.707107 0" size="0.002" rgba="0 0 0 0"/>
                {_arm_geom('r_tool_rod', f'type="cylinder" size="{TOOL_ROD_RADIUS} {rod_half}" pos="{rod_center_x} -0.000218121 -0.0981274" quat="0.707107 0 0.707107 0"', 'r_tool', 3)}
                {_arm_geom('r_tool_grip_sleeve', f'type="cylinder" size="{TOOL_GRIP_SLEEVE_RADIUS} {TOOL_GRIP_SLEEVE_HALF_LENGTH}" pos="{sleeve_center_x} -0.000218121 -0.0981274" quat="0.707107 0 0.707107 0"', 'r_joint', 2)}
                <body name="r_moving_jaw" pos="0.0202 0.0188 -0.0234"
                      quat="0.707107 0.707107 -1.85362e-08 1.85362e-08" gravcomp="1">
                  <joint name="r_j6" type="hinge" axis="0 0 1"
                         range="{ARM_JOINT_RANGES[5,0]} {ARM_JOINT_RANGES[5,1]}"
                         damping="0.6" frictionloss="0.052" armature="0.028"/>
                  {v['jaw']}
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    """


def _workspace_guide_xml() -> str:
    xmin, xmax = PUSHER_X_RANGE
    ymin, ymax = PUSHER_Y_RANGE
    thickness = 0.0012
    z = TABLE_TOP_Z + 0.001
    rgba = "0.20 0.42 0.65 0.28"
    return f"""
    <geom name="workspace_bottom" type="box" pos="{(xmin+xmax)/2} {ymin} {z}"
          size="{(xmax-xmin)/2} {thickness} {thickness}" rgba="{rgba}"
          group="1" contype="0" conaffinity="0"/>
    <geom name="workspace_top" type="box" pos="{(xmin+xmax)/2} {ymax} {z}"
          size="{(xmax-xmin)/2} {thickness} {thickness}" rgba="{rgba}"
          group="1" contype="0" conaffinity="0"/>
    <geom name="workspace_left" type="box" pos="{xmin} {(ymin+ymax)/2} {z}"
          size="{thickness} {(ymax-ymin)/2} {thickness}" rgba="{rgba}"
          group="1" contype="0" conaffinity="0"/>
    <geom name="workspace_right" type="box" pos="{xmax} {(ymin+ymax)/2} {z}"
          size="{thickness} {(ymax-ymin)/2} {thickness}" rgba="{rgba}"
          group="1" contype="0" conaffinity="0"/>
    """


def _compiler_xml(mesh_dir: Optional[str]) -> str:
    mesh_attr = f' meshdir="{mesh_dir}"' if mesh_dir else ""
    return f'<compiler angle="radian" autolimits="true" inertiafromgeom="true"{mesh_attr}/>'


def _build_pusht_xml():
    mesh_dir = _resolve_so101_mesh_dir()
    mesh_mode = mesh_dir is not None
    top_center_y = T_STEM_HALF_LENGTH + T_HALF_H
    model_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="pusht_so101_visual">
  {_compiler_xml(mesh_dir)}
  <option timestep="{PHYSICS_TIMESTEP}" gravity="0 0 -9.81" integrator="implicitfast">
    <flag contact="enable" energy="disable"/>
  </option>
  <size njmax="1200" nconmax="250"/>

  <default>
    <geom friction="{FRICTION} 0.01 0.001" solimp="0.90 0.95 0.001" solref="0.012 1"/>
    <velocity ctrllimited="true" ctrlrange="{-PUSHER_MAX_SPEED} {PUSHER_MAX_SPEED}" kv="4.0"/>
    <default class="goal_geom">
      <geom contype="0" conaffinity="0" group="1" rgba="0.20 0.60 0.95 0.65"/>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.35 0.35 0.35" specular="0.15 0.15 0.15"/>
    <rgba haze="0.12 0.15 0.20 1"/>
    <global offwidth="{RENDER_SIZE}" offheight="{RENDER_SIZE}" fovy="45"/>
  </visual>

  <asset>
    {_so101_asset_xml(mesh_mode)}
    <texture name="hatch_tex" type="2d" file="{texture_path('hatch_blue.png')}"/>
    <material name="goal_hatch" texture="hatch_tex" rgba="0.20 0.60 0.95 0.65" texrepeat="6 6"/>
  </asset>

  <worldbody>
    <light name="key" pos="-0.15 -0.25 0.75" dir="0.1 0.2 -1" diffuse="0.85 0.85 0.82"/>
    <light name="fill" pos="0.25 0.30 0.45" dir="-0.3 -0.3 -1" diffuse="0.40 0.43 0.48"/>
    <geom name="floor" type="plane" pos="0 0 -0.045" size="2 2 0.1" material="floor_mat"/>
    <geom name="table" type="box" pos="{TABLE_CENTER_X} 0 {-TABLE_HALF_SIZE[2]}"
          size="{TABLE_HALF_SIZE[0]} {TABLE_HALF_SIZE[1]} {TABLE_HALF_SIZE[2]}"
          material="table_mat" friction="1.0 0.01 0.001"/>
    {_workspace_guide_xml()}

    <body name="goal" mocap="true" pos="{GOAL_OFFSET[0]} {GOAL_OFFSET[1]} 0.0008"
          euler="0 0 {GOAL_Z_ROT}">
      <geom name="goal_stem" material="goal_hatch" contype="0" conaffinity="0" group="1" type="box"
            size="{T_HALF_H} {T_STEM_HALF_LENGTH} 0.0003"/>
      <geom name="goal_top" material="goal_hatch" contype="0" conaffinity="0" group="1" type="box"
            size="{T_HALF_W} {T_HALF_H} 0.0003"
            pos="0 {top_center_y} 0"/>
    </body>

    <body name="block" pos="0 0 {T_HALF_HEIGHT + 0.0005}">
      <joint name="block_x" type="slide" axis="1 0 0" limited="true"
             range="{OBJECT_X_RANGE[0]} {OBJECT_X_RANGE[1]}" damping="1.0"/>
      <joint name="block_y" type="slide" axis="0 1 0" limited="true"
             range="{OBJECT_Y_RANGE[0]} {OBJECT_Y_RANGE[1]}" damping="1.0"/>
      <joint name="block_yaw" type="hinge" axis="0 0 1" damping="0.08"/>
      <geom name="block_stem" type="box"
            size="{T_HALF_H} {T_STEM_HALF_LENGTH} 0.0003"
            rgba="0.36 0.43 0.55 1" mass="{BLOCK_MASS * 0.55}"/>
      <geom name="block_top" type="box"
            size="{T_HALF_W} {T_HALF_H} 0.0003"
            pos="0 {top_center_y} 0" rgba="0.36 0.43 0.55 1"
            mass="{BLOCK_MASS * 0.45}"/>
    </body>

    <body name="pusher" pos="0 0 {PUSHER_BODY_Z}">
      <joint name="root_x" type="slide" axis="1 0 0" limited="true"
             range="{PUSHER_X_RANGE[0]} {PUSHER_X_RANGE[1]}" damping="0.06"/>
      <joint name="root_y" type="slide" axis="0 1 0" limited="true"
             range="{PUSHER_Y_RANGE[0]} {PUSHER_Y_RANGE[1]}" damping="0.06"/>
      <geom name="pusher_collision" type="cylinder"
            size="{PUSHER_RADIUS} {PUSHER_COLLISION_HALF_HEIGHT}"
            rgba="1 0 0 0" mass="0.030"/>
      <geom name="pusher_marker" type="sphere" pos="0 0 {PUSHER_MARKER_Z}"
            size="{PUSHER_MARKER_RADIUS}" rgba="0.95 0.05 0.04 1"
            contype="0" conaffinity="0" group="4" mass="0.0001"/>
      <site name="pusher" pos="0 0 0" size="0.001" rgba="0 0 0 0"/>
    </body>

    {_so101_body_xml(mesh_mode)}
  </worldbody>

  <actuator>
    <velocity name="x_motor" joint="root_x"/>
    <velocity name="y_motor" joint="root_y"/>
  </actuator>

  <sensor>
    <framepos name="position" objtype="body" objname="block" reftype="body" refname="goal"/>
    <framequat name="orientation" objtype="body" objname="block" reftype="body" refname="goal"/>
  </sensor>
</mujoco>
"""
    return model_xml, mesh_mode



class _SO101VisualMixin:
    """Shared visual-only arm IK, workspace guard, rendering, and recording."""

    def _init_visual_overlay(self, render_mode, max_episode_steps, mesh_mode):
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.arm_visual_backend = "official_so101_meshes" if mesh_mode else "procedural_so101"

        self._viewer = None
        self._renderer = None
        self._elapsed = 0
        self._record_frames = None
        self._record_paths = None

        self._camera_2d = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self._camera_2d)
        self._camera_2d.lookat[:] = np.array([0.005, 0.0, 0.0])
        self._camera_2d.distance = 0.19
        self._camera_2d.azimuth = 90.0
        self._camera_2d.elevation = -90.0

        self._camera_3d = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self._camera_3d)
        self._camera_3d.lookat[:] = np.array([-0.135, 0.0, 0.105])
        self._camera_3d.distance = 0.58
        self._camera_3d.azimuth = 134.0
        self._camera_3d.elevation = -25.0


        self._option_2d = mujoco.MjvOption()
        self._option_2d.geomgroup[:] = 1
        self._option_2d.geomgroup[2] = 0  # hide the arm body, keep group-3 tool rod

        self._option_3d = mujoco.MjvOption()
        self._option_3d.geomgroup[:] = 1

        self._setup_arm_ik(HOME_PUSHER_XY)

    def _joint_addresses(self, names: Iterable[str]):
        ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names],
            dtype=np.int32,
        )
        if np.any(ids < 0):
            raise RuntimeError(f"Missing joint(s): {tuple(names)}")
        return (
            ids,
            self.model.jnt_qposadr[ids].astype(np.int32).copy(),
            self.model.jnt_dofadr[ids].astype(np.int32).copy(),
        )

    @staticmethod
    def _skew(v):
        x, y, z = v
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    def _setup_arm_ik(self, initial_pusher_xy):
        self._arm_tcp_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "arm_tcp"
        )
        (
            self._arm_joint_ids,
            self._arm_qpos_adr,
            self._arm_dof_adr,
        ) = self._joint_addresses(ARM_JOINT_NAMES)
        _, self._pusher_qpos_adr, self._pusher_dof_adr = self._joint_addresses(
            ("root_x", "root_y")
        )

        model_ranges = self.model.jnt_range[self._arm_joint_ids]
        if not np.allclose(model_ranges, ARM_JOINT_RANGES, atol=1e-8):
            raise RuntimeError(f"Unexpected SO-101 joint ranges: {model_ranges!r}")

        self._ik_qpos_adr = self._arm_qpos_adr[:5]
        self._ik_dof_adr = self._arm_dof_adr[:5]
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        self.data.qpos[self._pusher_qpos_adr] = np.asarray(initial_pusher_xy)
        self._set_arm_qpos(ARM_INIT_SEED)
        mujoco.mj_forward(self.model, self.data)
        target = np.array(
            [
                self.data.site_xpos[self._pusher_site, 0],
                self.data.site_xpos[self._pusher_site, 1],
                TOOL_TCP_Z,
            ],
            dtype=np.float64,
        )

        from scipy.optimize import least_squares

        def residual(q):
            self._set_arm_qpos(q)
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self._arm_tcp_site]
            axis = self._tool_axis()
            return np.concatenate(
                [
                    pos - target,
                    IK_ORIENTATION_LENGTH * (axis - WORLD_TOOL_AXIS),
                    0.0015 * (q - ARM_INIT_SEED),
                ]
            )

        result = least_squares(
            residual,
            ARM_INIT_SEED,
            bounds=(ARM_JOINT_RANGES[:5, 0], ARM_JOINT_RANGES[:5, 1]),
            method="trf",
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=800,
        )
        q = np.clip(result.x, ARM_JOINT_RANGES[:5, 0], ARM_JOINT_RANGES[:5, 1])
        self._set_arm_qpos(q)
        mujoco.mj_forward(self.model, self.data)
        pos_err, tilt_deg = self._current_arm_errors(target)
        if (
            not np.isfinite(pos_err)
            or pos_err > IK_INIT_POS_TOL
            or tilt_deg > IK_INIT_TILT_TOL_DEG
        ):
            raise RuntimeError(
                "SO-101 initialization IK failed: "
                f"position_error={pos_err:.6g} m, tilt={tilt_deg:.4f} deg, "
                f"message={result.message}"
            )

        self._arm_init_q = q.copy()
        self._arm_q_warm = q.copy()
        self._arm_target_clipped = False
        self._arm_position_error = pos_err
        self._arm_tilt_deg = tilt_deg

    def _set_arm_qpos(self, q5):
        q5 = np.asarray(q5, dtype=np.float64)
        self.data.qpos[self._ik_qpos_adr] = q5
        self.data.qpos[self._arm_qpos_adr[5]] = ARM_GRIPPER_Q
        self.data.qvel[self._arm_dof_adr] = 0.0
        self.data.qacc_warmstart[self._arm_dof_adr] = 0.0

    def _tool_axis(self):
        return self.data.site_xmat[self._arm_tcp_site].reshape(3, 3)[:, 2].copy()

    def _current_arm_errors(self, target_xyz):
        pos_err = float(
            np.linalg.norm(target_xyz - self.data.site_xpos[self._arm_tcp_site])
        )
        dot = float(np.clip(np.dot(self._tool_axis(), WORLD_TOOL_AXIS), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(dot)))
        return pos_err, tilt_deg

    def _project_target_to_safe_annulus(self, target_xy):
        target_xy = np.asarray(target_xy, dtype=np.float64).copy()
        delta = target_xy - ARM_BASE_POS[:2]
        radius = float(np.linalg.norm(delta))
        clipped = False
        if radius < 1e-12:
            delta = np.array([1.0, 0.0])
            radius = 1.0
        safe_r = float(np.clip(radius, ARM_SAFE_RADIUS[0], ARM_SAFE_RADIUS[1]))
        if abs(safe_r - radius) > 1e-12:
            target_xy = ARM_BASE_POS[:2] + delta * (safe_r / radius)
            clipped = True
        return target_xy, clipped

    def _evaluate_ik_state(self, q, target_xyz):
        self._set_arm_qpos(q)
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self._arm_tcp_site].copy()
        axis = self._tool_axis()
        e_pos = target_xyz - pos
        e_axis = WORLD_TOOL_AXIS - axis
        score = float(
            np.dot(e_pos, e_pos)
            + (IK_ORIENTATION_LENGTH ** 2) * np.dot(e_axis, e_axis)
        )
        dot = float(np.clip(np.dot(axis, WORLD_TOOL_AXIS), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(dot)))
        return score, e_pos, e_axis, axis, tilt_deg

    def _solve_ik_dls(self, target_xyz, q_warm):
        lower = ARM_JOINT_RANGES[:5, 0] + IK_JOINT_MARGIN
        upper = ARM_JOINT_RANGES[:5, 1] - IK_JOINT_MARGIN
        q = np.clip(np.asarray(q_warm, dtype=np.float64), lower, upper)

        score, e_pos, e_axis, axis, tilt_deg = self._evaluate_ik_state(q, target_xyz)
        best_q = q.copy()
        best_score = score

        for _ in range(IK_MAX_ITERS):
            if (
                np.linalg.norm(e_pos) <= IK_TRACK_POS_TOL
                and tilt_deg <= IK_TRACK_TILT_TOL_DEG
            ):
                break

            self._jacp.fill(0.0)
            self._jacr.fill(0.0)
            mujoco.mj_jacSite(
                self.model,
                self.data,
                self._jacp,
                self._jacr,
                self._arm_tcp_site,
            )
            j_pos = self._jacp[:, self._ik_dof_adr]
            j_rot = self._jacr[:, self._ik_dof_adr]
            j_axis = -self._skew(axis) @ j_rot
            j_task = np.vstack([j_pos, IK_ORIENTATION_LENGTH * j_axis])
            error = np.concatenate([e_pos, IK_ORIENTATION_LENGTH * e_axis])

            singular_values = np.linalg.svd(j_task, compute_uv=False)
            sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
            singular_ratio = max(
                0.0, (IK_SINGULAR_VALUE - sigma_min) / IK_SINGULAR_VALUE
            )
            damping = IK_DAMPING_MIN + IK_DAMPING_EXTRA * singular_ratio**2

            hessian = j_task.T @ j_task
            hessian += (damping**2 + IK_POSTURE_GAIN) * np.eye(5)
            gradient = j_task.T @ error
            gradient += IK_POSTURE_GAIN * (ARM_INIT_SEED - q)
            try:
                dq = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            max_step = float(np.max(np.abs(dq)))
            if max_step > IK_MAX_JOINT_STEP:
                dq *= IK_MAX_JOINT_STEP / max_step

            accepted = False
            alpha = 1.0
            for _ in range(7):
                candidate = np.clip(q + alpha * dq, lower, upper)
                cand_state = self._evaluate_ik_state(candidate, target_xyz)
                if cand_state[0] <= score + 1e-14:
                    q = candidate
                    score, e_pos, e_axis, axis, tilt_deg = cand_state
                    accepted = True
                    if score < best_score:
                        best_score = score
                        best_q = q.copy()
                    break
                alpha *= 0.5
            if not accepted:
                break

        self._set_arm_qpos(best_q)
        mujoco.mj_forward(self.model, self.data)
        return best_q

    def _track_pusher_with_arm(self):
        mujoco.mj_forward(self.model, self.data)
        pusher_xy = self.data.site_xpos[self._pusher_site, :2].copy()
        safe_xy, clipped = self._project_target_to_safe_annulus(pusher_xy)
        target = np.array([safe_xy[0], safe_xy[1], TOOL_TCP_Z], dtype=np.float64)
        q = self._solve_ik_dls(target, self._arm_q_warm)
        self._arm_q_warm = q.copy()
        self._set_arm_qpos(q)
        mujoco.mj_forward(self.model, self.data)
        self._arm_target_clipped = clipped
        self._arm_position_error, self._arm_tilt_deg = self._current_arm_errors(target)

    def _arm_tcp_xy_error(self):
        return float(
            np.linalg.norm(
                self.data.site_xpos[self._pusher_site, :2]
                - self.data.site_xpos[self._arm_tcp_site, :2]
            )
        )

    def _arm_diagnostics(self):
        return {
            "arm_tcp_position_error": float(self._arm_position_error),
            "arm_tcp_xy_error": self._arm_tcp_xy_error(),
            "arm_tool_tilt_deg": float(self._arm_tilt_deg),
            "arm_target_clipped": bool(self._arm_target_clipped),
            "arm_visual_backend": self.arm_visual_backend,
            "arm_tcp_z": float(self.data.site_xpos[self._arm_tcp_site, 2]),
            "tool_rod_bottom_z": float(TOOL_ROD_BOTTOM_Z),
        }

    def _normalized_action_to_velocity(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        velocity = action * PUSHER_MAX_SPEED
        q = self.data.qpos[self._pusher_qpos_adr].copy()
        dt = float(self.model.opt.timestep * FRAME_SKIP)
        low = np.array([PUSHER_X_RANGE[0], PUSHER_Y_RANGE[0]])
        high = np.array([PUSHER_X_RANGE[1], PUSHER_Y_RANGE[1]])
        velocity = np.minimum(velocity, (high - q) / dt)
        velocity = np.maximum(velocity, (low - q) / dt)
        return np.clip(velocity, -PUSHER_MAX_SPEED, PUSHER_MAX_SPEED)

    def _advance_physics(self, action):
        self._set_arm_qpos(self._arm_q_warm)
        self.data.ctrl[:] = self._normalized_action_to_velocity(action)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        low = np.array([PUSHER_X_RANGE[0], PUSHER_Y_RANGE[0]])
        high = np.array([PUSHER_X_RANGE[1], PUSHER_Y_RANGE[1]])
        self.data.qpos[self._pusher_qpos_adr] = np.clip(
            self.data.qpos[self._pusher_qpos_adr], low, high
        )
        self._track_pusher_with_arm()

    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

    @property
    def recording_paths(self):
        return None if self._record_paths is None else dict(self._record_paths)

    def start_recording(self, path):
        root, ext = os.path.splitext(path)
        if not ext:
            ext = ".mp4"
            root = path
        self._record_paths = {
            "2d": f"{root}_2d{ext}",
            "3d": f"{root}_3d{ext}",
        }
        for output in self._record_paths.values():
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        self._record_frames = {"2d": [], "3d": []}

    def _capture_recording_frame(self):
        if self._record_frames is None:
            return
        self._record_frames["2d"].append(self.render("rgb_array", camera="2d"))
        self._record_frames["3d"].append(self.render("rgb_array", camera="3d"))

    def stop_recording(self):
        if self._record_frames is not None and self._record_paths is not None:
            import imageio.v2 as imageio
            for view in ("2d", "3d"):
                frames = self._record_frames[view]
                if frames:
                    imageio.mimsave(
                        self._record_paths[view], frames, fps=RECORD_FPS, macro_block_size=None
                    )
        paths = self.recording_paths
        self._record_frames = None
        self._record_paths = None
        return paths

    def render(self, mode=None, camera="2d"):
        mode = mode or self.render_mode
        if mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mujoco_viewer
                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if mode != "rgb_array":
            return None
        if camera not in {"2d", "3d"}:
            raise ValueError("camera must be '2d' or '3d'")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=RENDER_SIZE, width=RENDER_SIZE
            )
        cam = self._camera_2d if camera == "2d" else self._camera_3d
        option = self._option_2d if camera == "2d" else self._option_3d
        # MuJoCo's free-camera projection is controlled by model.vis.global_.
        # Switch it only while building the scene so the 2D view is genuinely
        # orthographic: a vertical tool projects to a concentric circle rather
        # than a perspective streak.
        old_orthographic = int(self.model.vis.global_.orthographic)
        old_fovy = float(self.model.vis.global_.fovy)
        try:
            self.model.vis.global_.orthographic = int(camera == "2d")
            self.model.vis.global_.fovy = (
                ORTHO_VIEW_HEIGHT if camera == "2d" else PERSPECTIVE_FOVY
            )
            self._renderer.update_scene(self.data, camera=cam, scene_option=option)
        finally:
            self.model.vis.global_.orthographic = old_orthographic
            self.model.vis.global_.fovy = old_fovy

        if camera == "2d":
            # A clean task-only view: no arm shadows/reflections/haze.
            for flag in (
                mujoco.mjtRndFlag.mjRND_SHADOW,
                mujoco.mjtRndFlag.mjRND_REFLECTION,
                mujoco.mjtRndFlag.mjRND_FOG,
                mujoco.mjtRndFlag.mjRND_HAZE,
            ):
                self._renderer.scene.flags[flag] = 0
        return self._renderer.render()

    def close(self):
        self.stop_recording()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class PushTEnv(_SO101VisualMixin):
    """Planar PushT physics with a stable visual-only SO-101 overlay."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, max_episode_steps=200):
        xml, mesh_mode = _build_pusht_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._block_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block"
        )
        self._goal_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )
        self._pusher_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher"
        )
        self._pusher_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pusher"
        )
        self._init_visual_overlay(render_mode, max_episode_steps, mesh_mode)
        self._setup_pseudo_renderer()

    def _setup_pseudo_renderer(self):
        res, uv_half_width = 128, 0.070
        self._res = res
        self._uv_half_width = uv_half_width
        one_d = np.arange(res, dtype=np.float64) - res / 2
        x_grid = np.tile(one_d.reshape(1, res), (res, 1)) + 0.5
        y_grid = -np.tile(one_d.reshape(res, 1), (1, res)) + 0.5
        uv_grid = np.stack([x_grid, y_grid]) / ((res / 2) / uv_half_width)
        self._homo_uv = np.concatenate(
            [uv_grid, np.ones((1, res, res), dtype=np.float64)], axis=0
        )

        stem = np.array(
            [
                [-T_HALF_H, T_STEM_HALF_LENGTH],
                [T_HALF_H, T_STEM_HALF_LENGTH],
                [-T_HALF_H, -T_STEM_HALF_LENGTH],
                [T_HALF_H, -T_STEM_HALF_LENGTH],
            ]
        )
        top_center_y = T_STEM_HALF_LENGTH + T_HALF_H
        top = np.array(
            [
                [-T_HALF_W, top_center_y + T_HALF_H],
                [T_HALF_W, top_center_y + T_HALF_H],
                [-T_HALF_W, top_center_y - T_HALF_H],
                [T_HALF_W, top_center_y - T_HALF_H],
            ]
        )
        scale = (res / 2) / uv_half_width
        stem_px = (stem * scale + res / 2).astype(int)
        top_px = (top * scale + res / 2).astype(int)
        tee = np.zeros((res, res), dtype=np.uint8)
        tee.T[
            stem_px[0, 0] : stem_px[1, 0],
            stem_px[2, 1] : stem_px[0, 1],
        ] = 1
        tee.T[
            top_px[0, 0] : top_px[1, 0],
            top_px[2, 1] : top_px[0, 1],
        ] = 1
        self._tee_mask = np.flip(tee, axis=0).astype(bool)
        self._goal_area = int(self._tee_mask.sum())
        c, s = np.cos(GOAL_Z_ROT), np.sin(GOAL_Z_ROT)
        goal_to_world = np.eye(3)
        goal_to_world[0, 0] = c
        goal_to_world[0, 1] = -s
        goal_to_world[1, 0] = s
        goal_to_world[1, 1] = c
        goal_to_world[0, 2] = GOAL_OFFSET[0]
        goal_to_world[1, 2] = GOAL_OFFSET[1]
        self._world_to_goal = np.linalg.inv(goal_to_world)

    def _block_xy(self):
        return self.data.qpos[:2].copy()

    def _block_yaw(self):
        return float(self.data.qpos[2])

    def _compute_intersection(self):
        xy = self._block_xy()
        yaw = self._block_yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        block_to_world = np.eye(3)
        block_to_world[0, 0] = c
        block_to_world[0, 1] = -s
        block_to_world[1, 0] = s
        block_to_world[1, 1] = c
        block_to_world[0, 2] = xy[0]
        block_to_world[1, 2] = xy[1]
        block_to_goal = self._world_to_goal @ block_to_world
        transformed = (block_to_goal @ self._homo_uv.reshape(3, -1)).reshape(
            3, self._res, self._res
        )
        points = transformed[:2] / transformed[2]
        pixels = points[:, self._tee_mask]
        indices = (
            pixels * (self._res / 2) / self._uv_half_width + self._res / 2
        ).astype(int)
        indices[0] = np.clip(indices[0], 0, self._res - 1)
        indices[1] = np.clip(indices[1], 0, self._res - 1)
        rendered = np.zeros((self._res, self._res), dtype=np.uint8)
        rendered[indices[0], indices[1]] = 1
        rendered = rendered.T[::-1].astype(bool)
        return float((rendered & self._tee_mask).sum()) / self._goal_area

    def get_obs(self):
        yaw = self._block_yaw()
        return {
            "block_xy": self._block_xy(),
            "block_yaw": np.array([yaw]),
            "block_quat": np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
            "pusher_xy": self.data.qpos[3:5].copy(),
            "goal_xy": GOAL_OFFSET.copy(),
        }

    def compute_dense_reward(self, info):
        yaw = self._block_yaw()
        rotation_reward = np.cos(yaw - GOAL_Z_ROT)
        reward = (((rotation_reward + 1.0) / 2.0) ** 2) / 2.0
        distance = np.linalg.norm(self._block_xy() - GOAL_OFFSET)
        reward += ((1.0 - np.tanh(5.0 * distance)) ** 2) / 2.0
        pusher_dist = np.linalg.norm(self.data.qpos[3:5] - self._block_xy())
        reward += np.sqrt(max(0.0, 1.0 - np.tanh(5.0 * pusher_dist))) / 20.0
        if info.get("success"):
            reward = 3.0
        return float(reward)

    def evaluate(self):
        return {"success": bool(self._compute_intersection() >= INTERSECTION_THRESH)}

    def step(self, action):
        self._advance_physics(action)
        info = self.evaluate()
        info.update(self._arm_diagnostics())
        reward = self.compute_dense_reward(info)
        obs = self.get_obs()
        self._elapsed += 1
        terminated = info["success"]
        truncated = self._elapsed >= self.max_episode_steps
        self._capture_recording_frame()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the task; optional direction presets make both push directions explicit.

        ``options={"direction": "left_to_right"}`` and
        ``options={"direction": "right_to_left"}`` place the T and pusher in
        mirrored poses.  Explicit ``block_xy``, ``block_yaw`` and ``pusher_xy``
        values override the preset.
        """
        rng = np.random.default_rng(seed)
        options = {} if options is None else dict(options)
        mujoco.mj_resetData(self.model, self.data)

        direction = options.get("direction")
        if direction == "left_to_right":
            default_block_xy = np.array([-0.025, 0.0])
            default_pusher_xy = np.array([-0.052, 0.0])
            default_yaw = 0.0
        elif direction == "right_to_left":
            default_block_xy = np.array([0.025, 0.0])
            default_pusher_xy = np.array([0.052, 0.0])
            default_yaw = 0.0
        elif direction is None:
            default_block_xy = SPAWN_CENTER + rng.uniform(
                -SPAWN_HALF_SIZE, SPAWN_HALF_SIZE
            )
            default_pusher_xy = HOME_PUSHER_XY
            default_yaw = rng.uniform(-np.pi, np.pi)
        else:
            raise ValueError("direction must be left_to_right or right_to_left")

        block_xy = np.asarray(options.get("block_xy", default_block_xy), dtype=np.float64)
        pusher_xy = np.asarray(options.get("pusher_xy", default_pusher_xy), dtype=np.float64)
        block_yaw = float(options.get("block_yaw", default_yaw))
        if block_xy.shape != (2,) or pusher_xy.shape != (2,):
            raise ValueError("block_xy and pusher_xy must have shape (2,)")

        self.data.qpos[:2] = np.clip(
            block_xy,
            [OBJECT_X_RANGE[0], OBJECT_Y_RANGE[0]],
            [OBJECT_X_RANGE[1], OBJECT_Y_RANGE[1]],
        )
        self.data.qpos[2] = block_yaw
        self.data.qpos[3:5] = np.clip(
            pusher_xy,
            [PUSHER_X_RANGE[0], PUSHER_Y_RANGE[0]],
            [PUSHER_X_RANGE[1], PUSHER_Y_RANGE[1]],
        )
        self._arm_q_warm = self._arm_init_q.copy()
        self._set_arm_qpos(self._arm_q_warm)
        mujoco.mj_forward(self.model, self.data)
        self._track_pusher_with_arm()
        self._elapsed = 0
        self._capture_recording_frame()
        return self.get_obs(), self._arm_diagnostics()


if __name__ == "__main__":
    env = PushTEnv(max_episode_steps=100)
    obs, info = env.reset(seed=0)
    print(
        f"block={obs['block_xy']}, yaw={obs['block_yaw'][0]:.3f}, "
        f"arm_err={info['arm_tcp_position_error']:.6g} m, "
        f"tilt={info['arm_tool_tilt_deg']:.3f} deg, "
        f"visual={info['arm_visual_backend']}"
    )
    env.close()
