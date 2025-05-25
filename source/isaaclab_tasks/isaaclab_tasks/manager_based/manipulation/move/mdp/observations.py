# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def box_positions_in_world_frame(
#     env: ManagerBasedRLEnv,
#     box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
# ) -> torch.Tensor:
#     """The position of the  box in the world frame."""
#     box: RigidObject = env.scene[box_cfg.name]
    
#     return box.data.root_pos_w


def assemble_inner_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    assemble_inner_cfg: SceneEntityCfg = SceneEntityCfg("assemble_inner"),
) -> torch.Tensor:
    """The position of the assemble_inner in the world frame."""
    assemble_inner: RigidObject = env.scene[assemble_inner_cfg.name]
    
    return assemble_inner.data.root_pos_w


def assemble_outer_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    assemble_outer_cfg: SceneEntityCfg = SceneEntityCfg("assemble_outer"),
) -> torch.Tensor:
    """The position of the assemble_outer in the world frame."""
    assemble_outer: RigidObject = env.scene[assemble_outer_cfg.name]
    
    return assemble_outer.data.root_pos_w


def instance_randomize_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)

    return torch.cat((cube_1_pos_w, cube_2_pos_w, cube_3_pos_w), dim=1)


# def box_orientations_in_world_frame(
#     env: ManagerBasedRLEnv,
#     box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
# ):
#     """The orientation of the cubes in the world frame."""
#     box: RigidObject = env.scene[box_cfg.name]

#     return box.data.root_quat_w


def assemble_inner_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    assemble_inner_cfg: SceneEntityCfg = SceneEntityCfg("assemble_inner"),
):
    """The orientation of the cubes in the world frame."""
    assemble_inner: RigidObject = env.scene[assemble_inner_cfg.name]

    return assemble_inner.data.root_quat_w


def assemble_outer_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    assemble_outer_cfg: SceneEntityCfg = SceneEntityCfg("assemble_outer"),
):
    """The orientation of the cubes in the world frame."""
    assemble_outer: RigidObject = env.scene[assemble_outer_cfg.name]

    return assemble_outer.data.root_quat_w


def instance_randomize_cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    return torch.cat((cube_1_quat_w, cube_2_quat_w, cube_3_quat_w), dim=1)


# def object_obs(
#     env: ManagerBasedRLEnv,
#     box_cfg: SceneEntityCfg = SceneEntityCfg(""),
#     desk_cfg: SceneEntityCfg = SceneEntityCfg("desk"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ):
#     """
#     Object observations (in world frame):
#         box pos,
#         box quat,
#         gripper to box,
#         box to desk,
#     """
    
#     box: RigidObject = env.scene[box_cfg.name]
#     desk: RigidObject = env.scene[desk_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

#     box_pos_w = box.data.root_pos_w
#     box_quat_w = box.data.root_quat_w
    
#     desk_pos_w = desk.data.root_pos_w
#     ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    
#     gripper_to_box = box_pos_w - ee_pos_w
#     box_to_desk = box_pos_w - desk_pos_w

#     return torch.cat(
#         (
#             box_pos_w - env.scene.env_origins,
#             box_quat_w,
#             gripper_to_box,
#             box_to_desk,
#         ),
#         dim=1,
#     )


def object_obs(
    env: ManagerBasedRLEnv,
    assemble_inner_cfg: SceneEntityCfg = SceneEntityCfg("assemble_inner"),
    assenmble_outer_cfg: SceneEntityCfg = SceneEntityCfg("assemble_outer"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        assemble_inner pos,
        assemble_inner quat,
        assemble_outer pos,
        assemble_outer quat,
        gripper to assemble_outer,
        assenmble_outer to assemble_inner,
    """
    assemble_inner: RigidObject = env.scene[assemble_inner_cfg.name]
    assemble_outer: RigidObject = env.scene[assenmble_outer_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    assemble_inner_pos_w = assemble_inner.data.root_pos_w
    assemble_inner_quat_w = assemble_inner.data.root_quat_w

    assemble_outer_pos_w = assemble_inner.data.root_pos_w
    assemble_inner_quat_w = assemble_outer.data.root_quat_w
    
    gripper_to_assemble_outer = assemble_outer_pos_w - ee_frame.data.target_pos_w[:, 0, :]
    assemble_outer_to_assemble_inner = assemble_outer_pos_w - assemble_inner_pos_w

    return torch.cat(
        (
            assemble_inner_pos_w - env.scene.env_origins,
            assemble_inner_quat_w,
            assemble_outer_pos_w - env.scene.env_origins,
            assemble_inner_quat_w,
            gripper_to_assemble_outer,
            assemble_outer_to_assemble_inner,
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
        cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 3.5,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )
    
    # print("grasped:", grasped)

    return grasped


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    inner_object_cfg: SceneEntityCfg,
    outer_object_cfg: SceneEntityCfg,
    desk_cfg: SceneEntityCfg,
    xy_threshold: float = 0.011345,
    height_threshold: float = 1.07,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    inner_object: RigidObject = env.scene[inner_object_cfg.name]
    outer_object: RigidObject = env.scene[outer_object_cfg.name]
    desk: RigidObject = env.scene[desk_cfg.name]
    
    inner_object_pos = inner_object.data.root_pos_w
    outer_object_pos = outer_object.data.root_pos_w
    desk_pos = desk.data.root_pos_w
    
    pos_x_diff = inner_object_pos[:, 0] - outer_object_pos[:, 0]
    pos_y_diff = inner_object_pos[:, 1] - outer_object_pos[:, 1]
    height_diff = outer_object_pos[:, 2] - desk_pos[:, 2]

    # pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    # height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    # xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(abs(pos_x_diff) < xy_threshold, abs(pos_y_diff) < xy_threshold)
    
    stacked = torch.logical_and(stacked, height_diff < height_threshold)
    
    # stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
    )
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
    )

    return stacked
