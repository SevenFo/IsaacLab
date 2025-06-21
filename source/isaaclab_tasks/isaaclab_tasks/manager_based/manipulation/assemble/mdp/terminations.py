# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def box_stacked(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
#     desk_cfg: SceneEntityCfg = SceneEntityCfg("desk"),
#     # xy_threshold: float = 0.05,
#     xy_threshold: float = 2.04,
#     height_threshold: float = 0.91,
#     height_diff: float = 0.0468,
#     gripper_open_val: torch.tensor = torch.tensor([0.04]),
#     atol=0.0001,
#     rtol=0.0001,
# ):
#     robot: Articulation = env.scene[robot_cfg.name]
#     box: RigidObject = env.scene[box_cfg.name]
#     desk: RigidObject = env.scene[desk_cfg.name]

#     pos_diff_desk_box = desk.data.root_pos_w - box.data.root_pos_w

#     # Compute position difference in x-y plane
#     xy_dist_desk_box = torch.norm(pos_diff_desk_box[:, :2], dim=1)
    
#     # Compute height difference
#     h_dist_desk_box = torch.norm(pos_diff_desk_box[:, 2:], dim=1)
    
#     # print(f"desk_pos: {desk.data.root_pos_w}, box_pos: {box.data.root_pos_w}, xy_dist: {xy_dist_desk_box}, h_dist: {h_dist_desk_box}")

#     # Check positions
#     stacked = xy_dist_desk_box < xy_threshold
#     stacked = torch.logical_and(h_dist_desk_box - height_diff < height_threshold, stacked)

#     # Check gripper positions
#     stacked = torch.logical_and(
#         torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
#     )
#     stacked = torch.logical_and(
#         torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
#     )

#     return stacked


def outer_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    inner_object_cfg: SceneEntityCfg = SceneEntityCfg("assemble_inner"),
    outer_object_cfg: SceneEntityCfg = SceneEntityCfg("assemble_outer"),
    desk_cfg: SceneEntityCfg = SceneEntityCfg("desk"),
    xy_threshold: float = 0.011345,
    height_threshold: float = 1.07,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
):
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
    
    stacked = torch.logical_and(abs(pos_x_diff) < xy_threshold, abs(pos_y_diff) < xy_threshold)
    
    stacked = torch.logical_and(stacked, height_diff < height_threshold)
    
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
    )
    
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
    )
    # print(f"stacked: {stacked}")

    return stacked
