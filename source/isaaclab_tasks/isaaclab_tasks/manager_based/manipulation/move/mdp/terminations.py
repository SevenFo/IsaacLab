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


def spanner_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spanner_cfg: SceneEntityCfg = SceneEntityCfg("spanner"),
    desk_cfg: SceneEntityCfg = SceneEntityCfg("desk"),
    height_threshold: float = 0.005,
    height_diff: float = 1.0,
    gripper_open_val: torch.tensor = torch.tensor([-0.03]),
    gripper_threshold: float = 0.005,
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    spanner: RigidObject = env.scene[spanner_cfg.name]
    desk: RigidObject = env.scene[desk_cfg.name]

    pos_diff_desk_spanner = spanner.data.root_pos_w - desk.data.root_pos_w
    
    # Compute height difference
    h_dist_desk_spanner = torch.norm(pos_diff_desk_spanner[:, 2:], dim=1)

    # Check positions
    stacked = h_dist_desk_spanner - height_diff < height_threshold

    # Check gripper positions
    stacked = torch.logical_and(
        robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device) < gripper_threshold, stacked
    )
    stacked = torch.logical_and(
        robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device) < gripper_threshold, stacked
    )
    
    # if torch.any(stacked):
    #     print("Spanner stacked")

    return stacked


# def outer_stacked(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     inner_object_cfg: SceneEntityCfg = SceneEntityCfg("assemble_inner"),
#     outer_object_cfg: SceneEntityCfg = SceneEntityCfg("assemble_outer"),
#     desk_cfg: SceneEntityCfg = SceneEntityCfg("desk"),
#     xy_threshold: float = 0.011345,
#     height_threshold: float = 1.07,
#     gripper_open_val: torch.tensor = torch.tensor([0.04]),
# ):
#     robot: Articulation = env.scene[robot_cfg.name]
#     inner_object: RigidObject = env.scene[inner_object_cfg.name]
#     outer_object: RigidObject = env.scene[outer_object_cfg.name]
#     desk: RigidObject = env.scene[desk_cfg.name]

#     inner_object_pos = inner_object.data.root_pos_w
#     outer_object_pos = outer_object.data.root_pos_w
#     desk_pos = desk.data.root_pos_w
    
#     pos_x_diff = inner_object_pos[:, 0] - outer_object_pos[:, 0]
#     pos_y_diff = inner_object_pos[:, 1] - outer_object_pos[:, 1]
#     height_diff = outer_object_pos[:, 2] - desk_pos[:, 2]
    
#     stacked = torch.logical_and(abs(pos_x_diff) < xy_threshold, abs(pos_y_diff) < xy_threshold)
    
#     stacked = torch.logical_and(stacked, height_diff < height_threshold)
    
#     stacked = torch.logical_and(
#         torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
#     )
    
#     stacked = torch.logical_and(
#         torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
#     )
#     # print(f"stacked: {stacked}")

#     return stacked
