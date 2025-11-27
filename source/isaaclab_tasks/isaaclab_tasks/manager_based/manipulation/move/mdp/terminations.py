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


def leave_button(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """当press_button从True变为False时返回True（falling edge）。"""
    # 当前帧press状态
    press = env.observation_manager.compute_group("subtask_terms")["press"]
    # 读取上一步状态
    if not hasattr(env, "_last_press_for_termination"):
        env._last_press_for_termination = torch.zeros_like(
            press, dtype=torch.bool, device=press.device
        )
    last_press = env._last_press_for_termination

    # 检测falling edge: 上一步为True，本步为False
    leave = last_press & (~press)

    # 更新缓存
    env._last_press_for_termination = press.clone()

    # leave = torch.tensor([True],device=env.device)
    return leave


def leave_spanner(
    env: ManagerBasedRLEnv,
    spanner_cfg: SceneEntityCfg = SceneEntityCfg("spanner"),
    desk_cfg: SceneEntityCfg = SceneEntityCfg("desk"),
    height_threshold: float = 0.005,
    height_diff: float = 1.25,
) -> torch.Tensor:
    spanner: RigidObject = env.scene[spanner_cfg.name]
    desk: RigidObject = env.scene[desk_cfg.name]

    pos_diff_desk_spanner = spanner.data.root_pos_w - desk.data.root_pos_w

    # Compute height difference
    h_dist_desk_spanner = torch.norm(pos_diff_desk_spanner[:, 2:], dim=1)

    leave_1 = abs(h_dist_desk_spanner) - height_diff > height_threshold

    """当grasp_spanner从True变为False时返回True（falling edge）。"""
    # 当前帧press状态
    grasp = env.observation_manager.compute_group("subtask_terms")["grasp"]

    # 检测2: grasp
    leave_2 = grasp

    leave = leave_1 & leave_2

    # leave = torch.tensor([True],device=env.device)
    return leave


# def leave_button(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """检测 press 的下降沿，并在检测到后让 leave 永远保持 True"""
#     press = env.observation_manager.compute_group("subtask_terms")['press']

#     # 初始化缓存
#     if not hasattr(env, "_last_press_for_termination"):
#         env._last_press_for_termination = torch.zeros_like(press, dtype=torch.bool)
#     if not hasattr(env, "_leave_triggered"):
#         env._leave_triggered = torch.zeros_like(press, dtype=torch.bool)

#     last_press = env._last_press_for_termination
#     leave_triggered = env._leave_triggered  # 是否已触发下降沿的标志

#     # 检测下降沿（仅在未触发时检测）
#     new_leave = (~leave_triggered) & (last_press & ~press)

#     # 更新 leave_triggered：如果检测到下降沿，则永远保持 True
#     leave_triggered = leave_triggered | new_leave

#     # 更新缓存
#     env._last_press_for_termination = press.clone()
#     env._leave_triggered = leave_triggered.clone()

#     print(f"press:{press},leave_triggered:{leave_triggered}")

#     return leave_triggered  # 只要触发过下降沿，就永远返回 True


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
        robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)
        < gripper_threshold,
        stacked,
    )
    stacked = torch.logical_and(
        robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)
        < gripper_threshold,
        stacked,
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
