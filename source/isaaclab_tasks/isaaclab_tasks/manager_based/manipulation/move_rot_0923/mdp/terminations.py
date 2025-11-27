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
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

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


def press_button(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 0.1,
    time_threshold: float = 0,
) -> torch.Tensor:
    """Check if the button is pressed by the specified robot."""
    contactsensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    if contactsensor.data.force_matrix_w is not None:
        force_norms = torch.norm(contactsensor.data.force_matrix_w, dim=-1)
        max_forces = torch.max(force_norms.view(force_norms.shape[0], -1), dim=1).values
    else:
        max_forces = torch.zeros(contactsensor.num_instances, device=env.device)

    pressed = torch.logical_and(
        max_forces > force_threshold,
        contactsensor.data.current_contact_time.squeeze(-1) > time_threshold,
    )

    # print(
    #     f"pressed:{pressed},max_forces:{max_forces.item():.2f},time:{contactsensor.data.current_contact_time.squeeze(-1).item():.2f}"
    # )·
    return pressed


def box_open(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Check if the button is pressed by the specified robot."""
    heavybox: Articulation = env.scene["heavy_box"]
    joint_indices, joint_names = heavybox.find_joints("boxjoint")

    box_opened = (
        heavybox.data.joint_pos[:, joint_indices].squeeze().item() - 0.30
    ) > 0.01
    box_opened = torch.tensor([box_opened])
    # print(
    #     f"box_opened:{box_opened}, joint_diff: {(heavybox.data.joint_pos[:, joint_indices].squeeze().item() - 0.30) > 0.01:.2f}"
    # )
    return box_opened


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    x_threshold_1: float = 0.140,
    x_threshold_2: float = 0.165,
    y_threshold_1: float = -0.015,
    y_threshold_2: float = 0,
    z_threshold_1: float = -0.02,
    z_threshold_2: float = 0.118,
    gripper_open_val: torch.tensor = torch.tensor([-0.48]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (B, 4)

    # 物体在世界坐标系下相对末端的向量
    obj_vec_w = object_pos_w - ee_pos_w  # (B, 3)

    # 将obj_vec_w从世界坐标系变换到ee_frame本体坐标系下
    ee_quat_conj = torch.cat([ee_quat_w[:, :1], -ee_quat_w[:, 1:]], dim=1)  # (B, 4)

    # 将物体向量从世界坐标系转换到末端执行器坐标系
    obj_vec_ee = math_utils.quat_apply(ee_quat_conj, obj_vec_w)  # (B, 3)

    # 空间约束
    x_ok = (obj_vec_ee[:, 0] >= x_threshold_1) & (obj_vec_ee[:, 0] <= x_threshold_2)
    y_ok = (obj_vec_ee[:, 1] >= y_threshold_1) & (obj_vec_ee[:, 1] <= y_threshold_2)
    z_ok = (obj_vec_ee[:, 2] >= z_threshold_1) & (obj_vec_ee[:, 2] <= z_threshold_2)
    # print(
    #     f"[object_grasped] x: {obj_vec_ee[:, 0]}, y: {obj_vec_ee[:, 1]}, z: {obj_vec_ee[:, 2]}"
    # )
    spatial_ok = x_ok & y_ok & z_ok

    # 爪夹闭合条件
    gripper1_ok = (
        robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)
    ) < gripper_threshold
    gripper2_ok = (
        robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)
    ) < gripper_threshold
    gripper_ok = gripper1_ok & gripper2_ok

    grasped = spatial_ok & gripper_ok

    # if torch.any(grasped):
    #     print("Object grasped")

    return grasped


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
    # 当前帧grasp状态
    grasp = env.observation_manager.compute_group("subtask_terms")["grasp"]

    # 检测2: grasp
    leave_2 = grasp

    leave = leave_1 & leave_2

    # leave = torch.tensor([True],device=env.device)
    return leave


def lift_eef(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    height_threshold: float = 0.005,
    height_diff: float = 3.25,
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_z = ee_pos_w[:, 2]

    gripper_height_ok = ee_pos_z - height_diff > height_threshold
    put_ok = env.observation_manager.compute_group("subtask_terms")["put"]

    lift = gripper_height_ok & put_ok

    return lift


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
