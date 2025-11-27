# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import cv2
from typing import TYPE_CHECKING, Dict, Union

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, Camera, ContactSensor
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """The position of the  box in the world frame."""
    box: Articulation = env.scene[box_cfg.name]
    
    return box.data.root_state_w[:, :3]


def box_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """The orientation of the cubes in the world frame."""
    box: Articulation = env.scene[box_cfg.name]

    return box.data.root_state_w[:, 3:7]


def spanner_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    spanner_cfg: SceneEntityCfg = SceneEntityCfg("spanner"),
) -> torch.Tensor:
    """The position of the spanner in the world frame."""
    spanner: RigidObject = env.scene[spanner_cfg.name]
    
    return spanner.data.root_pos_w


def spanner_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    spanner_cfg: SceneEntityCfg = SceneEntityCfg("spanner"),
):
    """The orientation of the spanner in the world frame."""
    spanner: RigidObject = env.scene[spanner_cfg.name]

    return spanner.data.root_quat_w


def object_obs(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    desk_cfg: SceneEntityCfg = SceneEntityCfg("desk"),
    spanner_cfg: SceneEntityCfg = SceneEntityCfg("spanner"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    button_cfg: SceneEntityCfg = SceneEntityCfg("button"),
):
    """
    Object observations (in world frame):
        box pos,
        box quat,
        spanner pos,
        spanner quat,
        gripper to box,
        box to desk,
        gripper to spanner,
        spanner to desk
    """
    
    box: Articulation = env.scene[box_cfg.name]
    spanner: RigidObject = env.scene[spanner_cfg.name]
    desk: RigidObject = env.scene[desk_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    button: RigidObject = env.scene[button_cfg.name]

    box_pos_w = box.data.root_state_w[:, :3]
    box_quat_w = box.data.root_state_w[:, 3:7]

    spanner_pos_w = spanner.data.root_pos_w
    spanner_quat_w = spanner.data.root_quat_w

    desk_pos_w = desk.data.root_pos_w
    
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    
    button_pos_w = button.data.root_pos_w
    button_quat_w = button.data.root_quat_w
    
    gripper_to_box = box_pos_w - ee_pos_w
    box_to_desk = desk_pos_w - box_pos_w
    gripper_to_spanner = spanner_pos_w - ee_pos_w
    spanner_to_desk = desk_pos_w - spanner_pos_w
    gripper_to_button = button_pos_w - ee_pos_w

    # box_joint = box.data.joint_pos[:, 0].unsqueeze(1)  # 获取箱子关节位置

    return torch.cat(
        (
            box_pos_w - env.scene.env_origins,
            box_quat_w,
            spanner_pos_w - env.scene.env_origins,
            spanner_quat_w,
            button_pos_w - env.scene.env_origins,
            button_quat_w,
            gripper_to_box,
            box_to_desk,
            gripper_to_spanner,
            spanner_to_desk,
            gripper_to_button,
            # box_joint,
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
    # print(f"x: {obj_vec_ee[:, 0]}, y: {obj_vec_ee[:, 1]}, z: {obj_vec_ee[:, 2]}")
    spatial_ok = x_ok & y_ok & z_ok

    # 爪夹闭合条件
    gripper1_ok = (robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) < gripper_threshold
    gripper2_ok = (robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) < gripper_threshold
    gripper_ok = gripper1_ok & gripper2_ok

    grasped = spatial_ok & gripper_ok
    
    # if torch.any(grasped):
    #     print("Object grasped")

    return grasped


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

    pressed = torch.logical_and(max_forces > force_threshold, contactsensor.data.current_contact_time.squeeze(-1) > time_threshold)
    # print(f"pressed:{pressed}")
    return pressed


def camera_rgbd(
    env: ManagerBasedRLEnv,
    frontcamera_cfg: SceneEntityCfg = SceneEntityCfg("frontcamera"),
    sidecamera_cfg: SceneEntityCfg = SceneEntityCfg("sidecamera"),
    wristcamera_cfg: SceneEntityCfg = SceneEntityCfg("wristcamera"),
    depth_scale: float = 10.0,  # 深度范围上限（单位：米）
) -> torch.Tensor:
    """
    返回压缩后的RGBD张量 (B, 3, H, W, 4)，数据类型为uint16
    - RGB通道：0-65535对应0-1
    - Depth通道：0-65535对应0-depth_scale米
    """
    # 获取相机对象
    cameras = {
        "front": env.scene[frontcamera_cfg.name],
        "side": env.scene[sidecamera_cfg.name],
        "wrist": env.scene[wristcamera_cfg.name]
    }
    
    def ensure_4d(t: torch.Tensor) -> torch.Tensor:
        """确保张量是 (B, H, W, C) 格式"""
        return t.unsqueeze(-1) if t.ndim == 3 else t

    def compress_to_uint16(rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        压缩数据到uint16:
        - RGB: [0,1] -> [0,65535]
        - Depth: [0,depth_scale] -> [0,65535]
        """
        rgb_uint16 = (rgb.clamp(0, 1) * 65535).round().to(torch.uint16)
        depth_uint16 = (depth.clamp(0, depth_scale) * (65535 / depth_scale)).round().to(torch.uint16)
        return torch.cat([rgb_uint16, depth_uint16], dim=-1)

    # 处理每个相机
    compressed_data = []
    for cam in cameras.values():
        # 获取原始数据 (B, H, W, C)
        rgb = ensure_4d(cam.data.output["rgb"].float())  # (B, H, W, 3)
        depth = ensure_4d(cam.data.output["depth"].float())  # (B, H, W, 1)
        
        # 压缩并保持形状 (B, H, W, 4)
        rgbd = compress_to_uint16(rgb, depth)
        compressed_data.append(rgbd.unsqueeze(1))  # (B, 1, H, W, 4)

    # 拼接所有相机 (B, 3, H, W, 4)
    return torch.cat(compressed_data, dim=1)