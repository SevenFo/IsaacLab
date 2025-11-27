# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera
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


def ee_frame_pos(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_pos_gripper(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_l = ee_frame.data.target_pos_w[:, 1, :] - env.scene.env_origins[:, 0:3]
    ee_frame_pos_r = ee_frame.data.target_pos_w[:, 2, :] - env.scene.env_origins[:, 0:3]
    ee_frame_pos = (ee_frame_pos_l + ee_frame_pos_r) / 2
    return ee_frame_pos


def ee_frame_quat(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def ee_frame_quat_gripper(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """
    计算抓夹两个手指之间的四元数平均值。
    使用归一化的算术平均方法来近似两个四元数的平均值。
    这对于接近的四元数是一个合理的近似。
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat_l = ee_frame.data.target_quat_w[:, 1, :]  # 左手指四元数
    ee_frame_quat_r = ee_frame.data.target_quat_w[:, 2, :]  # 右手指四元数

    # 确保两个四元数的符号一致（选择较短的路径）
    # 如果点积为负，翻转其中一个四元数的符号
    dot_product = torch.sum(ee_frame_quat_l * ee_frame_quat_r, dim=1, keepdim=True)
    ee_frame_quat_r = torch.where(dot_product < 0, -ee_frame_quat_r, ee_frame_quat_r)

    # 算术平均并归一化
    ee_frame_quat_avg = (ee_frame_quat_l + ee_frame_quat_r) / 2.0
    ee_frame_quat = ee_frame_quat_avg / torch.norm(
        ee_frame_quat_avg, dim=1, keepdim=True
    )

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def grasp_box(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    x_threshold_1: float = 0.2,
    x_threshold_2: float = 0.24,
    y_threshold_1: float = 0.034,
    y_threshold_2: float = 0.045,
    z_threshold_1: float = -0.05,
    z_threshold_2: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([-0.48]),
    gripper_threshold: float = 0.005,
):
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    box: Articulation = env.scene[box_cfg.name]
    # shelf: RigidObject = env.scene[shelf_cfg.name]

    box_pos_w = box.data.root_state_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (B, 4)

    # 箱子在世界坐标系下相对末端的向量
    obj_vec_w = box_pos_w - ee_pos_w  # (B, 3)

    # 将obj_vec_w从世界坐标系变换到ee_frame本体坐标系下
    ee_quat_conj = torch.cat([ee_quat_w[:, :1], -ee_quat_w[:, 1:]], dim=1)  # (B, 4)

    # 将物体向量从世界坐标系转换到末端执行器坐标系
    obj_vec_ee = math_utils.quat_apply(ee_quat_conj, obj_vec_w)  # (B, 3)

    # 空间约束
    x_ok = (obj_vec_ee[:, 0] >= x_threshold_1) & (obj_vec_ee[:, 0] <= x_threshold_2)
    y_ok = (obj_vec_ee[:, 1] >= y_threshold_1) & (obj_vec_ee[:, 1] <= y_threshold_2)
    z_ok = (obj_vec_ee[:, 2] >= z_threshold_1) & (obj_vec_ee[:, 2] <= z_threshold_2)

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
    # print(f"roted:{roted}")

    return grasped


def rot_box(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    desk_cfg: SceneEntityCfg,
    height_threshold: float = 0.005,
    height_diff: float = 1.18,
    yaw_threshold: float = 0.1,
    gripper_open_val: torch.tensor = torch.tensor([-0.48]),
    gripper_threshold: float = 0.005,
):
    robot: Articulation = env.scene[robot_cfg.name]
    box: Articulation = env.scene[box_cfg.name]
    desk: RigidObject = env.scene[desk_cfg.name]

    pos_diff_desk_box = box.data.root_state_w[:, :3] - desk.data.root_pos_w
    box_quat = box.data.root_state_w[:, 3:7]
    box_euler = math_utils.euler_xyz_from_quat(box_quat)
    box_euler_tensor = torch.stack(box_euler, dim=1)

    h_dist_desk_box = torch.norm(pos_diff_desk_box[:, 2:], dim=1)
    h_ok = h_dist_desk_box - height_diff > height_threshold

    # Check if yaw is close to 0 or 2π (360°)
    # Since angles wrap around, 0° and 360° (2π) represent the same orientation
    yaw_rad = box_euler_tensor[:, 2]
    yaw_ok = (torch.abs(yaw_rad) < yaw_threshold) | (
        torch.abs(yaw_rad - 2 * torch.pi) < yaw_threshold
    )

    spatial_ok = yaw_ok & h_ok

    # 爪夹闭合条件
    gripper1_ok = (
        robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)
    ) < gripper_threshold
    gripper2_ok = (
        robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)
    ) < gripper_threshold
    gripper_ok = gripper1_ok & gripper2_ok

    grasped = spatial_ok & gripper_ok

    return grasped


def put_box(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    desk_cfg: SceneEntityCfg = SceneEntityCfg("desk"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold_1: float = 1.05,
    x_threshold_2: float = 1.36,
    y_threshold_1: float = -3.53,
    y_threshold_2: float = -3.35,
    yaw_threshold: float = 0.2,
    gripper_open_val: torch.tensor = torch.tensor([-0.03]),
    gripper_threshold: float = 0.005,
    height_threshold: float = 0.005,
    height_diff: float = 1.13,
) -> torch.Tensor:
    box: Articulation = env.scene[box_cfg.name]
    desk: RigidObject = env.scene[desk_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    pos_diff_desk_box = box.data.root_state_w[:, :3] - desk.data.root_pos_w
    box_pos_relative = box.data.root_state_w[:, :3] - env.scene.env_origins[:, :3]
    box_quat = box.data.root_state_w[:, 3:7]
    box_euler = math_utils.euler_xyz_from_quat(box_quat)
    box_euler_tensor = torch.stack(box_euler, dim=1)

    # Compute height difference
    h_dist_desk_box = torch.norm(pos_diff_desk_box[:, 2:], dim=1)
    # print(h_dist_desk_box)

    x_ok = (box_pos_relative[:, 0] >= x_threshold_1) & (
        box_pos_relative[:, 0] <= x_threshold_2
    )
    y_ok = (box_pos_relative[:, 1] >= y_threshold_1) & (
        box_pos_relative[:, 1] <= y_threshold_2
    )
    h_ok = h_dist_desk_box - height_diff < height_threshold

    # Check if yaw is close to 0 or 2π (360°)
    # Since angles wrap around, 0° and 360° (2π) represent the same orientation
    yaw_rad = box_euler_tensor[:, 2]
    yaw_ok = (torch.abs(yaw_rad) < yaw_threshold) | (
        torch.abs(yaw_rad - 2 * torch.pi) < yaw_threshold
    )

    # h_ok = torch.ones(len(box_pos_relative), dtype=torch.bool)

    # print(
    #     f"[put_box ]x_ok:{x_ok[0].item()}({box_pos_relative[:, 0].item():.2f}),y_ok:{y_ok[0].item()},h_ok:{h_ok[0].item()},yaw_ok:{yaw_ok[0].item()}:yaw={yaw_rad[0].item():.4f} (threshold={yaw_threshold})"
    # )

    put = x_ok & y_ok & h_ok & yaw_ok

    put = torch.logical_and(
        robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)
        < gripper_threshold,
        put,
    )
    put = torch.logical_and(
        robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)
        < gripper_threshold,
        put,
    )

    return put


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
        "wrist": env.scene[wristcamera_cfg.name],
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
        depth_uint16 = (
            (depth.clamp(0, depth_scale) * (65535 / depth_scale))
            .round()
            .to(torch.uint16)
        )
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


def camera_pointcloud(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    normalize: bool = True,
    visualize: bool = False,
) -> torch.Tensor:
    """ """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]  # type: ignore
    # distance_to_image_plane
    if "depth" not in sensor.data.output:
        return torch.zeros(0, 3, device=env.device)

    # obtain the input image
    depth_images = sensor.data.output["depth"]
    points_3d_cam = math_utils.unproject_depth(
        depth_images, sensor.data.intrinsic_matrices
    )

    # TODO current w.r.t. world or not env_zero (not worked in muilti-env)
    points_3d_world = math_utils.transform_points(
        points_3d_cam, sensor.data.pos_w, sensor.data.quat_w_ros
    )
    if visualize:
        if points_3d_world.size()[0] > 0:
            ...

    # rgb/depth image normalization
    if normalize:
        points_3d_world[points_3d_world == float("inf")] = 0

    # print(f"min: {points_3d_world.min()}, max: {points_3d_world.max()}")
    # print(f"points_3d_world: {points_3d_world.shape}")

    return points_3d_world.clone()
