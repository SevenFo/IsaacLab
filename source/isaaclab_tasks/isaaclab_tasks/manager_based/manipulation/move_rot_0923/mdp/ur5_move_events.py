# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_view_settings(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Set the view settings for the environment."""
    env.sim.set_camera_view(
        eye=(-1.81532, 0.97065, 4.73775), target=(1.215, -3.200, 2.860)
    )


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(
        env.num_envs, 1
    )


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # joint_indices, joint_names = asset.find_joints(["shoulder_lift_joint", "elbow_joint"])
    # joint_pos[:, joint_indices] += math_utils.sample_gaussian(mean, std, (len(env_ids), len(joint_indices)), joint_pos.device)
    joint_pos += math_utils.sample_gaussian(
        mean, std, joint_pos.shape, joint_pos.device
    )

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    ## ur5
    joint_pos[:, -8:] = asset.data.default_joint_pos[env_ids, -8:]
    ## franka
    # joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def sample_object_poses(
    pose_range: dict[str, tuple[float, float]] = {},
):
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    pose_list = []

    sample = [random.uniform(range[0], range[1]) for range in range_list]
    pose_list.append(sample)

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    pose_range: dict[str, tuple[float, float]] = {},
):
    if env_ids is None or len(asset_cfgs) == 0:
        return

    asset_cfg1, asset_cfg2 = asset_cfgs[0], asset_cfgs[1]
    asset1 = env.scene[asset_cfg1.name]
    asset2 = env.scene[asset_cfg2.name]

    # 定义asset2相对于asset1的初始位置偏移和旋转
    rel_pos = torch.tensor(
        [-10, -0.04, -10], device=env.device
    )  # asset2相对于asset1的位置偏移
    rel_rot = math_utils.quat_from_euler_xyz(  # asset2相对于asset1的旋转(绕x轴90度)
        torch.tensor([0.0], device=env.device),
        torch.tensor([0.0], device=env.device),
        torch.tensor([math.pi / 2], device=env.device),
    )

    for cur_env in env_ids.tolist():
        # 生成第一个物体的随机位姿
        pose1 = sample_object_poses(pose_range=pose_range)[0]

        # 转换为张量并写入仿真
        pose_tensor1 = torch.tensor([pose1], device=env.device)
        positions1 = pose_tensor1[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientations1 = math_utils.quat_from_euler_xyz(
            pose_tensor1[:, 3], pose_tensor1[:, 4], pose_tensor1[:, 5]
        )

        # 设置asset1的位姿
        asset1.write_root_pose_to_sim(
            torch.cat([positions1, orientations1], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device),
        )
        asset1.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device),
        )

        # 计算asset2的位姿
        # 位置 = asset1的位置 + 相对位置偏移
        positions2 = positions1 - rel_pos.unsqueeze(0)
        # 旋转 = asset1的旋转 * 相对旋转
        orientations2 = math_utils.quat_mul(orientations1, rel_rot)

        # # 设置asset2的位姿
        asset2.write_root_pose_to_sim(
            torch.cat([positions2, orientations2], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device),
        )
        asset2.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device),
        )


def randomize_single_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    pose_range: dict[str, tuple[float, float]] = {},
):
    if env_ids is None or len(asset_cfgs) == 0:
        return

    asset_cfg1 = asset_cfgs[0]
    asset1 = env.scene[asset_cfg1.name]

    for cur_env in env_ids.tolist():
        # 生成第一个物体的随机位姿
        pose1 = sample_object_poses(pose_range=pose_range)[0]

        # 转换为张量并写入仿真
        pose_tensor1 = torch.tensor([pose1], device=env.device)
        positions1 = pose_tensor1[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientations1 = math_utils.quat_from_euler_xyz(
            pose_tensor1[:, 3], pose_tensor1[:, 4], pose_tensor1[:, 5]
        )

        # 设置asset1的位姿
        asset1.write_root_pose_to_sim(
            torch.cat([positions1, orientations1], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device),
        )
        asset1.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device),
        )


def set_boxjoint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    target_joint_pos: float,
):
    """Reset the box joint position to a specified value."""
    box: Articulation = env.scene[asset_cfg.name]

    joint_indices, joint_names = box.find_joints("boxjoint")

    target_positions = torch.full(
        size=(len(env_ids), len(joint_indices)),
        fill_value=target_joint_pos,
        device=env.device,
        dtype=torch.float,
    )

    box.set_joint_position_target(
        target=target_positions, joint_ids=joint_indices, env_ids=env_ids
    )

    box.write_data_to_sim()


# def set_shelf_position(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     asset_cfg: SceneEntityCfg,
#     target_shelf_state: torch.Tensor,
# ):
#     """Set the shelf to a specified root state."""
#     # retrieve the shelf asset
#     shelf: RigidObject = env.scene[asset_cfg.name]

#     # 确保 env_ids 是张量且正确处理维度
#     if isinstance(env_ids, int):
#         env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)
#     elif not isinstance(env_ids, torch.Tensor):
#         env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

#     # 如果 env_ids 是标量，转换为1D张量
#     if env_ids.dim() == 0:
#         env_ids = env_ids.unsqueeze(0)

#     # 确保 target_shelf_state 是二维的 (len(env_ids), 13)
#     if target_shelf_state.dim() == 1:
#         target_shelf_state = target_shelf_state.unsqueeze(0)

#     # 确保目标状态与 env_ids 数量匹配
#     if target_shelf_state.shape[0] != len(env_ids):
#         # 如果只有一个目标状态，复制到所有 env_ids
#         if target_shelf_state.shape[0] == 1:
#             target_shelf_state = target_shelf_state.repeat(len(env_ids), 1)
#         else:
#             raise ValueError(
#                 f"target_shelf_state shape {target_shelf_state.shape} does not match "
#                 f"number of env_ids {len(env_ids)}"
#             )

#     # 确保目标状态在正确的设备上
#     target_shelf_state = target_shelf_state.to(device=env.device, dtype=torch.float)

#     # write to simulation
#     shelf.write_root_state_to_sim(target_shelf_state, env_ids=env_ids)


def set_shelf_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    target_shelf_state: torch.Tensor,
):
    """Set the shelf to a specified root state."""
    # retrieve the shelf asset
    shelf: RigidObject = env.scene[asset_cfg.name]

    # 确保 env_ids 是张量且正确处理维度
    if isinstance(env_ids, int):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # 如果 env_ids 是标量，转换为1D张量
    if env_ids.dim() == 0:
        env_ids = env_ids.unsqueeze(0)

    num_envs = len(env_ids)

    # 确保 target_shelf_state 在正确的设备上
    target_shelf_state = target_shelf_state.to(device=env.device, dtype=torch.float)

    # 如果 target_shelf_state 是1D，复制到所有环境
    if target_shelf_state.dim() == 1:
        target_shelf_state = target_shelf_state.unsqueeze(0).repeat(num_envs, 1)

    # 确保目标状态数量与环境数量匹配
    if target_shelf_state.shape[0] != num_envs:
        raise ValueError(
            f"target_shelf_state shape {target_shelf_state.shape} does not match "
            f"number of env_ids {num_envs}"
        )

    # 像 randomize_object_pose 一样，逐个环境设置位姿
    for i, cur_env in enumerate(env_ids.tolist()):
        # 获取当前环境的目标状态
        cur_state = target_shelf_state[i].unsqueeze(0)

        # 分离位置和旋转
        positions = cur_state[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientations = cur_state[:, 3:7]

        # 设置货架的位姿
        shelf.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device),
        )

        # 设置货架的速度（如果有速度信息）
        if cur_state.shape[1] >= 13:
            velocities = cur_state[:, 7:13]
            shelf.write_root_velocity_to_sim(
                velocities, env_ids=torch.tensor([cur_env], device=env.device)
            )
        else:
            # 如果没有速度信息，设置为零速度
            shelf.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )


def reset_last_leave(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    """Reset the last press state."""
    # env._last_leave_for_termination = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "_leave_triggered"):
        env._leave_triggered.fill_(False)
