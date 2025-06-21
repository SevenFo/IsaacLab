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
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def set_view_settings(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor
):
    """Set the view settings for the environment."""
    env.sim.set_camera_view(
        eye = (-1.81532, 0.97065, 4.73775),
        target =(1.215, -3.200, 2.860)
    )
    

def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


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
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def sample_object_poses(
    pose_range: dict[str, tuple[float, float]] = {},
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
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
    rel_pos = torch.tensor([0.0, -0.04, 0.001], device=env.device)  # asset2相对于asset1的位置偏移
    rel_rot = math_utils.quat_from_euler_xyz(  # asset2相对于asset1的旋转(绕x轴90度)
        torch.tensor([0.0], device=env.device),
        torch.tensor([0.0], device=env.device),
        torch.tensor([math.pi/2], device=env.device)
    )
    
    for cur_env in env_ids.tolist():
        # 生成第一个物体的随机位姿
        pose1 = sample_object_poses(pose_range=pose_range)[0]

        # 转换为张量并写入仿真
        pose_tensor1 = torch.tensor([pose1], device=env.device)
        positions1 = pose_tensor1[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientations1 = math_utils.quat_from_euler_xyz(pose_tensor1[:, 3], pose_tensor1[:, 4], pose_tensor1[:, 5])
        
        # print(f"positions1: {positions1}, orientations1: {orientations1}")
        
        # 设置asset1的位姿
        asset1.write_root_pose_to_sim(
            torch.cat([positions1, orientations1], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
        )
        asset1.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
        )

        # 计算asset2的位姿
        # 位置 = asset1的位置 + 相对位置偏移
        positions2 = positions1 - rel_pos.unsqueeze(0)
        # 旋转 = asset1的旋转 * 相对旋转
        orientations2 = math_utils.quat_mul(orientations1, rel_rot)
        
        # print(f"positions2: {positions2}, orientations2: {orientations2}")
        
        # # 设置asset2的位姿
        asset2.write_root_pose_to_sim(
            torch.cat([positions2, orientations2], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
        )
        asset2.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
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
            dtype=torch.float
        )
    
    box.set_joint_position_target(
        target=target_positions,
        joint_ids=joint_indices,
        env_ids=env_ids
    )
    
    box.write_data_to_sim()
    
    # print(f"target_pos:{box.data.joint_pos_target},joint_pos:{box.data.joint_pos}")
    