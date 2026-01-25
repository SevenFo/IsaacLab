# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual Test 场景事件处理函数

这个模块只包含用于视觉测试场景的事件函数，不包含配置类。
配置类定义在 config/ur5/move_joint_pos_env_cfg_visual_test.py 中。
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ==============================================================================
# 全局场景状态
# ==============================================================================

# 当前活跃的测试场景
# 可选值：None, "red_box", "spanner", "hand"
# 当为 None 时，所有场景事件都不执行位置随机化
_current_scene: str | None = None


def set_current_scene(scene_name: str | None):
    """设置当前活跃的测试场景

    Args:
        scene_name: 场景名称 ("red_box", "spanner", "hand") 或 None
    """
    global _current_scene
    _current_scene = scene_name
    print(f"[VisualTestEvents] Current scene set to: {_current_scene}")


def get_current_scene() -> str | None:
    """获取当前活跃的测试场景"""
    return _current_scene


# ==============================================================================
# 常量定义
# ==============================================================================

# 隐藏位置
HIDDEN_POS = (100.0, 100.0, -50.0)

# 箱子尺寸估计（用于防重叠计算）
# 红色工具箱大约尺寸：长 ~0.35m, 宽 ~0.2m, 高 ~0.15m
BOX_SIZE = (0.35, 0.2, 0.15)
BOX_MIN_DISTANCE = 0.4  # 箱子之间的最小距离（考虑旋转后对角线）

# 工具尺寸估计
SPANNER_SIZE = (0.25, 0.05, 0.02)  # 扳手
SCREWDRIVER_SIZE = (0.2, 0.03, 0.03)  # 螺丝刀
HAMMER_SIZE = (0.3, 0.08, 0.03)  # 锤子
TOOL_MIN_DISTANCE = 0.25  # 工具之间的最小距离

# 干扰箱子颜色定义 (用于材质随机化)
# 使用非红色的颜色，避免与目标红色箱子混淆
DISTRACTOR_COLORS = [
    (0.0, 0.8, 0.2),  # 绿色
    (0.2, 0.4, 0.9),  # 蓝色
    (1.0, 0.6, 0.0),  # 橙色
    (0.6, 0.0, 0.8),  # 紫色
    (0.9, 0.9, 0.0),  # 黄色
    (0.0, 0.8, 0.8),  # 青色
]


# ==============================================================================
# 辅助函数
# ==============================================================================


def _randomize_asset_color(
    asset: "Articulation", color: tuple[float, float, float], env_id: int = 0
):
    """随机化资产的显示颜色

    使用 Isaac Lab 的 visual_material 方式设置颜色

    Args:
        asset: Articulation 资产实例
        color: RGB 颜色元组 (0-1 范围)
        env_id: 环境 ID（用于获取正确的 prim 路径）
    """
    try:
        import isaaclab.sim as sim_utils

        # 通过 root_physx_view.link_paths 获取具体的 prim 路径
        if not hasattr(asset, "root_physx_view") or asset.root_physx_view is None:
            print("[VisualTestEvents] Asset has no root_physx_view, skipping color")
            return

        link_paths = asset.root_physx_view.link_paths
        if env_id >= len(link_paths) or len(link_paths[env_id]) == 0:
            print(f"[VisualTestEvents] No link paths for env_id {env_id}")
            return

        # 取第一个 link 的路径，提取资产根路径
        first_link_path = link_paths[env_id][0]
        path_parts = first_link_path.split("/")
        if len(path_parts) >= 5:
            prim_path = "/".join(path_parts[:5])
        else:
            prim_path = first_link_path

        print(f"[VisualTestEvents] Setting color {color} for prim: {prim_path}")

        # 获取 stage
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            print(f"[VisualTestEvents] Prim not valid: {prim_path}")
            return

        # 为这个资产创建一个唯一的材质路径
        asset_name = path_parts[4] if len(path_parts) >= 5 else "asset"
        material_path = f"/World/Looks/RandomColor_{asset_name}_{env_id}"

        # 创建 PreviewSurface 材质
        material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        material_cfg.func(material_path, material_cfg)

        # 绑定材质到资产的所有 mesh
        sim_utils.bind_visual_material(prim_path, material_path, stage=stage)

        print(f"[VisualTestEvents] Applied material {material_path} to {prim_path}")

    except Exception as e:
        # 颜色随机化失败不影响主流程
        import traceback

        print(f"[VisualTestEvents] Color randomization failed: {e}")
        traceback.print_exc()


def _check_position_valid(
    new_pos: tuple[float, float],
    existing_positions: list[tuple[float, float]],
    min_distance: float,
) -> bool:
    """检查新位置是否与已有位置保持足够距离

    Args:
        new_pos: 新位置 (x, y)
        existing_positions: 已有位置列表
        min_distance: 最小距离

    Returns:
        是否有效（不与任何已有位置重叠）
    """
    for pos in existing_positions:
        dist = math.sqrt((new_pos[0] - pos[0]) ** 2 + (new_pos[1] - pos[1]) ** 2)
        if dist < min_distance:
            return False
    return True


def _generate_non_overlapping_position(
    pose_range: dict,
    existing_positions: list[tuple[float, float]],
    min_distance: float,
    max_attempts: int = 50,
    offset_range: tuple[float, float] = (-0.3, 0.3),
) -> tuple[float, float] | None:
    """生成不与已有位置重叠的新位置

    Args:
        pose_range: 位置范围字典
        existing_positions: 已有位置列表
        min_distance: 最小距离
        max_attempts: 最大尝试次数
        offset_range: 相对于 pose_range 的偏移范围

    Returns:
        有效位置 (x, y)，如果找不到则返回 None
    """
    for _ in range(max_attempts):
        x = random.uniform(*pose_range["x"]) + random.uniform(*offset_range)
        y = random.uniform(*pose_range["y"]) + random.uniform(*offset_range)

        if _check_position_valid((x, y), existing_positions, min_distance):
            return (x, y)

    return None


def _reset_articulation_joint(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_name: str,
    joint_name: str = "boxjoint",
    joint_pos: float = 0.0,
):
    """重置 Articulation 的关节位置

    Args:
        env: 环境实例
        env_ids: 环境 ID
        asset_name: 资产名称
        joint_name: 关节名称
        joint_pos: 目标关节位置
    """
    try:
        asset: Articulation = env.scene[asset_name]
        # 获取关节索引
        joint_ids = asset.find_joints(joint_name)[0]
        if len(joint_ids) > 0:
            # 设置关节位置
            joint_pos_tensor = torch.full(
                (len(env_ids), len(joint_ids)),
                joint_pos,
                device=env.device,
            )
            joint_vel_tensor = torch.zeros_like(joint_pos_tensor)
            asset.write_joint_state_to_sim(
                joint_pos_tensor, joint_vel_tensor, joint_ids, env_ids
            )
    except (KeyError, IndexError, RuntimeError):
        pass  # 资产或关节可能不存在


# ==============================================================================
# 场景随机化函数
# ==============================================================================


def randomize_red_box_test_scene(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict,
    asset_cfg: SceneEntityCfg,
    distractor_cfgs: list[SceneEntityCfg],
    hide_tool_cfgs: list[SceneEntityCfg] | None = None,
    hide_alice_cfg: SceneEntityCfg | None = None,
):
    """场景 1：随机化红色工具箱和干扰箱子的位置（防重叠）

    场景隔离：隐藏工具和 alice，只显示箱子

    注意：只有当 _current_scene == "red_box" 时才执行

    Args:
        env: 环境实例
        env_ids: 需要重置的环境 ID
        pose_range: 位置范围字典，包含 x, y, z, yaw 的范围
        asset_cfg: 目标资产（红色工具箱）的配置
        distractor_cfgs: 干扰箱子的配置列表
        hide_tool_cfgs: 需要隐藏的工具配置列表
        hide_alice_cfg: 需要隐藏的 alice 配置
    """
    # 检查当前场景是否匹配
    if _current_scene != "red_box":
        return

    print("[VisualTestEvents] Executing red_box scene randomization")

    # ===== 场景隔离：隐藏其他场景的资产 =====
    hidden_pose = torch.tensor(
        [[HIDDEN_POS[0], HIDDEN_POS[1], HIDDEN_POS[2], 1.0, 0.0, 0.0, 0.0]],
        device=env.device,
    )

    # 隐藏工具
    if hide_tool_cfgs:
        for tool_cfg in hide_tool_cfgs:
            try:
                asset = env.scene[tool_cfg.name]
                asset.write_root_pose_to_sim(
                    hidden_pose.repeat(len(env_ids), 1), env_ids
                )
            except KeyError:
                pass

    # 隐藏 alice
    if hide_alice_cfg:
        try:
            alice = env.scene[hide_alice_cfg.name]
            alice.write_root_pose_to_sim(hidden_pose.repeat(len(env_ids), 1), env_ids)
        except KeyError:
            pass

    # ===== 随机化箱子 =====
    # 随机决定显示几个干扰箱子（0-3 个）
    num_distractors = random.randint(1, len(distractor_cfgs))

    # 隐藏所有干扰箱子
    hidden_pose = torch.tensor(
        [[HIDDEN_POS[0], HIDDEN_POS[1], HIDDEN_POS[2], 1.0, 0.0, 0.0, 0.0]],
        device=env.device,
    )
    for distractor_cfg in distractor_cfgs:
        try:
            asset = env.scene[distractor_cfg.name]
            asset.write_root_pose_to_sim(hidden_pose.repeat(len(env_ids), 1), env_ids)
            # 重置关节位置（让箱子合上）
            _reset_articulation_joint(
                env, env_ids, distractor_cfg.name, "boxjoint", 0.0
            )
        except KeyError:
            pass

    # 记录已使用的位置
    used_positions: list[tuple[float, float]] = []

    # 随机化目标箱子位置
    asset = env.scene[asset_cfg.name]
    pos_x = random.uniform(*pose_range["x"])
    pos_y = random.uniform(*pose_range["y"])
    pos_z = (
        pose_range["z"][0] if isinstance(pose_range["z"], tuple) else pose_range["z"]
    )
    yaw = random.uniform(*pose_range["yaw"])

    used_positions.append((pos_x, pos_y))

    quat = quat_from_euler_xyz(
        torch.tensor([0.0], device=env.device),
        torch.tensor([0.0], device=env.device),
        torch.tensor([yaw], device=env.device),
    )
    pose = torch.tensor(
        [
            [
                pos_x,
                pos_y,
                pos_z,
                quat[0, 0].item(),
                quat[0, 1].item(),
                quat[0, 2].item(),
                quat[0, 3].item(),
            ]
        ],
        device=env.device,
    )
    asset.write_root_pose_to_sim(pose.repeat(len(env_ids), 1), env_ids)

    # 重置目标箱子的关节（让箱子合上）
    _reset_articulation_joint(env, env_ids, asset_cfg.name, "boxjoint", 0.0)

    # 随机化选中的干扰箱子位置（防重叠）
    for i in range(num_distractors):
        try:
            distractor = env.scene[distractor_cfgs[i].name]

            # 生成不重叠的位置
            new_pos = _generate_non_overlapping_position(
                pose_range,
                used_positions,
                BOX_MIN_DISTANCE,
                max_attempts=50,
                offset_range=(-0.35, 0.35),
            )

            if new_pos is None:
                # 找不到有效位置，跳过这个干扰箱子
                continue

            dist_x, dist_y = new_pos
            used_positions.append(new_pos)

            dist_yaw = random.uniform(*pose_range["yaw"])

            dist_quat = quat_from_euler_xyz(
                torch.tensor([0.0], device=env.device),
                torch.tensor([0.0], device=env.device),
                torch.tensor([dist_yaw], device=env.device),
            )
            dist_pose = torch.tensor(
                [
                    [
                        dist_x,
                        dist_y,
                        pos_z,
                        dist_quat[0, 0].item(),
                        dist_quat[0, 1].item(),
                        dist_quat[0, 2].item(),
                        dist_quat[0, 3].item(),
                    ]
                ],
                device=env.device,
            )
            distractor.write_root_pose_to_sim(
                dist_pose.repeat(len(env_ids), 1), env_ids
            )

            # 重置干扰箱子的关节（让箱子合上）
            _reset_articulation_joint(
                env, env_ids, distractor_cfgs[i].name, "boxjoint", 0.0
            )

            # 颜色随机化已禁用 - 现在使用预定义颜色的 USD 文件
            # 如需启用动态颜色随机化（备用方案），取消下面两行注释：
            # random_color = random.choice(DISTRACTOR_COLORS)
            # _randomize_asset_color(distractor, random_color, env_id=0)

        except KeyError:
            pass


def randomize_spanner_test_scene(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict,
    spanner_cfg: SceneEntityCfg,
    distractor_tool_cfgs: list[SceneEntityCfg],
    hide_box_cfgs: list[SceneEntityCfg] | None = None,
    hide_alice_cfg: SceneEntityCfg | None = None,
):
    """场景 2：随机化扳手和干扰工具的位置（防重叠）

    场景隔离：隐藏所有箱子和 alice，只显示工具

    注意：只有当 _current_scene == "spanner" 时才执行

    Args:
        env: 环境实例
        env_ids: 需要重置的环境 ID
        pose_range: 位置范围字典，包含 x, y, z, roll, pitch, yaw 的范围
        spanner_cfg: 扳手资产的配置
        distractor_tool_cfgs: 干扰工具的配置列表
        hide_box_cfgs: 需要隐藏的箱子配置列表
        hide_alice_cfg: 需要隐藏的 alice 配置
    """
    # 检查当前场景是否匹配
    if _current_scene != "spanner":
        return

    print("[VisualTestEvents] Executing spanner scene randomization")

    # ===== 场景隔离：隐藏其他场景的资产 =====
    hidden_pose = torch.tensor(
        [[HIDDEN_POS[0], HIDDEN_POS[1], HIDDEN_POS[2], 1.0, 0.0, 0.0, 0.0]],
        device=env.device,
    )

    # 隐藏所有箱子
    if hide_box_cfgs:
        for box_cfg in hide_box_cfgs:
            try:
                asset = env.scene[box_cfg.name]
                asset.write_root_pose_to_sim(
                    hidden_pose.repeat(len(env_ids), 1), env_ids
                )
                # 重置关节（确保箱子是关闭状态）
                _reset_articulation_joint(env, env_ids, box_cfg.name, "boxjoint", 0.0)
            except KeyError:
                pass

    # 隐藏 alice
    if hide_alice_cfg:
        try:
            alice = env.scene[hide_alice_cfg.name]
            alice.write_root_pose_to_sim(hidden_pose.repeat(len(env_ids), 1), env_ids)
        except KeyError:
            pass

    # ===== 随机化工具 =====
    # 随机决定显示几个干扰工具（0-2 个）
    num_distractors = random.randint(1, len(distractor_tool_cfgs))

    # 隐藏所有干扰工具
    hidden_pose = torch.tensor(
        [[HIDDEN_POS[0], HIDDEN_POS[1], HIDDEN_POS[2], 1.0, 0.0, 0.0, 0.0]],
        device=env.device,
    )
    for distractor_cfg in distractor_tool_cfgs:
        try:
            asset = env.scene[distractor_cfg.name]
            asset.write_root_pose_to_sim(hidden_pose.repeat(len(env_ids), 1), env_ids)
        except KeyError:
            pass

    # 记录已使用的位置
    used_positions: list[tuple[float, float]] = []

    # 随机化扳手位置和姿态
    asset = env.scene[spanner_cfg.name]
    pos_x = random.uniform(*pose_range["x"])
    pos_y = random.uniform(*pose_range["y"])
    pos_z = (
        pose_range["z"][0] if isinstance(pose_range["z"], tuple) else pose_range["z"]
    )
    roll = random.uniform(*pose_range.get("roll", (0, 0)))
    pitch = random.uniform(*pose_range.get("pitch", (0, 0)))
    yaw = random.uniform(*pose_range.get("yaw", (-3.14, 3.14)))

    used_positions.append((pos_x, pos_y))

    quat = quat_from_euler_xyz(
        torch.tensor([roll], device=env.device),
        torch.tensor([pitch], device=env.device),
        torch.tensor([yaw], device=env.device),
    )
    pose = torch.tensor(
        [
            [
                pos_x,
                pos_y,
                pos_z,
                quat[0, 0].item(),
                quat[0, 1].item(),
                quat[0, 2].item(),
                quat[0, 3].item(),
            ]
        ],
        device=env.device,
    )
    asset.write_root_pose_to_sim(pose.repeat(len(env_ids), 1), env_ids)

    # 随机化选中的干扰工具位置（防重叠）
    for i in range(num_distractors):
        try:
            distractor = env.scene[distractor_tool_cfgs[i].name]

            # 生成不重叠的位置
            new_pos = _generate_non_overlapping_position(
                pose_range,
                used_positions,
                TOOL_MIN_DISTANCE,
                max_attempts=50,
                offset_range=(-0.25, 0.25),
            )

            if new_pos is None:
                continue

            dist_x, dist_y = new_pos
            used_positions.append(new_pos)

            dist_roll = random.uniform(*pose_range.get("roll", (0, 0)))
            dist_pitch = random.uniform(*pose_range.get("pitch", (0, 0)))
            dist_yaw = random.uniform(*pose_range.get("yaw", (-3.14, 3.14)))

            dist_quat = quat_from_euler_xyz(
                torch.tensor([dist_roll], device=env.device),
                torch.tensor([dist_pitch], device=env.device),
                torch.tensor([dist_yaw], device=env.device),
            )
            dist_pose = torch.tensor(
                [
                    [
                        dist_x,
                        dist_y,
                        pos_z,
                        dist_quat[0, 0].item(),
                        dist_quat[0, 1].item(),
                        dist_quat[0, 2].item(),
                        dist_quat[0, 3].item(),
                    ]
                ],
                device=env.device,
            )
            distractor.write_root_pose_to_sim(
                dist_pose.repeat(len(env_ids), 1), env_ids
            )

        except KeyError:
            pass


def randomize_hand_test_scene(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict,
    alice_cfg: SceneEntityCfg,
    hide_box_cfgs: list[SceneEntityCfg] | None = None,
    hide_tool_cfgs: list[SceneEntityCfg] | None = None,
):
    """场景 3：随机化 Alice 的位置和姿态（人手检测）

    场景隔离：隐藏所有箱子和工具，只显示 alice

    注意：只有当 _current_scene == "hand" 时才执行

    位置和关节状态参考自 alice_control_skills.py 的 move_to_operation_position:
    - 位置：[0.3067, -3.4, 2.7]
    - 四元数：[0.5, 0.5, 0.5, 0.5]
    - 关节：手臂举起，手掌朝上

    Args:
        env: 环境实例
        env_ids: 需要重置的环境 ID
        pose_range: 位置范围字典，包含 x, y, z 的范围
        alice_cfg: Alice 资产的配置
        hide_box_cfgs: 需要隐藏的箱子配置列表
        hide_tool_cfgs: 需要隐藏的工具配置列表
    """
    # 检查当前场景是否匹配
    if _current_scene != "hand":
        return

    print("[VisualTestEvents] Executing hand scene randomization")

    # ===== 场景隔离：隐藏其他场景的资产 =====
    hidden_pose = torch.tensor(
        [[HIDDEN_POS[0], HIDDEN_POS[1], HIDDEN_POS[2], 1.0, 0.0, 0.0, 0.0]],
        device=env.device,
    )

    # 隐藏所有箱子
    if hide_box_cfgs:
        for box_cfg in hide_box_cfgs:
            try:
                asset = env.scene[box_cfg.name]
                asset.write_root_pose_to_sim(
                    hidden_pose.repeat(len(env_ids), 1), env_ids
                )
                # 重置关节（确保箱子是关闭状态）
                _reset_articulation_joint(env, env_ids, box_cfg.name, "boxjoint", 0.0)
            except KeyError:
                pass

    # 隐藏所有工具
    if hide_tool_cfgs:
        for tool_cfg in hide_tool_cfgs:
            try:
                asset = env.scene[tool_cfg.name]
                asset.write_root_pose_to_sim(
                    hidden_pose.repeat(len(env_ids), 1), env_ids
                )
            except KeyError:
                pass

    # ===== 随机化 Alice 位置 =====
    alice: Articulation = env.scene[alice_cfg.name]

    # 位置随机化（小幅度）
    pos_x = random.uniform(*pose_range["x"])
    pos_y = random.uniform(*pose_range["y"])
    pos_z = random.uniform(*pose_range["z"])

    # 使用与 move_to_operation_position 相同的四元数 [0.5, 0.5, 0.5, 0.5]
    # 这个四元数使 Alice 面朝正确的方向
    quat = (0.5, 0.5, 0.5, 0.5)

    # 设置 root pose (pos + quat + 6 个速度分量)
    root_state = torch.tensor(
        [[pos_x, pos_y, pos_z, quat[0], quat[1], quat[2], quat[3], 0, 0, 0, 0, 0, 0]],
        device=env.device,
        dtype=torch.float32,
    )
    alice.write_root_state_to_sim(root_state.repeat(len(env_ids), 1), env_ids)

    # ===== 设置 Alice 关节状态（手臂举起，手掌朝上）=====
    # 参考 alice_control_skills.py 的 move_to_operation_position
    try:
        # 获取关节索引
        joint_names_to_set = [
            "D6Joint_RightArm:0",
            "D6Joint_RightArm:1",
            "D6Joint_RightArm:2",
            "D6Joint_RightForeArm:0",
            "D6Joint_RightForeArm:1",
            "D6Joint_RightForeArm:2",
            "D6Joint_RightWrist:0",
            "D6Joint_RightWrist:1",
            "D6Joint_RightWrist:2",
        ]

        # 对应的关节角度值（弧度）
        joint_values = {
            "D6Joint_RightArm:0": 0.0,
            "D6Joint_RightArm:1": math.radians(66.7),
            "D6Joint_RightArm:2": math.radians(50.7),
            "D6Joint_RightForeArm:0": 0.0,
            "D6Joint_RightForeArm:1": math.radians(25.9),
            "D6Joint_RightForeArm:2": math.radians(-23.2),
            "D6Joint_RightWrist:0": math.radians(-150)
            + math.radians(random.uniform(-5, 30)),
            "D6Joint_RightWrist:1": math.radians(10)
            + math.radians(random.uniform(-10, 10)),
            "D6Joint_RightWrist:2": math.radians(8)
            + math.radians(random.uniform(-10, 10)),
        }

        # 获取当前关节位置并修改特定关节
        joint_pos = alice.data.joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)

        for joint_name, value in joint_values.items():
            try:
                joint_ids = alice.find_joints(joint_name)[0]
                if len(joint_ids) > 0:
                    joint_pos[:, joint_ids[0]] = value
            except (KeyError, IndexError):
                pass

        # 写入关节状态
        alice.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        alice.set_joint_position_target(joint_pos, env_ids=env_ids)

    except Exception:
        # 如果关节设置失败，只设置位置
        pass


# ==============================================================================
# 资产显示/隐藏函数
# ==============================================================================


def hide_assets(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
):
    """隐藏指定的资产列表

    通过将资产移动到地下远离场景的位置来实现隐藏。

    Args:
        env: 环境实例
        env_ids: 需要操作的环境 ID
        asset_cfgs: 要隐藏的资产配置列表
    """
    hidden_pose = torch.tensor(
        [[HIDDEN_POS[0], HIDDEN_POS[1], HIDDEN_POS[2], 1.0, 0.0, 0.0, 0.0]],
        device=env.device,
    )

    for asset_cfg in asset_cfgs:
        try:
            asset = env.scene[asset_cfg.name]
            asset.write_root_pose_to_sim(hidden_pose.repeat(len(env_ids), 1), env_ids)
        except KeyError:
            pass


def show_assets(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    default_positions: list[tuple] | None = None,
):
    """显示指定的资产列表

    Args:
        env: 环境实例
        env_ids: 需要操作的环境 ID
        asset_cfgs: 要显示的资产配置列表
        default_positions: 每个资产的默认位置列表 (x, y, z)
    """
    for i, asset_cfg in enumerate(asset_cfgs):
        try:
            asset = env.scene[asset_cfg.name]
            if default_positions and i < len(default_positions):
                pos = default_positions[i]
                pose = torch.tensor(
                    [[pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]], device=env.device
                )
            else:
                pose = torch.tensor(
                    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=env.device
                )
            asset.write_root_pose_to_sim(pose.repeat(len(env_ids), 1), env_ids)
        except KeyError:
            pass
