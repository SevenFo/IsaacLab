# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual Test 环境配置

这个配置基于原有的 move_joint_pos_env_cfg.py，添加了用于视觉测试的额外资产和事件。

测试场景：
1. 红色工具箱检测测试 - 目标红色箱子 + 干扰箱子
2. 扳手检测测试 - 目标扳手 + 干扰工具
3. 人手检测测试 - Alice 人物模型
"""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, ContactSensorCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.move_rot_0923 import mdp
from isaaclab_tasks.manager_based.manipulation.move_rot_0923.mdp import (
    ur5_move_events,
    visual_test_events,
)
from isaaclab_tasks.manager_based.manipulation.move_rot_0923.move_env_cfg import (
    MoveEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG
from ..robots.universal_robots import UR5_CFG  # isort: skip
from ..robots.toolbox import TOOLBOX_CFG  # isort: skip
from ..robots.colored_toolbox import (  # isort: skip
    BLUE_TOOLBOX_CFG,
    ORANGE_TOOLBOX_CFG,
    GREEN_TOOLBOX_CFG,
)
from ..robots.alice import ALICE_CFG


# 隐藏位置常量
HIDDEN_POS = (100.0, 100.0, -50.0)


@configclass
class VisualTestEventCfg:
    """Visual Test 专用事件配置

    注意：这些事件在默认情况下都会执行，测试脚本需要根据场景选择性禁用某些事件。

    场景隔离说明：
    - 场景 1 (红色箱子): 显示 box + 干扰箱子，隐藏工具和 alice
    - 场景 2 (扳手): 显示工具，隐藏所有箱子和 alice
    - 场景 3 (人手): 显示 alice，隐藏所有箱子和工具
    """

    # ========== 基础事件 ==========
    randomize_ur5_joint_state = EventTerm(
        func=ur5_move_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    view_settings = EventTerm(
        func=ur5_move_events.set_view_settings,
        mode="startup",
        params={},
    )

    # 重置箱子关节 (使用 box 作为目标红色工具箱)
    reset_boxjoint_position = EventTerm(
        func=ur5_move_events.set_boxjoint_pose,
        mode="reset",
        params={
            "target_joint_pos": 0.0,
            "asset_cfg": SceneEntityCfg("box"),
        },
    )

    # ========== 场景 1：红色工具箱检测测试 ==========
    # 位置参考 TOOLBOX_CFG 初始位置：(1.215, -3.45, 2.9)
    red_box_test_randomize = EventTerm(
        func=visual_test_events.randomize_red_box_test_scene,
        mode="reset",
        params={
            "pose_range": {
                "x": (1.15, 1.35),
                "y": (-3.50, -3.35),
                "z": (2.9, 2.9),
                "yaw": (-1.75, -1.40),
            },
            "asset_cfg": SceneEntityCfg("box"),  # 红色工具箱（目标）
            "distractor_cfgs": [
                SceneEntityCfg("distractor_box_1"),
                SceneEntityCfg("distractor_box_2"),
                SceneEntityCfg("distractor_box_3"),
            ],
            # 场景隔离：隐藏其他场景的资产
            "hide_tool_cfgs": [
                SceneEntityCfg("spanner"),
                SceneEntityCfg("screwdriver"),
                SceneEntityCfg("hammer"),
            ],
            "hide_alice_cfg": SceneEntityCfg("alice"),
        },
    )

    # ========== 场景 2：扳手检测测试 ==========
    # 位置参考桌面区域，与箱子位置相同
    spanner_test_randomize = EventTerm(
        func=visual_test_events.randomize_spanner_test_scene,
        mode="reset",
        params={
            "pose_range": {
                "x": (1.15, 1.35),
                "y": (-3.50, -3.35),
                "z": (2.9, 2.9),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "spanner_cfg": SceneEntityCfg("spanner"),
            "distractor_tool_cfgs": [
                SceneEntityCfg("screwdriver"),
                SceneEntityCfg("hammer"),
            ],
            # 场景隔离：隐藏所有箱子和 alice
            "hide_box_cfgs": [
                SceneEntityCfg("box"),
                SceneEntityCfg("distractor_box_1"),
                SceneEntityCfg("distractor_box_2"),
                SceneEntityCfg("distractor_box_3"),
            ],
            "hide_alice_cfg": SceneEntityCfg("alice"),
        },
    )

    # ========== 场景 3：人手检测测试 ==========
    # Alice 位置参考自 alice_control_skills.py 的 move_to_operation_position
    # operation_position = [0.3067, -3.4, 2.7, 0.5, 0.5, 0.5, 0.5, ...]
    hand_test_randomize = EventTerm(
        func=visual_test_events.randomize_hand_test_scene,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.35),  # 原始位置 0.3067 ± 0.05m
                "y": (-3.45, -3.35),  # 原始位置 -3.4 ± 0.05m
                "z": (2.65, 2.75),  # 原始位置 2.7 ± 0.05m
            },
            "alice_cfg": SceneEntityCfg("alice"),
            # 场景隔离：隐藏所有箱子和工具
            "hide_box_cfgs": [
                SceneEntityCfg("box"),
                SceneEntityCfg("distractor_box_1"),
                SceneEntityCfg("distractor_box_2"),
                SceneEntityCfg("distractor_box_3"),
            ],
            "hide_tool_cfgs": [
                SceneEntityCfg("spanner"),
                SceneEntityCfg("screwdriver"),
                SceneEntityCfg("hammer"),
            ],
        },
    )


@configclass
class UR5BoxMoveEnvCfg(MoveEnvCfg):
    """Visual Test 环境配置类

    基于原有的 MoveEnvCfg，添加了用于视觉测试的额外资产：
    - 干扰箱子 (distractor_box_1/2/3)
    - 干扰工具 (screwdriver, hammer)
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 使用 Visual Test 专用事件配置
        self.events = VisualTestEventCfg()

        # Set UR5 as robot
        self.scene.robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ========== 原有资产 ==========
        # 红色工具箱（目标资产）
        self.scene.box = TOOLBOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Box")

        # 箱子组件（用于关节控制）
        self.scene.button = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box/red_toolbox/Toolbox/box_lower/Button/button",
        )
        self.scene.bar = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box/red_toolbox/Toolbox/box_upper/bar/bar",
        )

        # Alice
        self.scene.alice = ALICE_CFG.replace(prim_path="{ENV_REGEX_NS}/Alice")

        # ========== 干扰资产池 ==========
        # 使用预定义颜色的工具箱，避免运行时动态修改颜色
        # 蓝色工具箱
        self.scene.distractor_box_1 = BLUE_TOOLBOX_CFG.replace(
            prim_path="{ENV_REGEX_NS}/DistractorBox1",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=HIDDEN_POS,
                joint_pos={"boxjoint": 0.0},
            ),
        )
        # 橙色工具箱
        self.scene.distractor_box_2 = ORANGE_TOOLBOX_CFG.replace(
            prim_path="{ENV_REGEX_NS}/DistractorBox2",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=HIDDEN_POS,
                joint_pos={"boxjoint": 0.0},
            ),
        )
        # 绿色工具箱
        self.scene.distractor_box_3 = GREEN_TOOLBOX_CFG.replace(
            prim_path="{ENV_REGEX_NS}/DistractorBox3",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=HIDDEN_POS,
                joint_pos={"boxjoint": 0.0},
            ),
        )

        # 干扰工具的物理属性
        tool_rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=True,
            kinematic_enabled=True,
        )

        self.scene.screwdriver = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Screwdriver",
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/tools/Screwdriver.usd",
                rigid_props=tool_rigid_props,
                activate_contact_sensors=True,
                scale=(0.057, 0.057, 0.057),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=HIDDEN_POS),
        )

        self.scene.hammer = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Hammer",
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/tools/Hammer.usd",
                rigid_props=tool_rigid_props,
                activate_contact_sensors=True,
                scale=(0.85, 0.85, 0.85),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=HIDDEN_POS),
        )

        # ========== Actions ==========
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow.*", "wrist.*"],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "finger_joint_.*",
                "left_inner_finger_joint",
                "right_inner_finger_joint",
            ],
            open_command_expr={
                "finger_joint_.*": 0.0,
                "left_inner_finger_joint": -0.785,
                "right_inner_finger_joint": -0.785,
            },
            close_command_expr={
                "finger_joint_.*": 0.7,
                "left_inner_finger_joint": -0.785,
                "right_inner_finger_joint": -0.785,
            },
        )

        # ========== 场景资产 ==========
        spanner_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000,
            max_linear_velocity=1000,
            max_depenetration_velocity=5,
            disable_gravity=False,
        )

        desk_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=True,
        )

        plane_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=True,
        )

        self.scene.spanner = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Spanner",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=HIDDEN_POS, rot=(0.70711, 0.70711, 0, 0)
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0409/spanner.usd",
                scale=(1.1, 1.6, 1.3),
                rigid_props=spanner_properties,
            ),
        )

        self.scene.desk = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Desk",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.73461, -2.82622, 1.82532), rot=(0, 0, 0, 1)
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/Desk.usd",
                scale=(1, 1, 1),
                rigid_props=desk_properties,
            ),
        )

        self.scene.plane_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plane_1",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.04, 0.08, 0), rot=(1, 0, 0, 0)
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/plane_1.usd",
                scale=(1, 1, 1),
                rigid_props=plane_properties,
            ),
        )

        self.scene.plane_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plane_2",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.04, 0.08, 0), rot=(1, 0, 0, 0)
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/plane_2.usd",
                scale=(1, 1, 1),
                rigid_props=plane_properties,
            ),
        )

        self.scene.plane_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plane_3",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.04, 0.08, 0), rot=(1, 0, 0, 0)
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/plane_3.usd",
                scale=(1, 1, 1),
                rigid_props=plane_properties,
            ),
        )

        self.scene.plane_4 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plane_4",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.04, 0.08, 0), rot=(1, 0, 0, 0)
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/plane_4.usd",
                scale=(1, 1, 1),
                rigid_props=plane_properties,
            ),
        )

        # ========== 末端执行器跟踪坐标系配置 ==========
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/base_link_gripper",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/right_inner_finger",
                    name="rightfinger",
                    offset=OffsetCfg(pos=(0.15, 0.0425, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/left_inner_finger",
                    name="leftfinger",
                    offset=OffsetCfg(pos=(0.15, -0.0425, 0.0)),
                ),
            ],
        )

        # ========== 相机配置 ==========
        self.scene.topcamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/TopCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=28),
            offset=CameraCfg.OffsetCfg(
                pos=(1.175, -3.7, 4.04),
                rot=(0.99027, 0.13917, 0, 0),
                convention="opengl",
            ),
            width=224,
            height=224,
            data_types=["rgb"],
            debug_vis=True,
        )

        self.scene.sidecamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=23),
            offset=CameraCfg.OffsetCfg(
                pos=(0.07763, -3.7, 3.63566),
                rot=(0.57923, 0.40558, -0.40558, -0.57923),
                convention="opengl",
            ),
            width=224,
            height=224,
            data_types=["rgb"],
            debug_vis=True,
        )

        self.scene.wristcamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link/WristCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=12.5),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.07, 0.0, 0.0),
                rot=(0, 0.70711, -0.70711, 0),
                convention="opengl",
            ),
            width=224,
            height=224,
            data_types=["rgb"],
            debug_vis=True,
        )

        self.scene.inspector_side = CameraCfg(
            prim_path="{ENV_REGEX_NS}/InsSideCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=23),
            offset=CameraCfg.OffsetCfg(
                pos=(0.07763, -3.91718, 3.77474),
                rot=(0.65996, 0.4312, -0.33461, -0.51629),
                convention="opengl",
            ),
            width=640,
            height=640,
            data_types=["rgb", "depth"],
            debug_vis=True,
        )

        self.scene.inspector_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/InsTopCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=20),
            offset=CameraCfg.OffsetCfg(
                pos=(1.7, -3.95, 3.5),
                rot=(0.85009, 0.35166, 0.17222, 0.35216),
                convention="opengl",
            ),
            width=640,
            height=640,
            data_types=["rgb", "depth"],
            debug_vis=True,
        )

        self.scene.leftcamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/LeftCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=23),
            offset=CameraCfg.OffsetCfg(
                pos=(0.74745, -3.77492, 3.29499),
                rot=(0.8925, 0.37505, -0.09707, -0.23101),
                convention="opengl",
            ),
            width=480,
            height=480,
            data_types=["rgb", "depth"],
            debug_vis=True,
        )

        # ========== 接触传感器 ==========
        self.scene.contact_sensor = ContactSensorCfg(
            track_air_time=True,
            prim_path="{ENV_REGEX_NS}/Box/red_toolbox/Toolbox/box_lower/Button/button",
            force_threshold=0.1,
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Robot/ur5/gripper/right_inner_finger",
                "{ENV_REGEX_NS}/Robot/ur5/gripper/left_inner_finger",
            ],
            update_period=0.01,
        )
