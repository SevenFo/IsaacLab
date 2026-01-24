from isaaclab.assets import RigidObjectCfg
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
from ..robots.toolbox_heavy import HEAVY_TOOLBOX_CFG  # isort: skip
from ..robots.alice import ALICE_CFG


@configclass
class EventCfg:
    """Visual Test 专用事件配置"""

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

    # ========== 场景1：红色工具箱检测测试 ==========
    # 随机化红色工具箱和干扰箱子的位置
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
            "asset_cfg": SceneEntityCfg("heavy_box"),  # 红色工具箱（目标）
            "distractor_cfgs": [
                SceneEntityCfg("distractor_box_1"),
                SceneEntityCfg("distractor_box_2"),
                SceneEntityCfg("distractor_box_3"),
            ],
        },
    )

    # ========== 场景2：扳手检测测试 ==========
    # 随机化扳手和干扰工具的位置
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
                "yaw": (-3.14, 3.14),  # 全方位旋转
            },
            "spanner_cfg": SceneEntityCfg("spanner"),
            "distractor_tool_cfgs": [
                SceneEntityCfg("screwdriver"),
                SceneEntityCfg("hammer"),
            ],
        },
    )

    # 在扳手测试时隐藏所有箱子
    hide_boxes_for_spanner_test = EventTerm(
        func=visual_test_events.hide_assets,
        mode="reset",
        params={
            "asset_cfgs": [
                SceneEntityCfg("box"),
                SceneEntityCfg("heavy_box"),
                SceneEntityCfg("distractor_box_1"),
                SceneEntityCfg("distractor_box_2"),
                SceneEntityCfg("distractor_box_3"),
            ],
        },
    )

    # ========== 场景3：人手检测测试 ==========
    # Alice 随机小幅度移动
    hand_test_randomize = EventTerm(
        func=visual_test_events.randomize_hand_test_scene,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.35, 0.45),  # ±0.05m
                "y": (0.9, 1.1),  # ±0.1m
                "z": (1.65, 1.75),  # ±0.05m
                "yaw": (1.47, 1.67),  # ±0.1 rad
                "pitch": (1.47, 1.67),  # ±0.1 rad
                "roll": (-0.05, 0.05),  # ±0.05 rad
            },
            "alice_cfg": SceneEntityCfg("alice"),
        },
    )

    # 在人手测试时隐藏所有其他资产
    hide_objects_for_hand_test = EventTerm(
        func=visual_test_events.hide_assets,
        mode="reset",
        params={
            "asset_cfgs": [
                SceneEntityCfg("box"),
                SceneEntityCfg("heavy_box"),
                SceneEntityCfg("distractor_box_1"),
                SceneEntityCfg("distractor_box_2"),
                SceneEntityCfg("distractor_box_3"),
                SceneEntityCfg("spanner"),
                SceneEntityCfg("screwdriver"),
                SceneEntityCfg("hammer"),
            ],
        },
    )

    # 重置箱子关节
    reset_boxjoint_position = EventTerm(
        func=ur5_move_events.set_boxjoint_pose,
        mode="reset",
        params={
            "target_joint_pos": 0.0,
            "asset_cfg": SceneEntityCfg("heavy_box"),
        },
    )


@configclass
class UR5BoxMoveEnvCfg(MoveEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set UR5 as robot
        self.scene.robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ========== 目标资产 ==========
        # 原有的箱子（用于场景切换时隐藏）
        self.scene.box = TOOLBOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Box")

        # 红色工具箱（场景1的目标）
        self.scene.heavy_box = HEAVY_TOOLBOX_CFG.replace(
            prim_path="{ENV_REGEX_NS}/HeavyBox"
        )

        # 扳手（场景2的目标）
        spanner_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.spanner = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Spanner",
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0409/spanner.usd",
                scale=(1.1, 1.6, 1.3),
                rigid_props=spanner_properties,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(100.0, 100.0, -50.0),  # 默认隐藏
            ),
        )

        # Alice（场景3的目标）
        self.scene.alice = ALICE_CFG

        # ========== 干扰资产池 ==========
        # 场景1的干扰箱子（3个，不同颜色）
        for i in range(1, 4):
            setattr(
                self.scene,
                f"distractor_box_{i}",
                TOOLBOX_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/DistractorBox{i}"),
            )

        # 场景2的干扰工具
        self.scene.screwdriver = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Screwdriver",
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/tools/Screwdriver.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(100.0, 100.0, -50.0),  # 默认隐藏
            ),
        )

        self.scene.hammer = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Hammer",
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/tools/Hammer.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(100.0, 100.0, -50.0),  # 默认隐藏
            ),
        )

        # ========== 箱子组件（用于关节控制）==========
        self.scene.heavy_button = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/HeavyBox/red_toolbox/Toolbox/box_lower/Button/button",
        )

        self.scene.heavy_bar = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/HeavyBox/red_toolbox/Toolbox/box_upper/bar/bar",
        )

        self.scene.button = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box/red_toolbox/Toolbox/box_lower/Button/button",
        )

        self.scene.bar = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box/red_toolbox/Toolbox/box_upper/bar/bar",
        )

        self.scene.alice = ALICE_CFG.replace(prim_path="{ENV_REGEX_NS}/Alice")

        # self.scene.alice_hand = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Alice/Noitom_Hips/Noitom_Spine/Noitom_Spine1/Noitom_Spine2/Noitom_RightShoulder/Noitom_RightArm/SM_RightArm/Noitom_RightForeArm/Noitom_RightHand",

        # Set actions for the specific robot type (ur5)
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
                pos=[1.4, -2.5, 0.0], rot=[0.70711, 0.70711, 0, 0]
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
                pos=[0.73461, -2.82622, 1.82532], rot=[0, 0, 0, 1]
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
                pos=[0.04, 0.08, 0], rot=[1, 0, 0, 0]
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
                pos=[0.04, 0.08, 0], rot=[1, 0, 0, 0]
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
                pos=[0.04, 0.08, 0], rot=[1, 0, 0, 0]
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
                pos=[0.04, 0.08, 0], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/plane_4.usd",
                scale=(1, 1, 1),
                rigid_props=plane_properties,
            ),
        )

        # 末端执行器跟踪坐标系配置
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
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/right_inner_finger",
                    name="rightfinger",
                    offset=OffsetCfg(
                        pos=(0.15, 0.0425, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/left_inner_finger",
                    name="leftfinger",
                    offset=OffsetCfg(
                        pos=(0.15, -0.0425, 0.0),
                    ),
                ),
            ],
        )

        # add frontcemaera
        self.scene.topcamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/TopCamera",
            # spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=20),
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=28),
            offset=CameraCfg.OffsetCfg(
                pos=(1.175, -3.7, 4.04),
                rot=(0.99027, 0.13917, 0, 0),
                # pos=(1.7, -3.95, 3.5),
                # rot=(0.85009, 0.35166, 0.17222, 0.35216),
                convention="opengl",
            ),
            # width=256,
            # height=256,
            width=224,
            height=224,
            data_types=["rgb"],
            debug_vis=True,
        )

        # add sidecamera
        self.scene.sidecamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCamera",
            # spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=20),
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=23),
            offset=CameraCfg.OffsetCfg(
                pos=(0.07763, -3.7, 3.63566),
                rot=(0.57923, 0.40558, -0.40558, -0.57923),
                # pos=(0.07763, -3.91718, 3.77474),
                # rot=(0.65996, 0.4312, -0.33461, -0.51629),
                convention="opengl",
            ),
            # width=256,
            # height=256,
            width=224,
            height=224,
            data_types=["rgb"],
            debug_vis=True,
        )

        # add wristcamera
        self.scene.wristcamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link/WristCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=12.5),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.07, 0.0, 0.0),
                rot=(0, 0.70711, -0.70711, 0),
                # pos=(-0.07, 0.0, 0.0),
                # rot=(0, 0.70711, -0.70711, 0),
                convention="opengl",
            ),
            # width=256,
            # height=256,
            width=224,
            height=224,
            data_types=["rgb"],
            debug_vis=True,
        )

        self.scene.inspector_side = CameraCfg(
            prim_path="{ENV_REGEX_NS}/InsSideCamera",
            spawn=PinholeCameraCfg(
                vertical_aperture=None, focal_length=23
            ),  # focal_length=
            offset=CameraCfg.OffsetCfg(
                pos=(0.07763, -3.91718, 3.77474),
                rot=(0.65996, 0.4312, -0.33461, -0.51629),
                convention="opengl",
            ),
            width=640,
            height=640,
            data_types=["rgb"],
            debug_vis=True,
        )
        self.scene.inspector_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/InsTopCamera",
            spawn=PinholeCameraCfg(
                vertical_aperture=None, focal_length=20
            ),  # focal_length=28
            offset=CameraCfg.OffsetCfg(
                pos=(1.7, -3.95, 3.5),
                rot=(0.85009, 0.35166, 0.17222, 0.35216),
                convention="opengl",
            ),
            width=640,
            height=640,
            data_types=["rgb"],
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

        # add contact_sensor
        self.scene.contact_sensor = ContactSensorCfg(
            track_air_time=True,
            prim_path="{ENV_REGEX_NS}/HeavyBox/red_toolbox/Toolbox/box_lower/Button/button",
            force_threshold=0.1,
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Robot/ur5/gripper/right_inner_finger",
                "{ENV_REGEX_NS}/Robot/ur5/gripper/left_inner_finger",
            ],
            update_period=0.01,
        )
