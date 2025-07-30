from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import (
    FrameTransformerCfg,
    ContactSensorCfg,
    CameraCfg,
)
from isaaclab.sim.spawners import PinholeCameraCfg

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from ....move_20250708 import mdp
from ....move_20250708.mdp import ur5_move_events
from ....move_20250708.move_env_cfg_rgb import MoveEnvCfg
from .robots.alice import ALICE_CFG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.universal_robots import UR5_CFG
from isaaclab_assets.robots.toolbox import TOOLBOX_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events.
    事件触发器"""

    # 随机化UR5机械臂的关节状态
    randomize_ur5_joint_state = EventTerm(
        func=ur5_move_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            # "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 随机化箱子、扳手位姿
    randomize_object_positions = EventTerm(
        func=ur5_move_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (1.15, 1.28),
                "y": (-3.48, -3.4),
                "z": (2.9, 2.9),
                "yaw": (-0.1, 0.1),
            },
            # "pose_range": {"x": (1.215, 1.215), "y": (-3.45, -3.45), "z": (2.9, 2.9), "yaw": (0, 0)},
            "asset_cfgs": [SceneEntityCfg("box"), SceneEntityCfg("spanner")],
        },
    )

    # 重置时将箱子关节置0
    reset_boxjoint_position = EventTerm(
        func=ur5_move_events.set_boxjoint_pose,
        mode="reset",
        params={
            "target_joint_pos": 0.0,
            "asset_cfg": SceneEntityCfg("box"),
        },
    )

    # reset_last_press = EventTerm(
    #     func=ur5_move_events.reset_last_leave,
    #     mode="reset",
    #     params={},
    # )

    # 按钮按下时触发箱盖打开
    button_pressed = EventTerm(
        func=ur5_move_events.set_boxjoint_pose,
        mode="press",
        params={
            "target_joint_pos": 0.33,
            "asset_cfg": SceneEntityCfg("box"),
        },
    )

    view_settings = EventTerm(
        func=ur5_move_events.set_view_settings,
        mode="startup",
        params={},
    )


@configclass
class UR5BoxMoveEnvCfg(MoveEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        # 事件触发器
        self.events = EventCfg()

        # Set UR5 as robot
        self.scene.robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.box = TOOLBOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Box")

        self.scene.alice = ALICE_CFG.replace(prim_path="{ENV_REGEX_NS}/Alice")

        self.scene.button = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box/red_toolbox/Toolbox/box_lower/Button/button",
        )

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
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
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

        # alice_properties = RigidBodyPropertiesCfg(
        #     solver_position_iteration_count=16,
        #     solver_velocity_iteration_count=1,
        #     max_angular_velocity=1000.0,
        #     max_linear_velocity=1000.0,
        #     max_depenetration_velocity=5.0,
        #     disable_gravity=True,
        # )

        self.scene.spanner = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Spanner",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[1.4, -2.5, 2.9], rot=[0.70711, 0.70711, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0409/spanner.usd",
                scale=(1.3, 1.6, 1.3),
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

        # self.scene.alice = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Alice",
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=[0.4067, -3.1, 1.6], rot=[0.5, 0.5, 0.5, 0.5]
        #     ),
        #     spawn=UsdFileCfg(
        #         usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0409/PN_Stickman_v12_ThumbInward.usd",
        #         scale=(0.01, 0.01, 0.01),
        #         rigid_props=alice_properties,
        #     ),
        # )

        # self.scene.alice_right_forearm = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Alice/PN_Stickman_v12_ThumbInward/Noitom_Hips/Noitom_Spine/Noitom_Spine1/Noitom_Spine2/Noitom_RightShoulder/Noitom_RightArm/Noitom_RightForeArm",
        # )

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
            spawn=PinholeCameraCfg(
                vertical_aperture=None, focal_length=20
            ),  # focal_length=28
            offset=CameraCfg.OffsetCfg(
                pos=(1.175, -3.7, 4.04),  # (1.7, -3.95, 3.5),
                rot=(0.99027, 0.13917, 0, 0),  # (0.85009, 0.35166, 0.17222, 0.35216),
                convention="opengl",
            ),
            width=256,
            height=256,
            # width=480,
            # height=480,
            data_types=["rgb", "depth"],
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
            # width=256,
            # height=256,
            width=480,
            height=480,
            data_types=["rgb", "depth"],
            debug_vis=True,
        )
        # add sidecamera
        self.scene.sidecamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCamera",
            spawn=PinholeCameraCfg(
                vertical_aperture=None, focal_length=20
            ),  # focal_length=23
            offset=CameraCfg.OffsetCfg(
                pos=(0.07763, -3.7, 3.63566),  # (0.07763, -3.91718, 3.77474),
                rot=(
                    0.57923,
                    0.40558,
                    -0.40558,
                    -0.57923,
                ),  # (0.65996, 0.4312, -0.33461, -0.51629),
                convention="opengl",
            ),
            width=256,
            height=256,
            data_types=["rgb", "depth"],
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
            width=480,
            height=480,
            data_types=["rgb", "depth"],
            debug_vis=True,
        )
        # add wristcamera
        self.scene.wristcamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link/WristCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=12.5),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.07, 0.0, 0.0),
                rot=(0, 0.70711, -0.70711, 0),
                convention="opengl",
            ),
            width=256,
            height=256,
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
        # add contact_sensor
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
