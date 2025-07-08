from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, ContactSensorCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.move import mdp
from isaaclab_tasks.manager_based.manipulation.move.mdp import ur5_move_events
from isaaclab_tasks.manager_based.manipulation.move.move_env_cfg import MoveEnvCfg

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
                "y": (-3.4, -3.48),
                "z": (2.9, 2.9),
                "yaw": (-0.1, 0.1),
            },
            "asset_cfgs": [SceneEntityCfg("box"), SceneEntityCfg("spanner")],
        },
    )

    # 重置时将箱子关节置0
    reset_boxjoint_position = EventTerm(
        func=ur5_move_events.set_boxjoint_pose,
        mode="reset",
        params={
            "target_joint_pos": 0.0,
            # "target_joint_pos": 0.33,
            "asset_cfg": SceneEntityCfg("box"),
        },
    )

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
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/left_inner_finger",
                    name="leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

        # add frontcemaera
        self.scene.frontcamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=12.5),
            offset=CameraCfg.OffsetCfg(
                pos=(1.175, -2.29804, 3.19415),
                rot=(0, 0, 0.6157, 0.7880),
                convention="opengl",
            ),
            # offset=CameraCfg.OffsetCfg(pos=(1.175, -2.43562, 3.30584), rot=(1, 0, 0, 0), convention="opengl"),
            width=256,
            height=256,
            data_types=["rgb"],
            debug_vis=True,
        )

        # add sidecamera
        self.scene.sidecamera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCamera",
            spawn=PinholeCameraCfg(vertical_aperture=None, focal_length=12.5),
            # offset=CameraCfg.OffsetCfg(pos=(0.07763, -3.56398, 3.63566), rot=(0.5792, -0.4056, -0.4056, -0.5792), convention="opengl"),
            offset=CameraCfg.OffsetCfg(
                pos=(0.07763, -3.56398, 3.63566),
                rot=(0.57923, 0.40558, -0.40558, -0.57923),
                convention="opengl",
            ),
            width=256,
            height=256,
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
                convention="opengl",
            ),
            width=256,
            height=256,
            data_types=["rgb"],
            debug_vis=True,
        )

        # # add contact_sensor
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
