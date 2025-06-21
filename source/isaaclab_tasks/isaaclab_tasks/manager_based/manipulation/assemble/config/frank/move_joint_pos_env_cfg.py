import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.manager_based.manipulation.assemble import mdp
from isaaclab_tasks.manager_based.manipulation.assemble.move_env_cfg import (
    MoveEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class EventCfg:
    """Configuration for events.
    事件触发器"""

    # Franka机械臂的初始位置
    init_franka_arm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [
                0.0444,
                -0.1894,
                -0.1107,
                -2.5148,
                0.0044,
                2.3775,
                0.6952,
                0.0400,
                0.0400,
            ],
        },
    )

    # 随机化Franka机械臂的关节状态
    randomize_franka_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 随机化assemble_inner的位姿
    # 固定assemble_inner的位姿
    randomize_assemble_inner_pose = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            # "pose_range": {"x": (0.489, 0.754), "y": (-2.000, -2.047), "z": (2.818, 2.818), "yaw": (-0.5, 0.5)},
            "pose_range": {
                "x": (0.600, 0.600),
                "y": (-2.000, -2.000),
                "z": (2.818, 2.818),
                "yaw": (0, 0),
            },
            "asset_cfgs": [SceneEntityCfg("assemble_inner")],
        },
    )

    # 随机化assemble_outer的位姿
    # 减小assemble_outer的位姿的随机化程度
    randomize_assemble_outer_pose = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            # "pose_range": {
            #     "x": (0.58, 0.62),
            #     "y": (-1.87, -1.83),
            #     "z": (2.885, 2.885),
            #     "yaw": (2.6, 3.6),
            # },
            "pose_range": {
                "x": (0.6, 0.6),
                "y": (-1.85, -1.85),
                "z": (2.885, 2.885),
                "yaw": (3.14, 3.14),
            },
            "asset_cfgs": [SceneEntityCfg("assemble_outer")],
        },
    )


@configclass
class FrankBoxMoveEnvCfg(MoveEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        # 事件触发器r
        self.events = EventCfg()

        self._robot_prim_path = "{ENV_REGEX_NS}/Robot"

        # Set robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path=self._robot_prim_path)
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to lunar_base
        self.scene.lunar_base.spawn.semantic_tags = [("class", "scene")]

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        desk_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=True,
        )

        assemble_inner_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        assemble_outer_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.desk = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Desk",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.73461, -2.82622, 1.82532], rot=[0, 0, 0, 1]
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/Desk.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=desk_properties,
                semantic_tags=[("class", "desk")],
            ),
        )

        self.scene.assemble_inner = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Assenmble_inner",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.6, -2.0, 2.818], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_assemble_inner/assemble_inner.usdc",
                scale=(1.0, 1.0, 1.0),
                rigid_props=assemble_inner_properties,
                semantic_tags=[("class", "Assenmble_inner")],
            ),
        )

        self.scene.assemble_outer = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Assenmble_outer",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.6, -1.85, 2.885], rot=[0, 0, 0, 1]
            ),
            spawn=UsdFileCfg(
                usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_assemble_outer/assemble_outer.usdc",
                scale=(1.0, 1.0, 1.0),
                rigid_props=assemble_outer_properties,
                semantic_tags=[("class", "Assenmble_outer")],
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=f"{self._robot_prim_path}/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{self._robot_prim_path}/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{self._robot_prim_path}/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{self._robot_prim_path}/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        self.scene.front_camera = TiledCameraCfg(
            prim_path=f"{self._robot_prim_path}/panda_link0/front_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=0.193,  # 1.93mm
                focus_distance=400.0,
                f_stop=0.0,  # disable DOF
                horizontal_aperture=0.384,  # 基于分辨率1280×3μm计算的传感器宽度（1280×3μm=3.84mm=0.384cm）
                vertical_aperture=0.216,  # 基于分辨率720×3μm计算的传感器高度（720×3μm=2.16mm=0.216cm）
                clipping_range=(0.1, 1.0e5),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.62, 0.0, 2.35),
                rot=quat_from_euler_xyz(
                    roll=torch.deg2rad(torch.scalar_tensor(30)),
                    pitch=torch.deg2rad(torch.scalar_tensor(0)),
                    yaw=torch.deg2rad(torch.scalar_tensor(90)),
                )  # w-x-y-z # x-y-z-extrinsic == z-y-x-instrinsic != x-y-z-instrinsic (30,0,90) # orient
                .squeeze()
                .numpy(),  # extrinsic
                convention="opengl",  # forward: -z, up: +y,isaac sim default
            ),
        )
        self.scene.wrist_camera = TiledCameraCfg(
            prim_path=f"{self._robot_prim_path}/panda_hand/wrist_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=0.193,  # 1.93mm
                focus_distance=400.0,
                f_stop=0.0,  # disable DOF
                horizontal_aperture=0.384,  # 基于分辨率1280×3μm计算的传感器宽度（1280×3μm=3.84mm=0.384cm）
                vertical_aperture=0.216,  # 基于分辨率720×3μm计算的传感器高度（720×3μm=2.16mm=0.216cm）
                clipping_range=(0.1, 1.0e5),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-0.05, 0.0, 0.0),
                rot=(
                    math.sqrt(2) / 2,
                    0,
                    0,
                    math.sqrt(2) / 2,
                ),  # w-x-y-z # x-y-z-intrinsic (0,0,90) # orient
                convention="ros",  # +z: forward, -y: up
            ),
        )
        self.scene.inspector_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/inspector_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=0.493,  # 1.93mm
                focus_distance=400.0,
                f_stop=0.0,  # disable DOF
                horizontal_aperture=0.384,  # 基于分辨率1280×3μm计算的传感器宽度（1280×3μm=3.84mm=0.384cm）
                vertical_aperture=0.216,  # 基于分辨率720×3μm计算的传感器高度（720×3μm=2.16mm=0.216cm）
                clipping_range=(0.1, 1.0e5),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-1.08, -2.0, 3.9),
                rot=(
                    0.61538,
                    0.34811,
                    -0.34828,
                    -0.61549,
                ),  # w-x-y-z # x-y-z-intrinsic (0,0,90) # orient
                convention="opengl",  # forward: -z, up: +y,isaac sim default
            ),
        )
