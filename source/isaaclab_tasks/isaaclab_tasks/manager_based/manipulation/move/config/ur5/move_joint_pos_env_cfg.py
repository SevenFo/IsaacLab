from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.move import mdp
from isaaclab_tasks.manager_based.manipulation.move.mdp import ur5_move_events
from isaaclab_tasks.manager_based.manipulation.move.move_env_cfg import (
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

    # UR5机械臂的初始关节位置
    # init_ur5_arm_pose = EventTerm(
    #     func=ur5_move_events.set_default_joint_pose,
    #     mode="startup",
    #     # 都要换算成弧度
    #     params={
    #         "default_pose": [-0.523, -1.856, 0.741, -0.156, -1.744, 1.570, 0.000, 0.000, -0.17968, 0.80768, 0.07501, 0.25643, 0.50414, -0.785],
    #     },
    # )

    # # 随机化UR5机械臂的关节状态
    # randomize_ur5_joint_state = EventTerm(
    #     func=ur5_move_events.randomize_joint_by_gaussian_offset,
    #     mode="reset",
    #     params={
    #         "mean": 0.0,
    #         "std": 0.02,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # Franka机械臂的初始位置
    init_franka_arm_pose = EventTerm(
        func=ur5_move_events.set_default_joint_pose,
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
        func=ur5_move_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 随机化箱子的位姿
    # randomize_box_positions = EventTerm(
    #     func=ur5_move_events.randomize_object_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (0.7, 1.0), "y": (-0.58, -0.47), "z": (2.52892, 2.52892), "yaw": (-0.5, 0.5)},
    #         "asset_cfgs": [SceneEntityCfg("box")]
    #     }
    # )

    # 随机化assemble_inner的位姿
    # 固定assemble_inner的位姿
    randomize_assemble_inner_pose = EventTerm(
        func=ur5_move_events.randomize_object_pose,
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
        func=ur5_move_events.randomize_object_pose,
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
class UR5BoxMoveEnvCfg(MoveEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        # 事件触发器r
        self.events = EventCfg()

        # Set UR5 as robot
        # self.scene.robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to ROOM_set
        self.scene.ROOM_set.spawn.semantic_tags = [("class", "scene")]

        # Set actions for the specific robot type (ur5)
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], scale=0.5, use_default_offset=True
        # )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["finger_joint_.*", "left_inner_finger_joint", "right_inner_finger_joint"],
        #     open_command_expr={"finger_joint_.*": 0.0, "left_inner_finger_joint": -0.785, "right_inner_finger_joint": -0.785},
        #     close_command_expr={"finger_joint_.*": 0.7, "left_inner_finger_joint": -0.785, "right_inner_finger_joint": -0.785},
        # )

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

        # box_properties = RigidBodyPropertiesCfg(
        #     solver_position_iteration_count=16,
        #     solver_velocity_iteration_count=1,
        #     max_angular_velocity=1000.0,
        #     max_linear_velocity=1000.0,
        #     max_depenetration_velocity=5.0,
        #     disable_gravity=False,
        # )

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

        # self.scene.box = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Box",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.75351, -0.40163, 2.51788], rot=[1, 0, 0, 0]),
        #     # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.72271, -1.85085, 3.10000], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/box_lower.usd",
        #                      scale=(0.2, 0.2, 0.2),
        #                      rigid_props=box_properties,
        #                      semantic_tags=[("class", "box")],
        #     ),
        # )

        # self.scene.box = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Box",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.75351, -0.42568, 2.52892], rot=[0.707, 0, 0, 0.707]),
        #     spawn=UsdFileCfg(usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/Toolbox_no_joint.usd",
        #                      scale=(0.2, 0.2, 0.2),
        #                      rigid_props=box_properties,
        #                      semantic_tags=[("class", "box")],
        #     ),
        # )

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

        # 末端执行器跟踪坐标系配置
        # marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # self.scene.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link",
        #     debug_vis=False,
        #     visualizer_cfg=marker_cfg,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/base_link_gripper",
        #             name="end_effector",
        #             offset=OffsetCfg(
        #                 pos=[0.0, 0.0, 0.0],
        #             ),
        #         ),
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/right_inner_finger",
        #             name="rightfinger",
        #             offset=OffsetCfg(
        #                 pos=(0.0, 0.0, 0.0),
        #             ),
        #         ),
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/ur5/gripper/left_inner_finger",
        #             name="leftfinger",
        #             offset=OffsetCfg(
        #                 pos=(0.0, 0.0, 0.0),
        #             ),
        #         ),
        #     ],
        # )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
