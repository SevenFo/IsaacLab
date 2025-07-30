# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

# from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp


##
# Scene definition
##
@configclass
class RoomSetSceneCfg(InteractiveSceneCfg):
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # conmtact sensor: will be populated by agent env cfg
    contact_sensor: ContactSensorCfg = MISSING

    # Room_set
    ROOM_set = AssetBaseCfg(
        prim_path="/World/ROOM_set",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/Collected_ROOM_set/ROOM_set.no.ur5.box.desk_clean_version.usd"
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)  # 上一次执行的动作
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 关节位置
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 关节速度
        object = ObsTerm(func=mdp.object_obs)
        box_positions = ObsTerm(func=mdp.box_positions_in_world_frame)  # 箱子的位置
        box_orientations = ObsTerm(
            func=mdp.box_orientations_in_world_frame
        )  # 箱子的朝向
        spanner_positions = ObsTerm(
            func=mdp.spanner_positions_in_world_frame
        )  # 扳手的位置
        spanner_orientations = ObsTerm(
            func=mdp.spanner_orientations_in_world_frame
        )  # 扳手的朝向
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)  # 末端执行器的位置
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)  # 末端执行器的朝向
        gripper_pos = ObsTerm(func=mdp.gripper_pos)  # 夹爪的位置
        eef_pos_gripper = ObsTerm(func=mdp.ee_frame_pos_gripper)
        camera_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("topcamera"), "normalize": False},
        )  # RGB相机图像
        camera_side = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("sidecamera"), "normalize": False},
        )  # RGB相机图像
        inspector_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("inspector_top"), "normalize": False},
        )  # RGB相机图像
        inspector_side = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("inspector_side"), "normalize": False},
        )  # RGB相机图像

        camera_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wristcamera"), "normalize": False},
        )  # RGB相机图像
        camera_left = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("leftcamera"), "normalize": False},
        )  # RGB相机图像
        # pointcloud_camera_top = ObsTerm(
        #     func=mdp.camera_pointcloud,
        #     params={"sensor_cfg": SceneEntityCfg("topcamera"), "normalize": True},
        # )  # 点云
        # pointcloud_camera_side = ObsTerm(
        #     func=mdp.camera_pointcloud,
        #     params={"sensor_cfg": SceneEntityCfg("sidecamera"), "normalize": True},
        # )  # 点云
        # pointcloud_camera_wrist = ObsTerm(
        #     func=mdp.camera_pointcloud,
        #     params={"sensor_cfg": SceneEntityCfg("wristcamera"), "normalize": True},
        # )  # 点云
        pointcloud_camera_left = ObsTerm(
            func=mdp.camera_pointcloud,
            params={"sensor_cfg": SceneEntityCfg("leftcamera"), "normalize": True},
        )  # 点云

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # subtask_1
        press = ObsTerm(
            func=mdp.press_button,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_sensor"),
            },
        )

        # subtask_2
        # grasp = ObsTerm(
        #     func=mdp.object_grasped,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("spanner"),
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # subtask_3
    # success = DoneTerm(func=mdp.spanner_stacked)
    success = DoneTerm(func=mdp.leave_button)


@configclass
class MoveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: RoomSetSceneCfg = RoomSetSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 3000.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
