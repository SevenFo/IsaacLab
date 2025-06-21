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
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import  UsdFileCfg
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

    # Room_set
    ROOM_set = AssetBaseCfg(
        prim_path="/World/ROOM_set",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/Collected_ROOM_set/ROOM_set.no.ur5.box.desk_clean_version.usd"),
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

        actions = ObsTerm(func=mdp.last_action)        # 上一次执行的动作                                                                             
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)      # 关节位置
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)      # 关节速度
        object = ObsTerm(func=mdp.object_obs)            # assemble位置、角度，末端执行器相对箱子位置，箱子相对桌子位置
        assemble_inner_positions = ObsTerm(func=mdp.assemble_inner_positions_in_world_frame)    # assemble_inner的位置
        assemble_inner_orientations = ObsTerm(func=mdp.assemble_inner_orientations_in_world_frame)    # assemble_inner的朝向
        assemble_outer_positions = ObsTerm(func=mdp.assemble_outer_positions_in_world_frame)    # assemble_inner的位置
        assemble_outer_orientations = ObsTerm(func=mdp.assemble_outer_orientations_in_world_frame)    # assemble_inner的朝向
        # box_positions = ObsTerm(func=mdp.box_positions_in_world_frame)    # 箱子的位置
        # box_orientations = ObsTerm(func=mdp.box_orientations_in_world_frame)    # 箱子的朝向
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)     # 末端执行器的位置
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)      # 末端执行器的朝向
        gripper_pos = ObsTerm(func=mdp.gripper_pos)      # 夹爪的位置
     
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

        grasp = ObsTerm(
            func=mdp.object_grasped,    
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("assemble_outer"),
            },
        )
        # stack = ObsTerm(
        #     func=mdp.object_stacked,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "inner_object_cfg": SceneEntityCfg("assemble_inner"),
        #         "outer_object_cfg": SceneEntityCfg("assemble_outer"),
        #         "desk_cfg": SceneEntityCfg("desk"),
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

    # box_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("box")}
    # )

    success = DoneTerm(func=mdp.outer_stacked)


@configclass
class MoveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: RoomSetSceneCfg = RoomSetSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
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
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
