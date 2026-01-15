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
import isaaclab.sim as sim_utils

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
        eef_quat_gripper = ObsTerm(func=mdp.ee_frame_quat_gripper)
        camera_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("topcamera"), "normalize": False},
        )  # RGB 相机图像
        camera_side = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("sidecamera"), "normalize": False},
        )  # RGB 相机图像
        camera_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wristcamera"), "normalize": False},
        )  # RGB 相机图像
        inspector_side = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("inspector_side"), "normalize": False},
        )  # RGB 相机图像
        inspector_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("inspector_top"), "normalize": False},
        )  # RGB 相机图像
        camera_left = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("leftcamera"), "normalize": False},
        )  # RGB 相机图像
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

        # camera_rgbd = ObsTerm(func=mdp.camera_rgbd)     # RGBD 相机图像

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # subtask_1
        lift = ObsTerm(
            func=mdp.grasp_box,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "box_cfg": SceneEntityCfg("box"),
            },
        )

        # subtask_2
        rot = ObsTerm(
            func=mdp.rot_box,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "box_cfg": SceneEntityCfg("box"),
                "desk_cfg": SceneEntityCfg("desk"),
            },
        )

        # subtask_3
        put = ObsTerm(func=mdp.put_box)
        press = ObsTerm(
            func=mdp.press_button,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_sensor"),
            },
        )
        grasp = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("spanner"),
            },
        )

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
    success = DoneTerm(func=mdp.lift_eef)


@configclass
class MoveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: RoomSetSceneCfg = RoomSetSceneCfg(
        num_envs=1, env_spacing=0, replicate_physics=False
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
        self.decimation = 5  # should be in [10,5]
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        # self.sim.use_fabric = False
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render = sim_utils.RenderCfg(
            enable_ambient_occlusion=True,
            enable_dl_denoiser=True,
            enable_dlssg=True,
            enable_reflections=True,
            enable_direct_lighting=True,
            antialiasing_mode="DLSS",
            dlss_mode=2,
            enable_translucency=True,
            enable_global_illumination=True,
            enable_shadows=True,
        )
        self.sim.render.enable_dl_denoiser = (
            True  # 开启 AI 降噪（最关键，能瞬间消除大部分颗粒噪点）
        )
        # --- 采样与光照质量 ---
        self.sim.render.samples_per_pixel = (
            4  # 增加采样数（默认 1 太低，设为 4 或更高可以大幅提升光影细腻度）
        )
        self.sim.render.enable_direct_lighting = True
        self.sim.render.enable_shadows = True
        # --- 抗锯齿与图像重建 ---
        self.sim.render.antialiasing_mode = "DLAA"  # 使用 DLAA（基于 AI 的抗锯齿，不缩减分辨率，画质比 DLSS 更清晰、无抖动）
        # --- 间接光照处理（权衡项） ---
        self.sim.render.enable_reflections = (
            True  # 开启反射（开启会增加噪点压力，但配合 Denoiser 效果较好）
        )
        self.sim.render.enable_global_illumination = (
            True  # 开启全局光照（这是最大的噪点来源，开启后必须确保 Denoiser 为 True）
        )
        self.sim.render.enable_ambient_occlusion = True  # 环境光遮蔽，增加细节阴影
