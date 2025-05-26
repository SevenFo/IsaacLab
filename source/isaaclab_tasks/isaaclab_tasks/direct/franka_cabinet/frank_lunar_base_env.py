# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from collections import OrderedDict
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    RigidObjectCfg,
    RigidObject,
    AssetBaseCfg,
)
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, FrameTransformer
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import (
    DifferentialInverseKinematicsActionCfg,
    BinaryJointPositionActionCfg,
    BinaryJointPositionAction,
)
from isaaclab_tasks.direct.franka_cabinet.lunar_base_env import (
    LunarBaseSceneCfg,
    LunarBaseEnv,
    LunarBaseEnvCfg,
)
from isaaclab_assets.robots.franka import (
    FRANKA_PANDA_HIGH_PD_CFG,
    FRANKA_PANDA_CFG,
)  # isort: skip
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.manager_based.manipulation.move import mdp
from isaaclab.managers import SceneEntityCfg


@configclass
class FrankLunarBaseSceneCfg(LunarBaseSceneCfg):
    """Configuration for a multi-object scene."""

    lunar_base = AssetBaseCfg(
        prim_path="/World/ROOM_set",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0, 0, 0], rot=[0, 0, 0, 0]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/Collected_ROOM_set/ROOM_set.no.ur5.box.desk_clean_version.usd"
        ),
    )

    # sensors
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        # visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
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

    # rigid object

    desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.73461, -2.82622, 1.82532), rot=(0, 0, 0, 1)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0403/Desk.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            semantic_tags=[("class", "desk")],
        ),
    )

    assemble_inner = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Assenmble_inner",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -2.0, 2.818), rot=(1, 0, 0, 0)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_assemble_inner/assemble_inner.usdc",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False
            ),
            semantic_tags=[("class", "Assenmble_inner")],
        ),
    )

    assemble_outer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Assenmble_outer",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -1.85, 2.885), rot=(0, 0, 0, 1)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_assemble_outer/assemble_outer.usdc",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False
            ),
            semantic_tags=[("class", "Assenmble_outer")],
        ),
    )

    def __post_init__(self):
        self.gripper_camera.prim_path = (
            "{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera"
        )
        self.gripper_camera.offset.pos = (0.05, 0.0, 0.0)
        self.front_camera.prim_path = (
            "{ENV_REGEX_NS}/Robot/panda_link0/top_camera"
        )
        self.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.robot.init_state.pos = (0.60000, -1.27000, 2.79427)
        self.robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)


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

    # 随机化箱子的位姿
    # randomize_box_positions = EventTerm(
    #     func=mdp.randomize_object_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (0.7, 1.0), "y": (-0.58, -0.47), "z": (2.52892, 2.52892), "yaw": (-0.5, 0.5)},
    #         "asset_cfgs": [SceneEntityCfg("box")]
    #     }
    # )

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
class FrankLunarBaseEnvCfg(LunarBaseEnvCfg):
    # env

    scene: FrankLunarBaseSceneCfg = FrankLunarBaseSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=True
    )
    events = EventCfg()

    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(7,),
        dtype=np.float32,
    )  # 6 for end effector, 1 for gripper

    def __post_init__(self):
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = (
            1024 * 1024 * 4
        )
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.dikconfig = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.107)
            ),
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
        )
        self.gripper_action_cfg = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )


class FrankLunarBaseEnv(LunarBaseEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankLunarBaseEnvCfg

    def __init__(
        self,
        cfg: FrankLunarBaseEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.grepper_action = BinaryJointPositionAction(
            self.cfg.gripper_action_cfg, self
        )

    def _setup_scene(self):
        self._robot = self.scene.articulations.get("robot")
        self._front_camera = self.scene.sensors.get("front_camera")
        self._gripper_camera = self.scene.sensors.get("gripper_camera")
        # clone and replicate # TODO still need?
        self._ee_frame = self.scene.sensors.get("ee_frame")
        self._assemble_inner = self.scene.rigid_objects.get("assemble_inner")
        self._assemble_outer = self.scene.rigid_objects.get("assemble_outer")
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions):
        env_action_dict = {
            "end_effector": actions[:, 0:6],
            "gripper": actions[:, 6:],
        }
        self.grepper_action.process_actions(env_action_dict["gripper"])
        self.dik_action.process_actions(env_action_dict["end_effector"])

    def _apply_action(self):
        self.dik_action.apply_actions()
        self.grepper_action.apply_actions()

    def step(self, action: torch.Tensor | np.ndarray):
        # step the simulation
        return super().step(action)

    def _get_observations(self):
        obs = super()._get_observations()

        def joint_pos_rel():
            return (
                self._robot.data.joint_pos[:, :]
                - self._robot.data.default_joint_pos[:, :]
            )

        def joint_vel_rel():
            return (
                self._robot.data.joint_vel[:, :]
                - self._robot.data.default_joint_vel[:, :]
            )

        def last_action():
            return self._last_action.clone()

        def ee_frame_pos():
            return (
                self._ee_frame.data.target_pos_w[:, 0, :]
                - self.scene.env_origins[:, 0:3]
            )

        def ee_frame_quat():
            return self._ee_frame.data.target_quat_w[:, 0, :]

        def gripper_pos():
            return torch.cat(
                (
                    self._robot.data.joint_pos[:, -1].clone().unsqueeze(1),
                    -1
                    * self._robot.data.joint_pos[:, -2].clone().unsqueeze(1),
                ),
                dim=1,
            )

        def object_obs():
            # TODO
            assert self._assemble_inner and self._assemble_outer
            assert type(self._ee_frame) is FrameTransformer

            assemble_inner: RigidObject = self._assemble_inner
            assemble_outer: RigidObject = self._assemble_outer
            ee_frame: FrameTransformer = self._ee_frame

            assemble_inner_pos_w = assemble_inner.data.root_pos_w
            assemble_inner_quat_w = assemble_inner.data.root_quat_w

            assemble_outer_pos_w = assemble_inner.data.root_pos_w
            assemble_inner_quat_w = assemble_outer.data.root_quat_w

            gripper_to_assemble_outer = (
                assemble_outer_pos_w - ee_frame.data.target_pos_w[:, 0, :]
            )
            assemble_outer_to_assemble_inner = (
                assemble_outer_pos_w - assemble_inner_pos_w
            )

            return torch.cat(
                (
                    assemble_inner_pos_w - self.scene.env_origins,
                    assemble_inner_quat_w,
                    assemble_outer_pos_w - self.scene.env_origins,
                    assemble_inner_quat_w,
                    gripper_to_assemble_outer,
                    assemble_outer_to_assemble_inner,
                ),
                dim=1,
            )

        mimic_policy_obs_dict = OrderedDict()
        mimic_policy_obs_dict["eef_pos"] = ee_frame_pos()
        mimic_policy_obs_dict["eef_quat"] = ee_frame_quat()
        mimic_policy_obs_dict["gripper_pos"] = gripper_pos()
        mimic_policy_obs_dict["object"] = object_obs()
        obs.update(
            {
                "policy": mimic_policy_obs_dict,
            }
        )
        return obs
