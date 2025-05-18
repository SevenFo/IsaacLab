# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
import math

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)


@configclass
class FrankaCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2

    action_space = gym.spaces.Dict(
        {
            "end_effector": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,),
                dtype=np.float32,
            ),
            "gripper": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    )

    observation_space = gym.spaces.Dict(
        {
            "end_effector": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,),
                dtype=np.float32,
            ),
            "gripper": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
            "front_rgb": gym.spaces.Box(
                low=0.0,
                high=255.0,
                shape=(480, 640, 3),
                dtype=np.uint8,
            ),
            "top_rgb": gym.spaces.Box(
                low=0.0,
                high=255.0,
                shape=(480, 640, 3),
                dtype=np.uint8,
            ),
            "front_depth": gym.spaces.Box(
                low=0.0,
                high=1.0e5,
                shape=(480, 640, 1),
                dtype=np.float32,
            ),
            "top_depth": gym.spaces.Box(
                low=0.0,
                high=1.0e5,
                shape=(480, 640, 1),
                dtype=np.float32,
            ),
        }
    )

    state_space = gym.spaces.Dict(
        {
            "joint_positions": gym.spaces.Box(
                low=-np.pi,
                high=np.pi,
                shape=(7,),
                dtype=np.float32,
            ),
        }
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/data/shared_folder/gripper/ur5_with_robotiq_gripper/ur5_with_gripper_single_articulation.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                articulation_enabled=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6256159257955498, -1.2960845679032278, 2.9002112950938577),
            joint_pos={
                ".*shoulder_pan.*": float(np.deg2rad(3.8)),  # 1.56,  # all HAA
                ".*shoulder_lift.*": float(
                    np.deg2rad(-118.1)
                ),  # 1.56,  # both front HFE
                ".*elbow_joint.*": float(np.deg2rad(45)),  # 1.56,  # both hind HFE
                ".*wrist_1.*": float(np.deg2rad(-84.2)),  # 1.56,  # both front KFE
                ".*wrist_2.*": float(np.deg2rad(272)),  # 1.56,  # both hind KFE
                ".*wrist_3.*": float(np.deg2rad(93)),  # 1.56,  # both front TFE
                # ".*outer_finger_joint": 0.0,  # both front TFE
                # ".*inner_finger_joint": 0.0,  # both hind TFE
                # '.*inner_finger_knuckle_joint': 0.0,  # both hind TFE
                # "finger_joint": 0.0,  # both front TFE
                # "right_outer_knuckle_joint":
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*", ".*wrist_.*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[
                    "finger_joint",
                    ".*outer_finger_joint",
                    ".*inner_finger_joint",
                    ".*inner_finger_knuckle_joint",
                    "right_outer_knuckle_joint",
                ],
                velocity_limit=1.0,  # 示例值
                effort_limit=10.0,  # 示例值，具体请查阅 Robotiq 夹爪规格
                stiffness=100.0,  # 示例值
                damping=20.0,  # 示例值 (例如，约 2*sqrt(stiffness))
            ),
        },
    )

    # ground plane
    terrain = AssetBaseCfg(
        prim_path="/World/LunarBase",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0416/ROOM_set_fix.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    gripper_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link/wrist_camera",
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
    top_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link/top_camera",
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

    dikconfig = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist_.*"],
        body_name="gripper",
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0)  # TODO
        ),
    )


class FrankaCabinetEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaCabinetEnvCfg

    def __init__(
        self, cfg: FrankaCabinetEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(
            env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device
        ):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 0
        ].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 1
        ].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        stage = get_current_stage()
        # hand_pose = get_env_local_pose(
        #     self.scene.env_origins[0],
        #     UsdGeom.Xformable(
        #         stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")
        #     ),
        #     self.device,
        # )
        # lfinger_pose = get_env_local_pose(
        #     self.scene.env_origins[0],
        #     UsdGeom.Xformable(
        #         stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")
        #     ),
        #     self.device,
        # )
        # rfinger_pose = get_env_local_pose(
        #     self.scene.env_origins[0],
        #     UsdGeom.Xformable(
        #         stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")
        #     ),
        #     self.device,
        # )

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.dik_action = DifferentialInverseKinematicsAction(self.cfg.dikconfig, self)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.dik_action.process_actions(actions)

    def _apply_action(self):
        self.dik_action.apply_actions()
        # self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        # self._compute_intermediate_values()

        return self._compute_rewards()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(
            joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # TODO
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        pass

    def _compute_rewards(
        self,
    ):
        self.extras["log"] = {}

        return torch.zeros(self.num_envs, device=self.device)

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return (
            global_franka_rot,
            global_franka_pos,
            global_drawer_rot,
            global_drawer_pos,
        )
