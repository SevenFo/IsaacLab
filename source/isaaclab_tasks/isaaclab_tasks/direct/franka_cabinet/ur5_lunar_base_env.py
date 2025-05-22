# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
import math
from collections import OrderedDict

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)


@configclass
class LunarBaseScene(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=None),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0, color=(0.75, 0.75, 0.75)
        ),
    )

    lunar_base = AssetBaseCfg(
        prim_path="/World/LunarBase",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0416/ROOM_set_fix.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Isaac Sim Property Panel 中的 Transform.Orien 属性欧拉角采用的模式为 X-Y-Z-intrinsic
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
    front_camera = TiledCameraCfg(
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

    # # rigid object
    # object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Object",
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=[
    #             sim_utils.ConeCfg(
    #                 radius=0.3,
    #                 height=0.6,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(
    #                     diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
    #                 ),
    #             ),
    #             sim_utils.CuboidCfg(
    #                 size=(0.3, 0.3, 0.3),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(
    #                     diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
    #                 ),
    #             ),
    #             sim_utils.SphereCfg(
    #                 radius=0.3,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(
    #                     diffuse_color=(0.0, 0.0, 1.0), metallic=0.2
    #                 ),
    #             ),
    #         ],
    #         random_choice=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=4,
    #             solver_velocity_iteration_count=0,
    #             disable_gravity=False,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    # )

    # articulation
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
                ".*elbow_joint.*": float(
                    np.deg2rad(45)
                ),  # 1.56,  # both hind HFE
                ".*wrist_1.*": float(
                    np.deg2rad(-84.2)
                ),  # 1.56,  # both front KFE
                ".*wrist_2.*": float(
                    np.deg2rad(272)
                ),  # 1.56,  # both hind KFE
                ".*wrist_3.*": float(
                    np.deg2rad(93)
                ),  # 1.56,  # both front TFE
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


@configclass
class LunarBaseEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2

    action_space = gym.spaces.Dict(
        {
            # delat pose: pos, rot, axis-angle
            "end_effector": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(6,),
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
            "policy": gym.spaces.Dict(
                {
                    "end_effector": gym.spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(6,),
                        dtype=np.float32,
                    ),
                    "gripper": gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(2,),
                        dtype=np.float32,
                    ),
                }
            ),
            "rgb": gym.spaces.Dict(
                {
                    "gripper_rgb": gym.spaces.Box(
                        low=0.0,
                        high=255.0,
                        shape=(480, 640, 3),
                        dtype=np.uint8,
                    ),
                    "front_rgb": gym.spaces.Box(
                        low=0.0,
                        high=255.0,
                        shape=(480, 640, 3),
                        dtype=np.uint8,
                    ),
                }
            ),
            "depth": gym.spaces.Dict(
                {
                    "gripper_depth": gym.spaces.Box(
                        low=0.0,
                        high=1.0e5,
                        shape=(480, 640, 1),
                        dtype=np.float32,
                    ),
                    "front_depth": gym.spaces.Box(
                        low=0.0,
                        high=1.0e5,
                        shape=(480, 640, 1),
                        dtype=np.float32,
                    ),
                }
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
        dt=1 / 10,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        render=sim_utils.RenderCfg(
            antialiasing_mode="DLAA",
            enable_dl_denoiser=True,
            samples_per_pixel=2,
            enable_reflections=True,
            enable_global_illumination=True,
        ),
    )
    # scene
    scene: InteractiveSceneCfg = LunarBaseScene(
        num_envs=1, env_spacing=3.0, replicate_physics=True
    )

    dikconfig = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist_.*"],
        body_name="base_link_gripper",
        scale=0.5,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0)  # TODO
        ),
    )

    dof_velocity_scale = 0.1


class LunarBaseEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: LunarBaseEnvCfg

    def __init__(
        self, cfg: LunarBaseEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(
            env_pos: torch.Tensor,
            xformable: UsdGeom.Xformable,
            device: torch.device,
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

        self.robot_dof_speed_scales = torch.ones_like(
            self.robot_dof_lower_limits
        )

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

        self.robot_grasp_rot = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.robot_grasp_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        self.dik_action = DifferentialInverseKinematicsAction(
            self.cfg.dikconfig, self
        )

    def _setup_scene(self):
        self._robot = self.scene.articulations.get("robot")
        self._front_camera = self.scene.sensors.get("front_camera")
        self._gripper_camera = self.scene.sensors.get("gripper_camera")
        # clone and replicate # TODO still need?
        self.scene.clone_environments(copy_from_source=False)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        # tensor([[0.0724, 0.1214, 0.9045]], device='cuda:0')
        # tensor([[-0.9691, -0.0589, -0.2302, -0.0654]], device='cuda:0')
        self.dik_action.process_actions(actions)

    def _apply_action(self):
        self.dik_action.apply_actions()
        # self._robot.set_joint_position_target(self.robot_dof_targets)

    def step(self, action: dict | None):
        if action is not None:
            end_effector_action = action["end_effector"]
            gripper_action = action["gripper"]
            if isinstance(end_effector_action, np.ndarray):
                end_effector_action = torch.from_numpy(end_effector_action)
            return super().step(end_effector_action)
        else:
            return self.step_without_applying_action()

    def step_without_applying_action(self) -> VecEnvStepReturn:
        """执行环境的一个步骤而不应用任何动作。

        与 `step()` 类似，但跳过动作处理，仅推进物理状态并更新观测值、奖励等。
        """

        # 检查是否需要在物理循环中渲染
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # 执行物理步进循环
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # 不应用动作，但需更新场景数据到模拟（如传感器、状态等）
            # self.scene.write_data_to_sim()
            # 模拟一步
            self.sim.step(render=False)
            # 渲染检查
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # 更新场景数据（如传感器、状态等）
            self.scene.update(dt=self.physics_dt)

        # 更新环境计数器
        self.episode_length_buf += 1  # 当前episode步数（每个env）
        self.common_step_counter += 1  # 总步数（所有env共用）

        # 获取done标志和奖励
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # 重置终止的环境
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # 更新关节运动学
            self.scene.write_data_to_sim()
            self.sim.forward()
            # 如果有传感器，重新渲染
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # 应用间隔事件
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 获取观测值
        self.obs_buf = self._get_observations()

        # 添加观测噪声
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(
                self.obs_buf["policy"]
            )

        # 返回观测、奖励、重置标志和额外信息
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        # self._compute_intermediate_values()

        return self._compute_rewards()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[
            env_ids
        ] + sample_uniform(
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
        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids
        )

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # Camera data
        gripper_camera_rgb = self._gripper_camera.data.output["rgb"]
        front_camera_rgb = self._front_camera.data.output["rgb"]
        gripper_camera_depth = self._gripper_camera.data.output[
            "distance_to_image_plane"
        ]
        front_camera_depth = self._front_camera.data.output[
            "distance_to_image_plane"
        ]

        # For non-policy observations (e.g., for human rendering, separate critics)
        full_rgb_obs = {
            "gripper_rgb": gripper_camera_rgb,
            "front_rgb": front_camera_rgb,
        }
        full_depth_obs = {
            "gripper_depth": gripper_camera_depth,
            "front_depth": front_camera_depth,  # Original key was front_camera_depth
        }

        # --- Prepare policy observations (MUST BE A DICTIONARY) ---
        policy_obs_dict = OrderedDict()

        # Arm joint positions and velocities
        # Assuming first N joints are arm, next M are gripper
        num_arm_joints = 6  # For UR5 (shoulder_pan to wrist_3)
        # Update this based on your self._robot.num_joints and actuator groups

        # Get all joint positions and velocities
        all_joint_pos = self._robot.data.joint_pos
        all_joint_vel = self._robot.data.joint_vel

        # Arm joints
        arm_joint_pos = all_joint_pos[:, :num_arm_joints]
        arm_joint_vel = all_joint_vel[:, :num_arm_joints]

        # Gripper joints (example, assuming they are after arm joints)
        # You need to know the indices of your gripper joints
        # Example: if gripper joints are indices 6 through 11 (for a 6-DOF gripper)
        gripper_joint_indices_start = num_arm_joints
        # num_gripper_joints = self._robot.num_joints - num_arm_joints # if all remaining are gripper
        # For your 'gripper' actuator group which has 8 joint_names_expr patterns:
        # "finger_joint", ".*outer_finger_joint"(2), ".*inner_finger_joint"(2), ".*inner_finger_knuckle_joint"(2), "right_outer_knuckle_joint"
        # The proactive joint id is finger_joint (idx:6), remained are passive joints
        # For simplicity, let's hardcode an example. This needs to be correct for your robot.
        gripper_joint_indices = torch.tensor(
            [6], device=self.device
        )  # proactive gripper joint idx is 6
        if self._robot.num_joints > gripper_joint_indices.max():  # Basic check
            gripper_qpos = all_joint_pos[:, gripper_joint_indices]
            gripper_qvel = all_joint_vel[:, gripper_joint_indices]
        else:  # Fallback if indices are wrong, to prevent crash. Log error ideally.
            print(
                "WARNING: Gripper joint indices seem out of bounds. Using zeros for gripper obs."
            )
            num_gripper_dof_example = 1  
            gripper_qpos = torch.zeros(
                (self.num_envs, num_gripper_dof_example), device=self.device
            )
            gripper_qvel = torch.zeros(
                (self.num_envs, num_gripper_dof_example), device=self.device
            )

        # Scale them (example, adjust scaling as needed or if policy handles unscaled)
        # Using -1 to 1 scaling based on limits:
        arm_dof_pos_scaled = (
            2.0
            * (arm_joint_pos - self.robot_dof_lower_limits[:num_arm_joints])
            / (
                self.robot_dof_upper_limits[:num_arm_joints]
                - self.robot_dof_lower_limits[:num_arm_joints]
            )
            - 1.0
        )
        arm_dof_vel_scaled = (
            arm_joint_vel * self.cfg.dof_velocity_scale
        )  # Or another scaling factor

        # Gripper scaling (assuming limits for gripper joints are also in robot_dof_lower/upper_limits)
        gripper_dof_pos_scaled = (
            2.0
            * (
                gripper_qpos
                - self.robot_dof_lower_limits[gripper_joint_indices]
            )
            / (
                self.robot_dof_upper_limits[gripper_joint_indices]
                - self.robot_dof_lower_limits[gripper_joint_indices]
            )
            - 1.0
        )
        gripper_dof_vel_scaled = (
            gripper_qvel * self.cfg.dof_velocity_scale
        )  # Adjust scaling factor if different

        # Populate policy_obs_dict with keys your Robomimic model expects
        # COMMON KEYS (ADJUST TO YOUR TRAINING CONFIG):
        policy_obs_dict["robot0_joint_pos"] = (
            arm_dof_pos_scaled  # Or "robot_qpos", "arm_qpos" etc.
        )
        policy_obs_dict["robot0_joint_vel"] = (
            arm_dof_vel_scaled  # Or "robot_qvel", "arm_qvel"
        )
        policy_obs_dict["robot0_gripper_qpos"] = (
            gripper_dof_pos_scaled  # Or "gripper_qpos"
        )
        policy_obs_dict["robot0_gripper_qvel"] = (
            gripper_dof_vel_scaled  # Or "gripper_qvel"
        )

        # Example: End-effector pose (if used in training)
        # self.dik_action.target_ee_pose_w is (N, 13) [pos, quat, vel, ang_vel]
        # You might need to get the current EE pose, not target.
        gripper_body_idx = 6 # TODO
        eef_body_idx = self.dik_action._body_idx # TODO
        current_ee_pose_w = self._robot.data.body_state_w[:, eef_body_idx, :7] # pos, quat (w,x,y,z)
        policy_obs_dict["robot0_eef_pos"] = current_ee_pose_w[:, :3]
        policy_obs_dict["robot0_eef_quat"] = current_ee_pose_w[:, 3:7]
        current_gripper_pos = self._robot.data.body_pos_w[:, gripper_body_idx,:]
        policy_obs_dict["robot0_gripper_pos"] = current_gripper_pos

        # Example: Object poses (if used in training)
        if hasattr(self, "_object_to_assemble"): # Assuming you have objects
           object_pose = self._object_to_assemble.data.root_state_w[:, :7]
           policy_obs_dict["object_pose"] = object_pose

        # Example: Image observations (if used in training AND configured in Robomimic)
        # The keys here MUST match Robomimic config (e.g., config.observation.modalities.rgb.obs_keys)
        policy_obs_dict["agentview_image"] = front_camera_rgb # if "agentview_image" was a key
        policy_obs_dict["robot0_eye_in_hand_image"] = gripper_camera_rgb # if "robot0_eye_in_hand_image" was a key

        # Clamp observations if necessary (often done by policy wrapper or normalization)
        for k_obs, v_obs in policy_obs_dict.items():
            policy_obs_dict[k_obs] = torch.clamp(
                v_obs, -5.0, 5.0
            )  # Generic clamp, adjust per obs type

        return {
            "policy": policy_obs_dict,  # This is the dictionary the Robomimic policy will receive
            "rgb": full_rgb_obs,
            "depth": full_depth_obs,
        }

    # auxiliary methods

    def _compute_intermediate_values(
        self, env_ids: torch.Tensor | None = None
    ):
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
            drawer_rot,
            drawer_pos,
            drawer_local_grasp_rot,
            drawer_local_grasp_pos,
        )

        return (
            global_franka_rot,
            global_franka_pos,
            global_drawer_rot,
            global_drawer_pos,
        )
