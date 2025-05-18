# lunar_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
import torch
import math
import numpy as np

import omni.usd
from pxr import Gf, Sdf

# Isaac Lab imports
from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# Gymnasium for RL interface
import gymnasium as gym
from gymnasium.spaces import Box


##
# Scene Configuration
##
@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    lunar_base = AssetBaseCfg(
        prim_path="/World/LunarBase",
        spawn=sim_utils.UsdFileCfg(
            # !! MODIFY THIS PATH TO YOUR ACTUAL USD FILE !!
            usd_path="/data/shared_folder/IssacAsserts/Projects/Collected_ROOM_set_fix_0416/ROOM_set_fix.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    # rigid object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
                    ),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
                    ),
                ),
                sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.0, 1.0), metallic=0.2
                    ),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),  # Increased iterations
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    # articulation
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # !! MODIFY THIS PATH TO YOUR ACTUAL USD FILE !!
            usd_path="/data/shared_folder/gripper/ur5_with_robotiq_gripper/ur5_with_gripper_single_articulation.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),  # Increased iterations
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),  # Adjusted base height if needed
            joint_pos={
                ".*shoulder_pan.*": 1.56,
                ".*shoulder_lift.*": 1.56,
                ".*elbow_joint.*": 1.56,
                ".*wrist_1.*": 1.56,
                ".*wrist_2.*": 1.56,
                ".*wrist_3.*": 1.56,
                # !! ADJUST GRIPPER JOINT NAMES AND INIT VALUES FOR YOUR MODEL !!
                "finger_joint": 0.0,  # Example for Robotiq 2F-85 like
                # "right_outer_knuckle_joint": 0.0, # ... and other gripper joints if any
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*shoulder_pan_joint",
                    ".*shoulder_lift_joint",
                    ".*elbow_joint",
                    ".*wrist_1_joint",
                    ".*wrist_2_joint",
                    ".*wrist_3_joint",
                ],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                # !! ADJUST GRIPPER JOINT NAMES FOR YOUR MODEL !!
                joint_names_expr=["finger_joint"],  # Example for Robotiq 2F-85 like
                velocity_limit=5.0,
                effort_limit=100.0,
                stiffness=400.0,
                damping=10.0,
            ),
        },
    )


##
# LunarBaseEnv Class for Reinforcement Learning
##
class LunarBaseEnv(gym.Env):
    """
    LunarBaseEnv for reinforcement learning, managing multiple simulation environments
    within Isaac Sim, conforming to the Gymnasium API.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}  # Approx

    def __init__(
        self,
        scene_cfg: MultiObjectSceneCfg = MultiObjectSceneCfg(),
        num_envs: int = 1,  # Default to 1 for easier debugging and server use
        headless: bool = True,
        device: str = "cuda:0",
        max_episode_length: int = 250,
        env_spacing: float = 3.0,
        replicate_physics: bool = True,
        # Any other AppLauncher args can be passed via app_launcher_args
        **app_launcher_args,
    ):
        """
        Args:
            scene_cfg: Configuration for the interactive scene.
            num_envs: Number of parallel environments.
            headless: Whether to run Isaac Sim in headless mode.
            device: PyTorch device for simulation and tensors.
            max_episode_length: Maximum number of steps per episode.
            env_spacing: Spacing between environments.
            replicate_physics: Whether to replicate physics across environments.
            **app_launcher_args: Additional arguments for AppLauncher.
        """
        self.device = torch.device(device)
        self.num_envs = num_envs
        self._max_episode_length = max_episode_length
        self._is_closed = False

        # Initialize AppLauncher
        parser = argparse.ArgumentParser()  # Dummy parser for AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        default_args_list = ["--device", device]  # Pass device to app launcher
        if headless:
            default_args_list.append("--headless")
        # Add any other custom app_launcher_args
        for k, v_arg in app_launcher_args.items():
            arg_name = f"--{k.replace('_', '-')}"
            if isinstance(v_arg, bool) and v_arg:
                default_args_list.append(arg_name)
            elif not isinstance(v_arg, bool):
                default_args_list.append(arg_name)
                default_args_list.append(str(v_arg))
        args_cli_for_app = parser.parse_args(default_args_list)

        self.app_launcher = AppLauncher(args_cli_for_app)
        self.simulation_app = self.app_launcher.app

        # Simulation context
        sim_cfg_args = {"device": device}
        if hasattr(scene_cfg, "dt") and scene_cfg.dt is not None:
            sim_cfg_args["dt"] = scene_cfg.dt
        else:  # Default dt if not in scene_cfg
            sim_cfg_args["dt"] = 1.0 / 60.0  # Example default
        self.sim_cfg = sim_utils.SimulationCfg(**sim_cfg_args)
        self.sim = SimulationContext(self.sim_cfg)

        # Update scene_cfg with runtime parameters
        scene_cfg.num_envs = self.num_envs
        scene_cfg.env_spacing = env_spacing
        scene_cfg.replicate_physics = replicate_physics
        self.scene_cfg_instance = scene_cfg

        # Scene
        self.scene = InteractiveScene(self.scene_cfg_instance)

        # Set main camera (optional, good for debugging if not headless)
        self.sim.set_camera_view([3.5, 0.0, 4.0], [0.0, 0.0, 1.0])  # Adjusted view

        # Randomize colors
        self._randomize_shape_color(self.scene_cfg_instance.object.prim_path)

        # Store references to assets
        self.robot: Articulation = self.scene["robot"]
        self.object: RigidObject = self.scene["object"]

        # --- Define action and observation spaces ---
        self.arm_actuator = self.robot.actuators["arm"]
        self.gripper_actuator = self.robot.actuators["gripper"]
        num_arm_actions = self.arm_actuator.num_actions
        num_gripper_actions = self.gripper_actuator.num_actions
        self.num_actions = num_arm_actions + num_gripper_actions

        action_low = -np.ones(self.num_actions, dtype=np.float32)
        action_high = np.ones(self.num_actions, dtype=np.float32)
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation: arm_pos, arm_vel, gripper_pos, gripper_vel, object_root_state (13)
        obs_dim = (num_arm_actions * 2) + (num_gripper_actions * 2) + 13
        obs_low = -np.inf * np.ones(obs_dim, dtype=np.float32)
        obs_high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Buffers
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int
        )
        self.sim_dt = self.sim.get_physics_dt()  # Get actual physics dt

        # Initial reset to prime the simulation
        self.sim.reset()  # This also calls scene.reset() internally after world.reset()
        print(
            f"[LunarBaseEnv]: Setup complete. Num Envs: {self.num_envs}, Device: {self.device}, Sim DT: {self.sim_dt:.6f}"
        )
        print(f"  Action Space Dim: {self.num_actions}, Obs Space Dim: {obs_dim}")
        print(
            f"  Arm Actuator Joints ({num_arm_actions}): {self.arm_actuator.joint_names}"
        )
        print(
            f"  Gripper Actuator Joints ({num_gripper_actions}): {self.gripper_actuator.joint_names}"
        )

    def _randomize_shape_color(self, prim_path_expr: str):
        """Randomize the color of the geometry."""
        stage = omni.usd.get_context().get_stage()
        prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
        with Sdf.ChangeBlock():
            for prim_path_str in prim_paths:
                prim = stage.GetPrimAtPath(prim_path_str)
                if not prim.IsValid():
                    continue

                # Try standard material diffuse color
                geom_prim = prim.GetChild("geometry")  # Common structure
                if geom_prim:
                    material_prim = geom_prim.GetChild("material")
                    if material_prim:
                        shader_prim = material_prim.GetChild("Shader")
                        if shader_prim:
                            color_attr = shader_prim.GetAttribute(
                                "inputs:diffuse_color"
                            )  # Note: sometimes diffuse_color
                            if not color_attr:
                                color_attr = shader_prim.GetAttribute(
                                    "inputs:diffuseColor"
                                )
                            if color_attr.IsValid():
                                color_attr.Set(
                                    Gf.Vec3f(
                                        random.random(),
                                        random.random(),
                                        random.random(),
                                    )
                                )
                                continue  # Found and set

                # Fallback to primvars:displayColor if above fails
                display_color_attr = prim.GetAttribute("primvars:displayColor")
                if (
                    not display_color_attr
                ):  # Create if it doesn't exist for some geom types
                    # Need to know the schema of the geometry to create it correctly.
                    # For simple shapes from sim_utils, they often have a 'geometry' child.
                    geom_child = prim.GetChild("geometry")
                    if geom_child:  # Check if 'geometry' child exists
                        # Check for specific geometry type like Sphere, Cube, Cone to target correctly
                        # This part is tricky without knowing the exact prim structure of spawned assets
                        # Let's assume a direct 'primvars:displayColor' on the 'geometry' child if it's a GPrim
                        target_prim_for_display_color = geom_child
                        # Try to find a GPrim child like 'Cube', 'Sphere' under 'geometry'
                        for child_name in [
                            "Cube",
                            "Sphere",
                            "Cone",
                            "Cylinder",
                        ]:  # Common gprim names
                            potential_gprim = geom_child.GetChild(child_name)
                            if potential_gprim and potential_gprim.IsA(
                                Sdf.Schema.GetPrimDefinition(
                                    "UsdGeomGprim"
                                ).GetPrimType()
                            ):
                                target_prim_for_display_color = potential_gprim
                                break

                        display_color_attr = target_prim_for_display_color.GetAttribute(
                            "primvars:displayColor"
                        )
                        if not display_color_attr:
                            display_color_attr = (
                                target_prim_for_display_color.CreateAttribute(
                                    "primvars:displayColor",
                                    Sdf.ValueTypeNames.Color3fArray,
                                    custom=False,
                                )
                            )

                if display_color_attr and display_color_attr.IsValid():
                    display_color_attr.Set(
                        [Gf.Vec3f(random.random(), random.random(), random.random())]
                    )
                # else:
                #     print(f"Warning: Could not find or create color attribute for {prim_path_str}")

    def _get_observations(self) -> torch.Tensor:
        """Gathers observations from the environment."""
        # Ensure buffers are up-to-date from simulation
        # self.scene.update(self.sim_dt) is called by self.sim.step() -> world.render() -> scene.update()
        # or explicitly if needed before a sim step.
        # For getting observations, data should be from the *last* sim step.

        # Robot arm data
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_actuator.joint_indices]
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_actuator.joint_indices]

        # Robot gripper data
        gripper_joint_pos = self.robot.data.joint_pos[
            :, self.gripper_actuator.joint_indices
        ]
        gripper_joint_vel = self.robot.data.joint_vel[
            :, self.gripper_actuator.joint_indices
        ]

        # Object data (pos_w, quat_w, lin_vel_w, ang_vel_w) -> 13 elements
        # Using root_state_w for world frame data, consistent with most RL practices
        object_root_state = (
            self.object.data.root_state_w.clone()
        )  # shape (num_envs, 13)

        obs = torch.cat(
            [
                arm_joint_pos,
                arm_joint_vel,
                gripper_joint_pos,
                gripper_joint_vel,
                object_root_state,
            ],
            dim=-1,
        )
        return obs.to(self.device)

    def _compute_rewards(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes rewards for the current state."""
        # Object Z position (world frame) is the 3rd element (index 2) of object_root_state
        # object_root_state is the last 13 elements of obs.
        object_z_pos = self.object.data.root_pos_w[
            :, 2
        ]  # More direct: from self.object.data

        # Example: Reward for keeping the object above a certain height
        target_height = 1.0  # Target height in meters
        height_reward_scale = 1.0
        height_reward = height_reward_scale * torch.exp(
            -5.0 * ((object_z_pos - target_height) ** 2)
        )  # Gaussian-like

        # Penalty for large arm joint velocities (effort)
        effort_penalty_scale = -0.005
        arm_vel_penalty = effort_penalty_scale * torch.sum(
            self.robot.data.joint_vel[:, self.arm_actuator.joint_indices] ** 2, dim=-1
        )

        rewards = height_reward + arm_vel_penalty
        return rewards.to(self.device)

    def _check_terminated_truncated(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Checks for episode termination or truncation."""
        # Terminated: Task-specific conditions (e.g., object fell, goal reached)
        object_z_pos = self.object.data.root_pos_w[:, 2]
        # Terminate if object falls below ground (e.g., 0.05m, assuming ground is at 0)
        # or if it goes too high (e.g., unstable)
        terminated = (object_z_pos < 0.05) | (object_z_pos > 3.0)

        # Truncated: Episode length limit
        self.episode_length_buf += 1  # Increment for all active envs
        truncated = self.episode_length_buf >= self._max_episode_length

        return terminated.to(self.device), truncated.to(self.device)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Resets the environment to an initial state."""
        if self._is_closed:
            raise RuntimeError("Cannot reset a closed environment.")
        super().reset(seed=seed)  # For RNG if needed by Gymnasium's utils

        # Reset object state
        obj_default_root_state = self.object.data.default_root_state.clone()
        # Add some randomization to initial object XY position for varied starts
        xy_noise = (
            torch.rand((self.num_envs, 2), device=self.device) - 0.5
        ) * 0.4  # Small XY noise [-0.2, 0.2]
        obj_default_root_state[:, 0:2] += xy_noise
        obj_default_root_state[:, :3] += self.scene.env_origins  # Add env origins
        self.object.write_root_state_to_sim(
            obj_default_root_state
        )  # Writes pose and velocity

        # Reset robot state (root and joints)
        robot_default_root_state = self.robot.data.default_root_state.clone()
        robot_default_root_state[:, :3] += self.scene.env_origins
        self.robot.write_root_state_to_sim(
            robot_default_root_state
        )  # Writes pose and velocity

        joint_pos, joint_vel = (
            self.robot.data.default_joint_pos.clone(),
            self.robot.data.default_joint_vel.clone(),
        )
        # Optional: Add noise to initial joint positions for exploration
        # pos_noise_joints = (torch.rand_like(joint_pos) - 0.5) * 0.1 # Small noise
        # joint_pos += pos_noise_joints
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Reset scene (this calls managers, event proc, etc. AND updates buffers)
        # Crucially, scene.reset() will call world.reset() which then calls
        # the reset_idx_fn of various managers.
        self.scene.reset()  # This should handle updating data buffers correctly after reset.

        # Reset episode length buffer for all envs
        self.episode_length_buf[:] = 0

        # Get initial observations AFTER all resets and scene updates
        obs = self._get_observations()
        info = {}  # Standard to return an empty info dict for Gymnasium

        # Randomize colors again on reset if desired
        # self._randomize_shape_color(self.scene_cfg_instance.object.prim_path)

        return obs, info

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Applies an action and steps the simulation."""
        if self._is_closed:
            raise RuntimeError("Cannot step a closed environment.")

        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        if (
            actions.ndim == 1
        ):  # If a single action (num_actions,) is passed for num_envs=1
            actions = actions.unsqueeze(0)  # Make it (1, num_actions)

        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Actions shape mismatch. Expected ({self.num_envs}, {self.num_actions}), got {actions.shape}"
            )

        # Actions are expected to be normalized [-1, 1]
        # Split actions for arm and gripper
        arm_actions_normalized = actions[:, : self.arm_actuator.num_actions]
        gripper_actions_normalized = actions[:, self.arm_actuator.num_actions :]

        # Convert normalized actions to target joint positions
        # For ImplicitActuatorCfg, command is target position.
        # We can implement delta-पोजिशन control or absolute position control.
        # Let's try absolute position control, mapping [-1,1] to a reasonable joint range.
        # Get current joint positions to map relative actions or to define target ranges
        # current_arm_pos = self.robot.data.joint_pos[:, self.arm_actuator.joint_indices]
        # current_gripper_pos = self.robot.data.joint_pos[:, self.gripper_actuator.joint_indices]

        # Example: Scale arm actions to +/- PI/2 radians from current or default
        # This is a simplification. A better way is to use joint limits.
        # arm_joint_mid_points = self.robot.data.default_joint_pos[:, self.arm_actuator.joint_indices]
        # arm_action_scale = math.pi / 2.0
        # arm_target_pos = arm_joint_mid_points + arm_actions_normalized * arm_action_scale

        # Simpler: actions are target joint positions directly, assuming agent learns the scale
        # or actions are delta positions. Let's assume actions are scaled deltas.
        action_scale_arm = 0.1  # Max change per step (radians)
        action_scale_gripper = 0.05  # Max change per step (radians for gripper)

        arm_target_pos = (
            self.robot.data.joint_pos[:, self.arm_actuator.joint_indices]
            + arm_actions_normalized * action_scale_arm
        )

        # For gripper, map [-1, 1] to [0, 0.8] (approx Robotiq 2F-85 range, 0=open)
        # Let's use delta for gripper as well for consistency
        gripper_target_pos = (
            self.robot.data.joint_pos[:, self.gripper_actuator.joint_indices]
            + gripper_actions_normalized * action_scale_gripper
        )
        # Clamp gripper targets to a valid range, e.g., [0, 0.8]
        gripper_target_pos = torch.clamp(
            gripper_target_pos, 0.0, 0.83
        )  # 0.83 approx fully closed for Robotiq

        self.arm_actuator.set_command(arm_target_pos)
        if self.gripper_actuator.num_actions > 0:
            self.gripper_actuator.set_command(gripper_target_pos)

        # Perform simulation step
        self.scene.write_data_to_sim()  # Write actuator commands
        self.sim.step(
            render=not self.app_launcher.args.headless
        )  # simulation_app.update handles render

        # Get results (obs, reward, done)
        # sim.step() calls scene.update() which updates all data buffers
        obs = self._get_observations()
        rewards = self._compute_rewards(obs)
        terminated, truncated = self._check_terminated_truncated()
        info = {}  # Standard to return an empty info dict

        # Handle auto-reset for environments that are done
        # This is typical in vectorized environments for continuous training
        # If an env is 'terminated' or 'truncated', its episode_length_buf is reset below
        # and its state is reset in _reset_done_envs.
        # The returned obs, rewards, terminated, truncated should be for the *current* step,
        # before any auto-reset that prepares for the *next* step.
        # RL libraries often handle this: if done, they use the current info and then call reset.
        # However, for Isaac Lab's typical VecEnv pattern, we reset internally.

        done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(done_env_ids) > 0:
            self._reset_done_envs(done_env_ids)
            # After reset, the obs for these envs will be the new initial obs.
            # Some RL algos might prefer the obs *before* reset for the done step.
            # This implementation (like IsaacGymEnvs) provides the obs *after* reset for done envs.
            # If you need pre-reset obs, you'd need to cache it or adjust logic.
            # For now, this is consistent with many Isaac Lab examples.
            new_obs_for_done_envs = self._get_observations_for_envs(done_env_ids)
            obs[done_env_ids] = new_obs_for_done_envs

        return obs, rewards, terminated, truncated, info

    def _get_observations_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Gathers observations for specified environment IDs."""
        if env_ids.numel() == 0:
            return torch.empty((0, self.observation_space.shape[0]), device=self.device)

        arm_joint_pos = self.robot.data.joint_pos[env_ids][
            :, self.arm_actuator.joint_indices
        ]
        arm_joint_vel = self.robot.data.joint_vel[env_ids][
            :, self.arm_actuator.joint_indices
        ]
        gripper_joint_pos = self.robot.data.joint_pos[env_ids][
            :, self.gripper_actuator.joint_indices
        ]
        gripper_joint_vel = self.robot.data.joint_vel[env_ids][
            :, self.gripper_actuator.joint_indices
        ]
        object_root_state = self.object.data.root_state_w[env_ids].clone()

        obs_subset = torch.cat(
            [
                arm_joint_pos,
                arm_joint_vel,
                gripper_joint_pos,
                gripper_joint_vel,
                object_root_state,
            ],
            dim=-1,
        )
        return obs_subset.to(self.device)

    def _reset_done_envs(self, env_ids: torch.Tensor):
        """Resets specified environments that are 'done'."""
        if env_ids.numel() == 0:  # Check if tensor is empty
            return

        num_done_envs = len(env_ids)

        # Object reset for specific envs
        obj_default_root_state_subset = self.object.data.default_root_state[
            env_ids
        ].clone()
        xy_noise = (torch.rand((num_done_envs, 2), device=self.device) - 0.5) * 0.4
        obj_default_root_state_subset[:, 0:2] += xy_noise
        obj_default_root_state_subset[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(
            obj_default_root_state_subset, env_ids=env_ids
        )

        # Robot reset for specific envs
        robot_default_root_state_subset = self.robot.data.default_root_state[
            env_ids
        ].clone()
        robot_default_root_state_subset[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(
            robot_default_root_state_subset, env_ids=env_ids
        )

        joint_pos_subset = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel_subset = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(
            joint_pos_subset, joint_vel_subset, env_ids=env_ids
        )

        # Reset scene managers for these envs
        # This will call update internally for these env_ids
        self.scene.reset(env_ids=env_ids)

        # Reset episode length buffer for these envs
        self.episode_length_buf[env_ids] = 0

    def render(self, mode="human"):  # Gymnasium render modes
        """Isaac Sim handles rendering automatically if not headless."""
        if self._is_closed:
            # print("Warning: Attempting to render a closed environment.")
            return None
        # In Isaac Sim, rendering is continuous if not headless, driven by sim.step() or app.update().
        # This method is more for API compatibility.
        if mode == "rgb_array":
            # This would require setting up a camera and capturing an image using isaaclab.sim.RenderProduct
            # For now, returning None or raising NotImplementedError.
            # print("Warning: 'rgb_array' rendering not fully implemented yet for LunarBaseEnv.")
            return None
        elif mode == "human":
            # Simulation is already rendering if not headless and sim.step() is called.
            pass
        else:
            super().render(mode=mode)  # Handles unknown modes

    def close(self):
        """Cleans up resources and closes the simulation app."""
        if not self._is_closed:
            print("[LunarBaseEnv]: Closing simulation environment...")
            # It's generally better to let AppLauncher handle the full app shutdown
            # self.sim.stop() # This just stops physics stepping
            if self.app_launcher:
                self.app_launcher.close()  # This closes the entire Isaac Sim application
            self._is_closed = True
            print("[LunarBaseEnv]: Environment closed.")
        # else:
        # print("[LunarBaseEnv]: Environment already closed.")

    @property
    def unwrapped(self) -> gym.Env:
        """Returns the base environment, bypassing any wrappers."""
        return self
