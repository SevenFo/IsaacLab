# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn multiple objects in multiple environments.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/multi_asset.py --num_envs 2048

"""

from __future__ import annotations


import random
import math
import torch
import asyncio  # 用于UI更新
import logging
import numpy as np
import os

logging.basicConfig(
    level=logging.INFO, format=f"{__file__}: %(levelname)s: %(message)s"
)

import omni.usd
import omni.log
from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz

# from custom_ui import IntegratedCustomPanel
from utils import save_images_grid
##
# Randomization events.
##


def randomize_shape_color(prim_path_expr: str):
    """Randomize the color of the geometry."""
    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = prim_spec.GetAttributeAtPath(
                prim_path + "/geometry/material/Shader.inputs:diffuseColor"
            )
            color_spec.default = Gf.Vec3f(
                random.random(), random.random(), random.random()
            )


##
# Scene Configuration
##


@configclass
class LunarBaseEnvConfig(DirectRLEnvCfg):
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


import gymnasium as gym


class LunarBaseEnv(gym.Env):
    def __init__(
        self, scene_cfg: LunarBaseEnvConfig, simulation_context: SimulationContext
    ):
        super().__init__()
        self.scene_cfg = scene_cfg
        self.sim = simulation_context
        self.sim.set_camera_view(
            (2.08, -1.12, 3.95),
            (0.6256159257955498, -1.2960845679032278, 2.9002112950938577),
        )

        with Timer("[INFO] Time to create scene: "):
            self.scene = InteractiveScene(self.scene_cfg)

        with Timer("[INFO] Time to randomize scene: "):
            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            # randomize_shape_color(scene_cfg.object.prim_path)
            pass

        self.robot: Articulation = self.scene["robot"]
        self.sim_dt = self.sim.get_physics_dt()
        self.step_count = 0

        # Play the simulator
        self.sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")

        self.arm_actuator: ImplicitActuatorCfg = self.robot.actuators["arm"]
        self.gripper_actuator: ImplicitActuatorCfg = self.robot.actuators["gripper"]

        self.num_env = self.scene_cfg.num_envs
        self.action_space = gym.spaces.Dict(
            {
                "end_effector": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_env, 7),
                    dtype=np.float32,
                ),
                "gripper": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_env, 1),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "end_effector": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_env, 7),
                    dtype=np.float32,
                ),
                "gripper": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_env, 1),
                    dtype=np.float32,
                ),
                "front_rgb": gym.spaces.Box(
                    low=0.0,
                    high=255.0,
                    shape=(self.num_env, 480, 640, 3),
                    dtype=np.uint8,
                ),
                "top_rgb": gym.spaces.Box(
                    low=0.0,
                    high=255.0,
                    shape=(self.num_env, 480, 640, 3),
                    dtype=np.uint8,
                ),
                "front_depth": gym.spaces.Box(
                    low=0.0,
                    high=1.0e5,
                    shape=(self.num_env, 480, 640, 1),
                    dtype=np.float32,
                ),
                "top_depth": gym.spaces.Box(
                    low=0.0,
                    high=1.0e5,
                    shape=(self.num_env, 480, 640, 1),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self):
        with Timer("[INFO] Time to reset scene: "):
            self.step_count = 0
            robotroot_state = self.robot.data.default_root_state.clone()
            robot_default_joint_pos = self.robot.data.default_joint_pos.clone()
            robotroot_state[:, :3] += self.scene.env_origins
            self.robot.write_root_pose_to_sim(robotroot_state[:, :7])
            self.robot.write_root_velocity_to_sim(robotroot_state[:, 7:])
            # -- joint state
            joint_pos, joint_vel = (
                self.robot.data.default_joint_pos.clone(),
                self.robot.data.default_joint_vel.clone(),
            )
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

            self.scene.reset()
            print("[INFO]: Resetting scene state...")


##
# Simulation Loop
##


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    global g_custom_ui_panel  # 引用全局UI实例
    # Extract scene entities
    # note: we only do this here for readability.
    # rigid_object: RigidObject = scene["object"]
    # Define simulation stepping
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    # root_state = rigid_object.data.default_root_state.clone()
    robot_default_joint_pos = robot.data.default_joint_pos.clone()
    while simulation_app.is_running():
        # Reset
        if count % 250 == 0:
            # reset counter
            count = 0
            # -- root state
            robotroot_state = robot.data.default_root_state.clone()
            robot_default_joint_pos = robot.data.default_joint_pos.clone()
            robotroot_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(robotroot_state[:, :7])
            robot.write_root_velocity_to_sim(robotroot_state[:, 7:])
            # -- joint state
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO]: Resetting scene state...")

        if count % 10 == 0:
            gripper_camera = scene["gripper_camera"].data.output["rgb"]
            save_images_grid(
                gripper_camera,
                subtitles=[f"Cam{i}" for i in range(gripper_camera.shape[0])],
                title="Tiled RGB Image",
                filename=os.path.join(
                    "./output/vis", "gripper_camera", f"{count:04d}.jpg"
                ),
            )
            top_camera = scene["top_camera"].data.output["rgb"]
            save_images_grid(
                top_camera,
                subtitles=[f"Cam{i}" for i in range(top_camera.shape[0])],
                title="Tiled RGB Image",
                filename=os.path.join("./output/vis", "top_camera", f"{count:04d}.jpg"),
            )

        # Apply action to robot
        robot_default_joint_pos[:, [6]] = (
            torch.ones_like(robot_default_joint_pos[:, [6]])
            * count
            / 200
            * 40
            / 360
            * torch.pi
            * 2
        )
        robot.set_joint_position_target(robot_default_joint_pos[:, :6], list(range(6)))
        robot.set_joint_position_target(robot_default_joint_pos[:, [6]], [6])
        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(
        (2.08, -1.12, 3.95),
        (0.6256159257955498, -1.2960845679032278, 2.9002112950938577),
    )
    # Design scene
    scene_cfg = LunarBaseEnvConfig(
        num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False
    )
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    with Timer("[INFO] Time to randomize scene: "):
        # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
        # Note: Just need to acquire the right attribute about the property you want to set
        # Here is an example on setting color randomly
        # randomize_shape_color(scene_cfg.object.prim_path)
        pass

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main execution
    try:
        main()
    except Exception as e:
        print(f"Main execution error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 关闭 sim app
        if simulation_app:  # 检查 simulation_app 是否已定义
            simulation_app.close()
