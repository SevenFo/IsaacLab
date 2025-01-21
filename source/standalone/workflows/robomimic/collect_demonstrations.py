# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect demonstrations for Isaac Lab environments."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=1,
    help="Number of episodes to store in the dataset.",
)
parser.add_argument(
    "--filename", type=str, default="hdf_dataset", help="Basename of output file."
)
# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# from omni.isaac.core.utils.extensions import enable_extension

# EXTENSIONS = [
#     "omni.anim.skelJoint",
# ]

# for ext in EXTENSIONS:
#     enable_extension(ext)
"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse, Se3Gamepad
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

from omni.isaac.lab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.sensors.camera import TiledCamera
from omni.isaac.lab.sensors.camera.utils import (
    save_images_to_file,
    create_pointcloud_from_depth,
    transform_points,
)
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.utils.math import (
    subtract_frame_transforms,
    quat_rotate,
    euler_xyz_from_quat,
    quat_from_matrix,
    quat_mul,
    quat_from_angle_axis,
    quat_inv,
)
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core.utils.viewports import set_camera_view
import omni.kit.hotkeys.core


marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    prim_path="/Visuals/myMarkers",
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        ),
    },
)
my_visual_markers = VisualizationMarkers(marker_cfg)


def DeRegisterAllHotKeys():
    hotkey_registry = omni.kit.hotkeys.core.get_hotkey_registry()
    discovered_hotkeys = hotkey_registry.get_all_hotkeys()
    print(f"There are now {len(discovered_hotkeys)} hotkeys.")
    delete_list = discovered_hotkeys.copy()
    for hotkey in delete_list:
        try:
            hotkey_registry.deregister_hotkey(hotkey)
        except Exception as e:
            print(f"Failed to deregister hotkey:{hotkey} {e}")
    discovered_hotkeys = hotkey_registry.get_all_hotkeys()
    print(f"After deletion there are now {len(discovered_hotkeys)} hotkeys.")


def _set_camera_view_to_target_object():
    viewport = get_active_viewport()
    # viewport.set_active_camera(path)
    set_camera_view(
        eye=np.array([1.6, 0.0, 0.6]),
        target=np.array([0, 0, 0]),
        viewport_api=viewport,
    )


def pre_process_actions(
    delta_pose: torch.Tensor, gripper_command: bool
) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(
            (delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device
        )
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def get_frame_transforms(
    frame_name, robot, rotation, translation, relative_translation: bool = True
):
    """
    rotation: torch.Tensor, shape=(4,), dtype=torch.float, w.r.t. the relative frame (i.e. last frame)
    translation: torch.Tensor, shape=(3,), dtype=torch.float, w.r.t. the relative frame (i.e. last frame)
    """
    frame_idx = [
        idx for idx, name in enumerate(robot.body_names) if name == frame_name
    ][0]
    frame_pos = robot.data.body_pos_w[:, frame_idx, :].cpu()
    frame_quat = robot.data.body_quat_w[:, frame_idx, :].cpu()
    offset = torch.tensor(
        translation, dtype=frame_pos.dtype, device=frame_pos.device
    ).expand_as(frame_pos)
    trans_quat = quat_mul(
        frame_quat, rotation
    )  # 右乘为 inner rotation (i.e. body-fixed rotation), 左乘为 outer rotation (i.e. space-fixed rotation)
    if relative_translation:
        trans_pos = frame_pos + quat_rotate(
            trans_quat, offset
        )  # 先将 offset 旋转到 frame 的坐标系下，再加上 frame 的位置
    else:
        trans_pos = frame_pos + offset
    return trans_pos, trans_quat


def get_image_observation(env: ManagerBasedEnv):
    """F
    Due to current limitations in the renderer, we can have only one TiledCamera instance in the scene. For use cases that require a setup with more than one camera, we can imitate the multi-camera behavior by moving the location of the camera in between render calls in a step.
    For example, in a stereo vision setup, the below snippet can be implemented:

    # render image from "first" camera
    camera_data_1 = self._tiled_camera.data.output["rgb"].clone() / 255.0
    # update camera transform to the "second" camera location
    self._tiled_camera.set_world_poses(
        positions=pos,
        orientations=rot,
        convention="world"
    )
    # step the renderer
    self.sim.render()
    self._tiled_camera.update(0, force_recompute=True)
    # render image from "second" camera
    camera_data_2 = self._tiled_camera.data.output["rgb"].clone() / 255.0

    As for robot env, we should trans front tiled camera (default position) to the wrist of the robot and render the image.

    - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
    - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
    - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

    default camera axis: -Z (forward), +Y (up), +X (right)
    """

    matrix_from_ros_to_world = torch.Tensor([[[0, 0, 1], [-1, 0, 0], [0, -1, 0]]])
    q_z_n90 = quat_from_angle_axis(
        torch.tensor([-0.5 * torch.pi]), torch.tensor([0, 0, 1], dtype=torch.float)
    )
    q_z_90 = quat_from_angle_axis(
        torch.tensor([0.5 * torch.pi]), torch.tensor([0, 0, 1], dtype=torch.float)
    )
    q_x_180 = quat_from_angle_axis(
        torch.tensor([torch.pi]), torch.tensor([1, 0, 0], dtype=torch.float)
    )
    q_y_60 = quat_from_angle_axis(
        torch.tensor([1 / 3 * torch.pi]), torch.tensor([0, 1, 0], dtype=torch.float)
    )
    q_y_30 = quat_from_angle_axis(
        torch.tensor([1 / 6 * torch.pi]), torch.tensor([0, 1, 0], dtype=torch.float)
    )
    q_unit = torch.tensor([[1, 0, 0, 0]], dtype=torch.float)

    trans_posw, trans_quatw = get_frame_transforms(
        "panda_hand",
        robot := env.scene.articulations.get("robot"),  # type: ignore
        rotation=quat_from_angle_axis(
            torch.tensor([0.5 * torch.pi]), torch.tensor([0, 0, 1], dtype=torch.float)
        ),
        translation=torch.tensor([0.0, 0.1, 0.0], dtype=torch.float),
    )

    camera_trans_dict = {
        "wrist": get_frame_transforms(
            "panda_hand",
            robot := env.scene.articulations.get("robot"),  # type: ignore
            rotation=quat_mul(q_x_180, q_z_n90),
            translation=torch.tensor([0.0, 0.05, -0.06], dtype=torch.float),
        ),
        "front": get_frame_transforms(
            "panda_link0",
            robot := env.scene.articulations.get("robot"),  # type: ignore
            rotation=quat_mul(q_y_60, q_z_90),  # inner rotation: y_60 -> z_90
            translation=torch.tensor([1.6, 0.0, 0.6], dtype=torch.float),
            relative_translation=False,
        ),
    }

    # world_lcoation = torch.tensor([0, 0, 0])
    # wrist_cam_location = camera_trans_dict["wrist"][0].squeeze()
    # front_cam_location = camera_trans_dict["front"][0].squeeze()

    # world_rotation = torch.tensor([1, 0, 0, 0], dtype=torch.float)
    # wrist_cam_rotation = camera_trans_dict["wrist"][1].squeeze()
    # front_cam_rotation = camera_trans_dict["front"][1].squeeze()

    # locations = torch.stack(
    #     [world_lcoation, wrist_cam_location, front_cam_location], dim=0
    # )
    # orientations = torch.stack(
    #     [world_rotation, wrist_cam_rotation, front_cam_rotation], dim=0
    # )

    # my_visual_markers.visualize(
    #     translations=locations, orientations=orientations, marker_indices=[0, 0, 0]
    # )

    camera: TiledCamera = env.scene.sensors["tiled_camera"]  # type: ignore
    camera_data_dict = {}
    for key, value in camera_trans_dict.items():
        # 设置相机位置和方向
        camera.set_world_poses(
            positions=value[0], orientations=value[1], convention="opengl"
        )
        # 渲染新视角
        env.sim.render()
        camera.update(0, force_recompute=True)

        intrinsic_matrices = camera.data.intrinsic_matrices.clone()

        # 获取并处理图像数据
        rgb_data = (
            camera.data.output["rgb"].clone() / 255.0
        )  # torch.Size([1, 256, 256, 3])
        depth_data = camera.data.output[
            "distance_to_image_plane"
        ].clone()  # torch.Size([1, 256, 256, 1])
        # pointcloud_data = create_pointcloud_from_depth(
        #     intrinsic_matrix=intrinsic_matrices.squeeze(0),  # torch.Size([3, 3])
        #     depth=depth_data.squeeze(0).squeeze(-1),  # torch.Size([256, 256])
        #     orientation=q_x_180,  # transform to camera coordinate (not camera optical coordinate)
        # )
        # pointcloud_data = transform_points(
        #     pointcloud_data,
        #     orientation=value[1].squeeze().tolist(),
        #     position=value[0].squeeze().tolist(),
        # )  # transform to world coordinate
        # save_images_to_file(rgb_data, f"camera_rgb_{key}.png")
        # save_images_to_file(
        #     (depth_data.clone() - depth_data.min())
        #     / (depth_data.max() - depth_data.min()),
        #     f"camera_depth_{key}.png",
        # )
        # np.savetxt(
        #     f"camera_pointcloud_{key}.txt",
        #     pointcloud_data.numpy(force=True),  # type: ignore
        #     delimiter=",",
        # )
        camera_data_dict[f"{key}_rgb"] = rgb_data
        camera_data_dict[f"{key}_depth"] = depth_data
        # camera_data_dict[f"{key}_pointcloud"] = pointcloud_data

    return camera_data_dict


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    DeRegisterAllHotKeys()

    assert (
        args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
        or args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-Cam-v0"
    ), "Only 'Isaac-Lift-Cube-Franka-IK-Rel-v0' is supported currently."
    # parse configuration
    env_cfg: FrankaCubeLiftEnvCfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.observations.rgb.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    env_cfg.commands.object_pose.debug_vis = False  # disable debug visualization
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # set camera view to target object
    _set_camera_view_to_target_object()
    env.unwrapped.sim.step()

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=0.04, rot_sensitivity=0.08)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'."
        )
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper
    print(teleop_interface)

    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=args_cli.num_envs,
        env_config={"teleop_device": args_cli.teleop_device},
    )

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces
    teleop_interface.reset()
    collector_interface.reset()

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not collector_interface.is_stopped():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(
                delta_pose, dtype=torch.float, device=args_cli.device
            ).repeat(args_cli.num_envs, 1)
            # compute actions based on environment
            actions = pre_process_actions(delta_pose, gripper_command)

            # TODO: Deal with the case when reset is triggered by teleoperation device.
            #   The observations need to be recollected.
            # store signals before stepping
            # -- obs
            for key, value in obs_dict["policy"].items():
                collector_interface.add(f"obs/{key}", value)
            # cam_obs = get_image_observation(env.unwrapped)
            # for key, value in cam_obs.items():
            #     collector_interface.add(f"obs/{key}", value)
            # -- actions
            collector_interface.add("actions", actions)

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            # robomimic only cares about policy observations
            # store signals from the environment
            # -- next_obs
            for key, value in obs_dict["policy"].items():
                collector_interface.add(f"next_obs/{key}", value)
            # cam_obs = get_image_observation(env.unwrapped)
            # for key, value in cam_obs.items():
            #     collector_interface.add(f"next_obs/{key}", value)
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)

            # -- is success label
            collector_interface.add(
                "success",
                env.unwrapped.termination_manager.get_term("object_reached_goal"),
            )

            # flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)

            # check if enough data is collected
            if collector_interface.is_stopped():
                break

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
