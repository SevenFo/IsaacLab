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
# parser.add_argument("--active_gpu", type=int, default=2)
# parser.add_argument("--physics_gpu", type=int, default=0)
parser.add_argument("--multi_gpu", action="store_false", default=True)
parser.add_argument("--max_gpu_count", type=int, default=3)
# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.core.utils.extensions import enable_extension

EXTENSIONS = [
    "omni.anim.skelJoint",
]

for ext in EXTENSIONS:
    enable_extension(ext)

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
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.xforms import reset_and_set_xform_ops
import omni.kit.hotkeys.core

from pxr import Gf

if __name__ == "__main__":
    assert (
        args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
        or args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-Cam-v0"
        or args_cli.task == "Isaac-Lift-Cube-Franka-IK-Abs-Cam-v0"
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

    # add people to env
    people_prim = add_reference_to_stage(
        usd_path=f"omniverse://localhost/Projects/lunarbase/chars/astro/astro.usd",
        prim_path="/World/House/People",
    )
    reset_and_set_xform_ops(
        prim=people_prim,
        translation=Gf.Vec3d(45.0, -70.0, 0.0),
        orientation=Gf.Quatd(0, 0, 0, 1.0),
        scale=Gf.Vec3d(100.0, 100.0, 100.0),
    )

    obs_dict, _ = env.reset()

    while simulation_app.is_running():
        env.unwrapped.sim.step()
