# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--env_config_file",
    type=str,
    default=None,
    help="Env Config YAML file Path, use to update default env config",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    if args_cli.env_config_file:
        with open(args_cli.env_config_file, "r") as f:
            env_new_cfg = yaml.safe_load(f)

        def dynamic_set_attr(object: object, kwargs: dict, path: list[str]):
            for k, v in kwargs.items():
                if k in object.__dict__:
                    if isinstance(v, dict):
                        next_path = path.copy()
                        next_path.append(k)
                        dynamic_set_attr(
                            object.__getattribute__(k), v, next_path
                        )
                    else:
                        print(
                            f"set {'.'.join(path + [k])} from {object.__getattribute__(k)} to {v}"
                        )
                        object.__setattr__(k, v)

        dynamic_set_attr(env_cfg, env_new_cfg, path=["env_cfg"])
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            device = env.unwrapped.device  # 获取设备信息
            actions = {
                key: torch.zeros(space.shape, device=device)
                for key, space in env.action_space.spaces.items()
            }
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
