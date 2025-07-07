# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to replay demonstrations from a file and record them into a new dataset.

This script loads episodes from an existing HDF5 file (which may only contain actions),
replays them in the simulator, and records the full data (actions, observations, states)
into a new HDF5 file. This is useful for completing datasets that were partially recorded.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay and record demonstrations in Isaac Lab environments.")
parser.add_argument(
    "--input_dataset_file", type=str, required=True, help="Input dataset file to be replayed (source)."
)
parser.add_argument(
    "--output_dataset_file", type=str, required=True, help="Output dataset file to save the re-recorded episodes."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes in parallel.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task, overriding the one from the dataset.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    """Replay episodes from a file and record them into a new one."""

    # --- 1. Load Input Dataset ---
    if not os.path.exists(args_cli.input_dataset_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_dataset_file} does not exist.")
    
    input_dataset_handler = HDF5DatasetFileHandler()
    input_dataset_handler.open(args_cli.input_dataset_file)
    env_name = input_dataset_handler.get_env_name()
    episode_count = input_dataset_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the input dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if not episode_indices_to_replay:
        episode_indices_to_replay = list(range(episode_count))

    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    # --- 2. Configure Environment for Recording ---
    num_envs = args_cli.num_envs
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Disable terminations as we are replaying fixed-length episodes
    env_cfg.terminations = {}

    # Configure observations to be recorded individually
    env_cfg.observations.policy.concatenate_terms = False

    # ** KEY CHANGE: Enable the recorder to save data to the new output file **
    output_dir = os.path.dirname(args_cli.output_dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_dataset_file))[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    env_cfg.recorders = ActionStateRecorderManagerCfg(
        dataset_export_dir_path=output_dir,
        dataset_filename=output_file_name,
    )

    # create environment from loaded config
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    # reset before starting
    env.reset()

    # --- 3. Replay and Record Loop ---
    episode_names = list(input_dataset_handler.get_episode_names())
    replayed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting() and episode_indices_to_replay:
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            is_first_step_in_batch = True
            has_more_actions = True

            while has_more_actions:
                actions = torch.zeros(env.action_space.shape, device=env.device)
                has_more_actions = False
                
                # Buffer for envs that just finished an episode
                envs_finished_episode = []

                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    
                    if env_next_action is None:
                        # This env has finished its current episode.
                        # Mark it for potential export and loading of a new one.
                        if not is_first_step_in_batch:
                            envs_finished_episode.append(env_id)
                        
                        # Try to load the next available episode
                        next_episode_index = None
                        if episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)

                        if next_episode_index is not None:
                            print(f"Env {env_id}: Loading episode #{next_episode_index} to replay...")
                            episode_data = input_dataset_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            
                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            
                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_more_actions = True # We have a new episode to process
                        else:
                            # No more episodes to assign to this env
                            continue
                    
                    if env_next_action is not None:
                        actions[env_id] = env_next_action
                        has_more_actions = True
                
                # ** KEY CHANGE: Export completed episodes before the next step **
                if envs_finished_episode:
                    env_ids_to_export = torch.tensor(envs_finished_episode, device=env.device)
                    print(f"Exporting completed episodes for envs: {envs_finished_episode}")
                    # Mark the completed episodes as successful
                    success_flags = torch.ones((len(envs_finished_episode), 1), dtype=torch.bool, device=env.device)
                    env.recorder_manager.set_success_to_episodes(env_ids_to_export, success_flags)
                    # Export the episodes to the new HDF5 file
                    env.recorder_manager.export_episodes(env_ids_to_export)
                    replayed_episode_count += len(envs_finished_episode)


                if not has_more_actions:
                    break

                # The recorder automatically captures data before this step
                env.step(actions)
                is_first_step_in_batch = False

            # --- 4. Final Export ---
            # After the loop, export any remaining episodes that were fully replayed but not yet exported
            final_envs_to_export = []
            for env_id in range(num_envs):
                # Check if the env has valid data (i.e., it was used) and hasn't been exported yet
                if env_episode_data_map[env_id].get_num_steps() > 0 and env_episode_data_map[env_id].next_action_index >= env_episode_data_map[env_id].get_num_steps():
                    final_envs_to_export.append(env_id)

            if final_envs_to_export:
                env_ids_to_export = torch.tensor(final_envs_to_export, device=env.device)
                print(f"Exporting final episodes for envs: {final_envs_to_export}")
                success_flags = torch.ones((len(final_envs_to_export), 1), dtype=torch.bool, device=env.device)
                env.recorder_manager.set_success_to_episodes(env_ids_to_export, success_flags)
                env.recorder_manager.export_episodes(env_ids_to_export)
                replayed_episode_count += len(final_envs_to_export)

            # If all episodes from the list are processed, break the main while loop
            if not episode_indices_to_replay:
                break

    # Close environment after replay is complete
    plural_trailing_s = "s" if replayed_episode_count != 1 else ""
    print(f"\nFinished replaying and recording {replayed_episode_count} episode{plural_trailing_s}.")
    print(f"New dataset saved to: {args_cli.output_dataset_file}")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()