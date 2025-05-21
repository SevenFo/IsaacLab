# task_methods.py

import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from collections import OrderedDict
import numpy as np

# Assuming LunarBaseEnv and DirectRLEnv are type hints known in this context
# If not, you might need to use typing.TypeVar or forward references if they are complex types
# For simplicity, let's assume they can be imported or are understood by type checker.
# from isaaclab_tasks.direct.franka_cabinet.ur5_lunar_base_env import LunarBaseEnv # For type hint
# from isaaclab.envs.direct_rl_env import DirectRLEnv # For type hint

# To avoid circular dependencies if LunarBaseEnv imports from task_methods,
# use string literals for type hints or `if typing.TYPE_CHECKING:`
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from isaaclab_tasks.direct.franka_cabinet.ur5_lunar_base_env import (
        LunarBaseEnv,
    )
    from isaaclab.envs.direct_rl_env import DirectRLEnv


def assemble(env: Union["LunarBaseEnv", "DirectRLEnv"], kwargs: dict) -> bool:
    """
    Method to solve assemble task using a pre-trained Robomimic policy.
    This function is called externally, passing the environment instance.

    Args:
        env: An initialized Isaac Lab environment instance (e.g., LunarBaseEnv).
             It's CRITICAL that this environment's _get_observations() returns
             obs_dict["policy"] as a dictionary of tensors, and that
             env_cfg.observations.policy.concatenate_terms was False during its setup
             if Robomimic's ObsUtils relies on that.
        kwargs: A dictionary containing parameters:
            - "checkpoint_path" (str): Path to the Robomimic policy checkpoint.
            - "horizon" (int, optional): Max steps per episode. Defaults to 800.
            - "policy_device" (str, optional): Device for policy ("cuda" or "cpu").
                                             Defaults to "cuda" if available, else "cpu".

    Returns:
        bool: True if the task was considered solved, False otherwise.
    """
    checkpoint_path = kwargs.get("checkpoint_path")
    if not checkpoint_path:
        print("[assemble_task] ERROR: Missing 'checkpoint_path' in kwargs.")
        # Consider raising ValueError("Missing 'checkpoint_path'...")
        return False

    horizon = kwargs.get("horizon", 800)

    specified_policy_device = kwargs.get("policy_device")
    if specified_policy_device:
        policy_device = TorchUtils.get_torch_device(
            try_to_use_cuda=(specified_policy_device == "cuda")
        )
    else:
        policy_device = (
            env.device
        )  # Use env's device if policy_device not specified

    print(f"[assemble_task] Using policy device: {policy_device}")
    print(f"[assemble_task] Environment device: {env.device}")
    print(f"[assemble_task] Loading policy from: {checkpoint_path}")

    # Load Robomimic policy
    try:
        policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path, device=policy_device, verbose=True
        )
    except Exception as e:
        print(
            f"[assemble_task] ERROR: Failed to load policy from checkpoint: {e}"
        )
        return False

    if hasattr(policy, "eval") and callable(policy.eval):
        policy.eval()

    print("[assemble_task] Starting policy rollout...")

    # Get current observation using the environment's public/internal method
    # This assumes _get_observations() returns the latest state correctly.
    obs_dict = env._get_observations()

    policy.start_episode()
    task_successfully_completed = False

    # --- Observation Structure Check ---
    if "policy" not in obs_dict:
        print(
            "[assemble_task] ERROR: 'policy' key missing in environment observations."
        )
        return False

    policy_input_source = obs_dict["policy"]
    if not isinstance(policy_input_source, dict):
        print(
            f"[assemble_task] ERROR: obs_dict['policy'] is not a dictionary (type: {type(policy_input_source)}). "
            "Robomimic policy expects a dictionary of observations under 'policy' key. "
            "Ensure `env._get_observations()` provides this structure and that "
            "`env_cfg.observations.policy.concatenate_terms = False` was likely set if using Robomimic's ObsUtils."
        )
        return False
    # --- End Observation Structure Check ---

    for i in range(horizon):
        # 1. Prepare observations for the policy
        current_policy_obs = OrderedDict()
        for (
            key,
            tensor_val,
        ) in (
            policy_input_source.items()
        ):  # policy_input_source is obs_dict["policy"]
            if not isinstance(tensor_val, torch.Tensor):
                print(
                    f"[assemble_task] ERROR: Observation value for key '{key}' is not a Tensor, but {type(tensor_val)}"
                )
                return False
            current_policy_obs[key] = tensor_val.squeeze(0).to(policy_device)

        # 2. Get action from policy
        with torch.no_grad():
            action_np = policy(current_policy_obs)

        if not isinstance(action_np, np.ndarray):
            print(
                f"[assemble_task] ERROR: Policy output is not a numpy array, but {type(action_np)}"
            )
            return False

        # 3. Format action for environment (specific to LunarBaseEnv's action dict)
        # This part makes the task method somewhat coupled to the env's action_space structure.
        # For a truly generic task_method, action formatting might need to be more abstract
        # or the environment would need to accept a flat action array.
        try:
            end_effector_shape = env.action_space["end_effector"].shape[0]
            gripper_shape = env.action_space["gripper"].shape[0]
            expected_action_dim = end_effector_shape + gripper_shape

            if action_np.shape[0] != expected_action_dim:
                print(
                    f"[assemble_task] ERROR: Policy action dimension mismatch. "
                    f"Expected {expected_action_dim}, got {action_np.shape[0]}."
                )
                return False

            env_action_dict = {
                "end_effector": torch.from_numpy(
                    action_np[0:end_effector_shape]
                )
                .to(env.device)
                .unsqueeze(0),
                "gripper": torch.from_numpy(
                    action_np[
                        end_effector_shape : end_effector_shape + gripper_shape
                    ]
                )
                .to(env.device)
                .unsqueeze(0),
            }
        except (AttributeError, KeyError) as e:
            print(
                f"[assemble_task] ERROR: Could not format action for environment. Env action_space structure mismatch or issue: {e}"
            )
            return False

        # 4. Step the environment
        # This directly calls the env's step method.
        obs_dict, reward, terminated, truncated, info = env.step(
            env_action_dict
        )
        policy_input_source = obs_dict["policy"]  # Update for next iteration

        is_terminated = (
            terminated.item()
            if isinstance(terminated, torch.Tensor)
            else terminated
        )
        is_truncated = (
            truncated.item()
            if isinstance(truncated, torch.Tensor)
            else truncated
        )

        if is_terminated or is_truncated:
            if is_terminated and not is_truncated:
                print(
                    f"[assemble_task] Episode terminated successfully by environment logic after {i + 1} steps."
                )
                task_successfully_completed = True
            elif is_truncated:
                print(
                    f"[assemble_task] Episode truncated (e.g., time limit) after {i + 1} steps."
                )
            else:
                print(
                    f"[assemble_task] Episode terminated and truncated after {i + 1} steps."
                )
            break
    else:
        print(
            f"[assemble_task] Episode reached horizon ({horizon} steps) without termination."
        )

    print(
        f"[assemble_task] Policy rollout finished. Task success: {task_successfully_completed}"
    )
    return task_successfully_completed


# You can add other task methods here if needed
# def another_task(env, kwargs):
#     pass

AVAILABLE_TASK_METHODS = {
    "assemble": assemble,
    # "another_task": another_task,
}
