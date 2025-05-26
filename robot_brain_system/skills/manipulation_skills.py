"""
Manipulation skills for robot arm control.
Primarily features the 'assemble_object' skill using a pre-trained policy.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, Generator, Tuple, Optional, Union, TYPE_CHECKING

from ..core.types import SkillType, ExecutionMode
from ..core.skill_manager import skill_register

# Attempt to import Robomimic, make it optional
ROBOMIMIC_AVAILABLE = False
try:
    import robomimic.utils.torch_utils as TorchUtils
    import robomimic.utils.file_utils as FileUtils
    from collections import OrderedDict  # Used by robomimic example

    ROBOMIMIC_AVAILABLE = True
except ImportError:
    print(
        "[Skill: assemble_object] Warning: robomimic library not found. Assemble skill will not be functional."
    )


# Type hinting for Isaac Lab environment if available
if TYPE_CHECKING:
    from omni.isaac.lab.envs import (
        ManagerBasedRLEnv,
    )  # Or the specific env type you use


@skill_register(
    name="assemble_object",
    skill_type=SkillType.FUNCTION,  # Could be SkillType.POLICY if you have a separate handler
    execution_mode=ExecutionMode.GENERATOR,
    description="Assemble an object using a pre-trained Robomimic policy.",
    timeout=300.0,  # 5 minutes, adjust as needed
    requires_env=True,
)
def assemble_object(env: Any) -> Generator[None, None, bool]:
    """
    Assemble an object using a pre-trained Robomimic policy.
    This skill runs the policy step-by-step within the Isaac simulator.

    Expected params: None
    """
    if not ROBOMIMIC_AVAILABLE:
        print(
            "[Skill: assemble_object] Robomimic library not available. Cannot execute."
        )
        yield None, None, None, None, None
        return False

    print("[Skill: assemble_object] Starting...")

    checkpoint_path = "assets/model_epoch_4000.pth"
    if not checkpoint_path:
        print(
            "[Skill: assemble_object] Error: 'checkpoint_path' parameter is missing."
        )
        yield None, None, None, None, None
        return False

    if not os.path.exists(checkpoint_path):
        print(
            f"[Skill: assemble_object] Error: Checkpoint path '{checkpoint_path}' does not exist."
        )
        yield None, None, None, None, None
        return False

    horizon = 800
    # Determine policy device: use specified, else try to use env's device, else cpu
    # The env object passed here is the gym.Env wrapper.
    # Accessing unwrapped.device is specific to Isaac Lab's DirectRLEnv or ManagerBasedRLEnv.
    # Be cautious if env type varies.
    default_device_str = "cpu"
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "device"):
        default_device_str = str(env.unwrapped.device)

    specified_policy_device_str = "cuda"
    policy_device = TorchUtils.get_torch_device(
        try_to_use_cuda=(specified_policy_device_str.startswith("cuda"))
    )

    print(f"[Skill: assemble_object] Using policy device: {policy_device}")
    print(f"[Skill: assemble_object] Environment device: {default_device_str}")
    print(f"[Skill: assemble_object] Loading policy from: {checkpoint_path}")

    try:
        policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path, device=policy_device, verbose=True
        )
        if hasattr(policy, "eval") and callable(policy.eval):
            policy.eval()
    except Exception as e:
        print(f"[Skill: assemble_object] Error: Failed to load policy: {e}")
        yield None, None, None, None, None
        return False

    print("[Skill: assemble_object] Policy loaded. Starting rollout...")

    # Reset environment and get initial observation
    # The environment should be reset by the system or a dedicated "reset" skill before calling this.
    # However, if this skill is meant to be a full episode, it might reset.
    # For now, assume obs comes from a previous env.reset() or env.step()
    # The standard is that env.reset() is called before a sequence of skills or tasks.
    # We need the *current* observation. A skill usually doesn't reset the env itself.
    # Let's assume the `env` is ready and an observation can be obtained or was just produced.
    # If `env` is the raw Isaac Lab env, it might have a method like `env.get_observations()`
    # or the observation is part of the last `step` call.
    # The `assemble_task_generator` from `intelligent_robot_system` does `obs_dict, _ = env.reset()`.
    # This is okay if the skill is truly episodic.

    # Let's adapt to get current obs. If the env is a gym.Env, there isn't a standard get_obs.
    # We must rely on the obs from the last step or reset.
    # The `SkillExecutor` in the subprocess won't provide obs to `next(generator.send(obs))`.
    # The skill itself gets `env` and calls `env.step()` to get new obs.
    # So, an initial observation is needed to start the policy.
    # This is tricky. The `intelligent_robot_system` example did `obs_dict, _ = env.reset()`. Let's follow that for now.

    obs_dict, info = (
        env.reset()
    )  # This resets the specific env instance passed.
    # Important: if this skill is part of a longer sequence, this reset might be undesirable.
    # Consider if `env.get_observations()` is a better fit if available and skill is not fully episodic.
    yield obs_dict, None, None, None, info
    policy.start_episode()
    task_successfully_completed = False

    # Check observation structure (critical for Robomimic)
    # This structure depends on how your Isaac Lab environment is configured.
    # The example from `intelligent_robot_system` uses `obs_dict['policy']`.
    # We need to ensure `env.reset()` and `env.step()` return observations in this format.
    # The `ReachEnvCfg` example likely returns obs in `obs_dict["policy"]`.

    policy_obs_key = (
        "policy"  # Common key in Isaac Lab tasks for policy inputs
    )
    if policy_obs_key not in obs_dict:
        print(
            f"[Skill: assemble_object] Error: Key '{policy_obs_key}' not found in initial observations."
        )
        yield None, None, None, None, None
        return False

    policy_input_source = obs_dict[policy_obs_key]
    if not isinstance(policy_input_source, dict):
        print(
            f"[Skill: assemble_object] Error: obs_dict['{policy_obs_key}'] is not a dict, but {type(policy_input_source)}."
        )
        yield None, None, None, None, None
        return False

    for i in range(horizon):
        current_policy_obs = OrderedDict()
        for key, tensor_val in policy_input_source.items():
            if not isinstance(tensor_val, torch.Tensor):
                print(
                    f"[Skill: assemble_object] Warning: Obs value for key '{key}' is not a Tensor."
                )
                continue  # Or handle appropriately
            current_policy_obs[key] = tensor_val.squeeze(0).to(
                policy_device
            )  # Robomimic expects unsqueezed

        with torch.no_grad():
            action_np = policy(current_policy_obs)  # Get action from policy

        if not isinstance(action_np, np.ndarray):
            print(
                f"[Skill: assemble_object] Error: Policy output is not a numpy array (got {type(action_np)})."
            )
            yield None, None, None, None, None
            return False

        # Action needs to be formatted for the environment's action space
        # For Isaac Lab, typically a torch tensor on the env's device, batched.
        action_tensor = (
            torch.from_numpy(action_np).to(env.unwrapped.device).unsqueeze(0)
        )

        # Step the environment
        obs_dict, reward, terminated, truncated, info = env.step(action_tensor)

        # Update policy input source for next iteration
        if policy_obs_key not in obs_dict:
            print(
                f"[Skill: assemble_object] Error: Key '{policy_obs_key}' not found in new observations after step."
            )
            yield None, None, None, None, None
            return False
        policy_input_source = obs_dict[policy_obs_key]

        # Yield control back to the executor loop in the subprocess
        yield obs_dict, reward, terminated, truncated, info

        # Check for termination conditions
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
            print(
                f"[Skill: assemble_object] Episode ended at step {i + 1}. Terminated: {is_terminated}, Truncated: {is_truncated}"
            )
            # Determine success based on your task's success criteria (e.g., reward, info flags)
            # This is a simplified check. Real success might come from info or specific reward.
            if (
                is_terminated and not is_truncated
            ):  # Assuming terminated means success for this policy
                task_successfully_completed = True
            break
    else:  # Loop finished without break (horizon reached)
        print(
            f"[Skill: assemble_object] Episode reached horizon ({horizon} steps) without termination."
        )

    print(
        f"[Skill: assemble_object] Rollout finished. Task success: {task_successfully_completed}"
    )
    return task_successfully_completed


# Ensure os module is imported if not already
import os
