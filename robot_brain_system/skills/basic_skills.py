"""
Basic skills for robot control.
"""

import time
import numpy as np
import torch  # Keep for type hinting / potential use by skills
from typing import Dict, Any, Generator, Optional

# Assuming Action and Observation types are primarily for IPC,
# skills will now directly use env.step() and standard Isaac Lab obs/action formats.
from ..core.types import (
    SkillType,
    ExecutionMode,
)  # Action, Observation (these might become less relevant for in-skill use)
from ..core.skill_manager import skill_register


@skill_register(
    name="reset_to_home",
    skill_type=SkillType.FUNCTION,
    execution_mode=ExecutionMode.GENERATOR,
    description="Reset robot to home position",
    timeout=10.0,
    requires_env=True,  # This skill now needs the environment
)
def reset_to_home(
    env: Any, params: Dict[str, Any]
) -> Generator[None, None, bool]:
    """
    Reset robot to home position using generator mode.
    The skill now directly interacts with the 'env' provided.
    """
    print("[Skill] reset_to_home: Starting...")

    # Get home position from params or use default
    # The format of home_position needs to match the environment's action space.
    # For Isaac-Reach-Franka-v0, this might be a target end-effector pose.
    # The original example used a 6D array. Let's assume it's a target pose.
    # For simplicity, we'll use the environment's reset method.
    # A true "move to home" would be more complex.

    # A more robust reset_to_home would involve a sequence of actions
    # to move the robot to a predefined joint configuration or EE pose.
    # For now, we use the environment's own reset mechanism as a proxy.
    try:
        # Most Isaac Lab envs return obs_dict, info_dict
        obs_dict, info_dict = env.reset()
        print("[Skill] reset_to_home: Environment reset triggered.")

        # Yield a few times to let the reset settle if it's not instantaneous
        for _ in range(5):  # Simulate a short settling time
            yield
            # If env has a render or update call needed: env.render() or simulation_app.update()
            # This depends on how the subprocess loop is structured.
            # The IsaacSimulator subprocess loop will handle sim_app.update().

        print("[Skill] reset_to_home: Completed")
        return True  # Assuming env.reset() is successful

    except Exception as e:
        print(f"[Skill] reset_to_home: Error during environment reset: {e}")
        return False


# @skill_register(
#     name="wait",
#     skill_type=SkillType.FUNCTION,
#     execution_mode=ExecutionMode.GENERATOR,
#     description="Wait for specified duration",
#     timeout=None,  # No timeout for wait skill
#     requires_env=True,  # Needs env to potentially send no-op actions or just for consistency
# )
def wait_skill(
    env: Any, params: Dict[str, Any]
) -> Generator[None, None, bool]:
    """
    Wait for a specified duration.
    During the wait, the skill will yield, allowing the simulator to step.
    It can optionally send no-op actions to the environment if needed.
    """
    duration = params.get("duration", 1.0)  # seconds
    print(f"[Skill] wait: Waiting for {duration} seconds...")

    start_time = time.time()
    # Assuming env.action_space is available from the gym.Env wrapper
    # This is a very generic no-op. Specific envs might need different no-ops.
    # Example: zero velocities for a velocity-controlled robot.
    # For Isaac-Reach-Franka-v0, action is likely [dx, dy, dz, dax, day, daz, gripper_state]
    # A no-op could be all zeros.

    # Construct a no-op action based on the environment's action space
    # This requires knowing the action space structure.
    # For simplicity, if the env is an Isaac Lab env, it will keep executing the last action if no new one is sent.
    # So, simply yielding is often enough. If explicit no-op is needed:
    # no_op_action_np = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    # no_op_action_tensor = torch.from_numpy(no_op_action_np).to(env.unwrapped.device).unsqueeze(0)

    while time.time() - start_time < duration:
        # Optionally, send a no-op action to the environment
        # obs, reward, term, trunc, info = env.step(no_op_action_tensor)
        yield  # Yield control to the main simulation loop
        # The IsaacSimulator._isaac_simulation_entry loop will call sim_app.update()
        # and skill_executor.step(), which calls next(this_generator).
        # A small sleep inside the generator might make it less CPU intensive if the
        # outer loop doesn't have its own sleep.
        # However, time.sleep() here would block the *skill's* execution,
        # not the whole subprocess directly unless the subprocess only steps this skill.
        # The outer loop's `time.sleep(0.001)` in _isaac_simulation_entry is better.

    print(f"[Skill] wait: Completed after {duration}s")
    return True


# @skill_register(
#     name="emergency_stop",
#     skill_type=SkillType.FUNCTION,
#     execution_mode=ExecutionMode.DIRECT,  # This can remain direct if it doesn't need env stepping
#     description="Emergency stop - immediate halt (conceptual)",
#     timeout=1.0,
#     requires_env=False,  # Assuming this can be handled at a higher level or via direct sim commands
# )
def emergency_stop(
    params: Dict[str, Any],
) -> bool:  # Env not needed if requires_env=False
    """
    Emergency stop skill. In a real system, this would send immediate stop commands.
    """
    print("[Skill] emergency_stop: EMERGENCY STOP ACTIVATED!")
    reason = params.get("reason", "User requested")
    print(f"[Skill] emergency_stop: Reason - {reason}")

    # In a real implementation, this might involve:
    # - Sending a specific command to the simulator to halt physics or apply brakes.
    # - Setting a global flag that other components check.
    # If it needs to interact with `env` to stop motors, set requires_env=True
    # and change signature to `emergency_stop(env, params)`.
    # For now, keeping it as a conceptual non-env skill.
    return True


# @skill_register(
#     name="get_current_state",
#     skill_type=SkillType.FUNCTION,
#     execution_mode=ExecutionMode.GENERATOR,  # Or DIRECT if env has a direct get_obs method
#     description="Get current robot state and observations",
#     timeout=5.0,
#     requires_env=True,
# )
def get_current_state(
    env: Any, params: Dict[str, Any]
) -> Generator[None, None, Dict[str, Any]]:
    """
    Get current robot state by performing a no-op step and returning observation.
    Returns the latest observation data.
    """
    print("[Skill] get_current_state: Querying current state...")

    # To get the most recent state, often a no-op step is performed.
    # Create a no-op action (e.g., zero velocities or maintain current pose)
    # This is highly dependent on the specific environment's action space.
    # As a generic approach for Isaac Lab envs, taking a zero action is common.
    # Example for a 7-DoF action space (e.g. delta EE pose + gripper)
    no_op_action_np = np.zeros(
        env.action_space.shape, dtype=env.action_space.dtype
    )
    action_tensor = (
        torch.from_numpy(no_op_action_np).to(env.unwrapped.device).unsqueeze(0)
    )  # Batch for single env

    obs_dict, reward, terminated, truncated, info = env.step(action_tensor)
    yield  # Yield once after stepping

    # Extract useful information from observation
    # The obs_dict structure depends on the Isaac Lab environment.
    # Often, there's a "policy" key for RL-ready observations.
    policy_obs = obs_dict.get("policy")
    if isinstance(policy_obs, torch.Tensor):
        policy_obs_np = policy_obs.cpu().numpy()
    elif isinstance(policy_obs, dict):  # If obs["policy"] is a dict of tensors
        policy_obs_np = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in policy_obs.items()
        }
    else:
        policy_obs_np = policy_obs

    state_info = {
        "timestamp": time.time(),  # Approximate, actual sim time might be in info
        "observation_policy_data": policy_obs_np,  # Or serialize further
        "full_observation_dict_keys": list(obs_dict.keys()),
        "info": info,  # This might contain sim-specific details
    }

    print(f"[Skill] get_current_state: Retrieved state.")
    return state_info


# Helper function for creating movement actions (if needed by skills, less common now)
# Skills will typically form their own action tensors.
# def create_movement_action(...)

