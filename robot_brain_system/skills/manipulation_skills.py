"""
Manipulation skills for robot arm control.
Primarily features the 'assemble_object' skill using a pre-trained policy.
"""

import time
import numpy as np
import torch
# Ensure os module is imported if not already
import os
from typing import Dict, Any, Generator, Tuple, Optional, Union, TYPE_CHECKING

from ..core.types import SkillType, ExecutionMode, PolicyFunction, Action
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
    name="press_button",
    skill_type=SkillType.POLICY,  # Could be SkillType.POLICY if you have a separate handler
    execution_mode=ExecutionMode.STEPACTION,
    timeout=300.0,  # 5 minutes, adjust as needed
    criterion={
        "successed": "The red box is opened",
        "failed": "".join(["The gripper posisiton is far away from the box and yellow button, ",
                           "or the gripper was pressed on areas other than the yellow button,"
                   "or the gripper is lingering (not moving) for several monitoring rounds, ",
                   "or any other gripper state that is not reasonable to execute the skill."]),
        "progress": "The gripper is on a reasonable state to execute the skill, such as: moving towards the red box and yellow button, etc.",
    },
    requires_env=True,
)
class PressButton: 
    """
    Assemble an object using a pre-trained Robomimic policy.
    This skill runs the policy step-by-step within the Isaac simulator.

    Expected params: None, NO NEED TO PASS ANY PARAMS, the skill will automatically get nessessary parameters from the environment.
    """       
    def __init__(self, policy_device: str = 'cuda', **running_params):
        self.policy_device = policy_device
        self.running_params = running_params
        if not ROBOMIMIC_AVAILABLE:
            print(  
                "[Skill: press_button] Robomimic library not available. Cannot execute."
            )
            return None

        print("[Skill: press_button] Starting...")

        checkpoint_path = "assets/skills/press.pth"
        if not os.path.exists(checkpoint_path):
            print(
                f"[Skill: press_button] Error: Checkpoint path '{checkpoint_path}' does not exist."
            )
            return None


        print(f"[Skill: press_button] Using policy device: {policy_device}")
        print(f"[Skill: press_button] Loading policy from: {checkpoint_path}")

        try:
            policy, _ = FileUtils.policy_from_checkpoint(
                ckpt_path=checkpoint_path, device=policy_device, verbose=True
            )
        except Exception as e:
            print(f"[Skill: press_button] Error: Failed to load policy: {e}")
            return None

        print("[Skill: press_button] Policy loaded")

        self.policy = policy
        self.policy.start_episode()

    def select_action(self, obs_dict: dict) -> Action:
        policy_obs_key = (
            "policy"  # Common key in Isaac Lab tasks for policy inputs
        )
        if policy_obs_key not in obs_dict:
            print(
                f"[Skill: PressButton] Error: Key '{policy_obs_key}' not found in initial observations."
            )
            return Action([], metadata={"info":'error'})

        policy_input_source = obs_dict[policy_obs_key]

        current_policy_obs = OrderedDict()
        for key, tensor_val in policy_input_source.items():
            if not isinstance(tensor_val, torch.Tensor):
                print(
                    f"[Skill: PressButton] Warning: Obs value for key '{key}' is not a Tensor."
                )
                continue  # Or handle appropriately
            current_policy_obs[key] = tensor_val.squeeze(0).to(
                self.policy_device
            )  # Robomimic expects squeeze

        with torch.no_grad():
            action_np = self.policy(current_policy_obs)  # Get action from policy

        if not isinstance(action_np, np.ndarray):
            print(
                f"[Skill: PressButton] Error: Policy output is not a numpy array (got {type(action_np)})."
            )
            return Action([], metadata={"info":'error'})
        
        return Action(torch.from_numpy(action_np).unsqueeze(0), metadata={"info":'success'})

        
