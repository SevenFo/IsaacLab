"""
Manipulation skills for robot arm control.
Primarily features the 'assemble_object' skill using a pre-trained policy.
"""

import numpy as np
import torch
import math
from torchvision.transforms import Resize, Compose
from torchvision.transforms.functional import convert_image_dtype
from scipy.spatial.transform import Rotation as R

# Ensure os module is imported if not already
import os
from typing import TYPE_CHECKING, List

import isaaclab.utils.math as math_utils
# TODO 不要包含 isaacsim 相关的package，后续可以从 sub process 获取 skill description
# from isaaclab.assets.rigid_object import RigidObject

from ..core.types import SkillType, ExecutionMode, Action, BaseSkill
from ..core.skill_manager import skill_register, get_skill_registry

# Attempt to import Robomimic, make it optional
ROBOMIMIC_AVAILABLE = False
try:
    import robomimic.utils.torch_utils as TorchUtils
    import robomimic.utils.file_utils as FileUtils
    from collections import OrderedDict  # Used by robomimic example

    ROBOMIMIC_AVAILABLE = True
except ImportError:
    print(
        "[Skill] Warning: robomimic library not found. Assemble skill will not be functional."
    )

# Type hinting for Isaac Lab environment if available
if TYPE_CHECKING:
    pass  # Or the specific env type you use


def HWC_to_CHW(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an image tensor from HWC (Height, Width, Channels) format to CHW (Channels, Height, Width).
    This is often required for compatibility with PyTorch models.
    """
    if image.dim() == 3:  # Check if the input is a 3D tensor (HWC)
        image = image.permute(2, 0, 1)  # Change to CHW format
    elif image.dim() == 4:  # Check if the input is a batch of images
        image = image.permute(0, 3, 1, 2)  # Change to BCHW format
    else:
        raise ValueError("Input tensor must be either HWC or BCHW format.")
    return image
    return pixcel_normalize(image)  # Normalize the image to [0, 1] range


def pixcel_normalize(image: torch.Tensor) -> torch.Tensor:
    """Resize and normalize an image tensor to have pixel values in the range [0, 1]."""

    if image.dtype != torch.uint8:
        print(
            f"[Warning] Image is not in uint8 format ({image.dtype}), remaining in the original format."
        )
        return image

    return convert_image_dtype(image, dtype=torch.float32)  # Normalize to [0, 1] range


@skill_register(
    name="open_box",
    skill_type=SkillType.POLICY,  # Could be SkillType.POLICY if you have a separate handler
    execution_mode=ExecutionMode.STEPACTION,
    timeout=300.0,  # 5 minutes, adjust as needed
    criterion={
        "successed": "红色箱子的上盖滑开，并且可以看到箱子里面有一个黄色扳手，否则说明技能正在执行！",
        # "failed": "".join(
        #     ["gripper state that is not reasonable to execute the skill."]
        # ),  # The gripper posisiton is far away from the box and yellow button, ", "or the gripper was pressed on areas other than the yellow button," "or the gripper is lingering (not moving) for several monitoring rounds, ", "or any other
        "progress": "",  # "The gripper is on a reasonable state to execute the skill",  # , such as: moving towards the red box and yellow button, etc.
    },
    requires_env=True,
)
class PressButton(BaseSkill):
    """This skill is used for openning a red box by moving the end-effector, It will automatically move the end-effector to the red box and open it by pressing the yellow button on the box.
    红色箱子盖子上有一个黑色的把手，箱子的盖子可以左右滑动，按下红色的按钮之后，箱子的盖子会滑开，里面有一把黄色的扳手
    Expected params: None, NO NEED TO PASS ANY PARAMS, the skill will automatically get nessessary parameters from the environment.
    """

    def __init__(self, policy_device: str = "cuda", **running_params):
        super().__init__()
        self.policy_device = policy_device
        self.running_params = running_params
        if not ROBOMIMIC_AVAILABLE:
            print(
                "[Skill: press_button] Robomimic library not available. Cannot execute."
            )
            return None

        print("[Skill: press_button] Starting...")

        checkpoint_path = self.cfg.get("model_path", "assets/skills/press.pth")
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
        self.num_steps = 0  # Initialize step counter
        self.policy = policy
        self.policy.start_episode()
        self.resize_fn = Compose([HWC_to_CHW, Resize([256, 256]), pixcel_normalize])

    def select_action(self, obs_dict: dict) -> Action:
        if self.num_steps >= 100:
            print(
                "[Skill: PressButton] Maximum steps reached, stopping skill execution."
            )
            return Action(
                [], metadata={"info": "timeout", "reason": "max_steps_reached"}
            )
        policy_obs_key = "policy"  # Common key in Isaac Lab tasks for policy inputs
        if policy_obs_key not in obs_dict:
            print(
                f"[Skill: PressButton] Error: Key '{policy_obs_key}' not found in initial observations."
            )
            raise ValueError(
                f"Expected key '{policy_obs_key}' not found in obs_dict: {obs_dict.keys()}"
            )

        policy_input_source = obs_dict[policy_obs_key]
        # Resize rgb obs to match policy input requirements
        rgb_obs_keys = [
            key for key in policy_input_source.keys() if key.startswith("camera_")
        ]
        for rbg_key in rgb_obs_keys:
            policy_input_source[rbg_key] = self.resize_fn(policy_input_source[rbg_key])
            # print(
            #     f"[Skill: PressButton] Resized {rbg_key} to shape {policy_input_source[rbg_key].shape}, min, max: {policy_input_source[rbg_key].min()} - {policy_input_source[rbg_key].max()}"
            # )

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
            raise ValueError(f"Expected numpy array from policy, got {type(action_np)}")
        self.num_steps += 1
        return Action(
            torch.from_numpy(action_np).unsqueeze(0),
            metadata={"info": "success", "reason": "none"},
        )


@skill_register(
    name="grasp_spanner",
    skill_type=SkillType.POLICY,  # Could be SkillType.POLICY if you have a separate handler
    execution_mode=ExecutionMode.STEPACTION,
    timeout=300.0,  # 5 minutes, adjust as needed
    criterion={
        # "successed": "the spanner is graspped by the gripper and moving above the box, the spanner must be high enough to avoid collision with the box.",
        "successed": "机械臂抓夹抓住黄色扳手，并离桌面一定距离，避免与箱子发生碰撞。",
        "failed": "".join(["抓夹闭合，但是没有抓住扳手"]),
        "progress": "The gripper is on a reasonable state to execute the skill, such as: grasping the spanner etc.",
    },
    requires_env=True,
)
class GraspSpanner(BaseSkill):
    """
    GRAB & LIFT SPANNER: Grasp tool from red box and lift it

    [HARD RULE] REQUIRED BEFORE EXECUTION:
    MUST HAVE `move end effector to home` TO THIS SPOT AS LAST STEP!
    Which means cant insert anyother skill between `move_to_home` and `grasp_spanner`, anyother skill only canbe inserted before `move_to_home` or after `grasp_spanner`.
    Right example: open_box -> move_to_home -> grasp_spanner -> other skill -> move_to_home -> grasp_spanner
    Wrong example: move_to_home -> open_box -> grasp_spanner -> other skill -> grasp_spanner
    Once you want to execute grasp_spanner, you must have `move_to_home` (or near the home is ok) as last step, otherwise it will fail.
    PARAMETERS: None
    """

    def __init__(self, policy_device: str = "cuda", **running_params):
        super().__init__()
        self.policy_device = policy_device
        self.running_params = running_params
        if not ROBOMIMIC_AVAILABLE:
            print(
                "[Skill: GraspSpanner] Robomimic library not available. Cannot execute."
            )
            return None

        print("[Skill: GraspSpanner] Starting...")

        checkpoint_path = self.cfg.get("model_path", "assets/skills/grasp.pth")

        if not os.path.exists(checkpoint_path):
            print(
                f"[Skill: GraspSpanner] Error: Checkpoint path '{checkpoint_path}' does not exist."
            )
            return None

        print(f"[Skill: GraspSpanner] Using policy device: {policy_device}")
        print(f"[Skill: GraspSpanner] Loading policy from: {checkpoint_path}")

        try:
            policy, _ = FileUtils.policy_from_checkpoint(
                ckpt_path=checkpoint_path, device=policy_device, verbose=True
            )
        except Exception as e:
            print(f"[Skill: GraspSpanner] Error: Failed to load policy: {e}")
            return None

        print("[Skill: GraspSpanner] Policy loaded")

        self.policy = policy
        self.policy.start_episode()
        self.num_steps = 0  # Initialize step counter
        self.resize_fn = Compose([HWC_to_CHW, Resize([256, 256]), pixcel_normalize])

    def select_action(self, obs_dict: dict) -> Action:
        if self.num_steps >= 200:
            print(
                "[Skill: GraspSpanner] Maximum steps reached, stopping skill execution."
            )
            return Action(
                [], metadata={"info": "timeout", "reason": "max_steps_reached"}
            )
        # 早期版本的 GraspSpanner 没有使用和 button 相关的特征，需要进行转化
        policy_obs_key = "policy"  # Common key in Isaac Lab tasks for policy inputs
        if policy_obs_key not in obs_dict:
            print(
                f"[Skill: GraspSpanner] Error: Key '{policy_obs_key}' not found in initial observations."
            )
            raise ValueError(
                f"Expected key '{policy_obs_key}' not found in obs_dict: {obs_dict.keys()}"
            )

        policy_input_source = obs_dict[policy_obs_key]
        rgb_obs_keys = [
            key for key in policy_input_source.keys() if key.startswith("camera_")
        ]
        for rbg_key in rgb_obs_keys:
            policy_input_source[rbg_key] = self.resize_fn(policy_input_source[rbg_key])
        # policy_input_source["object"] = torch.cat(
        #     [
        #         policy_input_source["object"][..., 0:7],
        #         policy_input_source["object"][..., 14:33],
        #     ],
        #     dim=-1,
        # )
        current_policy_obs = OrderedDict()
        for key, tensor_val in policy_input_source.items():
            if not isinstance(tensor_val, torch.Tensor):
                print(
                    f"[Skill: GraspSpanner] Warning: Obs value for key '{key}' is not a Tensor."
                )
                continue  # Or handle appropriately
            current_policy_obs[key] = tensor_val.squeeze(0).to(
                self.policy_device
            )  # Robomimic expects squeeze

        with torch.no_grad():
            action_np = self.policy(current_policy_obs)  # Get action from policy

        if not isinstance(action_np, np.ndarray):
            print(
                f"[Skill: GraspSpanner] Error: Policy output is not a numpy array (got {type(action_np)})."
            )
            raise ValueError(f"Expected numpy array from policy, got {type(action_np)}")
        self.num_steps += 1
        return Action(
            torch.from_numpy(action_np).unsqueeze(0),
            metadata={"info": "success", "reason": "none"},
        )


# --- HELPER FUNCTION (quat_to_axis_angle_torch) from the previous answer ---
# (This remains unchanged)
def quat_to_axis_angle_torch(q: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Convert a quaternion to an axis-angle representation using PyTorch.
    ... (full implementation from previous response) ...
    """
    qw = q[..., 0]
    qv = q[..., 1:4]
    angle = 2 * torch.acos(torch.clamp(qw, -1.0, 1.0))
    sin_half_angle = torch.sin(angle / 2)
    is_not_zero_angle = sin_half_angle.abs() > epsilon
    axis = torch.where(
        is_not_zero_angle.unsqueeze(-1),
        qv / sin_half_angle.unsqueeze(-1),
        torch.tensor([1.0, 0.0, 0.0], device=q.device, dtype=q.dtype),
    )
    axis_angle_vec = axis * angle.unsqueeze(-1)
    return axis_angle_vec


@skill_register(
    name="move_to_target_pose",
    skill_type=SkillType.POLICY,
    execution_mode=ExecutionMode.STEPACTION,
    timeout=300.0,
    enable_monitoring=False,  # Disable monitoring for this skill
    requires_env=True,
    criterion={
        "successed": "This skill always succeeds as it was controlled by the low-level controller, which is always successful.",
        "progress": "",
    },
)
class MoveToTarget(BaseSkill):
    """Moves the robot's end-effector to a specified target pose. in samecase, this skill can be helpful for trying a failed skill.

    Args:
        target_pose (List[float]): The target pose [x, y, z, qw, qx, qy, qz].
    """

    def __init__(
        self,
        policy_device: str = "cuda",
        **running_params,
    ):
        super().__init__()
        if "target_pose" not in running_params:
            assert "target_object" in running_params, "No moving target is specified."
        target_object = running_params.get("target_object", None)
        self.gripper_state: float = running_params.get(
            "gripper_state", 1.0
        )  # gripper_state (float, optional): Command for the gripper. Defaults to 0.0.
        pos_gain: float = running_params.get(
            "pos_gain", 1.0
        )  # pos_gain (float, optional): Proportional gain for position control. Defaults to 1.0
        rot_gain: float = running_params.get(
            "rot_gain", 0.5
        )  # rot_gain (float, optional): Proportional gain for rotation control. Defaults to 0.5
        max_pos_vel: float = running_params.get(
            "max_pos_vel", 0.2
        )  # max_pos_vel (float, optional): Maximum linear velocity (m/s). Defaults to 0.2.
        max_rot_vel: float = running_params.get(
            "max_rot_vel", 0.5
        )  # max_rot_vel (float, optional): Maximum angular velocity (rad/s). Defaults to 0.5.
        offset: List[float] = running_params.get("offset", 0.1)
        self.z_offset = offset
        enable_verification: bool = True
        print("[Skill: MoveToTarget] Initializing...")
        self.device = policy_device
        self.enable_verification = enable_verification
        if target_object is None:
            target_pose: List[float] = running_params["target_pose"]
            if not isinstance(target_pose, list) or len(target_pose) != 7:
                raise ValueError(
                    f"MoveToTarget requires 'target_pose' to be a list of 7 floats, but got {target_pose}"
                )
            self.target_pose = torch.tensor(
                target_pose, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            self.target_pos = self.target_pose[:, :3]
            self.target_quat = self.target_pose[:, 3:7]
            print(
                f"[Skill: MoveToTarget] Target pose set to: {self.target_pose.cpu().numpy().tolist()}"
            )
        else:
            self.target_pose = None
            self.target_object = target_object
            print(f"[Skill: MoveToTarget] Target object set to: {self.target_object}")
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain
        self.max_pos_vel = max_pos_vel
        self.max_rot_vel = max_rot_vel
        self.epsilon = 1e-6
        self.control_mode = "relative"  # Default control mode

        if self.enable_verification:
            print(
                "[Skill: MoveToTarget] VERIFICATION MODE IS ENABLED. SciPy will be used to check torch results."
            )

        print("[Skill: MoveToTarget] Initialized successfully.")

    def _verify_rotation_calculation(
        self, torch_rot_vec: torch.Tensor, error_quat: torch.Tensor
    ):
        """
        Calculates the rotation vector using SciPy and asserts its closeness to the torch version.
        This is a slow, for-testing-only method.
        """
        # 1. Get data on CPU as NumPy arrays
        error_quat_np = error_quat.cpu().numpy()

        # 2. Convert from (w, x, y, z) to SciPy's (x, y, z, w) format
        error_quat_np_xyzw = error_quat_np[:, [1, 2, 3, 0]]

        # 3. Perform SciPy calculation
        scipy_rot_vec_np = R.from_quat(error_quat_np_xyzw).as_rotvec()

        # 4. Compare with the torch implementation's result
        torch_rot_vec_np = torch_rot_vec.cpu().numpy()

        # 5. Assert that they are close. atol (absolute tolerance) handles the near-zero case.
        assert np.allclose(torch_rot_vec_np, scipy_rot_vec_np, atol=1e-2), (
            f"Rotation mismatch!\nTorch: {torch_rot_vec_np}\nSciPy: {scipy_rot_vec_np}"
        )

    def select_action(self, obs_dict: dict) -> Action:
        eef_pose_key = ["eef_pos_gripper", "eef_quat"]
        if "policy" not in obs_dict or not all(
            key in obs_dict["policy"] for key in eef_pose_key
        ):
            print(
                f"[Skill: MoveToTarget] Error: Key '{eef_pose_key}' not found in observations."
            )
            return Action(
                [],
                metadata={
                    "info": "error",
                    "reason": f"{eef_pose_key} has not been tracking in the environment, Non-visual skill can not be executed without it.",
                },
            )

        # 直接获取位置和姿态，无需先拼接再分割
        current_pos = obs_dict["policy"]["eef_pos"].clone()
        current_quat = obs_dict["policy"]["eef_quat"].clone()
        current_gripper_pos = obs_dict["policy"]["gripper_pos"].clone()

        # 确保维度正确并移动到设备
        if current_pos.dim() == 1:
            current_pos = current_pos.unsqueeze(0)
        if current_quat.dim() == 1:
            current_quat = current_quat.unsqueeze(0)

        current_pos = current_pos.to(self.device)
        current_quat = current_quat.to(self.device)

        # --- Position Control (unchanged) ---
        if self.target_pose is None:
            _k = f"{self.target_object}_aabb"
            if _k not in obs_dict["policy"]:
                print(
                    f"[Skill: MoveToTarget] Error: Target object '{self.target_object}' not found in observations."
                )
                return Action(
                    [],
                    metadata={
                        "info": "error",
                        "reason": f"Target object '{self.target_object}' not found in observations.",
                    },
                )
            # Get the target position from the AABB center
            target_aabb = obs_dict["policy"][_k]
            target_center = target_aabb.get_center()
            target_center[2] += self.z_offset  # Add z offset
            target_pos = torch.tensor(target_center, device=self.device).unsqueeze(
                0
            )  # [x, y, z]
            target_quat = current_quat.clone()
            if self.cfg.get("visualize", False):
                print(
                    f"[Skill: MoveToTarget] Target AABB 8p: {np.array2string(np.asarray(target_aabb.get_box_points()), precision=2, separator=', ')}"
                )
        else:
            target_pos = self.target_pos.clone()
            target_quat = self.target_quat.clone()
        if self.cfg.get("visualize", False):
            print(
                f"[Skill: MoveToTarget] Current position: {current_pos.cpu().numpy().tolist()}, Target position: {target_pos.cpu().numpy().tolist()}"
            )
            print(
                f"[Skill: MoveToTarget] Current quaternion: {current_quat.cpu().numpy().tolist()}, Target quaternion: {target_quat.cpu().numpy().tolist()}"
            )

        pos_error = target_pos - current_pos
        desired_pos_vel = pos_error * self.pos_gain
        pos_vel_norm = torch.linalg.norm(desired_pos_vel, dim=1, keepdim=True)
        scaled_pos_vel = desired_pos_vel * torch.min(
            torch.ones_like(pos_vel_norm),
            self.max_pos_vel / (pos_vel_norm + self.epsilon),
        )
        delta_pos = scaled_pos_vel

        # --- Rotation Control (with verification logic) ---
        # Calculate the error quaternion using PyTorch
        error_quat = math_utils.quat_mul(
            target_quat, math_utils.quat_conjugate(current_quat)
        )
        error_quat = torch.where(error_quat[:, 0:1] < 0, -error_quat, error_quat)

        # Convert error quaternion to an axis-angle vector using our high-performance torch function
        total_rot_vec = quat_to_axis_angle_torch(error_quat)

        # -- VERIFICATION BLOCK --
        if self.enable_verification:
            # This function will throw an AssertionError if the results don't match
            self._verify_rotation_calculation(total_rot_vec, error_quat)
        # -- END VERIFICATION BLOCK --

        # check if the delta is too small
        if (
            torch.linalg.norm(total_rot_vec, dim=-1) + torch.linalg.norm(delta_pos)
            < 0.1
        ):
            if self.control_mode == "relative":
                zero_action = torch.zeros(
                    size=(current_pos.shape[0], 7), device=self.device
                )
            else:
                raise NotImplementedError(
                    "[Skill: MoveToTarget] Absolute control mode is not implemented yet."
                )
                # TODO 假设绝对控制的 action 为 joint posisition,
                zero_action = torch.zeros(
                    size=(current_pos.shape[0], 7), device=self.device
                )
            zero_action[..., -1] = self.gripper_state
            print("[Skill: MoveToTarget] Delta is too small, skill finished.")
            return Action(
                zero_action,
                metadata={
                    "info": "finished",
                    "reason": f"end-effector is close enough to {self.target_pose if self.target_pose is not None else self.target_object}, skill finished.",
                },
            )

        # Continue with the pure torch result
        desired_rot_vel = total_rot_vec * self.rot_gain
        rot_vel_norm = torch.linalg.norm(desired_rot_vel, dim=1, keepdim=True)
        scaled_rot_vel = desired_rot_vel * torch.min(
            torch.ones_like(rot_vel_norm),
            self.max_rot_vel / (rot_vel_norm + self.epsilon),
        )
        delta_rot = scaled_rot_vel

        # --- Combine and create final action ---
        delta_pose_action = torch.cat([delta_pos, delta_rot], dim=1)
        current_gripper_action = current_gripper_pos.mean(dim=1, keepdim=True).to(
            self.device
        )
        current_gripper_action[current_gripper_action > 0.5] = 1.0
        current_gripper_action[current_gripper_action <= 0.5] = 0.0
        action = torch.cat(
            [delta_pose_action, current_gripper_action],
            dim=1,
        )
        return Action(action, metadata={"info": "success", "reason": "none"})


registry = get_skill_registry()
registry.register_skill(
    skill_type=SkillType.POLICY,
    execution_mode=ExecutionMode.STEPACTION,
    name="move_to_target_object",
    function=MoveToTarget,
    description="""Moves the robot's end-effector to the center pose of the target object.
This skill can not understand the visual information, only automatically retrieve the target object's center from the environment observation by the specified target object name.
So the target object related observation must be provided in the environment observation.
For example, `object_tracking` skill can be used to add the target object tracking information to the environment observation. However you need to call the `object_tracking` skill before this skill.
Args:
    target_object (str): The name of the target object.
    gripper_state (float, optional): Command for the gripper after arrivival. 1 means open, 0 means close. default is 1.""",
    timeout=300,
    enable_monitoring=False,  # Disable monitoring for this skill
    requires_env=True,
)


class AliceControl(BaseSkill):
    """Moves the robot's end-effector to a specified target pose. in samecase, this skill can be helpful for trying a failed skill.

    Args:
        target_pose (List[float]): The target pose [x, y, z, qw, qx, qy, qz].
    """

    def __init__(
        self,
        alice_right_forearm_rigid_entity: "RigidObject",
        policy_device: str = "cuda",
    ):
        super().__init__()

        # self.alice_right_forearm_rigid_entity: "RigidObject" = (
        #     alice_right_forearm_rigid_entity
        # )
        # bodies = self.alice_right_forearm_rigid_entity.body_names
        # print(f"[Skill: AliceControl] Found bodies in Alice's right forearm: {bodies}")
        # print("[Skill: AliceControl] Initialized successfully.")

    def apply_action(self, env) -> bool:
        # 获取当前关节目标位置
        current_target = env.scene["alice"].data.joint_pos_target.clone()

        # D6Joint_1:1 对应的是索引 2
        joint_idx = 2

        # 定义增量和限制范围（弧度）
        increment = math.radians(0.1)
        lower_limit = math.radians(55)
        upper_limit = math.radians(75)

        # 初始化或检查方向张量（每个实例一个方向）
        if (
            not hasattr(self, "direction")
            or self.direction.shape[0] != current_target.shape[0]
        ):
            self.direction = torch.ones(
                current_target.shape[0], device=env.device, dtype=torch.float32
            )

        # 获取当前角度
        current_angles = current_target[:, joint_idx]

        # 计算下一个角度
        next_angles = current_angles + increment * self.direction

        # 检测是否超出边界并反转方向
        exceeded_upper = next_angles > upper_limit
        exceeded_lower = next_angles < lower_limit

        # 更新方向（到达边界则反转）
        self.direction[exceeded_upper | exceeded_lower] *= -1

        # 限制角度在范围内
        next_angles = torch.clamp(next_angles, lower_limit, upper_limit)

        # 更新目标角度
        current_target[:, joint_idx] = next_angles

        # 将目标位置设置回环境
        env.scene["alice"].set_joint_position_target(current_target)

        # 写入模拟器，立即生效
        env.scene["alice"].write_joint_state_to_sim(
            current_target,
            torch.zeros_like(current_target, device=env.device),
        )
        return True
