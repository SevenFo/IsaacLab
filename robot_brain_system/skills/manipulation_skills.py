"""
Manipulation skills for robot arm control.
Primarily features the 'assemble_object' skill using a pre-trained policy.
"""

import numpy as np
import requests
import torch
import math
import random
import time
import msgpack
import msgpack_numpy as msgnp
from torchvision.transforms import Compose
from torchvision.transforms.functional import convert_image_dtype
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict

msgnp.patch()  # 服务端也要启用 numpy 支持

# Ensure os module is imported if not already
from typing import TYPE_CHECKING, Any, Dict, List

import isaaclab.utils.math as math_utils

# TODO 不要包含 isaacsim 相关的 package，后续可以从 sub process 获取 skill description
# from isaaclab.assets.rigid_object import RigidObject

from ..core.types import SkillType, ExecutionMode, Action, BaseSkill
from ..core.skill_manager import skill_register, get_skill_registry
from ..skills.observation_skills import ObjectTracking
from robot_brain_system.ui.console import global_console


# Type hinting for Isaac Lab environment if available
if TYPE_CHECKING:
    pass  # Or the specific env type you use


def HWC_to_CHW(image: torch.Tensor, normalize=False) -> torch.Tensor:
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
    return pixcel_normalize(image) if normalize else image


def pixcel_normalize(image: torch.Tensor) -> torch.Tensor:
    """Resize and normalize an image tensor to have pixel values in the range [0, 1]."""

    if image.dtype != torch.uint8:
        global_console.log(
            "skill",
            f"[Warning] Image is not in uint8 format ({image.dtype}), remaining in the original format.",
        )
        return image

    return convert_image_dtype(image, dtype=torch.float32)  # Normalize to [0, 1] range


def _sample_object_poses(
    pose_range: dict[str, tuple[float, float]] = {},
):
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    pose_list = []

    sample = [random.uniform(range[0], range[1]) for range in range_list]
    pose_list.append(sample)

    return pose_list


def quat_normalize(quat):
    return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)


class GO1Client:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def predict_action(self, payload: Dict[str, Any]) -> np.ndarray | None:
        """使用 msgpack 传输 numpy 数据，效率高且支持二进制"""
        try:
            # 1. 使用 msgpack 序列化（自动处理 numpy 数组）
            packed = msgpack.packb(payload, use_bin_type=True)
        except Exception as e:
            global_console.log("skill", f"[GO1Client] Failed to serialize payload: {e}")
            return None

        # 2. 发送二进制数据
        response = requests.post(
            f"http://{self.host}:{self.port}/act",
            data=packed,
            headers={"Content-Type": "application/msgpack"},
            timeout=10.0,  # 添加超时保护
        )

        if response.status_code != 200:
            global_console.log(
                "skill", f"[GO1Client] Request failed: {response.status_code}"
            )
            global_console.log("skill", f"[GO1Client] Error: {response.text}")
            return None

        # 3. 解包响应
        try:
            action = msgpack.unpackb(response.content, raw=False)

            action = np.array(action, copy=True)

            # action[...,0], action[...,1] = action[...,1], -action[...,0]
            # action[...,3], action[...,4] = action[...,4], -action[3]

            return action

        except Exception as e:
            global_console.log(
                "skill", f"[GO1Client] Failed to deserialize response: {e}"
            )
            return None


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
    """这个技能能够自动控制机械臂的末端执行器移动到红色箱子前方，并按下箱子上的按钮，从而打开箱子。
    这个技能的执行前提是箱子**完全**位于绿色的标志框中，并且箱子按钮朝向正对机械臂基座方向，两个条件缺一不可，否则请使用其他技能移动箱子到合适位置再执行该技能
    执行该技能无需先移动到红色箱子上方。
    Expected params: None, NO NEED TO PASS ANY PARAMS, the skill will automatically get necessary parameters from the environment.
    """

    def __init__(self, policy_device: str = "cuda", **running_params):
        super().__init__()
        self.policy_device = policy_device
        self.running_params = running_params

        global_console.log("skill", "[Skill: PressButton] Starting...")

        self.action_client = GO1Client(
            host=self.cfg.get("host", "localhost"), port=self.cfg.get("port", 2000)
        )
        self.ctrl_freq = 10.0

        global_console.log("skill", "[Skill: PressButton] Policy loaded")
        self.num_steps = 0
        self.trans_fn = Compose([lambda x: x])
        self.camera_keys = {
            "camera_side": "top",
            "camera_wrist": "left",
        }  # isaaclab obs -> policy input
        self.state_key = "joint_pos"
        self.state_index = (..., [0, 1, 2, 3, 4, 5, -1])

    def switch_to_heavy_box(
        self,
    ):
        """Switch the light box to heavy box in specified envs."""
        env = self.env.unwrapped
        if getattr(env, "switch_heavybox", None) is not None:
            global_console.log(
                "skill",
                f"[Skill: PressButton] Environment already switched to heavy box. env.switch_heavybox={getattr(env, 'switch_heavybox')}",
            )
            return
        light_box = env.scene["box"]
        heavy_box = env.scene["heavy_box"]
        spanner = env.scene["spanner"]
        cur_env = 0
        # Get the current pose and velocity of the light box
        light_box_pos = light_box.data.root_pos_w[cur_env : cur_env + 1]
        light_box_quat = light_box.data.root_quat_w[cur_env : cur_env + 1]
        light_box_lin_vel = light_box.data.root_lin_vel_w[cur_env : cur_env + 1]
        light_box_ang_vel = light_box.data.root_ang_vel_w[cur_env : cur_env + 1]

        # Get heavy box default state
        heavy_box_default_state = heavy_box.data.default_root_state[
            cur_env : cur_env + 1
        ].clone()

        # Set the heavy box to the same pose and velocity
        heavy_box_target_pose = torch.cat([light_box_pos, light_box_quat], dim=-1)

        # 设置扳手的位置
        spanner_rel_pos = torch.tensor(
            [0, -0.04, 0.001], device=env.device
        )  # asset2 相对于 asset1 的位置偏移
        spanner_rel_rot = math_utils.quat_from_euler_xyz(  # asset2 相对于 asset1 的旋转 (绕 x 轴 90 度)
            torch.tensor([0.0], device=env.device),
            torch.tensor([0.0], device=env.device),
            torch.tensor([math.pi / 2], device=env.device),
        )
        spanner_pos = heavy_box_target_pose[:, :3] - spanner_rel_pos.unsqueeze(0)
        # 旋转 = asset1 的旋转 * 相对旋转
        spanner_orient = math_utils.quat_mul(
            heavy_box_target_pose[:, 3:7], spanner_rel_rot
        )

        heavy_box.write_root_pose_to_sim(
            heavy_box_target_pose,
            env_ids=torch.tensor([cur_env], device=env.device),
        )
        heavy_box.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device),
        )
        # Move light box to heavy box's default position
        light_box.write_root_pose_to_sim(
            heavy_box_default_state[:, :7],
            env_ids=torch.tensor([cur_env], device=env.device),
        )
        light_box.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device),
        )
        # # 设置 asset2 的位姿
        spanner.write_root_pose_to_sim(
            torch.cat([spanner_pos, spanner_orient], dim=-1),
            env_ids=torch.tensor([0], device=env.device),
        )
        spanner.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([0], device=env.device),
        )
        setattr(env, "switch_heavybox", True)

    def initialize(self, *args, **kwargs):
        """
        Initialize the skill by setting up the environment and zero action.
        This method is called before the skill starts executing.
        """
        global_console.log(
            "skill", f"[Skill: PressButton] Initializing skill with:{args},{kwargs}"
        )
        super().initialize(*args, **kwargs)
        self.switch_to_heavy_box()
        env = self.env
        robot = env.scene["robot"]
        joint_pos = robot.data.default_joint_pos
        joint_vel = robot.data.default_joint_vel

        robot.set_joint_position_target(joint_pos)
        robot.set_joint_velocity_target(joint_vel)

        robot.write_joint_state_to_sim(
            joint_pos,
            torch.zeros_like(joint_vel, device=env.device),
        )
        self.zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )
        self.zero_action[..., -1] = +1.0  # Ensure gripper is open
        global_console.log("skill", "[Skill: PressButton] switched to heavy box")
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(self.zero_action)
            time.sleep(0.1)
        global_console.log("skill", "[Skill: PressButton] Initialization complete.")
        return obs

    def reset_env(self, random_reset=True):
        # reset spanner pos
        env = self.env
        spanner = env.scene["spanner"]
        box = env.scene["box"]
        # 定义 asset2 相对于 asset1 的初始位置偏移和旋转
        rel_pos = torch.tensor(
            [0.0, -0.04, 0.001], device=env.device
        )  # asset2 相对于 asset1 的位置偏移
        rel_rot = math_utils.quat_from_euler_xyz(  # asset2 相对于 asset1 的旋转 (绕 x 轴 90 度)
            torch.tensor([0.0], device=env.device),
            torch.tensor([0.0], device=env.device),
            torch.tensor([math.pi / 2], device=env.device),
        )
        pose1 = _sample_object_poses(
            pose_range={
                "x": (1.215, 1.215),
                "y": (-3.45, -3.45),
                "z": (2.9, 2.9),
                "yaw": (0, 0),
            }
        )[0]
        pose_tensor1 = torch.tensor([pose1], device=env.device)
        positions1 = (
            box.data.root_state_w[0:1, 0:3]
            if not random_reset
            else pose_tensor1[:, 0:3] + env.scene.env_origins[0, 0:3]
        )
        orientations1 = (
            box.data.root_state_w[0:1, 3:7]
            if not random_reset
            else math_utils.quat_from_euler_xyz(
                pose_tensor1[:, 3], pose_tensor1[:, 4], pose_tensor1[:, 5]
            )
        )
        if random_reset:
            # 设置 asset1 的位姿
            box.write_root_pose_to_sim(
                torch.cat([positions1, orientations1], dim=-1),
                env_ids=torch.tensor([0], device=env.device),
            )
            box.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([0], device=env.device),
            )
        for i in range(1):
            obs, reward, terminated, truncated, info = env.step(self.zero_action)
            time.sleep(0.1)

        positions2 = positions1 - rel_pos.unsqueeze(0)
        # 旋转 = asset1 的旋转 * 相对旋转
        orientations2 = math_utils.quat_mul(orientations1, rel_rot)

        # # 设置 asset2 的位姿
        spanner.write_root_pose_to_sim(
            torch.cat([positions2, orientations2], dim=-1),
            env_ids=torch.tensor([0], device=env.device),
        )
        spanner.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([0], device=env.device),
        )
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(self.zero_action)
            time.sleep(0.1)

    def select_action(self, obs_dict: dict) -> Action:
        if self.num_steps >= 200:
            global_console.log(
                "skill",
                "[Skill: PressButton] Maximum steps reached, stopping skill execution.",
            )
            return Action(
                [], metadata={"info": "timeout", "reason": "max_steps_reached"}
            )
        policy_obs_key = "policy"  # Common key in Isaac Lab tasks for policy inputs
        if policy_obs_key not in obs_dict:
            global_console.log(
                "skill",
                f"[Skill: PressButton] Error: Key '{policy_obs_key}' not found in initial observations.",
            )
            raise ValueError(
                f"Expected key '{policy_obs_key}' not found in obs_dict: {obs_dict.keys()}"
            )

        policy_input_source = obs_dict[policy_obs_key]
        rgb_obs_keys = [
            key for key in policy_input_source.keys() if key in self.camera_keys.keys()
        ]
        current_policy_obs = OrderedDict()
        for rgb_obs_key in rgb_obs_keys:
            current_policy_obs[self.camera_keys[rgb_obs_key]] = self.trans_fn(
                policy_input_source[rgb_obs_key]
            )
        current_policy_obs["state"] = policy_input_source[self.state_key][
            self.state_index
        ][:, None, :]

        # 转换为 numpy 数组（msgpack 可以直接处理）
        for key, tensor_val in current_policy_obs.items():
            if isinstance(tensor_val, torch.Tensor):
                current_policy_obs[key] = tensor_val.squeeze(0).cpu().numpy()

        current_policy_obs["ctrl_freqs"] = np.array([self.ctrl_freq], dtype=np.float32)
        current_policy_obs["instruction"] = "Press the button attached to the box."

        action_np = self.action_client.predict_action(
            current_policy_obs
        )  # Get action from policy

        if not isinstance(action_np, np.ndarray):
            global_console.log(
                "skill",
                f"[Skill: PressButton] Error: Policy output is not a numpy array (got {type(action_np)}).",
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
        "successed": "机械臂抓夹抓住黄色扳手即可",
        "failed": "".join(
            [
                "grasp has closed while not graspping the spanner (if the gripper is not closed, it will be considered as progress). The prerequisite for failed is that the gripper has already closed."
            ]
        ),
        "progress": "The gripper is on a reasonable state to execute the skill, such as: The robot arm moved towards the spanner. The robot attempted to grasp the object. The robot gripper is opening. The robot is trying to grasp the spanner etc.",
    },
    requires_env=True,
)
class GraspSpanner(BaseSkill):
    """
    GRAB & LIFT SPANNER: Grasp spanner and lift it. 如果场景中没有扳手，请勿执行该技能。
    This skill can not move spanner to any other position. If you want to move spanner to other position, plz call other skill

    PARAMETERS: None
    """

    def __init__(self, policy_device: str = "cuda", **running_params):
        super().__init__()
        self.policy_device = policy_device
        self.running_params = running_params

        global_console.log("skill", "[Skill: GraspSpanner] Starting...")

        self.action_client = GO1Client(
            host=self.cfg.get("host", "localhost"), port=self.cfg.get("port", 2000)
        )
        self.ctrl_freq = 10.0

        global_console.log("skill", "[Skill: GraspSpanner] Policy loaded")

        self.num_steps = 0
        self.trans_fn = Compose([lambda x: x])
        self.camera_keys = {
            "camera_side": "top",
            "camera_wrist": "left",
        }  # isaaclab obs -> policy input
        self.state_key = "joint_pos"
        self.state_index = (..., [0, 1, 2, 3, 4, 5, -1])

    def initialize(self, *args, **kwargs):
        """
        Initialize the skill by setting up the environment and zero action.
        This method is called before the skill starts executing.
        """
        global_console.log(
            "skill",
            "[Skill: GraspSpanner] Initializing skill make joints move to default positions...",
        )
        super().initialize(*args, **kwargs)
        env = self.env
        self.zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )
        self.zero_action[..., -1] = +1.0  # Ensure gripper is open

        robot = env.scene["robot"]

        joint_pos = robot.data.default_joint_pos
        joint_vel = robot.data.default_joint_vel

        robot.set_joint_position_target(joint_pos)
        robot.set_joint_velocity_target(joint_vel)
        robot.write_joint_state_to_sim(
            joint_pos,
            torch.zeros_like(joint_vel, device=env.device),
        )
        for i in range(30):
            obs, reward, terminated, truncated, info = env.step(self.zero_action)

        return obs

    def reset_env(self, random_reset=True):
        # reset spanner pos
        env = self.env
        spanner = env.scene["spanner"]
        box = env.scene["box"]
        # 定义 asset2 相对于 asset1 的初始位置偏移和旋转
        rel_pos = torch.tensor(
            [0.0, -0.04, 0.001], device=env.device
        )  # asset2 相对于 asset1 的位置偏移
        rel_rot = math_utils.quat_from_euler_xyz(  # asset2 相对于 asset1 的旋转 (绕 x 轴 90 度)
            torch.tensor([0.0], device=env.device),
            torch.tensor([0.0], device=env.device),
            torch.tensor([math.pi / 2], device=env.device),
        )
        pose1 = _sample_object_poses(
            pose_range={
                "x": (1.215, 1.215),
                "y": (-3.45, -3.45),
                "z": (2.9, 2.9),
                "yaw": (0, 0),
            }
        )[0]
        pose_tensor1 = torch.tensor([pose1], device=env.device)
        positions1 = (
            box.data.root_state_w[0:1, 0:3]
            if not random_reset
            else pose_tensor1[:, 0:3] + env.scene.env_origins[0, 0:3]
        )
        orientations1 = (
            box.data.root_state_w[0:1, 3:7]
            if not random_reset
            else math_utils.quat_from_euler_xyz(
                pose_tensor1[:, 3], pose_tensor1[:, 4], pose_tensor1[:, 5]
            )
        )
        if random_reset:
            # 设置 asset1 的位姿
            box.write_root_pose_to_sim(
                torch.cat([positions1, orientations1], dim=-1),
                env_ids=torch.tensor([0], device=env.device),
            )
            box.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([0], device=env.device),
            )
        for i in range(1):
            obs, reward, terminated, truncated, info = env.step(self.zero_action)
            time.sleep(0.1)

        positions2 = positions1 - rel_pos.unsqueeze(0)
        # 旋转 = asset1 的旋转 * 相对旋转
        orientations2 = math_utils.quat_mul(orientations1, rel_rot)

        # # 设置 asset2 的位姿
        spanner.write_root_pose_to_sim(
            torch.cat([positions2, orientations2], dim=-1),
            env_ids=torch.tensor([0], device=env.device),
        )
        spanner.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([0], device=env.device),
        )
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(self.zero_action)
            time.sleep(0.1)

    def select_action(self, obs_dict: dict) -> Action:
        if self.num_steps >= 300:
            global_console.log(
                "skill",
                "[Skill: GraspSpanner] Maximum steps reached, stopping skill execution.",
            )
            return Action(
                [], metadata={"info": "timeout", "reason": "max_steps_reached"}
            )
        policy_obs_key = "policy"  # Common key in Isaac Lab tasks for policy inputs
        if policy_obs_key not in obs_dict:
            global_console.log(
                "skill",
                f"[Skill: GraspSpanner] Error: Key '{policy_obs_key}' not found in initial observations.",
            )
            raise ValueError(
                f"Expected key '{policy_obs_key}' not found in obs_dict: {obs_dict.keys()}"
            )

        policy_input_source = obs_dict[policy_obs_key]
        rgb_obs_keys = [
            key for key in policy_input_source.keys() if key in self.camera_keys.keys()
        ]
        current_policy_obs = OrderedDict()
        for rgb_obs_key in rgb_obs_keys:
            current_policy_obs[self.camera_keys[rgb_obs_key]] = self.trans_fn(
                policy_input_source[rgb_obs_key]
            )
        current_policy_obs["state"] = policy_input_source[self.state_key][
            self.state_index
        ][:, None, :]

        # 转换为 numpy 数组（msgpack 可以直接处理）
        for key, tensor_val in current_policy_obs.items():
            if isinstance(tensor_val, torch.Tensor):
                current_policy_obs[key] = tensor_val.squeeze(0).cpu().numpy()

        current_policy_obs["ctrl_freqs"] = np.array([self.ctrl_freq], dtype=np.float32)
        current_policy_obs["instruction"] = "Grasp the spanner inside the box"

        action_np = self.action_client.predict_action(
            current_policy_obs
        )  # Get action from policy

        if not isinstance(action_np, np.ndarray):
            global_console.log(
                "skill",
                f"[Skill: GraspSpanner] Error: Policy output is not a numpy array (got {type(action_np)}).",
            )
            raise ValueError(f"Expected numpy array from policy, got {type(action_np)}")
        self.num_steps += 1
        return Action(
            torch.from_numpy(action_np).unsqueeze(0),
            metadata={"info": "success", "reason": "none"},
        )


@skill_register(
    name="move_box_to_suitable_position",
    skill_type=SkillType.POLICY,  # Could be SkillType.POLICY if you have a separate handler
    execution_mode=ExecutionMode.STEPACTION,
    timeout=300.0,  # 5 minutes, adjust as needed
    criterion={
        # "successed": "the spanner is graspped by the gripper and moving above the box, the spanner must be high enough to avoid collision with the box.",
        "successed": "箱子位于桌面的绿色的标志框中",
        "failed": "".join([""]),
        "progress": "箱子被提起并且正在移动到目标位置等合理的状态",
    },
    requires_env=True,
)
class LiftAndMoveBox(BaseSkill):
    """这个技能用于将箱子提起并移动到目标框中，并且让箱子的朝向转动到易于机械臂按下按钮打开箱子的方向。该技能的前提是视野中已经有一个箱子，如果没有箱子则无法执行该技能。

    Expected params: None, NO NEED TO PASS ANY PARAMS.
    """

    def __init__(self, policy_device: str = "cuda", **running_params):
        super().__init__()
        self.policy_device = policy_device
        self.running_params = running_params

        global_console.log("skill", "[Skill: LiftAndMoveBox] Starting...")
        self.action_client = GO1Client(
            host=self.cfg.get("host", "localhost"), port=self.cfg.get("port", 2000)
        )

        global_console.log("skill", "[Skill: LiftAndMoveBox] Policy loaded")

        self.num_steps = 0
        self.trans_fn = Compose([lambda x: x])
        self.camera_keys = {
            "camera_side": "top",
            "camera_wrist": "left",
        }  # isaaclab obs -> policy input
        self.state_key = "joint_pos"
        self.state_index = (..., [0, 1, 2, 3, 4, 5, -1])

    def initialize(self, *args, **kwargs):
        """
        Initialize the skill by setting up the environment and zero action.
        This method is called before the skill starts executing.
        """
        global_console.log(
            "skill",
            "[Skill: LiftAndMoveBox] Initializing skill make joints move to default positions...",
        )
        super().initialize(*args, **kwargs)
        env = self.env
        zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )
        zero_action[..., -1] = +1.0  # Ensure gripper is open

        robot = env.scene["robot"]

        joint_pos = robot.data.default_joint_pos
        joint_vel = robot.data.default_joint_vel

        robot.set_joint_position_target(joint_pos)
        robot.set_joint_velocity_target(joint_vel)
        robot.write_joint_state_to_sim(
            joint_pos,
            torch.zeros_like(joint_vel, device=env.device),
        )
        for i in range(1):
            obs, reward, terminated, truncated, info = env.step(zero_action)

        self.ctrl_freq = 10.0
        # global_console.log("skill",
        #     f"[Skill: LiftAndMoveBox] env decimation: {env.cfg.decimation}, dim_dt: {env.cfg.sim.dt}"
        # )
        self.max_steps = int(self.cfg.get("timeout", 60.0) * self.ctrl_freq)
        return obs

    def select_action(self, obs_dict: dict) -> Action:
        if self.num_steps >= self.max_steps:
            global_console.log(
                "skill",
                "[Skill: LiftAndMoveBox] Maximum steps reached, stopping skill execution.",
            )
            return Action(
                [], metadata={"info": "timeout", "reason": "max_steps_reached"}
            )
        policy_obs_key = "policy"  # Common key in Isaac Lab tasks for policy inputs
        if policy_obs_key not in obs_dict:
            global_console.log(
                "skill",
                f"[Skill: LiftAndMoveBox] Error: Key '{policy_obs_key}' not found in initial observations.",
            )
            raise ValueError(
                f"Expected key '{policy_obs_key}' not found in obs_dict: {obs_dict.keys()}"
            )

        policy_input_source = obs_dict[policy_obs_key]
        rgb_obs_keys = [
            key for key in policy_input_source.keys() if key in self.camera_keys.keys()
        ]
        current_policy_obs = OrderedDict()
        for rgb_obs_key in rgb_obs_keys:
            current_policy_obs[self.camera_keys[rgb_obs_key]] = self.trans_fn(
                policy_input_source[rgb_obs_key]
            )
        current_policy_obs["state"] = policy_input_source[self.state_key][
            self.state_index
        ][:, None, :]
        # 转换为 numpy 数组（msgpack 可以直接处理）
        for key, tensor_val in current_policy_obs.items():
            if isinstance(tensor_val, torch.Tensor):
                current_policy_obs[key] = tensor_val.squeeze(0).cpu().numpy()

        current_policy_obs["ctrl_freqs"] = np.array([self.ctrl_freq], dtype=np.float32)
        current_policy_obs["instruction"] = "Move the object to the target position."

        action_np = self.action_client.predict_action(
            current_policy_obs
        )  # Get action from policy

        if not isinstance(action_np, np.ndarray):
            global_console.log(
                "skill",
                f"[Skill: LiftAndMoveBox] Error: Policy output is not a numpy array (got {type(action_np)}).",
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
        "successed": "This skill always succeeds as it was controlled by the low-level controller, which is always successful. This skill does not require visual evidence to double-check even if there is no visual evidence just consider there are enough visual evidence.",
        "progress": "",
    },
)
class MoveToTarget(BaseSkill):
    """Moves the robot's end-effector to a specified target pose. in samecase, this skill can be helpful for trying a failed skill.

    Args:
        target_pose (List[float]): The target pose [x, y, z, qw, qx, qy, qz].
    """

    # gripper_state (List[float], optional): 1 means open, 0 means close. First element is the gripper state at the start of the skill and this continues until the skill is completed, second element is the gripper state at the end of the skill. Defaults to [1.0, 1.0]."""

    def __init__(
        self,
        policy_device: str = "cuda",
        **running_params,
    ):
        super().__init__()
        if "target_pose" not in running_params:
            assert "target_object" in running_params or "target" in running_params, (
                "No moving target is specified."
            )
        target_object = running_params.get("target_object", None) or running_params.get(
            "target", None
        )
        self.gripper_state: float = int(
            running_params.get("gripper_state", 1.0)
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
        offset: List[float] = running_params.get("offset", 0.08)
        self.dt = running_params.get("dt", 0.3)  # 默认 50Hz
        self.z_offset = offset
        enable_verification: bool = True
        global_console.log("skill", "[Skill: MoveToTarget] Initializing...")
        self.device = policy_device
        self.enable_verification = enable_verification
        self.target_object = target_object
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
            global_console.log(
                "skill",
                f"[Skill: MoveToTarget] Target pose set to: {self.target_pose.cpu().numpy().tolist()}",
            )
        else:
            self.target_pose = None
            self.target_object = target_object
            global_console.log(
                "skill",
                f"[Skill: MoveToTarget] Target object set to: {self.target_object}",
            )
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain
        self.max_pos_vel = max_pos_vel
        self.max_rot_vel = max_rot_vel
        self.epsilon = 1e-6
        self.control_mode = "relative"  # Default control mode
        self.delay_steps = 2  # Number of steps to delay before ending the skill
        self.step_counter = 0  # Initialize step counter
        self.finishing = False
        if self.enable_verification:
            global_console.log(
                "skill",
                "[Skill: MoveToTarget] VERIFICATION MODE IS ENABLED. SciPy will be used to check torch results.",
            )

        global_console.log("skill", "[Skill: MoveToTarget] Initialized successfully.")

    def initialize(self, *args, **kwargs):
        """
        Initialize the skill by setting up the environment and zero action.
        This method is called before the skill starts executing.
        """
        global_console.log(
            "skill",
            "[Skill: GraspSpanner] Initializing skill make joints move to default positions...",
        )
        obs_dict = super().initialize(*args, **kwargs)
        env = self.env
        if self.target_object is not None:
            self.obj_tracker = ObjectTracking(env=env, target_object=self.target_object)
            obs_dict = self.obj_tracker.initialize(env)
        self.filtered_target_pos = None
        self.filtered_target_quat = None
        return obs_dict

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
        if self.target_object is not None:
            obs_dict = self.obj_tracker(obs_dict)
        self.step_counter += 1
        eef_pose_key = ["eef_pos_gripper", "eef_quat"]
        if "policy" not in obs_dict or not all(
            key in obs_dict["policy"] for key in eef_pose_key
        ):
            global_console.log(
                "skill",
                f"[Skill: MoveToTarget] Error: Key '{eef_pose_key}' not found in observations.",
            )
            return Action(
                [],
                metadata={
                    "info": "error",
                    "reason": f"{eef_pose_key} has not been tracking in the environment, Non-visual skill can not be executed without it.",
                },
            )

        # 直接获取位置和姿态，无需先拼接再分割
        current_pos = obs_dict["policy"]["eef_pos_gripper"].clone().to(self.env.device)
        current_quat = obs_dict["policy"]["eef_quat"].clone().to(self.env.device)
        current_gripper_pos = (
            obs_dict["policy"]["gripper_pos"].clone().to(self.env.device)
        )

        # 确保维度正确并移动到设备
        if current_pos.dim() == 1:
            current_pos = current_pos.unsqueeze(0)
        if current_quat.dim() == 1:
            current_quat = current_quat.unsqueeze(0)

        current_pos = current_pos.to(self.device)
        current_quat = current_quat.to(self.device)

        alpha = self.cfg.get("target_smoothing_alpha", 0.3)

        # --- Position Control ---
        if self.target_pose is None:
            _k = f"{self.target_object}_aabb"
            if _k not in obs_dict["policy"]:
                global_console.log(
                    "skill",
                    f"[Skill: MoveToTarget] Error: Target object '{self.target_object}' not found in observations.",
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
            raw_target_pos = torch.tensor(target_center, device=self.device).unsqueeze(
                0
            )  # [x, y, z]
            # self.env.move_target_visualizer.visualize(target_center[None, ...])
            raw_target_quat = current_quat.clone()
            if self.cfg.get("visualize", False):
                # global_console.log("skill",
                #     f"[Skill: MoveToTarget] Target AABB 8p: {np.array2string(np.asarray(target_aabb.get_box_points()), precision=2, separator=', ')}"
                # )
                ...
        else:
            raw_target_pos = self.target_pos.clone()
            raw_target_quat = self.target_quat.clone()

        # --- 初始化平滑目标（如果尚未初始化）---
        if not hasattr(self, "filtered_target_pos") or self.filtered_target_pos is None:
            self.filtered_target_pos = raw_target_pos.clone()
        if (
            not hasattr(self, "filtered_target_quat")
            or self.filtered_target_quat is None
        ):
            self.filtered_target_quat = raw_target_quat.clone()

        self.filtered_target_pos = (
            alpha * raw_target_pos + (1 - alpha) * self.filtered_target_pos
        )
        self.filtered_target_quat = (
            alpha * raw_target_quat + (1 - alpha) * self.filtered_target_quat
        )

        self.filtered_target_quat = quat_normalize(self.filtered_target_quat)
        # 使用平滑后的目标
        target_pos = raw_target_pos
        target_quat = raw_target_quat

        if self.cfg.get("visualize", False):
            # global_console.log("skill",
            #     f"[Skill: MoveToTarget] Current position: {np.array2string(current_pos.cpu().numpy(), precision=2)}, Target position: {np.array2string(target_pos.cpu().numpy(), precision=2)}"
            # )
            # global_console.log("skill",
            #     f"[Skill: MoveToTarget] Current quaternion: {np.array2string(current_quat.cpu().numpy(), precision=2)}, Target quaternion: {np.array2string(target_quat.cpu().numpy(), precision=2)}"
            # )
            ...
        self.filtered_target_pos = (
            alpha * raw_target_pos + (1 - alpha) * self.filtered_target_pos
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
            torch.linalg.norm(total_rot_vec, dim=-1) + torch.linalg.norm(pos_error)
            < 0.03
            or self.finishing
        ):
            self.finishing = True
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
            global_console.log(
                "skill", "[Skill: MoveToTarget] Delta is too small, skill finishing."
            )

            if self.delay_steps > 0:
                self.delay_steps -= 1
                return Action(
                    zero_action,
                    metadata={
                        "info": "success",
                        "reason": f"end-effector is close enough to {self.target_pose if self.target_pose is not None else self.target_object}, skill finishing.",
                    },
                )
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
        delta_pose_action = torch.cat([delta_pos * self.dt, delta_rot * self.dt], dim=1)

        # 在移动过程中保持当前抓夹状态，而不是设置为期望状态
        # 判断当前抓夹是开着还是关着的：如果平均绝对值大于阈值，说明抓夹是打开的
        current_gripper_open_level = current_gripper_pos.abs().mean(dim=1, keepdim=True)
        # 保持当前状态：大于 0.5 认为是关闭的 (-1.0)，小于等于 0.5 认为是打开的 (+1.0)
        gripper_action = torch.where(current_gripper_open_level > 0.5, -1.0, 1.0)
        gripper_action = -torch.ones_like(gripper_action, device=gripper_action.device)

        R_world_to_base = torch.tensor(
            [
                [0, -1, 0],  # 世界 X → 基 Y
                [1, 0, 0],  # 世界 Y → 基-X
                [0, 0, 1],  # 世界 Z → 基 Z
            ],
            device=delta_pose_action.device,
            dtype=delta_pose_action.dtype,
        )

        delta_pose_action[..., :3] = delta_pose_action[..., :3] @ R_world_to_base
        delta_pose_action[..., 3:] = delta_pose_action[..., 3:] @ R_world_to_base
        action = torch.cat(
            [delta_pose_action, gripper_action],
            dim=1,
        )
        global_console.log(
            "skill",
            f"[MoveToTarget]: action: {np.array2string(action.cpu().numpy(), precision=2)}",
        )
        return Action(action, metadata={"info": "success", "reason": "none"})

    def cleanup(self):
        self.obj_tracker.cleanup()


registry = get_skill_registry()
registry.register_skill(
    skill_type=SkillType.POLICY,
    execution_mode=ExecutionMode.STEPACTION,
    name="move_to_target_object",
    function=MoveToTarget,
    description="""Moves the robot's end-effector to the pose of the target object.
Args:
    target_object (str): The name of the target object.
    gripper_state (float, optional): Command for the gripper after arrivival. 1 means open, -1 means close. default is 1. For example, if you want to drop the object after arrivival, set it to 1. If you want to grasp the object after arrivival, set it to -1.""",
    timeout=300,
    enable_monitoring=False,  # Disable monitoring for this skill
    requires_env=True,
)
# For example, `object_tracking` skill can be used to add the target object tracking information to the environment observation. However you need to call the `object_tracking` skill before this skill, so that the necessary observation information can be available when this skill starts.
# This skill can not understand the visual information, only automatically retrieve the target object's pose from the environment observation by the specified target object name.
# So the target object related observation must be provided in the environment observation.
