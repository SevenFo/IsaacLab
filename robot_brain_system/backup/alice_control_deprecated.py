"""Archived AliceControlDeprecated skill kept for reference.

This module retains the legacy AliceControl implementation that directly interacted
with the environment without the EnvProxy abstraction. It is no longer imported
by default but can be reviewed or revived if needed.
"""

import asyncio
import math
from typing import Optional

import torch
from PIL import Image

from isaaclab.utils.math import euler_xyz_from_quat

from ..core.types import BaseSkill
from ..utils.logging_utils import get_logger
from .MotionCaptureReceiverv2 import MotionCaptureReceiver


class AliceControlDeprecated(BaseSkill):
    """Legacy Alice control skill kept for historical reference."""

    def __init__(self, mode: str = "fixed"):
        super().__init__()
        self.logger = get_logger("skills.alice_control")
        self.motion_capture_receiver: Optional[MotionCaptureReceiver] = None
        self.latest_mocap_data = {}
        self.mode = mode  # "fixed" or "dynamic"

    def _update_mocap_data(self, data: dict):
        """[回调函数] 当接收器收到新数据时，此方法会被异步调用。"""
        self.latest_mocap_data = data

    def initialize(self, env, zero_action):
        self.logger.info("Initializing AliceControl Skill...")

        self.logger.info("Alice robot initialized to starting pose.")
        alice = env.scene["alice"]
        alice.write_root_state_to_sim(
            torch.tensor(
                [[0.4067, -3.3, 1.8, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0]],
                device=env.device,
                dtype=torch.float32,
            ),
        )
        init_alice_joint_position_target = torch.zeros_like(alice.data.joint_pos_target)
        init_alice_joint_position_target[:, :9] = torch.tensor(
            [
                0.0,
                math.radians(66.7),
                math.radians(50.7),
                0.0,
                math.radians(25.9),
                math.radians(-23.2),
                math.radians(-141.8),
                math.radians(-11.0),
                math.radians(-41.7),
            ],
            device=env.device,
        )
        alice.set_joint_position_target(
            init_alice_joint_position_target,
        )
        alice.write_joint_state_to_sim(
            init_alice_joint_position_target,
            torch.zeros_like(init_alice_joint_position_target, device=env.device),
        )

        if self.motion_capture_receiver is None:
            self.logger.info("Starting MotionCaptureReceiver server...")
            self.motion_capture_receiver = MotionCaptureReceiver(
                data_handler_callback=self._update_mocap_data
            )
            asyncio.ensure_future(self.motion_capture_receiver.start_server())
            self.logger.info("MotionCaptureReceiver server task has been scheduled.")

        zero_action[..., -1] = -1.0
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(zero_action)
        frame = obs["policy"]["camera_left"][0].cpu().numpy()
        Image.fromarray(frame).save("alice_initialization.png")
        return obs

    def _apply_fixed_action(self, env):
        current_target = env.scene["alice"].data.joint_pos_target.clone()
        joint_idx = 2
        increment = math.radians(0.1)
        lower_limit = math.radians(55)
        upper_limit = math.radians(75)

        if (
            not hasattr(self, "direction")
            or self.direction.shape[0] != current_target.shape[0]
        ):
            self.direction = torch.ones(
                current_target.shape[0], device=env.device, dtype=torch.float32
            )

        current_angles = current_target[:, joint_idx]
        next_angles = current_angles + increment * self.direction
        exceeded_upper = next_angles > upper_limit
        exceeded_lower = next_angles < lower_limit
        self.direction[exceeded_upper | exceeded_lower] *= -1
        next_angles = torch.clamp(next_angles, lower_limit, upper_limit)
        current_target[:, joint_idx] = next_angles

        env.scene["alice"].set_joint_position_target(current_target)
        env.scene["alice"].write_joint_state_to_sim(
            current_target,
            torch.zeros_like(current_target, device=env.device),
        )

    def _create_mocap_to_robot_map(self, env):
        return {
            "RightArm": {
                "joint_indices": [0, 1, 2],
                "axis_mapping": [
                    (0, 1.0, 0.0),
                    (2, 1.0, 0.0),
                    (1, 1.0, 0.0),
                ],
            },
            "RightForeArm": {
                "joint_indices": [3, 4, 5],
                "axis_mapping": [
                    (0, 1.0, 0.0),
                    (2, 1.0, 0.0),
                    (1, 1.0, 0.0),
                ],
            },
            "RightHand": {
                "joint_indices": [6, 7, 8],
                "axis_mapping": [
                    (0, 1.0, 0.0),
                    (2, 1.0, 0.0),
                    (1, 1.0, 0.0),
                ],
            },
        }

    def _apply_rotation_map(self, current_target, mocap_bone_name, mapping_info):
        if mocap_bone_name not in self.latest_mocap_data:
            return

        try:
            rot_data = self.latest_mocap_data[mocap_bone_name]["local_rotation"]
            mocap_quat = torch.tensor(
                rot_data, dtype=torch.float32, device=current_target.device
            )
            roll, pitch, yaw = euler_xyz_from_quat(mocap_quat.unsqueeze(0))
            euler_rad = (roll.item(), pitch.item(), yaw.item())
            joint_indices = mapping_info["joint_indices"]
            axis_mappings = mapping_info["axis_mapping"]
            for i in range(len(joint_indices)):
                joint_idx = joint_indices[i]
                mocap_axis_idx, scale, offset = axis_mappings[i]
                target_rad = euler_rad[mocap_axis_idx] * scale + offset
                current_target[:, joint_idx] = target_rad
        except (KeyError, IndexError) as exc:
            self.logger.warning(
                f"Error processing mocap map for '{mocap_bone_name}': {exc}"
            )

    def _apply_dynamic_action(self, env):
        if not self.latest_mocap_data:
            return False

        alice = env.scene["alice"]
        current_target = alice.data.joint_pos_target.clone()
        if not hasattr(self, "mocap_map"):
            self.mocap_map = self._create_mocap_to_robot_map(env)
            self.logger.info(f"Alice DOF Names: {alice.dof_names}")
            self.logger.info(f"Created Mocap Map: {self.mocap_map}")

        for mocap_bone_name, mapping_info in self.mocap_map.items():
            self._apply_rotation_map(current_target, mocap_bone_name, mapping_info)

        alice.set_joint_position_target(current_target)
        return True

    def apply_action(self, env) -> bool:
        if self.mode == "fixed":
            self._apply_fixed_action(env)
        elif self.mode == "dynamic":
            if not self._apply_dynamic_action(env):
                self.logger.warning("No valid mocap data received, skipping action.")
        return True
