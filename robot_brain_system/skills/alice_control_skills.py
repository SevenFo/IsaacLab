import threading
import time
import torch
import math
from PIL import Image
import asyncio
from isaaclab.utils.math import euler_xyz_from_quat, quat_mul

from ..core.types import BaseSkill
from ..utils.logging_utils import get_logger
from .MotionCaptureReceiverv2 import MotionCaptureReceiver
from robot_brain_system.core.env_proxy import EnvProxy
from robot_brain_system.ui.console import global_console


class AliceControl(BaseSkill):
    """控制 Alice 机器人的技能。支持固定动作执行 (fixed) 和 动捕实时重定向 (dynamic)。"""

    def __init__(self, device="cuda", mode="fixed"):
        super().__init__()
        self.logger = get_logger("skills.alice_control")
        self.motion_capture_receiver = None
        self.latest_mocap_data = {}
        self.mode = mode  # "fixed" or "dynamic"
        self._mocap_lock = (
            threading.Lock()
        )  # To safely update/read mocap data across threads
        # 内部状态
        self.joint_names_to_indices = {}
        self.mocap_map = {}
        self._mapping_initialized = False

    def _update_mocap_data(self, data: dict):
        """[回调函数] 当接收器收到新数据时，此方法会被异步调用。"""
        with self._mocap_lock:
            self.latest_mocap_data = data

    def initialize(
        self,
        env: EnvProxy,
    ):
        super().initialize(env)
        global_console.log("skill", "Initializing AliceControl Skill...")
        alice = self.env.scene["alice"]

        # --- 1. 定义机器人结构映射 (核心元数据) ---
        # 创建一个从动捕骨骼到机器人关节的映射字典。
        # 这是整个重定向逻辑的核心，您需要在这里定义所有映射关系。
        # 使用 find_joints 来获取实际索引。
        # 假设关节名称格式为 "D6Joint_LinkName:0", "D6Joint_LinkName:1", "D6Joint_LinkName:2" 对于 D6Joint,
        # 和 "RevoluteJoint_LinkName" 对于 RevoluteJoint。
        # 动捕欧拉角索引：0=X (roll), 1=Y (pitch), 2=Z (yaw) 从 euler_xyz_from_quat
        # axis_mapping: (mocap_axis_idx, scale, offset_rad)
        self.prelim_map = {
            "RightArm": {
                "joint_names": [
                    "D6Joint_RightArm:0",
                    "D6Joint_RightArm:1",
                    "D6Joint_RightArm:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "RightForeArm": {
                "joint_names": [
                    "D6Joint_RightForeArm:0",
                    "D6Joint_RightForeArm:1",
                    "D6Joint_RightForeArm:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "RightHand": {
                "joint_names": [
                    "D6Joint_RightWrist:0",
                    "D6Joint_RightWrist:1",
                    "D6Joint_RightWrist:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            # --- 右手手指 ---
            "RightHandThumb1": {
                "joint_names": [
                    "D6Joint_RightHandThumb1:0",
                    "D6Joint_RightHandThumb1:1",
                    "D6Joint_RightHandThumb1:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "RightHandThumb2": {
                "joint_names": ["RevoluteJoint_RightHandThumb2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandThumb3": {
                "joint_names": ["RevoluteJoint_RightHandThumb3"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            # "RightHandIndex": {
            #     "joint_names": ["RevoluteJoint_RightHandIndex"],
            #     "axis_mapping": [(2, 1.0, 0.0)],
            # },
            "RightHandIndex1": {
                "joint_names": ["RevoluteJoint_RightHandIndex1"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandIndex2": {
                "joint_names": ["RevoluteJoint_RightHandIndex2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandIndex3": {
                "joint_names": ["RevoluteJoint_RightHandIndex3"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            # "RightInHandMiddle": {
            #     "joint_names": ["RevoluteJoint_RightInHandMiddle"],
            #     "axis_mapping": [(2, 1.0, 0.0)],
            # },
            "RightHandMiddle1": {
                "joint_names": ["RevoluteJoint_RightHandMiddle1"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandMiddle2": {
                "joint_names": ["RevoluteJoint_RightHandMiddle2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandMiddle3": {
                "joint_names": ["RevoluteJoint_RightHandMiddle3"],
                "axis_mapping": [(2, 1.0, 0.0)],  # 对于手指，只需要映射 z 轴角度
            },
            # "RightInHandRing": {
            #     "joint_names": ["RevoluteJoint_RightInHandRing"],
            #     "axis_mapping": [(2, 1.0, 0.0)],
            # },
            "RightHandRing1": {
                "joint_names": ["RevoluteJoint_RightHandRing1"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandRing2": {
                "joint_names": ["RevoluteJoint_RightHandRing2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandRing3": {
                "joint_names": ["RevoluteJoint_RightHandRing3"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            # "RightInHandPinky": {
            #     "joint_names": ["RevoluteJoint_RightInHandPinky"],
            #     "axis_mapping": [(2, 1.0, 0.0)],
            # },
            "RightHandPinky1": {
                "joint_names": ["RevoluteJoint_RightHandPinky1"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandPinky2": {
                "joint_names": ["RevoluteJoint_RightHandPinky2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "RightHandPinky3": {
                "joint_names": ["RevoluteJoint_RightHandPinky3"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftArm": {
                "joint_names": [
                    "D6Joint_LeftArm:0",
                    "D6Joint_LeftArm:1",
                    "D6Joint_LeftArm:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "LeftForeArm": {
                "joint_names": [
                    "D6Joint_LeftForeArm:0",
                    "D6Joint_LeftForeArm:1",
                    "D6Joint_LeftForeArm:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "LeftHand": {
                "joint_names": [
                    "D6Joint_LeftWrist:0",
                    "D6Joint_LeftWrist:1",
                    "D6Joint_LeftWrist:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            # --- 左手手指 ---
            "LeftHandThumb1": {
                "joint_names": ["RevoluteJoint_LeftHandThumb1"],
                "axis_mapping": [(0, 1.0, 0.0)],
            },
            "LeftHandThumb2": {
                "joint_names": ["RevoluteJoint_LeftHandThumb2"],
                "axis_mapping": [(0, 1.0, 0.0)],
            },
            "LeftHandThumb3": {
                "joint_names": ["RevoluteJoint_LeftHandThumb3"],
                "axis_mapping": [(0, 1.0, 0.0)],
            },
            # "LeftInHandIndex": {
            #     "joint_names": ["RevoluteJoint_LeftInHandIndex"],
            #     "axis_mapping": [(2, 1.0, 0.0)],
            # },
            "LeftHandIndex1": {
                "joint_names": ["RevoluteJoint_LeftHandIndex1"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftHandIndex2": {
                "joint_names": ["RevoluteJoint_LeftHandIndex2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftHandIndex3": {
                "joint_names": ["RevoluteJoint_LeftHandIndex3"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            # "LeftInHandMiddle": {
            #     "joint_names": ["RevoluteJoint_LeftInHandMiddle"],
            #     "axis_mapping": [(2, 1.0, 0.0)],
            # },
            "LeftHandMiddle1": {
                "joint_names": ["RevoluteJoint_LeftHandMiddle1"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftHandMiddle2": {
                "joint_names": ["RevoluteJoint_LeftHandMiddle2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftHandMiddle3": {
                "joint_names": ["RevoluteJoint_LeftHandMiddle3"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            # "LeftInHandRing": {
            #     "joint_names": ["RevoluteJoint_LeftInHandRing"],
            #     "axis_mapping": [(0, 1.0, 0.0)],
            # },
            "LeftHandRing1": {
                "joint_names": ["RevoluteJoint_LeftHandRing1"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftHandRing2": {
                "joint_names": ["RevoluteJoint_LeftHandRing2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftHandRing3": {
                "joint_names": ["RevoluteJoint_LeftHandRing3"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            # "LeftInHandPinky": {
            #     "joint_names": ["RevoluteJoint_LeftInHandPinky"],
            #     "axis_mapping": [(2, 1.0, 0.0)],
            # },
            "LeftHandPinky1": {
                "joint_names": ["RevoluteJoint_LeftHandPinky1"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftHandPinky2": {
                "joint_names": ["RevoluteJoint_LeftHandPinky2"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            "LeftHandPinky3": {
                "joint_names": ["RevoluteJoint_LeftHandPinky3"],
                "axis_mapping": [(2, 1.0, 0.0)],
            },
            # --- 躯干 (Spine1, Neck, Neck1 已移除) ---
            "Spine": {
                "joint_names": [
                    "D6Joint_Spine:0",
                    "D6Joint_Spine:1",
                    "D6Joint_Spine:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "Spine2": {
                "joint_names": [
                    "D6Joint_Spine2:0",
                    "D6Joint_Spine2:1",
                    "D6Joint_Spine2:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "Head": {
                "joint_names": ["D6Joint_Neck:0", "D6Joint_Neck:1", "D6Joint_Neck:2"],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            # --- 右腿 ---
            "RightUpLeg": {
                "joint_names": [
                    "D6Joint_RightUpLeg:0",
                    "D6Joint_RightUpLeg:1",
                    "D6Joint_RightUpLeg:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "RightLeg": {
                "joint_names": ["RevoluteJoint_RightKnee"],
                "axis_mapping": [(0, 1.0, 0.0)],
            },
            "RightFoot": {
                "joint_names": ["RevoluteJoint_RightAnkle"],
                "axis_mapping": [(0, 1.0, 0.0)],
            },
            # --- 左腿 ---
            "LeftUpLeg": {
                "joint_names": [
                    "D6Joint_LeftUpLeg:0",
                    "D6Joint_LeftUpLeg:1",
                    "D6Joint_LeftUpLeg:2",
                ],
                "axis_mapping": [(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)],
            },
            "LeftLeg": {
                "joint_names": ["RevoluteJoint_LeftKnee"],
                "axis_mapping": [(0, 1.0, 0.0)],
            },
            "LeftFoot": {
                "joint_names": ["RevoluteJoint_LeftAnkle"],
                "axis_mapping": [(0, 1.0, 0.0)],
            },
            # Hips 骨骼通常用于驱动根骨骼 (root state)，不在关节映射中处理，代码中已有单独逻辑
        }

        # --- 2. 动态获取索引 ---
        # 展平所有关节名
        all_requested_names = []
        for info in self.prelim_map.values():
            all_requested_names.extend(info["joint_names"])

        # 去重并查找
        all_requested_names = list(set(all_requested_names))
        try:
            response = alice.find_joints(all_requested_names)
            if response.get("success", False):
                indices, names = response["result"]
                self.joint_names_to_indices = dict(zip(names, indices))
                global_console.log(
                    "skill",
                    f"Successfully discovered {len(names)} joints from simulation.",
                )
            else:
                self.joint_names_to_indices = self.cfg.get("joint_name_to_indices", {})
                global_console.log(
                    "warning", "Simulation find_joints failed. Used config backup."
                )
        except Exception as e:
            self.logger.error(f"Error during joint discovery: {e}")

        # --- 3. 构建动捕专用的结构化 Map ---
        # 此时已经有了索引，直接填充 self.mocap_map
        for bone_name, info in self.prelim_map.items():
            valid_indices = []
            for j_name in info["joint_names"]:
                if j_name in self.joint_names_to_indices:
                    valid_indices.append(self.joint_names_to_indices[j_name])

            if len(valid_indices) == len(info["joint_names"]):
                self.mocap_map[bone_name] = {
                    "joint_indices": valid_indices,
                    "axis_mapping": info["axis_mapping"],
                }

        # --- 4. 基础机器人状态初始化 (Root/Joints) ---
        global_console.log("skill", "Alice robot initialized to starting pose.")
        self.initial_root_state = torch.tensor(
            [
                [-1.8, -2.5, 2.8, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0]
            ],  # [-1.8, 0.95, 2.8, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0]
            device=self.env.device,
            dtype=torch.float32,
        )
        alice.write_root_state_to_sim(self.initial_root_state)

        self.init_alice_joint_position_target = torch.zeros_like(
            alice.data.joint_pos_target
        )
        alice.set_joint_position_target(
            self.init_alice_joint_position_target,
        )
        # write the joint state to sim to directly change the joint position at one step
        alice.write_joint_state_to_sim(
            self.init_alice_joint_position_target,
            torch.zeros_like(self.init_alice_joint_position_target, device=env.device),
        )

        # --- 5. 初始化动捕服务器 (仅 dynamic 模式) ---
        if self.mode == "dynamic":
            self._compile_mocap_mapping()
            global_console.log("skill", "Starting MotionCaptureReceiver server...")
            self.motion_capture_receiver = MotionCaptureReceiver(
                data_handler_callback=self._update_mocap_data
            )

            # 启动服务器在单独线程中运行，以避免阻塞主模拟循环
            def run_mocap_server():
                asyncio.run(self.motion_capture_receiver.start_server())

            self._server_thread = threading.Thread(target=run_mocap_server, daemon=True)
            self._server_thread.start()
            global_console.log(
                "skill",
                "MotionCaptureReceiver server task has been scheduled in background thread.",
            )

        # 保存初始状态快照
        obs = env.update(return_obs=True)
        self.init_root_pose = self.initial_root_state[:, :3].clone().squeeze()
        self.init_root_quat = self.initial_root_state[:, 3:7].clone().squeeze()
        frame = (
            obs.data["policy"]["camera_left"][0].cpu().numpy()
        )  # Ensure the observation is processed
        Image.fromarray(frame).save("alice_initialization.png")
        return obs

    def move_to_operation_position(self):
        """将 Alice 移动到操作位置的示例方法。"""
        self.mode = "fixed"
        alice = self.env.scene["alice"]
        operation_position = torch.tensor(
            [[0.4067, -3.2, 2.7, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0]],
            device=self.env.device,
            dtype=torch.float32,
        )
        init_joint_position_target = self.init_alice_joint_position_target.clone()
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightArm:0"]
        ] = 0.0
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightArm:1"]
        ] = math.radians(66.7)
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightArm:2"]
        ] = math.radians(50.7)
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightForeArm:0"]
        ] = 0.0
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightForeArm:1"]
        ] = math.radians(25.9)
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightForeArm:2"]
        ] = math.radians(-23.2)
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightWrist:0"]
        ] = math.radians(-141.8)
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightWrist:1"]
        ] = math.radians(-11.0)
        init_joint_position_target[
            :, self.joint_names_to_indices["D6Joint_RightWrist:2"]
        ] = math.radians(-41.7)
        alice.write_root_state_to_sim(operation_position)
        alice.set_joint_position_target(init_joint_position_target)
        # write the joint state to sim to directly change the joint position at one step
        alice.write_joint_state_to_sim(
            init_joint_position_target,
            torch.zeros_like(init_joint_position_target, device=self.env.device),
        )
        for i in range(10):
            obs = self.env.update(return_obs=True)
        frame = (
            obs.data["policy"]["camera_left"][0].cpu().numpy()
        )  # Ensure the observation is processed
        Image.fromarray(frame).save("alice_operation_position.png")
        global_console.log("skill", "Moved Alice to operation position.")

    def move_to_play_position(self):
        """将 Alice 移动到操作位置的示例方法。"""
        self.mode = "dynamic"
        alice = self.env.scene["alice"]
        operation_position = torch.tensor(
            [[-1.8, 0.95, 2.8, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0]],
            device=self.env.device,
            dtype=torch.float32,
        )
        alice.write_root_state_to_sim(operation_position)
        self.env.update(return_obs=False)
        global_console.log("skill", "Moved Alice to play position.")

    def _apply_fixed_action(self, return_obs: bool = False, update_sim: bool = False):
        """执行固定的演示动作（例如让右臂在一定范围内摆动）。"""
        env = self.env
        alice = env.scene["alice"]

        # 检查是否有关节索引
        joint_key = "D6Joint_RightArm:1"
        if joint_key not in self.joint_names_to_indices:
            return env.update(return_obs=return_obs) if update_sim else None

        current_target = alice.data.joint_pos_target.clone()
        joint_idx = self.joint_names_to_indices[joint_key]

        # 运动逻辑
        increment = math.radians(0.001)
        lower_limit = math.radians(55)
        upper_limit = math.radians(75)

        if not hasattr(self, "direction"):
            self.direction = torch.ones(current_target.shape[0], device=env.device)

        # 获取当前角度
        current_angles = current_target[:, joint_idx]
        # 计算下一个角度
        next_angles = current_angles + increment * self.direction
        # 边界检测与反转
        exceeded_upper = next_angles > upper_limit
        exceeded_lower = next_angles < lower_limit
        # 更新方向（到达边界则反转）
        self.direction[exceeded_upper | exceeded_lower] *= -1
        # 限制角度在范围内
        next_angles = torch.clamp(next_angles, lower_limit, upper_limit)
        # 更新目标角度
        current_target[:, joint_idx] = next_angles

        # 将目标位置设置回环境
        alice.set_joint_position_target(current_target)
        # 写入模拟器，立即生效
        alice.write_joint_state_to_sim(
            current_target,
            torch.zeros_like(current_target, device=self.env.device),
        )
        if update_sim:
            return self.env.update(return_obs=return_obs)
        return None

    def _compile_mocap_mapping(self):
        """
        将字典形式的 mocap_map 编译为扁平的 Tensor，以便进行向量化计算。
        只需在第一次收到数据或初始化时运行一次。
        """
        if not hasattr(self, "mocap_map") or not self.mocap_map:
            return

        # 临时列表用于收集数据
        map_bone_names = []  # 需要读取的动捕骨骼名称列表
        robot_joint_indices = []  # 对应的机器人关节索引
        source_bone_indices = []  # 该关节对应 map_bone_names 中的第几个骨骼
        source_axis_indices = []  # 取欧拉角的哪个轴 (0,1,2)
        scales = []
        offsets = []

        # 遍历生成的 mocap_map
        # mocap_map 结构：bone_name -> {"joint_indices": [...], "axis_mapping": [(axis, scale, offset), ...]}
        for bone_idx, (bone_name, info) in enumerate(self.mocap_map.items()):
            map_bone_names.append(bone_name)

            joint_idxs = info["joint_indices"]
            axis_maps = info["axis_mapping"]

            # 展开这个骨骼下的所有关节映射
            for i in range(len(joint_idxs)):
                robot_joint_indices.append(joint_idxs[i])
                source_bone_indices.append(bone_idx)  # 指向 map_bone_names 的索引

                axis, scale, offset = axis_maps[i]
                source_axis_indices.append(axis)
                scales.append(scale)
                offsets.append(offset)

        # 转换为 Tensor 并移动到 GPU (假设 self.env.device 可用)
        device = self.env.device
        self._map_bone_names = map_bone_names  # Python 列表，用于从字典取数据

        # 形状为 (N_mappings,)
        self._t_robot_joint_idxs = torch.tensor(
            robot_joint_indices, device=device, dtype=torch.long
        )
        self._t_source_bone_idxs = torch.tensor(
            source_bone_indices, device=device, dtype=torch.long
        )
        self._t_source_axis_idxs = torch.tensor(
            source_axis_indices, device=device, dtype=torch.long
        )
        self._t_scales = torch.tensor(scales, device=device, dtype=torch.float32)
        self._t_offsets = torch.tensor(offsets, device=device, dtype=torch.float32)

        self._mapping_initialized = True
        global_console.log(
            "skill",
            f"Mocap mapping compiled: {len(self._t_robot_joint_idxs)} joint mappings optimized.",
        )

    def _apply_dynamic_action(self, return_obs: bool = False, update_sim=False):
        cal_t = time.time()
        env = self.env

        with self._mocap_lock:
            if not self.latest_mocap_data:
                return None
            local_mocap_data = self.latest_mocap_data

        alice = env.scene["alice"]

        if not self._mapping_initialized:
            return self.env.update(return_obs=return_obs) if update_sim else None

        if not hasattr(self, "init_hips_pos") and "Hips" in local_mocap_data:
            pos_data = local_mocap_data["Hips"]["local_position"]
            self.init_hips_pos = (
                torch.tensor(pos_data, device=env.device, dtype=torch.float32) / 100
            )

        # --- 向量化核心开始 ---

        # 1. 批量提取四元数
        # 我们按照 self._map_bone_names 的顺序构建一个 (B, 4) 的 Tensor
        quat_list = []
        try:
            for name in self._map_bone_names:
                # 假设 local_rotation 是 list 或 array [w, x, y, z]
                quat_list.append(local_mocap_data[name]["local_rotation"])
        except KeyError as e:
            self.logger.warning(f"Missing bone data: {e}")
            return self.env.update(return_obs=return_obs) if update_sim else None

        # 转换为 Tensor: (N_unique_bones, 4)
        all_quats = torch.tensor(quat_list, device=env.device, dtype=torch.float32)

        # 2. 批量计算欧拉角 -> (N_unique_bones, 3)
        # euler_xyz_from_quat 接受 (..., 4) 并返回 (..., 3) (Roll, Pitch, Yaw)
        roll, pitch, yaw = euler_xyz_from_quat(all_quats)
        all_eulers = torch.stack([roll, pitch, yaw], dim=-1).squeeze()

        # 3. 使用预计算的索引提取需要的值 -> (N_mappings,)
        # 从 all_eulers 中选出对应的骨骼 (source_bone) 和对应的轴 (source_axis)
        target_raw_angles = all_eulers[
            self._t_source_bone_idxs, self._t_source_axis_idxs
        ]

        # 4. 批量应用线性变换 (Scale & Offset)
        target_vals = target_raw_angles * self._t_scales + self._t_offsets

        # 5. 批量处理角度限制 (Wrap to -pi, pi)
        # 替代原本的 if target > pi ... elif target < -pi ...
        # 公式：(x + pi) % (2*pi) - pi
        target_vals = torch.remainder(target_vals + math.pi, 2 * math.pi) - math.pi

        # 6. 将计算结果一次性赋值给目标 Tensor
        # current_target: (num_envs, num_joints)
        current_target = self.init_alice_joint_position_target.clone()

        # 假设只有一个环境实例，或者所有环境都做相同动作：
        # 使用索引将 target_vals 填入 current_target 的对应列
        current_target[:, self._t_robot_joint_idxs] = target_vals

        # --- 向量化核心结束 ---

        # --- 处理 Root (Hips) ---
        if "Hips" in local_mocap_data:
            pos_data = local_mocap_data["Hips"]["local_position"]
            rot_data = local_mocap_data["Hips"]["local_rotation"]

            root_pos_raw = (
                torch.tensor(pos_data, device=env.device, dtype=torch.float32) / 100
            )
            root_quat_raw = torch.tensor(
                rot_data, device=env.device, dtype=torch.float32
            )

            hip_pos_diff = root_pos_raw - self.init_hips_pos
            root_pos = self.init_root_pose + hip_pos_diff[[2, 0, 1]]  # Swap axes logic
            root_quat = quat_mul(self.init_root_quat, root_quat_raw)

            root_lin_vel = torch.zeros(3, device=env.device)
            root_ang_vel = torch.zeros(3, device=env.device)
            root_state = torch.cat(
                [root_pos, root_quat, root_lin_vel, root_ang_vel]
            ).unsqueeze(0)
        else:
            root_state = self.initial_root_state.clone()

        end_cal_t = time.time()
        # global_console.log("skill", f"Vectorized Cal time: {end_cal_t - cal_t:.6f} sec")

        with env:
            alice.set_joint_position_target(current_target)
            alice.set_joint_effort_target(
                torch.zeros_like(current_target, device=env.device)
            )
            alice.set_joint_velocity_target(
                torch.zeros_like(current_target, device=env.device)
            )
            alice.write_joint_state_to_sim(
                current_target,
                torch.zeros_like(current_target, device=env.device),
            )
            alice.write_root_state_to_sim(root_state)
            obs = self.env.update(return_obs=return_obs) if update_sim else None

        return obs

    def update(self, return_obs: bool = False, update_sim: bool = False):
        if self.mode == "fixed":
            return self._apply_fixed_action(
                return_obs=return_obs, update_sim=update_sim
            )
        elif self.mode == "dynamic":
            obs = self._apply_dynamic_action(
                return_obs=return_obs, update_sim=update_sim
            )
            if not obs:
                # self.logger.warning("No valid mocap data received, skipping action.")
                return None
        return obs
