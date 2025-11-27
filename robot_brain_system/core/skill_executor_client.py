"""
Client-side Skill Executor.
Runs in robot_brain_system process, uses EnvProxy to access remote environment.
"""

from datetime import datetime
import os
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from PIL import Image
import cv2

from robot_brain_system.core.types import SkillStatus
from robot_brain_system.core.skill_manager import SkillRegistry
from robot_brain_system.core.env_proxy import EnvProxy
from robot_brain_system.utils.metric_utils import with_time
from robot_brain_system.skills.alice_control_skills import AliceControl


class SkillExecutorClient:
    """
    Client-side skill executor that runs skills locally but accesses remote environment.
    This replaces the server-side SkillExecutor for the decoupled architecture.

    Key differences from original SkillExecutor:
    - Runs in robot_brain_system process (not server)
    - Uses EnvProxy instead of direct env access
    - Skills execute locally, only env queries are remote
    """

    def __init__(self, skill_registry: SkillRegistry, env_proxy: EnvProxy):
        """
        Initialize client-side skill executor.

        Args:
            skill_registry: Registry of available skills
            env_proxy: Proxy for remote environment access
        """
        self.enable_alice = False
        self.registry = skill_registry
        self.env_proxy = env_proxy

        # Current skill state
        self.current_skill: Optional[Callable] = None
        self.current_skill_name: Optional[str] = None
        self.current_skill_params: Optional[Dict[str, Any]] = None
        self.preaction_skills: list[Callable] = []

        # Execution status
        self.status: SkillStatus = SkillStatus.IDLE
        self.status_info: str = ""

        # Device info
        self.env_device = env_proxy.device
        self.skill_success_item_map = {}

        # Auto-execution thread
        self.execution_thread: Optional[threading.Thread] = None
        self.is_executing = False
        self.execution_lock = threading.Lock()
        self.skill_step_interval = 0.0001  # 50 Hz (20ms)

        # alice
        self.alice_control = AliceControl(mode="dynamic")
        self.alice_control.initialize(self.env_proxy)
        self.enable_alice = True

        # --- [新增] 录制状态变量 ---
        self.is_recording = False
        self.recording_keys = []
        self.recording_root_dir = ""
        self.recording_idx = 0

    def initialize_skill(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        policy_device=None,
        obs_dict: dict = {},
    ):
        """
        Initialize and start skill execution.

        Args:
            skill_name: Name of skill to execute
            parameters: Skill parameters
            policy_device: Device for policy (default: env device)
            obs_dict: Initial observation dictionary

        Returns:
            Tuple of (success: bool, obs_dict: dict)
        """
        self.status = SkillStatus.NOT_STARTED
        self.status_info = ""
        self._reset_current_skill_state()

        if not policy_device:
            policy_device = self.env_device

        if self.is_running():
            print(
                f"[SkillExecutorClient] Another skill '{self.current_skill_name}' is already running. Terminating it."
            )
            self.terminate_current_skill()

        skill_def = self.registry.get_skill(skill_name)
        if not skill_def:
            print(
                f"[SkillExecutorClient] Skill '{skill_name}' not found, available skills: {self.registry.list_skills()}"
            )
            self.status = SkillStatus.FAILED
            return False, obs_dict

        try:
            from robot_brain_system.core.types import ExecutionMode

            if skill_name == "move_box_to_suitable_position":
                self.env_proxy._simulator.set_env_decimation(10)
            else:
                self.env_proxy._simulator.set_env_decimation(5)

            # if skill_name == "move_to_target_object":
            #     self.move_alice_to_operation_position()

            if skill_def.execution_mode == ExecutionMode.STEPACTION:
                # Create skill instance (passes env_proxy as 'env')
                self.current_skill = skill_def.function(policy_device, **parameters)

                # Initialize skill with env_proxy (skills see it as regular env)
                obs_dict = (
                    self.current_skill.initialize(
                        self.env_proxy,  # Pass proxy instead of env.unwrapped
                    )
                    or obs_dict
                )

                self.current_skill_name = skill_name
                self.current_skill_params = parameters
                self.status = SkillStatus.RUNNING

                # Start auto-execution thread
                self._start_execution_thread()

                print(f"[SkillExecutorClient] Started policy skill: {skill_name}")
                return True, obs_dict

            elif skill_def.execution_mode == ExecutionMode.PREACTION:
                # PREACTION skills are initialized and added to preaction_skills list
                # They will be called in step() to preprocess observations
                self.current_skill_name = skill_name
                self.current_skill_params = parameters
                self.status = SkillStatus.RUNNING

                # Create skill instance and initialize it
                skill_instance = skill_def.function(
                    policy_device, self.env_proxy, **parameters
                )

                # Initialize the skill with env_proxy
                obs_dict = skill_instance.initialize(self.env_proxy) or obs_dict

                # Add to preaction skills list for step() processing
                self.preaction_skills.append(skill_instance)

                # PREACTION skills complete initialization immediately
                self.status = SkillStatus.IDLE
                self.status_info = "PREACTION skill initialized successfully."
                print(
                    f"[SkillExecutorClient] Initialized PREACTION skill: {skill_name}"
                )
                return True, obs_dict

            else:
                print(
                    f"[SkillExecutorClient] Unknown execution mode: {skill_def.execution_mode}"
                )
                self.status = SkillStatus.FAILED
                return False, obs_dict

        except Exception as e:
            print(f"[SkillExecutorClient] Error initializing skill '{skill_name}': {e}")
            import traceback

            traceback.print_exc()
            self.status = SkillStatus.FAILED
            self.status_info = str(e)
            return False, obs_dict

    @with_time
    def step(self, obs_dict: dict) -> tuple[Any, SkillStatus, str]:
        """
        Execute one step of current skill.

        Args:
            obs_dict: Current observation dictionary

        Returns:
            Tuple of (action, status, status_info)
        """
        # First, preprocess observation with all PREACTION skills
        for preaction_skill in self.preaction_skills:
            obs_dict = preaction_skill(obs_dict)

        if not self.is_running():
            return None, self.status, self.status_info

        try:
            # Unified interface: skills must implement `select_action(obs_dict)`
            # which returns an `Action` object with metadata indicating status.
            action = self.current_skill.select_action(obs_dict)

            # Expect an Action object
            try:
                info = action.metadata.get("info", "success")
                reason = action.metadata.get("reason", "")
            except Exception:
                # If action doesn't follow the Action interface, mark as error
                raise TypeError(
                    "Skill.select_action must return an Action with metadata['info']"
                )

            # Map metadata info to SkillStatus
            if info == "error":
                new_status = SkillStatus.FAILED
            elif info == "finished":
                new_status = SkillStatus.COMPLETED
            elif info == "timeout":
                new_status = SkillStatus.TIMEOUT
            else:
                new_status = SkillStatus.RUNNING

            self.status = new_status
            self.status_info = reason

            return action, self.status, self.status_info

        except Exception as e:
            print(f"[SkillExecutorClient] Error in skill step: {e}")
            import traceback

            traceback.print_exc()
            self.status = SkillStatus.FAILED
            self.status_info = str(e)
            return None, self.status, self.status_info

    def is_running(self) -> bool:
        """Check if a skill is currently running."""
        return self.status == SkillStatus.RUNNING

    def terminate_current_skill(
        self, skill_status: SkillStatus = SkillStatus.INTERRUPTED, status_info: str = ""
    ):
        """
        Terminate current skill execution.

        Args:
            skill_status: Final status for the skill
            status_info: Additional status information
        """
        if self.current_skill:
            print(f"[SkillExecutorClient] Terminating skill: {self.current_skill_name}")

            # Stop execution thread if running
            self._stop_execution_thread()

            try:
                if hasattr(self.current_skill, "cleanup"):
                    self.current_skill.cleanup()
            except Exception as e:
                print(f"[SkillExecutorClient] Error during skill cleanup: {e}")

        self.status = skill_status
        self.status_info = status_info
        self._reset_current_skill_state()

    def change_current_skill_status(self, status: SkillStatus):
        """
        Change current skill status (e.g., pause/resume).

        Args:
            status: New status
        """
        if self.current_skill:
            self.status = status
            print(
                f"[SkillExecutorClient] Changed skill '{self.current_skill_name}' status to {status.value}"
            )
        else:
            print("[SkillExecutorClient] No active skill to change status")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current executor status.

        Returns:
            Dictionary with status information
        """
        return {
            "status": self.status.value
            if isinstance(self.status, SkillStatus)
            else str(self.status),
            "skill_name": self.current_skill_name,
            "skill_params": self.current_skill_params,
            "status_info": self.status_info,
        }

    def _reset_current_skill_state(self):
        """Reset current skill state."""
        self.current_skill = None
        self.current_skill_name = None
        self.current_skill_params = None

    def _start_execution_thread(self):
        """Start the auto-execution thread for skill."""
        if self.execution_thread and self.execution_thread.is_alive():
            print("[SkillExecutorClient] Execution thread already running")
            return

        self.is_executing = True
        self.execution_thread = threading.Thread(
            target=self._execution_loop, daemon=True, name="SkillExecutionThread"
        )
        self.execution_thread.start()
        print("[SkillExecutorClient] Started execution thread")

    def _stop_execution_thread(self):
        """Stop the auto-execution thread."""
        self.is_executing = False
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=2.0)
            print("[SkillExecutorClient] Stopped execution thread")

    def _execution_loop(self):
        """
        Auto-execution loop that runs in a separate thread.
        This mimics the behavior of the old SERVER-side skill_executor loop.
        """
        print(
            f"[SkillExecutorClient] Execution loop started for skill: {self.current_skill_name}"
        )

        while self.is_executing and self.status == SkillStatus.RUNNING:
            try:
                # Get current observation from env_proxy
                obs = (
                    (
                        self.alice_control.update(return_obs=True, update_sim=True)
                        or self.env_proxy.get_current_observation()
                    )
                    if self.enable_alice
                    else self.env_proxy.get_current_observation()
                )
                if obs is None:
                    time.sleep(self.skill_step_interval)
                    continue
                obs_dict = obs.data

                # Execute one step of the skill (this handles PREACTION and STEPACTION)
                action, new_status, status_info = self.step(obs_dict)

                # Check if skill finished
                if new_status != SkillStatus.RUNNING:
                    print(
                        f"[SkillExecutorClient] Skill '{self.current_skill_name}' finished with status: {new_status.value}"
                    )
                    break

                # Apply action to environment if skill produced one
                if action is not None:
                    from robot_brain_system.core.types import Action as ActionType

                    # Ensure the server can evaluate success criteria by providing skill name
                    if action.metadata is None:
                        action.metadata = {}
                    action.metadata.setdefault("skill_name", self.current_skill_name)
                    raw_data = action.data

                    if len(raw_data.shape) == 2:
                        raw_data = raw_data.unsqueeze(1)  # (num_envs, 1, action_dim)
                    elif len(raw_data.shape) != 3:
                        raise ValueError(
                            f"Unsupported action shape {raw_data.shape}, expect 2D tensor"
                        )

                    for chunk_id in range(raw_data.shape[1]):
                        chunk_action = raw_data[:, chunk_id, :]
                        action_chunk = ActionType(
                            data=chunk_action, metadata=action.metadata
                        )
                        if self.enable_alice:
                            self.alice_control.update(
                                return_obs=False, update_sim=False
                            )
                        step_result = self.env_proxy.step(action_chunk)
                        # Server may return success flag in info
                        if step_result is not None:
                            obs, reward, terminated, truncated, info = step_result
                            if self.is_recording:
                                self._save_obs_frame(obs)
                            if isinstance(info, dict) and info.get("skill_success"):
                                with self.execution_lock:
                                    self.status = SkillStatus.COMPLETED
                                    self.status_info = (
                                        "Succeeded by server success_item"
                                    )
                                print(
                                    f"[SkillExecutorClient] Skill '{self.current_skill_name}' succeeded (server-reported)"
                                )
                                break
                        if chunk_id == 15:
                            break
                # Check if skill finished
                if new_status != SkillStatus.RUNNING:
                    print(
                        f"[SkillExecutorClient] Skill '{self.current_skill_name}' finished with status: {new_status.value}"
                    )
                    break
                # High-frequency execution
                time.sleep(self.skill_step_interval)

            except Exception as e:
                print(f"[SkillExecutorClient] Error in execution loop: {e}")
                import traceback

                traceback.print_exc()
                with self.execution_lock:
                    self.status = SkillStatus.FAILED
                    self.status_info = str(e)
                break
        # if self.current_skill_name == "move_to_target_object":
        #     self.env_proxy.release_env_lock()
        self.is_executing = False
        print(
            f"[SkillExecutorClient] Execution loop stopped for skill: {self.current_skill_name}"
        )

    def move_alice_to_operation_position(self):
        """Move Alice to operation position using AliceControl."""
        self.alice_control.move_to_operation_position()

    def move_alice_to_play_position(self):
        """Move Alice to play position using AliceControl."""
        self.alice_control.move_to_play_position()

    def start_recording(self, keys: List[str], save_root: str = "camera_data"):
        """开启录制"""
        with self.execution_lock:
            self.recording_keys = keys
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_root_dir = os.path.join(save_root, timestamp)
            self.recording_idx = 0

            # 预创建文件夹
            for key in keys:
                os.makedirs(os.path.join(self.recording_root_dir, key), exist_ok=True)

            self.is_recording = True
            print(
                f"[SkillExecutor] Start recording {keys} to {self.recording_root_dir}"
            )

    def stop_recording(self):
        """停止录制"""
        self.is_recording = False
        print(f"[SkillExecutor] Stop recording. Saved {self.recording_idx} frames.")

    def _save_obs_frame(self, obs):
        """使用用户指定的简化逻辑保存图片"""
        if not self.is_recording:
            return
        try:
            for key in self.recording_keys:
                # 直接按照您的要求提取数据
                # 假设 obs.data["policy"] 存在且格式正确
                if "policy" in obs.data and key in obs.data["policy"]:
                    frame_data = obs.data["policy"][key][0].cpu().numpy()
                elif key in obs.data:
                    # 兼容性回退：如果不在 policy 下，尝试直接从 data 获取
                    val = obs.data[key]
                    frame_data = val[0].cpu().numpy() if hasattr(val, "cpu") else val[0]
                else:
                    continue

                # 保存图片
                save_path = os.path.join(
                    self.recording_root_dir, key, f"{self.recording_idx}.jpg"
                )
                Image.fromarray(frame_data).save(save_path)

            self.recording_idx += 1
        except Exception as e:
            print(f"[Recording Error] {e}")

    @staticmethod
    def create_video_from_folder(
        folder_path: str, output_path: str = None, fps: float = 10.0
    ):
        """
        将指定文件夹内的所有 jpg 图片合成 MP4 视频。
        默认 fps=10 (即 0.1s 一张图)。
        """
        import glob

        # 获取所有图片并按数字顺序排序
        files = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )

        if not files:
            print(f"No images found in {folder_path}")
            return

        # 读取第一张图获取尺寸
        first_img = cv2.imread(files[0])
        if first_img is None:
            print("Failed to read first image.")
            return
        height, width, _ = first_img.shape

        # 确定输出路径
        if output_path is None:
            output_path = folder_path.rstrip("/\\") + ".mp4"

        # 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Generating video: {output_path} with {len(files)} frames...")
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                out.write(img)

        out.release()
        print("Video generation complete.")


def create_skill_executor_client(
    skill_registry: SkillRegistry, env_proxy: EnvProxy
) -> SkillExecutorClient:
    """
    Factory function to create client-side skill executor.

    Args:
        skill_registry: Registry of available skills
        env_proxy: Proxy for remote environment access

    Returns:
        SkillExecutorClient instance
    """
    return SkillExecutorClient(skill_registry, env_proxy)
