"""
Main robot brain system that orchestrates all components.
"""

import queue
import time
import os
import threading
from typing import (
    Optional,
    Dict,
    Any,
    cast,
)  # Generator might not be needed at this level anymore
import traceback  # For debugging
from PIL import Image
from omegaconf import DictConfig, OmegaConf

from .types import (
    SystemStatus,
    SystemState,
    SkillStatus,
)
from .isaac_simulator import IsaacSimulator
from .brain import QwenVLBrain
from .skill_manager import get_skill_registry
from .skill_executor_client import SkillExecutorClient
from ..ui.console import global_console

# Import skills to register them in client-side registry for brain planning
import robot_brain_system.skills  # noqa: F401


class RobotBrainSystem:
    """
    Main system that orchestrates the Isaac simulator, skills, and brain.
    """

    def __init__(self, config: DictConfig):
        self.config = cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))
        self.visualize = True
        self.log_path = config["monitoring"]["log_dir"]
        self.state = SystemState()
        self.pending_human_feedback: Optional[str] = None  # 存储待处理的人类反馈
        self.scene_mode = self.config.get("simulator", {}).get("scene_mode", "default")
        self.scene_ready = self.scene_mode != "missing_box_human_intervention_test"

        # Core components
        self.simulator: Optional[IsaacSimulator] = None
        self.env_proxy = None  # EnvProxy for remote env access
        self.skill_executor: Optional[SkillExecutorClient] = (
            None  # Client-side skill executor
        )
        self.skill_registry: "SkillRegistry" = None  # Global registry
        self.brain: QwenVLBrain = QwenVLBrain(config.get("brain", {}))

        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.is_shutdown_requested = (
            False  # Changed from self.is_shutdown to avoid conflict with method
        )

        # ui
        self.console = global_console
        self.console.set_status_provider(self._provide_ui_status)

        self.print("[RobotBrainSystem] Initialized")

    def print(self, message: str):
        """Helper to print messages to both console and standard output."""
        self.console.log("system", message)

    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.print("[RobotBrainSystem] Starting Console UI...")
            self.console.start()
            self.console.log(
                "system", "UI started early for initialization monitoring."
            )

            self.console.log("system", "Initializing components...")

            # 1. Initialize Isaac simulator (CLIENT proxy)
            sim_config_dict = self.config["simulator"]
            sim_config_dict["skills"] = self.config["skills"]
            self.simulator = IsaacSimulator(sim_config=sim_config_dict)

            if not self.simulator.initialize():
                self.print("[RobotBrainSystem] Failed to initialize simulator")
                self.state.status = SystemStatus.ERROR
                self.state.error_message = "Simulator initialization failed"
                return False
            self.console.log("system", "Simulator initialized successfully.")

            # 2. Create env_proxy for remote environment access
            from .env_proxy import create_env_proxy

            self.env_proxy = create_env_proxy(
                self.simulator, scene_mode=self.scene_mode
            )
            self.console.log("system", "EnvProxy created successfully.")

            # 缺箱子测试场景：启动即将箱子/扳手移走，等待人工重置
            if self.scene_mode == "missing_box_human_intervention_test":
                try:
                    self.env_proxy.reset_box_and_spanner("far")
                    self.scene_ready = False
                    self.console.log(
                        "system",
                        "Scene mode=missing_box_human_intervention_test: box/spanner moved far for intervention test.",
                    )
                except Exception as e:
                    self.console.log("error", f"Failed to place far-mode assets: {e}")

            # 3. Get global skill registry (CLIENT-side for planning and execution)
            self.skill_registry = get_skill_registry()
            if self.skill_registry is None:
                self.console.log(
                    "error", "[RobotBrainSystem] Failed to get global skill registry"
                )
                self.state.status = SystemStatus.ERROR
                self.state.error_message = "Failed to get skill registry"
                return False

            self.console.log(
                "system",
                f"Loaded {len(self.skill_registry.list_skills())} skills for brain planning.",
            )

            # 4. Create skill_executor_client for CLIENT-side skill execution
            self.skill_executor = SkillExecutorClient(
                self.skill_registry,
                self.env_proxy,
                input_queue=self.console.input_queue,  # [NEW] 传入输入队列引用
            )
            self.console.log("system", "SkillExecutorClient created successfully.")

            # 5. Initialize brain and connect to skill registry
            self.brain.initialize()
            self.brain.set_skill_registry(self.skill_registry)
            self.brain.set_system_state(state=self.state)

            self.state.status = SystemStatus.IDLE
            self.console.log("system", "Initialization complete")

            os.makedirs(self.log_path, exist_ok=True)
            self.brain.log_path = self.log_path  # Set log path for brain
            self.brain.visualize = self.visualize  # Set visualization flag

            return True

        except Exception as e:
            self.print(f"[RobotBrainSystem] Initialization failed: {e}")
            traceback.print_exc()
            self.state.error_message = str(e)
            self.state.status = SystemStatus.ERROR
            return False

    def start(self) -> bool:
        """Start the main system loop."""
        if self.state.status == SystemStatus.ERROR:
            self.console.log(
                "error", "[RobotBrainSystem] Cannot start - system in error state"
            )
            return False
        if self.state.is_running:
            self.console.log("system", "[RobotBrainSystem] System already running.")
            return True

        try:
            self.is_shutdown_requested = False
            self.state.is_running = True

            # Start main loop in separate thread
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()

            self.console.log("system", "System started successfully.")
            return True

        except Exception as e:
            self.console.log("error", f"[RobotBrainSystem] Failed to start: {e}")
            traceback.print_exc()
            self.state.is_running = False
            return False

    def shutdown(self):
        """Shutdown the system."""
        self.print("[RobotBrainSystem] Shutting down...")
        if self.is_shutdown_requested and not self.state.is_running:
            self.print("[RobotBrainSystem] Already shutdown or shutdown in progress.")
            return

        self.is_shutdown_requested = True  # Signal main loop to stop
        self.state.is_running = False  # Prevent new operations

        # Interrupt any running tasks in the brain
        if self.brain and (self.brain.has_task() or self.brain.has_plan()):
            self.brain.interrupt_task("System shutdown")

        # Terminate any skill in the CLIENT-side skill_executor
        if self.skill_executor and self.skill_executor.is_running():
            self.print("[RobotBrainSystem] Terminating skill in skill_executor...")
            self.skill_executor.terminate_current_skill()

        # Shutdown simulator (this will also stop its server process)
        if self.simulator and self.simulator.is_initialized:
            self.simulator.shutdown()

        # Wait for main thread
        if self.main_thread and self.main_thread.is_alive():
            self.print("[RobotBrainSystem] Waiting for main loop to terminate...")
            self.main_thread.join(timeout=10)
            if self.main_thread.is_alive():
                self.print(
                    "[RobotBrainSystem] Warning: Main loop did not terminate in time."
                )

        self.state.status = SystemStatus.SHUTDOWN
        self.print("[RobotBrainSystem] Shutdown complete")

    def execute_task(self, instruction: str) -> bool:
        """
        Execute a high-level task instruction.
        The brain will parse, plan, and then the system will start the first skill.
        """
        if not self.state.is_running:
            self.print("[RobotBrainSystem] System is not running. Cannot execute task.")
            return False

        # Check if Brain already has a task
        if self.brain.has_task():
            self.print(
                "[RobotBrainSystem] Brain is busy with another task. Please wait or interrupt."
            )
            return False

        # Check if a skill is running in the skill_executor from a previous plan
        if self.skill_executor.is_running():
            self.print(
                f"[RobotBrainSystem] SkillExecutor is busy executing skill: {self.skill_executor.current_skill_name}. Please wait or interrupt."
            )
            return False

        self.print(f"[RobotBrainSystem] Received task instruction: {instruction}")
        try:
            # 状态转换：IDLE -> THINKING
            self.state.status = SystemStatus.THINKING
            obs = self.env_proxy.get_latest_observation()
            if obs is None:
                self.print(
                    "[RobotBrainSystem] No observations available from simulator. Cannot execute task."
                )
                self.state.status = SystemStatus.ERROR
                self.state.error_message = "No observations available"
                return False
            inspector_rgb = obs.data["policy"]["inspector_side"][0].cpu().numpy()
            if self.visualize:
                import matplotlib.pyplot as plt

                plt.imshow(inspector_rgb)
                plt.axis("off")
                plt.savefig(
                    os.path.join(self.log_path, f"{time.time()}_execute_task_input.png")
                )
            image_data = Image.fromarray(inspector_rgb)

            # Brain 解析并存储任务
            task = self.brain.parse_task(instruction, image_data)

            # Brain 规划任务 (Brain 不修改系统状态，但会存储 task 和 plan)
            plan = self.brain.execute_task(task)
            if not plan or not plan.steps:
                self.print("[RobotBrainSystem] Brain did not produce a valid plan.")
                self.state.status = SystemStatus.ERROR  # 规划失败视为错误
                self.state.error_message = "No plan generated"
                self.brain.interrupt_task("No plan generated")  # Reset brain state
                return False

            # 记录到历史（但不再在 state 中存储 current_task/current_plan 副本）
            self.state.plan_history.append(plan)

            # 状态转换：THINKING -> EXECUTING
            self.state.status = SystemStatus.EXECUTING
            self.print(
                f"[RobotBrainSystem] Brain planned {len(plan.steps)} skills. Starting execution..."
            )

            return True

        except Exception as e:
            self.print(f"[RobotBrainSystem] Failed to start task execution: {e}")
            traceback.print_exc()
            self.state.error_message = str(e)
            self.state.status = SystemStatus.ERROR
            # Brain 异常时清理状态
            if self.brain.has_task():
                self.brain.interrupt_task(f"Error during task start: {e}")
            return False

    def interrupt_task(self, reason: str = "User interrupt"):
        """Interrupt the current high-level task being managed by the brain."""
        self.print(f"[RobotBrainSystem] Interrupting current task: {reason}")
        if self.brain:
            self.brain.interrupt_task(reason)  # This resets brain's plan

        # Terminate any skill running in the skill_executor_client
        if self.skill_executor and self.skill_executor.is_running():
            self.print(
                "[RobotBrainSystem] Sending terminate signal to skill_executor..."
            )
            self.skill_executor.terminate_current_skill()

        self.state.status = SystemStatus.IDLE
        self.state.sub_skill_status = {}

    def interrupt_skill(
        self,
        reason: str = "UNKOWN",
        skill_status: SkillStatus = SkillStatus.INTERRUPTED,
    ):
        brain_skill_info = self.brain.get_next_skill()
        if not brain_skill_info:
            self.print(
                "[RobotBrainSystem] No skill in the brain! ignore interrupt requirement!"
            )
            return True
        self.print(
            f"[RobotBrainSystem] Interrupting current skill:\n{brain_skill_info}, with status: {skill_status} and reason: {reason}\n"
        )

        # Terminate skill in skill_executor_client
        if self.skill_executor:
            executor_status = self.skill_executor.get_status()
            self.print(
                f"[RobotBrainSystem] Current skill_status in executor: {executor_status}"
            )

            # Verify brain and executor are in sync
            if brain_skill_info["name"] == executor_status.get("skill_name"):
                self.skill_executor.terminate_current_skill(
                    skill_status, status_info=reason
                )
                return True
            else:
                self.print(
                    f"[RobotBrainSystem] Warning: Brain skill {brain_skill_info['name']} != executor skill {executor_status.get('skill_name')}"
                )
                # Still terminate to be safe
                self.skill_executor.terminate_current_skill(
                    skill_status, status_info=reason
                )
                return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        sim_status = {}
        if self.simulator:
            sim_status = {
                "initialized": self.simulator.is_initialized,
                "is_running_subprocess": self.simulator.is_running,  # is the subprocess itself alive
                "device": self.simulator.device
                if hasattr(self.simulator, "device")
                else None,
                "num_envs": self.simulator.num_envs
                if hasattr(self.simulator, "num_envs")
                else 0,
                "skill_executor": self.skill_executor.get_status()
                if (self.skill_executor and self.simulator.is_initialized)
                else {"status": "not_initialized"},
            }

        brain_s = self.brain.get_status() if self.brain else {}

        # 从 Brain 获取当前任务信息
        current_task = self.brain.get_current_task() if self.brain else None

        # 根据 System 状态和 Brain 状态构造详细的当前操作描述
        current_system_op = self.state.status.value

        if self.state.status == SystemStatus.EXECUTING:
            # System 处于执行状态
            if brain_s.get("has_pending_skills"):
                # Brain 还有待执行的技能
                executor_status = sim_status.get("skill_executor", {})
                if executor_status.get("status") == "running":
                    # 技能正在运行
                    current_system_op = f"executing_plan (skill_running: {executor_status.get('skill_name')})"
                else:
                    # 等待启动下一个技能
                    next_skill_info = self.brain.get_next_skill()
                    if next_skill_info:
                        current_system_op = f"executing_plan (waiting_to_start_skill: {next_skill_info.get('name')})"
                    else:
                        current_system_op = "executing_plan (transitioning)"
            else:
                # Brain 没有待执行技能，计划即将完成
                current_system_op = "executing_plan (finishing)"

        elif self.state.status == SystemStatus.THINKING:
            current_system_op = "thinking (brain_planning)"

        return {
            "system": {
                "status": current_system_op,
                "is_running_main_loop": self.state.is_running,
                "error_message": self.state.error_message,
                "current_high_level_task_id": current_task.id if current_task else None,
                "current_high_level_task_desc": current_task.description
                if current_task
                else None,
            },
            "simulator": sim_status,
            "brain": brain_s,
            "skills_global_registry": {
                "registered_count": len(self.skill_registry.list_skills()),
                "available_list": self.skill_registry.list_skills()[:10],  # Show a few
            },
        }

    def get_available_skills(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available skills from the global registry."""
        skills_info = {}
        for skill_name in self.skill_registry.list_skills():
            skill_info = self.skill_registry.get_skill_info(skill_name)
            if skill_info:
                skills_info[skill_name] = skill_info
        return skills_info

    def reset_simulation(self) -> bool:
        """Reset the simulation environment."""
        if not self.simulator or not self.simulator.is_initialized:
            self.print("[RobotBrainSystem] Simulator not ready for reset.")
            return False
        try:
            resp, _ = self.env_proxy.reset()
            if resp is not None:
                self.print("[RobotBrainSystem] Simulation reset successfully.")
                return True
            else:
                self.print(
                    "[RobotBrainSystem] Simulation reset failed to return observation."
                )
                return False
        except Exception as e:
            self.print(f"[RobotBrainSystem] Reset failed: {e}")
            traceback.print_exc()
            return False

    def reset(self) -> bool:
        self.print("[RobotBrainSystem] Resetting system...")

        # 1. Reset simulation environment
        self.reset_simulation()

        # 2. Reset Brain (includes BrainState with task_memory, and BrainMemory instances)
        self.brain.reset()

        # 3. Reset SystemState (creates new instance, clears all history)
        self.state = SystemState()
        self.is_shutdown_requested = False
        self.state.is_running = True

        # 4. Clear pending human feedback
        self.pending_human_feedback = None

        # 5. Clear EnvProxy observation buffer if exists
        if self.env_proxy:
            self.env_proxy.clear_observation_buffer()
            self.print("[RobotBrainSystem] EnvProxy observation buffer cleared")

        # 6. Reset SkillExecutor state if exists
        if self.skill_executor:
            # Ensure no thread is running
            if self.skill_executor.is_running():
                self.skill_executor.terminate_current_skill()
            # Reset executor status
            self.skill_executor.status = SkillStatus.IDLE
            self.skill_executor.status_info = ""
            self.skill_executor._reset_current_skill_state()
            self.print("[RobotBrainSystem] SkillExecutor state reset")

        # 7. Core components - use global registry
        self.skill_registry = get_skill_registry()
        self.print("[RobotBrainSystem] Initialized")
        if self.skill_registry is None:
            self.print("[RobotBrainSystem] Failed to get global skill registry")
            self.state.status = SystemStatus.ERROR
            self.state.error_message = "Failed to get skill registry"
            return False

        self.print(
            f"[RobotBrainSystem] get {len(self.skill_registry.list_skills())} skills from simulator."
        )

        # 8. Connect brain to skill registry (for planning purposes)
        self.brain.set_skill_registry(self.skill_registry)
        self.brain.set_system_state(state=self.state)

        self.state.status = SystemStatus.IDLE
        self.print("[RobotBrainSystem] Initialization complete")
        self.print("[RobotBrainSystem] Resetting finish")
        return True

    def _handle_skill_completion(
        self, result_status: str, state_info: str, need_summary=True
    ):
        """
        Handles the logic after a skill finishes, including summarization,
        state updates, and triggering a replan.
        """
        self.print(
            f"[RobotBrainSystem] Handling completion of skill [{self.state.skill_history[-1]['name']}] with status: {result_status}"
        )

        # 1. Update the skill history with the final result
        self.state.skill_history[-1]["result"] = result_status
        self.state.skill_history[-1]["status_info"] = state_info

        # 2. Get the latest observation for summarization and replanning
        obs = self.env_proxy.get_latest_observation()

        # 3. 通过 Brain 的 API 更新技能状态 (Brain 是 Source of Truth)
        skill_index = self.state.skill_history[-1]["index"]
        self.brain.mark_skill_status(skill_index, SkillStatus[result_status.upper()])

        if need_summary:
            self.print("[RobotBrainSystem] Generating execution summary with brain...")
            # summary_skill_execution 内部会自动更新 task_memory
            execution_summary = self.brain.summary_skill_execution(
                self.state.skill_history[-1], obs
            )
            self.state.skill_history[-1]["execution_summary"] = execution_summary
        else:
            # 对于不走 monitor 的技能，仍然记录一句摘要，避免空白
            if (
                self.state.skill_history[-1].get("name") == "human_intervention"
                and state_info
            ):
                self.state.skill_history[-1]["execution_summary"] = state_info
            else:
                self.state.skill_history[-1]["execution_summary"] = "N/A"

        if self.state.skill_history[-1]["name"] == "grasp_spanner":
            # 如果成功抓起来 spanner，则移动 Alice 到操作位置
            self.skill_executor.move_alice_to_operation_position()
            obs = self.env_proxy.get_latest_observation()

        # 4. Trigger the brain to replan based on the new state
        current_task = self.brain.get_current_task()
        current_plan = self.brain.get_current_plan()
        if current_task and obs and current_plan:
            self.print("[RobotBrainSystem] Triggering brain to replan...")
            # 获取并消费 pending human feedback
            human_feedback = self.pending_human_feedback or ""
            self.pending_human_feedback = None  # 消费后清除

            _ = self.brain.replan_task(  # replan_task 会修改 plan (in-place)
                current_task,
                current_plan,  # 从 Brain 获取当前计划
                self.state.skill_history,
                obs,
                human_feedback=human_feedback,
            )

        # Note: Observation history is now managed by EnvProxy buffer,
        # which is cleared automatically before each skill starts

    def _start_next_skill(self):
        """
        Gets the next pending skill from the brain's plan and asks the
        skill_executor_client to start it.
        """
        retry_time = 3
        for i in range(retry_time + 1):
            next_skill_to_run = self.brain.get_next_skill()
            if next_skill_to_run:
                skill_name = next_skill_to_run["name"]
                skill_params = next_skill_to_run["parameters"]
                self.print(
                    f"[RobotBrainSystem] Requesting skill_executor to start skill: {skill_name}"
                )

                # Clear observation buffer before starting new skill
                # This ensures buffer only contains observations from this skill
                self.env_proxy.clear_observation_buffer()

                # Get current observation for skill initialization
                obs = self.env_proxy.get_latest_observation()
                obs_dict = obs.data if obs else {}

                # Initialize skill using skill_executor_client
                success, obs_dict = self.skill_executor.initialize_skill(
                    skill_name,
                    parameters=skill_params,
                    policy_device=self.config.get("policy_device", "cuda"),
                    obs_dict=obs_dict,
                )

                self.state.skill_history.append(next_skill_to_run)

                if success:
                    self.print(
                        f"[RobotBrainSystem] Successfully started skill: {skill_name}"
                    )
                    return True
                else:
                    # Get skill status from skill_executor_client
                    skill_status = self.skill_executor.get_status()
                    status = skill_status.get("status", "")
                    status_info = skill_status.get("status_info", "")

                    if i < retry_time and status_info:
                        self.print(
                            f"[RobotBrainSystem] Warning: Skill '{skill_name}' failed with status: {status_info}, replanning..."
                        )
                        self._handle_skill_completion(
                            status,
                            status_info,
                            need_summary=False,
                        )
                        continue
                    else:
                        self.print(
                            f"[RobotBrainSystem] CRITICAL: Failed to start skill '{skill_name}' with unknown reason."
                        )
                        self.interrupt_task(f"Failed to start skill {skill_name}")
                        self.state.status = SystemStatus.ERROR
                        self.state.error_message = f"Failed to start skill {skill_name}"
                        return True
            else:
                # No more skills in the plan. Treat as plan completion.
                if self.brain.has_plan():
                    if self.brain.is_plan_complete():
                        self.print(
                            "[RobotBrainSystem] All skills in the plan are complete. Task finished."
                        )
                    else:
                        self.print(
                            "[RobotBrainSystem] Plan has no remaining skills but is not marked complete. Interrupting for safety."
                        )
                    self.brain.interrupt_task("Plan completed successfully")

                self.state.status = SystemStatus.IDLE
                return True

    def _provide_ui_status(self) -> dict:
        """提供 UI 状态栏显示的信息"""
        # 当前技能信息
        current_skill = "None"
        if self.skill_executor and self.skill_executor.is_running():
            current_skill = self.skill_executor.current_skill_name or "Unknown"
        elif self.state.skill_history:
            # 技能未运行时，显示最后一个技能的信息
            last_skill = self.state.skill_history[-1]
            last_result = last_skill.get("result", "?")
            current_skill = f"{last_skill.get('name', '?')} [{last_result}]"

        # 当前任务
        current_task = self.brain.get_current_task() if self.brain else None
        task_desc = current_task.description if current_task else None

        # Brain 状态 - 更细粒度
        if self.state.status == SystemStatus.THINKING:
            brain_state = "Planning"
        elif self.state.status == SystemStatus.EXECUTING:
            if self.skill_executor and self.skill_executor.is_running():
                brain_state = "Executing"
            else:
                brain_state = "Transitioning"
        elif self.state.status == SystemStatus.IDLE:
            brain_state = "Idle"
        elif self.state.status == SystemStatus.ERROR:
            brain_state = "Error"
        else:
            brain_state = self.state.status.name

        # 是否有待处理的 human feedback
        feedback_indicator = ""
        if self.pending_human_feedback:
            feedback_indicator = " [FB pending]"

        return {
            "system_status": self.state.status.name + feedback_indicator,
            "current_task": task_desc,
            "current_skill": current_skill,
            "brain_state": brain_state,
        }

    def _manage_skill_lifecycle(self, skill_status):
        """
        处理非运行状态下的技能逻辑：检查上一个技能结果，启动下一个技能。
        """
        # --- 3b. 检查上一个技能是否刚结束 ---
        if self.state.skill_history and len(self.state.skill_history) > 0:
            last_skill = self.state.skill_history[-1]

            # 检查是否还没有结果（意味着刚结束）
            if "result" not in last_skill or last_skill["result"] is None:
                enable_monitoring = last_skill.get("enable_monitoring", True)

                # 获取状态字符串
                skill_status_str = (
                    skill_status.value
                    if isinstance(skill_status, SkillStatus)
                    else str(skill_status)
                )
                skill_status_info = self.skill_executor.status_info

                # Capture human feedback; if 来自 human_intervention，则存入其摘要而不触发中断重规划提示
                if skill_status_info:
                    info_lower = skill_status_info.lower()
                    if info_lower.startswith("human feedback:"):
                        feedback_text = skill_status_info.split(":", 1)[1].strip()
                        if feedback_text:
                            if last_skill.get("name") == "human_intervention":
                                # 作为技能结果记录，不注入 replan "URGENT" 提示
                                last_skill["execution_summary"] = (
                                    f"Human feedback: {feedback_text}"
                                )
                            else:
                                self.pending_human_feedback = feedback_text
                                self.console.log(
                                    "system",
                                    f"Captured human feedback from skill: {feedback_text}",
                                )

                if skill_status in [
                    SkillStatus.COMPLETED,
                    SkillStatus.FAILED,
                    SkillStatus.TIMEOUT,
                    SkillStatus.INTERRUPTED,
                ]:
                    # 技能结束，处理完成逻辑
                    self._handle_skill_completion(
                        skill_status_str,
                        skill_status_info,
                        need_summary=enable_monitoring,
                    )
                elif skill_status == SkillStatus.IDLE:
                    # 罕见情况：技能以 IDLE 结束
                    # 这通常意味着技能在初始化阶段就结束了，视为成功完成
                    self.console.log(
                        "system",
                        "Skill finished with 'idle' status, treating as completed.",
                    )
                    self._handle_skill_completion(
                        "completed",
                        "Skill ended in IDLE state",
                        need_summary=False,  # IDLE 结束通常不需要总结
                    )
                elif skill_status == SkillStatus.PAUSED:
                    # 技能被暂停，等待恢复，不做处理
                    pass
                elif skill_status == SkillStatus.RUNNING:
                    # 技能仍在运行，不应该进入这个分支
                    self.console.log(
                        "system",
                        f"Warning: Skill {last_skill['name']} reported as RUNNING but is_running() is False",
                    )

        # --- 4. 检查并启动下一个技能 ---
        if self.brain.has_pending_skills():
            self._start_next_skill()
        else:
            # 所有技能完成，任务结束
            if self.state.status == SystemStatus.EXECUTING:
                self.console.log(
                    "system", "All skills in the plan are complete. Task finished."
                )
                self.state.status = SystemStatus.IDLE
                self.brain.interrupt_task("Plan completed successfully")

    def _main_loop(self):
        """
        Main system loop that runs in a separate thread.

        This loop is responsible for:
        1. Collecting observations from simulator (for monitoring)
        2. Brain monitoring at lower frequency (~0.5Hz)
        3. Managing skill lifecycle (starting next skill when current finishes)
        4. System state management

        Note: Skill execution itself is handled by SkillExecutorClient's
        auto-execution thread at high frequency (50Hz).
        """
        self.console.log("system", "Main loop active.")
        heartbeat_interval = 0.2  # seconds for monitoring and planning

        while not self.is_shutdown_requested:
            try:
                try:
                    # 如果有输入，立即返回；没输入则等待 heartbeat_interval
                    user_input = self.console.input_queue.get(
                        timeout=heartbeat_interval
                    )
                    self._handle_user_input(user_input)
                except queue.Empty:
                    pass  # 无输入，继续执行周期性任务

                if not self.state.is_running:  # Paused or shutting down
                    continue

                if self.state.status != SystemStatus.EXECUTING:
                    continue

                # --- We are in EXECUTING state, now manage the skill lifecycle ---

                # 2. Get the current skill status
                is_skill_running = self.skill_executor.is_running()
                skill_status = self.skill_executor.status

                if is_skill_running:
                    # 3a. A skill is currently running (execution happens in SkillExecutorClient thread).

                    # Check if we should monitor (less frequent)
                    # Get accumulated observation history from buffer
                    obs_history = self.env_proxy.get_and_clear_observation_buffer()

                    if self.brain.should_monitor(
                        obs_history, system_status=self.state.status
                    ):
                        self.console.log(
                            "brain",
                            f"Monitoring execution with {len(obs_history)} observations...",
                        )
                        # Pause skill execution for monitoring
                        self.skill_executor.change_current_skill_status(
                            SkillStatus.PAUSED
                        )

                        monitoring_result = self.brain.monitor_skill_execution(
                            obs_history
                        )

                        # Resume skill execution after monitoring
                        self.skill_executor.change_current_skill_status(
                            SkillStatus.RUNNING
                        )
                        self._handle_monitoring_result(monitoring_result)
                else:
                    self._manage_skill_lifecycle(skill_status)

                # 5. Final system state check
                if self.brain.state.error_message:  # Brain 有错误信息
                    self.console.log(
                        "error",
                        f"Brain encountered error: {self.brain.state.error_message}",
                    )
                    self.state.status = SystemStatus.ERROR
                    self.state.error_message = self.brain.state.error_message

            except Exception as e:
                self.console.log("error", f"Critical error in main loop: {e}")
                traceback.print_exc()
                self.state.error_message = str(e)
                self.state.status = SystemStatus.ERROR
                # Potentially try to recover or shutdown gracefully
                if self.brain:
                    self.brain.interrupt_task("Main loop critical error")
                time.sleep(1)  # Longer sleep on error

        self.print("[RobotBrainSystem] Main loop stopped.")

    def _handle_monitoring_result(self, monitoring_result: Dict[str, Any]):
        """Handle brain monitoring result.
        Brain monitor may determine the status of a skill as succcessed, failed, progress , etc.,
            which are defined in the criterion property when regirstering one skill.
        It should be noticed that, when one skill function ended normally,
            it also return the status of skill termination, maybe success, timeout, etc.,
            which are defined in the skill function inside.
        This function only handle the skill status determined by brain monitor, nor the original status returned by skill function self.
        While if monitoring dicision is success or failed, system may interrupt one skill directly, instead of waiting for
            skill to finish executing itself, this (or other) action may changed the status of one skill that return by
            the skill executor. (The original skill status also return to skill exector, and we only get statue status info from skill exectuor)
        """
        result = monitoring_result.get("result", "progress")
        reason = monitoring_result.get("reason", "")
        self.print(
            f"[RobotBrainSystem] Brain monitoring result: {result}, Reason: {reason}"
        )

        if result == "failed":
            self.interrupt_skill(
                f"Brain decision: {reason}", skill_status=SkillStatus.FAILED
            )
        elif result == "success" or result == "successed":
            self.print(f"[RobotBrainSystem] Brain decided skill is success: {reason}")
            self.interrupt_skill(
                f"Brain decision: {reason}", skill_status=SkillStatus.COMPLETED
            )
        elif result == "not enough":
            # not enough observation to determine
            pass
        elif result == "grogress":
            # Observation history is managed by EnvProxy buffer
            # No need to clear state.obs_history (deprecated)
            pass

    # [NEW] 用户输入路由逻辑
    def _handle_user_input(self, user_input: str):
        """统一处理用户输入：指令 vs 自然语言"""
        user_input = user_input.strip()
        if not user_input:
            return

        # 1. 编辑/命令模式
        if user_input.startswith("/"):
            self._handle_slash_command(user_input)
            return

        # 2. 技能交互模式 (HumanIntervention)
        # 如果当前正在运行特定技能，且技能需要输入，则优先转发并阻止系统级中断逻辑
        current_skill_name = "None"
        if self.skill_executor.is_running():
            current_skill_name = self.skill_executor.current_skill_name

        if current_skill_name == "human_intervention":
            self.skill_executor.inject_input(user_input)
            return

        # 3. 默认模式：自然语言交互
        if self.state.status == SystemStatus.IDLE:
            self.console.log("system", f"New Task Request: {user_input}")
            # threading.Thread(target=self.execute_task, args=(user_input,)).start()
            self.execute_task(user_input)
            return

        elif self.state.status == SystemStatus.EXECUTING:
            self.console.log(
                "system", f"⚠️  INTERRUPTION: Human Feedback received: '{user_input}'"
            )
            # 启动反馈重规划
            # threading.Thread(
            #     target=self._trigger_feedback_replan, args=(user_input,)
            # ).start()
            return self._trigger_feedback_replan(user_input)

        else:
            self.console.log(
                "info", f"System busy ({self.state.status.name}), input ignored."
            )

    # [NEW] 反馈重规划逻辑
    def _trigger_feedback_replan(self, feedback_text: str):
        """
        存储人类反馈并打断当前技能。

        不直接调用 replan_task，而是：
        1. 存储 feedback 到 pending_human_feedback
        2. 终止当前技能（状态设为 INTERRUPTED）
        3. 让状态机自然流转：
           - _manage_skill_lifecycle 检测到技能结束
           - _handle_skill_completion 被调用
           - 其中会消费 pending_human_feedback 并传递给 replan_task

        这样避免了重复 replan 的问题。
        """
        # 1. 存储反馈，等待状态机消费
        self.pending_human_feedback = feedback_text
        self.console.log(
            "system", "Human feedback stored, will be used in next replan."
        )

        # 2. 终止当前技能（如果正在运行）
        if self.skill_executor.is_running():
            self.skill_executor.terminate_current_skill(
                SkillStatus.INTERRUPTED, status_info=f"Human Feedback: {feedback_text}"
            )
            # 不需要手动调用 replan，状态机会自动处理
        else:
            # 技能未运行，可能在等待下一个技能启动
            # 此时直接触发 replan
            self.console.log(
                "system", "No skill running, triggering immediate replan..."
            )
            try:
                obs = self.env_proxy.get_latest_observation()
                current_task = self.brain.get_current_task()
                current_plan = self.brain.get_current_plan()

                if current_task and current_plan and obs:
                    human_feedback = self.pending_human_feedback or ""
                    self.pending_human_feedback = None

                    self.brain.replan_task(
                        current_task,
                        current_plan,
                        self.state.skill_history,
                        obs,
                        human_feedback=human_feedback,
                    )
            except Exception as e:
                self.console.log("error", f"Immediate replan failed: {e}")

    # [NEW] Slash 命令处理
    def _handle_slash_command(self, command_str: str):
        """处理 /cmd 参数 指令"""
        parts = command_str[1:].split(" ")
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ["exit", "quit"]:
            self.shutdown()
        elif cmd in ["stop", "abort"]:
            self.interrupt_task("User command /stop")
        elif cmd in ["plan", "ls"]:
            current_plan = self.brain.get_current_plan() if self.brain else None
            if current_plan:
                self.console.log("brain", "\n" + current_plan.pretty_print())
            else:
                self.console.log("info", "No active plan.")
        elif cmd == "del":
            current_plan = self.brain.get_current_plan() if self.brain else None
            if current_plan and args:
                try:
                    idx = int(args[0])
                    current_plan.delete_skill(idx)
                    self.console.log("brain", f"Deleted step {idx}.")
                except Exception:
                    self.console.log("error", "Usage: /del <index>")
            else:
                self.console.log("info", "No active plan or missing index.")
        elif cmd == "help":
            self.console.log(
                "info",
                "/stop - Abort task\n/plan - Show plan\n/del <idx> - Delete step",
            )
        elif cmd == "reset_box":
            mode = "normal"
            if args and args[0] in ["far", "normal"]:
                mode = args[0]
            try:
                self.env_proxy.reset_box_and_spanner(
                    mode,
                    snapshot_path=os.path.join(
                        self.log_path, f"{time.time()}_reset_box.png"
                    ),
                )
                self.scene_ready = mode == "normal"
                self.console.log(
                    "system",
                    f"""/reset_box applied with mode={mode}. scene_ready={
                        self.scene_ready
                    }. Snapshot={
                        os.path.join(self.log_path, f"{time.time()}_reset_box.png")
                        if mode == "normal"
                        else "skipped"
                    }""",
                )

            except Exception as e:
                self.console.log("error", f"/reset_box failed: {e}")

        elif cmd == "reset_spanner":
            # 人为干扰使得扳手位置重置
            try:
                self.env_proxy.reset_spanner_position()
                self.console.log(
                    "system",
                    """/reset_spanner applied.""",
                )
            except Exception as e:
                self.console.log("error", f"/reset_spanner failed: {e}")
        elif cmd == "reset":
            try:
                self.reset()
                self.console.log("system", "/reset completed.")
            except Exception as e:
                self.console.log("error", f"/reset failed: {e}")
        else:
            self.console.log("error", f"Unknown command: {cmd}")


if __name__ == "__main__":
    import os

    print("TEST ROBOT BRAIN SYSTEM")
    from robot_brain_system.configs.config import DEVELOPMENT_CONFIG

    system = RobotBrainSystem(DEVELOPMENT_CONFIG)
    result = system.initialize()
    system.start()
    # home tensor([[ 1.1291, -3.8319,  3.6731]], device='cuda:2') tensor([[-0.6167,  0.3308, -0.3199, -0.6386]], device='cuda:2')
    # [1.1283, -3.8319,  3.6731, -0.6167,  0.3308, -0.3199, -0.6386]
    system.execute_task(
        "open the red box, move end-effector to home position, and than grasp the spanner in the red box, home position is [1.1283, -3.8319,  3.6731, -0.6167,  0.3308, -0.3199, -0.6386]"
    )
    try:
        while system.state.is_running:
            time.sleep(1)
            status = system.get_status()
            print(f"Current System Status: {status['system']['status']}")
            if status["system"]["status"] == "idle":
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down system...")
    system.interrupt_task("User requested shutdown")
    system.shutdown()
    print("Robot Brain System test completed.")
