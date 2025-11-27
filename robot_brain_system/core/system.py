"""
Main robot brain system that orchestrates all components.
"""

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

        print("[RobotBrainSystem] Initialized")

    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            print("[RobotBrainSystem] Initializing components...")

            # 1. Initialize Isaac simulator (CLIENT proxy)
            sim_config_dict = self.config["simulator"]
            sim_config_dict["skills"] = self.config["skills"]
            self.simulator = IsaacSimulator(sim_config=sim_config_dict)

            if not self.simulator.initialize():
                print("[RobotBrainSystem] Failed to initialize simulator")
                self.state.status = SystemStatus.ERROR
                self.state.error_message = "Simulator initialization failed"
                return False
            print("[RobotBrainSystem] Simulator initialized successfully.")

            # 2. Create env_proxy for remote environment access
            from .env_proxy import create_env_proxy

            self.env_proxy = create_env_proxy(self.simulator)
            print("[RobotBrainSystem] EnvProxy created successfully.")

            # 3. Get global skill registry (CLIENT-side for planning and execution)
            self.skill_registry = get_skill_registry()
            if self.skill_registry is None:
                print("[RobotBrainSystem] Failed to get global skill registry")
                self.state.status = SystemStatus.ERROR
                self.state.error_message = "Failed to get skill registry"
                return False

            print(
                f"[RobotBrainSystem] Loaded {len(self.skill_registry.list_skills())} skills for brain planning."
            )

            # 4. Create skill_executor_client for CLIENT-side skill execution
            self.skill_executor = SkillExecutorClient(
                self.skill_registry, self.env_proxy
            )
            print("[RobotBrainSystem] SkillExecutorClient created successfully.")

            # 5. Initialize brain and connect to skill registry
            self.brain.initialize()
            self.brain.set_skill_registry(self.skill_registry)
            self.brain.set_system_state(state=self.state)

            self.state.status = SystemStatus.IDLE
            print("[RobotBrainSystem] Initialization complete")

            os.makedirs(self.log_path, exist_ok=True)
            self.brain.log_path = self.log_path  # Set log path for brain
            self.brain.visualize = self.visualize  # Set visualization flag

            return True

        except Exception as e:
            print(f"[RobotBrainSystem] Initialization failed: {e}")
            traceback.print_exc()
            self.state.error_message = str(e)
            self.state.status = SystemStatus.ERROR
            return False

    def start(self) -> bool:
        """Start the main system loop."""
        if self.state.status == SystemStatus.ERROR:
            print("[RobotBrainSystem] Cannot start - system in error state")
            return False
        if self.state.is_running:
            print("[RobotBrainSystem] System already running.")
            return True

        try:
            self.is_shutdown_requested = False
            self.state.is_running = True

            # Start main loop in separate thread
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()

            print("[RobotBrainSystem] System started")
            return True

        except Exception as e:
            print(f"[RobotBrainSystem] Failed to start: {e}")
            traceback.print_exc()
            self.state.is_running = False
            return False

    def shutdown(self):
        """Shutdown the system."""
        print("[RobotBrainSystem] Shutting down...")
        if self.is_shutdown_requested and not self.state.is_running:
            print("[RobotBrainSystem] Already shutdown or shutdown in progress.")
            return

        self.is_shutdown_requested = True  # Signal main loop to stop
        self.state.is_running = False  # Prevent new operations

        # Interrupt any running tasks in the brain
        if self.brain and (self.brain.has_task() or self.brain.has_plan()):
            self.brain.interrupt_task("System shutdown")

        # Terminate any skill in the CLIENT-side skill_executor
        if self.skill_executor and self.skill_executor.is_running():
            print("[RobotBrainSystem] Terminating skill in skill_executor...")
            self.skill_executor.terminate_current_skill()

        # Shutdown simulator (this will also stop its server process)
        if self.simulator and self.simulator.is_initialized:
            self.simulator.shutdown()

        # Wait for main thread
        if self.main_thread and self.main_thread.is_alive():
            print("[RobotBrainSystem] Waiting for main loop to terminate...")
            self.main_thread.join(timeout=10)
            if self.main_thread.is_alive():
                print(
                    "[RobotBrainSystem] Warning: Main loop did not terminate in time."
                )

        self.state.status = SystemStatus.SHUTDOWN
        print("[RobotBrainSystem] Shutdown complete")

    def execute_task(self, instruction: str) -> bool:
        """
        Execute a high-level task instruction.
        The brain will parse, plan, and then the system will start the first skill.
        """
        if not self.state.is_running:
            print("[RobotBrainSystem] System is not running. Cannot execute task.")
            return False

        # Check if Brain already has a task
        if self.brain.has_task():
            print(
                "[RobotBrainSystem] Brain is busy with another task. Please wait or interrupt."
            )
            return False

        # Check if a skill is running in the skill_executor from a previous plan
        if self.skill_executor.is_running():
            print(
                f"[RobotBrainSystem] SkillExecutor is busy executing skill: {self.skill_executor.current_skill_name}. Please wait or interrupt."
            )
            return False

        print(f"[RobotBrainSystem] Received task instruction: {instruction}")
        try:
            # 状态转换: IDLE -> THINKING
            self.state.status = SystemStatus.THINKING
            obs = self.env_proxy.get_latest_observation()
            if obs is None:
                print(
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
            self.state.current_task = self.brain.parse_task(instruction, image_data)

            # Brain规划任务 (Brain不修改系统状态)
            plan = self.brain.execute_task(self.state.current_task)
            if not plan or not plan.steps:
                print("[RobotBrainSystem] Brain did not produce a valid plan.")
                self.state.status = SystemStatus.ERROR  # 规划失败视为错误
                self.state.error_message = "No plan generated"
                self.brain.interrupt_task("No plan generated")  # Reset brain state
                return False

            self.state.plan_history.append(plan)

            # 状态转换: THINKING -> EXECUTING
            self.state.status = SystemStatus.EXECUTING
            print(
                f"[RobotBrainSystem] Brain planned {len(plan.steps)} skills. Starting execution..."
            )

            return True

        except Exception as e:
            print(f"[RobotBrainSystem] Failed to start task execution: {e}")
            traceback.print_exc()
            self.state.error_message = str(e)
            self.state.status = SystemStatus.ERROR
            # Brain异常时清理状态
            if self.brain.has_task():
                self.brain.interrupt_task(f"Error during task start: {e}")
            return False

    def interrupt_task(self, reason: str = "User interrupt"):
        """Interrupt the current high-level task being managed by the brain."""
        print(f"[RobotBrainSystem] Interrupting current task: {reason}")
        if self.brain:
            self.brain.interrupt_task(reason)  # This resets brain's plan

        # Terminate any skill running in the skill_executor_client
        if self.skill_executor and self.skill_executor.is_running():
            print("[RobotBrainSystem] Sending terminate signal to skill_executor...")
            self.skill_executor.terminate_current_skill()

        self.state.status = SystemStatus.IDLE
        self.state.current_task = None
        self.state.sub_skill_status = {}

    def interrupt_skill(
        self,
        reason: str = "UNKOWN",
        skill_status: SkillStatus = SkillStatus.INTERRUPTED,
    ):
        brain_skill_info = self.brain.get_next_skill()
        if not brain_skill_info:
            print(
                "[RobotBrainSystem] No skill in the brain! ignore interrupt requirement!"
            )
            return True
        print(
            f"[RobotBrainSystem] Interrupting current skill:\n{brain_skill_info}, with status: {skill_status} and reason: {reason}\n"
        )

        # Terminate skill in skill_executor_client
        if self.skill_executor:
            executor_status = self.skill_executor.get_status()
            print(
                f"[RobotBrainSystem] Current skill_status in executor: {executor_status}"
            )

            # Verify brain and executor are in sync
            if brain_skill_info["name"] == executor_status.get("skill_name"):
                self.skill_executor.terminate_current_skill(
                    skill_status, status_info=reason
                )
                return True
            else:
                print(
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

        # 根据System状态和Brain状态构造详细的当前操作描述
        current_system_op = self.state.status.value

        if self.state.status == SystemStatus.EXECUTING:
            # System处于执行状态
            if brain_s.get("has_pending_skills"):
                # Brain还有待执行的技能
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
                # Brain没有待执行技能,计划即将完成
                current_system_op = "executing_plan (finishing)"

        elif self.state.status == SystemStatus.THINKING:
            current_system_op = "thinking (brain_planning)"

        return {
            "system": {
                "status": current_system_op,
                "is_running_main_loop": self.state.is_running,
                "error_message": self.state.error_message,
                "current_high_level_task_id": self.state.current_task.id
                if self.state.current_task
                else None,
                "current_high_level_task_desc": self.state.current_task.description
                if self.state.current_task
                else None,
            },
            "simulator": sim_status,
            "brain": brain_s,
            "skills_global_registry": {
                "registered_count": len(self.skill_registry.list_skills()),
                "available_list": self.skill_registry.list_skills()[:10],  # Show a few
            },
            "last_observation_snippet": str(self.state.last_observation.data)[:100]
            if self.state.last_observation
            and hasattr(self.state.last_observation, "data")
            else None,
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
            print("[RobotBrainSystem] Simulator not ready for reset.")
            return False
        try:
            resp = self.simulator.reset_env()
            if resp:
                print("[RobotBrainSystem] Simulation reset successfully.")
                return True
            else:
                print(
                    "[RobotBrainSystem] Simulation reset failed to return observation."
                )
                return False
        except Exception as e:
            print(f"[RobotBrainSystem] Reset failed: {e}")
            traceback.print_exc()
            return False

    def reset(self) -> bool:
        print("[RobotBrainSystem] Resetting system...")
        self.reset_simulation()
        self.brain.reset()

        self.state = SystemState()
        self.is_shutdown_requested = False
        self.state.is_running = True

        # Core components - use global registry
        self.skill_registry = get_skill_registry()
        print("[RobotBrainSystem] Initialized")
        if self.skill_registry is None:
            print("[RobotBrainSystem] Failed to get global skill registry")
            self.state.status = SystemStatus.ERROR
            self.state.error_message = "Failed to get skill registry"
            return False

        print(
            f"[RobotBrainSystem] get {len(self.skill_registry.list_skills())} skills from simulator."
        )

        # Connect brain to skill registry (for planning purposes)
        self.brain.set_skill_registry(self.skill_registry)
        self.brain.set_system_state(state=self.state)

        self.state.status = SystemStatus.IDLE
        print("[RobotBrainSystem] Initialization complete")
        print("[RobotBrainSystem] Resetting finish")
        return True

    def _handle_skill_completion(
        self, result_status: str, state_info: str, need_summary=True
    ):
        """
        Handles the logic after a skill finishes, including summarization,
        state updates, and triggering a replan.
        """
        print(
            f"[RobotBrainSystem] Handling completion of skill [{self.state.skill_history[-1]['name']}] with status: {result_status}"
        )

        # 1. Update the skill history with the final result
        self.state.skill_history[-1]["result"] = result_status
        self.state.skill_history[-1]["status_info"] = state_info

        # 2. Get the latest observation for summarization and replanning
        # (Similar to execute_task, we only need the latest observation)
        obs = self.env_proxy.get_latest_observation()
        if obs:
            self.state.last_observation = obs

        # 3. Ask the brain to summarize the execution
        # This now correctly uses the last completed skill from history
        last_plan = self.state.plan_history[-1]
        last_plan.mark_status(
            self.state.skill_history[-1]["index"], SkillStatus[result_status.upper()]
        )  # Update the plan's status, as well as the current plan in brain's state

        if need_summary:
            print("[RobotBrainSystem] Generating execution summary with brain...")
            execution_summary = self.brain.summary_skill_execution(
                self.state.skill_history[-1], obs
            )
            self.state.skill_history[-1]["execution_summary"] = execution_summary
        else:
            print("[RobotBrainSystem] Skipping execution summary as per configuration.")
            self.state.skill_history[-1]["execution_summary"] = "N/A"

        # 4. Trigger the brain to replan based on the new state
        if self.state.current_task and obs:
            print("[RobotBrainSystem] Triggering brain to replan...")
            _ = self.brain.replan_task(  # replan_task 会修改 last_plan (in-place)
                self.state.current_task,
                last_plan,  # Pass the current, now-updated plan
                self.state.skill_history,  # Pass the history of attempts for this plan
                obs,
            )

            # TODO
            # TODO ATTENTION!!
            # if (
            #     new_plan and new_plan.steps != last_plan.steps
            # ):  # Check if the plan was actually modified
            #     self.state.plan_history.append(new_plan)
            #     self.state.skill_history = []  # Reset skill history for the new plan
            #     print("[RobotBrainSystem] New plan generated by brain.")
            # else:
            #     print(
            #         "[RobotBrainSystem] Brain decided to continue with the current plan or task is complete."
            #     )

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
                print(
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
                    print(
                        f"[RobotBrainSystem] Successfully started skill: {skill_name}"
                    )
                    return True
                else:
                    # Get skill status from skill_executor_client
                    skill_status = self.skill_executor.get_status()
                    status = skill_status.get("status", "")
                    status_info = skill_status.get("status_info", "")

                    if i < retry_time and status_info:
                        print(
                            f"[RobotBrainSystem] Warning: Skill '{skill_name}' failed with status: {status_info}, replanning..."
                        )
                        self._handle_skill_completion(
                            status,
                            status_info,
                            need_summary=False,
                        )
                        continue
                    else:
                        print(
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
                        print(
                            "[RobotBrainSystem] All skills in the plan are complete. Task finished."
                        )
                    else:
                        print(
                            "[RobotBrainSystem] Plan has no remaining skills but is not marked complete. Interrupting for safety."
                        )
                    self.brain.interrupt_task("Plan completed successfully")

                self.state.status = SystemStatus.IDLE
                self.state.current_task = None
                return True

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
        print("[RobotBrainSystem] Main loop started.")
        loop_interval = 0.5  # seconds for monitoring and planning

        while not self.is_shutdown_requested:
            try:
                if not self.state.is_running:  # Paused or shutting down
                    time.sleep(loop_interval)
                    continue

                # 1. Update latest observation (optional, for state tracking)
                if self.env_proxy:
                    latest_obs = self.env_proxy.get_latest_observation()
                    if latest_obs:
                        self.state.last_observation = latest_obs

                if self.state.status != SystemStatus.EXECUTING:
                    time.sleep(loop_interval)
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
                        print(
                            f"[RobotBrainSystem] Brain monitoring execution with {len(obs_history)} observations..."
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

                    # Normal loop interval for monitoring
                    time.sleep(loop_interval)

                else:
                    # 3b. No skill is running. Either one just finished, or we need to start the first one.

                    # Check if a skill *was* running and has now finished
                    if self.state.skill_history and len(self.state.skill_history) > 0:
                        last_skill = self.state.skill_history[-1]

                        # Check if this skill was just finished (has no result yet)
                        if "result" not in last_skill or last_skill["result"] is None:
                            enable_monitoring = last_skill.get(
                                "enable_monitoring", "True"
                            )

                            # Convert SkillStatus enum to string
                            skill_status_str = (
                                skill_status.value
                                if isinstance(skill_status, SkillStatus)
                                else str(skill_status)
                            )
                            skill_status_info = self.skill_executor.status_info

                            if skill_status in [
                                SkillStatus.COMPLETED,
                                SkillStatus.FAILED,
                                SkillStatus.TIMEOUT,
                                SkillStatus.INTERRUPTED,
                            ]:
                                # Skill finished, handle completion
                                self._handle_skill_completion(
                                    skill_status_str,
                                    skill_status_info,
                                    need_summary=enable_monitoring,
                                )
                            elif skill_status == SkillStatus.IDLE:
                                # Skill finished with idle status
                                print(
                                    "[RobotBrainSystem] Skill finished with 'idle' status."
                                )
                                last_skill["result"] = skill_status_str
                                last_skill["status_info"] = skill_status_info
                                last_skill["execution_summary"] = "N/A"
                                if self.state.plan_history:
                                    last_plan = self.state.plan_history[-1]
                                    last_plan.mark_status(
                                        last_skill["index"],
                                        SkillStatus.COMPLETED,
                                    )
                            else:
                                # Unexpected status
                                print(
                                    f"[RobotBrainSystem] Warning: Skill {last_skill['name']} finished with unexpected status: {skill_status_str}"
                                )

                    # 4. Start the next skill in the plan
                    if self.brain.has_pending_skills():  # 使用新的辅助方法
                        self._start_next_skill()
                    else:
                        # 所有技能完成,任务结束
                        print(
                            "[RobotBrainSystem] All skills in the plan are complete. Task finished."
                        )
                        self.state.status = SystemStatus.IDLE
                        self.state.current_task = None
                        self.brain.interrupt_task("Plan completed successfully")

                    # Normal loop interval when no skill is running
                    time.sleep(loop_interval)

                # 5. Final system state check
                if self.brain.state.error_message:  # Brain有错误信息
                    print(
                        f"[RobotBrainSystem] Brain encountered error: {self.brain.state.error_message}"
                    )
                    self.state.status = SystemStatus.ERROR
                    self.state.error_message = self.brain.state.error_message

            except Exception as e:
                print(f"[RobotBrainSystem] Critical error in main loop: {e}")
                traceback.print_exc()
                self.state.error_message = str(e)
                self.state.status = SystemStatus.ERROR
                # Potentially try to recover or shutdown gracefully
                if self.brain:
                    self.brain.interrupt_task("Main loop critical error")
                time.sleep(1)  # Longer sleep on error

        print("[RobotBrainSystem] Main loop stopped.")

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
        print(f"[RobotBrainSystem] Brain monitoring result: {result}, Reason: {reason}")

        if result == "failed":
            self.interrupt_skill(
                f"Brain decision: {reason}", skill_status=SkillStatus.FAILED
            )
        elif result == "success" or result == "successed":
            print(f"[RobotBrainSystem] Brain decided skill is success: {reason}")
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
