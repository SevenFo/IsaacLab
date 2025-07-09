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
)  # Generator might not be needed at this level anymore
import traceback  # For debugging
from PIL import Image
import matplotlib
import multiprocessing

matplotlib.use("Agg")
multiprocessing.set_start_method("spawn", force=True)  # Ensure compatibility

from .types import (
    SystemStatus,
    Action,
    SystemState,
    SkillStatus,
)
from .isaac_simulator import IsaacSimulator
from .skill_manager import (
    SkillRegistry,
    SkillExecutor,
    get_skill_registry,
)  # Local SkillExecutor might be for non-env skills
from .brain import QwenVLBrain


class RobotBrainSystem:
    """
    Main system that orchestrates the Isaac simulator, skills, and brain.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.visualize = True
        self.log_path = config.get("log_path", "logs")
        self.state = SystemState()

        # Core components
        self.simulator: Optional[IsaacSimulator] = None
        self.skill_registry: SkillRegistry = get_skill_registry()  # Global registry
        # This SkillExecutor might be for skills not requiring the env, or removed if all skills go to subprocess
        # self.local_skill_executor: SkillExecutor = SkillExecutor(self.skill_registry)
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

            # Initialize Isaac simulator
            sim_config_from_main = self.config.get("simulator", {})
            # scene_cfg_path could be part of sim_config_from_main or None
            self.simulator = IsaacSimulator(sim_config=sim_config_from_main)

            if (
                not self.simulator.initialize()
            ):  # This now starts the subprocess and waits for "ready"
                print("[RobotBrainSystem] Failed to initialize simulator")
                self.state.status = SystemStatus.ERROR
                self.state.error_message = "Simulator initialization failed"
                return False
            print("[RobotBrainSystem] Simulator initialized successfully.")

            self.brain.initialize()

            # Import skills to register them (skills are auto-registered via decorators)
            try:
                import robot_brain_system.skills  # noqa: F401

                print("[RobotBrainSystem] Skills imported successfully")
            except Exception as e:
                print(f"[RobotBrainSystem] Warning: Failed to import skills: {e}")
                traceback.print_exc()

            print(
                f"[RobotBrainSystem] Registered {len(self.skill_registry.list_skills())} skills globally."
            )

            # Connect brain to skill registry (for planning purposes)
            self.brain.set_skill_registry(self.skill_registry)
            self.brain.set_system_state(state=self.state)

            self.state.status = SystemStatus.IDLE
            print("[RobotBrainSystem] Initialization complete")

            self.log_path = os.path.join(self.log_path, time.strftime("%Y%m%d_%H%M%S"))
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
        if self.brain and self.brain.state.status not in [
            SystemStatus.IDLE,
            SystemStatus.ERROR,
        ]:
            self.brain.interrupt_task("System shutdown")

        # Terminate any skill in the simulator subprocess
        if self.simulator and self.simulator.is_initialized:
            skill_status = self.simulator.get_skill_executor_status()
            if skill_status.get("is_running"):
                print("[RobotBrainSystem] Terminating skill in simulator subprocess...")
                self.simulator.terminate_current_skill()

            # Shutdown simulator (this will also stop its process)
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

        # Check if another high-level task (via brain) is already in progress
        if self.brain.state.status not in [
            SystemStatus.IDLE,
            SystemStatus.ERROR,
        ]:
            print(
                f"[RobotBrainSystem] Brain is busy with another task: {self.brain.state.status.value}"
            )
            # Optionally, allow queueing or interrupt current brain task
            return False

        # Check if a skill is running in the simulator from a previous plan
        sim_skill_status = (
            self.simulator.get_skill_executor_status() if self.simulator else {}
        )
        if sim_skill_status.get("is_running"):
            print(
                f"[RobotBrainSystem] Simulator is busy executing skill: {sim_skill_status.get('current_skill')}. Please wait or interrupt."
            )
            return False

        print(f"[RobotBrainSystem] Received task instruction: {instruction}")
        try:
            self.state.status = SystemStatus.THINKING
            obss = self.simulator.get_observation()
            if (not self.state.obs_history) and (not obss):
                print(
                    "[RobotBrainSystem] No observations available from simulator. Cannot execute task."
                )
                self.state.status = SystemStatus.ERROR
                self.state.error_message = "No observations available"
                return False
            obs = obss[-1] if obss else self.state.obs_history[-1]
            inspector_rgb = obs.data["policy"]["camera_top"][0].cpu().numpy()
            if self.visualize:
                import matplotlib.pyplot as plt

                plt.imshow(inspector_rgb)
                plt.axis("off")
                plt.savefig(os.path.join(self.log_path, "execute_task_input.png"))
            image_data = Image.fromarray(inspector_rgb)
            self.state.current_task = self.brain.parse_task(instruction, image_data)

            # Brain plans the task. execute_task in brain sets its internal state.
            plan = self.brain.execute_task(self.state.current_task)
            if not plan or not plan.skill_sequence:
                print("[RobotBrainSystem] Brain did not produce a valid plan.")
                self.state.status = SystemStatus.IDLE
                self.brain.interrupt_task("No plan generated")  # Reset brain state
                return False
            self.state.plan_history.append(plan)

            self.state.status = (
                SystemStatus.EXECUTING
            )  # System is now executing the brain's plan
            print(
                f"[RobotBrainSystem] Brain planned {len(plan.skill_sequence)} skills. Starting execution..."
            )

            # The _main_loop will pick up the first skill from brain.get_next_skill()
            # and request IsaacSimulator to run it.
            return True

        except Exception as e:
            print(f"[RobotBrainSystem] Failed to start task execution: {e}")
            traceback.print_exc()
            self.state.error_message = str(e)
            self.state.status = SystemStatus.ERROR
            if self.brain.state.status != SystemStatus.IDLE:
                self.brain.interrupt_task(f"Error during task start: {e}")
            return False

    def interrupt_task(self, reason: str = "User interrupt"):
        """Interrupt the current high-level task being managed by the brain."""
        print(f"[RobotBrainSystem] Interrupting current task: {reason}")
        if self.brain:
            self.brain.interrupt_task(reason)  # This resets brain's plan

        # Terminate any skill running in the simulator subprocess
        if self.simulator and self.simulator.is_initialized:
            sim_skill_status = self.simulator.get_skill_executor_status()
            if sim_skill_status.get("is_running"):
                print(
                    "[RobotBrainSystem] Sending terminate signal to skill in simulator..."
                )
                self.simulator.terminate_current_skill()

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
                "[RobotBrainSystem] No skill in the brain! ignore interrpt requirement!"
            )
            return True
        print(
            f"[RobotBrainSystem] Interrupting current skill:\n{brain_skill_info}, with status: {skill_status} and reason: {reason}\n"
        )
        # Terminate any skill running in the simulator subprocess
        # Skill is_running status will be False if interrupt command worked, then, this status will be indicated in _main_loop to handle the skill status
        # Task status well be no changed
        # brain not track skill status  only record current skill info!
        if self.simulator and self.simulator.is_initialized:
            sim_skill_status = self.simulator.get_skill_executor_status()
            # TODO
            # when the skill is finished while dicision havnt been marking, it would occure belowe situation, NEED TO RECORED LAST SKILL IN SKILL EXECUTION
            # assert brain_skill_info['name'] == sim_skill_status['current_skill'], f"brain skill info {brain_skill_info['name']} != sim skill status {sim_skill_status['current_skill']}"
            if sim_skill_status.get("is_running"):
                print(
                    "[RobotBrainSystem] Sending terminate signal to skill in simulator..."
                )
                return self.simulator.terminate_current_skill(skill_status)

            else:
                print(
                    f"[RobotBrainSystem] Skill is not in running status, directly change the state of executor to {skill_status}"
                )
                return True
        else:
            raise RuntimeError("Simulator is not initialzed!")

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
                "skill_executor": self.simulator.get_skill_executor_status()
                if self.simulator.is_initialized
                else {"status": "not_initialized"},
            }

        brain_s = self.brain.get_status() if self.brain else {}
        # If brain is executing, but sim skill executor is idle, it means brain is planning the next skill.
        current_system_op = self.state.status.value
        if (
            self.state.status == SystemStatus.EXECUTING
            and brain_s.get("status") == SystemStatus.EXECUTING
        ):
            if not sim_status.get("skill_executor", {}).get("is_running"):
                current_skill_name_from_brain = brain_s.get(
                    "current_task"
                )  # This is confusing, brain.get_next_skill() is better
                next_skill_info = self.brain.get_next_skill()
                if next_skill_info:
                    current_system_op = f"executing_plan (waiting_to_start_skill: {next_skill_info.get('name')})"
                else:  # Plan might be finishing or brain is advancing
                    current_system_op = "executing_plan (brain_advancing_or_plan_done)"
            else:  # Skill is running in sim
                current_system_op = f"executing_plan (skill_running_in_sim: {sim_status.get('skill_executor', {}).get('current_skill')})"
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

    def step_simulation(self, action: Action) -> bool:
        """Manually step the simulation (for testing/teleop)."""
        assert False, "Unavailable now!"
        if not self.simulator or not self.simulator.is_initialized:
            print("[RobotBrainSystem] Simulator not ready for manual step.")
            return False

        try:
            result = self.simulator.step_env(action)  # step_env now returns a tuple
            if result:
                obs, reward, terminated, truncated, info = result
                self.state.last_observation = obs
                # self.state.last_action = action # Action object, not raw data
                print(
                    f"[RobotBrainSystem] Manual step - Obs: {str(obs.data)[:50]}, Reward: {reward}, Done: {terminated or truncated}"
                )
                return True
            else:
                print(
                    "[RobotBrainSystem] Manual step failed to get result from simulator."
                )
                return False
        except Exception as e:
            print(f"[RobotBrainSystem] Manual step failed: {e}")
            traceback.print_exc()
            return False

    def reset_simulation(self) -> bool:
        """Reset the simulation environment."""
        if not self.simulator or not self.simulator.is_initialized:
            print("[RobotBrainSystem] Simulator not ready for reset.")
            return False
        try:
            obss = self.simulator.reset_env()
            if obss:
                self.state.last_observation = obss[-1]
                self.state.obs_history = []
                # self.state.last_action = None
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

    def _main_loop(self):
        """Main system loop that runs in a separate thread."""
        print("[RobotBrainSystem] Main loop started.")
        loop_interval = 0.5  # seconds, adjust as needed

        while not self.is_shutdown_requested:
            loop_start_time = time.time()
            try:
                if not self.state.is_running:  # Paused or shutting down
                    time.sleep(loop_interval)
                    continue

                # 1. Get current observation from simulator if available
                if self.simulator and self.simulator.is_initialized:
                    obss = self.simulator.get_observation()
                    if obss:
                        self.state.last_observation = obss[-1]
                        self.state.obs_history.extend(obss)
                        if len(self.state.obs_history) > 100:
                            self.state.obs_history.pop(0)
                        print(f"obs history length: {len(self.state.obs_history)}")
                    else:
                        if obss == []:
                            # No observations available, might be a sim issue
                            print(
                                "[RobotBrainSystem] Warning: No observations received from simulator."
                            )
                        else:
                            # This could happen if pipe is broken or sim is resetting
                            print(
                                "[RobotBrainSystem] Warning: Failed to get observation from simulator."
                            )
                            # Potentially handle simulator comms error here

                # 2. Handle brain monitoring if a skill is active in the brain
                if (
                    self.brain.state.status == SystemStatus.EXECUTING
                    or self.brain.state.status == SystemStatus.MONITORING
                ):
                    if self.brain.should_monitor():  # Brain decides if it's time
                        print("[RobotBrainSystem] Brain monitoring execution...")

                        decision = self.brain.monitor_skill_execution(
                            self.state.obs_history
                        )

                        self._handle_monitoring_decision(decision)
                        # if (
                        #     self.brain.state.status == SystemStatus.IDLE
                        # ):  # Monitoring decided task is complete/interrupted
                        #     self.state.status = SystemStatus.IDLE
                        #     self.state.current_task = None

                # 3. Manage skill execution based on brain's plan
                if (
                    self.state.status == SystemStatus.EXECUTING
                    and self.brain.state.status == SystemStatus.EXECUTING
                ):
                    # TODO 没必要通过simulation去获取skill state啊
                    sim_skill_exec_status = (
                        self.simulator.get_skill_executor_status()
                        if self.simulator
                        else {}
                    )
                    self.state.sub_skill_status = sim_skill_exec_status
                    print(
                        f"[RobotBrainSystem] Current skill execution status: {sim_skill_exec_status['status']}"
                    )
                    if not sim_skill_exec_status.get("is_running"):
                        # Current skill in sim finished or no skill started yet for current brain step.
                        # Check if the last skill in sim was successful (if one was running)
                        last_sim_skill_state = sim_skill_exec_status.get("status")
                        if len(self.state.skill_history) == 0:
                            print(
                                "[RobotBrainSystem] No skill be executing now, start to exec first skill."
                            )
                        elif (
                            last_sim_skill_state == "completed"
                            or last_sim_skill_state == "success"
                        ):  # Successful completion from SkillStatus enum
                            current_skill = self.state.skill_history[-1]
                            print(
                                f"[RobotBrainSystem] Subprocess skill [{current_skill['name']}] reported COMPLETED."
                            )
                            self.state.skill_history[-1]["result"] = (
                                last_sim_skill_state
                            )
                            self.brain.advance_skill()  # Tell brain to move to next skill
                        elif (
                            last_sim_skill_state == "failed"
                            or last_sim_skill_state == "timeout"
                        ):
                            # NEED TO TOGGLE BRAIN TO HANDLE THIS SITUATION
                            current_skill = self.state.skill_history[-1]
                            print(
                                f"[RobotBrainSystem] Subprocess skill [{current_skill['name']}] reported FAILED."
                            )
                            self.state.skill_history[-1]["result"] = (
                                last_sim_skill_state
                            )
                            # Brain needs to handle this failure (e.g. replan, abort)
                            # For now, we'll tell brain to advance, and monitoring should catch it or brain handles error.
                            # Or, directly set brain to error or make it replan.
                            # This interaction needs refinement.
                            # A simple approach: if skill fails, task fails.
                            # self.brain.interrupt_task(
                            #     f"Skill {sim_skill_exec_status.get('current_skill')} failed in subprocess."
                            # )
                            # self.state.status = SystemStatus.IDLE  # Or ERROR
                            # self.state.error_message = f"Skill {sim_skill_exec_status.get('current_skill')} failed."
                            # continue to next loop iteration to stop further processing of this plan
                            last_plan_info = self.state.plan_history[-1]
                            last_skill_info = self.state.skill_history[-1]
                            last_skill_execution_summary = (
                                self.brain.summary_skill_execution(last_skill_info)
                            )
                            self.state.skill_history[-1]["execution_summary"] = (
                                last_skill_execution_summary
                            )
                            obss = self.simulator.get_observation()
                            if obss:
                                self.state.obs_history.extend(obss)
                            else:
                                self.state.obs_history = self.state.obs_history[-1:]
                            obs = self.state.obs_history[-1]
                            assert self.state.current_task
                            new_plan = self.brain.replan_task(
                                self.state.current_task,
                                last_plan_info,
                                self.state.skill_history,
                                obs,
                            )
                            if new_plan:
                                self.state.plan_history.append(new_plan)
                                self.state.skill_history = []
                        elif last_sim_skill_state == "interrupted":
                            assert False, "unsupported now"
                            print(
                                f"[RobotBrainSystem] Subprocess skill '{sim_skill_exec_status.get('current_skill', 'Unknown')}' was INTERRUPTED."
                            )
                            # This might have been triggered by brain's monitoring decision or externally.
                            # Brain should already be aware or will be reset.
                            # No need to advance_skill if brain itself triggered it.
                            pass  # Brain state should reflect this already

                        # Get next skill from brain's plan
                        next_skill_to_run = self.brain.get_next_skill()
                        if next_skill_to_run:
                            skill_name = next_skill_to_run["name"]
                            skill_params = next_skill_to_run["parameters"]
                            print(
                                f"[RobotBrainSystem] Requesting simulator to start skill: {skill_name} with params {skill_params}"
                            )
                            if self.simulator:
                                success = self.simulator.start_skill_non_blocking(
                                    skill_name, skill_params
                                )
                                if not success:
                                    print(
                                        f"[RobotBrainSystem] Failed to start skill '{skill_name}' in simulator."
                                    )
                                    self.brain.interrupt_task(
                                        f"Failed to start skill {skill_name}"
                                    )
                                    self.state.status = SystemStatus.ERROR
                                    self.state.error_message = (
                                        f"Failed to start skill {skill_name}"
                                    )
                                self.state.skill_history.append(next_skill_to_run)
                        else:
                            # No more skills in brain's plan, and last sim skill (if any) is done.
                            # TODO IT SHOULD BE DOUBLE CHECKED THE TASK IS FINISHED NOR NOT, NOT SUPPORTED NOW
                            if (
                                self.brain.state.status == SystemStatus.EXECUTING
                            ):  # Brain thinks it has more or just finished
                                print(
                                    "[RobotBrainSystem] Brain has no next skill, but is still in EXECUTING state. Assuming plan completion."
                                )
                                self.brain.state.status = (
                                    SystemStatus.IDLE
                                )  # Mark brain as done with this plan

                            if (
                                self.brain.state.status == SystemStatus.IDLE
                            ):  # Brain confirms plan is done
                                print(
                                    "[RobotBrainSystem] All skills in brain's plan completed."
                                )
                                self.state.status = SystemStatus.IDLE
                                self.state.current_task = (
                                    None  # Clear the high-level task
                                )
                    else:
                        # Skill is running in simulator, print status
                        # print(f"[RobotBrainSystem] Skill '{sim_skill_exec_status.get('current_skill')}' is running in simulator. Status: {sim_skill_exec_status.get('status')}")
                        pass

                # Handle overall system state transitions based on brain & simulator
                if (
                    self.brain.state.status == SystemStatus.IDLE
                    and self.state.status == SystemStatus.EXECUTING
                ):
                    # Brain finished its plan, system should go idle if sim is also idle
                    sim_skill_exec_status = (
                        self.simulator.get_skill_executor_status()
                        if self.simulator
                        else {}
                    )
                    if not sim_skill_exec_status.get("is_running"):
                        print(
                            "[RobotBrainSystem] Brain and Simulator are idle. Task complete. System transitioning to IDLE."
                        )
                        self.state.status = SystemStatus.IDLE
                        self.state.current_task = None
                elif self.brain.state.status == SystemStatus.ERROR:
                    print(
                        "[RobotBrainSystem] Brain entered ERROR state. System transitioning to ERROR."
                    )
                    self.state.status = SystemStatus.ERROR
                    self.state.error_message = self.brain.state.error_message

                # Sleep to control loop frequency
                elapsed_time = time.time() - loop_start_time
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

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

    def _handle_monitoring_decision(self, decision: Dict[str, Any]):
        """Handle brain monitoring decisions.
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
        action = decision.get("action", "continue")
        reason = decision.get("reason", "")
        print(
            f"[RobotBrainSystem] Brain monitoring decision: {action}, Reason: {reason}"
        )

        if action == "failed":
            self.interrupt_skill(
                f"Brain decision: {reason}", skill_status=SkillStatus.FAILED
            )
        elif action == "retry":
            assert False, "Unsuportted now!"
            # Retry logic for a skill running in the subprocess is complex.
            # The brain might decide to retry the *current skill in its plan*.
            # This would mean:
            # 1. Terminate current skill in sim: self.simulator.terminate_current_skill()
            # 2. Brain's current_skill_index should NOT advance.
            # 3. The main loop will then try to re-start the same skill.
            print(
                f"[RobotBrainSystem] Brain decided to retry current skill: {reason}. Terminating and re-queueing."
            )
            if self.simulator and self.simulator.is_initialized:
                current_sim_skill = self.simulator.get_skill_executor_status().get(
                    "current_skill"
                )
                if current_sim_skill:  # A skill is actually running
                    self.simulator.terminate_current_skill()
                # The brain's current_skill_index is NOT advanced by `interrupt_task`
                # So the _main_loop should pick up the same skill again.
                # We need to make sure brain.get_next_skill() returns the current one again.
                # This might require a new method in brain, e.g., brain.retry_current_skill_in_plan()
                # For now, we rely on the main loop to pick it up after termination.
            else:
                print("[RobotBrainSystem] Cannot retry, simulator not available.")
        elif action == "successed":
            print(f"[RobotBrainSystem] Brain decided skill is successed: {reason}")
            self.interrupt_skill(
                f"Brain decision: {reason}", skill_status=SkillStatus.COMPLETED
            )
        elif action == "not enough":
            # not enough observation to determine
            pass
        elif action == "continue":
            pass
            self.state.obs_history = [self.state.obs_history[-1]]
            print(
                f"[RobotBrainSystem] Clear system.state.obs_history after monitoring, current len: {len(self.state.obs_history)}"
            )


if __name__ == "__main__":
    import os

    print("TEST ROBOT BRAIN SYSTEM")
    from robot_brain_system.configs.config import DEVELOPMENT_CONFIG

    system = RobotBrainSystem(DEVELOPMENT_CONFIG)
    result = system.initialize()
    system.start()
    system.execute_task(
        "press the yellow button"
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
