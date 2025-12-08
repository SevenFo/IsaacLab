#!/usr/bin/env python3
"""
Test script for HumanIntervention skill with UI integration.

This demonstrates:
1. Starting the full RobotBrainSystem with Textual UI
2. Manually triggering the human_intervention skill
3. UI dialog popup functionality
4. Human feedback handling

Usage:
    cd IsaacLab
    python -m robot_brain_system.examples.test_human_intervention

Note:
    This script runs the full system with UI. For a simpler test without UI,
    the HumanIntervention skill can be tested by manually typing 'c', 'f', or
    feedback text when prompted in the console.
"""

import time
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

from robot_brain_system.ui.console import global_console


def run_test_logic(system):
    """
    Test logic that runs in a background thread.
    Directly triggers the human_intervention skill.
    """
    # Wait for system to be ready
    time.sleep(3)

    global_console.log("info", "=" * 50)
    global_console.log("info", "HumanIntervention Skill Test")
    global_console.log("info", "=" * 50)

    # Get skill executor
    skill_executor = system.skill_executor
    if not skill_executor:
        global_console.log("error", "❌ SkillExecutor not available")
        return

    # List available skills
    global_console.log("info", "\n📋 Available skills:")
    for skill_name in skill_executor.registry.list_skills():
        global_console.log("info", f"   - {skill_name}")

    # Check if human_intervention is available
    if "human_intervention" not in skill_executor.registry.list_skills():
        global_console.log("error", "❌ human_intervention skill not found!")
        return

    # Start human_intervention skill
    global_console.log("brain", "\n🚀 Starting human_intervention skill...")
    global_console.log("info", "   The UI should show an intervention dialog")

    # Initialize the skill (will start its execution thread internally)
    success, _ = skill_executor.initialize_skill(
        "human_intervention",
        {
            "reason": "This is a test! The robot needs your guidance to proceed with the task."
        },
    )

    if not success:
        global_console.log("error", "❌ Failed to start skill")
        return

    global_console.log("success", "✅ Skill started")
    global_console.log("info", "\n📝 Instructions:")
    global_console.log("info", "   - Type 'c' to mark as success (SUCCESS)")
    global_console.log("info", "   - Type 'f' to mark as failed (FAILED)")
    global_console.log(
        "info", "   - Type any other text to provide feedback (triggers INTERRUPTED)"
    )
    global_console.log("info", "   - Press Ctrl+C or type /exit to quit")

    # Poll skill status
    timeout = 120
    start_time = time.time()

    while True:
        time.sleep(1)
        status = skill_executor.get_status()

        # is_running is derived from status == RUNNING
        is_running = skill_executor.is_running()
        skill_status = status.get("status", "unknown")
        status_info = status.get("status_info", "")

        elapsed = time.time() - start_time

        if not is_running:
            global_console.log("success", "\n✅ Skill finished!")
            global_console.log("info", f"   Final status: {skill_status}")
            global_console.log("info", f"   Status info: {status_info}")
            global_console.log("info", f"   Time: {elapsed:.1f}s")
            break

        if elapsed > timeout:
            global_console.log("warning", f"\n⚠️ Test timeout ({timeout}s)")
            skill_executor.terminate_current_skill()
            break

    global_console.log("info", "\nTest complete. Type /exit to quit.")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig):
    """Main entry with Hydra config."""

    def worker_task():
        """Backend worker that runs in background thread."""
        # Import here to avoid circular imports
        from robot_brain_system.core.system import RobotBrainSystem

        # Create system
        global_console.log("system", "Initializing RobotBrainSystem...")
        system = RobotBrainSystem(cfg)

        if not system.initialize():
            global_console.log("error", "❌ Failed to initialize system")
            return

        global_console.log("success", "✅ System initialized")

        # Start main loop so input routing and monitoring are active
        if not system.start():
            global_console.log("error", "❌ Failed to start system loop")
            return

        # Ensure UI exit triggers system shutdown
        global_console.set_shutdown_callback(system.shutdown)

        # Run test logic
        run_test_logic(system)

        # Keep system alive until UI exits
        try:
            while True:
                time.sleep(1)
        except:
            pass
        finally:
            global_console.log("system", "Shutting down...")
            system.shutdown()

    # Run UI with worker task
    global_console.run(worker_task)


if __name__ == "__main__":
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    main()
