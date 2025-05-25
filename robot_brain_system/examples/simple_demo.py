#!/usr/bin/env python3
"""
Simple demo to test the robot brain system.
"""

import time
import sys
import os
import traceback
import os

os.environ["DISPLAY"] = ":0"  # Set DISPLAY for GUI applications if needed

# Add the parent directory to the path to import the robot_brain_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_brain_system.core.system import (
    RobotBrainSystem,
    SystemStatus,
)  # Import SystemStatus Enum
from robot_brain_system.configs.config import DEVELOPMENT_CONFIG
from robot_brain_system.core.skill_manager import (
    get_skill_registry,
)  # For listing skills

# Import skills to ensure they are registered globally for the subprocess
import robot_brain_system.skills


def main():
    """Main demo function."""
    print("=" * 50)
    print("Robot Brain System Demo")
    print("=" * 50)

    system = None  # Define system here for finally block

    try:
        # Initialize the system
        print("\n1. Initializing Robot Brain System...")
        system = RobotBrainSystem(DEVELOPMENT_CONFIG)

        if not system.initialize():
            print("❌ Failed to initialize system")
            return False

        print("✅ System initialized successfully")

        # Check available skills (from global registry)
        print("\n2. Available Skills (Globally Registered):")
        skill_registry = get_skill_registry()
        skills = skill_registry.list_skills()

        if not skills:
            print(
                "⚠️  No skills found. Make sure skills are properly registered."
            )
            # Not necessarily a failure for the demo if some skills are there
        else:
            for skill_name in skills:
                skill_info = skill_registry.get_skill_info(skill_name)
                print(
                    f"   - {skill_name}: {skill_info['description']} ({skill_info['type']}, {skill_info['execution_mode']})"
                )

        # Start the system (starts the main loop thread)
        print("\n3. Starting system...")
        if not system.start():
            print("❌ Failed to start system")
            return False
        print("✅ System started successfully")

        # Wait a moment for system to stabilize
        time.sleep(2)

        # Execute a simple task via the brain
        print("\n4. Executing test task via Brain...")
        # This task should ideally resolve to 'reset_to_home' or a similar simple skill
        task_instruction = "Reset the robot to its starting home position"

        if system.execute_task(task_instruction):
            print(f"✅ Task started: {task_instruction}")

            # Monitor task execution
            print("\n5. Monitoring task execution (Brain + Simulator)...")
            start_time = time.time()
            # Max wait time for the entire high-level task
            max_wait_time_task = 60  # seconds

            while time.time() - start_time < max_wait_time_task:
                status_data = system.get_status()
                main_system_status_str = status_data["system"]["status"]
                brain_status_str = status_data["brain"].get(
                    "status", "unknown"
                )
                sim_skill_status = status_data["simulator"].get(
                    "skill_executor", {}
                )
                sim_skill_name = sim_skill_status.get("current_skill", "None")
                sim_skill_state = sim_skill_status.get("status", "idle")

                print(
                    f"   System: {main_system_status_str}, Brain: {brain_status_str}, SimSkill: {sim_skill_name}({sim_skill_state})"
                )

                # Check if the overall system (driven by brain's plan) is idle or errored
                if (
                    system.state.status == SystemStatus.IDLE
                    or system.state.status == SystemStatus.ERROR
                ):
                    break

                time.sleep(1)

            final_status_data = system.get_status()
            print(
                f"\n✅ Task execution attempt completed. Final System Status: {final_status_data['system']['status']}"
            )
            if system.state.status == SystemStatus.ERROR:
                print(
                    f"   Error Message: {final_status_data['system']['error_message']}"
                )
                # return False # Let it proceed to shutdown

        else:
            print(f"❌ Failed to start task: {task_instruction}")
            # return False # Let it proceed to shutdown

        # Test individual skill execution *directly in the simulator subprocess*
        print(
            "\n6. Testing individual skill execution in simulator subprocess..."
        )

        # Test direct skill execution (if skill is non-env, or for testing sim's direct exec)
        # For requires_env=False skills, one might have a local executor or send a special command
        # For this demo, let's test a skill that runs in the simulator
        if "emergency_stop" in skills:  # Assuming this is requires_env=False
            print("   Testing 'emergency_stop' (local direct execution)...")
            # This would use system.local_skill_executor if it's meant for non-env skills
            # success = system.local_skill_executor.execute_skill("emergency_stop", {"reason": "Demo test"}, env=None)
            # print(f"   ✅ emergency_stop local result: {success}")
            # OR if emergency_stop is also handled by sim's executor (e.g. to stop sim physics)
            print(
                "   (Skipping direct execution of 'emergency_stop' in this demo structure, handled by sim if needed)"
            )

        if "reset_to_home" in skills:
            print(
                "   Attempting to run 'reset_to_home' directly in simulator (non-blocking start)..."
            )
            if system.simulator.start_skill_non_blocking(
                "reset_to_home", {"home_position": [0.0] * 6}
            ):
                print("   ✅ 'reset_to_home' started in simulator.")
                time.sleep(1)  # Give it a moment
                # Monitor this specific skill
                skill_wait_start = time.time()
                while (
                    time.time() - skill_wait_start < 15
                ):  # Max 15s for this skill
                    sim_skill_status = (
                        system.simulator.get_skill_executor_status()
                    )
                    print(
                        f"      Sim Skill '{sim_skill_status.get('current_skill')}': {sim_skill_status.get('status')}"
                    )
                    if not sim_skill_status.get("is_running"):
                        break
                    time.sleep(0.5)
                final_sim_skill_status = (
                    system.simulator.get_skill_executor_status()
                )
                print(
                    f"   ✅ 'reset_to_home' final sim status: {final_sim_skill_status.get('status')}"
                )
            else:
                print("   ❌ Failed to start 'reset_to_home' in simulator.")

        print("\n7. Demo completed! 🎉")

        # Keep system running for a bit to observe, then shutdown is in finally
        print(
            "\n   System will continue running for 5 seconds before automatic shutdown..."
        )
        time.sleep(5)
        return True

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        print("\n8. Shutting down system...")
        if system:  # Check if system was initialized
            system.shutdown()
        print("✅ Shutdown completed")


if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(
        f"\nDemo {'completed successfully' if success else 'failed or had issues'}"
    )
    sys.exit(exit_code)
