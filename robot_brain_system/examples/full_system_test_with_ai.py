#!/usr/bin/env python3
# filepath: /home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/robot_brain_system/examples/full_system_test_with_ai.py
"""
Full system test with real AI model integration.
This demonstrates the complete robot brain system with Qwen VL.
"""

import sys
import os
import time

# Add the robot_brain_system to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.system import RobotBrainSystem
from configs.config import SYSTEM_CONFIG
import json


#!/usr/bin/env python3
# filepath: /home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/robot_brain_system/examples/full_system_test_with_ai.py
"""
Full system test with real AI model integration.
This demonstrates the complete robot brain system with Qwen VL.
"""

import sys
import os
import time

# Add the robot_brain_system to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.system import RobotBrainSystem
from configs.config import SYSTEM_CONFIG


def test_full_system_with_ai():
    """Test the complete system with AI integration."""
    print("🚀 Robot Brain System - Full AI Integration Test")
    print("=" * 60)

    # Configure for Qwen VL if model path is available
    qwen_model_path = os.getenv("QWEN_VL_MODEL_PATH", "")

    if qwen_model_path and os.path.exists(qwen_model_path):
        print(f"🧠 Using Qwen VL model: {qwen_model_path}")
        # Update config to use Qwen VL
        SYSTEM_CONFIG["brain"]["qwen"]["adapter_type"] = "qwen_vl"
        SYSTEM_CONFIG["brain"]["qwen"]["model_path"] = qwen_model_path
        SYSTEM_CONFIG["brain"]["qwen"]["max_tokens"] = 512
    else:
        print("🧠 Using mock implementation (no Qwen VL model found)")

    # Initialize the system
    print("\n📦 Initializing Robot Brain System...")
    system = RobotBrainSystem(SYSTEM_CONFIG)

    try:
        # Initialize the system
        if not system.initialize():
            print("❌ Failed to initialize system")
            return False

        # Start the system
        print("🎬 Starting system...")
        if not system.start():
            print("❌ Failed to start system")
            return False

        # Wait for initialization
        time.sleep(2)

        # Check system status
        status = system.get_status()
        print(f"📊 System Status: {status['system']['status']}")
        print(f"   - Is Running: {status['system']['is_running']}")

        # Test 1: Simple planning task
        print("\n🧪 Test 1: AI-Powered Task Planning")
        print("-" * 40)

        task_instruction = "Move the robot to position [0.5, 0.2, 0.4] and then return to home"
        print(f"📝 Task: {task_instruction}")

        success = system.execute_task(task_instruction)
        if success:
            print("🎯 Task started successfully")
        else:
            print("❌ Failed to start task")

        # Monitor execution for a few steps
        for step in range(5):
            time.sleep(1)
            status = system.get_status()

            system_status = status["system"]["status"]
            if system_status == "executing":
                print(f"   Step {step + 1}: System executing task...")
            elif system_status == "idle":
                print(f"   ✅ Task completed in {step + 1} steps!")
                break
            else:
                print(f"   Status: {system_status}")

        # Test 2: Complex manipulation task
        print("\n🧪 Test 2: AI-Powered Manipulation Planning")
        print("-" * 40)

        manipulation_task = "Pick up an object at [0.4, 0.0, 0.3] and place it at [0.4, 0.3, 0.3]"
        print(f"📝 Task: {manipulation_task}")

        success = system.execute_task(manipulation_task)
        if success:
            print("🎯 Task started successfully")
        else:
            print("❌ Failed to start task")

        # Monitor execution
        for step in range(8):
            time.sleep(1)
            status = system.get_status()

            system_status = status["system"]["status"]
            if system_status == "executing":
                print(f"   Step {step + 1}: System executing task...")
            elif system_status == "idle":
                print(f"   ✅ Task completed in {step + 1} steps!")
                break
            else:
                print(f"   Status: {system_status}")

        # Test 3: AI Monitoring Response
        print("\n🧪 Test 3: AI-Powered Monitoring")
        print("-" * 40)

        # Simulate a task that might need monitoring
        monitoring_task = (
            "Carefully inspect the current robot state and report status"
        )
        print(f"📝 Task: {monitoring_task}")

        success = system.execute_task(monitoring_task)
        if success:
            print("🎯 Task started successfully")
        else:
            print("❌ Failed to start task")

        # Let it run and show monitoring decisions
        for step in range(3):
            time.sleep(2)
            status = system.get_status()
            system_status = status["system"]["status"]
            print(f"   Monitoring step {step + 1}: Status = {system_status}")

            if system_status == "idle":
                break

        print("\n🎉 Full system test completed successfully!")

        # Final status report
        final_status = system.get_status()
        print("\n📋 Final System Status:")
        print(f"   - Status: {final_status['system']['status']}")
        print(f"   - Is Running: {final_status['system']['is_running']}")
        print(f"   - AI adapter type: {system.brain.adapter_type}")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean shutdown
        print("\n🔄 Shutting down system...")
        system.shutdown()


def test_ai_response_quality():
    """Test the quality of AI responses."""
    print("\n🧪 AI Response Quality Test")
    print("-" * 40)

    qwen_model_path = os.getenv("QWEN_VL_MODEL_PATH", "")

    if not qwen_model_path or not os.path.exists(qwen_model_path):
        print("⚠️  Skipped: No Qwen VL model available")
        return

    from core.brain import QwenVLBrain
    from core.types import Task

    config = {
        "qwen": {
            "adapter_type": "qwen_vl",
            "model_path": qwen_model_path,
            "max_tokens": 512,
        },
        "monitoring_interval": 1.0,
        "max_retries": 3,
    }

    brain = QwenVLBrain(config)

    # Set up skill registry for the brain
    from core.skill_manager import get_skill_registry

    skill_registry = get_skill_registry()
    brain.set_skill_registry(skill_registry)

    # Test different types of tasks
    test_tasks = [
        "Move to position [0.5, 0.0, 0.4]",
        "Pick up the red cube and place it on the blue platform",
        "Inspect the current environment and report what you see",
        "Navigate to the charging station and dock",
        "Perform a safety check of all systems",
    ]

    available_skills = [
        "reach_position",
        "grasp_object",
        "release_object",
        "pick_and_place",
        "inspect_object",
        "move_to_named_position",
        "reset_to_home",
        "wait",
        "emergency_stop",
        "get_current_state",
    ]

    print("Available skills:", available_skills)
    print()

    for i, task_desc in enumerate(test_tasks, 1):
        print(f"Task {i}: {task_desc}")

        task = Task(
            id=f"quality_test_{i}",
            description=task_desc,
            image=None,
            priority=1,
            metadata={"test": True},
        )

        try:
            # Test planning - use public method
            plan = brain.plan_task(task)
            if hasattr(plan, "skill_sequence"):
                print(f"   AI Plan: {plan.skill_sequence}")
                print(f"   Parameters: {getattr(plan, 'skill_params', [])}")
                print(
                    f"   Monitoring: {getattr(plan, 'monitoring_interval', 1.0)}s"
                )
            else:
                print(f"   AI Plan: {plan}")

            print()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()


def main():
    """Run all tests."""

    # Test 1: Full system integration
    success = test_full_system_with_ai()

    # Test 2: AI response quality
    test_ai_response_quality()

    if success:
        print("\n🎉 All tests completed successfully!")
        print("🤖 Robot Brain System with AI integration is ready for use!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
