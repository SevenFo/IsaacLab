#!/usr/bin/env python3
# filepath: /home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/robot_brain_system/examples/test_model_adapters.py
"""
Test script for model adapter integration.
This script tests different model adapters without requiring the full system.
"""

import sys
import os

# Add the robot_brain_system to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.brain import QwenVLBrain
from core.types import Task
import time


def test_mock_adapter():
    """Test the mock adapter (should always work)."""
    print("🧪 Testing Mock Adapter...")

    config = {
        "qwen": {
            "adapter_type": "mock",
            "max_tokens": 512,
        },
        "monitoring_interval": 1.0,
        "max_retries": 3,
    }

    brain = QwenVLBrain(config)

    # Create a test task
    task = Task(
        id="test_mock_task",
        description="Pick up the red cube and place it on the blue platform",
        image=None,  # No image for mock test
        priority=1,
        metadata={"test": True},
    )

    print(f"   Task: {task.description}")

    try:
        # Test planning
        plan = brain._query_qwen_for_plan(
            task, ["pick_and_place", "reach_position", "grasp_object"]
        )
        print(f"   ✅ Planning successful: {plan.skill_sequence}")

        # Test monitoring
        current_skill = {
            "name": "pick_and_place",
            "parameters": {"target": "red_cube"},
        }
        decision = brain._query_qwen_for_monitoring(task, current_skill, None)
        print(f"   ✅ Monitoring successful: {decision['action']}")

        return True

    except Exception as e:
        print(f"   ❌ Mock adapter test failed: {e}")
        return False


def test_openai_adapter():
    """Test the OpenAI adapter (requires API key)."""
    print("🧪 Testing OpenAI Adapter...")

    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("   ⚠️  Skipped: OPENAI_API_KEY environment variable not set")
        return None

    config = {
        "qwen": {
            "adapter_type": "openai",
            "api_key": api_key,
            "model": "gpt-4o-mini",  # Use cheaper model for testing
            "max_tokens": 256,
        },
        "monitoring_interval": 1.0,
        "max_retries": 3,
    }

    brain = QwenVLBrain(config)

    if brain.adapter_type == "mock":
        print("   ⚠️  Fell back to mock (OpenAI adapter initialization failed)")
        return False

    # Create a test task
    task = Task(
        id="test_openai_task",
        description="Pick up the red cube and place it on the blue platform",
        image=None,
        priority=1,
        metadata={"test": True},
    )

    print(f"   Task: {task.description}")

    try:
        # Test planning
        plan = brain._query_qwen_for_plan(
            task, ["pick_and_place", "reach_position", "grasp_object"]
        )
        print(f"   ✅ Planning successful: {plan.skill_sequence}")

        # Test monitoring
        current_skill = {
            "name": "pick_and_place",
            "parameters": {"target": "red_cube"},
        }
        decision = brain._query_qwen_for_monitoring(task, current_skill, None)
        print(f"   ✅ Monitoring successful: {decision['action']}")

        return True

    except Exception as e:
        print(f"   ❌ OpenAI adapter test failed: {e}")
        return False


def test_qwen_vl_adapter():
    """Test the Qwen VL adapter (requires local model)."""
    print("🧪 Testing Qwen VL Adapter...")

    # Check if model path is available
    model_path = os.getenv("QWEN_VL_MODEL_PATH", "")
    if not model_path:
        print("   ⚠️  Skipped: QWEN_VL_MODEL_PATH environment variable not set")
        return None

    if not os.path.exists(model_path):
        print(f"   ⚠️  Skipped: Model path {model_path} does not exist")
        return None

    config = {
        "qwen": {
            "adapter_type": "qwen_vl",
            "model_path": model_path,
            "max_tokens": 256,
        },
        "monitoring_interval": 1.0,
        "max_retries": 3,
    }

    brain = QwenVLBrain(config)

    if brain.adapter_type == "mock":
        print(
            "   ⚠️  Fell back to mock (Qwen VL adapter initialization failed)"
        )
        return False

    # Create a test task
    task = Task(
        id="test_qwen_task",
        description="Pick up the red cube and place it on the blue platform",
        image=None,
        priority=1,
        metadata={"test": True},
    )

    print(f"   Task: {task.description}")

    try:
        # Test planning
        plan = brain._query_qwen_for_plan(
            task, ["pick_and_place", "reach_position", "grasp_object"]
        )
        print(f"   ✅ Planning successful: {plan.skill_sequence}")

        # Test monitoring
        current_skill = {
            "name": "pick_and_place",
            "parameters": {"target": "red_cube"},
        }
        decision = brain._query_qwen_for_monitoring(task, current_skill, None)
        print(f"   ✅ Monitoring successful: {decision['action']}")

        return True

    except Exception as e:
        print(f"   ❌ Qwen VL adapter test failed: {e}")
        return False


def main():
    """Run all adapter tests."""
    print("🚀 Robot Brain System - Model Adapter Integration Tests")
    print("=" * 60)

    results = {}

    # Test mock adapter (should always work)
    results["mock"] = test_mock_adapter()
    print()

    # Test OpenAI adapter (optional)
    results["openai"] = test_openai_adapter()
    print()

    # Test Qwen VL adapter (optional)
    results["qwen_vl"] = test_qwen_vl_adapter()
    print()

    # Summary
    print("📊 Test Results Summary:")
    print("-" * 30)

    total_tests = 0
    passed_tests = 0

    for adapter, result in results.items():
        if result is not None:
            total_tests += 1
            if result:
                passed_tests += 1
                print(f"   ✅ {adapter.upper()}: PASSED")
            else:
                print(f"   ❌ {adapter.upper()}: FAILED")
        else:
            print(f"   ⚠️  {adapter.upper()}: SKIPPED")

    print(f"\nPassed: {passed_tests}/{total_tests} tests")

    if passed_tests == total_tests and total_tests > 0:
        print("🎉 All available adapter tests passed!")
        return 0
    elif passed_tests > 0:
        print("⚠️  Some tests passed, some failed or were skipped")
        return 1
    else:
        print("❌ No tests passed")
        return 2


if __name__ == "__main__":
    sys.exit(main())
