#!/usr/bin/env python3
"""
Quick test script for the robot brain system components.

This script tests the individual components without requiring Isaac Lab.
"""

import sys
import os


def test_skill_registration():
    """Test skill registration system."""
    print("Testing skill registration...")

    from robot_brain_system.core.skill_manager import (
        SkillRegistry,
        skill_register,
    )
    from robot_brain_system.core.types import SkillType, ExecutionMode

    # Create registry
    registry = SkillRegistry()

    # Register a test skill
    @skill_register(
        name="test_skill",
        skill_type=SkillType.FUNCTION,
        execution_mode=ExecutionMode.DIRECT,
        description="A test skill",
    )
    def test_skill(params):
        return True

    # Register it manually
    registry.register_skill(
        name="test_skill",
        skill_type=SkillType.FUNCTION,
        execution_mode=ExecutionMode.DIRECT,
        function=test_skill,
        description="A test skill",
    )

    # Test retrieval
    skill = registry.get_skill("test_skill")
    assert skill is not None, "Skill not found"
    assert skill.name == "test_skill", "Skill name mismatch"

    print("✅ Skill registration test passed")
    return True


def test_skill_imports():
    """Test that skills can be imported and registered."""
    print("Testing skill imports...")

    try:
        # This should register all skills
        import robot_brain_system.skills

        from robot_brain_system.core.skill_manager import get_skill_registry

        registry = get_skill_registry()

        skills = registry.list_skills()
        print(f"   Found {len(skills)} registered skills:")

        for skill_name in skills:
            skill_info = registry.get_skill_info(skill_name)
            print(f"     - {skill_name}: {skill_info['description']}")

        assert len(skills) > 0, "No skills were registered"
        print("✅ Skill imports test passed")
        return True

    except Exception as e:
        print(f"❌ Skill imports test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_types():
    """Test type definitions."""
    print("Testing type definitions...")

    from robot_brain_system.core.types import (
        Action,
        Observation,
        SkillType,
        ExecutionMode,
        SkillDefinition,
        SystemStatus,
    )
    import numpy as np

    # Test Action
    action = Action(data=np.array([1, 2, 3]), metadata={"test": True})
    assert action.data is not None, "Action data is None"

    # Test Observation
    obs = Observation(data={"position": [1, 2, 3]}, timestamp=123.456)
    assert obs.get("position") == [1, 2, 3], "Observation get failed"

    # Test enums
    assert SkillType.FUNCTION.value == "function", "SkillType enum error"
    assert ExecutionMode.DIRECT.value == "direct", "ExecutionMode enum error"

    print("✅ Types test passed")
    return True


def test_brain_component():
    """Test brain component initialization."""
    print("Testing brain component...")

    try:
        from robot_brain_system.core.brain import QwenVLBrain
        from robot_brain_system.core.skill_manager import SkillRegistry

        # Test brain initialization
        config = {
            "qwen": {"model": "test-model", "api_key": "test-key"},
            "monitoring_interval": 1.0,
        }

        brain = QwenVLBrain(config)
        assert brain.model_name == "test-model", "Brain config error"

        # Test skill registry connection
        registry = SkillRegistry()
        brain.set_skill_registry(registry)

        print("✅ Brain component test passed")
        return True

    except Exception as e:
        print(f"❌ Brain component test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Robot Brain System Component Tests")
    print("=" * 50)

    tests = [
        test_types,
        test_skill_registration,
        test_skill_imports,
        test_brain_component,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        print(
            f"\n{test_func.__name__.replace('test_', '').replace('_', ' ').title()}:"
        )
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_func.__name__} failed")
        except Exception as e:
            print(f"❌ {test_func.__name__} crashed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
