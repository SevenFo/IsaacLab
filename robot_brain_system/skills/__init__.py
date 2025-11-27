"""
Skills package for the robot brain system.
"""

# Import skills to register them with the global registry
try:
    from . import simple_test_skills  # Phase 3 test skills (work with EnvProxy)
    from . import basic_skills
    from . import manipulation_skills  # Re-enabled for migration testing
    from . import observation_skills  # Re-enabled for ObjectTracking test

    print("[Skills] Successfully imported skill modules")
except ImportError as e:
    import traceback

    traceback.print_exc()
    print(f"[Skills] Warning: Failed to import some skill modules: {e}")

__all__ = [
    "simple_test_skills",
    "basic_skills",
    "manipulation_skills",
    "observation_skills",
]
