"""
Skills package for the robot brain system.
"""

# Import skills to register them with the global registry
try:
    from . import basic_skills
    from . import manipulation_skills

    print("[Skills] Successfully imported skill modules")
except ImportError as e:
    print(f"[Skills] Warning: Failed to import some skill modules: {e}")

__all__ = ["basic_skills", "manipulation_skills"]
